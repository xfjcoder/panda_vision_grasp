"""Behavior cloning with auxiliary FK-derived pose-error losses, targeting the two learned
biases found in bc06/bc08's rollouts: the gripper closes ~0.049m above the object's center
(README item 19) and its closing axis is often several degrees off the object's flat-face
normal (item 22) -- both frozen across DAgger rounds, both the kind of continuous per-episode
target that's fully varied and always-correct in the demo labels, but imperfectly reproduced by
the trained policy.

--aux z   : vertical offset from the scene-CONSTANT target grasp height (table height + the
            object's fixed half-size -- only x/y is ever randomized, so this needs no
            privileged per-episode data at all). This is exactly bc08's original recipe.
--aux yaw : gripper-closing-axis yaw (mod 90 degrees, the box's cross-section is square) minus
            a per-EPISODE target, approximated as that episode's own final-step yaw (the
            post-lift hold phase keeps the arm static for many steps, so this is a stable,
            cheap proxy for "the orientation this episode actually settled on" without needing
            to save object pose during data collection). Caveat: for a badly-diverged DAgger
            rollout where even the FINAL orientation is off, this proxy inherits that error --
            imperfect, but still informative on average since most of the aggregated dataset's
            episodes do converge reasonably close by the post-lift hold.
--aux both: both heads trained jointly, each contributing its own weighted MSE term.

Both labels come from forward-kinematics-ing each transition's OWN proprio (`mj_kinematics`,
~0.002ms/call, no physics needed) -- cheap enough to precompute for the whole dataset up front,
no per-batch mujoco calls during training.

Reuses train_bc.py's exact MSE-on-action-mean recipe and PPO MultiInputPolicy network -- the
only addition is one or two small linear heads reading the actor's latent features
(`policy.mlp_extractor.forward_actor(...)`, the same tensor `get_distribution` computes
internally but doesn't expose), each with its OWN separate optimizer (NOT added as a param group
on `policy.optimizer` -- that broke `PPO.load`'s optimizer-state restore the first time this was
tried, see bc08's writeup; gradients still flow into the shared trunk the same way regardless of
which optimizer object calls `.step()`). Discarded at eval time -- eval_policy.py loads the saved
PPO model unchanged, exactly as for every other bc0N checkpoint.

Usage:
    python3 train_bc_auxz.py --run-name bc09 --demos data/demos_dagger4.npz --aux yaw
    python3 train_bc_auxz.py --run-name bc10 --demos data/demos_dagger4.npz --aux both
    python3 train_bc_auxz.py --run-name bc11 --demos data/demos_dagger5.npz --aux z
    python3 train_bc_auxz.py --run-name bc12 --demos data/demos_dagger5.npz --aux both
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from grasp_env import EPISODE_STEPS, GraspPixelEnv, N_ARM

HERE = Path(__file__).parent


def gripper_yaw_from_qpos(env, arm_qpos):
    env.data.qpos[:N_ARM] = arm_qpos
    mj.mj_kinematics(env.model, env.data)
    m = env.data.site_xmat[env.site_id].reshape(3, 3)[:, 1]
    return np.arctan2(m[1], m[0])


def wrap_quarter_pi(err):
    """Wrap an angular error to the box's 90-degree rotational symmetry, result in
    [-pi/4, pi/4] -- same convention as the item-22 diagnostic's align_err."""
    diff = err % (np.pi / 2)
    return ((diff + np.pi / 4) % (np.pi / 2)) - np.pi / 4


def compute_z_error_labels(env, proprio):
    """FK each transition's own joint angles to the gripper site's world z, relative to the
    constant target grasp height. No physics step needed, mj_kinematics alone is sufficient."""
    target_z = env._table_top_z() + env.obj_half_height
    z_errors = np.empty(len(proprio), dtype=np.float32)
    for i in range(len(proprio)):
        env.data.qpos[:N_ARM] = proprio[i, :N_ARM]
        mj.mj_kinematics(env.model, env.data)
        z_errors[i] = env.data.site_xpos[env.site_id][2] - target_z
    return z_errors


def compute_yaw_error_labels(env, proprio):
    """Per-episode target yaw = that episode's own final-step gripper yaw (the post-lift hold
    phase is static, a stable proxy for the orientation the episode actually settled on). Label
    for every transition in that episode = its own current yaw minus the target, wrapped to the
    box's 90-degree symmetry."""
    n = len(proprio)
    n_eps = n // EPISODE_STEPS
    assert n_eps * EPISODE_STEPS == n, "expected every episode to contribute EPISODE_STEPS rows"

    yaws = np.empty(n, dtype=np.float32)
    for i in range(n):
        yaws[i] = gripper_yaw_from_qpos(env, proprio[i, :N_ARM])

    yaw_errors = np.empty(n, dtype=np.float32)
    for ep in range(n_eps):
        s, e = ep * EPISODE_STEPS, (ep + 1) * EPISODE_STEPS
        target = yaws[e - 1]
        yaw_errors[s:e] = [wrap_quarter_pi(y - target) for y in yaws[s:e]]
    return yaw_errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--demos", type=str, default="data/demos.npz")
    ap.add_argument("--aux", choices=["z", "yaw", "both"], default="z",
                     help="Which auxiliary FK-derived pose-error head(s) to train jointly with "
                          "the main action MSE.")
    ap.add_argument("--aux-weight", type=float, default=1.0,
                     help="Weight on EACH normalized auxiliary MSE, relative to the main action "
                          "MSE (losses are summed, not averaged).")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    use_z = args.aux in ("z", "both")
    use_yaw = args.aux in ("yaw", "both")

    run_dir = HERE / "runs" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    data = np.load(args.demos)
    images, wrist_images = data["images"], data["wrist_images"]
    proprio, demo_actions = data["proprio"], data["actions"]
    n = len(demo_actions)
    print(f"loaded {n} demo transitions from {args.demos}, aux={args.aux}")

    fk_env = GraspPixelEnv()
    aux_targets = {}
    if use_z:
        z_errors = compute_z_error_labels(fk_env, proprio)
        z_mean, z_std = z_errors.mean(), z_errors.std() + 1e-6
        aux_targets["z"] = (z_errors - z_mean) / z_std
        print(f"aux z-error labels: raw mean={z_mean:.4f} std={z_std:.4f}")
    if use_yaw:
        yaw_errors = compute_yaw_error_labels(fk_env, proprio)
        yaw_mean, yaw_std = yaw_errors.mean(), yaw_errors.std() + 1e-6
        aux_targets["yaw"] = (yaw_errors - yaw_mean) / yaw_std
        print(f"aux yaw-error labels: raw mean={np.degrees(yaw_mean):.3f}deg "
              f"std={np.degrees(yaw_std):.3f}deg")
    fk_env.close()

    env = DummyVecEnv([lambda: GraspPixelEnv()])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma,
                        norm_obs_keys=["proprio"])
    env.obs_rms["proprio"].update(proprio)

    model = PPO("MultiInputPolicy", env, learning_rate=args.lr, gamma=args.gamma,
                device="cpu", seed=args.seed, verbose=0)
    policy = model.policy
    policy.set_training_mode(True)

    latent_dim_pi = policy.mlp_extractor.latent_dim_pi
    aux_heads = {name: nn.Linear(latent_dim_pi, 1) for name in aux_targets}
    # Each head gets its OWN optimizer, not a param group added to policy.optimizer -- see
    # module docstring / bc08's writeup for why (PPO.load's optimizer-state restore breaks
    # otherwise). Gradients still reach the shared trunk from the combined loss regardless.
    aux_optimizers = {name: torch.optim.Adam(head.parameters(), lr=args.lr)
                       for name, head in aux_heads.items()}

    rng = np.random.default_rng(args.seed)
    n_batches = max(n // args.batch_size, 1)
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        epoch_action_loss = 0.0
        epoch_aux_loss = {name: 0.0 for name in aux_targets}
        for b in range(n_batches):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            obs_batch = {
                "image": images[idx],
                "wrist_image": wrist_images[idx],
                "proprio": env.normalize_obs({"proprio": proprio[idx]})["proprio"],
            }
            obs_tensor, _ = policy.obs_to_tensor(obs_batch)
            action_tensor = torch.as_tensor(demo_actions[idx], dtype=torch.float32)

            features = policy.extract_features(obs_tensor, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            dist = policy._get_action_dist_from_latent(latent_pi)
            mean_actions = dist.distribution.mean
            action_loss = F.mse_loss(mean_actions, action_tensor)

            loss = action_loss
            aux_losses = {}
            for name, target_arr in aux_targets.items():
                aux_target = torch.as_tensor(target_arr[idx], dtype=torch.float32)
                aux_pred = aux_heads[name](latent_pi).squeeze(-1)
                aux_losses[name] = F.mse_loss(aux_pred, aux_target)
                loss = loss + args.aux_weight * aux_losses[name]

            policy.optimizer.zero_grad()
            for opt in aux_optimizers.values():
                opt.zero_grad()
            loss.backward()
            policy.optimizer.step()
            for opt in aux_optimizers.values():
                opt.step()
            epoch_action_loss += action_loss.item()
            for name in aux_targets:
                epoch_aux_loss[name] += aux_losses[name].item()
        aux_str = "  ".join(f"aux_{name}_mse={epoch_aux_loss[name] / n_batches:.5f}"
                             for name in aux_targets)
        print(f"epoch {epoch:3d}  action_mse={epoch_action_loss / n_batches:.5f}  {aux_str}")

    model.save(run_dir / "checkpoints" / "final")
    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    print(f"done, saved to {run_dir}/checkpoints/")


if __name__ == "__main__":
    main()
