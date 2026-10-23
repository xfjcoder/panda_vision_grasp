"""Behavior cloning with an auxiliary vertical-grasp-height loss, targeting the bug the user
spotted in bc06's gallery GIFs: the gripper closes ~0.049m above the object's center (near its
top tip) instead of at its center, frozen across two full cumulative DAgger rounds (bc04, bc06
both ~0.049m; see README item 19). The object's target grasp height is a scene CONSTANT (table
height + half the object's fixed size, never randomized -- only x/y varies), so the auxiliary
label needs no privileged per-episode data: for every demo transition, forward-kinematics the
proprio's own joint angles to get the gripper site's current world z, subtract the constant
target z, and that is the label. Cheap: ~0.002ms/call via mj_kinematics (not mj_forward -- no
physics needed, just the kinematic tree), so the whole 72k-transition dataset's labels are
precomputed once before training, no per-batch mujoco calls needed.

A verified-strong signal motivated trying this over another blanket DAgger round: a dedicated
probe (see README item 20) found wrist_cam's apparent cube size is highly, monotonically
sensitive to exactly this vertical offset (361px at the correct height down to 28-75px at 3-8cm
too high, in the actual 64x64 training resolution) while workspace_cam carries ~none (constant
~1215px regardless of height) -- the depth information to fix this is available, this auxiliary
task is a direct attempt to force the shared visual encoder to actually extract and use it,
rather than leaving it an incidental byproduct of the main per-step action-regression loss.

Reuses train_bc.py's exact MSE-on-action-mean recipe and PPO MultiInputPolicy network -- the
only addition is a small linear head (`aux_head`) reading the actor's latent features
(`policy.mlp_extractor.forward_actor(...)`, the same tensor `get_distribution` computes
internally but doesn't expose) to predict the normalized z-error, added to the same optimizer via
`add_param_group` so one `backward()`/`step()` updates both the action head and the shared
trunk from the combined loss. Discarded at eval time -- eval_policy.py loads the saved PPO model
unchanged, exactly as for every other bc0N checkpoint.

Usage:
    python3 train_bc_auxz.py --run-name bc08 --demos data/demos_dagger4.npz --aux-weight 1.0
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

from grasp_env import GraspPixelEnv, N_ARM

HERE = Path(__file__).parent


def compute_z_error_labels(env, proprio):
    """FK each transition's own joint angles (arm dims only -- gripper/object qpos irrelevant to
    the arm's forward kinematics) to the gripper site's world z, relative to the constant target
    grasp height. No physics step needed, mj_kinematics alone is sufficient and ~500x faster."""
    target_z = env._table_top_z() + env.obj_half_height
    z_errors = np.empty(len(proprio), dtype=np.float32)
    for i in range(len(proprio)):
        env.data.qpos[:N_ARM] = proprio[i, :N_ARM]
        mj.mj_kinematics(env.model, env.data)
        z_errors[i] = env.data.site_xpos[env.site_id][2] - target_z
    return z_errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--demos", type=str, default="data/demos.npz")
    ap.add_argument("--aux-weight", type=float, default=1.0,
                     help="Weight on the normalized auxiliary z-error MSE, relative to the main "
                          "action MSE (both losses are summed, not averaged).")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    data = np.load(args.demos)
    images, wrist_images = data["images"], data["wrist_images"]
    proprio, demo_actions = data["proprio"], data["actions"]
    n = len(demo_actions)
    print(f"loaded {n} demo transitions from {args.demos}")

    fk_env = GraspPixelEnv()
    z_errors = compute_z_error_labels(fk_env, proprio)
    fk_env.close()
    z_err_mean, z_err_std = z_errors.mean(), z_errors.std() + 1e-6
    z_err_norm = (z_errors - z_err_mean) / z_err_std
    print(f"aux z-error labels: raw mean={z_err_mean:.4f} std={z_err_std:.4f} "
          f"(normalized to mean 0 std 1 for training)")

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
    aux_head = nn.Linear(latent_dim_pi, 1)
    # A SEPARATE optimizer for aux_head, not an added param group on policy.optimizer -- PPO.load
    # restores optimizer state with exact_match=True, which fails if the saved optimizer has more
    # param groups than a freshly-constructed one (confirmed: "different number of parameter
    # groups" on eval-time load). Gradients still flow into the shared trunk from the combined
    # loss regardless of which optimizer object steps which params, so this loses nothing.
    aux_optimizer = torch.optim.Adam(aux_head.parameters(), lr=args.lr)

    rng = np.random.default_rng(args.seed)
    n_batches = max(n // args.batch_size, 1)
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        epoch_action_loss = 0.0
        epoch_aux_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            obs_batch = {
                "image": images[idx],
                "wrist_image": wrist_images[idx],
                "proprio": env.normalize_obs({"proprio": proprio[idx]})["proprio"],
            }
            obs_tensor, _ = policy.obs_to_tensor(obs_batch)
            action_tensor = torch.as_tensor(demo_actions[idx], dtype=torch.float32)
            aux_target = torch.as_tensor(z_err_norm[idx], dtype=torch.float32)

            features = policy.extract_features(obs_tensor, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            dist = policy._get_action_dist_from_latent(latent_pi)
            mean_actions = dist.distribution.mean
            action_loss = F.mse_loss(mean_actions, action_tensor)

            aux_pred = aux_head(latent_pi).squeeze(-1)
            aux_loss = F.mse_loss(aux_pred, aux_target)

            loss = action_loss + args.aux_weight * aux_loss

            policy.optimizer.zero_grad()
            aux_optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()
            aux_optimizer.step()
            epoch_action_loss += action_loss.item()
            epoch_aux_loss += aux_loss.item()
        print(f"epoch {epoch:3d}  action_mse={epoch_action_loss / n_batches:.5f}  "
              f"aux_mse={epoch_aux_loss / n_batches:.5f}")

    model.save(run_dir / "checkpoints" / "final")
    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    print(f"done, saved to {run_dir}/checkpoints/")


if __name__ == "__main__":
    main()
