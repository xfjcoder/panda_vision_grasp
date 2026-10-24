"""Behavior cloning with hold-phase-weighted action MSE, targeting the lift-phase lateral drift
bottleneck found once alignment (items 19-25) plateaued near expert precision: z-offset and yaw
error kept improving every DAgger round through `bc17`, but success/reward peaked two rounds
earlier (`bc15`) and declined -- confirmed via per-step trace (grip/object xy every 5 steps from
the close moment) that the remaining failure is still the item-15-era pattern, a small per-step
bias in the network's own output, integrated open-loop over the ~90-step hold phase -- just now
inconsistent instead of universal (some trials self-correct mid-hold, e.g. `bc15` seed 6's
success; others still accumulate monotonically to episode end, e.g. `bc17` seed 8).

Every per-step BC run so far (`bc01`-`bc17`) weights every one of the EPISODE_STEPS transitions
in an episode equally in the MSE, even though hold-phase steps (~75% of the episode) are exactly
where this specific, still-unsolved bias compounds -- the approach-phase steps that make up the
other ~25% are already handled well (that's what DAgger's rounds have been fixing, per item 25).
This reweights the SAME single action-MSE loss used by every prior bc0N run (no second
objective, no new head -- avoiding the multi-task interference items 21-23 found when stacking
auxiliary losses on this trunk) so hold-phase transitions count --hold-weight times as much.

Hold-phase detection reuses the exact has_opened/close_start convention used by every diagnostic
in this project (gripper_qpos crosses above 0.9 then below 0.75): weight 1.0 before that episode's
close_start, --hold-weight after, normalized so the dataset-wide mean weight is exactly 1.0 (keeps
the overall loss scale, and therefore the effective gradient magnitude at a given lr, comparable
to the unweighted baseline -- only the RELATIVE emphasis across transitions changes).

Usage:
    python3 train_bc_holdweight.py --run-name bc18 --demos data/demos_dagger10.npz --hold-weight 4.0
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from grasp_env import EPISODE_STEPS, GraspPixelEnv

HERE = Path(__file__).parent


def compute_hold_weights(proprio, hold_weight):
    n = len(proprio)
    n_eps = n // EPISODE_STEPS
    assert n_eps * EPISODE_STEPS == n, "expected every episode to contribute EPISODE_STEPS rows"

    gripper_qpos = proprio[:, -1]
    weights = np.ones(n, dtype=np.float32)
    for ep in range(n_eps):
        s, e = ep * EPISODE_STEPS, (ep + 1) * EPISODE_STEPS
        has_opened = False
        close_start = None
        for i in range(s, e):
            if gripper_qpos[i] > 0.9:
                has_opened = True
            if has_opened and close_start is None and gripper_qpos[i] < 0.75:
                close_start = i
                break
        if close_start is not None:
            weights[close_start:e] = hold_weight
    weights /= weights.mean()  # keep overall loss scale comparable to the unweighted baseline
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--demos", type=str, default="data/demos.npz")
    ap.add_argument("--hold-weight", type=float, default=4.0,
                     help="Relative MSE weight for post-close-moment (hold-phase) transitions, "
                          "before dataset-wide normalization to mean 1.0.")
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

    weights = compute_hold_weights(proprio, args.hold_weight)
    print(f"hold-phase weighting: {int((weights > 1.0).sum())}/{n} transitions up-weighted "
          f"(raw hold_weight={args.hold_weight}, normalized mean={weights.mean():.4f})")

    env = DummyVecEnv([lambda: GraspPixelEnv()])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma,
                        norm_obs_keys=["proprio"])
    env.obs_rms["proprio"].update(proprio)

    model = PPO("MultiInputPolicy", env, learning_rate=args.lr, gamma=args.gamma,
                device="cpu", seed=args.seed, verbose=0)
    policy = model.policy
    policy.set_training_mode(True)

    rng = np.random.default_rng(args.seed)
    n_batches = max(n // args.batch_size, 1)
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            obs_batch = {
                "image": images[idx],
                "wrist_image": wrist_images[idx],
                "proprio": env.normalize_obs({"proprio": proprio[idx]})["proprio"],
            }
            obs_tensor, _ = policy.obs_to_tensor(obs_batch)
            action_tensor = torch.as_tensor(demo_actions[idx], dtype=torch.float32)
            weight_tensor = torch.as_tensor(weights[idx], dtype=torch.float32)

            dist = policy.get_distribution(obs_tensor)
            mean_actions = dist.distribution.mean
            per_sample_mse = F.mse_loss(mean_actions, action_tensor, reduction="none").mean(dim=1)
            loss = (per_sample_mse * weight_tensor).mean()

            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch:3d}  weighted_mse={epoch_loss / n_batches:.5f}")

    model.save(run_dir / "checkpoints" / "final")
    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    print(f"done, saved to {run_dir}/checkpoints/")


if __name__ == "__main__":
    main()
