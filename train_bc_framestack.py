"""Frame-stacked-wrist-camera BC trainer (README item 31) -- tests the second of two untried
structural directions flagged after item 30's plateau confirmation (the other is residual-action
learning, see train_bc_residual.py).

Builds the PPO policy against `FrameStackWristEnv(GraspPixelEnv())` so its wrist-camera CNN
takes a 64x64x6 input (previous + current frame stacked on the channel axis) instead of
64x64x3, giving the CNN visual access to slip DIRECTION (not just slip position) for the first
time -- see framestack_env.py's module docstring for the full argument. Training data for the
offline regression comes from `stack_wrist_offline`, which reconstructs the exact same stacking
from an already-collected demos npz's `wrist_images` array -- no re-collection needed.

Otherwise identical to train_bc2.py: same net-arch/hold-weight/val-split machinery, no
early-stop option (item 29 found it backfires for this task; item 30 defaulted it off).

Usage:
    python3 train_bc_framestack.py --run-name bc30 --demos data/demos_dagger10.npz --hold-weight 1.5
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from framestack_env import FrameStackWristEnv, stack_wrist_offline
from grasp_env import EPISODE_STEPS, GraspPixelEnv
from train_bc2 import compute_hold_weights, episode_val_split

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--demos", type=str, default="data/demos.npz")
    ap.add_argument("--net-arch", type=str, default="64,64")
    ap.add_argument("--hold-weight", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net_arch = [int(x) for x in args.net_arch.split(",")]

    run_dir = HERE / "runs" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    data = np.load(args.demos)
    images, proprio, demo_actions = data["images"], data["proprio"], data["actions"]
    n = len(demo_actions)
    print(f"loaded {n} demo transitions from {args.demos}, net_arch={net_arch}, "
          f"hold_weight={args.hold_weight}, FRAME-STACKED wrist camera (64x64x6)")
    wrist_images = stack_wrist_offline(data["wrist_images"], EPISODE_STEPS)
    print(f"stacked wrist_images: {data['wrist_images'].shape} -> {wrist_images.shape}")

    weights = compute_hold_weights(proprio, args.hold_weight)
    train_idx, val_idx = episode_val_split(n, args.val_frac, args.seed)
    print(f"train/val split: {len(train_idx)} train rows ({len(train_idx)//EPISODE_STEPS} eps), "
          f"{len(val_idx)} val rows ({len(val_idx)//EPISODE_STEPS} eps)")

    env = DummyVecEnv([lambda: FrameStackWristEnv(GraspPixelEnv())])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma,
                        norm_obs_keys=["proprio"])
    env.obs_rms["proprio"].update(proprio[train_idx])

    model = PPO("MultiInputPolicy", env, learning_rate=args.lr, gamma=args.gamma,
                device="cpu", seed=args.seed, verbose=0,
                policy_kwargs=dict(net_arch=net_arch))
    policy = model.policy
    policy.set_training_mode(True)

    def batch_mse(idx, weighted=False):
        obs_batch = {
            "image": images[idx],
            "wrist_image": wrist_images[idx],
            "proprio": env.normalize_obs({"proprio": proprio[idx]})["proprio"],
        }
        obs_tensor, _ = policy.obs_to_tensor(obs_batch)
        action_tensor = torch.as_tensor(demo_actions[idx], dtype=torch.float32)
        dist = policy.get_distribution(obs_tensor)
        mean_actions = dist.distribution.mean
        if weighted:
            per_sample_mse = F.mse_loss(mean_actions, action_tensor, reduction="none").mean(dim=1)
            weight_tensor = torch.as_tensor(weights[idx], dtype=torch.float32)
            return (per_sample_mse * weight_tensor).mean()
        return F.mse_loss(mean_actions, action_tensor)

    rng = np.random.default_rng(args.seed)
    n_train = len(train_idx)
    n_batches = max(n_train // args.batch_size, 1)
    best_val_mse, best_epoch = float("inf"), -1
    for epoch in range(args.epochs):
        perm = train_idx[rng.permutation(n_train)]
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            loss = batch_mse(idx, weighted=True)
            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()
            epoch_loss += loss.item()

        policy.set_training_mode(False)
        if len(val_idx):
            val_losses, val_counts = [], []
            with torch.no_grad():
                for b in range(0, len(val_idx), args.batch_size):
                    vidx = val_idx[b:b + args.batch_size]
                    val_losses.append(batch_mse(vidx, weighted=False).item() * len(vidx))
                    val_counts.append(len(vidx))
            val_mse = sum(val_losses) / sum(val_counts)
        else:
            val_mse = float("nan")
        policy.set_training_mode(True)

        improved = len(val_idx) and val_mse < best_val_mse
        if improved:
            best_val_mse, best_epoch = val_mse, epoch
        print(f"epoch {epoch:3d}  train_weighted_mse={epoch_loss / n_batches:.5f}  "
              f"val_mse={val_mse:.5f}{'  (new best)' if improved else ''}")

    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    model.save(run_dir / "checkpoints" / "last_epoch")
    (run_dir / "checkpoints" / "final.zip").write_bytes(
        (run_dir / "checkpoints" / "last_epoch.zip").read_bytes())
    print(f"(best val_mse={best_val_mse:.5f} was at epoch {best_epoch}, saved anyway as the "
          f"full {args.epochs}-epoch model per item 29's early-stop finding)")
    print(f"done, saved to {run_dir}/checkpoints/")


if __name__ == "__main__":
    main()
