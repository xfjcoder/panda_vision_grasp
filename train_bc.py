"""Behavior-cloning pretraining on top of a PPO policy's own network, using the demo dataset
collect_demos.py produced from Stage 2's scripted controller. Deliberately reuses the exact same
PPO model object (architecture, optimizer) that train.py would build for RL fine-tuning -- so the
weights this script produces can be loaded straight into `train.py --resume` afterward, same as
resuming any other checkpoint, rather than needing a separate network/weight-transfer step.

Supervises the policy's action-distribution MEAN (via `policy.get_distribution(...).distribution
.mean`) against the demonstrated action at each timestep -- the direct analog of what
`policy.predict(obs, deterministic=True)` returns, via plain MSE.

Usage:
    python3 train_bc.py --run-name bc01 --demos data/demos.npz --epochs 20
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from grasp_env import GraspPixelEnv

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--demos", type=str, default="data/demos.npz")
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

    env = DummyVecEnv([lambda: GraspPixelEnv()])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma,
                        norm_obs_keys=["proprio"])
    # Fit normalization stats directly from the demo dataset's proprio, rather than from live
    # env steps -- same statistics train.py's --resume-vecnormalize will then continue updating
    # from, so RL fine-tuning starts from a consistent normalization, not a fresh one.
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

            dist = policy.get_distribution(obs_tensor)
            mean_actions = dist.distribution.mean
            loss = F.mse_loss(mean_actions, action_tensor)

            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch:3d}  mse={epoch_loss / n_batches:.5f}")

    model.save(run_dir / "checkpoints" / "final")
    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    print(f"done, saved to {run_dir}/checkpoints/")


if __name__ == "__main__":
    main()
