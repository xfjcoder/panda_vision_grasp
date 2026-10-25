"""Residual-action BC trainer (README item 31) -- tests the first of two untried structural
directions flagged after item 30's plateau confirmation (the other is frame-stacking the wrist
camera, see train_bc_framestack.py).

Mechanism: compute the per-timestep MEAN action across training episodes (shape
(EPISODE_STEPS, N_ARM+1)), a cheap, privileged-info-free stand-in for "the typical thing to do
at this point in the episode." Train the network to regress the RESIDUAL
(demo_action - mean_action[t]) instead of the raw action. At rollout time (eval_policy.py,
demo_stage5_rl_failure.py, collect_dagger.py, via residual_utils.load_mean_action) the policy's
output gets mean_action[t] added back before being clipped and stepped.

Why this should help, per item 29's root-cause finding: aggregate held-out MSE is dominated by
the easy majority of transitions (steady holding/approach, nearly identical across episodes),
which is now almost entirely absorbed by the fixed mean-action baseline for free. Every bit of
the network's remaining gradient signal goes toward the harder, state-dependent corrective
minority -- a structural version of what hold-phase reweighting (items 26/27) already validated
by hand-tuned sample weight.

Otherwise identical to train_bc2.py: same net-arch/hold-weight/val-split/early-stop machinery,
same CLI surface plus nothing new (the residual transform is unconditional, not a flag, since
it's cheap and the whole point of this script).

Usage:
    python3 train_bc_residual.py --run-name bc29 --demos data/demos_dagger10.npz --hold-weight 1.5
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from grasp_env import EPISODE_STEPS, GraspPixelEnv
from train_bc2 import compute_hold_weights, episode_val_split

HERE = Path(__file__).parent


def compute_mean_action(demo_actions, train_idx):
    """Per-timestep mean action across TRAIN episodes only (val rows excluded, same leakage
    discipline as env.obs_rms's train-only update in train_bc2.py)."""
    n_train_eps = len(train_idx) // EPISODE_STEPS
    train_actions = demo_actions[train_idx].reshape(n_train_eps, EPISODE_STEPS, -1)
    return train_actions.mean(axis=0)  # (EPISODE_STEPS, N_ARM+1)


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
    images, wrist_images = data["images"], data["wrist_images"]
    proprio, demo_actions = data["proprio"], data["actions"]
    n = len(demo_actions)
    print(f"loaded {n} demo transitions from {args.demos}, net_arch={net_arch}, "
          f"hold_weight={args.hold_weight}, RESIDUAL action target")

    weights = compute_hold_weights(proprio, args.hold_weight)
    train_idx, val_idx = episode_val_split(n, args.val_frac, args.seed)
    print(f"train/val split: {len(train_idx)} train rows ({len(train_idx)//EPISODE_STEPS} eps), "
          f"{len(val_idx)} val rows ({len(val_idx)//EPISODE_STEPS} eps)")

    mean_action = compute_mean_action(demo_actions, train_idx)
    np.save(run_dir / "checkpoints" / "mean_action.npy", mean_action)
    # Residual target = actual action minus the per-timestep mean, broadcast back out over every
    # episode (rows are contiguous EPISODE_STEPS-blocks per episode, same convention as
    # compute_hold_weights/episode_val_split).
    n_eps_total = n // EPISODE_STEPS
    mean_tiled = np.tile(mean_action, (n_eps_total, 1))
    residual_actions = (demo_actions - mean_tiled).astype(np.float32)
    print(f"mean |mean_action|={np.abs(mean_action).mean():.4f}  "
          f"mean |residual|={np.abs(residual_actions).mean():.4f}  "
          f"mean |raw action|={np.abs(demo_actions).mean():.4f}")

    env = DummyVecEnv([lambda: GraspPixelEnv()])
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
        action_tensor = torch.as_tensor(residual_actions[idx], dtype=torch.float32)
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
        print(f"epoch {epoch:3d}  train_weighted_residual_mse={epoch_loss / n_batches:.5f}  "
              f"val_residual_mse={val_mse:.5f}{'  (new best)' if improved else ''}")

    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    model.save(run_dir / "checkpoints" / "last_epoch")
    # No early-stop option here -- item 29 found it backfires for this task, and item 30's
    # fix (default off) is the validated behavior; always save the full-length model as final.
    (run_dir / "checkpoints" / "final.zip").write_bytes(
        (run_dir / "checkpoints" / "last_epoch.zip").read_bytes())
    print(f"(best val_residual_mse={best_val_mse:.5f} was at epoch {best_epoch}, saved anyway as "
          f"the full {args.epochs}-epoch model per item 29's early-stop finding)")
    print(f"done, saved to {run_dir}/checkpoints/ (including mean_action.npy for "
          f"eval_policy.py/demo_stage5_rl_failure.py/collect_dagger.py to add back at rollout)")


if __name__ == "__main__":
    main()
