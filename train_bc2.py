"""Second-generation BC trainer, built to close three gaps that have been constant across every
one of this project's 20+ `bc0N` checkpoints (`bc01`-`bc20`), never varied once:

1. **Network capacity.** Every checkpoint used PPO's RL-sized default policy MLP: two CNNs each
   embed one 64x64 camera to 256 dims, concatenated with proprio into a 527-dim vector, then
   squeezed through `Linear(527->64)->Tanh->Linear(64->64)->Tanh->Linear(64->8)` -- a 64-unit
   bottleneck downstream of two full image encoders. Every multi-task finding in this project
   (item 23's auxiliary-loss interference, items 26/27's hold-phase-reweighting tradeoffs) is
   consistent with this being a real capacity ceiling, never directly tested because the
   architecture itself was never varied -- only data, loss terms, and sample weights were.
   `--net-arch` (comma-separated ints, e.g. `256,256`) controls the actor MLP's hidden sizes via
   `policy_kwargs`; default `64,64` reproduces the old (untested) architecture exactly, so this
   script is also a drop-in, verifiable replacement for `train_bc.py`/`train_bc_holdweight.py`.
2. **No train/val split, ever.** Every prior run's only signal was its own TRAINING loss curve,
   which can't distinguish "still fitting" from "memorizing" -- the real signal (rollout
   success) only showed up after the fact, in a separate eval script. Holds out `--val-frac` of
   EPISODES (not individual transitions -- adjacent frames within an episode are highly
   correlated, so a transition-level split would leak) at a fixed random seed, reports both
   train and held-out val MSE every epoch.
3. **No hyperparameter sweep, ever.** `epochs=20, batch_size=256, lr=3e-4` have been every
   single run's defaults since `bc01`. Exposed as the same CLI flags as before, now easy to vary
   deliberately instead of by omission.

Keeps `bc19`'s hold-phase action-MSE reweighting (`--hold-weight`, see `train_bc_holdweight.py`'s
module docstring for the mechanism and item 26/27's results) as an option, defaulting to 1.0
(uniform, i.e. plain BC) so architecture/hyperparameter changes can be tested in isolation first.

Usage:
    python3 train_bc2.py --run-name bc21 --demos data/demos_dagger10.npz --hold-weight 1.5
    python3 train_bc2.py --run-name bc22 --demos data/demos_dagger10.npz --hold-weight 1.5 --net-arch 256,256
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
    if hold_weight == 1.0:
        return np.ones(len(proprio), dtype=np.float32)
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
    weights /= weights.mean()
    return weights


def episode_val_split(n, val_frac, seed):
    """Random EPISODE-level split (not transition-level, to avoid leaking correlated adjacent
    frames across train/val). Returns (train_row_idx, val_row_idx)."""
    n_eps = n // EPISODE_STEPS
    assert n_eps * EPISODE_STEPS == n, "expected every episode to contribute EPISODE_STEPS rows"
    rng = np.random.default_rng(seed)
    ep_order = rng.permutation(n_eps)
    n_val_eps = max(1, int(n_eps * val_frac))
    val_eps = set(ep_order[:n_val_eps].tolist())
    train_rows, val_rows = [], []
    for ep in range(n_eps):
        s, e = ep * EPISODE_STEPS, (ep + 1) * EPISODE_STEPS
        (val_rows if ep in val_eps else train_rows).extend(range(s, e))
    return np.array(train_rows), np.array(val_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--demos", type=str, default="data/demos.npz")
    ap.add_argument("--net-arch", type=str, default="64,64",
                     help="Comma-separated actor-MLP hidden layer sizes, e.g. '256,256'. "
                          "Default '64,64' exactly reproduces every prior bc0N checkpoint's "
                          "(untested) architecture.")
    ap.add_argument("--hold-weight", type=float, default=1.0,
                     help="Hold-phase action-MSE reweighting, see train_bc_holdweight.py. "
                          "1.0 = uniform/plain BC.")
    ap.add_argument("--val-frac", type=float, default=0.1,
                     help="Fraction of EPISODES (not transitions) held out for validation.")
    ap.add_argument("--early-stop", action="store_true",
                     help="Save the best-held-out-val-MSE epoch as checkpoints/final instead "
                          "of the last epoch. DEFAULT OFF, and deliberately so: tested this "
                          "directly (bc23 vs bc21, bc24 vs bc22) and it backfired badly in both "
                          "cases (bc23: 0/40 success vs bc21's 11/40) -- aggregate held-out MSE "
                          "bottoms out around epoch 1-2 here because the easy majority of "
                          "transitions (steady holding) gets fit almost immediately, while the "
                          "rare, high-variance, corrective transitions that actual task success "
                          "depends on keep costing MORE aggregate MSE as they're fit further, "
                          "exactly because they're legitimately harder to generalize, not noise. "
                          "Kept as an opt-in flag for diagnostic use (the train/val curve itself "
                          "is still a real, useful signal, see the printed numbers every epoch) "
                          "-- just don't use it to pick the deployed checkpoint for this task.")
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
          f"hold_weight={args.hold_weight}")

    weights = compute_hold_weights(proprio, args.hold_weight)
    train_idx, val_idx = episode_val_split(n, args.val_frac, args.seed)
    print(f"train/val split: {len(train_idx)} train rows ({len(train_idx)//EPISODE_STEPS} eps), "
          f"{len(val_idx)} val rows ({len(val_idx)//EPISODE_STEPS} eps)")

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
            if args.early_stop:
                model.save(run_dir / "checkpoints" / "best")
        print(f"epoch {epoch:3d}  train_weighted_mse={epoch_loss / n_batches:.5f}  "
              f"val_mse={val_mse:.5f}{'  (new best)' if improved else ''}")

    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    model.save(run_dir / "checkpoints" / "last_epoch")
    if args.early_stop and best_epoch >= 0:
        # "final" is the canonical name eval_policy.py/demo_stage5_rl_failure.py load -- point
        # it at the EARLY-STOPPED best-val checkpoint, not necessarily the last epoch trained.
        (run_dir / "checkpoints" / "final.zip").write_bytes(
            (run_dir / "checkpoints" / "best.zip").read_bytes())
        print(f"best val_mse={best_val_mse:.5f} at epoch {best_epoch} -- saved as "
              f"checkpoints/final (checkpoints/last_epoch is the full-{args.epochs}-epoch "
              f"model, kept for comparison)")
    else:
        # Default: "final" = the full-length-trained model, same convention as every prior
        # bc0N checkpoint -- validated better than early stopping for this task (see --early-stop).
        (run_dir / "checkpoints" / "final.zip").write_bytes(
            (run_dir / "checkpoints" / "last_epoch.zip").read_bytes())
        if best_epoch >= 0:
            print(f"(best val_mse={best_val_mse:.5f} was at epoch {best_epoch}, but "
                  f"--early-stop was not passed -- checkpoints/final is the full "
                  f"{args.epochs}-epoch model, matching every prior bc0N checkpoint)")
    print(f"done, saved to {run_dir}/checkpoints/")


if __name__ == "__main__":
    main()
