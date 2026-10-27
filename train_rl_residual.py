"""Trains lever #3's residual-RL policy (README item 32): a fresh PPO head whose output is a
small bounded correction added to a frozen `bc21`-style base policy's own prediction (see
`residual_env.ResidualActionEnv`), rather than a raw action. Mirrors `train.py`'s conventions
(PPO + SubprocVecEnv + VecMonitor + VecNormalize, per-run args.json) with the residual wrapper
inserted around each `GraspPixelEnv` before VecMonitor/VecNormalize, same place item 31's
`FrameStackWristEnv` sits.

Trained from scratch (not resumed from any BC/RL checkpoint) -- the point of this architecture is
that even an untrained residual head starts close to the frozen base's own ~11/40 performance,
since early rollouts are dominated by the base action plus small noise, not PPO's usual
from-scratch collapse.

Usage:
    python3 train_rl_residual.py --run-name grasp24 --timesteps 200000 --base-run-dir runs/bc21
"""
import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from grasp_env import GraspPixelEnv
from residual_env import ResidualActionEnv

HERE = Path(__file__).parent


def linear_schedule(initial):
    def fn(progress_remaining):
        return progress_remaining * initial
    return fn


def make_env(seed, base_run_dir, residual_scale, base_steps):
    def _init():
        env = GraspPixelEnv()
        env = ResidualActionEnv(env, base_run_dir, residual_scale=residual_scale,
                                 base_steps=base_steps, gripper_latch_delay=-1)
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-freq", type=int, default=20000)
    ap.add_argument("--base-run-dir", type=str, default="runs/bc21",
                     help="Frozen base policy's run directory -- its final.zip/"
                          "vecnormalize_final.pkl supply the deterministic mean-action "
                          "prediction the residual is added to.")
    ap.add_argument("--base-steps", type=int, default=None,
                     help="Use an intermediate checkpoint of --base-run-dir instead of final.")
    ap.add_argument("--residual-scale", type=float, default=0.15,
                     help="Residual action magnitude bound -- the RL policy's raw output "
                          "(std=1 at init) is multiplied by this before being added to the "
                          "base action and clipped to [-1, 1].")
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    env_fns = [make_env(args.seed + i, args.base_run_dir, args.residual_scale, args.base_steps)
               for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env, info_keywords=("success",))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma,
                        norm_obs_keys=["proprio"])

    n_steps = args.n_steps
    batch_size = args.batch_size
    if (n_steps * args.n_envs) % batch_size != 0:
        batch_size = n_steps * args.n_envs
        print(f"n_steps*n_envs not divisible by batch_size, using batch_size={batch_size}")

    model = PPO(
        "MultiInputPolicy", env,
        learning_rate=linear_schedule(args.lr),
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        device="cpu",
        tensorboard_log=str(run_dir / "logs"),
        seed=args.seed,
        verbose=1,
    )

    save_freq = max(args.checkpoint_freq // args.n_envs, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)
    model.save(run_dir / "checkpoints" / "final")
    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    print(f"done, saved to {run_dir}/checkpoints/")


if __name__ == "__main__":
    main()
