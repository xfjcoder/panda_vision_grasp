"""Trains a PPO policy for GraspPixelEnv (pixels + proprioception -> action, no privileged
state). Mirrors barrett_wam/go1_rl_walk's own conventions (PPO + SubprocVecEnv + VecNormalize +
VecMonitor, per-run args.json), with two things specific to a Dict{image, proprio} observation:
"MultiInputPolicy" (stable_baselines3's CNN+MLP combined extractor for Dict obs spaces) instead
of "MlpPolicy", and `norm_obs_keys=["proprio"]` so VecNormalize's running mean/std normalization
is applied only to proprioception, not to the raw 0-255 image (which the CNN extractor already
assumes and internally rescales itself -- normalizing it again would feed the wrong distribution
in).

Usage:
    python3 train.py --run-name grasp01 --timesteps 500000 --n-envs 8
"""
import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from grasp_env import GraspPixelEnv

HERE = Path(__file__).parent


def linear_schedule(initial):
    def fn(progress_remaining):
        return progress_remaining * initial
    return fn


def make_env(seed):
    def _init():
        env = GraspPixelEnv()
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--resume-vecnormalize", type=str, default=None)
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    env_fns = [make_env(args.seed + i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)
    if args.resume_vecnormalize:
        env = VecNormalize.load(args.resume_vecnormalize, env)
        env.training = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma,
                            norm_obs_keys=["proprio"])

    n_steps = args.n_steps
    batch_size = args.batch_size
    if (n_steps * args.n_envs) % batch_size != 0:
        batch_size = n_steps * args.n_envs
        print(f"n_steps*n_envs not divisible by batch_size, using batch_size={batch_size}")

    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu",
                          learning_rate=linear_schedule(args.lr))
        print(f"resumed from {args.resume}")
    else:
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

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.timesteps // (args.n_envs * 10), 1),
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
