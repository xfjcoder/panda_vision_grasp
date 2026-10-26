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

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from grasp_env import GraspPixelEnv

HERE = Path(__file__).parent

# Curriculum stages: object-position range scale (0 = fixed at TABLE_XY_RANGE's center, 1 = full
# range). grasp03 (ent_coef=0.01, no curriculum) still got 0/20 -- the policy never attempted a
# grasp at all, with rise staying at noise level across every trial, consistent with the full
# random-position range from step 0 leaving too little early reward gradient to find the
# descend-grasp-lift motion in the first place. Start narrow, widen once it's working.
SCALE_SCHEDULE = [0.15, 0.4, 0.7, 1.0]


def linear_schedule(initial):
    def fn(progress_remaining):
        return progress_remaining * initial
    return fn


def make_env(seed, curriculum_scale=1.0):
    def _init():
        env = GraspPixelEnv(curriculum_scale=curriculum_scale)
        env.reset(seed=seed)
        return env
    return _init


class CurriculumCallback(BaseCallback):
    """Widens every env's object-position range once the rolling success rate (recorded via
    VecMonitor's info_keywords=("success",)) clears `threshold`."""

    def __init__(self, threshold, check_freq, verbose=1):
        super().__init__(verbose)
        self.threshold = threshold
        self.check_freq = check_freq
        self.stage = 0

    def _on_step(self):
        if self.n_calls % self.check_freq != 0 or self.stage >= len(SCALE_SCHEDULE) - 1:
            return True
        if len(self.model.ep_info_buffer) < 20:
            return True
        success_rate = float(np.mean([ep["success"] for ep in self.model.ep_info_buffer]))
        if success_rate >= self.threshold:
            self.stage += 1
            scale = SCALE_SCHEDULE[self.stage]
            self.training_env.env_method("set_curriculum_scale", scale)
            if self.verbose:
                print(f"curriculum: success_rate={success_rate:.2f} over last "
                      f"{len(self.model.ep_info_buffer)} eps -> stage {self.stage}, scale={scale}")
        return True


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
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--curriculum-threshold", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--resume-vecnormalize", type=str, default=None)
    ap.add_argument("--checkpoint-freq", type=int, default=None,
                     help="Save a checkpoint (model + vecnormalize) every this many raw "
                          "timesteps, regardless of --timesteps. Default None reproduces the "
                          "old behavior (10 checkpoints total, spacing scales with --timesteps) "
                          "-- set this explicitly (e.g. 20000) for a long run where you want a "
                          "fixed, timesteps-independent cadence to evaluate a plateau curve.")
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    initial_scale = SCALE_SCHEDULE[0] if args.curriculum else 1.0
    env_fns = [make_env(args.seed + i, curriculum_scale=initial_scale) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env, info_keywords=("success",))
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
        # verbose is a saved constructor arg -- override explicitly, or a checkpoint made with
        # verbose=0 (e.g. train_bc.py's BC phase) silently carries that into every future
        # resume, with no progress tables printed and no indication training isn't just hung.
        model = PPO.load(args.resume, env=env, device="cpu", verbose=1,
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

    if args.checkpoint_freq is not None:
        save_freq = max(args.checkpoint_freq // args.n_envs, 1)
    else:
        save_freq = max(args.timesteps // (args.n_envs * 10), 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )
    callbacks = [checkpoint_cb]
    if args.curriculum:
        callbacks.append(CurriculumCallback(threshold=args.curriculum_threshold, check_freq=n_steps))

    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks))
    model.save(run_dir / "checkpoints" / "final")
    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    print(f"done, saved to {run_dir}/checkpoints/")


if __name__ == "__main__":
    main()
