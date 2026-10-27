"""Evaluates a lever #3 residual-RL checkpoint (train_rl_residual.py) over N fixed seeds -- same
40-trial convention as eval_policy.py, so numbers are directly comparable to champion `bc21` and
every `bc_coef`-aux checkpoint in README item 32.

Usage:
    python3 eval_policy_residual.py --run-name grasp24 --trials 40
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from grasp_env import EPISODE_STEPS, GraspPixelEnv
from residual_env import ResidualActionEnv

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--base-run-dir", type=str, default="runs/bc21")
    ap.add_argument("--base-steps", type=int, default=None)
    ap.add_argument("--residual-scale", type=float, default=0.15)
    ap.add_argument("--gripper-latch-delay", type=int, default=5,
                     help="Default ON (5), matching every other checkpoint's reported number "
                          "in this project (item 28) -- applied to the FINAL blended action "
                          "inside ResidualActionEnv, not the raw residual output.")
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_name
    if args.steps is not None:
        model_stem = f"ppo_{args.steps}_steps"
        vecnorm_name = f"ppo_vecnormalize_{args.steps}_steps.pkl"
    else:
        model_stem = "final"
        vecnorm_name = "vecnormalize_final.pkl"
    policy = PPO.load(run_dir / "checkpoints" / model_stem)

    env = GraspPixelEnv()
    env = ResidualActionEnv(env, args.base_run_dir, residual_scale=args.residual_scale,
                             base_steps=args.base_steps,
                             gripper_latch_delay=args.gripper_latch_delay)
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(str(run_dir / "checkpoints" / vecnorm_name), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    results = []
    for i in range(args.trials):
        seed = args.base_seed + i
        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0
        info = {}
        for t in range(EPISODE_STEPS):
            obs_norm = norm_env.normalize_obs(obs)
            action, _ = policy.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        results.append(dict(seed=seed, success=info["success"], rise=info["rise"],
                             drift=info["drift"], ep_reward=ep_reward))
        print(f"trial {i:2d} seed={seed:4d} rise={info['rise']:.4f}m drift={info['drift']:.4f}m "
              f"ep_reward={ep_reward:.2f} success={info['success']}")

    n_success = sum(r["success"] for r in results)
    mean_reward = float(np.mean([r["ep_reward"] for r in results]))
    mean_rise = float(np.mean([r["rise"] for r in results]))
    print(f"\nSummary over {args.trials} trials: {n_success}/{args.trials} succeeded "
          f"(champion bc21: 11/40)")
    print(f"SUMMARY_PARSEABLE run={args.run_name} trials={args.trials} success={n_success} "
          f"mean_reward={mean_reward:.4f} mean_rise={mean_rise:.4f}")
    env.close()


if __name__ == "__main__":
    main()
