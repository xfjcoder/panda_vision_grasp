"""Evaluates a trained GraspPixelEnv policy over N fixed seeds -- same random_table_xy
convention as eval_stage2_reach_grasp.py's scripted baseline, so the two numbers are directly
comparable (20/20 scripted with privileged state vs. this, pixels + proprioception only).

Usage:
    python3 eval_policy.py --run-name grasp01 --trials 20
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from grasp_env import GraspPixelEnv, EPISODE_STEPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.run_name
    policy = PPO.load(run_dir / "checkpoints" / "final")
    env = GraspPixelEnv()
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    results = []
    for i in range(args.trials):
        seed = args.base_seed + i
        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0
        info = {}
        for _ in range(EPISODE_STEPS):
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
    print(f"\nSummary over {args.trials} trials: {n_success}/{args.trials} succeeded "
          f"(scripted privileged-state baseline: 20/20)")
    env.close()


if __name__ == "__main__":
    main()
