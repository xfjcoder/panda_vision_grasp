"""Renders a trained RL policy's rollout from the same workspace_cam view documented in the
README's Stage 5 failure writeups (grasp01-grasp04, all 0/20) -- visual evidence for what "the
policy reaches toward the object but never completes a clean grasp+lift" actually looks like,
not just the eval numbers.

Uses a separate high-res renderer for the recorded frames; the policy itself still only ever
sees the normal 64x64 GraspPixelEnv observation, exactly as during training/eval.

Usage:
    python3 demo_stage5_rl_failure.py --run-name grasp04 --seed 0 --record media/stage5_rl_failure.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from grasp_env import EPISODE_STEPS, GraspPixelEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.run_name
    policy = PPO.load(run_dir / "checkpoints" / "final")
    env = GraspPixelEnv()
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    record_renderer = mj.Renderer(env.model, height=480, width=640)
    frames = []

    def render():
        record_renderer.update_scene(env.data, camera="workspace_cam")
        frames.append(record_renderer.render().copy())

    obs, _ = env.reset(seed=args.seed)
    render()
    info = {}
    for t in range(EPISODE_STEPS):
        obs_norm = norm_env.normalize_obs(obs)
        action, _ = policy.predict(obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        render()
        if terminated or truncated:
            break

    print(f"run={args.run_name} seed={args.seed} rise={info['rise']:.4f}m "
          f"drift={info['drift']:.4f}m success={info['success']}")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=50, loop=0)
        print(f"Saved {len(frames)} frames to {args.record}")

    record_renderer.close()
    env.close()


if __name__ == "__main__":
    main()
