"""Renders a trained RL policy's rollout from the same workspace_cam view documented in the
README's Stage 5 failure writeups (grasp01-grasp04, all 0/20) -- visual evidence for what "the
policy reaches toward the object but never completes a clean grasp+lift" actually looks like,
not just the eval numbers.

Uses separate high-res renderer(s) for the recorded frames; the policy itself still only ever
sees the normal 64x64 GraspPixelEnv observation from workspace_cam, exactly as during
training/eval -- wrist_cam (--dual-view) is purely a visualization add-on, see NOTICE.md.

Usage:
    python3 demo_stage5_rl_failure.py --run-name grasp04 --seed 0 --record media/stage5_rl_failure.gif
    python3 demo_stage5_rl_failure.py --run-name grasp08 --seed 0 --dual-view --record media/grasp08_dual.gif
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
    ap.add_argument("--dual-view", action="store_true",
                     help="Also render wrist_cam (eye-in-hand) side by side with workspace_cam. "
                          "Visualization only -- the policy never sees this view.")
    args = ap.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.run_name
    policy = PPO.load(run_dir / "checkpoints" / "final")
    env = GraspPixelEnv()
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    workspace_renderer = mj.Renderer(env.model, height=480, width=640)
    wrist_renderer = mj.Renderer(env.model, height=480, width=480) if args.dual_view else None
    frames = []

    def render():
        workspace_renderer.update_scene(env.data, camera="workspace_cam")
        frame = workspace_renderer.render().copy()
        if wrist_renderer is not None:
            wrist_renderer.update_scene(env.data, camera="wrist_cam")
            wrist_frame = wrist_renderer.render().copy()
            frame = np.concatenate([frame, wrist_frame], axis=1)
        frames.append(frame)

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
        # GitHub's README renderer won't animate (or sometimes won't even display) GIFs above
        # ~10MB -- half-resolution + a shared/optimized palette keeps every rendered GIF well
        # under that regardless of episode length or dual-view width.
        half_res = [f[::2, ::2] for f in frames]
        imgs = [Image.fromarray(f) for f in half_res]
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=50, loop=0,
                     optimize=True)
        print(f"Saved {len(frames)} frames to {args.record}")

    workspace_renderer.close()
    if wrist_renderer is not None:
        wrist_renderer.close()
    env.close()


if __name__ == "__main__":
    main()
