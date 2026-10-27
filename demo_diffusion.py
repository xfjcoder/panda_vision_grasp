"""Renders a diffusion-BC policy's rollout (train_bc_diffusion.py), same dual-view convention as
demo_chunked.py/demo_stage5_rl_failure.py.

Usage:
    python3 demo_diffusion.py --run-name bc31_diffusion --seed 0 --dual-view --record media/bc31_diffusion.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
import torch
from PIL import Image

from eval_policy_diffusion import load_model, predict_chunk
from grasp_env import EPISODE_STEPS, GraspPixelEnv
from ik import N_ARM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--dual-view", action="store_true")
    ap.add_argument("--gripper-latch-delay", type=int, default=5)
    args = ap.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.run_name
    model, chunk_size, proprio_mean, proprio_std = load_model(run_dir)
    env = GraspPixelEnv()

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
    generator = torch.Generator().manual_seed(args.seed)
    render()
    info = {}
    has_opened = False
    close_start = None
    t = 0
    while t < EPISODE_STEPS:
        chunk = predict_chunk(model, proprio_mean, proprio_std, obs, generator)
        for action in chunk:
            if args.gripper_latch_delay >= 0:
                g = obs["proprio"][-1]
                if g > 0.9:
                    has_opened = True
                if has_opened and close_start is None and g < 0.75:
                    close_start = t
                if close_start is not None and t >= close_start + args.gripper_latch_delay:
                    action[N_ARM] = -1.0
            obs, reward, terminated, truncated, info = env.step(action)
            render()
            t += 1
            if terminated or truncated or t >= EPISODE_STEPS:
                break
        if terminated or truncated:
            break

    print(f"run={args.run_name} seed={args.seed} rise={info['rise']:.4f}m "
          f"drift={info['drift']:.4f}m success={info['success']}")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
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
