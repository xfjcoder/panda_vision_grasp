"""Renders a chunked-BC policy's rollout (train_bc_chunked.py), same dual-view convention as
demo_stage5_rl_failure.py -- separate high-res renderer(s) for the GIF; the policy itself still
only ever sees the normal 64x64 GraspPixelEnv observation, re-queried once per CHUNK_SIZE steps.

Usage:
    python3 demo_chunked.py --run-name bc05_chunked --seed 0 --dual-view --record media/bc05_chunked.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from eval_policy_chunked import GRIP_CLOSED_THRESH, load_model, predict_chunk
from grasp_env import EPISODE_STEPS, GraspPixelEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--dual-view", action="store_true")
    ap.add_argument("--adaptive", action="store_true")
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
    render()
    info = {}
    t = 0
    while t < EPISODE_STEPS:
        chunk = predict_chunk(model, proprio_mean, proprio_std, obs)
        commit_len = chunk_size
        if args.adaptive and obs["proprio"][-1] < GRIP_CLOSED_THRESH:
            commit_len = 1
        for a in chunk[:commit_len]:
            obs, reward, terminated, truncated, info = env.step(a)
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
        # GitHub's README renderer won't animate (or sometimes won't even display) GIFs above
        # ~10MB -- half-resolution + an optimized palette keeps every rendered GIF well under
        # that regardless of episode length or dual-view width (see demo_stage5_rl_failure.py).
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
