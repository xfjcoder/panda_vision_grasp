"""Renders a lever #3 residual-RL checkpoint's rollout -- same workspace_cam/wrist_cam dual-view
convention as demo_stage5_rl_failure.py.

Usage:
    python3 demo_residual.py --run-name grasp24 --seed 0 --record media/grasp24_residual.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from grasp_env import EPISODE_STEPS, GraspPixelEnv
from residual_env import ResidualActionEnv

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--base-run-dir", type=str, default="runs/bc21")
    ap.add_argument("--base-steps", type=int, default=None)
    ap.add_argument("--residual-scale", type=float, default=0.15)
    ap.add_argument("--gripper-latch-delay", type=int, default=5)
    ap.add_argument("--dual-view", action="store_true")
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
    residual_env = ResidualActionEnv(env, args.base_run_dir, residual_scale=args.residual_scale,
                                      base_steps=args.base_steps,
                                      gripper_latch_delay=args.gripper_latch_delay)
    vec_env = DummyVecEnv([lambda: residual_env])
    norm_env = VecNormalize.load(str(run_dir / "checkpoints" / vecnorm_name), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    sim = env  # GraspPixelEnv itself, not wrapped -- ResidualActionEnv doesn't shadow .model/.data
    workspace_renderer = mj.Renderer(sim.model, height=480, width=640)
    wrist_renderer = mj.Renderer(sim.model, height=480, width=480) if args.dual_view else None
    frames = []

    def render():
        workspace_renderer.update_scene(sim.data, camera="workspace_cam")
        frame = workspace_renderer.render().copy()
        if wrist_renderer is not None:
            wrist_renderer.update_scene(sim.data, camera="wrist_cam")
            wrist_frame = wrist_renderer.render().copy()
            frame = np.concatenate([frame, wrist_frame], axis=1)
        frames.append(frame)

    obs, _ = residual_env.reset(seed=args.seed)
    render()
    info = {}
    for t in range(EPISODE_STEPS):
        obs_norm = norm_env.normalize_obs(obs)
        action, _ = policy.predict(obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = residual_env.step(action)
        render()
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
    residual_env.close()


if __name__ == "__main__":
    main()
