"""Evaluates a diffusion-BC checkpoint (train_bc_diffusion.py) over N fixed seeds -- same
40-trial convention as eval_policy.py, so numbers are directly comparable to champion `bc21`.

Re-queries the model only once per `chunk_size` steps (a "re-plan"), executing the sampled
action chunk open-loop in between -- same loop structure as eval_policy_chunked.py, now with a
DDPM draw instead of an MSE-regressed block. `chunk_size=1` checkpoints (the original,
diagnosed-broken per-step variant -- see train_bc_diffusion.py's module docstring) fall back to
re-planning every step, same as eval_policy.py.

Usage:
    python3 eval_policy_diffusion.py --run-name bc32_diffusion_chunked --trials 40
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from grasp_env import EPISODE_STEPS, GraspPixelEnv
from ik import N_ARM
from train_bc_diffusion import DiffusionBCNet


def load_model(run_dir):
    ckpt = torch.load(Path(run_dir) / "checkpoints" / "diffusion_final.pt", weights_only=False)
    model = DiffusionBCNet(chunk_size=ckpt.get("chunk_size", 1))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, model.chunk_size, ckpt["proprio_mean"], ckpt["proprio_std"]


def predict_chunk(model, proprio_mean, proprio_std, obs, generator):
    with torch.no_grad():
        image_t = torch.as_tensor(obs["image"]).unsqueeze(0)
        wrist_t = torch.as_tensor(obs["wrist_image"]).unsqueeze(0)
        proprio_norm = (obs["proprio"] - proprio_mean) / proprio_std
        proprio_t = torch.as_tensor(proprio_norm, dtype=torch.float32).unsqueeze(0)
        cond = model.encode(image_t, wrist_t, proprio_t)
        chunk = model.sample(cond, generator=generator)[0].numpy()
    return chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--gripper-latch-delay", type=int, default=5,
                     help="Same post-hoc gripper-command hysteresis as eval_policy.py (item 28), "
                          "default ON (5) to match every other checkpoint's reported number -- "
                          "pass a negative value to disable.")
    args = ap.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.run_name
    model, chunk_size, proprio_mean, proprio_std = load_model(run_dir)
    env = GraspPixelEnv()

    results = []
    for i in range(args.trials):
        seed = args.base_seed + i
        obs, _ = env.reset(seed=seed)
        generator = torch.Generator().manual_seed(seed)
        ep_reward = 0.0
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
                ep_reward += reward
                t += 1
                if terminated or truncated or t >= EPISODE_STEPS:
                    break
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
