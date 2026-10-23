"""Evaluates a chunked-BC checkpoint (train_bc_chunked.py) over N fixed seeds -- same convention
as eval_policy.py (random_table_xy, seeds 0..N-1) so the numbers are directly comparable.

Re-queries the model only once per CHUNK_SIZE steps (a "re-plan"), executing the predicted
action block open-loop in between -- that's the whole point of action chunking, so this eval
loop is NOT a drop-in replacement for eval_policy.py's per-step predict/step loop.

Usage:
    python3 eval_policy_chunked.py --run-name bc05_chunked --trials 20
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from grasp_env import EPISODE_STEPS, GraspPixelEnv
from train_bc_chunked import ChunkedBCNet


def load_model(run_dir):
    ckpt = torch.load(Path(run_dir) / "checkpoints" / "chunked_final.pt", weights_only=False)
    model = ChunkedBCNet(chunk_size=ckpt["chunk_size"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["chunk_size"], ckpt["proprio_mean"], ckpt["proprio_std"]


def predict_chunk(model, proprio_mean, proprio_std, obs):
    with torch.no_grad():
        image_t = torch.as_tensor(obs["image"]).unsqueeze(0)
        wrist_t = torch.as_tensor(obs["wrist_image"]).unsqueeze(0)
        proprio_norm = (obs["proprio"] - proprio_mean) / proprio_std
        proprio_t = torch.as_tensor(proprio_norm, dtype=torch.float32).unsqueeze(0)
        chunk = model(image_t, wrist_t, proprio_t)[0].numpy()
    return np.clip(chunk, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.run_name
    model, chunk_size, proprio_mean, proprio_std = load_model(run_dir)
    env = GraspPixelEnv()

    results = []
    for i in range(args.trials):
        seed = args.base_seed + i
        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0
        info = {}
        t = 0
        while t < EPISODE_STEPS:
            chunk = predict_chunk(model, proprio_mean, proprio_std, obs)
            for a in chunk:
                obs, reward, terminated, truncated, info = env.step(a)
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
    print(f"\nSummary over {args.trials} trials: {n_success}/{args.trials} succeeded "
          f"(scripted privileged-state baseline: 20/20)")
    env.close()


if __name__ == "__main__":
    main()
