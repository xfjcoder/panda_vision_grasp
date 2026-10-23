"""Evaluates a chunked-BC checkpoint (train_bc_chunked.py) over N fixed seeds -- same convention
as eval_policy.py (random_table_xy, seeds 0..N-1) so the numbers are directly comparable.

Re-queries the model only once per CHUNK_SIZE steps (a "re-plan"), executing the predicted
action block open-loop in between -- that's the whole point of action chunking, so this eval
loop is NOT a drop-in replacement for eval_policy.py's per-step predict/step loop.

--adaptive: proprio-keyed chunk boundaries, tried as a 2nd iteration on chunking after
bc05_chunked's fixed-10-step-stride trace revealed exactly where it breaks. Traced the worst
bc05_chunked trial (seed 0) and found the failure wasn't stale-chunk execution after some
external event -- it was the model's OWN freshly-predicted chunk (observed once the gripper
was already closed around the object) choosing to reopen partway through its own 10 steps,
and the policy had no way to notice and react before all 10 were spent. The one phase where an
open-loop commitment is genuinely risky is exactly the one proprio can detect for free: once
the gripper has started closing (`proprio`'s last element, `gripper_qpos` normalized to
[0, 1], drops below GRIP_CLOSED_THRESH), a bad chunk can release a held object with no
intervention until the next re-plan. Before that point (still descending, nothing gripped yet)
there's nothing irreversible to protect against, so the full chunk stays safe to commit to.
--adaptive uses the SAME already-trained bc05_chunked checkpoint -- committing to only 1 step
of the predicted chunk (instead of all CHUNK_SIZE) whenever the gripper reads closed, falling
back to per-step responsiveness exactly where the earlier failure happened, while keeping the
full chunk (and its smoothing benefit) everywhere else. No retraining involved -- this isolates
the execution-policy change as the only variable.

Usage:
    python3 eval_policy_chunked.py --run-name bc05_chunked --trials 20
    python3 eval_policy_chunked.py --run-name bc05_chunked --trials 20 --adaptive
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from grasp_env import EPISODE_STEPS, GraspPixelEnv
from train_bc_chunked import ChunkedBCNet

GRIP_CLOSED_THRESH = 0.7  # normalized gripper_qpos (1.0=open, 0.0=closed); below this the
                           # gripper has started closing -- commit length drops to 1


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
    ap.add_argument("--adaptive", action="store_true",
                     help="Commit only 1 step of the predicted chunk once the gripper reads "
                          "closed (proprio-keyed re-planning), full chunk otherwise.")
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
            commit_len = chunk_size
            if args.adaptive and obs["proprio"][-1] < GRIP_CLOSED_THRESH:
                commit_len = 1
            for a in chunk[:commit_len]:
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
