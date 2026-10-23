"""Action-chunking BC, as an alternative to per-step DAgger (bc02/bc04) for the same lift-phase
drift problem.

Why try this instead of a 3rd DAgger round: every per-step BC/DAgger attempt so far
(bc01-bc04) shares one structural property -- the network makes a fresh, independent
prediction at EVERY one of the 120 control steps, so any small per-step bias has 120
independent chances to nudge the trajectory off course before it compounds into visible
drift (confirmed directly: bc03's drift grows almost linearly for ~90 consecutive steps once
the gripper closes). Action chunking (ACT-style) instead predicts a whole BLOCK of
CHUNK_SIZE future actions from a single observation, then executes that block open-loop before
re-observing. Two effects, both relevant here: (1) only EPISODE_STEPS/CHUNK_SIZE independent
predictions per episode instead of EPISODE_STEPS, directly cutting the number of places error
can enter; (2) each chunk is one coherent regression target (a whole short trajectory segment),
which tends to be smoother than 10 independently-regressed single steps. The tradeoff this
accepts: less visual feedback during the chunk, since the network commits to CHUNK_SIZE steps
before it can react to anything new in the image.

Deliberately NOT built on SB3's PPO/MultiInputPolicy (unlike train_bc.py) -- a multi-step
action-chunk head doesn't fit ActorCriticPolicy's single-action-distribution design, so this is
a small standalone CNN+MLP regressor trained with plain MSE, independent of any RL fine-tuning
path. That also means a chunked checkpoint can't be loaded by train.py/eval_policy.py --
eval_policy_chunked.py and demo_chunked.py are its dedicated eval/demo counterparts.

Usage:
    python3 train_bc_chunked.py --run-name bc05_chunked --demos data/demos_dualcam.npz --epochs 40
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grasp_env import EPISODE_STEPS
from ik import N_ARM

HERE = Path(__file__).parent
PROPRIO_DIM = 2 * N_ARM + 1
ACTION_DIM = N_ARM + 1
CHUNK_SIZE = 10  # 0.5s at CONTROL_DT=0.05 -- 12 re-plans/episode instead of 120


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, 3, 64, 64)).shape[1]
        self.linear = nn.Sequential(nn.Linear(flat_dim, out_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.conv(x))


class ChunkedBCNet(nn.Module):
    """One obs (image, wrist_image, proprio) -> a (CHUNK_SIZE, ACTION_DIM) action block,
    squashed to [-1, 1] with tanh to match the env's action bounds."""

    def __init__(self, chunk_size=CHUNK_SIZE, img_embed=128, proprio_embed=32, hidden=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.image_enc = ImageEncoder(img_embed)
        self.wrist_enc = ImageEncoder(img_embed)
        self.proprio_enc = nn.Sequential(nn.Linear(PROPRIO_DIM, proprio_embed), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(img_embed * 2 + proprio_embed, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, chunk_size * ACTION_DIM),
        )

    def forward(self, image_u8, wrist_image_u8, proprio):
        image = image_u8.float().permute(0, 3, 1, 2) / 255.0
        wrist = wrist_image_u8.float().permute(0, 3, 1, 2) / 255.0
        z = torch.cat([self.image_enc(image), self.wrist_enc(wrist), self.proprio_enc(proprio)],
                       dim=-1)
        out = torch.tanh(self.head(z))
        return out.view(-1, self.chunk_size, ACTION_DIM)


def build_chunks(images, wrist_images, proprio, actions, chunk_size):
    n_eps = len(actions) // EPISODE_STEPS
    n_chunks_per_ep = EPISODE_STEPS // chunk_size
    starts = np.array([ep * EPISODE_STEPS + c * chunk_size
                        for ep in range(n_eps) for c in range(n_chunks_per_ep)])
    action_chunks = np.stack([actions[s:s + chunk_size] for s in starts])
    return images[starts], wrist_images[starts], proprio[starts], action_chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--demos", type=str, default="data/demos_dualcam.npz")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    data = np.load(args.demos)
    images, wrist_images = data["images"], data["wrist_images"]
    proprio, actions = data["proprio"], data["actions"]
    images, wrist_images, proprio, action_chunks = build_chunks(
        images, wrist_images, proprio, actions, args.chunk_size)
    n = len(action_chunks)
    print(f"loaded {len(actions)} demo transitions from {args.demos}, "
          f"built {n} chunks of size {args.chunk_size}")

    proprio_mean = proprio.mean(axis=0)
    proprio_std = proprio.std(axis=0) + 1e-6

    torch.manual_seed(args.seed)
    model = ChunkedBCNet(chunk_size=args.chunk_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rng = np.random.default_rng(args.seed)
    n_batches = max(n // args.batch_size, 1)
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            image_t = torch.as_tensor(images[idx])
            wrist_t = torch.as_tensor(wrist_images[idx])
            proprio_t = torch.as_tensor((proprio[idx] - proprio_mean) / proprio_std,
                                         dtype=torch.float32)
            target = torch.as_tensor(action_chunks[idx], dtype=torch.float32)

            pred = model(image_t, wrist_t, proprio_t)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch:3d}  mse={epoch_loss / n_batches:.5f}")

    torch.save({
        "model_state": model.state_dict(),
        "chunk_size": args.chunk_size,
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
    }, run_dir / "checkpoints" / "chunked_final.pt")
    print(f"done, saved to {run_dir}/checkpoints/chunked_final.pt")


if __name__ == "__main__":
    main()
