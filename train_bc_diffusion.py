"""Lever #4 from README item 32's borrowable-methods list: a generative BC action head
(Diffusion-Policy-style) that models the action DISTRIBUTION via a conditional DDPM, instead of
regressing a single mean action with MSE. The mode-averaging failure items 29/31 kept finding
(multimodal demos blurred into an average that's valid for neither mode) is structural to
MSE-against-a-point-estimate, not fixable by reweighting/capacity/architecture tweaks within that
framework (item 31's residual-action variant, `bc29`, hit the identical overfitting shape as
plain MSE despite a different target parameterization) -- a generative head is the structurally
different fix the literature (Diffusion Policy, Chi et al. 2023) proposes.

**`--chunk-size` (default 10, matching `train_bc_chunked.py`'s `CHUNK_SIZE`), not 1.** A first
attempt at `chunk_size=1` (resample a fresh single-step action from independent noise every
control step) scored 1/40 even with item 28's gripper-latch hack applied -- traced directly to
~54% sign-flip rate on the ARM channels step to step (vs. this project's ~2% scripted baseline),
because every control step draws an entirely independent noise sequence through the reverse
diffusion chain, with nothing enforcing temporal coherence between one step's sample and the
next. This is why every real Diffusion-Policy-style system pairs a generative head with action
CHUNKING: one diffusion draw produces a whole coherent block of future actions, executed
open-loop, so within-chunk jitter is eliminated by construction (one sample, not CHUNK_SIZE
independent ones) -- same chunking mechanism as `train_bc_chunked.py`, now generating the chunk
via DDPM instead of MSE regression.

Deliberately NOT built on SB3's PPO/MultiInputPolicy, same reasoning as train_bc_chunked.py -- a
diffusion head doesn't fit ActorCriticPolicy's single-Gaussian design. Standalone CNN+MLP
encoder (`ImageEncoder` reused directly from train_bc_chunked.py) feeding a small denoising MLP
conditioned on (encoder output, noisy flattened action chunk, diffusion timestep). Keeps `bc21`'s
own hold-phase action-MSE reweighting (`compute_hold_weights`, reused directly from
train_bc2.py) so the generative-head change is isolated as the only lever, same "change one
thing" discipline as every other item-31/32 comparison.

N_DIFFUSION_STEPS=50 is a standard small DDPM schedule -- cheap enough that sampling (T forward
passes through the small denoiser MLP, reusing one CNN encode per re-plan) doesn't meaningfully
slow down a 40-trial eval sweep even with chunking's much lower re-plan rate.

Usage:
    python3 train_bc_diffusion.py --run-name bc32_diffusion_chunked --demos data/demos_dagger10.npz --hold-weight 1.5
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grasp_env import EPISODE_STEPS
from ik import N_ARM
from train_bc2 import compute_hold_weights, episode_val_split
from train_bc_chunked import ImageEncoder, build_chunks

HERE = Path(__file__).parent
PROPRIO_DIM = 2 * N_ARM + 1
ACTION_DIM = N_ARM + 1
N_DIFFUSION_STEPS = 50
TIME_EMBED_DIM = 16


def chunk_episode_val_split(n_chunks_total, n_chunks_per_ep, val_frac, seed):
    """Same episode-level (not chunk-level) train/val split as train_bc2.py's
    episode_val_split, just parameterized by chunks-per-episode instead of
    transitions-per-episode -- build_chunks(random_starts=False) lays chunks out in the same
    fixed (episode, chunk-index) order, so dividing by n_chunks_per_ep recovers the episode."""
    n_eps = n_chunks_total // n_chunks_per_ep
    rng = np.random.default_rng(seed)
    ep_order = rng.permutation(n_eps)
    n_val_eps = max(1, int(n_eps * val_frac))
    val_eps = set(ep_order[:n_val_eps].tolist())
    train_rows, val_rows = [], []
    for ep in range(n_eps):
        s, e = ep * n_chunks_per_ep, (ep + 1) * n_chunks_per_ep
        (val_rows if ep in val_eps else train_rows).extend(range(s, e))
    return np.array(train_rows), np.array(val_rows)


class DiffusionBCNet(nn.Module):
    def __init__(self, chunk_size=1, img_embed=128, proprio_embed=32, hidden=256,
                 t_steps=N_DIFFUSION_STEPS):
        super().__init__()
        self.t_steps = t_steps
        self.chunk_size = chunk_size
        self.out_dim = chunk_size * ACTION_DIM
        self.image_enc = ImageEncoder(img_embed)
        self.wrist_enc = ImageEncoder(img_embed)
        self.proprio_enc = nn.Sequential(nn.Linear(PROPRIO_DIM, proprio_embed), nn.ReLU())
        self.time_embed = nn.Embedding(t_steps, TIME_EMBED_DIM)
        cond_dim = img_embed * 2 + proprio_embed
        self.denoiser = nn.Sequential(
            nn.Linear(cond_dim + self.out_dim + TIME_EMBED_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.out_dim),
        )

        betas = torch.linspace(1e-4, 0.02, t_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def encode(self, image_u8, wrist_image_u8, proprio):
        image = image_u8.float().permute(0, 3, 1, 2) / 255.0
        wrist = wrist_image_u8.float().permute(0, 3, 1, 2) / 255.0
        return torch.cat([self.image_enc(image), self.wrist_enc(wrist),
                           self.proprio_enc(proprio)], dim=-1)

    def predict_noise(self, cond, action_noisy, t):
        z = torch.cat([cond, action_noisy, self.time_embed(t)], dim=-1)
        return self.denoiser(z)

    def diffusion_loss(self, cond, action0, weights=None):
        b = action0.shape[0]
        action0 = action0.reshape(b, self.out_dim)
        t = torch.randint(0, self.t_steps, (b,), device=action0.device)
        noise = torch.randn_like(action0)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
        action_noisy = torch.sqrt(alpha_bar_t) * action0 + torch.sqrt(1 - alpha_bar_t) * noise
        noise_pred = self.predict_noise(cond, action_noisy, t)
        per_sample_mse = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=1)
        if weights is not None:
            return (per_sample_mse * weights).mean()
        return per_sample_mse.mean()

    @torch.no_grad()
    def sample(self, cond, generator=None):
        b = cond.shape[0]
        a = torch.randn(b, self.out_dim, generator=generator)
        for t in reversed(range(self.t_steps)):
            t_batch = torch.full((b,), t, dtype=torch.long)
            eps_hat = self.predict_noise(cond, a, t_batch)
            alpha_t, alpha_bar_t, beta_t = self.alphas[t], self.alpha_bars[t], self.betas[t]
            mean = (a - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_hat) / torch.sqrt(alpha_t)
            if t > 0:
                noise = torch.randn(b, self.out_dim, generator=generator)
                a = mean + torch.sqrt(beta_t) * noise
            else:
                a = mean
        a = torch.clamp(a, -1.0, 1.0)
        return a.view(b, self.chunk_size, ACTION_DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--demos", type=str, default="data/demos_dagger10.npz")
    ap.add_argument("--chunk-size", type=int, default=10,
                     help="Actions per diffusion draw, executed open-loop before re-planning. "
                          "1 reproduces the original (broken -- see module docstring) per-step "
                          "behavior; 10 matches train_bc_chunked.py's default.")
    ap.add_argument("--chunk-stride", type=int, default=None,
                     help="Spacing between chunk starts. Default (None) = chunk_size, i.e. the "
                          "original non-overlapping grid (21.6k distinct chunk-start images for "
                          "demos_dagger10.npz) that bc32/bc33 found too sparse for the CNN "
                          "encoder to learn real conditioning on, collapsing to near-constant "
                          "output regardless of input. --chunk-stride 1 instead lays a chunk "
                          "start at every transition (216k rows, chunk_size/stride as many as "
                          "the default), restoring full image coverage while keeping each "
                          "training target a coherent chunk_size-step block.")
    ap.add_argument("--hold-weight", type=float, default=1.5)
    ap.add_argument("--val-frac", type=float, default=0.1)
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
    n = len(actions)
    print(f"loaded {n} demo transitions from {args.demos}, hold_weight={args.hold_weight}, "
          f"chunk_size={args.chunk_size}, t_steps={N_DIFFUSION_STEPS}")

    weights_all = compute_hold_weights(proprio, args.hold_weight)
    if args.chunk_size > 1:
        stride = args.chunk_stride or args.chunk_size
        n_eps = n // EPISODE_STEPS
        offsets = np.arange(0, EPISODE_STEPS - args.chunk_size + 1, stride)
        n_chunks_per_ep = len(offsets)
        # weights_all is per-transition -- re-index to one weight per chunk (its start
        # transition) with the same stride build_chunks uses, so the two stay aligned.
        starts = np.array([ep * EPISODE_STEPS + o for ep in range(n_eps) for o in offsets])
        weights_all = weights_all[starts]
        images, wrist_images, proprio, actions = build_chunks(
            images, wrist_images, proprio, actions, args.chunk_size, random_starts=False,
            stride=stride)
        n = len(actions)
        print(f"built {n} chunks of size {args.chunk_size}, stride={stride}")
        train_idx, val_idx = chunk_episode_val_split(n, n_chunks_per_ep, args.val_frac, args.seed)
    else:
        train_idx, val_idx = episode_val_split(n, args.val_frac, args.seed)
    print(f"train/val split: {len(train_idx)} train rows, {len(val_idx)} val rows")

    proprio_mean = proprio.mean(axis=0)
    proprio_std = proprio.std(axis=0) + 1e-6

    torch.manual_seed(args.seed)
    model = DiffusionBCNet(chunk_size=args.chunk_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def batch_loss(idx, weighted):
        image_t = torch.as_tensor(images[idx])
        wrist_t = torch.as_tensor(wrist_images[idx])
        proprio_t = torch.as_tensor((proprio[idx] - proprio_mean) / proprio_std,
                                     dtype=torch.float32)
        action_t = torch.as_tensor(actions[idx], dtype=torch.float32)
        cond = model.encode(image_t, wrist_t, proprio_t)
        if weighted:
            weight_t = torch.as_tensor(weights_all[idx], dtype=torch.float32)
            return model.diffusion_loss(cond, action_t, weights=weight_t)
        return model.diffusion_loss(cond, action_t)

    rng = np.random.default_rng(args.seed)
    n_train = len(train_idx)
    n_batches = max(n_train // args.batch_size, 1)
    for epoch in range(args.epochs):
        perm = train_idx[rng.permutation(n_train)]
        epoch_loss = 0.0
        model.train()
        for b in range(n_batches):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            loss = batch_loss(idx, weighted=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        if len(val_idx):
            val_losses, val_counts = [], []
            with torch.no_grad():
                for b in range(0, len(val_idx), args.batch_size):
                    vidx = val_idx[b:b + args.batch_size]
                    val_losses.append(batch_loss(vidx, weighted=False).item() * len(vidx))
                    val_counts.append(len(vidx))
            val_mse = sum(val_losses) / sum(val_counts)
        else:
            val_mse = float("nan")
        print(f"epoch {epoch:3d}  train_weighted_noise_mse={epoch_loss / n_batches:.5f}  "
              f"val_noise_mse={val_mse:.5f}")

    torch.save({
        "model_state": model.state_dict(),
        "chunk_size": args.chunk_size,
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
    }, run_dir / "checkpoints" / "diffusion_final.pt")
    print(f"done, saved to {run_dir}/checkpoints/diffusion_final.pt")


if __name__ == "__main__":
    main()
