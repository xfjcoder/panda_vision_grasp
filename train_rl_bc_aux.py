"""RL fine-tuning from a BC checkpoint, with a DAPG-style (Rajeswaran et al. 2018, "Learning
Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations") auxiliary
BC regression loss added to every PPO minibatch update, on top of item 32's std-fix.

Item 32 found and fixed one root cause of RL-fine-tune-from-BC failing (bc21's uncalibrated
log_std making on-policy rollouts pure noise), then swept the resulting std hyperparameter to a
real local peak (0.03, grasp11: 4/40) -- still well below bc21's own 11/40, and a 400k-step/20
checkpoint extended run (grasp15) found no plateau-above-peak, just noisy oscillation around the
same ceiling. The diagnosis for THAT ceiling: even with sane exploration noise, nothing in
vanilla PPO's loss pulls the policy back toward bc21's precise, demonstrated behavior once
gradient steps start moving it -- only the initial weights encode that, and they decay away
immediately. This script adds that pull back explicitly: every PPO minibatch step also samples a
batch from the same demo data bc21 was trained on (data/demos_dagger10.npz, same hold-phase
reweighting bc21 used) and adds a weighted-MSE term to the total loss, with a coefficient that
decays linearly over the fine-tune run (starts strong, lets RL take over as training progresses).

BCRegularizedPPO.train() is a near-verbatim copy of stable_baselines3==2.9.0's PPO.train() with
the one addition marked -- SB3 doesn't expose a hook for an auxiliary loss term, so this is
necessarily version-coupled; re-diff against PPO.train() if the SB3 version changes.

Usage:
    python3 train_rl_bc_aux.py --run-name grasp16 --timesteps 200000 \\
        --resume runs/bc21_stdfix003/checkpoints/final.zip \\
        --resume-vecnormalize runs/bc21_stdfix003/checkpoints/vecnormalize_final.pkl \\
        --checkpoint-freq 20000 --bc-coef-init 1.0 --bc-coef-final 0.0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from grasp_env import GraspPixelEnv
from train_bc2 import compute_hold_weights

HERE = Path(__file__).parent


def linear_schedule(initial):
    def fn(progress_remaining):
        return progress_remaining * initial
    return fn


def make_env(seed):
    def _init():
        env = GraspPixelEnv()
        env.reset(seed=seed)
        return env
    return _init


class BCRegularizedPPO(PPO):
    """PPO with an added DAPG-style auxiliary BC loss, see module docstring."""

    def _excluded_save_params(self):
        # Without this, BaseAlgorithm.save() pickles every public instance attribute (minus its
        # own exclusion list) on EVERY checkpoint/final save -- including the multi-GB
        # _bc_images/_bc_wrist_images arrays set by set_bc_data(), ballooning each checkpoint
        # from a few MB to ~5GB+ and filling the disk within a handful of CheckpointCallback
        # saves (hit directly: grasp16's first attempt filled a 469G partition from 166GB+21GB
        # of bloated checkpoints before crashing on train_rl_bc_aux.py's own first few updates).
        return super()._excluded_save_params() + [
            "_bc_images", "_bc_wrist_images", "_bc_proprio", "_bc_actions", "_bc_weights",
            "_bc_rng",
        ]

    def set_bc_data(self, demos_path, hold_weight, bc_coef_init, bc_coef_final, bc_batch_size):
        data = np.load(demos_path)
        self._bc_images = data["images"]
        self._bc_wrist_images = data["wrist_images"]
        self._bc_proprio = data["proprio"]
        self._bc_actions = data["actions"]
        self._bc_weights = compute_hold_weights(self._bc_proprio, hold_weight)
        self.bc_coef_init = bc_coef_init
        self.bc_coef_final = bc_coef_final
        self.bc_batch_size = bc_batch_size
        self._bc_rng = np.random.default_rng(self.seed if self.seed is not None else 0)
        print(f"loaded {len(self._bc_actions)} BC-regularizer transitions from {demos_path}, "
              f"hold_weight={hold_weight}, bc_coef {bc_coef_init}->{bc_coef_final}, "
              f"bc_batch_size={bc_batch_size}")

    def _bc_loss(self):
        idx = self._bc_rng.integers(0, len(self._bc_actions), size=self.bc_batch_size)
        obs_batch = {
            "image": self._bc_images[idx],
            "wrist_image": self._bc_wrist_images[idx],
            "proprio": self.env.normalize_obs({"proprio": self._bc_proprio[idx]})["proprio"],
        }
        obs_tensor, _ = self.policy.obs_to_tensor(obs_batch)
        action_tensor = th.as_tensor(self._bc_actions[idx], dtype=th.float32)
        dist = self.policy.get_distribution(obs_tensor)
        mean_actions = dist.distribution.mean
        per_sample_mse = F.mse_loss(mean_actions, action_tensor, reduction="none").mean(dim=1)
        weight_tensor = th.as_tensor(self._bc_weights[idx], dtype=th.float32)
        return (per_sample_mse * weight_tensor).mean()

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # --- added: linearly-decaying auxiliary BC loss coefficient, same schedule mechanism
        # the base class already uses for lr/clip_range (progress_remaining: 1.0 at the start of
        # THIS fine-tune run, 0.0 at the end).
        bc_coef = self.bc_coef_final + self._current_progress_remaining * (
            self.bc_coef_init - self.bc_coef_final)

        entropy_losses = []
        pg_losses, value_losses, bc_losses = [], [], []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # --- added: auxiliary BC term, pulls every update back toward bc21's demos.
                bc_loss = self._bc_loss()
                bc_losses.append(bc_loss.item())

                loss = (policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                        + bc_coef * bc_loss)

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                            self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/bc_coef", bc_coef)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, required=True,
                     help="Required (unlike train.py) -- this script only makes sense as a "
                          "fine-tune of an existing BC/RL checkpoint.")
    ap.add_argument("--resume-vecnormalize", type=str, required=True)
    ap.add_argument("--checkpoint-freq", type=int, default=None)
    ap.add_argument("--demos", type=str, default="data/demos_dagger10.npz",
                     help="Demo file the auxiliary BC loss samples from -- default matches "
                          "what bc21 itself was trained on.")
    ap.add_argument("--hold-weight", type=float, default=1.5,
                     help="Hold-phase reweighting for the auxiliary BC loss, same mechanism as "
                          "train_bc2.py. Default 1.5 matches bc21's own training.")
    ap.add_argument("--bc-coef-init", type=float, default=1.0,
                     help="Auxiliary BC loss coefficient at the START of this fine-tune run.")
    ap.add_argument("--bc-coef-final", type=float, default=0.0,
                     help="Auxiliary BC loss coefficient at the END of this fine-tune run -- "
                          "linearly interpolated with --bc-coef-init over --timesteps.")
    ap.add_argument("--bc-batch-size", type=int, default=256,
                     help="Demo transitions sampled per PPO minibatch update for the "
                          "auxiliary BC loss.")
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    env_fns = [make_env(args.seed + i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env, info_keywords=("success",))
    env = VecNormalize.load(args.resume_vecnormalize, env)
    env.training = True

    n_steps = args.n_steps
    batch_size = args.batch_size
    if (n_steps * args.n_envs) % batch_size != 0:
        batch_size = n_steps * args.n_envs
        print(f"n_steps*n_envs not divisible by batch_size, using batch_size={batch_size}")

    model = BCRegularizedPPO.load(args.resume, env=env, device="cpu", verbose=1,
                                   learning_rate=linear_schedule(args.lr), seed=args.seed)
    print(f"resumed from {args.resume}")
    model.set_bc_data(args.demos, args.hold_weight, args.bc_coef_init, args.bc_coef_final,
                       args.bc_batch_size)

    checkpoint_cb = None
    if args.checkpoint_freq is not None:
        save_freq = max(args.checkpoint_freq // args.n_envs, 1)
    else:
        save_freq = max(args.timesteps // (args.n_envs * 10), 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)
    model.save(run_dir / "checkpoints" / "final")
    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    print(f"done, saved to {run_dir}/checkpoints/")


if __name__ == "__main__":
    main()
