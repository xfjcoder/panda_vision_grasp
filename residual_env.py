"""Lever #3 from README item 32's borrowable-methods list: "a state-conditioned residual base
policy instead of our time-indexed mean_action.npy" (item 31's residual-action BC variant used a
FIXED per-timestep mean, blind to the actual state; this uses champion `bc21`'s own trained
network -- a genuine function of the full image+proprio observation -- as the base instead).

`ResidualActionEnv` wraps `GraspPixelEnv` so the RL policy never outputs a raw action: every
step, a frozen copy of `bc21` predicts its own (deterministic, mean-action) prediction for the
current observation, and the RL policy's output is treated as a small, bounded correction added
on top before clipping to the env's real [-1, 1] action range. This is standard residual RL
(Silver et al. 2018, Johannink et al. 2019) -- the architectural analogue of lever #1's loss-based
anchoring (DAPG): instead of a loss term pulling gradients back toward BC, the action space itself
is centered on BC, so even an untrained RL head starts near `bc21`'s own ~11/40 performance rather
than PPO's usual from-scratch collapse (items 9-14's `grasp09`-`grasp14` all started at 0-4/40).

`gripper_latch_delay` is OFF by default (-1) -- unlike every eval/demo script in this project,
which defaults it to 5 (item 28) -- because this wrapper can also be the TRAINING env, and baking
an eval-only post-hoc hack into training dynamics would be a different, undiscussed design
choice. Eval/demo callers explicitly opt in with `gripper_latch_delay=5` to stay comparable to
every other checkpoint's reported number; `train_rl_residual.py` leaves it off so the policy has
to earn its own gripper timing through reward, same as every other RL training script here.
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from grasp_env import GraspPixelEnv
from ik import N_ARM


class ResidualActionEnv(gym.Wrapper):
    def __init__(self, env, base_run_dir, residual_scale=0.15, base_steps=None,
                 gripper_latch_delay=-1):
        super().__init__(env)
        run_dir = Path(base_run_dir)
        stem = f"ppo_{base_steps}_steps" if base_steps else "final"
        vecnorm_name = f"ppo_vecnormalize_{base_steps}_steps.pkl" if base_steps else \
            "vecnormalize_final.pkl"
        self.base_policy = PPO.load(run_dir / "checkpoints" / stem, device="cpu")
        base_vec_env = DummyVecEnv([lambda: GraspPixelEnv()])
        self.base_norm = VecNormalize.load(str(run_dir / "checkpoints" / vecnorm_name),
                                            base_vec_env)
        self.base_norm.training = False
        self.base_norm.norm_reward = False
        self.residual_scale = residual_scale
        self.gripper_latch_delay = gripper_latch_delay
        self._last_obs = None
        self._has_opened = False
        self._close_start = None
        self._t = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._has_opened = False
        self._close_start = None
        self._t = 0
        return obs, info

    def step(self, action):
        obs_norm = self.base_norm.normalize_obs(self._last_obs)
        base_action, _ = self.base_policy.predict(obs_norm, deterministic=True)
        final_action = np.clip(base_action + self.residual_scale * np.asarray(action), -1.0, 1.0)
        if self.gripper_latch_delay >= 0:
            g = self._last_obs["proprio"][-1]
            if g > 0.9:
                self._has_opened = True
            if self._has_opened and self._close_start is None and g < 0.75:
                self._close_start = self._t
            if self._close_start is not None and self._t >= self._close_start + \
                    self.gripper_latch_delay:
                final_action[N_ARM] = -1.0
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        self._last_obs = obs
        self._t += 1
        info["base_action"] = base_action
        info["residual_action"] = np.asarray(action)
        return obs, reward, terminated, truncated, info
