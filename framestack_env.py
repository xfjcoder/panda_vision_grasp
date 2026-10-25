"""Frame-stacked wrist-camera wrapper (README item 31's second structural direction, alongside
train_bc_residual.py): stacks the previous and current wrist_image frame along the channel axis
(64x64x6 instead of 64x64x3).

Why: grasp_env.py's proprio already includes joint velocities, so the arm's OWN motion isn't
blind to a single-frame policy. But how the GRASPED OBJECT is slipping relative to the gripper
-- exactly what `rise`/`drift` measure -- is purely visual, and a single still frame can show
slip POSITION but never slip DIRECTION (no visual motion cue at all). Two consecutive frames
make that directly visible to the CNN for the first time.

Only wraps wrist_image (the eye-in-hand view, where slip is actually visible) -- the workspace
image is left untouched, to keep this one hypothesis isolated.

`FrameStackWristEnv` is the live rollout wrapper (used identically at train, eval, demo, and
DAgger-collection time). `stack_wrist_offline` reconstructs the exact same stacking from an
already-collected demos npz's `wrist_images` array (no re-collection needed): decision index i
pairs `wrist_images[i-1]` (or a duplicate of `wrist_images[0]` when i is an episode's first step)
with `wrist_images[i]`, matching what the live wrapper produces step for step.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrameStackWristEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space["wrist_image"].shape
        spaces_dict = dict(env.observation_space.spaces)
        spaces_dict["wrist_image"] = spaces.Box(0, 255, (h, w, c * 2), dtype=np.uint8)
        self.observation_space = spaces.Dict(spaces_dict)
        self._prev_wrist = None

    def _stack(self, obs):
        obs = dict(obs)
        obs["wrist_image"] = np.concatenate([self._prev_wrist, obs["wrist_image"]], axis=-1)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_wrist = obs["wrist_image"]
        return self._stack(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        stacked = self._stack(obs)
        self._prev_wrist = obs["wrist_image"]
        return stacked, reward, terminated, truncated, info


def stack_wrist_offline(wrist_images, episode_steps):
    """Reconstructs FrameStackWristEnv's exact stacking from an already-collected flat
    wrist_images array (contiguous episode_steps-row blocks per episode, same convention as
    train_bc2.py's episode_val_split/compute_hold_weights)."""
    n = len(wrist_images)
    n_eps = n // episode_steps
    assert n_eps * episode_steps == n
    h, w, c = wrist_images.shape[1:]
    stacked = np.empty((n, h, w, c * 2), dtype=wrist_images.dtype)
    for ep in range(n_eps):
        s, e = ep * episode_steps, (ep + 1) * episode_steps
        prev = wrist_images[s]
        for i in range(s, e):
            stacked[i] = np.concatenate([prev, wrist_images[i]], axis=-1)
            prev = wrist_images[i]
    return stacked
