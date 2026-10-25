"""Shared helper for the residual-action BC variant (train_bc_residual.py, README item 31): the
policy predicts a correction on top of a fixed, precomputed per-timestep mean action
(mean_action.npy, written alongside the checkpoint at train time) instead of the raw action.

Why a per-timestep mean and not, say, the scripted expert's own trajectory: the expert's
schedule needs privileged object-position state (see collect_dagger.py), which isn't available
at eval time for a pixels-to-actions policy. The per-timestep mean action across training
episodes needs nothing but the step index, which every rollout loop already has, and it still
captures the "easy majority" (steady approach/close/lift schedule, roughly shared across
episodes) that item 29 found gets fit almost immediately -- leaving 100% of the network's
residual-prediction error to the harder, state-dependent corrective minority that actual task
success depends on.

Auto-detected: a checkpoint with no mean_action.npy is a plain (non-residual) run; callers get
None back and skip the add-back entirely, so this is a no-op for every pre-item-31 checkpoint.
"""
from pathlib import Path

import numpy as np


def load_mean_action(run_dir):
    path = Path(run_dir) / "checkpoints" / "mean_action.npy"
    return np.load(path) if path.exists() else None
