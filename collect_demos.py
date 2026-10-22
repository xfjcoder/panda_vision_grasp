"""Collects a behavior-cloning dataset by driving GraspPixelEnv's exact observation/action
pipeline with Stage 2's already-working scripted controller (20/20 with privileged state), using
the FULL object-position range (curriculum_scale=1.0, matching eval/deployment distribution) --
not the narrow curriculum used by grasp04.

Why: grasp01-grasp04 each diagnosed and fixed a real cold-start RL problem (partial learning,
exploration collapse, task scaffolding) without a single success -- strong evidence the remaining
gap is the precise grasp-closing maneuver being too hard to discover via pure exploration from
sparse reward. Rather than keep tuning cold-start RL, warm-start the policy on what the scripted
controller already knows how to do, then (separately, in train.py) fine-tune with PPO on top.

Each scripted phase (descend / close+hold / lift / post-lift hold) is re-derived at CONTROL_DT
(not physics-dt) resolution directly against the SAME env instance used for training, so observed
images/proprio and the resulting action sequence are exactly what GraspPixelEnv.step() would see.

Usage:
    python3 collect_demos.py --episodes 200 --out demos.npz
"""
import argparse

import numpy as np

from eval_stage2_reach_grasp import (CLOSE_SECONDS, DESCEND_SECONDS, GRIPPER_CLOSE, GRIPPER_OPEN,
                                      HOLD_SECONDS, LIFT_HEIGHT, LIFT_SECONDS)
from grasp_env import CONTROL_DT, EPISODE_STEPS, GraspPixelEnv, MAX_JOINT_RATE
from ik import solve_ik


def gripper_to_action(g):
    return float(np.clip(g / 255.0 * 2.0 - 1.0, -1.0, 1.0))


def build_scripted_targets(model, q_pregrasp, obj_pos0):
    """One (q_target, gripper_target) pair per control step, covering descend -> close+hold ->
    lift -> a static post-lift hold padding out the rest of the episode -- same phase structure
    as eval_stage2_reach_grasp.run_trial, just resampled onto CONTROL_DT control steps instead
    of raw physics-dt substeps."""
    q_grasp, ok, _ = solve_ik(model, obj_pos0, q_pregrasp, max_iters=150, tol=0.01)
    lifted = obj_pos0 + np.array([0, 0, LIFT_HEIGHT])
    q_lifted, ok2, _ = solve_ik(model, lifted, q_grasp, max_iters=150, tol=0.01)

    targets = []
    n_descend = max(int(DESCEND_SECONDS / CONTROL_DT), 1)
    for i in range(n_descend):
        frac = (i + 1) / n_descend
        targets.append((q_pregrasp + (q_grasp - q_pregrasp) * frac, GRIPPER_OPEN))

    n_close = max(int((CLOSE_SECONDS + HOLD_SECONDS) / CONTROL_DT), 1)
    for _ in range(n_close):
        targets.append((q_grasp.copy(), GRIPPER_CLOSE))

    n_lift = max(int(LIFT_SECONDS / CONTROL_DT), 1)
    for i in range(n_lift):
        frac = (i + 1) / n_lift
        targets.append((q_grasp + (q_lifted - q_grasp) * frac, GRIPPER_CLOSE))

    while len(targets) < EPISODE_STEPS:
        targets.append((q_lifted.copy(), GRIPPER_CLOSE))
    return targets[:EPISODE_STEPS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", type=str, default="demos.npz")
    args = ap.parse_args()

    env = GraspPixelEnv(curriculum_scale=1.0)
    images, proprios, actions = [], [], []
    n_success = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        q_pregrasp = env.q_target.copy()
        obj_pos0 = env.obj_pos0.copy()
        targets = build_scripted_targets(env.model, q_pregrasp, obj_pos0)

        q_target = q_pregrasp.copy()
        info = {}
        for q_next, g_next in targets:
            delta = np.clip((q_next - q_target) / MAX_JOINT_RATE, -1.0, 1.0)
            action = np.concatenate([delta, [gripper_to_action(g_next)]]).astype(np.float32)
            images.append(obs["image"])
            proprios.append(obs["proprio"])
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            q_target = env.q_target.copy()
            if terminated or truncated:
                break
        n_success += int(info["success"])
        if ep % 20 == 0 or ep == args.episodes - 1:
            print(f"episode {ep}: success={info['success']} ({n_success}/{ep + 1} so far)")

    print(f"collected {len(actions)} transitions over {args.episodes} episodes, "
          f"{n_success}/{args.episodes} scripted successes")
    np.savez(args.out, images=np.array(images, dtype=np.uint8),
              proprio=np.array(proprios, dtype=np.float32),
              actions=np.array(actions, dtype=np.float32))
    env.close()


if __name__ == "__main__":
    main()
