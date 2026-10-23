"""DAgger correction pass on top of `bc01`: rolls out `bc01`'s OWN policy (not the scripted
controller) so the states visited are the ones the learned policy actually drifts into, then
labels each visited state with the EXPERT's corrective action from that exact (possibly
already-drifted) state, rather than the original fixed trajectory's action.

Why: bc01 (pure behavior cloning on 200 scripted demo episodes, no drift-recovery examples)
reaches a small positive `rise` on its best trials but never a successful grasp -- confirmed via
GIF inspection (media/bc01.gif) that by the time it closes the gripper, the object sits outside
the finger line, not between the pads. Classic BC compounding error: the network only ever saw
states ON the expert's own trajectory, never "slightly off-center, here's how to correct" states,
so once its own small prediction errors drift it off that exact path there's no learned recovery
signal. DAgger's fix is exactly this: label the LEARNER's own visited states with fresh expert
actions, so the aggregated dataset finally contains recovery examples.

The "expert action from an arbitrary current state" query is cheap here because we have
privileged access to the object's ground-truth position and the scripted controller's absolute,
time-indexed joint-target schedule (build_scripted_targets, from collect_demos.py) -- that
schedule doesn't depend on where the arm actually is, so at any step t the expert's correction is
just "reach this step's scheduled absolute joint target from wherever you currently are":
    expert_action = clip((ideal_targets[t].q_target - actual_q_target) / MAX_JOINT_RATE, -1, 1)
computed against the ACTUAL `env.q_target` (which reflects the rollout policy's own prior
actions, including any drift), not the original demo's q_target.

**CORRECTION_CAP, added for the bc03 lift-phase-drift round (the first DAgger round, bc01->bc02,
didn't have this):** bc02's post-mortem (see README Stage 5 item 13) found that DAgger fixed the
targeted alignment error but made the aggregate-trained policy noisier overall (sign-flip rate
0.56 vs bc01's 0.02) -- because when the learner has drifted far off-path, the raw clip(-1, 1)
correction saturates at the full MAX_JOINT_RATE snap, producing training labels much larger and
jerkier than any smooth demo action (bc01's mean|action| was 0.031). Those outlier-magnitude
labels dominate the MSE loss and teach jerkiness even in states that didn't need it. Capping the
correction to CORRECTION_CAP keeps every label pointing the right corrective direction without
ever exceeding a few times the smooth-demo action scale, regardless of how far the learner has
drifted.

Usage:
    python3 collect_dagger.py --run-name bc03 --episodes 200 --out data/dagger03.npz
"""
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from collect_demos import build_scripted_targets, gripper_to_action
from grasp_env import EPISODE_STEPS, GraspPixelEnv, MAX_JOINT_RATE

CORRECTION_CAP = 0.15  # ~5x bc01's mean|action| of 0.031 -- enough to correct meaningfully
                        # faster than smooth demo motion, far short of a full-rate snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True,
                     help="Existing run (e.g. bc03) whose policy drives the rollout")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", type=str, default="data/dagger01.npz")
    args = ap.parse_args()

    run_dir = f"runs/{args.run_name}"
    policy = PPO.load(f"{run_dir}/checkpoints/final")
    env = GraspPixelEnv(curriculum_scale=1.0)
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(f"{run_dir}/checkpoints/vecnormalize_final.pkl", vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    images, wrist_images, proprios, actions = [], [], [], []
    n_would_succeed = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        q_pregrasp = env.q_target.copy()
        obj_pos0 = env.obj_pos0.copy()
        ideal_targets = build_scripted_targets(env.model, q_pregrasp, obj_pos0)

        actual_q_target = q_pregrasp.copy()
        info = {}
        for t in range(min(len(ideal_targets), EPISODE_STEPS)):
            ideal_q, ideal_g = ideal_targets[t]
            delta = np.clip((ideal_q - actual_q_target) / MAX_JOINT_RATE,
                             -CORRECTION_CAP, CORRECTION_CAP)
            expert_action = np.concatenate([delta, [gripper_to_action(ideal_g)]]).astype(np.float32)

            images.append(obs["image"])
            wrist_images.append(obs["wrist_image"])
            proprios.append(obs["proprio"])
            actions.append(expert_action)

            obs_norm = norm_env.normalize_obs(obs)
            learner_action, _ = policy.predict(obs_norm, deterministic=True)
            learner_action = np.clip(learner_action, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(learner_action)
            actual_q_target = env.q_target.copy()
            if terminated or truncated:
                break
        n_would_succeed += int(info["success"])
        if ep % 20 == 0 or ep == args.episodes - 1:
            print(f"episode {ep}: {args.run_name}-driven rollout success={info['success']} "
                  f"({n_would_succeed}/{ep + 1} so far)")

    print(f"collected {len(actions)} DAgger-relabeled transitions over {args.episodes} episodes "
          f"({args.run_name} itself succeeded on {n_would_succeed}/{args.episodes} of them)")
    np.savez(args.out, images=np.array(images, dtype=np.uint8),
              wrist_images=np.array(wrist_images, dtype=np.uint8),
              proprio=np.array(proprios, dtype=np.float32),
              actions=np.array(actions, dtype=np.float32))
    env.close()


if __name__ == "__main__":
    main()
