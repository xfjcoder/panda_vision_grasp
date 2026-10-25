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
from framestack_env import FrameStackWristEnv
from grasp_env import EPISODE_STEPS, GraspPixelEnv, MAX_JOINT_RATE, N_ARM
from residual_utils import load_mean_action

CORRECTION_CAP = 0.15  # ~5x bc01's mean|action| of 0.031 -- enough to correct meaningfully
                        # faster than smooth demo motion, far short of a full-rate snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True,
                     help="Existing run (e.g. bc03) whose policy drives the rollout")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", type=str, default="data/dagger01.npz")
    ap.add_argument("--seed-offset", type=int, default=0,
                     help="Episodes are seeded args.seed_offset + 0..episodes-1. Default 0 "
                          "reproduces every prior round's seeds (0..199) -- fine when the "
                          "champion changed round to round (items 18-25's by-hand rounds always "
                          "did), but collecting AGAIN from the SAME unpromoted champion with the "
                          "default offset re-visits the identical seeds and yields a byte-for-"
                          "byte-identical dataset (caught when dagger_loop.py's round 2 produced "
                          "a dagger12.npz identical to round 1's dagger11.npz, both from bc21 -- "
                          "see README item 30). dagger_loop.py now passes a fresh offset every "
                          "round so repeat collections from a plateaued champion actually explore "
                          "new states.")
    ap.add_argument("--gripper-latch-delay", type=int, default=5,
                     help="Apply the same post-hoc gripper-command latch as eval_policy.py "
                          "(README item 28) to the LEARNER action that drives which states get "
                          "visited -- does not touch the recorded label, which is always the "
                          "expert's own correct gripper command already. Keeps the visited-state "
                          "distribution consistent with how this checkpoint is actually run at "
                          "eval time. Pass a negative value to disable.")
    args = ap.parse_args()

    run_dir = f"runs/{args.run_name}"
    policy = PPO.load(f"{run_dir}/checkpoints/final")
    mean_action = load_mean_action(run_dir)
    env = GraspPixelEnv(curriculum_scale=1.0)
    sim = env  # keeps a direct reference for q_target/obj_pos0/model -- FrameStackWristEnv
               # doesn't forward those attributes (see demo_stage5_rl_failure.py's same fix)
    is_framestacked = policy.observation_space["wrist_image"].shape[0] == 6
    if is_framestacked:
        env = FrameStackWristEnv(env)  # item 31: trained against a stacked-wrist checkpoint
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(f"{run_dir}/checkpoints/vecnormalize_final.pkl", vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    images, wrist_images, proprios, actions = [], [], [], []
    n_would_succeed = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed_offset + ep)
        q_pregrasp = sim.q_target.copy()
        obj_pos0 = sim.obj_pos0.copy()
        ideal_targets = build_scripted_targets(sim.model, q_pregrasp, obj_pos0)

        actual_q_target = q_pregrasp.copy()
        info = {}
        has_opened = False
        close_start = None
        for t in range(min(len(ideal_targets), EPISODE_STEPS)):
            ideal_q, ideal_g = ideal_targets[t]
            delta = np.clip((ideal_q - actual_q_target) / MAX_JOINT_RATE,
                             -CORRECTION_CAP, CORRECTION_CAP)
            expert_action = np.concatenate([delta, [gripper_to_action(ideal_g)]]).astype(np.float32)

            images.append(obs["image"])
            # Always store the single CURRENT-frame wrist image, even when is_framestacked --
            # downstream files (train_bc2.py etc.) all expect plain 3-channel wrist_images, and
            # stacking is always re-derivable offline (stack_wrist_offline) from this.
            wrist_images.append(obs["wrist_image"][..., -3:] if is_framestacked else obs["wrist_image"])
            proprios.append(obs["proprio"])
            actions.append(expert_action)

            obs_norm = norm_env.normalize_obs(obs)
            learner_action, _ = policy.predict(obs_norm, deterministic=True)
            if mean_action is not None:
                learner_action = learner_action + mean_action[t]
            learner_action = np.clip(learner_action, -1.0, 1.0)
            if args.gripper_latch_delay >= 0:
                g = obs["proprio"][-1]
                if g > 0.9:
                    has_opened = True
                if has_opened and close_start is None and g < 0.75:
                    close_start = t
                if close_start is not None and t >= close_start + args.gripper_latch_delay:
                    learner_action[N_ARM] = -1.0
            obs, reward, terminated, truncated, info = env.step(learner_action)
            actual_q_target = sim.q_target.copy()
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
