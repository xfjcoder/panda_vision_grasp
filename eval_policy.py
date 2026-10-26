"""Evaluates a trained GraspPixelEnv policy over N fixed seeds -- same random_table_xy
convention as eval_stage2_reach_grasp.py's scripted baseline, so the two numbers are directly
comparable (20/20 scripted with privileged state vs. this, pixels + proprioception only).

Usage:
    python3 eval_policy.py --run-name grasp01 --trials 20
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from framestack_env import FrameStackWristEnv
from grasp_env import GraspPixelEnv, EPISODE_STEPS, N_ARM
from residual_utils import load_mean_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None,
                     help="Evaluate an intermediate CheckpointCallback snapshot "
                          "(checkpoints/ppo_<steps>_steps.zip + "
                          "ppo_vecnormalize_<steps>_steps.pkl) instead of the default "
                          "final/vecnormalize_final.pkl -- for plotting a plateau curve "
                          "across a single long training run's saved checkpoints.")
    ap.add_argument("--gripper-latch-delay", type=int, default=5,
                     help="Once proprio's gripper_qpos confirms a close (crosses below 0.75 "
                          "after having been open) and this many MORE steps have passed, "
                          "override the policy's own gripper command to fully-closed for the "
                          "rest of the episode -- a post-hoc hysteresis on just that one output "
                          "channel, using only the policy's own observable state, no "
                          "retraining. Targets a diagnosed issue: the learned gripper command "
                          "itself keeps oscillating after contact (6-13 sign flips per 30 "
                          "post-close steps vs the scripted expert's constant command), unlike "
                          "the smooth demos it was trained on. Delay matters a lot -- 0 (latch "
                          "immediately) actively hurts (cuts off what looks like a brief, "
                          "useful self-adjusting squeeze), 5 is the sweet spot found by sweeping "
                          "{0,5,10,15,20,30} on bc15/bc19, 15+ also hurts. DEFAULT ON (5) as of "
                          "item 28 -- pass a negative value (e.g. -1) to fully disable and "
                          "reproduce the original unmodified behavior, needed when comparing "
                          "against any pre-item-28 numbers in the README.")
    args = ap.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.run_name
    if args.steps is not None:
        model_stem = f"ppo_{args.steps}_steps"
        vecnorm_name = f"ppo_vecnormalize_{args.steps}_steps.pkl"
    else:
        model_stem = "final"
        vecnorm_name = "vecnormalize_final.pkl"
    policy = PPO.load(run_dir / "checkpoints" / model_stem)
    mean_action = load_mean_action(run_dir)
    env = GraspPixelEnv()
    if policy.observation_space["wrist_image"].shape[0] == 6:
        env = FrameStackWristEnv(env)  # item 31: trained against a stacked-wrist checkpoint
    vec_env = DummyVecEnv([lambda: env])
    norm_env = VecNormalize.load(str(run_dir / "checkpoints" / vecnorm_name), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    results = []
    for i in range(args.trials):
        seed = args.base_seed + i
        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0
        info = {}
        has_opened = False
        close_start = None
        for t in range(EPISODE_STEPS):
            obs_norm = norm_env.normalize_obs(obs)
            action, _ = policy.predict(obs_norm, deterministic=True)
            if mean_action is not None:
                action = action + mean_action[t]
            if args.gripper_latch_delay >= 0:
                g = obs["proprio"][-1]
                if g > 0.9:
                    has_opened = True
                if has_opened and close_start is None and g < 0.75:
                    close_start = t
                if close_start is not None and t >= close_start + args.gripper_latch_delay:
                    action[N_ARM] = -1.0
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        results.append(dict(seed=seed, success=info["success"], rise=info["rise"],
                             drift=info["drift"], ep_reward=ep_reward))
        print(f"trial {i:2d} seed={seed:4d} rise={info['rise']:.4f}m drift={info['drift']:.4f}m "
              f"ep_reward={ep_reward:.2f} success={info['success']}")

    n_success = sum(r["success"] for r in results)
    mean_reward = float(np.mean([r["ep_reward"] for r in results]))
    mean_rise = float(np.mean([r["rise"] for r in results]))
    print(f"\nSummary over {args.trials} trials: {n_success}/{args.trials} succeeded "
          f"(scripted privileged-state baseline: 20/20)")
    print(f"SUMMARY_PARSEABLE run={args.run_name} trials={args.trials} success={n_success} "
          f"mean_reward={mean_reward:.4f} mean_rise={mean_rise:.4f}")
    env.close()


if __name__ == "__main__":
    main()
