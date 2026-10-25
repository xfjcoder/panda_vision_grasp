"""Automates the DAgger iteration loop that items 18-25 ran by hand, one round at a time: roll
out the current champion -> collect fresh expert corrections -> merge cumulatively -> retrain ->
evaluate on a FIXED, larger seed count (less noise-prone than the 20-seed sweeps that overstated
early hold-weight gains, see item 27's regression-to-the-mean note) -> compare to the champion
under the IDENTICAL eval condition -> promote only if strictly better, else stop and report the
plateau instead of guessing whether to continue.

Why this exists: every DAgger round in this project so far needed a human (or the assistant) to
manually run collect_dagger.py, merge the npz files by hand, retrain, eval, and eyeball a
printed table to decide whether to keep going -- correct, but entirely manual every single time,
and the champion being compared against was often evaluated under a slightly different condition
(different trial count, before/after the gripper latch existed) than the new candidate. This
script removes the manual labor for routine continuation rounds and enforces an apples-to-apples
comparison, while keeping the same honest-negative-result discipline: a round that doesn't beat
its champion stops the loop rather than silently cherry-picking a worse checkpoint forward.

Promotion rule (deliberately simple and conservative): candidate promotes only if its success
count on the shared eval-trial set is HIGHER than the champion's, or EQUAL with a higher mean
reward. Anything else (including a tie on both) stops the loop -- "iteratively improving" means
every step must be a real improvement, not a lateral move.

Usage:
    python3 dagger_loop.py --champion bc19 --champion-data data/demos_dagger10.npz \
        --net-arch 64,64 --hold-weight 1.5 --max-rounds 5 --eval-trials 40 \
        --start-dagger-num 11 --start-run-num 23
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

SUMMARY_RE = re.compile(
    r"SUMMARY_PARSEABLE run=(\S+) trials=(\d+) success=(\d+) mean_reward=(-?[\d.]+) "
    r"mean_rise=(-?[\d.]+)"
)


def run(cmd, log_path):
    print(f"$ {' '.join(cmd)}")
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"!! command failed (exit {proc.returncode}), see {log_path}")
        sys.exit(1)
    return log_path.read_text()


def parse_eval_summary(eval_log_text):
    m = SUMMARY_RE.search(eval_log_text)
    if not m:
        raise RuntimeError(f"could not find SUMMARY_PARSEABLE line in eval output")
    return dict(run=m.group(1), trials=int(m.group(2)), success=int(m.group(3)),
                mean_reward=float(m.group(4)), mean_rise=float(m.group(5)))


def eval_run(run_name, trials):
    log_path = HERE / "runs" / f"{run_name}_loop_eval.log"
    text = run(["python3", "eval_policy.py", "--run-name", run_name, "--trials", str(trials)],
               log_path)
    return parse_eval_summary(text)


def merge_npz(a_path, b_path, out_path):
    a, b = np.load(a_path), np.load(b_path)
    merged = {k: np.concatenate([a[k], b[k]], axis=0) for k in a.keys()}
    np.savez(out_path, **merged)
    return len(merged["actions"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", type=str, required=True)
    ap.add_argument("--champion-data", type=str, required=True)
    ap.add_argument("--net-arch", type=str, default="64,64")
    ap.add_argument("--hold-weight", type=float, default=1.5)
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--eval-trials", type=int, default=40)
    ap.add_argument("--collect-episodes", type=int, default=200)
    ap.add_argument("--start-dagger-num", type=int, required=True,
                     help="First unused data/daggerNN.npz index -- avoids clobbering existing "
                          "files, no auto-detection to avoid silently overwriting something.")
    ap.add_argument("--start-run-num", type=int, required=True,
                     help="First unused bcNN run name -- same reasoning.")
    args = ap.parse_args()

    champion, champion_data = args.champion, args.champion_data
    print(f"=== evaluating starting champion {champion} ({args.eval_trials} trials) ===")
    champion_result = eval_run(champion, args.eval_trials)
    print(f"champion {champion}: success={champion_result['success']}/{args.eval_trials} "
          f"mean_reward={champion_result['mean_reward']:.3f}")

    history = [dict(round=0, run=champion, **{k: v for k, v in champion_result.items()
                                                if k != "run"}, promoted=True)]

    dagger_num, run_num = args.start_dagger_num, args.start_run_num
    for round_i in range(1, args.max_rounds + 1):
        print(f"\n=== round {round_i}: collecting corrections from {champion} ===")
        dagger_out = HERE / "data" / f"dagger{dagger_num:02d}.npz"
        seed_offset = dagger_num * args.collect_episodes
        run(["python3", "collect_dagger.py", "--run-name", champion,
             "--episodes", str(args.collect_episodes), "--out", str(dagger_out),
             "--seed-offset", str(seed_offset)],
            HERE / "runs" / f"dagger{dagger_num:02d}_collect.log")

        new_data = HERE / "data" / f"demos_dagger{dagger_num}.npz"
        n = merge_npz(champion_data, dagger_out, new_data)
        print(f"merged -> {new_data} ({n} transitions)")

        new_run = f"bc{run_num}"
        print(f"=== round {round_i}: training {new_run} (net_arch={args.net_arch}, "
              f"hold_weight={args.hold_weight}) ===")
        run(["python3", "train_bc2.py", "--run-name", new_run, "--demos", str(new_data),
             "--net-arch", args.net_arch, "--hold-weight", str(args.hold_weight)],
            HERE / "runs" / f"{new_run}_train.log")

        print(f"=== round {round_i}: evaluating {new_run} ({args.eval_trials} trials) ===")
        candidate_result = eval_run(new_run, args.eval_trials)
        print(f"{new_run}: success={candidate_result['success']}/{args.eval_trials} "
              f"mean_reward={candidate_result['mean_reward']:.3f}  (champion {champion}: "
              f"success={champion_result['success']}/{args.eval_trials} "
              f"mean_reward={champion_result['mean_reward']:.3f})")

        promoted = (candidate_result["success"] > champion_result["success"] or
                    (candidate_result["success"] == champion_result["success"] and
                     candidate_result["mean_reward"] > champion_result["mean_reward"]))
        history.append(dict(round=round_i, run=new_run,
                             **{k: v for k, v in candidate_result.items() if k != "run"},
                             promoted=promoted))

        if promoted:
            print(f"PROMOTED: {new_run} beats {champion} -- continuing the loop from {new_run}")
            champion, champion_data = new_run, str(new_data)
            champion_result = candidate_result
        else:
            print(f"PLATEAU: {new_run} did not beat {champion} -- stopping the loop here")
            break

        dagger_num += 1
        run_num += 1

    print("\n=== loop summary ===")
    print(f"{'round':>5} {'run':>6} {'success':>10} {'mean_reward':>12} {'promoted':>9}")
    for h in history:
        print(f"{h['round']:>5} {h['run']:>6} {h['success']:>7}/{h['trials']:<2} "
              f"{h['mean_reward']:>12.3f} {str(h['promoted']):>9}")
    print(f"\nFinal champion: {champion}")


if __name__ == "__main__":
    main()
