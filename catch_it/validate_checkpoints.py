"""
Author: Zwe Min Htet Aung
ABOUT: pick a training run's checkpoint on the validation throws (seeds 0-499), never the test throws.
Runs eval_catch.py on every snapshot of a run's nn/ folder (ep_*.pth by step, then best_reward_*.pth and
last.pth), writes a markdown table of the catch rates, and prints the best checkpoint on the last line:
  BEST <path> <catch rate>
Highest catch rate wins; a tie goes to the checkpoint listed first (the earlier snapshot).
A checkpoint whose result is already in --csv_dir is not evaluated again, so a stopped pipeline can resume.

Usage (from catch_it/):
  python validate_checkpoints.py --nn_dir outputs/rl_exp04_goal/<date>/<time>/nn --label goal \
    --csv_dir outputs/exp04_val --md results/exp04_goal_action/validation.md
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

from analyze_results import wilson

CATCH_IT = os.path.dirname(os.path.abspath(__file__))


def checkpoints(nn_dir):
    eps = sorted(glob.glob(os.path.join(nn_dir, "ep_*.pth")),
                 key=lambda p: int(re.search(r"ep_(\d+)_", os.path.basename(p)).group(1)))
    return eps + sorted(glob.glob(os.path.join(nn_dir, "best_reward_*.pth"))) + \
        [p for p in [os.path.join(nn_dir, "last.pth")] if os.path.exists(p)]


def step_of(path):
    m = re.search(r"_step_(\d+)m_", os.path.basename(path))
    return "{}M".format(int(m.group(1))) if m else "-"


def main():
    parser = argparse.ArgumentParser(description="Evaluate every checkpoint of a run on the validation seeds")
    parser.add_argument("--nn_dir", required=True, help="the run's nn/ folder")
    parser.add_argument("--label", required=True, help="short name of the run, e.g. goal or delta")
    parser.add_argument("--csv_dir", required=True, help="where the per-checkpoint CSVs go (keep it out of results/<exp>/)")
    parser.add_argument("--md", required=True, help="markdown file; this run's table is appended")
    parser.add_argument("--seeds", default="0:500")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cks = checkpoints(os.path.abspath(args.nn_dir))
    if not cks:
        sys.exit("no checkpoints in " + args.nn_dir)
    rows = []
    for ck in cks:
        tag = "{}_{}".format(args.label, os.path.splitext(os.path.basename(ck))[0])
        meta_path = os.path.join(args.csv_dir, args.label, "rl_{}.json".format(tag))
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
        if meta is None or meta.get("checkpoint") != ck or meta.get("seeds") != args.seeds:
            subprocess.run([sys.executable, "eval_catch.py", "--exp", args.label, "--controller", "rl",
                            "--checkpoint", ck, "--seeds", args.seeds, "--workers", str(args.workers),
                            "--out_dir", args.csv_dir, "--tag", tag],
                           cwd=CATCH_IT, check=True, stdout=subprocess.DEVNULL)
            meta = json.load(open(meta_path))
        n = meta["episodes"]
        k = round(meta["catch_rate"] * n)
        lo, hi = wilson(k, n)
        rows.append((ck, step_of(ck), meta["catch_rate"], lo, hi))
        print("{:45s} {:>5s}  catch {:.3f}".format(os.path.basename(ck), rows[-1][1], meta["catch_rate"]), flush=True)

    best = max(rows, key=lambda r: r[2])    # max keeps the first of equal values: the earlier snapshot
    with open(args.md, "a") as f:
        f.write("\n## {} ({} validation throws, seeds {})\n\n".format(args.label, n, args.seeds))
        f.write("Run: `{}`\n\n".format(os.path.relpath(os.path.abspath(args.nn_dir), CATCH_IT)))
        f.write("| checkpoint | step | catch | 95% CI |\n|---|---|---|---|\n")
        for ck, step, rate, lo, hi in rows:
            mark = " **(picked)**" if ck == best[0] else ""
            f.write("| {}{} | {} | {:.1f}% | {:.1f}-{:.1f}% |\n".format(
                os.path.basename(ck), mark, step, 100 * rate, 100 * lo, 100 * hi))
    print("BEST {} {:.4f}".format(best[0], best[2]))


if __name__ == "__main__":
    main()
