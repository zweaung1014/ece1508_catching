"""
Author: Zwe Min Htet Aung
ABOUT: summarize the per-episode CSVs written by eval_catch.py for one experiment (results/<exp>/).

Prints, per test condition (an experiment without conditions is one block):
1. Catch rate with a 95% Wilson interval, the outcome breakdown and effort metrics, per controller.
2. Paired McNemar tests against a reference controller (same seeds = same throws).
3. Classical diagnostics: prediction error, tracking error, and why each miss happened.
4. Catch rate per bin of the per-throw disturbances (velocity noise, drag, delay) when they were
   randomized, then of flight time, launch azimuth and required hoop travel. The disturbances are
   drawn independently, so each bin averages over the others.

Usage (from catch_it/):
  python analyze_results.py --exp exp02_randomized --markdown results/exp02_randomized/summary.md
  python analyze_results.py --exp exp01_one_factor --conditions nominal noise_x2
"""
import os
import argparse
import glob
import math
import numpy as np
import pandas as pd
from scipy.stats import binom

CATCH_IT = os.path.dirname(os.path.abspath(__file__))
ORDER = ["random", "classical_single", "classical_multi", "rl", "classical_perfect", "oracle"]
OUTCOMES = ["catch", "ball_floor", "ball_robot", "collision", "timeout"]
# Thresholds for attributing a classical miss (checked in this order)
MISS_PRED = 0.06    # m, predicted catch point off the true path by more than this
MISS_HOOP = 0.04    # m, hoop more than this from its planned point at the planned catch time
MISS_AXIS = 15.0    # deg, hoop axis more than this from the planned axis


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired 0/1 outcomes. Returns (a-only wins, b-only wins, p)."""
    only_a = int(np.sum((a == 1) & (b == 0)))
    only_b = int(np.sum((a == 0) & (b == 1)))
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * binom.cdf(min(only_a, only_b), n, 0.5))
    return only_a, only_b, p


def miss_reason(r):
    if r["feasible"] == 0:
        return "not reachable in time"
    if r["pred_err_perp"] > MISS_PRED:
        return "prediction"
    if r["hoop_err_at_tstar"] > MISS_HOOP or r["axis_err_at_tstar_deg"] > MISS_AXIS:
        return "arm tracking"
    return "other (rim, IK, bounce)"


def load(exp_dir, conditions):
    frames = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "**", "*.csv"), recursive=True)):
        df = pd.read_csv(path)
        if "condition" not in df.columns:              # one setting per experiment (exp02 on)
            df["condition"] = os.path.basename(os.path.normpath(exp_dir))
        df["run"] = os.path.splitext(os.path.basename(path))[0]   # controller name (+ tag)
        frames.append(df)
    if not frames:
        raise SystemExit("no CSVs found under " + exp_dir)
    df = pd.concat(frames, ignore_index=True)
    if conditions:
        df = df[df["condition"].isin(conditions)]
    if "drag_rate" in df.columns:
        df["drag_loss_pct"] = 100 * (1 - np.exp(-df["drag_rate"]))   # % of speed lost to drag over 1 s
    return df


def run_order(runs):
    def rank(r):
        for i, o in enumerate(ORDER):
            if r == o or r.startswith(o + "_"):
                return (i, r)
        return (99, r)
    return sorted(runs, key=rank)


def table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    out = [line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def summarize(df, ref, bins):
    out = []
    for cond in sorted(df["condition"].unique(), key=lambda c: (c != "nominal", c)):
        d = df[df["condition"] == cond]
        runs = run_order(d["run"].unique())
        out.append("\n## Condition: {}\n".format(cond))

        # 1. Catch rate and outcomes
        rows = []
        for run in runs:
            x = d[d["run"] == run]
            n, k = len(x), int(x["success"].sum())
            lo, hi = wilson(k, n)
            rows.append([run, n, "{:.1%}".format(k / n), "{:.1%}-{:.1%}".format(lo, hi)]
                        + ["{:.1%}".format((x["outcome"] == o).mean()) for o in OUTCOMES[1:]]
                        + ["{:.2f}".format(x["rim_hits"].mean()), "{:.2f}".format(x["ik_fails"].mean()),
                           "{:.2f}".format(x["hoop_travel"].mean()), "{:.3f}".format(x["action_smooth"].mean())])
        out.append(table(rows, ["controller", "n", "catch", "95% CI"] + OUTCOMES[1:]
                         + ["rim hits/ep", "IK fails/ep", "hoop travel (m)", "action change^2"]))

        # 2. Paired tests against the reference
        if ref in runs:
            r = d[d["run"] == ref].set_index("seed")["success"]
            rows = []
            for run in runs:
                if run == ref:
                    continue
                o = d[d["run"] == run].set_index("seed")["success"]
                common = r.index.intersection(o.index)
                a, b, p = mcnemar(r.loc[common].values, o.loc[common].values)
                rows.append([run, len(common), a, b, "{:.2g}".format(p)])
            out.append("\nPaired McNemar vs {} (same seeds):\n".format(ref))
            out.append(table(rows, ["controller", "paired n", "only {} caught".format(ref), "only other caught", "p"]))

        # 3. Classical diagnostics
        cl = [run for run in runs if run.startswith("classical")]
        if cl:
            rows, reasons = [], {}
            for run in cl:
                x = d[d["run"] == run]
                rows.append([run,
                             "{:.1f}".format(100 * x["pred_err_perp"].median()), "{:.1f}".format(100 * x["pred_err_perp"].quantile(0.9)),
                             "{:.1f}".format(100 * x["hoop_err_at_tstar"].median()),
                             "{:.0f}".format(1000 * x["decision_delay"].median()),
                             "{:.1%}".format(x["feasible"].mean()), "{:.2f}".format(x["g_hat"].median())])
                miss = x[x["success"] == 0]
                reasons[run] = miss.apply(miss_reason, axis=1).value_counts() if len(miss) else pd.Series(dtype=int)
            out.append("\nClassical diagnostics (prediction error = predicted catch point vs true path, across the path):\n")
            out.append(table(rows, ["controller", "pred err median (cm)", "pred err p90 (cm)", "hoop err median (cm)",
                                    "first plan after release (ms)", "plan reachable", "g estimate (median)"]))
            names = sorted({k for s in reasons.values() for k in s.index})
            if names:
                out.append("\nWhy the classical controllers missed (count of misses):\n")
                out.append(table([[run] + [int(reasons[run].get(k, 0)) for k in names] for run in cl], ["controller"] + names))

        # 4. Catch rate per bin
        for col, edges, label, names in bins:
            if col not in d.columns or d[col].nunique() < 2:
                continue
            cats = pd.cut(d[col], edges, labels=names, include_lowest=True)
            g = d.assign(bin=cats).groupby(["bin", "run"], observed=False)["success"].mean().unstack("run")
            g = g[[run for run in runs if run in g.columns]]
            rows = [[str(b)] + ["{:.0%}".format(v) if not np.isnan(v) else "-" for v in g.loc[b].values] for b in g.index]
            counts = d[d["run"] == runs[0]].assign(bin=cats).groupby("bin", observed=False).size()
            rows = [r + [int(counts.loc[b])] for r, b in zip(rows, g.index)]
            out.append("\nCatch rate by {}:\n".format(label))
            out.append(table(rows, [label] + list(g.columns) + ["n"]))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Summarize eval_catch.py results")
    parser.add_argument("--exp", required=True, help="experiment folder under --results_dir")
    parser.add_argument("--results_dir", default=os.path.join(CATCH_IT, "results"))
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--ref", default="rl", help="reference controller for the paired tests")
    parser.add_argument("--markdown", default="", help="also write the summary to this file")
    args = parser.parse_args()
    df = load(os.path.join(args.results_dir, args.exp), args.conditions)
    ref = args.ref if args.ref in df["run"].unique() else "classical_multi"
    # (column, bin edges, table label, bin names or None for the interval)
    bins = [
        ("vel_noise_std", [0.05, 0.0933, 0.1367, 0.18], "velocity noise std (m/s)", ["0.05-0.09", "0.09-0.14", "0.14-0.18"]),
        ("drag_loss_pct", [0, 4, 8, 14], "speed lost to drag over 1 s (%)", ["0-4", "4-8", "8-13"]),
        ("act_delay_steps", [-0.5, 0.5, 1.5, 2.5], "action delay (steps of 40 ms)", ["0", "1", "2"]),
        ("flight_time", [0.6, 0.7, 0.8, 0.9, 1.0], "flight time (s)", None),
        ("launch_az_deg", [-60, -30, 0, 30, 60], "launch azimuth (deg)", None),
        ("home_to_target", [0.0, 0.2, 0.3, 0.4, 0.5, 0.8], "home-to-target distance (m)", None),
    ]
    text = summarize(df, ref, bins)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write("# HoopCatch evaluation summary: {}\n".format(args.exp) + text + "\n")
        print("\nsaved", args.markdown)


if __name__ == "__main__":
    main()
