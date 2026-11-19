"""
Author: Zwe Min Htet Aung
ABOUT: training curves of a PPO run, to see whether the policy under- or over-fits and whether training is
stable. Reads the run's TensorBoard log and, if given, the validation results of its snapshots
(validate_checkpoints.py) and of a reference controller on the same validation throws.

Panels (x = training steps):
  1. Catch rate: training (stochastic policy, last 200 training episodes) vs validation (mean action, the
     500 held-out validation throws, 95% CI), and the reference controller on those same throws
  2. Episode reward (training)          3. Value (critic) loss          4. Policy (actor) loss
  5. KL between the rollout policy and the updated one, with the early-stop line
  6. Exploration std of the policy (range over the 5 actions)

How to read panel 1:
  - both curves low and flat -> under-fitting (the policy has not learned the task)
  - training high, validation clearly lower -> the policy depends on its exploration noise or on the
    training throws (over-fitting); every training episode draws a new throw, so this gap is usually small
  - both rise, then fall -> unstable training (good policies are lost), not over-fitting

Writes <out>.png, <out>_train.csv (one row per update) and <out>_val.csv (one row per snapshot).

Usage (from catch_it/):
  python plot_training.py --run_dir outputs/rl_exp05/<date>/<time> --val_dir outputs/exp05_val/rl \
      --ref_json outputs/exp05_val/val_ref/classical_multi.json --out results/exp05_ball_noise/training_curves
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from analyze_results import wilson

# Reference palette (dataviz skill, light mode; validated: CVD dE 24.7, contrast >= 3:1)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRAIN, VAL = "#2a78d6", "#eb6834"

TAGS = {
    "train_catch": "metrics/episode_success_per_step",
    "episode_reward": "metrics/episode_rewards_per_step",
    "critic_loss": "losses/critic_loss",
    "actor_loss": "losses/actor_loss",
    "kl": "info/kl",
    "std_min": "policy/std_min",
    "std_max": "policy/std_max",
    "sigma_mean": "info/sigma_mean",   # older trainer (larry-catch): mean std only
}


def read_tb(run_dir):
    files = glob.glob(os.path.join(run_dir, "tb", "events*"))
    if not files:
        raise SystemExit("no TensorBoard log in " + os.path.join(run_dir, "tb"))
    ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
    ea.Reload()
    have = ea.Tags()["scalars"]
    cols = {}
    for name, tag in TAGS.items():
        if tag in have:
            cols[name] = pd.Series({e.step: e.value for e in ea.Scalars(tag)})
    df = pd.DataFrame(cols)
    df.index.name = "step"
    return df.sort_index()


def steps_per_update(run_dir):
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    return int(cfg["train"]["ppo"]["horizon_length"]) * int(cfg["num_envs"])


def read_val(val_dir, batch, last_step):
    """Validation catch rate of every periodic snapshot (ep_*) and last.pth, at the step it was saved."""
    rows = []
    for path in glob.glob(os.path.join(val_dir, "rl_*.json")):
        meta = json.load(open(path))
        name = os.path.basename(meta["checkpoint"])
        m = re.match(r"ep_(\d+)_", name)
        if m:
            step = (int(m.group(1)) + 1) * batch      # agent_steps starts at one batch and grows by one per update
        elif name == "last.pth":
            step = last_step
        else:
            continue                                  # best_reward_*: its step is not recorded in the name
        n = meta["episodes"]
        lo, hi = wilson(round(meta["catch_rate"] * n), n)
        rows.append({"step": step, "checkpoint": name, "val_catch": meta["catch_rate"], "lo": lo, "hi": hi, "n": n})
    return pd.DataFrame(rows).sort_values("step").drop_duplicates("step", keep="first") if rows else pd.DataFrame()


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def line(ax, x, y, color, window, clip=True):
    """Raw values thin and light, rolling mean on top. clip: y range from the 1st-99.5th percentile, so a
    few spikes do not flatten the rest (they run off the top)."""
    ax.plot(x, y, color=color, linewidth=0.8, alpha=0.25)
    ax.plot(x, pd.Series(y).rolling(window, min_periods=1).mean(), color=color, linewidth=2)
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if clip and len(y) > 10:
        lo, hi = np.percentile(y, [1, 99.5])
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(lo - pad, hi + pad)


def main():
    p = argparse.ArgumentParser(description="Plot the training curves of a PPO run")
    p.add_argument("--run_dir", required=True, help="the run folder (has tb/ and config.yaml)")
    p.add_argument("--val_dir", default="", help="validation JSONs of the run's snapshots (validate_checkpoints.py)")
    p.add_argument("--ref_json", default="", help="a reference controller's eval JSON on the same validation throws")
    p.add_argument("--ref_label", default="classical_multi")
    p.add_argument("--out", required=True, help="output path without extension")
    p.add_argument("--window", type=int, default=20, help="updates in the rolling mean")
    args = p.parse_args()

    tb = read_tb(args.run_dir)
    x = tb.index.values / 1e6
    val = read_val(args.val_dir, steps_per_update(args.run_dir), int(tb.index.max())) if args.val_dir else pd.DataFrame()
    ref = json.load(open(args.ref_json)) if args.ref_json and os.path.exists(args.ref_json) else None

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), facecolor=SURFACE)
    ax = axes[0, 0]
    style(ax, "Catch rate (%)")
    if "train_catch" in tb:
        line(ax, x, 100 * tb["train_catch"].values, TRAIN, args.window, clip=False)
        ax.annotate("training", (x[-1], 100 * tb["train_catch"].rolling(args.window, min_periods=1).mean().values[-1]),
                    xytext=(4, 6), textcoords="offset points", color=INK, fontsize=8, va="center")
    if len(val):
        vx = val["step"].values / 1e6
        # clip: at 500/500 the Wilson upper bound rounds to just under 1.0, a tiny negative error bar
        yerr = [np.clip(100 * (val["val_catch"] - val["lo"]), 0, None), np.clip(100 * (val["hi"] - val["val_catch"]), 0, None)]
        ax.errorbar(vx, 100 * val["val_catch"], yerr=yerr,
                    color=VAL, linewidth=2, marker="o", markersize=5, capsize=0, elinewidth=1,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate("validation", (vx[-1], 100 * val["val_catch"].values[-1]), xytext=(4, -6),
                    textcoords="offset points", color=INK, fontsize=8, va="center")
    if ref is not None:
        ax.axhline(100 * ref["catch_rate"], color=INK2, linewidth=1.2, linestyle="--")
        ax.annotate("{} {:.1f}%".format(args.ref_label, 100 * ref["catch_rate"]), (0.45 * x[-1], 100 * ref["catch_rate"]),
                    xytext=(0, -11), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_ylim(0, 102)
    handles = [plt.Line2D([], [], color=TRAIN, linewidth=2, label="training (stochastic, last 200 episodes)")]
    if len(val):
        handles.append(plt.Line2D([], [], color=VAL, linewidth=2, marker="o", label="validation (mean action, 500 held-out throws, 95% CI)"))
    if ref is not None:
        handles.append(plt.Line2D([], [], color=INK2, linewidth=1.2, linestyle="--", label=args.ref_label + " on the validation throws"))

    panels = [(axes[0, 1], "episode_reward", "Episode reward (training)"),
              (axes[0, 2], "critic_loss", "Value (critic) loss"),
              (axes[1, 0], "actor_loss", "Policy (actor) loss"),
              (axes[1, 1], "kl", "KL, rollout policy vs updated policy (log scale)")]
    for ax, col, title in panels:
        style(ax, title)
        if col in tb:
            line(ax, x, tb[col].values, TRAIN, args.window, clip=(col != "kl"))
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylim(bottom=max(1e-5, float(np.nanmin(tb["kl"].values[tb["kl"].values > 0])) / 2) if "kl" in tb else 1e-5)
    axes[1, 1].axhline(0.032, color=INK2, linewidth=1.2, linestyle="--")
    axes[1, 1].annotate("early stop (KL > 0.032)", (0, 0.032), xytext=(2, 4), textcoords="offset points", color=INK2, fontsize=8)

    ax = axes[1, 2]
    style(ax, "Exploration std (range over the 5 actions)")
    if "std_min" in tb and "std_max" in tb:
        ax.fill_between(x, tb["std_min"], tb["std_max"], color=TRAIN, alpha=0.25, linewidth=0)
        ax.plot(x, (tb["std_min"] + tb["std_max"]) / 2, color=TRAIN, linewidth=2)
    elif "sigma_mean" in tb:
        ax.plot(x, tb["sigma_mean"], color=TRAIN, linewidth=2)
    for ax in axes[1]:
        ax.set_xlabel("training steps (millions)", color=INK2, fontsize=8)
    fig.suptitle("{}  (thin: every update; thick: {}-update rolling mean)".format(
        os.path.relpath(args.run_dir), args.window), x=0.01, ha="left", fontsize=10, color=INK2)
    # Legend of the catch-rate panel, below the panels so it never covers a curve
    fig.legend(handles=handles, fontsize=8, frameon=False, loc="lower center", ncol=len(handles), labelcolor=INK)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out + ".png", dpi=130, facecolor=SURFACE)
    tb.to_csv(args.out + "_train.csv")
    if len(val):
        val.to_csv(args.out + "_val.csv", index=False)
    print("saved", args.out + ".png")


if __name__ == "__main__":
    main()
