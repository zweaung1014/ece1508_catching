"""
Author: Zwe Min Htet Aung
ABOUT: report figures for one experiment: its test results (eval_catch.py CSVs in results/<exp>/) and the
training curves of its RL run (the CSVs plot_training.py wrote). Writes PNGs into results/<exp>/figures/:
  test_catch_rates.png  catch rate of every controller on the test throws, with 95% CIs
  catch_vs_noise.png    catch rate vs the ball's position / velocity noise level: rl and the classical catchers
  misses_map.png        where the missed throws were aimed (top view) and how noisy their ball readings were
  how_they_move.png     IK failures, hoop travel and action change per throw (secondary; not in the decision rule)
  learning_curve.png    the RL run's training vs validation catch rate, with the reference catcher
Colors follow the controller in every figure: rl blue, classical_multi orange, classical_single aqua, others gray
(dataviz reference palette, light mode; validated all-pairs: CVD dE 9.2, normal-vision dE 24.0).

Usage (from catch_it/):
  python plot_results.py --exp exp05_ball_noise --val_ref outputs/exp05_val/val_ref/classical_multi.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_results import wilson

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MUTED = "#a3a29c"                     # controllers shown for context only
COLOR = {"rl": "#2a78d6", "classical_multi": "#eb6834", "classical_single": "#1baf7a"}
ORDER = ["rl", "classical_multi", "classical_perfect", "oracle", "classical_single", "random"]
DESC = {"rl": "RL (PPO, one noisy frame)",
        "classical_multi": "classical_multi (fits all readings)",
        "classical_perfect": "classical_perfect (true ball state)",
        "oracle": "oracle (knows the aim point)",
        "classical_single": "classical_single (one reading)",
        "random": "random actions"}
MARK = {"rl": "o", "classical_multi": "^", "classical_single": "s"}   # same marker per controller everywhere
DODGE = {"rl": -0.012, "classical_multi": 0.012}                     # sideways offset, fraction of the x range
DPI = 200


def style(ax, title, xgrid=True, ygrid=True):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print("saved", path)


def rate(df):
    k, n = int(df["success"].sum()), len(df)
    lo, hi = wilson(k, n)
    return k / n, lo, hi, n


def fig_catch_rates(runs, out_dir):
    order = [c for c in ORDER if c in runs]
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    n = len(runs[order[0]])
    style(ax, "Catch rate on the {} test throws (bars: 95% CI)".format(n), ygrid=False)
    for i, c in enumerate(order):
        p, lo, hi, _ = rate(runs[c])
        y = len(order) - 1 - i
        ax.barh(y, 100 * p, height=0.62, color=COLOR.get(c, MUTED))
        ax.errorbar(100 * p, y, xerr=[[max(0, 100 * (p - lo))], [max(0, 100 * (hi - p))]],
                    color=INK, elinewidth=1, capsize=3)
        ax.text(100 * p + 2, y, "{:.1f}%".format(100 * p), va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DESC.get(c, c) for c in reversed(order)], fontsize=9, color=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("catch rate (%)", color=INK2, fontsize=9)
    fig.tight_layout()
    save(fig, out_dir, "test_catch_rates.png")


def by_bin(df, col, edges):
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = df[(df[col] >= a) & ((df[col] < b) | (b == edges[-1]))]
        if len(s):
            p, lo, hi, n = rate(s)
            rows.append(((a + b) / 2, p, lo, hi, n))
    return np.array(rows)


def fig_catch_vs_noise(runs, out_dir):
    show = [c for c in ["rl", "classical_multi", "classical_single"] if c in runs]
    panels = [("pos_noise_std", 1000, np.linspace(5, 20, 6), "Ball position noise std (mm)"),
              ("vel_noise_std", 1, np.linspace(0.1, 0.4, 7), "Ball velocity noise std (m/s)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax, (col, scale, edges, title) in zip(axes, panels):
        style(ax, "Catch rate by " + title[0].lower() + title[1:])
        for c in show:
            df = runs[c].assign(_x=runs[c][col] * scale)
            r = by_bin(df, "_x", edges)
            r[:, 0] += DODGE.get(c, 0) * (edges[-1] - edges[0])  # rl and classical_multi overlap near 100%
            ax.fill_between(r[:, 0], 100 * r[:, 2], 100 * r[:, 3], color=COLOR[c], alpha=0.18, linewidth=0)
            ax.plot(r[:, 0], 100 * r[:, 1], color=COLOR[c], linewidth=2, marker=MARK[c], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3 if c == "rl" else 2)
            dy = {"rl": 7, "classical_multi": -8}.get(c, 8)        # rl and classical_multi overlap near 100%
            ax.annotate(c, (r[-1, 0], 100 * r[-1, 1]), xytext=(6, dy), textcoords="offset points",
                        fontsize=8, color=INK, va="center")
        ax.set_ylim(0, 104)
        ax.set_xlim(edges[0], edges[-1] + 0.22 * (edges[-1] - edges[0]))
        ax.set_xticks(edges)
        ax.set_xlabel(title + ", drawn per throw; bands: 95% CI", color=INK2, fontsize=8)
    axes[0].set_ylabel("catch rate (%)", color=INK2, fontsize=9)
    handles = [plt.Line2D([], [], color=COLOR[c], linewidth=2, marker=MARK[c], label=DESC[c]) for c in show]
    fig.legend(handles=handles, fontsize=8, frameon=False, loc="lower center", ncol=len(handles), labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out_dir, "catch_vs_noise.png")


def fig_misses(runs, out_dir):
    base = runs["rl"]
    tx = base["target_r"] * np.sin(np.radians(base["target_az_deg"]))    # az = atan2(x, y) in eval_catch.py
    ty = base["target_r"] * np.cos(np.radians(base["target_az_deg"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE)
    marks = [(c, MARK[c]) for c in ("rl", "classical_multi")]
    ax = axes[0]
    style(ax, "Where the missed throws were aimed (top view)")
    ax.scatter(tx, ty, s=3, color=MUTED, alpha=0.35, linewidths=0)
    ax.scatter([0], [0], marker="s", s=40, color=INK)
    ax.annotate("arm base", (0, 0), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax2 = axes[1]
    style(ax2, "Missed throws: aim distance vs velocity noise")
    ax2.scatter(base["target_r"], base["vel_noise_std"], s=3, color=MUTED, alpha=0.35, linewidths=0)
    handles = [plt.Line2D([], [], color=MUTED, marker="o", linestyle="", markersize=4,
                          label="all {} test throws".format(len(base)))]
    for c, m in marks:
        if c not in runs:
            continue
        miss = runs[c].set_index("seed")["success"] == 0
        idx = base["seed"].map(miss).fillna(False).values.astype(bool)
        kw = dict(s=55, marker=m, color=COLOR[c], edgecolors=SURFACE, linewidths=1.2, zorder=3)
        ax.scatter(tx[idx], ty[idx], **kw)
        ax2.scatter(base["target_r"][idx], base["vel_noise_std"][idx], **kw)
        handles.append(plt.Line2D([], [], color=COLOR[c], marker=m, linestyle="", markersize=7,
                                  label="{} missed ({})".format(c, int(idx.sum()))))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m), arm-base frame", color=INK2, fontsize=8)
    ax.set_ylabel("y (m)", color=INK2, fontsize=8)
    ax2.set_xlabel("aim point distance from the arm base, horizontal (m)", color=INK2, fontsize=8)
    ax2.set_ylabel("ball velocity noise std (m/s)", color=INK2, fontsize=8)
    fig.legend(handles=handles, fontsize=8, frameon=False, loc="lower center", ncol=len(handles), labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out_dir, "misses_map.png")


def fig_motion(runs, out_dir):
    show = [c for c in ["rl", "classical_multi", "classical_perfect"] if c in runs]
    metrics = [("ik_fails", "IK failures per throw"), ("hoop_travel", "Hoop travel per throw (m)"),
               ("action_smooth", "Squared action change per step")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), facecolor=SURFACE)
    for ax, (col, title) in zip(axes, metrics):
        style(ax, title, xgrid=False)
        for i, c in enumerate(show):
            v = runs[c][col]
            m, ci = v.mean(), 1.96 * v.std() / np.sqrt(len(v))
            ax.bar(i, m, width=0.62, color=COLOR.get(c, MUTED))
            ax.errorbar(i, m, yerr=ci, color=INK, elinewidth=1, capsize=3)
            ax.annotate("{:.2f}".format(m), (i, m + ci), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(show)))
        ax.set_xticklabels([c.replace("_", "_\n") for c in show], fontsize=8, color=INK)
        ax.margins(y=0.15)
    fig.suptitle("How the hoop moves, test throws (secondary measures, not part of the decision rule; bars: 95% CI)",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out_dir, "how_they_move.png")


def fig_learning(res_dir, val_ref, picked, out_dir):
    tr_path, va_path = (os.path.join(res_dir, "training_curves_" + s + ".csv") for s in ("train", "val"))
    if not (os.path.exists(tr_path) and os.path.exists(va_path)):
        print("skip learning_curve.png: no training_curves_*.csv in", res_dir)
        return
    tr, va = pd.read_csv(tr_path), pd.read_csv(va_path)
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax, "RL catch rate during training")
    x = tr["step"] / 1e6
    ax.plot(x, 100 * tr["train_catch"], color=INK2, linewidth=0.8, alpha=0.2)
    ax.plot(x, 100 * tr["train_catch"].rolling(20, min_periods=1).mean(), color=INK2, linewidth=1.6)
    vx = va["step"] / 1e6
    ax.errorbar(vx, 100 * va["val_catch"],
                yerr=[np.clip(100 * (va["val_catch"] - va["lo"]), 0, None), np.clip(100 * (va["hi"] - va["val_catch"]), 0, None)],
                color=COLOR["rl"], linewidth=2, marker="o", markersize=5, elinewidth=1, capsize=0,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    handles = [plt.Line2D([], [], color=INK2, linewidth=1.6, label="training (exploring policy, last 200 throws)"),
               plt.Line2D([], [], color=COLOR["rl"], linewidth=2, marker="o",
                          label="validation (mean action, 500 held-out throws, 95% CI)")]
    if val_ref and os.path.exists(val_ref):
        ref = json.load(open(val_ref))["catch_rate"]
        ax.axhline(100 * ref, color=COLOR["classical_multi"], linewidth=1.4, linestyle="--")
        handles.append(plt.Line2D([], [], color=COLOR["classical_multi"], linewidth=1.4, linestyle="--",
                                  label="classical_multi on the same validation throws ({:.1f}%)".format(100 * ref)))
    if picked:
        row = va[va["checkpoint"] == picked]
        if len(row):
            px, py = row["step"].values[0] / 1e6, 100 * row["val_catch"].values[0]
            ax.annotate("picked: {:.0f}M steps, {:.1f}%".format(px, py), (px, py), xytext=(0, -38),
                        textcoords="offset points", ha="center", fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.set_ylim(0, 104)
    ax.set_xlabel("training steps (millions)", color=INK2, fontsize=9)
    ax.set_ylabel("catch rate (%)", color=INK2, fontsize=9)
    fig.legend(handles=handles, fontsize=8, frameon=False, loc="lower center", ncol=2, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out_dir, "learning_curve.png")


def main():
    p = argparse.ArgumentParser(description="Report figures for one experiment")
    p.add_argument("--exp", required=True, help="folder name under results/")
    p.add_argument("--val_ref", default="", help="reference controller's eval JSON on the validation throws")
    args = p.parse_args()
    res_dir = os.path.join("results", args.exp)
    out_dir = os.path.join(res_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    runs = {}
    for c in ORDER:
        csv = os.path.join(res_dir, c + ".csv")
        if os.path.exists(csv) and os.path.exists(os.path.join(res_dir, c + ".json")):
            runs[c] = pd.read_csv(csv).sort_values("seed").reset_index(drop=True)
    picked = ""
    if os.path.exists(os.path.join(res_dir, "picked.txt")):
        picked = os.path.basename(open(os.path.join(res_dir, "picked.txt")).read().split()[1])
    fig_catch_rates(runs, out_dir)
    fig_catch_vs_noise(runs, out_dir)
    if "rl" in runs:
        fig_misses(runs, out_dir)
    fig_motion(runs, out_dir)
    fig_learning(res_dir, args.val_ref, picked, out_dir)


if __name__ == "__main__":
    main()
