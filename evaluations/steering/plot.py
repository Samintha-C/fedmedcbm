"""Fig. "steering" (figures/steering_main.pdf): one column per dataset, the head rule at NEC~5, three seeds.

Top row, the share of test images (judge and head both right) the independent judge assigns to the
target class, against the edit's size relative to the feature (||dh|| / ||h||):
  - concept edit, target top-1: linear judge with +/- 1 s.d. over seeds, and the MLP judge;
  - concept edit, target in the top 5 (linear judge);
  - controls (linear judge): a random direction of the same size, and the edit from a third class's explanation.
Bottom row, the two controls alone on a 0-5% axis, because they overlap near zero in the top row.
All top-1 curves start at 0: the judge is right before the edit and the target is another class.
The dotted line is scale 1, where the edited concepts reach their calibrated values.

Lab figure style (Lato, soft text); each line also has its own marker or dash, so identity is not
colour alone.
usage: python3 -I plot.py --results <steer.py results dir> --out <dir>
"""
import argparse
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

LATO = "/usr/local/texlive/2026/texmf-dist/fonts/truetype/typoland/lato"
if os.path.isdir(LATO):
    for f in ("Lato-Regular.ttf", "Lato-Bold.ttf", "Lato-Light.ttf"):
        font_manager.fontManager.addfont(os.path.join(LATO, f))
GREEN, LIGHTGREY = "#388C4E", "#BDBDBD"
BLUE, OCHRE, RED, PURPLE = "#3D74C4", "#B8860B", "#D1495B", "#7B5EA7"
HEAD, BODY, GRID = "#434343", "#595959", "#E6E6E6"
plt.rcParams.update({
    "font.family": "Lato" if os.path.isdir(LATO) else "sans-serif", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
    "axes.edgecolor": LIGHTGREY, "axes.labelcolor": BODY, "xtick.color": BODY, "ytick.color": BODY,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42,
})
CVPR_TEXTWIDTH_IN = 496.856 / 72.27
DATASETS = [("c10", "CIFAR-10"), ("c100", "CIFAR-100"), ("places365", "Places365"), ("imagenet", "ImageNet")]
SEEDS = (42, 31, 7)
MK = dict(markersize=3, markeredgecolor="white", markeredgewidth=0.4)


def load(results, ds, judge):
    return [json.load(open(os.path.join(results, f"steer_{ds}_s{s}_{judge}_both_right.json"))) for s in SEEDS]


def series(runs, direction, key):
    scales = sorted(runs[0]["scales"], key=float)
    vals = [[100 * r["scales"][s][direction][key] for r in runs] for s in scales]
    return [st.mean(v) for v in vals], [st.stdev(v) for v in vals]


def edit_sizes(runs):
    scales = sorted(runs[0]["scales"], key=float)
    return [st.mean(r["scales"][s]["edit_norm_rel"] for r in runs) for s in scales], scales.index("1")


def style(ax):
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="steer.py's results dir (<data>/saved/results)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(2, 4, figsize=(CVPR_TEXTWIDTH_IN, 2.05), sharex="col", gridspec_kw=dict(height_ratios=[3.4, 1]))
    for i, (ds, name) in enumerate(DATASETS):
        ax, zoom = axes[0][i], axes[1][i]
        lin, mlp = load(args.results, ds, "linear"), load(args.results, ds, "mlp")
        x, at_one = edit_sizes(lin)
        x0 = [0.0] + x
        rnd = [0.0] + series(lin, "random", "judge_top1")[0]
        third = [0.0] + series(lin, "other class", "judge_top1")[0]
        ax.axvline(x[at_one], color=LIGHTGREY, linestyle=(0, (1, 2)), linewidth=0.8, zorder=0, label="Scale 1 (calibrated edit)")
        m, s = series(lin, "concepts", "judge_top1")
        ax.fill_between(x0, [0.0] + [a - b for a, b in zip(m, s)], [0.0] + [a + b for a, b in zip(m, s)],
                        color=GREEN, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(x0, [0.0] + m, color=GREEN, linewidth=1.5, marker="o", label="Concept edit, linear judge", zorder=5, **MK)
        ax.plot(x0, [0.0] + series(mlp, "concepts", "judge_top1")[0], color=BLUE, linewidth=1.1, marker="s",
                label="Concept edit, MLP judge", zorder=4, **MK)
        ax.plot(x, series(lin, "concepts", "judge_top5")[0], color=OCHRE, linestyle="--", linewidth=1.1,
                label="Concept edit, target in top 5", zorder=3)
        for a in (ax, zoom):
            a.plot(x0, rnd, color=RED, linewidth=1.0, marker="D", label="Random direction", zorder=2, **dict(MK, markersize=2.6))
            a.plot(x0, third, color=PURPLE, linewidth=1.0, linestyle="--", marker="^", label="Third class's edit", zorder=2, **MK)
        ax.set_title(f"{name}  (NEC {st.mean(r['nec'] for r in lin):.1f})", color=HEAD, loc="left", pad=3)
        ax.set_xlim(0, max(x) * 1.04)
        style(ax)
        zoom.axvline(x[at_one], color=LIGHTGREY, linestyle=(0, (1, 2)), linewidth=0.8, zorder=0)
        zoom.set_ylim(-0.3, 5)
        zoom.set_yticks([0, 5])
        zoom.grid(axis="y", color=GRID, linewidth=0.5)
        zoom.set_axisbelow(True)
        for sp in ("top", "right"):
            zoom.spines[sp].set_visible(False)
        zoom.set_xlabel(r"Edit size  $\|\Delta h\|\,/\,\|h\|$", labelpad=1.5)
        if i:
            ax.tick_params(labelleft=False)
            zoom.tick_params(labelleft=False)
    axes[0][0].set_ylabel("Judge on target (%)")
    axes[1][0].set_ylabel("Zoom (%)")
    fig.align_ylabels(axes[:, 0])
    h, l = axes[0][0].get_legend_handles_labels()
    order = [1, 2, 3, 4, 5, 0]
    fig.legend([h[k] for k in order], [l[k] for k in order], loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.03), handlelength=2.0, columnspacing=0.9, handletextpad=0.45,
               labelcolor=BODY, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.925), w_pad=0.8, h_pad=0.35)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "steering_main.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print("wrote", out)


if __name__ == "__main__":
    main()
