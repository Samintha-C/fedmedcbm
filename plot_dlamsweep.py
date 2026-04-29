"""
Plot accuracy and sparsity vs. dual_lam across datasets from a dlamsweep run.

Usage (default paths point at PVC):
    python plot_dlamsweep.py

Override paths:
    python plot_dlamsweep.py \
        --base_dir /sc-rwx-vol/fedmedcbm/models \
        --output_dir /sc-rwx-vol/fedmedcbm/eval_results/visualizations/dlam_sweep
"""

import argparse
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

# ── Global style ──────────────────────────────────────────────────────────────
rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans"],
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "xtick.minor.size":  2,
    "ytick.minor.size":  2,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.85",
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

# Muted, colour-blind-friendly palette
COLORS  = {"cifar10": "#4C72B0", "cifar100": "#DD8452", "cub": "#55A868"}
MARKERS = {"cifar10": "o",       "cifar100": "s",        "cub": "^"}
LABELS  = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "cub": "CUB-200"}

DATASETS   = ["cifar10", "cifar100", "cub"]
LAM_VALUES = [0.0005, 0.001, 0.002, 0.005, 0.01]
LAM_STRS   = ["0.0005", "0.001", "0.002", "0.005", "0.01"]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_metrics(base_dir, dataset, lam_str):
    lam_dir = os.path.join(base_dir, dataset, "dlamsweep", f"lam{lam_str}")
    if not os.path.isdir(lam_dir):
        return None
    subdirs = sorted(
        d for d in os.listdir(lam_dir)
        if os.path.isdir(os.path.join(lam_dir, d))
    )
    if not subdirs:
        return None
    metrics_path = os.path.join(lam_dir, subdirs[-1], "metrics.txt")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        return json.load(f)


def collect(base_dir):
    data = {ds: {"lams": [], "accuracy": [], "sparsity": []} for ds in DATASETS}
    for ds in DATASETS:
        for lam, lam_str in zip(LAM_VALUES, LAM_STRS):
            m = load_metrics(base_dir, ds, lam_str)
            if m is None:
                print(f"[WARN] no metrics for {ds} lam={lam_str} — skipping")
                continue
            acc      = float(m["metrics"]["test_accuracy"]) * 100
            sparsity = float(m["sparsity"]["Percentage non-zero"]) * 100
            data[ds]["lams"].append(lam)
            data[ds]["accuracy"].append(acc)
            data[ds]["sparsity"].append(sparsity)
            print(f"  {ds} lam={lam_str}: acc={acc:.2f}%  sparsity={sparsity:.2f}%")
    return data


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _style_ax(ax, grid_axis="both"):
    ax.grid(True, which="major", linestyle="--", linewidth=0.5,
            color="0.88", alpha=0.9, axis=grid_axis)
    ax.set_axisbelow(True)


def _save(fig, output_dir, filename):
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 1: separate line charts ───────────────────────────────────────────────
def plot_metric(data, key, ylabel, title, filename, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ds in DATASETS:
        xs, ys = data[ds]["lams"], data[ds][key]
        if not xs:
            continue
        ax.plot(xs, ys, marker=MARKERS[ds], color=COLORS[ds],
                linewidth=2, markersize=7, label=LABELS[ds],
                markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.5)))
    ax.set_xlabel("Regularisation strength  λ₁")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.legend(loc="best")
    _style_ax(ax)
    plt.tight_layout()
    _save(fig, output_dir, filename)


# ── Plot 2: Pareto scatter ─────────────────────────────────────────────────────
def plot_pareto(data, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    min_lam, max_lam = min(LAM_VALUES), max(LAM_VALUES)

    for ds in DATASETS:
        xs, ys, lams = data[ds]["sparsity"], data[ds]["accuracy"], data[ds]["lams"]
        if not xs:
            continue
        # Draw a faint connecting line so the λ trajectory is readable
        ax.plot(xs, ys, color=COLORS[ds], linewidth=1, alpha=0.35, zorder=1)
        sizes = [50 + 200 * (lam - min_lam) / (max_lam - min_lam) for lam in lams]
        ax.scatter(xs, ys, s=sizes, color=COLORS[ds], marker=MARKERS[ds],
                   alpha=0.9, zorder=3, label=LABELS[ds],
                   edgecolors="white", linewidths=0.8)
        for x, y, lam in zip(xs, ys, lams):
            ax.annotate(
                f"λ={lam}", xy=(x, y),
                xytext=(6, 4), textcoords="offset points",
                fontsize=8, color=COLORS[ds],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"),
            )

    ax.set_xlabel("Non-zero weights  (%)  ←  sparser")
    ax.set_ylabel("Test accuracy  (%)")
    ax.set_title("Accuracy–Sparsity Frontier  (dot size ∝ λ₁)", pad=10)
    ax.legend(loc="lower right")
    _style_ax(ax)
    plt.tight_layout()
    _save(fig, output_dir, "dlamsweep_pareto.png")


# ── Plot 3: Connected dot panels ──────────────────────────────────────────────
def plot_connected_dots(data, output_dir):
    n = len(DATASETS)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 5), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax, ds in zip(axes, DATASETS):
        xs, ys, lams = data[ds]["sparsity"], data[ds]["accuracy"], data[ds]["lams"]
        color = COLORS[ds]

        if not xs:
            ax.set_title(LABELS[ds], color=color, pad=8)
            continue

        # Trajectory line
        ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.45, zorder=2)

        for i, (x, y, lam) in enumerate(zip(xs, ys, lams)):
            t = i / max(len(lams) - 1, 1)
            # Dots lighten at small λ, saturate at large λ
            dot_alpha = 0.35 + 0.65 * t
            ax.scatter(x, y, s=90, color=color, marker=MARKERS[ds],
                       alpha=dot_alpha, zorder=3,
                       edgecolors="white", linewidths=0.8)
            # Alternate label offset to reduce overlap
            dy = 6 if i % 2 == 0 else -14
            ax.annotate(
                f"λ={lam}", xy=(x, y),
                xytext=(5, dy), textcoords="offset points",
                fontsize=7.5, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.65, ec="none"),
            )

        ax.set_xlabel("Non-zero weights  (%)")
        ax.set_title(LABELS[ds], color=color, pad=8, fontweight="semibold")
        _style_ax(ax, grid_axis="both")
        # Only left panel gets a y-label
        if ax is axes[0]:
            ax.set_ylabel("Test accuracy  (%)")
        else:
            ax.tick_params(labelleft=False)

    fig.suptitle(
        "Accuracy–Sparsity Tradeoff  (λ₁ increases along each curve  →)",
        fontsize=13, y=1.02,
    )
    _save(fig, output_dir, "dlamsweep_connected.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",   default="/sc-rwx-vol/fedmedcbm/models")
    parser.add_argument("--output_dir", default="/sc-rwx-vol/fedmedcbm/eval_results/visualizations/dlam_sweep")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Collecting metrics...")
    data = collect(args.base_dir)

    plot_metric(data, "accuracy", "Test accuracy  (%)",
                "Test Accuracy vs. Regularisation Strength",
                "dlamsweep_accuracy.png", args.output_dir)

    plot_metric(data, "sparsity", "Non-zero weights  (%)",
                "Sparsity vs. Regularisation Strength",
                "dlamsweep_sparsity.png", args.output_dir)

    plot_pareto(data, args.output_dir)
    plot_connected_dots(data, args.output_dir)

    # Mirror all plots into each dataset's dlamsweep dir
    all_fnames = (
        "dlamsweep_accuracy.png",
        "dlamsweep_sparsity.png",
        "dlamsweep_pareto.png",
        "dlamsweep_connected.png",
    )
    for ds in DATASETS:
        ds_out = os.path.join(args.base_dir, ds, "dlamsweep")
        if os.path.isdir(ds_out):
            for fname in all_fnames:
                src = os.path.join(args.output_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(ds_out, fname))
                    print(f"Mirrored {fname} → {ds_out}/")

    print("\nDone.")
    print("\nTo copy results locally, run:")
    print("kubectl cp wenglab-interpretable-ai/sc-rwx-copy-pod:"
          "/sc-rwx-vol/fedmedcbm/eval_results/visualizations/dlam_sweep "
          "/Users/saminthachandrasiri/Research/TrustworthyMLLab/fed_lfc_cbm/visualizations/dlam_sweep")


if __name__ == "__main__":
    main()
