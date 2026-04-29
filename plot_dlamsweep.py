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


DATASETS = ["cifar10", "cifar100", "cub"]
COLORS   = {"cifar10": "steelblue", "cifar100": "darkorange", "cub": "forestgreen"}
MARKERS  = {"cifar10": "o",         "cifar100": "s",           "cub": "^"}

# All lam values we expect; missing ones are silently skipped.
LAM_VALUES = [0.0005, 0.001, 0.002, 0.005, 0.01]
LAM_STRS   = ["0.0005", "0.001", "0.002", "0.005", "0.01"]


def load_metrics(base_dir, dataset, lam_str):
    lam_dir = os.path.join(base_dir, dataset, "dlamsweep", f"lam{lam_str}")
    if not os.path.isdir(lam_dir):
        return None
    # Pick the most recent model subdir (alphabetical sort → timestamp order)
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


def plot_metric(data, key, ylabel, title, filename, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds in DATASETS:
        xs = data[ds]["lams"]
        ys = data[ds][key]
        if not xs:
            continue
        ax.plot(xs, ys, marker=MARKERS[ds], color=COLORS[ds], linewidth=2,
                markersize=7, label=ds)
    ax.set_xscale("log")
    ax.set_xlabel("dual_lam (λ₁)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_pareto(data, output_dir):
    """Option 1: Pareto scatter — sparsity on x, accuracy on y, λ encoded as dot size."""
    fig, ax = plt.subplots(figsize=(8, 6))
    # Size range: smallest λ → smallest dot, largest λ → largest dot
    min_lam, max_lam = min(LAM_VALUES), max(LAM_VALUES)

    for ds in DATASETS:
        xs = data[ds]["sparsity"]   # % non-zero (lower = sparser)
        ys = data[ds]["accuracy"]
        lams = data[ds]["lams"]
        if not xs:
            continue
        sizes = [40 + 180 * (lam - min_lam) / (max_lam - min_lam) for lam in lams]
        sc = ax.scatter(xs, ys, s=sizes, color=COLORS[ds], marker=MARKERS[ds],
                        alpha=0.85, label=ds, zorder=3)
        # Annotate each point with its λ value
        for x, y, lam in zip(xs, ys, lams):
            ax.annotate(f"λ={lam}", (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=7.5, color=COLORS[ds])

    ax.set_xlabel("Non-zero Weights (%) — lower is sparser →", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("FedDualAvg: Accuracy–Sparsity Pareto (dot size ∝ λ₁)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "dlamsweep_pareto.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_connected_dots(data, output_dir):
    """Option 3: Connected dot plot — one panel per dataset, x=sparsity, y=accuracy,
    dots connected as λ increases, each dot labelled with its λ value."""
    n = len(DATASETS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)

    for ax, ds in zip(axes, DATASETS):
        xs = data[ds]["sparsity"]
        ys = data[ds]["accuracy"]
        lams = data[ds]["lams"]
        if not xs:
            ax.set_title(ds, fontsize=12)
            continue

        # Connect in order of increasing λ (already sorted)
        ax.plot(xs, ys, color=COLORS[ds], linewidth=1.5, zorder=2, alpha=0.6)
        for i, (x, y, lam) in enumerate(zip(xs, ys, lams)):
            # Fade dots from light (small λ) to full color (large λ)
            alpha = 0.4 + 0.6 * i / max(len(lams) - 1, 1)
            ax.scatter(x, y, s=80, color=COLORS[ds], marker=MARKERS[ds],
                       alpha=alpha, zorder=3)
            ax.annotate(f"λ={lam}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color=COLORS[ds])

        ax.set_xlabel("Non-zero Weights (%)", fontsize=11)
        ax.set_title(ds, fontsize=12, color=COLORS[ds])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Test Accuracy (%)", fontsize=11)
    fig.suptitle("FedDualAvg: Accuracy–Sparsity Tradeoff by Dataset (λ₁ increases →)",
                 fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "dlamsweep_connected.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",   default="/sc-rwx-vol/fedmedcbm/models")
    parser.add_argument("--output_dir", default="/sc-rwx-vol/fedmedcbm/eval_results/visualizations/dlam_sweep")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Collecting metrics...")
    data = collect(args.base_dir)

    # Separate accuracy and sparsity lines
    plot_metric(data, "accuracy", "Test Accuracy (%)",
                "FedDualAvg: Test Accuracy vs. dual_lam",
                "dlamsweep_accuracy.png", args.output_dir)

    plot_metric(data, "sparsity", "Non-zero Weights (%)",
                "FedDualAvg: Sparsity vs. dual_lam",
                "dlamsweep_sparsity.png", args.output_dir)

    # Pareto scatter (accuracy vs sparsity, λ as dot size)
    plot_pareto(data, args.output_dir)

    # Connected dot plot (one panel per dataset)
    plot_connected_dots(data, args.output_dir)

    # Mirror all plots into each dataset's own dlamsweep dir for easy reference
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
    print(f"kubectl cp wenglab-interpretable-ai/sc-rwx-copy-pod:/sc-rwx-vol/fedmedcbm/eval_results/visualizations/dlam_sweep "
          f"/Users/saminthachandrasiri/Research/TrustworthyMLLab/fed_lfc_cbm/visualizations/dlam_sweep")


if __name__ == "__main__":
    main()
