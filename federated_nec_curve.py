"""
Federated NEC curve (Option B): assemble an accuracy-vs-NEC curve from a set of
feddualavg runs (e.g. a dual_lam sweep).

Each run contributes ONE genuinely-federated operating point:
    NEC      = Non-zero weights / num_classes   (avg effective concepts per class)
    accuracy = test accuracy of the federated global model

This is the federated analogue of VLG-CBM's NEC table. Unlike the built-in NEC
eval (evaluations/sparse_utils.measure_acc), which retrains a *centralized* SAGA
head on pooled features, every point here is a real federated model read straight
from its metrics.txt — no retraining, no centralization.

NEC / sparsity conventions match VLG-CBM exactly (sparse_utils.py:53):
    NEC = num_concepts * (nnz / total_weights) = nnz / num_classes

Usage (default: dlamsweep layout on the PVC):
    python federated_nec_curve.py \
        --base_dir /sc-rwx-vol/fedmedcbm/models \
        --output_dir /sc-rwx-vol/fedmedcbm/eval_results/visualizations/fed_nec

Generic mode (recursively scan any tree of run dirs, group as one curve):
    python federated_nec_curve.py --runs_dir /path/to/runs --label cifar100
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans"],
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

# Match plot_dlamsweep.py styling
COLORS  = {"cifar10": "#4C72B0", "cifar100": "#DD8452", "cub": "#55A868",
           "imagenet": "#C44E52", "places365": "#8172B3"}
MARKERS = {"cifar10": "o", "cifar100": "s", "cub": "^", "imagenet": "D", "places365": "v"}
LABELS  = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "cub": "CUB-200",
           "imagenet": "ImageNet", "places365": "Places365"}

DATASETS = ["cifar10", "cifar100", "cub", "imagenet", "places365"]


# ── Point extraction ────────────────────────────────────────────────────────
def _num_classes_from_metrics(m, run_dir):
    """num_classes, preferring per_class_accuracies length, then args.txt."""
    pca = m.get("per_class_accuracies")
    if isinstance(pca, dict):
        inner = pca.get("Per class accuracy")
        if isinstance(inner, dict) and inner:
            return len(inner)
    args_path = os.path.join(run_dir, "args.txt")
    if os.path.exists(args_path):
        try:
            with open(args_path) as f:
                a = json.load(f)
            if a.get("num_classes"):
                return int(a["num_classes"])
        except Exception:
            pass
    return None


def read_point(metrics_path):
    """Parse one run's metrics.txt into a federated NEC point, or None."""
    try:
        with open(metrics_path) as f:
            m = json.load(f)
    except Exception as e:
        print(f"  [skip] unreadable {metrics_path}: {e}")
        return None

    sp = m.get("sparsity")
    metrics = m.get("metrics", {})
    if not sp or "test_accuracy" not in metrics:
        # No sparse head / not a federated sparse run (e.g. centralized fedavg
        # without a sparsity block) — can't place it on the NEC axis.
        return None

    nnz = int(sp["Non-zero weights"])
    total = int(sp["Total weights"])
    run_dir = os.path.dirname(metrics_path)
    num_classes = _num_classes_from_metrics(m, run_dir)
    if not num_classes:
        print(f"  [skip] cannot determine num_classes for {metrics_path}")
        return None

    nec = nnz / num_classes
    return {
        "run": os.path.basename(run_dir),
        "method": m.get("final_layer_method", "unknown"),
        "dual_lam": m.get("dual_lam"),
        "nec": nec,
        "accuracy_pct": float(metrics["test_accuracy"]) * 100.0,
        "nnz": nnz,
        "total_weights": total,
        "num_classes": num_classes,
        "num_concepts": total // num_classes,
        "pct_nonzero": float(sp.get("Percentage non-zero", nnz / total)) * 100.0,
    }


def _latest_run_metrics(run_parent):
    """Given a dir whose children are run dirs, return the latest run's metrics.txt."""
    subdirs = sorted(d for d in glob.glob(os.path.join(run_parent, "*")) if os.path.isdir(d))
    for d in reversed(subdirs):
        mp = os.path.join(d, "metrics.txt")
        if os.path.exists(mp):
            return mp
    return None


def collect_dlamsweep(base_dir):
    """Scan {base_dir}/{dataset}/dlamsweep/lam*/{run}/metrics.txt → {dataset: [points]}."""
    data = {}
    for ds in DATASETS:
        sweep_dir = os.path.join(base_dir, ds, "dlamsweep")
        if not os.path.isdir(sweep_dir):
            continue
        points = []
        for lam_dir in sorted(glob.glob(os.path.join(sweep_dir, "lam*"))):
            mp = _latest_run_metrics(lam_dir)
            if mp is None:
                print(f"  [warn] no metrics.txt under {lam_dir}")
                continue
            pt = read_point(mp)
            if pt:
                points.append(pt)
                print(f"  {ds} λ={pt['dual_lam']}: NEC={pt['nec']:.2f}  acc={pt['accuracy_pct']:.2f}%")
        if points:
            data[ds] = sorted(points, key=lambda p: p["nec"])
    return data


def collect_runs_dir(runs_dir, label):
    """Recursively find all metrics.txt under runs_dir, group as one curve."""
    points = []
    for mp in sorted(glob.glob(os.path.join(runs_dir, "**", "metrics.txt"), recursive=True)):
        pt = read_point(mp)
        if pt:
            points.append(pt)
            print(f"  {label} λ={pt['dual_lam']}: NEC={pt['nec']:.2f}  acc={pt['accuracy_pct']:.2f}%")
    return {label: sorted(points, key=lambda p: p["nec"])} if points else {}


# ── Outputs ─────────────────────────────────────────────────────────────────
def write_csv(data, out_csv):
    rows = []
    for ds, points in data.items():
        for p in points:
            rows.append({"dataset": ds, **p})
    if not rows:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = ["dataset", "method", "dual_lam", "nec", "accuracy_pct",
            "nnz", "total_weights", "num_classes", "num_concepts", "pct_nonzero", "run"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"\nCSV written to {out_csv}")


def plot_curve(data, out_png):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for ds, points in data.items():
        if not points:
            continue
        xs = [p["nec"] for p in points]
        ys = [p["accuracy_pct"] for p in points]
        color = COLORS.get(ds, None)
        marker = MARKERS.get(ds, "o")
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2, markersize=7,
                label=LABELS.get(ds, ds), markeredgecolor="white", markeredgewidth=0.8)
        for p in points:
            lam = p["dual_lam"]
            if lam is not None:
                ax.annotate(f"λ={lam}", xy=(p["nec"], p["accuracy_pct"]),
                            xytext=(5, 4), textcoords="offset points",
                            fontsize=7.5, color=color,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"))

    ax.set_xlabel("NEC  (effective concepts per class)  ←  more interpretable")
    ax.set_ylabel("Test accuracy  (%)")
    ax.set_title("Federated Accuracy–NEC Curve  (each point = a real federated model)", pad=10)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="0.88", alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=False))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Plot written to {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/sc-rwx-vol/fedmedcbm/models",
                        help="Root models dir (dlamsweep layout). Ignored if --runs_dir is set.")
    parser.add_argument("--runs_dir", default=None,
                        help="Recursively scan this dir for metrics.txt and treat as one curve.")
    parser.add_argument("--label", default="runs",
                        help="Curve label for --runs_dir mode.")
    parser.add_argument("--output_dir",
                        default="/sc-rwx-vol/fedmedcbm/eval_results/visualizations/fed_nec")
    args = parser.parse_args()

    print("Collecting federated NEC points...")
    if args.runs_dir:
        data = collect_runs_dir(args.runs_dir, args.label)
    else:
        data = collect_dlamsweep(args.base_dir)

    if not data:
        print("No federated NEC points found.")
        return

    write_csv(data, os.path.join(args.output_dir, "federated_nec.csv"))
    plot_curve(data, os.path.join(args.output_dir, "federated_nec_curve.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
