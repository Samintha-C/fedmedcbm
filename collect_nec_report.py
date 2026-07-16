"""
Bundle the federated dlam/NEC sweep results into a single self-contained PDF
(accuracy-NEC curve + full operating-point table), plus the CSV.

Reuses the collection logic in federated_nec_curve.py. No GPU / no retraining —
reads existing metrics.txt files off the PVC.

Usage:
  python collect_nec_report.py \
      --base_dir /sc-rwx-vol/fedmedcbm/models \
      --output_dir /sc-rwx-vol/fedmedcbm/eval_results/visualizations/fed_nec

Generic mode (one curve from an arbitrary tree of run dirs):
  python collect_nec_report.py --runs_dir /path/to/runs --label cifar100
"""

import argparse
import datetime
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from federated_nec_curve import (
    collect_dlamsweep, collect_runs_dir, write_csv, COLORS, MARKERS, LABELS,
)


def _curve_figure(data):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for ds, points in data.items():
        if not points:
            continue
        xs = [p["nec"] for p in points]
        ys = [p["accuracy_pct"] for p in points]
        ax.plot(xs, ys, marker=MARKERS.get(ds, "o"), color=COLORS.get(ds),
                linewidth=2, markersize=7, label=LABELS.get(ds, ds),
                markeredgecolor="white", markeredgewidth=0.8)
        for p in points:
            if p.get("dual_lam") is not None:
                ax.annotate(f"λ={p['dual_lam']}", (p["nec"], p["accuracy_pct"]),
                            xytext=(5, 4), textcoords="offset points", fontsize=7,
                            color=COLORS.get(ds))
    ax.set_xlabel("NEC  (effective concepts per class)  ←  more interpretable")
    ax.set_ylabel("Test accuracy  (%)")
    ax.set_title("Federated Accuracy–NEC Curve  (each point = a real federated model)")
    ax.grid(True, linestyle="--", linewidth=0.5, color="0.88")
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    return fig


def _table_figure(data):
    cols = ["dataset", "dual_lam", "NEC", "acc %", "nnz", "concepts"]
    rows = []
    for ds, points in data.items():
        for p in points:
            rows.append([
                ds,
                "" if p.get("dual_lam") is None else str(p["dual_lam"]),
                f"{p['nec']:.2f}",
                f"{p['accuracy_pct']:.2f}",
                str(p["nnz"]),
                str(p["num_concepts"]),
            ])
    n = max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(8.5, min(11, 1.5 + 0.32 * n)))
    ax.axis("off")
    ax.set_title("Federated NEC sweep — all operating points", pad=12)
    tbl = ax.table(cellText=rows or [["(no data)"] + [""] * (len(cols) - 1)],
                   colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    return fig


def build_pdf(data, out_pdf):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with PdfPages(out_pdf) as pdf:
        d = pdf.infodict()
        d["Title"] = "Federated NEC sweep report"
        d["CreationDate"] = datetime.datetime.now()
        for fig in (_curve_figure(data), _table_figure(data)):
            fig.text(0.99, 0.01, f"generated {stamp}", ha="right", va="bottom",
                     fontsize=6, color="0.5")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"PDF written to {out_pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/sc-rwx-vol/fedmedcbm/models")
    parser.add_argument("--runs_dir", default=None,
                        help="Recursively scan this dir instead of the dlamsweep layout.")
    parser.add_argument("--label", default="runs", help="Curve label for --runs_dir mode.")
    parser.add_argument("--output_dir",
                        default="/sc-rwx-vol/fedmedcbm/eval_results/visualizations/fed_nec")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Collecting federated NEC points...")
    data = collect_runs_dir(args.runs_dir, args.label) if args.runs_dir \
        else collect_dlamsweep(args.base_dir)
    if not data:
        print("No federated NEC points found.")
        return

    write_csv(data, os.path.join(args.output_dir, "federated_nec.csv"))
    build_pdf(data, os.path.join(args.output_dir, "federated_nec_report.pdf"))
    print("\nDone.")


if __name__ == "__main__":
    main()
