"""Sankey diagram of concept-to-class final-layer weights.

The Label-free CBM repo (evaluate_cbm.ipynb) only prints lines in SankeyMATIC's
text format and asks you to paste them into sankeymatic.com by hand. This renders
the diagram directly so it can go straight into the paper, and keeps the text
export as `--sankeymatic` for compatibility.

Conventions follow LF-CBM: negative weights are shown as "NOT <concept>" with the
absolute value as the flow width.

Three input modes:

  --load_dir <ckpt>     read final.pt / concepts.txt from a trained checkpoint
  --spec <json>         hand-specified flows (no torch required)
  --sankeymatic         emit SankeyMATIC text instead of a figure

Spec format:
    {"flows": [["writing implement", "quill", 15.13],
               ["inkpot", "quill", 4.20]]}

Usage:
    python plot_concept_sankey.py --load_dir <ckpt> --classes quill hourglass \
        --top_k 6 --output sankey.pdf
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualizations.contribution_panel import RC, LABEL_COLOR, SPINE_COLOR

# Qualitative palette for target classes. Muted to match the rest of the figures.
CLASS_COLORS = ["#C44E52", "#4C72B0", "#55A868", "#8172B3", "#DD8452",
                "#937860", "#DA8BC3", "#8C8C8C"]


def _ribbon(ax, x0, x1, ys0, ys1, yt0, yt1, color, alpha, negative=False, n=80):
    """Fill a smooth band from a source span [ys0,ys1] to a target span [yt0,yt1]."""
    t = np.linspace(0.0, 1.0, n)
    s = 3 * t**2 - 2 * t**3          # smoothstep, zero slope at both ends
    x = x0 + (x1 - x0) * t
    top = ys0 + (yt0 - ys0) * s
    bot = ys1 + (yt1 - ys1) * s
    ax.fill_between(x, top, bot, color=color, alpha=alpha, linewidth=0,
                    zorder=2)
    if negative:
        # Dashed edge distinguishes suppressed ("NOT") concepts from supporting
        # ones even in greyscale print.
        ax.plot(x, top, color=color, linewidth=0.7, linestyle=(0, (3, 2)),
                alpha=min(1.0, alpha + 0.30), zorder=3)
        ax.plot(x, bot, color=color, linewidth=0.7, linestyle=(0, (3, 2)),
                alpha=min(1.0, alpha + 0.30), zorder=3)


def draw_sankey(ax, flows, node_gap=0.30, node_width=0.045, x0=0.0, x1=1.0,
                label_size=11, alpha=0.62, value_labels=False):
    """Render bipartite concept -> class flows.

    Args:
        flows: list of (source, target, weight). Negative weights are rendered
               as "NOT <source>" with |weight| width, per LF-CBM.
    """
    prepared = []
    for src, tgt, w in flows:
        neg = w < 0
        prepared.append({
            "src": f"NOT {src}" if neg else src,
            "tgt": tgt,
            "w": abs(float(w)),
            "neg": neg,
        })
    prepared = [f for f in prepared if f["w"] > 0]
    if not prepared:
        raise SystemExit("No non-zero flows to draw.")

    # Preserve first-seen order so callers control layout by ordering `flows`.
    sources, targets = [], []
    for f in prepared:
        if f["src"] not in sources:
            sources.append(f["src"])
        if f["tgt"] not in targets:
            targets.append(f["tgt"])

    src_total = {s: sum(f["w"] for f in prepared if f["src"] == s) for s in sources}
    tgt_total = {t: sum(f["w"] for f in prepared if f["tgt"] == t) for t in targets}

    # Scale so each column spans the same height regardless of side totals.
    def _layout(names, totals, gap):
        total = sum(totals[n] for n in names)
        span = total + gap * max(len(names) - 1, 0)
        y, out = 0.0, {}
        for n in names:
            out[n] = [y, y + totals[n]]
            y += totals[n] + gap
        return out, span

    unit_gap = node_gap * (sum(src_total.values()) / max(len(sources), 1))
    src_pos, src_span = _layout(sources, src_total, unit_gap)
    unit_gap_t = node_gap * (sum(tgt_total.values()) / max(len(targets), 1))
    tgt_pos, tgt_span = _layout(targets, tgt_total, unit_gap_t)

    # Centre the shorter column against the taller one.
    span = max(src_span, tgt_span)
    for pos, s in ((src_pos, src_span), (tgt_pos, tgt_span)):
        shift = (span - s) / 2.0
        for k in pos:
            pos[k] = [pos[k][0] + shift, pos[k][1] + shift]

    color_of = {t: CLASS_COLORS[i % len(CLASS_COLORS)] for i, t in enumerate(targets)}

    # Draw ribbons in descending width so thin flows stay visible on top.
    src_cursor = {s: src_pos[s][0] for s in sources}
    tgt_cursor = {t: tgt_pos[t][0] for t in targets}
    for f in sorted(prepared, key=lambda f: -f["w"]):
        ys0 = src_cursor[f["src"]]
        ys1 = ys0 + f["w"]
        yt0 = tgt_cursor[f["tgt"]]
        yt1 = yt0 + f["w"]
        src_cursor[f["src"]] = ys1
        tgt_cursor[f["tgt"]] = yt1
        _ribbon(ax, x0 + node_width, x1 - node_width, ys0, ys1, yt0, yt1,
                color_of[f["tgt"]], alpha * (0.72 if f["neg"] else 1.0),
                negative=f["neg"])

    # Node bars and labels
    for s in sources:
        y0, y1 = src_pos[s]
        ax.fill_between([x0, x0 + node_width], y0, y1, color=LABEL_COLOR,
                        alpha=0.75, linewidth=0, zorder=4)
        label = s
        if value_labels:
            label = f"{s}  ({src_total[s]:.2f})"
        ax.text(x0 - 0.015, (y0 + y1) / 2, label, ha="right", va="center",
                fontsize=label_size, color=LABEL_COLOR, zorder=5)

    for t in targets:
        y0, y1 = tgt_pos[t]
        ax.fill_between([x1 - node_width, x1], y0, y1, color=color_of[t],
                        alpha=0.90, linewidth=0, zorder=4)
        label = t
        if value_labels:
            label = f"{t}  ({tgt_total[t]:.2f})"
        ax.text(x1 + 0.015, (y0 + y1) / 2, label, ha="left", va="center",
                fontsize=label_size + 1, color=LABEL_COLOR, zorder=5)

    ax.set_xlim(x0 - 0.30, x1 + 0.22)
    ax.set_ylim(-0.02 * span, span * 1.02)
    ax.invert_yaxis()
    ax.axis("off")
    return span


def render(flows, output_path, width=9.0, height=None, title=None, **kw):
    n_src = len({f[0] for f in flows})
    if height is None:
        height = max(2.6, 0.42 * n_src + 1.0)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(width, height))
        draw_sankey(ax, flows, **kw)
        if title:
            ax.set_title(title, fontsize=13, color=LABEL_COLOR, pad=14)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}  ({len(flows)} flows, {n_src} concepts)")


def sankeymatic_text(flows):
    """LF-CBM compatible text for https://sankeymatic.com/build/."""
    lines = []
    for src, tgt, w in flows:
        name = f"NOT {src}" if w < 0 else src
        lines.append(f"{name} [{abs(w):.4f}] {tgt}")
    return "\n".join(lines)


def _class_names(load_dir, num_classes, dataset=None):
    """Resolve human-readable class names for a checkpoint.

    train_vlg.py does not write classes.txt, so the usual source is
    data_utils.get_classes() keyed on the dataset recorded in args.txt.
    """
    classes_path = os.path.join(load_dir, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            return [l.strip() for l in f if l.strip()]

    if dataset is None:
        args_path = os.path.join(load_dir, "args.txt")
        if os.path.exists(args_path):
            try:
                with open(args_path) as f:
                    dataset = json.load(f).get("dataset")
            except Exception:
                dataset = None
    if dataset:
        try:
            from data import data_utils
            return data_utils.get_classes(dataset)
        except Exception as e:
            print(f"[warn] could not load class names for {dataset!r}: {e}")

    return [str(i) for i in range(num_classes)]


def flows_from_checkpoint(load_dir, class_names, top_k, min_weight, dataset=None):
    """Extract concept->class flows from a trained final layer."""
    import torch  # imported lazily so --spec mode needs no torch

    final_sd = torch.load(os.path.join(load_dir, "final.pt"), map_location="cpu")
    W = final_sd["weight"].float()                      # [num_classes, num_concepts]
    with open(os.path.join(load_dir, "concepts.txt")) as f:
        concepts = [l.strip() for l in f if l.strip()]

    classes = _class_names(load_dir, W.shape[0], dataset)

    flows = []
    for cname in class_names:
        if cname in classes:
            ci = classes.index(cname)
        elif cname.isdigit():
            ci = int(cname)
            cname = classes[ci] if ci < len(classes) else cname
        else:
            raise SystemExit(
                f"Class {cname!r} not found. Available (first 20): {classes[:20]}"
            )
        row = W[ci]
        order = torch.argsort(row.abs(), descending=True)
        taken = 0
        for j in order:
            w = float(row[j])
            if abs(w) < min_weight:
                break
            name = concepts[j] if j < len(concepts) else f"concept_{int(j)}"
            flows.append((name, cname, w))
            taken += 1
            if taken >= top_k:
                break
    return flows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--load_dir", help="Trained checkpoint (final.pt, concepts.txt)")
    src.add_argument("--spec", help="JSON file with a 'flows' list")
    p.add_argument("--classes", nargs="+", default=None,
                   help="Class names (or indices) to include, checkpoint mode")
    p.add_argument("--dataset", default=None,
                   help="Override the dataset used to resolve class names "
                        "(defaults to the value in the checkpoint's args.txt)")
    p.add_argument("--top_k", type=int, default=6,
                   help="Max concepts per class")
    p.add_argument("--min_weight", type=float, default=0.05,
                   help="Ignore weights below this magnitude (LF-CBM uses 0.05)")
    p.add_argument("--output", default="concept_sankey.pdf")
    p.add_argument("--title", default=None)
    p.add_argument("--width", type=float, default=9.0)
    p.add_argument("--height", type=float, default=None)
    p.add_argument("--value_labels", action="store_true",
                   help="Append the summed weight to each node label")
    p.add_argument("--sankeymatic", action="store_true",
                   help="Print SankeyMATIC text instead of rendering")
    args = p.parse_args()

    if args.spec:
        with open(args.spec) as f:
            flows = [tuple(x) for x in json.load(f)["flows"]]
    else:
        if not args.classes:
            raise SystemExit("--classes is required with --load_dir")
        flows = flows_from_checkpoint(args.load_dir, args.classes,
                                      args.top_k, args.min_weight,
                                      dataset=args.dataset)

    if args.sankeymatic:
        print(sankeymatic_text(flows))
        return

    render(flows, args.output, width=args.width, height=args.height,
           title=args.title, value_labels=args.value_labels)


if __name__ == "__main__":
    main()
