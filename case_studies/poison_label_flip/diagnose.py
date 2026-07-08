"""
Diagnostic visualizations for the poisoned-client case study.

Loads per-client final-layer primal snapshots written by train_vlg.py
(--phase3_snapshot_dir) and produces:
  1. Per-client weight heatmaps (all clients + global, side-by-side)
  2. Top-K concept table for the two affected classes
  3. Per-client cosine-divergence-from-global scores

All outputs are written to out_dir.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_snapshots(snapshot_dir: str, num_clients: int, file_tag: str = "primal") -> dict:
    snapshots = {}
    for i in range(num_clients):
        path = os.path.join(snapshot_dir, f"client_{i}_{file_tag}.pt")
        if os.path.exists(path):
            snapshots[f"client_{i}"] = torch.load(path, map_location="cpu")
    global_path = os.path.join(snapshot_dir, f"global_{file_tag}.pt")
    if os.path.exists(global_path):
        snapshots["global"] = torch.load(global_path, map_location="cpu")
    return snapshots


def plot_weight_heatmaps(
    snapshot_dir: str,
    concept_names: list,
    class_names: list,
    num_clients: int,
    adversary_client_id: int,
    out_dir: str,
    file_tag: str = "primal",
):
    snapshots = _load_snapshots(snapshot_dir, num_clients, file_tag)
    if not snapshots:
        print("[diagnose] No snapshot files found.")
        return

    keys = [f"client_{i}" for i in range(num_clients) if f"client_{i}" in snapshots]
    if "global" in snapshots:
        keys.append("global")

    n_panels = len(keys)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    # Shared color scale across all panels
    all_weights = torch.cat([snapshots[k]["weight"].flatten() for k in keys])
    vmax = float(all_weights.abs().quantile(0.99))
    vmin = -vmax

    for ax, key in zip(axes, keys):
        w = snapshots[key]["weight"].float().numpy()  # [num_classes, num_concepts]
        im = ax.imshow(w, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xticks([])

        label = key
        if key.startswith("client_"):
            cid = int(key.split("_")[1])
            label = f"Client {cid}"
            if cid == adversary_client_id:
                label += " ★"
                ax.set_title(label, color="red", fontsize=9, fontweight="bold")
            else:
                ax.set_title(label, fontsize=9)
        else:
            ax.set_title("Global", fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=axes[-1], fraction=0.05, pad=0.04, label="Weight")
    fig.suptitle("Concept→Class Final-Layer Weights per Client\n(★ = poisoned client)",
                 fontsize=11)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "weight_heatmaps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[diagnose] Heatmaps saved to {out_path}")


def print_topk_concepts(
    snapshot_dir: str,
    concept_names: list,
    class_names: list,
    num_clients: int,
    adversary_client_id: int,
    source_class: str,
    target_class: str,
    k: int = 5,
    out_dir: str = None,
    file_tag: str = "primal",
):
    snapshots = _load_snapshots(snapshot_dir, num_clients, file_tag)
    if not snapshots:
        return

    src_idx = class_names.index(source_class) if source_class in class_names else None
    tgt_idx = class_names.index(target_class) if target_class in class_names else None
    focus_indices = [i for i in [src_idx, tgt_idx] if i is not None]

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"Top-{k} concepts for '{source_class}' and '{target_class}' per client")
    lines.append(f"{'='*70}")

    keys = [f"client_{i}" for i in range(num_clients) if f"client_{i}" in snapshots]
    keys.append("global")

    for key in keys:
        w = snapshots[key]["weight"].float()
        cid_label = key
        if key.startswith("client_"):
            cid = int(key.split("_")[1])
            cid_label = f"Client {cid}" + (" [POISONED]" if cid == adversary_client_id else "")
        else:
            cid_label = "Global"

        lines.append(f"\n  {cid_label}:")
        for cls_idx in focus_indices:
            cls_name = class_names[cls_idx]
            row = w[cls_idx]  # [num_concepts]
            topk_vals, topk_idxs = torch.topk(row.abs(), k=min(k, len(concept_names)))
            top_concepts = [(concept_names[i], float(row[i])) for i in topk_idxs]
            formatted = ", ".join(f"{c} ({v:+.3f})" for c, v in top_concepts)
            lines.append(f"    [{cls_name:>10}]: {formatted}")

    output = "\n".join(lines)
    print(output)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "topk_concepts.txt")
        with open(out_path, "w") as f:
            f.write(output + "\n")
        print(f"[diagnose] Top-K table saved to {out_path}")


def compute_divergence_scores(
    snapshot_dir: str,
    num_clients: int,
    adversary_client_id: int,
    out_dir: str = None,
    file_tag: str = "primal",
) -> dict:
    snapshots = _load_snapshots(snapshot_dir, num_clients, file_tag)
    if "global" not in snapshots:
        print(f"[diagnose] No global_{file_tag}.pt found; skipping divergence.")
        return {}

    g = snapshots["global"]["weight"].float().flatten()
    g_norm = g / (g.norm() + 1e-8)

    scores = {}
    print("\n=== Per-Client Cosine Divergence from Global Final Layer ===")
    print(f"  {'Client':<12} {'Cos-sim':>10}  {'Divergence':>12}  {'Flagged':>8}")
    print(f"  {'-'*12} {'-'*10}  {'-'*12}  {'-'*8}")

    for i in range(num_clients):
        key = f"client_{i}"
        if key not in snapshots:
            continue
        c = snapshots[key]["weight"].float().flatten()
        c_norm = c / (c.norm() + 1e-8)
        cos_sim = float(torch.dot(g_norm, c_norm))
        div = 1.0 - cos_sim
        flagged = "✓ OUTLIER" if i == adversary_client_id else ""
        scores[key] = {"cos_sim": cos_sim, "divergence": div}
        print(f"  Client {i:<6} {cos_sim:>10.4f}  {div:>12.4f}  {flagged}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "divergence_scores.json")
        with open(out_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"[diagnose] Divergence scores saved to {out_path}")

    return scores


def compute_column_divergence(
    snapshot_dir: str,
    class_names: list,
    num_clients: int,
    adversary_client_id: int,
    focus_classes: list,
    out_dir: str = None,
    file_tag: str = "primal",
) -> dict:
    """Per-class-row cosine divergence from the global head, restricted to the
    poisoned classes. The global scalar dilutes a 2-class attack across all classes;
    this isolates it. Each class row W[c] is that class's weights over concepts."""
    snapshots = _load_snapshots(snapshot_dir, num_clients, file_tag)
    if "global" not in snapshots:
        return {}

    focus_idx = [(c, class_names.index(c)) for c in focus_classes if c in class_names]
    g = snapshots["global"]["weight"].float()

    scores = {}
    print("\n=== Per-Class-Column Divergence from Global (poisoned classes only) ===")
    header = f"  {'Client':<12}" + "".join(f"{cn:>16}" for cn, _ in focus_idx) + f"{'Flagged':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i in range(num_clients):
        key = f"client_{i}"
        if key not in snapshots:
            continue
        w = snapshots[key]["weight"].float()
        row = {}
        line = f"  Client {i:<6}"
        for cn, ci in focus_idx:
            gc = g[ci] / (g[ci].norm() + 1e-8)
            cc = w[ci] / (w[ci].norm() + 1e-8)
            div = 1.0 - float(torch.dot(gc, cc))
            row[cn] = div
            line += f"{div:>16.4f}"
        line += f"{'✓ OUTLIER' if i == adversary_client_id else '':>12}"
        scores[key] = row
        print(line)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "column_divergence.json")
        with open(out_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"[diagnose] Column divergence saved to {out_path}")

    return scores


def run_diagnostics(
    snapshot_dir: str,
    concept_names: list,
    class_names: list,
    num_clients: int,
    adversary_client_id: int,
    source_class: str,
    target_class: str,
    topk: int = 5,
    out_dir: str = None,
    file_tag: str = "primal",
):
    if out_dir is None:
        out_dir = os.path.join(snapshot_dir, "diagnostics")

    print(f"\n{'#'*70}")
    print(f"# Poison Case Study Diagnostics  ({file_tag})")
    print(f"# Adversary: Client {adversary_client_id}  |  Flip: {source_class} -> {target_class}")
    print(f"{'#'*70}")

    plot_weight_heatmaps(snapshot_dir, concept_names, class_names,
                         num_clients, adversary_client_id, out_dir, file_tag=file_tag)

    print_topk_concepts(snapshot_dir, concept_names, class_names,
                        num_clients, adversary_client_id,
                        source_class, target_class, k=topk, out_dir=out_dir, file_tag=file_tag)

    compute_divergence_scores(snapshot_dir, num_clients, adversary_client_id, out_dir, file_tag=file_tag)

    compute_column_divergence(snapshot_dir, class_names, num_clients, adversary_client_id,
                              focus_classes=[source_class, target_class],
                              out_dir=out_dir, file_tag=file_tag)

    print(f"\n[diagnose] All outputs written to {out_dir}")
