"""Generate concept coherence visualizations from a trained VLG-CBM checkpoint.

Visualization 1 — Top-k activated images per concept neuron (VLG-CBM style):
  For selected concepts, show the k images with highest normalized activation.

Visualization 2 — Concept contribution bar chart per image (LF-CBM style):
  For selected images, show the top contributing concepts to the predicted class.

Usage:
    python visualize_concept_coherence.py \
        --load_dir /path/to/checkpoint \
        --dataset cifar10 \
        --output_dir /path/to/output \
        --top_k 5 \
        --num_concepts 8 \
        --num_images 6
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Label-free-CBM"))

from data import data_utils


# ── Loading ──────────────────────────────────────────────────────────────────

def load_checkpoint(load_dir, device="cpu"):
    """Load concepts, normalization stats, final layer, and concept features."""
    # Concepts
    with open(os.path.join(load_dir, "concepts.txt")) as f:
        concepts = [l.strip() for l in f if l.strip()]

    # Normalization
    mean = torch.load(os.path.join(load_dir, "train_concept_features_mean.pt"), map_location=device)
    std = torch.load(os.path.join(load_dir, "train_concept_features_std.pt"), map_location=device).clamp(min=1e-8)

    # Final layer
    final_sd = torch.load(os.path.join(load_dir, "final.pt"), map_location=device)
    W = final_sd["weight"]  # [K, C]
    b = final_sd["bias"]    # [K]

    # Pre-saved concept features (CBL output, pre-normalization)
    val_feats = torch.load(os.path.join(load_dir, "val_concept_features.pt"), map_location=device)  # [N, C]
    val_labels = torch.load(os.path.join(load_dir, "val_concept_labels.pt"), map_location=device)    # [N]

    # Normalize
    val_normed = (val_feats - mean) / std  # [N, C]

    return concepts, val_normed, val_labels, W, b


def load_images(dataset_name):
    """Load the raw PIL dataset for image display."""
    try:
        return data_utils.get_data(f"{dataset_name}_test", preprocess=None)
    except Exception:
        return data_utils.get_data(f"{dataset_name}_val", preprocess=None)


# ── Visualization 1: Top-k images per concept ───────────────────────────────

def plot_topk_images_per_concept(val_normed, concepts, pil_dataset, concept_indices,
                                 top_k=5, output_dir=None):
    """For each selected concept, show the top-k images with highest activation."""
    n_concepts = len(concept_indices)
    fig, axes = plt.subplots(n_concepts, top_k, figsize=(2.2 * top_k, 2.5 * n_concepts))
    if n_concepts == 1:
        axes = axes[np.newaxis, :]

    for row, cidx in enumerate(concept_indices):
        activations = val_normed[:, cidx]  # [N]
        topk_vals, topk_idxs = activations.topk(top_k)

        for col in range(top_k):
            ax = axes[row, col]
            img_idx = topk_idxs[col].item()
            img = pil_dataset[img_idx][0]
            if not isinstance(img, Image.Image):
                # Handle tensor images
                if isinstance(img, torch.Tensor):
                    img = img.permute(1, 2, 0).numpy()
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(img)

            ax.imshow(img)
            ax.set_title(f"{topk_vals[col]:.2f}", fontsize=9)
            ax.axis("off")

        # Row label
        cname = concepts[cidx] if cidx < len(concepts) else f"concept_{cidx}"
        axes[row, 0].set_ylabel(f"C{cidx}: {cname}", fontsize=9, rotation=0,
                                labelpad=max(80, 8 * len(cname)), ha="right", va="center")

    fig.suptitle("Top-k activated images per concept neuron", fontsize=13, y=1.01)
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, "topk_images_per_concept.pdf")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def select_diverse_concepts(W, val_normed, concepts, n=8):
    """Select concepts that are diverse and informative for visualization.

    Picks concepts that:
    1. Have high max activation (the concept is strongly expressed somewhere)
    2. Have high final-layer weight magnitude (the concept matters for classification)
    3. Span different concept names (avoids showing 5 variations of "red")
    """
    C = val_normed.shape[1]

    # Score = max activation across samples * max weight magnitude across classes
    max_act = val_normed.max(dim=0).values     # [C]
    max_weight = W.abs().max(dim=0).values      # [C]
    score = max_act * max_weight

    # Sort by score, then greedily pick concepts with distinct first words
    order = score.argsort(descending=True)
    selected = []
    seen_prefixes = set()
    for idx in order:
        idx = idx.item()
        if idx >= len(concepts):
            continue
        prefix = concepts[idx].split()[0].lower()
        if prefix not in seen_prefixes:
            selected.append(idx)
            seen_prefixes.add(prefix)
        if len(selected) >= n:
            break

    # If we didn't get enough, fill without prefix constraint
    if len(selected) < n:
        for idx in order:
            idx = idx.item()
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= n:
                break

    return selected


# ── Visualization 2: Concept contribution bar chart ──────────────────────────

def plot_concept_contributions(val_normed, val_labels, W, b, concepts, classes,
                               pil_dataset, image_indices, max_display=7,
                               output_dir=None):
    """For each selected image, show the top concept contributions to predicted class."""
    n_images = len(image_indices)
    fig, axes = plt.subplots(n_images, 2, figsize=(12, 3.2 * n_images),
                              gridspec_kw={"width_ratios": [1, 2.5]})
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for row, img_idx in enumerate(image_indices):
        a = val_normed[img_idx]              # [C]
        logits = a @ W.T + b                  # [K]
        pred_class = logits.argmax().item()
        gt_class = val_labels[img_idx].item()
        conf = F.softmax(logits, dim=0)

        # Contribution of each concept to predicted class
        contributions = (a * W[pred_class]).cpu().numpy()  # [C]

        # Top concepts by absolute contribution
        order = np.argsort(np.abs(contributions))[::-1]
        top_idxs = order[:max_display]
        remaining = contributions[order[max_display:]].sum()

        values = contributions[top_idxs]
        names = []
        for ci in top_idxs:
            name = concepts[ci] if ci < len(concepts) else f"concept_{ci}"
            if a[ci] < 0:
                name = "NOT " + name
            names.append(name)
        names.append(f"Sum of {len(contributions) - max_display} others")
        values = np.append(values, remaining)

        # Image panel
        ax_img = axes[row, 0]
        img = pil_dataset[img_idx][0]
        if not isinstance(img, Image.Image):
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img)
        ax_img.imshow(img)
        ax_img.axis("off")

        gt_name = classes[gt_class] if gt_class < len(classes) else str(gt_class)
        pred_name = classes[pred_class] if pred_class < len(classes) else str(pred_class)
        ax_img.set_title(f"GT: {gt_name}", fontsize=9)

        # Bar chart panel
        ax_bar = axes[row, 1]
        y_pos = np.arange(len(values))
        colors = ["#E74C3C" if v > 0 else "#3498DB" for v in values]
        ax_bar.barh(y_pos, values, color=colors, edgecolor="none", height=0.65)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(names, fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Concept contribution", fontsize=9)
        ax_bar.set_title(
            f"Pred: {pred_name}  |  Conf: {conf[pred_class]:.3f}  |  "
            f"Logit: {logits[pred_class]:.2f}  |  Bias: {b[pred_class]:.2f}",
            fontsize=9,
        )

        # Annotate bars
        for i, v in enumerate(values):
            ax_bar.text(v + 0.02 * np.sign(v), i, f"{v:+.2f}",
                        va="center", fontsize=7, color=colors[i])

        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, "concept_contributions.pdf")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def select_diverse_images(val_normed, val_labels, W, b, classes, n=6):
    """Select correctly classified images spanning different classes."""
    logits = val_normed @ W.T + b              # [N, K]
    preds = logits.argmax(dim=1)               # [N]
    correct = (preds == val_labels).nonzero(as_tuple=True)[0]

    # Pick one image per class (highest confidence)
    confs = F.softmax(logits, dim=1)
    selected = []
    seen_classes = set()
    # Sort by confidence descending
    conf_vals, conf_order = confs[correct, preds[correct]].sort(descending=True)
    for rank in range(len(conf_order)):
        idx = correct[conf_order[rank]].item()
        cls = preds[idx].item()
        if cls not in seen_classes:
            selected.append(idx)
            seen_classes.add(cls)
        if len(selected) >= n:
            break

    return selected


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Concept coherence visualizations")
    parser.add_argument("--load_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5, help="Images per concept (Viz 1)")
    parser.add_argument("--num_concepts", type=int, default=8, help="Concepts to show (Viz 1)")
    parser.add_argument("--num_images", type=int, default=6, help="Images to explain (Viz 2)")
    parser.add_argument("--concept_indices", type=int, nargs="+", default=None,
                        help="Manual concept indices for Viz 1 (overrides auto-selection)")
    parser.add_argument("--image_indices", type=int, nargs="+", default=None,
                        help="Manual image indices for Viz 2 (overrides auto-selection)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.load_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading checkpoint...")
    concepts, val_normed, val_labels, W, b = load_checkpoint(args.load_dir)
    print(f"  Concepts: {len(concepts)}  |  Samples: {val_normed.shape[0]}  |  Classes: {W.shape[0]}")

    print("Loading images...")
    pil_dataset = load_images(args.dataset)

    classes = data_utils.get_classes(args.dataset)

    # ── Viz 1: Top-k images per concept ────────────────────────────────────
    if args.concept_indices:
        concept_indices = args.concept_indices
    else:
        concept_indices = select_diverse_concepts(W, val_normed, concepts, n=args.num_concepts)

    print(f"\nVisualization 1: Top-{args.top_k} images for {len(concept_indices)} concepts")
    for ci in concept_indices:
        print(f"  C{ci}: {concepts[ci]}")

    plot_topk_images_per_concept(
        val_normed, concepts, pil_dataset, concept_indices,
        top_k=args.top_k, output_dir=output_dir,
    )

    # ── Viz 2: Concept contributions per image ─────────────────────────────
    if args.image_indices:
        image_indices = args.image_indices
    else:
        image_indices = select_diverse_images(val_normed, val_labels, W, b, classes, n=args.num_images)

    print(f"\nVisualization 2: Concept contributions for {len(image_indices)} images")
    for ii in image_indices:
        gt = classes[val_labels[ii].item()]
        pred_cls = (val_normed[ii] @ W.T + b).argmax().item()
        pred = classes[pred_cls]
        print(f"  Image {ii}: GT={gt}, Pred={pred}")

    plot_concept_contributions(
        val_normed, val_labels, W, b, concepts, classes,
        pil_dataset, image_indices, output_dir=output_dir,
    )

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
