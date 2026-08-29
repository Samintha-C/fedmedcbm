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
        --top_k 4 \
        --num_concepts 12 \
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
from evaluate_steerability import load_full_model, load_concepts
from torch.utils.data import DataLoader
from visualizations.contribution_panel import (RC as CONTRIB_RC, LABEL_COLOR as CONTRIB_LABEL_COLOR,
                                                draw_contribution_panel)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_and_extract(load_dir, dataset_name, device="cpu", batch_size=128):
    """Load model, run on test set, return aligned activations + PIL dataset."""
    model, saved_args, num_concepts, num_classes = load_full_model(load_dir, device)
    concepts = load_concepts(load_dir, saved_args)

    # Final layer weights
    W = model.final_layer.weight.data.cpu()
    b = model.final_layer.bias.data.cpu()

    # Get preprocessed test set for model inference
    if hasattr(model.backbone, "preprocess"):
        preprocess = model.backbone.preprocess
    else:
        _, preprocess = data_utils.get_target_model(
            model.backbone.backbone.__class__.__name__.lower(), "cpu"
        )
        if preprocess is None:
            import torchvision.transforms as T
            preprocess = T.Compose([
                T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    try:
        test_data = data_utils.get_data(f"{dataset_name}_test", preprocess=preprocess)
    except Exception:
        test_data = data_utils.get_data(f"{dataset_name}_val", preprocess=preprocess)

    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    # Extract normalized concept activations
    all_a, all_y = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            h = model.backbone(images)
            c = model.cbl(h)
            a = model.normalization(c)
            all_a.append(a.cpu())
            all_y.append(labels if torch.is_tensor(labels) else torch.tensor(labels))

    val_normed = torch.cat(all_a, dim=0)
    val_labels = torch.cat(all_y, dim=0)

    # Load raw PIL images (same test set, no transform) — indices will match
    try:
        pil_dataset = data_utils.get_data(f"{dataset_name}_test", preprocess=None)
    except Exception:
        pil_dataset = data_utils.get_data(f"{dataset_name}_val", preprocess=None)

    return concepts, val_normed, val_labels, W, b, pil_dataset


# ── Visualization 1: Top-k images per concept ───────────────────────────────

def plot_topk_images_per_concept(val_normed, concepts, pil_dataset, concept_indices,
                                 top_k=4, output_dir=None, width=5.5,
                                 label_fontsize=11, value_fontsize=10,
                                 filename="topk_images_per_concept.pdf"):
    """For each selected concept, show the top-k images with highest activation.

    Drawn at the paper's printed width so \\includegraphics does not scale the text
    down. The concept name sits ABOVE its row rather than beside it, and every
    thumbnail is rendered in a square box of identical size regardless of the
    source aspect ratio, so rows are visually comparable.
    """
    n_concepts = len(concept_indices)
    cell = width / top_k
    # "activation: 0.98" at 10pt is ~1.1in wide, so the cell must be at least that
    # or adjacent labels collide; top_k=4 over 5.5in gives 1.375in cells.
    # Extra height per row carries the heading above and the values below.
    fig, axes = plt.subplots(n_concepts, top_k,
                             figsize=(width, n_concepts * (cell + 0.78)))
    axes = np.atleast_2d(axes)
    if n_concepts == 1 and axes.shape[0] != 1:
        axes = axes.reshape(1, -1)

    with plt.rc_context(CONTRIB_RC):
        for row, cidx in enumerate(concept_indices):
            activations = val_normed[:, cidx]
            topk_vals, topk_idxs = activations.topk(top_k)
            cname = concepts[cidx] if cidx < len(concepts) else f"concept_{cidx}"

            for col in range(top_k):
                ax = axes[row, col]
                img = pil_dataset[topk_idxs[col].item()][0]
                if not isinstance(img, Image.Image):
                    if isinstance(img, torch.Tensor):
                        img = img.permute(1, 2, 0).numpy()
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                        img = Image.fromarray(img)
                # Square centre crop then a fixed resize: CUB photos vary in
                # aspect ratio, and without this the thumbnails differ in size.
                w, h = img.size
                m = min(w, h)
                img = img.crop(((w - m) // 2, (h - m) // 2,
                                (w + m) // 2, (h + m) // 2)).resize((224, 224))
                ax.imshow(img)
                # Below the image: the row heading owns the space above. Split
                # over two lines because "activation: 0.99" at 10pt is ~1.35in,
                # which fills the whole cell and touches the neighbouring label.
                ax.text(0.5, -0.04, f"activation\n{topk_vals[col]:.2f}",
                        transform=ax.transAxes, ha="center", va="top",
                        linespacing=1.25,
                        fontsize=value_fontsize, color=CONTRIB_LABEL_COLOR)
                ax.axis("off")

            # Concept name ABOVE the row, spanning its full width.
            left = axes[row, 0].get_position().x0
            right = axes[row, -1].get_position().x1
            top = axes[row, 0].get_position().y1
            fig.text((left + right) / 2, top + 0.012,
                     f"C{cidx}: {cname}", ha="center", va="bottom",
                     fontsize=label_fontsize, fontweight="bold",
                     color=CONTRIB_LABEL_COLOR)

    if output_dir:
        path = os.path.join(output_dir, filename)
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
                               output_dir=None, min_contribution=1e-2,
                               show_others=False, show_prediction=True):
    """For each selected image, show the top concept contributions to predicted class.

    Concepts whose contribution is below ``min_contribution`` in magnitude are
    dropped rather than drawn as flat 0.00 bars, and the "Sum of N others" row is
    off by default: both are visual noise on sparse final layers where only a
    handful of concepts carry any weight at all.
    """
    n_images = len(image_indices)
    with plt.rc_context(CONTRIB_RC):
        fig, axes = plt.subplots(n_images, 2, figsize=(12, 3.0 * n_images),
                                  gridspec_kw={"width_ratios": [1, 2.5]})
        if n_images == 1:
            axes = axes[np.newaxis, :]
        _draw_contribution_rows(
            axes, val_normed, val_labels, W, b, concepts, classes, pil_dataset,
            image_indices, max_display, min_contribution, show_others,
            show_prediction,
        )
        plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, "concept_contributions.pdf")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def _draw_contribution_rows(axes, val_normed, val_labels, W, b, concepts, classes,
                            pil_dataset, image_indices, max_display,
                            min_contribution, show_others, show_prediction):
    for row, img_idx in enumerate(image_indices):
        a = val_normed[img_idx]              # [C]
        logits = a @ W.T + b                  # [K]
        pred_class = logits.argmax().item()
        gt_class = val_labels[img_idx].item()

        # Contribution of each concept to predicted class
        contributions = (a * W[pred_class]).cpu().numpy()  # [C]

        # Only display concepts that are actively expressed (a > 0).
        # Below-average concepts (a < 0) can still contribute via negative weights,
        # but the "NOT X" framing is misleading on z-scored activations.
        a_np = a.cpu().numpy()
        eligible = np.where(a_np > 0)[0]
        eligible_sorted = eligible[np.argsort(np.abs(contributions[eligible]))[::-1]]
        top_idxs = eligible_sorted[:max_display]

        # Drop concepts that contribute nothing. A sparse final layer leaves most
        # of the eligible set at exactly 0.00, which renders as a stack of empty
        # ticks (e.g. "motor", "mottled brown plumage" on an ImageNet quill).
        top_idxs = top_idxs[np.abs(contributions[top_idxs]) >= min_contribution]

        values = contributions[top_idxs]
        names = [concepts[ci] if ci < len(concepts) else f"concept_{ci}" for ci in top_idxs]

        if show_others:
            shown_mask = np.zeros(len(contributions), dtype=bool)
            shown_mask[top_idxs] = True
            names.append(f"Sum of {(~shown_mask).sum()} others")
            values = np.append(values, contributions[~shown_mask].sum())

        img = pil_dataset[img_idx][0]
        if not isinstance(img, Image.Image):
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img)

        gt_name = classes[gt_class] if gt_class < len(classes) else str(gt_class)
        pred_name = classes[pred_class] if pred_class < len(classes) else str(pred_class)

        draw_contribution_panel(
            axes[row, 0], axes[row, 1], img, names, values,
            gt_name=gt_name,
            pred_name=pred_name if show_prediction else None,
        )


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
    parser.add_argument("--top_k", type=int, default=4,
                        help="Images per concept (Viz 1). Above 4 the per-image "
                             "activation labels collide at the 5.5in paper width.")
    parser.add_argument("--num_concepts", type=int, default=8, help="Concepts to show (Viz 1)")
    parser.add_argument("--num_images", type=int, default=6, help="Images to explain (Viz 2)")
    parser.add_argument("--concept_indices", type=int, nargs="+", default=None,
                        help="Manual concept indices for Viz 1 (overrides auto-selection)")
    parser.add_argument("--image_indices", type=int, nargs="+", default=None,
                        help="Manual image indices for Viz 2 (overrides auto-selection)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_display", type=int, default=7,
        help="Max concept bars per image (Viz 2)")
    parser.add_argument("--min_contribution", type=float, default=1e-2,
        help="Drop concepts contributing less than this in magnitude (Viz 2)")
    parser.add_argument("--show_others", action="store_true",
        help="Add the 'Sum of N others' bar back (Viz 2)")
    parser.add_argument("--hide_prediction", action="store_true",
        help="Drop the 'Predicted: <class>' title above the bars (Viz 2)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or os.path.join(args.load_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model and extracting test-set activations...")
    concepts, val_normed, val_labels, W, b, pil_dataset = load_and_extract(
        args.load_dir, args.dataset, device=device, batch_size=args.batch_size,
    )
    print(f"  Concepts: {len(concepts)}  |  Samples: {val_normed.shape[0]}  |  Classes: {W.shape[0]}")

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

    # Also emit one file per concept. The paper includes single concepts at
    # \textwidth (main text and appendix show different ones), and cropping them
    # out of the combined sheet by hand is what shrank the fonts previously.
    for ci in concept_indices:
        plot_topk_images_per_concept(
            val_normed, concepts, pil_dataset, [ci],
            top_k=args.top_k, output_dir=output_dir,
            filename=f"concept_c{ci}.pdf",
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
        max_display=args.max_display,
        min_contribution=args.min_contribution,
        show_others=args.show_others,
        show_prediction=not args.hide_prediction,
    )

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
