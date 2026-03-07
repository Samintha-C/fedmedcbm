"""Evaluate a trained CBL (projection layer) against DINO binary annotations.

Loads a Phase 1 checkpoint, runs inference on the CIFAR-100 val set, and
compares sigmoid(CBL logits) against ground truth DINO concept labels.

Usage:
    python evaluate_cbl.py \
        --load_dir /path/to/phase1_checkpoint \
        --annotation_dir /path/to/annotations \
        --dataset cifar100
"""
import argparse
import json
import os
import sys

import torch
import numpy as np

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Label-free-CBM"))

from data import data_utils
from models.fed_vlgcbm import Backbone, BackboneCLIP, ConceptLayer


def load_phase1(load_dir, device):
    """Load backbone + CBL from a Phase 1 checkpoint directory.

    Supports two checkpoint formats:
    - Federated VLG (our format): backbone.pt + cbl.pt + args.txt with "backbone" key
    - Vanilla VLG-CBM (original paper): cbl.pt only + args.txt with "backbone"/"concept_set" keys.
      The backbone is not saved when cbl_finetune=False; pretrained weights are used as-is.
    """
    args_path = os.path.join(load_dir, "args.txt")
    if os.path.exists(args_path):
        with open(args_path) as f:
            saved_args = json.load(f)
    else:
        saved_args = {}

    backbone_name = saved_args.get("backbone", "resnet50")
    feature_layer = saved_args.get("feature_layer", "layer4")

    if backbone_name.startswith("clip_"):
        backbone = BackboneCLIP(backbone_name, use_penultimate=saved_args.get("use_clip_penultimate", True), device=device)
    else:
        backbone = Backbone(backbone_name, feature_layer, device)

    backbone_pt = os.path.join(load_dir, "backbone.pt")
    if os.path.exists(backbone_pt):
        backbone.backbone.load_state_dict(
            torch.load(backbone_pt, map_location=device)
        )
    else:
        print(f"  [INFO] backbone.pt not found — using pretrained weights (cbl_finetune=False checkpoint)")

    # Infer num_concepts and num_hidden directly from cbl.pt state dict,
    # so this works for both our format (args.txt has cbl_hidden_layers) and
    # vanilla VLG-CBM checkpoints (may not have that key).
    cbl_sd = torch.load(os.path.join(load_dir, "cbl.pt"), map_location=device)
    weight_keys = sorted(k for k in cbl_sd if k.endswith(".weight"))
    num_concepts = cbl_sd[weight_keys[0]].shape[0]  # first linear outputs num_concepts
    num_hidden = len(weight_keys) - 1  # 0 for no hidden layers, 1 for one hidden, etc.

    cbl = ConceptLayer(backbone.output_dim, num_concepts, num_hidden=num_hidden, bias=True, device=device)
    cbl.load_state_dict(cbl_sd)

    backbone.eval()
    cbl.eval()
    return backbone, cbl, saved_args, num_concepts


def evaluate_cbl_dino(backbone, cbl, dino_loader, concepts, device):
    """Compare CBL predictions against DINO binary annotations.

    Returns dict with per-concept precision/recall/F1/AUC and aggregates.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    all_cbl_logits = []
    all_dino_labels = []

    with torch.no_grad():
        for images, concept_one_hot, targets in dino_loader:
            images = images.to(device)
            features = backbone(images)
            logits = cbl(features)
            all_cbl_logits.append(logits.cpu())
            all_dino_labels.append(concept_one_hot)

    all_cbl_logits = torch.cat(all_cbl_logits, dim=0)   # [N, C]
    all_dino_labels = torch.cat(all_dino_labels, dim=0)  # [N, C] binary

    cbl_probs = torch.sigmoid(all_cbl_logits)
    cbl_binary = (cbl_probs > 0.5).float()

    N, C = cbl_probs.shape

    per_concept = []
    for c in range(C):
        gt = all_dino_labels[:, c].numpy()
        prob = cbl_probs[:, c].numpy()
        prevalence = float(gt.mean())
        n_pos = int(gt.sum())

        metrics_c = {
            "concept": concepts[c] if c < len(concepts) else f"concept_{c}",
            "prevalence": prevalence,
            "n_positive": n_pos,
            "auc_roc": None,
            # F1 at fixed 0.5 threshold
            "f1_at_05": float(f1_score(gt, (prob > 0.5).astype(float), zero_division=0)),
            "precision_at_05": float(precision_score(gt, (prob > 0.5).astype(float), zero_division=0)),
            "recall_at_05": float(recall_score(gt, (prob > 0.5).astype(float), zero_division=0)),
            # F1 at optimal threshold (sweep 0.01–0.99)
            "f1_optimal": 0.0,
            "threshold_optimal": None,
        }

        if n_pos > 0 and n_pos < N:
            metrics_c["auc_roc"] = float(roc_auc_score(gt, prob))
            # Find threshold maximising F1
            best_f1, best_thr = 0.0, 0.5
            for thr in np.linspace(0.01, 0.99, 99):
                f = f1_score(gt, (prob >= thr).astype(float), zero_division=0)
                if f > best_f1:
                    best_f1, best_thr = f, float(thr)
            metrics_c["f1_optimal"] = float(best_f1)
            metrics_c["threshold_optimal"] = float(best_thr)

        per_concept.append(metrics_c)

    # Aggregate only over concepts that have at least one positive annotation
    annotated = [m for m in per_concept if m["n_positive"] > 0]
    all_aucs = [m["auc_roc"] for m in annotated if m["auc_roc"] is not None]
    all_prevalences = [m["prevalence"] for m in per_concept]
    overall_acc = float((cbl_binary == all_dino_labels).float().mean())

    return {
        "num_samples": N,
        "num_concepts": C,
        "concepts_with_any_positive": len(annotated),
        "overall_binary_accuracy": overall_acc,
        # F1 at 0.5 threshold — only over annotated concepts
        "mean_f1_at_05": float(np.mean([m["f1_at_05"] for m in annotated])) if annotated else None,
        "median_f1_at_05": float(np.median([m["f1_at_05"] for m in annotated])) if annotated else None,
        # F1 at per-concept optimal threshold — upper bound on F1 performance
        "mean_f1_optimal": float(np.mean([m["f1_optimal"] for m in annotated])) if annotated else None,
        "median_f1_optimal": float(np.median([m["f1_optimal"] for m in annotated])) if annotated else None,
        # AUC — threshold-independent discrimination quality
        "mean_auc_roc": float(np.mean(all_aucs)) if all_aucs else None,
        "median_auc_roc": float(np.median(all_aucs)) if all_aucs else None,
        "mean_prevalence": float(np.mean(all_prevalences)),
        "per_concept": per_concept,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CBL against DINO binary annotations")
    parser.add_argument("--load_dir", type=str, required=True,
        help="Phase 1 checkpoint directory (with backbone.pt, cbl.pt, args.txt)")
    parser.add_argument("--annotation_dir", type=str, required=True,
        help="DINO annotation directory containing cifar100_val/")
    parser.add_argument("--dataset", type=str, default=None,
        help="Dataset name (auto-detected from args.txt if omitted)")
    parser.add_argument("--concept_file", type=str, default=None,
        help="Concept file (auto-detected from args.txt if omitted)")
    parser.add_argument("--dino_confidence_threshold", type=float, default=0.10,
        help="DINO detection confidence threshold")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--save_results", type=str, default=None,
        help="Path to save results JSON (default: <load_dir>/cbl_evaluation_dino.json)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading Phase 1 checkpoint from {args.load_dir}")
    backbone, cbl, saved_args, num_concepts = load_phase1(args.load_dir, device)

    dataset_name = args.dataset or saved_args.get("dataset", "cifar100")

    # Load concepts: prefer the checkpoint's concepts.txt (reflects post-filter concepts
    # the model actually trained on) over the input concept file (which may be pre-filter).
    checkpoint_concepts_path = os.path.join(args.load_dir, "concepts.txt")
    if os.path.exists(checkpoint_concepts_path):
        with open(checkpoint_concepts_path) as f:
            concepts = [l.strip() for l in f if l.strip()]
        print(f"Loaded {len(concepts)} concepts from checkpoint concepts.txt (CBL has {num_concepts} outputs)")
    else:
        concept_file = args.concept_file or saved_args.get("concept_file") or saved_args.get("concept_set")
        if concept_file is None:
            raise FileNotFoundError("No concept file found. Specify --concept_file or place concepts.txt in load_dir.")
        concepts = data_utils.get_concepts(concept_file, saved_args.get("filter_set"))
        print(f"Loaded {len(concepts)} concepts from {concept_file} (CBL has {num_concepts} outputs)")

    if len(concepts) != num_concepts:
        print(f"  [WARNING] Concept count mismatch: {len(concepts)} concepts vs {num_concepts} CBL outputs. "
              f"The model may have been trained with concept filtering.")

    from torch.utils.data import DataLoader
    from data.concept_dataset_vlg import DinoConceptDataset

    if hasattr(backbone, "preprocess"):
        preprocess = backbone.preprocess
    else:
        _, preprocess = data_utils.get_target_model(saved_args.get("backbone", "resnet50"), device)

    # Load the official held-out val/test set
    try:
        test_data = data_utils.get_data(f"{dataset_name}_test", preprocess=None)
    except Exception:
        test_data = data_utils.get_data(f"{dataset_name}_val", preprocess=None)

    dino_dataset = DinoConceptDataset(
        dataset_name, test_data, concepts,
        annotation_dir=args.annotation_dir, split_suffix="val",
        confidence_threshold=args.dino_confidence_threshold,
        preprocess=preprocess,
    )
    dino_loader = DataLoader(dino_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"\nEvaluating CBL vs DINO annotations on {dataset_name} val set ({len(dino_dataset)} samples)...")
    results = evaluate_cbl_dino(backbone, cbl, dino_loader, concepts, device)

    print(f"\n{'='*60}")
    print(f"CBL vs DINO Evaluation Results")
    print(f"{'='*60}")
    print(f"Samples: {results['num_samples']}  Concepts: {results['num_concepts']}")
    print(f"Concepts with any positive annotation: {results['concepts_with_any_positive']}/{results['num_concepts']}")
    print(f"Mean concept prevalence: {results['mean_prevalence']:.4f}")
    print(f"\nOverall binary accuracy:  {results['overall_binary_accuracy']:.4f}")
    print(f"Mean F1 (at 0.5 thr):     {results['mean_f1_at_05']:.4f}")
    print(f"Median F1 (at 0.5 thr):   {results['median_f1_at_05']:.4f}")
    print(f"Mean F1 (optimal thr):    {results['mean_f1_optimal']:.4f}")
    print(f"Median F1 (optimal thr):  {results['median_f1_optimal']:.4f}")
    if results['mean_auc_roc'] is not None:
        print(f"Mean AUC-ROC:             {results['mean_auc_roc']:.4f}")
        print(f"Median AUC-ROC:           {results['median_auc_roc']:.4f}")

    sorted_by_f1 = sorted(results["per_concept"], key=lambda x: x["f1_optimal"], reverse=True)

    print(f"\nTop 10 concepts by F1 (optimal threshold):")
    for i, m in enumerate(sorted_by_f1[:10]):
        auc_str = f"AUC={m['auc_roc']:.3f}" if m['auc_roc'] is not None else "AUC=N/A"
        thr_str = f"thr={m['threshold_optimal']:.2f}" if m['threshold_optimal'] is not None else "thr=N/A"
        print(f"  {i+1:2d}. {m['concept']:40s}  F1*={m['f1_optimal']:.3f}  F1@.5={m['f1_at_05']:.3f}  {auc_str}  {thr_str}  prev={m['prevalence']:.3f}")

    print(f"\nBottom 10 concepts by F1 (optimal threshold, annotated only):")
    annotated_sorted = [m for m in sorted_by_f1 if m["n_positive"] > 0]
    for i, m in enumerate(annotated_sorted[-10:]):
        auc_str = f"AUC={m['auc_roc']:.3f}" if m['auc_roc'] is not None else "AUC=N/A"
        thr_str = f"thr={m['threshold_optimal']:.2f}" if m['threshold_optimal'] is not None else "thr=N/A"
        print(f"  {i+1:2d}. {m['concept']:40s}  F1*={m['f1_optimal']:.3f}  F1@.5={m['f1_at_05']:.3f}  {auc_str}  {thr_str}  prev={m['prevalence']:.3f}")

    results["load_dir"] = args.load_dir
    results["dataset"] = dataset_name
    results["annotation_dir"] = args.annotation_dir
    results["confidence_threshold"] = args.dino_confidence_threshold

    save_path = args.save_results or os.path.join(args.load_dir, "cbl_evaluation_dino.json")
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
