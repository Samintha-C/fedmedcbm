"""Evaluate a trained CBL (projection layer) against concept ground truth.

Two modes:
  --mode clip   Compare CBL predictions against CLIP cosine similarities (default)
  --mode dino   Compare CBL predictions against DINO binary annotations

Usage:
    python evaluate_cbl.py --load_dir /path/to/checkpoint --dataset cifar100
    python evaluate_cbl.py --load_dir /path/to/checkpoint --mode dino --annotation_dir /path/to/annotations
"""
import argparse
import json
import os
import sys

import torch
import numpy as np
import clip

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Label-free-CBM"))

from data import data_utils
from data.data_utils import get_classes
from utils.concepts import generate_clip_text_embeddings
from models.fed_vlgcbm import Backbone, BackboneCLIP, ConceptLayer


def load_phase1(load_dir, device):
    """Load backbone + CBL from a Phase 1 checkpoint directory."""
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

    backbone.backbone.load_state_dict(
        torch.load(os.path.join(load_dir, "backbone.pt"), map_location=device)
    )

    # Infer num_concepts from cbl.pt
    cbl_sd = torch.load(os.path.join(load_dir, "cbl.pt"), map_location=device)
    # First linear layer weight shape: [out_features, in_features]
    first_weight = None
    for k, v in cbl_sd.items():
        if "weight" in k:
            first_weight = v
            break
    num_concepts = first_weight.shape[0]
    num_hidden = saved_args.get("cbl_hidden_layers", 0)

    cbl = ConceptLayer(backbone.output_dim, num_concepts, num_hidden=num_hidden, bias=True, device=device)
    cbl.load_state_dict(cbl_sd)

    backbone.eval()
    cbl.eval()
    return backbone, cbl, saved_args, num_concepts


def evaluate_cbl(backbone, cbl, clip_model, concept_embeddings, data_loader, device, top_k=5):
    """Compare CBL predictions against CLIP similarities on a dataset.

    Returns dict with per-concept and aggregate metrics.
    """
    all_cbl_preds = []
    all_clip_sims = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch[0].to(device)

            # CBL predictions
            features = backbone(images)
            cbl_preds = cbl(features)  # [B, num_concepts]

            # CLIP ground truth
            clip_img = clip_model.encode_image(images).float()
            clip_img = clip_img / clip_img.norm(dim=-1, keepdim=True)
            clip_sims = clip_img @ concept_embeddings.T  # [B, num_concepts]

            all_cbl_preds.append(cbl_preds.cpu())
            all_clip_sims.append(clip_sims.cpu())

    all_cbl_preds = torch.cat(all_cbl_preds, dim=0)  # [N, C]
    all_clip_sims = torch.cat(all_clip_sims, dim=0)   # [N, C]

    N, C = all_cbl_preds.shape

    # --- Per-concept Pearson correlation ---
    cbl_np = all_cbl_preds.numpy()
    clip_np = all_clip_sims.numpy()

    per_concept_corr = []
    for c in range(C):
        cc = np.corrcoef(cbl_np[:, c], clip_np[:, c])[0, 1]
        per_concept_corr.append(float(cc) if not np.isnan(cc) else 0.0)
    per_concept_corr = np.array(per_concept_corr)

    # --- Top-k precision per sample ---
    # For each image: are the CBL's top-k concepts also in CLIP's top-k?
    cbl_topk = torch.topk(all_cbl_preds, k=top_k, dim=1).indices  # [N, k]
    clip_topk = torch.topk(all_clip_sims, k=top_k, dim=1).indices  # [N, k]

    topk_precisions = []
    for i in range(N):
        overlap = len(set(cbl_topk[i].tolist()) & set(clip_topk[i].tolist()))
        topk_precisions.append(overlap / top_k)
    avg_topk_precision = float(np.mean(topk_precisions))

    # --- MSE and cosine similarity ---
    mse = float(torch.nn.functional.mse_loss(all_cbl_preds, all_clip_sims).item())
    # Row-wise cosine similarity (per sample)
    cos_sim = torch.nn.functional.cosine_similarity(all_cbl_preds, all_clip_sims, dim=1)
    avg_cos_sim = float(cos_sim.mean().item())

    # --- Rank correlation (Spearman) per concept ---
    from scipy.stats import spearmanr
    per_concept_spearman = []
    for c in range(C):
        rho, _ = spearmanr(cbl_np[:, c], clip_np[:, c])
        per_concept_spearman.append(float(rho) if not np.isnan(rho) else 0.0)
    per_concept_spearman = np.array(per_concept_spearman)

    return {
        "num_samples": N,
        "num_concepts": C,
        "mse": mse,
        "avg_cosine_similarity": avg_cos_sim,
        f"avg_top{top_k}_precision": avg_topk_precision,
        "pearson_correlation": {
            "mean": float(per_concept_corr.mean()),
            "median": float(np.median(per_concept_corr)),
            "min": float(per_concept_corr.min()),
            "max": float(per_concept_corr.max()),
            "std": float(per_concept_corr.std()),
        },
        "spearman_correlation": {
            "mean": float(per_concept_spearman.mean()),
            "median": float(np.median(per_concept_spearman)),
            "min": float(per_concept_spearman.min()),
            "max": float(per_concept_spearman.max()),
            "std": float(per_concept_spearman.std()),
        },
        "per_concept_pearson": per_concept_corr.tolist(),
        "per_concept_spearman": per_concept_spearman.tolist(),
    }


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
        pred = cbl_binary[:, c].numpy()
        prob = cbl_probs[:, c].numpy()
        prevalence = float(gt.mean())

        metrics_c = {
            "concept": concepts[c] if c < len(concepts) else f"concept_{c}",
            "prevalence": prevalence,
            "precision": float(precision_score(gt, pred, zero_division=0)),
            "recall": float(recall_score(gt, pred, zero_division=0)),
            "f1": float(f1_score(gt, pred, zero_division=0)),
            "auc_roc": None,
        }
        if 0 < gt.sum() < len(gt):
            metrics_c["auc_roc"] = float(roc_auc_score(gt, prob))
        per_concept.append(metrics_c)

    all_f1s = [m["f1"] for m in per_concept]
    all_aucs = [m["auc_roc"] for m in per_concept if m["auc_roc"] is not None]
    all_prevalences = [m["prevalence"] for m in per_concept]
    overall_acc = float((cbl_binary == all_dino_labels).float().mean())

    return {
        "num_samples": N,
        "num_concepts": C,
        "overall_binary_accuracy": overall_acc,
        "mean_f1": float(np.mean(all_f1s)),
        "median_f1": float(np.median(all_f1s)),
        "mean_auc_roc": float(np.mean(all_aucs)) if all_aucs else None,
        "median_auc_roc": float(np.median(all_aucs)) if all_aucs else None,
        "mean_prevalence": float(np.mean(all_prevalences)),
        "concepts_with_any_positive": sum(1 for p in all_prevalences if p > 0),
        "per_concept": per_concept,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CBL against concept ground truth")
    parser.add_argument("--load_dir", type=str, required=True, help="Phase 1 checkpoint directory (with backbone.pt, cbl.pt)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (auto-detected from args.txt if omitted)")
    parser.add_argument("--concept_file", type=str, default=None, help="Concept file (auto-detected from args.txt if omitted)")
    parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="CLIP model name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k for precision metric")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--mode", type=str, default="clip", choices=["clip", "dino"],
        help="Evaluation mode: clip (vs CLIP similarities) or dino (vs DINO binary annotations)")
    parser.add_argument("--annotation_dir", type=str, default=None,
        help="DINO annotation directory (required for --mode dino)")
    parser.add_argument("--dino_confidence_threshold", type=float, default=0.10,
        help="DINO detection confidence threshold")
    parser.add_argument("--save_results", type=str, default=None, help="Path to save results JSON (default: <load_dir>/cbl_evaluation.json)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load Phase 1 checkpoint
    print(f"Loading Phase 1 checkpoint from {args.load_dir}")
    backbone, cbl, saved_args, num_concepts = load_phase1(args.load_dir, device)

    dataset_name = args.dataset or saved_args.get("dataset", "cifar100")

    # Load concepts
    concept_file = args.concept_file or saved_args.get("concept_file")
    if concept_file is None:
        concept_path = os.path.join(args.load_dir, "concepts.txt")
        if os.path.exists(concept_path):
            with open(concept_path) as f:
                concepts = [l.strip() for l in f if l.strip()]
        else:
            raise FileNotFoundError("No concept file found. Specify --concept_file.")
    else:
        concepts = data_utils.get_concepts(concept_file, saved_args.get("filter_set"))
    print(f"Loaded {len(concepts)} concepts (CBL has {num_concepts} outputs)")

    if args.mode == "dino":
        if args.annotation_dir is None:
            raise ValueError("--annotation_dir is required for --mode dino")

        from torch.utils.data import DataLoader
        from data.concept_dataset_vlg import DinoConceptDataset

        # Get backbone preprocessing
        if hasattr(backbone, "preprocess"):
            preprocess = backbone.preprocess
        else:
            _, preprocess = data_utils.get_target_model(saved_args.get("backbone", "resnet50"), device)

        # Load the official test/val set (held-out data)
        try:
            test_data = data_utils.get_data(f"{dataset_name}_test", preprocess=None)
        except Exception:
            test_data = data_utils.get_data(f"{dataset_name}_val", preprocess=None)

        # Wrap with DinoConceptDataset using val annotations
        dino_dataset = DinoConceptDataset(
            dataset_name, test_data, concepts,
            annotation_dir=args.annotation_dir, split_suffix="val",
            confidence_threshold=args.dino_confidence_threshold,
            preprocess=preprocess,
        )
        dino_loader = DataLoader(dino_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        print(f"\nEvaluating CBL vs DINO annotations on {dataset_name} test set ({len(dino_dataset)} samples)...")
        results = evaluate_cbl_dino(backbone, cbl, dino_loader, concepts, device)

        # Print summary
        print(f"\n{'='*60}")
        print(f"CBL vs DINO Evaluation Results")
        print(f"{'='*60}")
        print(f"Samples: {results['num_samples']}  Concepts: {results['num_concepts']}")
        print(f"Concepts with any positive annotation: {results['concepts_with_any_positive']}/{results['num_concepts']}")
        print(f"Mean concept prevalence: {results['mean_prevalence']:.4f}")
        print(f"\nOverall binary accuracy:  {results['overall_binary_accuracy']:.4f}")
        print(f"Mean F1:                  {results['mean_f1']:.4f}")
        print(f"Median F1:                {results['median_f1']:.4f}")
        if results['mean_auc_roc'] is not None:
            print(f"Mean AUC-ROC:             {results['mean_auc_roc']:.4f}")
            print(f"Median AUC-ROC:           {results['median_auc_roc']:.4f}")

        # Top/bottom 10 by F1
        sorted_by_f1 = sorted(results["per_concept"], key=lambda x: x["f1"], reverse=True)
        print(f"\nTop 10 concepts by F1:")
        for i, m in enumerate(sorted_by_f1[:10]):
            auc_str = f"AUC={m['auc_roc']:.3f}" if m['auc_roc'] is not None else "AUC=N/A"
            print(f"  {i+1:2d}. {m['concept']:40s}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  {auc_str}  prev={m['prevalence']:.3f}")

        print(f"\nBottom 10 concepts by F1:")
        for i, m in enumerate(sorted_by_f1[-10:]):
            auc_str = f"AUC={m['auc_roc']:.3f}" if m['auc_roc'] is not None else "AUC=N/A"
            print(f"  {i+1:2d}. {m['concept']:40s}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  {auc_str}  prev={m['prevalence']:.3f}")

        # Save
        results["load_dir"] = args.load_dir
        results["dataset"] = dataset_name
        results["mode"] = "dino"
        results["annotation_dir"] = args.annotation_dir
        results["confidence_threshold"] = args.dino_confidence_threshold

        save_path = args.save_results or os.path.join(args.load_dir, "cbl_evaluation_dino.json")
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")
        return

    # --- CLIP mode (original behavior) ---
    clip_name = args.clip_name or saved_args.get("clip_name", "ViT-B/16")

    # Load CLIP
    print(f"Loading CLIP {clip_name}...")
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    clip_model.eval()

    # Generate concept text embeddings
    concept_embeddings = generate_clip_text_embeddings(concepts, clip_model, device)
    print(f"Concept embeddings: {concept_embeddings.shape}")

    # Load test data with backbone's preprocessing (not CLIP's)
    if hasattr(backbone, "preprocess"):
        target_preprocess = backbone.preprocess
    else:
        _, target_preprocess = data_utils.get_target_model(saved_args.get("backbone", "resnet50"), device)

    # We need images preprocessed for BOTH backbone and CLIP.
    # Since they may differ, load raw data and apply both.
    # Use CLIP preprocessing for CLIP, backbone preprocessing for backbone.
    # Simplest: use backbone preprocess for the dataloader, run CLIP on same images.
    # CLIP's encode_image expects CLIP-preprocessed inputs, so we need a dual approach.

    # Load data with CLIP preprocessing (CLIP needs its own transforms)
    from torch.utils.data import DataLoader

    # For the backbone: use its preprocess
    try:
        test_data_backbone = data_utils.get_data(f"{dataset_name}_test", preprocess=target_preprocess)
    except Exception:
        test_data_backbone = data_utils.get_data(f"{dataset_name}_val", preprocess=target_preprocess)

    # For CLIP: use clip preprocess on the same images
    try:
        test_data_clip = data_utils.get_data(f"{dataset_name}_test", preprocess=clip_preprocess)
    except Exception:
        test_data_clip = data_utils.get_data(f"{dataset_name}_val", preprocess=clip_preprocess)

    # Create paired dataloader
    class PairedDataset(torch.utils.data.Dataset):
        def __init__(self, ds_backbone, ds_clip):
            self.ds_backbone = ds_backbone
            self.ds_clip = ds_clip
        def __len__(self):
            return len(self.ds_backbone)
        def __getitem__(self, idx):
            img_bb, label = self.ds_backbone[idx]
            img_clip, _ = self.ds_clip[idx]
            return img_bb, img_clip, label

    paired = PairedDataset(test_data_backbone, test_data_clip)
    loader = DataLoader(paired, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Evaluate
    print(f"\nEvaluating on {dataset_name} test set ({len(paired)} samples)...")
    all_cbl_preds = []
    all_clip_sims = []

    with torch.no_grad():
        for img_bb, img_clip, labels in loader:
            img_bb = img_bb.to(device)
            img_clip = img_clip.to(device)

            # CBL predictions
            features = backbone(img_bb)
            cbl_preds = cbl(features)

            # CLIP ground truth
            clip_img = clip_model.encode_image(img_clip).float()
            clip_img = clip_img / clip_img.norm(dim=-1, keepdim=True)
            clip_sims = clip_img @ concept_embeddings.T

            all_cbl_preds.append(cbl_preds.cpu())
            all_clip_sims.append(clip_sims.cpu())

    all_cbl_preds = torch.cat(all_cbl_preds, dim=0)
    all_clip_sims = torch.cat(all_clip_sims, dim=0)
    N, C = all_cbl_preds.shape
    print(f"Collected predictions: {N} samples, {C} concepts")

    # Compute metrics
    cbl_np = all_cbl_preds.numpy()
    clip_np = all_clip_sims.numpy()

    # Per-concept Pearson correlation
    per_concept_pearson = []
    for c in range(C):
        cc = np.corrcoef(cbl_np[:, c], clip_np[:, c])[0, 1]
        per_concept_pearson.append(float(cc) if not np.isnan(cc) else 0.0)
    per_concept_pearson = np.array(per_concept_pearson)

    # Per-concept Spearman correlation
    from scipy.stats import spearmanr
    per_concept_spearman = []
    for c in range(C):
        rho, _ = spearmanr(cbl_np[:, c], clip_np[:, c])
        per_concept_spearman.append(float(rho) if not np.isnan(rho) else 0.0)
    per_concept_spearman = np.array(per_concept_spearman)

    # MSE
    mse = float(torch.nn.functional.mse_loss(all_cbl_preds, all_clip_sims).item())

    # Per-sample cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(all_cbl_preds, all_clip_sims, dim=1)
    avg_cos_sim = float(cos_sim.mean().item())

    # Top-k precision
    top_k = args.top_k
    cbl_topk = torch.topk(all_cbl_preds, k=top_k, dim=1).indices
    clip_topk = torch.topk(all_clip_sims, k=top_k, dim=1).indices
    topk_prec = []
    for i in range(N):
        overlap = len(set(cbl_topk[i].tolist()) & set(clip_topk[i].tolist()))
        topk_prec.append(overlap / top_k)
    avg_topk_prec = float(np.mean(topk_prec))

    # Bottom concepts (worst Pearson)
    worst_idx = np.argsort(per_concept_pearson)[:10]
    best_idx = np.argsort(per_concept_pearson)[-10:][::-1]

    # Print results
    print(f"\n{'='*50}")
    print(f"CBL Evaluation Results")
    print(f"{'='*50}")
    print(f"MSE (CBL vs CLIP):          {mse:.6f}")
    print(f"Avg Cosine Similarity:      {avg_cos_sim:.4f}")
    print(f"Avg Top-{top_k} Precision:       {avg_topk_prec:.4f}")
    print(f"\nPearson Correlation (per-concept):")
    print(f"  Mean:   {per_concept_pearson.mean():.4f}")
    print(f"  Median: {np.median(per_concept_pearson):.4f}")
    print(f"  Std:    {per_concept_pearson.std():.4f}")
    print(f"  Min:    {per_concept_pearson.min():.4f}")
    print(f"  Max:    {per_concept_pearson.max():.4f}")
    print(f"\nSpearman Correlation (per-concept):")
    print(f"  Mean:   {per_concept_spearman.mean():.4f}")
    print(f"  Median: {np.median(per_concept_spearman):.4f}")
    print(f"  Std:    {per_concept_spearman.std():.4f}")
    print(f"  Min:    {per_concept_spearman.min():.4f}")
    print(f"  Max:    {per_concept_spearman.max():.4f}")

    print(f"\nTop 10 best-aligned concepts (Pearson):")
    for i, idx in enumerate(best_idx):
        print(f"  {i+1}. {concepts[idx]:40s}  r={per_concept_pearson[idx]:.4f}")

    print(f"\nTop 10 worst-aligned concepts (Pearson):")
    for i, idx in enumerate(worst_idx):
        print(f"  {i+1}. {concepts[idx]:40s}  r={per_concept_pearson[idx]:.4f}")

    # Save results
    results = {
        "load_dir": args.load_dir,
        "dataset": dataset_name,
        "num_samples": N,
        "num_concepts": C,
        "clip_name": clip_name,
        "mse": mse,
        "avg_cosine_similarity": avg_cos_sim,
        f"avg_top{top_k}_precision": avg_topk_prec,
        "pearson": {
            "mean": float(per_concept_pearson.mean()),
            "median": float(np.median(per_concept_pearson)),
            "std": float(per_concept_pearson.std()),
            "min": float(per_concept_pearson.min()),
            "max": float(per_concept_pearson.max()),
        },
        "spearman": {
            "mean": float(per_concept_spearman.mean()),
            "median": float(np.median(per_concept_spearman)),
            "std": float(per_concept_spearman.std()),
            "min": float(per_concept_spearman.min()),
            "max": float(per_concept_spearman.max()),
        },
        "per_concept_pearson": per_concept_pearson.tolist(),
        "per_concept_spearman": per_concept_spearman.tolist(),
        "concepts": concepts,
    }

    save_path = args.save_results or os.path.join(args.load_dir, "cbl_evaluation.json")
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
