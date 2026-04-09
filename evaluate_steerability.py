"""Evaluate steerability and interpretability of a trained VLG-CBM model.

Experiment 1: Test-time concept intervention curves
  - Progressively replace predicted concept activations with DINO ground-truth values
  - Measures accuracy improvement under importance-ranked vs random interventions

Experiment 3: Decision faithfulness (top-k pruning)
  - Zero out all but top-k weights per class in the final layer
  - Measures fraction of predictions changed vs original model

Usage:
    python evaluate_steerability.py \\
        --load_dir /path/to/checkpoint \\
        --dataset cifar100 \\
        --annotation_dir /path/to/annotations \\
        --output_dir /path/to/results \\
        --device cuda
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Label-free-CBM"))

from data import data_utils
from data.concept_dataset_vlg import DinoConceptDataset
from models.fed_vlgcbm import (
    Backbone, BackboneCLIP, ConceptLayer,
    NormalizationLayer, FinalLayer, FedVLGCBM,
)
from evaluate_cbl import load_phase1


# ── Model loading ─────────────────────────────────────────────────────────────

def load_full_model(load_dir, device):
    """Load backbone + CBL + normalization + final layer from a checkpoint dir."""
    backbone, cbl, saved_args, num_concepts = load_phase1(load_dir, device)

    norm_layer = NormalizationLayer.from_pretrained(load_dir, device=device)

    final_sd = torch.load(os.path.join(load_dir, "final.pt"), map_location=device)
    # final.pt state dict has keys "weight" [num_classes, num_concepts] and "bias" [num_classes]
    num_classes = final_sd["weight"].shape[0]
    final_layer = FinalLayer(num_concepts, num_classes, device=device)
    final_layer.load_state_dict(final_sd)

    model = FedVLGCBM(backbone, cbl, normalization=norm_layer, final_layer=final_layer)
    model.eval()
    model.to(device)

    return model, saved_args, num_concepts, num_classes


def load_concepts(load_dir, saved_args):
    concepts_path = os.path.join(load_dir, "concepts.txt")
    if os.path.exists(concepts_path):
        with open(concepts_path) as f:
            return [l.strip() for l in f if l.strip()]
    concept_file = saved_args.get("concept_file") or saved_args.get("concept_set")
    if concept_file is None:
        raise FileNotFoundError("concepts.txt not found and no concept_file in args.txt")
    return data_utils.get_concepts(concept_file, saved_args.get("filter_set"))


def build_test_loader(dataset_name, model, concepts, annotation_dir, batch_size, num_workers):
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
        raw_test = data_utils.get_data(f"{dataset_name}_test", preprocess=None)
    except Exception:
        raw_test = data_utils.get_data(f"{dataset_name}_val", preprocess=None)

    dino_dataset = DinoConceptDataset(
        dataset_name, raw_test, concepts,
        annotation_dir=annotation_dir, split_suffix="val",
        confidence_threshold=0.10, preprocess=preprocess,
    )
    loader = torch.utils.data.DataLoader(
        dino_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return loader, len(dino_dataset)


# ── Shared inference: collect concept activations + labels ────────────────────

def collect_activations(model, loader, device):
    """Run backbone + CBL + norm on all test images.

    Returns:
        a_tilde  [N, C]  concept activations (z-score normalized)
        dino_gt  [N, C]  DINO binary labels
        targets  [N]     integer class labels
    """
    all_a, all_dino, all_y = [], [], []
    with torch.no_grad():
        for images, concept_one_hot, targets in tqdm(loader, desc="Extracting activations", leave=False):
            images = images.to(device)
            h = model.backbone(images)
            c = model.cbl(h)
            a = model.normalization(c)
            all_a.append(a.cpu())
            all_dino.append(concept_one_hot.cpu())
            all_y.append(targets if torch.is_tensor(targets) else torch.tensor(targets))
    return (
        torch.cat(all_a, dim=0),   # [N, C]
        torch.cat(all_dino, dim=0),  # [N, C]
        torch.cat(all_y, dim=0),   # [N]
    )


# ── Experiment 1: Concept intervention curves ─────────────────────────────────

def run_intervention_experiment(a_tilde, dino_gt, targets, W_f, b_f, max_interventions=50, n_random_trials=5):
    """
    Args:
        a_tilde   [N, C]  normalized concept activations
        dino_gt   [N, C]  DINO binary labels (0.0 or 1.0)
        targets   [N]     true class labels
        W_f       [K, C]  final layer weights
        b_f       [K]     final layer bias
        max_interventions: max number of concepts to intervene on
        n_random_trials: number of random permutation seeds for the random-order baseline
    """
    N, C = a_tilde.shape
    max_interventions = min(max_interventions, C)

    # Original predictions (no intervention)
    logits_orig = a_tilde @ W_f.T + b_f  # [N, K]
    preds_orig = logits_orig.argmax(dim=1)  # [N]

    importance_acc = []
    random_acc = []

    for n in tqdm(range(max_interventions + 1), desc="Intervention sweep (importance)"):
        correct = 0
        for i in range(N):
            a_i = a_tilde[i].clone()
            dino_i = dino_gt[i]         # [C] binary
            pred_i = preds_orig[i].item()
            w_pred = W_f[pred_i]        # [C]

            # Rank by importance for this sample's predicted class
            rank = torch.argsort(w_pred.abs(), descending=True)

            # Intervene on top-n concepts
            for j in range(n):
                c_idx = rank[j].item()
                a_i[c_idx] = 1.0 if dino_i[c_idx].item() > 0.5 else -1.0

            pred_intervened = (a_i @ W_f.T + b_f).argmax().item()
            correct += int(pred_intervened == targets[i].item())
        importance_acc.append(correct / N)

    # Random order: average over multiple trials
    random_acc_trials = []
    for seed in range(n_random_trials):
        rng = torch.Generator()
        rng.manual_seed(seed)
        trial_acc = []
        for n in range(max_interventions + 1):
            correct = 0
            for i in range(N):
                a_i = a_tilde[i].clone()
                dino_i = dino_gt[i]
                rank = torch.randperm(C, generator=rng)
                for j in range(n):
                    c_idx = rank[j].item()
                    a_i[c_idx] = 1.0 if dino_i[c_idx].item() > 0.5 else -1.0
                pred_intervened = (a_i @ W_f.T + b_f).argmax().item()
                correct += int(pred_intervened == targets[i].item())
            trial_acc.append(correct / N)
        random_acc_trials.append(trial_acc)

    random_acc = np.mean(random_acc_trials, axis=0).tolist()

    avg_nonzero = float((W_f.abs() > 1e-5).sum(dim=1).float().mean().item())

    return {
        "importance_order": {
            "n_interventions": list(range(max_interventions + 1)),
            "accuracy": importance_acc,
        },
        "random_order": {
            "n_interventions": list(range(max_interventions + 1)),
            "accuracy": random_acc,
        },
        "num_test_samples": N,
        "avg_nonzero_weights_per_class": avg_nonzero,
    }


def plot_intervention(results, dataset_name, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available — skipping intervention plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    imp = results["importance_order"]
    rnd = results["random_order"]
    ax.plot(imp["n_interventions"], [a * 100 for a in imp["accuracy"]],
            label="Importance order", color="steelblue", linewidth=2)
    ax.plot(rnd["n_interventions"], [a * 100 for a in rnd["accuracy"]],
            label="Random order", color="coral", linewidth=2, linestyle="--")
    ax.set_xlabel("Number of intervened concepts")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Test-time concept intervention: {dataset_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "intervention_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved intervention curve to {out_path}")


# ── Experiment 3: Decision faithfulness (top-k pruning) ──────────────────────

def run_faithfulness_experiment(a_tilde, targets, W_f, b_f, k_values=(3, 5, 10, 15, 20, 25, 30)):
    N, C = a_tilde.shape
    K = W_f.shape[0]

    # Original predictions
    logits_orig = a_tilde @ W_f.T + b_f  # [N, K]
    preds_orig = logits_orig.argmax(dim=1)  # [N]
    acc_original = float((preds_orig == targets).float().mean().item())

    avg_nonzero = float((W_f.abs() > 1e-5).sum(dim=1).float().mean().item())

    results_by_k = {}
    for k in tqdm(k_values, desc="Faithfulness sweep (top-k)"):
        # Build pruned weight matrix: keep top-k per row by |magnitude|
        W_pruned = W_f.clone()
        for cls in range(K):
            row = W_pruned[cls]
            if k < C:
                # Zero out all but top-k
                threshold_val = row.abs().topk(k).values.min()
                mask = row.abs() < threshold_val
                W_pruned[cls][mask] = 0.0

        logits_pruned = a_tilde @ W_pruned.T + b_f
        preds_pruned = logits_pruned.argmax(dim=1)

        n_changed = int((preds_pruned != preds_orig).sum().item())
        pct_changed = n_changed / N
        acc_pruned = float((preds_pruned == targets).float().mean().item())

        results_by_k[str(k)] = {
            "pct_changed": pct_changed,
            "acc_pruned": acc_pruned,
            "n_changed": n_changed,
        }

    return {
        "original_accuracy": acc_original,
        "results_by_k": results_by_k,
        "num_test_samples": N,
        "model_nec": avg_nonzero,
    }


def plot_faithfulness(results, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available — skipping faithfulness plot")
        return

    k_vals = sorted(int(k) for k in results["results_by_k"])
    pct_changed = [results["results_by_k"][str(k)]["pct_changed"] * 100 for k in k_vals]
    acc_pruned = [results["results_by_k"][str(k)]["acc_pruned"] * 100 for k in k_vals]
    acc_original = results["original_accuracy"] * 100
    nec = results["model_nec"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1, color2 = "tomato", "steelblue"
    ax1.plot(k_vals, pct_changed, color=color1, linewidth=2, marker="o", label="% predictions changed")
    ax1.set_xlabel("k (concepts retained per class)")
    ax1.set_ylabel("% predictions changed vs original", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(k_vals, acc_pruned, color=color2, linewidth=2, marker="s", linestyle="--", label="Pruned accuracy (%)")
    ax2.axhline(acc_original, color=color2, linestyle=":", alpha=0.6, label=f"Original acc ({acc_original:.1f}%)")
    ax2.set_ylabel("Test accuracy (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # NEC reference line
    ax1.axvline(nec, color="gray", linestyle="--", alpha=0.7, label=f"NEC={nec:.1f}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Decision faithfulness: top-k pruning")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "faithfulness_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved faithfulness curve to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Steerability and faithfulness evaluation for VLG-CBM")
    parser.add_argument("--load_dir", type=str, required=True,
        help="Trained checkpoint directory (backbone.pt, cbl.pt, final.pt, ...)")
    parser.add_argument("--dataset", type=str, default=None,
        help="Dataset name (auto-detected from args.txt if omitted)")
    parser.add_argument("--annotation_dir", type=str, required=True,
        help="DINO annotation directory")
    parser.add_argument("--output_dir", type=str, default=None,
        help="Directory to save results (default: <load_dir>/steerability/)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_interventions", type=int, default=50,
        help="Max number of concepts to intervene on (Experiment 1)")
    parser.add_argument("--n_random_trials", type=int, default=5,
        help="Number of random permutation seeds for Experiment 1 baseline")
    parser.add_argument("--k_values", type=int, nargs="+", default=[3, 5, 10, 15, 20, 25, 30],
        help="k values for top-k pruning (Experiment 3)")
    parser.add_argument("--skip_intervention", action="store_true",
        help="Skip Experiment 1 (intervention curves)")
    parser.add_argument("--skip_faithfulness", action="store_true",
        help="Skip Experiment 3 (faithfulness pruning)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or os.path.join(args.load_dir, "steerability")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {args.load_dir}")
    model, saved_args, num_concepts, num_classes = load_full_model(args.load_dir, device)
    concepts = load_concepts(args.load_dir, saved_args)
    dataset_name = args.dataset or saved_args.get("dataset", "cifar100")

    print(f"Dataset: {dataset_name}  |  Concepts: {num_concepts}  |  Classes: {num_classes}")
    if len(concepts) != num_concepts:
        print(f"  [WARN] {len(concepts)} concept names vs {num_concepts} CBL outputs — using {num_concepts} concepts")
        concepts = concepts[:num_concepts]

    W_f = model.final_layer.weight.data.cpu()  # [num_classes, num_concepts]
    b_f = model.final_layer.bias.data.cpu()    # [num_classes]
    nec = float((W_f.abs() > 1e-5).sum(dim=1).float().mean().item())
    print(f"Final layer NEC (avg nonzero per class): {nec:.2f}")

    print(f"\nBuilding test dataloader...")
    loader, n_test = build_test_loader(
        dataset_name, model, concepts, args.annotation_dir,
        args.batch_size, args.num_workers,
    )
    print(f"Test samples: {n_test}")

    print(f"\nCollecting concept activations on test set...")
    a_tilde, dino_gt, targets = collect_activations(model, loader, device)
    # Move to CPU for per-sample loops
    a_tilde = a_tilde.cpu()
    dino_gt = dino_gt.cpu()
    targets = targets.cpu()

    # ── Experiment 1 ──────────────────────────────────────────────────────────
    if not args.skip_intervention:
        max_iv = min(args.max_interventions, num_concepts)
        print(f"\n{'='*60}")
        print(f"Experiment 1: Concept intervention curves (max={max_iv})")
        print(f"{'='*60}")

        intervention_results = run_intervention_experiment(
            a_tilde, dino_gt, targets, W_f, b_f,
            max_interventions=max_iv,
            n_random_trials=args.n_random_trials,
        )

        out_path = os.path.join(output_dir, "intervention_results.json")
        with open(out_path, "w") as f:
            json.dump(intervention_results, f, indent=2)
        print(f"  Saved to {out_path}")

        plot_intervention(intervention_results, dataset_name, output_dir)

        baseline_acc = intervention_results["importance_order"]["accuracy"][0]
        max_acc = intervention_results["importance_order"]["accuracy"][-1]
        print(f"  Baseline accuracy (0 interventions): {baseline_acc:.4f}")
        print(f"  Accuracy after {max_iv} importance interventions: {max_acc:.4f}")
        print(f"  Gain: {max_acc - baseline_acc:+.4f}")
    else:
        intervention_results = None
        print("\nSkipping Experiment 1 (--skip_intervention)")

    # ── Experiment 3 ──────────────────────────────────────────────────────────
    if not args.skip_faithfulness:
        print(f"\n{'='*60}")
        print(f"Experiment 3: Decision faithfulness (top-k pruning)")
        print(f"{'='*60}")

        faithfulness_results = run_faithfulness_experiment(
            a_tilde, targets, W_f, b_f,
            k_values=args.k_values,
        )

        out_path = os.path.join(output_dir, "faithfulness_results.json")
        with open(out_path, "w") as f:
            json.dump(faithfulness_results, f, indent=2)
        print(f"  Saved to {out_path}")

        plot_faithfulness(faithfulness_results, output_dir)
    else:
        faithfulness_results = None
        print("\nSkipping Experiment 3 (--skip_faithfulness)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Model:    {args.load_dir}")
    print(f"Dataset:  {dataset_name}  ({n_test} test samples)")
    print(f"NEC:      {nec:.2f}")

    if intervention_results is not None:
        imp_acc = intervention_results["importance_order"]["accuracy"]
        rnd_acc = intervention_results["random_order"]["accuracy"]
        print(f"\nExperiment 1 — Intervention curves:")
        print(f"  Baseline accuracy:           {imp_acc[0]:.4f}")
        for n in [5, 10, 20, 50]:
            if n <= len(imp_acc) - 1:
                print(f"  After {n:2d} importance intervens:  {imp_acc[n]:.4f}  (random: {rnd_acc[n]:.4f})")

    if faithfulness_results is not None:
        print(f"\nExperiment 3 — Decision faithfulness:")
        print(f"  Original accuracy: {faithfulness_results['original_accuracy']:.4f}")
        print(f"  {'k':>4}  {'% changed':>10}  {'acc_pruned':>10}")
        for k in sorted(int(k) for k in faithfulness_results["results_by_k"]):
            r = faithfulness_results["results_by_k"][str(k)]
            print(f"  {k:>4}  {r['pct_changed']*100:>9.2f}%  {r['acc_pruned']:>10.4f}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
