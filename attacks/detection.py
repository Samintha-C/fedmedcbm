"""
Intervention-based detection for federated CBM poisoning.

For each client, each class k:
  - "Importance-ordered intervention": zero the bottom `fraction` of concepts by |W[k,:]|
    and measure accuracy on local class-k val samples.
  - "Random intervention": zero a random `fraction` of concepts (averaged over trials).
  - Score[k] = importance_acc - random_acc

High score → model's concept-class associations for class k are coherent.
Low score  → associations are noisy (consistent with label-flip poisoning).

Server-side suspicion aggregation:
  For each class k, compute mean/std of Score[i,k] across clients.
  A client's suspicion score is the mean positive z-score deviation (how much
  below federation average each of their per-class scores is).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch


def compute_intervention_scores(
    val_feats: torch.Tensor,       # [N, num_concepts]
    val_labels: torch.Tensor,      # [N]
    weight_matrix: torch.Tensor,   # [num_classes, num_concepts]
    num_classes: int,
    fraction: float = 0.5,
    n_trials: int = 5,
) -> Dict[int, float]:
    """
    Compute per-class coherence scores for one client.

    Returns a dict {class_k: score}.  Score is NaN for classes with no val samples.
    """
    W = weight_matrix.detach()  # [num_classes, num_concepts]
    num_concepts = W.shape[1]
    n_zero = max(1, int(fraction * num_concepts))
    device = W.device

    val_feats = val_feats.to(device)
    val_labels = val_labels.to(device)

    scores: Dict[int, float] = {}
    for k in range(num_classes):
        mask = val_labels == k
        if mask.sum() == 0:
            scores[k] = math.nan
            continue

        feats_k = val_feats[mask]       # [n_k, num_concepts]
        labels_k = val_labels[mask]     # [n_k]

        # Importance mask: keep top-(1-fraction) concepts by |W[k,:]|
        importance_order = W[k].abs().argsort(descending=True)
        keep_mask = torch.zeros(num_concepts, device=device)
        keep_mask[importance_order[: num_concepts - n_zero]] = 1.0

        feats_imp = feats_k * keep_mask.unsqueeze(0)
        logits_imp = feats_imp @ W.T
        acc_imp = (logits_imp.argmax(dim=1) == labels_k).float().mean().item()

        # Random baseline: average over n_trials random masks
        acc_rand_sum = 0.0
        for _ in range(n_trials):
            perm = torch.randperm(num_concepts, device=device)
            rand_mask = torch.zeros(num_concepts, device=device)
            rand_mask[perm[: num_concepts - n_zero]] = 1.0
            feats_rand = feats_k * rand_mask.unsqueeze(0)
            logits_rand = feats_rand @ W.T
            acc_rand_sum += (logits_rand.argmax(dim=1) == labels_k).float().mean().item()
        acc_rand = acc_rand_sum / n_trials

        scores[k] = acc_imp - acc_rand

    return scores


def compute_suspicion(
    all_client_scores: List[Dict[int, float]],
    num_classes: int,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Aggregate per-client per-class scores into suspicion signals.

    Returns:
        suspicion_scores  : List[float] length n_clients — higher = more suspicious
        class_mean_scores : List[float] length num_classes — federation mean per class
        class_var_scores  : List[float] length num_classes — federation variance per class
    """
    import math as _math

    n_clients = len(all_client_scores)

    # Build score matrix [n_clients, num_classes]; NaN where no val samples
    matrix = [[all_client_scores[i].get(k, _math.nan) for k in range(num_classes)]
              for i in range(n_clients)]

    # Per-class stats ignoring NaN
    class_means: List[float] = []
    class_stds: List[float] = []
    for k in range(num_classes):
        vals = [matrix[i][k] for i in range(n_clients) if not _math.isnan(matrix[i][k])]
        if vals:
            mean_k = sum(vals) / len(vals)
            var_k = sum((v - mean_k) ** 2 for v in vals) / max(len(vals), 1)
            class_means.append(mean_k)
            class_stds.append(var_k ** 0.5)
        else:
            class_means.append(0.0)
            class_stds.append(0.0)

    # Per-client suspicion: mean of positive z-deviations across classes
    suspicion: List[float] = []
    for i in range(n_clients):
        deviations = []
        for k in range(num_classes):
            s = matrix[i][k]
            if _math.isnan(s):
                continue
            std_k = class_stds[k] if class_stds[k] > 1e-6 else 1e-6
            # Positive z-deviation: client i scored lower than the mean
            z = (class_means[k] - s) / std_k
            deviations.append(max(0.0, z))
        suspicion.append(sum(deviations) / max(len(deviations), 1))

    class_vars = [s ** 2 for s in class_stds]
    return suspicion, class_means, class_vars


def flag_clients(suspicion_scores: List[float], threshold: float) -> List[int]:
    """Return indices of clients whose suspicion score exceeds the threshold."""
    return [i for i, s in enumerate(suspicion_scores) if s >= threshold]


def top_suspicious_classes(
    class_means: List[float],
    class_vars: List[float],
    k: int = 5,
) -> List[int]:
    """
    Return the k class indices most likely to be under attack.

    Combines low mean (corrupted associations) and high variance (client disagreement).
    """
    num_classes = len(class_means)
    # Normalise each signal to [0, 1] so they're comparable
    max_var = max(class_vars) if max(class_vars) > 1e-8 else 1.0
    min_mean = min(class_means)
    range_mean = max(class_means) - min_mean if max(class_means) > min_mean else 1.0

    scores = []
    for c in range(num_classes):
        low_mean = 1.0 - (class_means[c] - min_mean) / range_mean   # higher = lower mean
        high_var = class_vars[c] / max_var
        scores.append((low_mean + high_var, c))

    scores.sort(reverse=True)
    return [c for _, c in scores[:k]]
