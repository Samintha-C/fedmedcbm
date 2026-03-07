"""Utility functions for advanced CBL diagnostics.

Six analyses beyond standard F1 / AUC:
  1. diag_client_geometry   — activation subspace geometry across client partitions
  2. diag_calibration       — ECE, reliability bins, logit distribution statistics
  3. diag_cohens_d          — per-concept logit separation quality
  4. diag_coactivation      — pairwise Pearson correlation and effective rank
  5. diag_class_conditional — class-mean concept patterns and linear-probe accuracy
  6. diag_jaccard_agreement — per-concept Jaccard similarity between two CBLs

All functions accept pre-extracted numpy arrays (logits, labels) so they are
independent of the data-loading infrastructure and easy to unit-test.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import subspace_angles as _subspace_angles
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _dirichlet_split(
    class_labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[list[int]]:
    """Partition sample indices via per-class Dirichlet draw (standard FL setup)."""
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for cls in np.unique(class_labels):
        idx = np.where(class_labels == cls)[0]
        props = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for k, split in enumerate(np.split(idx, cuts)):
            client_indices[k].extend(split.tolist())
    return client_indices


def _pca_basis(X: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors of centred X, returned as columns [d, k]."""
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T


def _effective_rank(X: np.ndarray) -> float:
    """Effective rank = exp(entropy of normalised singular values) of matrix X."""
    _, sv, _ = np.linalg.svd(X, full_matrices=False)
    sv = sv[sv > 1e-10]
    if len(sv) == 0:
        return 1.0
    sv_n = sv / sv.sum()
    return float(np.exp(-np.sum(sv_n * np.log(sv_n + 1e-12))))


def _ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict]]:
    """Expected Calibration Error over flat (probability, binary label) arrays."""
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    n = len(probs)
    for b in range(n_bins):
        m = (probs >= boundaries[b]) & (probs < boundaries[b + 1])
        if m.sum() == 0:
            continue
        conf = float(probs[m].mean())
        acc  = float(labels[m].mean())
        ece += m.sum() * abs(conf - acc)
        bins.append({
            "lo": float(boundaries[b]),
            "hi": float(boundaries[b + 1]),
            "avg_conf": conf,
            "avg_acc":  acc,
            "n":        int(m.sum()),
        })
    return ece / n, bins


def _cohens_d(pos: np.ndarray, neg: np.ndarray) -> float | None:
    n1, n2 = len(pos), len(neg)
    if n1 < 2 or n2 < 2:
        return None
    pooled_var = ((n1 - 1) * pos.var() + (n2 - 1) * neg.var()) / (n1 + n2 - 2)
    return float((pos.mean() - neg.mean()) / (np.sqrt(pooled_var) + 1e-10))


# ── Diagnostic 1: Activation geometry across client partitions ────────────────

def diag_client_geometry(
    train_logits: np.ndarray,
    train_class_labels: np.ndarray,
    num_clients: int = 5,
    alpha: float = 0.5,
    seed: int = 42,
    top_k: int = 10,
) -> dict:
    """Activation subspace geometry across Dirichlet-simulated client partitions.

    Tests the non-IID subspace hypothesis: do client data partitions live in
    near-orthogonal concept subspaces (centralised CBL) or more-aligned ones
    (federated CBL)?

    Args:
        train_logits:       [N, C] pre-extracted concept-space logits.
        train_class_labels: [N]   integer class label per training sample.
        num_clients:        Number of client partitions to simulate.
        alpha:              Dirichlet concentration (lower = more heterogeneous).
        seed:               RNG seed for reproducible partitioning.
        top_k:              Leading PCA directions per client used for subspace
                            angle computation.

    Returns:
        dict containing cosine similarity matrix, aggregate stats, and
        (when scipy is available) principal angles between all client-pair subspaces.
    """
    probs = _sigmoid(train_logits)                                      # [N, C]
    client_indices = _dirichlet_split(train_class_labels, num_clients, alpha, seed)
    client_probs   = [probs[idxs] for idxs in client_indices]

    # Mean-activation cosine-similarity matrix
    means   = np.stack([a.mean(axis=0) for a in client_probs])          # [K, C]
    norms   = np.linalg.norm(means, axis=1, keepdims=True) + 1e-10
    means_n = means / norms
    cos_mat = (means_n @ means_n.T).tolist()                            # [K, K]

    K   = num_clients
    off = [cos_mat[i][j] for i in range(K) for j in range(i + 1, K)]

    # Principal subspace angles between all client pairs
    actual_k = min(top_k, train_logits.shape[1],
                   *(max(1, len(a) - 1) for a in client_probs))
    principal_angles: dict = {}
    if _HAS_SCIPY and actual_k > 0:
        for i in range(K):
            for j in range(i + 1, K):
                b1  = _pca_basis(client_probs[i], actual_k)
                b2  = _pca_basis(client_probs[j], actual_k)
                ang = np.degrees(_subspace_angles(b1, b2))
                principal_angles[f"client_{i}_vs_{j}"] = {
                    "min_angle_deg":  float(ang.min()),
                    "mean_angle_deg": float(ang.mean()),
                    "max_angle_deg":  float(ang.max()),
                    "angles_deg":     ang.tolist(),
                }
    elif not _HAS_SCIPY:
        principal_angles["_warning"] = "scipy unavailable — principal angles skipped"

    return {
        "client_sizes":             [len(i) for i in client_indices],
        "cosine_similarity_matrix": cos_mat,
        "mean_pairwise_cosine":     float(np.mean(off)),
        "min_pairwise_cosine":      float(np.min(off)),
        "std_pairwise_cosine":      float(np.std(off)),
        "principal_angles":         principal_angles,
        "top_k_used":               actual_k,
        "scipy_available":          _HAS_SCIPY,
    }


# ── Diagnostic 2: Calibration ─────────────────────────────────────────────────

def diag_calibration(
    logits: np.ndarray,
    dino_labels: np.ndarray,
    concepts: list[str],
    n_bins: int = 10,
) -> dict:
    """Calibration: global ECE, per-concept ECE, and logit distribution statistics.

    The logit_statistics section directly diagnoses the F1@0.5 collapse: if
    frac_pos_below_zero is high, positive-class logits cluster below zero and
    a sigmoid threshold of 0.5 misses almost everything.

    Args:
        logits:      [N, C] raw concept logits (pre-sigmoid).
        dino_labels: [N, C] binary DINO ground-truth labels.
        concepts:    List of C concept name strings.
        n_bins:      Number of reliability-diagram bins.
    """
    probs = _sigmoid(logits)
    N, C  = probs.shape

    global_ece, global_bins = _ece(probs.ravel(), dino_labels.ravel(), n_bins)

    per_concept_ece: list[dict] = []
    for c in range(C):
        n_pos = int(dino_labels[:, c].sum())
        if n_pos == 0:
            continue
        ece_c, _ = _ece(probs[:, c], dino_labels[:, c], n_bins)
        per_concept_ece.append({
            "concept":    concepts[c] if c < len(concepts) else f"concept_{c}",
            "ece":        float(ece_c),
            "n_positive": n_pos,
        })
    per_concept_ece.sort(key=lambda x: -x["ece"])

    # Logit-distribution statistics split by positive / negative DINO label
    pos_mask   = dino_labels.astype(bool)
    pos_logits = logits[pos_mask]
    neg_logits = logits[~pos_mask]

    hist_pos, edges = np.histogram(pos_logits, bins=50, range=(-10.0, 10.0))
    hist_neg, _     = np.histogram(neg_logits, bins=50, range=(-10.0, 10.0))

    return {
        "global_ece":              float(global_ece),
        "mean_per_concept_ece":    float(np.mean([x["ece"] for x in per_concept_ece])) if per_concept_ece else None,
        "global_reliability_bins": global_bins,
        "per_concept_ece_worst20": per_concept_ece[:20],
        "logit_statistics": {
            "pos_mean":            float(pos_logits.mean())     if len(pos_logits) else None,
            "pos_std":             float(pos_logits.std())      if len(pos_logits) else None,
            "pos_median":          float(np.median(pos_logits)) if len(pos_logits) else None,
            "neg_mean":            float(neg_logits.mean()),
            "neg_std":             float(neg_logits.std()),
            "neg_median":          float(np.median(neg_logits)),
            "frac_pos_below_zero": float((pos_logits < 0).mean()) if len(pos_logits) else None,
            "frac_neg_above_zero": float((neg_logits > 0).mean()),
        },
        "logit_histogram": {
            "bin_edges":  edges.tolist(),
            "pos_counts": hist_pos.tolist(),
            "neg_counts": hist_neg.tolist(),
        },
    }


# ── Diagnostic 3: Per-concept Cohen's d ──────────────────────────────────────

def diag_cohens_d(
    logits: np.ndarray,
    dino_labels: np.ndarray,
    concepts: list[str],
) -> dict:
    """Per-concept Cohen's d: logit separation quality between positive and negative samples.

    Distinguishes reduced-separation (distributions overlap more → lower d for all
    concepts) from outlier-driven AUC drops (a few concepts with d ≈ 0 drag the mean).

    Args:
        logits:      [N, C] raw concept logits.
        dino_labels: [N, C] binary DINO ground-truth labels.
        concepts:    List of C concept name strings.
    """
    results: list[dict] = []
    for c in range(logits.shape[1]):
        gt    = dino_labels[:, c].astype(bool)
        n_pos = int(gt.sum())
        if n_pos == 0:
            continue
        pos_l, neg_l = logits[gt, c], logits[~gt, c]
        d = _cohens_d(pos_l, neg_l)
        results.append({
            "concept":        concepts[c] if c < len(concepts) else f"concept_{c}",
            "cohens_d":       d,
            "n_positive":     n_pos,
            "n_negative":     int((~gt).sum()),
            "mean_pos_logit": float(pos_l.mean()),
            "mean_neg_logit": float(neg_l.mean()),
            "std_pos_logit":  float(pos_l.std()),
            "std_neg_logit":  float(neg_l.std()),
        })
    results.sort(key=lambda x: -(x["cohens_d"] or -999.0))

    valid_ds = [r["cohens_d"] for r in results if r["cohens_d"] is not None]
    return {
        "mean_cohens_d":    float(np.mean(valid_ds))   if valid_ds else None,
        "median_cohens_d":  float(np.median(valid_ds)) if valid_ds else None,
        "frac_d_above_1_0": float(np.mean(np.array(valid_ds) > 1.0)) if valid_ds else None,
        "frac_d_above_0_5": float(np.mean(np.array(valid_ds) > 0.5)) if valid_ds else None,
        "per_concept":      results,
    }


# ── Diagnostic 4: Concept co-activation structure ────────────────────────────

def diag_coactivation(logits: np.ndarray) -> dict:
    """Concept co-activation: pairwise Pearson correlation and effective rank.

    Higher inter-concept correlation + lower effective rank → CBL encodes
    redundant information (concepts fire together instead of independently).
    This can explain why a CBL with worse DINO F1 still works for classification:
    correlated concepts provide more paths through a sparse final layer.

    Args:
        logits: [N, C] raw concept logits (pre-sigmoid).
    """
    probs    = _sigmoid(logits)
    C        = probs.shape[1]
    corr     = np.corrcoef(probs.T)              # [C, C]
    off_mask = ~np.eye(C, dtype=bool)
    abs_off  = np.abs(corr[off_mask])

    pairs = [
        (i, j, float(corr[i, j]))
        for i in range(C) for j in range(i + 1, C)
    ]
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    return {
        "effective_rank":           _effective_rank(probs),
        "effective_rank_fraction":  _effective_rank(probs) / C,
        "mean_abs_pairwise_corr":   float(abs_off.mean()),
        "median_abs_pairwise_corr": float(np.median(abs_off)),
        "frac_pairs_above_0_3":     float((abs_off > 0.3).mean()),
        "frac_pairs_above_0_5":     float((abs_off > 0.5).mean()),
        "top20_correlated_pairs":   [
            {"i": i, "j": j, "corr": r} for i, j, r in pairs[:20]
        ],
    }


# ── Diagnostic 5: Class-conditional concept patterns + linear probe ───────────

def diag_class_conditional(
    train_logits: np.ndarray,
    train_class_labels: np.ndarray,
    val_logits: np.ndarray,
    val_class_labels: np.ndarray,
) -> dict:
    """Class-conditional concept patterns and linear-probe classification accuracy.

    Measures how discriminative the concept representation is for classification,
    independent of federated optimisation. Two outcomes:
      - Centralised CBL → higher probe accuracy: concept rep is better; the
        problem is purely in federated final-layer optimisation.
      - Federated CBL → comparable/higher probe accuracy: federated CBL creates
        a more classification-friendly representation despite worse DINO F1.

    Args:
        train_logits:       [N_train, C] concept logits for training split.
        train_class_labels: [N_train]   integer class labels for training split.
        val_logits:         [N_val, C]  concept logits for validation split.
        val_class_labels:   [N_val]     integer class labels for validation split.
    """
    if not _HAS_SKLEARN:
        return {"error": "scikit-learn not available — install with: pip install scikit-learn"}

    train_probs = _sigmoid(train_logits)
    val_probs   = _sigmoid(val_logits)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_probs)
    X_val   = scaler.transform(val_probs)

    lr = LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs",
        multi_class="multinomial", random_state=42,
    )
    lr.fit(X_train, train_class_labels)
    val_acc = float(lr.score(X_val, val_class_labels))

    # Per-class mean activation vector in val set
    classes      = np.unique(val_class_labels)
    class_means  = [val_probs[val_class_labels == c].mean(axis=0) for c in classes]
    arr          = np.stack(class_means)                     # [num_classes, C]
    norms        = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    cos_inter    = (arr / norms) @ (arr / norms).T
    np.fill_diagonal(cos_inter, 0.0)
    n_cls        = len(classes)
    mean_inter_cos = float(cos_inter.sum() / (n_cls * (n_cls - 1)))

    # Top concepts by mean absolute LR coefficient across all classes
    lr_mag  = np.abs(lr.coef_).mean(axis=0)
    top_idx = np.argsort(lr_mag)[::-1][:20].tolist()

    return {
        "linear_probe_val_accuracy":              val_acc,
        "mean_inter_class_cosine_of_class_means": mean_inter_cos,
        "top20_concept_indices_by_lr_weight":     top_idx,
        "top20_lr_weights":                       lr_mag[top_idx].tolist(),
        "per_class_mean_activations":             {
            int(c): class_means[i].tolist() for i, c in enumerate(classes)
        },
    }


# ── Diagnostic 6: Concept-level Jaccard agreement between two CBLs ────────────

def diag_jaccard_agreement(
    logits1: np.ndarray,
    logits2: np.ndarray,
    concepts: list[str],
    threshold: float = 0.5,
) -> dict:
    """Per-concept Jaccard similarity between the activated sample sets of two CBLs.

    Low Jaccard for a concept → the two CBLs disagree on which samples activate
    it, meaning federation has distorted that concept's semantics the most. This
    guides targeted fixes (e.g., re-weighting or freezing specific concepts).

    Args:
        logits1:   [N, C1] raw logits from the primary CBL.
        logits2:   [N, C2] raw logits from the comparison CBL (same N samples).
        concepts:  Concept name strings (min(C1, C2) entries used).
        threshold: Sigmoid threshold for binarising activations.
    """
    p1 = _sigmoid(logits1) > threshold
    p2 = _sigmoid(logits2) > threshold
    C  = min(logits1.shape[1], logits2.shape[1])

    results: list[dict] = []
    for c in range(C):
        inter   = int((p1[:, c] & p2[:, c]).sum())
        union   = int((p1[:, c] | p2[:, c]).sum())
        jaccard = inter / union if union > 0 else 1.0
        results.append({
            "concept":        concepts[c] if c < len(concepts) else f"concept_{c}",
            "jaccard":        float(jaccard),
            "n_active_cbl1":  int(p1[:, c].sum()),
            "n_active_cbl2":  int(p2[:, c].sum()),
            "n_intersection": inter,
            "n_union":        union,
        })
    results.sort(key=lambda x: x["jaccard"])

    jaccards = [r["jaccard"] for r in results]
    return {
        "mean_jaccard":        float(np.mean(jaccards)),
        "median_jaccard":      float(np.median(jaccards)),
        "frac_below_0_3":      float(np.mean(np.array(jaccards) < 0.3)),
        "frac_below_0_5":      float(np.mean(np.array(jaccards) < 0.5)),
        "bottom20_by_jaccard": results[:20],
        "top20_by_jaccard":    results[-20:],
        "per_concept":         results,
    }
