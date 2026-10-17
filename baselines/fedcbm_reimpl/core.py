"""FedCBM (Yang & Long, 2023), following the F-CM authors' reimplementation.

Reference: github.com/francescoTheSantis/F-CMs, src/fedcbm.py (MIT), commit
31cbb14. That code trains one concept at a time in a numpy loop, which is fine
for their ~10-100 concepts but days of CPU at ours (up to ~4.3K concepts and
~1.2M images). This file is the same math in torch, all concepts at once.
selftest.py checks it against the reference.

Stage A (shared concept bank).
    One bias-free linear SVM per concept on frozen embeddings, labels in
    {-1, +1}. Each round every client computes the hinge subgradient
    -sum_{i: y_i <h_i, w> < 1} y_i h_i over its samples, the server sums them,
    divides by the total sample count, adds l2 * w, and takes ONE step of size
    lr (reference: local_epochs = 1). A concept with no positives or no
    negatives anywhere keeps w = 0, as in the reference. Concepts never
    interact, so stacking them into W [C, D] changes nothing but speed.

Projection.
    z_c(h) = <h, w_c> / ||w_c||^2, and a zero-norm row divides by 1.

Stage B (personalized heads).
    Each client fits its own multinomial logistic regression on its projected
    scores, over the classes it actually holds, and the heads are never
    aggregated. The reference uses sklearn LogisticRegression(lbfgs, C=1,
    max_iter=200). fit_head() minimizes the same objective sklearn does,
    mean cross-entropy + ||W||^2 / (2 C n) with an unpenalized intercept, using
    torch L-BFGS.

Evaluation.
    The reference default scores every client's head on ONE global test set
    and reports the mean over clients. personalized_test_split() adds per-client
    test sets that follow each client's training class mix.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Head:
    """A client's private head over the classes it holds (`classes` sorted)."""
    weight: torch.Tensor  # [K_m, C]
    bias: torch.Tensor    # [K_m]
    classes: torch.Tensor  # [K_m] global class ids
    constant: Optional[int] = None  # reference _ConstantHead: one class only

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        if self.constant is not None:
            return torch.full((z.shape[0],), self.constant, dtype=torch.long, device=z.device)
        w = self.weight.to(z.device, z.dtype)
        b = self.bias.to(z.device, z.dtype)
        return self.classes.to(z.device)[(z @ w.T + b).argmax(dim=1)]


def _rows(n: int, chunk: int):
    for s in range(0, n, chunk):
        yield s, min(s + chunk, n)


def train_cav_bank(
    feats: torch.Tensor,
    concepts: torch.Tensor,
    rows: Optional[torch.Tensor] = None,
    rounds: int = 200,
    lr: float = 0.01,
    l2: float = 1e-4,
    device: str = "cuda",
    chunk: int = 32768,
    log_every: int = 50,
    log: Callable[[str], None] = print,
    balanced: bool = False,
    bias: bool = False,
    stats: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, object]:
    """Stage A on every training row the clients hold.

    feats [N, D] (any float dtype) and concepts [N, C] (bool or 0/1) may live on
    CPU or GPU. `rows` selects the rows held by clients (None = all). The
    reference sums the per-client subgradients and divides by the total count,
    which equals one pass over the union of the clients' rows, so the client
    split does not enter Stage A at all.

    Defaults reproduce the reference exactly. The three options below are our
    STRONGER variant, for labels as imbalanced as ours (~1-6% positive), where
    the bias-free reference collapses to scoring every image "absent":
      balanced  weight each concept's positives by n / (2 n_pos) and negatives by
                n / (2 n_neg) (sklearn's class_weight="balanced"), as training a
                CAV on equal-sized concept and random sets does. Needs each
                concept's global positive count: one C-vector per client.
      bias      an unregularized intercept per concept, as in the original
                CAV / linear-SVM recipe (the reference drops it).
      stats     {"mean", "std"} from feature_stats(): standardize features first,
                with statistics aggregated from per-client sums.
    """
    if rows is not None:
        feats, concepts = feats[rows], concepts[rows]
    n, d = feats.shape
    c = concepts.shape[1]

    pos = torch.zeros(c, dtype=torch.long)
    for s, e in _rows(n, 1 << 16):
        pos += concepts[s:e].to(torch.long).sum(dim=0).cpu()
    trainable = (pos > 0) & (pos < n)
    mask = trainable.to(device, torch.float32).unsqueeze(1)

    w = torch.zeros(c, d, dtype=torch.float32, device=device)
    b = torch.zeros(c, dtype=torch.float32, device=device)
    history: List[Dict[str, float]] = []
    mu = sd = None
    if stats is not None:
        mu, sd = stats["mean"].to(device), stats["std"].to(device)
    if balanced:
        posf = pos.to(device, torch.float32)
        w_pos = torch.where(posf > 0, n / (2.0 * posf.clamp(min=1)), torch.zeros_like(posf))
        w_neg = torch.where(posf < n, n / (2.0 * (n - posf).clamp(min=1)), torch.zeros_like(posf))

    def one_pass(w_cur: torch.Tensor, b_cur: torch.Tensor, want_grad: bool):
        g = torch.zeros_like(w_cur) if want_grad else None
        gb = torch.zeros_like(b_cur) if want_grad else None
        hinge = torch.zeros(c, dtype=torch.float64, device=device)
        active = torch.zeros(c, dtype=torch.float64, device=device)
        for s, e in _rows(n, chunk):
            x = feats[s:e].to(device, non_blocking=True).float()
            if mu is not None:
                x = (x - mu) / sd
            # Out of place: when concepts are already float on `device`, .to()/.float()
            # return the SAME storage, and an in-place mul_/sub_ would overwrite the
            # labels every round (caught by selftest on CPU).
            y = concepts[s:e].to(device, dtype=torch.float32, non_blocking=True) * 2.0 - 1.0
            score = x @ w_cur.T
            if bias:
                score = score + b_cur
            margin = y * score
            act = margin < 1.0
            hinge += torch.clamp(1.0 - margin, min=0.0).sum(dim=0, dtype=torch.float64)
            active += act.sum(dim=0, dtype=torch.float64)
            if want_grad:
                coef = -(y * act)
                if balanced:
                    coef = coef * torch.where(y > 0, w_pos, w_neg)
                g.addmm_(coef.T, x)
                if bias:
                    gb += coef.sum(dim=0)
        return g, gb, hinge / n, active / n

    for r in range(1, rounds + 1):
        g, gb, hinge, _ = one_pass(w, b, want_grad=True)
        w = (w - lr * (g / n + l2 * w)) * mask
        if bias:
            b = (b - lr * (gb / n)) * mask.squeeze(1)
        if log_every and (r == 1 or r == rounds or r % log_every == 0):
            # hinge here is at the PRE-update w of round r (one pass per round).
            t = trainable.to(device)
            reg = l2 * (w * w).sum(dim=1)
            log(f"[FedCBM][Stage A] round {r}/{rounds}  mean hinge (pre-step) "
                f"{hinge[t].mean().item():.5f}  mean ||w||^2*l2 {reg[t].mean().item():.3e}")
            history.append({"round": r, "mean_hinge_pre_step": float(hinge[t].mean())})

    _, _, hinge, active = one_pass(w, b, want_grad=False)
    t = trainable.to(device)
    obj = hinge + l2 * (w * w).sum(dim=1).double()
    return {
        "W": w,
        "b": b if bias else None,
        "stats": stats,
        "trainable": trainable,
        "pos_counts": pos,
        "final_objective_mean": float(obj[t].mean()),
        "final_hinge_mean": float(hinge[t].mean()),
        "final_active_fraction_mean": float(active[t].mean()),
        "history": history,
    }


@torch.no_grad()
def feature_stats(feats: torch.Tensor, rows: Optional[torch.Tensor] = None,
                  chunk: int = 65536) -> Dict[str, torch.Tensor]:
    """Mean and std over the clients' rows, from summed per-row statistics
    (what clients would send as sum / sum of squares). Same formula as our
    Phase-2 normalization: std = sqrt(max(E[x^2] - E[x]^2, 1e-8))."""
    n = feats.shape[0] if rows is None else rows.numel()
    s1 = torch.zeros(feats.shape[1], dtype=torch.float64)
    s2 = torch.zeros(feats.shape[1], dtype=torch.float64)
    for s, e in _rows(n, chunk):
        x = (feats[s:e] if rows is None else feats[rows[s:e]]).double().cpu()
        s1 += x.sum(dim=0)
        s2 += (x * x).sum(dim=0)
    mean = s1 / n
    std = (s2 / n - mean * mean).clamp(min=1e-8).sqrt()
    return {"mean": mean.float(), "std": std.float()}


@torch.no_grad()
def project(feats: torch.Tensor, w: torch.Tensor, rows: Optional[torch.Tensor] = None,
            chunk: int = 32768, out_device: Optional[str] = None,
            b: Optional[torch.Tensor] = None,
            stats: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    """z = (h W^T + b) / ||w||^2 per concept; a zero-norm row divides by 1
    (reference). b and stats are only set by the stronger variant; with them
    unset this is exactly the reference projection. The sign of z is the
    concept prediction; a per-concept offset in z is absorbed by the heads'
    class intercepts, so it does not change what a head can fit."""
    device = w.device
    out_device = out_device or str(device)
    norms = (w * w).sum(dim=1)
    denom = torch.where(norms > 1e-12, norms, torch.ones_like(norms))
    mu = sd = None
    if stats is not None:
        mu, sd = stats["mean"].to(device), stats["std"].to(device)
    idx = rows if rows is not None else None
    n = feats.shape[0] if idx is None else idx.shape[0]
    out = torch.empty(n, w.shape[0], dtype=torch.float32, device=out_device)
    for s, e in _rows(n, chunk):
        x = (feats[s:e] if idx is None else feats[idx[s:e]]).to(device).float()
        if mu is not None:
            x = (x - mu) / sd
        score = x @ w.T
        if b is not None:
            score = score + b
        out[s:e] = (score / denom).to(out_device)
    return out


def fit_head(z: torch.Tensor, y: torch.Tensor, C: float = 1.0, max_iter: int = 200,
             tol: float = 1e-4, dtype: torch.dtype = torch.float32) -> Head:
    """sklearn LogisticRegression(lbfgs, C) objective, fit with torch L-BFGS.

    sklearn (>=1.1) minimizes mean CE + ||W||^2 / (2 C n) over the classes in y,
    intercept unpenalized, stopping at max |grad| <= tol or max_iter iterations
    (scipy L-BFGS-B, history 10). Classes absent from y get no output, as in
    sklearn, so predictions are restricted to the client's own classes.
    """
    classes = torch.unique(y.cpu())
    if classes.numel() < 2:
        return Head(weight=torch.zeros(0), bias=torch.zeros(0), classes=classes,
                    constant=int(classes[0]))
    dev = z.device
    zz = z.to(dtype)
    target = torch.searchsorted(classes.to(dev), y.to(dev))
    n, f = zz.shape
    k = classes.numel()
    W = torch.zeros(k, f, dtype=dtype, device=dev, requires_grad=True)
    b = torch.zeros(k, dtype=dtype, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, max_eval=int(max_iter * 1.25),
                            tolerance_grad=tol, tolerance_change=1e-12, history_size=10,
                            line_search_fn="strong_wolfe")
    reg = 1.0 / (2.0 * C * n)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(zz @ W.T + b, target) + reg * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return Head(weight=W.detach(), bias=b.detach(), classes=classes)


@torch.no_grad()
def accuracy(head: Head, z: torch.Tensor, y: torch.Tensor, chunk: int = 65536) -> float:
    correct = 0
    for s, e in _rows(z.shape[0], chunk):
        correct += int((head.predict(z[s:e]) == y[s:e].to(z.device)).sum())
    return correct / max(z.shape[0], 1)


def personalized_test_split(test_labels: torch.Tensor, client_train_labels: Sequence[torch.Tensor],
                            num_classes: int, seed: int) -> List[torch.Tensor]:
    """Per-client test sets following each client's training class mix.

    For every class k, its test images are shuffled and divided among clients
    in proportion to how many class-k training images each client holds
    (largest-remainder rounding). A class no client holds is split evenly.
    The pieces partition the test set, so size-weighted mean accuracy over
    clients is the accuracy of the personalized system on the full test set.
    """
    m = len(client_train_labels)
    counts = np.zeros((m, num_classes), dtype=np.int64)
    for i, lab in enumerate(client_train_labels):
        counts[i] = np.bincount(lab.cpu().numpy(), minlength=num_classes)
    g = torch.Generator().manual_seed(seed)
    labels = test_labels.cpu()
    parts: List[List[torch.Tensor]] = [[] for _ in range(m)]
    for k in range(num_classes):
        idx = (labels == k).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        col = counts[:, k].astype(np.float64)
        share = col / col.sum() if col.sum() > 0 else np.full(m, 1.0 / m)
        raw = share * idx.numel()
        alloc = np.floor(raw).astype(np.int64)
        for j in np.argsort(-(raw - alloc), kind="stable")[: idx.numel() - alloc.sum()]:
            alloc[j] += 1
        start = 0
        for i in range(m):
            parts[i].append(idx[start:start + alloc[i]])
            start += alloc[i]
    return [torch.cat(p) if p else torch.zeros(0, dtype=torch.long) for p in parts]


def client_stats(client_train_labels: Sequence[torch.Tensor], split: Sequence[torch.Tensor],
                 num_classes: int) -> List[Dict[str, float]]:
    """Each client's data conditions: training size, classes held, effective number
    of classes exp(H(p_m)), total-variation distance of its label mix from the
    federation's, the largest class's share, and its per-client test split size."""
    counts = torch.stack([torch.bincount(l.cpu(), minlength=num_classes).double() for l in client_train_labels])
    glob = counts.sum(dim=0) / counts.sum()
    out = []
    for c, ix in zip(counts, split):
        p = c / c.sum().clamp(min=1)
        nz = p[p > 0]
        out.append({"n_train": int(c.sum()), "n_test": int(ix.numel()), "classes_held": int((c > 0).sum()),
                    "eff_classes": float(torch.exp(-(nz * nz.log()).sum())),
                    "tv_from_global": float(0.5 * (p - glob).abs().sum()), "top_class_share": float(p.max())})
    return out


@torch.no_grad()
def concept_accuracy(z: torch.Tensor, concepts: torch.Tensor, trainable: torch.Tensor,
                     chunk: int = 32768) -> Dict[str, float]:
    """Reference metric: predict present iff projected score >= 0, per concept,
    averaged over trainable concepts. Also balanced accuracy, since concepts are
    rare and 'always absent' already scores high on plain accuracy."""
    c = z.shape[1]
    tp = torch.zeros(c, dtype=torch.float64)
    tn = torch.zeros(c, dtype=torch.float64)
    p = torch.zeros(c, dtype=torch.float64)
    n_all = z.shape[0]
    for s, e in _rows(n_all, chunk):
        pred = (z[s:e] >= 0).cpu()
        gt = concepts[s:e].cpu().bool()
        tp += (pred & gt).sum(dim=0, dtype=torch.float64)
        tn += (~pred & ~gt).sum(dim=0, dtype=torch.float64)
        p += gt.sum(dim=0, dtype=torch.float64)
    neg = n_all - p
    acc = (tp + tn) / n_all
    tpr = torch.where(p > 0, tp / p.clamp(min=1), torch.full_like(tp, float("nan")))
    tnr = torch.where(neg > 0, tn / neg.clamp(min=1), torch.full_like(tn, float("nan")))
    bal = (tpr + tnr) / 2
    t = trainable.cpu()
    return {"concept_accuracy_mean": float(acc[t].mean()),
            "concept_balanced_accuracy_mean": float(torch.nanmean(bal[t]))}


# ── NEC-constrained heads ──────────────────────────────────────────────────────
@torch.no_grad()
def fit_sparse_path(z: torch.Tensor, y: torch.Tensor, targets: Sequence[float], alpha: float = 0.99,
                    n_lambda: int = 40, min_ratio: float = 1e-3, iters: int = 100,
                    tol: float = 1e-6, refit_iters: int = 0) -> Dict[float, Dict[str, object]]:
    """One client's sparse head at each NEC target, from an elastic-net path.

    Objective (glm_saga's, as in VLG-CBM's sparse layer):
        mean CE + lam * ((1 - alpha)/2 ||W||^2 + alpha ||W||_1), intercept free,
    over the classes this client holds, on its projected scores standardized
    with its OWN mean/std (the analogue of the normalized concept activations
    our sparse layer sees). Solved by FISTA with step 1/L, warm-started down a
    geometric lam path from lam_max (all-zero W). NEC = nonzeros / classes held.
    For each target, the path point whose NEC is nearest is returned, mapped
    back to raw projected scores (the zero pattern is unchanged).

    refit_iters > 0 also returns "head_refit": the same support refit without
    the penalty (FISTA on mean CE, zero pattern fixed, from the path point),
    as our head is refit on FedDualAvg's support. At low NEC the L1 shrinkage,
    not the support, is what costs accuracy (C100 NEC5 val 0.43-0.48 -> 0.61-0.64).
    "empty_classes" counts held classes whose support is empty.
    """
    classes = torch.unique(y.cpu())
    dev = z.device
    k = classes.numel()
    if k < 2:
        const = Head(torch.zeros(0), torch.zeros(0), classes, constant=int(classes[0]))
        return {t: {"head": const, "head_refit": const, "nec": 0.0, "lam": float("nan"), "empty_classes": 0}
                for t in targets}
    tgt = torch.searchsorted(classes.to(dev), y.to(dev))
    n, f = z.shape
    mu, sd = z.mean(dim=0), z.std(dim=0).clamp(min=1e-6)
    zs = (z - mu) / sd
    Y = F.one_hot(tgt, k).float()
    # Lipschitz constant of the mean-CE gradient: 0.5 * sigma_max(zs)^2 / n.
    v = torch.randn(f, device=dev)
    for _ in range(30):
        v = zs.T @ (zs @ v)
        v = v / v.norm()
    s2 = float((zs @ v).norm() ** 2) / n
    prior = Y.mean(dim=0).clamp(min=1e-12)
    W = torch.zeros(k, f, device=dev)
    b = prior.log() - prior.log().mean()
    G0 = (torch.softmax(zs @ W.T + b, dim=1) - Y).T @ zs / n
    lam_max = float(G0.abs().max()) / alpha
    lams = [lam_max * r for r in torch.logspace(0, float(np.log10(min_ratio)), n_lambda).tolist()]
    path = []
    for lam in lams:
        L = 0.5 * s2 + lam * (1 - alpha)
        Wy, by, t_m = W.clone(), b.clone(), 1.0
        for _ in range(iters):
            P = torch.softmax(zs @ Wy.T + by, dim=1) - Y
            gW = P.T @ zs / n + lam * (1 - alpha) * Wy
            gb = P.mean(dim=0)
            W_new = Wy - gW / L
            W_new = torch.sign(W_new) * torch.clamp(W_new.abs() - lam * alpha / L, min=0.0)
            b_new = by - gb / L
            t_n = (1 + (1 + 4 * t_m * t_m) ** 0.5) / 2
            Wy = W_new + ((t_m - 1) / t_n) * (W_new - W)
            by = b_new + ((t_m - 1) / t_n) * (b_new - b)
            delta = float((W_new - W).abs().max())
            W, b, t_m = W_new, b_new, t_n
            if delta < tol:
                break
        nec = float((W != 0).sum()) / k
        path.append((lam, nec, W.clone(), b.clone()))
        if nec > 1.5 * max(targets):
            break
    out = {}
    for t in targets:
        lam, nec, Wt, bt = min(path, key=lambda p: abs(p[1] - t))
        fits = {"head": (Wt, bt)}
        if refit_iters:
            fits["head_refit"] = _refit_support(zs, Y, Wt, bt, 0.5 * s2, refit_iters, tol)
        out[t] = {"nec": nec, "lam": lam, "empty_classes": int(((Wt != 0).sum(dim=1) == 0).sum())}
        for key, (Wk, bk) in fits.items():
            Wr = Wk / sd
            out[t][key] = Head(weight=Wr, bias=bk - Wr @ mu, classes=classes)
    return out


@torch.no_grad()
def _refit_support(zs: torch.Tensor, Y: torch.Tensor, W: torch.Tensor, b: torch.Tensor,
                   L: float, iters: int, tol: float):
    """Mean CE over W's zero pattern, no penalty: FISTA from (W, b), step 1/L."""
    M = (W != 0).float()
    n = zs.shape[0]
    Wy, by, t_m = W.clone(), b.clone(), 1.0
    for _ in range(iters):
        P = torch.softmax(zs @ Wy.T + by, dim=1) - Y
        W_new = Wy - (P.T @ zs / n) * M / L
        b_new = by - P.mean(dim=0) / L
        t_n = (1 + (1 + 4 * t_m * t_m) ** 0.5) / 2
        Wy = W_new + ((t_m - 1) / t_n) * (W_new - W)
        by = b_new + ((t_m - 1) / t_n) * (b_new - b)
        delta = float((W_new - W).abs().max())
        W, b, t_m = W_new, b_new, t_n
        if delta < tol:
            break
    return W, b


def support_stats(heads: Sequence[Head], num_classes: int, num_concepts: int) -> Dict[str, float]:
    """What an auditor of the federation reads, per class: the UNION of the
    clients' supports (distinct concepts used for that class anywhere) and the
    mean pairwise Jaccard overlap of the supports of clients that hold it."""
    masks, holds = [], []
    for h in heads:
        m = torch.zeros(num_classes, num_concepts, dtype=torch.bool)
        held = torch.zeros(num_classes, dtype=torch.bool)
        if h.constant is None:
            m[h.classes] = (h.weight != 0).cpu()
            held[h.classes] = True
        masks.append(m)
        holds.append(held)
    union = torch.zeros(num_classes, num_concepts, dtype=torch.bool)
    for m in masks:
        union |= m
    any_held = torch.stack(holds).any(dim=0)
    jac = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            both = holds[i] & holds[j]
            inter = (masks[i] & masks[j]).sum(dim=1).float()
            uni = (masks[i] | masks[j]).sum(dim=1).float()
            ok = both & (uni > 0)
            if ok.any():
                jac.append((inter[ok] / uni[ok]))
    return {"union_nec": float(union.sum(dim=1)[any_held].float().mean()),
            "support_jaccard": float(torch.cat(jac).mean()) if jac else float("nan")}
