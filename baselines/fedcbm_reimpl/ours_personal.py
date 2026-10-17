"""Ours, personalized: our federated concept layer, then a private head per
client, fitted with the SAME solver and protocol as NEC-constrained FedCBM
(nec.py).

The only difference from FedCBM-NEC is the concept layer: here it is the CBL
our pipeline trained federatedly (Phase 1) plus its Phase-2 normalization,
loaded from a finished main_fed.py run; there it is FedCBM's SVM bank. The
partition, the per-client heads (dense at every C in --head_C_grid, sparse
from core.fit_sparse_path at every NEC target), the per-client test splits and
the scoring are the same calls with the same seed, so this metrics.json lines
up with FEDCBM_NEC's cell by cell.

Phase 1 trains under the partition, so each partition has its own run:
  --runs iid=<dir> dirichlet=<dir>
where <dir> holds cbl.pt or has exactly one child that does, e.g.
models/SEED_MAIN/<ds>/iid_s7_nec50 or models/<ds>/dlamsweep_iid/lam0.0013.

Each run is checked against this process before anything is fitted; any
failure exits non-zero:
  1. its args.txt matches our dataset, seed, split, clients, backbone and
     partition (and is not a finetuned-backbone or poisoned run);
  2. row order: its saved train labels equal ours in the order train_vlg.py
     wrote them (client 0's rows in partition order, then client 1, ...), and
     its saved val labels equal ours;
  3. concept features recomputed here from cbl.pt and the saved normalization
     match its saved train_concept_features.pt on sampled rows;
  4. its own global head (final.pt) on our test features reproduces the test
     accuracy in its metrics.txt.

Usage: run.py's data flags, plus --runs as above.
"""

import datetime
import glob
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import core  # noqa: E402
import nec  # noqa: E402
import run  # noqa: E402


def resolve_run(path):
    if os.path.isfile(os.path.join(path, "cbl.pt")):
        return path
    kids = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(path, "*", "cbl.pt")))
    if len(kids) != 1:
        sys.exit(f"ERROR: expected exactly one run with cbl.pt under {path}, found {len(kids)}")
    return kids[0]


def load_run(path, device):
    sd = torch.load(os.path.join(path, "cbl.pt"), map_location="cpu")
    if set(sd) != {"model.0.weight", "model.0.bias"}:
        sys.exit(f"ERROR: {path}: only a single-linear CBL (cbl_hidden_layers 0) is supported, got {sorted(sd)}")
    final = torch.load(os.path.join(path, "final.pt"), map_location="cpu")
    with open(os.path.join(path, "args.txt")) as f:
        ra = json.load(f)
    with open(os.path.join(path, "metrics.txt")) as f:
        rm = json.load(f)
    with open(os.path.join(path, "concepts.txt")) as f:
        concepts = f.read().split("\n")
    return {"path": path, "args": ra, "metrics": rm, "concepts": concepts,
            "W": sd["model.0.weight"].float().to(device), "b": sd["model.0.bias"].float().to(device),
            "mean": torch.load(os.path.join(path, "train_concept_features_mean.pt"), map_location="cpu").float().to(device),
            "std": torch.load(os.path.join(path, "train_concept_features_std.pt"), map_location="cpu").float().to(device),
            "final_W": final["weight"].float(), "final_b": final["bias"].float()}


@torch.no_grad()
def concept_scores(feats, r, rows=None, chunk=32768):
    """Normalized concept activations, as train_vlg.py Phase 2 computes them:
    (cbl(h) - mean) / std."""
    n = feats.shape[0] if rows is None else rows.numel()
    out = torch.empty(n, r["W"].shape[0], dtype=torch.float32, device=r["W"].device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        x = (feats[s:e] if rows is None else feats[rows[s:e]]).to(r["W"].device).float()
        out[s:e] = (x @ r["W"].T + r["b"] - r["mean"]) / r["std"]
    return out


def check_args(r, args, p, alpha=None):
    ra = dict({"label_flip_client": -1}, **r["args"])  # runs from before --label_flip_client had no flip
    want = {"dataset": args.dataset, "seed": args.seed, "val_split": args.val_split,
            "num_clients": args.num_clients, "backbone": args.backbone, "iid": p == "iid",
            "cbl_hidden_layers": 0, "cbl_finetune": False, "label_flip_client": -1}
    if p != "iid":
        want["alpha"] = args.alpha if alpha is None else alpha
    if args.backbone.startswith("clip_"):
        want["use_clip_penultimate"] = args.use_clip_penultimate
    else:
        want["feature_layer"] = args.feature_layer
    bad = {k: {"run": ra.get(k), "here": v} for k, v in want.items() if ra.get(k) != v}
    if bad:
        sys.exit(f"ERROR: {r['path']} does not match this {p} run: {bad}")


def load_saved(path, mmap):
    if mmap:
        try:  # avoids reading Places365/ImageNet's multi-GB file to check a few rows
            return torch.load(path, map_location="cpu", mmap=True)
        except (RuntimeError, TypeError):
            pass
    return torch.load(path, map_location="cpu")


def check_rows(r, d, ci, args):
    """Checks 2 and 3: train_vlg.py saved Phase-2 rows client by client, each
    client's block in its partition order, so saved row i is train row held[i]."""
    tr, va = d["tr"], d["va"]
    held = torch.cat([torch.as_tensor(c, dtype=torch.long) for c in ci])
    y_saved = load_saved(os.path.join(r["path"], "train_concept_labels.pt"), mmap=False)
    out = {"saved_train_rows": int(y_saved.shape[0]), "held_rows": int(held.numel())}
    out["train_labels_equal"] = (y_saved.shape[0] == held.numel()
                                 and torch.equal(y_saved.long(), tr._target_cache[held].long()))
    yv_saved = load_saved(os.path.join(r["path"], "val_concept_labels.pt"), mmap=False)
    out["val_labels_equal"] = torch.equal(yv_saved.long(), va._target_cache.long())
    if not (out["train_labels_equal"] and out["val_labels_equal"]):
        sys.exit(f"ERROR: {r['path']}: saved labels are not in this process's client order: {out}")

    saved = load_saved(os.path.join(r["path"], "train_concept_features.pt"), mmap=True)
    if tuple(saved.shape) != (held.numel(), r["W"].shape[0]):
        sys.exit(f"ERROR: {r['path']}: saved features {tuple(saved.shape)}, expected {(held.numel(), r['W'].shape[0])}")
    pos = torch.randperm(held.numel(), generator=torch.Generator().manual_seed(0))[:args.check_rows]
    pos = torch.sort(pos).values
    ours = concept_scores(tr._feature_cache, r, rows=held[pos]).cpu()
    ref = saved[pos].float()
    rel = (ours - ref).abs() / (1.0 + ref.abs())
    out["feature_rows_checked"] = int(pos.numel())
    out["feature_max_rel_err"] = float(rel.max())
    out["feature_mean_abs_err"] = float((ours - ref).abs().mean())
    del saved
    if out["feature_max_rel_err"] > args.check_tol:
        sys.exit(f"ERROR: {r['path']}: recomputed concept features differ from the saved ones: {out}")
    return out


def main():
    ap = run.build_parser()
    ap.add_argument("--runs", nargs="+", required=True, metavar="PARTITION=DIR")
    ap.add_argument("--nec_targets", type=float, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--head_C_grid", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--path_lambdas", type=int, default=40)
    ap.add_argument("--path_iters", type=int, default=100)
    ap.add_argument("--check_rows", type=int, default=4096)
    ap.add_argument("--check_tol", type=float, default=1e-2,
                    help="max |ours - saved| / (1 + |saved|) over checked feature entries (saved is fp16)")
    ap.add_argument("--acc_tol", type=float, default=0.005,
                    help="max |global-head test acc here - metrics.txt| (check 4)")
    args = ap.parse_args()
    runs = dict(s.split("=", 1) for s in args.runs)
    missing = [p for p in args.partitions if p not in runs]
    if missing:
        sys.exit(f"ERROR: no --runs entry for partition(s) {missing}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    tag = datetime.datetime.now().strftime("%b%d-%H:%M").lower()
    save_dir = os.path.join(args.save_dir, f"ours-personal-{args.dataset}-s{args.seed}-{tag}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    t0 = time.time()

    # Every check reads the run before the slow per-client fits start.
    loaded = {}
    for p in args.partitions:
        r = load_run(resolve_run(runs[p]), device)
        check_args(r, args, p)
        loaded[p] = r
        print(f"[run] {p}: {r['path']}  ({r['W'].shape[0]} concepts)", flush=True)

    d = run.load_data(args, device)
    tr, va, te = d["tr"], d["va"], d["te"]
    nk = len(d["classes"])
    y_val, y_test = va._target_cache.to(device), te._target_cache.to(device)
    out = {"dataset": args.dataset, "seed": args.seed, "num_clients": args.num_clients,
           "method": "ours, personalized: federated CBL + Phase-2 normalization from the run, "
                     "private per-client heads (same solver as nec.py)",
           "num_classes": nk, "nec_targets": args.nec_targets, "head_C_grid": args.head_C_grid,
           "runs": {}, "dense": {}, "sparse": {}}

    for p in args.partitions:
        r = loaded[p]
        nc = r["W"].shape[0]
        ci = run.split_dataset_for_federated(d["train_ds"], args.num_clients, iid=(p == "iid"),
                                             alpha=args.alpha, seed=args.seed)
        rec = {"path": r["path"], "num_concepts": nc, "checks": check_rows(r, d, ci, args)}
        print(f"[check] {p}: {rec['checks']}", flush=True)
        z_val, z_test = concept_scores(va._feature_cache, r), concept_scores(te._feature_cache, r)

        # Check 4, and the reference row: the run's own GLOBAL head (FedDualAvg).
        gh = core.Head(weight=r["final_W"], bias=r["final_b"], classes=torch.arange(nk))
        acc, ref_acc = core.accuracy(gh, z_test, y_test), float(r["metrics"]["metrics"]["test_accuracy"])
        rec["global_head"] = {"test_acc": acc, "metrics_txt_test_acc": ref_acc,
                              "val_acc": core.accuracy(gh, z_val, y_val),
                              "nec": float((r["final_W"].abs() > 1e-5).sum()) / nk}
        print(f"[check] {p}: global head test acc {acc:.4f} here vs {ref_acc:.4f} in metrics.txt "
              f"(NEC {rec['global_head']['nec']:.1f})", flush=True)
        if abs(acc - ref_acc) > args.acc_tol:
            sys.exit(f"ERROR: {r['path']}: global head does not reproduce its test accuracy: {rec['global_head']}")

        dense = {C: [] for C in args.head_C_grid}
        sparse = {t: [] for t in args.nec_targets}
        nec_ach = {t: [] for t in args.nec_targets}
        labels = []
        for m in range(args.num_clients):
            idx = torch.as_tensor(ci[m], dtype=torch.long)
            y = tr._target_cache[idx]
            labels.append(y)
            z = concept_scores(tr._feature_cache, r, rows=idx)
            th = time.time()
            for C in args.head_C_grid:
                dense[C].append(core.fit_head(z, y, C=C, max_iter=max(50, args.head_max_iter), tol=args.head_tol))
            td = time.time() - th
            res = core.fit_sparse_path(z, y, args.nec_targets, n_lambda=args.path_lambdas, iters=args.path_iters)
            for t in args.nec_targets:
                sparse[t].append(res[t]["head"])
                nec_ach[t].append(res[t]["nec"])
            print(f"[{p}] client {m}: dense x{len(args.head_C_grid)} {td:.0f}s, path {time.time() - th - td:.0f}s, "
                  f"NEC achieved { {t: round(res[t]['nec'], 2) for t in args.nec_targets} }", flush=True)
            del z
            torch.cuda.empty_cache()
        split = core.personalized_test_split(te._target_cache, labels, nk, seed=args.seed)
        rec["global_head"]["per_client_split_acc"] = [
            core.accuracy(gh, z_test[ix.to(device)], y_test[ix.to(device)]) if ix.numel() else float("nan")
            for ix in split]
        for C in args.head_C_grid:
            out["dense"].setdefault(str(C), {})[p] = nec.score(dense[C], z_val, y_val, z_test, y_test, split)
        for t in args.nec_targets:
            s = nec.score(sparse[t], z_val, y_val, z_test, y_test, split)
            s["nec_mean_over_clients"] = sum(nec_ach[t]) / len(nec_ach[t])
            s["nec_per_client"] = nec_ach[t]
            s.update(core.support_stats(sparse[t], nk, nc))
            out["sparse"].setdefault(str(t), {})[p] = s
            print(f"[{p}] NEC {t:g}: achieved {s['nec_mean_over_clients']:.2f}, union {s['union_nec']:.1f}, "
                  f"jaccard {s['support_jaccard']:.3f} | global {s['global_test_mean_client_acc']:.4f} "
                  f"per-client {s['personalized_test_acc_weighted']:.4f}", flush=True)
        for C in args.head_C_grid:
            s = out["dense"][str(C)][p]
            print(f"[{p}] dense C={C:g}: global {s['global_test_mean_client_acc']:.4f} "
                  f"per-client {s['personalized_test_acc_weighted']:.4f} (val {s['global_val_mean_client_acc']:.4f})")
        out["runs"][p] = rec
        del z_val, z_test
        torch.cuda.empty_cache()

    sel = "iid" if "iid" in args.partitions else args.partitions[0]
    out["dense_selected_C"] = max(args.head_C_grid, key=lambda C: out["dense"][str(C)][sel]["global_val_mean_client_acc"])
    print(f"[dense] selected C={out['dense_selected_C']:g} (on {sel} val)")
    out["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {save_dir}")


if __name__ == "__main__":
    main()
