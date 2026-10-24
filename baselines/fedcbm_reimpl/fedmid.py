"""FedMiD vs FedDualAvg on the head of a finished main_fed.py run (council W2, Table 3).

FedMiD (Yuan et al. 2021, the primal counterpart of FedDualAvg): every local step is a
proximal SGD step, soft(w - eta_c g, eta_c lam), and the server averages the clients' primal
weights. FedDualAvg is ours_head.feddualavg, the replication of train_vlg.py that reproduces the
logged runs. Both use the same steps, batches, data order and best-round-on-val rule, so the only
difference is where the l1 prox sits relative to the averaging.

Data and checks are ours_head.py's (concept layer and normalization from the run, rows in
client-block order). Per partition, lambda runs over a geometric grid from the tab:main NEC~50
lambda to the NEC~5 lambda (--grid_points) plus multiples of the NEC~5 lambda (--beyond), where
FedMiD needs larger lambda to become sparse. --fda_points ends runs FedDualAvg only at the two
tab:main lambdas (the large datasets; their other FedDualAvg operating points are in Table 6).
Recorded per run: test accuracy, achieved NEC, best round, per-class test accuracy; for FedMiD
also the clients' mean NEC, the union of their supports and the support all clients share.

Usage: run.py's data flags, plus --runs iid=<dir> dirichlet=<dir> and --lam_targets as ours_head.py.
"""
import datetime
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import core  # noqa: E402
import ours_personal as op  # noqa: E402
import run  # noqa: E402
from ours_head import feddualavg, soft  # noqa: E402


def fedmid(z, y, zv, yv, nk, lam, clients, rounds=200, epochs=3, bs=512, eta_c=0.01, eta_s=1.0):
    dev = z.device
    g = torch.Generator(device="cpu").manual_seed(0)
    n = float(sum(len(c) for c in clients))
    wts = [len(c) / n for c in clients]
    w, b = torch.zeros(nk, z.shape[1], device=dev), torch.zeros(nk, device=dev)
    best, best_acc, best_nnz, diag = None, -1.0, None, {}
    for r in range(rounds):
        dw, db, sup = torch.zeros_like(w), torch.zeros_like(b), []
        for ci, ix in enumerate(clients):
            lw, lb = w.clone(), b.clone()
            for _ in range(epochs):
                perm = ix[torch.randperm(len(ix), generator=g).to(dev)]
                for s in range(0, len(perm), bs):
                    b_ix = perm[s:s + bs]
                    zb_ = z[b_ix].float()
                    P = torch.softmax(zb_ @ lw.T + lb, 1)
                    P[torch.arange(len(b_ix), device=dev), y[b_ix]] -= 1.0
                    lw = soft(lw - eta_c * (P.T @ zb_ / len(b_ix)), eta_c * lam)
                    lb -= eta_c * P.mean(0)
            sup.append(lw.abs() > 1e-5)
            dw += wts[ci] * (lw - w)
            db += wts[ci] * (lb - b)
        w += eta_s * dw
        b += eta_s * db
        acc = core.accuracy(core.Head(w, b, torch.arange(nk)), zv, yv)
        nnz = int((w.abs() > 1e-5).sum())
        if acc > best_acc or (acc == best_acc and best_nnz is not None and nnz < best_nnz):
            s = torch.stack(sup)
            best, best_acc, best_nnz = (w.clone(), b.clone(), r + 1), acc, nnz
            diag = {"client_nec_mean": float(s.sum()) / len(sup) / nk,
                    "client_union_nec": float(s.any(0).sum()) / nk,
                    "client_shared_nec": float(s.all(0).sum()) / nk}
        del sup
    return best, diag


def main():
    ap = run.build_parser()
    ap.add_argument("--runs", nargs="+", required=True, metavar="PARTITION=DIR")
    ap.add_argument("--lam_targets", nargs="+", required=True, metavar="NEC:LAM",
                    help="as ours_head.py: iid.5:0.022 iid.50:0.0013 dirichlet.5:... dirichlet.50:...")
    ap.add_argument("--grid_points", type=int, default=5,
                    help="geometric grid from the NEC~50 lambda to the NEC~5 lambda, ends included")
    ap.add_argument("--beyond", type=float, nargs="*", default=[2.3, 4.5, 9.0],
                    help="extra lambdas as multiples of the NEC~5 lambda")
    ap.add_argument("--fda_points", choices=["all", "ends"], default="all")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--check_rows", type=int, default=4096)
    ap.add_argument("--check_tol", type=float, default=1e-2)
    ap.add_argument("--acc_tol", type=float, default=0.005)
    args = ap.parse_args()
    runs = dict(s.split("=", 1) for s in args.runs)
    lam_t = {}
    for s in args.lam_targets:
        k, lam = s.split(":")
        part, _, nec = k.rpartition(".")
        for p in ([part] if part else args.partitions):
            lam_t.setdefault(p, {})[int(nec)] = float(lam)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    tag = datetime.datetime.now().strftime("%b%d-%H:%M").lower()
    save_dir = os.path.join(args.save_dir, f"fedmid-{args.dataset}-s{args.seed}-{tag}")
    os.makedirs(save_dir, exist_ok=True)
    t0 = time.time()
    loaded = {}
    for p, _, alpha in run.arms(args):
        r = op.load_run(op.resolve_run(runs[p]), device)
        op.check_args(r, args, p, alpha)
        loaded[p] = r
    d = run.load_data(args, device)
    tr, va, te = d["tr"], d["va"], d["te"]
    nk = len(d["classes"])
    out = {"dataset": args.dataset, "seed": args.seed, "rounds": args.rounds,
           "method": "FedMiD (local prox-SGD + primal averaging) vs FedDualAvg, same steps",
           "lam_targets": lam_t, "partitions": {}}
    for p, iid, alpha in run.arms(args):
        r = loaded[p]
        ci = run.split_dataset_for_federated(d["train_ds"], args.num_clients, iid=iid,
                                             alpha=alpha, seed=args.seed)
        rec = {"run": r["path"], "checks": op.check_rows(r, d, ci, args), "runs": []}
        held = torch.cat([torch.as_tensor(c, dtype=torch.long) for c in ci])
        z = torch.empty(held.numel(), r["W"].shape[0], dtype=torch.float16, device=device)
        for s in range(0, held.numel(), 65536):
            z[s:s + 65536] = op.concept_scores(tr._feature_cache, r, rows=held[s:s + 65536]).half()
        y = tr._target_cache[held].to(device)
        clients, off = [], 0
        for c in ci:
            clients.append(torch.arange(off, off + len(c), device=device))
            off += len(c)
        zv, yv = op.concept_scores(va._feature_cache, r), va._target_cache.to(device)
        zt, yt = op.concept_scores(te._feature_cache, r), te._target_cache.to(device)
        rec["class_counts"] = [torch.bincount(tr._target_cache[torch.as_tensor(c, dtype=torch.long)],
                                              minlength=nk).tolist() for c in ci]
        lo, hi = lam_t[p][50], lam_t[p][5]
        n = args.grid_points
        grid = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)] if n > 1 else [hi]
        grid += [hi * m for m in args.beyond]
        ends = {grid[0], grid[n - 1]} if n > 1 else {hi}

        def score(W, b):
            pred = (zt @ W.T + b).argmax(1)
            hit = torch.bincount(yt[pred == yt], minlength=nk).float()
            tot = torch.bincount(yt, minlength=nk).float().clamp(min=1)
            return 100 * float((pred == yt).float().mean()), (100 * hit / tot).tolist()

        for lam in grid:
            for name in ("feddualavg", "fedmid"):
                if name == "feddualavg" and args.fda_points == "ends" and lam not in ends:
                    continue
                t1 = time.time()
                if name == "feddualavg":
                    W, b, _ = feddualavg(z, y, zv, yv, nk, lam, clients, rounds=args.rounds)
                    rnd, diag = None, {}
                else:
                    (W, b, rnd), diag = fedmid(z, y, zv, yv, nk, lam, clients, rounds=args.rounds)
                acc, per_class = score(W, b)
                one = {"method": name, "lam": lam, "best_round": rnd, "test_acc": acc,
                       "val_acc": 100 * core.accuracy(core.Head(W, b, torch.arange(nk)), zv, yv),
                       "nec": float((W.abs() > 1e-5).sum()) / nk, **diag,
                       "per_class_test_acc": per_class, "seconds": round(time.time() - t1, 1)}
                rec["runs"].append(one)
                extra = "".join(f" {k} {v:.1f}" for k, v in diag.items())
                print(f"[{p}] {name:10s} lam {lam:<10.3g} test {acc:.2f} NEC {one['nec']:8.2f}"
                      f"{'' if rnd is None else f' (round {rnd})'}{extra} [{one['seconds']}s]", flush=True)
                out["partitions"][p] = rec
                with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                    json.dump(out, f, indent=1)
        del z
        torch.cuda.empty_cache()
    out["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[done] {save_dir}")


if __name__ == "__main__":
    main()
