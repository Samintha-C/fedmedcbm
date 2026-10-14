"""Our head, improved: FedDualAvg -> per-class budget -> masked-FedAvg refit, plus
per-client logit adjustment. Post hoc on finished main_fed.py runs (no Phase 1 rerun).

For each run (one per partition, as ours_personal.py) the concept layer is loaded and
checked by ours_personal.py's checks (args, row order, recomputed features, global head
reproduces metrics.txt). Then, per NEC target:
  fda        FedDualAvg replicated from train_vlg.py (200 rounds x 3 local epochs, batch
             512, eta_c 0.01, eta_s 1, linear threshold schedule, best round on val),
             lam bracketed at 0.6/1/1.6 x the given value, nearest achieved NEC kept;
  perclass   each class keeps its top-k concepts by |dual| (k = target), then masked
             FedAvg (20 rounds x 10 local full-batch GD steps, no L1): NEC is exactly k;
  logit-adj  per client, b_m = b + log p_m(y) - log p(y) (training label counts, +1
             smoothing); weights untouched, so every client shares one explanation.
Scored on the global test set and on per-client test splits (core.personalized_test_split,
same seed, as ours_personal.py / nec.py).

Usage: run.py's data flags, plus --runs iid=<dir> dirichlet=<dir> and
       --lam_targets 5:<lam> 50:<lam> (the tab:main lambdas for that partition)
"""
import datetime
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import core  # noqa: E402
import ours_personal as op  # noqa: E402
import run  # noqa: E402


def soft(z, t):
    return torch.sign(z) * torch.clamp(z.abs() - t, min=0.0)


def feddualavg(z, y, zv, yv, nk, lam, clients, rounds=200, epochs=3, bs=512, eta_c=0.01, eta_s=1.0):
    dev = z.device
    g = torch.Generator(device="cpu").manual_seed(0)
    n = float(sum(len(c) for c in clients))
    wts = [len(c) / n for c in clients]
    zw, zb = torch.zeros(nk, z.shape[1], device=dev), torch.zeros(nk, device=dev)
    K = epochs * -(-len(clients[0]) // bs)
    best, best_acc, best_nnz = None, -1.0, None
    for r in range(rounds):
        dw, db = torch.zeros_like(zw), torch.zeros_like(zb)
        for ci, ix in enumerate(clients):
            lw, lb, step = zw.clone(), zb.clone(), 0
            for _ in range(epochs):
                perm = ix[torch.randperm(len(ix), generator=g).to(dev)]
                for s in range(0, len(perm), bs):
                    b_ix = perm[s:s + bs]
                    w = soft(lw, eta_s * eta_c * (r * K + step + 1) * lam)
                    zb_ = z[b_ix].float()
                    P = torch.softmax(zb_ @ w.T + lb, 1)
                    P[torch.arange(len(b_ix), device=dev), y[b_ix]] -= 1.0
                    lw -= eta_c * (P.T @ zb_ / len(b_ix))
                    lb -= eta_c * P.mean(0)
                    step += 1
            dw += wts[ci] * (lw - zw)
            db += wts[ci] * (lb - zb)
        zw += eta_s * dw
        zb += eta_s * db
        w_s = soft(zw, eta_s * eta_c * (r + 1) * K * lam)
        acc = core.accuracy(core.Head(w_s, zb, torch.arange(nk)), zv, yv)
        nnz = int((w_s.abs() > 1e-5).sum())
        if acc > best_acc or (acc == best_acc and best_nnz is not None and nnz < best_nnz):
            best, best_acc, best_nnz = (w_s.clone(), zb.clone(), zw.clone()), acc, nnz
    return best


def lipschitz(z, rows=65536):
    zz = z[torch.randperm(z.shape[0], device=z.device)[:rows]].float()
    v = torch.randn(zz.shape[1], device=z.device)
    for _ in range(30):
        v = zz.T @ (zz @ v)
        v = v / v.norm()
    return 0.5 * float((zz @ v).norm() ** 2) / zz.shape[0]


def refit(W, b, M, z, y, nk, clients, rounds=20, local=10):
    lr = 1.0 / lipschitz(z)
    n = float(sum(len(c) for c in clients))
    W = W * M
    for _ in range(rounds):
        Wn, bn = torch.zeros_like(W), torch.zeros_like(b)
        for ix in clients:
            zc, Y = z[ix].float(), F.one_hot(y[ix], nk).float()
            Wm, bm = W.clone(), b.clone()
            for _ in range(local):
                P = torch.softmax(zc @ Wm.T + bm, 1) - Y
                Wm = Wm - lr * (P.T @ zc / len(ix)) * M
                bm = bm - lr * P.mean(0)
            Wn += Wm * len(ix) / n
            bn += bm * len(ix) / n
            del zc, Y
        W, b = Wn, bn
    return W, b


def main():
    ap = run.build_parser()
    ap.add_argument("--runs", nargs="+", required=True, metavar="PARTITION=DIR")
    ap.add_argument("--lam_targets", nargs="+", required=True, metavar="NEC:LAM",
                    help="per partition: iid.5:0.022 iid.50:0.0013 dirichlet.5:... (or NEC:LAM for both)")
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
    save_dir = os.path.join(args.save_dir, f"ours-head-{args.dataset}-s{args.seed}-{tag}")
    os.makedirs(save_dir, exist_ok=True)
    t0 = time.time()
    loaded = {}
    for p in args.partitions:
        r = op.load_run(op.resolve_run(runs[p]), device)
        op.check_args(r, args, p)
        loaded[p] = r
    d = run.load_data(args, device)
    tr, va, te = d["tr"], d["va"], d["te"]
    nk = len(d["classes"])
    out = {"dataset": args.dataset, "seed": args.seed, "method": "ours head: fda / per-class+refit / logit-adj",
           "lam_targets": lam_t, "partitions": {}}
    for p in args.partitions:
        r = loaded[p]
        ci = run.split_dataset_for_federated(d["train_ds"], args.num_clients, iid=(p == "iid"),
                                             alpha=args.alpha, seed=args.seed)
        rec = {"run": r["path"], "checks": op.check_rows(r, d, ci, args), "targets": {}}
        held = torch.cat([torch.as_tensor(c, dtype=torch.long) for c in ci])
        # rows in client-block order, fp16 (Places365 / ImageNet would need ~17 / 22 GB in fp32)
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
        labels = [tr._target_cache[torch.as_tensor(c, dtype=torch.long)] for c in ci]
        split = core.personalized_test_split(te._target_cache, labels, nk, seed=args.seed)
        prior = (torch.bincount(y, minlength=nk).float() + 1) / (len(y) + nk)

        def gacc(W, b):
            return 100 * core.accuracy(core.Head(W, b, torch.arange(nk)), zt, yt)

        def pacc(heads):
            corr = tot = 0
            for (W, b), ix in zip(heads, split):
                if ix.numel():
                    ix = ix.to(device)
                    corr += int(((zt[ix] @ W.T + b).argmax(1) == yt[ix]).sum())
                    tot += ix.numel()
            return 100 * corr / max(tot, 1)

        for target, base in sorted(lam_t[p].items()):
            cands = []
            for m in (0.6, 1.0, 1.6):
                W, b, zw = feddualavg(z, y, zv, yv, nk, base * m, clients)
                nec = float((W.abs() > 1e-5).sum()) / nk
                cands.append((abs(nec - target), m, nec, W, b, zw))
            _, m, nec, W, b, zw = min(cands, key=lambda c: c[0])
            M = torch.zeros_like(zw)
            M.scatter_(1, zw.abs().topk(min(target, zw.shape[1]), dim=1).indices, 1.0)
            Wr, br = refit(W, b, M, z, y, nk, clients)
            la = []
            for l in labels:
                pm = (torch.bincount(l.to(device), minlength=nk).float() + 1) / (len(l) + nk)
                la.append((Wr, br + pm.log() - prior.log()))
            res = {"lam": base * m, "fda": {"nec": nec, "global_test": gacc(W, b)},
                   "perclass_refit": {"nec": float((Wr * M != 0).sum()) / nk, "global_test": gacc(Wr, br),
                                      "per_client": pacc([(Wr, br)] * len(ci))},
                   "logit_adjust": {"per_client": pacc(la),
                                    "global_test_mean": sum(gacc(*h) for h in la) / len(la)},
                   "union_nec": float((Wr * M != 0).sum()) / nk}
            rec["targets"][str(target)] = res
            print(f"[{p}] NEC~{target}: fda {res['fda']['global_test']:.2f}@{nec:.1f} (lam x{m}) | "
                  f"per-class+refit {res['perclass_refit']['global_test']:.2f} (per-client "
                  f"{res['perclass_refit']['per_client']:.2f}) | logit-adj per-client "
                  f"{res['logit_adjust']['per_client']:.2f}", flush=True)
        out["partitions"][p] = rec
        del z
        torch.cuda.empty_cache()
    out["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {save_dir}")


if __name__ == "__main__":
    main()
