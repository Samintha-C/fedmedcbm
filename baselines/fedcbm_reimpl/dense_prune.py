"""Table 3's dense and post-hoc rows at matched NEC, on the same runs as the head rule.
Post hoc on finished main_fed.py runs (no Phase 1 rerun), set up exactly as ours_head.py.

For each run the concept layer is loaded and checked by ours_personal.py's checks, and the
Phase-2 concept features are rebuilt in client-block order from the run's own Dirichlet split.
Then, on those features:
  fedavg     a dense head trained as train_vlg.py's fedavg_l1 at lambda 0: FedAvg, each client
             a fresh Adam (lr 1e-3, no weight decay) per round, 200 rounds x 3 local epochs,
             batch 512, best round on val;
  fedprox    the same with train_vlg.py's FedProx term, mu/2 ||theta - theta_global||^2 over
             weight and bias (mu 0.01);
  prune      the fedavg head cut to each NEC target, then fine-tuned on that support by the same
             FedAvg loop (200 rounds x 3 epochs, best round on val) with the other weights held at
             zero, the fine-tune lr picked on val from --prune_lrs. At the dense head's own lr (1e-3)
             the fine-tune stops short of the support's ceiling (C10 s42 NEC 20: 69.2 against 78.1
             for a centralized fit on the same mask run to convergence). Local check (C10 s42 /
             C100 s31): val picks 1e-1 on C10 (NEC 5 / 20: 38.1 / 77.0) and 3e-2 on C100 (16.8 /
             49.0, ceilings 17.2 / 51.9), so the default grid brackets the best lr. Two cuts: perclass keeps
             each class's top-k |w| (NEC exactly k); global keeps the k*K largest |w| over the
             whole head (NEC k on average, as a global threshold would). The cut head untouched is
             reported too, and with --gd_refit_rounds > 0 ours_head.refit's masked GD.
  oracle     central_path: the sparse layer solved centrally on the pooled rows of the same run
             (VLG-CBM's objective, core.fit_sparse_path's solver), then refit on its support. What a
             sparse head on this concept layer reaches when the data can be pooled. Checked against
             core.fit_sparse_path on C10 / C100 (same NEC, lambda and accuracy at every target).
  thresh     FedAvg as fedavg above, with train_vlg.py's fedavg_l1 server step: the averaged head is
             soft-thresholded every round, at the level that leaves k*K nonzeros (NEC exactly k).
  fedmid     fedmid.fedmid (local prox-SGD, primal averaging) over a lambda bracket (--fedmid_lams x
             --mults), the run with the NEC nearest each target kept.
  elastic    FedDualAvg with train_vlg.py's elastic-net prox (l2 = --elastic_l2_ratio x l1), bracketed
             around the head rule's lambda, nearest NEC kept.
  central    the head rule itself (ours_head.feddualavg, then ours_head.refit 100 rounds on its own
             support) with every training row held by one client: our optimizer without federation,
             the centralized reference for Table 3 (central_path's FISTA does not converge on Places365 /
             ImageNet). With one client FedDualAvg's threshold no longer scales by client 0's step count
             over the federation's size-weighted mean (n_0 / (sum n_m^2 / N), from the run's split), so
             the head rule's lambda is multiplied by that ratio and bracketed by --central_mults.
  thresh, fedmid and elastic are each refit on their own support as the head rule is (ours_head.refit,
  --refit_rounds 100); the head before the refit is recorded too.
Scored on the global test set. The head rule at the same NEC is OURS_HEAD_OWN / OURS_HEAD_NEC1020
on the same runs.

Usage: run.py's data flags, plus --runs dirichlet=<dir> [iid=<dir>], --necs 5 10 20 and
       --parts dense prune oracle thresh fedmid elastic (prune needs dense; fedmid / elastic need
       --fedmid_lams / --elastic_lams NEC:LAM ...).
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
import fedmid as fm  # noqa: E402
import ours_head  # noqa: E402
import ours_personal as op  # noqa: E402
import run  # noqa: E402


def fedavg_dense(z, y, zv, yv, nk, clients, mu=0.0, rounds=200, epochs=3, bs=512, lr=1e-3, mask=None, init=None,
                 budget=None):
    """train_vlg.py's fedavg_l1 (lambda 0) or fedprox (mu > 0) Phase 3 on rows in client-block order.
    With mask (and init, the head it was cut from), the same loop trains only the masked weights:
    zero-gradient coordinates get zero Adam steps, so the support never changes. With budget=k,
    train_vlg.py's fedavg_l1 server step: after averaging, soft-threshold the head every round, at
    the level that leaves k*K nonzeros (so NEC is exactly k without searching a fixed threshold)."""
    dev = z.device
    g = torch.Generator(device="cpu").manual_seed(0)
    torch.manual_seed(0)
    lin = torch.nn.Linear(z.shape[1], nk)
    W, b = lin.weight.detach().to(dev), lin.bias.detach().to(dev)
    if mask is not None:
        W, b = init[0] * mask, init[1].clone()
    n = float(sum(len(c) for c in clients))
    wts = [len(c) / n for c in clients]
    best, best_acc, hist = None, -1.0, []
    for r in range(rounds):
        nW, nb = torch.zeros_like(W), torch.zeros_like(b)
        for ci, ix in enumerate(clients):
            lw, lb = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
            opt = torch.optim.Adam([lw, lb], lr=lr)
            for _ in range(epochs):
                perm = ix[torch.randperm(len(ix), generator=g).to(dev)]
                for s in range(0, len(perm), bs):
                    b_ix = perm[s:s + bs]
                    w = lw if mask is None else lw * mask
                    loss = torch.nn.functional.cross_entropy(z[b_ix].float() @ w.T + lb, y[b_ix])
                    if mu > 0:
                        loss = loss + 0.5 * mu * ((lw - W).pow(2).sum() + (lb - b).pow(2).sum())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            nW += wts[ci] * lw.detach()
            nb += wts[ci] * lb.detach()
        W, b = nW, nb
        if budget is not None:
            theta = W.abs().flatten().topk(min(budget * nk + 1, W.numel())).values[-1]
            W = ours_head.soft(W, theta)
        acc = core.accuracy(core.Head(W, b, torch.arange(nk)), zv, yv)
        hist.append(acc)
        if acc > best_acc:
            best, best_acc = (W.clone(), b.clone(), r + 1), acc
    return best[0], best[1], {"best_round": best[2], "val": best_acc, "val_by_round": hist}


def cut(W, k, mode):
    """Mask keeping each class's top-k |w| (perclass) or the k*K largest |w| overall (global)."""
    M = torch.zeros_like(W)
    if mode == "perclass":
        M.scatter_(1, W.abs().topk(min(k, W.shape[1]), dim=1).indices, 1.0)
    else:
        flat = M.view(-1)
        flat[W.abs().view(-1).topk(min(k * W.shape[0], W.numel())).indices] = 1.0
    return M


def feddualavg_en(z, y, zv, yv, nk, lam, l2_ratio, clients, rounds=200, epochs=3, bs=512, eta_c=0.01, eta_s=1.0):
    """ours_head.feddualavg with train_vlg.py's elastic-net prox (--dual_l2_lam): the l1 soft
    threshold at t*lam, divided by 1 + t*lam*l2_ratio, at every step and at the server."""
    dev = z.device
    g = torch.Generator(device="cpu").manual_seed(0)
    n = float(sum(len(c) for c in clients))
    wts = [len(c) / n for c in clients]
    zw, zb = torch.zeros(nk, z.shape[1], device=dev), torch.zeros(nk, device=dev)
    K = epochs * -(-len(clients[0]) // bs)
    prox = lambda v, t: ours_head.soft(v, t * lam) / (1.0 + t * lam * l2_ratio)  # noqa: E731
    best, best_acc, best_nnz = None, -1.0, None
    for r in range(rounds):
        dw, db = torch.zeros_like(zw), torch.zeros_like(zb)
        for ci, ix in enumerate(clients):
            lw, lb, step = zw.clone(), zb.clone(), 0
            for _ in range(epochs):
                perm = ix[torch.randperm(len(ix), generator=g).to(dev)]
                for s in range(0, len(perm), bs):
                    b_ix = perm[s:s + bs]
                    w = prox(lw, eta_s * eta_c * (r * K + step + 1))
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
        w_s = prox(zw, eta_s * eta_c * (r + 1) * K)
        acc = core.accuracy(core.Head(w_s, zb, torch.arange(nk)), zv, yv)
        nnz = int((w_s.abs() > 1e-5).sum())
        if acc > best_acc or (acc == best_acc and best_nnz is not None and nnz < best_nnz):
            best, best_acc, best_nnz = (w_s.clone(), zb.clone(), r + 1), acc, nnz
    return best


def parse_lams(items):
    """['5:0.02', '20:0.005'] -> {5: 0.02, 20: 0.005}"""
    return {int(k): float(v) for k, v in (i.split(":") for i in items)}


@torch.no_grad()
def central_path(z, y, nk, targets, alpha=0.99, n_lambda=60, min_ratio=1e-3, iters=100, tol=1e-6,
                 refit_iters=1000, chunk=65536):
    """core.fit_sparse_path on the pooled rows (one "client" holding everything), in row chunks so
    ImageNet's features stay fp16 on the card. Same objective (glm_saga's, as VLG-CBM's sparse
    layer: mean CE + lam ((1 - alpha)/2 ||W||^2 + alpha ||W||_1) on standardized scores), FISTA
    with step 1/L down a warm-started geometric lam path; per target the path point with the
    nearest NEC, then that support refit without the penalty. Weights are mapped back to raw
    scores. Returns {target: {nec, lam, head (W, b), head_refit (W, b)}}."""
    dev = z.device
    n, f = z.shape
    s1, s2 = torch.zeros(f, dtype=torch.float64, device=dev), torch.zeros(f, dtype=torch.float64, device=dev)
    for s in range(0, n, chunk):
        x = z[s:s + chunk].float()
        s1 += x.sum(0).double()
        s2 += (x * x).sum(0).double()
    mu = (s1 / n).float()
    sd = ((s2 - n * (s1 / n) ** 2) / max(n - 1, 1)).clamp(min=0).sqrt().float().clamp(min=1e-6)

    def rows(s):
        return (z[s:s + chunk].float() - mu) / sd

    def grad(W, b):
        gW, gb = torch.zeros_like(W), torch.zeros_like(b)
        for s in range(0, n, chunk):
            x = rows(s)
            P = torch.softmax(x @ W.T + b, dim=1)
            P[torch.arange(len(x), device=dev), y[s:s + chunk]] -= 1.0
            gW += P.T @ x
            gb += P.sum(0)
        return gW / n, gb / n

    v = torch.randn(f, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    for _ in range(30):
        u = torch.zeros(f, device=dev)
        for s in range(0, n, chunk):
            x = rows(s)
            u += x.T @ (x @ v)
        v = u / u.norm()
    sq = sum(float((rows(s) @ v).pow(2).sum()) for s in range(0, n, chunk)) / n
    prior = torch.bincount(y, minlength=nk).float().clamp(min=1) / n
    W, b = torch.zeros(nk, f, device=dev), prior.log() - prior.log().mean()
    lam_max = float(grad(W, b)[0].abs().max()) / alpha
    best = {}
    for lam in [lam_max * r for r in torch.logspace(0, torch.log10(torch.tensor(min_ratio)).item(), n_lambda).tolist()]:
        L = 0.5 * sq + lam * (1 - alpha)
        Wy, by, t_m = W.clone(), b.clone(), 1.0
        for _ in range(iters):
            gW, gb = grad(Wy, by)
            W_new = Wy - (gW + lam * (1 - alpha) * Wy) / L
            W_new = torch.sign(W_new) * torch.clamp(W_new.abs() - lam * alpha / L, min=0.0)
            b_new = by - gb / L
            t_n = (1 + (1 + 4 * t_m * t_m) ** 0.5) / 2
            Wy = W_new + ((t_m - 1) / t_n) * (W_new - W)
            by = b_new + ((t_m - 1) / t_n) * (b_new - b)
            delta = float((W_new - W).abs().max())
            W, b, t_m = W_new, b_new, t_n
            if delta < tol:
                break
        nec = float((W != 0).sum()) / nk
        for t in targets:
            if t not in best or abs(nec - t) < abs(best[t]["nec"] - t):
                best[t] = {"nec": nec, "lam": lam, "W": W.clone(), "b": b.clone()}
        if nec > 1.5 * max(targets):
            break
    out = {}
    for t, p in best.items():
        M = (p["W"] != 0).float()
        Wr, br = p["W"].clone(), p["b"].clone()
        Wy, by, t_m = Wr.clone(), br.clone(), 1.0
        for _ in range(refit_iters):
            gW, gb = grad(Wy, by)
            W_new = Wy - gW * M / (0.5 * sq)
            b_new = by - gb / (0.5 * sq)
            t_n = (1 + (1 + 4 * t_m * t_m) ** 0.5) / 2
            Wy = W_new + ((t_m - 1) / t_n) * (W_new - Wr)
            by = b_new + ((t_m - 1) / t_n) * (b_new - br)
            delta = float((W_new - Wr).abs().max())
            Wr, br, t_m = W_new, b_new, t_n
            if delta < tol:
                break
        raw = lambda W, b: (W / sd, b - (W / sd) @ mu)  # noqa: E731
        out[t] = {"nec": p["nec"], "lam": p["lam"], "head": raw(p["W"], p["b"]), "head_refit": raw(Wr, br)}
    return out


def main():
    ap = run.build_parser()
    ap.add_argument("--runs", nargs="+", required=True, metavar="PARTITION=DIR")
    ap.add_argument("--parts", nargs="+", choices=["dense", "prune", "oracle", "thresh", "fedmid", "elastic", "central"],
                    default=["dense", "prune", "oracle"])
    ap.add_argument("--fedmid_lams", nargs="+", default=[], metavar="NEC:LAM",
                    help="FedMiD lambda per NEC target (interpolated from the FEDMID runs), bracketed by --mults")
    ap.add_argument("--elastic_lams", nargs="+", default=[], metavar="NEC:LAM",
                    help="elastic-net FedDualAvg l1 lambda per NEC target (the head rule's), bracketed by --mults")
    ap.add_argument("--central_lams", nargs="+", default=[], metavar="NEC:LAM",
                    help="the head rule's lambda per NEC target; scaled to one client, bracketed by --mults")
    ap.add_argument("--mults", type=float, nargs="+", default=[0.6, 1.0, 1.6])
    ap.add_argument("--central_mults", type=float, nargs="+", default=[0.3, 0.45, 0.65, 1.0],
                    help="bracket for central; one client lands at a lower NEC than the step ratio predicts "
                         "(local check, C100 s31: x0.6 already gives NEC 6.6 for target 10)")
    ap.add_argument("--elastic_l2_ratio", type=float, default=1.0, help="lambda_2 / lambda_1 (train_vlg --dual_l2_lam)")
    ap.add_argument("--refit_rounds", type=int, default=100, help="own-support refit for every sparse row (head rule: 100)")
    ap.add_argument("--necs", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--prune_modes", nargs="+", choices=["perclass", "global"], default=["perclass", "global"])
    ap.add_argument("--prune_lrs", type=float, nargs="+", default=[1e-2, 3e-2, 1e-1],
                    help="Adam lr grid for the masked fine-tune, picked on val")
    ap.add_argument("--gd_refit_rounds", type=int, default=0,
                    help="> 0: also report ours_head.refit (masked GD) for this many rounds")
    ap.add_argument("--oracle_lambdas", type=int, default=60)
    ap.add_argument("--fedprox_mu", type=float, default=0.01)
    ap.add_argument("--check_rows", type=int, default=4096)
    ap.add_argument("--check_tol", type=float, default=1e-2)
    ap.add_argument("--acc_tol", type=float, default=0.005)
    args = ap.parse_args()
    runs = dict(s.split("=", 1) for s in args.runs)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    tag = datetime.datetime.now().strftime("%b%d-%H:%M").lower()
    save_dir = os.path.join(args.save_dir, f"dense-prune-{args.dataset}-s{args.seed}-{tag}")
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
    out = {"dataset": args.dataset, "seed": args.seed, "method": "dense FedAvg / FedProx; FedAvg cut to NEC + masked fine-tune; centralized sparse oracle; FedAvg + l1 threshold, FedMiD, elastic-net FedDualAvg (each + own-support refit)",
           "parts": args.parts, "necs": args.necs, "prune_lrs": args.prune_lrs, "fedprox_mu": args.fedprox_mu,
           "mults": args.mults, "elastic_l2_ratio": args.elastic_l2_ratio, "refit_rounds": args.refit_rounds,
           "partitions": {}}
    for p, iid, alpha in run.arms(args):
        r = loaded[p]
        ci = run.split_dataset_for_federated(d["train_ds"], args.num_clients, iid=iid, alpha=alpha, seed=args.seed)
        rec = {"run": r["path"], "checks": op.check_rows(r, d, ci, args)}
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

        def gacc(W, b):
            return 100 * core.accuracy(core.Head(W, b, torch.arange(nk)), zt, yt)

        def nec(W):
            return float((W.abs() > 1e-5).sum()) / nk

        dense = {}
        for name, mu in (("fedavg", 0.0), ("fedprox", args.fedprox_mu)):
            if "dense" not in args.parts and not (name == "fedavg" and "prune" in args.parts):
                continue
            th = time.time()
            W, b, info = fedavg_dense(z, y, zv, yv, nk, clients, mu=mu)
            dense[name] = (W, b)
            info.update(global_test=gacc(W, b), nec=nec(W), seconds=round(time.time() - th, 1))
            del info["val_by_round"]
            rec[name] = info
            print(f"[{p}] dense {name}: global test {info['global_test']:.2f} (best round {info['best_round']}, "
                  f"val {100 * info['val']:.2f}, NEC {info['nec']:.0f}, {info['seconds']:.0f}s)", flush=True)
        if "prune" in args.parts:
            W, b = dense["fedavg"]
            rec["prune"] = {}
            for mode in args.prune_modes:
                for k in args.necs:
                    th = time.time()
                    M = cut(W, k, mode)
                    fits = {lr: fedavg_dense(z, y, zv, yv, nk, clients, lr=lr, mask=M, init=(W, b))
                            for lr in args.prune_lrs}
                    lr = max(fits, key=lambda a: fits[a][2]["val"])
                    Wf, bf, info = fits[lr]
                    res = {"nec": float((Wf * M != 0).sum()) / nk, "global_test": gacc(Wf, bf), "lr": lr,
                           "best_round": info["best_round"], "val": info["val"],
                           "by_lr": {str(a): {"val": f[2]["val"], "global_test": gacc(f[0], f[1]),
                                              "best_round": f[2]["best_round"]} for a, f in fits.items()},
                           "no_refit_global_test": gacc(W * M, b),
                           "empty_classes": int((M.sum(1) == 0).sum())}
                    if args.gd_refit_rounds > 0:
                        res["gd_refit_global_test"] = gacc(*ours_head.refit(W, b, M, z, y, nk, clients,
                                                                            rounds=args.gd_refit_rounds))
                    res["seconds"] = round(time.time() - th, 1)
                    rec["prune"].setdefault(mode, {})[str(k)] = res
                    print(f"[{p}] prune {mode} NEC {k}: fine-tuned {res['global_test']:.2f} @ {res['nec']:.2f} "
                          f"(lr {lr:g}, round {res['best_round']}; cut only {res['no_refit_global_test']:.2f}, "
                          f"empty classes {res['empty_classes']}, {res['seconds']:.0f}s)", flush=True)
        def refit_own(W, b):
            """The head rule's last step: refit on this head's own support (ours_head.refit, own_100)."""
            return ours_head.refit(W, b, (W.abs() > 1e-5).float(), z, y, nk, clients, rounds=args.refit_rounds)

        def nearest(cands, k):
            """cands: [(lam, W, b, best_round)]; the run whose NEC is nearest k, then its refit."""
            pts = [{"lam": lam, "nec": nec(W), "global_test": gacc(W, b), "best_round": br} for lam, W, b, br in cands]
            i = min(range(len(pts)), key=lambda j: abs(pts[j]["nec"] - k))
            Wr, br_ = refit_own(cands[i][1], cands[i][2])
            return {"nec": pts[i]["nec"], "lam": pts[i]["lam"], "no_refit_global_test": pts[i]["global_test"],
                    "global_test": gacc(Wr, br_), "bracket": pts}

        if "thresh" in args.parts:
            rec["thresh"] = {}
            for k in args.necs:
                th = time.time()
                W, b, info = fedavg_dense(z, y, zv, yv, nk, clients, budget=k)
                Wr, br = refit_own(W, b)
                rec["thresh"][str(k)] = res = {"nec": nec(W), "no_refit_global_test": gacc(W, b), "global_test": gacc(Wr, br),
                                               "best_round": info["best_round"], "val": info["val"],
                                               "seconds": round(time.time() - th, 1)}
                print(f"[{p}] FedAvg + l1 threshold NEC {k}: refit {res['global_test']:.2f} @ {res['nec']:.2f} (before "
                      f"refit {res['no_refit_global_test']:.2f}, round {res['best_round']}, {res['seconds']:.0f}s)", flush=True)
        for part, lams in (("fedmid", parse_lams(args.fedmid_lams)), ("elastic", parse_lams(args.elastic_lams))):
            if part not in args.parts:
                continue
            rec[part] = {}
            for k in args.necs:
                th = time.time()
                cands = []
                for m in args.mults:
                    lam = lams[k] * m
                    if part == "fedmid":
                        (W, b, br), _ = fm.fedmid(z, y, zv, yv, nk, lam, clients)
                    else:
                        W, b, br = feddualavg_en(z, y, zv, yv, nk, lam, args.elastic_l2_ratio, clients)
                    cands.append((lam, W, b, br))
                res = nearest(cands, k)
                res["seconds"] = round(time.time() - th, 1)
                rec[part][str(k)] = res
                print(f"[{p}] {part} NEC~{k}: refit {res['global_test']:.2f} @ {res['nec']:.2f} (before refit "
                      f"{res['no_refit_global_test']:.2f}, lam {res['lam']:.3g}; bracket NEC "
                      f"{[round(q['nec'], 1) for q in res['bracket']]}, {res['seconds']:.0f}s)", flush=True)
        if "central" in args.parts:
            sizes = torch.tensor([float(len(c)) for c in clients])
            ratio = float(sizes[0] / ((sizes ** 2).sum() / sizes.sum()))
            pooled = [torch.arange(held.numel(), device=device)]
            rec["central"] = {"lam_ratio": ratio}
            for k, lam0 in parse_lams(args.central_lams).items():
                th = time.time()
                cands = []
                for m in args.central_mults:
                    lam = lam0 * ratio * m
                    W, b, _ = ours_head.feddualavg(z, y, zv, yv, nk, lam, pooled)
                    cands.append((lam, W, b, None))
                pts = [{"lam": lam, "nec": nec(W), "global_test": gacc(W, b)} for lam, W, b, _ in cands]
                i = min(range(len(pts)), key=lambda j: abs(pts[j]["nec"] - k))
                W, b = cands[i][1], cands[i][2]
                Wr, br = ours_head.refit(W, b, (W.abs() > 1e-5).float(), z, y, nk, pooled, rounds=args.refit_rounds)
                res = {"nec": pts[i]["nec"], "lam": pts[i]["lam"], "no_refit_global_test": pts[i]["global_test"],
                       "global_test": gacc(Wr, br), "bracket": pts, "seconds": round(time.time() - th, 1)}
                rec["central"][str(k)] = res
                print(f"[{p}] central head rule NEC~{k}: refit {res['global_test']:.2f} @ {res['nec']:.2f} (before "
                      f"refit {res['no_refit_global_test']:.2f}, lam {res['lam']:.3g} = x{ratio:.2f} x bracket; bracket "
                      f"NEC {[round(q['nec'], 1) for q in pts]}, {res['seconds']:.0f}s)", flush=True)
        if "oracle" in args.parts:
            th = time.time()
            paths = central_path(z, y, nk, args.necs, n_lambda=args.oracle_lambdas)
            rec["oracle"] = {str(k): {"nec": v["nec"], "lam": v["lam"], "path_global_test": gacc(*v["head"]),
                                      "global_test": gacc(*v["head_refit"])} for k, v in paths.items()}
            rec["oracle_seconds"] = round(time.time() - th, 1)
            for k, v in rec["oracle"].items():
                print(f"[{p}] oracle NEC~{k}: refit {v['global_test']:.2f} @ {v['nec']:.2f} (path "
                      f"{v['path_global_test']:.2f}, lam {v['lam']:.3g})", flush=True)
            print(f"[{p}] oracle took {rec['oracle_seconds']:.0f}s", flush=True)
        out["partitions"][p] = rec
        del z
        torch.cuda.empty_cache()
    out["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {save_dir}")


if __name__ == "__main__":
    main()
