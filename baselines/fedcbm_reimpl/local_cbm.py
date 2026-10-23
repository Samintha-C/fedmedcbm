"""Fully local CBMs against ours: each client trains its own concept layer on its
own data only, then its own head, and is compared with our federated concept
layer under the SAME private heads (reviewer T014 / T019.2: what federating each
layer buys).

Three-way comparison, per partition:
  ours     federated CBL + federated sparse head. Read from the run's own global
           head here (check 4 of ours_personal.py) and from OURS_HEAD_OWN; not
           refitted.
  fed_cbl  federated CBL + a private head per client: ours_personal.py's heads
           plus nec.py's support refit. Differs from ours only in the head.
  local    each client's own CBL, trained exactly as one client of Phase 1 with no
           one to average with: the same initialization for every client, the
           run's rounds x local epochs, Adam, BCE with its pos_weight, the ortho
           penalty and best-on-val restore (train_cbl / validate_cbl), then its own
           normalization and its own private head. Differs from fed_cbl only in
           the concept layer.
Phase-1 settings are read from the federated run's args.txt, so local and
federated training differ only in the averaging. Heads for fed_cbl and local are
the same calls (core.fit_head over --head_C_grid; core.fit_sparse_path with the
support refit), scored the same way (global test, mean over clients; per-client
test splits). Also recorded, for what an auditor would see:
  support   union NEC and pairwise Jaccard of the clients' per-class supports
            (core.support_stats; concept indices name the same concept everywhere);
  concepts  per CBL, mean concept AUC against the DINO test labels; for the local
            CBLs, the agreement of two clients' activations on the test images (per
            concept Pearson correlation, mean over concepts and client pairs) and
            each one's agreement with the federated CBL. One federated CBL agrees
            with itself, so its value is 1.

Usage: run.py's data flags, plus --runs as ours_personal.py (with --alphas, one
run per arm as ours_head.py: --runs iid=<dir> a0.05=<dir> ...).
"""

import copy
import datetime
import importlib.util
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import core  # noqa: E402
import nec  # noqa: E402
import ours_personal as op  # noqa: E402
import run  # noqa: E402
from models.fed_vlgcbm import ConceptLayer, train_cbl, validate_cbl  # noqa: E402

# Phase-1 settings copied from the federated run (args.txt key -> value used here).
PHASE1_KEYS = ("num_rounds", "cbl_epochs", "cbl_lr", "cbl_batch_size", "cbl_pos_weight",
               "cbl_optimizer", "weight_decay", "ortho_lambda", "cbl_loss_type", "cbl_auto_weight")


def load_get_loss():
    """utils/loss_vlg.py by path, as train_vlg.py does (Label-free-CBM also has a utils)."""
    spec = importlib.util.spec_from_file_location(
        "fed_loss_vlg", os.path.join(run._FED_ROOT, "utils", "loss_vlg.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_loss


def phase1_settings(r):
    ra = r["args"]
    hp = {k: ra.get(k) for k in PHASE1_KEYS}
    if hp["cbl_epochs"] is None:
        hp["cbl_epochs"] = ra["local_epochs"]
    if hp["cbl_loss_type"] != "bce" or hp["cbl_auto_weight"]:
        sys.exit(f"ERROR: {r['path']}: local CBLs mirror plain BCE only, run has {hp}")
    return hp


def train_local_cbl(tr, rows, val_loader, init, hp, loss_fn, device):
    """One client of Phase 1 with nobody to average with: num_rounds calls of train_cbl
    (fresh Adam each, best-epoch restore on val), keeping the best round on val."""
    cbl = copy.deepcopy(init)
    loader = DataLoader(Subset(tr, rows), batch_size=hp["cbl_batch_size"], shuffle=True, num_workers=0)
    identity = torch.nn.Identity()
    best, best_state = float("inf"), None
    for _ in range(hp["num_rounds"]):
        train_cbl(identity, cbl, loader, epochs=hp["cbl_epochs"], loss_fn=loss_fn, lr=hp["cbl_lr"],
                  weight_decay=hp["weight_decay"], device=str(device), optimizer_name=hp["cbl_optimizer"],
                  val_loader=val_loader, lambda_ortho=hp["ortho_lambda"])
        v = validate_cbl(identity, cbl, val_loader, loss_fn, str(device))
        if v < best:
            best, best_state = v, copy.deepcopy(cbl.state_dict())
    cbl.load_state_dict(best_state)
    return cbl, best


@torch.no_grad()
def local_run(cbl, feats, rows, device, chunk=32768):
    """The {W, b, mean, std} of op.concept_scores for a local CBL, normalized with this
    client's own statistics (train_vlg.py Step 2: population variance, clamp 1e-8)."""
    W = cbl.model[0].weight.detach().float().to(device)
    b = cbl.model[0].bias.detach().float().to(device)
    s1 = torch.zeros(W.shape[0], dtype=torch.float64, device=device)
    s2 = torch.zeros_like(s1)
    for s in range(0, rows.numel(), chunk):
        a = (feats[rows[s:s + chunk]].to(device).float() @ W.T + b).double()
        s1 += a.sum(0)
        s2 += (a * a).sum(0)
    mean = s1 / rows.numel()
    std = (s2 / rows.numel() - mean ** 2).clamp(min=1e-8).sqrt()
    return {"W": W, "b": b, "mean": mean.float(), "std": std.float()}


@torch.no_grad()
def concept_auc(z, labels):
    """Mean over concepts (with both classes present) of the rank AUC of z against labels."""
    y = labels.bool().to(z.device)
    pos = y.sum(0).double()
    neg = y.shape[0] - pos
    ranks = z.argsort(0).argsort(0).double() + 1
    auc = ((ranks * y).sum(0) - pos * (pos + 1) / 2) / (pos * neg).clamp(min=1)
    ok = (pos > 0) & (neg > 0)
    return float(auc[ok].mean())


@torch.no_grad()
def agreement(za, zb):
    """Mean over concepts of the Pearson correlation of two CBLs' test activations."""
    a, b = za - za.mean(0), zb - zb.mean(0)
    r = (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp(min=1e-12)
    return float(r.mean())


def score(heads, zv, yv, zt, yt, split):
    """nec.score with one feature set per client (local CBLs, held on the CPU and moved to
    the labels' device one client at a time: five ImageNet sets do not fit a 24 GB GPU) or
    the same one for all."""
    dev = yt.device
    g_val = [core.accuracy(h, zv[m].to(dev), yv) for m, h in enumerate(heads)]
    g_test = [core.accuracy(h, zt[m].to(dev), yt) for m, h in enumerate(heads)]
    correct, total, each = 0.0, 0, []
    for m, (h, ix) in enumerate(zip(heads, split)):
        each.append(None)
        if ix.numel():
            ix = ix.to(dev)
            each[-1] = core.accuracy(h, zt[m].to(dev)[ix], yt[ix])
            correct += each[-1] * ix.numel()
            total += ix.numel()
    return {"global_val_mean_client_acc": sum(g_val) / len(g_val),
            "global_test_mean_client_acc": sum(g_test) / len(g_test),
            "personalized_test_acc_weighted": correct / max(total, 1),
            "client_global_test_acc": g_test, "personalized_test_acc_each": each}


def fit_heads(z, y, args):
    dense = [core.fit_head(z, y, C=C, max_iter=max(50, args.head_max_iter), tol=args.head_tol)
             for C in args.head_C_grid]
    path = core.fit_sparse_path(z, y, args.nec_targets, n_lambda=args.path_lambdas, iters=args.path_iters,
                                refit_iters=args.path_refit_iters)
    return dense, path


def main():
    ap = run.build_parser()
    ap.add_argument("--runs", nargs="+", required=True, metavar="PARTITION=DIR")
    ap.add_argument("--nec_targets", type=float, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--head_C_grid", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--path_lambdas", type=int, default=40)
    ap.add_argument("--path_iters", type=int, default=100)
    ap.add_argument("--path_refit_iters", type=int, default=1000,
                    help="support refit without the penalty, as nec.py; 1000 = our refit's 100 rounds x 10")
    ap.add_argument("--local_init", choices=["independent", "shared"], default="independent",
                    help="independent: each client draws its own initialization, as clients that never "
                         "coordinate would; shared: one initialization for all (the first LOCAL_CBM runs). "
                         "On small clients (CUB) a shared init keeps local CBLs correlated through it, so "
                         "agreement and support overlap read high")
    ap.add_argument("--check_rows", type=int, default=4096)
    ap.add_argument("--check_tol", type=float, default=1e-2)
    ap.add_argument("--acc_tol", type=float, default=0.005)
    args = ap.parse_args()
    runs = dict(s.split("=", 1) for s in args.runs)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    tag = datetime.datetime.now().strftime("%b%d-%H:%M").lower()
    save_dir = os.path.join(args.save_dir, f"local-cbm-{args.dataset}-s{args.seed}-{tag}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    t0 = time.time()

    loaded = {}
    for p, _, alpha in run.arms(args):
        r = op.load_run(op.resolve_run(runs[p]), device)
        op.check_args(r, args, p, alpha)
        loaded[p] = r
        print(f"[run] {p}: {r['path']}  ({r['W'].shape[0]} concepts)", flush=True)
    d = run.load_data(args, device)
    tr, va, te = d["tr"], d["va"], d["te"]
    nk = len(d["classes"])
    y_val, y_test = va._target_cache.to(device), te._target_cache.to(device)
    get_loss = load_get_loss()
    out = {"dataset": args.dataset, "seed": args.seed, "num_clients": args.num_clients,
           "method": "fed_cbl: federated CBL + private heads; local: own CBL + own private head",
           "num_classes": nk, "nec_targets": args.nec_targets, "head_C_grid": args.head_C_grid,
           "local_init": args.local_init,
           "runs": {}, "concepts": {}, "client_stats": {}, "fed_cbl": {}, "local": {}}

    for p, iid, alpha in run.arms(args):
        r = loaded[p]
        nc = r["W"].shape[0]
        hp = phase1_settings(r)
        ci = run.split_dataset_for_federated(d["train_ds"], args.num_clients, iid=iid, alpha=alpha, seed=args.seed)
        rec = {"path": r["path"], "num_concepts": nc, "phase1": hp, "checks": op.check_rows(r, d, ci, args)}
        z_val, z_test = op.concept_scores(va._feature_cache, r), op.concept_scores(te._feature_cache, r)
        gh = core.Head(weight=r["final_W"], bias=r["final_b"], classes=torch.arange(nk))
        acc, ref = core.accuracy(gh, z_test, y_test), float(r["metrics"]["metrics"]["test_accuracy"])
        rec["global_head"] = {"test_acc": acc, "metrics_txt_test_acc": ref,
                              "nec": float((r["final_W"].abs() > 1e-5).sum()) / nk}
        print(f"[check] {p}: {rec['checks']} | global head {acc:.4f} vs {ref:.4f} in metrics.txt", flush=True)
        if abs(acc - ref) > args.acc_tol:
            sys.exit(f"ERROR: {r['path']}: global head does not reproduce its test accuracy: {rec['global_head']}")
        # Held on the CPU like the local sets and moved per client by score(): on ImageNet the
        # pair is 2 GB of GPU memory the head fits need.
        z_val, z_test = z_val.cpu(), z_test.cpu()

        loss_fn = get_loss("bce", nc, 0, [0] * nc, hp["cbl_pos_weight"], False, device=str(device))
        val_loader = DataLoader(va, batch_size=hp["cbl_batch_size"], shuffle=False, num_workers=0)
        # Independent seeds also avoid seed alone, which is the federated Dirichlet run's own
        # init (train_vlg seeds, and its Dirichlet split draws from numpy, not torch).
        inits = []
        for s in ([args.seed * 1000 + m for m in range(args.num_clients)]
                  if args.local_init == "independent" else [args.seed]):
            torch.manual_seed(s)
            inits.append(ConceptLayer(r["W"].shape[1], nc, num_hidden=0, bias=True, device=str(device)))

        heads = {v: {"dense": [], "sparse": {t: [] for t in args.nec_targets},
                     "refit": {t: [] for t in args.nec_targets}, "nec": {t: [] for t in args.nec_targets},
                     "empty": {t: [] for t in args.nec_targets}} for v in ("fed_cbl", "local")}
        zl_val, zl_test, labels, val_loss = [], [], [], []
        for m in range(args.num_clients):
            idx = torch.as_tensor(ci[m], dtype=torch.long)
            y = tr._target_cache[idx]
            labels.append(y)
            th = time.time()
            cbl, vl = train_local_cbl(tr, ci[m], val_loader, inits[m % len(inits)], hp, loss_fn, device)
            val_loss.append(vl)
            loc = local_run(cbl, tr._feature_cache, idx, device)
            tl = time.time() - th
            zl_val.append(op.concept_scores(va._feature_cache, loc).cpu())
            zl_test.append(op.concept_scores(te._feature_cache, loc).cpu())
            del cbl
            torch.cuda.empty_cache()
            # One CBL's training features on the GPU at a time (ImageNet: 4.2 GB each; holding
            # both ran a 22 GB card out of memory in fit_sparse_path, 2026-10-23).
            for v, rv in (("fed_cbl", r), ("local", loc)):
                z = op.concept_scores(tr._feature_cache, rv, rows=idx)
                dense, path = fit_heads(z, y, args)
                heads[v]["dense"].append(dense)
                for t in args.nec_targets:
                    heads[v]["sparse"][t].append(path[t]["head"])
                    heads[v]["refit"][t].append(path[t]["head_refit"])
                    heads[v]["nec"][t].append(path[t]["nec"])
                    heads[v]["empty"][t].append(path[t]["empty_classes"])
                del z, path
                torch.cuda.empty_cache()
            print(f"[{p}] client {m}: {len(ci[m])} rows, local CBL {tl:.0f}s (val loss {vl:.4f}), heads "
                  f"{time.time() - th - tl:.0f}s, NEC achieved fed { {t: round(heads['fed_cbl']['nec'][t][-1], 1) for t in args.nec_targets} } "
                  f"local { {t: round(heads['local']['nec'][t][-1], 1) for t in args.nec_targets} }", flush=True)

        split = core.personalized_test_split(te._target_cache, labels, nk, seed=args.seed)
        out["client_stats"][p] = core.client_stats(labels, split, nk)
        feats = {"fed_cbl": ([z_val] * args.num_clients, [z_test] * args.num_clients), "local": (zl_val, zl_test)}
        for v in ("fed_cbl", "local"):
            zv, zt = feats[v]
            res = out[v]
            for i, C in enumerate(args.head_C_grid):
                res.setdefault("dense", {}).setdefault(str(C), {})[p] = score(
                    [h[i] for h in heads[v]["dense"]], zv, y_val, zt, y_test, split)
            for key in ("sparse", "refit"):
                for t in args.nec_targets:
                    s = score(heads[v][key][t], zv, y_val, zt, y_test, split)
                    s["nec_mean_over_clients"] = sum(heads[v]["nec"][t]) / args.num_clients
                    s["nec_per_client"] = heads[v]["nec"][t]
                    s["empty_classes_per_client"] = heads[v]["empty"][t]
                    s.update(core.support_stats(heads[v][key][t], nk, nc))
                    res.setdefault("sparse_refit" if key == "refit" else "sparse", {}).setdefault(str(t), {})[p] = s
                    print(f"[{p}] {v} {key} NEC {t:g}: achieved {s['nec_mean_over_clients']:.2f}, union "
                          f"{s['union_nec']:.1f}, jaccard {s['support_jaccard']:.3f} | global "
                          f"{s['global_test_mean_client_acc']:.4f} per-client {s['personalized_test_acc_weighted']:.4f}",
                          flush=True)
        lab, zt_cpu = te._annotation_cache, z_test.cpu()
        pairs = [(i, j) for i in range(args.num_clients) for j in range(i + 1, args.num_clients)]
        out["concepts"][p] = {
            "fed_auc": concept_auc(zt_cpu, lab),
            "local_auc_per_client": [concept_auc(z, lab) for z in zl_test],
            "local_pairwise_agreement": sum(agreement(zl_test[i], zl_test[j]) for i, j in pairs) / len(pairs),
            "local_vs_fed_agreement": [agreement(z, zt_cpu) for z in zl_test],
            "local_cbl_val_loss": val_loss}
        c = out["concepts"][p]
        print(f"[{p}] concepts: fed AUC {c['fed_auc']:.3f}, local AUC {[round(a, 3) for a in c['local_auc_per_client']]}, "
              f"local pairwise agreement {c['local_pairwise_agreement']:.3f}, local vs fed "
              f"{[round(a, 3) for a in c['local_vs_fed_agreement']]}", flush=True)
        out["runs"][p] = rec
        del z_val, z_test, zt_cpu, zl_val, zl_test, feats
        torch.cuda.empty_cache()

    out["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {save_dir}")


if __name__ == "__main__":
    main()
