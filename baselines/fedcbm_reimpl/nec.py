"""NEC-constrained FedCBM: our stronger FedCBM with sparse per-client heads.

Reuses the concept bank the strong run already selected and saved for this
seed (cav_bank.pt under --bank_root/s<seed>), so Stage A is not repeated. Then,
per partition and per client:
  dense heads   sklearn-style LR at every C in --head_C_grid (Yang & Long tune
                C over {0.01, 0.1, 1.0}); C chosen by mean client accuracy on the
                full val set, IID partition, as in run.py;
  sparse heads  one elastic-net path per client (core.fit_sparse_path), keeping
                the point nearest each NEC target; with --path_refit_iters > 0, also
                that support refit without the penalty ("sparse_refit").
Every head set is scored on the global test set (mean over clients) and on the
per-client test splits; sparse sets also report the union NEC across clients
and the support overlap (core.support_stats).

Usage: run.py's data flags, plus
  --bank_root /sc-rwx-vol/fedmedcbm/models/FEDCBM_STRONG/<dataset>
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
import run  # noqa: E402


def score(heads, z_val, y_val, z_test, y_test, split):
    g_val = [core.accuracy(h, z_val, y_val) for h in heads]
    g_test = [core.accuracy(h, z_test, y_test) for h in heads]
    correct, total, each = 0.0, 0, []
    for h, ix in zip(heads, split):
        each.append(None)
        if ix.numel():
            ix = ix.to(z_test.device)
            each[-1] = core.accuracy(h, z_test[ix], y_test[ix])
            correct += each[-1] * ix.numel()
            total += ix.numel()
    return {"global_val_mean_client_acc": sum(g_val) / len(g_val),
            "global_test_mean_client_acc": sum(g_test) / len(g_test),
            "personalized_test_acc_weighted": correct / max(total, 1),
            "client_global_test_acc": g_test, "personalized_test_acc_each": each}


def main():
    ap = run.build_parser()
    ap.add_argument("--bank_root", required=True)
    ap.add_argument("--nec_targets", type=float, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--head_C_grid", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--path_lambdas", type=int, default=40)
    ap.add_argument("--path_iters", type=int, default=100)
    ap.add_argument("--path_refit_iters", type=int, default=0,
                    help="> 0: also refit each sparse head's support without the penalty")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    banks = sorted(glob.glob(os.path.join(args.bank_root, f"s{args.seed}", "*", "cav_bank.pt")))
    if not banks:
        sys.exit(f"ERROR: no cav_bank.pt under {args.bank_root}/s{args.seed} (has the strong run finished?)")
    bank = torch.load(banks[-1], map_location="cpu", weights_only=False)
    tag = datetime.datetime.now().strftime("%b%d-%H:%M").lower()
    save_dir = os.path.join(args.save_dir, f"fedcbm-nec-{args.dataset}-s{args.seed}-{tag}")
    os.makedirs(save_dir, exist_ok=True)
    t0 = time.time()

    d = run.load_data(args, device)
    assert d["concepts"] == bank["concepts"], "concept list differs from the bank's (seed/filter mismatch)"
    tr, va, te = d["tr"], d["va"], d["te"]
    nk, nc = len(d["classes"]), len(d["concepts"])
    stage_a = {"W": bank["W"].float().to(device),
               "b": None if bank["b"] is None else bank["b"].float().to(device),
               "stats": bank["stats"]}
    z_val, z_test = run.proj(va._feature_cache, stage_a, args), run.proj(te._feature_cache, stage_a, args)
    y_val, y_test = va._target_cache.to(device), te._target_cache.to(device)
    out = {"dataset": args.dataset, "seed": args.seed, "bank": banks[-1], "bank_selected": bank.get("selected"),
           "num_concepts": nc, "num_classes": nk, "nec_targets": args.nec_targets,
           "head_C_grid": args.head_C_grid, "dense": {}, "sparse": {}}

    for p, iid, alpha in run.arms(args):
        ci = run.split_dataset_for_federated(d["train_ds"], args.num_clients, iid=iid,
                                             alpha=alpha, seed=args.seed)
        dense = {C: [] for C in args.head_C_grid}
        sparse = {t: [] for t in args.nec_targets}
        refit = {t: [] for t in args.nec_targets}
        nec_ach = {t: [] for t in args.nec_targets}
        empty = {t: [] for t in args.nec_targets}
        labels = []
        for m in range(args.num_clients):
            idx = torch.as_tensor(ci[m], dtype=torch.long)
            y = tr._target_cache[idx]
            labels.append(y)
            z = run.proj(tr._feature_cache, stage_a, args, rows=idx)
            th = time.time()
            for C in args.head_C_grid:
                dense[C].append(core.fit_head(z, y, C=C, max_iter=max(50, args.head_max_iter), tol=args.head_tol))
            td = time.time() - th
            res = core.fit_sparse_path(z, y, args.nec_targets, n_lambda=args.path_lambdas, iters=args.path_iters,
                                       refit_iters=args.path_refit_iters)
            for t in args.nec_targets:
                sparse[t].append(res[t]["head"])
                nec_ach[t].append(res[t]["nec"])
                empty[t].append(res[t]["empty_classes"])
                if args.path_refit_iters:
                    refit[t].append(res[t]["head_refit"])
            print(f"[{p}] client {m}: dense x{len(args.head_C_grid)} {td:.0f}s, path {time.time() - th - td:.0f}s, "
                  f"NEC achieved { {t: round(res[t]['nec'], 2) for t in args.nec_targets} }", flush=True)
            del z
            torch.cuda.empty_cache()
        split = core.personalized_test_split(te._target_cache, labels, nk, seed=args.seed)
        out.setdefault("client_stats", {})[p] = core.client_stats(labels, split, nk)
        for C in args.head_C_grid:
            out["dense"].setdefault(str(C), {})[p] = score(dense[C], z_val, y_val, z_test, y_test, split)
        for key, heads in (("sparse", sparse), ("sparse_refit", refit)):
            if key == "sparse_refit" and not args.path_refit_iters:
                continue
            for t in args.nec_targets:
                r = score(heads[t], z_val, y_val, z_test, y_test, split)
                r["nec_mean_over_clients"] = sum(nec_ach[t]) / len(nec_ach[t])
                r["nec_per_client"] = nec_ach[t]
                r["empty_classes_per_client"] = empty[t]
                r.update(core.support_stats(heads[t], nk, nc))
                out.setdefault(key, {}).setdefault(str(t), {})[p] = r
                print(f"[{p}] {key} NEC {t:g}: achieved {r['nec_mean_over_clients']:.2f}, union {r['union_nec']:.1f}, "
                      f"jaccard {r['support_jaccard']:.3f}, empty classes {empty[t]} | global "
                      f"{r['global_test_mean_client_acc']:.4f} per-client {r['personalized_test_acc_weighted']:.4f}",
                      flush=True)
        for C in args.head_C_grid:
            r = out["dense"][str(C)][p]
            print(f"[{p}] dense C={C:g}: global {r['global_test_mean_client_acc']:.4f} "
                  f"per-client {r['personalized_test_acc_weighted']:.4f} (val {r['global_val_mean_client_acc']:.4f})")

    names = [a[0] for a in run.arms(args)]
    sel = "iid" if "iid" in names else names[0]
    out["dense_selected_C"] = max(args.head_C_grid, key=lambda C: out["dense"][str(C)][sel]["global_val_mean_client_acc"])
    print(f"[dense] selected C={out['dense_selected_C']:g} (on {sel} val)")
    out["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {save_dir}")


if __name__ == "__main__":
    main()
