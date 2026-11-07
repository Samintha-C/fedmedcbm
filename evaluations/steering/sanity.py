"""Sanity check for steer.py: can the test fail? The same steering on concept layers whose columns
carry no image-level concept information, each with the head rule at NEC~5 fit the same way.

Layers (all linear, all with z-scored activations):
  ours            the run's federated CBL and its saved own_100 head;
  random          random unit directions in feature space, as many as the run has concepts;
  central_true    a CBL trained centrally on the true DINO labels with the run's Phase-1 settings;
  class_shuffled  the same training, with each class's label rows permuted among that class's images.
                  A column still knows which classes it belongs to and nothing about the image.
central_true is the reference for class_shuffled, so the comparison is not also federated vs central.

Heads for the new layers: FedDualAvg on the run's client split, lambda searched from the tab:main value
to reach NEC~5, then the 100-round refit on its own support. Steering is steer.py's edit on one image
set: test images the judge and every head get right, targets from classes with a positive-weight
concept in every head. Calibration: DINO rule on the layer's training labels; random has no labels of
its own, so it uses the 95th / 5th percentile, and is also run with the DINO rule. Each edit is
reported at its own size and rescaled per image to ours ("@ours norm"), so a layer cannot win by a
larger edit. CIFAR-10 / CIFAR-100 only: new heads need the full training split.

usage: python3 -I sanity.py --data <data root> --dataset c10 --seed 42
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, SPLIT, calib, concept_edit, draw, feddualavg, layer_concepts, load_seed, nec, train_judge  # noqa: E402
import local_cbm as lc  # noqa: E402  (common.py puts baselines/fedcbm_reimpl on the path)
import ours_head  # noqa: E402
from data.data_utils import split_dataset_for_federated  # noqa: E402
from models.fed_vlgcbm import ConceptLayer  # noqa: E402

TAB_LAM = {"c10": 0.022, "c100": 0.01}


class _Targets:
    """Just enough of a dataset for split_dataset_for_federated (it reads .targets)."""
    def __init__(self, t):
        self.targets = t

    def __len__(self):
        return len(self.targets)


def shuffle_within_class(lab, y, seed):
    g = torch.Generator().manual_seed(seed)
    out = lab.clone()
    for k in y.unique().tolist():
        ix = (y == k).nonzero().squeeze(1)
        out[ix] = lab[ix[torch.randperm(len(ix), generator=g)]]
    return out


def fit_head(L, Xtr, ytr, Xva, yva, ci, nk, lam0, tag):
    """FedDualAvg with lambda searched to NEC~5, then the own-support refit (the head rule)."""
    held = torch.cat(ci)
    z, y = layer_concepts(Xtr[held], L), ytr[held]
    sizes = [len(c) for c in ci]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    clients = [torch.arange(s, s + n) for s, n in zip(starts, sizes)]
    zv = layer_concepts(Xva, L)
    lam, tried, lo, hi = lam0, [], None, None
    for _ in range(6):
        t = time.time()
        (W, b, _), _, _ = feddualavg(z, y, zv, yva, nk, lam, clients)
        n = nec(W)
        tried.append((lam, n, W, b))
        print(f"  [{tag}] lam {lam:.5f}: NEC {n:.2f} ({time.time() - t:.0f}s)", flush=True)
        if 4.25 <= n <= 5.75:
            break
        if n > 5:
            lo = lam
        else:
            hi = lam
        lam = (lo * hi) ** 0.5 if lo and hi else lam * min(4.0, max(0.25, n / 5))
    lam, _, W, b = min(tried, key=lambda r: abs(r[1] - 5))
    Wr, br = ours_head.refit(W, b, (W.abs() > 1e-5).float(), z, y, nk, clients, rounds=100)
    return Wr, br, lam


def train_layer(lab_tr, lab_va, Xtr, ytr, Xva, yva, run, seed):
    hp = lc.phase1_settings({"args": json.load(open(os.path.join(run, "args.txt"))), "path": run})
    nc = lab_tr.shape[1]
    loss_fn = lc.load_get_loss()("bce", nc, 0, [0] * nc, hp["cbl_pos_weight"], False, device="cpu")
    tr = TensorDataset(Xtr, lab_tr.float(), ytr)
    val = DataLoader(TensorDataset(Xva, lab_va.float(), yva), batch_size=hp["cbl_batch_size"], shuffle=False)
    torch.manual_seed(seed * 1000)
    init = ConceptLayer(Xtr.shape[1], nc, num_hidden=0, bias=True, device="cpu")
    cbl, _ = lc.train_local_cbl(tr, list(range(len(ytr))), val, init, hp, loss_fn, torch.device("cpu"))
    r = lc.local_run(cbl, Xtr, torch.arange(len(ytr)), torch.device("cpu"))
    return {"Wc": r["W"], "bc": r["b"], "mean": r["mean"], "std": r["std"]}


def build(data, ds, seed, L0, W0, b0, f, lab_tr, lab_va, cache):
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = f["train"], f["val"], f["test"]
    nk, nc = int(ytr.max()) + 1, L0["Wc"].shape[0]
    run = os.path.join(data, "saved", ds, f"s{seed}")
    args = json.load(open(os.path.join(run, "args.txt")))
    assert args["seed"] == seed and args["alpha"] == 0.5 and args["num_clients"] == 5, args
    ci = [torch.as_tensor(c, dtype=torch.long) for c in
          split_dataset_for_federated(_Targets(ytr.numpy()), 5, iid=False, alpha=0.5, seed=seed)]
    sizes = json.load(open(os.path.join(run, "training_metrics.json")))["client_data_sizes"]
    assert [len(c) for c in ci] == sizes, ([len(c) for c in ci], sizes)
    out = {"layers": {"ours": L0}, "heads": {"ours": (W0, b0)}, "labels": {"ours": lab_tr}, "lam": {}}

    g = torch.Generator().manual_seed(seed)
    R = torch.randn(nc, Xtr.shape[1], generator=g)
    R = R / R.norm(dim=1, keepdim=True)
    a = Xtr @ R.T
    out["layers"]["random"] = {"Wc": R, "bc": torch.zeros(nc), "mean": a.mean(0), "std": a.std(0)}
    out["labels"]["random"] = lab_tr

    t = time.time()
    out["layers"]["central_true"] = train_layer(lab_tr, lab_va, Xtr, ytr, Xva, yva, run, seed)
    out["labels"]["central_true"] = lab_tr
    print(f"[{ds} s{seed}] central_true CBL trained ({time.time() - t:.0f}s)", flush=True)
    t = time.time()
    sh_tr, sh_va = shuffle_within_class(lab_tr, ytr, seed), shuffle_within_class(lab_va, yva, seed + 1)
    assert torch.equal(sh_tr[ytr == 0].sum(0), lab_tr[ytr == 0].sum(0))
    out["layers"]["class_shuffled"] = train_layer(sh_tr, sh_va, Xtr, ytr, Xva, yva, run, seed)
    out["labels"]["class_shuffled"] = sh_tr
    print(f"[{ds} s{seed}] class_shuffled CBL trained ({time.time() - t:.0f}s)", flush=True)

    for v in ("random", "central_true", "class_shuffled"):
        W, b, lam = fit_head(out["layers"][v], Xtr, ytr, Xva, yva, ci, nk, TAB_LAM[ds], f"{ds} s{seed} {v}")
        out["heads"][v], out["lam"][v] = (W, b), lam
    torch.save(out, cache)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data root (see common.py)")
    ap.add_argument("--out", default=None, help="layers cache + results dir (default <data>/saved/sanity)")
    ap.add_argument("--dataset", required=True, choices=sorted(TAB_LAM))
    ap.add_argument("--seed", type=int, required=True, choices=sorted(SPLIT))
    ap.add_argument("--judges", nargs="+", default=["linear", "mlp"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ds, seed = args.dataset, args.seed
    L0, heads, metrics, f, lab_tr = load_seed(args.data, ds, seed)
    W0, b0 = heads["own_100"]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = f["train"], f["val"], f["test"]
    nk = W0.shape[0]
    keep = torch.load(os.path.join(args.data, CACHE[ds][1].format(SPLIT[seed][0])), map_location="cpu").sum(0) > 0
    lab_va = torch.load(os.path.join(args.data, CACHE[ds][1].format(SPLIT[seed][1])), map_location="cpu")[:, keep]
    d = args.out or os.path.join(args.data, "saved", "sanity")
    os.makedirs(d, exist_ok=True)
    B = build(args.data, ds, seed, L0, W0, b0, f, lab_tr, lab_va, os.path.join(d, f"layers_{ds}_s{seed}.pt"))

    head = lambda v, F: layer_concepts(F, B["layers"][v]) @ B["heads"][v][0].T + B["heads"][v][1]  # noqa: E731
    arms = ["ours", "central_true", "class_shuffled", "random"]
    info = {}
    with torch.no_grad():
        for v in arms:
            W = B["heads"][v][0]
            info[v] = {"test": 100 * float((head(v, Xte).argmax(1) == yte).float().mean()), "nec": nec(W),
                       "classes_with_pos": int(((W > 1e-5).sum(1) > 0).sum()), "lam": B["lam"].get(v),
                       "auc_true_labels_val": float(lc.concept_auc(layer_concepts(Xva, B["layers"][v]), lab_va))}
            print(f"[{ds} s{seed}] {v:15s} test {info[v]['test']:.2f} NEC {info[v]['nec']:.2f} "
                  f"classes {info[v]['classes_with_pos']}/{nk} concept AUC vs true labels {info[v]['auc_true_labels_val']:.3f}", flush=True)
    logged = metrics["refit_variants"]["own_100"]["global_test"]
    assert abs(info["ours"]["test"] - logged) < 0.015, (info["ours"]["test"], logged)

    cal = {}
    for v in arms:
        a_tr = layer_concepts(Xtr, B["layers"][v])
        cal[v] = calib(a_tr, B["labels"][v])
        if v == "random":
            cal["random_dino"] = cal[v]
            cal[v] = (torch.quantile(a_tr, 0.95, dim=0), torch.quantile(a_tr, 0.05, dim=0))
    runs = {"ours": "ours", "central_true": "central_true", "class_shuffled": "class_shuffled", "random": "random", "random_dino": "random"}
    unit = lambda v: v / v.norm(dim=1, keepdim=True).clamp(min=1e-12)  # noqa: E731
    out = {"dataset": ds, "seed": seed, "layers": info, "judges": {}}
    for kind in args.judges:
        judge, vacc = train_judge(kind, Xtr, ytr, Xva, yva, nk)
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            right = judge(Xte).argmax(1) == yte
            for v in arms:
                right &= head(v, Xte).argmax(1) == yte
            ok = right.nonzero().squeeze(1)
            pool = len(ok)
            ok = ok[torch.randperm(pool)[:args.n]]
            h, y = Xte[ok], yte[ok]
            valid = torch.stack([(B["heads"][v][0] > 1e-5).any(1) for v in arms]).all(0).nonzero().squeeze(1)
            yp = draw(valid, y[:, None], g)
            yq = draw(valid, torch.stack([y, yp], 1), g)
            res = {"pool": pool, "n": len(y), "valid_targets": len(valid), "arms": {}}
            print(f"[{ds} s{seed}] {kind} judge: all-right pool {pool}, n {len(y)}, targets {len(valid)}/{nk}", flush=True)
            ours_nrm = {}
            for name, v in runs.items():
                L, (W, b) = B["layers"][v], B["heads"][v]
                on, off = cal[name]
                res["arms"][name] = {}
                for s in args.scales:
                    dh = concept_edit(h, y, yp, L, W, on, off, s)
                    nrm = dh.norm(dim=1, keepdim=True)
                    if name == "ours":
                        ours_nrm[s] = nrm
                    dirs = {"concepts": dh, "@ours norm": unit(dh) * ours_nrm[s], "random": unit(torch.randn_like(h)) * nrm,
                            "other class": concept_edit(h, y, yq, L, W, on, off, s)}
                    r = {"edit_norm_rel": float((nrm.squeeze(1) / h.norm(dim=1)).mean())}
                    for dn, dd in dirs.items():
                        lj = judge(h + dd)
                        jp = lj.argmax(1)
                        r[dn] = {"judge_top1": float((jp == yp).float().mean()),
                                 "judge_top5": float((lj.topk(min(5, nk), 1).indices == yp[:, None]).any(1).float().mean()),
                                 "judge_keeps_source": float((jp == y).float().mean()),
                                 "head_top1": float((head(v, h + dd).argmax(1) == yp).float().mean())}
                        if dn == "other class":
                            r[dn]["judge_to_edit_class"] = float((jp == yq).float().mean())
                    res["arms"][name][f"{s:g}"] = r
                    print(f"  {kind:6s} {name:15s} scale {s:<3g} |dh|/|h| {r['edit_norm_rel']:.2f} | judge->target "
                          f"{100 * r['concepts']['judge_top1']:5.1f} @ours norm {100 * r['@ours norm']['judge_top1']:5.1f} | "
                          f"head {100 * r['concepts']['head_top1']:5.1f} | random {100 * r['random']['judge_top1']:4.1f} | "
                          f"third ->target {100 * r['other class']['judge_top1']:4.1f} ->own {100 * r['other class']['judge_to_edit_class']:5.1f}", flush=True)
        out["judges"][kind] = dict(res, judge_val=vacc)
    json.dump(out, open(os.path.join(d, f"sanity_{ds}_s{seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
