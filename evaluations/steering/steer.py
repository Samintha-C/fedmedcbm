"""Steering one seed's head: is the model steerable through the concepts its explanations name?

Head: own_100 at NEC~5 from OURS_HEAD_SAVED (FedDualAvg at the tab:main lambda, then 100 rounds of
masked FedAvg on its own support), with that seed's CBL and normalization. Its test accuracy is
recomputed here and has to match the job's metrics.json before anything is steered.

For a test image of class y and a target y', the edit sets y''s positive-weight concepts to mu+ and
y''s negative-weight concepts and y's positive-only concepts to mu- (common.calib), by the minimal-norm
change to the backbone features; scale s moves each edited concept a fraction s of the way. An
independent judge (linear or MLP probe on backbone features, class labels only) reads the result.
Images: test images the judge and the head both get right (--images judge_right drops the head
condition). Targets and third classes are drawn from the classes with a positive-weight concept.
Controls, per image:
  random       a random direction at the size of that image's concept edit;
  other class  the edit built from a third class's explanation (judge_to_edit_class says where the
               judge goes instead).

usage: python3 -I steer.py --data <data root> --dataset c10 --seed 42 --judge linear
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, SPLIT, calib, concept_edit, draw, layer_concepts, load_seed, nec, train_judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data root (see common.py)")
    ap.add_argument("--out", default=None, help="results dir (default <data>/saved/results)")
    ap.add_argument("--dataset", required=True, choices=sorted(CACHE))
    ap.add_argument("--seed", type=int, required=True, choices=sorted(SPLIT))
    ap.add_argument("--judge", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--images", default="both_right", choices=["both_right", "judge_right"])
    args = ap.parse_args()
    ds, seed = args.dataset, args.seed
    L, heads, metrics, f, lab = load_seed(args.data, ds, seed)
    W, b = heads["own_100"]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = f["train"], f["val"], f["test"]
    nk = W.shape[0]
    head = lambda F: layer_concepts(F, L) @ W.T + b  # noqa: E731
    acc = 100 * float((head(Xte).argmax(1) == yte).float().mean())
    logged = metrics["refit_variants"]["own_100"]["global_test"]
    assert abs(acc - logged) < 0.015, (acc, logged)
    on, off = calib(layer_concepts(Xtr, L), lab)
    valid = ((W > 1e-5).sum(1) > 0).nonzero().squeeze(1)
    judge, vacc = train_judge(args.judge, Xtr, ytr, Xva, yva, nk)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        pred = judge(Xte).argmax(1)
        right = pred == yte
        if args.images == "both_right":
            right &= head(Xte).argmax(1) == yte
    ok = right.nonzero().squeeze(1)
    pool = len(ok)
    ok = ok[torch.randperm(pool)[:args.n]]
    h, y = Xte[ok], yte[ok]
    yp = draw(valid, y[:, None], g)
    yq = draw(valid, torch.stack([y, yp], 1), g)
    out = {"dataset": ds, "seed": seed, "judge": args.judge, "images": args.images, "head_test": acc, "nec": nec(W),
           "judge_val": vacc, "judge_test": float((pred == yte).float().mean()), "pool": pool, "n": len(y),
           "classes_with_pos": len(valid), "num_classes": nk, "scales": {}}
    print(f"[{ds} s{seed}] head {acc:.2f} NEC {out['nec']:.2f} | {args.judge} judge test {100 * out['judge_test']:.1f} | "
          f"{args.images} pool {pool}, n {len(y)} | classes with a positive concept {len(valid)}/{nk}", flush=True)
    unit = lambda v: v / v.norm(dim=1, keepdim=True).clamp(min=1e-12)  # noqa: E731
    with torch.no_grad():
        for s in args.scales:
            dh = concept_edit(h, y, yp, L, W, on, off, s)
            nrm = dh.norm(dim=1, keepdim=True)
            dirs = {"concepts": dh, "random": unit(torch.randn_like(h)) * nrm, "other class": concept_edit(h, y, yq, L, W, on, off, s)}
            r = {"edit_norm_rel": float((nrm.squeeze(1) / h.norm(dim=1)).mean())}
            for name, d in dirs.items():
                lj = judge(h + d)
                jp = lj.argmax(1)
                r[name] = {"judge_top1": float((jp == yp).float().mean()), "judge_keeps_source": float((jp == y).float().mean()),
                           "judge_top5": float((lj.topk(min(5, nk), 1).indices == yp[:, None]).any(1).float().mean()),
                           "head_top1": float((head(h + d).argmax(1) == yp).float().mean())}
                if name == "other class":
                    r[name]["judge_to_edit_class"] = float((jp == yq).float().mean())
            out["scales"][f"{s:g}"] = r
            c, rd, oc = r["concepts"], r["random"], r["other class"]
            print(f"  scale {s:g}: |dh|/|h| {r['edit_norm_rel']:.2f} | judge->target {100 * c['judge_top1']:5.1f} "
                  f"(top5 {100 * c['judge_top5']:5.1f}, keeps source {100 * c['judge_keeps_source']:5.1f}) head {100 * c['head_top1']:5.1f} | "
                  f"random {100 * rd['judge_top1']:4.1f} (keeps source {100 * rd['judge_keeps_source']:5.1f}) | "
                  f"third: ->target {100 * oc['judge_top1']:4.1f} ->own {100 * oc['judge_to_edit_class']:5.1f}", flush=True)
    d = args.out or os.path.join(args.data, "saved", "results")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"steer_{ds}_s{seed}_{args.judge}_{args.images}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
