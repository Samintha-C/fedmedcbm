"""Does a concept layer know about the image, or only about the class? For the layers sanity.py
built, on test images with the true DINO test labels (never used to train or select a layer): concept
AUC over all images, and within one class at a time, so class identity cannot help. Pairs (class,
concept) need at least 5 images with the label and 5 without. "support" restricts to the pairs where
the concept has positive weight for that class in the layer's own head.

usage: python3 -I within_class_auc.py --data <data root> [--sanity <sanity dir>]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, SPLIT, TEST, TEST_ANN, layer_concepts  # noqa: E402


def auc_cols(z, lab):
    """Rank AUC per column, NaN where a column has fewer than 5 positives or negatives."""
    r = z.argsort(0).argsort(0).double() + 1
    pos, neg = lab.sum(0).double(), (~lab).sum(0).double()
    a = ((r * lab).sum(0) - pos * (pos + 1) / 2) / (pos * neg).clamp(min=1)
    a[(pos < 5) | (neg < 5)] = float("nan")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data root (see common.py)")
    ap.add_argument("--sanity", default=None, help="sanity.py's output dir (default <data>/saved/sanity)")
    args = ap.parse_args()
    sanity = args.sanity or os.path.join(args.data, "saved", "sanity")
    out = {}
    for ds in ("c10", "c100"):
        c = torch.load(os.path.join(args.data, TEST[ds]), map_location="cpu")
        Xte, yte = c["features"].float(), c["targets"].long()
        full = torch.load(os.path.join(args.data, TEST_ANN[ds]), map_location="cpu")
        for seed in SPLIT:
            keep = torch.load(os.path.join(args.data, CACHE[ds][1].format(SPLIT[seed][0])), map_location="cpu").sum(0) > 0
            lab = full[:, keep].bool()
            B = torch.load(os.path.join(sanity, f"layers_{ds}_s{seed}.pt"), weights_only=False)
            for v in ("ours", "central_true", "class_shuffled", "random"):
                z = layer_concepts(Xte, B["layers"][v])
                W = B["heads"][v][0]
                glob = auc_cols(z, lab)
                per = torch.stack([auc_cols(z[yte == k], lab[yte == k]) for k in range(W.shape[0])])
                sup = (W > 1e-5) & ~per.isnan()
                rec = {"global": float(glob[~glob.isnan()].mean()), "within": float(per[~per.isnan()].mean()),
                       "within_pairs": int((~per.isnan()).sum()),
                       "within_support": float(per[sup].mean()) if sup.any() else None, "support_pairs": int(sup.sum())}
                out.setdefault(ds, {}).setdefault(v, {})[str(seed)] = rec
                ws = "n/a" if rec["within_support"] is None else f"{rec['within_support']:.3f}"
                print(f"{ds} s{seed} {v:15s} global {rec['global']:.3f} | within class {rec['within']:.3f} "
                      f"({rec['within_pairs']} pairs) | within, head support {ws} ({rec['support_pairs']} pairs)", flush=True)
    json.dump(out, open(os.path.join(sanity, "within_class_auc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
