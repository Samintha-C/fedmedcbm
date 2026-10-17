"""FedCBM baseline on our data, under our federated protocol.

Same frozen backbone, same train/val split, same client partition, same DINO
concept labels and the same concept filter as a main_fed.py run with the same
--dataset/--seed/--iid/--alpha flags. Every data step below calls the function
train_vlg.py calls, in the same order; nothing in the main pipeline is changed.
The FedCBM math is in core.py (checked against the F-CM authors' code by
selftest.py).

Stage A depends only on the training rows, which every partition covers in
full, so it runs once per seed and is shared by all the partitions requested.

Usage (flags as in naut/gen_seed_main.py):
  python baselines/fedcbm_reimpl/run.py --dataset cifar100 \
      --concept_file concept_files/cifar100_filtered.txt \
      --backbone resnet50 --feature_layer layer4 --val_split 0.1 \
      --seed 42 --partitions iid dirichlet \
      --annotation_dir /sc-rwx-vol/fedmedcbm/annotations \
      --annotation_cache_dir /sc-rwx-vol/fedmedcbm/annotation_cache \
      --save_dir /tmp/out
"""

import argparse
import datetime
import gc
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_FED_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_FED_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import core  # noqa: E402
import data.data_utils as data_utils  # noqa: E402
from data.concept_dataset_vlg import DinoConceptDataset  # noqa: E402
from data.data_utils import get_classes, get_data, split_dataset_for_federated  # noqa: E402
from models.fed_vlgcbm import Backbone, BackboneCLIP  # noqa: E402
from training_utils import get_preprocess  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser()
    # data / backbone: same names and meaning as main_fed.py
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--concept_file", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--clip_name", default="ViT-B/16")
    ap.add_argument("--use_clip_penultimate", action="store_true")
    ap.add_argument("--feature_layer", default="layer4")
    ap.add_argument("--num_clients", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--annotation_dir", required=True)
    ap.add_argument("--annotation_cache_dir", required=True)
    ap.add_argument("--dino_confidence_threshold", type=float, default=0.10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--extract_batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--partitions", nargs="+", default=["iid", "dirichlet"],
                    choices=["iid", "dirichlet"])
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="heads scripts: replace the dirichlet partition by one per alpha, named a<alpha>")
    # FedCBM: defaults are the F-CM reference config (conf/learning/FedCBM.yaml,
    # trainer.max_epochs 200)
    ap.add_argument("--svm_rounds", type=int, default=200)
    ap.add_argument("--svm_lr", type=float, nargs="+", default=[0.01],
                    help="one value = fixed; several = grid, chosen on validation")
    ap.add_argument("--svm_l2", type=float, default=1e-4)
    # Stronger variant (all off = the reference). See core.train_cav_bank.
    ap.add_argument("--svm_balanced", action="store_true")
    ap.add_argument("--svm_bias", action="store_true")
    ap.add_argument("--svm_standardize", nargs="+", choices=["no", "yes"], default=["no"],
                    help="one value = fixed; both = part of the validation grid")
    ap.add_argument("--variant_name", default="faithful")
    ap.add_argument("--head_C", type=float, default=1.0)
    ap.add_argument("--head_max_iter", type=int, default=200)
    ap.add_argument("--head_tol", type=float, default=1e-4)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--resident", choices=["auto", "yes", "no"], default="auto",
                    help="keep train features + concept labels on the GPU during Stage A")
    # optional: compare against the F-CM reference on real data (small datasets only)
    ap.add_argument("--fcm_repo", default=None)
    ap.add_argument("--reference_check_concepts", type=int, default=0)
    return ap


def parse():
    return build_parser().parse_args()


def _step(label, t0):
    print(f"[TIMING] {label}: {time.time() - t0:.1f}s", flush=True)
    return time.time()


def build_backbone(args, device):
    preprocess = get_preprocess(args.backbone)
    if args.backbone.startswith("clip_"):
        bb = BackboneCLIP(args.backbone, use_penultimate=args.use_clip_penultimate, device=str(device))
    else:
        bb = Backbone(args.backbone, args.feature_layer, str(device))
    for p in bb.parameters():
        p.requires_grad = False
    bb.eval()
    return bb, preprocess


def load_data(args, device):
    """Mirror of train_vlg.py lines ~192-302 and ~454-463, plus its test loader."""
    t = time.time()
    backbone, preprocess = build_backbone(args, device)
    classes = get_classes(args.dataset)
    concepts = data_utils.get_concepts(args.concept_file, None)

    full_train = get_data(f"{args.dataset}_train", preprocess=None)
    n_val = int(args.val_split * len(full_train))
    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [len(full_train) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    assert not (set(train_ds.indices) & set(val_ds.indices))
    t = _step("dataset + train/val split", t)

    kw = dict(annotation_dir=args.annotation_dir, confidence_threshold=args.dino_confidence_threshold,
              preprocess=preprocess, cache_dir=args.annotation_cache_dir)
    tr = DinoConceptDataset(args.dataset, train_ds, concepts, split_suffix="train", **kw)
    va = DinoConceptDataset(args.dataset, val_ds, concepts, split_suffix="train", **kw)
    t = _step("annotations (train, val)", t)

    # VLG-CBM dynamic filter, exactly as train_vlg.py: drop zero-count concepts.
    counts = torch.zeros(len(concepts), dtype=torch.long)
    for s in range(0, tr._annotation_cache.shape[0], 65536):
        counts += tr._annotation_cache[s:s + 65536].sum(dim=0).long()
    keep = counts > 0
    if not bool(keep.all()):
        concepts = [c for c, k in zip(concepts, keep.tolist()) if k]
        tr._annotation_cache = tr._annotation_cache[:, keep]
        va._annotation_cache = va._annotation_cache[:, keep]
        tr.concepts = va.concepts = concepts
        tr.concept_set = va.concept_set = set(concepts)
        gc.collect()
    print(f"[concepts] {len(concepts)} after the zero-count filter")

    xb = min(1024, args.extract_batch_size)
    for ds in (tr, va):
        ds.populate_feature_cache(backbone=backbone, device=str(device), backbone_name=args.backbone,
                                  batch_size=xb, num_workers=args.num_workers, prefetch_factor=4)
    t = _step("features (train, val)", t)

    try:
        test_data = get_data(f"{args.dataset}_test", None)
    except Exception:
        test_data = get_data(f"{args.dataset}_val", None)
    te = DinoConceptDataset(args.dataset, test_data, concepts, split_suffix="val", **kw)
    te.populate_feature_cache(backbone=backbone, device=str(device), backbone_name=args.backbone,
                              batch_size=xb, num_workers=args.num_workers, prefetch_factor=4)
    t = _step("test annotations + features", t)
    del backbone
    torch.cuda.empty_cache()
    return dict(classes=classes, concepts=concepts, train_ds=train_ds, val_ds=val_ds,
                tr=tr, va=va, te=te)


def reference_check(args, d, rows, stage_a, lr, out):
    """Real-data check against the F-CM reference on a few concepts (Stage A).
    Only meaningful for the faithful configuration (the reference has no bias,
    weighting or standardization)."""
    import numpy as np
    from selftest import load_reference
    ref = load_reference(args.fcm_repo)
    tr = d["tr"]
    trainable = stage_a["trainable"].nonzero(as_tuple=True)[0]
    pos = stage_a["pos_counts"][trainable]
    pick = trainable[torch.argsort(pos, descending=True)][:args.reference_check_concepts]
    h = tr._feature_cache[rows].float().numpy()
    res = {}
    for j in pick.tolist():
        y = np.where(tr._annotation_cache[rows, j].numpy(), 1.0, -1.0).astype(np.float32)
        w_ref = ref._train_single_cav([(h, y, h.shape[0])], h.shape[1], args.svm_rounds, 1,
                                      lr, args.svm_l2).w
        w = stage_a["W"][j].cpu().numpy()
        res[d["concepts"][j]] = float(np.abs(w - w_ref).max() / max(np.abs(w_ref).max(), 1e-12))
        print(f"[reference check] {d['concepts'][j]!r}: max rel |w - w_ref| = {res[d['concepts'][j]]:.2e}")
    out["reference_check_stage_a"] = res


def arms(args):
    """(name, iid, alpha) per partition the heads scripts evaluate. With --alphas,
    "dirichlet" becomes one partition per alpha, named a<alpha> (a0.05, a1, ...)."""
    out = []
    for p in args.partitions:
        if p == "dirichlet" and args.alphas:
            out += [(f"a{a:g}", False, a) for a in args.alphas]
        else:
            out.append((p, p == "iid", args.alpha))
    return out


def proj(feats, stage_a, args, rows=None):
    return core.project(feats, stage_a["W"], rows=rows, chunk=args.chunk,
                        b=stage_a["b"], stats=stage_a["stats"])


def eval_partition(p, ci, vi, stage_a, d, args, z_val, y_val, z_test, y_test):
    """Fit one private head per client and score it four ways: its own train
    and val split, the full (global) val set, the global test set, and its own
    class-matched test split."""
    tr, va, te = d["tr"], d["va"], d["te"]
    heads, rec, train_labels = [], {"clients": []}, []
    for m in range(args.num_clients):
        idx = torch.as_tensor(ci[m], dtype=torch.long)
        y = tr._target_cache[idx]
        train_labels.append(y)
        z = proj(tr._feature_cache, stage_a, args, rows=idx)
        th = time.time()
        head = core.fit_head(z, y, C=args.head_C, max_iter=max(50, args.head_max_iter), tol=args.head_tol)
        vidx = torch.as_tensor(vi[m], dtype=torch.long)
        zv = proj(va._feature_cache, stage_a, args, rows=vidx)
        rec["clients"].append({
            "n_train": int(idx.numel()), "n_classes_held": int(torch.unique(y).numel()),
            "train_acc": core.accuracy(head, z, y),
            "val_acc": core.accuracy(head, zv, va._target_cache[vidx]),
            "global_val_acc": core.accuracy(head, z_val, y_val),
            "global_test_acc": core.accuracy(head, z_test, y_test),
            "head_fit_seconds": round(time.time() - th, 1)})
        heads.append(head)
        del z, zv
        torch.cuda.empty_cache()
        print(f"[{p}] client {m}: {rec['clients'][-1]}", flush=True)

    mean = lambda k: sum(c[k] for c in rec["clients"]) / len(rec["clients"])
    # Reference protocol: every head on the one global test set, mean over clients.
    rec["global_test_mean_client_acc"] = mean("global_test_acc")
    rec["global_val_mean_client_acc"] = mean("global_val_acc")
    # Personalized protocol: each head on its own class-matched test split.
    split = core.personalized_test_split(te._target_cache, train_labels, len(d["classes"]), seed=args.seed)
    correct, total, per = 0, 0, []
    for m, ix in enumerate(split):
        ix = ix.to(z_test.device)
        a = core.accuracy(heads[m], z_test[ix], y_test[ix]) if ix.numel() else float("nan")
        per.append(a)
        rec["clients"][m]["personal_test_acc"] = a
        rec["clients"][m]["n_personal_test"] = int(ix.numel())
        if ix.numel():
            correct += a * ix.numel()
            total += ix.numel()
    rec["personalized_test_acc_weighted"] = correct / max(total, 1)
    rec["personalized_test_acc_macro"] = sum(x for x in per if x == x) / max(sum(x == x for x in per), 1)
    print(f"[{p}] global-test mean client acc {rec['global_test_mean_client_acc']:.4f}  "
          f"personalized acc {rec['personalized_test_acc_weighted']:.4f}  "
          f"(global-val {rec['global_val_mean_client_acc']:.4f})", flush=True)
    return rec


def main():
    args = parse()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False  # keep Stage A in true fp32, like the reference
    tag = datetime.datetime.now().strftime("%b%d-%H:%M").lower()
    save_dir = os.path.join(args.save_dir, f"fedcbm-{args.variant_name}-{args.dataset}-s{args.seed}-{tag}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    t0 = time.time()
    faithful = not (args.svm_balanced or args.svm_bias or "yes" in args.svm_standardize)

    d = load_data(args, device)
    tr, va, te = d["tr"], d["va"], d["te"]
    out = {"dataset": args.dataset, "seed": args.seed, "num_clients": args.num_clients,
           "num_concepts": len(d["concepts"]), "num_classes": len(d["classes"]),
           "variant": args.variant_name, "faithful_to_reference": faithful,
           "method": "FedCBM (Yang & Long 2023), F-CM reference reimplementation, torch port"
                     + ("" if faithful else "; stronger variant (see fedcbm block)"),
           "fedcbm": {k: getattr(args, k) for k in
                      ("svm_rounds", "svm_lr", "svm_l2", "svm_balanced", "svm_bias", "svm_standardize",
                       "head_C", "head_max_iter", "head_tol")},
           "partitions": {}}

    # ── client partitions (same call and seed as train_vlg.py) ────────────────
    parts = {}
    for p in args.partitions:
        ci = split_dataset_for_federated(d["train_ds"], args.num_clients, iid=(p == "iid"),
                                         alpha=args.alpha, seed=args.seed)
        vi = split_dataset_for_federated(d["val_ds"], args.num_clients, iid=(p == "iid"),
                                         alpha=args.alpha, seed=args.seed)
        held = torch.cat([torch.as_tensor(c, dtype=torch.long) for c in ci])
        assert held.numel() == torch.unique(held).numel(), "clients overlap"
        parts[p] = (ci, vi, held)
        print(f"[partition] {p}: client sizes {[len(c) for c in ci]}, "
              f"{held.numel()}/{len(d['train_ds'])} train rows held")

    # Stage A uses the rows the clients hold. Every partition must hold the same
    # rows for the shared bank to be valid; if not, fail loudly rather than mix.
    helds = [torch.sort(v[2]).values for v in parts.values()]
    assert all(torch.equal(helds[0], h) for h in helds), "partitions hold different rows"
    rows = None if helds[0].numel() == len(d["train_ds"]) else helds[0]

    feats, conc = tr._feature_cache, tr._annotation_cache
    need = feats.numel() * feats.element_size() + conc.numel() * conc.element_size()
    resident = args.resident == "yes" or (
        args.resident == "auto" and device.type == "cuda"
        and need < 0.45 * torch.cuda.get_device_properties(device).total_memory)
    if resident:
        feats, conc = feats.to(device), conc.to(device)
    print(f"[Stage A] N={feats.shape[0] if rows is None else rows.numel()} D={feats.shape[1]} "
          f"C={conc.shape[1]}  resident_on_gpu={resident} ({need / 1024**3:.1f} GiB)")
    stats = core.feature_stats(tr._feature_cache, rows) if "yes" in args.svm_standardize else None

    # ── Stage A grid, chosen on validation ────────────────────────────────────
    # Selection: mean over clients of each head's accuracy on the full val set,
    # on the IID partition (where the global and per-client tasks coincide).
    # Only that partition's heads are fit per configuration; the others are fit
    # once, for the winner. One configuration = no selection (faithful default).
    sel_part = "iid" if "iid" in parts else next(iter(parts))
    y_val, y_test = va._target_cache.to(device), te._target_cache.to(device)
    grid, best = [], None
    for std in args.svm_standardize:
        for lr in args.svm_lr:
            cfg = {"lr": lr, "standardize": std == "yes"}
            t = time.time()
            stage_a = core.train_cav_bank(feats, conc, rows=rows, rounds=args.svm_rounds, lr=lr,
                                          l2=args.svm_l2, device=str(device), chunk=args.chunk,
                                          balanced=args.svm_balanced, bias=args.svm_bias,
                                          stats=stats if cfg["standardize"] else None)
            t = _step(f"Stage A {cfg}", t)
            z_val = proj(va._feature_cache, stage_a, args)
            z_test = proj(te._feature_cache, stage_a, args)
            cval = core.concept_accuracy(z_val, va._annotation_cache, stage_a["trainable"], chunk=args.chunk)
            ci, vi, _ = parts[sel_part]
            rec = eval_partition(sel_part, ci, vi, stage_a, d, args, z_val, y_val, z_test, y_test)
            entry = {**cfg, "selection_partition": sel_part,
                     "global_val_mean_client_acc": rec["global_val_mean_client_acc"],
                     "concept_balanced_accuracy_val": cval["concept_balanced_accuracy_mean"],
                     "stage_a": {k: stage_a[k] for k in ("final_objective_mean", "final_hinge_mean",
                                                         "final_active_fraction_mean")}}
            grid.append(entry)
            print(f"[grid] {entry}", flush=True)
            if best is None or rec["global_val_mean_client_acc"] > best[0]:
                best = (rec["global_val_mean_client_acc"], cfg, stage_a, rec, z_val, z_test)
            else:
                del stage_a, z_val, z_test
            torch.cuda.empty_cache()
    del feats, conc
    torch.cuda.empty_cache()
    _, cfg, stage_a, rec, z_val, z_test = best
    out["selected"] = cfg
    out["grid"] = grid
    print(f"[grid] selected {cfg}", flush=True)

    out["stage_a"] = {k: stage_a[k] for k in ("final_objective_mean", "final_hinge_mean",
                                              "final_active_fraction_mean", "history")}
    out["stage_a"]["trained_concepts"] = int(stage_a["trainable"].sum())
    torch.save({"W": stage_a["W"].cpu().half(),
                "b": None if stage_a["b"] is None else stage_a["b"].cpu(),
                "stats": stage_a["stats"], "concepts": d["concepts"], "trainable": stage_a["trainable"],
                "selected": cfg}, os.path.join(save_dir, "cav_bank.pt"))
    if args.fcm_repo and args.reference_check_concepts > 0:
        if faithful:
            reference_check(args, d, rows if rows is not None else torch.arange(len(d["train_ds"])),
                            stage_a, cfg["lr"], out)
        else:
            print("[reference check] skipped: the stronger variant has no reference counterpart")
    out["concept_metrics_test"] = core.concept_accuracy(z_test, te._annotation_cache,
                                                        stage_a["trainable"], chunk=args.chunk)
    print(f"[concepts] test {out['concept_metrics_test']}")

    # ── Stage B for every partition (the selection partition is already done) ─
    for p, (ci, vi, _) in parts.items():
        out["partitions"][p] = rec if p == sel_part else eval_partition(
            p, ci, vi, stage_a, d, args, z_val, y_val, z_test, y_test)

    out["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {save_dir}")


if __name__ == "__main__":
    main()
