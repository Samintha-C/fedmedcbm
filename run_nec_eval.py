#!/usr/bin/env python3
import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from evaluations.sparse_utils import measure_acc
from glm_saga.elasticnet import IndexedTensorDataset
from evaluate_fed_cbm import load_fed_vlg_cbm, _is_vlg_checkpoint


def run_nec_eval(load_dir, device="cuda", nec_lam_max=0.01, nec_measure_level=(5, 10, 15, 20, 25, 30), saga_batch_size=512, saga_step_size=0.1, saga_n_iters=2000):
    if not _is_vlg_checkpoint(load_dir):
        raise ValueError(f"{load_dir} is not a VLG checkpoint (missing cbl.pt or final.pt)")
    model, args_dict = load_fed_vlg_cbm(load_dir, device)
    args = argparse.Namespace(**args_dict)
    num_concepts = args.num_concepts
    num_classes = args.num_classes
    concepts = open(os.path.join(load_dir, "concepts.txt")).read().split("\n")

    train_feats_path = os.path.join(load_dir, "train_concept_features.pt")
    if os.path.exists(train_feats_path):
        train_feats = torch.load(os.path.join(load_dir, "train_concept_features.pt"))
        train_labels = torch.load(os.path.join(load_dir, "train_concept_labels.pt"))
        val_feats = torch.load(os.path.join(load_dir, "val_concept_features.pt"))
        val_labels = torch.load(os.path.join(load_dir, "val_concept_labels.pt"))
        train_concept_loader = DataLoader(
            IndexedTensorDataset(train_feats, train_labels),
            batch_size=saga_batch_size,
            shuffle=True,
        )
        val_concept_loader = DataLoader(
            TensorDataset(val_feats, val_labels),
            batch_size=saga_batch_size,
            shuffle=False,
        )
    else:
        from data.concept_dataset_vlg import get_concept_dataloader
        from models.fed_vlgcbm import get_final_layer_dataset
        preprocess = model.backbone.preprocess
        train_cbl = get_concept_dataloader(
            args.dataset, "train", concepts, preprocess=preprocess,
            val_split=getattr(args, "val_split", 0.1), batch_size=getattr(args, "cbl_batch_size", 32),
            num_workers=getattr(args, "num_workers", 4), shuffle=True, use_allones=True, seed=getattr(args, "seed", 42),
        )
        val_cbl = get_concept_dataloader(
            args.dataset, "val", concepts, preprocess=preprocess,
            val_split=getattr(args, "val_split", 0.1), batch_size=getattr(args, "cbl_batch_size", 32),
            num_workers=getattr(args, "num_workers", 4), shuffle=False, use_allones=True, seed=getattr(args, "seed", 42),
        )
        train_concept_loader, val_concept_loader, _ = get_final_layer_dataset(
            model.backbone, model.cbl, train_cbl, val_cbl,
            save_dir=load_dir, load_dir=None,
            batch_size=saga_batch_size, device=device,
        )

    from data.concept_dataset_vlg import get_concept_dataloader
    test_cbl = get_concept_dataloader(
        args.dataset, "test", concepts, preprocess=model.backbone.preprocess,
        batch_size=getattr(args, "batch_size", 32), num_workers=getattr(args, "num_workers", 4),
        shuffle=False, use_allones=True,
    )

    test_feats, test_labels = [], []
    with torch.no_grad():
        for features, _, labels in tqdm(test_cbl):
            features = features.to(device)
            logits = model.normalization(model.cbl(model.backbone(features)))
            test_feats.append(logits.cpu())
            test_labels.append(labels)
    test_feats = torch.cat(test_feats, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    test_concept_loader = DataLoader(TensorDataset(test_feats, test_labels), batch_size=saga_batch_size, shuffle=False)

    path, truncated_weights, _ = measure_acc(
        num_concepts, num_classes, len(train_concept_loader.dataset),
        train_concept_loader, val_concept_loader, test_concept_loader,
        saga_step_size=saga_step_size, saga_n_iters=saga_n_iters,
        device=device, max_lam=nec_lam_max, measure_level=nec_measure_level,
    )
    sparsity_list = [(p["weight"].abs() > 1e-5).float().mean().item() for p in path]
    nec_col = [num_concepts * s for s in sparsity_list]
    acc_col = [p["metrics"]["acc_test"] for p in path]
    pd.DataFrame({"NEC": nec_col, "Accuracy": acc_col}).to_csv(os.path.join(load_dir, "metrics.csv"), index=False)
    for nec_val, (W, b) in truncated_weights.items():
        torch.save(W, os.path.join(load_dir, f"W_g@NEC={nec_val:d}.pt"))
        torch.save(b, os.path.join(load_dir, f"b_g@NEC={nec_val:d}.pt"))
    return truncated_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nec_lam_max", type=float, default=0.01)
    parser.add_argument("--nec_measure_level", type=str, default="5,10,15,20,25,30")
    parser.add_argument("--saga_batch_size", type=int, default=512)
    parser.add_argument("--saga_step_size", type=float, default=0.1)
    parser.add_argument("--saga_n_iters", type=int, default=2000)
    args = parser.parse_args()
    nec_measure_level = tuple(int(x) for x in args.nec_measure_level.split(","))
    run_nec_eval(
        args.load_dir, device=args.device, nec_lam_max=args.nec_lam_max,
        nec_measure_level=nec_measure_level, saga_batch_size=args.saga_batch_size,
        saga_step_size=args.saga_step_size, saga_n_iters=args.saga_n_iters,
    )


if __name__ == "__main__":
    main()
