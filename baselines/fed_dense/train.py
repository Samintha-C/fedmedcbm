"""Federated dense baseline: the standard CBM pipeline with the CBL removed.

  Phase 1: SKIPPED (no concept layer)
  Phase 2: federated normalization + feature extraction (raw backbone embeddings)
  Phase 3: federated dense head via genuine FedAvg (backbone_dim -> num_classes)

Phase 2 normalization and Phase 3 FedAvg are faithful ports of the corresponding
blocks in train_vlg.py (the fedavg_thresh federated loop, with the thresholding /
FedMask steps removed). Backbone is frozen throughout, matching the CBM runs.
"""

import datetime
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from data.data_utils import (
    get_data, get_classes, split_dataset_for_federated, print_client_distribution,
)
from models.fed_vlgcbm import Backbone, BackboneCLIP, NormalizationLayer
from training_utils import get_preprocess, save_args, save_metrics_txt, save_training_metrics
from evaluate_fed_cbm import get_accuracy_cbm, get_per_class_accuracy_vlg
from baselines.fed_dense.model import DenseClassifier


def _build_backbone(args, device):
    if args.backbone.startswith("clip_"):
        preprocess = get_preprocess(args.backbone)
        backbone = BackboneCLIP(
            args.backbone,
            use_penultimate=getattr(args, "use_clip_penultimate", True),
            device=str(device),
        )
    else:
        preprocess = get_preprocess(args.backbone)
        backbone = Backbone(args.backbone, getattr(args, "feature_layer", "layer4"), str(device))
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return backbone, preprocess


@torch.no_grad()
def _extract_features(backbone, dataset, device, batch_size, num_workers):
    """Run frozen backbone over a dataset, return (feats[N,D] cpu, labels[N] cpu)."""
    feats, labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    for images, y in loader:
        emb = backbone(images.to(device)).cpu()
        feats.append(emb)
        labels.append(y)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def train_fed_dense(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    classes = get_classes(args.dataset)
    num_classes = len(classes)
    args.num_classes = num_classes

    date_tag = datetime.datetime.now().strftime("%b%d-%H:%M").lower()
    run_name = f"c{args.num_clients}r{args.num_rounds}-dense-{args.dataset}-fedavg-{date_tag}"
    save_dir = os.path.join(args.save_dir, "fully_trained", run_name)
    os.makedirs(save_dir, exist_ok=True)
    save_args(save_dir, args)

    # ── Backbone (frozen) ────────────────────────────────────────────────────
    backbone, preprocess = _build_backbone(args, device)
    backbone_dim = backbone.output_dim
    print(f"[dense] backbone={args.backbone} frozen, feature_dim={backbone_dim}")

    # ── Data + identical train/val split + federated partition ───────────────
    full_train = get_data(f"{args.dataset}_train", preprocess=preprocess)
    val_split = getattr(args, "val_split", 0.1)
    n_val = int(val_split * len(full_train))
    n_train = len(full_train) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Split train dataset: {n_train} train, {n_val} val (val_split={val_split})")

    client_indices = split_dataset_for_federated(
        train_dataset, args.num_clients,
        iid=getattr(args, "iid", False), alpha=args.alpha, seed=args.seed,
    )
    print_client_distribution(train_dataset, client_indices, num_classes=num_classes)

    # ── Phase 2a: per-client feature extraction (frozen backbone) ────────────
    bs = getattr(args, "extract_batch_size", 256)
    nw = getattr(args, "num_workers", 2)
    client_feats, client_labels, client_data_sizes = [], [], []
    for i in range(args.num_clients):
        sub = Subset(train_dataset, client_indices[i])
        f, y = _extract_features(backbone, sub, device, bs, nw)
        client_feats.append(f)
        client_labels.append(y)
        client_data_sizes.append(len(sub))
        print(f"[dense] client {i}: extracted {len(sub)} features")
    total_samples = sum(client_data_sizes)
    client_weights = [n / total_samples for n in client_data_sizes]

    val_feats, val_labels = _extract_features(backbone, val_dataset, device, bs, nw)

    # ── Phase 2b: federated normalization (parallel statistics) ──────────────
    # Each client contributes local sum/sq-sum/count; server aggregates.
    g_sum = torch.zeros(backbone_dim)
    g_sq = torch.zeros(backbone_dim)
    g_n = 0
    for f in client_feats:
        g_sum += f.sum(dim=0)
        g_sq += (f ** 2).sum(dim=0)
        g_n += f.shape[0]
    g_mean = g_sum / g_n
    g_var = (g_sq / g_n) - (g_mean ** 2)
    g_std = g_var.clamp(min=1e-8).sqrt()
    norm_layer = NormalizationLayer(g_mean, g_std, device=str(device))

    def _normalize(t):
        return (t - g_mean) / g_std

    saga_bs = getattr(args, "saga_batch_size", 512)
    client_loaders = [
        DataLoader(TensorDataset(_normalize(client_feats[i]), client_labels[i]),
                   batch_size=saga_bs, shuffle=True)
        for i in range(args.num_clients)
    ]
    val_loader = DataLoader(TensorDataset(_normalize(val_feats), val_labels),
                            batch_size=saga_bs, shuffle=False)

    # ── Phase 3: federated dense head (genuine FedAvg) ───────────────────────
    head = nn.Linear(backbone_dim, num_classes).to(device)
    client_heads = [nn.Linear(backbone_dim, num_classes).to(device)
                    for _ in range(args.num_clients)]
    ce_loss = nn.CrossEntropyLoss()

    final_rounds = getattr(args, "final_rounds", 200)
    final_epochs = getattr(args, "final_epochs", 3)
    final_lr = getattr(args, "final_lr", 1e-3)
    final_wd = getattr(args, "final_weight_decay", 1e-4)
    print(f"[dense] FedAvg head: lr={final_lr} weight_decay={final_wd}")

    metrics = {"rounds": [], "avg_client_loss": [], "val_accuracy": [], "best_val_accuracy": []}
    best_val_acc = 0.0
    best_head_state = None

    for round_num in range(final_rounds):
        round_losses = []
        for i in range(args.num_clients):
            client_heads[i].load_state_dict(head.state_dict())
            client_heads[i].train()
            opt = torch.optim.Adam(client_heads[i].parameters(), lr=final_lr, weight_decay=final_wd)
            epoch_loss, n_batches = 0.0, 0
            for _ in range(final_epochs):
                for feats, labels in client_loaders[i]:
                    feats, labels = feats.to(device), labels.to(device)
                    loss = ce_loss(client_heads[i](feats), labels)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()
                    n_batches += 1
            round_losses.append(epoch_loss / max(n_batches, 1))

        # Server: weighted average of client heads (FedAvg)
        global_state = {}
        for key in head.state_dict().keys():
            param = client_heads[0].state_dict()[key]
            if param.dtype.is_floating_point:
                global_state[key] = torch.zeros_like(param)
                for i in range(args.num_clients):
                    global_state[key] += client_weights[i] * client_heads[i].state_dict()[key]
            else:
                global_state[key] = param.clone()
        head.load_state_dict(global_state)

        # Eval on pooled val (== size-weighted per-client accuracy)
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                preds = head(feats).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(total, 1)

        avg_loss = sum(round_losses) / len(round_losses)
        if (round_num + 1) % 10 == 0 or round_num == final_rounds - 1:
            print(f"=== Round {round_num + 1}/{final_rounds} === avg_loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}

        metrics["rounds"].append(round_num + 1)
        metrics["avg_client_loss"].append(avg_loss)
        metrics["val_accuracy"].append(float(val_acc))
        metrics["best_val_accuracy"].append(float(best_val_acc))

    if best_head_state is not None:
        head.load_state_dict(best_head_state)
    print(f"[dense] best val accuracy: {best_val_acc:.4f}")

    # ── Final evaluation on test set ─────────────────────────────────────────
    model = DenseClassifier(backbone, normalization=norm_layer, head=head).to(device)
    model.eval()
    test_data = get_data(f"{args.dataset}_val", preprocess=preprocess)
    print("\n=== Test evaluation ===")
    accuracy = get_accuracy_cbm(model, test_data, device,
                                batch_size=getattr(args, "batch_size", 250),
                                num_workers=nw)
    per_class = get_per_class_accuracy_vlg(model, test_data, device, classes,
                                           batch_size=getattr(args, "batch_size", 250),
                                           num_workers=nw)
    print(f"Test accuracy: {float(accuracy):.4f}  (overall {per_class['Overall accuracy']}%)")

    # ── Save (metrics.txt schema matches CBM runs; no sparsity block) ────────
    metrics_txt = {
        "per_class_accuracies": per_class,
        "metrics": {"test_accuracy": float(accuracy)},
        "method": "fed_dense",
        "backbone": args.backbone,
        "feature_dim": backbone_dim,
        "num_clients": args.num_clients,
        "final_rounds": final_rounds,
        "final_epochs": final_epochs,
        "final_lr": final_lr,
        "final_weight_decay": final_wd,
    }
    save_metrics_txt(save_dir, metrics_txt)
    save_training_metrics(save_dir, metrics)
    torch.save(head.state_dict(), os.path.join(save_dir, "head.pt"))
    norm_layer.save_model(save_dir)
    print(f"\n[dense] Saved to {save_dir}")
    return save_dir
