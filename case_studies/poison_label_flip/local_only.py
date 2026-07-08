"""Local-only diagnostic: fit an INDEPENDENT dense final-layer head per client.

Unlike the convergence snapshot (which reads each client's last-round primal from
the federated feddualavg loop, heavily pulled toward consensus), this fits a fresh
head from scratch on each client's OWN normalized concept features + labels — no
aggregation, no federation, no L1. The poison manifests at full magnitude because
nothing averages client 0 back toward the honest clients.

Called after Phase 2 (features extracted + federated-normalized). Writes files with
the same names/format the diagnose.py pipeline expects, tagged "localonly":
    client_{i}_localonly.pt, global_localonly.pt   (each: {"weight", "bias"})
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _fit_head(feats, labels, num_concepts, num_classes, device,
              epochs, lr, weight_decay, batch_size):
    # Derive input width from the tensor so a concept-filtered CBL can't mismatch.
    head = nn.Linear(feats.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(feats, labels), batch_size=batch_size, shuffle=True)
    head.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = ce(head(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


def fit_local_only_heads(all_train_feats, all_train_labels, client_data_sizes,
                         num_concepts, num_classes, device, out_dir,
                         epochs=50, lr=1e-3, weight_decay=1e-4, batch_size=512):
    """Fit one standalone head per client (on its own data block) + a pooled global
    reference. all_train_feats/labels are the Phase-2 normalized concept features,
    concatenated in client order — sliced back by client_data_sizes."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Local-only diagnostic: fitting {len(client_data_sizes)} independent heads ===")

    offset = 0
    for i, n in enumerate(client_data_sizes):
        f = all_train_feats[offset:offset + n]
        y = all_train_labels[offset:offset + n]
        offset += n
        head = _fit_head(f, y, num_concepts, num_classes, device,
                         epochs, lr, weight_decay, batch_size)
        torch.save({"weight": head.weight.data.cpu(), "bias": head.bias.data.cpu()},
                   os.path.join(out_dir, f"client_{i}_localonly.pt"))
        print(f"[local-only] client {i}: standalone head fit on {n} samples")

    # Pooled global reference (all clients' data together).
    head_g = _fit_head(all_train_feats, all_train_labels, num_concepts, num_classes,
                       device, epochs, lr, weight_decay, batch_size)
    torch.save({"weight": head_g.weight.data.cpu(), "bias": head_g.bias.data.cpu()},
               os.path.join(out_dir, "global_localonly.pt"))
    print(f"[local-only] global: pooled head fit on {sum(client_data_sizes)} samples")
    print(f"[local-only] heads saved to {out_dir}")
