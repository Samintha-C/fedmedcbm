import json
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR

import clip

from data import data_utils
from glm_saga.elasticnet import glm_saga, IndexedTensorDataset


class Backbone(nn.Module):
    feature_vals = {}

    def __init__(self, backbone_name: str, feature_layer: str, device: str = "cuda"):
        super().__init__()
        target_model, target_preprocess = data_utils.get_target_model(backbone_name, device)

        def hook(module, input, output):
            self.feature_vals[output.device] = output

        getattr(target_model, feature_layer).register_forward_hook(hook)
        self.backbone = target_model
        self.preprocess = target_preprocess
        self.output_dim = data_utils.BACKBONE_ENCODING_DIMENSION[backbone_name]

    def forward(self, x):
        out = self.backbone(x)
        return self.feature_vals[out.device].mean(dim=[2, 3])

    def save_model(self, save_dir):
        torch.save(self.backbone.state_dict(), os.path.join(save_dir, "backbone.pt"))


class BackboneCLIP(nn.Module):
    def __init__(self, backbone_name: str, use_penultimate: bool = True, device: str = "cuda"):
        super().__init__()
        target_model, target_preprocess = clip.load(backbone_name[5:], device=device)
        if use_penultimate:
            target_model = target_model.visual
            N = target_model.attnpool.c_proj.in_features
            identity = nn.Linear(N, N, dtype=torch.float16, device=device)
            nn.init.zeros_(identity.bias)
            identity.weight.data.copy_(torch.eye(N))
            target_model.attnpool.c_proj = identity
            self.output_dim = data_utils.BACKBONE_ENCODING_DIMENSION.get(
                f"{backbone_name}_penultimate", data_utils.BACKBONE_ENCODING_DIMENSION.get(backbone_name, 768)
            )
        else:
            target_model = target_model.visual
            self.output_dim = data_utils.BACKBONE_ENCODING_DIMENSION.get(backbone_name, 512)
        self.backbone = target_model.float()
        self.preprocess = target_preprocess

    def forward(self, x):
        return self.backbone(x).float()

    def save_model(self, save_dir):
        torch.save(self.backbone.state_dict(), os.path.join(save_dir, "backbone.pt"))


class ConceptLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_hidden: int = 0,
                 bias: bool = True, device: str = "cuda"):
        super().__init__()
        layers = [nn.Linear(in_features, out_features, bias=bias)]
        for _ in range(num_hidden):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(out_features, out_features, bias=bias))
        self.model = nn.Sequential(*layers).to(device)
        self.out_features = out_features

    def forward(self, x):
        return self.model(x)

    def save_model(self, save_dir):
        torch.save(self.state_dict(), os.path.join(save_dir, "cbl.pt"))


class NormalizationLayer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, device: str = "cuda"):
        super().__init__()
        self.register_buffer("mean", mean.to(device))
        self.register_buffer("std", std.to(device).clamp(min=1e-8))

    def forward(self, x):
        return (x - self.mean) / self.std

    def save_model(self, save_dir):
        torch.save(self.mean.cpu(), os.path.join(save_dir, "train_concept_features_mean.pt"))
        torch.save(self.std.cpu(), os.path.join(save_dir, "train_concept_features_std.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        mean = torch.load(os.path.join(load_path, "train_concept_features_mean.pt"), map_location=device)
        std = torch.load(os.path.join(load_path, "train_concept_features_std.pt"), map_location=device)
        return cls(mean, std, device=device)


class FinalLayer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, device: str = "cuda"):
        super().__init__(in_features, out_features, bias=True)
        self.to(device)

    def save_model(self, save_dir):
        torch.save(self.state_dict(), os.path.join(save_dir, "final.pt"))


class FedVLGCBM(nn.Module):
    def __init__(self, backbone, cbl, normalization=None, final_layer=None):
        super().__init__()
        self.backbone = backbone
        self.cbl = cbl
        self.normalization = normalization
        self.final_layer = final_layer

    def forward(self, x, return_concepts=False):
        h = self.backbone(x)
        c = self.cbl(h)
        if self.normalization is not None:
            c = self.normalization(c)
        if self.final_layer is not None:
            logits = self.final_layer(c)
            return (logits, c) if return_concepts else logits
        return c


def validate_cbl(backbone, cbl, val_loader, loss_fn, device="cuda"):
    val_loss = 0.0
    with torch.no_grad():
        for features, concept_one_hot, _ in val_loader:
            features = features.to(device)
            concept_one_hot = concept_one_hot.to(device)
            concept_logits = cbl(backbone(features))
            val_loss += loss_fn(concept_logits, concept_one_hot).item()
    return val_loss / len(val_loader)


def train_cbl(backbone, cbl, train_loader, val_loader, epochs, loss_fn, lr=1e-3, weight_decay=1e-5,
              device="cuda", finetune=False, optimizer_name="sgd", backbone_lr=1e-3):
    if optimizer_name == "sgd":
        opt = torch.optim.SGD(cbl.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        opt = torch.optim.Adam(cbl.parameters(), lr=lr, weight_decay=weight_decay)
    if finetune:
        opt.add_param_group({"params": backbone.parameters(), "lr": backbone_lr})

    best_val_loss = float("inf")
    best_cbl_state = None
    best_backbone_state = None

    for epoch in range(epochs):
        train_loss = 0.0
        for features, concept_one_hot, _ in train_loader:
            features = features.to(device)
            concept_one_hot = concept_one_hot.to(device)
            if finetune:
                backbone.train()
                embeddings = backbone(features)
            else:
                with torch.no_grad():
                    embeddings = backbone(features)
            concept_logits = cbl(embeddings)
            loss = loss_fn(concept_logits, concept_one_hot)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        backbone.eval()
        train_loss /= len(train_loader)
        val_loss = validate_cbl(backbone, cbl, val_loader, loss_fn, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_cbl_state = {k: v.cpu().clone() for k, v in cbl.state_dict().items()}
            best_backbone_state = {k: v.cpu().clone() for k, v in backbone.state_dict().items()}

    if best_cbl_state is not None:
        cbl.load_state_dict(best_cbl_state)
    if best_backbone_state is not None:
        backbone.load_state_dict(best_backbone_state)
    return cbl, backbone


def test_model(loader, backbone, cbl, normalization, final_layer, device="cuda"):
    correct = 0
    total = 0
    for features, _, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            logits = final_layer(normalization(cbl(backbone(features))))
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
    return correct / total


def per_class_accuracy(model, loader, classes, device="cuda"):
    correct = torch.zeros(len(classes)).to(device)
    total = torch.zeros(len(classes)).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for features, _, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1)
            for pred, target in zip(preds, targets):
                total[target] += 1
                if pred == target:
                    correct[target] += 1
    pca = (correct / total).nan_to_num_(nan=0.0)
    tot = total.sum()
    overall = (correct.sum() / tot).item() * 100.0 if tot.item() > 0 else 0.0
    return {
        "Per class accuracy": {classes[i]: f"{pca[i].item()*100.0:.2f}" for i in range(len(classes))},
        "Overall accuracy": f"{overall:.2f}",
        "Datapoints": f"{tot.item()}",
    }


def get_final_layer_dataset(backbone, cbl, train_loader, val_loader, save_dir, load_dir=None,
                            batch_size=256, device="cuda", filter=None):
    if load_dir is None:
        with torch.no_grad():
            train_feats, train_labels = [], []
            for features, _, labels in tqdm(train_loader):
                features = features.to(device)
                logits = cbl(backbone(features))
                train_feats.append(logits.cpu())
                train_labels.append(labels)
            train_feats = torch.cat(train_feats, dim=0)
            train_labels = torch.cat(train_labels, dim=0)

            val_feats, val_labels = [], []
            for features, _, labels in tqdm(val_loader):
                features = features.to(device)
                logits = cbl(backbone(features))
                val_feats.append(logits.cpu())
                val_labels.append(labels)
            val_feats = torch.cat(val_feats, dim=0)
            val_labels = torch.cat(val_labels, dim=0)

            mean = train_feats.mean(dim=0)
            std = train_feats.std(dim=0).clamp(min=1e-8)
            train_feats = (train_feats - mean) / std
            val_feats = (val_feats - mean) / std
            norm_layer = NormalizationLayer(mean, std, device=device)
    else:
        train_feats = torch.load(os.path.join(load_dir, "train_concept_features.pt"))
        train_labels = torch.load(os.path.join(load_dir, "train_concept_labels.pt"))
        val_feats = torch.load(os.path.join(load_dir, "val_concept_features.pt"))
        val_labels = torch.load(os.path.join(load_dir, "val_concept_labels.pt"))
        norm_layer = NormalizationLayer.from_pretrained(load_dir, device=device)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_feats, os.path.join(save_dir, "train_concept_features.pt"))
    torch.save(train_labels, os.path.join(save_dir, "train_concept_labels.pt"))
    torch.save(val_feats, os.path.join(save_dir, "val_concept_features.pt"))
    torch.save(val_labels, os.path.join(save_dir, "val_concept_labels.pt"))
    norm_layer.save_model(save_dir)

    if filter is not None:
        train_feats = train_feats[:, filter]
        val_feats = val_feats[:, filter]

    train_ds = IndexedTensorDataset(train_feats, train_labels)
    val_ds = TensorDataset(val_feats, val_labels)
    train_loader_out = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader_out = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader_out, val_loader_out, norm_layer


def train_sparse_final(linear, indexed_train_loader, val_loader, n_iters, lam, step_size=0.1, device="cuda"):
    num_classes = linear.weight.shape[0]
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    metadata = {"max_reg": {"nongrouped": lam}}
    out = glm_saga(
        linear, indexed_train_loader, step_size, n_iters, 0.99,
        epsilon=1, k=1, val_loader=val_loader, do_zero=False, metadata=metadata,
        n_ex=len(indexed_train_loader.dataset), n_classes=num_classes, verbose=True,
    )
    return out


def train_dense_final(model, indexed_train_loader, val_loader, n_iters, lr=0.001, device="cuda"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(opt, gamma=0.95)
    ce = nn.CrossEntropyLoss()
    for epoch in range(n_iters):
        model.train()
        for inputs, targets, _ in indexed_train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = ce(model(inputs), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_acc += (model(inputs).argmax(1) == targets).float().mean().item()
        val_acc /= len(val_loader)
        scheduler.step()
    return {
        "path": [{
            "weight": model.weight.detach().cpu().clone(),
            "bias": model.bias.detach().cpu().clone(),
            "lr": lr, "lam": -1.0, "alpha": -1.0, "time": -1.0,
            "metrics": {"val_accuracy": val_acc},
        }]
    }
