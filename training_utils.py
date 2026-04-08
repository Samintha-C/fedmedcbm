import json
import os
import random
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

import sys
import importlib.util

# Add current directory to path FIRST, before any other imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import local utils modules directly from files to avoid conflicts with Label-free-CBM's utils.py
utils_concepts_path = os.path.join(current_dir, 'utils', 'concepts.py')
data_utils_path = os.path.join(current_dir, 'data', 'data_utils.py')

spec_concepts = importlib.util.spec_from_file_location("fed_utils_concepts", utils_concepts_path)
spec_data = importlib.util.spec_from_file_location("fed_data_utils", data_utils_path)

fed_utils_concepts = importlib.util.module_from_spec(spec_concepts)
fed_data_utils = importlib.util.module_from_spec(spec_data)

spec_concepts.loader.exec_module(fed_utils_concepts)
spec_data.loader.exec_module(fed_data_utils)

# Import functions with unique names to avoid conflicts
load_concepts_from_file = fed_utils_concepts.load_concepts_from_file
load_or_generate_concept_embeddings = fed_utils_concepts.load_or_generate_concept_embeddings
get_data = fed_data_utils.get_data
get_classes = fed_data_utils.get_classes
get_resnet_preprocess = fed_data_utils.get_resnet_preprocess
get_clip_preprocess = fed_data_utils.get_clip_preprocess
split_dataset_for_federated = fed_data_utils.split_dataset_for_federated
print_client_distribution = fed_data_utils.print_client_distribution


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def federated_averaging(models, client_weights=None, exclude_prefixes=None):
    if client_weights is None:
        client_weights = [1.0 / len(models)] * len(models)

    global_state = {}
    for key in models[0].state_dict().keys():
        param = models[0].state_dict()[key]

        # Skip frozen parameters (e.g. backbone) — just copy from first model
        if exclude_prefixes and any(key.startswith(p) for p in exclude_prefixes):
            global_state[key] = param.clone()
            continue

        # Only average float parameters (weights, biases)
        # For non-float parameters (Long tensors like batch norm counts), copy from first model
        if param.dtype.is_floating_point:
            global_state[key] = torch.zeros_like(param)
            for i, model in enumerate(models):
                global_state[key] += client_weights[i] * model.state_dict()[key]
        else:
            # For non-float parameters, just copy from first model
            global_state[key] = param.clone()

    return global_state


def train_client_local(
    model,
    train_loader,
    concept_embeddings,
    clip_model,
    epochs,
    lr,
    weight_decay,
    sparsity_lambda,
    device
):
    model.train()
    optimizer = torch.optim.Adam(model.projection.parameters(), lr=lr, weight_decay=weight_decay)
    sparsity_loss_fn = L1SparsityLoss(lambda_l1=sparsity_lambda)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for images, _ in train_loader:
            images = images.to(device)

            with torch.no_grad():
                image_features = model.backbone(images)
                clip_image_features = clip_model.encode_image(images)
                clip_image_features = clip_image_features.float()  # Convert to float32
                clip_image_features = clip_image_features / clip_image_features.norm(dim=-1, keepdim=True)
                clip_targets = clip_image_features @ concept_embeddings.T

            concept_pred = model.projection(image_features)

            loss = cosine_similarity_cubed_loss(concept_pred, clip_targets)
            sparsity_loss = sparsity_loss_fn(concept_pred, clip_targets, model.get_projection_params())
            total_loss = loss + sparsity_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1

    return epoch_loss / num_batches if num_batches > 0 else 0.0


def train_final_layer_local(model, train_loader, epochs, lr, device):
    model.train()
    optimizer = torch.optim.Adam(model.final_layer.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            if model.normalization is not None:
                with torch.no_grad():
                    image_features = model.backbone(images)
                    concept_pred = model.projection(image_features)
                    concept_features = model.normalization(concept_pred)
            else:
                with torch.no_grad():
                    image_features = model.backbone(images)
                    concept_pred = model.projection(image_features)
                    concept_features = concept_pred

            logits = model.final_layer(concept_features)
            loss = ce_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

    return epoch_loss / num_batches if num_batches > 0 else 0.0


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images, return_concepts=True)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def save_args(save_dir, args):
    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        json.dump(vars(args), f, indent=2)


def save_concepts(save_dir, concepts):
    with open(os.path.join(save_dir, "concepts.txt"), "w") as f:
        if concepts:
            f.write(concepts[0])
            for concept in concepts[1:]:
                f.write('\n' + concept)


def save_training_metrics(save_dir, training_metrics):
    with open(os.path.join(save_dir, "training_metrics.json"), "w") as f:
        json.dump(training_metrics, f, indent=2)
    print(f"\nTraining metrics saved to {os.path.join(save_dir, 'training_metrics.json')}")


def save_metrics_txt(save_dir, metrics_data):
    try:
        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            json.dump(metrics_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save metrics.txt: {e}")


def get_preprocess(backbone_name):
    if backbone_name.startswith("clip_"):
        return get_clip_preprocess()
    else:
        return get_resnet_preprocess()


def create_client_data_loaders(train_dataset, client_indices, num_clients, batch_size, num_workers):
    client_train_loaders = []
    client_data_sizes = []

    for i in range(num_clients):
        client_train_subset = Subset(train_dataset, client_indices[i])
        train_loader = DataLoader(
            client_train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        client_train_loaders.append(train_loader)
        client_data_sizes.append(len(client_train_subset))

    total_samples = sum(client_data_sizes)
    client_weights = [size / total_samples for size in client_data_sizes]

    return client_train_loaders, client_data_sizes, client_weights


def log_mem(tag: str = ""):
    """Print GPU and CPU memory usage at the current point in training."""
    msg = f"[MEM] {tag}"
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        msg += f"  GPU alloc={alloc:.2f}GB reserved={reserved:.2f}GB"
    try:
        import psutil, os as _os
        proc = psutil.Process(_os.getpid())
        rss = proc.memory_info().rss / 1024 ** 3
        msg += f"  CPU RSS={rss:.2f}GB"
    except Exception:
        pass
    print(msg)


def init_log(log_dir, run_name, is_phase1):
    """Create structured log file at job start. Returns path or None."""
    if log_dir is None:
        return None
    phase_subdir = "p1" if is_phase1 else "p2"
    log_path = os.path.join(log_dir, phase_subdir, f"{run_name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        with open(log_path, "w") as f:
            json.dump({"run_name": run_name, "started_at": datetime.datetime.now().isoformat(),
                       "status": "started"}, f, indent=2)
    except Exception:
        pass
    return log_path


def update_log(log_path, metrics_dict):
    """Overwrite the structured log file with current metrics."""
    if log_path is None:
        return
    try:
        with open(log_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
    except Exception:
        pass
