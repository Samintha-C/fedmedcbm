import os
import sys
import torch
import numpy as np
from torch.utils.data import Subset, random_split
from torchvision import datasets, transforms, models
import clip

lf_cbm_path = os.path.join(os.path.dirname(__file__), '../../Label-free-CBM')
if os.path.exists(lf_cbm_path):
    sys.path.insert(0, lf_cbm_path)
    import data_utils as lf_cbm_data_utils
else:
    raise ImportError(f"Label-free-CBM not found at {lf_cbm_path}. Please ensure Label-free-CBM is in the parent directory.")

def get_data(dataset_name, preprocess=None, root=None):
    return lf_cbm_data_utils.get_data(dataset_name, preprocess)

def get_target_model(target_name, device):
    return lf_cbm_data_utils.get_target_model(target_name, device)

def get_classes(dataset_name):
    local_label_file = os.path.join(os.path.dirname(__file__), f"{dataset_name}_classes.txt")
    if os.path.exists(local_label_file):
        label_file = local_label_file
    else:
        label_file = lf_cbm_data_utils.LABEL_FILES.get(dataset_name)
        if label_file is None:
            if dataset_name == "cifar10":
                label_file = lf_cbm_data_utils.LABEL_FILES["cifar10"]
            elif dataset_name == "cifar100":
                label_file = lf_cbm_data_utils.LABEL_FILES["cifar100"]
            elif dataset_name == "imagenet":
                label_file = lf_cbm_data_utils.LABEL_FILES["imagenet"]
            elif dataset_name == "cub":
                label_file = lf_cbm_data_utils.LABEL_FILES["cub"]
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
    
    with open(label_file, "r") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes

def get_resnet_preprocess():
    return lf_cbm_data_utils.get_resnet_imagenet_preprocess()

def get_clip_preprocess():
    _, preprocess = clip.load("ViT-B/16")
    return preprocess

def get_dataset_targets(dataset):
    """Extract targets/labels from dataset, handling different dataset types."""
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        return np.array(dataset.labels)
    else:
        # Fallback: iterate through dataset (slower)
        targets = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            targets.append(label)
        return np.array(targets)


def split_dataset_dirichlet(dataset, num_clients, alpha=0.5, seed=42, min_samples=10):
    """
    Splits dataset indices among clients using a Dirichlet distribution.

    This is the standard approach in federated learning research (FedML, LEAF, FedAvg paper).

    Args:
        dataset: PyTorch dataset (must have .targets or .labels attribute)
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (controls heterogeneity)
               - alpha -> 0: Extreme non-IID (each client gets ~1 class)
               - alpha ~ 0.1-0.5: Realistic non-IID (clients specialize but have some overlap)
               - alpha ~ 1.0: Moderate non-IID
               - alpha -> inf: IID (uniform distribution)
        seed: Random seed for reproducibility
        min_samples: Minimum samples per client (to avoid empty clients)

    Returns:
        list: [client_0_indices, client_1_indices, ...]
    """
    np.random.seed(seed)

    targets = get_dataset_targets(dataset)
    num_classes = int(np.max(targets)) + 1
    num_samples = len(targets)

    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]

    # Partition each class independently using Dirichlet
    for c in range(num_classes):
        # Get all indices for class c
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)

        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to actual counts
        # Ensure at least some samples go to clients with non-zero proportion
        proportions = proportions / proportions.sum()
        counts = (proportions * len(idx_c)).astype(int)

        # Distribute remaining samples (due to rounding)
        remainder = len(idx_c) - counts.sum()
        for i in range(remainder):
            counts[i % num_clients] += 1

        # Split indices according to counts
        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(idx_c[start:end].tolist())
            start = end

    # Shuffle each client's indices
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])

    return client_indices


def split_dataset_for_federated(dataset, num_clients, iid=True, alpha=0.5, seed=42):
    """
    Split dataset for federated learning with IID or non-IID distribution.

    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        iid: If True, use IID partitioning. If False, use Dirichlet non-IID.
        alpha: Dirichlet concentration parameter (only used when iid=False)
               - alpha ~ 0.1: Extreme non-IID (clients specialize heavily)
               - alpha ~ 0.5: Moderate non-IID (realistic heterogeneity)
               - alpha ~ 1.0: Mild non-IID
               - alpha >= 100: Nearly IID
        seed: Random seed

    Returns:
        list: [client_0_indices, client_1_indices, ...]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if iid:
        # IID: Random uniform split
        total_size = len(dataset)
        indices = torch.randperm(total_size).tolist()
        chunk_size = total_size // num_clients
        client_indices = []

        for i in range(num_clients):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_clients - 1 else total_size
            client_indices.append(indices[start_idx:end_idx])

        return client_indices
    else:
        # Non-IID: Dirichlet distribution
        return split_dataset_dirichlet(dataset, num_clients, alpha=alpha, seed=seed)


def print_client_distribution(dataset, client_indices, num_classes=None):
    """
    Print the class distribution for each client (useful for debugging).

    Args:
        dataset: PyTorch dataset
        client_indices: List of index lists per client
        num_classes: Number of classes (auto-detected if None)
    """
    targets = get_dataset_targets(dataset)
    if num_classes is None:
        num_classes = int(np.max(targets)) + 1

    print("\n=== Client Data Distribution ===")
    for client_id, indices in enumerate(client_indices):
        client_labels = targets[indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        total = sum(counts)
        print(f"Client {client_id}: {total} samples, distribution: {dist}")
    print("================================\n")
