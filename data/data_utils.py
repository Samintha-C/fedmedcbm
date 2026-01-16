import os
import sys
import torch
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

def split_dataset_for_federated(dataset, num_clients, iid=True, seed=42):
    torch.manual_seed(seed)
    
    if iid:
        total_size = len(dataset)
        indices = torch.randperm(total_size)
        splits = torch.split(indices, total_size // num_clients)
        return [split.tolist() for split in splits]
    else:
        sorted_indices = sorted(range(len(dataset)), key=lambda i: dataset[i][1])
        total_size = len(dataset)
        chunk_size = total_size // num_clients
        client_indices = []
        
        for i in range(num_clients):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_clients - 1 else total_size
            client_indices.append(sorted_indices[start_idx:end_idx])
        
        return client_indices
