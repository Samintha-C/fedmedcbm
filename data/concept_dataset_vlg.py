import os
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from .data_utils import get_data, get_classes


class AllOneConceptDataset(Dataset):
    def __init__(self, dataset_name: str, torch_dataset: Dataset, concepts: List[str],
                 preprocess=None):
        self.torch_dataset = torch_dataset
        self.concepts = concepts
        self.preprocess = preprocess
        classes = get_classes(dataset_name)
        self.per_class_concepts = len(concepts) // len(classes)

    def __len__(self):
        return len(self.torch_dataset)

    def __getitem__(self, idx):
        image, target = self.torch_dataset[idx]
        if self.preprocess:
            image = self.preprocess(image)
        concept_one_hot = torch.zeros(len(self.concepts), dtype=torch.float)
        concept_one_hot[target * self.per_class_concepts : (target + 1) * self.per_class_concepts] = 1
        return image, concept_one_hot, target


def get_concept_dataloader(
    dataset_name: str,
    split: str,
    concepts: List[str],
    preprocess=None,
    val_split: Optional[float] = 0.1,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = False,
    label_dir: str = "outputs",
    use_allones: bool = True,
    seed: int = 42,
):
    assert use_allones, "Only use_allones=True is supported (no DINO)"
    train_suffix = "train"
    val_suffix = "val"
    if split == "test":
        try:
            test_data = get_data(f"{dataset_name}_test", None)
        except Exception:
            test_data = get_data(f"{dataset_name}_{val_suffix}", None)
        base_dataset = AllOneConceptDataset(
            dataset_name,
            test_data,
            concepts,
            preprocess=preprocess,
        )
        loader = DataLoader(base_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        return loader
    base_dataset = AllOneConceptDataset(
        dataset_name,
        get_data(f"{dataset_name}_{train_suffix}", None),
        concepts,
        preprocess=preprocess,
    )
    n_val = int(val_split * len(base_dataset))
    n_train = len(base_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        base_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    dataset = train_ds if split == "train" else val_ds
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


