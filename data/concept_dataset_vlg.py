import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from .data_utils import get_data, get_classes, format_concept


class DinoConceptDataset(Dataset):
    """
    Uses pre-generated Grounding DINO annotations (same format as VLG-CBM).
    Layout: annotation_dir/{dataset_name}_{split_suffix}/{idx}.json
    JSON: list with first elem {"img_path": ...}, rest {"label": str, "logit": float, "box": [...]}.
    Mirrors VLG-CBM ConceptDataset.__getitem__all (no crop_to_concept).
    """
    def __init__(
        self,
        dataset_name: str,
        torch_dataset: Dataset,
        concepts: List[str],
        annotation_dir: str,
        split_suffix: str = "train",
        confidence_threshold: float = 0.10,
        preprocess=None,
    ):
        self.torch_dataset = torch_dataset
        self.concepts = concepts
        self.concept_set = set(concepts)
        self.preprocess = preprocess
        self.confidence_threshold = confidence_threshold
        self.dir = os.path.join(annotation_dir, f"{dataset_name}_{split_suffix}")

        # Pre-load all annotations into RAM to avoid per-sample CephFS reads.
        self._annotation_cache = self._preload_annotations()

    def _cache_path(self) -> str:
        """Deterministic path for the serialised annotation tensor."""
        import hashlib
        concept_hash = hashlib.md5("|".join(self.concepts).encode()).hexdigest()[:12]
        fname = f"_cache_c{len(self.concepts)}_t{self.confidence_threshold:.3f}_{concept_hash}.pt"
        # Store next to the per-split annotation directory.
        return os.path.join(os.path.dirname(self.dir), os.path.basename(self.dir) + fname)

    def _preload_annotations(self) -> torch.Tensor:
        """Load all annotation JSONs and convert to a [N, num_concepts] float tensor.

        On first call the result is persisted to a .pt cache file so subsequent
        runs (including across Kubernetes restarts) skip the NFS file scan.
        Uses a thread pool to parallelise I/O on the first (cache-miss) run.
        """
        import concurrent.futures

        cache_file = self._cache_path()
        n = len(self.torch_dataset)
        num_concepts = len(self.concepts)

        if os.path.isfile(cache_file):
            print(f"[DinoConceptDataset] Loading annotation cache: {cache_file}", flush=True)
            tensor = torch.load(cache_file, weights_only=True)
            if tensor.shape == (n, num_concepts):
                return tensor
            print(f"[DinoConceptDataset] Cache shape mismatch {tensor.shape}, rebuilding.", flush=True)

        print(
            f"[DinoConceptDataset] Building annotation cache for {n} samples "
            f"({num_concepts} concepts) — this is a one-time NFS scan.",
            flush=True,
        )

        indices = (
            [int(self.torch_dataset.indices[i]) for i in range(n)]
            if hasattr(self.torch_dataset, "indices")
            else list(range(n))
        )

        concepts = self.concepts
        conf_thresh = self.confidence_threshold
        ann_dir = self.dir

        cache = torch.zeros(n, num_concepts, dtype=torch.float)

        def _load_one(args):
            local_idx, real_idx = args
            path = os.path.join(ann_dir, f"{real_idx}.json")
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                return local_idx, None
            bbxs = [b for b in data[1:] if b.get("logit", 0) > conf_thresh]
            present = set(format_concept(b.get("label", "")) for b in bbxs)
            row = torch.tensor(
                [1.0 if c in present else 0.0 for c in concepts],
                dtype=torch.float,
            )
            return local_idx, row

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            futures = pool.map(_load_one, enumerate(indices))
            for local_idx, row in tqdm(futures, total=n, desc="Loading DINO annotations", unit="files"):
                if row is not None:
                    cache[local_idx] = row

        print(f"[DinoConceptDataset] Saving annotation cache to {cache_file}", flush=True)
        torch.save(cache, cache_file)
        return cache

    def __len__(self):
        return len(self.torch_dataset)

    def __getitem__(self, idx):
        image, target = self.torch_dataset[idx]
        concept_one_hot = self._annotation_cache[idx]
        if self.preprocess:
            image = self.preprocess(image)
        return image, concept_one_hot, target


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
    annotation_dir: Optional[str] = None,
    confidence_threshold: float = 0.10,
):
    train_suffix = "train"
    val_suffix = "val"
    use_dino = annotation_dir is not None and os.path.isdir(annotation_dir)
    if use_dino:
        use_allones = False

    if split == "test":
        try:
            test_data = get_data(f"{dataset_name}_test", None)
        except Exception:
            test_data = get_data(f"{dataset_name}_{val_suffix}", None)
        if use_dino:
            base_dataset = DinoConceptDataset(
                dataset_name, test_data, concepts,
                annotation_dir=annotation_dir, split_suffix=val_suffix,
                confidence_threshold=confidence_threshold, preprocess=preprocess,
            )
        else:
            base_dataset = AllOneConceptDataset(
                dataset_name, test_data, concepts, preprocess=preprocess,
            )
        return DataLoader(base_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    base_data = get_data(f"{dataset_name}_{train_suffix}", None)
    if use_dino:
        base_dataset = DinoConceptDataset(
            dataset_name, base_data, concepts,
            annotation_dir=annotation_dir, split_suffix=train_suffix,
            confidence_threshold=confidence_threshold, preprocess=preprocess,
        )
    else:
        base_dataset = AllOneConceptDataset(
            dataset_name, base_data, concepts, preprocess=preprocess,
        )
    n_val = int(val_split * len(base_dataset))
    n_train = len(base_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        base_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    dataset = train_ds if split == "train" else val_ds
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


