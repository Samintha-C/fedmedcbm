import hashlib
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

    Annotations are stored as a single stacked bool tensor [N, num_concepts]
    (4x smaller than float32; matters for places365-scale data). __getitem__
    casts to float at access time.

    Optional feature cache: when populated via populate_feature_cache(),
    __getitem__ returns pre-extracted backbone embeddings instead of raw
    images, letting the Phase 1 training loop skip the backbone entirely.
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
        cache_dir: Optional[str] = None,
    ):
        self.torch_dataset = torch_dataset
        self.concepts = concepts
        self.concept_set = set(concepts)
        self.preprocess = preprocess
        self.confidence_threshold = confidence_threshold
        self.dir = os.path.join(annotation_dir, f"{dataset_name}_{split_suffix}")
        self.dataset_name = dataset_name
        self.split_suffix = split_suffix
        self.cache_dir = cache_dir
        self._feature_cache: Optional[torch.Tensor] = None
        self._target_cache: Optional[torch.Tensor] = None

        cached = self._try_load_cache()
        if cached is not None:
            self._annotation_cache = cached
        else:
            self._annotation_cache = self._preload_annotations()
            self._save_cache(self._annotation_cache)

    def _real_indices(self) -> List[int]:
        n = len(self.torch_dataset)
        if hasattr(self.torch_dataset, "indices"):
            return [int(self.torch_dataset.indices[i]) for i in range(n)]
        return list(range(n))

    def _cache_path(self) -> Optional[str]:
        """Filename encodes all identity-defining inputs. If a file with this
        name exists, the contents are guaranteed to match the current dataset
        config — no need for separate metadata validation."""
        if self.cache_dir is None:
            return None
        concepts_hash = hashlib.md5("\n".join(self.concepts).encode()).hexdigest()[:12]
        idx_blob = torch.tensor(self._real_indices(), dtype=torch.long).numpy().tobytes()
        idx_hash = hashlib.md5(idx_blob).hexdigest()[:12]
        fname = (
            f"{self.dataset_name}_{self.split_suffix}"
            f"_conf{self.confidence_threshold:.3f}"
            f"_c{concepts_hash}_s{idx_hash}.pt"
        )
        return os.path.join(self.cache_dir, fname)

    def _try_load_cache(self) -> Optional[torch.Tensor]:
        path = self._cache_path()
        if path is None or not os.path.exists(path):
            return None
        try:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"[annotation cache] failed to load {path}: {e}; will rebuild")
            return None
        expected = (len(self.torch_dataset), len(self.concepts))
        if tuple(tensor.shape) != expected:
            print(f"[annotation cache] shape mismatch ({tuple(tensor.shape)} vs {expected}); will rebuild")
            return None
        print(f"[annotation cache] loaded {path} ({tuple(tensor.shape)}, {tensor.dtype})")
        return tensor

    def _save_cache(self, tensor: torch.Tensor) -> None:
        path = self._cache_path()
        if path is None:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(tensor, path)
            print(f"[annotation cache] saved {path} ({tuple(tensor.shape)}, {tensor.dtype})")
        except Exception as e:
            print(f"[annotation cache] failed to save {path}: {e}")

    def _preload_annotations(self) -> torch.Tensor:
        """Load all annotation JSONs into a single stacked bool tensor
        [N, num_concepts]. Uses a thread pool so NFS per-file latency
        doesn't dominate."""
        import concurrent.futures

        indices = self._real_indices()
        n = len(indices)
        num_concepts = len(self.concepts)
        conf_thresh = self.confidence_threshold
        ann_dir = self.dir
        concepts = self.concepts

        stacked = torch.zeros((n, num_concepts), dtype=torch.bool)

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
            row = torch.tensor([c in present for c in concepts], dtype=torch.bool)
            return local_idx, row

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for local_idx, row in pool.map(_load_one, enumerate(indices)):
                if row is not None:
                    stacked[local_idx] = row
        return stacked

    def _feature_cache_path(self, backbone_name: str) -> Optional[str]:
        if self.cache_dir is None:
            return None
        safe_name = backbone_name.replace("/", "_").replace(":", "_")
        idx_blob = torch.tensor(self._real_indices(), dtype=torch.long).numpy().tobytes()
        idx_hash = hashlib.md5(idx_blob).hexdigest()[:12]
        fname = f"{self.dataset_name}_{self.split_suffix}_{safe_name}_feat_s{idx_hash}.pt"
        return os.path.join(self.cache_dir, fname)

    def populate_feature_cache(
        self,
        backbone,
        device: str,
        backbone_name: str,
        batch_size: int = 256,
        num_workers: int = 4,
        prefetch_factor: int = 4,
    ) -> None:
        """Pre-extract and cache backbone embeddings + targets. Caller must
        ensure the backbone is frozen — cached features become invalid
        otherwise. After this runs, __getitem__ yields embeddings instead
        of raw images; the Phase 1 loop should branch on features.dim()."""
        path = self._feature_cache_path(backbone_name)
        expected_n = len(self.torch_dataset)

        if path is not None and os.path.exists(path):
            try:
                blob = torch.load(path, map_location="cpu", weights_only=True)
                if blob["features"].shape[0] == expected_n:
                    self._feature_cache = blob["features"]
                    self._target_cache = blob["targets"]
                    print(f"[feature cache] loaded {path} "
                          f"({tuple(self._feature_cache.shape)}, {self._feature_cache.dtype})")
                    return
                print(f"[feature cache] row mismatch "
                      f"({blob['features'].shape[0]} vs {expected_n}); rebuilding")
            except Exception as e:
                print(f"[feature cache] failed to load {path}: {e}; rebuilding")

        torch_dataset = self.torch_dataset
        preprocess = self.preprocess

        class _PreprocessDataset(Dataset):
            def __len__(self_inner):
                return len(torch_dataset)
            def __getitem__(self_inner, idx):
                img, tgt = torch_dataset[idx]
                if preprocess is not None:
                    img = preprocess(img)
                return img, tgt

        # pin_memory=False: extraction is a one-shot, GPU-bound op. Pinning
        # balloons CPU RSS (page-locked memory accounting is lossy in cgroup v2)
        # and gave us OOMKills on CLIP backbones with batch_size * 4 loaders.
        # prefetch_factor keeps workers further ahead of the GPU during NFS stalls.
        _pf = prefetch_factor if num_workers > 0 else None
        loader = DataLoader(
            _PreprocessDataset(), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False,
            prefetch_factor=_pf,
        )

        print(f"[feature cache] extracting backbone features "
              f"for {expected_n} samples (batch_size={batch_size})...", flush=True)
        backbone.eval()
        feats_parts, targets_parts = [], []
        with torch.no_grad():
            for images, tgts in tqdm(loader, desc="extract features"):
                images = images.to(device, non_blocking=True)
                emb = backbone(images).detach().to("cpu", dtype=torch.float16)
                feats_parts.append(emb)
                targets_parts.append(tgts.to(torch.int64))

        self._feature_cache = torch.cat(feats_parts, dim=0)
        self._target_cache = torch.cat(targets_parts, dim=0)
        print(f"[feature cache] extracted "
              f"{tuple(self._feature_cache.shape)}, {self._feature_cache.dtype}")

        if path is not None:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(
                    {"features": self._feature_cache, "targets": self._target_cache},
                    path,
                )
                print(f"[feature cache] saved {path}")
            except Exception as e:
                print(f"[feature cache] failed to save {path}: {e}")

    def __len__(self):
        return len(self.torch_dataset)

    def __getitem__(self, idx):
        if self._feature_cache is not None:
            image = self._feature_cache[idx].float()
            target = int(self._target_cache[idx])
        else:
            image, target = self.torch_dataset[idx]
            if self.preprocess:
                image = self.preprocess(image)
        concept_one_hot = self._annotation_cache[idx].float()
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
    cache_dir: Optional[str] = None,
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
                cache_dir=cache_dir,
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
            cache_dir=cache_dir,
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


