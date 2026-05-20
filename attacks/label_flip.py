import torch


def apply_label_flip(labels: torch.Tensor, source_class: int, target_class: int) -> torch.Tensor:
    """Return a copy of `labels` with source_class relabeled as target_class."""
    labels = labels.clone()
    labels[labels == source_class] = target_class
    return labels
