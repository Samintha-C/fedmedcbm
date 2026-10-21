from typing import List
import torch
import torch.nn as nn

nINF = -100


class CosCubedLoss(nn.Module):
    """Cosine-cubed similarity loss from Label-free CBM (Oikarinen & Das, 2023).

    For each concept column, mean-center across the batch, cube element-wise
    (sharpens peaks, suppresses near-zero), L2-normalize per column, then
    compute cosine similarity.  Returned loss is the negative mean similarity
    so that minimising the loss maximises alignment.

    This is scale-invariant by construction, which makes it robust to federated
    averaging where absolute activation magnitudes shift across clients.
    """

    def forward(self, x, y):
        # x: concept logits  [batch_size, num_concepts]
        # y: concept targets  [batch_size, num_concepts]  (binary one-hot or continuous)
        x = x.float()
        y = y.float()

        # Mean-center each concept column across the batch
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        # Cube — sharpens high activations, suppresses near-zero
        x = x ** 3
        y = y ** 3

        # L2-normalize each column (across samples)
        x = x / (x.norm(p=2, dim=0, keepdim=True).clamp(min=1e-8))
        y = y / (y.norm(p=2, dim=0, keepdim=True).clamp(min=1e-8))

        # Per-concept cosine similarity, then negate for loss
        similarities = (x * y).sum(dim=0)        # [num_concepts]
        return -similarities.mean()


class TwoWayLoss(nn.Module):
    def __init__(self, Tp=4.0, Tn=1.0):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x / self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x / self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x / self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x / self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]
        return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
            torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()


class BalancedHingeLoss(nn.Module):
    """Per-concept hinge loss on the CBL logits, labels mapped to {-1, +1}, as FedCBM's
    concept SVMs use. Each concept's positives and negatives get equal total weight,
    n / (2 n_pos) and n / (2 n_neg) (sklearn's class_weight="balanced", the FedCBM
    baseline's balanced variant in baselines/fedcbm_reimpl/core.py): unweighted hinge on
    ~1-6% positives scores every image absent. cbl_pos_weight is not used.
    squared=True gives the squared hinge, which is smooth."""

    def __init__(self, concept_counts, num_samples, squared=False):
        super().__init__()
        pos = torch.as_tensor(concept_counts, dtype=torch.float32)
        n = float(num_samples)
        self.register_buffer("w_pos", torch.where(pos > 0, n / (2.0 * pos.clamp(min=1)), torch.zeros_like(pos)))
        self.register_buffer("w_neg", torch.where(pos < n, n / (2.0 * (n - pos).clamp(min=1)), torch.zeros_like(pos)))
        self.squared = squared

    def forward(self, x, y):
        h = torch.clamp(1.0 - (y * 2.0 - 1.0) * x, min=0.0)
        if self.squared:
            h = h * h
        return (h * torch.where(y > 0, self.w_pos, self.w_neg)).mean()


def get_loss(loss_type: str, num_concepts: int, num_samples: int, concept_counts: List[float],
             cbl_pos_weight: float, cbl_auto_weight: bool = False, tp: float = 4.0, device="cuda"):
    if loss_type == "bce":
        if cbl_auto_weight:
            pos_count = torch.tensor(concept_counts).to(device)
            neg_count = num_samples - pos_count
            scale = (neg_count / pos_count.clamp(min=1)) * cbl_pos_weight
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=scale).to(device)
        else:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cbl_pos_weight] * num_concepts)).to(device)
    elif loss_type == "twoway":
        loss_fn = TwoWayLoss(Tp=tp)
    elif loss_type == "cos_cubed":
        loss_fn = CosCubedLoss()
    elif loss_type in ("hinge", "sqhinge"):
        loss_fn = BalancedHingeLoss(concept_counts, num_samples, squared=loss_type == "sqhinge").to(device)
    else:
        raise NotImplementedError(f"Loss {loss_type} is not implemented")
    return loss_fn
