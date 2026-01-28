from typing import List
import torch
import torch.nn as nn

nINF = -100


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
    else:
        raise NotImplementedError(f"Loss {loss_type} is not implemented")
    return loss_fn
