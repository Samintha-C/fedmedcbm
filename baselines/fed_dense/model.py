"""Dense (non-CBM) federated classifier: frozen backbone + linear head.

Mirrors the CBM's frozen-backbone setup exactly, minus the concept bottleneck.
The forward signature matches FedVLGCBM so the existing evaluation helpers
(get_accuracy_cbm, get_per_class_accuracy_vlg) work unchanged.
"""

import torch.nn as nn


class DenseClassifier(nn.Module):
    """frozen backbone -> (optional) feature normalization -> dense linear head."""

    def __init__(self, backbone, normalization=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.normalization = normalization
        self.head = head

    def forward(self, x, return_concepts=False):
        # x.dim()==4: raw image batch -> run backbone. Else: pre-extracted feats.
        h = self.backbone(x) if x.dim() == 4 else x
        if self.normalization is not None:
            h = self.normalization(h)
        logits = self.head(h)
        # return_concepts kept for eval-helper compatibility; "concepts" here are
        # just the normalized backbone features (no semantic meaning).
        return (logits, h) if return_concepts else logits
