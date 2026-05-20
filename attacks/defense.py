"""
Server-side defense strategies for federated CBM poisoning.

Three modes:
  "detection_only"   — identify suspicious clients but do not change aggregation weights.
  "client_exclusion" — set flagged clients' contribution to zero and renormalize.
  "reweighting"      — scale flagged clients' contribution by (1 - decay * suspicion),
                       giving a smooth rather than binary penalty.

"none" is handled by the caller (no detection is run at all).
"""

from typing import List


def apply_defense(
    mode: str,
    client_weights: List[float],
    suspicion_scores: List[float],
    flagged_clients: List[int],
    reweight_decay: float = 0.7,
) -> List[float]:
    """
    Return effective aggregation weights for this round.

    For "detection_only" the original weights are returned unchanged — the caller
    still logs the detection result but does not act on it.
    """
    effective = list(client_weights)

    if mode == "detection_only" or mode == "none":
        return effective

    if mode == "client_exclusion":
        for i in flagged_clients:
            effective[i] = 0.0

    elif mode == "reweighting":
        for i, sus in enumerate(suspicion_scores):
            # Multiply each client's weight by a decay proportional to suspicion.
            # Decay is clamped to [0, 1] so no client gets a negative weight.
            scale = max(0.0, 1.0 - reweight_decay * sus)
            effective[i] *= scale

    # Renormalize so weights sum to 1 (skip if all are zero to avoid div-by-zero)
    total = sum(effective)
    if total > 1e-8:
        effective = [w / total for w in effective]

    return effective
