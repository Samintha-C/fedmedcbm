from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LabelFlipConfig:
    source_class: int
    target_class: int


@dataclass
class AttackConfig:
    # Currently "label_flip"; future: "gradient", "backdoor", "concept_poison"
    attack_type: str
    label_flip: Optional[LabelFlipConfig] = None


@dataclass
class DefenseConfig:
    # "none" | "detection_only" | "client_exclusion" | "reweighting"
    mode: str = "none"
    detection_interval: int = 10
    # z-score threshold above which a client is flagged
    suspicion_threshold: float = 1.5
    # for reweighting: effective_weight *= (1 - decay * suspicion)
    reweight_decay: float = 0.7


@dataclass
class PoisoningContext:
    # Maps client index → AttackConfig.  Clients not in this dict are clean.
    client_attacks: Dict[int, AttackConfig]
    defense: DefenseConfig
    num_classes: int
    # Fraction of least-important concepts zeroed in intervention check
    intervention_fraction: float = 0.5
    # Number of random-zeroing trials for the baseline
    intervention_trials: int = 5
    # Populated during training; read back by the experiment after the run
    detection_log: List[dict] = field(default_factory=list)
