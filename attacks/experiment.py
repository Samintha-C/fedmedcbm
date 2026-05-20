"""
Poisoning experiment orchestrator for federated CBM.

Runs a sweep over (n_malicious, defense_mode) configurations using pre-extracted
Phase 1+2 concept features.  For each config it:
  1. Builds a PoisoningContext (attack assignments + defense policy).
  2. Calls simulate_federated_training_vlg with that context.
  3. Loads the saved model and computes evaluation metrics.
  4. Aggregates detection precision/recall from the context's detection_log.

Usage (from evaluations/poisoning_eval.py):
    exp = PoisoningExperiment(base_args, output_dir)
    results = exp.run_sweep()
    exp.save_results(results)
    exp.print_summary(results)
"""

import copy
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attacks.config import AttackConfig, DefenseConfig, LabelFlipConfig, PoisoningContext


# ---------------------------------------------------------------------------
# cifar100 attack pairs: (source_class, target_class)
# snake=77, lobster=45, mountain=49, bus=13, oak_tree=52, tractor=88
# Chosen so source and target have clearly distinct concept profiles.
# ---------------------------------------------------------------------------
CIFAR100_ATTACK_PAIRS: List[Tuple[int, int]] = [
    (77, 45),   # snake → lobster   (client 0)
    (49, 13),   # mountain → bus    (client 1)
    (52, 88),   # oak_tree → tractor (client 2)
]

CIFAR100_CLASS_NAMES: Dict[int, str] = {
    13: "bus", 45: "lobster", 49: "mountain",
    52: "oak_tree", 77: "snake", 88: "tractor",
}

DEFENSE_MODES = ["none", "detection_only", "client_exclusion", "reweighting"]
N_MALICIOUS_LIST = [0, 1, 2, 3]


class PoisoningExperiment:
    def __init__(
        self,
        base_args,
        output_dir: str,
        attack_pairs: Optional[List[Tuple[int, int]]] = None,
        defense_modes: Optional[List[str]] = None,
        n_malicious_list: Optional[List[int]] = None,
        detection_interval: int = 10,
        suspicion_threshold: float = 1.5,
        reweight_decay: float = 0.7,
    ):
        self.base_args = base_args
        self.output_dir = output_dir
        self.attack_pairs = attack_pairs or CIFAR100_ATTACK_PAIRS
        self.defense_modes = defense_modes or DEFENSE_MODES
        self.n_malicious_list = n_malicious_list or N_MALICIOUS_LIST
        self.detection_interval = detection_interval
        self.suspicion_threshold = suspicion_threshold
        self.reweight_decay = reweight_decay
        os.makedirs(output_dir, exist_ok=True)
        # Resolve num_classes from the dataset if not already set on base_args
        if not hasattr(base_args, "num_classes") or base_args.num_classes is None:
            try:
                from training_utils import get_classes
                base_args.num_classes = len(get_classes(base_args.dataset))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def _build_context(self, n_malicious: int, defense_mode: str) -> PoisoningContext:
        """Assign the first n_malicious clients as attackers using self.attack_pairs."""
        client_attacks: Dict[int, AttackConfig] = {}
        for i in range(n_malicious):
            src, tgt = self.attack_pairs[i]
            client_attacks[i] = AttackConfig(
                attack_type="label_flip",
                label_flip=LabelFlipConfig(source_class=src, target_class=tgt),
            )
        defense = DefenseConfig(
            mode=defense_mode,
            detection_interval=self.detection_interval,
            suspicion_threshold=self.suspicion_threshold,
            reweight_decay=self.reweight_decay,
        )
        return PoisoningContext(
            client_attacks=client_attacks,
            defense=defense,
            num_classes=self.base_args.num_classes if hasattr(self.base_args, "num_classes") else 100,
        )

    # ------------------------------------------------------------------
    # Run one configuration
    # ------------------------------------------------------------------

    def run_configuration(
        self,
        n_malicious: int,
        defense_mode: str,
    ) -> dict:
        """
        Run Phase 3 for one (n_malicious, defense_mode) configuration.
        Returns a results dict with training metrics + detection log.
        """
        from train_vlg import simulate_federated_training_vlg

        config_key = f"n{n_malicious}_{defense_mode}"
        config_save_dir = os.path.join(self.output_dir, config_key)
        os.makedirs(config_save_dir, exist_ok=True)

        args = copy.deepcopy(self.base_args)
        args.save_dir = config_save_dir

        ctx = self._build_context(n_malicious, defense_mode)

        print(f"\n{'='*60}")
        print(f"Running: n_malicious={n_malicious}, defense={defense_mode}")
        if ctx.client_attacks:
            for ci, atk in ctx.client_attacks.items():
                lf = atk.label_flip
                print(f"  Client {ci}: {lf.source_class} → {lf.target_class}")
        print(f"{'='*60}")

        simulate_federated_training_vlg(args, poisoning_ctx=ctx)

        # Locate the saved final layer
        model_dirs = sorted(
            glob.glob(os.path.join(config_save_dir, "fully_trained", "*")),
            key=os.path.getmtime,
        )
        model_dir = model_dirs[-1] if model_dirs else None

        result = {
            "config_key": config_key,
            "n_malicious": n_malicious,
            "defense_mode": defense_mode,
            "attack_pairs": [list(self.attack_pairs[i]) for i in range(n_malicious)],
            "model_dir": model_dir,
            "detection_log": ctx.detection_log,
        }

        # Load saved metrics.txt if present
        if model_dir:
            metrics_path = os.path.join(model_dir, "metrics.txt")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    result["train_metrics"] = json.load(f)

        return result

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------

    def compute_attack_metrics(
        self,
        result: dict,
        test_loader: DataLoader,
        global_model,            # FedVLGCBM with backbone/cbl/normalization loaded
        device: str,
        baseline_result: Optional[dict] = None,
    ) -> dict:
        """
        Given a trained model (Phase 3 complete) and test loader, compute:
          - clean_accuracy: test acc on non-attacked classes
          - attack_success_rate: per attack pair, fraction of source_class classified as target
          - per_class_accuracy: acc on all classes
          - per_class_degradation: drop vs. baseline (if provided)
          - sparsity: % non-zero final layer weights
          - top5_concepts_attacked: concept indices for attacked target classes
        """
        model_dir = result.get("model_dir")
        if model_dir is None or not os.path.exists(os.path.join(model_dir, "final.pt")):
            return {"error": "model not found"}

        # Load final layer weights
        final_state = torch.load(
            os.path.join(model_dir, "final.pt"), map_location=device, weights_only=True
        )
        final_layer = copy.deepcopy(global_model.final_layer)
        final_layer.load_state_dict(final_state)
        final_layer.to(device).eval()

        # Collect all predictions and ground-truth labels
        all_preds, all_labels = _collect_predictions(
            test_loader, global_model.backbone, global_model.cbl,
            global_model.normalization, final_layer, device,
        )

        num_classes = all_preds.max().item() + 1
        num_classes = max(num_classes, final_layer.weight.shape[0])

        # Per-class accuracy
        per_class_acc: Dict[int, float] = {}
        for k in range(num_classes):
            mask = all_labels == k
            if mask.sum() == 0:
                per_class_acc[k] = None
            else:
                per_class_acc[k] = (all_preds[mask] == k).float().mean().item()

        # Attacked classes
        attacked_sources = [p[0] for p in result["attack_pairs"]]
        attacked_targets = [p[1] for p in result["attack_pairs"]]
        attacked_classes = set(attacked_sources + attacked_targets)

        clean_classes = [k for k in range(num_classes) if k not in attacked_classes
                         and per_class_acc.get(k) is not None]
        clean_accuracy = (
            sum(per_class_acc[k] for k in clean_classes) / len(clean_classes)
            if clean_classes else None
        )

        # Overall test accuracy
        test_accuracy = (all_preds == all_labels).float().mean().item()

        # Attack success rate: fraction of source_class test samples predicted as target_class
        attack_success_rates: Dict[str, float] = {}
        for src, tgt in result["attack_pairs"]:
            src_mask = all_labels == src
            if src_mask.sum() == 0:
                attack_success_rates[f"{src}_to_{tgt}"] = None
            else:
                asr = (all_preds[src_mask] == tgt).float().mean().item()
                attack_success_rates[f"{src}_to_{tgt}"] = asr

        # Per-class degradation vs. baseline
        per_class_degradation: Optional[Dict[int, float]] = None
        if baseline_result and "eval_metrics" in baseline_result:
            base_pca = baseline_result["eval_metrics"].get("per_class_accuracy", {})
            per_class_degradation = {}
            for k in range(num_classes):
                base_k = base_pca.get(str(k)) or base_pca.get(k)
                curr_k = per_class_acc.get(k)
                if base_k is not None and curr_k is not None:
                    per_class_degradation[k] = float(base_k) - curr_k

        # Sparsity
        W = final_layer.weight.data
        nnz = int((W.abs() > 1e-5).sum().item())
        total = int(W.numel())
        sparsity_pct_nonzero = nnz / total

        # Top-5 concept indices for attacked target classes (qualitative)
        top5_concepts: Dict[str, List[int]] = {}
        for src, tgt in result["attack_pairs"]:
            w_tgt = W[tgt].abs()
            top5 = w_tgt.topk(min(5, w_tgt.shape[0])).indices.tolist()
            top5_concepts[f"target_{tgt}"] = top5

        return {
            "test_accuracy": test_accuracy,
            "clean_accuracy": clean_accuracy,
            "attack_success_rates": attack_success_rates,
            "per_class_accuracy": {k: v for k, v in per_class_acc.items() if v is not None},
            "per_class_degradation": per_class_degradation,
            "sparsity_nnz": nnz,
            "sparsity_total": total,
            "sparsity_pct_nonzero": sparsity_pct_nonzero,
            "top5_concepts_attacked_targets": top5_concepts,
        }

    def compute_detection_metrics(self, result: dict) -> dict:
        """
        Compute detection precision/recall from the detection log.

        True malicious = clients listed in result["attack_pairs"] (indices 0..n_malicious-1).
        """
        detection_log = result.get("detection_log", [])
        n_malicious = result["n_malicious"]
        true_malicious = set(range(n_malicious))

        if not detection_log:
            return {"precision": None, "recall": None, "n_detection_rounds": 0}

        tp_total, fp_total, fn_total = 0, 0, 0
        for entry in detection_log:
            flagged = set(entry.get("flagged_clients", []))
            tp = len(flagged & true_malicious)
            fp = len(flagged - true_malicious)
            fn = len(true_malicious - flagged)
            tp_total += tp
            fp_total += fp
            fn_total += fn

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else None
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else None

        return {
            "precision": precision,
            "recall": recall,
            "n_detection_rounds": len(detection_log),
            "tp_total": tp_total,
            "fp_total": fp_total,
            "fn_total": fn_total,
        }

    # ------------------------------------------------------------------
    # Full sweep
    # ------------------------------------------------------------------

    def run_sweep(self) -> Dict[str, dict]:
        results: Dict[str, dict] = {}

        for n_malicious in self.n_malicious_list:
            for defense_mode in self.defense_modes:
                # Skip defense modes when there's no attack
                if n_malicious == 0 and defense_mode != "none":
                    continue
                key = f"n{n_malicious}_{defense_mode}"
                result_path = os.path.join(self.output_dir, f"{key}_result.json")
                if os.path.exists(result_path):
                    print(f"[sweep] {key}: loading cached result")
                    with open(result_path) as f:
                        results[key] = json.load(f)
                    continue

                result = self.run_configuration(n_malicious, defense_mode)
                result["detection_metrics"] = self.compute_detection_metrics(result)

                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2, default=_json_default)
                results[key] = result
                print(f"[sweep] {key}: saved to {result_path}")

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def save_results(self, results: Dict[str, dict]) -> None:
        path = os.path.join(self.output_dir, "poisoning_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)
        print(f"Results saved to {path}")

    def print_summary(self, results: Dict[str, dict]) -> None:
        _print_summary_table(results, self.attack_pairs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_predictions(
    test_loader: DataLoader,
    backbone, cbl, normalization, final_layer,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_preds, all_labels = [], []
    backbone.eval(); cbl.eval(); normalization.eval(); final_layer.eval()
    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch[0].to(device), batch[1]
            feats = backbone(imgs)
            concepts = cbl(feats)
            norm_concepts = normalization(concepts)
            logits = final_layer(norm_concepts)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    return torch.cat(all_preds), torch.cat(all_labels)


def _json_default(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (torch.Tensor,)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _print_summary_table(results: Dict[str, dict], attack_pairs: List[Tuple[int, int]]) -> None:
    header = f"{'Config':<30} {'TestAcc':>8} {'CleanAcc':>9} {'ASR_0':>7} {'ASR_1':>7} {'ASR_2':>7} {'Sparsity':>9} {'DetPrec':>8} {'DetRecall':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for key, r in sorted(results.items()):
        em = r.get("eval_metrics", {})
        dm = r.get("detection_metrics", {})
        test_acc = em.get("test_accuracy")
        clean_acc = em.get("clean_accuracy")
        asrs = em.get("attack_success_rates", {})
        sparsity = em.get("sparsity_pct_nonzero")
        prec = dm.get("precision")
        recall = dm.get("recall")

        def _fmt(v, pct=True):
            if v is None:
                return "  N/A"
            return f"{v*100:7.2f}" if pct else f"{v:7.4f}"

        asr_vals = [_fmt(asrs.get(f"{src}_to_{tgt}")) for src, tgt in attack_pairs]
        while len(asr_vals) < 3:
            asr_vals.append("  N/A")

        print(
            f"{key:<30} {_fmt(test_acc):>8} {_fmt(clean_acc):>9} "
            f"{asr_vals[0]:>7} {asr_vals[1]:>7} {asr_vals[2]:>7} "
            f"{_fmt(sparsity):>9} {_fmt(prec):>8} {_fmt(recall):>9}"
        )

    print("=" * len(header))
