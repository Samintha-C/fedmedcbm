"""
Poisoning defense evaluation for federated CBM.

Runs a sweep of Phase 3 training under label-flip attacks with different defense
modes and records accuracy, attack success rate, detection precision/recall, and
concept selection corruption.

Requires pre-extracted Phase 1+2 features (use --load_pretrained_vlg pointing to
a directory with backbone.pt, cbl.pt, normalization.pt, and cached concept features).

Example — full sweep:
    python evaluations/poisoning_eval.py \\
        --config naut/generated_jobs/fedcbm-cifar100-c5-fda.yaml \\
        --load_pretrained_vlg /sc-rwx-vol/fedmedcbm/models/cifar100/feddualavg/seed42 \\
        --output_dir /sc-rwx-vol/fedmedcbm/poisoning/cifar100

Single configuration (useful for parallel Kubernetes jobs):
    python evaluations/poisoning_eval.py ... \\
        --single_n_malicious 2 --single_defense client_exclusion
"""

import argparse
import copy
import json
import math
import os
import sys

# Ensure project root is on sys.path
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)


def _parse_args():
    """
    Two-level parsing:
      1. Extract poisoning-specific flags.
      2. Forward everything else to main_fed._parse_args() to get the base training args.
    """
    poison_parser = argparse.ArgumentParser(add_help=False)
    poison_parser.add_argument("--output_dir", type=str, required=True)
    poison_parser.add_argument("--n_malicious", type=int, nargs="+", default=[0, 1, 2, 3])
    poison_parser.add_argument("--defense_modes", type=str, nargs="+",
                               default=["none", "detection_only", "client_exclusion", "reweighting"])
    poison_parser.add_argument("--detection_interval", type=int, default=10)
    poison_parser.add_argument("--suspicion_threshold", type=float, default=1.5)
    poison_parser.add_argument("--reweight_decay", type=float, default=0.7)
    poison_parser.add_argument("--single_n_malicious", type=int, default=None)
    poison_parser.add_argument("--single_defense", type=str, default=None)

    poison_args, remaining = poison_parser.parse_known_args()

    # Build base training args from the remaining argv
    from main_fed import _parse_args as _base_parse
    base_args = _base_parse(remaining)

    return poison_args, base_args


def main():
    poison_args, base_args = _parse_args()

    if poison_args.single_n_malicious is not None and poison_args.single_defense is not None:
        _run_single(base_args, poison_args)
    else:
        _run_sweep(base_args, poison_args)


def _run_single(base_args, poison_args):
    from attacks.experiment import PoisoningExperiment

    n = poison_args.single_n_malicious
    d = poison_args.single_defense

    exp = PoisoningExperiment(
        base_args=base_args,
        output_dir=poison_args.output_dir,
        n_malicious_list=[n],
        defense_modes=[d],
        detection_interval=poison_args.detection_interval,
        suspicion_threshold=poison_args.suspicion_threshold,
        reweight_decay=poison_args.reweight_decay,
    )
    result = exp.run_configuration(n, d)
    result["detection_metrics"] = exp.compute_detection_metrics(result)
    _try_compute_eval_metrics({f"n{n}_{d}": result}, base_args, exp, poison_args.output_dir)

    out_path = os.path.join(poison_args.output_dir, f"n{n}_{d}_result.json")
    _save_json(result, out_path)
    print(f"Result saved to {out_path}")


def _run_sweep(base_args, poison_args):
    from attacks.experiment import PoisoningExperiment

    exp = PoisoningExperiment(
        base_args=base_args,
        output_dir=poison_args.output_dir,
        n_malicious_list=poison_args.n_malicious,
        defense_modes=poison_args.defense_modes,
        detection_interval=poison_args.detection_interval,
        suspicion_threshold=poison_args.suspicion_threshold,
        reweight_decay=poison_args.reweight_decay,
    )

    results = exp.run_sweep()
    _try_compute_eval_metrics(results, base_args, exp, poison_args.output_dir)

    exp.save_results(results)
    exp.print_summary(results)


def _try_compute_eval_metrics(results, base_args, exp, output_dir):
    """
    Compute test-set metrics for each result dict in-place.
    Requires backbone+CBL accessible from --load_pretrained_vlg.
    Silently skips if anything is unavailable.
    """
    try:
        import torch
        from training_utils import set_seed

        pretrained_dir = getattr(base_args, "load_pretrained_vlg", None)
        if not pretrained_dir or not os.path.exists(pretrained_dir):
            print("[poisoning_eval] Skipping eval metrics: --load_pretrained_vlg not set or not found.")
            return

        device = base_args.device if torch.cuda.is_available() else "cpu"
        set_seed(base_args.seed)

        global_model = _load_global_model(pretrained_dir, base_args, device)
        if global_model is None:
            print("[poisoning_eval] Could not load global model; skipping eval metrics.")
            return

        test_loader = _build_test_loader(base_args)
        if test_loader is None:
            print("[poisoning_eval] Could not build test loader; skipping eval metrics.")
            return

        baseline = results.get("n0_none")

        for key, result in results.items():
            if not result.get("model_dir"):
                continue
            try:
                eval_metrics = exp.compute_attack_metrics(
                    result, test_loader, global_model, device, baseline_result=baseline
                )
                result["eval_metrics"] = eval_metrics
                rpath = os.path.join(output_dir, f"{key}_result.json")
                _save_json(result, rpath)
                acc = eval_metrics.get("test_accuracy")
                print(f"[eval] {key}: test_acc={acc:.4f}" if acc is not None else f"[eval] {key}: done")
            except Exception as e:
                print(f"[eval] {key}: failed — {e}")
    except Exception as e:
        print(f"[poisoning_eval] eval metrics unavailable: {e}")


def _load_global_model(pretrained_dir: str, args, device: str):
    try:
        from models.fed_vlgcbm import Backbone, BackboneCLIP, ConceptLayer, NormalizationLayer, FinalLayer, FedVLGCBM
        from training_utils import get_classes
        import torch

        classes = get_classes(args.dataset)
        num_classes = len(classes)

        backbone_name = getattr(args, "backbone", "resnet50")
        saved_args_path = os.path.join(pretrained_dir, "args.json")
        if os.path.exists(saved_args_path):
            with open(saved_args_path) as f:
                saved = json.load(f)
            backbone_name = saved.get("backbone", backbone_name)

        use_clip = "clip" in backbone_name.lower()
        if use_clip:
            backbone = BackboneCLIP(backbone_name, device=device)
        else:
            backbone = Backbone(backbone_name, device=device)

        cbl_path = os.path.join(pretrained_dir, "cbl.pt")
        norm_path = os.path.join(pretrained_dir, "normalization.pt")
        final_path = os.path.join(pretrained_dir, "final.pt")

        if not os.path.exists(cbl_path):
            print(f"[load_model] cbl.pt not found in {pretrained_dir}")
            return None

        cbl_state = torch.load(cbl_path, map_location=device, weights_only=True)
        w_keys = [k for k in cbl_state if "weight" in k]
        num_concepts = cbl_state[w_keys[-1]].shape[0] if w_keys else None
        if num_concepts is None:
            print("[load_model] cannot infer num_concepts from cbl.pt")
            return None

        num_hidden = getattr(args, "cbl_hidden_layers", 0)
        cbl = ConceptLayer(backbone.out_features, num_concepts, num_hidden, device=device)
        cbl.load_state_dict(cbl_state)

        norm = NormalizationLayer(num_concepts, device=device)
        if os.path.exists(norm_path):
            norm.load_state_dict(torch.load(norm_path, map_location=device, weights_only=True))

        final = FinalLayer(num_concepts, num_classes, device=device)
        if os.path.exists(final_path):
            final.load_state_dict(torch.load(final_path, map_location=device, weights_only=True))

        return FedVLGCBM(backbone, cbl, norm, final)
    except Exception as e:
        print(f"[load_model] failed: {e}")
        return None


def _build_test_loader(args):
    try:
        import torch
        from training_utils import get_data, get_preprocess
        backbone_name = getattr(args, "backbone", "resnet50")
        preprocess = get_preprocess(backbone_name)
        data_dir = getattr(args, "data_dir", "./data")
        _, _, test_dataset = get_data(args.dataset, preprocess, data_dir)
        return torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"[build_test_loader] failed: {e}")
        return None


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Not serializable: {type(obj)}")


if __name__ == "__main__":
    main()
