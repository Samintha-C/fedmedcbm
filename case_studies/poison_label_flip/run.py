"""
Entrypoint for the poisoned-client label-flip case study.

Reads a YAML config, builds the main_fed.py CLI call with poison + snapshot
flags, runs federated training as a subprocess, then invokes diagnostics.

Usage:
  python case_studies/poison_label_flip/run.py \
      --config case_studies/poison_label_flip/configs/cifar10_dog2cat.yaml \
      --save_dir /tmp/workspace/saved_models \
      --annotation_dir /sc-rwx-vol/fedmedcbm/annotations \
      --annotation_cache_dir /sc-rwx-vol/fedmedcbm/annotation_cache \
      [--snapshot_dir /sc-rwx-vol/fedmedcbm/case_studies/cifar10_dog2cat/snapshots] \
      [--out_dir /sc-rwx-vol/fedmedcbm/case_studies/cifar10_dog2cat/diagnostics] \
      [--log_dir /sc-rwx-vol/fedmedcbm/job_logs] \
      [--skip_train]  # reuse existing snapshot, jump straight to diagnostics
"""

import argparse
import json
import os
import subprocess
import sys

import yaml


def _load_concept_names(concept_file: str) -> list:
    with open(concept_file) as f:
        return [line.strip() for line in f if line.strip()]


def _get_class_names(dataset: str, fed_root: str) -> list:
    """Call data_utils.get_classes via a small subprocess to avoid importing
    the full torch + CLIP stack in this lightweight entrypoint."""
    script = (
        "import sys; sys.path.insert(0, '.'); "
        "from data.data_utils import get_classes; "
        f"print('\\n'.join(get_classes('{dataset}')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=fed_root,
    )
    if result.returncode != 0:
        raise RuntimeError(f"get_classes failed:\n{result.stderr}")
    return [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]


def build_train_cmd(cfg: dict, save_dir: str, snapshot_dir: str,
                    annotation_dir: str, annotation_cache_dir: str,
                    log_dir: str, fed_root: str) -> list:
    flip_map_json = json.dumps(cfg["flip_map"])

    cmd = [
        sys.executable, "main_fed.py",
        "--dataset",              cfg["dataset"],
        "--concept_file",         cfg["concept_file"],
        "--backbone",             cfg["backbone"],
        "--clip_name",            cfg["clip_name"],
        "--num_clients",          str(cfg["num_clients"]),
        "--num_rounds",           str(cfg["num_rounds"]),
        "--local_epochs",         str(cfg["local_epochs"]),
        "--batch_size",           str(cfg["batch_size"]),
        "--lr",                   str(cfg["lr"]),
        "--weight_decay",         str(cfg["weight_decay"]),
        "--alpha",                str(cfg["alpha"]),
        "--seed",                 str(cfg["seed"]),
        "--device",               "cuda",
        "--save_dir",             save_dir,
        "--use_vlg",
        "--val_split",            str(cfg["val_split"]),
        "--cbl_loss_type",        cfg["cbl_loss_type"],
        "--cbl_lr",               str(cfg["cbl_lr"]),
        "--cbl_epochs",           str(cfg["cbl_epochs"]),
        "--cbl_batch_size",       str(cfg["cbl_batch_size"]),
        "--cbl_optimizer",        cfg["cbl_optimizer"],
        "--cbl_hidden_layers",    str(cfg["cbl_hidden_layers"]),
        "--cbl_pos_weight",       str(cfg["cbl_pos_weight"]),
        "--cbl_twoway_tp",        str(cfg["cbl_twoway_tp"]),
        "--cbl_bb_lr_rate",       str(cfg["cbl_bb_lr_rate"]),
        "--ortho_lambda",         str(cfg["ortho_lambda"]),
        "--final_layer_method",   cfg["final_layer_method"],
        "--final_rounds",         str(cfg["final_rounds"]),
        "--final_epochs",         str(cfg["final_epochs"]),
        "--final_lr",             str(cfg["final_lr"]),
        "--dual_lam",             str(cfg["dual_lam"]),
        "--dual_eta_s",           str(cfg["dual_eta_s"]),
        "--dual_eta_c",           str(cfg["dual_eta_c"]),
        "--no_nec_eval",
        "--num_workers",          "2",
        # Poison flags
        "--label_flip_client",    str(cfg["adversary_client_id"]),
        "--label_flip_map",       flip_map_json,
        # Snapshot flag
        "--phase3_snapshot_dir",  snapshot_dir,
    ]

    if cfg.get("use_clip_penultimate"):
        cmd.append("--use_clip_penultimate")
    if cfg.get("iid"):
        cmd.append("--iid")
    if annotation_dir:
        cmd += ["--annotation_dir", annotation_dir]
    if annotation_cache_dir:
        cmd += ["--annotation_cache_dir", annotation_cache_dir]
    if log_dir:
        cmd += ["--log_dir", log_dir]

    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to case-study YAML config (relative to fed_lfc_cbm/)")
    parser.add_argument("--save_dir", default="/tmp/workspace/saved_models")
    parser.add_argument("--snapshot_dir", default=None,
                        help="Where to write per-client snapshots. "
                             "Defaults to save_dir/../snapshots/poison_label_flip/")
    parser.add_argument("--out_dir", default=None,
                        help="Where to write diagnostic figures/tables. "
                             "Defaults to snapshot_dir/diagnostics/")
    parser.add_argument("--annotation_dir", default=None)
    parser.add_argument("--annotation_cache_dir", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training; run diagnostics on an existing snapshot_dir.")
    args = parser.parse_args()

    # fed_lfc_cbm root = two levels up from this file
    fed_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    config_path = args.config if os.path.isabs(args.config) else os.path.join(fed_root, args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    snapshot_dir = args.snapshot_dir or os.path.join(
        args.save_dir, "..", "case_studies", "poison_label_flip", "snapshots"
    )
    snapshot_dir = os.path.normpath(snapshot_dir)
    out_dir = args.out_dir or os.path.join(snapshot_dir, "diagnostics")

    if not args.skip_train:
        cmd = build_train_cmd(
            cfg, args.save_dir, snapshot_dir,
            args.annotation_dir, args.annotation_cache_dir,
            args.log_dir, fed_root,
        )
        print(f"\n[run] Training command:\n  {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=fed_root)
        if result.returncode != 0:
            print(f"[run] Training failed with exit code {result.returncode}")
            sys.exit(result.returncode)
        print("[run] Training complete.")

    # ── Diagnostics ─────────────────────────────────────────────────────────
    sys.path.insert(0, fed_root)
    from case_studies.poison_label_flip.diagnose import run_diagnostics

    concept_file = cfg["concept_file"]
    if not os.path.isabs(concept_file):
        concept_file = os.path.join(fed_root, concept_file)
    concept_names = _load_concept_names(concept_file)
    class_names = _get_class_names(cfg["dataset"], fed_root)

    run_diagnostics(
        snapshot_dir=snapshot_dir,
        concept_names=concept_names,
        class_names=class_names,
        num_clients=cfg["num_clients"],
        adversary_client_id=cfg["adversary_client_id"],
        source_class=cfg["source_class"],
        target_class=cfg["target_class"],
        topk=cfg.get("topk_concepts", 5),
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
