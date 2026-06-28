"""Entrypoint for the federated dense (non-CBM) baseline.

Reads a YAML config, merges in path/runtime overrides, and runs train_fed_dense.

Usage:
  python baselines/fed_dense/run.py \
      --config baselines/fed_dense/configs/cifar10.yaml \
      --save_dir /tmp/workspace/saved_models \
      [--device cuda] [--seed 42]
"""

import argparse
import os
import sys
import types

import yaml

# Ensure project root on path
_FED_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _FED_ROOT not in sys.path:
    sys.path.insert(0, _FED_ROOT)

from baselines.fed_dense.train import train_fed_dense


# Defaults for fields a config may omit.
_DEFAULTS = {
    "device": "cuda",
    "num_workers": 2,
    "iid": False,
    "alpha": 0.5,
    "val_split": 0.1,
    "batch_size": 250,
    "extract_batch_size": 256,
    "saga_batch_size": 512,
    "final_rounds": 200,
    "final_epochs": 3,
    "final_lr": 1e-3,
    "final_weight_decay": 1e-4,
    "use_clip_penultimate": False,
    "feature_layer": "layer4",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to baseline YAML config (relative to fed_lfc_cbm/ ok)")
    parser.add_argument("--save_dir", default="saved_models")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    cli = parser.parse_args()

    config_path = cli.config if os.path.isabs(cli.config) else os.path.join(_FED_ROOT, cli.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    merged = dict(_DEFAULTS)
    merged.update(cfg)
    merged["save_dir"] = cli.save_dir
    if cli.device is not None:
        merged["device"] = cli.device
    if cli.seed is not None:
        merged["seed"] = cli.seed

    args = types.SimpleNamespace(**merged)
    print(f"[run] fed_dense config: {config_path}")
    print(f"[run] dataset={args.dataset} backbone={args.backbone} "
          f"clients={args.num_clients} rounds={args.final_rounds}")
    train_fed_dense(args)


if __name__ == "__main__":
    main()
