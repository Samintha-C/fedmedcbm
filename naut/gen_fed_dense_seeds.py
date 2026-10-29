#!/usr/bin/env python3
"""Emit the seed repeats of the no-concept FedAvg row of Table 2 (X5): baselines/fed_dense on
seeds 42 / 31 / 7, one more pair per dataset, same configs.

The existing row (88.2 / 67.2 / 73.8 / 42.5 / 72.0) is one fed_dense run per dataset
(models/<ds>/baselines/fed_dense/, args.txt: Dirichlet 0.5, seed 42, CIFAR-100 31). fed_dense draws
its own Dirichlet split (train.py), so R7's equal-chunks bug does not apply to it. This adds the
other two seeds of each dataset (gen_seed_main.SEEDS), settings unchanged ("seeds only").

Features come from the main pipeline's backbone-feature cache (--feature_cache_dir, the
--annotation_cache_dir of train_vlg.py): fed_dense draws the same random_split and uses the same
backbone and preprocessing, so the cache holds the features it would extract. Checked 2026-10-29:
the expected file names below exist on the PVC for CIFAR-10 / CIFAR-100 / Places365 / ImageNet at
all three seeds (train split, val split, test set), and locally on CIFAR-100 s31 the cached test
rows equal a fresh extraction to 0.15% (fp16). CUB's cache names do not match (its loader indexes
differently), so CUB extracts as before, which takes minutes. Each Job checks its cache files up
front and stops if one is missing, because a cold Places365 extraction on CPU would take days.

Sizing, CPU only: the head is FedAvg over in-memory features (DataLoader on CPU tensors), which
leaves a GPU mostly idle. Measured locally: a 2048 -> 1000 head with Adam at batch 512 costs
12.7 us per sample on 10 threads, so ImageNet's 200 rounds x 3 epochs x 1.22M rows is ~2.6 h of
compute, Places365 (365 classes, 1.71M rows) ~2.4 h. RAM: Places365's train cache is 7.0 GB fp16 and
slicing it into clients briefly holds two copies (~15 GB peak, ~8 GB after), ImageNet 5.0 GB (~11 GB
peak); CIFAR / CUB hold under 0.2 GB of features (joint-pilot CIFAR pods peaked at 2.9 GB).

Usage:
    python naut/gen_fed_dense_seeds.py > naut/generated_jobs/fed-dense-seeds.yaml
"""

import hashlib
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "baselines", "fedcbm_reimpl", "naut"))
from gen_seed_main import CFG, SEEDS  # noqa: E402
from gen_jobs import CIFAR_CACHE, LFC_COMMIT, MODELS  # noqa: E402

CACHE = "/sc-rwx-vol/fedmedcbm/annotation_cache"
# Rows of each split, for the cache file names (train before the val split, test = "<ds>_val").
SIZES = {"cifar10": (50000, 0.1, 10000), "cifar100": (50000, 0.1, 10000),
         "places365": (1803460, 0.05, 36500), "imagenet": (1281167, 0.05, 50000)}
BACKBONE = {"cifar10": "clip_RN50", "cifar100": "resnet50", "places365": "resnet50", "imagenet": "resnet50"}
# (cpu, memory, /dev/shm); Places365 and ImageNet get one Job per seed.
SIZED = {"cifar10": ("8", "4Gi", "1Gi"), "cifar100": ("8", "4Gi", "1Gi"), "cub": ("8", "4Gi", "1Gi"),
         "places365": ("8", "24Gi", "1Gi"), "imagenet": ("8", "16Gi", "1Gi")}
LARGE = ("places365", "imagenet")


def _h(idx):
    return hashlib.md5(torch.tensor(idx, dtype=torch.long).numpy().tobytes()).hexdigest()[:12]


def cache_files(ds, seed):
    """Cache files fed_dense will read at this seed (DinoConceptDataset naming), or [] for CUB."""
    if ds not in SIZES:
        return []
    n, vs, nt = SIZES[ds]
    nv = int(vs * n)
    tr, va = torch.utils.data.random_split(range(n), [n - nv, nv], generator=torch.Generator().manual_seed(seed))
    bb = BACKBONE[ds]
    return [f"{ds}_train_{bb}_feat_s{_h(list(tr.indices))}.pt", f"{ds}_train_{bb}_feat_s{_h(list(va.indices))}.pt",
            f"{ds}_val_{bb}_feat_s{_h(list(range(nt)))}.pt"]


JOB = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: fed-dense-{ds}{jtag}
  namespace: wenglab-interpretable-ai
  labels:
    experiment: fed-lfc-cbm
    task: fed-dense-seeds
    dataset: {ds}
spec:
  ttlSecondsAfterFinished: 86400
  backoffLimit: 0
  template:
    spec:
      containers:
        - name: cpu-container
          image: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
          command: ["bash", "-c"]
          args:
            - |
              set -eo pipefail
              apt-get update -q && apt-get install -y -q git

              WORKSPACE=/tmp/workspace
              mkdir -p $WORKSPACE && cd $WORKSPACE
              git clone https://github.com/Samintha-C/fedmedcbm.git $WORKSPACE/fedmedcbm
              git clone https://github.com/Trustworthy-ML-Lab/Label-free-CBM.git $WORKSPACE/Label-free-CBM
              git -C $WORKSPACE/Label-free-CBM checkout -q {lfc}

              export OMP_NUM_THREADS={cpu} MKL_NUM_THREADS={cpu}
              export PIP_CACHE_DIR=/tmp/pip-cache
              mkdir -p $PIP_CACHE_DIR
              pip install --cache-dir $PIP_CACHE_DIR -q ftfy regex tqdm numpy pandas scikit-learn pyyaml
              pip install --cache-dir $PIP_CACHE_DIR -q git+https://github.com/openai/CLIP.git
              pip install --cache-dir $PIP_CACHE_DIR -q pytorchcv || true

              cd $WORKSPACE/fedmedcbm
              {data}
              {check}
              mkdir -p /root/.cache && cp -rn {cifar_cache}/. /root/.cache/ 2>/dev/null || true

              for SEED in {seeds}; do
                D={models}/FED_DENSE/{ds}/s$SEED
                if find "$D" -name metrics.txt 2>/dev/null | grep -q .; then
                  echo "[skip] {ds} seed $SEED: already on the PVC"; continue
                fi
                case $SEED in
{cases}
                esac
                for F in $NEED; do
                  if [ ! -f {cache}/$F ]; then echo "ERROR: feature cache {cache}/$F missing"; exit 1; fi
                done
                echo "###### fed_dense | {ds} | dirichlet 0.5 | seed $SEED | CPU ######"
                SAVE=$WORKSPACE/s$SEED
                mkdir -p "$SAVE"
                python baselines/fed_dense/run.py --config baselines/fed_dense/configs/{ds}.yaml \\
                  --save_dir "$SAVE" --device cpu --seed $SEED --feature_cache_dir {cache}
                MD=$(find "$SAVE/fully_trained" -maxdepth 1 -mindepth 1 -type d | head -1)
                if [ -z "$MD" ]; then echo "ERROR: no model dir produced"; exit 1; fi
                mkdir -p "$D" && cp -r "$MD" "$D/"
                A=$(grep -o '"test_accuracy": [0-9.]*' "$MD/metrics.txt" | head -1 | cut -d' ' -f2)
                awk -v a="$A" -v s=$SEED 'BEGIN{{printf "RESULT fed_dense {ds} dirichlet s%s acc=%.2f\\n", s, a*100}}'
              done
          resources:
            requests:
              cpu: "{cpu}"
              memory: "{mem}"
            limits:
              cpu: "{cpu}"
              memory: "{mem}"
          volumeMounts:
            - name: sc-rwx-vol
              mountPath: /sc-rwx-vol
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: sc-rwx-vol
          persistentVolumeClaim:
            claimName: sc-rwx-vol
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: {dshm}
      restartPolicy: Never
"""

HEADER = """# No-concept FedAvg row (Table 2, X5): baselines/fed_dense at the two missing seeds per dataset.
# GENERATED BY naut/gen_fed_dense_seeds.py -- edit that, not this file.
#
# CPU-only. CIFAR-10 / CIFAR-100 / CUB: one Job each, seeds in sequence. Places365 / ImageNet: one
# Job per seed. Features from the main pipeline's cache ({cache}); CUB extracts.
# RESUMABLE: a seed is skipped if its output is already on the PVC.
# Output: {models}/FED_DENSE/<dataset>/s<seed>/<run>/   (the seed-42 runs stay in <ds>/baselines/fed_dense/)
# Needs the fed_dense --feature_cache_dir commit pushed to Samintha-C/fedmedcbm before apply.
"""


def main():
    out = [HEADER.format(cache=CACHE, models=MODELS)]
    for ds, seeds in SEEDS.items():
        cpu, mem, dshm = SIZED[ds]
        groups = [[s] for s in seeds] if ds in LARGE else [seeds]
        for group in groups:
            cases = "\n".join(f"                  {s}) NEED=\"{' '.join(cache_files(ds, s))}\" ;;" for s in group)
            out.append(JOB.format(
                ds=ds, jtag=f"-s{group[0]}" if ds in LARGE else "", seeds=" ".join(map(str, group)),
                cases=cases, cache=CACHE, data=CFG[ds]["data"] or "true",
                check=f'test -e {CFG[ds]["check"]} || {{ echo "ERROR: {CFG[ds]["check"]} missing"; exit 1; }}'
                      if CFG[ds]["check"] else "true",
                models=MODELS, cifar_cache=CIFAR_CACHE, lfc=LFC_COMMIT, cpu=cpu, mem=mem, dshm=dshm,
            ))
    sys.stdout.write("".join(out))


if __name__ == "__main__":
    main()
