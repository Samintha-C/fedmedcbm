#!/usr/bin/env python3
"""Emit the joint-CBM pilot: one CPU-only Job per (dataset, joint_gamma), stacked into one YAML.

Each Job runs, in one pod:
  1. main_fed.py with --joint_cbm: Phase 1 trains the CBL jointly with a dense head on its raw
     concept logits (joint_gamma * concept BCE + (1 - joint_gamma) * class CE, both FedAvg'd),
     then Phases 2-3 unchanged (plain FedDualAvg at the NEC~50 lambda). metrics.txt also holds
     the joint dense head's test accuracy (joint_cbm.joint_head_test_accuracy).
  2. ours_head.py on that run: the head rule (FedDualAvg -> refit on its own support, 100 rounds)
     at NEC~5 and NEC~50, as OURS_HEAD_OWN. The lambda bracket (0.6 / 1 / 1.6 x) is kept because
     the jointly trained CBL can move the lambda-to-NEC map.
Compare with OURS_HEAD_OWN/<ds>/s42 (dirichlet), the sequential CBL on the same seed and partition.

Why CPU: CIFAR's backbone features are cached on the PVC for seed 42, so no stage keeps a GPU
busy (Phase 3 FedDualAvg runs at 6-24% GPU). The one backbone pass left is test evaluation
(three passes over 10k test images, including the joint head's).

Why the gamma grid: on our trained CBLs the concept BCE is 0.01-0.05 (CIFAR-10 / CIFAR-100,
pos_weight 0.2) while a concept head's class CE is ~0.3-1.5, so gamma 0.5 (F-CM's default
weight) is dominated by the class loss; 0.99 puts the two terms on a similar scale.

Config is imported from gen_seed_main (data, Phase-1 settings, lambdas) and gen_jobs (pins,
CIFAR cache), so the pilot cannot drift from tab:main.

Usage:
    python naut/gen_joint_pilot.py > naut/generated_jobs/joint-pilot.yaml
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "baselines", "fedcbm_reimpl", "naut"))
from gen_seed_main import CFG, LAMS, CLASSES, COMMON  # noqa: E402
from gen_jobs import CIFAR_CACHE, LFC_COMMIT, MODELS, pick  # noqa: E402

COMMON_CPU = COMMON.replace("--device cuda", "--device cpu")
# Measured on local-cbm-indep CIFAR pods (Prometheus, 2026-10-27): peak 1.1-2.8 GiB, up to 8 cores.
# main_fed.py adds five client copies of the backbone in Phase 1 (~2 GiB for CLIP RN50).
CPU, MEM, DSHM = "8", "6Gi", "1Gi"

JOB = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: joint-pilot-{ds}-g{gtag}
  namespace: wenglab-interpretable-ai
  labels:
    experiment: fed-lfc-cbm
    task: joint-pilot
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
              pip install --cache-dir $PIP_CACHE_DIR -q ftfy regex tqdm numpy pandas scikit-learn
              pip install --cache-dir $PIP_CACHE_DIR -q git+https://github.com/openai/CLIP.git
              pip install --cache-dir $PIP_CACHE_DIR -q pytorchcv || true
              pip install --cache-dir $PIP_CACHE_DIR -q psutil || true

              cd $WORKSPACE/fedmedcbm
              OUT={models}/JOINT_PILOT/{ds}
              D=$OUT/g{gtag}_dir_s{seed}
              H=$OUT/g{gtag}_dir_s{seed}_head
              mkdir -p $OUT
              mkdir -p /root/.cache && cp -rn {cifar_cache}/. /root/.cache/ 2>/dev/null || true

              if find "$D" -name metrics.txt 2>/dev/null | grep -q .; then
                echo "[skip] {ds} gamma {g}: joint run already on the PVC"
              else
                echo "###### {ds} | dirichlet 0.5 | seed {seed} | joint CBM gamma {g} | CPU ######"
                SAVE=$WORKSPACE/full
                mkdir -p "$SAVE"
                python main_fed.py --dataset {ds} --concept_file {concepts} {bb} \\
                  --alpha 0.5 --seed {seed} --save_dir "$SAVE" --dual_lam {lam50} \\
                  --joint_cbm --joint_gamma {g} \\
                  {args} {common}
                MD=$(find "$SAVE/fully_trained" -maxdepth 1 -mindepth 1 -type d | head -1)
                if [ -z "$MD" ]; then echo "ERROR: no model dir produced"; exit 1; fi
                mkdir -p "$D" && cp -r "$MD" "$D/"
                echo "[ok] -> $D"
              fi
              M=$(find "$D" -name metrics.txt | head -1)
              A=$(grep -o '"test_accuracy": [0-9.]*' "$M" | head -1 | cut -d' ' -f2)
              J=$(grep -o '"joint_head_test_accuracy": [0-9.]*' "$M" | head -1 | cut -d' ' -f2)
              NZ=$(grep -o '"Non-zero weights": [0-9]*' "$M" | head -1 | cut -d' ' -f3)
              awk -v a="$A" -v j="$J" -v nz="$NZ" -v k={classes} \\
                'BEGIN{{printf "RESULT {ds} g{g} plain FedDualAvg acc=%.2f NEC=%.2f | joint dense head acc=%.2f\\n", a*100, nz/k, j*100}}'

              if find "$H" -name metrics.json 2>/dev/null | grep -q .; then
                echo "[skip] {ds} gamma {g}: head rule already on the PVC"; exit 0
              fi
              echo "###### {ds} | gamma {g} | head rule (FedDualAvg -> own-support refit) | CPU ######"
              HSAVE=$WORKSPACE/head
              mkdir -p "$HSAVE"
              python baselines/fedcbm_reimpl/ours_head.py --dataset {ds} --concept_file {concepts} {bb} \\
                --val_split {val_split} --num_workers {workers} --seed {seed} --num_clients 5 --alpha 0.5 \\
                --partitions dirichlet \\
                --annotation_dir /sc-rwx-vol/fedmedcbm/annotations \\
                --annotation_cache_dir /sc-rwx-vol/fedmedcbm/annotation_cache \\
                --save_dir "$HSAVE" --device cpu --runs dirichlet=$D \\
                --lam_mults 0.6 1.0 1.6 --refit_rounds 100 --supports own \\
                --lam_targets dirichlet.5:{lam5} dirichlet.50:{lam50}
              MH=$(find "$HSAVE" -maxdepth 1 -mindepth 1 -type d | head -1)
              if [ -z "$MH" ]; then echo "ERROR: no head-rule output"; exit 1; fi
              mkdir -p "$H" && cp -r "$MH" "$H/"
              echo "[ok] -> $H"
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

HEADER = """# Joint-CBM pilot: CBL + dense head trained jointly in Phase 1, then our head rule.
# GENERATED BY naut/gen_joint_pilot.py -- edit that, not this file.
#
# One CPU-only Job per (dataset, gamma); seed 42, Dirichlet alpha 0.5. No GPU is requested.
# RESUMABLE: each stage is skipped if its output is already on the PVC.
# Output: {models}/JOINT_PILOT/<dataset>/g<gamma>_dir_s42/<run>/       (main_fed.py run, joint_head.pt)
#         {models}/JOINT_PILOT/<dataset>/g<gamma>_dir_s42_head/<run>/  (ours_head.py metrics.json)
# Needs the --joint_cbm commit pushed to Samintha-C/fedmedcbm before apply.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    ap.add_argument("--gammas", nargs="+", default=["0.5", "0.9", "0.99"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = [HEADER.format(models=MODELS)]
    for ds in a.datasets:
        if not ds.startswith("cifar"):
            sys.exit(f"{ds}: only CIFAR has the cached features and sizing this CPU template assumes")
        cfg = CFG[ds]
        for g in a.gammas:
            out.append(JOB.format(
                ds=ds, g=g, gtag=g.replace(".", ""), seed=a.seed,
                lam50=LAMS[ds]["dirichlet"][50], lam5=LAMS[ds]["dirichlet"][5],
                concepts=cfg["concepts"], bb=cfg["bb"], args=cfg["args"], common=COMMON_CPU,
                val_split=pick("val_split", cfg["args"]), workers=pick("num_workers", cfg["args"]),
                classes=CLASSES[ds], models=MODELS, cifar_cache=CIFAR_CACHE, lfc=LFC_COMMIT,
                cpu=CPU, mem=MEM, dshm=DSHM,
            ))
    sys.stdout.write("".join(out))


if __name__ == "__main__":
    main()
