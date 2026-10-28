#!/usr/bin/env python3
"""Emit the CIFAR-100 CLIP-RN50 rerun (X27 / D6): ours on VLG-CBM's CIFAR-100 backbone, IID.

tab:main runs CIFAR-100 on ImageNet ResNet-50 layer4, while VLG-CBM's CIFAR-100 number (65.7)
uses CLIP-RN50, so "exceeds VLG-CBM on CIFAR-100" crosses backbones (council R1 / R2 / AC).
This reruns ours with CIFAR-10's backbone flags (CLIP-RN50 penultimate, VLG-CBM's choice) and
CIFAR-100's Phase-1 settings, IID only (the VLG-CBM comparison is centralized, so IID is the
matched partition), seeds 42 / 31 / 7.

One CPU-only Job, seeds in sequence. Each seed:
  1. main_fed.py, full pipeline, plain FedDualAvg at the CIFAR-100 IID NEC~50 lambda. The CLIP
     features of CIFAR-100 are probably not cached yet, so the first pass extracts them on CPU.
     Seeds run one after another because all three would write the same test-set cache file
     (the save is not atomic).
  2. ours_head.py on that run: the head rule (FedDualAvg -> refit on its own support, 100
     rounds) at NEC~5 and NEC~50. The backbone change can move the lambda-to-NEC map, so the
     bracket is wide (0.3 to 3.3 x the ResNet-50 lambdas). per-class+refit gives NEC exactly 5
     and 50 whatever the bracket lands on.
Compare with OURS_HEAD_OWN/cifar100 (IID, ResNet-50) and VLG-CBM's CIFAR-100 row.

Caution: an earlier federated CIFAR-100 run on CLIP-RN50 (hidden layers 0, before the head
rule) gave 59 against 65.7 on ResNet-50.

Sizing as gen_joint_pilot.py (8 CPU / 6Gi): its CLIP-RN50 CIFAR-10 pods used 59-99% CPU and
30-65% RAM.

Usage:
    python naut/gen_c100_clip.py > naut/generated_jobs/c100-clip.yaml
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "baselines", "fedcbm_reimpl", "naut"))
from gen_seed_main import CFG, LAMS, CLASSES, COMMON  # noqa: E402
from gen_jobs import CIFAR_CACHE, LFC_COMMIT, MODELS, pick  # noqa: E402

DS = "cifar100"
SEEDS = [42, 31, 7]
BB = CFG["cifar10"]["bb"]  # --backbone clip_RN50 --use_clip_penultimate --clip_name RN50
LAM_MULTS = "0.3 0.55 1.0 1.8 3.3"
COMMON_CPU = COMMON.replace("--device cuda", "--device cpu")
CPU, MEM, DSHM = "8", "6Gi", "1Gi"

JOB = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: c100-clip-iid
  namespace: wenglab-interpretable-ai
  labels:
    experiment: fed-lfc-cbm
    task: c100-clip
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
              OUT={models}/C100_CLIP/{ds}
              mkdir -p $OUT
              mkdir -p /root/.cache && cp -rn {cifar_cache}/. /root/.cache/ 2>/dev/null || true

              for SEED in {seeds}; do
                D=$OUT/iid_s$SEED
                H=$OUT/iid_s$SEED"_head"
                if find "$D" -name metrics.txt 2>/dev/null | grep -q .; then
                  echo "[skip] {ds} seed $SEED: run already on the PVC"
                else
                  echo "###### {ds} | IID | seed $SEED | CLIP-RN50 | CPU ######"
                  SAVE=$WORKSPACE/full_s$SEED
                  mkdir -p "$SAVE"
                  python main_fed.py --dataset {ds} --concept_file {concepts} {bb} \\
                    --iid --seed $SEED --save_dir "$SAVE" --dual_lam {lam50} \\
                    {args} {common}
                  MD=$(find "$SAVE/fully_trained" -maxdepth 1 -mindepth 1 -type d | head -1)
                  if [ -z "$MD" ]; then echo "ERROR: no model dir produced"; exit 1; fi
                  mkdir -p "$D" && cp -r "$MD" "$D/"
                  echo "[ok] -> $D"
                fi
                M=$(find "$D" -name metrics.txt | head -1)
                A=$(grep -o '"test_accuracy": [0-9.]*' "$M" | head -1 | cut -d' ' -f2)
                NZ=$(grep -o '"Non-zero weights": [0-9]*' "$M" | head -1 | cut -d' ' -f3)
                awk -v a="$A" -v nz="$NZ" -v k={classes} -v s=$SEED \\
                  'BEGIN{{printf "RESULT {ds} clip iid s%s plain FedDualAvg acc=%.2f NEC=%.2f\\n", s, a*100, nz/k}}'

                if find "$H" -name metrics.json 2>/dev/null | grep -q .; then
                  echo "[skip] {ds} seed $SEED: head rule already on the PVC"; continue
                fi
                echo "###### {ds} | seed $SEED | head rule (FedDualAvg -> own-support refit) | CPU ######"
                HSAVE=$WORKSPACE/head_s$SEED
                mkdir -p "$HSAVE"
                python baselines/fedcbm_reimpl/ours_head.py --dataset {ds} --concept_file {concepts} {bb} \\
                  --val_split {val_split} --num_workers {workers} --seed $SEED --num_clients 5 --alpha 0.5 \\
                  --partitions iid \\
                  --annotation_dir /sc-rwx-vol/fedmedcbm/annotations \\
                  --annotation_cache_dir /sc-rwx-vol/fedmedcbm/annotation_cache \\
                  --save_dir "$HSAVE" --device cpu --runs iid=$D \\
                  --lam_mults {lam_mults} --refit_rounds 100 --supports own \\
                  --lam_targets iid.5:{lam5} iid.50:{lam50}
                MH=$(find "$HSAVE" -maxdepth 1 -mindepth 1 -type d | head -1)
                if [ -z "$MH" ]; then echo "ERROR: no head-rule output"; exit 1; fi
                mkdir -p "$H" && cp -r "$MH" "$H/"
                echo "[ok] -> $H"
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

HEADER = """# CIFAR-100 on CLIP-RN50 (VLG-CBM's backbone), IID, ours + the head rule (X27 / D6).
# GENERATED BY naut/gen_c100_clip.py -- edit that, not this file.
#
# One CPU-only Job; seeds {seeds} run in sequence (they share the test-set feature cache file).
# RESUMABLE: each stage is skipped if its output is already on the PVC.
# Output: {models}/C100_CLIP/cifar100/iid_s<seed>/<run>/        (main_fed.py run)
#         {models}/C100_CLIP/cifar100/iid_s<seed>_head/<run>/   (ours_head.py metrics.json)
"""


def main():
    cfg = CFG[DS]
    seeds = " ".join(map(str, SEEDS))
    sys.stdout.write(HEADER.format(seeds=seeds, models=MODELS) + JOB.format(
        ds=DS, seeds=seeds, lam50=LAMS[DS]["iid"][50], lam5=LAMS[DS]["iid"][5],
        concepts=cfg["concepts"], bb=BB, args=cfg["args"], common=COMMON_CPU,
        val_split=pick("val_split", cfg["args"]), workers=pick("num_workers", cfg["args"]),
        classes=CLASSES[DS], models=MODELS, cifar_cache=CIFAR_CACHE, lfc=LFC_COMMIT,
        lam_mults=LAM_MULTS, cpu=CPU, mem=MEM, dshm=DSHM,
    ))


if __name__ == "__main__":
    main()
