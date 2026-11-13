#!/usr/bin/env python3
"""Emit centralized VLG-CBM on our CIFAR-100 backbone (ImageNet ResNet-50, layer4), seeds 42 / 31 / 7
(2026-11-12).

Why: VLG-CBM's published CIFAR-100 ANEC-5 (65.73) reads CLIP RN50, while our CIFAR-100 reads ImageNet
ResNet-50 layer4. VLG-CBM reports CIFAR only on CLIP RN50 (paper Table 2, App. B), so the matched
number has to be run. Paired with gen_c100_clip_rn50.py (ours on CLIP RN50).

VLG-CBM's own code and settings: the official repo at 84031a2 (its latest commit, 2025-06-04) and its
configs/cifar100.json with exactly four keys changed, each asserted in the Job:
  backbone clip_RN50 -> resnet50 (torchvision IMAGENET1K_V1, our CIFAR-100 backbone; feature_layer is
  already layer4), use_clip_penultimate true -> false, seed 28527 -> 42 / 31 / 7, save_dir.
Everything else stays VLG-CBM's: hidden layer 1, DINO threshold 0.15, 40 CBL epochs, lr 5e-4,
pos weight 0.2 with auto weight, SAGA settings. The DINO annotations are the ones our pipeline reads
(/sc-rwx-vol/fedmedcbm/annotations, VLG-CBM's {dataset}_{train,val}/{idx}.json layout), as in the
April job naut/generated_jobs/vlgcbm-centralized-rn50-cifar100.yaml (hl 0, threshold 0.10, never
evaluated at NEC 5; its output under models/cifar100/ is left alone).

Per seed: train_cbm.py, then sparse_evaluation.py (the GLM-SAGA path, accuracy at NEC 5, 10, ..., 30;
ACC@5 is the ANEC-5 VLG-CBM publishes), written to anec.csv in the model dir. Output only under
/sc-rwx-vol/fedmedcbm/models/VLGCBM_RN50/cifar100/s<seed>/<model dir>/. RESUMABLE: a seed whose anec.csv
is already on the PVC is skipped.

One GPU pod, seeds in sequence. VLG-CBM's SAGA stages (its own final layer, then the evaluation path)
use little GPU, and sparse_evaluation.py must run on the device that trained the model (cbl.pt is
loaded without map_location), so it cannot move to a CPU job. One pod keeps at most one pod in a
low-GPU phase at a time (NRP: at most 4 violating pods per user).

Sizing: 4 CPU (VLG-CBM's 6 data workers on 4 cores during CBL training; ~1 core during SAGA, still
over the 20% floor), 8Gi RAM (CIFAR-100 in memory, ResNet-50, 45k x 892 concept features), 2Gi
/dev/shm (6 workers x prefetch 2 x 32 images at 224 px is ~0.25 GB).

usage: python3 naut/gen_vlgcbm_rn50.py > naut/generated_jobs/vlgcbm-rn50-c100.yaml
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import gen_vit_b16 as V  # noqa: E402  (node exclusions)
from gen_jobs import CIFAR_CACHE, MODELS  # noqa: E402

VLG_COMMIT = "84031a2"
SEEDS = [42, 31, 7]
OUT = f"{MODELS}/VLGCBM_RN50/cifar100"
CPU, MEM, DSHM = "4", "8Gi", "2Gi"

JOB = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: vlgcbm-rn50-cifar100
  namespace: wenglab-interpretable-ai
  labels:
    experiment: fed-lfc-cbm
    task: vlgcbm-rn50
    dataset: cifar100
spec:
  ttlSecondsAfterFinished: 86400
  backoffLimit: 0
  template:
    spec:
      containers:
        - name: gpu-container
          image: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
          command: ["bash", "-c"]
          args:
            - |
              set -eo pipefail
              apt-get update -q && apt-get install -y -q git libgl1 libglib2.0-0

              WORKSPACE=/tmp/workspace
              mkdir -p $WORKSPACE && cd $WORKSPACE
              git clone https://github.com/Trustworthy-ML-Lab/VLG-CBM.git $WORKSPACE/VLG-CBM
              git -C $WORKSPACE/VLG-CBM checkout -q {commit}
              echo "VLG-CBM commit: $(git -C $WORKSPACE/VLG-CBM rev-parse --short HEAD)"

              export PIP_CACHE_DIR=/tmp/pip-cache
              mkdir -p $PIP_CACHE_DIR
              pip install --cache-dir $PIP_CACHE_DIR -q -r $WORKSPACE/VLG-CBM/requirements.txt
              pip install --cache-dir $PIP_CACHE_DIR -q git+https://github.com/openai/CLIP.git
              python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('GPU:', torch.cuda.get_device_name(0))"

              cd $WORKSPACE/VLG-CBM
              mkdir -p annotations
              ln -sfn /sc-rwx-vol/fedmedcbm/annotations/cifar100_train annotations/cifar100_train
              ln -sfn /sc-rwx-vol/fedmedcbm/annotations/cifar100_val annotations/cifar100_val
              for S in train val; do
                if [ ! -f annotations/cifar100_$S/0.json ]; then echo "ERROR: no annotations/cifar100_$S/0.json"; exit 1; fi
              done
              export DATASET_FOLDER=$WORKSPACE/datasets
              mkdir -p $DATASET_FOLDER
              cp -rn {cifar_cache}/. $DATASET_FOLDER/ 2>/dev/null || true

              OUT={out}
              mkdir -p $OUT
              FAILS=""
              for SEED in {seeds}; do
                D=$OUT/s$SEED
                if find "$D" -name anec.csv 2>/dev/null | grep -q .; then
                  echo "[skip] seed $SEED: already on the PVC"; continue
                fi
                echo "###### VLG-CBM | cifar100 | ResNet-50 layer4 | seed $SEED | train_cbm.py ######"
                python - "$SEED" <<'PYEOF'
              import json, sys
              seed = int(sys.argv[1])
              c = json.load(open("configs/cifar100.json"))
              assert (c["backbone"], c["use_clip_penultimate"], c["feature_layer"]) == ("clip_RN50", True, "layer4"), c
              assert (c["cbl_hidden_layers"], c["cbl_confidence_threshold"], c["cbl_epochs"]) == (1, 0.15, 40), c
              c.update(backbone="resnet50", use_clip_penultimate=False, seed=seed,
                       save_dir=f"saved_models/cifar100_rn50_s{{seed}}")
              json.dump(c, open(f"configs/cifar100_rn50_s{{seed}}.json", "w"), indent=2)
              print(json.dumps(c))
              PYEOF
                set +e
                python train_cbm.py --config configs/cifar100_rn50_s$SEED.json
                RC=$?; set -e
                MD=$(find saved_models/cifar100_rn50_s$SEED -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
                if [ $RC -ne 0 ] || [ -z "$MD" ]; then echo "[FAIL] seed $SEED train_cbm rc=$RC"; FAILS="$FAILS s$SEED-train"; continue; fi

                echo "###### VLG-CBM | seed $SEED | sparse_evaluation.py (ANEC) ######"
                set +e
                python sparse_evaluation.py --load_path "$MD" --result_file "$MD/anec.csv"
                RC=$?; set -e
                if [ $RC -ne 0 ] || [ ! -f "$MD/anec.csv" ]; then echo "[FAIL] seed $SEED sparse_evaluation rc=$RC"; FAILS="$FAILS s$SEED-eval"; fi
                mkdir -p "$D" && cp -r "$MD" "$D/"
                echo "[ok] -> $D"
                if [ -f "$MD/anec.csv" ]; then
                  python -c "import pandas as p; r=p.read_csv('$MD/anec.csv').iloc[-1]; print('RESULT vlgcbm-rn50 cifar100 s$SEED ANEC-5=%.2f ANEC-avg=%.2f' % (100*r['ACC@5'], 100*r['AVGACC']))"
                fi
              done
              if [ -n "$FAILS" ]; then echo "FAILED:$FAILS"; exit 1; fi
          resources:
            requests:
              cpu: "{cpu}"
              memory: "{mem}"
              nvidia.com/gpu: "1"
            limits:
              cpu: "{cpu}"
              memory: "{mem}"
              nvidia.com/gpu: "1"
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
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: NotIn
                    values:
{bad_nodes}
                  - key: topology.kubernetes.io/zone
                    operator: NotIn
                    values:
{bad_zones}
      restartPolicy: Never
"""

HEADER = """# Centralized VLG-CBM on our CIFAR-100 backbone (ImageNet ResNet-50 layer4), seeds {seeds}.
# GENERATED BY naut/gen_vlgcbm_rn50.py -- edit that, not this file.
#
# Code: Trustworthy-ML-Lab/VLG-CBM @ {commit}, configs/cifar100.json with backbone, use_clip_penultimate,
# seed and save_dir changed. One GPU Job, seeds in sequence. RESUMABLE (a seed with anec.csv is skipped).
# Output: {out}/s<seed>/<model dir>/  (train.log, args.txt, cbl.pt, metrics.csv = NEC path, anec.csv)
"""


def main():
    seeds = " ".join(map(str, SEEDS))
    job = JOB.format(commit=VLG_COMMIT, cifar_cache=CIFAR_CACHE, out=OUT, seeds=seeds, cpu=CPU, mem=MEM, dshm=DSHM,
                     bad_nodes="\n".join(f"                      - {n}" for n in V.BAD_NODES),
                     bad_zones="\n".join(f"                      - {z}" for z in V.BAD_ZONES))
    assert "VLGCBM_RN50/cifar100" in job and "centralized_vlgcbm" not in job
    sys.stdout.write(HEADER.format(seeds=seeds, commit=VLG_COMMIT, out=OUT) + job)


if __name__ == "__main__":
    main()
