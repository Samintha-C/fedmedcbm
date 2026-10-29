#!/usr/bin/env python3
"""Emit the joint-CBM row of Table 2 at the chosen gamma 0.99 (X6, decided 2026-10-29): all five
datasets, Dirichlet 0.5, seeds 42 / 31 / 7. Same pipeline as gen_joint_pilot.py.

Three kinds of Job:
  headwide  CIFAR-10 / CIFAR-100: ours_head.py again on the pilot's g099 runs, with a wide lambda
            bracket. The pilot's 0.6 / 1 / 1.6 bracket landed the NEC~5 head at 3.6 to 6.3 and the
            NEC~50 head at 127 to 183 on CIFAR-100: the joint CBL moves the lambda-to-NEC map, most
            at NEC~50 (1.6 x the tab:main lambda still keeps 127 to 137). The pilot's _head
            outputs stay; this writes _headwide.
  cub       CPU, as the pilot: main_fed.py --joint_cbm (plain FedDualAvg at the NEC~50 lambda),
            then the head rule with the wide bracket at NEC~5 and ~50.
  large     Places365 / ImageNet, GPU: main_fed.py --joint_cbm with --final_rounds 1, then the head
            rule at NEC~5 only (Table 2's budget) with the pilot bracket. main_fed's own Phase 3 is
            the "plain FedDualAvg" column, which Table 2 does not use, and it runs at 6 to 24% GPU
            for about 4 h on Places365; ours_head.py redoes FedDualAvg for the head rule anyway.
            One round is safe: the FedDualAvg schedule divides by the number of rounds, not by
            rounds - 1, and ours_head.py's check 4 only compares the run's head with its metrics.txt.
Backbone features and DINO annotations are in the PVC cache for all three seeds of every dataset
(the tab:main runs), so no Job extracts training features.

Sizing (Prometheus, 2026-10-29). joint-pilot CIFAR pods: 8 cores, peak 2.8 to 3.0 GiB of 6Gi.
ours-head-nec1020: ImageNet 96% GPU on average and 19 to 20 GiB peak, Places365 50 to 60% GPU and
21 to 22 GiB, about 1 core. local-cbm-indep ImageNet (local CBL training on cached features, as
joint Phase 1): 77 to 97% GPU, 25 to 46 GiB, 1 to 2 cores. Hence 2 CPU / 48Gi / 1 GPU for the large
Jobs (as local-indep-large), 8 CPU / 4Gi for CUB and the CIFAR head reruns.

Usage:
    python naut/gen_joint_g099.py > naut/generated_jobs/joint-g099.yaml
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "baselines", "fedcbm_reimpl", "naut"))
from gen_seed_main import CFG, LAMS, CLASSES, COMMON  # noqa: E402
from gen_jobs import CIFAR_CACHE, CUB_NEC5_LAMS, LFC_COMMIT, MODELS, pick  # noqa: E402

G, GTAG = "0.99", "099"
SEEDS = [42, 31, 7]
WIDE = "0.6 0.8 1.0 1.25 1.6 2.2 3 4.5 6.5 9"
PILOT = "0.6 1.0 1.6"
COMMON_CPU = COMMON.replace("--device cuda", "--device cpu")
# (cpu, memory, /dev/shm)
SIZED = {"cifar10": ("8", "4Gi", "1Gi"), "cifar100": ("8", "4Gi", "1Gi"), "cub": ("8", "4Gi", "1Gi"),
         "places365": ("2", "48Gi", "4Gi"), "imagenet": ("2", "48Gi", "4Gi")}

SETUP = """          image: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
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
              {data}
              {check}
              mkdir -p /root/.cache && cp -rn {cifar_cache}/. /root/.cache/ 2>/dev/null || true
              OUT={models}/JOINT_PILOT/{ds}
              D=$OUT/g{gtag}_dir_s{seed}
              H=$OUT/g{gtag}_dir_s{seed}_{hsuffix}
"""

MAIN = """
              if find "$D" -name metrics.txt 2>/dev/null | grep -q .; then
                echo "[skip] {ds} seed {seed}: joint run already on the PVC"
              else
                echo "###### {ds} | dirichlet 0.5 | seed {seed} | joint CBM gamma {g} | {dev} ######"
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
              J=$(grep -o '"joint_head_test_accuracy": [0-9.]*' "$M" | head -1 | cut -d' ' -f2)
              awk -v j="$J" 'BEGIN{{printf "RESULT {ds} g{g} s{seed} joint dense head acc=%.2f\\n", j*100}}'
"""

HEAD = """
              if find "$H" -name metrics.json 2>/dev/null | grep -q .; then
                echo "[skip] {ds} seed {seed}: head rule already on the PVC"; exit 0
              fi
              if ! find "$D" -name metrics.txt 2>/dev/null | grep -q .; then
                echo "ERROR: no joint run at $D"; exit 1
              fi
              echo "###### {ds} | gamma {g} | seed {seed} | head rule, lam x ({mults}) | {dev} ######"
              HSAVE=$WORKSPACE/head
              mkdir -p "$HSAVE"
              python baselines/fedcbm_reimpl/ours_head.py --dataset {ds} --concept_file {concepts} {bb} \\
                --val_split {val_split} --num_workers {workers} --seed {seed} --num_clients 5 --alpha 0.5 \\
                --partitions dirichlet \\
                --annotation_dir /sc-rwx-vol/fedmedcbm/annotations \\
                --annotation_cache_dir /sc-rwx-vol/fedmedcbm/annotation_cache \\
                --save_dir "$HSAVE" --device {device} --runs dirichlet=$D \\
                --lam_mults {mults} --refit_rounds 100 --supports own \\
                --lam_targets {targets}
              MH=$(find "$HSAVE" -maxdepth 1 -mindepth 1 -type d | head -1)
              if [ -z "$MH" ]; then echo "ERROR: no head-rule output"; exit 1; fi
              mkdir -p "$H" && cp -r "$MH" "$H/"
              echo "[ok] -> $H"
"""

GPU_RES = """              nvidia.com/gpu: "1"
"""

AFFINITY = """      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                      # as ours-head-nec1020 / local-cbm-indep-large: all >= 24 GB
                      - NVIDIA-GeForce-RTX-3090
                      - NVIDIA-A10
                      - NVIDIA-L40
                      - NVIDIA-L40S
                      - Tesla-V100-SXM2-32GB
                      - NVIDIA-RTX-5000-Ada-Generation
                      - NVIDIA-GeForce-RTX-4090
                      - NVIDIA-L4
                      - Quadro-RTX-6000
                      - NVIDIA-RTX-A5000
                      - NVIDIA-TITAN-RTX
                  - key: kubernetes.io/hostname
                    operator: NotIn
                    values:
                      - nautilus-ext-gpu01.fullerton.edu  # "GPU is lost" admission errors, 2026-10-15
"""

JOB = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: wenglab-interpretable-ai
  labels:
    experiment: fed-lfc-cbm
    task: joint-g099
    dataset: {ds}
spec:
  ttlSecondsAfterFinished: 86400
  backoffLimit: 0
  template:
    spec:
      containers:
        - name: {cname}
{body}          resources:
            requests:
              cpu: "{cpu}"
              memory: "{mem}"
{gpu}            limits:
              cpu: "{cpu}"
              memory: "{mem}"
{gpu}          volumeMounts:
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
{affinity}      restartPolicy: Never
"""

HEADER = """# Joint CBM at gamma 0.99 (X6): Table 2's "Ours, joint CBL" row, Dirichlet 0.5, seeds 42 / 31 / 7.
# GENERATED BY naut/gen_joint_g099.py -- edit that, not this file.
#
# joint-g099-headwide-<cifar>-s<seed>  CPU, head rule again on the pilot's runs, wide lambda bracket
# joint-g099-cub-s<seed>               CPU, joint run + head rule (wide bracket, NEC~5 and ~50)
# joint-g099-<places365|imagenet>-s<seed>  GPU, joint run (Phase 3 cut to 1 round) + head rule at NEC~5
# RESUMABLE: each stage is skipped if its output is already on the PVC.
# Output: {models}/JOINT_PILOT/<dataset>/g099_dir_s<seed>/<run>/           (main_fed.py run)
#         {models}/JOINT_PILOT/<dataset>/g099_dir_s<seed>_head/<run>/      (ours_head.py, CUB / large)
#         {models}/JOINT_PILOT/<cifar>/g099_dir_s<seed>_headwide/<run>/    (ours_head.py, CIFAR rerun)
"""


def lam5(ds):
    return LAMS[ds]["dirichlet"][5] or CUB_NEC5_LAMS["dirichlet"]


def job(ds, seed, kind):
    cfg = CFG[ds]
    cpu, mem, dshm = SIZED[ds]
    large = kind == "large"
    lam50 = LAMS[ds]["dirichlet"][50]
    fmt = dict(ds=ds, seed=seed, g=G, gtag=GTAG, concepts=cfg["concepts"], bb=cfg["bb"], lam50=lam50,
               val_split=pick("val_split", cfg["args"]), workers=pick("num_workers", cfg["args"]),
               dev="GPU" if large else "CPU", device="cuda" if large else "cpu",
               mults=PILOT if large else WIDE,
               targets=(f"dirichlet.5:{lam5(ds)}" if large
                        else f"dirichlet.5:{lam5(ds)} dirichlet.50:{lam50}"))
    body = SETUP.format(lfc=LFC_COMMIT, cpu=cpu, data=cfg["data"] or "true",
                        check=(f'test -e {cfg["check"]} || {{ echo "ERROR: {cfg["check"]} missing"; exit 1; }}'
                               if cfg["check"] else "true"),
                        cifar_cache=CIFAR_CACHE, models=MODELS, ds=ds, gtag=GTAG, seed=seed,
                        hsuffix="headwide" if kind == "headwide" else "head")
    if kind != "headwide":
        common = (COMMON + " --final_rounds 1") if large else COMMON_CPU
        body += MAIN.format(args=cfg["args"], common=common, classes=CLASSES[ds], **fmt)
    body += HEAD.format(**fmt)
    name = f"joint-g{GTAG}-headwide-{ds}-s{seed}" if kind == "headwide" else f"joint-g{GTAG}-{ds}-s{seed}"
    return JOB.format(name=name, ds=ds, cname="gpu-container" if large else "cpu-container", body=body,
                      cpu=cpu, mem=mem, dshm=dshm, gpu=GPU_RES if large else "",
                      affinity=AFFINITY if large else "")


def main():
    out = [HEADER.format(models=MODELS)]
    for ds, kind in (("cifar10", "headwide"), ("cifar100", "headwide"), ("cub", "cub"),
                     ("places365", "large"), ("imagenet", "large")):
        for seed in SEEDS:
            out.append(job(ds, seed, kind))
    sys.stdout.write("".join(out))


if __name__ == "__main__":
    main()
