#!/usr/bin/env python3
"""Emit one CPU-only Job per NEC~5 arm, all stacked into a single YAML.

Why CPU: Phase 3 trains a (K x C) linear layer over cached concept features.
Measured on places365, 200 FedDualAvg rounds take 2h16m-2h23m at ~2% GPU
utilisation, so the GPU contributes almost nothing and the pod trips the
"<40% for 3 hours" policy for no benefit. A pod that requests no GPU is not
subject to that policy at all.

Why one Job per arm rather than one Job looping over arms: the arms are
independent once their nec50 source is on the PVC, so separate Jobs run in
parallel and each pod is short.

--load_pretrained_vlg skips Phases 1-2, so the Phase-2-on-CPU hang recorded in
the SAGA diagnostics does not apply. The one remaining backbone pass is test
evaluation; on CPU budget an extra 15-30 min per arm.

Config is imported from gen_seed_main so the two cannot drift.

Usage:
    python naut/gen_phase3_cpu.py --datasets places365 --partitions iid \\
        > naut/generated_jobs/submit/08-phase3-cpu.yaml
"""

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_seed_main import CFG, LAMS, SEEDS, CLASSES, COMMON, DATA  # noqa  # noqa: E402

# COMMON hardcodes --device cuda; Phase 3 here is CPU.
COMMON_CPU = COMMON.replace("--device cuda", "--device cpu")

CPU = "16"
MEM = {"cifar10": "16Gi", "cifar100": "16Gi", "cub": "16Gi",
       "places365": "48Gi", "imagenet": "64Gi"}

JOB = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: fedcbm-p3{tag}-{ds}-{ptag}-s{seed}
  namespace: wenglab-interpretable-ai
  labels:
    experiment: fed-lfc-cbm
    task: phase3-cpu
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

              export PIP_CACHE_DIR=/tmp/pip-cache
              mkdir -p $PIP_CACHE_DIR
              pip install --cache-dir $PIP_CACHE_DIR -q ftfy regex tqdm numpy pandas
              pip install --cache-dir $PIP_CACHE_DIR -q git+https://github.com/openai/CLIP.git
              pip install --cache-dir $PIP_CACHE_DIR -q pytorchcv || true
              pip install --cache-dir $PIP_CACHE_DIR -q psutil || true

              # Hand torch the whole CPU request; the default thread count is
              # derived from host cores, not the cgroup limit.
              export OMP_NUM_THREADS={cpu}
              export MKL_NUM_THREADS={cpu}

              cd $WORKSPACE/fedmedcbm
              OUT=/sc-rwx-vol/fedmedcbm/models/SEED_MAIN/{ds}
              D5=$OUT/{ptag}_s{seed}_nec5
              {data}

              if find "$D5" -name metrics.txt 2>/dev/null | grep -q .; then
                echo "[skip] {ds} {ptag}_s{seed}_nec5 already complete"; exit 0
              fi

              PIN=$(find "$OUT/{ptag}_s{seed}_nec50" -name train_concept_features.pt 2>/dev/null | head -1)
              if [ -z "$PIN" ]; then
                echo "ERROR: {ds} {ptag}_s{seed}_nec50 is not on the PVC yet."
                echo "This arm reuses its cached concept features. Run it after that arm lands."
                exit 1
              fi
              PIN=$(dirname "$PIN")

              echo "###### {ds} | {ptag} | seed {seed} | NEC~5 (lam={lam}) | PHASE 3 | CPU ######"
              SAVE=$WORKSPACE/p3
              mkdir -p "$SAVE"
              python main_fed.py --dataset {ds} --concept_file {concepts} {bb} \\
                {flag} --seed {seed} --save_dir "$SAVE" --dual_lam {lam} \\
                --load_pretrained_vlg "$PIN" \\
                {args} {common}

              MD=$(find "$SAVE/fully_trained" -maxdepth 1 -mindepth 1 -type d | head -1)
              if [ -z "$MD" ]; then echo "ERROR: no model dir produced"; exit 1; fi
              mkdir -p "$D5" && cp -r "$MD" "$D5/"
              echo "[ok] -> $D5"

              M=$(find "$D5" -name metrics.txt | head -1)
              A=$(grep -o '"test_accuracy": [0-9.]*' "$M" | head -1 | cut -d' ' -f2)
              NZ=$(grep -o '"Non-zero weights": [0-9]*' "$M" | head -1 | cut -d' ' -f3)
              awk -v a="$A" -v nz="$NZ" -v k={classes} \\
                'BEGIN{{printf "RESULT {ds} {ptag}_s{seed}_nec5  acc=%.2f  NEC=%.2f\\n", a*100, nz/k}}'
          resources:
            requests:
              cpu: "{cpu}"
              memory: "{mem}"{gpuline}
            limits:
              cpu: "{cpu}"
              memory: "{mem}"{gpuline}
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
            sizeLimit: "8Gi"{affinity}
      restartPolicy: Never
"""

HEADER = """# CPU-only Phase 3 (NEC~5) arms, one Job each so they run in parallel.
# Generated by naut/gen_phase3_cpu.py -- do not edit by hand.
#
# No GPU is requested, so these pods are outside the GPU-utilisation policy.
# Each arm needs its matching nec50 arm already on the PVC and exits 1 if not.
#
# MEASURED 2026-09-16: CPU Phase 3 is ~5.6x slower than GPU (3.9 min/round vs
# ~0.7), i.e. ~13h for 200 rounds instead of ~2h20m. Use --gpu unless the pod
# is specifically being kept off the GPU-utilisation policy and 13h is fine.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["places365"])
    ap.add_argument("--partitions", nargs="+", default=["iid", "dir"],
                    choices=["iid", "dir"])
    ap.add_argument("--gpu", action="store_true",
                    help="request a GPU and use --device cuda; CPU Phase 3 measured\n                         ~5.6x slower (3.9 min/round vs 0.7)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="default: whatever gen_seed_main.SEEDS lists")
    a = ap.parse_args()

    tag2part = {"dir": ("dirichlet", "--alpha 0.5"), "iid": ("iid", "--iid")}
    AFF = ('''
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                      - NVIDIA-GeForce-RTX-3090
                      - NVIDIA-A10
                      - NVIDIA-A40
                      - NVIDIA-A100-SXM4-80GB
                      - NVIDIA-A100-PCIE-40GB''')
    tag = "gpu" if a.gpu else "cpu"
    common = COMMON if a.gpu else COMMON_CPU
    gpuline = '\n              nvidia.com/gpu: "1"' if a.gpu else ""
    affinity = AFF if a.gpu else ""
    out = [HEADER]
    for ds in a.datasets:
        cfg = CFG[ds]
        for ptag in a.partitions:
            part, flag = tag2part[ptag]
            lam = LAMS[ds][part][5]
            if lam is None:
                print(f"# {ds} {ptag}: no NEC~5 cell, skipped", file=sys.stderr)
                continue
            for seed in (a.seeds or SEEDS[ds]):
                out.append(JOB.format(
                    ds=ds, ptag=ptag, seed=seed, lam=lam, flag=flag,
                    concepts=cfg["concepts"], bb=cfg["bb"], args=cfg["args"],
                    common=common, tag=tag, gpuline=gpuline, affinity=affinity,
                    data=cfg["data"] or ": # no dataset setup needed",
                    classes=CLASSES[ds], cpu=CPU, mem=MEM[ds],
                ))
    sys.stdout.write("".join(out))


if __name__ == "__main__":
    main()
