"""Generate seed repeats for every cell of tab:main.

tab:main currently reports one run per cell. This adds two more seeds to each,
giving mean +/- s.d. over three seeds for all 18 cells (both partitions, NEC~5
and NEC~50, five datasets; CUB has no NEC~5 cell because it collapses first).

WHY THIS IS 20 FULL RUNS AND NOT 36
-----------------------------------
A genuine seed repeat has to redo Phase 1: the seed sets the data partition and
the CBL initialization, so --load_pretrained_vlg cannot be used to make a
*different* seed cheap. But within one (dataset, partition, seed) the two NEC
cells differ only in the Phase-3 lambda, so Phase 1 need only run once. Each
arm below therefore trains the full pipeline at the NEC~50 lambda, saves the
concept features, and then runs the NEC~5 lambda as a Phase-3-only job off that
same CBL. That is 20 full pipelines + 16 cheap arms instead of 36 full ones,
which matters mostly for ImageNet and Places365.

Note the existing dlamsweep jobs do NOT do this -- they retrain Phase 1 for
every lambda. Nothing is wrong with those results; this is just cheaper.

CONFIGS
-------
Every per-dataset hyperparameter below was extracted from the corresponding
block of fedcbm-dlamsweep-fda.yaml / fedcbm-dlamsweep-iid-fda.yaml, including
the commented-out blocks, rather than assumed. They differ more than expected:
ImageNet uses val_split 0.05, cbl_lr 0.01, cbl_epochs 7, cbl_pos_weight 8000;
Places365 uses batch 128 and pos_weight 50; CUB uses resnet18_cub with
feature_layer features.final_pool and cbl_epochs 15. IID and Dirichlet differ
only by --iid versus --alpha 0.5.

LAMBDAS
-------
Taken from the vlg-collect-nec run of 2026-08-30: for each cell, the lambda
whose achieved NEC was nearest the target.

Usage:
    python naut/gen_seed_main.py > naut/generated_jobs/fedcbm-seed-main.yaml
"""

DATA = "/sc-rwx-vol/fedmedcbm/datasets"

# Per-dataset: base seed already run, plus the two to add.
# CIFAR-100's existing runs use seed 31; every other dataset uses 42.
SEEDS = {
    "cifar10":   [31, 7],
    "cifar100":  [42, 7],
    "cub":       [31, 7],
    "places365": [31, 7],
    "imagenet":  [31, 7],
}

# lambda for (partition, NEC target). None = cell does not exist.
LAMS = {
    "cifar10":   {"dirichlet": {50: "0.0013",    5: "0.022"},
                  "iid":       {50: "0.0013",    5: "0.022"}},
    "cifar100":  {"dirichlet": {50: "0.00086",   5: "0.01"},
                  "iid":       {50: "0.00069",   5: "0.0088"}},
    # CUB collapses below NEC~9, so there is no NEC~5 cell to seed.
    "cub":       {"dirichlet": {50: "0.003",     5: None},
                  "iid":       {50: "0.003",     5: None}},
    "places365": {"dirichlet": {50: "0.0001",    5: "0.001"},
                  "iid":       {50: "0.000122",  5: "0.001"}},
    "imagenet":  {"dirichlet": {50: "0.000077",  5: "0.0005"},
                  "iid":       {50: "0.000091",  5: "0.0012"}},
}

CFG = {
    "cifar10": dict(
        concepts="concept_files/cifar10_filtered.txt",
        bb="--backbone clip_RN50 --use_clip_penultimate --clip_name RN50",
        args="--val_split 0.1 --batch_size 32 --cbl_lr 5e-4 --cbl_epochs 5 "
             "--cbl_batch_size 32 --cbl_pos_weight 0.2 --num_workers 2",
        data="", check="", mem="16Gi", dshm="6Gi"),
    "cifar100": dict(
        concepts="concept_files/cifar100_filtered.txt",
        bb="--backbone resnet50 --clip_name ViT-B/16 --feature_layer layer4",
        args="--val_split 0.1 --batch_size 32 --cbl_lr 5e-4 --cbl_epochs 3 "
             "--cbl_batch_size 32 --cbl_pos_weight 0.2 --num_workers 2",
        data="", check="", mem="16Gi", dshm="6Gi"),
    "cub": dict(
        concepts="concept_files/cub_filtered.txt",
        bb="--backbone resnet18_cub --clip_name ViT-B/16 --feature_layer features.final_pool",
        args="--val_split 0.1 --batch_size 32 --cbl_lr 5e-4 --cbl_epochs 15 "
             "--cbl_batch_size 32 --cbl_pos_weight 0.2 --num_workers 2",
        data=f"mkdir -p data && ln -sfn {DATA}/CUB data/CUB",
        check="data/CUB/test", mem="16Gi", dshm="6Gi"),
    "places365": dict(
        concepts="concept_files/places365_filtered.txt",
        bb="--backbone resnet50 --clip_name ViT-B/16 --feature_layer layer4",
        args="--val_split 0.05 --batch_size 128 --cbl_lr 5e-4 --cbl_epochs 4 "
             "--cbl_batch_size 128 --cbl_pos_weight 50 --num_workers 4",
        data=f"export PLACES365_ROOT={DATA}/places365",
        check=f"{DATA}/places365/data_256_standard", mem="64Gi", dshm="8Gi"),
    "imagenet": dict(
        concepts="concept_files/imagenet_filtered.txt",
        bb="--backbone resnet50 --clip_name ViT-B/16 --feature_layer layer4",
        args="--val_split 0.05 --batch_size 64 --cbl_lr 0.01 --cbl_epochs 7 "
             "--cbl_batch_size 64 --cbl_pos_weight 8000.0 --num_workers 4",
        data=f"export IMAGENET_ROOT={DATA}/imagenet_kaggle_download/ILSVRC/Data/CLS-LOC",
        check=f"{DATA}/imagenet_kaggle_download/ILSVRC/Data/CLS-LOC/val",
        mem="64Gi", dshm="8Gi"),
}

COMMON = ("--num_clients 5 --num_rounds 5 --local_epochs 5 --lr 1e-3 "
          "--weight_decay 1e-5 --device cuda --use_vlg "
          "--annotation_dir /sc-rwx-vol/fedmedcbm/annotations "
          "--annotation_cache_dir /sc-rwx-vol/fedmedcbm/annotation_cache "
          "--log_dir /sc-rwx-vol/fedmedcbm/job_logs "
          "--cbl_loss_type bce --cbl_optimizer adam --cbl_hidden_layers 0 "
          "--cbl_twoway_tp 4.0 --cbl_bb_lr_rate 1.0 --ortho_lambda 100 "
          "--final_layer_method feddualavg --final_rounds 200 --final_epochs 3 "
          "--final_lr 1e-3 --no_nec_eval")

HEADER = """# Seed repeats for every cell of tab:main -- two extra seeds per cell.
#
# GENERATED BY naut/gen_seed_main.py -- edit that, not this file.
#
# One Job per dataset. Within a Job, each (partition, seed) trains the full
# pipeline once at the NEC~50 lambda and then reuses that CBL for the NEC~5
# lambda via --load_pretrained_vlg, so Phase 1 runs 20 times rather than 36.
#
# Seeds: existing runs use 42 (31 for CIFAR-100); this adds the other two of
# {42, 31, 7} so every cell ends with three.
#
# CUB has no NEC~5 arm: the model collapses below NEC~9 (29.5% at NEC~1.4), so
# that cell is a dash in tab:main and there is nothing to seed.
#
# RESUMABLE: each arm skips itself if its output already has metrics.txt.
# Output: /sc-rwx-vol/fedmedcbm/models/SEED_MAIN/<dataset>/<partition>_s<seed>_nec<target>/
"""

JOB = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: fedcbm-seedmain-{ds}
  namespace: wenglab-interpretable-ai
  labels:
    experiment: fed-lfc-cbm
    task: seed-main
    dataset: {ds}
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

              cd $WORKSPACE/fedmedcbm
              DS={ds}
              OUT=/sc-rwx-vol/fedmedcbm/models/SEED_MAIN/$DS
              mkdir -p $OUT

              {data}
              CHK="{check}"
              if [ -n "$CHK" ] && [ ! -e "$CHK" ]; then
                echo "ERROR: raw images not reachable at $CHK"; exit 1
              fi

              # $1 partition-flag  $2 tag  $3 seed  $4 lam50  $5 lam5
              run_pair () {{
                PART="$1"; PTAG="$2"; SEED="$3"; L50="$4"; L5="$5"

                D50=$OUT/${{PTAG}}_s${{SEED}}_nec50
                if find "$D50" -name metrics.txt 2>/dev/null | grep -q .; then
                  echo "[skip] $DS/${{PTAG}}_s${{SEED}}_nec50"
                else
                  echo ""
                  echo "############ $DS | $PTAG | seed $SEED | NEC~50 (lam=$L50) | FULL ############"
                  SAVE=$WORKSPACE/full_${{PTAG}}_${{SEED}}
                  mkdir -p "$SAVE"
                  set +e
                  python main_fed.py --dataset $DS --concept_file {concepts} {bb} \\
                    $PART --seed $SEED --save_dir "$SAVE" --dual_lam $L50 \\
                    {args} {common}
                  RC=$?; set -e
                  if [ $RC -ne 0 ]; then echo "[FAIL] $DS/$PTAG/s$SEED/nec50 rc=$RC"; return 0; fi
                  MD=$(find "$SAVE/fully_trained" -maxdepth 1 -mindepth 1 -type d | head -1)
                  if [ -z "$MD" ]; then echo "[FAIL] $DS/$PTAG/s$SEED no model dir"; return 0; fi
                  mkdir -p "$D50" && cp -r "$MD" "$D50/"
                  echo "[ok] -> $D50"
                fi

                # NEC~5 reuses the CBL just trained: same seed, same partition,
                # only the Phase-3 lambda differs, so Phase 1 need not repeat.
                if [ -z "$L5" ]; then return 0; fi
                D5=$OUT/${{PTAG}}_s${{SEED}}_nec5
                if find "$D5" -name metrics.txt 2>/dev/null | grep -q .; then
                  echo "[skip] $DS/${{PTAG}}_s${{SEED}}_nec5"; return 0
                fi
                PIN=$(find "$D50" -name train_concept_features.pt 2>/dev/null | head -1)
                if [ -z "$PIN" ]; then
                  echo "[FAIL] $DS/$PTAG/s$SEED: no cached features to reuse for NEC~5"; return 0
                fi
                PIN=$(dirname "$PIN")
                echo ""
                echo "############ $DS | $PTAG | seed $SEED | NEC~5 (lam=$L5) | PHASE 3 ############"
                SAVE5=$WORKSPACE/p3_${{PTAG}}_${{SEED}}
                mkdir -p "$SAVE5"
                set +e
                python main_fed.py --dataset $DS --concept_file {concepts} {bb} \\
                  $PART --seed $SEED --save_dir "$SAVE5" --dual_lam $L5 \\
                  --load_pretrained_vlg "$PIN" \\
                  {args} {common}
                RC=$?; set -e
                if [ $RC -ne 0 ]; then echo "[FAIL] $DS/$PTAG/s$SEED/nec5 rc=$RC"; return 0; fi
                MD5=$(find "$SAVE5/fully_trained" -maxdepth 1 -mindepth 1 -type d | head -1)
                if [ -z "$MD5" ]; then echo "[FAIL] $DS/$PTAG/s$SEED/nec5 no model dir"; return 0; fi
                mkdir -p "$D5" && cp -r "$MD5" "$D5/"
                echo "[ok] -> $D5"
                rm -rf "$SAVE5"
              }}

{calls}
              echo ""
              echo "=========== SUMMARY: $DS ==========="
              NDONE=$(find $OUT -name metrics.txt 2>/dev/null | wc -l)
              for D in $(ls -d $OUT/*/ 2>/dev/null | sort); do
                M=$(find "$D" -name metrics.txt | head -1); [ -z "$M" ] && continue
                A=$(grep -o '"test_accuracy": [0-9.]*' "$M" | head -1 | cut -d' ' -f2)
                NZ=$(grep -o '"Non-zero weights": [0-9]*' "$M" | head -1 | cut -d' ' -f3)
                awk -v n="$(basename $D)" -v a="$A" -v nz="$NZ" -v k={classes} \\
                  'BEGIN{{printf "  %-28s acc=%6.2f  NEC=%8.2f\\n", n, a*100, nz/k}}'
              done
              echo "$NDONE/{ncells} cells have results."
              if [ "$NDONE" -eq 0 ]; then echo "ERROR: no cell produced results"; exit 1; fi
          resources:
            requests:
              cpu: "4"
              memory: "{mem}"
              nvidia.com/gpu: "1"
            limits:
              cpu: "4"
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
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                      - NVIDIA-GeForce-RTX-3090
                      - NVIDIA-A10
                      - NVIDIA-A40
                      - NVIDIA-A100-SXM4-80GB
                      - NVIDIA-A100-PCIE-40GB
      restartPolicy: Never
"""

CLASSES = {"cifar10": 10, "cifar100": 100, "cub": 200, "places365": 365, "imagenet": 998}


def main():
    out = [HEADER]
    for ds, cfg in CFG.items():
        calls, n = [], 0
        for part, ptag, flag in (("dirichlet", "dir", "--alpha 0.5"), ("iid", "iid", "--iid")):
            for seed in SEEDS[ds]:
                l50 = LAMS[ds][part][50]
                l5 = LAMS[ds][part][5] or ""
                calls.append(f'              run_pair "{flag}" {ptag} {seed} {l50} "{l5}"')
                n += 1 + (1 if l5 else 0)
        out.append(JOB.format(
            ds=ds, concepts=cfg["concepts"], bb=cfg["bb"], args=cfg["args"],
            common=COMMON, data=cfg["data"] or ": # no dataset setup needed",
            check=cfg["check"], mem=cfg["mem"], dshm=cfg["dshm"],
            calls="\n".join(calls), classes=CLASSES[ds], ncells=n,
        ))
    print("".join(out), end="")


if __name__ == "__main__":
    main()
