#!/usr/bin/env python3
"""Emit the CLIP ViT-B/16 runs for the supplement (Lily, 2026-11-12): CIFAR-100 and CUB, Dirichlet 0.5,
seeds 42 / 31 / 7, on the sandbox code.

Code: sandbox-fed-cbm at 371153a, copied from the PVC (/sc-rwx-vol/fedmedcbm/sanity_1111/sandbox-fed-cbm,
checked file by file against the commit on 2026-11-12; SANDBOX_COMMIT holds the hash), as the sanity
jobs did (naut/sanity_1111/). Outputs only under /sc-rwx-vol/fedmedcbm/models/VIT_B16/<ds>/.

Per seed, each stage skipped if its output is already on the PVC:
  1. main_fed.py, Phases 1-3, plain FedDualAvg at the dataset's Dirichlet NEC~50 lambda, the backbone
     swapped for CLIP ViT-B/16. Everything else is the dataset's tab:main setting (gen_seed_main.CFG).
     No --use_clip_penultimate: that path patches `attnpool`, which only CLIP ResNets have, so the
     concept layer reads CLIP's 512-d image embedding. Grounding DINO labels are reused as they are.
  2. heads/ours_head.py, the head rule (FedDualAvg -> refit on its own support, 100 rounds) at
     NEC~5 and NEC~20. The lambdas are ResNet's (gen_jobs.dir_lams), bracketed 0.4-2.5x because the
     backbone moves the lambda-to-NEC map. The local smoke test (CIFAR-100 s31, 2026-11-12) gave
     NEC 5.7 at lambda 0.0088, about where ResNet-50 lands, so CIFAR-100 should sit mid-bracket.
  3. heads/dense_prune.py, Table 3's dense FedAvg / FedProx heads and FedAvg + l1 threshold
     (exact NEC 5 and 20, own-support refit). The sandbox dropped the per-class cut + fine-tune part.
  4. baselines/fed_dense/run.py, FedAvg without a concept layer, from the dataset's config with the
     backbone line set to clip_ViT-B/16 (fed_dense's default use_clip_penultimate is already False).

Seeds run in sequence in one Job per dataset: the first head-rule step extracts the test-set features
and all seeds would write the same cache file.

Local smoke test (2026-11-12, macOS, CPU): 74.6% at NEC 5.7, CIFAR-100 s31 Dirichlet, plain FedDualAvg
(ResNet-50 at the same lambda: 64.7 @ 6.0). Extraction ran at ~40 img/s on 14 cores.

Sizing (CPU only, as the sanity CPU retrains and gen_c100_clip.py):
  CIFAR-100: 8 CPU, 8Gi, 3Gi /dev/shm. main_fed's extraction runs batch 256 x 4 workers x prefetch 4,
    ~2.4 GB of shm (it bus-errored on 1Gi in c100-clip), and shm counts against the memory limit.
    CLIP-RN50 CIFAR pods used 59-99% CPU and 1.8-3.9 GB RAM (gen_joint_pilot).
  CUB: 4 CPU, 6Gi, 3Gi. Same extraction shm, a smaller dataset; ours-head-nec1020 CUB pods used
    1.4-2.6 of 8 cores, so 8 would sit under the 20% CPU floor in the head stages.

usage: python3 naut/gen_vit_b16.py > naut/generated_jobs/vit-b16.yaml
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FED = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_FED, "baselines", "fedcbm_reimpl", "naut"))
from gen_seed_main import CFG, LAMS, CLASSES, COMMON  # noqa: E402
from gen_jobs import CIFAR_CACHE, LFC_COMMIT, MODELS, dir_lams  # noqa: E402

SANDBOX = os.path.join(os.path.dirname(_FED), "sandbox-fed-cbm")
SANDBOX_COMMIT = "371153a"
CODE = "/sc-rwx-vol/fedmedcbm/sanity_1111/sandbox-fed-cbm"
OUT_ROOT = f"{MODELS}/VIT_B16"
SEEDS = [42, 31, 7]
DATASETS = ("cifar100", "cub")
BB = "--backbone clip_ViT-B/16 --clip_name ViT-B/16"
LAM_MULTS = "0.4 0.65 1.0 1.6 2.5"
NECS = (5, 20)
SIZE = {"cifar100": ("8", "8Gi", "3Gi"), "cub": ("4", "6Gi", "3Gi")}
BAD_NODES = ["cph-dgx-node2.humboldt.edu", "cph-dgx-node4.humboldt.edu", "cph-dgx-node9.humboldt.edu",
             "k8s-haosu-17.sdsc.optiputer.net", "nautilus-ext-gpu01.fullerton.edu"]  # as gen_sanity.py
BAD_ZONES = ["humboldt"]
# The sandbox's main_fed.py dropped Phase 4, so --no_nec_eval is gone; the rest of COMMON is unchanged.
MAIN_COMMON = COMMON.replace(" --no_nec_eval", "").replace("--device cuda", "--device cpu")
ANN = "--annotation_dir /sc-rwx-vol/fedmedcbm/annotations --annotation_cache_dir /sc-rwx-vol/fedmedcbm/annotation_cache"

JOB = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: vit-b16-{ds}
  namespace: wenglab-interpretable-ai
  labels:
    experiment: fed-lfc-cbm
    task: vit-b16
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
              cp -r {code} $WORKSPACE/fedmedcbm
              COMMIT=$(cat $WORKSPACE/fedmedcbm/SANDBOX_COMMIT)
              echo "sandbox-fed-cbm commit: $COMMIT"
              if [ "$COMMIT" != "{commit}" ]; then echo "ERROR: expected sandbox commit {commit}"; exit 1; fi
              git clone https://github.com/Trustworthy-ML-Lab/Label-free-CBM.git $WORKSPACE/Label-free-CBM
              git -C $WORKSPACE/Label-free-CBM checkout -q {lfc}

              export OMP_NUM_THREADS={cpu} MKL_NUM_THREADS={cpu}
              export PIP_CACHE_DIR=/tmp/pip-cache
              mkdir -p $PIP_CACHE_DIR
              pip install --cache-dir $PIP_CACHE_DIR -q ftfy regex tqdm numpy pandas scikit-learn pyyaml
              pip install --cache-dir $PIP_CACHE_DIR -q git+https://github.com/openai/CLIP.git
              pip install --cache-dir $PIP_CACHE_DIR -q pytorchcv || true
              pip install --cache-dir $PIP_CACHE_DIR -q psutil || true

              cd $WORKSPACE/fedmedcbm
              DS={ds}
              OUT={out_root}/$DS
              mkdir -p $OUT
              {data}
              CHK="{check}"
              if [ -n "$CHK" ] && [ ! -e "$CHK" ]; then echo "ERROR: raw images not reachable at $CHK"; exit 1; fi
              sed -e 's|^backbone: .*|backbone: clip_ViT-B/16|' baselines/fed_dense/configs/$DS.yaml > $WORKSPACE/fed_dense_vit.yaml
              grep -q '^backbone: clip_ViT-B/16$' $WORKSPACE/fed_dense_vit.yaml

              FAILS=""
              for SEED in {seeds}; do
                D=$OUT/dir_s$SEED
                if find "$D" -name metrics.txt 2>/dev/null | grep -q .; then
                  echo "[skip] $DS seed $SEED: run already on the PVC"
                else
                  echo "###### ViT-B/16 | $DS | dirichlet 0.5 | seed $SEED | main_fed (Phases 1-3) | CPU ######"
                  SAVE=$WORKSPACE/full_s$SEED
                  mkdir -p "$SAVE"
                  set +e
                  python main_fed.py --dataset $DS --concept_file {concepts} {bb} \\
                    --alpha 0.5 --seed $SEED --save_dir "$SAVE" --dual_lam {lam50} \\
                    {args} {common}
                  RC=$?; set -e
                  MD=$(find "$SAVE/fully_trained" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
                  if [ $RC -ne 0 ] || [ -z "$MD" ]; then echo "[FAIL] $DS seed $SEED main_fed rc=$RC"; FAILS="$FAILS s$SEED-run"; continue; fi
                  mkdir -p "$D" && cp -r "$MD" "$D/"
                  echo "[ok] -> $D"
                fi
                R=$(find "$D" -name metrics.txt | head -1 | xargs dirname)
                A=$(grep -o '"test_accuracy": [0-9.]*' "$R/metrics.txt" | head -1 | cut -d' ' -f2)
                NZ=$(grep -o '"Non-zero weights": [0-9]*' "$R/metrics.txt" | head -1 | cut -d' ' -f3)
                awk -v a="$A" -v nz="$NZ" -v k={classes} -v s=$SEED \\
                  'BEGIN{{printf "RESULT vit-b16 {ds} dir s%s plain FedDualAvg (NEC~50 lambda) acc=%.2f NEC=%.2f\\n", s, a*100, nz/k}}'

                H=$OUT/dir_s$SEED"_head"
                if find "$H" -name metrics.json 2>/dev/null | grep -q .; then
                  echo "[skip] $DS seed $SEED: head rule already on the PVC"
                else
                  echo "###### ViT-B/16 | $DS | seed $SEED | head rule at NEC~5 / ~20 ######"
                  HSAVE=$WORKSPACE/head_s$SEED
                  mkdir -p "$HSAVE"
                  set +e
                  python heads/ours_head.py --dataset $DS --concept_file {concepts} {bb} \\
                    {head_args} --seed $SEED \\
                    --partitions dirichlet --runs dirichlet=$R --save_dir "$HSAVE" \\
                    --lam_mults {lam_mults} --refit_rounds 100 --supports own \\
                    --lam_targets {lam_targets}
                  RC=$?; set -e
                  MH=$(find "$HSAVE" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
                  if [ $RC -ne 0 ] || [ -z "$MH" ]; then echo "[FAIL] $DS seed $SEED head rule rc=$RC"; FAILS="$FAILS s$SEED-head"
                  else mkdir -p "$H" && cp -r "$MH" "$H/" && echo "[ok] -> $H"; fi
                fi

                T=$OUT/dir_s$SEED"_t3"
                if find "$T" -name metrics.json 2>/dev/null | grep -q .; then
                  echo "[skip] $DS seed $SEED: Table 3 rows already on the PVC"
                else
                  echo "###### ViT-B/16 | $DS | seed $SEED | Table 3: dense FedAvg / FedProx, FedAvg + l1 threshold ######"
                  TSAVE=$WORKSPACE/t3_s$SEED
                  mkdir -p "$TSAVE"
                  set +e
                  python heads/dense_prune.py --dataset $DS --concept_file {concepts} {bb} \\
                    {head_args} --seed $SEED \\
                    --partitions dirichlet --runs dirichlet=$R --save_dir "$TSAVE" \\
                    --parts dense thresh --necs {necs}
                  RC=$?; set -e
                  MT=$(find "$TSAVE" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
                  if [ $RC -ne 0 ] || [ -z "$MT" ]; then echo "[FAIL] $DS seed $SEED Table 3 rc=$RC"; FAILS="$FAILS s$SEED-t3"
                  else mkdir -p "$T" && cp -r "$MT" "$T/" && echo "[ok] -> $T"; fi
                fi

                F=$OUT/dir_s$SEED"_fed_dense"
                if find "$F" -name metrics.txt 2>/dev/null | grep -q .; then
                  echo "[skip] $DS seed $SEED: FedAvg without concepts already on the PVC"
                else
                  echo "###### ViT-B/16 | $DS | seed $SEED | FedAvg without a concept layer ######"
                  FSAVE=$WORKSPACE/fd_s$SEED
                  mkdir -p "$FSAVE"
                  set +e
                  python baselines/fed_dense/run.py --config $WORKSPACE/fed_dense_vit.yaml \\
                    --save_dir "$FSAVE" --device cpu --seed $SEED --feature_cache_dir /sc-rwx-vol/fedmedcbm/annotation_cache
                  RC=$?; set -e
                  MF=$(find "$FSAVE/fully_trained" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
                  if [ $RC -ne 0 ] || [ -z "$MF" ]; then echo "[FAIL] $DS seed $SEED fed_dense rc=$RC"; FAILS="$FAILS s$SEED-fed_dense"
                  else
                    mkdir -p "$F" && cp -r "$MF" "$F/" && echo "[ok] -> $F"
                    A=$(grep -o '"test_accuracy": [0-9.]*' "$MF/metrics.txt" | head -1 | cut -d' ' -f2)
                    awk -v a="$A" -v s=$SEED 'BEGIN{{printf "RESULT vit-b16 {ds} dir s%s FedAvg without concepts acc=%.2f\\n", s, a*100}}'
                  fi
                fi
                {cifar_out}
              done

              echo ""
              echo "=========== SUMMARY: $DS ==========="
              for SEED in {seeds}; do python heads/summarize.py $OUT/dir_s$SEED"_head" || true; python heads/summarize.py $OUT/dir_s$SEED"_t3" || true; done
              if [ -n "$FAILS" ]; then echo "FAILED:$FAILS"; exit 1; fi
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

HEADER = """# CLIP ViT-B/16 runs for the supplement: CIFAR-100 and CUB, Dirichlet 0.5, seeds {seeds}.
# GENERATED BY naut/gen_vit_b16.py -- edit that, not this file.
#
# Code: sandbox-fed-cbm @ {commit}, copied from {code}.
# One CPU-only Job per dataset; seeds run in sequence. RESUMABLE: each stage is skipped if its output exists.
# Output: {out_root}/<ds>/dir_s<seed>/<run>/          main_fed.py run (plain FedDualAvg, NEC~50 lambda)
#         {out_root}/<ds>/dir_s<seed>_head/<run>/     head rule, NEC~5 and ~20 (own_100 entries)
#         {out_root}/<ds>/dir_s<seed>_t3/<run>/       dense FedAvg / FedProx, FedAvg + l1 threshold at NEC 5 and 20
#         {out_root}/<ds>/dir_s<seed>_fed_dense/<run>/ FedAvg without a concept layer
"""


def flags_of(*paths):
    names = set()
    for p in paths:
        names |= set(re.findall(r'add_argument\(\s*"--([a-z0-9_]+)"', open(os.path.join(SANDBOX, p)).read()))
    return names


def check_flags(cmd, allowed, what):
    used = set(re.findall(r"(?<![\w-])--([a-z0-9_]+)", cmd))
    bad = sorted(used - allowed)
    assert not bad, f"{what}: flags not in the sandbox script: {bad}"


def main():
    main_flags = flags_of("main_fed.py")
    head_flags = flags_of("heads/run.py", "heads/ours_head.py")
    t3_flags = flags_of("heads/run.py", "heads/dense_prune.py")
    fd_flags = flags_of("baselines/fed_dense/run.py")
    seeds = " ".join(map(str, SEEDS))
    out = [HEADER.format(seeds=seeds, commit=SANDBOX_COMMIT, code=CODE, out_root=OUT_ROOT)]
    for ds in DATASETS:
        cfg = CFG[ds]
        lams = dir_lams(ds)
        cpu, mem, dshm = SIZE[ds]
        val_split = re.search(r"--val_split (\S+)", cfg["args"]).group(1)
        workers = re.search(r"--num_workers (\S+)", cfg["args"]).group(1)
        head_args = (f"--val_split {val_split} --num_workers {workers} --num_clients 5 --alpha 0.5 {ANN} "
                     f"--device cpu --extract_batch_size 64")
        cifar = ds.startswith("cifar")
        job = JOB.format(
            ds=ds, code=CODE, commit=SANDBOX_COMMIT, lfc=LFC_COMMIT, out_root=OUT_ROOT, seeds=seeds,
            data=(f"mkdir -p /root/.cache && cp -rn {CIFAR_CACHE}/. /root/.cache/ 2>/dev/null || true" if cifar
                  else cfg["data"] or ": # no dataset setup needed"),
            check=cfg["check"],
            cifar_out=(f"cp -rn /root/.cache/cifar-* {CIFAR_CACHE}/ 2>/dev/null || true" if cifar else ": # not CIFAR"),
            concepts=cfg["concepts"], bb=BB, lam50=LAMS[ds]["dirichlet"][50], args=cfg["args"], common=MAIN_COMMON,
            classes=CLASSES[ds], head_args=head_args, lam_mults=LAM_MULTS,
            lam_targets=" ".join(f"dirichlet.{k}:{lams[k]:g}" for k in NECS),
            necs=" ".join(map(str, NECS)), cpu=cpu, mem=mem, dshm=dshm,
            bad_nodes="\n".join(f"                      - {n}" for n in BAD_NODES),
            bad_zones="\n".join(f"                      - {z}" for z in BAD_ZONES),
        )
        blocks = re.findall(r"python (\S+\.py) (.*?)(?:\n\s*RC=)", job, flags=re.S)
        assert len(blocks) == 4, f"{ds}: expected 4 python stages, found {len(blocks)}"
        for script, rest in blocks:
            allowed = {"main_fed.py": main_flags, "heads/ours_head.py": head_flags,
                       "heads/dense_prune.py": t3_flags, "baselines/fed_dense/run.py": fd_flags}[script]
            check_flags(rest, allowed, f"{ds} {script}")
        assert "use_clip_penultimate" not in job and "no_nec_eval" not in job
        out.append(job)
    sys.stdout.write("".join(out))


if __name__ == "__main__":
    main()
