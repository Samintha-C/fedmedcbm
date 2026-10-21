#!/usr/bin/env python3
"""lambda_ortho ablation: one Job per (dataset, lambda_ortho).

Reviewer ask (Overleaf T036.2 / T036.3, 2026-10-20): ablate lambda_ortho, and say why it
is so much larger than the sparsity lambda. The one earlier sweep (March 2026, PVC
eval_results/cbl_eval_dino_p1c5r5-e3-filtered-bce-ol{25,50,75,100,200}.json) was
CIFAR-100 Phase 1 only, over a factor of 8, with no head: mean concept AUC 0.879-0.881
throughout, effective rank of the concept activations 213 -> 242. This one spans 0 and
four decades and adds the head.

Each Job runs Dirichlet alpha=0.5 at the tab:main base seed (CIFAR-100 31, the rest 42),
so the PVC feature cache is reused:
  1. main_fed.py, full pipeline, --ortho_lambda O, Phase 3 at the tab:main NEC~50 lambda.
     O=100 is the tab:main config, so its accuracy should sit near the dlamsweep run
     printed in the summary (on the small datasets it matched to 0.03).
  2. Geometry of the trained CBL rows: L_ortho itself (the quantity lambda multiplies),
     mean / max |cos| between concept directions, effective rank of W.
  3. evaluate_cbl.py: concept AUC / F1 against DINO on the test split, calibration,
     Cohen's d, co-activation (effective rank, mean |corr|), as in the March sweep.
  4. ours_head.py, the head rule, at NEC~5 and NEC~50: FedDualAvg bracketed at
     0.6/1/1.6 x the tab:main lambda (lambda_ortho may move the NEC), nearest NEC kept,
     then the own-support refit (refit_variants.own_100).
Each step skips itself if its output exists, so a rerun only fills gaps.

CIFAR-10 / CIFAR-100 / CUB: CPU only. On a GPU these runs used 0.5 CPU and 1-9% GPU;
on 4 cores they took 1.8-2.3 h (CIFAR), 20-55 min (CUB).

Places365 / ImageNet: GPU. Phase 1 there is the C x C Gram product of the ortho penalty
(2543 / 4750 concepts) over 1.7M / 1.2M rows, days on CPU; measured on the posaware runs
(2026-10-15) it holds 44% / 89% GPU. Two changes keep the GPU busy:
  - main_fed.py's own Phase 3 runs 1 round (--final_rounds 1): its 200 rounds sat at 6-17%
    GPU for hours. ours_head.py replicates the 200-round FedDualAvg at 76-96% GPU, so read
    its "fda" lines. The 1-round head only serves ours_head's feature check.
  - evaluate_cbl.py is skipped: two passes over the test JPEGs on NFS (~15 img/s) would
    idle the GPU for 1-2 h. Step 2 still runs.
Expected: Places365 ~4 h, ImageNet ~12 h per Job (Phase 1 ~5 h + the bracketed head).

Requests, from measured use (NRP: CPU 20-200%, RAM 20-150% of request, GPU > 40%):
  CPU jobs  4 cores; 6Gi CIFAR (mean 1.8-2.4, max 3.2 GiB), 4Gi CUB (mean 0.9-1.9, max 2.2)
  GPU jobs  2 cores (~1 used); 48Gi (main_fed Phase 2 peaks 35 / 43.5 GiB, ours_head on
            Places365 42 GiB; Phase 1 sits at 9-16 / 11.3 GiB); >= 24 GB cards (ImageNet's
            head uses 22 GiB of GPU memory)

SKIP: jobs that need not run are written commented out, with the reason above each
(--all writes everything live):
  - CIFAR-10 / CIFAR-100 / CUB: ran 2026-10-21, results in cvpr2027/RESULTS_STATUS.md.
  - Places365 / ImageNet o100: the tab:main config, already on the PVC (dlamsweep run,
    OURS_HEAD_OWN s42, both at the tab:main lambda); o100 reproduced tab:main on all three
    small datasets. Saves ~16 GPU-hours.

Output: /sc-rwx-vol/fedmedcbm/models/ORTHO_SWEEP/<dataset>/o<lambda>/
  <run>/                     main_fed.py run (metrics.txt, cbl.pt, concept features)
  cbl_weight_geometry.json
  cbl_eval_dino.json         (CPU datasets only)
  head/<run>/metrics.json    ours_head.py

Usage:
    python naut/gen_ortho_sweep.py > naut/generated_jobs/fedcbm-orthosweep.yaml
"""

import argparse
import os
import sys

_NAUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _NAUT)
sys.path.insert(0, os.path.join(os.path.dirname(_NAUT), "baselines", "fedcbm_reimpl", "naut"))
from gen_seed_main import CFG, CLASSES, COMMON, LAMS, SEEDS  # noqa: E402
import gen_jobs  # noqa: E402

DATASETS = ("cifar10", "cifar100", "cub", "places365", "imagenet")
GPU_DATASETS = ("places365", "imagenet")
ORTHO = ["0", "10", "100", "1000", "10000"]
# tab:main's base seed: the one of {42, 31, 7} that gen_seed_main did not add.
BASE_SEED = {ds: ({42, 31, 7} - set(SEEDS[ds])).pop() for ds in DATASETS}
MODELS = "/sc-rwx-vol/fedmedcbm/models"
# (cpu, memory, /dev/shm); measured use in the docstring.
SIZE = {"cifar10": ("4", "6Gi", "1Gi"), "cifar100": ("4", "6Gi", "1Gi"), "cub": ("4", "4Gi", "1Gi"),
        "places365": ("2", "48Gi", "4Gi"), "imagenet": ("2", "48Gi", "4Gi")}
SKIP = {ds: (ORTHO, "ran 2026-10-21, results in cvpr2027/RESULTS_STATUS.md")
        for ds in ("cifar10", "cifar100", "cub")}
SKIP.update({ds: (["100"], "tab:main config, already on the PVC (dlamsweep run + OURS_HEAD_OWN s42)")
             for ds in GPU_DATASETS})

assert COMMON.count("--ortho_lambda 100") == 1 and COMMON.count("--device cuda") == 1
assert COMMON.count("--final_rounds 200") == 1
COMMON_CPU = COMMON.replace("--ortho_lambda 100", "--ortho_lambda $O").replace("--device cuda", "--device cpu")
COMMON_GPU = COMMON.replace("--ortho_lambda 100", "--ortho_lambda $O").replace("--final_rounds 200", "--final_rounds 1")

# gen_jobs' GPU affinity (>= 24 GB nvidia.com/gpu products, flaky host excluded), so the two stay in step.
AFFINITY = "      affinity:\n" + gen_jobs.JOB.split("      affinity:\n", 1)[1].split("      restartPolicy: Never", 1)[0]

# Step 2, inline so the job needs nothing that is not already in the repo.
GEOM = ("import json,sys,torch,torch.nn.functional as F; "
        "W=torch.load(sys.argv[1],map_location='cpu')['model.0.weight'].float(); "
        "C=W.shape[0]; I=torch.eye(C); Wn=F.normalize(W,dim=1); G=Wn@Wn.T; off=G[I==0].abs(); "
        "s=torch.linalg.svdvals(W); p=s/s.sum(); "
        "json.dump({'num_concepts':C,'L_ortho':float((G-I).pow(2).mean()),"
        "'mean_abs_cos':float(off.mean()),'max_abs_cos':float(off.max()),"
        "'frac_abs_cos_above_0.5':float((off>0.5).float().mean()),"
        "'eff_rank_W':float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum()))},"
        "open(sys.argv[2],'w'),indent=2)")

EVAL_CPU = """              # 3. concept quality vs DINO on the test split (no train-set pass)
              if [ -e $D/cbl_eval_dino.json ]; then
                echo "[skip] evaluate_cbl.py"
              else
                python evaluate_cbl.py --load_dir $RUN --dataset $DS \\
                  --annotation_dir /sc-rwx-vol/fedmedcbm/annotations --dino_confidence_threshold 0.10 \\
                  --batch_size 128 --num_workers 2 --device cpu \\
                  --run_diagnostics --skip_client_geometry --skip_linear_probe \\
                  --save_results $WORKSPACE/cbl_eval_dino.json \\
                  && cp $WORKSPACE/cbl_eval_dino.json $D/ \\
                  || { echo "[FAIL] evaluate_cbl.py"; NF=$((NF+1)); }
              fi
"""
EVAL_GPU = """              # 3. evaluate_cbl.py not run here: two passes over the test JPEGs on NFS would
              #    idle the GPU for 1-2 h (see naut/gen_ortho_sweep.py).
              echo "[skip] evaluate_cbl.py (GPU dataset)"
"""

FDA_CPU = """              for M in $RUN/metrics.txt $REFM; do
                A=$(grep -o '"test_accuracy": [0-9.]*' $M | head -1 | cut -d' ' -f2)
                NZ=$(grep -o '"Non-zero weights": [0-9]*' $M | head -1 | cut -d' ' -f3)
                awk -v n="$M" -v a="$A" -v nz="$NZ" -v k=$K \\
                  'BEGIN{printf "  FedDualAvg acc=%6.2f  NEC=%6.2f  %s\\n", a*100, nz/k, n}'
              done
              echo "  (second line: tab:main run, ortho_lambda 100, GPU)"
"""
FDA_GPU = """              A=$(grep -o '"test_accuracy": [0-9.]*' $REFM | head -1 | cut -d' ' -f2)
              NZ=$(grep -o '"Non-zero weights": [0-9]*' $REFM | head -1 | cut -d' ' -f3)
              awk -v n="$REFM" -v a="$A" -v nz="$NZ" -v k=$K \\
                'BEGIN{printf "  tab:main (ortho 100) FedDualAvg acc=%6.2f  NEC=%6.2f  %s\\n", a*100, nz/k, n}'
              echo "  main_fed.py ran 1 Phase-3 round here; this run's FedDualAvg is the fda lines above"
"""

HEADER = """# lambda_ortho ablation: {datasets} x lambda_ortho in {lams}.
#
# GENERATED BY naut/gen_ortho_sweep.py -- edit that, not this file.
#
# One Job per (dataset, lambda_ortho); Dirichlet alpha=0.5, tab:main base seed.
# Per Job: main_fed.py -> CBL weight geometry -> evaluate_cbl.py (CPU datasets only) ->
# ours_head.py at NEC~5 / ~50. CIFAR-10/100 and CUB on CPU; Places365 and ImageNet on GPU.
# Jobs that need not run are commented out, with the reason above each (--all: all live).
# RESUMABLE: each step skips itself if its output exists.
# Output: /sc-rwx-vol/fedmedcbm/models/ORTHO_SWEEP/<dataset>/o<lambda>/
"""

JOB = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: fedcbm-ortho-{ds}-o{o}
  namespace: wenglab-interpretable-ai
  labels:
    experiment: fed-lfc-cbm
    task: ortho-sweep
    dataset: {ds}
spec:
  ttlSecondsAfterFinished: 86400
  backoffLimit: 0
  template:
    spec:
      containers:
        - name: {kind}-container
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
              pip install --cache-dir $PIP_CACHE_DIR -q ftfy regex tqdm numpy pandas scikit-learn scipy
              pip install --cache-dir $PIP_CACHE_DIR -q git+https://github.com/openai/CLIP.git
              pip install --cache-dir $PIP_CACHE_DIR -q pytorchcv || true
              pip install --cache-dir $PIP_CACHE_DIR -q psutil || true

              cd $WORKSPACE/fedmedcbm
              DS={ds}
              O={o}
              SEED={seed}
              K={classes}
              D={models}/ORTHO_SWEEP/$DS/o$O
              REFM=$(find {ref} -name metrics.txt -print -quit 2>/dev/null || true)
              mkdir -p $D
              {data}
              {cifar_in}
              CHK="{check}"
              if [ -n "$CHK" ] && [ ! -e "$CHK" ]; then
                echo "ERROR: raw images not reachable at $CHK"; exit 1
              fi
              NF=0

              echo ""
              echo "############ $DS | ortho_lambda $O | dirichlet 0.5 | seed $SEED ############"
              # 1. full pipeline; Phase 3 at the tab:main NEC~50 lambda
              if [ -n "$(find $D -mindepth 2 -maxdepth 2 -name metrics.txt -print -quit)" ]; then
                echo "[skip] main_fed.py"
              else
                SAVE=$WORKSPACE/run
                mkdir -p $SAVE
                python main_fed.py --dataset $DS --concept_file {concepts} {bb} \\
                  --alpha 0.5 --seed $SEED --save_dir $SAVE --dual_lam {lam50} \\
                  {args} {common}
                MD=$(find $SAVE/fully_trained -maxdepth 1 -mindepth 1 -type d | head -1)
                if [ -z "$MD" ]; then echo "ERROR: main_fed.py left no model dir"; exit 1; fi
                cp -r "$MD" $D/
                echo "[ok] main_fed.py -> $D/$(basename $MD)"
              fi
              RUN=$(dirname "$(find $D -mindepth 2 -maxdepth 2 -name cbl.pt -print -quit)")

              # 2. geometry of the CBL rows (what the penalty acts on)
              if [ -e $D/cbl_weight_geometry.json ]; then
                echo "[skip] weight geometry"
              else
                python -c "{geom}" $RUN/cbl.pt $D/cbl_weight_geometry.json \\
                  || {{ echo "[FAIL] weight geometry"; NF=$((NF+1)); }}
              fi

{eval_step}
              # 4. the head rule at NEC~5 and NEC~50
              if [ -n "$(find $D/head -name metrics.json -print -quit 2>/dev/null)" ]; then
                echo "[skip] ours_head.py"
              else
                python baselines/fedcbm_reimpl/ours_head.py --dataset $DS --concept_file {concepts} {bb} \\
                  --val_split {val_split} --num_workers {workers} --seed $SEED --num_clients 5 --alpha 0.5 \\
                  --partitions dirichlet \\
                  --annotation_dir /sc-rwx-vol/fedmedcbm/annotations \\
                  --annotation_cache_dir /sc-rwx-vol/fedmedcbm/annotation_cache \\
                  --save_dir $WORKSPACE/head --device {device} --runs dirichlet=$RUN \\
                  --supports own --refit_rounds 100{lam_targets} \\
                  && mkdir -p $D/head && cp -r $WORKSPACE/head/. $D/head/ \\
                  || {{ echo "[FAIL] ours_head.py"; NF=$((NF+1)); }}
              fi

              echo ""
              echo "=========== SUMMARY: $DS ortho_lambda $O ==========="
{fda}              cat $D/cbl_weight_geometry.json 2>/dev/null || true
              grep -E '^  "(mean_auc_roc|mean_f1_optimal)"|"(effective_rank|mean_abs_pairwise_corr|mean_cohens_d)"' \\
                $D/cbl_eval_dino.json 2>/dev/null || true
              if [ $NF -gt 0 ]; then echo "ERROR: $NF step(s) failed"; exit 1; fi
          resources:
            requests:
              cpu: "{cpu}"
              memory: "{mem}"{gpu}
            limits:
              cpu: "{cpu}"
              memory: "{mem}"{gpu}
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
{affinity}      restartPolicy: Never
"""


def pick(flag, s):
    tok = s.split()
    return tok[tok.index(f"--{flag}") + 1]


def comment_out(job, reason):
    return (f"# [not submitted: {reason}; --all to emit live]\n"
            + "".join(f"# {ln}\n" if ln else "#\n" for ln in job.rstrip("\n").split("\n")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--lams", nargs="+", default=ORTHO)
    ap.add_argument("--all", action="store_true", help="write the SKIP jobs live too")
    a = ap.parse_args()
    out = [HEADER.format(datasets=", ".join(a.datasets), lams="{" + ", ".join(a.lams) + "}")]
    for ds in a.datasets:
        cfg = CFG[ds]
        gpu = ds in GPU_DATASETS
        cpu, mem, dshm = SIZE[ds]
        lam50 = LAMS[ds]["dirichlet"][50]
        cifar_in = (f"mkdir -p /root/.cache {gen_jobs.CIFAR_CACHE} && "
                    f"cp -rn {gen_jobs.CIFAR_CACHE}/. /root/.cache/ 2>/dev/null || true"
                    if ds.startswith("cifar") else ": # not CIFAR")
        for o in a.lams:
            job = JOB.format(
                ds=ds, o=o, seed=BASE_SEED[ds], models=MODELS, lfc=gen_jobs.LFC_COMMIT,
                kind="gpu" if gpu else "cpu", device="cuda" if gpu else "cpu",
                concepts=cfg["concepts"], bb=cfg["bb"], args=cfg["args"],
                common=COMMON_GPU if gpu else COMMON_CPU,
                data=cfg["data"] or ": # no dataset setup needed", check=cfg["check"], cifar_in=cifar_in,
                lam50=lam50, geom=GEOM, lam_targets=gen_jobs.lam_targets(ds),
                eval_step=EVAL_GPU if gpu else EVAL_CPU, fda=FDA_GPU if gpu else FDA_CPU,
                val_split=pick("val_split", cfg["args"]), workers=pick("num_workers", cfg["args"]),
                ref=f"{MODELS}/{ds}/dlamsweep/lam{lam50}", classes=CLASSES[ds],
                cpu=cpu, mem=mem, dshm=dshm,
                gpu='\n              nvidia.com/gpu: "1"' if gpu else "",
                affinity=AFFINITY if gpu else "")
            skip_lams, reason = SKIP.get(ds, ([], ""))
            out.append(comment_out(job, reason) if o in skip_lams and not a.all else job)
    print("".join(out), end="")


if __name__ == "__main__":
    main()
