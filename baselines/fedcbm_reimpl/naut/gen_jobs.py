"""Generate the FedCBM-reimplementation jobs: one Job per dataset.

Per-dataset backbone, concept file, val split and dataset setup are imported
from naut/gen_seed_main.py, so these runs see exactly the data tab:main does.
Each Job runs seeds 42, 31 and 7 (the three seeds of tab:main), and each seed
produces both partitions (IID and Dirichlet alpha=0.5) from one shared Stage A.

Each Job first runs selftest.py against the F-CM authors' code (pinned commit)
and stops if it fails. CIFAR-10, CIFAR-100 and CUB also compare Stage A on 8
real concepts against the reference; Places365 and ImageNet skip that (the
reference's numpy loop is too slow at their size).

Variants:
  faithful  the reference as published (F-CM's defaults). On our per-image DINO
            labels (1-6% positive) it predicts every concept absent; appendix only.
  balanced  THE REPRESENTATIVE FedCBM BASELINE. Algorithm 1 of Yang & Long
            unchanged (bias-free subgradient SVM, lr 0.01, 200 rounds, CAV
            projection, private heads); each concept's positives and negatives get
            equal total weight, as in the concept sets of FedCBM's source for Broden
            (Abid et al. 2022, CCE: |P| = |N| = 100). That is the only change.
  nec       nec.py on the balanced bank: the paper's C-tuned dense heads
            ({0.01, 0.1, 1.0}) plus NEC-matched sparse heads.
  personal  ours with private per-client heads (ours_personal.py).
  head-own  ours, the head rule: FedDualAvg -> refit on its own support (ours_head.py).
  nec-refit nec, plus each sparse head refit on its support without the penalty.
The earlier "strong" variant (balanced + bias + standardization, lr chosen on
val; FEDCBM_STRONG) is retired; its flags are in each run's args.json.

Usage:
    python baselines/fedcbm_reimpl/naut/gen_jobs.py faithful > baselines/fedcbm_reimpl/naut/fedcbm-reimpl.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py balanced > baselines/fedcbm_reimpl/naut/fedcbm-balanced.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py nec      > baselines/fedcbm_reimpl/naut/fedcbm-nec.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py personal > baselines/fedcbm_reimpl/naut/ours-personal.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py head-own > baselines/fedcbm_reimpl/naut/ours-head-own.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py nec-refit > baselines/fedcbm_reimpl/naut/fedcbm-nec-refit.yaml
"""

import os
import re
import sys

_FED = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_FED, "naut"))
from gen_seed_main import CFG, LAMS  # noqa: E402
from gen_seed_main import SEEDS as SEED_MAIN_SEEDS  # noqa: E402

SEEDS = [42, 31, 7]
MODELS = "/sc-rwx-vol/fedmedcbm/models"
FCM_COMMIT = "31cbb14"
LFC_COMMIT = "fa2869d"
REF_CHECK = {"cifar10": 8, "cifar100": 8, "cub": 8, "places365": 0, "imagenet": 0}
VARIANTS = {
    "faithful": dict(job="fedcbm-reimpl", out="FEDCBM_REIMPL", flags="", refcheck=True),
    "balanced": dict(job="fedcbm-balanced", out="FEDCBM_BALANCED", refcheck=False, cpu_small=True,
                     flags="--svm_balanced --variant_name balanced"),
    # Heads on the balanced run's saved concept bank (nec.py). Needs
    # FEDCBM_BALANCED/<ds>/s<seed>: submit per dataset after its balanced job.
    "nec": dict(job="fedcbm-nec", out="FEDCBM_BALANCED_NEC", refcheck=False, script="nec.py", cpu_small=True,
                flags="--bank_root /sc-rwx-vol/fedmedcbm/models/FEDCBM_BALANCED/$DS "
                      "--nec_targets 5 10 20 50 --head_C_grid 0.01 0.1 1.0"),
    # Ours with private per-client heads (ours_personal.py): our federated CBL from
    # the tab:main run of each (seed, partition), then nec.py's heads. $RUNS is set
    # per seed by run_dirs() below.
    "personal": dict(job="ours-personal", out="OURS_PERSONAL", refcheck=False, script="ours_personal.py",
                     flags="--runs $RUNS --nec_targets 5 10 20 50 --head_C_grid 0.01 0.1 1.0",
                     note="# Ours, personalized: our federated CBL + private sparse heads per client\n"
                          "# (ours_personal.py), the counterpart of the nec variant.\n"),
    # Our improved head (ours_head.py) on the tab:main runs, and on the --cbl_agg posaware
    # reruns (naut/gen_seed_main.py posaware -> SEED_POSAWARE; submit after those finish).
    "head": dict(job="ours-head", out="OURS_HEAD", refcheck=False, script="ours_head.py", cpu_small=True,
                 flags="--runs $RUNS",
                 note="# Ours, improved head: FedDualAvg -> per-class budget -> refit, + logit adjustment.\n"),
    "head-posaware": dict(job="ours-head-posaware", out="OURS_HEAD_POSAWARE", refcheck=False,
                          script="ours_head.py", cpu_small=True, flags="--runs $RUNS",
                          note="# Ours, improved head, on the posaware concept layers (SEED_POSAWARE).\n"),
    # Diagnosis of the ImageNet NEC5 drop (2026-10-15): own vs per-class support, 20 vs 100
    # refit rounds, lam fixed at x1.0. Seed 42 only; apply the imagenet job alone.
    "head-diag": dict(job="ours-head-diag", out="OURS_HEAD_DIAG", refcheck=False, script="ours_head.py",
                      cpu_small=True, seeds=[42],
                      flags="--runs $RUNS --lam_mults 1.0 --refit_rounds 20 100 --supports perclass own",
                      note="# Diagnosis: own vs per-class support, refit 20 vs 100 rounds (seed 42).\n"),
    # The head rule picked from head-diag (2026-10-15): FedDualAvg at the tab:main lambda, then
    # refit on its own support for 100 rounds. Three seeds; seed 42 should reproduce head-diag's own_100.
    "head-own": dict(job="ours-head-own", out="OURS_HEAD_OWN", refcheck=False, script="ours_head.py",
                     cpu_small=True, sized=True,
                     flags="--runs $RUNS --lam_mults 1.0 --refit_rounds 100 --supports own",
                     note="# Ours, the head rule: FedDualAvg -> refit on its own support, 100 rounds, + logit adjustment.\n"),
}
# nec, with each sparse head also refit on its own support without the penalty: the L1 path
# alone collapses at low NEC (ImageNet NEC5 13%). 1000 steps = our refit's 100 rounds x 10 local.
VARIANTS["nec-refit"] = dict(VARIANTS["nec"], job="fedcbm-nec-refit", out="FEDCBM_BALANCED_NEC_REFIT", sized=True,
                             flags=VARIANTS["nec"]["flags"] + " --path_refit_iters 1000",
                             note="# FedCBM-NEC + support refit (core.fit_sparse_path refit_iters), reported next to the path heads.\n")
# Requests for sized=True variants, from measured use of the same scripts (NRP Prometheus,
# 2026-10-14/15): CPU pods used 4-7 of 8 cores and <= 1.5 GiB; GPU pods ~1 core, 94-100% GPU,
# peak RAM 12 GiB (ImageNet), 13-42 GiB (Places365). NRP: CPU 20-200%, RAM 20-150% of request.
# (cpu, memory, /dev/shm); Places365 and ImageNet get one Job per seed.
SIZED = {"cifar10": ("8", "4Gi", "1Gi"), "cifar100": ("8", "4Gi", "1Gi"), "cub": ("8", "3Gi", "1Gi"),
         "places365": ("2", "48Gi", "4Gi"), "imagenet": ("2", "20Gi", "4Gi")}
# CUB has no NEC~5 lambda in gen_seed_main.LAMS; these are the CUB_NEC5 bracket's.
CUB_NEC5_LAMS = {"iid": "0.020", "dirichlet": "0.019"}


def lam_targets(ds):
    out = []
    for part in ("iid", "dirichlet"):
        for nec in (5, 50):
            lam = LAMS[ds][part][nec] or (CUB_NEC5_LAMS[part] if ds == "cub" and nec == 5 else None)
            out.append(f"{part}.{nec}:{lam}")
    return " --lam_targets " + " ".join(out)
# torchvision downloads CIFAR from Toronto at ~80 KB/s (35 min each). Keep one
# copy on the PVC and seed the pod's cache from it; the first job populates it.
CIFAR_CACHE = "/sc-rwx-vol/fedmedcbm/datasets/torchvision_cifar"
# GPUs on Nautilus are often all taken. These datasets' backbone features are
# cached on the PVC, so what is left (Stage A, per-client heads) runs on CPU in
# minutes; variants with cpu_small=True get CPU-only Jobs for them.
CPU_DATASETS = ("cifar10", "cifar100", "cub")
CPU_CORES = 8


def pin_threads(job, cores):
    """torch's thread count pinned to the CPU limit (in a container it
    otherwise sees every host core and oversubscribes)."""
    old = "              export PIP_CACHE_DIR=/tmp/pip-cache\n"
    assert job.count(old) == 1, f"pin_threads: expected 1x {old!r}"
    return job.replace(old, f"              export OMP_NUM_THREADS={cores} MKL_NUM_THREADS={cores}\n" + old)


def to_cpu(job, cores=CPU_CORES):
    """The same Job without a GPU: no GPU request, no GPU-type affinity,
    --device cpu, and threads pinned to the CPU limit."""
    def sub(old, new, count):
        nonlocal job
        assert job.count(old) == count, f"to_cpu: expected {count}x {old!r}"
        job = job.replace(old, new)
    sub('              nvidia.com/gpu: "1"\n', "", 2)
    sub("--device cuda", "--device cpu", 1)
    job = pin_threads(job, cores)
    head, tail = job.split("      affinity:\n", 1)
    job = head + "      restartPolicy: Never" + tail.split("      restartPolicy: Never", 1)[1]
    assert "nvidia" not in job, "to_cpu: GPU reference left"
    return job


def pick(flag, s):
    m = re.search(rf"--{flag}\s+(\S+)", s)
    return m.group(1) if m else None


def run_dirs(ds, posaware=False):
    """Shell `case` setting $RUNS to the tab:main run of each (seed, partition):
    the base seed's NEC~50 run is in dlamsweep[_iid]/lam<lambda>, the two seeds
    added by gen_seed_main.py are in SEED_MAIN/<ds>/<part>_s<seed>_nec50."""
    base = (set(SEEDS) - set(SEED_MAIN_SEEDS[ds])).pop()
    lines = ["", "case $SEED in"]
    for seed in SEEDS:
        if posaware:
            iid = f"{MODELS}/SEED_POSAWARE/{ds}/iid_s{seed}_nec50"
            dr = f"{MODELS}/SEED_POSAWARE/{ds}/dir_s{seed}_nec50"
        elif seed == base:
            iid = f"{MODELS}/{ds}/dlamsweep_iid/lam{LAMS[ds]['iid'][50]}"
            dr = f"{MODELS}/{ds}/dlamsweep/lam{LAMS[ds]['dirichlet'][50]}"
        else:
            iid = f"{MODELS}/SEED_MAIN/{ds}/iid_s{seed}_nec50"
            dr = f"{MODELS}/SEED_MAIN/{ds}/dir_s{seed}_nec50"
        lines.append(f'  {seed}) RUNS="iid={iid} dirichlet={dr}" ;;')
    lines.append("esac")
    return "\n                ".join(lines)


HEADER = """# FedCBM (Yang & Long 2023) on our data and protocol: the F-CM authors'
# reimplementation (github.com/francescoTheSantis/F-CMs, src/fedcbm.py),
# ported to torch in baselines/fedcbm_reimpl/.
#
# GENERATED BY baselines/fedcbm_reimpl/naut/gen_jobs.py -- edit that, not this file.
#
# One Job per dataset; seeds 42, 31, 7; both partitions per seed.
# RESUMABLE: a seed is skipped if its output already has metrics.json.
# Output: /sc-rwx-vol/fedmedcbm/models/FEDCBM_REIMPL/<dataset>/s<seed>/<run>/metrics.json
#   partitions.<iid|dirichlet>.global_test_mean_client_acc   (F-CM reference protocol)
#   partitions.<iid|dirichlet>.personalized_test_acc_weighted (per-client test splits)
"""

JOB = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: {job}-{ds}
  namespace: wenglab-interpretable-ai
  labels:
    experiment: fed-lfc-cbm
    task: {job}
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
              git -C $WORKSPACE/Label-free-CBM checkout -q {lfc}
              git clone https://github.com/francescoTheSantis/F-CMs.git $WORKSPACE/F-CMs
              git -C $WORKSPACE/F-CMs checkout -q {fcm}

              export PIP_CACHE_DIR=/tmp/pip-cache
              mkdir -p $PIP_CACHE_DIR
              pip install --cache-dir $PIP_CACHE_DIR -q ftfy regex tqdm numpy pandas scikit-learn
              pip install --cache-dir $PIP_CACHE_DIR -q git+https://github.com/openai/CLIP.git
              pip install --cache-dir $PIP_CACHE_DIR -q pytorchcv || true
              pip install --cache-dir $PIP_CACHE_DIR -q psutil || true

              cd $WORKSPACE/fedmedcbm
              echo "=== selftest vs F-CM reference ==="
              set +e
              python baselines/fedcbm_reimpl/selftest.py --fcm_repo $WORKSPACE/F-CMs > $WORKSPACE/selftest.log 2>&1
              RC=$?; set -e
              grep -v -iE "warn|STOP: TOTAL|Increase the number|scikit-learn.org|Please also refer|n_iter_i" $WORKSPACE/selftest.log || true
              if [ $RC -ne 0 ]; then echo "ERROR: selftest failed, not running"; exit 1; fi

              DS={ds}
              OUT=/sc-rwx-vol/fedmedcbm/models/{out}/$DS
              mkdir -p $OUT
              {data}
              {cifar_in}
              CHK="{check}"
              if [ -n "$CHK" ] && [ ! -e "$CHK" ]; then
                echo "ERROR: raw images not reachable at $CHK"; exit 1
              fi

              for SEED in {seeds}; do
                D=$OUT/s$SEED
                if find "$D" -name metrics.json 2>/dev/null | grep -q .; then
                  echo "[skip] $DS seed $SEED"; continue
                fi
                echo ""
                echo "############ FedCBM reimpl | $DS | seed $SEED ############"
                SAVE=$WORKSPACE/out_s$SEED{runs}
                set +e
                python baselines/fedcbm_reimpl/{script} --dataset $DS --concept_file {concepts} {bb} \\
                  --val_split {val_split} --num_workers {workers} --seed $SEED --num_clients 5 --alpha 0.5 \\
                  --partitions iid dirichlet \\
                  --annotation_dir /sc-rwx-vol/fedmedcbm/annotations \\
                  --annotation_cache_dir /sc-rwx-vol/fedmedcbm/annotation_cache \\
                  --save_dir "$SAVE" --device cuda {flags}{refcheck}
                RC=$?; set -e
                if [ $RC -ne 0 ]; then echo "[FAIL] $DS seed $SEED rc=$RC"; continue; fi
                MD=$(find "$SAVE" -maxdepth 1 -mindepth 1 -type d | head -1)
                mkdir -p "$D" && cp -r "$MD" "$D/"
                echo "[ok] -> $D"
                {cifar_out}
              done

              echo ""
              echo "=========== SUMMARY: $DS ==========="
              python baselines/fedcbm_reimpl/summarize.py $OUT
              N=$(for S in {seeds}; do find $OUT/s$S -name metrics.json 2>/dev/null; done | wc -l)
              echo "$N/{nseeds} seeds have results."
              if [ "$N" -eq 0 ]; then echo "ERROR: no results"; exit 1; fi
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
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                      # all >= 24 GB; 11-16 GB cards left out (ImageNet's per-client heads).
                      # A40 / A100 need their own resource names on NRP, so they are not listed.
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
      restartPolicy: Never
"""


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "faithful"
    v = VARIANTS[variant]
    note = v.get("note", "")
    if v.get("cpu_small"):
        note += f"# {', '.join(CPU_DATASETS)}: CPU only ({CPU_CORES} cores), see CPU_DATASETS in gen_jobs.py.\n"
    if v.get("sized"):
        note += "# Requests sized from measured use (SIZED in gen_jobs.py); Places365 / ImageNet: one Job per seed.\n"
    out = [HEADER.replace("# One Job per dataset;", f"# Variant: {variant}.\n{note}# One Job per dataset;")
           .replace("FEDCBM_REIMPL", v["out"])]
    for ds, cfg in CFG.items():
        cpu_only = bool(v.get("cpu_small")) and ds in CPU_DATASETS
        k = REF_CHECK[ds] if v["refcheck"] else 0
        cifar = ds.startswith("cifar")
        cifar_in = (f'mkdir -p /root/.cache {CIFAR_CACHE} && cp -rn {CIFAR_CACHE}/. /root/.cache/ 2>/dev/null || true'
                    if cifar else ": # not CIFAR")
        cifar_out = (f'cp -rn /root/.cache/cifar-* {CIFAR_CACHE}/ 2>/dev/null || true'
                     if cifar else ": # not CIFAR")
        refcheck = (f"\\\n                  --fcm_repo $WORKSPACE/F-CMs --reference_check_concepts {k}"
                    if k else "")
        seeds = v.get("seeds", SEEDS)
        cpu, mem, dshm = SIZED[ds] if v.get("sized") else (
            str(CPU_CORES) if cpu_only else cfg.get("cpu", "4"), cfg["mem"], cfg["dshm"])
        groups = [[s] for s in seeds] if v.get("sized") and not cpu_only else [seeds]
        for grp in groups:
            job = JOB.format(
                ds=ds, job=v["job"], out=v["out"],
                flags=v["flags"] + (lam_targets(ds) if variant.startswith("head") else ""), script=v.get("script", "run.py"),
                runs=(run_dirs(ds, posaware=variant == "head-posaware")
                      if variant in ("personal", "head", "head-posaware", "head-diag", "head-own") else ""),
                cifar_in=cifar_in, cifar_out=cifar_out,
                lfc=LFC_COMMIT, fcm=FCM_COMMIT, concepts=cfg["concepts"], bb=cfg["bb"],
                val_split=pick("val_split", cfg["args"]), workers=pick("num_workers", cfg["args"]),
                data=cfg["data"] or ": # no dataset setup needed", check=cfg["check"],
                seeds=" ".join(map(str, grp)), nseeds=len(grp), refcheck=refcheck,
                cpu=cpu, mem=mem, dshm=dshm)
            if len(groups) > 1:  # one Job per seed
                old = f"  name: {v['job']}-{ds}\n"
                assert job.count(old) == 1
                job = job.replace(old, f"  name: {v['job']}-{ds}-s{grp[0]}\n")
            if cpu_only:
                job = to_cpu(job, int(cpu))
            elif v.get("sized"):
                job = pin_threads(job, int(cpu))
            out.append(job)
    print("".join(out), end="")


if __name__ == "__main__":
    main()
