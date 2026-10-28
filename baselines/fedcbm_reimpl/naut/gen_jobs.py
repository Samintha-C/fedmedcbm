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
  head-nec1020  head-own at NEC~10 and ~20 (lambdas interpolated between tab:main's NEC~5 / ~50).
  nec-refit nec, plus each sparse head refit on its support without the penalty.
  head-alpha / nec-alpha  head-own / nec-refit per client across Dirichlet alpha (alpha-sweep runs).
  local / local-indep / local-alpha  fully local CBMs vs our federated CBL, same private heads
            (local_cbm.py), on the tab:main runs / the alpha-sweep runs.
  fedmid    FedMiD vs FedDualAvg over a lambda grid on the head (fedmid.py), each dataset's
            base tab:main run (the seed whose run is in dlamsweep).
The earlier "strong" variant (balanced + bias + standardization, lr chosen on
val; FEDCBM_STRONG) is retired; its flags are in each run's args.json.

Usage:
    python baselines/fedcbm_reimpl/naut/gen_jobs.py faithful > baselines/fedcbm_reimpl/naut/fedcbm-reimpl.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py balanced > baselines/fedcbm_reimpl/naut/fedcbm-balanced.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py nec      > baselines/fedcbm_reimpl/naut/fedcbm-nec.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py personal > baselines/fedcbm_reimpl/naut/ours-personal.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py head-own > baselines/fedcbm_reimpl/naut/ours-head-own.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py head-nec1020 > baselines/fedcbm_reimpl/naut/ours-head-nec1020.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py nec-refit > baselines/fedcbm_reimpl/naut/fedcbm-nec-refit.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py head-alpha > baselines/fedcbm_reimpl/naut/ours-head-alpha.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py nec-alpha > baselines/fedcbm_reimpl/naut/fedcbm-nec-alpha.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py local > baselines/fedcbm_reimpl/naut/local-cbm.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py local-indep > baselines/fedcbm_reimpl/naut/local-cbm-indep.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py local-indep-large > baselines/fedcbm_reimpl/naut/local-cbm-indep-large.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py local-alpha > baselines/fedcbm_reimpl/naut/local-cbm-alpha.yaml
    python baselines/fedcbm_reimpl/naut/gen_jobs.py fedmid > baselines/fedcbm_reimpl/naut/fedmid.yaml
"""

import math
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
# Per-client results across Dirichlet alpha: the heads above on the Fig-4 alpha-sweep runs
# (models/<ds>/alphasweep/{a0.05,a0.1,a0.5,a1.0,iid}, one seed each). FedCBM's bank does not
# depend on the partition, so that seed's bank is reused. Ours keeps the lam bracket so each
# alpha lands near NEC 5 / 50. Both save per-client accuracy and data conditions.
ALPHAS = [0.05, 0.1, 0.5, 1.0]
# The alpha sweep's seed per dataset (args.txt of every alphasweep run): CIFAR-100 ran 31, the
# rest 42. Both variants must use it, so FedCBM's clients are ours.
ALPHA_SWEEP_SEED = {"cifar10": 42, "cifar100": 31, "cub": 42, "places365": 42, "imagenet": 42}
_ALPHA_FLAGS = " --alphas " + " ".join(f"{a:g}" for a in ALPHAS)
VARIANTS["head-alpha"] = dict(job="ours-head-alpha", out="OURS_HEAD_ALPHA", refcheck=False, script="ours_head.py",
                              cpu_small=True, sized=True, seed_of=ALPHA_SWEEP_SEED, alpha_runs=True,
                              datasets=("cifar10", "cifar100", "cub"),
                              flags="--runs $RUNS --supports own --refit_rounds 100" + _ALPHA_FLAGS,
                              note="# Ours (head rule) per client across Dirichlet alpha, on the alpha-sweep runs.\n")
VARIANTS["nec-alpha"] = dict(VARIANTS["nec-refit"], job="fedcbm-nec-alpha", out="FEDCBM_BALANCED_NEC_ALPHA",
                             seed_of=ALPHA_SWEEP_SEED, datasets=("cifar10", "cifar100", "cub"),
                             flags=VARIANTS["nec-refit"]["flags"] + _ALPHA_FLAGS,
                             note="# FedCBM-NEC (+ refit) per client across Dirichlet alpha (the alpha sweep's seed).\n")
# Fully local CBMs (local_cbm.py): each client trains its own CBL with the tab:main run's Phase-1
# settings, then fed_cbl (our CBL) and local get the same private heads with the support refit.
# One Job per seed: a local CBL per client costs about one Phase 1, so a seed is a few hours.
VARIANTS["local"] = dict(job="local-cbm", out="LOCAL_CBM", refcheck=False, script="local_cbm.py",
                         cpu_small=True, sized=True, per_seed=True, datasets=("cifar10", "cifar100", "cub"),
                         flags="--runs $RUNS --nec_targets 5 10 20 50 --head_C_grid 0.01 0.1 1.0 "
                               "--path_refit_iters 1000 --local_init shared",
                         note="# Fully local CBMs vs our federated CBL, same private heads (+ refit), "
                              "union NEC, concept AUC and agreement.\n")
# The first LOCAL_CBM runs (2026-10-22) gave every client one shared initialization; on CUB's small
# clients that keeps local CBLs correlated through it. local-indep is the same comparison with an
# independent initialization per client. Measured on LOCAL_CBM: CUB used 0.6-0.7 GiB of 3Gi (the
# 20% floor), CIFAR 1.0-1.8 of 4Gi, 3.9-7.1 of 8 cores; CUB gets 2Gi here.
_LOCAL_INDEP = VARIANTS["local"]["flags"].replace("--local_init shared", "--local_init independent")
VARIANTS["local-indep"] = dict(VARIANTS["local"], job="local-cbm-indep", out="LOCAL_CBM_INDEP",
                               flags=_LOCAL_INDEP, mem_of={"cub": "2Gi"},
                               note="# Fully local CBMs, independent init per client, vs our federated CBL.\n")
VARIANTS["local-alpha"] = dict(VARIANTS["local"], job="local-cbm-alpha", out="LOCAL_CBM_ALPHA",
                               seed_of=ALPHA_SWEEP_SEED, alpha_runs=True, mem_of={"cub": "2Gi"},
                               flags=_LOCAL_INDEP + _ALPHA_FLAGS,
                               note="# Fully local CBMs (independent init) vs our federated CBL across "
                                    "Dirichlet alpha (alpha-sweep runs).\n")
# local-indep on the two large datasets: GPU, one Job per seed, same LOCAL_CBM_INDEP root and job
# names. ImageNet needs ~30 GiB of host memory for five local CBLs' val/test features (held on the
# CPU so they do not fill a 24 GB card), so 48Gi as Places365; GPU use follows Phase 1 (44% / 89%
# on the posaware runs) and the per-client paths (94-99% on fedcbm-nec).
VARIANTS["local-indep-large"] = dict(VARIANTS["local-indep"], datasets=("places365", "imagenet"),
                                     mem_of={"imagenet": "48Gi"},
                                     note="# Fully local CBMs, independent init, Places365 / ImageNet (GPU).\n")
# FedMiD vs FedDualAvg (fedmid.py) on each dataset's base tab:main run, the one in dlamsweep[_iid]
# (C10 s42 and C100 s31 match the local test of 2026-10-24). One FedDualAvg run took ~25 min on
# Places365 and ~1 h on ImageNet in head-own, so those two run the Dirichlet arm only, FedDualAvg at
# the two tab:main lambdas (Table 6 holds its other operating points) and FedMiD at five lambdas.
BASE_SEED = {ds: (set(SEEDS) - set(SEED_MAIN_SEEDS[ds])).pop() for ds in CFG}
_FEDMID_LARGE = " --partitions dirichlet --grid_points 2 --beyond 3 9 27 --fda_points ends"
VARIANTS["fedmid"] = dict(job="fedmid", out="FEDMID", refcheck=False, script="fedmid.py",
                          cpu_small=True, sized=True, seed_of=BASE_SEED, flags="--runs $RUNS", mem_of={"cub": "2Gi"},
                          flags_of={"places365": _FEDMID_LARGE, "imagenet": _FEDMID_LARGE},
                          note="# FedMiD (local prox-SGD + primal averaging) vs FedDualAvg on the head, lambda grid.\n")
# The head rule at NEC~10 and ~20 on the tab:main runs (D4: the cost-of-interpretability claim at
# NEC 5 / 10 / 20; X7: Table 3 at matched NEC). Same runs and rule as head-own, lambdas from
# lam_targets_mid with the default 0.6 / 1 / 1.6 bracket; per-class+refit gives NEC exactly 10 and 20.
# One Job per seed. Places365 and ImageNet run the Dirichlet arm only (a FedDualAvg run is ~25 min
# / ~1 h there, and this takes 6 per seed).
_DIR_ONLY = " --partitions dirichlet"
VARIANTS["head-nec1020"] = dict(VARIANTS["head-own"], job="ours-head-nec1020", out="OURS_HEAD_NEC1020",
                                per_seed=True, mem_of={"cub": "2Gi"},
                                flags="--runs $RUNS --refit_rounds 100 --supports own",
                                flags_of={"places365": _DIR_ONLY, "imagenet": _DIR_ONLY},
                                note="# Ours, the head rule (own-support refit, 100 rounds) at NEC~10 and ~20.\n")
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


def lam_targets_mid(ds, necs=(10, 20)):
    """Lambdas for NEC between the tab:main targets: log-linear in log NEC between the NEC~5 and
    NEC~50 lambdas (FedDualAvg's NEC falls roughly as a power of lambda); the bracket absorbs the rest."""
    out = []
    for part in ("iid", "dirichlet"):
        l5 = float(LAMS[ds][part][5] or CUB_NEC5_LAMS[part])
        l50 = float(LAMS[ds][part][50])
        for nec in necs:
            t = math.log(nec / 5) / math.log(10)
            out.append(f"{part}.{nec}:{l5 ** (1 - t) * l50 ** t:.3g}")
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


def alpha_run_dirs(ds):
    """$RUNS for the alpha-sweep runs (ALPHA_SWEEP_SEED): iid=... a0.05=... (keys as run.arms names them)."""
    runs = [f"iid={MODELS}/{ds}/alphasweep/iid"] + [f"a{a:g}={MODELS}/{ds}/alphasweep/a{a}" for a in ALPHAS]
    return f'\n                RUNS="{" ".join(runs)}"'


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
        if ds not in v.get("datasets", CFG):
            continue
        cpu_only = bool(v.get("cpu_small")) and ds in CPU_DATASETS
        k = REF_CHECK[ds] if v["refcheck"] else 0
        cifar = ds.startswith("cifar")
        cifar_in = (f'mkdir -p /root/.cache {CIFAR_CACHE} && cp -rn {CIFAR_CACHE}/. /root/.cache/ 2>/dev/null || true'
                    if cifar else ": # not CIFAR")
        cifar_out = (f'cp -rn /root/.cache/cifar-* {CIFAR_CACHE}/ 2>/dev/null || true'
                     if cifar else ": # not CIFAR")
        refcheck = (f"\\\n                  --fcm_repo $WORKSPACE/F-CMs --reference_check_concepts {k}"
                    if k else "")
        seeds = [v["seed_of"][ds]] if "seed_of" in v else v.get("seeds", SEEDS)
        cpu, mem, dshm = SIZED[ds] if v.get("sized") else (
            str(CPU_CORES) if cpu_only else cfg.get("cpu", "4"), cfg["mem"], cfg["dshm"])
        mem = v.get("mem_of", {}).get(ds, mem)
        groups =[[s] for s in seeds] if (v.get("sized") and not cpu_only) or v.get("per_seed") else [seeds]
        for grp in groups:
            job = JOB.format(
                ds=ds, job=v["job"], out=v["out"],
                flags=v["flags"] + (lam_targets_mid(ds) if variant == "head-nec1020" else
                                    lam_targets(ds) if variant.startswith("head") or variant == "fedmid" else "")
                + v.get("flags_of", {}).get(ds, ""), script=v.get("script", "run.py"),
                runs=(alpha_run_dirs(ds) if v.get("alpha_runs") else
                      run_dirs(ds, posaware=variant == "head-posaware")
                      if variant in ("personal", "head", "head-posaware", "head-diag", "head-own", "head-nec1020",
                                     "local", "local-indep", "local-indep-large", "fedmid") else ""),
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
