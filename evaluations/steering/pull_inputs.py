"""Read-only copy of the steering inputs from the PVC into a data root (layout in common.py).

Needs the copier pod running (naut/copy.yaml, pod sc-rwx-copy-pod); start and delete it yourself.
Per dataset and seed: the run's concept layer files, its OURS_HEAD_SAVED heads.pt + metrics.json, the
feature and DINO label caches of its training split (CIFAR) or of the split's validation portion
(Places365, ImageNet), and the test caches. kubectl exec drops long transfers, sometimes with exit
code 0, so every file is copied in 64 MB chunks, each checked for length, and the whole file is
checked against md5sum on the pod. Files already present at the right size are skipped.

usage: python3 -I pull_inputs.py --data <data root> [--datasets c10 c100 places365 imagenet] [--check]
  --check   only compare sizes on the PVC with the local copies; transfers nothing
"""
import argparse
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, SEEDS, SPLIT, TEST, TEST_ANN, VAL  # noqa: E402

NS, POD = "wenglab-interpretable-ai", "sc-rwx-copy-pod"
PVC = "/sc-rwx-vol/fedmedcbm"
LONG = {"c10": "cifar10", "c100": "cifar100", "places365": "places365", "imagenet": "imagenet"}
# the run behind each seed's Table 2 head (base seed in dlamsweep, the others in SEED_MAIN)
RUNS = {"c10": {42: "cifar10/dlamsweep/lam0.0013/c5r5-vlg-cifar10-feddualavg-jul18-23:51",
                31: "SEED_MAIN/cifar10/dir_s31_nec50/c5r5-vlg-cifar10-feddualavg-sep12-17:54",
                7: "SEED_MAIN/cifar10/dir_s7_nec50/c5r5-vlg-cifar10-feddualavg-sep12-18:10"},
        "c100": {42: "SEED_MAIN/cifar100/dir_s42_nec50/c5r5-vlg-cifar100-feddualavg-sep12-17:54",
                 31: "cifar100/dlamsweep/lam0.00086/c5r5-vlg-cifar100-feddualavg-jul18-23:52",
                 7: "SEED_MAIN/cifar100/dir_s7_nec50/c5r5-vlg-cifar100-feddualavg-sep12-18:09"},
        "places365": {42: "places365/dlamsweep/lam0.0001/c5r5-vlg-places365-feddualavg-jul17-05:37",
                      31: "SEED_MAIN/places365/dir_s31_nec50/c5r5-vlg-places365-feddualavg-sep12-17:54",
                      7: "SEED_MAIN/places365/dir_s7_nec50/c5r5-vlg-places365-feddualavg-sep14-08:13"},
        "imagenet": {42: "imagenet/dlamsweep/lam0.000077/c5r5-vlg-imagenet-feddualavg-sep03-19:14",
                     31: "SEED_MAIN/imagenet/dir_s31_nec50/c5r5-vlg-imagenet-feddualavg-sep12-17:54",
                     7: "SEED_MAIN/imagenet/dir_s7_nec50/c5r5-vlg-imagenet-feddualavg-sep18-14:48"}}
RUN_FILES = ("cbl.pt", "train_concept_features_mean.pt", "train_concept_features_std.pt", "args.txt",
             "concepts.txt", "metrics.txt")
MB = 1 << 20
CHUNK = 64


def ex(*cmd, **kw):
    return subprocess.run(["kubectl", "exec", "-n", NS, POD, "--", *cmd], **kw)


def size(path):
    r = ex("stat", "-c", "%s", path, capture_output=True, text=True)
    return int(r.stdout) if r.returncode == 0 else None


def pull(src, dst, check, optional=False):
    n = size(src)
    if n is None:
        if optional:
            print(f"absent (optional): {src}", flush=True)
            return True
        print(f"MISSING on PVC: {src}", flush=True)
        return False
    have = os.path.getsize(dst) if os.path.exists(dst) else None
    if check:
        print(f"{'ok  ' if have == n else 'DIFF'} {dst} (PVC {n}, local {have})", flush=True)
        return have == n
    if have == n:
        print(f"have {dst} ({n})", flush=True)
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with open(tmp, "wb") as f:
        i = 0
        while i * MB < n:
            want = min(CHUNK * MB, n - i * MB)
            for attempt in range(20):
                r = ex("dd", f"if={src}", f"bs={MB}", f"skip={i}", f"count={CHUNK}", "status=none", capture_output=True)
                if r.returncode == 0 and len(r.stdout) == want:
                    break
                print(f"retry {attempt + 1} {os.path.basename(src)} at {i} MB: rc {r.returncode}, {len(r.stdout)} of {want}", flush=True)
            else:
                sys.exit(f"gave up on {src}")
            f.write(r.stdout)
            i += CHUNK
    remote = ex("md5sum", src, capture_output=True, text=True, check=True).stdout.split()[0]
    h = hashlib.md5()
    with open(tmp, "rb") as f:
        for b in iter(lambda: f.read(MB), b""):
            h.update(b)
    if h.hexdigest() != remote:
        sys.exit(f"md5 mismatch {src}")
    os.replace(tmp, dst)
    print(f"pulled {dst} ({n})", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    ok = True
    cache = lambda name: (f"{PVC}/annotation_cache/{os.path.basename(name)}", os.path.join(args.data, name))  # noqa: E731
    for ds in args.datasets:
        files = [cache(TEST[ds])] + ([cache(TEST_ANN[ds])] if ds in TEST_ANN else [])
        for seed in SEEDS:
            run, out = f"{PVC}/models/{RUNS[ds][seed]}", os.path.join(args.data, "saved", ds, f"s{seed}")
            for f in RUN_FILES:
                ok &= pull(f"{run}/{f}", os.path.join(out, f), args.check)
            ok &= pull(f"{run}/removed_concepts.txt", os.path.join(out, "removed_concepts.txt"), args.check,
                       optional=ds not in VAL)
            if ds in VAL:
                ok &= pull(f"{run}/val_concept_labels.pt", os.path.join(out, "val_concept_labels.pt"), args.check)
            else:  # client sizes, for sanity.py's check of the client split
                ok &= pull(f"{run}/training_metrics.json", os.path.join(out, "training_metrics.json"), args.check)
            r = ex("sh", "-c", f"ls -d {PVC}/models/OURS_HEAD_SAVED/{LONG[ds]}/s{seed}/*/", capture_output=True, text=True)
            dirs = r.stdout.split()
            if len(dirs) != 1:
                print(f"MISSING or ambiguous OURS_HEAD_SAVED/{LONG[ds]}/s{seed}: {dirs}", flush=True)
                ok = False
            else:
                for f in ("heads.pt", "metrics.json"):
                    ok &= pull(dirs[0] + f, os.path.join(out, f), args.check)
            hashes = [VAL[ds][seed]] if ds in VAL else list(SPLIT[seed])
            files += [cache(c.format(h)) for h in hashes for c in CACHE[ds]]
        for src, dst in dict.fromkeys(files):
            ok &= pull(src, dst, args.check)
    print("ALL OK" if ok else "SOME FILES MISSING OR DIFFERENT", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
