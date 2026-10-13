"""Print FedCBM-reimplementation results under a directory, one line per seed,
then mean +/- s.d. over seeds per partition and protocol.

Usage: python baselines/fedcbm_reimpl/summarize.py /sc-rwx-vol/fedmedcbm/models/FEDCBM_REIMPL/cifar100
"""

import glob
import json
import os
import statistics
import sys

PROTOCOLS = (("global_test_mean_client_acc", "global test, mean of client heads"),
             ("personalized_test_acc_weighted", "per-client test splits"))


def main(root):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True))]
    if not runs:
        print(f"no metrics.json under {root}")
        return
    agg = {}
    for m in runs:
        cells = []
        for part, r in m["partitions"].items():
            for key, _ in PROTOCOLS:
                agg.setdefault((part, key), []).append(100 * r[key])
            cells.append(f"{part}: {100 * r[PROTOCOLS[0][0]]:.2f} | {100 * r[PROTOCOLS[1][0]]:.2f}")
        print(f"  {m['dataset']} s{m['seed']}  ({m['num_concepts']} concepts)  " + "   ".join(cells))
    print("  mean +/- s.d. over seeds:")
    for (part, key), v in sorted(agg.items()):
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        label = dict(PROTOCOLS)[key]
        print(f"    {part:9s} {label:34s} {statistics.mean(v):6.2f} +/- {sd:.2f}  (n={len(v)})")


if __name__ == "__main__":
    main(sys.argv[1])
