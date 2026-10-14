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
    nec_runs = [m for m in runs if "sparse" in m]
    head_runs = [m for m in runs if any("targets" in v for v in m.get("partitions", {}).values())]
    runs = [m for m in runs if "partitions" in m and not any("targets" in v for v in m["partitions"].values())]
    head_agg = {}
    for m in head_runs:  # ours_head.py output: per partition and NEC target
        for p, rec in m["partitions"].items():
            for t, r in rec["targets"].items():
                cells = {"fda": r["fda"]["global_test"], "perclass_refit": r["perclass_refit"]["global_test"],
                         "per_client": r["perclass_refit"]["per_client"], "logit_adj": r["logit_adjust"]["per_client"]}
                for k, v in cells.items():
                    head_agg.setdefault((m['dataset'], p, int(t), k), []).append(v)
                print(f"  {m['dataset']} s{m['seed']} {p} NEC~{t}: fda {cells['fda']:.2f}@{r['fda']['nec']:.1f} | "
                      f"per-class+refit {cells['perclass_refit']:.2f} | per-client {cells['per_client']:.2f} | "
                      f"logit-adj per-client {cells['logit_adj']:.2f}")
    for (ds, p, t, k), v in sorted(head_agg.items()):
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"    {ds:9s} {p:9s} NEC~{t:<3d} {k:15s} {statistics.mean(v):6.2f} +/- {sd:.2f}  (n={len(v)})")
    for m in nec_runs:  # nec.py output: sparse heads per NEC target + dense C grid
        for t, by_p in m["sparse"].items():
            print(f"  {m['dataset']} s{m['seed']} NEC {float(t):g}: " + "   ".join(
                f"{p}: {100 * r['global_test_mean_client_acc']:.2f} | {100 * r['personalized_test_acc_weighted']:.2f} "
                f"(achieved {r['nec_mean_over_clients']:.1f}, union {r['union_nec']:.1f})" for p, r in by_p.items()))
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
