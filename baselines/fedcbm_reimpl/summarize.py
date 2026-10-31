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
    t3_runs = [m for m in runs if m.get("method", "").startswith("dense FedAvg")]
    for m in t3_runs:  # dense_prune.py output (Table 3): one line per seed and part
        for p, rec in m["partitions"].items():
            cells = [f"{k} {rec[k]['global_test']:.2f}" for k in ("fedavg", "fedprox") if k in rec]
            cells += [f"{mode} {k}: {r['global_test']:.2f}" for mode, by in rec.get("prune", {}).items()
                      for k, r in by.items()]
            cells += [f"oracle {k}: {r['global_test']:.2f}@{r['nec']:.1f}" for k, r in rec.get("oracle", {}).items()]
            cells += [f"{part} {k}: {r['global_test']:.2f}@{r['nec']:.1f}" for part in ("thresh", "fedmid", "elastic", "central")
                      for k, r in rec.get(part, {}).items() if k != "lam_ratio"]
            print(f"s{m['seed']} {p}: " + " | ".join(cells))
    runs = [m for m in runs if m not in t3_runs]
    if not runs:
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
                for name, v in r.get("refit_variants", {}).items():  # e.g. own_100, the head rule
                    cells[name] = v["global_test"]
                    cells[f"{name}_la"] = v["logit_adj_per_client"]
                for k, v in cells.items():
                    head_agg.setdefault((m['dataset'], p, int(t), k), []).append(v)
                print(f"  {m['dataset']} s{m['seed']} {p} NEC~{t}: fda {cells['fda']:.2f}@{r['fda']['nec']:.1f} | "
                      f"per-class+refit {cells['perclass_refit']:.2f} | per-client {cells['per_client']:.2f} | "
                      f"logit-adj per-client {cells['logit_adj']:.2f}" + "".join(
                          f" | {name} {v['global_test']:.2f}@{v['nec']:.1f} (logit-adj {v['logit_adj_per_client']:.2f})"
                          for name, v in r.get("refit_variants", {}).items()))
    for (ds, p, t, k), v in sorted(head_agg.items()):
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"    {ds:9s} {p:9s} NEC~{t:<3d} {k:15s} {statistics.mean(v):6.2f} +/- {sd:.2f}  (n={len(v)})")
    nec_agg = {}
    for m in nec_runs:  # nec.py output: sparse heads per NEC target (+ refit on that support) + dense C grid
        for key in ("sparse", "sparse_refit"):
            for t, by_p in m.get(key, {}).items():
                for p, r in by_p.items():
                    for proto, _ in PROTOCOLS:
                        nec_agg.setdefault((key, float(t), p, proto), []).append(100 * r[proto])
                print(f"  {m['dataset']} s{m['seed']} {key} NEC {float(t):g}: " + "   ".join(
                    f"{p}: {100 * r['global_test_mean_client_acc']:.2f} | {100 * r['personalized_test_acc_weighted']:.2f} "
                    f"(achieved {r['nec_mean_over_clients']:.1f}, union {r['union_nec']:.1f})" for p, r in by_p.items()))
    for (key, t, p, proto), v in sorted(nec_agg.items()):
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"    {key:12s} NEC {t:<4g} {p:9s} {dict(PROTOCOLS)[proto]:34s} {statistics.mean(v):6.2f} +/- {sd:.2f}  (n={len(v)})")
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
