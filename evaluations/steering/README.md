# Steering evaluation

Code behind the steering results of the CVPR 2027 draft (Sec. 4.7, Fig. `steering_main.pdf`, App. "Steering Details") and the sanity check on the concept layer.

For a test image of class `y` and a target class `y'`, the edit switches on the concepts in `y'`'s explanation (and off the ones against it) by the smallest change to the backbone features. A judge that never sees the concepts (a linear or MLP probe on the backbone features) then classifies the edited features. Controls: a random direction of the same size, and the same edit built from a third class's explanation.

| File | What it does |
|---|---|
| `common.py` | Data layout, loading one seed, calibration, the edit, the judge, FedDualAvg for the sanity heads |
| `pull_inputs.py` | Read-only copy of the inputs from the PVC through the copier pod; `--check` only compares |
| `steer.py` | Steering for one dataset, seed and judge |
| `plot.py` | The paper figure, from `steer.py`'s results |
| `sanity.py` | The same steering on random, centrally trained and class-shuffled concept layers (CIFAR only) |
| `within_class_auc.py` | Concept AUC within one class at a time, for the layers `sanity.py` built |

## Inputs

The heads come from the `ours-head-saved` jobs (`baselines/fedcbm_reimpl/naut/ours-head-saved.yaml`, `ours_head.py --save_heads`), which write `OURS_HEAD_SAVED/<dataset>/s<seed>/<run>/heads.pt` next to `metrics.json`. The concept layers are the runs behind Table 2 (listed in `pull_inputs.py`).

Start the copier pod (`kubectl apply -f naut/copy.yaml`), then pull into a data root of your choice:

```bash
python3 -I evaluations/steering/pull_inputs.py --data <data root>
```

About 5.4 GB for all four datasets. Places365 and ImageNet use the validation portion of each seed's training split (about 90k and 64k images, never test) instead of the 5 to 7 GB full training caches. Delete the copier pod afterwards (`kubectl delete pod sc-rwx-copy-pod -n wenglab-interpretable-ai`).

## Run

```bash
for ds in c10 c100 places365 imagenet; do for s in 42 31 7; do for j in linear mlp; do
  python3 -I evaluations/steering/steer.py --data <data root> --dataset $ds --seed $s --judge $j
done; done; done
python3 -I evaluations/steering/plot.py --results <data root>/saved/results --out <figure dir>
```

Each `steer.py` run first recomputes the head's test accuracy and stops unless it matches the job's `metrics.json`. Results go to `<data root>/saved/results/steer_<ds>_s<seed>_<judge>_both_right.json`. Everything runs on CPU in minutes per run.

Sanity check and within-class AUC (CIFAR-10 and CIFAR-100 only, since new heads need the full training split):

```bash
for ds in c10 c100; do for s in 42 31 7; do
  python3 -I evaluations/steering/sanity.py --data <data root> --dataset $ds --seed $s
done; done
python3 -I evaluations/steering/within_class_auc.py --data <data root>
```

`sanity.py` trains two concept layers and fits three heads per seed, so it takes several minutes per seed on CIFAR-10 and longer on CIFAR-100. It caches them as `layers_<ds>_s<seed>.pt` in `<data root>/saved/sanity/`, where its results also go.

## Determinism

Judges, image draws and target draws are seeded, so a rerun on the same inputs reproduces every number exactly.
