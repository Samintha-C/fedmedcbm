"""Prepare CheXpert-small for main_fed.py (X28): F-CM's split plus concept labels as annotations.

Writes, from the unzipped CheXpert-v1.0-small directory:
  $CHEXPERT_ROOT/fedcbm/chexpert_train.csv   F-CM's train + val (80%), our "chexpert_train"
  $CHEXPERT_ROOT/fedcbm/chexpert_val.csv     F-CM's test (20%), our test set ("chexpert_val")
  $CHEXPERT_ROOT/fedcbm/split_info.json      sizes, class balance, concept prevalence
  <annotation_dir>/chexpert_{train,val}/<i>.json
                                             the 13 CheXpert labels of row i in the format
                                             DinoConceptDataset reads (each present concept
                                             as a detection with logit 1.0)
The split and label handling are data/chexpert.py's (F-CM's). Output paths must lie outside
this repo, since the data and its labels may not be redistributed.

usage: CHEXPERT_ROOT=<.../CheXpert-v1.0-small> python case_studies/chexpert/prepare.py \
           --annotation_dir <dir outside the repo>
"""
import argparse
import json
import os
import sys

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from data.chexpert import (CONCEPT_NAMES, PREP_SUBDIR, SPLIT_SEED, TARGET_NAME,  # noqa: E402
                           chexpert_root, clean_and_split)


def outside_repo(path):
    real, repo = os.path.realpath(path), os.path.realpath(_REPO)
    if os.path.commonpath([real, repo]) == repo:
        sys.exit(f"{path} is inside the repo ({repo}); CheXpert data must stay outside it")
    return path


def write_annotations(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, (img, row) in enumerate(zip(df["img_id"], df[CONCEPT_NAMES].to_numpy())):
        rec = [{"img_path": img}] + [{"label": c, "logit": 1.0, "box": []}
                                     for c, v in zip(CONCEPT_NAMES, row) if v == 1]
        with open(os.path.join(out_dir, f"{i}.json"), "w") as f:
            json.dump(rec, f)


def summary(df):
    return {"n": len(df), "any_finding": int(df[TARGET_NAME].sum()),
            "no_finding": int((df[TARGET_NAME] == 0).sum()),
            "concept_prevalence": {c: round(float(df[c].mean()), 4) for c in CONCEPT_NAMES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="default: $CHEXPERT_ROOT")
    ap.add_argument("--annotation_dir", required=True)
    a = ap.parse_args()
    root = outside_repo(a.root or chexpert_root())
    ann = outside_repo(a.annotation_dir)

    tr, va, te = clean_and_split(root, seed=SPLIT_SEED)
    train = pd.concat([tr, va], ignore_index=True)
    test = te.reset_index(drop=True)
    missing = [p for p in pd.concat([train, test])["img_id"] if not os.path.exists(os.path.join(root, p))]
    if missing:
        sys.exit(f"{len(missing)} images missing under {root}, e.g. {missing[0]}")

    prep = os.path.join(root, PREP_SUBDIR)
    os.makedirs(prep, exist_ok=True)
    cols = ["img_id", "subject_id", TARGET_NAME] + CONCEPT_NAMES
    train[cols].to_csv(os.path.join(prep, "chexpert_train.csv"), index=False)
    test[cols].to_csv(os.path.join(prep, "chexpert_val.csv"), index=False)

    write_annotations(train, os.path.join(ann, "chexpert_train"))
    write_annotations(test, os.path.join(ann, "chexpert_val"))

    info = {"split_seed": SPLIT_SEED, "fcm_split": {"train": len(tr), "val": len(va), "test": len(te)},
            "train": summary(train), "test": summary(test),
            "patients_overlap": len(set(train["subject_id"]) & set(test["subject_id"]))}
    with open(os.path.join(prep, "split_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))
    print(f"[ok] CSVs in {prep}, annotations in {ann}")


if __name__ == "__main__":
    main()
