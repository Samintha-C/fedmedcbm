"""CheXpert (small) for the F-CM head-to-head (X28): F-CM's split and image handling.

Everything that decides the data follows F-CM (github.com/francescoTheSantis/F-CMs, commit
31cbb14, src/data/datasets/cheXpert.py and src/data/preprocessing.py):
  - clean_and_split: the active lines of their clean_and_split_data, unchanged. Frontal views
    only, the last visit per patient, NaN concepts absent and uncertain (-1) present (U-Ones), task = any
    finding (the No Finding label inverted), the majority class undersampled to the minority's
    size (random_state 42), then a seeded shuffle and a 70 / 10 / 20 split.
  - load_image: their _CheXpert.__getitem__ image steps (center square crop, resize to 224,
    histogram equalization, grayscale replicated to RGB).
  - CXR_PREPROCESS: their test_transform (Resize(224) + ToTensor, no ImageNet normalization).
    Their encoder (resnet18_cxr) is ImageNet ResNet-18 pooled after layer4: InputImgEncoder
    keeps only the ResNet and drops the projector built next to it. Here that is
    `--backbone resnet18_cxr --feature_layer layer4` (data_utils.get_target_model).

Two deliberate differences, both because our pipeline fixes the test set across seeds:
  - one split, at F-CM's seed 42. Our "train" set is their train + val (80%); main_fed draws
    its own val from it with --val_split 0.125 per seed. Our test set ("chexpert_val") is
    their test set (20%).
  - F-CM embeds the training images once through their train_transform (a random affine and
    flip per image); here every split goes through the test transform.

Paths: CHEXPERT_ROOT is the unzipped CheXpert-v1.0-small directory (train/, valid/,
train.csv, valid.csv). case_studies/chexpert/prepare.py writes the split CSVs to
$CHEXPERT_ROOT/fedcbm/ and the concept labels as annotation JSONs. The data and anything
derived from it stay outside the repo (research-use agreement).
"""
import os

import numpy as np
import pandas as pd
import PIL
import PIL.ImageOps
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TARGET_NAME = "No Finding"
CONCEPT_NAMES = ['Enlarged Cardiomediastinum',
                 'Cardiomegaly',
                 'Lung Opacity',
                 'Lung Lesion',
                 'Edema',
                 'Consolidation',
                 'Pneumonia',
                 'Atelectasis',
                 'Pneumothorax',
                 'Pleural Effusion',
                 'Pleural Other',
                 'Fracture',
                 'Support Devices']
SPLIT_SEED = 42
PREP_SUBDIR = "fedcbm"
transResize = 224

CXR_PREPROCESS = transforms.Compose([
    transforms.Resize(transResize),
    transforms.ToTensor()
])


def chexpert_root():
    root = os.environ.get("CHEXPERT_ROOT")
    if not root:
        raise RuntimeError("Set CHEXPERT_ROOT to the unzipped CheXpert-v1.0-small directory")
    return root


def clean_and_split(data_path, seed=SPLIT_SEED):
    """F-CM's clean_and_split_data (cheXpert.py), active lines unchanged, returning the frames
    instead of writing custom_*.csv into the dataset directory."""
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "valid.csv"))
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    full_df = full_df.reset_index(drop=True)

    # Filter frontal and AP/PA images
    full_df = full_df[full_df["Frontal/Lateral"] == "Frontal"]
    full_df = full_df.drop(columns=["Frontal/Lateral", "AP/PA"])

    # Eliminate duplicates
    subject_ids = []
    for index, row in full_df.iterrows():
        path_parts = row['Path'].split("/")
        patient_part = next(part for part in path_parts if "patient" in part)
        subject_ids.append(patient_part[len("patient"):])
    full_df['subject_id'] = subject_ids

    # Droping duplicate patient recordings except for the last visit
    full_df = full_df.drop_duplicates(subset=['subject_id'], keep='last')
    full_df = full_df.sort_values(by=['subject_id']).reset_index(drop=True)

    # Fill NaNs as absent, uncertain (-1) as present (U-Ones strategy from CheXpert paper)
    full_df[CONCEPT_NAMES] = full_df[CONCEPT_NAMES].fillna(0)
    full_df[CONCEPT_NAMES] = full_df[CONCEPT_NAMES].replace(-1, 1)
    # if full_df[TERGET_NAME] is 0 replace it with 1, otherwise put 0
    full_df[TARGET_NAME] = np.where(full_df[TARGET_NAME] == 1, 0, 1)

    # Extract relative path
    full_df["img_id"] = full_df["Path"].apply(lambda x: x.split("/", 1)[1] if "/" in x else x)
    full_df = full_df.drop(columns=["Path"])

    majority = full_df[full_df[TARGET_NAME] == 1]
    minority = full_df[full_df[TARGET_NAME] == 0]
    target_majority_size = min(len(majority), int(len(minority)))
    if target_majority_size < len(majority):
        majority_sampled = majority.sample(n=target_majority_size, random_state=42)
        print(f"[CheXpert] Majority reduced: {len(majority)}->{len(majority_sampled)}")
    else:
        majority_sampled = majority
    full_df_balanced = pd.concat([minority, majority_sampled], ignore_index=True)
    print(f"[CheXpert] Total balanced: {len(full_df_balanced)}")

    full_df_all = full_df_balanced.copy()
    # Shuffle and split into train, val and test
    full_df_all = full_df_all.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(full_df_all) * (0.7))  # 70% train
    val_idx = int(len(full_df_all) * (0.8))     # 10% val, 20% test
    return (full_df_all.iloc[:split_idx], full_df_all.iloc[split_idx:val_idx],
            full_df_all.iloc[val_idx:])


def load_image(path):
    """F-CM's _CheXpert.__getitem__ image steps, before the transform."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    r_min = max(0, (height - width) / 2)
    r_max = min(height, (height + width) / 2)
    c_min = max(0, (width - height) / 2)
    c_max = min(width, (width + height) / 2)
    image = image.crop((c_min, r_min, c_max, r_max))
    image = image.resize((transResize, transResize))
    image = PIL.ImageOps.equalize(image)
    return image.convert('L').convert('RGB')


class CheXpert(Dataset):
    """One prepared split ("train" = F-CM train + val, "val" = F-CM test). Returns
    (image, target) with target 1 = any finding; .targets for get_dataset_targets."""

    def __init__(self, split, transform=None, root=None):
        self.root = root or chexpert_root()
        csv = os.path.join(self.root, PREP_SUBDIR, f"chexpert_{split}.csv")
        if not os.path.exists(csv):
            raise FileNotFoundError(f"{csv} missing: run case_studies/chexpert/prepare.py first")
        self.df = pd.read_csv(csv)
        self.transform = transform
        self.targets = self.df[TARGET_NAME].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = load_image(os.path.join(self.root, self.df["img_id"].iloc[idx]))
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx]
