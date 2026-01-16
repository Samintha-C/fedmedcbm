# Data Directory

This directory contains symlinks to data files from the Label-free-CBM repository.

## Structure

- `concept_sets/` - Symlink to Label-free-CBM concept files
- `*_classes.txt` - Symlinks to class label files (cifar10, cifar100, imagenet)
- `data_utils.py` - Data loading utilities that reuse Label-free-CBM functions

## Concept Files

Concept files are accessed via symlinks from `Label-free-CBM/data/concept_sets/`. 
Common concept files include:
- `cifar10_filtered.txt`
- `cifar100_filtered.txt`
- `imagenet_filtered.txt`

These can be used directly in the training script:
```bash
python main_fed.py --concept_file cifar10_filtered.txt ...
```

## Class Files

Class label files are also symlinked from Label-free-CBM for easy access.
