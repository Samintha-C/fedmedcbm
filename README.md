# Fed-LFC: Federated Label-Free Concept Bottleneck Model

Federated Learning framework for Label-Free Concept Bottleneck Models, reusing components from [Label-free-CBM](https://github.com/Trustworthy-ML-Lab/Label-free-CBM).

## Project Structure

```
fed_lfc_cbm/
├── data/
│   ├── __init__.py
│   └── data_utils.py      # Dataset loaders (reuses Label-free-CBM)
├── models/
│   ├── __init__.py
│   ├── backbones.py       # ResNet50 and CLIP-ViT wrappers (uses Label-free-CBM models)
│   └── fed_lfc.py         # Core FedLFC_CBM class
├── utils/
│   ├── __init__.py
│   ├── concepts.py        # Concept loading (uses Label-free-CBM concept files)
│   └── losses.py          # Sparsity regularization (Elastic Net/L1)
├── main_fed.py            # Federated Learning Simulation Loop
├── run_example.sh         # Example run script
└── requirements.txt       # Dependencies
```

## Installation

### Local Setup

```bash
pip install -r requirements.txt
```

### Nautilus Setup

1. Deploy the pod:
```bash
kubectl apply -f naut/medcbm.yaml
kubectl exec -it medcbm -n wenglab-interpretable-ai -- bash
```

2. Inside the pod, clone and setup:
```bash
cd /sc-cbint-vol
git clone <repo-url> fed_lfc_cbm
cd fed_lfc_cbm
bash setup_nautilus.sh
```

**Note**: Ensure `Label-free-CBM` is cloned in the same parent directory as `fed_lfc_cbm` for the symlinks to work.

## Usage

### Quick Start with CIFAR-10

```bash
python main_fed.py \
    --dataset cifar10 \
    --concept_file cifar10_filtered.txt \
    --backbone resnet50 \
    --clip_name ViT-B/16 \
    --num_clients 5 \
    --num_rounds 10 \
    --local_epochs 5 \
    --final_rounds 5 \
    --final_epochs 3 \
    --batch_size 32 \
    --lr 1e-3 \
    --final_lr 1e-3 \
    --iid \
    --seed 42
```

The concept file path is automatically resolved relative to `Label-free-CBM/data/concept_sets/` if not found as an absolute path.

### Available Datasets

- `cifar10` - CIFAR-10 dataset
- `cifar100` - CIFAR-100 dataset  
- `imagenet` - ImageNet dataset

### Concept Files

Uses concept files from Label-free-CBM. Common options:
- `cifar10_filtered.txt`
- `cifar100_filtered.txt`
- `imagenet_filtered.txt`

## Key Features

- **Federated Learning**: Simulates multiple clients training locally and aggregating updates using FedAvg
- **Label-Free Concepts**: Uses GPT-generated text concepts and CLIP embeddings from Label-free-CBM
- **Reuses Label-free-CBM**: Leverages pretrained backbones, data loaders, and concept sets
- **Two-Phase Training**: 
  1. Projection layer training (aligns backbone features to CLIP concept embeddings)
  2. Final layer training (maps concepts to class predictions)
- **Sparsity Regularization**: Supports L1 regularization for concept sparsity
- **Flexible Backbones**: Supports ResNet50 and CLIP-ViT backbones via Label-free-CBM
