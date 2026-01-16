#!/bin/bash

# Example script to run federated training on CIFAR-10
# Using concept file from Label-free-CBM

python main_fed.py \
    --dataset cifar10 \
    --concept_file cifar10_filtered.txt \
    --backbone resnet50 \
    --clip_name ViT-B/16 \
    --num_clients 5 \
    --num_rounds 10 \
    --local_epochs 5 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --sparsity_lambda 1e-4 \
    --iid \
    --seed 42 \
    --device cuda
