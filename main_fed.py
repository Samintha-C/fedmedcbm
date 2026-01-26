import argparse
import json
import os
import random
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm

import clip
import sys
import importlib.util

# Add current directory to path FIRST, before any other imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import local utils modules directly from files to avoid conflicts with Label-free-CBM's utils.py
utils_concepts_path = os.path.join(current_dir, 'utils', 'concepts.py')
utils_losses_path = os.path.join(current_dir, 'utils', 'losses.py')
data_utils_path = os.path.join(current_dir, 'data', 'data_utils.py')

spec_concepts = importlib.util.spec_from_file_location("fed_utils_concepts", utils_concepts_path)
spec_losses = importlib.util.spec_from_file_location("fed_utils_losses", utils_losses_path)
spec_data = importlib.util.spec_from_file_location("fed_data_utils", data_utils_path)

fed_utils_concepts = importlib.util.module_from_spec(spec_concepts)
fed_utils_losses = importlib.util.module_from_spec(spec_losses)
fed_data_utils = importlib.util.module_from_spec(spec_data)

spec_concepts.loader.exec_module(fed_utils_concepts)
spec_losses.loader.exec_module(fed_utils_losses)
spec_data.loader.exec_module(fed_data_utils)

# Now import models (which may add Label-free-CBM to path)
from models.fed_lfc import FedLFC_CBM

# Import functions with unique names to avoid conflicts
load_concepts_from_file = fed_utils_concepts.load_concepts_from_file
load_or_generate_concept_embeddings = fed_utils_concepts.load_or_generate_concept_embeddings
cosine_similarity_cubed_loss = fed_utils_losses.cosine_similarity_cubed_loss
L1SparsityLoss = fed_utils_losses.L1SparsityLoss
get_data = fed_data_utils.get_data
get_classes = fed_data_utils.get_classes
get_resnet_preprocess = fed_data_utils.get_resnet_preprocess
get_clip_preprocess = fed_data_utils.get_clip_preprocess
split_dataset_for_federated = fed_data_utils.split_dataset_for_federated


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def federated_averaging(models, client_weights=None):
    if client_weights is None:
        client_weights = [1.0 / len(models)] * len(models)
    
    global_state = {}
    for key in models[0].state_dict().keys():
        param = models[0].state_dict()[key]
        
        # Only average float parameters (weights, biases)
        # For non-float parameters (Long tensors like batch norm counts), copy from first model
        if param.dtype.is_floating_point:
            global_state[key] = torch.zeros_like(param)
            for i, model in enumerate(models):
                global_state[key] += client_weights[i] * model.state_dict()[key]
        else:
            # For non-float parameters, just copy from first model
            global_state[key] = param.clone()
    
    return global_state


def train_client_local(
    model,
    train_loader,
    concept_embeddings,
    clip_model,
    epochs,
    lr,
    weight_decay,
    sparsity_lambda,
    device
):
    model.train()
    optimizer = torch.optim.Adam(model.projection.parameters(), lr=lr, weight_decay=weight_decay)
    sparsity_loss_fn = L1SparsityLoss(lambda_l1=sparsity_lambda)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for images, _ in train_loader:
            images = images.to(device)
            
            with torch.no_grad():
                image_features = model.backbone(images)
                clip_image_features = clip_model.encode_image(images)
                clip_image_features = clip_image_features.float()  # Convert to float32
                clip_image_features = clip_image_features / clip_image_features.norm(dim=-1, keepdim=True)
                clip_targets = clip_image_features @ concept_embeddings.T
            
            concept_pred = model.projection(image_features)
            
            loss = cosine_similarity_cubed_loss(concept_pred, clip_targets)
            sparsity_loss = sparsity_loss_fn(concept_pred, clip_targets, model.get_projection_params())
            total_loss = loss + sparsity_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
    
    return epoch_loss / num_batches if num_batches > 0 else 0.0


def train_final_layer_local(model, train_loader, epochs, lr, device):
    model.train()
    optimizer = torch.optim.Adam(model.final_layer.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            if model.normalization is not None:
                with torch.no_grad():
                    image_features = model.backbone(images)
                    concept_pred = model.projection(image_features)
                    concept_features = model.normalization(concept_pred)
            else:
                with torch.no_grad():
                    image_features = model.backbone(images)
                    concept_pred = model.projection(image_features)
                    concept_features = concept_pred
            
            logits = model.final_layer(concept_features)
            loss = ce_loss(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
    
    return epoch_loss / num_batches if num_batches > 0 else 0.0


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits, _ = model(images, return_concepts=True)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def simulate_federated_training(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    save_dir = os.path.join(
        args.save_dir,
        f"fed_lfc_{args.dataset}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    concepts = load_concepts_from_file(args.concept_file)
    concept_embeddings = load_or_generate_concept_embeddings(
        concepts,
        clip_name=args.clip_name,
        device=device,
        cache_dir=args.cache_dir
    )
    num_concepts = concept_embeddings.size(0)
    
    clip_model, _ = clip.load(args.clip_name, device=device)
    
    print(f"Loaded {num_concepts} concepts")
    
    classes = get_classes(args.dataset)
    num_classes = len(classes)
    
    args.num_concepts = num_concepts
    args.num_classes = num_classes
    
    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    with open(os.path.join(save_dir, "concepts.txt"), "w") as f:
        if concepts:
            f.write(concepts[0])
            for concept in concepts[1:]:
                f.write('\n' + concept)
    print(f"Dataset: {args.dataset}, Classes: {num_classes}")
    
    if args.backbone.startswith("clip_"):
        preprocess = get_clip_preprocess()
    else:
        preprocess = get_resnet_preprocess()
    
    train_dataset = get_data(f"{args.dataset}_train", preprocess=preprocess)
    val_dataset = get_data(f"{args.dataset}_val", preprocess=preprocess)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    client_indices = split_dataset_for_federated(
        train_dataset, args.num_clients, iid=args.iid, seed=args.seed
    )
    
    global_model = FedLFC_CBM(
        backbone_type=args.backbone,
        clip_name=args.clip_name,
        num_concepts=num_concepts,
        num_classes=num_classes,
        use_clip_penultimate=args.use_clip_penultimate,
        proj_hidden_layers=args.proj_hidden_layers,
        device=device
    )
    
    client_models = [
        FedLFC_CBM(
            backbone_type=args.backbone,
            clip_name=args.clip_name,
            num_concepts=num_concepts,
            num_classes=num_classes,
            use_clip_penultimate=args.use_clip_penultimate,
            proj_hidden_layers=args.proj_hidden_layers,
            device=device
        ) for _ in range(args.num_clients)
    ]
    
    client_train_loaders = []
    client_data_sizes = []
    
    for i in range(args.num_clients):
        client_train_subset = Subset(train_dataset, client_indices[i])
        train_loader = DataLoader(
            client_train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        client_train_loaders.append(train_loader)
        client_data_sizes.append(len(client_train_subset))
    
    total_samples = sum(client_data_sizes)
    client_weights = [size / total_samples for size in client_data_sizes]
    
    global_test_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print("\n=== Phase 1: Training Projection Layer ===")
    best_proj_loss = float('inf')
    
    for round_num in range(args.num_rounds):
        print(f"\n=== Federated Round {round_num + 1}/{args.num_rounds} ===")
        
        for client_id in range(args.num_clients):
            client_models[client_id].load_state_dict(global_model.state_dict())
            
            train_loss = train_client_local(
                client_models[client_id],
                client_train_loaders[client_id],
                concept_embeddings,
                clip_model,
                epochs=args.local_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                sparsity_lambda=args.sparsity_lambda,
                device=device
            )
            
            print(f"Client {client_id}: Train Loss = {train_loss:.4f}")
        
        global_state = federated_averaging(client_models, client_weights)
        global_model.load_state_dict(global_state)
        
        if train_loss < best_proj_loss:
            best_proj_loss = train_loss
            torch.save(global_model.state_dict(), os.path.join(save_dir, "best_projection.pt"))
    
    print("\n=== Phase 2: Computing Normalization Statistics ===")
    with torch.no_grad():
        all_concept_features = []
        global_model.eval()
        for client_loader in client_train_loaders:
            for images, _ in client_loader:
                images = images.to(device)
                image_features = global_model.backbone(images)
                concept_pred = global_model.projection(image_features)
                all_concept_features.append(concept_pred.cpu())
        
        all_concept_features = torch.cat(all_concept_features, dim=0)
        proj_mean = all_concept_features.mean(dim=0, keepdim=True)
        proj_std = all_concept_features.std(dim=0, keepdim=True) + 1e-8
        
        global_model.set_normalization(proj_mean, proj_std)
        for client_model in client_models:
            client_model.set_normalization(proj_mean, proj_std)
        
        torch.save(proj_mean, os.path.join(save_dir, "proj_mean.pt"))
        torch.save(proj_std, os.path.join(save_dir, "proj_std.pt"))
    
    print("\n=== Phase 3: Training Final Layer ===")
    best_accuracy = 0.0
    
    for round_num in range(args.final_rounds):
        print(f"\n=== Final Layer Round {round_num + 1}/{args.final_rounds} ===")
        
        for client_id in range(args.num_clients):
            client_models[client_id].load_state_dict(global_model.state_dict())
            
            train_loss = train_final_layer_local(
                client_models[client_id],
                client_train_loaders[client_id],
                epochs=args.final_epochs,
                lr=args.final_lr,
                device=device
            )
            
            print(f"Client {client_id}: Final Layer Loss = {train_loss:.4f}")
        
        global_state = federated_averaging(client_models, client_weights)
        global_model.load_state_dict(global_state)
        
        accuracy = evaluate_model(global_model, global_test_loader, device)
        print(f"Global Test Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(global_model.state_dict(), os.path.join(save_dir, "best_model.pt"))
    
    print(f"\nBest Accuracy: {best_accuracy:.4f}")
    torch.save(global_model.state_dict(), os.path.join(save_dir, "final_model.pt"))


def main():
    parser = argparse.ArgumentParser(description="Federated Label-Free Concept Bottleneck Model")
    
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet"], help="Dataset name")
    parser.add_argument("--concept_file", type=str, required=True, help="Path to concept file")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone type: resnet50 or clip_ViT-B/16")
    parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="CLIP model name")
    parser.add_argument("--use_clip_penultimate", action="store_true", help="Use CLIP penultimate layer")
    
    parser.add_argument("--num_clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--local_epochs", type=int, default=5, help="Local training epochs per round")
    parser.add_argument("--iid", action="store_true", default=True, help="Use IID data distribution (default: True)")
    
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for projection layer")
    parser.add_argument("--final_lr", type=float, default=1e-3, help="Learning rate for final layer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--sparsity_lambda", type=float, default=1e-4, help="Sparsity regularization")
    parser.add_argument("--proj_hidden_layers", type=int, default=0, help="Hidden layers in projection")
    parser.add_argument("--final_rounds", type=int, default=5, help="Number of rounds for final layer training")
    parser.add_argument("--final_epochs", type=int, default=3, help="Epochs per round for final layer training")
    
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Save directory")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for embeddings")
    
    args = parser.parse_args()
    simulate_federated_training(args)


if __name__ == "__main__":
    main()
