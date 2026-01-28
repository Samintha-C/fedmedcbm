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
print_client_distribution = fed_data_utils.print_client_distribution


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
    elif args.backbone == "resnet18_cub":
        preprocess = get_resnet_preprocess()
    else:
        preprocess = get_resnet_preprocess()
    
    train_dataset = get_data(f"{args.dataset}_train", preprocess=preprocess)
    val_dataset = get_data(f"{args.dataset}_val", preprocess=preprocess)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Partition data among clients
    if args.iid:
        print(f"\nUsing IID data distribution")
    else:
        print(f"\nUsing Non-IID data distribution (Dirichlet alpha={args.alpha})")

    client_indices = split_dataset_for_federated(
        train_dataset, args.num_clients, iid=args.iid, alpha=args.alpha, seed=args.seed
    )

    # Print client data distribution for debugging
    print_client_distribution(train_dataset, client_indices, num_classes=num_classes)
    
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
    projection_metrics = {
        "rounds": [],
        "client_losses": [],
        "avg_client_loss": [],
        "best_proj_loss": []
    }
    
    for round_num in range(args.num_rounds):
        print(f"\n=== Federated Round {round_num + 1}/{args.num_rounds} ===")
        
        round_client_losses = []
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
            
            round_client_losses.append(train_loss)
            print(f"Client {client_id}: Train Loss = {train_loss:.4f}")
        
        global_state = federated_averaging(client_models, client_weights)
        global_model.load_state_dict(global_state)
        
        avg_loss = sum(round_client_losses) / len(round_client_losses)
        projection_metrics["rounds"].append(round_num + 1)
        projection_metrics["client_losses"].append(round_client_losses)
        projection_metrics["avg_client_loss"].append(avg_loss)
        
        if avg_loss < best_proj_loss:
            best_proj_loss = avg_loss
            torch.save(global_model.state_dict(), os.path.join(save_dir, "best_projection.pt"))
        
        projection_metrics["best_proj_loss"].append(best_proj_loss)
    
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
    final_layer_metrics = {
        "rounds": [],
        "client_losses": [],
        "avg_client_loss": [],
        "global_accuracy": [],
        "best_accuracy": []
    }
    
    for round_num in range(args.final_rounds):
        print(f"\n=== Final Layer Round {round_num + 1}/{args.final_rounds} ===")
        
        round_client_losses = []
        for client_id in range(args.num_clients):
            client_models[client_id].load_state_dict(global_model.state_dict())
            
            train_loss = train_final_layer_local(
                client_models[client_id],
                client_train_loaders[client_id],
                epochs=args.final_epochs,
                lr=args.final_lr,
                device=device
            )
            
            round_client_losses.append(train_loss)
            print(f"Client {client_id}: Final Layer Loss = {train_loss:.4f}")
        
        global_state = federated_averaging(client_models, client_weights)
        global_model.load_state_dict(global_state)
        
        accuracy = evaluate_model(global_model, global_test_loader, device)
        print(f"Global Test Accuracy: {accuracy:.4f}")
        
        avg_loss = sum(round_client_losses) / len(round_client_losses)
        final_layer_metrics["rounds"].append(round_num + 1)
        final_layer_metrics["client_losses"].append(round_client_losses)
        final_layer_metrics["avg_client_loss"].append(avg_loss)
        final_layer_metrics["global_accuracy"].append(float(accuracy))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(global_model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        
        final_layer_metrics["best_accuracy"].append(float(best_accuracy))
    
    print(f"\nBest Accuracy: {best_accuracy:.4f}")
    torch.save(global_model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    
    training_metrics = {
        "projection_phase": projection_metrics,
        "final_layer_phase": final_layer_metrics,
        "num_clients": args.num_clients,
        "client_data_sizes": client_data_sizes,
        "client_weights": client_weights,
        "iid": args.iid,
        "alpha": args.alpha if not args.iid else None,
        "best_final_accuracy": float(best_accuracy)
    }
    
    with open(os.path.join(save_dir, "training_metrics.json"), "w") as f:
        json.dump(training_metrics, f, indent=2)
    
    print(f"\nTraining metrics saved to {os.path.join(save_dir, 'training_metrics.json')}")


def simulate_federated_training_vlg(args):
    import copy
    _loss_vlg_spec = importlib.util.spec_from_file_location("fed_loss_vlg", os.path.join(current_dir, "utils", "loss_vlg.py"))
    _loss_vlg_mod = importlib.util.module_from_spec(_loss_vlg_spec)
    _loss_vlg_spec.loader.exec_module(_loss_vlg_mod)
    get_loss_vlg = _loss_vlg_mod.get_loss
    from data import data_utils
    from data.concept_dataset_vlg import AllOneConceptDataset, get_concept_dataloader
    from models.fed_vlgcbm import (
        Backbone, BackboneCLIP, ConceptLayer, NormalizationLayer, FinalLayer, FedVLGCBM,
        train_cbl, validate_cbl, get_final_layer_dataset, train_sparse_final, train_dense_final, test_model, per_class_accuracy,
    )

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(
        args.save_dir,
        f"fed_vlg_{args.dataset}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    )
    os.makedirs(save_dir, exist_ok=True)

    concepts = data_utils.get_concepts(args.concept_file, getattr(args, "filter_set", None))
    num_concepts = len(concepts)
    classes = get_classes(args.dataset)
    num_classes = len(classes)
    args.num_concepts = num_concepts
    args.num_classes = num_classes

    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(save_dir, "concepts.txt"), "w") as f:
        f.write("\n".join(concepts))

    if args.backbone.startswith("clip_"):
        preprocess = get_clip_preprocess()
        backbone = BackboneCLIP(args.backbone, use_penultimate=getattr(args, "use_clip_penultimate", True), device=str(device))
    else:
        preprocess = get_resnet_preprocess()
        backbone = Backbone(args.backbone, getattr(args, "feature_layer", "layer4"), str(device))

    cbl = ConceptLayer(
        backbone.output_dim, num_concepts,
        num_hidden=getattr(args, "cbl_hidden_layers", 0),
        bias=True, device=str(device)
    )
    global_model = FedVLGCBM(backbone, cbl, normalization=None, final_layer=None)
    global_model.to(device)

    full_train_dataset = get_data(f"{args.dataset}_train", preprocess=None)
    
    val_split = getattr(args, "val_split", 0.1)
    n_val = int(val_split * len(full_train_dataset))
    n_train = len(full_train_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Split full train dataset: {len(train_dataset)} train, {len(val_dataset)} val (val_split={val_split})")
    
    client_indices = split_dataset_for_federated(
        train_dataset, args.num_clients, iid=args.iid, alpha=args.alpha, seed=args.seed
    )
    print_client_distribution(train_dataset, client_indices, num_classes=num_classes)

    base_cbl_dataset = AllOneConceptDataset(args.dataset, train_dataset, concepts, preprocess)
    val_cbl_dataset = AllOneConceptDataset(args.dataset, val_dataset, concepts, preprocess)
    val_cbl_loader = DataLoader(
        val_cbl_dataset,
        batch_size=getattr(args, "cbl_batch_size", 32),
        num_workers=args.num_workers,
        shuffle=False
    )
    client_train_loaders = []
    client_data_sizes = []
    for i in range(args.num_clients):
        sub = Subset(base_cbl_dataset, client_indices[i])
        client_train_loaders.append(DataLoader(
            sub, batch_size=getattr(args, "cbl_batch_size", 32),
            shuffle=True, num_workers=args.num_workers, pin_memory=True
        ))
        client_data_sizes.append(len(sub))
    total_samples = sum(client_data_sizes)
    client_weights = [n / total_samples for n in client_data_sizes]

    test_loader = get_concept_dataloader(
        args.dataset, "test", concepts, preprocess=preprocess, use_allones=True,
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    num_train = len(base_cbl_dataset)
    concept_counts = [num_train // num_classes] * num_concepts
    loss_fn = get_loss_vlg(
        getattr(args, "cbl_loss_type", "bce"), num_concepts, num_train, concept_counts,
        getattr(args, "cbl_pos_weight", 0.2), not getattr(args, "no_cbl_auto_weight", False),
        tp=getattr(args, "cbl_twoway_tp", 4.0), device=str(device)
    )

    client_models = [copy.deepcopy(global_model) for _ in range(args.num_clients)]
    for m in client_models:
        m.to(device)

    print("\n=== Phase 1: Federated CBL training ===")
    projection_metrics = {"rounds": [], "client_losses": [], "avg_client_loss": [], "best_val_loss": []}
    best_val_loss = float("inf")
    for round_num in range(args.num_rounds):
        round_losses = []
        for i in range(args.num_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            _, _ = train_cbl(
                client_models[i].backbone, client_models[i].cbl,
                client_train_loaders[i], val_cbl_loader,
                epochs=getattr(args, "cbl_epochs", args.local_epochs),
                loss_fn=loss_fn, lr=getattr(args, "cbl_lr", args.lr),
                weight_decay=args.weight_decay, device=str(device),
                finetune=getattr(args, "cbl_finetune", False),
                optimizer_name=getattr(args, "cbl_optimizer", "adam"),
                backbone_lr=getattr(args, "cbl_bb_lr_rate", 1.0) * getattr(args, "cbl_lr", args.lr),
            )
            vl = validate_cbl(client_models[i].backbone, client_models[i].cbl, val_cbl_loader, loss_fn, str(device))
            round_losses.append(vl)
        global_state = federated_averaging(client_models, client_weights)
        global_model.load_state_dict(global_state)
        avg_loss = sum(round_losses) / len(round_losses)
        projection_metrics["rounds"].append(round_num + 1)
        projection_metrics["client_losses"].append(round_losses)
        projection_metrics["avg_client_loss"].append(avg_loss)
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
        projection_metrics["best_val_loss"].append(best_val_loss)
        print(f"Round {round_num + 1} avg val loss: {avg_loss:.4f}")

    full_train_cbl_loader = DataLoader(
        base_cbl_dataset, batch_size=getattr(args, "saga_batch_size", 512),
        shuffle=True, num_workers=args.num_workers
    )
    print("\n=== Phase 2: Final-layer dataset and normalization ===")
    train_concept_loader, val_concept_loader, norm_layer = get_final_layer_dataset(
        global_model.backbone, global_model.cbl,
        full_train_cbl_loader, val_cbl_loader,
        save_dir, load_dir=None, batch_size=getattr(args, "saga_batch_size", 512), device=str(device)
    )
    global_model.normalization = norm_layer

    print("\n=== Phase 3: Final layer (sparse GLM-SAGA or dense) ===")
    final_layer = FinalLayer(num_concepts, num_classes, device=str(device))
    if getattr(args, "dense", False):
        out = train_dense_final(
            final_layer, train_concept_loader, val_concept_loader,
            n_iters=getattr(args, "saga_n_iters", 2000), lr=getattr(args, "dense_lr", 0.001), device=str(device)
        )
    else:
        out = train_sparse_final(
            final_layer, train_concept_loader, val_concept_loader,
            n_iters=getattr(args, "saga_n_iters", 2000), lam=getattr(args, "saga_lam", 0.0007),
            step_size=getattr(args, "saga_step_size", 0.1), device=str(device)
        )
    w = out["path"][0]["weight"] if out.get("path") else out.get("best", {}).get("weight")
    b = out["path"][0]["bias"] if out.get("path") else out.get("best", {}).get("bias")
    if w is not None:
        final_layer.weight.data.copy_(w.to(device))
    if b is not None:
        final_layer.bias.data.copy_(b.to(device))
    global_model.final_layer = final_layer

    test_acc = test_model(test_loader, global_model.backbone, global_model.cbl, global_model.normalization, global_model.final_layer, str(device))
    print(f"Test accuracy: {test_acc:.4f}")

    global_model.backbone.save_model(save_dir)
    global_model.cbl.save_model(save_dir)
    global_model.normalization.save_model(save_dir)
    global_model.final_layer.save_model(save_dir)

    pca = per_class_accuracy(global_model, test_loader, classes, str(device))
    sparsity_vlg = {"Non-zero weights": (global_model.final_layer.weight.data.abs() > 1e-5).sum().item(), "Total weights": global_model.final_layer.weight.data.numel(), "Percentage non-zero": (global_model.final_layer.weight.data.abs() > 1e-5).float().mean().item()}
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        json.dump({"per_class_accuracies": pca, "lam": getattr(args, "saga_lam", -1.0), "lr": getattr(args, "dense_lr", -1.0), "alpha": -1.0, "time": -1.0, "metrics": {"test_accuracy": float(test_acc)}, "sparsity": sparsity_vlg}, f, indent=2)

    if getattr(args, "run_nec_eval", True) and not getattr(args, "dense", False):
        print("\n=== Phase 4: NEC evaluation ===")
        import pandas as pd
        from evaluations.sparse_utils import measure_acc
        test_feats, test_labels = [], []
        with torch.no_grad():
            for features, _, labels in tqdm(test_loader):
                features = features.to(device)
                logits = global_model.normalization(global_model.cbl(global_model.backbone(features)))
                test_feats.append(logits.cpu())
                test_labels.append(labels)
        test_feats = torch.cat(test_feats, dim=0)
        test_labels = torch.cat(test_labels, dim=0)
        test_concept_loader = DataLoader(
            TensorDataset(test_feats, test_labels),
            batch_size=getattr(args, "saga_batch_size", 512),
            shuffle=False,
        )
        nec_measure_level = getattr(args, "nec_measure_level", (5, 10, 15, 20, 25, 30))
        path, truncated_weights, _ = measure_acc(
            num_concepts, num_classes, len(train_concept_loader.dataset),
            train_concept_loader, val_concept_loader, test_concept_loader,
            saga_step_size=getattr(args, "saga_step_size", 0.1),
            saga_n_iters=getattr(args, "saga_n_iters", 2000),
            device=str(device),
            max_lam=getattr(args, "nec_lam_max", 0.01),
            measure_level=nec_measure_level,
        )
        sparsity_list = [(p["weight"].abs() > 1e-5).float().mean().item() for p in path]
        nec_col = [num_concepts * s for s in sparsity_list]
        acc_col = [p["metrics"]["acc_test"] for p in path]
        pd.DataFrame({"NEC": nec_col, "Accuracy": acc_col}).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
        for nec_val, (W, b) in truncated_weights.items():
            torch.save(W, os.path.join(save_dir, f"W_g@NEC={nec_val:d}.pt"))
            torch.save(b, os.path.join(save_dir, f"b_g@NEC={nec_val:d}.pt"))

    training_metrics = {
        "projection_phase": projection_metrics,
        "num_clients": args.num_clients, "client_data_sizes": client_data_sizes, "client_weights": client_weights,
        "iid": args.iid, "alpha": args.alpha if not args.iid else None,
        "best_final_accuracy": float(test_acc),
    }
    with open(os.path.join(save_dir, "training_metrics.json"), "w") as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Federated Label-Free Concept Bottleneck Model")
    
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet", "cub"], help="Dataset name")
    parser.add_argument("--concept_file", type=str, required=True, help="Path to concept file")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone type: resnet50 or clip_ViT-B/16")
    parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="CLIP model name")
    parser.add_argument("--use_clip_penultimate", action="store_true", help="Use CLIP penultimate layer")
    parser.add_argument("--use_vlg", action="store_true", help="Use VLG-CBM training (AllOne concepts, BCE/TwoWay loss, SAGA final layer)")
    
    parser.add_argument("--num_clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--local_epochs", type=int, default=5, help="Local training epochs per round")
    parser.add_argument("--iid", action="store_true", help="Use IID data distribution")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for non-IID partitioning (smaller = more heterogeneous). "
                        "Only used when --iid is not set. Typical values: 0.1 (extreme), 0.5 (moderate), 1.0 (mild), 100 (near-IID)")
    
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
    
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split (VLG)")
    parser.add_argument("--feature_layer", type=str, default="layer4", help="Backbone feature layer (VLG, non-CLIP)")
    parser.add_argument("--cbl_loss_type", type=str, default="bce", choices=["bce", "twoway"], help="CBL loss (VLG)")
    parser.add_argument("--cbl_lr", type=float, default=5e-4, help="CBL learning rate (VLG)")
    parser.add_argument("--cbl_epochs", type=int, default=20, help="CBL epochs per client round (VLG)")
    parser.add_argument("--cbl_batch_size", type=int, default=32, help="CBL batch size (VLG)")
    parser.add_argument("--cbl_optimizer", type=str, default="adam", choices=["adam", "sgd"], help="CBL optimizer (VLG)")
    parser.add_argument("--cbl_hidden_layers", type=int, default=0, help="CBL hidden layers (VLG)")
    parser.add_argument("--cbl_pos_weight", type=float, default=0.2, help="BCE positive weight (VLG)")
    parser.add_argument("--no_cbl_auto_weight", action="store_true", help="Disable BCE auto positive weighting (VLG)")
    parser.add_argument("--cbl_twoway_tp", type=float, default=4.0, help="TwoWay loss Tp (VLG)")
    parser.add_argument("--cbl_finetune", action="store_true", help="Finetune backbone in CBL (VLG)")
    parser.add_argument("--cbl_bb_lr_rate", type=float, default=1.0, help="Backbone LR scale in CBL (VLG)")
    parser.add_argument("--saga_lam", type=float, default=0.0007, help="SAGA sparsity lambda (VLG)")
    parser.add_argument("--saga_n_iters", type=int, default=2000, help="SAGA iterations (VLG)")
    parser.add_argument("--saga_step_size", type=float, default=0.1, help="SAGA step size (VLG)")
    parser.add_argument("--saga_batch_size", type=int, default=512, help="SAGA batch size (VLG)")
    parser.add_argument("--no_nec_eval", action="store_true", help="Skip NEC evaluation (Phase 4)")
    parser.add_argument("--nec_lam_max", type=float, default=0.01, help="NEC path max lambda (VLG)")
    parser.add_argument("--nec_measure_level", type=str, default="5,10,15,20,25,30", help="NEC levels, comma-separated (VLG)")
    parser.add_argument("--dense", action="store_true", help="Train dense final layer (VLG)")
    parser.add_argument("--dense_lr", type=float, default=0.001, help="Learning rate for dense final layer (VLG)")
    
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("--config", type=str, default=None)
    config_pre, remaining = config_parser.parse_known_args()
    if config_pre.config is not None:
        with open(config_pre.config, "r") as f:
            parser.set_defaults(**json.load(f))
    
    args = parser.parse_args(remaining)
    args.run_nec_eval = not getattr(args, "no_nec_eval", False)
    nm = getattr(args, "nec_measure_level", (5, 10, 15, 20, 25, 30))
    args.nec_measure_level = tuple(int(x) for x in (nm.split(",") if isinstance(nm, str) else nm))
    if getattr(args, "use_vlg", False):
        simulate_federated_training_vlg(args)
    else:
        simulate_federated_training(args)


if __name__ == "__main__":
    main()
