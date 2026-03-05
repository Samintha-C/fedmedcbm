import json
import os
import sys
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on sys.path before local imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import clip
from glm_saga.elasticnet import soft_threshold, IndexedTensorDataset, glm_saga

from models.fed_lfc import FedLFC_CBM
from training_utils import (
    set_seed, federated_averaging, train_client_local, train_final_layer_local,
    evaluate_model, save_args, save_concepts, save_training_metrics, save_metrics_txt,
    get_preprocess, create_client_data_loaders,
    load_concepts_from_file, load_or_generate_concept_embeddings,
    get_data, get_classes, split_dataset_for_federated, print_client_distribution,
)


def simulate_federated_training(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(
        args.save_dir,
        f"fed_lfc_{args.dataset}_c{args.num_clients}_r{args.num_rounds}_fr{args.final_rounds}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
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

    save_args(save_dir, args)
    save_concepts(save_dir, concepts)
    print(f"Dataset: {args.dataset}, Classes: {num_classes}")

    preprocess = get_preprocess(args.backbone)

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

    client_train_loaders, client_data_sizes, client_weights = create_client_data_loaders(
        train_dataset, client_indices, args.num_clients, args.batch_size, args.num_workers
    )

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

        global_state = federated_averaging(client_models, client_weights, exclude_prefixes=["backbone."])
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

    final_layer_method = getattr(args, 'final_layer_method', 'fedavg')

    if final_layer_method == "hybrid_saga":
        print("\n=== Phase 3: Hybrid Sparse Final Layer (GLM-SAGA) ===")
        # Extract normalized concept features from all clients
        all_concept_features = []
        all_labels = []
        global_model.eval()
        with torch.no_grad():
            for client_loader in client_train_loaders:
                for images, labels in client_loader:
                    images = images.to(device)
                    features = global_model.backbone(images)
                    concepts = global_model.projection(features)
                    concepts_norm = global_model.normalization(concepts)
                    all_concept_features.append(concepts_norm.cpu())
                    all_labels.append(labels)
        all_concept_features = torch.cat(all_concept_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Val concept features
        val_concept_features = []
        val_labels_list = []
        val_loader_temp = DataLoader(val_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers)
        with torch.no_grad():
            for images, labels in val_loader_temp:
                images = images.to(device)
                features = global_model.backbone(images)
                concepts = global_model.projection(features)
                concepts_norm = global_model.normalization(concepts)
                val_concept_features.append(concepts_norm.cpu())
                val_labels_list.append(labels)
        val_concept_features = torch.cat(val_concept_features, dim=0)
        val_labels_all = torch.cat(val_labels_list, dim=0)

        # Create SAGA-compatible data loaders
        saga_batch_size = getattr(args, 'saga_batch_size', 512)
        train_ds = IndexedTensorDataset(all_concept_features, all_labels)
        train_loader_saga = DataLoader(train_ds, batch_size=saga_batch_size, shuffle=True)
        val_ds = TensorDataset(val_concept_features, val_labels_all)
        val_loader_saga = DataLoader(val_ds, batch_size=saga_batch_size, shuffle=False)

        # Train sparse final layer with GLM-SAGA
        final_linear = nn.Linear(num_concepts, num_classes).to(device)
        final_linear.weight.data.zero_()
        final_linear.bias.data.zero_()

        saga_lam = getattr(args, 'saga_lam', 0.0007)
        metadata = {"max_reg": {"nongrouped": saga_lam}}
        out = glm_saga(
            final_linear, train_loader_saga,
            getattr(args, 'saga_step_size', 0.1),
            getattr(args, 'saga_n_iters', 2000),
            0.99,
            epsilon=1, k=1,
            val_loader=val_loader_saga,
            do_zero=False,
            metadata=metadata,
            n_ex=len(all_concept_features),
            n_classes=num_classes,
        )

        # Load sparse weights into the model
        w = out["path"][0]["weight"]
        b = out["path"][0]["bias"]
        global_model.final_layer.linear.weight.data.copy_(w.to(device))
        global_model.final_layer.linear.bias.data.copy_(b.to(device))

        best_accuracy = evaluate_model(global_model, global_test_loader, device)
        print(f"Hybrid SAGA Test Accuracy: {best_accuracy:.4f}")

        # Report sparsity
        nnz = (global_model.final_layer.linear.weight.data.abs() > 1e-5).sum().item()
        total = global_model.final_layer.linear.weight.data.numel()
        print(f"Final layer sparsity: {nnz}/{total} non-zero ({nnz/total:.4f})")

        torch.save(global_model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        final_layer_metrics = {"method": "hybrid_saga", "accuracy": float(best_accuracy),
                               "sparsity_nnz": nnz, "sparsity_total": total}

    else:
        # FedAvg or FedAvg-Thresh path
        print(f"\n=== Phase 3: Training Final Layer ({final_layer_method}) ===")
        best_accuracy = 0.0
        final_layer_metrics = {
            "rounds": [],
            "client_losses": [],
            "avg_client_loss": [],
            "global_accuracy": [],
            "best_accuracy": [],
            "method": final_layer_method,
        }
        if final_layer_method == "fedavg_thresh":
            final_layer_metrics["threshold_lam"] = []

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

            global_state = federated_averaging(client_models, client_weights, exclude_prefixes=["backbone."])

            # Server-side soft thresholding for fedavg_thresh
            if final_layer_method == "fedavg_thresh":
                progress = round_num / max(args.final_rounds - 1, 1)
                lam = args.thresh_lam_start + progress * (args.thresh_lam_end - args.thresh_lam_start)
                for key in global_state:
                    if "final_layer" in key and "weight" in key:
                        global_state[key] = soft_threshold(global_state[key], lam)
                print(f"  Applied soft_threshold with lam={lam:.6f}")
                final_layer_metrics["threshold_lam"].append(float(lam))

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

        # Report sparsity for thresh method
        if final_layer_method == "fedavg_thresh":
            nnz = (global_model.final_layer.linear.weight.data.abs() > 1e-5).sum().item()
            total = global_model.final_layer.linear.weight.data.numel()
            print(f"Final layer sparsity: {nnz}/{total} non-zero ({nnz/total:.4f})")
            final_layer_metrics["sparsity_nnz"] = nnz
            final_layer_metrics["sparsity_total"] = total
            final_layer_metrics["sparsity_pct_nonzero"] = nnz / total

    print(f"\nBest Accuracy: {best_accuracy:.4f}")
    torch.save(global_model.state_dict(), os.path.join(save_dir, "final_model.pt"))

    # Save metrics.txt (consistent with VLG format)
    weight_data = global_model.final_layer.linear.weight.data
    sparsity_lfc = {
        "Non-zero weights": int((weight_data.abs() > 1e-5).sum().item()),
        "Total weights": int(weight_data.numel()),
        "Percentage non-zero": float((weight_data.abs() > 1e-5).float().mean().item()),
    }
    lfc_metrics_txt = {
        "final_layer_method": final_layer_method,
        "metrics": {"test_accuracy": float(best_accuracy)},
        "sparsity": sparsity_lfc,
    }
    if final_layer_method == "hybrid_saga":
        lfc_metrics_txt["saga_lam"] = getattr(args, "saga_lam", 0.0007)
    elif final_layer_method == "fedavg_thresh":
        lfc_metrics_txt["thresh_lam_start"] = args.thresh_lam_start
        lfc_metrics_txt["thresh_lam_end"] = args.thresh_lam_end
    lfc_metrics_txt["lr"] = args.final_lr
    save_metrics_txt(save_dir, lfc_metrics_txt)

    training_metrics = {
        "projection_phase": projection_metrics,
        "final_layer_phase": final_layer_metrics,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "final_rounds": args.final_rounds,
        "client_data_sizes": client_data_sizes,
        "client_weights": client_weights,
        "iid": args.iid,
        "alpha": args.alpha if not args.iid else None,
        "best_final_accuracy": float(best_accuracy),
        "final_layer_method": final_layer_method,
    }

    save_training_metrics(save_dir, training_metrics)
