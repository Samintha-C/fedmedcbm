import argparse
import json
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import importlib.util

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

utils_concepts_path = os.path.join(current_dir, 'utils', 'concepts.py')
data_utils_path = os.path.join(current_dir, 'data', 'data_utils.py')

spec_concepts = importlib.util.spec_from_file_location("fed_utils_concepts", utils_concepts_path)
spec_data = importlib.util.spec_from_file_location("fed_data_utils", data_utils_path)

fed_utils_concepts = importlib.util.module_from_spec(spec_concepts)
fed_data_utils = importlib.util.module_from_spec(spec_data)

spec_concepts.loader.exec_module(fed_utils_concepts)
spec_data.loader.exec_module(fed_data_utils)

from models.fed_lfc import FedLFC_CBM

load_concepts_from_file = fed_utils_concepts.load_concepts_from_file
get_data = fed_data_utils.get_data
get_classes = fed_data_utils.get_classes
get_target_model = fed_data_utils.get_target_model


def _is_vlg_checkpoint(load_dir):
    return (os.path.exists(os.path.join(load_dir, "cbl.pt")) and
            os.path.exists(os.path.join(load_dir, "final.pt")))


def load_fed_vlg_cbm(load_dir, device):
    from models.fed_vlgcbm import (
        Backbone, BackboneCLIP, ConceptLayer, NormalizationLayer, FinalLayer, FedVLGCBM,
    )
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)
    backbone_name = args["backbone"]
    if backbone_name.startswith("clip_"):
        backbone = BackboneCLIP(backbone_name, use_penultimate=args.get("use_clip_penultimate", True), device=str(device))
    else:
        backbone = Backbone(backbone_name, args.get("feature_layer", "layer4"), str(device))
    num_concepts = args.get("num_concepts")
    num_classes = args.get("num_classes")
    if num_concepts is None or num_classes is None:
        final_sd = torch.load(os.path.join(load_dir, "final.pt"), map_location=device)
        w = final_sd.get("weight", final_sd.get("linear.weight"))
        if w is None:
            w = list(final_sd.values())[0]
        num_classes, num_concepts = w.shape[0], w.shape[1]
        args["num_concepts"] = num_concepts
        args["num_classes"] = num_classes
    cbl = ConceptLayer(backbone.output_dim, num_concepts, num_hidden=args.get("cbl_hidden_layers", 0), bias=True, device=str(device))
    cbl.load_state_dict(torch.load(os.path.join(load_dir, "cbl.pt"), map_location=device))
    backbone.backbone.load_state_dict(torch.load(os.path.join(load_dir, "backbone.pt"), map_location=device))
    norm_layer = NormalizationLayer.from_pretrained(load_dir, device=str(device))
    final_layer = FinalLayer(num_concepts, num_classes, device=str(device))
    final_layer.load_state_dict(torch.load(os.path.join(load_dir, "final.pt"), map_location=device))
    model = FedVLGCBM(backbone, cbl, norm_layer, final_layer)
    model.to(device)
    model.eval()
    return model, args


def load_fed_cbm(load_dir, device):
    if _is_vlg_checkpoint(load_dir):
        return load_fed_vlg_cbm(load_dir, device)
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)
    
    model_path = os.path.join(load_dir, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(load_dir, "final_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found in {load_dir}. Expected 'best_model.pt' or 'final_model.pt'")
    
    state_dict = torch.load(model_path, map_location=device)
    
    if "num_concepts" not in args or "num_classes" not in args:
        print("Warning: args.txt missing num_concepts/num_classes. Inferring from model state_dict...")
        if "final_layer.linear.weight" in state_dict:
            num_classes = state_dict["final_layer.linear.weight"].shape[0]
            num_concepts = state_dict["final_layer.linear.weight"].shape[1]
        elif "final_layer.model.weight" in state_dict:
            num_classes = state_dict["final_layer.model.weight"].shape[0]
            num_concepts = state_dict["final_layer.model.weight"].shape[1]
        else:
            raise ValueError("Cannot infer num_concepts/num_classes from model. Please retrain with updated training script.")
        args["num_concepts"] = num_concepts
        args["num_classes"] = num_classes
        print(f"Inferred: num_concepts={num_concepts}, num_classes={num_classes}")
    
    model = FedLFC_CBM(
        backbone_type=args["backbone"],
        clip_name=args.get("clip_name", "ViT-B/16"),
        num_concepts=args["num_concepts"],
        num_classes=args["num_classes"],
        use_clip_penultimate=args.get("use_clip_penultimate", False),
        proj_hidden_layers=args.get("proj_hidden_layers", 0),
        device=device
    )
    
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if not k.startswith("normalization.")}
    
    model.load_state_dict(filtered_state_dict, strict=False)
    
    proj_mean_path = os.path.join(load_dir, "proj_mean.pt")
    proj_std_path = os.path.join(load_dir, "proj_std.pt")
    
    if os.path.exists(proj_mean_path) and os.path.exists(proj_std_path):
        proj_mean = torch.load(proj_mean_path, map_location=device)
        proj_std = torch.load(proj_std_path, map_location=device)
        model.set_normalization(proj_mean, proj_std)
    elif "normalization.mean" in state_dict and "normalization.std" in state_dict:
        proj_mean = state_dict["normalization.mean"]
        proj_std = state_dict["normalization.std"]
        model.set_normalization(proj_mean, proj_std)
    else:
        print("Warning: Normalization statistics not found. Model may not work correctly.")
    
    model.eval()
    return model, args


def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=2):
    correct = 0
    total = 0
    model.eval()
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device), return_concepts=True)
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu() == labels)
            total += len(labels)
    return correct / total


def get_per_class_accuracy_vlg(model, dataset, device, classes, batch_size=250, num_workers=2):
    correct = torch.zeros(len(classes)).to(device)
    total = torch.zeros(len(classes)).to(device)
    model.eval()
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=1)
        for p, t in zip(preds, labels):
            total[t] += 1
            if p == t:
                correct[t] += 1
    pca = (correct / total).nan_to_num_(nan=0.0)
    tot = total.sum()
    overall = (correct.sum() / tot).item() * 100.0 if tot.item() > 0 else 0.0
    return {
        "Per class accuracy": {classes[i]: f"{pca[i].item()*100.0:.2f}" for i in range(len(classes))},
        "Overall accuracy": f"{overall:.2f}",
        "Datapoints": f"{tot.item()}",
    }


def get_sparsity_vlg(final_layer):
    w = final_layer.weight.data.cpu()
    nnz = (w.abs() > 1e-5).sum().item()
    n = w.numel()
    return {"Non-zero weights": nnz, "Total weights": n, "Percentage non-zero": nnz / n}


def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2):
    preds = []
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device), return_concepts=True)
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds


def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels in tqdm(DataLoader(dataset, 500, num_workers=8, pin_memory=True)):
        with torch.no_grad():
            outs, concept_act = model(images.to(device), return_concepts=True)
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred = []
    for i in range(torch.max(pred) + 1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds == i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred


def get_detailed_concept_metrics(model, dataset, device, batch_size=250, num_workers=2):
    model.eval()
    all_concept_acts = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)):
            images = images.to(device)
            labels = labels.to(device)
            
            outs, concept_acts = model(images, return_concepts=True)
            preds = torch.argmax(outs, dim=1)
            
            all_concept_acts.append(concept_acts.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
    
    all_concept_acts = torch.cat(all_concept_acts, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    
    num_concepts = all_concept_acts.shape[1]
    num_classes = len(torch.unique(all_labels))
    
    concept_stats = []
    for c in range(num_concepts):
        concept_vals = all_concept_acts[:, c]
        concept_stats.append({
            "mean": float(concept_vals.mean().item()),
            "std": float(concept_vals.std().item()),
            "min": float(concept_vals.min().item()),
            "max": float(concept_vals.max().item()),
            "median": float(concept_vals.median().item())
        })
    
    concept_acts_by_class = []
    for class_id in range(num_classes):
        class_mask = (all_labels == class_id)
        if class_mask.sum() > 0:
            class_concept_acts = all_concept_acts[class_mask].mean(dim=0)
            concept_acts_by_class.append(class_concept_acts.tolist())
        else:
            concept_acts_by_class.append([0.0] * num_concepts)
    
    return {
        "concept_statistics": concept_stats,
        "concept_activations_by_class": concept_acts_by_class,
        "all_concept_activations_shape": list(all_concept_acts.shape)
    }


def get_weight_statistics(model, concepts, classes):
    fl = getattr(model.final_layer, "linear", model.final_layer)
    weights = fl.weight.data.cpu()
    bias = fl.bias.data.cpu()
    
    num_classes, num_concepts = weights.shape
    
    weight_stats_per_concept = []
    for c in range(num_concepts):
        concept_weights = weights[:, c]
        weight_stats_per_concept.append({
            "mean": float(concept_weights.mean().item()),
            "std": float(concept_weights.std().item()),
            "min": float(concept_weights.min().item()),
            "max": float(concept_weights.max().item()),
            "abs_mean": float(concept_weights.abs().mean().item()),
            "abs_sum": float(concept_weights.abs().sum().item())
        })
    
    top_concepts_per_class = []
    bottom_concepts_per_class = []
    
    for class_id in range(num_classes):
        class_weights = weights[class_id, :]
        top_k = min(10, num_concepts)
        top_indices = torch.topk(class_weights.abs(), k=top_k).indices.tolist()
        bottom_indices = torch.topk(class_weights.abs(), k=top_k, largest=False).indices.tolist()
        
        top_concepts_per_class.append([
            {
                "concept_idx": int(idx),
                "concept_name": concepts[idx],
                "weight": float(class_weights[idx].item())
            }
            for idx in top_indices
        ])
        
        bottom_concepts_per_class.append([
            {
                "concept_idx": int(idx),
                "concept_name": concepts[idx],
                "weight": float(class_weights[idx].item())
            }
            for idx in bottom_indices
        ])
    
    concept_importance = []
    for c in range(num_concepts):
        importance = weights[:, c].abs().sum().item()
        concept_importance.append({
            "concept_idx": c,
            "concept_name": concepts[c],
            "importance_score": float(importance)
        })
    
    concept_importance.sort(key=lambda x: x["importance_score"], reverse=True)
    
    weight_matrix = weights.tolist()
    
    return {
        "weight_statistics_per_concept": weight_stats_per_concept,
        "top_concepts_per_class": top_concepts_per_class,
        "bottom_concepts_per_class": bottom_concepts_per_class,
        "concept_importance_ranking": concept_importance,
        "bias_per_class": bias.tolist(),
        "weight_matrix": weight_matrix
    }


def get_concept_confusion_analysis(model, dataset, device, concepts, classes, batch_size=250, num_workers=2):
    model.eval()
    confusion_by_concept = {i: {} for i in range(len(concepts))}
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)):
            images = images.to(device)
            labels = labels.to(device)
            
            outs, concept_acts = model(images, return_concepts=True)
            preds = torch.argmax(outs, dim=1)
            
            for i in range(len(images)):
                label = int(labels[i].item())
                pred = int(preds[i].item())
                
                if label != pred:
                    concept_vals = concept_acts[i].cpu()
                    top_concept_idx = int(torch.argmax(concept_vals.abs()).item())
                    
                    key = f"{classes[label]}_vs_{classes[pred]}"
                    if key not in confusion_by_concept[top_concept_idx]:
                        confusion_by_concept[top_concept_idx][key] = 0
                    confusion_by_concept[top_concept_idx][key] += 1
    
    most_confused_concepts = []
    for c in range(len(concepts)):
        total_confusions = sum(confusion_by_concept[c].values())
        if total_confusions > 0:
            most_confused_concepts.append({
                "concept_idx": c,
                "concept_name": concepts[c],
                "total_confusions": total_confusions,
                "confusion_pairs": confusion_by_concept[c]
            })
    
    most_confused_concepts.sort(key=lambda x: x["total_confusions"], reverse=True)
    
    return {
        "most_confused_concepts": most_confused_concepts[:20],
        "confusion_by_concept": {str(k): v for k, v in confusion_by_concept.items()}
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Federated Label-Free Concept Bottleneck Model")
    parser.add_argument("--load_dir", type=str, required=True, help="Directory containing saved model")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--show_weights", action="store_true", help="Show final layer weights for classes")
    parser.add_argument("--show_sparsity", action="store_true", help="Show sparsity statistics")
    parser.add_argument("--save_results", type=str, default=None, help="Path to save evaluation results as JSON")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print('device', device)
    print(f"Loading model from {args.load_dir}")
    model, saved_args = load_fed_cbm(args.load_dir, device)
    
    dataset_name = saved_args["dataset"]
    
    concept_file = os.path.join(args.load_dir, "concepts.txt")
    if os.path.exists(concept_file):
        print(f"Loading concepts from {concept_file}")
        with open(concept_file, 'r') as f:
            concepts = [line.strip() for line in f.readlines() if line.strip()]
    else:
        saved_concept_file = saved_args.get("concept_file", None)
        if saved_concept_file:
            print(f"Loading concepts from saved concept_file: {saved_concept_file}")
            concepts = load_concepts_from_file(saved_concept_file)
        else:
            raise FileNotFoundError(f"Concept file not found at {concept_file}. Please ensure concepts.txt exists in {args.load_dir} or retrain the model.")
    classes = get_classes(dataset_name)
    
    if hasattr(model, "backbone") and hasattr(model.backbone, "preprocess"):
        target_preprocess = model.backbone.preprocess
    else:
        _, target_preprocess = get_target_model(saved_args["backbone"], device)

    if _is_vlg_checkpoint(args.load_dir):
        try:
            eval_d_probe = dataset_name + "_test"
            eval_data_t = get_data(eval_d_probe, preprocess=target_preprocess)
        except Exception:
            eval_d_probe = dataset_name + "_val"
            eval_data_t = get_data(eval_d_probe, preprocess=target_preprocess)
        print(f"VLG checkpoint: using {eval_d_probe} for metrics (matches VLG test accuracy when test exists)")
    else:
        eval_d_probe = dataset_name + "_val"
        eval_data_t = get_data(eval_d_probe, preprocess=target_preprocess)

    print("\n=== Measuring Accuracy ===")
    accuracy = get_accuracy_cbm(model, eval_data_t, device, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"Accuracy: {accuracy*100:.2f}%")

    per_class_accuracies = None
    sparsity_vlg = None
    if _is_vlg_checkpoint(args.load_dir):
        print("\n=== Per-class accuracy (VLG format) ===")
        per_class_accuracies = get_per_class_accuracy_vlg(model, eval_data_t, device, classes, batch_size=args.batch_size, num_workers=args.num_workers)
        sparsity_vlg = get_sparsity_vlg(model.final_layer)
        print(f"Overall: {per_class_accuracies['Overall accuracy']}%")
        metrics_txt_path = os.path.join(args.load_dir, "metrics.txt")
        if not os.path.exists(metrics_txt_path):
            vlg_metrics = {
                "per_class_accuracies": per_class_accuracies,
                "lam": -1.0, "lr": -1.0, "alpha": -1.0, "time": -1.0,
                "metrics": {"test_accuracy": float(accuracy)},
                "sparsity": sparsity_vlg,
            }
            with open(metrics_txt_path, "w") as f:
                json.dump(vlg_metrics, f, indent=2)
    
    print("\n=== Computing Detailed Concept Metrics ===")
    concept_metrics = get_detailed_concept_metrics(model, eval_data_t, device, batch_size=args.batch_size, num_workers=args.num_workers)

    print("\n=== Computing Weight Statistics ===")
    weight_metrics = get_weight_statistics(model, concepts, classes)

    print("\n=== Computing Concept Confusion Analysis ===")
    confusion_metrics = get_concept_confusion_analysis(model, eval_data_t, device, concepts, classes, batch_size=args.batch_size, num_workers=args.num_workers)
    
    results = {
        "model_dir": args.load_dir,
        "dataset": dataset_name,
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100),
        "num_concepts": len(concepts),
        "num_classes": len(classes),
        "backbone": saved_args.get("backbone", "unknown"),
        "clip_name": saved_args.get("clip_name", "unknown"),
        "concept_metrics": concept_metrics,
        "weight_metrics": weight_metrics,
        "confusion_metrics": confusion_metrics
    }
    # Include training config when present (from new naming / args.txt)
    if "num_clients" in saved_args:
        results["num_clients"] = saved_args["num_clients"]
    if "num_rounds" in saved_args:
        results["num_rounds"] = saved_args["num_rounds"]
    if "saga_n_iters" in saved_args:
        results["saga_n_iters"] = saved_args["saga_n_iters"]
    if "final_rounds" in saved_args:
        results["final_rounds"] = saved_args["final_rounds"]
    if per_class_accuracies is not None:
        results["per_class_accuracies"] = per_class_accuracies
    if sparsity_vlg is not None:
        results["sparsity"] = sparsity_vlg
    
    if args.show_weights:
        print("\n=== Final Layer Weights ===")
        final_weights = getattr(model.final_layer, "linear", model.final_layer).weight.data
        
        to_show = random.choices([i for i in range(len(classes))], k=min(3, len(classes)))
        for i in to_show:
            print(f"\nOutput class: {i} - {classes[i]}")
            print("Incoming weights:")
            for j in range(len(concepts)):
                weight = final_weights[i, j].item()
                if abs(weight) > 0.05:
                    concept_name = concepts[j]
                    if weight < 0:
                        concept_name = f"NOT {concept_name}"
                        weight = abs(weight)
                    print(f"{concept_name} [{weight:.4f}] {classes[i]}")
    
    if args.show_sparsity:
        print("\n=== Sparsity Statistics ===")
        final_weights = getattr(model.final_layer, "linear", model.final_layer).weight.data
        weight_contribs = torch.sum(torch.abs(final_weights), dim=0)
        num_concepts_used = torch.sum(weight_contribs > 1e-5).item()
        print(f"Num concepts with outgoing weights: {num_concepts_used}/{len(weight_contribs)}")
        if sparsity_vlg is None:
            results["sparsity"] = {
                "num_concepts_used": int(num_concepts_used),
                "total_concepts": int(len(weight_contribs)),
                "sparsity_ratio": float(num_concepts_used / len(weight_contribs))
            }
        
        top_weights, top_weight_ids = torch.topk(final_weights, k=5, dim=1)
        bottom_weights, bottom_weight_ids = torch.topk(final_weights, k=5, dim=1, largest=False)
        
        to_show = random.choices([i for i in range(len(classes))], k=min(2, len(classes)))
        for i in to_show:
            print(f"\nClass {i} - {classes[i]}")
            out = "Highest weights: "
            for j in range(top_weights.shape[1]):
                idx = int(top_weight_ids[i, j].cpu())
                out += f"{concepts[idx]}:{top_weights[i, j]:.3f}, "
            print(out)
            out = "Lowest weights: "
            for j in range(bottom_weights.shape[1]):
                idx = int(bottom_weight_ids[i, j].cpu())
                out += f"{concepts[idx]}:{bottom_weights[i, j]:.3f}, "
            print(out)
    
    if args.save_results:
        print('RESULTS')
        os.makedirs(os.path.dirname(args.save_results) if os.path.dirname(args.save_results) else ".", exist_ok=True)
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")
    else:
        results_path = os.path.join(args.load_dir, "evaluation_results.json")
        print('RESULTS', results_path)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
