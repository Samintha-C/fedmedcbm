import argparse
import json
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import sys
import importlib.util

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

utils_concepts_path = os.path.join(current_dir, 'utils', 'concepts.py')
data_utils_path = os.path.join(current_dir, 'data', 'data_utils.py')
plots_path = os.path.join(current_dir, 'visualization', 'plots.py')

spec_concepts = importlib.util.spec_from_file_location("fed_utils_concepts", utils_concepts_path)
spec_data = importlib.util.spec_from_file_location("fed_data_utils", data_utils_path)
spec_plots = importlib.util.spec_from_file_location("plots", plots_path)

fed_utils_concepts = importlib.util.module_from_spec(spec_concepts)
fed_data_utils = importlib.util.module_from_spec(spec_data)
plots = importlib.util.module_from_spec(spec_plots)

spec_concepts.loader.exec_module(fed_utils_concepts)
spec_data.loader.exec_module(fed_data_utils)
spec_plots.loader.exec_module(plots)

from models.fed_lfc import FedLFC_CBM
from evaluate_fed_cbm import load_fed_cbm

load_concepts_from_file = fed_utils_concepts.load_concepts_from_file
get_data = fed_data_utils.get_data
get_classes = fed_data_utils.get_classes
get_target_model = fed_data_utils.get_target_model


def visualize_individual_prediction(model, dataset, dataset_pil, idx, concepts, classes, device, save_dir=None):
    image, label = dataset_pil[idx]
    x, _ = dataset[idx]
    x = x.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs, concept_act = model(x, return_concepts=True)
        conf = F.softmax(outputs[0], dim=0)
        top_logit, top_class = torch.max(outputs[0], dim=0)
        
        print(f"Image {idx}: GT={classes[int(label)]}, Pred={classes[top_class]}, Confidence={conf[top_class]:.3f}")
        
        concept_vals, top_concepts = torch.topk(concept_act[0], k=5)
        print("Top concept activations:")
        for i in range(len(concept_vals)):
            print(f"  {concepts[int(top_concepts[i])]}: {concept_vals[i]:.3f}")
        
        contributions = concept_act[0] * model.final_layer.linear.weight[top_class, :]
        feature_names = [("NOT " if concept_act[0][i] < 0 else "") + concepts[i] for i in range(len(concepts))]
        values = contributions.cpu().numpy()
        max_display = min(int(sum(abs(values) > 0.005)) + 1, 10)
        
        bias = model.final_layer.linear.bias[top_class].item()
        title = f"Pred:{classes[top_class]} - Conf:{conf[top_class]:.3f} - Logit:{top_logit:.2f} - Bias:{bias:.2f}"
        
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"prediction_{idx}_{classes[top_class]}.png")
        
        plots.bar(values, feature_names, max_display=max_display, title=title, fontsize=12, save_path=save_path, show=False)
        
        return image, label, top_class, conf[top_class]


def visualize_weights(model, concepts, classes, save_dir=None):
    weights = model.final_layer.linear.weight.data.cpu().numpy()
    bias = model.final_layer.linear.bias.data.cpu().numpy()
    
    print(f"\nWeight matrix shape: {weights.shape}")
    print(f"Bias shape: {bias.shape}")
    
    save_path_heatmap = None
    save_path_top = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path_heatmap = os.path.join(save_dir, "weight_heatmap.png")
        save_path_top = os.path.join(save_dir, "top_concepts_per_class.png")
    
    plots.plot_weight_heatmap(weights, classes, concepts, top_k_concepts=20, save_path=save_path_heatmap, show=False)
    plots.plot_top_concepts_per_class(weights, classes, concepts, top_k=5, save_path=save_path_top, show=False)
    
    print(f"\nTop concepts per class:")
    for i, class_name in enumerate(classes):
        class_weights = weights[i, :]
        top_indices = np.argsort(np.abs(class_weights))[::-1][:5]
        print(f"\n{class_name}:")
        for idx in top_indices:
            weight = class_weights[idx]
            concept_name = concepts[idx]
            if weight < 0:
                concept_name = f"NOT {concept_name}"
            print(f"  {concept_name}: {abs(weight):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Federated CBM Results")
    parser.add_argument("--load_dir", type=str, required=True, help="Directory containing saved model")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of sample predictions to visualize")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save visualizations")
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Loading model from {args.load_dir}")
    model, saved_args = load_fed_cbm(args.load_dir, device)
    
    dataset_name = saved_args["dataset"]
    
    concept_file = os.path.join(args.load_dir, "concepts.txt")
    if os.path.exists(concept_file):
        with open(concept_file, 'r') as f:
            concepts = [line.strip() for line in f.readlines() if line.strip()]
    else:
        saved_concept_file = saved_args.get("concept_file", None)
        if saved_concept_file:
            concepts = load_concepts_from_file(saved_concept_file)
        else:
            raise FileNotFoundError(f"Concept file not found at {concept_file}")
    
    classes = get_classes(dataset_name)
    
    _, target_preprocess = get_target_model(saved_args["backbone"], device)
    
    val_d_probe = dataset_name + "_val"
    val_data_t = get_data(val_d_probe, preprocess=target_preprocess)
    val_pil_data = get_data(val_d_probe)
    
    print(f"\n=== Visualizing Weights ===")
    visualize_weights(model, concepts, classes, save_dir=args.save_dir)
    
    print(f"\n=== Visualizing Individual Predictions ===")
    sample_indices = random.sample(range(len(val_data_t)), min(args.num_samples, len(val_data_t)))
    
    for idx in sample_indices:
        try:
            visualize_individual_prediction(
                model, val_data_t, val_pil_data, idx, concepts, classes, device, 
                save_dir=os.path.join(args.save_dir, "predictions") if args.save_dir else None
            )
        except Exception as e:
            print(f"Error visualizing sample {idx}: {e}")
            continue
    
    print(f"\nVisualization complete!")
    if args.save_dir:
        print(f"Visualizations saved to {args.save_dir}")


if __name__ == "__main__":
    main()
