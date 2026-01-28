import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_evaluation_results(results_file):
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_concept_statistics(results, save_dir=None, show=True):
    concept_stats = results["concept_metrics"]["concept_statistics"]
    
    means = [s["mean"] for s in concept_stats]
    stds = [s["std"] for s in concept_stats]
    mins = [s["min"] for s in concept_stats]
    maxs = [s["max"] for s in concept_stats]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].hist(means, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Mean Activation', fontsize=11)
    axes[0, 0].set_ylabel('Number of Concepts', fontsize=11)
    axes[0, 0].set_title('Distribution of Mean Concept Activations', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(stds, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Std Activation', fontsize=11)
    axes[0, 1].set_ylabel('Number of Concepts', fontsize=11)
    axes[0, 1].set_title('Distribution of Concept Activation Std', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(means, stds, alpha=0.5, s=20)
    axes[1, 0].set_xlabel('Mean Activation', fontsize=11)
    axes[1, 0].set_ylabel('Std Activation', fontsize=11)
    axes[1, 0].set_title('Mean vs Std of Concept Activations', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(range(len(means)), sorted(means, reverse=True), 'o-', markersize=3)
    axes[1, 1].set_xlabel('Concept Rank', fontsize=11)
    axes[1, 1].set_ylabel('Mean Activation', fontsize=11)
    axes[1, 1].set_title('Sorted Mean Activations', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "concept_statistics.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_concept_activations_by_class(results, concepts, classes, save_dir=None, show=True, top_k=10):
    concept_acts_by_class = results["concept_metrics"]["concept_activations_by_class"]
    
    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, 1, figsize=(14, num_classes * 1.5))
    
    if num_classes == 1:
        axes = [axes]
    
    for class_id, class_name in enumerate(classes):
        class_acts = np.array(concept_acts_by_class[class_id])
        top_indices = np.argsort(np.abs(class_acts))[::-1][:top_k]
        
        top_concepts = [concepts[i] for i in top_indices]
        top_acts = class_acts[top_indices]
        
        colors = ['red' if a > 0 else 'blue' for a in top_acts]
        axes[class_id].barh(range(len(top_concepts)), top_acts, color=colors, alpha=0.7)
        axes[class_id].set_yticks(range(len(top_concepts)))
        axes[class_id].set_yticklabels(top_concepts, fontsize=8)
        axes[class_id].set_xlabel('Mean Activation', fontsize=9)
        axes[class_id].set_title(f'{class_name} - Top {top_k} Concepts', fontsize=10)
        axes[class_id].axvline(0, color='black', linestyle='-', linewidth=0.5)
        axes[class_id].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "concept_activations_by_class.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_weight_statistics(results, concepts, classes, save_dir=None, show=True):
    weight_stats = results["weight_metrics"]["weight_statistics_per_concept"]
    importance = results["weight_metrics"]["concept_importance_ranking"]
    
    abs_means = [s["abs_mean"] for s in weight_stats]
    abs_sums = [s["abs_sum"] for s in weight_stats]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].hist(abs_means, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Mean Absolute Weight', fontsize=11)
    axes[0, 0].set_ylabel('Number of Concepts', fontsize=11)
    axes[0, 0].set_title('Distribution of Mean Absolute Weights', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(abs_sums, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Sum of Absolute Weights', fontsize=11)
    axes[0, 1].set_ylabel('Number of Concepts', fontsize=11)
    axes[0, 1].set_title('Distribution of Sum of Absolute Weights', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    top_20_importance = importance[:20]
    top_names = [item["concept_name"][:30] for item in top_20_importance]
    top_scores = [item["importance_score"] for item in top_20_importance]
    
    axes[1, 0].barh(range(len(top_names)), top_scores, color='green', alpha=0.7)
    axes[1, 0].set_yticks(range(len(top_names)))
    axes[1, 0].set_yticklabels(top_names, fontsize=8)
    axes[1, 0].set_xlabel('Importance Score', fontsize=11)
    axes[1, 0].set_title('Top 20 Most Important Concepts', fontsize=12)
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    axes[1, 1].plot(range(len(importance)), [item["importance_score"] for item in importance], 'o-', markersize=2)
    axes[1, 1].set_xlabel('Concept Rank', fontsize=11)
    axes[1, 1].set_ylabel('Importance Score', fontsize=11)
    axes[1, 1].set_title('Concept Importance Ranking', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "weight_statistics.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_top_concepts_per_class(results, concepts, classes, save_dir=None, show=True, top_k=5):
    top_concepts = results["weight_metrics"]["top_concepts_per_class"]
    
    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, 1, figsize=(14, num_classes * 1.2))
    
    if num_classes == 1:
        axes = [axes]
    
    for class_id, class_name in enumerate(classes):
        class_top = top_concepts[class_id][:top_k]
        concept_names = [item["concept_name"][:40] for item in class_top]
        weights = [item["weight"] for item in class_top]
        
        colors = ['red' if w > 0 else 'blue' for w in weights]
        axes[class_id].barh(range(len(concept_names)), weights, color=colors, alpha=0.7)
        axes[class_id].set_yticks(range(len(concept_names)))
        axes[class_id].set_yticklabels(concept_names, fontsize=8)
        axes[class_id].set_xlabel('Weight Value', fontsize=9)
        axes[class_id].set_title(f'{class_name} - Top {top_k} Concepts by Weight', fontsize=10)
        axes[class_id].axvline(0, color='black', linestyle='-', linewidth=0.5)
        axes[class_id].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "top_concepts_per_class.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_concept_confusion(results, concepts, save_dir=None, show=True, top_k=15):
    most_confused = results["confusion_metrics"]["most_confused_concepts"][:top_k]
    
    if not most_confused:
        print("No confusion data available")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    concept_names = [item["concept_name"][:40] for item in most_confused]
    confusion_counts = [item["total_confusions"] for item in most_confused]
    
    ax.barh(range(len(concept_names)), confusion_counts, color='purple', alpha=0.7)
    ax.set_yticks(range(len(concept_names)))
    ax.set_yticklabels(concept_names, fontsize=9)
    ax.set_xlabel('Number of Confusions', fontsize=11)
    ax.set_title(f'Top {top_k} Most Confused Concepts', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "concept_confusion.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_weight_heatmap_detailed(results, concepts, classes, save_dir=None, show=True, top_k_concepts=30):
    if "weight_matrix" in results["weight_metrics"]:
        weight_matrix_full = np.array(results["weight_metrics"]["weight_matrix"])
        importance = results["weight_metrics"]["concept_importance_ranking"]
        top_concept_indices = [item["concept_idx"] for item in importance[:top_k_concepts]]
        
        weights_matrix = weight_matrix_full[:, top_concept_indices]
        top_concept_names = [concepts[i][:30] for i in top_concept_indices]
    else:
        importance = results["weight_metrics"]["concept_importance_ranking"]
        top_concept_indices = [item["concept_idx"] for item in importance[:top_k_concepts]]
        
        top_concept_names = [concepts[i][:30] for i in top_concept_indices]
        
        weights_data = []
        for class_id in range(len(classes)):
            class_weights = []
            for concept_idx in top_concept_indices:
                weight = 0.0
                for item in results["weight_metrics"]["top_concepts_per_class"][class_id]:
                    if item["concept_idx"] == concept_idx:
                        weight = item["weight"]
                        break
                if weight == 0.0:
                    for item in results["weight_metrics"]["bottom_concepts_per_class"][class_id]:
                        if item["concept_idx"] == concept_idx:
                            weight = item["weight"]
                            break
                if weight == 0.0:
                    weight_stats = results["weight_metrics"]["weight_statistics_per_concept"][concept_idx]
                    weight = weight_stats["mean"]
                class_weights.append(weight)
            weights_data.append(class_weights)
        
        weights_matrix = np.array(weights_data)
    
    fig, ax = plt.subplots(figsize=(max(14, top_k_concepts * 0.5), max(8, len(classes) * 0.5)))
    
    im = ax.imshow(weights_matrix, aspect='auto', cmap='RdBu_r', 
                   vmin=-np.abs(weights_matrix).max(), vmax=np.abs(weights_matrix).max())
    
    ax.set_xticks(np.arange(len(top_concept_names)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(top_concept_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(classes, fontsize=9)
    
    ax.set_xlabel('Concepts', fontsize=11)
    ax.set_ylabel('Classes', fontsize=11)
    ax.set_title(f'Weight Heatmap (Top {top_k_concepts} Concepts)', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Weight Value')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "weight_heatmap_detailed.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Detailed Evaluation Results")
    parser.add_argument("--results_file", type=str, required=True, help="Path to evaluation_results.json")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save visualizations")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        raise FileNotFoundError(f"Results file not found: {args.results_file}")
    
    results = load_evaluation_results(args.results_file)
    
    print("Loading concepts and classes...")
    results_dir = os.path.dirname(args.results_file)
    model_dir = results["model_dir"]
    dataset = results.get("dataset", "unknown")
    
    concepts_file = None
    search_paths = [
        os.path.join(model_dir, "concepts.txt"),
        os.path.join(results_dir, "concepts.txt"),
        os.path.join(results_dir, "saved_models", os.path.basename(model_dir), "concepts.txt")
    ]
    
    if os.path.exists(results_dir):
        saved_models_dir = os.path.join(results_dir, "saved_models")
        if os.path.exists(saved_models_dir):
            for subdir in os.listdir(saved_models_dir):
                potential_file = os.path.join(saved_models_dir, subdir, "concepts.txt")
                if os.path.exists(potential_file):
                    concepts_file = potential_file
                    break
    
    if not concepts_file:
        for path in search_paths:
            if os.path.exists(path):
                concepts_file = path
                break
    
    if not concepts_file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        label_free_cbm_path = os.path.join(parent_dir, "Label-free-CBM", "data", "concept_sets", f"{dataset}_filtered.txt")
        if os.path.exists(label_free_cbm_path):
            concepts_file = label_free_cbm_path
    
    if concepts_file and os.path.exists(concepts_file):
        with open(concepts_file, 'r') as f:
            concepts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(concepts)} concepts from {concepts_file}")
    else:
        print(f"Warning: concepts.txt not found. Searched: {search_paths}")
        print(f"Using generic concept names")
        concepts = [f"Concept {i}" for i in range(results["num_concepts"])]
    
    classes = [f"Class {i}" for i in range(results["num_classes"])]
    
    dataset = results.get("dataset", "unknown")
    if dataset == "cifar10":
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset == "cifar100":
        class_file = "fed_lfc_cbm/data/cifar100_classes.txt"
        if not os.path.exists(class_file):
            class_file = os.path.join(os.path.dirname(__file__), "data", "cifar100_classes.txt")
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
        else:
            print(f"Warning: {class_file} not found, using generic class names")
    elif dataset == "imagenet":
        class_file = "fed_lfc_cbm/data/imagenet_classes.txt"
        if not os.path.exists(class_file):
            class_file = os.path.join(os.path.dirname(__file__), "data", "imagenet_classes.txt")
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
        else:
            print(f"Warning: {class_file} not found, using generic class names")
    
    print(f"Dataset: {results['dataset']}")
    print(f"Accuracy: {results['accuracy_percent']:.2f}%")
    print(f"Concepts: {results['num_concepts']}, Classes: {results['num_classes']}")
    
    print("\nGenerating visualizations...")
    
    plot_concept_statistics(results, save_dir=args.save_dir, show=args.show)
    plot_concept_activations_by_class(results, concepts, classes, save_dir=args.save_dir, show=args.show)
    plot_weight_statistics(results, concepts, classes, save_dir=args.save_dir, show=args.show)
    plot_top_concepts_per_class(results, concepts, classes, save_dir=args.save_dir, show=args.show)
    plot_concept_confusion(results, concepts, save_dir=args.save_dir, show=args.show)
    plot_weight_heatmap_detailed(results, concepts, classes, save_dir=args.save_dir, show=args.show)
    
    print("\nVisualization complete!")
    if args.save_dir:
        print(f"All visualizations saved to {args.save_dir}")

if __name__ == "__main__":
    main()
