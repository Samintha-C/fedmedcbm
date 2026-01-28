import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_training_metrics(metrics_file):
    with open(metrics_file, 'r') as f:
        return json.load(f)

def plot_accuracy_over_rounds(metrics, save_dir=None, show=True):
    final_phase = metrics["final_layer_phase"]
    rounds = final_phase["rounds"]
    accuracies = final_phase["global_accuracy"]
    best_accuracies = final_phase["best_accuracy"]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, [a * 100 for a in accuracies], 'o-', label='Global Accuracy', linewidth=2, markersize=8)
    plt.plot(rounds, [a * 100 for a in best_accuracies], 's--', label='Best Accuracy', linewidth=2, markersize=6)
    plt.xlabel('Federated Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Federated Learning: Accuracy vs Rounds', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "accuracy_over_rounds.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_client_losses(metrics, save_dir=None, show=True):
    projection_phase = metrics["projection_phase"]
    final_phase = metrics["final_layer_phase"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    proj_rounds = projection_phase["rounds"]
    proj_client_losses = projection_phase["client_losses"]
    proj_avg_losses = projection_phase["avg_client_loss"]
    
    for client_id in range(len(proj_client_losses[0])):
        client_losses = [losses[client_id] for losses in proj_client_losses]
        ax1.plot(proj_rounds, client_losses, 'o-', label=f'Client {client_id}', alpha=0.7, markersize=4)
    ax1.plot(proj_rounds, proj_avg_losses, 'k--', label='Average', linewidth=2)
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Projection Layer: Client Losses', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    final_rounds = final_phase["rounds"]
    final_client_losses = final_phase["client_losses"]
    final_avg_losses = final_phase["avg_client_loss"]
    
    for client_id in range(len(final_client_losses[0])):
        client_losses = [losses[client_id] for losses in final_client_losses]
        ax2.plot(final_rounds, client_losses, 'o-', label=f'Client {client_id}', alpha=0.7, markersize=4)
    ax2.plot(final_rounds, final_avg_losses, 'k--', label='Average', linewidth=2)
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Final Layer: Client Losses', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "client_losses.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_loss_over_rounds(metrics, save_dir=None, show=True):
    projection_phase = metrics["projection_phase"]
    final_phase = metrics["final_layer_phase"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    proj_rounds = projection_phase["rounds"]
    proj_avg_losses = projection_phase["avg_client_loss"]
    proj_best_losses = projection_phase["best_proj_loss"]
    
    ax1.plot(proj_rounds, proj_avg_losses, 'o-', label='Average Loss', linewidth=2, markersize=6)
    ax1.plot(proj_rounds, proj_best_losses, 's--', label='Best Loss', linewidth=2, markersize=5)
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Projection Layer: Loss vs Rounds', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    final_rounds = final_phase["rounds"]
    final_avg_losses = final_phase["avg_client_loss"]
    
    ax2.plot(final_rounds, final_avg_losses, 'o-', label='Average Loss', linewidth=2, markersize=6, color='green')
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Final Layer: Loss vs Rounds', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "loss_over_rounds.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_client_data_distribution(metrics, save_dir=None, show=True):
    client_sizes = metrics["client_data_sizes"]
    client_weights = metrics["client_weights"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    client_ids = list(range(len(client_sizes)))
    ax1.bar(client_ids, client_sizes, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Client ID', fontsize=11)
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.set_title('Client Data Distribution', fontsize=12)
    ax1.set_xticks(client_ids)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(client_ids, [w * 100 for w in client_weights], color='coral', alpha=0.7)
    ax2.set_xlabel('Client ID', fontsize=11)
    ax2.set_ylabel('Weight (%)', fontsize=11)
    ax2.set_title('Federated Averaging Weights', fontsize=12)
    ax2.set_xticks(client_ids)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "client_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Federated Training Metrics")
    parser.add_argument("--metrics_file", type=str, required=True, help="Path to training_metrics.json")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save visualizations")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {args.metrics_file}")
    
    metrics = load_training_metrics(args.metrics_file)
    
    print("Generating visualizations...")
    print(f"Number of clients: {metrics['num_clients']}")
    print(f"IID distribution: {metrics['iid']}")
    print(f"Best final accuracy: {metrics['best_final_accuracy']*100:.2f}%")
    
    plot_accuracy_over_rounds(metrics, save_dir=args.save_dir, show=args.show)
    plot_client_losses(metrics, save_dir=args.save_dir, show=args.show)
    plot_loss_over_rounds(metrics, save_dir=args.save_dir, show=args.show)
    plot_client_data_distribution(metrics, save_dir=args.save_dir, show=args.show)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
