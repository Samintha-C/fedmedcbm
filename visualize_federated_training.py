import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_training_metrics(metrics_file):
    with open(metrics_file, 'r') as f:
        return json.load(f)

def plot_accuracy_over_rounds(metrics, save_dir=None, show=True):
    final_phase = metrics.get("final_layer_phase")
    if final_phase is None:
        return  # No final layer phase

    # Check if this is centralized training (hybrid_saga, fedavg for VLG) or federated
    if "rounds" not in final_phase:
        # Centralized training - no rounds to plot
        print("Skipping accuracy plot - centralized final layer training (no rounds)")
        return

    rounds = final_phase["rounds"]
    accuracies = final_phase.get("global_accuracy") or final_phase.get("val_accuracy", [])
    best_accuracies = final_phase.get("best_accuracy") or final_phase.get("best_val_accuracy", [])

    if not rounds or not accuracies:
        print("Skipping accuracy plot - no round data available")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, [a * 100 for a in accuracies], 'o-', label='Global Accuracy', linewidth=2, markersize=8)
    if best_accuracies:
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
    final_phase = metrics.get("final_layer_phase")

    # Check if final phase has federated rounds (not centralized)
    has_final_federated = (final_phase is not None and
                           "rounds" in final_phase and
                           "client_losses" in final_phase)

    n_axes = 2 if has_final_federated else 1
    fig, axes = plt.subplots(1, n_axes, figsize=(8 * n_axes, 6))
    if n_axes == 1:
        axes = [axes]
    ax1, ax2 = axes[0], (axes[1] if has_final_federated else None)

    proj_rounds = projection_phase["rounds"]
    proj_client_losses = projection_phase["client_losses"]
    proj_avg_losses = projection_phase["avg_client_loss"]

    for client_id in range(len(proj_client_losses[0])):
        client_losses = [losses[client_id] for losses in proj_client_losses]
        ax1.plot(proj_rounds, client_losses, 'o-', label=f'Client {client_id}', alpha=0.7, markersize=4)
    ax1.plot(proj_rounds, proj_avg_losses, 'k--', label='Average', linewidth=2)
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('CBL / Projection: Client Losses', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    if has_final_federated:
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
    final_phase = metrics.get("final_layer_phase")

    # Check if final phase has federated rounds
    has_final_federated = (final_phase is not None and
                           "rounds" in final_phase and
                           "avg_client_loss" in final_phase)

    n_axes = 2 if has_final_federated else 1
    fig, axes = plt.subplots(1, n_axes, figsize=(8 * n_axes, 6))
    if n_axes == 1:
        axes = [axes]
    ax1, ax2 = axes[0], (axes[1] if has_final_federated else None)

    proj_rounds = projection_phase["rounds"]
    proj_avg_losses = projection_phase["avg_client_loss"]
    proj_best_losses = projection_phase.get("best_proj_loss") or projection_phase.get("best_val_loss")
    if isinstance(proj_best_losses, list):
        pass
    else:
        proj_best_losses = [proj_best_losses] * len(proj_rounds) if proj_best_losses is not None else proj_avg_losses

    ax1.plot(proj_rounds, proj_avg_losses, 'o-', label='Average Loss', linewidth=2, markersize=6)
    ax1.plot(proj_rounds, proj_best_losses, 's--', label='Best Loss', linewidth=2, markersize=5)
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('CBL / Projection: Loss vs Rounds', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    if has_final_federated:
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

    print("=" * 60)
    print("FEDERATED TRAINING SUMMARY")
    print("=" * 60)
    print(f"Number of clients: {metrics.get('num_clients', '?')}")
    print(f"Number of rounds: {metrics.get('num_rounds', '?')}")
    print(f"IID distribution: {metrics.get('iid', '?')}")
    if not metrics.get('iid'):
        print(f"Dirichlet alpha: {metrics.get('alpha', '?')}")

    final_method = metrics.get('final_layer_method', 'unknown')
    print(f"Final layer method: {final_method}")

    final_phase = metrics.get('final_layer_phase')
    if final_phase:
        if final_method == 'hybrid_saga':
            print(f"  - Sparsity: {final_phase.get('sparsity_nnz', '?')}/{final_phase.get('sparsity_total', '?')} non-zero")
            if 'saga_lam' in final_phase:
                print(f"  - SAGA lambda: {final_phase['saga_lam']}")
        elif final_method == 'fedavg_thresh':
            print(f"  - Final rounds: {metrics.get('final_rounds', '?')}")
            if 'sparsity_nnz' in final_phase:
                print(f"  - Sparsity: {final_phase['sparsity_nnz']}/{final_phase['sparsity_total']} non-zero")
        elif final_method == 'fedavg':
            print(f"  - Final rounds: {metrics.get('final_rounds', '?')}")
            print(f"  - Dense final layer (no sparsity)")

    bfa = metrics.get('best_final_accuracy')
    if bfa is not None:
        print(f"Best final accuracy: {bfa*100:.2f}%")
    print("=" * 60)

    print("\nGenerating visualizations...")
    plot_accuracy_over_rounds(metrics, save_dir=args.save_dir, show=args.show)
    plot_client_losses(metrics, save_dir=args.save_dir, show=args.show)
    plot_loss_over_rounds(metrics, save_dir=args.save_dir, show=args.show)
    plot_client_data_distribution(metrics, save_dir=args.save_dir, show=args.show)

    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
