import argparse
import json
import os
import sys

# Ensure project root is on sys.path before local imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from train_vlg import simulate_federated_training_vlg


def main():
    parser = argparse.ArgumentParser(description="Federated Label-Free Concept Bottleneck Model")

    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet", "cub", "places365"], help="Dataset name")
    parser.add_argument("--concept_file", type=str, required=True, help="Path to concept file")
    parser.add_argument("--filter_set", type=str, default=None,
        help="Path to file listing concepts to remove (one per line). Used to match vanilla VLG-CBM concept filtering.")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone type: resnet50 or clip_ViT-B/16")
    parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="CLIP model name")
    parser.add_argument("--use_clip_penultimate", action="store_true", help="Use CLIP penultimate layer")
    parser.add_argument("--use_vlg", action="store_true", help="Use VLG-CBM training (AllOne concepts, BCE/TwoWay loss, SAGA final layer)")
    parser.add_argument("--annotation_dir", type=str, default=None,
        help="Path to folder containing cifar100_train/ and cifar100_val/ (pre-generated Grounding DINO annotations). If set, use DINO concept labels instead of AllOne.")
    parser.add_argument("--dino_confidence_threshold", type=float, default=0.10,
        help="Min logit for DINO annotations to count (only when --annotation_dir is set)")
    parser.add_argument("--annotation_cache_dir", type=str, default=None,
        help="Directory to cache preloaded DINO annotation tensors. First run writes; "
             "subsequent runs with matching (dataset, split, confidence, concepts, indices) "
             "skip JSON reads. Critical for places365 (~1.6M files over NFS).")
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
    parser.add_argument("--final_layer_method", type=str, default=None,
        choices=["fedavg", "fedavg_thresh", "hybrid_saga", "feddualavg"],
        help="Final layer training method: fedavg (dense FedAvg for LFC, dense centralized for VLG), "
             "fedavg_thresh (FedAvg + server-side thresholding), "
             "hybrid_saga (federated feature extraction + centralized GLM-SAGA), "
             "feddualavg (Federated Dual Averaging with group-lasso proximal). "
             "Default: fedavg for LFC, hybrid_saga for VLG")
    parser.add_argument("--thresh_lam_start", type=float, default=0.01,
        help="Starting group-threshold lambda (compared to column L2 norms)")
    parser.add_argument("--thresh_lam_end", type=float, default=0.12,
        help="Ending group-threshold lambda (compared to column L2 norms)")
    parser.add_argument("--dual_eta_s", type=float, default=1.0,
        help="Server learning rate for FedDualAvg")
    parser.add_argument("--dual_eta_c", type=float, default=0.01,
        help="Client learning rate for FedDualAvg")
    parser.add_argument("--dual_lam", type=float, default=0.001,
        help="Group-lasso regularization lambda for FedDualAvg")
    parser.add_argument("--dual_lam_end", type=float, default=0.01,
        help="Ending eta_tilde for burnin schedule (decays from dual_lam to this value)")
    parser.add_argument("--dual_schedule", type=str, default="linear",
        choices=["linear", "burnin"],
        help="eta_tilde schedule for FedDualAvg: "
             "linear (grows over time, original), "
             "burnin (starts high and decays, SAGA-style)")
    parser.add_argument("--dual_warmup_rounds", type=int, default=0,
        help="Number of initial FedDualAvg rounds with lambda=0 (no regularization). "
             "Lets the dense solution converge before sparsity pressure kicks in.")

    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Save directory")
    parser.add_argument("--log_dir", type=str, default=None,
        help="Directory for structured log files (creates p1/ and p2/ subdirs)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for embeddings")

    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split (VLG)")
    parser.add_argument("--feature_layer", type=str, default="layer4", help="Backbone feature layer (VLG, non-CLIP)")
    parser.add_argument("--cbl_loss_type", type=str, default="bce", choices=["bce", "twoway", "cos_cubed"], help="CBL loss (VLG)")
    parser.add_argument("--cbl_lr", type=float, default=5e-4, help="CBL learning rate (VLG)")
    parser.add_argument("--cbl_epochs", type=int, default=20, help="CBL epochs per client round (VLG)")
    parser.add_argument("--cbl_batch_size", type=int, default=32, help="CBL batch size (VLG)")
    parser.add_argument("--cbl_optimizer", type=str, default="adam", choices=["adam", "sgd"], help="CBL optimizer (VLG)")
    parser.add_argument("--cbl_hidden_layers", type=int, default=0, help="CBL hidden layers (VLG)")
    parser.add_argument("--cbl_pos_weight", type=float, default=0.2, help="BCE positive weight (VLG)")
    parser.add_argument("--cbl_auto_weight", action="store_true", help="Automatically weight positive examples by neg/pos ratio per concept (VLG). Matches VLG-CBM --cbl_auto_weight flag.")
    parser.add_argument("--cbl_twoway_tp", type=float, default=4.0, help="TwoWay loss Tp (VLG)")
    parser.add_argument("--cbl_finetune", action="store_true", help="Finetune backbone in CBL (VLG)")
    parser.add_argument("--ortho_lambda", type=float, default=0.0,
        help="Weight orthogonality regularization coefficient for CBL training (VLG). "
             "Penalises aligned concept projection vectors via off-diagonal Gram-matrix loss. "
             "0.0 disables (default). Suggested values to sweep: 0.1, 0.5, 1.0.")
    parser.add_argument("--cbl_bb_lr_rate", type=float, default=1.0, help="Backbone LR scale in CBL (VLG)")
    parser.add_argument("--saga_lam", type=float, default=0.0007, help="SAGA sparsity lambda (VLG)")
    parser.add_argument("--saga_n_iters", type=int, default=2000, help="SAGA iterations (VLG)")
    parser.add_argument("--saga_step_size", type=float, default=0.1, help="SAGA step size (VLG)")
    parser.add_argument("--saga_batch_size", type=int, default=512, help="SAGA batch size (VLG)")
    parser.add_argument("--phase1_only", action="store_true",
        help="Run only Phase 1 (CBL training), save backbone+CBL, then exit. "
             "Use --load_cbl_dir in a subsequent job to run Phase 2+3.")
    parser.add_argument("--load_cbl_dir", type=str, default=None,
        help="Path to a Phase-1-only checkpoint directory containing backbone.pt and cbl.pt. "
             "When set, skips Phase 1 and runs Phase 2+3 directly.")
    parser.add_argument("--load_pretrained_vlg", type=str, default=None,
        help="Path to a pretrained VLG directory (e.g. saved_models/fed_vlg_cifar100_...) containing "
             "pre-extracted concept features (train_concept_features.pt, train_concept_labels.pt, "
             "val_concept_features.pt, val_concept_labels.pt) and model weights (backbone.pt, cbl.pt, "
             "train_concept_features_mean.pt, train_concept_features_std.pt). "
             "When set, skips Phases 1-2 and runs Phase 3 directly using the loaded features.")
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
    if args.final_layer_method is None:
        args.final_layer_method = "hybrid_saga"
    simulate_federated_training_vlg(args)


if __name__ == "__main__":
    main()
