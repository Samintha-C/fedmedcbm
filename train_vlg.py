import copy
import gc
import json
import os
import sys
import datetime
import importlib.util
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

# Ensure project root is on sys.path before local imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from glm_saga.elasticnet import IndexedTensorDataset, group_threshold
from tqdm import tqdm

from data import data_utils
from data.data_utils import LabelFlipSubset
from data.concept_dataset_vlg import AllOneConceptDataset, DinoConceptDataset, get_concept_dataloader
from models.fed_vlgcbm import (
    Backbone, BackboneCLIP, ConceptLayer, NormalizationLayer, FinalLayer, FedVLGCBM,
    train_cbl, validate_cbl, get_final_layer_dataset, train_sparse_final, train_dense_final, test_model, per_class_accuracy,
)

from training_utils import (
    set_seed, federated_averaging, save_args, save_training_metrics, save_metrics_txt,
    get_preprocess, create_client_data_loaders, log_mem, init_log, update_log,
    get_classes, split_dataset_for_federated, print_client_distribution, get_data,
)

current_dir = os.path.dirname(os.path.abspath(__file__))


def _get_loss_vlg():
    _loss_vlg_spec = importlib.util.spec_from_file_location("fed_loss_vlg", os.path.join(current_dir, "utils", "loss_vlg.py"))
    _loss_vlg_mod = importlib.util.module_from_spec(_loss_vlg_spec)
    _loss_vlg_spec.loader.exec_module(_loss_vlg_mod)
    return _loss_vlg_mod.get_loss


def _build_run_name(a):
    """Build a run name like 'p1-c5r10-vlg-cifar100-feb25-14:30'."""
    now = datetime.datetime.now()
    date_tag = now.strftime("%b%d-%H:%M").lower()
    method = (getattr(a, "final_layer_method", None) or "hybrid_saga").replace("_", "")
    if getattr(a, "phase1_only", False):
        return f"p1-c{a.num_clients}r{a.num_rounds}-vlg-{a.dataset}-{date_tag}"
    elif getattr(a, "load_cbl_dir", None) or getattr(a, "load_pretrained_vlg", None):
        r = getattr(a, "final_rounds", a.num_rounds)
        return f"c{a.num_clients}r{r}-vlg-{a.dataset}-{method}-{date_tag}"
    else:
        return f"c{a.num_clients}r{a.num_rounds}-vlg-{a.dataset}-{method}-{date_tag}"


def _step(label: str, t0: float) -> float:
    """Print a labelled elapsed-time line and return the new start time."""
    elapsed = time.time() - t0
    print(f"[TIMING] {label}: {elapsed:.1f}s", flush=True)
    return time.time()


def soft_threshold(z, lam):
    """
    Element-wise soft thresholding for standard L1 sparsity.
    Allows each class to independently drop concepts.
    """
    return torch.sign(z) * torch.clamp(torch.abs(z) - lam, min=0.0)


def elasticnet_threshold(z, lam_l1, lam_l2):
    """
    Proximal operator for elastic net: R(W) = λ₁‖W‖₁ + λ₂‖W‖²_F.
    prox(z)_{k,c} = sign(z) * max(|z| - η·λ₁, 0) / (1 + η·λ₂)
    lam_l1 and lam_l2 are the already-scaled η·λ values.
    """
    return soft_threshold(z, lam_l1) / (1.0 + lam_l2)


def simulate_federated_training_vlg(args):
    get_loss_vlg = _get_loss_vlg()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_mem("training start")
    _run_name = _build_run_name(args)
    _is_phase1 = getattr(args, "phase1_only", False)
    if _is_phase1:
        save_dir = os.path.join(args.save_dir, "projection_layers", _run_name)
    else:
        save_dir = os.path.join(args.save_dir, "fully_trained", _run_name)
    os.makedirs(save_dir, exist_ok=True)
    _log_path = init_log(getattr(args, "log_dir", None), _run_name, _is_phase1)
    print(f"RUN_NAME={_run_name}")
    print(f"SAVE_DIR={save_dir}")

    concepts = data_utils.get_concepts(args.concept_file, getattr(args, "filter_set", None))
    # When resuming from a Phase-1 checkpoint, the saved concepts.txt is authoritative:
    # the checkpoint's CBL width is fixed to that list, and any discrepancy with
    # --concept_file (e.g. dynamic filter having dropped zero-count concepts during
    # Phase 1) would cause load_state_dict size mismatch downstream.
    _cbl_dir_preload = getattr(args, "load_cbl_dir", None)
    if _cbl_dir_preload is not None:
        _ckpt_concepts_path = os.path.join(_cbl_dir_preload, "concepts.txt")
        if os.path.exists(_ckpt_concepts_path):
            with open(_ckpt_concepts_path) as _f:
                _ckpt_concepts = [line.strip() for line in _f if line.strip()]
            if _ckpt_concepts != concepts:
                print(f"[concept filter] overriding concept list with checkpoint's concepts.txt "
                      f"({len(concepts)} -> {len(_ckpt_concepts)} concepts) to match CBL width")
                concepts = _ckpt_concepts
        else:
            print(f"[WARN] {_ckpt_concepts_path} not found; using --concept_file as-is "
                  f"(CBL load may fail on width mismatch)")
    num_concepts = len(concepts)
    classes = get_classes(args.dataset)
    num_classes = len(classes)
    args.num_concepts = num_concepts
    args.num_classes = num_classes

    save_args(save_dir, args)

    load_dir = getattr(args, "load_pretrained_vlg", None)
    _pretrained_mode = load_dir is not None

    if _pretrained_mode:
        print(f"\n=== Pretrained VLG mode: loading pre-extracted features from {load_dir} ===")
        if args.backbone.startswith("clip_"):
            preprocess = get_preprocess(args.backbone)
            backbone = BackboneCLIP(args.backbone, use_penultimate=getattr(args, "use_clip_penultimate", True), device="cpu")
        else:
            preprocess = get_preprocess(args.backbone)
            backbone = Backbone(args.backbone, getattr(args, "feature_layer", "layer4"), "cpu")
        all_train_feats = torch.load(
            os.path.join(load_dir, "train_concept_features.pt"), map_location="cpu"
        )
        all_train_labels = torch.load(
            os.path.join(load_dir, "train_concept_labels.pt"), map_location="cpu"
        )
        val_feats = torch.load(
            os.path.join(load_dir, "val_concept_features.pt"), map_location="cpu"
        )
        val_labels_all = torch.load(
            os.path.join(load_dir, "val_concept_labels.pt"), map_location="cpu"
        )
        num_concepts = all_train_feats.shape[1]
        args.num_concepts = num_concepts
        num_train = all_train_feats.shape[0]
        print(f"Loaded features: {num_train} train, {val_feats.shape[0]} val, {num_concepts} concepts")

        cbl = ConceptLayer(
            backbone.output_dim, num_concepts,
            num_hidden=getattr(args, "cbl_hidden_layers", 0),
            bias=True, device="cpu"
        )
        _bb_pt = os.path.join(load_dir, "backbone.pt")
        if os.path.exists(_bb_pt):
            backbone.backbone.load_state_dict(torch.load(_bb_pt, map_location="cpu"))
        else:
            print(f"  [INFO] backbone.pt not found in {load_dir} — using pretrained weights (cbl_finetune=False checkpoint)")
        cbl.load_state_dict(
            torch.load(os.path.join(load_dir, "cbl.pt"), map_location="cpu")
        )
        norm_layer = NormalizationLayer.from_pretrained(load_dir, device=str(device))
        print("Loaded backbone, CBL, and normalization from pretrained dir")

        global_model = FedVLGCBM(backbone, cbl, normalization=norm_layer, final_layer=None)
        global_model.cbl.to(device)
        # backbone stays on CPU; will be moved to device only for final evaluation

        # Uniform client split of pre-extracted train features
        base = num_train // args.num_clients
        rem = num_train % args.num_clients
        client_data_sizes = [base + (1 if i < rem else 0) for i in range(args.num_clients)]
        total_samples = sum(client_data_sizes)
        client_weights = [n / total_samples for n in client_data_sizes]

        saga_bs = getattr(args, "saga_batch_size", 512)
        train_concept_loader = DataLoader(
            IndexedTensorDataset(all_train_feats, all_train_labels),
            batch_size=saga_bs, shuffle=True
        )
        val_concept_loader = DataLoader(
            TensorDataset(val_feats, val_labels_all),
            batch_size=saga_bs, shuffle=False
        )
        projection_metrics = {}
    else:
        _t = time.time()
        if args.backbone.startswith("clip_"):
            preprocess = get_preprocess(args.backbone)
            backbone = BackboneCLIP(args.backbone, use_penultimate=getattr(args, "use_clip_penultimate", True), device=str(device))
        else:
            preprocess = get_preprocess(args.backbone)
            backbone = Backbone(args.backbone, getattr(args, "feature_layer", "layer4"), str(device))
        _t = _step("backbone init", _t)

        full_train_dataset = get_data(f"{args.dataset}_train", preprocess=None)
        _t = _step("get_data (dataset load)", _t)

        val_split = getattr(args, "val_split", 0.1)
        n_val = int(val_split * len(full_train_dataset))
        n_train = len(full_train_dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )
        print(f"Split full train dataset: {len(train_dataset)} train, {len(val_dataset)} val (val_split={val_split})")
        # Sanity: random_split should produce disjoint Subsets. Verify to rule out
        # any silent leakage from upstream dataset quirks.
        _train_idx_set = set(train_dataset.indices)
        _val_idx_set = set(val_dataset.indices)
        _overlap = _train_idx_set & _val_idx_set
        assert not _overlap, f"train/val index overlap detected: {len(_overlap)} samples"
        print(f"[sanity] train/val disjoint: |train|={len(_train_idx_set)} |val|={len(_val_idx_set)} overlap=0")
        _t = _step("train/val split", _t)

        client_indices = split_dataset_for_federated(
            train_dataset, args.num_clients, iid=args.iid, alpha=args.alpha, seed=args.seed
        )
        _t = _step("split_dataset_for_federated", _t)
        print_client_distribution(train_dataset, client_indices, num_classes=num_classes)
        _t = _step("print_client_distribution", _t)

        annotation_dir = getattr(args, "annotation_dir", None)
        dino_conf = getattr(args, "dino_confidence_threshold", 0.10)
        ann_cache_dir = getattr(args, "annotation_cache_dir", None)
        if annotation_dir and os.path.isdir(annotation_dir):
            print(f"[TIMING] Building train DinoConceptDataset ({len(train_dataset)} samples)...", flush=True)
            base_cbl_dataset = DinoConceptDataset(
                args.dataset, train_dataset, concepts,
                annotation_dir=annotation_dir, split_suffix="train",
                confidence_threshold=dino_conf, preprocess=preprocess,
                cache_dir=ann_cache_dir,
            )
            _t = _step("DinoConceptDataset train (annotation preload)", _t)
            print(f"[TIMING] Building val DinoConceptDataset ({len(val_dataset)} samples)...", flush=True)
            val_cbl_dataset = DinoConceptDataset(
                args.dataset, val_dataset, concepts,
                annotation_dir=annotation_dir, split_suffix="train",
                confidence_threshold=dino_conf, preprocess=preprocess,
                cache_dir=ann_cache_dir,
            )
            _t = _step("DinoConceptDataset val (annotation preload)", _t)
        else:
            base_cbl_dataset = AllOneConceptDataset(args.dataset, train_dataset, concepts, preprocess)
            val_cbl_dataset = AllOneConceptDataset(args.dataset, val_dataset, concepts, preprocess)
            _t = _step("AllOneConceptDataset init", _t)

        # VLG-CBM-style concept filtering: drop concepts with zero DINO presence in the
        # training set. Matches VLG-CBM/data/concept_dataset.py:get_filtered_concepts_and_counts.
        # Only applies when DINO annotations are in use (AllOne has no real counts).
        if annotation_dir and os.path.isdir(annotation_dir):
            log_mem("concept filter: start (before sum)")
            # Chunked sum across rows. Naive bool.sum(dim=0) on a 4GB+ tensor can
            # blow up to int64 intermediates (N × C × 8 bytes ≈ 35GB on places365)
            # depending on the kernel path. Streaming in chunks bounds the
            # working set to chunk_rows × C bytes for the bool view + small
            # int64 accumulator.
            _ann = base_cbl_dataset._annotation_cache
            _N, _C = _ann.shape
            _counts = torch.zeros(_C, dtype=torch.long)
            _chunk = 65536
            for _start in range(0, _N, _chunk):
                _counts += _ann[_start:_start + _chunk].sum(dim=0).long()
            log_mem("concept filter: after chunked sum")
            keep_mask = _counts > 0
            if not bool(keep_mask.all()):
                removed_concepts = [c for c, k in zip(concepts, keep_mask.tolist()) if not k]
                concepts = [c for c, k in zip(concepts, keep_mask.tolist()) if k]
                print(f"[concept filter] kept {len(concepts)}, removed {len(removed_concepts)} "
                      f"zero-count concepts (VLG-CBM dynamic filter)")
                with open(os.path.join(save_dir, "removed_concepts.txt"), "w") as f:
                    f.write("\n".join(removed_concepts))
                # Mask in-place. We delete the old reference and gc.collect()
                # before letting the indexed slice settle, so the parent tensor
                # is freed at the earliest possible moment (otherwise both old
                # and new tensors live concurrently → 2× annotation cache peak).
                log_mem("concept filter: before train mask")
                _new_train_ann = base_cbl_dataset._annotation_cache[:, keep_mask]
                base_cbl_dataset._annotation_cache = _new_train_ann
                del _new_train_ann
                gc.collect()
                log_mem("concept filter: after train mask")

                _new_val_ann = val_cbl_dataset._annotation_cache[:, keep_mask]
                val_cbl_dataset._annotation_cache = _new_val_ann
                del _new_val_ann
                gc.collect()
                log_mem("concept filter: after val mask")

                base_cbl_dataset.concepts = concepts
                val_cbl_dataset.concepts = concepts
                base_cbl_dataset.concept_set = set(concepts)
                val_cbl_dataset.concept_set = set(concepts)
                num_concepts = len(concepts)
                args.num_concepts = num_concepts
            else:
                print(f"[concept filter] all {len(concepts)} concepts have DINO presence; nothing removed")
            _t = _step("concept filter (dino zero-count)", _t)

        with open(os.path.join(save_dir, "concepts.txt"), "w") as f:
            f.write("\n".join(concepts))

        cbl = ConceptLayer(
            backbone.output_dim, num_concepts,
            num_hidden=getattr(args, "cbl_hidden_layers", 0),
            bias=True, device=str(device)
        )
        global_model = FedVLGCBM(backbone, cbl, normalization=None, final_layer=None)
        global_model.to(device)
        _t = _step("CBL + global model init", _t)

        # When the feature cache will be populated below, Phase 1's DataLoaders
        # serve from in-RAM tensors — no I/O or preprocessing happens per __getitem__.
        # In that regime, workers add zero throughput value but each fork inherits
        # the parent's ~11GB+ of cached tensors. On large datasets (places365,
        # imagenet) this triggers OOM under CoW edge cases (pin_memory pool growth,
        # tensor refcount touches dirtying many pages, etc.). Use num_workers=0 +
        # pin_memory=False to keep the loaders single-process and avoid duplication.
        # If feature cache won't be active (cbl_finetune or no DINO), keep workers.
        _will_use_feature_cache = (
            not getattr(args, "cbl_finetune", False)
            and annotation_dir and os.path.isdir(annotation_dir)
        )
        _p1_workers = 0 if _will_use_feature_cache else args.num_workers
        _p1_pin = not _will_use_feature_cache

        val_cbl_loader = DataLoader(
            val_cbl_dataset,
            batch_size=getattr(args, "cbl_batch_size", 32),
            num_workers=_p1_workers,
            shuffle=False
        )
        # ── Label-flip hook (case study) ─────────────────────────────────────
        _flip_client = getattr(args, "label_flip_client", -1)
        _flip_map_int = {}
        if _flip_client >= 0:
            _flip_map_raw = getattr(args, "label_flip_map", None)
            if _flip_map_raw:
                import json as _json
                _name_map = _json.loads(_flip_map_raw) if isinstance(_flip_map_raw, str) else _flip_map_raw
                _name_to_idx = {c: i for i, c in enumerate(classes)}
                _flip_map_int = {_name_to_idx[src]: _name_to_idx[tgt]
                                 for src, tgt in _name_map.items()
                                 if src in _name_to_idx and tgt in _name_to_idx}
                print(f"[poison] Client {_flip_client} label flip map (indices): {_flip_map_int}")

        client_train_loaders = []
        client_data_sizes = []
        for i in range(args.num_clients):
            sub = Subset(base_cbl_dataset, client_indices[i])
            if i == _flip_client and _flip_map_int:
                sub = LabelFlipSubset(sub, _flip_map_int)
                print(f"[poison] Wrapped client {i} dataset with LabelFlipSubset")
            client_train_loaders.append(DataLoader(
                sub, batch_size=getattr(args, "cbl_batch_size", 32),
                shuffle=True, num_workers=_p1_workers, pin_memory=_p1_pin
            ))
            client_data_sizes.append(len(sub))
        total_samples = sum(client_data_sizes)
        client_weights = [n / total_samples for n in client_data_sizes]
        _t = _step("DataLoader creation", _t)

        _cbl_dir = getattr(args, "load_cbl_dir", None)
        if _cbl_dir is not None:
            # Load a Phase-1-only checkpoint; skip CBL training entirely.
            _bb_pt = os.path.join(_cbl_dir, "backbone.pt")
            if os.path.exists(_bb_pt):
                global_model.backbone.backbone.load_state_dict(
                    torch.load(_bb_pt, map_location=str(device))
                )
            else:
                print(f"  [INFO] backbone.pt not found in {_cbl_dir} — using pretrained weights (cbl_finetune=False checkpoint)")
            global_model.cbl.load_state_dict(
                torch.load(os.path.join(_cbl_dir, "cbl.pt"), map_location=str(device))
            )
            print(f"Loaded CBL weights from {_cbl_dir}, skipping Phase 1")

            # Populate feature cache so the Phase 2 loops below (3 backbone passes)
            # skip the backbone. If the Phase 1 job wrote a cache to the same
            # --annotation_cache_dir, this is an on-disk hit and costs ~1s; otherwise
            # extraction runs once here. Gated on not cbl_finetune because a
            # finetuned backbone would invalidate any prior cache silently.
            if (not getattr(args, "cbl_finetune", False)
                    and annotation_dir and os.path.isdir(annotation_dir)):
                # Scale extraction batch by cbl_batch_size so small-pod datasets
                # (cifar100 @ 6Gi) don't OOM. Large datasets (places365 @ 32Gi)
                # get up to 1024. Formula: cbl_batch_size * 8, capped at 1024.
                _extract_bs = min(1024, getattr(args, "cbl_batch_size", 32) * 8)
                _extract_workers = max(args.num_workers, 4)
                base_cbl_dataset.populate_feature_cache(
                    backbone=backbone, device=str(device),
                    backbone_name=args.backbone,
                    batch_size=_extract_bs, num_workers=_extract_workers, prefetch_factor=4,
                )
                val_cbl_dataset.populate_feature_cache(
                    backbone=backbone, device=str(device),
                    backbone_name=args.backbone,
                    batch_size=_extract_bs, num_workers=_extract_workers, prefetch_factor=4,
                )
                _t = _step("backbone feature pre-extraction (Phase 2 only)", _t)

            projection_metrics = {}
        else:
            num_train = len(base_cbl_dataset)
            cbl_loss_type = getattr(args, "cbl_loss_type", "bce")

            # cos_cubed is parameterless — skip expensive concept count computation.
            if cbl_loss_type in ("cos_cubed",):
                concept_counts = [0] * num_concepts
                _t = _step("concept count (skipped for cos_cubed)", _t)
            else:
                use_dino = annotation_dir and os.path.isdir(annotation_dir)
                if use_dino:
                    # Compute per-concept positive counts directly from the preloaded
                    # annotation cache — avoids creating a DataLoader that would load
                    # and preprocess every image just to discard it.
                    print("Computing per-concept positive counts from DINO annotations...")
                    log_mem("concept count: before chunked sum")
                    _ann = base_cbl_dataset._annotation_cache
                    _N, _C = _ann.shape
                    _cc = torch.zeros(_C, dtype=torch.long)
                    _chunk = 65536
                    for _start in range(0, _N, _chunk):
                        _cc += _ann[_start:_start + _chunk].sum(dim=0).long()
                    concept_counts = _cc.tolist()
                    log_mem("concept count: after chunked sum")
                    print(f"  Concept pos counts: min={min(concept_counts):.0f} median={sorted(concept_counts)[len(concept_counts)//2]:.0f} max={max(concept_counts):.0f}")
                else:
                    per_class_concepts = num_concepts // num_classes
                    targets = data_utils.get_dataset_targets(train_dataset)
                    class_counts = [0] * num_classes
                    for label in targets:
                        class_counts[int(label)] += 1
                    concept_counts = []
                    for c in range(num_classes):
                        concept_counts.extend([class_counts[c]] * per_class_concepts)
                    orphan_count = num_train // num_classes
                    while len(concept_counts) < num_concepts:
                        concept_counts.append(orphan_count)
                _t = _step("concept count computation", _t)

            # Pre-extract backbone features once when frozen. Each client/epoch/round
            # below would otherwise rerun the backbone on the same images, which for
            # places365 is hundreds of millions of wasted forward passes. Only safe
            # when the backbone is not being finetuned.
            if (not getattr(args, "cbl_finetune", False)
                    and annotation_dir and os.path.isdir(annotation_dir)):
                _extract_bs = min(1024, getattr(args, "cbl_batch_size", 32) * 8)
                _extract_workers = max(args.num_workers, 4)
                base_cbl_dataset.populate_feature_cache(
                    backbone=backbone, device=str(device),
                    backbone_name=args.backbone,
                    batch_size=_extract_bs, num_workers=_extract_workers, prefetch_factor=4,
                )
                val_cbl_dataset.populate_feature_cache(
                    backbone=backbone, device=str(device),
                    backbone_name=args.backbone,
                    batch_size=_extract_bs, num_workers=_extract_workers, prefetch_factor=4,
                )
                _t = _step("backbone feature pre-extraction", _t)

            loss_fn = get_loss_vlg(
                cbl_loss_type, num_concepts, num_train, concept_counts,
                getattr(args, "cbl_pos_weight", 0.2), getattr(args, "cbl_auto_weight", False),
                tp=getattr(args, "cbl_twoway_tp", 4.0), device=str(device)
            )
            _t = _step("loss function init", _t)

            client_models = [copy.deepcopy(global_model) for _ in range(args.num_clients)]
            for m in client_models:
                m.to(device)
            log_mem("after Phase 1 client model init")
            _t = _step(f"deepcopy {args.num_clients} client models to {device}", _t)

            print("\n=== Phase 1: Federated CBL training ===")
            projection_metrics = {"rounds": [], "client_losses": [], "avg_client_loss": [], "best_val_loss": []}
            best_val_loss = float("inf")
            best_cbl_state = None
            cbl_finetune = getattr(args, "cbl_finetune", False)
            # Only exclude backbone from aggregation when it is frozen (not being finetuned).
            # If cbl_finetune=True the backbone is trained on each client, so its updates must be aggregated.
            cbl_exclude_prefixes = ["backbone."] if not cbl_finetune else None
            lambda_ortho = getattr(args, "ortho_lambda", 0.0)
            for round_num in range(args.num_rounds):
                round_losses = []
                round_concept_losses = []
                round_ortho_losses = []
                for i in range(args.num_clients):
                    client_models[i].load_state_dict(global_model.state_dict())
                    total_loss, concept_loss, ortho_loss = train_cbl(
                        client_models[i].backbone, client_models[i].cbl,
                        client_train_loaders[i],
                        epochs=getattr(args, "cbl_epochs", args.local_epochs),
                        loss_fn=loss_fn, lr=getattr(args, "cbl_lr", args.lr),
                        weight_decay=args.weight_decay, device=str(device),
                        finetune=cbl_finetune,
                        optimizer_name=getattr(args, "cbl_optimizer", "adam"),
                        backbone_lr=getattr(args, "cbl_bb_lr_rate", 1.0) * getattr(args, "cbl_lr", args.lr),
                        val_loader=val_cbl_loader,
                        lambda_ortho=lambda_ortho,
                    )
                    round_losses.append(total_loss)
                    round_concept_losses.append(concept_loss)
                    round_ortho_losses.append(ortho_loss)
                    if lambda_ortho > 0.0:
                        print(f"  Round {round_num + 1} Client {i}: concept_loss={concept_loss:.4f} ortho_loss={ortho_loss:.4f} total_loss={total_loss:.4f}", flush=True)
                global_state = federated_averaging(client_models, client_weights, exclude_prefixes=cbl_exclude_prefixes)
                global_model.load_state_dict(global_state)
                avg_train_loss = sum(round_losses) / len(round_losses)
                avg_concept_loss = sum(round_concept_losses) / len(round_concept_losses)
                avg_ortho_loss = sum(round_ortho_losses) / len(round_ortho_losses)
                # Server-side validation on aggregated model
                server_val_loss = validate_cbl(global_model.backbone, global_model.cbl, val_cbl_loader, loss_fn, str(device))
                projection_metrics["rounds"].append(round_num + 1)
                projection_metrics["client_losses"].append(round_losses)
                projection_metrics["avg_client_loss"].append(avg_train_loss)
                if server_val_loss < best_val_loss:
                    best_val_loss = server_val_loss
                    best_cbl_state = {k: v.clone() for k, v in global_model.state_dict().items()}
                projection_metrics["best_val_loss"].append(best_val_loss)
                if lambda_ortho > 0.0:
                    print(f"Round {round_num + 1} avg client: concept_loss={avg_concept_loss:.4f} ortho_loss={avg_ortho_loss:.4f} total_loss={avg_train_loss:.4f}, server val loss: {server_val_loss:.4f}")
                else:
                    print(f"Round {round_num + 1} avg client train loss: {avg_train_loss:.4f}, server val loss: {server_val_loss:.4f}")
                update_log(_log_path, {"status": "in_progress", "phase": "cbl_training",
                                        "round": round_num + 1, "total_rounds": args.num_rounds,
                                        "avg_train_loss": avg_train_loss, "server_val_loss": server_val_loss,
                                        "best_val_loss": best_val_loss})

            # Restore the best CBL checkpoint (by server val loss) before Phase 2
            if best_cbl_state is not None:
                global_model.load_state_dict(best_cbl_state)
                print(f"Restored best CBL checkpoint (val loss {best_val_loss:.4f})")

            # --phase1_only: save CBL and exit so Phase 2+3 can run in a separate job
            if getattr(args, "phase1_only", False):
                print(f"\nPhase 1 only mode — saving model to {save_dir}")
                global_model.backbone.save_model(save_dir)
                global_model.cbl.save_model(save_dir)
                with open(os.path.join(save_dir, "metrics.txt"), "w") as _f:
                    json.dump({"phase1_only": True, "best_val_loss": float(best_val_loss),
                               "num_rounds": args.num_rounds, "num_clients": args.num_clients}, _f, indent=2)
                update_log(_log_path, {"status": "completed", "phase": "cbl_training",
                                        "best_val_loss": float(best_val_loss),
                                        "num_rounds": args.num_rounds, "num_clients": args.num_clients,
                                        "completed_at": datetime.datetime.now().isoformat()})
                print("Phase 1 complete.")
                return

            # Free Phase 1 objects: N client model copies are the dominant GPU consumer.
            # best_cbl_state holds an extra full state-dict copy — release it too.
            log_mem("before Phase 1 cleanup")
            del client_models, best_cbl_state
            gc.collect()
            torch.cuda.empty_cache()
            log_mem("after Phase 1 cleanup")
            print("Freed client models before Phase 2")

        print("\n=== Phase 2: Federated normalization and concept feature extraction ===")
        saga_bs = getattr(args, "saga_batch_size", 512)

        # Phase 2 is GPU-bound (backbone inference), not I/O-bound.
        # Rebuild loaders with num_workers=0 so no worker processes are forked.
        # Worker forks each inherit the parent RSS (~1.5GB), causing CPU OOM on 8Gi nodes.
        p2_bs = getattr(args, "cbl_batch_size", 32)
        phase2_loaders = [
            DataLoader(client_train_loaders[i].dataset, batch_size=p2_bs, shuffle=False, num_workers=0)
            for i in range(args.num_clients)
        ]
        phase2_val_loader = DataLoader(val_cbl_loader.dataset, batch_size=p2_bs, shuffle=False, num_workers=0)
        log_mem("Phase 2 start (num_workers=0 loaders ready)")

        # Step 2a: Each client computes local concept feature statistics (mean, var, count)
        # Server aggregates via parallel statistics formula — no raw data leaves clients
        client_sums = []
        client_sq_sums = []
        client_counts = []
        global_model.eval()
        with torch.no_grad():
            for i in range(args.num_clients):
                local_sum = torch.zeros(num_concepts)
                local_sq_sum = torch.zeros(num_concepts)
                local_n = 0
                for features, _, _ in phase2_loaders[i]:
                    features = features.to(device)
                    # 4D → raw images, run backbone. 2D → feature cache active, skip it.
                    emb = global_model.backbone(features) if features.dim() == 4 else features
                    logits = global_model.cbl(emb).cpu()
                    local_sum += logits.sum(dim=0)
                    local_sq_sum += (logits ** 2).sum(dim=0)
                    local_n += logits.size(0)
                client_sums.append(local_sum)
                client_sq_sums.append(local_sq_sum)
                client_counts.append(local_n)

        # Server aggregates statistics
        total_n = sum(client_counts)
        global_sum = sum(client_sums)
        global_sq_sum = sum(client_sq_sums)
        global_mean = global_sum / total_n
        global_var = (global_sq_sum / total_n) - (global_mean ** 2)
        global_std = global_var.clamp(min=1e-8).sqrt()

        norm_layer = NormalizationLayer(global_mean, global_std, device=str(device))
        global_model.normalization = norm_layer

        # Step 2b: Extract normalized concept features for centralized methods and NEC eval
        # (hybrid_saga/fedavg need pooled features; fedavg_thresh extracts per-client features in Phase 3)
        all_train_feats, all_train_labels = [], []
        with torch.no_grad():
            for i in range(args.num_clients):
                for features, _, labels in phase2_loaders[i]:
                    features = features.to(device)
                    emb = global_model.backbone(features) if features.dim() == 4 else features
                    logits = norm_layer(global_model.cbl(emb)).cpu()
                    all_train_feats.append(logits)
                    all_train_labels.append(labels)
        all_train_feats = torch.cat(all_train_feats, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        log_mem(f"after Phase 2 train feat extraction ({all_train_feats.shape[0]} samples, {all_train_feats.element_size() * all_train_feats.nelement() / 1024**3:.2f}GB tensor)")

        val_feats, val_labels_all = [], []
        with torch.no_grad():
            for features, _, labels in phase2_val_loader:
                features = features.to(device)
                emb = global_model.backbone(features) if features.dim() == 4 else features
                logits = norm_layer(global_model.cbl(emb)).cpu()
                val_feats.append(logits)
                val_labels_all.append(labels)
        val_feats = torch.cat(val_feats, dim=0)
        val_labels_all = torch.cat(val_labels_all, dim=0)

        # Save concept features for reproducibility
        os.makedirs(save_dir, exist_ok=True)
        torch.save(all_train_feats, os.path.join(save_dir, "train_concept_features.pt"))
        torch.save(all_train_labels, os.path.join(save_dir, "train_concept_labels.pt"))
        torch.save(val_feats, os.path.join(save_dir, "val_concept_features.pt"))
        torch.save(val_labels_all, os.path.join(save_dir, "val_concept_labels.pt"))
        norm_layer.save_model(save_dir)

        # Local-only diagnostic (case study): fit an independent head per client on
        # its own normalized features. Gated on flag → zero cost for normal runs.
        _lo_dir = getattr(args, "local_only_diag_dir", None)
        if _lo_dir:
            from case_studies.poison_label_flip.local_only import fit_local_only_heads
            fit_local_only_heads(
                all_train_feats, all_train_labels, client_data_sizes,
                num_concepts, num_classes, str(device), _lo_dir,
                epochs=getattr(args, "local_only_epochs", 50),
            )

        train_concept_loader = DataLoader(
            IndexedTensorDataset(all_train_feats, all_train_labels),
            batch_size=saga_bs, shuffle=True
        )
        val_concept_loader = DataLoader(
            TensorDataset(val_feats, val_labels_all),
            batch_size=saga_bs, shuffle=False
        )

        # Precompute federated val split indices before releasing val_dataset.
        # fedavg_thresh Phase 3 needs these to slice val_feats per client.
        _val_indices_thresh = split_dataset_for_federated(
            val_dataset, args.num_clients, iid=args.iid, alpha=args.alpha, seed=args.seed
        )

        # Free raw image datasets/loaders — no longer needed after feature extraction.
        log_mem("before Phase 2 cleanup")
        del client_train_loaders, base_cbl_dataset, val_cbl_dataset, val_cbl_loader
        del full_train_dataset, train_dataset, val_dataset
        global_model.backbone.cpu()          # move backbone off GPU — only needed again for test_model at the end
        gc.collect()
        torch.cuda.empty_cache()
        log_mem("after Phase 2 cleanup (backbone on CPU)")
        print("Freed backbone / raw-image objects before Phase 3")

    test_loader = get_concept_dataloader(
        args.dataset, "test", concepts, preprocess=preprocess,
        use_allones=not (getattr(args, "annotation_dir", None) and os.path.isdir(getattr(args, "annotation_dir", ""))),
        annotation_dir=getattr(args, "annotation_dir", None),
        confidence_threshold=getattr(args, "dino_confidence_threshold", 0.10),
        cache_dir=getattr(args, "annotation_cache_dir", None),
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    vlg_final_method = args.final_layer_method
    log_mem("start of Phase 3")
    if vlg_final_method in ("hybrid_saga", "fedavg"):
        print(f"\n=== Phase 3: Final layer (sparse GLM-SAGA or dense) ===")
        final_layer = FinalLayer(num_concepts, num_classes, device=str(device))
        if getattr(args, "dense", False) or vlg_final_method == "fedavg":
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

        # Record final layer phase metrics for centralized methods
        nnz_central = int((final_layer.weight.data.abs() > 1e-5).sum().item())
        total_central = int(final_layer.weight.data.numel())
        vlg_central_metrics = {
            "method": vlg_final_method,
            "sparsity_nnz": nnz_central,
            "sparsity_total": total_central,
            "sparsity_pct_nonzero": nnz_central / total_central,
        }
        if vlg_final_method == "hybrid_saga":
            vlg_central_metrics["saga_lam"] = getattr(args, "saga_lam", 0.0007)
            vlg_central_metrics["saga_n_iters"] = getattr(args, "saga_n_iters", 2000)
        else:
            vlg_central_metrics["dense_lr"] = getattr(args, "dense_lr", 0.001)
        # Extract val accuracy from SAGA output if available
        if out.get("path") and out["path"][0].get("metrics"):
            vlg_central_metrics["val_metrics"] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in out["path"][0]["metrics"].items()
            }
        print(f"Final layer sparsity: {nnz_central}/{total_central} non-zero ({nnz_central/total_central:.4f})")

    elif vlg_final_method == "fedavg_thresh":
        print("\n=== Phase 3: Federated Final Layer with Thresholding (VLG) ===")
        # Reuse the normalized concept features already extracted in Phase 2.
        # all_train_feats was built by iterating clients in order 0..N, so slicing
        # by client_data_sizes recovers each client's block without a second backbone pass.
        client_concept_loaders = []
        offset = 0
        for i in range(args.num_clients):
            n = client_data_sizes[i]
            c_feats = all_train_feats[offset:offset + n]
            c_labels = all_train_labels[offset:offset + n]
            offset += n
            client_concept_loaders.append(DataLoader(
                TensorDataset(c_feats, c_labels),
                batch_size=saga_bs, shuffle=True
            ))

        # Per-client val concept features.
        # Normal mode: use same federated split indices as training to preserve label distribution.
        # Pretrained mode: uniform split (val_dataset not available when loading pre-extracted features).
        client_val_concept_loaders = []
        client_val_sizes = []
        if _pretrained_mode:
            n_val_total = val_feats.shape[0]
            base_val = n_val_total // args.num_clients
            rem_val = n_val_total % args.num_clients
            offset_val = 0
            for i in range(args.num_clients):
                n_v = base_val + (1 if i < rem_val else 0)
                v_feats_i = val_feats[offset_val:offset_val + n_v]
                v_labels_i = val_labels_all[offset_val:offset_val + n_v]
                client_val_concept_loaders.append(DataLoader(
                    TensorDataset(v_feats_i, v_labels_i),
                    batch_size=saga_bs, shuffle=False
                ))
                client_val_sizes.append(n_v)
                offset_val += n_v
        else:
            # _val_indices_thresh was precomputed before val_dataset was deleted
            for i in range(args.num_clients):
                idx = torch.tensor(_val_indices_thresh[i], dtype=torch.long)
                v_feats_i = val_feats[idx]
                v_labels_i = val_labels_all[idx]
                client_val_concept_loaders.append(DataLoader(
                    TensorDataset(v_feats_i, v_labels_i),
                    batch_size=saga_bs, shuffle=False
                ))
                client_val_sizes.append(len(idx))
        total_val = sum(client_val_sizes)
        client_val_weights = [n / total_val for n in client_val_sizes]

        # Initialize global and client final layers
        final_layer = FinalLayer(num_concepts, num_classes, device=str(device))
        client_final_layers = [FinalLayer(num_concepts, num_classes, device=str(device))
                               for _ in range(args.num_clients)]
        ce_loss = nn.CrossEntropyLoss()

        final_rounds = getattr(args, "final_rounds", 5)
        final_epochs = getattr(args, "final_epochs", 3)
        final_lr = getattr(args, "final_lr", 1e-3)

        thresh_metrics = {
            "rounds": [], "client_losses": [], "avg_client_loss": [],
            "val_accuracy": [], "best_val_accuracy": [], "threshold_lam": [],
            "mask_alive": [], "concepts_alive": [],
        }
        best_val_acc = 0.0
        best_fl_state = None
        # FedMask: binary mask (1 = alive, 0 = dead). Starts all-ones; monotonically shrinks.
        _weight_mask = {key: torch.ones_like(val)
                        for key, val in final_layer.state_dict().items() if "weight" in key}
        for round_num in range(final_rounds):
            print(f"\n=== VLG Final Layer Round {round_num + 1}/{final_rounds} ===")
            round_losses = []
            for i in range(args.num_clients):
                # Sync client with global
                client_final_layers[i].load_state_dict(final_layer.state_dict())
                client_final_layers[i].train()
                opt = torch.optim.Adam(client_final_layers[i].parameters(), lr=final_lr)
                epoch_loss = 0.0
                n_batches = 0
                for epoch in range(final_epochs):
                    for feats, labels in client_concept_loaders[i]:
                        feats, labels = feats.to(device), labels.to(device)
                        loss = ce_loss(client_final_layers[i](feats), labels)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        # FedMask: re-zero dead weights after each step
                        with torch.no_grad():
                            client_final_layers[i].weight.data *= _weight_mask["weight"]
                        epoch_loss += loss.item()
                        n_batches += 1
                round_losses.append(epoch_loss / max(n_batches, 1))
                print(f"  Client {i}: Loss = {round_losses[-1]:.4f}")

            # Aggregate final layers
            global_fl_state = {}
            for key in final_layer.state_dict().keys():
                param = client_final_layers[0].state_dict()[key]
                if param.dtype.is_floating_point:
                    global_fl_state[key] = torch.zeros_like(param)
                    for i in range(args.num_clients):
                        global_fl_state[key] += client_weights[i] * client_final_layers[i].state_dict()[key]
                else:
                    global_fl_state[key] = param.clone()

            # Server-side group thresholding (zeros entire concept columns by L2 norm)
            progress = round_num / max(final_rounds - 1, 1)
            lam = args.thresh_lam_start + progress * (args.thresh_lam_end - args.thresh_lam_start)
            for key in global_fl_state:
                if "weight" in key:
                    global_fl_state[key] = group_threshold(global_fl_state[key], lam)
                    # FedMask: update mask — once a concept column is dead, it stays dead
                    _weight_mask[key] *= (global_fl_state[key].abs() > 1e-8).float()
                    global_fl_state[key] *= _weight_mask[key]
            _mask_total = sum(m.numel() for m in _weight_mask.values())
            _mask_alive = sum((m > 0).sum().item() for m in _weight_mask.values())
            # Count alive concepts (columns with any non-zero weight)
            _concepts_alive = sum((_weight_mask[k].abs().sum(dim=0) > 0).sum().item()
                                  for k in _weight_mask)
            _concepts_total = sum(_weight_mask[k].shape[1] for k in _weight_mask)
            # Log column norm stats for tuning threshold
            for key in global_fl_state:
                if "weight" in key:
                    col_norms = global_fl_state[key].norm(p=2, dim=0)
                    print(f"  group_threshold lam={lam:.6f}  col_norms: min={col_norms.min():.4f} median={col_norms.median():.4f} max={col_norms.max():.4f}")
            print(f"  Weights: {_mask_alive}/{_mask_total} alive  Concepts: {_concepts_alive}/{_concepts_total}")

            final_layer.load_state_dict(global_fl_state)

            # Federated evaluation: each client evaluates on its own val split
            final_layer.eval()
            val_acc = 0.0
            with torch.no_grad():
                for i in range(args.num_clients):
                    client_correct, client_total = 0, 0
                    for feats, labels in client_val_concept_loaders[i]:
                        feats, labels = feats.to(device), labels.to(device)
                        preds = final_layer(feats).argmax(dim=1)
                        client_correct += (preds == labels).sum().item()
                        client_total += labels.size(0)
                    val_acc += client_val_weights[i] * (client_correct / max(client_total, 1))
            print(f"  Val Accuracy: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_fl_state = {k: v.clone() for k, v in final_layer.state_dict().items()}

            thresh_metrics["rounds"].append(round_num + 1)
            thresh_metrics["client_losses"].append(round_losses)
            thresh_metrics["avg_client_loss"].append(sum(round_losses) / len(round_losses))
            thresh_metrics["val_accuracy"].append(float(val_acc))
            thresh_metrics["best_val_accuracy"].append(float(best_val_acc))
            thresh_metrics["threshold_lam"].append(float(lam))
            thresh_metrics["mask_alive"].append(int(_mask_alive))
            thresh_metrics["concepts_alive"].append(int(_concepts_alive))
            update_log(_log_path, {"status": "in_progress", "phase": "final_layer_fedavg_thresh",
                                    "round": round_num + 1, "total_rounds": getattr(args, "final_rounds", 5),
                                    "val_accuracy": float(val_acc), "best_val_accuracy": float(best_val_acc),
                                    "threshold_lam": float(lam), "mask_alive": int(_mask_alive),
                                    "concepts_alive": int(_concepts_alive)})

        if best_fl_state is not None:
            # Apply the accumulated dead-concept mask to the best-accuracy checkpoint.
            # Concepts zeroed by server pruning in later rounds stay zeroed even if
            # the best val accuracy was achieved before those rounds.
            for key in _weight_mask:
                if key in best_fl_state:
                    best_fl_state[key] = best_fl_state[key] * _weight_mask[key]
            final_layer.load_state_dict(best_fl_state)
        global_model.final_layer = final_layer

        # Report sparsity
        nnz = (final_layer.weight.data.abs() > 1e-5).sum().item()
        total_w = final_layer.weight.data.numel()
        print(f"Final layer sparsity: {nnz}/{total_w} non-zero ({nnz/total_w:.4f})")

    elif vlg_final_method == "fedavg_l1":
        print("\n=== Phase 3: FedAvg + Element-wise L1 Soft-Threshold (naive baseline) ===")
        # Each client trains the final layer with CE only, server FedAvgs the
        # weights, then applies element-wise L1 soft-threshold post-aggregation.
        # No proximal step in the local objective, no FedMask — the server's
        # soft_threshold is the only sparsity pressure.
        client_concept_loaders = []
        offset = 0
        for i in range(args.num_clients):
            n = client_data_sizes[i]
            c_feats = all_train_feats[offset:offset + n]
            c_labels = all_train_labels[offset:offset + n]
            offset += n
            client_concept_loaders.append(DataLoader(
                TensorDataset(c_feats, c_labels),
                batch_size=saga_bs, shuffle=True
            ))

        # Per-client val concept features (mirrors fedavg_thresh)
        client_val_concept_loaders = []
        client_val_sizes = []
        if _pretrained_mode:
            n_val_total = val_feats.shape[0]
            base_val = n_val_total // args.num_clients
            rem_val = n_val_total % args.num_clients
            offset_val = 0
            for i in range(args.num_clients):
                n_v = base_val + (1 if i < rem_val else 0)
                v_feats_i = val_feats[offset_val:offset_val + n_v]
                v_labels_i = val_labels_all[offset_val:offset_val + n_v]
                client_val_concept_loaders.append(DataLoader(
                    TensorDataset(v_feats_i, v_labels_i),
                    batch_size=saga_bs, shuffle=False
                ))
                client_val_sizes.append(n_v)
                offset_val += n_v
        else:
            for i in range(args.num_clients):
                idx = torch.tensor(_val_indices_thresh[i], dtype=torch.long)
                v_feats_i = val_feats[idx]
                v_labels_i = val_labels_all[idx]
                client_val_concept_loaders.append(DataLoader(
                    TensorDataset(v_feats_i, v_labels_i),
                    batch_size=saga_bs, shuffle=False
                ))
                client_val_sizes.append(len(idx))
        total_val = sum(client_val_sizes)
        client_val_weights = [n / total_val for n in client_val_sizes]

        final_layer = FinalLayer(num_concepts, num_classes, device=str(device))
        client_final_layers = [FinalLayer(num_concepts, num_classes, device=str(device))
                               for _ in range(args.num_clients)]
        ce_loss = nn.CrossEntropyLoss()

        final_rounds = getattr(args, "final_rounds", 5)
        final_epochs = getattr(args, "final_epochs", 3)
        final_lr = getattr(args, "final_lr", 1e-3)
        l1_lam = float(args.fedavg_l1_lam)
        print(f"[fedavg_l1] post-aggregation L1 threshold λ={l1_lam}")

        l1_metrics = {
            "rounds": [], "client_losses": [], "avg_client_loss": [],
            "val_accuracy": [], "best_val_accuracy": [],
            "nnz_weights": [], "concepts_alive": [],
            "threshold_lam": l1_lam,
        }
        best_val_acc = 0.0
        best_fl_state = None

        for round_num in range(final_rounds):
            print(f"\n=== FedAvg+L1 Round {round_num + 1}/{final_rounds} ===")
            round_losses = []
            for i in range(args.num_clients):
                client_final_layers[i].load_state_dict(final_layer.state_dict())
                client_final_layers[i].train()
                opt = torch.optim.Adam(client_final_layers[i].parameters(), lr=final_lr)
                epoch_loss = 0.0
                n_batches = 0
                for epoch in range(final_epochs):
                    for feats, labels in client_concept_loaders[i]:
                        feats, labels = feats.to(device), labels.to(device)
                        loss = ce_loss(client_final_layers[i](feats), labels)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        epoch_loss += loss.item()
                        n_batches += 1
                round_losses.append(epoch_loss / max(n_batches, 1))
                print(f"  Client {i}: Loss = {round_losses[-1]:.4f}")

            # Server: weighted FedAvg of client final layers
            global_fl_state = {}
            for key in final_layer.state_dict().keys():
                param = client_final_layers[0].state_dict()[key]
                if param.dtype.is_floating_point:
                    global_fl_state[key] = torch.zeros_like(param)
                    for i in range(args.num_clients):
                        global_fl_state[key] += client_weights[i] * client_final_layers[i].state_dict()[key]
                else:
                    global_fl_state[key] = param.clone()

            # Element-wise L1 soft threshold on the averaged weight matrix.
            # Bias is left unregularized (matches feddualavg).
            for key in global_fl_state:
                if "weight" in key:
                    global_fl_state[key] = soft_threshold(global_fl_state[key], l1_lam)

            final_layer.load_state_dict(global_fl_state)

            # Federated val evaluation
            final_layer.eval()
            val_acc = 0.0
            with torch.no_grad():
                for i in range(args.num_clients):
                    client_correct, client_total = 0, 0
                    for feats, labels in client_val_concept_loaders[i]:
                        feats, labels = feats.to(device), labels.to(device)
                        preds = final_layer(feats).argmax(dim=1)
                        client_correct += (preds == labels).sum().item()
                        client_total += labels.size(0)
                    val_acc += client_val_weights[i] * (client_correct / max(client_total, 1))
            print(f"  Val Accuracy: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_fl_state = {k: v.clone() for k, v in final_layer.state_dict().items()}

            nnz_round = int((final_layer.weight.data.abs() > 1e-5).sum().item())
            concepts_alive_round = int(
                (final_layer.weight.data.abs().sum(dim=0) > 1e-5).sum().item()
            )

            l1_metrics["rounds"].append(round_num + 1)
            l1_metrics["client_losses"].append(round_losses)
            l1_metrics["avg_client_loss"].append(sum(round_losses) / len(round_losses))
            l1_metrics["val_accuracy"].append(float(val_acc))
            l1_metrics["best_val_accuracy"].append(float(best_val_acc))
            l1_metrics["nnz_weights"].append(nnz_round)
            l1_metrics["concepts_alive"].append(concepts_alive_round)
            update_log(_log_path, {"status": "in_progress", "phase": "final_layer_fedavg_l1",
                                    "round": round_num + 1, "total_rounds": final_rounds,
                                    "val_accuracy": float(val_acc), "best_val_accuracy": float(best_val_acc),
                                    "threshold_lam": l1_lam, "nnz_weights": nnz_round,
                                    "concepts_alive": concepts_alive_round})

        if best_fl_state is not None:
            final_layer.load_state_dict(best_fl_state)
        global_model.final_layer = final_layer

        nnz = (final_layer.weight.data.abs() > 1e-5).sum().item()
        total_w = final_layer.weight.data.numel()
        print(f"Final layer sparsity: {nnz}/{total_w} non-zero ({nnz/total_w:.4f})")

    elif vlg_final_method == "feddualavg":
        print("\n=== Phase 3: Federated Dual Averaging with Group Lasso (VLG) ===")
        # Reuse the normalized concept features already extracted in Phase 2.
        client_concept_loaders = []
        offset = 0
        for i in range(args.num_clients):
            n = client_data_sizes[i]
            c_feats = all_train_feats[offset:offset + n]
            c_labels = all_train_labels[offset:offset + n]
            offset += n
            client_concept_loaders.append(DataLoader(
                TensorDataset(c_feats, c_labels),
                batch_size=saga_bs, shuffle=True
            ))

        # Per-client val concept features
        client_val_concept_loaders = []
        client_val_sizes = []
        if _pretrained_mode:
            n_val_total = val_feats.shape[0]
            base_val = n_val_total // args.num_clients
            rem_val = n_val_total % args.num_clients
            offset_val = 0
            for i in range(args.num_clients):
                n_v = base_val + (1 if i < rem_val else 0)
                v_feats_i = val_feats[offset_val:offset_val + n_v]
                v_labels_i = val_labels_all[offset_val:offset_val + n_v]
                client_val_concept_loaders.append(DataLoader(
                    TensorDataset(v_feats_i, v_labels_i),
                    batch_size=saga_bs, shuffle=False
                ))
                client_val_sizes.append(n_v)
                offset_val += n_v
        else:
            for i in range(args.num_clients):
                idx = torch.tensor(_val_indices_thresh[i], dtype=torch.long)
                v_feats_i = val_feats[idx]
                v_labels_i = val_labels_all[idx]
                client_val_concept_loaders.append(DataLoader(
                    TensorDataset(v_feats_i, v_labels_i),
                    batch_size=saga_bs, shuffle=False
                ))
                client_val_sizes.append(len(idx))
        total_val = sum(client_val_sizes)
        client_val_weights = [n / total_val for n in client_val_sizes]

        # Initialize final layer and dual state
        final_layer = FinalLayer(num_concepts, num_classes, device=str(device))
        ce_loss = nn.CrossEntropyLoss()

        final_rounds = getattr(args, "final_rounds", 5)
        final_epochs = getattr(args, "final_epochs", 3)
        eta_s = args.dual_eta_s
        eta_c = args.dual_eta_c
        dual_lam = args.dual_lam
        dual_lam_end = getattr(args, "dual_lam_end", 0.01)
        dual_schedule = getattr(args, "dual_schedule", "linear")
        dual_warmup_rounds = getattr(args, "dual_warmup_rounds", 0)
        dual_l2_lam = getattr(args, "dual_l2_lam", 0.0)
        if dual_l2_lam > 0.0:
            # Elastic net: scale L2 threshold by the same ratio as L1 at each step
            _l2_ratio = dual_l2_lam / dual_lam
            _prox = lambda z, eta: elasticnet_threshold(z, eta, eta * _l2_ratio)
            print(f"[feddualavg] proximal: elastic net (λ₁={dual_lam}, λ₂={dual_l2_lam})")
        else:
            _prox = soft_threshold
            print(f"[feddualavg] proximal: L1 soft threshold (λ={dual_lam})")

        # Dual state: accumulated (negative) gradients — same shape as weight and bias
        z_weight = torch.zeros(num_classes, num_concepts, device=device)
        z_bias = torch.zeros(num_classes, device=device)

        # Count total local steps per round (K) for eta_tilde schedule
        # K = final_epochs * avg_batches_per_epoch; approximate from first client
        K_approx = final_epochs * len(client_concept_loaders[0])

        dual_metrics = {
            "rounds": [], "client_losses": [], "avg_client_loss": [],
            "val_accuracy": [], "best_val_accuracy": [],
            "concepts_alive": [], "nnz_weights": [],
            "eta_tilde": [],
        }
        best_val_acc = 0.0
        best_fl_state = None

        for round_num in range(final_rounds):
            print(f"\n=== FedDualAvg Round {round_num + 1}/{final_rounds} ===")
            round_losses = []
            client_z_deltas_w = []
            client_z_deltas_b = []

            for i in range(args.num_clients):
                # Each client starts from the global dual state
                z_local_w = z_weight.clone()
                z_local_b = z_bias.clone()

                client_loss_sum = 0.0
                n_steps = 0

                for epoch in range(final_epochs):
                    for feats, labels in client_concept_loaders[i]:
                        feats, labels = feats.to(device), labels.to(device)

                        # Step counter for eta_tilde schedule
                        if round_num < dual_warmup_rounds:
                            eta_tilde = 0.0
                        else:
                            effective_round = round_num - dual_warmup_rounds
                            effective_rounds = final_rounds - dual_warmup_rounds
                            effective_step = effective_round * K_approx + n_steps + 1
                            effective_total = effective_rounds * K_approx
                            if dual_schedule == "burnin":
                                progress = effective_step / effective_total
                                eta_tilde = dual_lam_end + (dual_lam - dual_lam_end) * (1.0 - progress)
                            else:
                                eta_tilde = eta_s * eta_c * effective_step * dual_lam

                        # 1. Primal recovery: w = prox(z, eta_tilde)
                        w_primal = _prox(z_local_w, eta_tilde)
                        b_primal = z_local_b.clone()  # no regularization on bias

                        # 2. Forward pass and gradient at primal point
                        w_param = w_primal.detach().requires_grad_(True)
                        b_param = b_primal.detach().requires_grad_(True)
                        logits = feats @ w_param.T + b_param
                        loss = ce_loss(logits, labels)
                        loss.backward()
                        grad_w = w_param.grad.detach()
                        grad_b = b_param.grad.detach()

                        # 3. Update dual: z -= eta_c * grad
                        z_local_w -= eta_c * grad_w
                        z_local_b -= eta_c * grad_b

                        client_loss_sum += loss.item()
                        n_steps += 1

                # Client dual delta
                delta_w = z_local_w - z_weight
                delta_b = z_local_b - z_bias
                client_z_deltas_w.append(delta_w)
                client_z_deltas_b.append(delta_b)
                round_losses.append(client_loss_sum / max(n_steps, 1))
                print(f"  Client {i}: Loss = {round_losses[-1]:.4f}")

                # Snapshot: retain per-client dual state from the last round
                _snapshot_dir = getattr(args, "phase3_snapshot_dir", None)
                if _snapshot_dir and round_num == final_rounds - 1:
                    if not hasattr(args, "_snap_z_states"):
                        args._snap_z_states = []
                    args._snap_z_states.append((z_local_w.clone().cpu(), z_local_b.clone().cpu()))

            # Server: weighted average of dual deltas
            avg_delta_w = torch.zeros_like(z_weight)
            avg_delta_b = torch.zeros_like(z_bias)
            for i in range(args.num_clients):
                avg_delta_w += client_weights[i] * client_z_deltas_w[i]
                avg_delta_b += client_weights[i] * client_z_deltas_b[i]

            z_weight += eta_s * avg_delta_w
            z_bias += eta_s * avg_delta_b

            # Server primal recovery
            if round_num < dual_warmup_rounds:
                eta_tilde_server = 0.0
            else:
                effective_round = round_num - dual_warmup_rounds
                effective_rounds = final_rounds - dual_warmup_rounds
                server_step = (effective_round + 1) * K_approx
                effective_total = effective_rounds * K_approx
                if dual_schedule == "burnin":
                    progress = server_step / effective_total
                    eta_tilde_server = dual_lam_end + (dual_lam - dual_lam_end) * (1.0 - progress)
                else:
                    eta_tilde_server = eta_s * eta_c * server_step * dual_lam
            w_server = _prox(z_weight, eta_tilde_server)
            b_server = z_bias.clone()

            # Write per-client primal snapshots at the last round
            if getattr(args, "phase3_snapshot_dir", None) and round_num == final_rounds - 1:
                _snap_dir = args.phase3_snapshot_dir
                os.makedirs(_snap_dir, exist_ok=True)
                for _ci, (_zw, _zb) in enumerate(getattr(args, "_snap_z_states", [])):
                    _w_snap = _prox(_zw.to(device), eta_tilde_server).cpu()
                    torch.save({"weight": _w_snap, "bias": _zb},
                               os.path.join(_snap_dir, f"client_{_ci}_primal.pt"))
                torch.save({"weight": w_server.cpu(), "bias": b_server.cpu()},
                           os.path.join(_snap_dir, "global_primal.pt"))
                print(f"[snapshot] Per-client primals saved to {_snap_dir}")

            # Load primal into final layer for evaluation
            with torch.no_grad():
                final_layer.weight.copy_(w_server)
                final_layer.bias.copy_(b_server)

            # Sparsity stats
            nnz = (w_server.abs() > 1e-5).sum().item()
            total_w = w_server.numel()
            col_norms = w_server.norm(p=2, dim=0)
            concepts_alive = (col_norms > 1e-5).sum().item()
            concepts_total = w_server.shape[1]
            print(f"  eta_tilde_server={eta_tilde_server:.6f}  col_norms: min={col_norms.min():.4f} median={col_norms.median():.4f} max={col_norms.max():.4f}")
            print(f"  Weights: {nnz}/{total_w} non-zero  Concepts: {concepts_alive}/{concepts_total}")

            # Federated evaluation
            final_layer.eval()
            val_acc = 0.0
            with torch.no_grad():
                for i in range(args.num_clients):
                    client_correct, client_total = 0, 0
                    for feats, labels in client_val_concept_loaders[i]:
                        feats, labels = feats.to(device), labels.to(device)
                        preds = final_layer(feats).argmax(dim=1)
                        client_correct += (preds == labels).sum().item()
                        client_total += labels.size(0)
                    val_acc += client_val_weights[i] * (client_correct / max(client_total, 1))
            print(f"  Val Accuracy: {val_acc:.4f}")
            # On val_acc ties, prefer the sparser iterate. Without this, once val_acc
            # saturates (common on tiny val splits) the first-saturating (denser) iterate
            # is kept and all subsequent L1 thresholding is effectively discarded.
            _cur_nnz = nnz
            _best_nnz = dual_metrics["nnz_weights"][-1] if dual_metrics.get("nnz_weights") and best_fl_state is not None else None
            if (val_acc > best_val_acc) or (
                val_acc == best_val_acc and best_fl_state is not None and _cur_nnz < int((best_fl_state["weight"].abs() > 1e-5).sum().item())
            ):
                best_val_acc = val_acc
                best_fl_state = {k: v.clone() for k, v in final_layer.state_dict().items()}

            dual_metrics["rounds"].append(round_num + 1)
            dual_metrics["client_losses"].append(round_losses)
            dual_metrics["avg_client_loss"].append(sum(round_losses) / len(round_losses))
            dual_metrics["val_accuracy"].append(float(val_acc))
            dual_metrics["best_val_accuracy"].append(float(best_val_acc))
            dual_metrics["concepts_alive"].append(int(concepts_alive))
            dual_metrics["nnz_weights"].append(int(nnz))
            dual_metrics["eta_tilde"].append(float(eta_tilde_server))
            update_log(_log_path, {"status": "in_progress", "phase": "final_layer_feddualavg",
                                    "round": round_num + 1, "total_rounds": final_rounds,
                                    "val_accuracy": float(val_acc), "best_val_accuracy": float(best_val_acc),
                                    "concepts_alive": int(concepts_alive), "nnz_weights": int(nnz),
                                    "eta_tilde": float(eta_tilde_server)})

        if best_fl_state is not None:
            final_layer.load_state_dict(best_fl_state)
        global_model.final_layer = final_layer

        # Report sparsity
        nnz = (final_layer.weight.data.abs() > 1e-5).sum().item()
        total_w = final_layer.weight.data.numel()
        print(f"Final layer sparsity: {nnz}/{total_w} non-zero ({nnz/total_w:.4f})")

    global_model.backbone.to(device)     # move backbone back to GPU for evaluation
    log_mem("before test evaluation (backbone back on GPU)")
    test_acc = test_model(test_loader, global_model.backbone, global_model.cbl, global_model.normalization, global_model.final_layer, str(device))
    print(f"Test accuracy: {test_acc:.4f}")

    # Save model immediately after final layer training completes
    print(f"\nSaving model to {save_dir}...")
    global_model.backbone.save_model(save_dir)
    global_model.cbl.save_model(save_dir)
    global_model.normalization.save_model(save_dir)
    global_model.final_layer.save_model(save_dir)
    print("Model saved successfully!")

    # Try to compute per-class accuracy, but don't fail if it errors
    pca = None
    try:
        pca = per_class_accuracy(global_model, test_loader, classes, str(device))
    except Exception as e:
        print(f"Warning: Failed to compute per-class accuracy: {e}")

    sparsity_vlg = {
        "Non-zero weights": int((global_model.final_layer.weight.data.abs() > 1e-5).sum().item()),
        "Total weights": int(global_model.final_layer.weight.data.numel()),
        "Percentage non-zero": float((global_model.final_layer.weight.data.abs() > 1e-5).float().mean().item()),
    }
    metrics_txt_data = {
        "final_layer_method": vlg_final_method,
        "per_class_accuracies": pca,
        "metrics": {"test_accuracy": float(test_acc)},
        "sparsity": sparsity_vlg,
    }
    if vlg_final_method == "hybrid_saga":
        metrics_txt_data["saga_lam"] = getattr(args, "saga_lam", 0.0007)
        metrics_txt_data["saga_n_iters"] = getattr(args, "saga_n_iters", 2000)
    elif vlg_final_method == "fedavg":
        metrics_txt_data["dense_lr"] = getattr(args, "dense_lr", 0.001)
    elif vlg_final_method == "fedavg_thresh":
        metrics_txt_data["thresh_lam_start"] = args.thresh_lam_start
        metrics_txt_data["thresh_lam_end"] = args.thresh_lam_end
        metrics_txt_data["final_rounds"] = getattr(args, "final_rounds", 5)
        metrics_txt_data["final_lr"] = getattr(args, "final_lr", 1e-3)
    elif vlg_final_method == "fedavg_l1":
        metrics_txt_data["fedavg_l1_lam"] = float(args.fedavg_l1_lam)
        metrics_txt_data["final_rounds"] = getattr(args, "final_rounds", 5)
        metrics_txt_data["final_lr"] = getattr(args, "final_lr", 1e-3)
    elif vlg_final_method == "feddualavg":
        metrics_txt_data["dual_eta_s"] = args.dual_eta_s
        metrics_txt_data["dual_eta_c"] = args.dual_eta_c
        metrics_txt_data["dual_lam"] = args.dual_lam
        metrics_txt_data["dual_schedule"] = getattr(args, "dual_schedule", "linear")
        metrics_txt_data["final_rounds"] = getattr(args, "final_rounds", 5)
        if getattr(args, "dual_schedule", "linear") == "burnin":
            metrics_txt_data["dual_lam_end"] = getattr(args, "dual_lam_end", 0.01)
        if getattr(args, "dual_warmup_rounds", 0) > 0:
            metrics_txt_data["dual_warmup_rounds"] = args.dual_warmup_rounds
    save_metrics_txt(save_dir, metrics_txt_data)

    # Wrap NEC evaluation in try-except so it doesn't prevent model saving
    if getattr(args, "run_nec_eval", True) and not getattr(args, "dense", False):
        try:
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
        except Exception as e:
            print(f"Warning: NEC evaluation failed: {e}")
            print("Model has already been saved. Continuing...")

    training_metrics = {
        "projection_phase": projection_metrics,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "final_rounds": getattr(args, "final_rounds", None),
        "final_epochs": getattr(args, "final_epochs", None),
        "client_data_sizes": client_data_sizes,
        "client_weights": client_weights,
        "iid": args.iid,
        "alpha": args.alpha if not args.iid else None,
        "best_final_accuracy": float(test_acc),
        "final_layer_method": vlg_final_method,
    }
    if vlg_final_method == "fedavg_thresh":
        training_metrics["final_layer_phase"] = thresh_metrics
    elif vlg_final_method == "fedavg_l1":
        training_metrics["final_layer_phase"] = l1_metrics
    elif vlg_final_method == "feddualavg":
        training_metrics["final_layer_phase"] = dual_metrics
    elif vlg_final_method in ("hybrid_saga", "fedavg"):
        training_metrics["final_layer_phase"] = vlg_central_metrics
    save_training_metrics(save_dir, training_metrics)
    update_log(_log_path, {"status": "completed", "phase": "fully_trained",
                            "test_accuracy": float(test_acc),
                            "final_layer_method": vlg_final_method,
                            "completed_at": datetime.datetime.now().isoformat()})
    print(f"Saved to {save_dir}")
