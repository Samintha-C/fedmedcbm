import torch
import clip
import os
import json


def load_concepts_from_file(concept_file):
    if not os.path.isabs(concept_file) and not os.path.exists(concept_file):
        lf_cbm_path = os.path.join(os.path.dirname(__file__), '../../Label-free-CBM/data/concept_sets', concept_file)
        if os.path.exists(lf_cbm_path):
            concept_file = lf_cbm_path
    
    with open(concept_file, 'r') as f:
        concepts = [line.strip() for line in f.readlines() if line.strip()]
    return concepts


def generate_clip_text_embeddings(concepts, clip_model, device="cuda"):
    text = clip.tokenize(concepts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.float()


def load_or_generate_concept_embeddings(concepts, clip_name="ViT-B/16", device="cuda", cache_dir=None):
    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, f"concept_embeddings_{clip_name.replace('/', '_')}.pt")
        if os.path.exists(cache_file):
            return torch.load(cache_file, map_location=device)
    
    clip_model, _ = clip.load(clip_name, device=device)
    embeddings = generate_clip_text_embeddings(concepts, clip_model, device)
    
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(embeddings, cache_file)
    
    return embeddings


def filter_concepts_by_activation(clip_features, concepts, top_k=5, threshold=0.25):
    highest = torch.mean(torch.topk(clip_features, dim=0, k=top_k)[0], dim=0)
    mask = highest > threshold
    filtered_concepts = [concepts[i] for i in range(len(concepts)) if mask[i]]
    return filtered_concepts, mask
