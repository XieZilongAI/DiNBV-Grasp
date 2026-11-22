#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Category encoding for open-world distance selection using CLIP text embeddings.
Any textual category name is mapped to a 512D semantic vector (no fixed label set),
which naturally supports new/unseen categories.
"""
import numpy as np

# Lazy load CLIP to avoid import cost when not used
_clip_model = None
_clip_tokenizer = None
_CLIP_DIM = 512


def _get_clip():
    global _clip_model, _clip_tokenizer
    if _clip_model is None:
        try:
            import open_clip
            _clip_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            _clip_model.eval()
        except Exception as e:
            raise RuntimeError(
                "CLIP not available. Install: pip install open_clip_torch. Error: " + str(e)
            )
    return _clip_model, _clip_tokenizer


def get_clip_embedding_dim():
    """Return CLIP text embedding dimension (512 for ViT-B-32)."""
    return _CLIP_DIM


def category_to_clip_embedding(category: str, device: str = "cuda"):
    """
    Encode a category name into a 512D CLIP text embedding.
    Works for arbitrary category strings (no fixed vocabulary).

    Args:
        category: Category name, e.g. "Faucet", "Coffee Mug", "Unknown object"
        device: "cuda" or "cpu"

    Returns:
        emb: np.ndarray of shape (512,) dtype float32, L2-normalized
    """
    import torch
    model, tokenizer = _get_clip()
    model = model.to(device)
    # Use natural language prompts to help CLIP understand the category.
    text = f"a {category}" if category and category != "Unknown" else "an object"
    with torch.no_grad():
        tokens = tokenizer([text]).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy().astype(np.float32).flatten()
    assert emb.shape == (_CLIP_DIM,), f"expected shape ({_CLIP_DIM},), got {emb.shape}"
    return emb


def category_to_clip_embedding_batch(categories: list, device: str = "cuda"):
    """
    Encode a batch of category names, more efficient than looping.

    Args:
        categories: list of category names
        device: "cuda" or "cpu"

    Returns:
        np.ndarray of shape (N, 512) float32
    """
    import torch
    model, tokenizer = _get_clip()
    model = model.to(device)
    texts = [f"a {c}" if c and c != "Unknown" else "an object" for c in categories]
    with torch.no_grad():
        tokens = tokenizer(texts)
        if hasattr(tokens, "to"):
            tokens = tokens.to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy().astype(np.float32)
    return emb


def extract_category_from_sample_name(sample_name: str):
    """
    Parse category from a sample folder name (kept for backward compatibility
    with the original dataset naming scheme).
    """
    parts = sample_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "Unknown"
