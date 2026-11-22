#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract features for distance selection using CLIP text embeddings
instead of a 37D one-hot category vector (open-world support).

Feature dimension:
  1035D = 523D (512D DINOv3-Base + 10D depth + 1D distance) + 512D CLIP category.
"""
from argparse import ArgumentParser
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from feature_extractor import DINOv3BaseFeatureExtractor
from category_encoding import (
    extract_category_from_sample_name,
    category_to_clip_embedding,
    get_clip_embedding_dim,
)

FEATURE_DIM = 523 + get_clip_embedding_dim()  # 1035


def extract_features_for_sample(sample_dir, data_dir, sample_name, extractor, device="cuda"):
    """
    Extract features for all distances of a single sample
    with category encoded by CLIP text embeddings.
    """
    features_dict = {}
    category = extract_category_from_sample_name(sample_name)
    clip_emb = category_to_clip_embedding(category, device=device)

    for dist in np.arange(1.0, 6.05, 0.1):
        dist_str = f"{dist:.1f}"
        try:
            img_path = os.path.join(data_dir, sample_name, f"camera_rgb_dist{dist_str}m.png")
            if not os.path.exists(img_path):
                continue
            depth_path = os.path.join(data_dir, sample_name, f"camera_depth_dist{dist_str}m.npy")
            if not os.path.exists(depth_path):
                continue

            image = np.array(Image.open(img_path).convert("RGB")) / 255.0
            depth = np.load(depth_path)
            features_523d = extractor.extract(image, depth, dist)
            features_1035d = np.concatenate([features_523d, clip_emb])
            features_dict[dist_str] = features_1035d
        except Exception as e:
            print(f"  Warning: failed to process {sample_name} at {dist_str}m: {e}")
            continue
    return features_dict


def main():
    parser = ArgumentParser(
        description="Batch feature extraction with CLIP category encoding (open-world)."
    )
    parser.add_argument("--eval_dir", type=str, required=True, help="evaluation directory")
    parser.add_argument("--data_dir", type=str, required=True, help="RGB-D data directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_prefix", type=str, default="features_dinov3_base_clip_")
    args = parser.parse_args()

    print("=" * 80)
    print("Batch feature extraction (1035D = 523D + 512D CLIP category) — open-world support")
    print("=" * 80)

    print("\n[1/3] Initialize DINOv3-Base feature extractor...")
    extractor = DINOv3BaseFeatureExtractor(device=args.device)
    print(f"  Feature dimension: 1035D (523D + 512D CLIP)")

    print("\n[2/3] Scanning samples...")
    sample_folders = [
        f for f in os.listdir(args.eval_dir)
        if os.path.isdir(os.path.join(args.eval_dir, f))
    ]
    print(f"  Found {len(sample_folders)} samples")

    print("\n[3/3] Extracting features...")
    total_features = 0
    total_samples = 0
    for sample_name in tqdm(sample_folders, desc="Processing samples"):
        sample_dir = os.path.join(args.eval_dir, sample_name)
        features_dict = extract_features_for_sample(
            sample_dir, args.data_dir, sample_name, extractor, args.device
        )
        if not features_dict:
            continue
        for dist_str, features in features_dict.items():
            out_path = os.path.join(sample_dir, f"{args.feature_prefix}dist{dist_str}.npy")
            np.save(out_path, features)
            total_features += 1
        total_samples += 1

    print(f"\nDone! samples: {total_samples}, features: {total_features}, dim: 1035D")
    print("=" * 80)


if __name__ == "__main__":
    main()
