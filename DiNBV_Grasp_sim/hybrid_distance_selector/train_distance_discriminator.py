#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a distance discriminator with CLIP category features.

Feature dimension:
  1035D = 523D (512D visual + 10D depth + 1D distance) + 512D CLIP text embedding.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser


class DistanceDiscriminatorWithCategory(nn.Module):
    """Distance discriminator with CLIP category features and self-attention."""

    def __init__(self, input_dim: int = 1035,
                 hidden_dims=[256, 128, 64],
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()

        self.input_dim = input_dim

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # MLP blocks
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LayerNorm(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_layer(x)

        # Attention expects 3D tensor: [batch, seq_len, dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]

        attn_out, _ = self.attention(x, x, x)
        x = attn_out.squeeze(1)  # [B, D]

        # MLP
        x = self.hidden_layers(x)

        # Output
        x = self.output_layer(x)
        return x


def split_dataset(eval_dir: str, train_ratio: float = 0.8, seed: int = 42):
    """Split scenes into train/test subsets."""
    sample_folders = [
        f for f in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, f))
    ]

    np.random.seed(seed)
    np.random.shuffle(sample_folders)

    split_idx = int(len(sample_folders) * train_ratio)
    train_samples = sample_folders[:split_idx]
    test_samples = sample_folders[split_idx:]

    print(f"  Split dataset: {len(train_samples)} train, {len(test_samples)} test")
    return train_samples, test_samples


class ContrastiveDistanceDataset(Dataset):
    """Dataset for distance discriminator with per-scene grouping."""

    def __init__(self, eval_dir: str,
                 sample_names=None,
                 feature_prefix: str = "features_dinov3_base_clip_"):
        self.eval_dir = eval_dir
        self.samples = []
        self.feature_prefix = feature_prefix
        self.scene_data = {}

        # Load all scenes
        if sample_names is None:
            sample_folders = [
                f for f in os.listdir(eval_dir)
                if os.path.isdir(os.path.join(eval_dir, f))
            ]
        else:
            sample_folders = sample_names

        for sample_name in sample_folders:
            sample_dir = os.path.join(eval_dir, sample_name)
            scene_samples = []

            # Load all distances for this scene
            for dist in np.arange(1.0, 6.05, 0.1):
                dist_str = f"{dist:.1f}"
                result_file = os.path.join(sample_dir, f"dist_{dist_str}_result.json")
                feature_file = os.path.join(
                    sample_dir, f"{self.feature_prefix}dist{dist_str}.npy"
                )

                if os.path.exists(result_file) and os.path.exists(feature_file):
                    with open(result_file, "r") as f:
                        result = json.load(f)
                        success = bool(result.get("mani_succ", False))

                    features = np.load(feature_file)

                    sample = {
                        "sample_name": sample_name,
                        "distance": dist,
                        "features": features,
                        "success": success,
                        "label": 1.0 if success else 0.0,
                    }
                    self.samples.append(sample)
                    scene_samples.append(sample)

            if scene_samples:
                self.scene_data[sample_name] = scene_samples

        print(
            f"Loaded {len(self.samples)} samples from {len(self.scene_data)} scenes"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            torch.tensor(item["features"], dtype=torch.float32),
            torch.tensor([item["label"]], dtype=torch.float32),
        )

    def get_contrastive_pair(self):
        """Sample a positive/negative pair within the same scene."""
        if not self.scene_data:
            return None, None

        scene_name = np.random.choice(list(self.scene_data.keys()))
        scene_samples = self.scene_data[scene_name]

        positive_samples = [s for s in scene_samples if s["success"]]
        negative_samples = [s for s in scene_samples if not s["success"]]
        if not positive_samples or not negative_samples:
            return None, None

        pos = np.random.choice(positive_samples)
        neg = np.random.choice(negative_samples)

        pos_feat = torch.tensor(pos["features"], dtype=torch.float32)
        neg_feat = torch.tensor(neg["features"], dtype=torch.float32)
        return pos_feat, neg_feat

    def get_random_scene(self):
        """Return all distances in a random scene (for scene-level ranking loss)."""
        if not self.scene_data:
            return None, None

        scene_name = np.random.choice(list(self.scene_data.keys()))
        scene_samples = self.scene_data[scene_name]

        features = np.array([s["features"] for s in scene_samples])
        labels = np.array([s["label"] for s in scene_samples])

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )


def evaluate_top1_accuracy(model, dataset, device):
    """Scene-level Top-1 accuracy: does the highest-scoring distance succeed?"""
    model.eval()
    top1_correct = 0
    total_scenes = 0

    with torch.no_grad():
        for scene_name, scene_samples in dataset.scene_data.items():
            if not scene_samples:
                continue

            features = np.array([s["features"] for s in scene_samples])
            success_labels = np.array([s["success"] for s in scene_samples])

            features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
            scores = model(features_tensor).cpu().numpy().flatten()

            best_idx = int(np.argmax(scores))
            if success_labels[best_idx]:
                top1_correct += 1
            total_scenes += 1

    return top1_correct / total_scenes if total_scenes > 0 else 0.0


def train_discriminator(args):
    """Main training loop for the distance discriminator."""
    print("=" * 80)
    print("Train distance discriminator with CLIP category features")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(
        f"Feature dimension: {args.input_dim}D "
        "(1035D = 523D + 512D CLIP category)"
    )

    # Train / test split
    if args.use_data_split:
        print(f"\nSplitting dataset (train_ratio={args.train_ratio}, seed={args.seed})...")
        train_samples, test_samples = split_dataset(
            args.eval_dir, args.train_ratio, args.seed
        )
    else:
        print("\nUsing all data for training (no explicit test split)...")
        train_samples = None
        test_samples = None

    # Build datasets
    print("\nLoading training data...")
    train_dataset = ContrastiveDistanceDataset(
        args.eval_dir, train_samples, args.feature_prefix
    )

    if args.use_data_split:
        print("\nLoading test data...")
        test_dataset = ContrastiveDistanceDataset(
            args.eval_dir, test_samples, args.feature_prefix
        )
    else:
        test_dataset = None

    if len(train_dataset) == 0:
        print("\nERROR: no training samples found!")
        print(f"  eval_dir: {args.eval_dir}")
        print(f"  feature_prefix: {args.feature_prefix}")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        test_loader = None

    # Model
    print("\nInitializing model...")
    model = DistanceDiscriminatorWithCategory(
        input_dim=args.input_dim,
        hidden_dims=[256, 128, 64],
        num_heads=4,
        dropout=0.2,
    ).to(device)
    print(f"  #params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    bce_loss = nn.BCELoss()
    margin_loss = nn.MarginRankingLoss(margin=0.2)

    os.makedirs(args.output_dir, exist_ok=True)
    best_test_acc = 0.0
    best_top1_acc = 0.0

    print(f"\nStart training for {args.epochs} epochs...")
    print("-" * 80)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            # Forward
            predictions = model(features)

            # Binary classification loss
            loss_bce = bce_loss(predictions, labels)

            # Scene-level ranking loss (probabilistic sampling)
            loss_ranking = torch.tensor(0.0, device=device)
            if np.random.rand() < 0.3:
                scene_feats, scene_labels = train_dataset.get_random_scene()
                if scene_feats is not None:
                    scene_feats = scene_feats.to(device)
                    scene_labels = scene_labels.to(device)

                    scene_scores = model(scene_feats).squeeze(-1)
                    success_mask = scene_labels > 0.5
                    failure_mask = scene_labels < 0.5
                    if success_mask.sum() > 0 and failure_mask.sum() > 0:
                        success_mean = scene_scores[success_mask].mean()
                        failure_mean = scene_scores[failure_mask].mean()
                        margin = 0.3
                        loss_ranking = torch.clamp(
                            margin - (success_mean - failure_mean), min=0.0
                        )

            # Contrastive loss between positive/negative distances
            loss_contrastive = torch.tensor(0.0, device=device)
            if np.random.rand() < 0.3:
                pos_feat, neg_feat = train_dataset.get_contrastive_pair()
                if pos_feat is not None and neg_feat is not None:
                    pos_feat = pos_feat.unsqueeze(0).to(device)
                    neg_feat = neg_feat.unsqueeze(0).to(device)
                    pos_score = model(pos_feat).view(-1)
                    neg_score = model(neg_feat).view(-1)
                    target = torch.ones(1, device=device)
                    loss_contrastive = margin_loss(pos_score, neg_score, target)

            loss = loss_bce + 1.0 * loss_ranking + 0.5 * loss_contrastive

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred_labels = (predictions > 0.5).float()
            train_correct += (pred_labels == labels).sum().item()
            train_total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs}, "
            f"Loss: {avg_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}",
            end="",
        )

        # Evaluate
        if test_loader is not None:
            model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0

            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(device)
                    labels = labels.to(device)

                    predictions = model(features)
                    loss_val = bce_loss(predictions, labels)
                    test_loss += loss_val.item()

                    pred_labels = (predictions > 0.5).float()
                    test_correct += (pred_labels == labels).sum().item()
                    test_total += labels.size(0)

            test_acc = test_correct / test_total
            top1_acc = evaluate_top1_accuracy(model, test_dataset, device)
            print(
                f", Test Acc: {test_acc:.4f}, Top-1 Acc: {top1_acc:.4f}",
                end="",
            )

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_test_acc = test_acc
                best_model_path = os.path.join(
                    args.output_dir, "discriminator_best.pth"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "test_acc": test_acc,
                        "top1_acc": top1_acc,
                        "input_dim": args.input_dim,
                    },
                    best_model_path,
                )
                print(" → best model updated", end="")

        print()

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"discriminator_epoch{epoch + 1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )

    # Final checkpoint
    final_model_path = os.path.join(args.output_dir, "discriminator_final.pth")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_model_path,
    )

    print("\n" + "=" * 80)
    print("Training finished!")
    print(f"  Best test accuracy (binary): {best_test_acc:.4f}")
    print(f"  Best Top-1 accuracy (scene-level): {best_top1_acc:.4f}")
    print(f"  Checkpoints saved to: {args.output_dir}")
    print("=" * 80)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory containing per-distance evaluation JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models/discriminator_ckpts",
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--feature_prefix",
        type=str,
        default="features_dinov3_base_clip_",
        help="Prefix of saved feature .npy files.",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=1035,
        help="Input feature dimension "
             "(1035D = 523D + 512D CLIP category).",
    )
    parser.add_argument(
        "--use_data_split",
        action="store_true",
        help="Whether to split scenes into train/test sets.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio (if use_data_split is set).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu').",
    )

    args = parser.parse_args()
    train_discriminator(args)


if __name__ == "__main__":
    main()

