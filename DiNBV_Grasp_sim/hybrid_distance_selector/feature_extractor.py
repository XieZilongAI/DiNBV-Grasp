"""
Feature extraction for camera distance selection.

Supported extractors:
  - DINOv3BaseFeatureExtractor (default): DINOv3 ViT-B/16,
    projects the 768D class token to 512D.
  - CNNFeatureExtractor: ResNet18 backbone (512D), kept as fallback.

Final feature vector per distance candidate:
  523D = 512D visual + 10D depth statistics + 1D normalized distance
"""
import numpy as np
import cv2
from PIL import Image
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# DINOv3-Base Feature Extractor (primary)
# ---------------------------------------------------------------------------

class DINOv3BaseFeatureExtractor:
    """
    Extract 512D visual features using the DINOv3-Base ViT-B/16.

    Backbone: vit_base_patch16_dinov3.lvd1689m (timm)
    Output:   768D class token → linear projection → 512D
    Total feature vector: 523D = 512D + 10D depth + 1D distance
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        import timm
        self.device = device

        # Removing the classifier exposes the pretrained model's class token.
        backbone = timm.create_model(
            'vit_base_patch16_dinov3.lvd1689m',
            pretrained=True,
            num_classes=0,   # returns 768D pooled features
        )
        backbone.eval()
        backbone.to(device)

        # Use the same projection across feature-extraction runs.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.projection = nn.Linear(backbone.num_features, 512, bias=False)
            nn.init.orthogonal_(self.projection.weight)
        self.projection.eval()
        self.projection.to(device)

        self.backbone = backbone

        # Use model-specific normalization from timm
        data_config = timm.data.resolve_model_data_config(backbone)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

        print(f"DINOv3BaseFeatureExtractor initialized on {device}")

    def extract_visual_features(self, image):
        """
        Extract 512D visual features from an RGB image.

        Args:
            image: np.ndarray (H, W, 3) float32 in [0, 1]

        Returns:
            np.ndarray of shape (512,)
        """
        pil_img = Image.fromarray((image * 255).astype(np.uint8))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_768 = self.backbone(tensor)           # (1, 768)
            feat_512 = self.projection(feat_768)       # (1, 512)

        return feat_512.cpu().numpy().flatten()

    def extract_depth_statistics(self, depth):
        """Extract 10D depth statistics (shared with CNNFeatureExtractor)."""
        return _extract_depth_statistics(depth)

    def extract(self, image, depth, distance):
        """
        Extract 523D feature vector.

        Args:
            image:    np.ndarray (H, W, 3) float32 [0, 1]
            depth:    np.ndarray (H, W)
            distance: float, camera distance in metres

        Returns:
            np.ndarray of shape (523,)
        """
        visual = self.extract_visual_features(image)
        depth_feat = self.extract_depth_statistics(depth)
        dist_feat = np.array([distance / 6.0], dtype=np.float32)
        return np.concatenate([visual, depth_feat, dist_feat])

    def extract_batch(self, images, depths, distances, batch_size=16):
        """Batch extraction."""
        features_list = []
        for i in range(0, len(images), batch_size):
            for img, depth, dist in zip(
                images[i:i + batch_size],
                depths[i:i + batch_size],
                distances[i:i + batch_size],
            ):
                features_list.append(self.extract(img, depth, dist))
        return np.stack(features_list)


# ---------------------------------------------------------------------------
# Shared depth-statistics helper
# ---------------------------------------------------------------------------

def _extract_depth_statistics(depth):
    """
    Extract 10D depth statistics from a depth map.

    Returns:
        np.ndarray of shape (10,)
    """
    features = []
    valid_depth = depth[(depth > 0.1) & (depth < 0.99)]

    if len(valid_depth) > 0:
        features.append(float(valid_depth.mean()))
        features.append(float(valid_depth.std()))
        features.append(float(valid_depth.min()))
        features.append(float(valid_depth.max()))

        depth_grad_x = np.abs(np.gradient(depth, axis=1))
        depth_grad_y = np.abs(np.gradient(depth, axis=0))
        features.append(float(depth_grad_x.mean()))
        features.append(float(depth_grad_y.mean()))

        depth_uint8 = (depth * 255).astype(np.uint8)
        sobelx = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        features.append(float(edge_magnitude.mean()))
        features.append(float(edge_magnitude.std()))

        features.append(len(valid_depth) / depth.size)
        features.append(float((valid_depth > valid_depth.mean()).sum() / len(valid_depth)))
    else:
        features.extend([0.0] * 10)

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# ResNet18-based extractor (fallback / legacy)
# ---------------------------------------------------------------------------

class CNNFeatureExtractor:
    """
    ResNet18-based feature extractor (fallback).
    Output: 523D = 512D CNN + 10D depth + 1D distance
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Identity()  # type: ignore  → 512D output
        self.model.eval()
        self.model.to(device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"CNNFeatureExtractor (ResNet18, fallback) initialized on {device}")

    def extract_cnn_features(self, image):
        image_uint8 = (image * 255).astype(np.uint8)
        tensor = self.transform(image_uint8).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        return features.cpu().numpy().flatten()

    def extract_depth_statistics(self, depth):
        return _extract_depth_statistics(depth)

    def extract(self, image, depth, distance):
        cnn_feat = self.extract_cnn_features(image)
        depth_feat = self.extract_depth_statistics(depth)
        dist_feat = np.array([distance / 6.0], dtype=np.float32)
        return np.concatenate([cnn_feat, depth_feat, dist_feat])

    def extract_batch(self, images, depths, distances, batch_size=16):
        features_list = []
        for i in range(0, len(images), batch_size):
            for img, depth, dist in zip(
                images[i:i + batch_size],
                depths[i:i + batch_size],
                distances[i:i + batch_size],
            ):
                features_list.append(self.extract(img, depth, dist))
        return np.stack(features_list)


# ---------------------------------------------------------------------------
# Simple 5D extractor (lightweight baseline, no deep network)
# ---------------------------------------------------------------------------

class SimpleFeatureExtractor:
    """5D handcrafted features, no GPU required. Runs in < 2 ms."""

    def extract(self, image, depth, distance):
        features = [distance / 6.0]
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(min(clarity / 1000.0, 1.0))
        features.append(gray.std() / 255.0)
        valid_depth = depth[(depth > 0.1) & (depth < 0.99)]
        if len(valid_depth) > 0:
            features.append(min(valid_depth.std() * 10, 1.0))
            features.append(len(valid_depth) / depth.size)
        else:
            features.extend([0.0, 0.0])
        return np.array(features, dtype=np.float32)

    def extract_batch(self, images, depths, distances):
        return np.stack([self.extract(i, d, dist)
                         for i, d, dist in zip(images, depths, distances)])
