"""
Hybrid distance selector - combines heuristic, discriminator, and history
Supports both 5D simple features and 523D CNN features
Inference time: <15ms (5D) or ~50-80ms (523D CNN)
"""
import numpy as np
import torch
import torch.nn as nn
from feature_extractor import SimpleFeatureExtractor, CNNFeatureExtractor
from build_history_cache import HistoryCache

class LightweightDistanceDiscriminator(nn.Module):
    """Lightweight discriminator for simple features"""
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


class DistanceDiscriminatorWithAttention(nn.Module):
    """Enhanced discriminator with multi-head attention for CNN features"""
    def __init__(self, input_dim=523, hidden_dims=[256, 128, 64], 
                 num_heads=4, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.attn_norm = nn.LayerNorm(hidden_dims[0])
        
        # MLP layers after attention
        mlp_layers = []
        for i in range(len(hidden_dims) - 1):
            mlp_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input projection
        h = self.input_proj(x)
        
        # Add sequence dimension for attention if needed
        if len(h.shape) == 2:
            h = h.unsqueeze(1)
        
        # Self-attention with residual connection
        attn_out, _ = self.attention(h, h, h)
        h = self.attn_norm(h + attn_out)
        
        # Remove sequence dimension
        h = h.squeeze(1)
        
        # MLP layers
        h = self.mlp(h)
        
        # Output
        output = self.output_layer(h)
        
        return output


class HybridDistanceSelector:
    """
    Hybrid distance selector combining:
    1. Heuristic rules (50% weight)
    2. Learned discriminator (30% weight)
    3. Historical prior (20% weight)
    
    Inference time: <15ms per frame
    """
    
    def __init__(self, discriminator_path=None, history_cache_path=None, 
                 weights=(0.5, 0.3, 0.2), use_cnn=False, device='cuda'):
        """
        Args:
            discriminator_path: path to trained discriminator model
            history_cache_path: path to history cache pickle
            weights: (w_heuristic, w_discriminator, w_history)
            use_cnn: if True, use CNN features (523D), else use simple features (5D)
            device: 'cuda' or 'cpu' for CNN feature extraction
        """
        self.use_cnn = use_cnn
        self.device = device
        
        # Feature extractor
        if use_cnn:
            print("Using CNNFeatureExtractor (523D features)")
            self.feature_extractor = CNNFeatureExtractor(device=device)
            self.feature_dim = 523
        else:
            print("Using SimpleFeatureExtractor (5D features)")
            self.feature_extractor = SimpleFeatureExtractor()
            self.feature_dim = 5
        
        # Component 1: Discriminator (optional)
        self.use_discriminator = False
        if discriminator_path:
            try:
                # Auto-detect model type based on feature dimension
                if use_cnn:
                    print("Loading DistanceDiscriminatorWithAttention")
                    self.discriminator = DistanceDiscriminatorWithAttention(
                        input_dim=523,
                        hidden_dims=[256, 128, 64],
                        num_heads=4,
                        dropout=0.2
                    )
                else:
                    print("Loading LightweightDistanceDiscriminator")
                    self.discriminator = LightweightDistanceDiscriminator(input_dim=5)
                
                self.discriminator.load_state_dict(torch.load(discriminator_path))
                self.discriminator.eval()
                if torch.cuda.is_available() and device == 'cuda':
                    self.discriminator = self.discriminator.to(device)
                self.use_discriminator = True
                print(f"Loaded discriminator from {discriminator_path}")
            except Exception as e:
                print(f"Warning: Could not load discriminator: {e}")
        
        # Component 3: History cache (optional)
        self.use_history = False
        if history_cache_path:
            try:
                self.history = HistoryCache(history_cache_path)
                self.use_history = True
                print("Loaded history cache from", history_cache_path)
            except Exception as e:
                print(f"Warning: Could not load history cache: {e}")
        
        # Weights
        self.w_heuristic, self.w_disc, self.w_history = weights
        
        # Normalize weights
        total = self.w_heuristic + (self.w_disc if self.use_discriminator else 0) + \
                (self.w_history if self.use_history else 0)
        self.w_heuristic /= total
        if self.use_discriminator:
            self.w_disc /= total
        if self.use_history:
            self.w_history /= total
        
        print(f"Hybrid selector initialized with weights: h={self.w_heuristic:.2f}, "
              f"d={self.w_disc:.2f}, hist={self.w_history:.2f}")
    
    def extract_features(self, image, depth, distance):
        """Extract features (5D or 523D depending on use_cnn)"""
        return self.feature_extractor.extract(image, depth, distance)
    
    def compute_value(self, image, depth, distance, category=None):
        """
        Compute overall value score
        Time: <15ms (5D) or ~50-80ms (523D CNN)
        
        Args:
            image: RGB image array (H, W, 3), normalized [0, 1]
            depth: Depth map array (H, W)
            distance: camera distance in meters
            category: object category (optional, for history prior)
        
        Returns:
            value: float in [0, 1]
        """
        # Extract features (5D or 523D)
        features = self.extract_features(image, depth, distance)
        
        # 1. Discriminator score
        if self.use_discriminator:
            with torch.no_grad():
                feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                if torch.cuda.is_available() and self.device == 'cuda':
                    feat_tensor = feat_tensor.to(self.device)
                disc_score = self.discriminator(feat_tensor).item()
        else:
            disc_score = 0.5
        
        # 2. Simple heuristic score
        # For CNN features (523D): use depth statistics at indices 512, 513
        # For simple features (5D): use clarity, contrast at indices 1, 2
        if self.use_cnn:
            # Extract depth statistics from CNN features
            # features[512:522] are depth statistics, features[522] is distance
            depth_mean = features[512]  # depth mean
            depth_std = features[513]   # depth std
            heuristic_score = depth_mean * depth_std * (1.0 - abs(distance - 3.5) / 3.5)
        else:
            # Use simple features
            clarity, contrast = features[1], features[2]
            dist_penalty = 1.0 - abs(distance - 3.5) / 3.5
            heuristic_score = clarity * contrast * max(dist_penalty, 0.1)
        
        # 3. History prior
        if self.use_history:
            prior_score = self.history.get_prior_score(distance, category)
        else:
            prior_score = 0.5
        
        # Weighted combination
        value = (self.w_disc * disc_score + 
                 self.w_heuristic * heuristic_score + 
                 self.w_history * prior_score)
        
        return value
    
    def select_best_distance(self, images_dict, aff_maps_dict, category=None):
        """
        Select best distance from multiple options
        
        Args:
            images_dict: {distance: image_array}
            aff_maps_dict: {distance: aff_map_array}
            category: object category (optional)
        
        Returns:
            best_distance: float
            all_scores: dict {distance: score}
        """
        scores = {}
        
        for distance in images_dict.keys():
            if distance not in aff_maps_dict:
                continue
            
            image = images_dict[distance]
            aff_map = aff_maps_dict[distance]
            
            score = self.compute_value(image, aff_map, distance, category)
            scores[distance] = score
        
        if not scores:
            return None, {}
        
        best_distance = max(scores.keys(), key=lambda d: scores[d])
        return best_distance, scores
    
    def should_stop_approaching(self, value_history, window=5, threshold=0.8, 
                                 decline_threshold=3):
        """
        Decide whether to stop approaching based on value history
        
        Args:
            value_history: list of recent value scores
            window: number of recent values to consider
            threshold: high value threshold for immediate stopping
            decline_threshold: number of consecutive declines to trigger stop
        
        Returns:
            should_stop: bool
            reason: str (explanation)
        """
        if len(value_history) < window:
            return False, "Not enough history"
        
        recent_values = value_history[-window:]
        
        # Strategy 1: Value peaked and consistently declining
        decline_count = 0
        for i in range(len(recent_values) - 1):
            if recent_values[i+1] < recent_values[i]:
                decline_count += 1
        
        if decline_count >= decline_threshold:
            return True, f"Value declining ({decline_count}/{window-1} steps)"
        
        # Strategy 2: Value above high threshold
        if recent_values[-1] > threshold:
            return True, f"High value reached ({recent_values[-1]:.3f})"
        
        # Strategy 3: Value plateau (variance very low)
        if len(recent_values) >= window:
            variance = np.var(recent_values)
            if variance < 0.001 and recent_values[-1] > 0.5:
                return True, f"Value plateau (var={variance:.4f})"
        
        return False, "Continue approaching"


# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--discriminator', type=str, default=None)
    parser.add_argument('--history_cache', type=str, default=None)
    args = parser.parse_args()
    
    # Initialize selector
    selector = HybridDistanceSelector(
        discriminator_path=args.discriminator,
        history_cache_path=args.history_cache
    )
    
    print("\nHybrid distance selector ready!")
    print("Use selector.compute_value(image, aff_map, distance) to evaluate a distance")
    print("Use selector.select_best_distance(images_dict, aff_maps_dict) to select from multiple options")



