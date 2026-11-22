"""
Benchmark and evaluate distance selector performance.

This script compares different strategies:
  - Random distance
  - Fixed distance heuristics
  - Learned discriminators (5D simple features, 523D CNN, 560D CNN+category)
"""
from argparse import ArgumentParser
import os
import json
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm

# 37 object categories (kept consistent with the original V3 setting)
CATEGORIES = [
    'Bottle', 'Box', 'Cart', 'Dispenser', 'Door', 'Eyeglasses', 'Fan', 
    'Faucet', 'FoldingChair', 'Globe', 'Kettle', 'Keyboard', 'Knife', 
    'KitchenPot', 'Lamp', 'Laptop', 'Lighter', 'Microwave', 'Mouse', 
    'Oven', 'Pen', 'Phone', 'Pliers', 'Printer', 'Refrigerator', 
    'Remote', 'Safe', 'Scissors', 'Stapler', 'Suitcase', 'Switch', 
    'Table', 'Toilet', 'TrashCan', 'USB', 'WashingMachine', 'Window'
]

# Categories in the Test-Seen split (seen during training)
TEST_SEEN_CATEGORIES = {
    'Safe', 'Door', 'Display', 'Refrigerator', 'Laptop', 'Lighter', 
    'Microwave', 'Mouse', 'Box', 'TrashCan', 'KitchenPot', 'Suitcase', 
    'Pliers', 'StorageFurniture', 'Remote', 'Bottle', 'FoldingChair', 
    'Toaster', 'Lamp', 'Dispenser', 'Eyeglasses', 'Pen', 'Printer', 
    'Keyboard', 'Fan', 'Knife', 'Dishwaher'
}

# Category-specific optimal distances (from earlier V3 analysis).
# Format: category -> (optimal_distance, tolerance)
CATEGORY_OPTIMAL_DISTANCES = {
    'Kettle': (3.6, 0.6),
    'Keyboard': (3.5, 0.5),
    'Mouse': (3.0, 0.5),
    'Eyeglasses': (2.5, 0.4),
    'Phone': (2.8, 0.5),
    'Faucet': (2.2, 0.6),
    'Table': (4.0, 0.8),
    'Microwave': (3.8, 0.7),
    'Bottle': (3.2, 0.6),
    'Box': (3.4, 0.6),
    'Door': (4.2, 0.8),
    'Refrigerator': (4.5, 0.9),
    'TrashCan': (3.8, 0.7),
    'WashingMachine': (4.2, 0.8),
}

def extract_category_from_sample_name(sample_name):
    """Extract category from sample folder name."""
    parts = sample_name.split('_')
    if len(parts) >= 2:
        return parts[1]
    return None

def extract_category_from_features(features):
    """Extract category from 560D features (last 37 dims are one-hot)."""
    if len(features) != 560:
        return None
    
    # Last 37 dims are category one-hot: features[523:560]
    category_onehot = features[523:560]
    category_idx = np.argmax(category_onehot)
    
    if category_onehot[category_idx] > 0.5:  # only treat as valid if prob > 0.5
        return CATEGORIES[category_idx]
    return None

def get_category_optimal_distance(category, history_cache_obj=None):
    """Get category-specific optimal distance (prefer history cache when available)."""
    # Prefer to read from history_cache if possible
    if history_cache_obj and category:
        try:
            opt_dist = history_cache_obj.get_optimal_distance(category)
            if opt_dist:
                # Tolerance from history can be tuned per category;
                # for now we use a default.
                tolerance = 0.7  # default tolerance
                return (opt_dist, tolerance)
        except:
            pass
    
    # Fall back to hard-coded category-specific distances
    if category and category in CATEGORY_OPTIMAL_DISTANCES:
        return CATEGORY_OPTIMAL_DISTANCES[category]
    
    # Finally fall back to a global optimal distance (3.5m)
    if history_cache_obj:
        try:
            opt_dist = history_cache_obj.get_optimal_distance(None)  # global optimum
            if opt_dist:
                return (opt_dist, 0.8)
        except:
            pass
    
    return (3.5, 0.8)

# Discriminator model definitions (must match training)
class LightweightDistanceDiscriminator(nn.Module):
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
    """Enhanced discriminator with multi-head attention (CNN-based, 523D)."""
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


class DistanceDiscriminatorWithCategory560(nn.Module):
    """Discriminator with category one-hot features (560D legacy model)."""

    def __init__(self, input_dim=560, hidden_dims=[256, 128, 64],
                 num_heads=4, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Hidden layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        
        # Attention layer (needs 3D input: [batch, seq, features])
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]
        
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.squeeze(1)  # [batch, features]
        
        # Hidden layers
        x = self.hidden_layers(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x

parser = ArgumentParser()
parser.add_argument('--eval_dir', type=str, required=True,
                    help='evaluation results directory')
parser.add_argument('--discriminator', type=str, required=True,
                    help='discriminator model path')
parser.add_argument('--history_cache', type=str, default=None,
                    help='history cache path')
parser.add_argument('--output', type=str, default='benchmark_results.json',
                    help='output file for results')
parser.add_argument('--feature_prefix', type=str, default='features_',
                    help='prefix for feature files (features_ for 5D, features_cnn_ for 523D)')
parser.add_argument('--use_attention', action='store_true',
                    help='use attention-based discriminator model')
parser.add_argument('--test_only', action='store_true',
                    help='only benchmark on test set (requires train/test split)')
args = parser.parse_args()

print("=== Distance Selector Benchmark ===\n")

# Determine input dimension from a sample feature file
sample_folders = [f for f in os.listdir(args.eval_dir) 
                 if os.path.isdir(os.path.join(args.eval_dir, f))]
sample_feature_file = None
for sample_name in sample_folders[:5]:  # Check first 5 samples
    sample_dir = os.path.join(args.eval_dir, sample_name)
    for dist in np.arange(1.0, 6.05, 0.1):
        dist_str = f"{dist:.1f}"
        feature_file = os.path.join(sample_dir, f'{args.feature_prefix}dist{dist_str}.npy')
        if os.path.exists(feature_file):
            sample_feature_file = feature_file
            break
    if sample_feature_file:
        break

if sample_feature_file is None:
    raise RuntimeError(f"No feature files found with prefix '{args.feature_prefix}'")

sample_features = np.load(sample_feature_file)
input_dim = len(sample_features)
print(f"Detected feature dimension: {input_dim}")

# Load discriminator
print("Loading discriminator...")

# Load checkpoint to check model structure
checkpoint = torch.load(args.discriminator, map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Get state_dict keys (convert to list if needed)
state_dict_keys = list(state_dict.keys())

# Detect model type from state_dict keys
if any('input_layer' in k for k in state_dict_keys) and any('hidden_layers' in k for k in state_dict_keys):
    # V3 model structure
    print("Using DistanceDiscriminatorWithCategory560 (CNN+category one-hot, 560D)")
    discriminator = DistanceDiscriminatorWithCategory560(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        num_heads=4,
        dropout=0.2
    )
elif any('input_proj' in k for k in state_dict_keys) and any('mlp' in k for k in state_dict_keys):
    # V2 model structure
    print("Using DistanceDiscriminatorWithAttention (V2)")
    discriminator = DistanceDiscriminatorWithAttention(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        num_heads=4,
        dropout=0.2
    )
elif args.use_attention and input_dim > 100:
    # Default to V2 structure
    print("Using DistanceDiscriminatorWithAttention (default)")
    discriminator = DistanceDiscriminatorWithAttention(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        num_heads=4,
        dropout=0.2
    )
else:
    print("Using LightweightDistanceDiscriminator")
    discriminator = LightweightDistanceDiscriminator(input_dim=input_dim)

# Load state dict
discriminator.load_state_dict(state_dict)
discriminator.eval()
print(f"Loaded discriminator from {args.discriminator}\n")

# Load history cache (optional)
history_cache = None
history_cache_obj = None
if args.history_cache and os.path.exists(args.history_cache):
    from build_history_cache import HistoryCache
    history_cache_obj = HistoryCache(args.history_cache)
    print(f"Loaded history cache from {args.history_cache}\n")

# Load evaluation data
sample_folders = [f for f in os.listdir(args.eval_dir) 
                 if os.path.isdir(os.path.join(args.eval_dir, f))]

# Filter to test set only if requested
if args.test_only:
    from train_distance_discriminator import split_dataset
    _, test_samples = split_dataset(args.eval_dir, train_ratio=0.8, seed=42)
    sample_folders = [s for s in sample_folders if s in test_samples]
    print(f"Test-only mode: benchmarking on {len(sample_folders)} test samples\n")
else:
    print(f"Found {len(sample_folders)} samples to benchmark\n")

# Baseline strategies
class BaselineSelector:
    """Baseline selection strategies"""
    
    @staticmethod
    def random_selector(distances):
        """Random selection"""
        return np.random.choice(distances)
    
    @staticmethod
    def fixed_selector(distances, fixed_dist=3.5):
        """Fixed distance (e.g., 3.5m)"""
        # Find closest available distance
        return min(distances, key=lambda d: abs(d - fixed_dist))
    
    @staticmethod
    def nearest_selector(distances):
        """Always select nearest distance"""
        return min(distances)
    
    @staticmethod
    def farthest_selector(distances):
        """Always select farthest distance"""
        return max(distances)

# Benchmark metrics
results = {
    'discriminator': {'success': [], 'time': [], 'seen': [], 'unseen': []},
    'heuristic': {'success': [], 'time': [], 'seen': [], 'unseen': []},
    'hybrid': {'success': [], 'time': [], 'seen': [], 'unseen': []},
    'random': {'success': [], 'time': [], 'seen': [], 'unseen': []},
    'fixed_3.5m': {'success': [], 'time': [], 'seen': [], 'unseen': []},
    'nearest': {'success': [], 'time': [], 'seen': [], 'unseen': []},
    'oracle': {'success': [], 'time': [], 'seen': [], 'unseen': []}  # Best possible
}

# Benchmark each sample
print("Running benchmark...")
for sample_name in tqdm(sample_folders):
    sample_dir = os.path.join(args.eval_dir, sample_name)
    
    # Extract category to determine if seen or unseen
    category = extract_category_from_sample_name(sample_name)
    is_seen = (category is not None and category in TEST_SEEN_CATEGORIES)
    
    # Collect all available distances, features, and success labels
    available_data = {}
    distance_features = {}
    
    for dist in np.arange(1.0, 6.05, 0.1):
        dist_str = f"{dist:.1f}"
        result_file = os.path.join(sample_dir, f'dist_{dist_str}_result.json')
        feature_file = os.path.join(sample_dir, f'{args.feature_prefix}dist{dist_str}.npy')
        
        if os.path.exists(result_file) and os.path.exists(feature_file):
            # Load success label
            with open(result_file, 'r') as f:
                result = json.load(f)
                success = bool(result.get('mani_succ', False))
            
            # Load features (5D or 523D)
            features = np.load(feature_file)
            
            available_data[dist] = {'success': success}
            distance_features[dist] = features
    
    if len(available_data) < 10:
        continue  # Skip samples with too few distances
    
    distances = list(available_data.keys())
    
    # === Oracle (Best Possible) ===
    oracle_dist = max(available_data.keys(), 
                     key=lambda d: available_data[d]['success'])
    oracle_result = available_data[oracle_dist]
    oracle_success = 1.0 if oracle_result['success'] else 0.0
    results['oracle']['success'].append(oracle_success)
    results['oracle']['time'].append(0.0)
    if is_seen:
        results['oracle']['seen'].append(oracle_success)
    else:
        results['oracle']['unseen'].append(oracle_success)
    
    # === Method 1: Pure Discriminator ===
    start_time = time.time()
    discriminator_scores = {}
    for dist in distances:
        feat = torch.tensor(distance_features[dist]).unsqueeze(0)
        # print("feat: ", feat, "feat.shape: ", feat.shape)
        with torch.no_grad():
            score = discriminator(feat).item()
        discriminator_scores[dist] = score
    
    disc_best_dist = max(discriminator_scores.keys(), 
                         key=lambda d: discriminator_scores[d])
    disc_time = (time.time() - start_time) * 1000  # ms
    
    disc_result = available_data[disc_best_dist]
    disc_success = 1.0 if disc_result['success'] else 0.0
    results['discriminator']['success'].append(disc_success)
    results['discriminator']['time'].append(disc_time)
    if is_seen:
        results['discriminator']['seen'].append(disc_success)
    else:
        results['discriminator']['unseen'].append(disc_success)
    
    # === Method 2: Improved heuristic (category + depth features) ===
    start_time = time.time()
    
    # Extract category information (from sample name or features)
    category = extract_category_from_sample_name(sample_name)
    if category is None and input_dim == 560:
        # Try to extract category from features (first distance feature)
        first_feat = distance_features[list(distances)[0]]
        category = extract_category_from_features(first_feat)
    
    # Get category-specific optimal distance (preferring history cache)
    optimal_dist, tolerance = get_category_optimal_distance(category, history_cache_obj)
    
    heuristic_scores = {}
    for dist in distances:
        feat = distance_features[dist]
        
        if len(feat) > 10:
            # CNN features (523D or 560D): use depth statistics
            # features[512:522] are depth statistics (10 dims)
            # features[522] is the distance feature (for 523D)
            # for 560D: features[512:522] depth stats, features[522] distance, features[523:560] category
            
            if len(feat) >= 523:
                depth_mean = feat[512]  # normalized depth mean
                depth_std = feat[513]   # normalized depth std
                depth_min = feat[514] if len(feat) > 514 else depth_mean
                depth_max = feat[515] if len(feat) > 515 else depth_mean
                
                # Depth-quality score: mean not too close or far; low std (consistent depth).
                # We assume a reasonable range corresponds roughly to 3–5m.
                depth_mean_score = 1.0 - abs(depth_mean - 0.5) * 2.0  # assume 0.5 is optimal depth
                depth_mean_score = np.clip(depth_mean_score, 0.1, 1.0)
                
                # Depth consistency score: smaller std is better (normalized)
                depth_consistency = 1.0 - np.clip(depth_std * 2.0, 0.0, 1.0)
                
                # Depth range score: prefer moderate ranges (not too extreme)
                depth_range = depth_max - depth_min if len(feat) > 515 else depth_std
                depth_range_score = 1.0 - np.clip((depth_range - 0.3) * 2.0, 0.0, 1.0)
                
                depth_score = (depth_mean_score * 0.4 + 
                              depth_consistency * 0.4 + 
                              depth_range_score * 0.2)
            else:
                depth_score = 0.5  # default fallback
            
            # Category-specific distance score (Gaussian around optimal distance)
            dist_score = np.exp(-0.5 * ((dist - optimal_dist) / tolerance) ** 2)
            
            # Final heuristic score
            heuristic_scores[dist] = depth_score * 0.6 + dist_score * 0.4
        else:
            # 5D simple features: clarity * contrast * distance_penalty
            clarity, contrast = feat[1], feat[2]
            # Use category-specific optimal distance
            dist_score = np.exp(-0.5 * ((dist - optimal_dist) / tolerance) ** 2)
            heuristic_scores[dist] = clarity * contrast * dist_score
    
    heur_best_dist = max(heuristic_scores.keys(), 
                         key=lambda d: heuristic_scores[d])
    heur_time = (time.time() - start_time) * 1000  # ms
    
    heur_result = available_data[heur_best_dist]
    heur_success = 1.0 if heur_result['success'] else 0.0
    results['heuristic']['success'].append(heur_success)
    results['heuristic']['time'].append(heur_time)
    if is_seen:
        results['heuristic']['seen'].append(heur_success)
    else:
        results['heuristic']['unseen'].append(heur_success)
    
    # === Method 3: Hybrid (Discriminator + Heuristic + History) ===
    start_time = time.time()
    hybrid_scores = {}
    for dist in distances:
        disc_score = discriminator_scores[dist]
        heur_score = heuristic_scores[dist]
        
        # History prior (if available via HistoryCache)
        if history_cache_obj:
            prior_score = history_cache_obj.get_prior_score(dist, category)
        else:
            prior_score = 0.5
        
        # Improved weighted combination: emphasize heuristic when it is strong;
        # if heuristic and discriminator agree (both high or both low), increase confidence.
        agreement = 1.0 - abs(disc_score - heur_score)
        
        # Dynamic weights: adjust balance between discriminator, heuristic, and prior
        # depending on agreement and heuristic quality.
        w_disc = 0.6
        w_heur = 0.25
        w_prior = 0.15
        
        # If discriminator and heuristic strongly agree, slightly boost their weights.
        if agreement > 0.7:
            w_disc = 0.65
            w_heur = 0.28
            w_prior = 0.07
        
        hybrid_scores[dist] = (w_disc * disc_score + 
                               w_heur * heur_score + 
                               w_prior * prior_score)
    
    hybrid_best_dist = max(hybrid_scores.keys(), 
                           key=lambda d: hybrid_scores[d])
    hybrid_time = (time.time() - start_time) * 1000  # ms
    
    hybrid_result = available_data[hybrid_best_dist]
    hybrid_success = 1.0 if hybrid_result['success'] else 0.0
    results['hybrid']['success'].append(hybrid_success)
    results['hybrid']['time'].append(hybrid_time)
    if is_seen:
        results['hybrid']['seen'].append(hybrid_success)
    else:
        results['hybrid']['unseen'].append(hybrid_success)
    
    # === Baseline 1: Random ===
    random_dist = np.random.choice(distances)
    random_result = available_data[random_dist]
    random_success = 1.0 if random_result['success'] else 0.0
    results['random']['success'].append(random_success)
    results['random']['time'].append(0.1)
    if is_seen:
        results['random']['seen'].append(random_success)
    else:
        results['random']['unseen'].append(random_success)
    
    # === Baseline 2: Fixed 3.5m ===
    fixed_dist = min(distances, key=lambda d: abs(d - 3.5))
    fixed_result = available_data[fixed_dist]
    fixed_success = 1.0 if fixed_result['success'] else 0.0
    results['fixed_3.5m']['success'].append(fixed_success)
    results['fixed_3.5m']['time'].append(0.1)
    if is_seen:
        results['fixed_3.5m']['seen'].append(fixed_success)
    else:
        results['fixed_3.5m']['unseen'].append(fixed_success)
    
    # === Baseline 3: Nearest ===
    nearest_dist = min(distances)
    nearest_result = available_data[nearest_dist]
    nearest_success = 1.0 if nearest_result['success'] else 0.0
    results['nearest']['success'].append(nearest_success)
    results['nearest']['time'].append(0.1)
    if is_seen:
        results['nearest']['seen'].append(nearest_success)
    else:
        results['nearest']['unseen'].append(nearest_success)

# Calculate statistics
print("\n=== Benchmark Results (Overall) ===\n")
print(f"{'Method':<20} {'Success Rate':<15} {'Avg Time (ms)':<15}")
print("-" * 55)

method_order = ['oracle', 'discriminator', 'hybrid', 'heuristic', 'fixed_3.5m', 'nearest', 'random']
for method in method_order:
    if len(results[method]['success']) == 0:
        continue
    
    success_rate = np.mean(results[method]['success'])
    avg_time = np.mean(results[method]['time'])
    
    print(f"{method:<20} {success_rate:<15.4f} {avg_time:<15.2f}")

# Calculate Test Seen and Test Unseen splits
print("\n=== Benchmark Results (Test Seen Split) ===\n")
print(f"{'Method':<20} {'Success Rate':<15} {'Sample Count':<15}")
print("-" * 55)

for method in method_order:
    if len(results[method]['seen']) == 0:
        continue
    
    seen_success_rate = np.mean(results[method]['seen'])
    seen_count = len(results[method]['seen'])
    
    print(f"{method:<20} {seen_success_rate:<15.4f} {seen_count:<15}")

print("\n=== Benchmark Results (Test Unseen Split) ===\n")
print(f"{'Method':<20} {'Success Rate':<15} {'Sample Count':<15}")
print("-" * 55)

for method in method_order:
    if len(results[method]['unseen']) == 0:
        continue
    
    unseen_success_rate = np.mean(results[method]['unseen'])
    unseen_count = len(results[method]['unseen'])
    
    print(f"{method:<20} {unseen_success_rate:<15.4f} {unseen_count:<15}")

# Improvement over baselines
print("\n=== Improvement vs Baselines ===\n")
disc_success = np.mean(results['discriminator']['success'])
hybrid_success = np.mean(results['hybrid']['success'])
random_success = np.mean(results['random']['success'])
fixed_success = np.mean(results['fixed_3.5m']['success'])
oracle_success = np.mean(results['oracle']['success'])

if random_success > 0:
    improvement_random = (disc_success - random_success) / random_success * 100
    print(f"Discriminator vs Random: {improvement_random:+.1f}%")

if fixed_success > 0:
    improvement_fixed = (disc_success - fixed_success) / fixed_success * 100
    print(f"Discriminator vs Fixed:  {improvement_fixed:+.1f}%")

if oracle_success > 0:
    gap_to_oracle = (oracle_success - disc_success) / oracle_success * 100
    print(f"Gap to Oracle:           {gap_to_oracle:.1f}%")

# Save results
with open(args.output, 'w') as f:
    json.dump({
        'results': {k: {
            'success_rate': float(np.mean(v['success'])) if len(v['success']) > 0 else 0.0,
            'avg_time_ms': float(np.mean(v['time'])) if len(v['time']) > 0 else 0.0,
            'std_time_ms': float(np.std(v['time'])) if len(v['time']) > 0 else 0.0,
            'test_seen': {
                'success_rate': float(np.mean(v['seen'])) if len(v['seen']) > 0 else 0.0,
                'sample_count': len(v['seen'])
            },
            'test_unseen': {
                'success_rate': float(np.mean(v['unseen'])) if len(v['unseen']) > 0 else 0.0,
                'sample_count': len(v['unseen'])
            }
        } for k, v in results.items() if len(v['success']) > 0},
        'sample_count': len([s for s in sample_folders if len(results['oracle']['success']) > 0]),
        'total_samples': len(sample_folders)
    }, f, indent=2)

print(f"\nResults saved to {args.output}")



