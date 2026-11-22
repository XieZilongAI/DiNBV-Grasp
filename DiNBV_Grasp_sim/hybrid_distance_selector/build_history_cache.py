"""
Build historical statistics cache from evaluation results
Used as prior knowledge for distance selection
"""
from argparse import ArgumentParser
import os
import json
import numpy as np
import pickle
from collections import defaultdict


class HistoryCache:
    """Helper class to use history cache"""
    
    def __init__(self, cache_file):
        with open(cache_file, 'rb') as f:
            self.cache = pickle.load(f)
    
    def get_prior_score(self, distance, category=None):
        """
        Get prior score for a distance based on historical success rate
        
        Args:
            distance: float, camera distance
            category: str, object category (optional)
        
        Returns:
            prior_score: float in [0, 1]
        """
        # Category-specific prior if available
        if category and category in self.cache['category_optimal']:
            opt_dist = self.cache['category_optimal'][category]['optimal_distance']
            # Gaussian around optimal distance
            score = np.exp(-0.5 * ((distance - opt_dist) / 0.7) ** 2)
            return score
        
        # Global prior
        if distance in self.cache['distance_success_rate']:
            return self.cache['distance_success_rate'][distance]
        
        # Interpolate if exact distance not in cache
        distances = sorted(self.cache['distance_success_rate'].keys())
        if distance < min(distances):
            return self.cache['distance_success_rate'][min(distances)]
        if distance > max(distances):
            return self.cache['distance_success_rate'][max(distances)]
        
        # Linear interpolation
        lower = max([d for d in distances if d <= distance])
        upper = min([d for d in distances if d >= distance])
        
        if lower == upper:
            return self.cache['distance_success_rate'][lower]
        
        weight = (distance - lower) / (upper - lower)
        score = (1 - weight) * self.cache['distance_success_rate'][lower] + \
                weight * self.cache['distance_success_rate'][upper]
        
        return score
    
    def get_optimal_distance(self, category=None):
        """Get optimal distance for a category"""
        if category and category in self.cache['category_optimal']:
            return self.cache['category_optimal'][category]['optimal_distance']
        return self.cache['global_optimal']


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--eval_dir', type=str, required=True, help='evaluation results directory')
    parser.add_argument('--output', type=str, default='history_cache.pkl', help='output cache file')
    args = parser.parse_args()

    print("Building history cache from evaluation results...")

    # Collect statistics
    stats = defaultdict(lambda: defaultdict(list))

    sample_folders = [f for f in os.listdir(args.eval_dir) if os.path.isdir(os.path.join(args.eval_dir, f))]

    for sample_name in sample_folders:
        sample_dir = os.path.join(args.eval_dir, sample_name)
        
        # Extract category from sample name
        parts = sample_name.split('_')
        category = parts[1] if len(parts) > 1 else 'unknown'
        
        # Estimate object size from sample
        # We'll use the affordance map at a reference distance (3.0m)
        ref_dist = 3.0
        dist_str = f"{ref_dist:.1f}"
        
        # Collect success/failure for each distance
        for dist in np.arange(1.0, 6.05, 0.1):
            dist_str = f"{dist:.1f}"
            result_file = os.path.join(sample_dir, f'dist_{dist_str}_result.json')
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    success = bool(result.get('mani_succ', False))
                    
                    # Record: category → distance → success rate
                    stats[category][dist].append(1.0 if success else 0.0)
                    stats['global'][dist].append(1.0 if success else 0.0)

    # Build cache
    cache = {
        'category_optimal': {},  # category → optimal_distance
        'global_optimal': None,  # global optimal distance
        'distance_success_rate': {}  # distance → success_rate
    }

    # Global statistics
    print("\n=== Global Statistics ===")
    global_optimal_dist = None
    global_optimal_rate = 0

    for dist in sorted(stats['global'].keys()):
        successes = stats['global'][dist]
        success_rate = np.mean(successes)
        cache['distance_success_rate'][float(dist)] = float(success_rate)
        
        if success_rate > global_optimal_rate:
            global_optimal_rate = success_rate
            global_optimal_dist = dist
        
        print(f"Distance {dist:.1f}m: {success_rate:.3f} ({len(successes)} samples)")

    cache['global_optimal'] = float(global_optimal_dist) if global_optimal_dist else 3.5
    print(f"\nGlobal optimal distance: {cache['global_optimal']:.1f}m (rate: {global_optimal_rate:.3f})")

    # Category-specific statistics
    print("\n=== Category-Specific Optimal Distances ===")
    for category in stats.keys():
        if category == 'global':
            continue
        
        cat_stats = stats[category]
        
        # Find optimal distance for this category
        cat_optimal_dist = None
        cat_optimal_rate = 0
        
        for dist in cat_stats.keys():
            successes = cat_stats[dist]
            if len(successes) < 5:  # Skip if too few samples
                continue
            
            success_rate = np.mean(successes)
            if success_rate > cat_optimal_rate:
                cat_optimal_rate = success_rate
                cat_optimal_dist = dist
        
        if cat_optimal_dist is not None:
            cache['category_optimal'][category] = {
                'optimal_distance': float(cat_optimal_dist),
                'success_rate': float(cat_optimal_rate),
                'sample_count': sum(len(cat_stats[d]) for d in cat_stats.keys())
            }
            print(f"{category}: {cat_optimal_dist:.1f}m ({cat_optimal_rate:.3f}, n={cache['category_optimal'][category]['sample_count']})")

    # Save cache
    with open(args.output, 'wb') as f:
        pickle.dump(cache, f)

    print(f"\nHistory cache saved to {args.output}")
    print(f"Total categories: {len(cache['category_optimal'])}")

