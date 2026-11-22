"""
Heuristic distance selector based on object size and affordance quality
Pure rule-based, no training required, <10ms inference
"""
import numpy as np
import cv2

class HeuristicDistanceSelector:
    """
    Pure heuristic method for distance selection
    Based on:
    1. Object size → optimal distance mapping
    2. Affordance quality (peak, concentration, coverage)
    3. Image clarity
    """
    
    def __init__(self):
        # Object size → optimal distance mapping (heuristic rules)
        # Format: (min_ratio, max_ratio) → (optimal_dist, tolerance)
        self.size_distance_map = [
            (0.0, 0.05, 2.0, 0.5),     # Very small objects: 2.0±0.5m
            (0.05, 0.10, 2.8, 0.6),    # Small objects: 2.8±0.6m
            (0.10, 0.15, 3.5, 0.7),    # Medium objects: 3.5±0.7m
            (0.15, 0.25, 4.5, 0.8),    # Large objects: 4.5±0.8m
            (0.25, 1.0, 5.0, 1.0),     # Very large objects: 5.0±1.0m
        ]
    
    def estimate_object_size(self, aff_map, threshold=0.3):
        """
        Estimate object size from affordance map
        Returns: ratio of object pixels to total pixels
        """
        obj_pixels = (aff_map > threshold).sum()
        ratio = obj_pixels / aff_map.size
        return ratio
    
    def get_optimal_distance_for_size(self, obj_size):
        """
        Get optimal distance based on object size
        Returns: (optimal_distance, tolerance)
        """
        for min_ratio, max_ratio, opt_dist, tolerance in self.size_distance_map:
            if min_ratio <= obj_size < max_ratio:
                return opt_dist, tolerance
        
        # Default
        return 3.5, 1.0
    
    def compute_distance_score(self, current_dist, optimal_dist, tolerance):
        """
        Compute distance score using Gaussian function
        Score is highest at optimal_dist, decreases with distance
        """
        score = np.exp(-0.5 * ((current_dist - optimal_dist) / tolerance) ** 2)
        return score
    
    def compute_affordance_quality(self, aff_map):
        """
        Compute affordance map quality score
        Returns: quality score in [0, 1]
        """
        # 1. Peak confidence
        peak = float(aff_map.max())
        
        # 2. Concentration (how focused is the affordance)
        positive_aff = aff_map[aff_map > 0.3]
        if len(positive_aff) > 0:
            # Higher concentration if peak is much larger than 90th percentile
            p90 = np.percentile(positive_aff, 90)
            concentration = peak / (p90 + 1e-6)
            concentration = min(concentration, 2.0) / 2.0  # Normalize to [0, 1]
        else:
            concentration = 0.0
        
        # 3. Coverage (not too small, not too large)
        coverage_ratio = (aff_map > 0.3).sum() / aff_map.size
        if coverage_ratio < 0.01:
            coverage_score = coverage_ratio * 100  # Penalize too small
        elif coverage_ratio > 0.3:
            coverage_score = 0.3 / coverage_ratio  # Penalize too large
        else:
            coverage_score = 1.0
        
        # Combine
        quality = peak * 0.5 + concentration * 0.3 + coverage_score * 0.2
        return quality
    
    def compute_image_clarity(self, image):
        """
        Compute image sharpness using Laplacian variance
        Returns: clarity score in [0, 1]
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize (typical range: 0-1000)
        clarity = min(laplacian_var / 1000.0, 1.0)
        return clarity
    
    def compute_value(self, image, aff_map, distance):
        """
        Compute overall value score for this distance
        Returns: value in [0, 1]
        Time: <10ms
        """
        # 1. Estimate object size
        obj_size = self.estimate_object_size(aff_map)
        
        # 2. Get optimal distance for this size
        optimal_dist, tolerance = self.get_optimal_distance_for_size(obj_size)
        
        # 3. Distance score
        distance_score = self.compute_distance_score(distance, optimal_dist, tolerance)
        
        # 4. Affordance quality
        quality_score = self.compute_affordance_quality(aff_map)
        
        # 5. Image clarity
        clarity_score = self.compute_image_clarity(image)
        
        # 6. Combine scores (weighted product)
        final_value = (
            distance_score ** 0.4 *    # 40% weight on distance
            quality_score ** 0.4 *     # 40% weight on affordance quality
            clarity_score ** 0.2       # 20% weight on clarity
        )
        
        return final_value
    
    def select_best_distance(self, images_dict, aff_maps_dict):
        """
        Select best distance from a dictionary of images and affordance maps
        
        Args:
            images_dict: {distance: image_array}
            aff_maps_dict: {distance: aff_map_array}
        
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
            
            score = self.compute_value(image, aff_map, distance)
            scores[distance] = score
        
        if not scores:
            return None, {}
        
        best_distance = max(scores, key=scores.get)
        return best_distance, scores
    
    def should_stop_approaching(self, value_history, window=5, threshold=0.8):
        """
        Decide if robot should stop approaching based on value history
        
        Args:
            value_history: list of recent value scores
            window: number of recent values to consider
            threshold: high value threshold for immediate stopping
        
        Returns:
            should_stop: bool
        """
        if len(value_history) < window:
            return False
        
        recent_values = value_history[-window:]
        
        # Strategy 1: Value peaked and declining
        if len(recent_values) >= 3:
            if recent_values[-1] < recent_values[-2] < recent_values[-3]:
                return True
        
        # Strategy 2: Value above threshold
        if recent_values[-1] > threshold:
            return True
        
        return False



