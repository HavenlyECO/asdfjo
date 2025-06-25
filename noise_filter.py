"""
Card detection noise filter
Cleans up false positives from card detection results
Uses domain-specific knowledge to filter out UI elements and non-card regions
"""

import os
import cv2
import numpy as np
import logging
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("noise_filter")

class NoiseFilter:
    """
    Specialized filter to remove non-card detections from model output
    Uses domain-specific knowledge about poker UI and card characteristics
    """
    
    def __init__(self):
        """Initialize the noise filter"""
        # Define the characteristics of real playing cards
        self.card_constraints = {
            'min_white_ratio': 0.25,      # Cards typically have significant white areas
            'max_saturation': 0.5,        # Cards typically have low saturation in most of the area
            'min_edge_density': 0.05,     # Cards have clear edges
            'max_edge_density': 0.3,      # But not too noisy
            'min_corner_contrast': 30,    # Cards have contrast between rank/suit and background
            'min_aspect_ratio': 0.6,      # Height divided by width (portrait orientation)
            'max_aspect_ratio': 1.8       # Not too narrow
        }
        
        # Define known UI regions to exclude
        self.ui_regions = [
            # Format: [name, [x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio]]
            ["chip_stacks", [0.4, 0.4, 0.6, 0.5]],
            ["pot_display", [0.4, 0.3, 0.6, 0.36]],
            ["table_border", [0.0, 0.0, 1.0, 0.1]],
            ["table_border", [0.0, 0.9, 1.0, 1.0]],
            ["player_ui", [0.0, 0.75, 0.3, 1.0]],
            ["chat_ui", [0.7, 0.75, 1.0, 1.0]]
        ]
        
        # Define expected card locations
        self.card_locations = [
            # Format: [name, [x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio], expected_count]
            ["community_cards", [0.3, 0.4, 0.7, 0.6], 5],
            ["hero_cards", [0.3, 0.7, 0.7, 0.9], 2],
            ["opponent_cards", [0.3, 0.1, 0.7, 0.3], 2]
        ]
        
    def filter_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out noise detections from model output
        
        Args:
            image: Original image
            detections: List of detection dictionaries with 'bbox', 'confidence', etc.
            
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []
            
        img_height, img_width = image.shape[:2]
        filtered_detections = []
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda d: d.get('confidence', 0), reverse=True)
        
        # Filter based on multiple criteria
        for detection in sorted_detections:
            # Extract bbox coordinates
            if 'bbox' not in detection:
                continue
                
            x, y, w, h = detection['bbox']
            
            # Convert to pixels if normalized
            if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                x = int(x * img_width)
                y = int(y * img_height)
                w = int(w * img_width)
                h = int(h * img_height)
            
            # Ensure coordinates are valid
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > img_width or y + h > img_height:
                continue
                
            # Extract the region
            region = image[int(y):int(y+h), int(x):int(x+w)]
            if region.size == 0:
                continue
                
            # Check if this region is in a UI area
            if self._is_in_ui_region(x, y, w, h, img_width, img_height):
                continue
                
            # Apply card-specific validation
            if self._is_valid_card(region):
                # Keep this detection
                filtered_detections.append(detection)
                
        # Check against expected card locations and counts
        filtered_detections = self._filter_by_card_locations(filtered_detections, img_width, img_height)
        
        return filtered_detections
    
    def _is_in_ui_region(self, x: float, y: float, w: float, h: float, 
                       img_width: int, img_height: int) -> bool:
        """Check if a detection falls within a known UI region"""
        # Calculate center point
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Convert to ratios
        center_x_ratio = center_x / img_width
        center_y_ratio = center_y / img_height
        
        # Check against each UI region
        for name, region in self.ui_regions:
            x_min, y_min, x_max, y_max = region
            
            if (x_min <= center_x_ratio <= x_max and 
                y_min <= center_y_ratio <= y_max):
                return True
                
        return False
    
    def _is_valid_card(self, card_img: np.ndarray) -> bool:
        """Check if an image region has characteristics of a playing card"""
        try:
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            
            # Calculate aspect ratio
            h, w = card_img.shape[:2]
            aspect_ratio = h / w if w > 0 else 0
            
            if not (self.card_constraints['min_aspect_ratio'] <= aspect_ratio <= 
                  self.card_constraints['max_aspect_ratio']):
                return False
            
            # Check for white content (cards have significant white areas)
            white_pixels = np.sum((hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 200))
            white_ratio = white_pixels / (h * w) if h * w > 0 else 0
            
            if white_ratio < self.card_constraints['min_white_ratio']:
                return False
                
            # Check saturation (playing cards are not highly saturated)
            mean_saturation = np.mean(hsv[:, :, 1]) / 255.0
            
            if mean_saturation > self.card_constraints['max_saturation']:
                return False
                
            # Check edge characteristics (cards have clear defined edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (h * w) if h * w > 0 else 0
            
            if not (self.card_constraints['min_edge_density'] <= edge_density <= 
                  self.card_constraints['max_edge_density']):
                return False
                
            # Check corner for rank/suit indicators (cards have text/symbols in corners)
            corner_size = min(h, w) // 4
            if corner_size > 0:
                top_left = gray[:corner_size, :corner_size]
                corner_std = np.std(top_left)
                
                if corner_std < self.card_constraints['min_corner_contrast']:
                    return False
            
            # Passed all tests
            return True
            
        except Exception as e:
            log.warning(f"Error in card validation: {e}")
            return False
    
    def _filter_by_card_locations(self, detections: List[Dict[str, Any]], 
                               img_width: int, img_height: int) -> List[Dict[str, Any]]:
        """Filter based on expected card locations"""
        if not detections:
            return []
            
        # Group detections by card location
        location_detections = {loc[0]: [] for loc in self.card_locations}
        unassigned = []
        
        for det in detections:
            x, y, w, h = det['bbox']
            center_x = x + w / 2
            center_y = y + h / 2
            center_x_ratio = center_x / img_width
            center_y_ratio = center_y / img_height
            
            assigned = False
            for name, region, _ in self.card_locations:
                x_min, y_min, x_max, y_max = region
                if (x_min <= center_x_ratio <= x_max and 
                    y_min <= center_y_ratio <= y_max):
                    location_detections[name].append(det)
                    assigned = True
                    break
                    
            if not assigned:
                unassigned.append(det)
        
        # Filter by expected counts and confidence
        result = []
        for name, _, expected_count in self.card_locations:
            if location_detections[name]:
                # Sort by confidence
                sorted_dets = sorted(location_detections[name], 
                                   key=lambda d: d.get('confidence', 0), reverse=True)
                # Take top N detections
                result.extend(sorted_dets[:expected_count])
        
        # Add any unassigned detections that are very confident
        for det in unassigned:
            if det.get('confidence', 0) > 0.9:  # Very high confidence threshold
                result.append(det)
        
        return result
    
    def filter_model_predictions(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        Filter a trained model to remove noise predictions
        
        Args:
            model_path: Path to the model file
            output_path: Path to save the filtered model (optional)
            
        Returns:
            Path to the filtered model
        """
        # Create default output path if not provided
        if output_path is None:
            base_path = Path(model_path)
            output_path = str(base_path.parent / f"{base_path.stem}_filtered{base_path.suffix}")
        
        try:
            # Load the model
            log.info(f"Loading model from {model_path}")
            
            # First, add PyTorch safeguards for loading
            self._add_torch_serialization_safeguard()
            
            # Load the model
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            
            # Set confidence threshold higher
            model.conf = 0.4  # Minimum confidence threshold
            model.iou = 0.45  # IoU threshold
            
            # Set max detections per image
            model.max_det = 9  # Maximum number of cards in a poker game
            
            # Save the filtered model
            log.info(f"Saving filtered model to {output_path}")
            torch.save(model.state_dict(), output_path)
            
            return output_path
        except Exception as e:
            log.error(f"Error filtering model: {e}")
            return model_path
            
    def _add_torch_serialization_safeguard(self):
        """Add PyTorch serialization safeguards to fix weights loading issue"""
        try:
            # Fix for the PyTorch 2.6+ weights_only issue
            import torch.serialization
            
            # Explicitly add ultralytics classes to the safe globals
            torch.serialization.add_safe_globals([
                'ultralytics.nn.tasks.DetectionModel',
                'ultralytics.nn.modules',
                'ultralytics.nn.tasks',
                'ultralytics.yolo.utils',
                'ultralytics.yolo.v8',
                'ultralytics.engine.model'
            ])
            
            # Set environment variable to use old loading behavior if needed
            os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
            
            log.info("Added PyTorch serialization safeguards")
        except Exception as e:
            log.warning(f"Could not add PyTorch serialization fixes: {e}")
    
    def process_test_image(self, image_path: str, model_path: str, output_dir: str) -> str:
        """
        Process a test image with the model and filter noise detections
        
        Args:
            image_path: Path to the test image
            model_path: Path to the model file
            output_dir: Directory to save output visualization
            
        Returns:
            Path to the output visualization
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load image
            log.info(f"Loading test image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                log.error(f"Could not read image: {image_path}")
                return ""
                
            # Load model
            self._add_torch_serialization_safeguard()
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            
            # Run inference
            log.info("Running inference")
            results = model(image)
            
            # Convert results to our format
            detections = []
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                detections.append({
                    'bbox': (float(x1), float(y1), float(w), float(h)),
                    'confidence': float(conf),
                    'class': int(cls),
                    'name': results.names[int(cls)]
                })
            
            # Draw original detections
            original_viz = image.copy()
            for det in detections:
                x, y, w, h = map(int, det['bbox'])
                conf = det['confidence']
                label = f"{det.get('name', 'card')}: {conf:.2f}"
                
                cv2.rectangle(original_viz, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(original_viz, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save original visualization
            original_path = os.path.join(output_dir, f"{Path(image_path).stem}_original.jpg")
            cv2.imwrite(original_path, original_viz)
            
            # Filter detections
            filtered_detections = self.filter_detections(image, detections)
            
            # Draw filtered detections
            filtered_viz = image.copy()
            for det in filtered_detections:
                x, y, w, h = map(int, det['bbox'])
                conf = det['confidence']
                label = f"{det.get('name', 'card')}: {conf:.2f}"
                
                cv2.rectangle(filtered_viz, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(filtered_viz, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Save filtered visualization
            filtered_path = os.path.join(output_dir, f"{Path(image_path).stem}_filtered.jpg")
            cv2.imwrite(filtered_path, filtered_viz)
            
            # Create side-by-side comparison
            h, w = image.shape[:2]
            comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
            comparison[:, :w] = original_viz
            comparison[:, w:] = filtered_viz
            
            # Add labels
            cv2.putText(comparison, "Original Detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Filtered Detections", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save comparison
            comparison_path = os.path.join(output_dir, f"{Path(image_path).stem}_comparison.jpg")
            cv2.imwrite(comparison_path, comparison)
            
            log.info(f"Original detections: {len(detections)}, Filtered: {len(filtered_detections)}")
            
            return comparison_path
            
        except Exception as e:
            log.error(f"Error processing test image: {e}")
            return ""

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Filter noise from card detection model")
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--output", help="Path to save the filtered model")
    parser.add_argument("--test-image", help="Path to a test image for visualization")
    parser.add_argument("--output-dir", default="filtered_results", help="Directory for output files")
    
    args = parser.parse_args()
    
    # Create noise filter
    noise_filter = NoiseFilter()
    
    # Process model
    filtered_model_path = noise_filter.filter_model_predictions(args.model, args.output)
    log.info(f"Filtered model saved to: {filtered_model_path}")
    
    # Process test image if provided
    if args.test_image:
        comparison_path = noise_filter.process_test_image(
            args.test_image, filtered_model_path, args.output_dir
        )
        if comparison_path:
            log.info(f"Comparison visualization saved to: {comparison_path}")

if __name__ == "__main__":
    main()