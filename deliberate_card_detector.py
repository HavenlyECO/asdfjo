"""
Deliberate card detection module with careful, methodical approach
Implements a human-like analysis strategy with patience and thoroughness
"""

import cv2
import numpy as np
import logging
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Configure module logging
log = logging.getLogger("deliberate_detector")

# Card definitions
SUITS = ['c', 'd', 'h', 's']  # clubs, diamonds, hearts, spades
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
CARD_CLASSES = [f"{rank}{suit}" for rank in RANKS for suit in SUITS]

class DeliberateCardDetector:
    """
    Card detector that carefully and methodically analyzes each frame
    Takes time to explore multiple detection strategies like a human would
    """
    
    def __init__(self, debug_dir="deliberate_debug"):
        """Initialize the deliberate detector with careful analysis parameters"""
        # Configure debug output
        self.debug_dir = debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Processing parameters
        self.min_processing_time = 5.0  # Minimum seconds to spend analyzing a frame
        self.max_attempts = 5  # Number of detection strategies to try
        
        # Card location contexts in poker games
        self.contexts = [
            {
                "name": "hero_cards",  # Player's own cards (bottom of screen)
                "region": (0.3, 0.7, 0.7, 1.0),  # (x_min, x_max, y_min, y_max) as ratios
                "expected_count": 2,
                "min_area_ratio": 0.005,
                "max_area_ratio": 0.05
            },
            {
                "name": "community_cards",  # Cards in the middle of the table
                "region": (0.2, 0.8, 0.35, 0.65),
                "expected_count": 5,
                "min_area_ratio": 0.003,
                "max_area_ratio": 0.035
            },
            {
                "name": "opponent_cards",  # Opponent's cards (top of screen)
                "region": (0.3, 0.7, 0.0, 0.35),
                "expected_count": 2,
                "min_area_ratio": 0.003,
                "max_area_ratio": 0.04
            },
            {
                "name": "full_table",  # All visible cards
                "region": (0.0, 1.0, 0.0, 1.0),
                "expected_count": 9,  # Maximum cards visible in poker (2 + 2 + 5)
                "min_area_ratio": 0.002,
                "max_area_ratio": 0.05
            }
        ]
        
        # Detection strategies in order of increasing complexity
        self.strategies = [
            self.detect_cards_basic,
            self.detect_cards_adaptive,
            self.detect_cards_color_filtered,
            self.detect_cards_multiscale,
            self.detect_cards_comprehensive
        ]
        
        log.info(f"Initialized deliberate detector with {len(self.strategies)} strategies")
    
    def detect_all_cards(self, image: np.ndarray, frame_id: str = None) -> List[Dict[str, Any]]:
        """
        Carefully analyze an image to find all cards in different contexts
        Takes time to ensure thorough analysis like a human would
        
        Args:
            image: Input image
            frame_id: Identifier for the frame (for debugging)
            
        Returns:
            List of detected card regions with context information
        """
        if image is None or image.size == 0:
            return []
            
        # Start timing
        start_time = time.time()
        
        # Create a unique identifier for this analysis session
        if frame_id is None:
            frame_id = f"frame_{int(start_time)}"
        
        # Create debug directory for this frame
        frame_debug_dir = os.path.join(self.debug_dir, frame_id)
        os.makedirs(frame_debug_dir, exist_ok=True)
        
        # Save original image for reference
        cv2.imwrite(os.path.join(frame_debug_dir, "original.jpg"), image)
        
        # Initialize results
        all_detected_cards = []
        best_strategy_results = []
        best_strategy_name = "none"
        best_strategy_count = 0
        
        # Try each context separately for targeted detection
        for context in self.contexts:
            context_name = context["name"]
            region = context["region"]
            expected_count = context["expected_count"]
            
            # Extract region coordinates
            x_min_ratio, x_max_ratio, y_min_ratio, y_max_ratio = region
            height, width = image.shape[:2]
            x_min = int(width * x_min_ratio)
            x_max = int(width * x_max_ratio)
            y_min = int(height * y_min_ratio)
            y_max = int(height * y_max_ratio)
            
            # Extract region and ensure valid
            if x_min >= x_max or y_min >= y_max:
                continue
                
            region_img = image[y_min:y_max, x_min:x_max]
            if region_img.size == 0:
                continue
            
            # Save region for debugging
            cv2.imwrite(os.path.join(frame_debug_dir, f"region_{context_name}.jpg"), region_img)
            
            log.info(f"Analyzing {context_name} region - expecting up to {expected_count} cards")
            
            # Try different strategies for this context
            best_context_results = []
            
            # Track the best strategy for this context
            best_context_strategy = None
            best_context_count = 0
            
            for i, strategy in enumerate(self.strategies):
                # Apply this detection strategy
                cards = strategy(region_img, context, os.path.join(frame_debug_dir, f"strategy_{i+1}_{context_name}"))
                
                # Adjust card coordinates to original image
                for card in cards:
                    card_x, card_y, card_w, card_h = card["bbox"]
                    card["bbox"] = (card_x + x_min, card_y + y_min, card_w, card_h)
                    card["context"] = context_name
                
                # Save results from this strategy
                cv2.imwrite(os.path.join(frame_debug_dir, f"results_{context_name}_strategy_{i+1}.jpg"), 
                           self._draw_detections(region_img, cards, adjust_coords=False))
                
                # Update best strategy for this context
                if len(cards) > best_context_count and len(cards) <= expected_count:
                    best_context_count = len(cards)
                    best_context_strategy = i + 1
                    best_context_results = cards
                
                log.info(f"Strategy {i+1} for {context_name}: found {len(cards)} cards")
                
                # If we found the expected number, no need to try more strategies
                if len(cards) == expected_count:
                    break
            
            # Track the overall best strategy
            if best_context_count > best_strategy_count:
                best_strategy_count = best_context_count
                best_strategy_name = f"{context_name}_strategy_{best_context_strategy}"
                best_strategy_results = best_context_results
            
            # Add best results from this context to overall results
            all_detected_cards.extend(best_context_results)
        
        # Create final debug image with all detections
        final_debug_img = self._draw_detections(image, all_detected_cards)
        cv2.imwrite(os.path.join(frame_debug_dir, "final_detections.jpg"), final_debug_img)
        
        # Ensure we've spent enough time on this frame (minimum processing time)
        elapsed_time = time.time() - start_time
        remaining_time = max(0, self.min_processing_time - elapsed_time)
        
        if remaining_time > 0:
            log.info(f"Waiting {remaining_time:.2f} seconds to ensure deliberate analysis")
            time.sleep(remaining_time)
        
        # Log final results
        total_time = time.time() - start_time
        log.info(f"Completed analysis in {total_time:.2f} seconds, found {len(all_detected_cards)} cards "
               f"(best: {best_strategy_name} with {best_strategy_count})")
        
        return all_detected_cards
    
    def detect_cards_basic(self, image: np.ndarray, context: Dict[str, Any], debug_prefix: str) -> List[Dict[str, Any]]:
        """Basic card detection using contour analysis"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binary threshold
        _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
        
        # Save debug images
        cv2.imwrite(f"{debug_prefix}_thresh.jpg", thresh)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours to find cards
        cards = self._process_contours(contours, image.shape, context)
        
        return cards
    
    def detect_cards_adaptive(self, image: np.ndarray, context: Dict[str, Any], debug_prefix: str) -> List[Dict[str, Any]]:
        """Card detection using adaptive thresholding"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Save debug images
        cv2.imwrite(f"{debug_prefix}_thresh.jpg", thresh)
        cv2.imwrite(f"{debug_prefix}_opening.jpg", opening)
        
        # Find contours
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours to find cards
        cards = self._process_contours(contours, image.shape, context)
        
        return cards
    
    def detect_cards_color_filtered(self, image: np.ndarray, context: Dict[str, Any], debug_prefix: str) -> List[Dict[str, Any]]:
        """Card detection using color filtering to isolate cards"""
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Playing cards are typically white/light colored
        # Define range for white/light colors
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 30, 255])
        
        # Create mask for white/light regions
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Clean up the mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        # Save debug images
        cv2.imwrite(f"{debug_prefix}_mask.jpg", white_mask)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours to find cards
        cards = self._process_contours(contours, image.shape, context)
        
        return cards
    
    def detect_cards_multiscale(self, image: np.ndarray, context: Dict[str, Any], debug_prefix: str) -> List[Dict[str, Any]]:
        """Card detection with multiple scale analysis"""
        all_cards = []
        
        # Generate multiple scales
        scales = [0.5, 0.75, 1.0, 1.5]
        
        for i, scale in enumerate(scales):
            # Skip invalid scales
            if scale <= 0:
                continue
                
            # Resize image for this scale
            if scale == 1.0:
                scaled_img = image.copy()
            else:
                height, width = image.shape[:2]
                scaled_width = int(width * scale)
                scaled_height = int(height * scale)
                scaled_img = cv2.resize(image, (scaled_width, scaled_height))
            
            # Process this scale with a basic method
            gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection for this scale
            edges = cv2.Canny(blurred, 50, 150)
            
            # Save debug image
            cv2.imwrite(f"{debug_prefix}_scale_{i+1}_edges.jpg", edges)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            cards = self._process_contours(contours, scaled_img.shape, context)
            
            # Adjust coordinates back to original scale
            if scale != 1.0:
                for card in cards:
                    x, y, w, h = card["bbox"]
                    card["bbox"] = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
            
            # Add to overall results
            all_cards.extend(cards)
        
        # Remove overlapping detections
        filtered_cards = self._filter_overlaps(all_cards)
        
        return filtered_cards
    
    def detect_cards_comprehensive(self, image: np.ndarray, context: Dict[str, Any], debug_prefix: str) -> List[Dict[str, Any]]:
        """Comprehensive card detection combining multiple techniques"""
        # Create a copy to work with
        working_img = image.copy()
        
        # Enhance contrast
        lab = cv2.cvtColor(working_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        enhanced_lab = cv2.merge((enhanced_l, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple thresholding techniques
        _, binary_thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)
        otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Combine thresholds
        combined_thresh = cv2.bitwise_or(binary_thresh, adaptive_thresh)
        combined_thresh = cv2.bitwise_or(combined_thresh, otsu_thresh)
        
        # Edge detection
        edges = cv2.Canny(gray, 30, 150)
        
        # Combine edges and threshold
        combined = cv2.bitwise_or(combined_thresh, edges)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Save debug images
        cv2.imwrite(f"{debug_prefix}_enhanced.jpg", enhanced)
        cv2.imwrite(f"{debug_prefix}_combined.jpg", combined)
        cv2.imwrite(f"{debug_prefix}_cleaned.jpg", cleaned)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours
        cards = self._process_contours(contours, image.shape, context)
        
        # Color-based validation
        validated_cards = []
        
        for card in cards:
            x, y, w, h = map(int, card["bbox"])
            
            # Ensure coordinates are within image bounds
            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                continue
                
            card_img = image[y:y+h, x:x+w]
            
            # Check if this looks like a card (typically has significant white content)
            hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            white_pixels = np.sum((hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 150))
            white_ratio = white_pixels / (w * h) if w * h > 0 else 0
            
            # Validate white ratio for cards
            if white_ratio > 0.3:
                validated_cards.append(card)
        
        return validated_cards
    
    def _process_contours(self, contours: List[np.ndarray], img_shape: Tuple[int, int, int], 
                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process contours to identify cards"""
        # Extract image dimensions
        img_height, img_width = img_shape[:2]
        img_area = img_height * img_width
        
        # Extract area constraints from context
        min_area_ratio = context.get("min_area_ratio", 0.005)
        max_area_ratio = context.get("max_area_ratio", 0.05)
        
        # Calculate absolute area constraints
        min_area = img_area * min_area_ratio
        max_area = img_area * max_area_ratio
        
        cards = []
        
        for contour in contours:
            # Skip small contours
            if len(contour) < 4:
                continue
                
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
                
            # Find approximated polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # Cards typically have 4 corners, but we'll allow some flexibility
            if len(approx) < 3 or len(approx) > 6:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip excessively small rectangles
            if w < 10 or h < 15:
                continue
                
            # Check aspect ratio - cards are typically taller than wide
            aspect_ratio = h / w if w > 0 else 0
            if not (1.0 <= aspect_ratio <= 2.0):
                continue
                
            # This looks like a card
            cards.append({
                "bbox": (x, y, w, h),
                "area": area,
                "aspect_ratio": aspect_ratio,
                "points": len(approx)
            })
        
        return cards
    
    def _filter_overlaps(self, cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter overlapping card detections"""
        if not cards:
            return []
            
        # Sort by area (largest first)
        sorted_cards = sorted(cards, key=lambda c: c.get("area", 0), reverse=True)
        
        filtered = []
        for card in sorted_cards:
            # Check if this card overlaps significantly with any already filtered card
            should_add = True
            
            for filtered_card in filtered:
                # Calculate overlap
                overlap_ratio = self._calculate_overlap(card["bbox"], filtered_card["bbox"])
                
                # If significant overlap, skip this card
                if overlap_ratio > 0.5:
                    should_add = False
                    break
            
            if should_add:
                filtered.append(card)
        
        return filtered
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], 
                         bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap
        
        # Calculate areas
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Calculate overlap ratio
        smaller_area = min(area1, area2)
        if smaller_area > 0:
            return intersection / smaller_area
        return 0.0
    
    def _draw_detections(self, image: np.ndarray, 
                      detections: List[Dict[str, Any]], 
                      adjust_coords: bool = True) -> np.ndarray:
        """Draw detection boxes on image"""
        # Create a copy to draw on
        result = image.copy()
        
        # Draw each detection
        for i, det in enumerate(detections):
            # Extract coordinates
            x, y, w, h = map(int, det["bbox"])
            
            # Draw rectangle
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add context label if available
            label = det.get("context", f"Card {i+1}")
            
            # Draw label
            cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result

    def _identify_card(self, card_img: np.ndarray) -> Dict[str, Any]:
        """Identify a card using specialized techniques"""
        # Ensure card image is valid
        if card_img is None or card_img.size == 0 or card_img.shape[0] < 20 or card_img.shape[1] < 20:
            return {"card_class": "unknown", "confidence": 0.0, "source": "invalid_image"}
        
        try:
            # Resize card for consistent processing while maintaining aspect ratio
            height, width = card_img.shape[:2]
            target_height = 280
            scale_factor = target_height / height
            target_width = int(width * scale_factor)
            card_img_resized = cv2.resize(card_img, (target_width, target_height))
            
            # Extract top-left corner where rank and suit are typically found
            corner_height = int(target_height * 0.25)  # 25% of card height
            corner_width = int(target_width * 0.25)   # 25% of card width
            
            # Ensure corner region is not empty
            corner_height = max(10, corner_height)
            corner_width = max(10, corner_width)
            
            # Make sure corner region stays within image bounds
            corner_height = min(corner_height, card_img_resized.shape[0] - 1)
            corner_width = min(corner_width, card_img_resized.shape[1] - 1)
            
            corner_img = card_img_resized[0:corner_height, 0:corner_width]
            
            # Identify suit using color analysis
            suit, suit_confidence = self._identify_suit(corner_img)
            
            # Identify rank using shape analysis
            rank, rank_confidence = self._identify_rank(corner_img)
            
            # Combine results
            if rank and suit:
                card_class = f"{rank}{suit}"
                confidence = (suit_confidence + rank_confidence) / 2
                
                # Add an additional check: verify this is a valid card class
                if card_class in CARD_CLASSES:
                    return {
                        "card_class": card_class,
                        "confidence": min(confidence, 0.98),  # Cap confidence at 0.98
                        "source": "shape_color_analysis",
                        "rank": rank,
                        "suit": suit
                    }
            
            # If not identified, return unknown
            return {"card_class": "unknown", "confidence": 0.0, "source": "unidentified"}
            
        except Exception as e:
            log.error(f"Card identification error: {e}")
            return {"card_class": "unknown", "confidence": 0.0, "source": "error"}
    
    def _identify_suit(self, corner_img: np.ndarray) -> Tuple[Optional[str], float]:
        """Identify card suit using specialized color and shape analysis"""
        # Validate input image
        if corner_img is None or corner_img.size == 0 or corner_img.shape[0] < 5 or corner_img.shape[1] < 5:
            return None, 0.0
            
        try:
            # Create a working copy
            corner = corner_img.copy()
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
            
            # Calculate red and black regions
            # Red in HSV has hue around 0 and 180
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = red_mask1 | red_mask2
            
            # Black in HSV has low value
            black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
            
            # Calculate red and black pixel percentages
            total_pixels = corner.shape[0] * corner.shape[1]
            if total_pixels > 0:  # Avoid division by zero
                red_ratio = np.count_nonzero(red_mask) / total_pixels
                black_ratio = np.count_nonzero(black_mask) / total_pixels
                
                # Determine if it's a red or black card
                if red_ratio > 0.1 and red_ratio > black_ratio:
                    # It's a red card (hearts or diamonds)
                    # Extract shapes from red areas
                    corner_red = cv2.bitwise_and(corner, corner, mask=red_mask)
                    gray_red = cv2.cvtColor(corner_red, cv2.COLOR_BGR2GRAY)
                    _, thresh_red = cv2.threshold(gray_red, 30, 255, cv2.THRESH_BINARY)
                    
                    # Find contours in red areas
                    contours, _ = cv2.findContours(thresh_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Calculate shape features
                    diamonds_score = 0
                    hearts_score = 0
                    
                    for contour in contours:
                        if len(contour) < 5 or cv2.contourArea(contour) < 20:  # Skip small contours
                            continue
                            
                        # Calculate shape metrics
                        perimeter = cv2.arcLength(contour, True)
                        area = cv2.contourArea(contour)
                        
                        if perimeter > 0:
                            # Shape metrics
                            circularity = 4 * np.pi * area / (perimeter ** 2)
                            
                            # Diamonds have more angular shapes (lower circularity)
                            # Hearts have more rounded shapes (higher circularity)
                            if circularity < 0.6:
                                diamonds_score += 1
                            else:
                                hearts_score += 1
                    
                    # Determine suit based on shape features
                    if diamonds_score > hearts_score:
                        return 'd', 0.7  # Diamond
                    elif hearts_score > diamonds_score:
                        return 'h', 0.7  # Heart
                    else:
                        # If scores tied, use color distribution
                        red_std = np.std(hsv[:, :, 0])
                        if red_std > 20:
                            return 'h', 0.6  # Heart - more color variation
                        else:
                            return 'd', 0.6  # Diamond - more uniform color
                            
                elif black_ratio > 0.1:
                    # It's a black card (clubs or spades)
                    
                    # Extract shapes from black areas
                    corner_black = cv2.bitwise_and(corner, corner, mask=black_mask)
                    gray_black = cv2.cvtColor(corner_black, cv2.COLOR_BGR2GRAY)
                    _, thresh_black = cv2.threshold(gray_black, 30, 255, cv2.THRESH_BINARY)
                    
                    # Find contours in black areas
                    contours, _ = cv2.findContours(thresh_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Count significant contours (indicative of club's three lobes)
                    significant_contours = sum(1 for c in contours if cv2.contourArea(c) > 15)
                    
                    if significant_contours >= 3:
                        return 'c', 0.6  # Club - multiple distinct shapes
                    else:
                        return 's', 0.6  # Spade - typically fewer distinct shapes
            
            # If simple analysis failed, default to spade (most common black card)
            return 's', 0.4
            
        except Exception as e:
            log.error(f"Suit identification error: {e}")
            return None, 0.0
    
    def _identify_rank(self, corner_img: np.ndarray) -> Tuple[Optional[str], float]:
        """Identify card rank using shape analysis and patterns"""
        # Validate input image
        if corner_img is None or corner_img.size == 0:
            return None, 0.0
            
        try:
            # Convert to grayscale
            gray_corner = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to isolate text/symbols
            _, thresh = cv2.threshold(gray_corner, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Use connected components for rank identification
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
            
            # Filter out very small components
            significant_components = 0
            large_components = 0
            for i in range(1, num_labels):  # Skip background (0)
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                if area > 10:  # Minimum area threshold
                    significant_components += 1
                    
                if area > 50 and (width > 10 or height > 10):
                    large_components += 1
            
            # Use component count to identify rank
            if large_components >= 2:
                # Complex ranks like face cards (K, Q, J) or 10
                if significant_components >= 5:
                    return 'K', 0.5  # King often has most components
                elif significant_components >= 4:
                    return 'Q', 0.5  # Queen has medium complexity
                elif significant_components >= 3:
                    return 'J', 0.5  # Jack has simpler design
                else:
                    return 'T', 0.5  # 10 has two large digits
                    
            elif significant_components >= 3:
                return '8', 0.4  # Could be 8, A, etc.
                
            elif significant_components == 2:
                return 'A', 0.4  # Could be A, 2, 3, etc.
                
            elif significant_components == 1:
                # Check size and position of the single component
                if stats[1, cv2.CC_STAT_AREA] > 30:
                    return '7', 0.4  # 7 often has a large connected area
                else:
                    return '2', 0.4  # Default to lowest card
            
            # If connected components analysis fails, make an educated guess
            return 'A', 0.3  # Default to Ace
            
        except Exception as e:
            log.error(f"Rank identification error: {e}")
            return None, 0.0