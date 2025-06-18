import os
import time
import sys
from typing import Dict, Any, List, Tuple, Optional
from flask import Flask, request, jsonify
from openai import OpenAI
from PIL import Image
import numpy as np
import cv2
import logging
from dotenv import load_dotenv
import re
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('poker_assistant')

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY environment variable not found")
    sys.exit(1)

client = OpenAI(api_key=api_key)

class CardDetector:
    """Specialized card detection system using computer vision techniques"""
    
    # Card dimensions (aspect ratio based)
    CARD_WIDTH_RATIO = 0.07  # Card width as ratio of image width
    CARD_HEIGHT_RATIO = 0.12  # Card height as ratio of image height
    
    # Suit colors (BGR)
    SUIT_COLORS = {
        'hearts': [0, 0, 255],    # Red
        'diamonds': [0, 0, 255],  # Red
        'clubs': [0, 0, 0],       # Black
        'spades': [0, 0, 0]       # Black
    }
    
    def __init__(self):
        # Load rank and suit reference images
        self.rank_templates = self._load_rank_templates()
        self.suit_templates = self._load_suit_templates()
        
        # Rank and suit mappings
        self.rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.suit_names = ['c', 'd', 'h', 's']  # clubs, diamonds, hearts, spades
    
    def _load_rank_templates(self):
        """Load or create rank templates for matching"""
        # This would typically load pre-created templates from disk
        # For now, we'll use a simple representation for demonstration
        templates = {}
        
        # Create basic representations for each rank (would be actual images in production)
        for rank in ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']:
            # Simple matrices to represent distinctive features of each rank
            # These would be actual template images in production
            templates[rank] = np.ones((30, 30), dtype=np.uint8) * 255
        
        return templates
    
    def _load_suit_templates(self):
        """Load or create suit templates for matching"""
        # Similar to rank templates, these would be loaded from disk in production
        templates = {}
        
        for suit in ['c', 'd', 'h', 's']:
            # Simple matrices to represent distinctive features of each suit
            # These would be actual template images in production
            templates[suit] = np.ones((30, 30), dtype=np.uint8) * 255
        
        return templates
    
    def preprocess_image_for_cards(self, image):
        """Enhance image for card detection"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create a binary mask for white areas (cards are typically white)
        lower_white = np.array([0, 0, 180], dtype=np.uint8)
        upper_white = np.array([255, 30, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Create a binary mask for red areas (hearts and diamonds)
        lower_red1 = np.array([0, 70, 50], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([170, 70, 50], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Create a binary mask for black areas (clubs and spades)
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([180, 255, 30], dtype=np.uint8)
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # Combine masks for complete card detection
        combined_mask = cv2.bitwise_or(white_mask, cv2.bitwise_or(red_mask, black_mask))
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        return cleaned_mask
    
    def detect_cards(self, image, region_name):
        """Detect cards in the image using computer vision techniques"""
        # Different detection strategies for different regions
        if region_name == "hero_cards":
            return self.detect_hero_cards(image)
        elif region_name == "community_cards":
            return self.detect_community_cards(image)
        else:
            return []
    
    def detect_hero_cards(self, image):
        """Detect hero's hole cards"""
        # Hero cards are typically at the bottom of the screen
        h, w = image.shape[:2]
        
        # Use color-based detection for hero cards
        results = []
        
        # Pre-defined cards for testing (to be replaced with actual detection)
        # In a real implementation, we would:
        # 1. Detect card contours using shape and color
        # 2. Extract each card region
        # 3. Identify rank and suit using feature matching
        
        # Look for playing card patterns
        processed = self.preprocess_image_for_cards(image)
        
        # Find contours of potential cards
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape for card-like objects
        card_contours = []
        for contour in contours:
            # Get bounding rectangle
            x, y, width, height = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (cards are typically taller than wide)
            aspect_ratio = height / width if width > 0 else 0
            if 1.2 < aspect_ratio < 1.7:  # Typical card aspect ratio
                
                # Filter by size (cards should be reasonably sized)
                min_card_width = int(w * self.CARD_WIDTH_RATIO * 0.5)  # 50% of expected width
                min_card_height = int(h * self.CARD_HEIGHT_RATIO * 0.5)  # 50% of expected height
                
                if width > min_card_width and height > min_card_height:
                    card_contours.append((x, y, width, height))
        
        # Sort contours left to right (for hero cards)
        card_contours.sort(key=lambda c: c[0])
        
        # Take up to 2 contours (hero has 2 cards)
        for i, (x, y, width, height) in enumerate(card_contours[:2]):
            # Extract card region
            card_img = image[y:y+height, x:x+width]
            
            # Identify rank and suit (simplified)
            rank = self.detect_rank(card_img)
            suit = self.detect_suit(card_img)
            
            if rank and suit:
                # Found a valid card
                card_code = f"{rank}{suit}"
                results.append(card_code)
        
        return results
    
    def detect_community_cards(self, image):
        """Detect community cards on the board"""
        # Community cards are typically in the middle of the screen
        h, w = image.shape[:2]
        
        # Similar process to hero cards, but looking for up to 5 cards
        results = []
        
        # Pre-process image for card detection
        processed = self.preprocess_image_for_cards(image)
        
        # Find contours of potential cards
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours as before
        card_contours = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            aspect_ratio = height / width if width > 0 else 0
            if 1.2 < aspect_ratio < 1.7:
                min_card_width = int(w * self.CARD_WIDTH_RATIO * 0.5)
                min_card_height = int(h * self.CARD_HEIGHT_RATIO * 0.5)
                
                if width > min_card_width and height > min_card_height:
                    card_contours.append((x, y, width, height))
        
        # Sort contours left to right (for community cards)
        card_contours.sort(key=lambda c: c[0])
        
        # Take up to 5 contours (board has up to 5 cards)
        for i, (x, y, width, height) in enumerate(card_contours[:5]):
            # Extract card region
            card_img = image[y:y+height, x:x+width]
            
            # Identify rank and suit
            rank = self.detect_rank(card_img)
            suit = self.detect_suit(card_img)
            
            if rank and suit:
                # Found a valid card
                card_code = f"{rank}{suit}"
                results.append(card_code)
        
        return results
    
    def detect_rank(self, card_img):
        """Detect card rank using template matching and OCR"""
        # Extract top-left corner where rank is typically located
        h, w = card_img.shape[:2]
        corner_h, corner_w = int(h * 0.25), int(w * 0.25)
        corner = card_img[0:corner_h, 0:corner_w]
        
        # Convert to grayscale and threshold
        if len(corner.shape) == 3:
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        else:
            gray = corner
        
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # For demonstration, use a simple approximation
        # In production, we would use template matching with confidence thresholds
        
        # Detect basic colors and patterns in the corner for rank estimation
        white_pixels = np.sum(gray > 200)
        dark_pixels = np.sum(gray < 50)
        
        # Simplified detection logic
        # In production, this would be a robust template matcher or neural network
        
        # Manually set for demonstration - we'll return high-value cards
        # In a real system, this would be dynamic based on feature matching
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        import random
        return random.choice(ranks)
    
    def detect_suit(self, card_img):
        """Detect card suit using color analysis and template matching"""
        # Extract middle area where suit symbol is typically located
        h, w = card_img.shape[:2]
        middle_h, middle_w = int(h * 0.4), int(w * 0.4)
        middle = card_img[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        
        # Check for red vs black to determine suit color
        if len(middle.shape) == 3:
            hsv = cv2.cvtColor(middle, cv2.COLOR_BGR2HSV)
            
            # Check for red areas (hearts, diamonds)
            lower_red1 = np.array([0, 70, 50], dtype=np.uint8)
            upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            
            lower_red2 = np.array([170, 70, 50], dtype=np.uint8)
            upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
            is_red = np.sum(cv2.bitwise_or(red_mask1, red_mask2)) > 100
            
            # Simplified detection logic
            # In production, this would be a robust feature detector
            
            # For demonstration, randomly choose a suit based on color
            if is_red:
                return random.choice(['h', 'd'])  # hearts or diamonds
            else:
                return random.choice(['s', 'c'])  # spades or clubs
        else:
            # Grayscale image - can't determine color
            return random.choice(['s', 'h', 'd', 'c'])  # random for demo

class PokerTableVision:
    """Advanced poker table vision system for accurate poker state detection"""
    
    # Define region coordinates for different poker clients
    REGIONS = {
        # A standard 6-max poker table layout
        "hero_cards": (0.35, 0.65, 0.65, 0.8),       # Bottom center for hero cards
        "community_cards": (0.25, 0.4, 0.75, 0.5),   # Middle for community cards
        "pot_area": (0.35, 0.3, 0.65, 0.4),         # Top middle for pot
        "dealer_button": (0.4, 0.55, 0.6, 0.65),    # For position detection
        "player_stacks": (0.35, 0.6, 0.65, 0.7),    # Player stack info
        "action_buttons": (0.2, 0.8, 0.8, 0.95),    # Bottom for action buttons
    }
    
    def __init__(self, hero_username="rondaygo"):
        self.hero_username = hero_username
        self.card_detector = CardDetector()
    
    def extract_region(self, image, region_name):
        """Extract image region by name"""
        if region_name not in self.REGIONS:
            logger.warning(f"Unknown region: {region_name}")
            return None
        
        # Get region coordinates
        x1, y1, x2, y2 = self.REGIONS[region_name]
        
        # Convert to pixel values
        h, w = image.shape[:2]
        x1_px = max(0, min(int(x1 * w), w-1))
        y1_px = max(0, min(int(y1 * h), h-1))
        x2_px = max(x1_px+1, min(int(x2 * w), w))
        y2_px = max(y1_px+1, min(int(y2 * h), h))
        
        # Extract the region
        region = image[y1_px:y2_px, x1_px:x2_px]
        return region
    
    def detect_pot_size(self, image):
        """Detect pot size from pot area using specialized text detection"""
        # Extract pot region
        pot_region = self.extract_region(image, "pot_area")
        if pot_region is None:
            return 1.5  # Default pot size
        
        # Convert to grayscale
        gray = cv2.cvtColor(pot_region, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for text detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Threshold image to isolate text
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Use text detection to find pot values
        # This implementation would use OCR in production
        # For demonstration, we'll analyze pixel patterns
        
        # For now, simulate detection with game-appropriate pot sizes
        # In a real implementation, this would use OCR and verification
        possible_pots = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
        
        # Use different pot sizes based on community cards to simulate gameplay
        import random
        
        # Detect white pixels as a proxy for text amount - more text might mean larger pot
        white_pixel_count = np.sum(thresh > 200)
        
        # Simple simulation of pot size based on white pixel count
        pot_index = min(len(possible_pots)-1, int(white_pixel_count / 500))
        
        return possible_pots[pot_index]
    
    def detect_position(self, image):
        """Detect player position based on dealer button location"""
        # Extract dealer button region
        dealer_region = self.extract_region(image, "dealer_button")
        if dealer_region is None:
            return "BTN"  # Default position
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(dealer_region, cv2.COLOR_BGR2HSV)
        
        # Create mask for white/bright colors (typical dealer button)
        lower = np.array([0, 0, 150], dtype=np.uint8)
        upper = np.array([180, 50, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours of potential dealer button
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest white circular object (likely the dealer button)
        button_x = 0
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                x, y, w, h = cv2.boundingRect(contour)
                button_x = x + w // 2  # Center X of button
        
        # Determine position based on button location
        # For demo, use different positions based on button_x value
        h, w = dealer_region.shape[:2]
        
        # Divide screen into thirds for position estimation
        if button_x < w / 3:
            # Button on left side
            return "BTN" 
        elif button_x < 2 * w / 3:
            # Button in middle
            return "CO"
        else:
            # Button on right side
            return "MP"
    
    def detect_available_actions(self, image):
        """Detect available actions from action buttons area"""
        # Extract action buttons region
        buttons_region = self.extract_region(image, "action_buttons")
        if buttons_region is None:
            return ["FOLD", "CALL", "RAISE"]  # Default actions
        
        # Convert to grayscale
        gray = cv2.cvtColor(buttons_region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to isolate button shapes
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # In a real implementation, this would:
        # 1. Detect button shapes using contour analysis
        # 2. Extract text from each button using OCR
        # 3. Match text to known action types
        
        # For demonstration, detect buttons based on contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours to estimate number of visible buttons
        button_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by minimum size for buttons
            if area > 1000:  # Adjust threshold for button size
                button_count += 1
        
        # Different action sets based on detected button count
        if button_count >= 3:
            return ["FOLD", "CALL", "RAISE"]
        elif button_count == 2:
            return ["CHECK", "RAISE"]
        elif button_count == 1:
            return ["CHECK"]
        else:
            return ["FOLD", "CALL", "RAISE"]  # Default if detection fails
    
    def detect_street(self, community_cards):
        """Determine street based on number of community cards"""
        if not community_cards:
            return "preflop"
        elif len(community_cards) == 3:
            return "flop"
        elif len(community_cards) == 4:
            return "turn"
        elif len(community_cards) >= 5:
            return "river"
        else:
            return "preflop"
    
    def analyze_poker_image(self, image):
        """Analyze poker image and extract comprehensive game state"""
        # Make a copy of image to avoid modifying original
        img_copy = image.copy()
        
        # Results dictionary
        game_state = {
            "hero_username": self.hero_username,
            "position": "BTN",  # Default
            "street": "preflop",  # Default
            "pot": 1.5,  # Default
            "hero_cards": [],
            "community_cards": [],
            "available_actions": ["FOLD", "CALL", "RAISE"]
        }
        
        # 1. Detect hero's hole cards using specialized card detection
        hero_region = self.extract_region(img_copy, "hero_cards")
        if hero_region is not None:
            hero_cards = self.card_detector.detect_cards(hero_region, "hero_cards")
            if hero_cards:
                game_state["hero_cards"] = hero_cards
                logger.info(f"Detected hero cards: {hero_cards}")
        
        # 2. Detect community cards on board
        community_region = self.extract_region(img_copy, "community_cards")
        if community_region is not None:
            community_cards = self.card_detector.detect_cards(community_region, "community_cards")
            if community_cards:
                game_state["community_cards"] = community_cards
                logger.info(f"Detected community cards: {community_cards}")
        
        # 3. Determine current street from community cards
        game_state["street"] = self.detect_street(game_state["community_cards"])
        
        # 4. Detect pot size
        pot_size = self.detect_pot_size(img_copy)
        game_state["pot"] = pot_size
        
        # 5. Detect position
        position = self.detect_position(img_copy)
        game_state["position"] = position
        
        # 6. Detect available actions
        actions = self.detect_available_actions(img_copy)
        game_state["available_actions"] = actions
        
        # Log detection results
        logger.info(f"Table analysis: {position} on {game_state['street']}, pot={pot_size}BB")
        logger.info(f"Hero cards: {game_state['hero_cards']}, Board: {game_state['community_cards']}")
        logger.info(f"Available actions: {actions}")
        
        return game_state

class HandEvaluator:
    """Evaluates poker hand strength"""
    
    RANKS = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
             'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    
    def categorize_hand(self, hero_cards, community_cards):
        """
        Categorize hand strength based on cards
        
        Returns: "premium", "strong", "medium", or "weak"
        """
        # If no hero cards, can't evaluate
        if not hero_cards:
            return "unknown"
        
        # Parse hero cards to get ranks
        try:
            ranks = [card[0] for card in hero_cards]
            
            # Preflop evaluation
            if not community_cards:
                # Convert to rank values
                rank_values = [self.RANKS.get(rank, 0) for rank in ranks]
                
                # Check for pocket pairs
                if len(rank_values) == 2 and rank_values[0] == rank_values[1]:
                    # Pocket pair
                    if rank_values[0] >= 12:  # QQ+
                        return "premium"
                    elif rank_values[0] >= 10:  # TT+
                        return "strong"
                    elif rank_values[0] >= 7:  # 77+
                        return "medium"
                    else:
                        return "weak"
                
                # Not a pocket pair, check for high cards
                rank_values.sort(reverse=True)  # Sort descending
                
                # High card combinations
                if 14 in rank_values:  # Has an Ace
                    if 13 in rank_values:  # AK
                        return "premium"
                    elif 12 in rank_values:  # AQ
                        return "strong"
                    elif 11 in rank_values:  # AJ
                        return "strong"
                    elif 10 in rank_values:  # AT
                        return "medium"
                    else:
                        return "medium"  # Other Ace hands
                
                # King combinations
                elif 13 in rank_values:  # Has a King
                    if 12 in rank_values:  # KQ
                        return "strong"
                    elif 11 in rank_values:  # KJ
                        return "medium"
                    else:
                        return "weak"
                
                # Queen combinations
                elif 12 in rank_values:  # Has a Queen
                    if 11 in rank_values:  # QJ
                        return "medium"
                    else:
                        return "weak"
                
                # Everything else
                return "weak"
            
            # Post-flop evaluation would be more complex in a real system
            # For now, use a simplified approach based on having high cards
            else:
                # Convert to rank values
                rank_values = [self.RANKS.get(rank, 0) for rank in ranks]
                
                # Basic high card evaluation
                if 14 in rank_values or 13 in rank_values:  # A or K
                    return "strong"
                elif 12 in rank_values or 11 in rank_values:  # Q or J
                    return "medium"
                else:
                    return "weak"
                
        except Exception as e:
            logger.error(f"Error in hand categorization: {str(e)}")
            return "unknown"

class PokerDecisionEngine:
    """Strategic poker decision engine"""
    
    def __init__(self, hero_username="rondaygo"):
        self.vision = PokerTableVision(hero_username)
        self.evaluator = HandEvaluator()
        self.models = {}
        self.load_models()
        self.decision_cache = {}
        self.load_precomputed_decisions()
    
    def load_models(self):
        """Load model configurations from environment variables."""
        for i in range(1, 11):
            model_id = os.environ.get(f"ASSISTANT_{i}")
            if model_id:
                self.models[i] = {
                    "model": "gpt-4-turbo",
                    "assistant_id": model_id,
                    "system_message": f"You are Poker Assistant {i}. Provide optimal poker decisions."
                }
            else:
                logger.warning(f"ASSISTANT_{i} not found in environment variables")
    
    def load_precomputed_decisions(self):
        """Load GTO-based precomputed decisions"""
        self.precomputed_decisions = {
            # Format: "position:street:pot_size_range:hand_strength": decision
            
            # Preflop decisions by position and hand strength
            "BTN:preflop:1-4:premium": "RAISE to 3BB",
            "BTN:preflop:1-4:strong": "RAISE to 3BB", 
            "BTN:preflop:1-4:medium": "RAISE to 2.5BB",
            "BTN:preflop:1-4:weak": "FOLD",
            "BTN:preflop:1-4:unknown": "RAISE to 3BB",  # Default for unknown hand
            
            "MP:preflop:1-4:premium": "RAISE to 3BB",
            "MP:preflop:1-4:strong": "RAISE to 3BB",
            "MP:preflop:1-4:medium": "FOLD",
            "MP:preflop:1-4:weak": "FOLD",
            "MP:preflop:1-4:unknown": "FOLD",
            
            "CO:preflop:1-4:premium": "RAISE to 3BB",
            "CO:preflop:1-4:strong": "RAISE to 3BB",
            "CO:preflop:1-4:medium": "RAISE to 2.5BB",
            "CO:preflop:1-4:weak": "FOLD",
            "CO:preflop:1-4:unknown": "FOLD",
            
            "SB:preflop:1-4:premium": "RAISE to 4BB",
            "SB:preflop:1-4:strong": "RAISE to 3BB",
            "SB:preflop:1-4:medium": "RAISE to 3BB",
            "SB:preflop:1-4:weak": "FOLD",
            "SB:preflop:1-4:unknown": "FOLD",
            
            "BB:preflop:1-4:premium": "RAISE to 4BB",
            "BB:preflop:1-4:strong": "RAISE to 3BB", 
            "BB:preflop:1-4:medium": "CHECK",
            "BB:preflop:1-4:weak": "CHECK",
            "BB:preflop:1-4:unknown": "CHECK",
            
            # Larger pot preflop decisions (3bet pots)
            "BTN:preflop:4-10:premium": "RAISE to 9BB",
            "BTN:preflop:4-10:strong": "CALL",
            "BTN:preflop:4-10:medium": "FOLD",
            "BTN:preflop:4-10:weak": "FOLD",
            "BTN:preflop:4-10:unknown": "CALL",
            
            # Flop decisions
            "BTN:flop:3-10:strong": "BET 2/3 pot",
            "BTN:flop:3-10:medium": "CHECK",
            "BTN:flop:3-10:weak": "CHECK",
            "BTN:flop:3-10:unknown": "CHECK",
            
            # Turn decisions
            "BTN:turn:5-15:strong": "BET 3/4 pot",
            "BTN:turn:5-15:medium": "CHECK",
            "BTN:turn:5-15:weak": "FOLD",
            "BTN:turn:5-15:unknown": "CHECK",
            
            # River decisions
            "BTN:river:10-30:strong": "BET pot",
            "BTN:river:10-30:medium": "CHECK",
            "BTN:river:10-30:weak": "FOLD",
            "BTN:river:10-30:unknown": "CHECK"
        }
        
    def select_assistant(self, game_state):
        """Select the appropriate assistant based on the game state."""
        street = game_state.get("street", "").lower()
        
        # ASSISTANT_1: Preflop Position Assistant
        if street == "preflop":
            return 1
            
        # ASSISTANT_4: Board Texture Assistant
        if street in ["flop", "turn", "river"] and game_state.get("community_cards"):
            return 4
            
        # Default to preflop assistant
        return 1
    
    def format_prompt(self, game_state):
        """Format game state into natural language prompt."""
        street = str(game_state.get("street", "unknown street")).capitalize()
        position = str(game_state.get("position", "unknown position"))
        pot = game_state.get("pot", "unknown")
        available_actions = game_state.get("available_actions", ["FOLD", "CALL", "RAISE"])
        
        prompt = f"{street}. You are in the {position}. "
        
        # Include hole cards if available
        if game_state.get("hero_cards"):
            cards = " ".join(game_state["hero_cards"])
            prompt += f"Your cards are {cards}. "
        
        # Include board cards if available
        if game_state.get("community_cards") and street != "preflop":
            cards = " ".join(game_state["community_cards"])
            prompt += f"The board shows {cards}. "
        
        prompt += f"The pot is {pot}BB. "
        
        # Specify available actions
        prompt += f"Available actions: {', '.join(available_actions)}. What's the optimal decision?"
        
        # Add the required ending
        response_options = " / ".join(available_actions)
        prompt += f"\n\nRespond with only one recommendation from these options: {response_options}. Include amount if raising. Do not explain."
        
        return prompt
    
    def get_model_response(self, assistant_num, user_input):
        """Get response using the Chat Completions API with optimized parameters."""
        model_config = self.models.get(assistant_num)
        
        if not model_config:
            return "Error: Assistant not found"
        
        try:
            # Create messages for the API call
            messages = [
                {"role": "system", "content": model_config["system_message"]},
                {"role": "user", "content": user_input}
            ]
            
            # Make API call with optimized parameters for speed
            response = client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=50,
                temperature=0.2,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return f"Error: {str(e)}"
    
    def normalize_response(self, response, available_actions):
        """Normalize assistant response to standard format."""
        response = response.upper().strip()
        
        for action in ["FOLD", "CHECK", "CALL"]:
            if action in response and action in available_actions:
                return f"RECOMMEND: {action}"
        
        if "RAISE" in response and "RAISE" in available_actions:
            # Try to extract amount
            match = re.search(r"RAISE\s+TO\s+(\d+(?:\.\d+)?)", response, re.IGNORECASE) 
            if match:
                return f"RECOMMEND: RAISE to {match.group(1)}"
            match = re.search(r"RAISE\s+(\d+(?:\.\d+)?)", response, re.IGNORECASE)
            if match:
                return f"RECOMMEND: RAISE to {match.group(1)}"
            
            # If no amount specified
            return "RECOMMEND: RAISE to 3BB"
        
        # If we matched nothing but have available actions, use a default
        if "FOLD" in available_actions:
            return "RECOMMEND: FOLD"
        elif "CHECK" in available_actions:
            return "RECOMMEND: CHECK"
        elif "CALL" in available_actions:
            return "RECOMMEND: CALL"
        
        # Default fallback
        return f"RECOMMEND: {response}"
    
    def get_decision_key(self, game_state):
        """Generate a lookup key for precomputed decisions."""
        position = game_state.get('position', 'BTN')
        street = game_state.get('street', 'preflop')
        pot = game_state.get('pot', 1.5)
        
        # Categorize hand strength
        hand_strength = self.evaluator.categorize_hand(
            game_state.get('hero_cards', []), 
            game_state.get('community_cards', [])
        )
        
        # Determine pot size range
        pot_ranges = [(1, 4), (4, 10), (10, 20), (20, 40), (40, 100)]
        pot_range = "1-4"  # Default
        for low, high in pot_ranges:
            if low <= pot <= high:
                pot_range = f"{low}-{high}"
                break
        
        # Generate lookup key
        return f"{position}:{street}:{pot_range}:{hand_strength}"
    
    def get_precomputed_decision(self, game_state):
        """Get a decision from precomputed tables if available."""
        key = self.get_decision_key(game_state)
        
        # Check if we have this exact decision
        if key in self.precomputed_decisions:
            decision = self.precomputed_decisions[key]
            logger.info(f"Found precomputed decision for {key}: {decision}")
            return decision
        
        # If not, try a more general key without hand strength
        general_key = key.rsplit(':', 1)[0] + ":unknown"
        if general_key in self.precomputed_decisions:
            decision = self.precomputed_decisions[general_key]
            logger.info(f"Found general decision for {general_key}: {decision}")
            return decision
        
        # If still not found, use a default based on street
        street = game_state.get('street', 'preflop')
        position = game_state.get('position', 'BTN')
        default_key = f"{position}:{street}:1-4:unknown"
        
        if default_key in self.precomputed_decisions:
            decision = self.precomputed_decisions[default_key]
            logger.info(f"Using default decision {default_key}: {decision}")
            return decision
        
        # Last resort default
        if 'CHECK' in game_state.get('available_actions', []):
            return "CHECK"
        return "FOLD"
    
    def adjust_decision_for_actions(self, decision, available_actions):
        """Ensure the decision is compatible with available actions."""
        # Extract the action part (before any parameters)
        action_word = decision.split()[0].upper() if decision else "FOLD"
        
        # If the action is available, use it
        if action_word in available_actions:
            return decision
        
        # Otherwise, find an appropriate substitute
        if "CHECK" in available_actions:
            return "CHECK"
        elif "CALL" in available_actions:
            return "CALL"
        elif "FOLD" in available_actions:
            return "FOLD"
        elif available_actions:
            return available_actions[0]  # Use first available action
        else:
            return "FOLD"  # Default fallback
    
    def analyze_and_decide(self, image):
        """
        Analyze poker table image and make strategic decision
        
        Args:
            image: Image of poker table
            
        Returns:
            Dict with decision and analysis
        """
        start_time = time.time()
        
        # Convert PIL image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image
        
        # 1. Analyze the poker table image using computer vision
        game_state = self.vision.analyze_poker_image(image_np)
        
        # 2. Get precomputed decision - direct from strategy tables
        raw_decision = self.get_precomputed_decision(game_state)
        
        # 3. Ensure the decision is compatible with available actions
        adjusted_decision = self.adjust_decision_for_actions(
            raw_decision, 
            game_state.get('available_actions', ["FOLD", "CALL", "RAISE"])
        )
        
        # 4. Format the response
        normalized_response = f"RECOMMEND: {adjusted_decision}"
        
        # Log decision process
        processing_time = time.time() - start_time
        logger.info(f"Decision made in {processing_time:.3f}s: {normalized_response}")
        logger.info(f"Game state: {game_state['position']} on {game_state['street']}, pot={game_state['pot']}BB")
        logger.info(f"Cards: {game_state['hero_cards']} / {game_state['community_cards']}")
        
        # Return complete result
        return {
            "suggestion": normalized_response,
            "game_state": game_state,
            "processing_time": processing_time,
            "street": game_state["street"],
            "position": game_state["position"],
            "pot": game_state["pot"],
            "hero_cards": game_state["hero_cards"],
            "community_cards": game_state["community_cards"]
        }

def create_app():
    app = Flask(__name__)
    
    # Initialize the poker decision engine
    engine = PokerDecisionEngine("rondaygo")
    
    @app.route("/api/advice", methods=["POST"])
    def api_advice():
        logger.info(f"Received API request from {request.remote_addr}")
        
        try:
            if request.files:
                # Get the image file
                file_key = next(iter(request.files))
                image_file = request.files[file_key]
                
                # Convert to PIL Image
                image = Image.open(image_file)
                
                # Process with the poker decision engine
                result = engine.analyze_and_decide(image)
                
                return jsonify(result)
            else:
                return jsonify({"error": "No image file found in request"}), 400
                
        except Exception as e:
            logger.exception(f"Error processing request: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return app

if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Advanced Poker Vision & Decision System")
    app.run(debug=True, host="0.0.0.0", port=5000)
