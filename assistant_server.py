import os
import time
import sys
import threading
from typing import Dict, Any, List, Tuple, Optional
from flask import Flask, request, jsonify
from openai import OpenAI
from PIL import Image
import numpy as np
import cv2
import pytesseract
import json
import logging
from dotenv import load_dotenv
import re
from collections import deque
import concurrent.futures

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

class PokerCard:
    """Represents a card with rank and suit"""
    
    # Card recognition templates
    RANKS = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 
        'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    
    SUITS = {
        'c': 'clubs', 'd': 'diamonds', 'h': 'hearts', 's': 'spades',
        '\u2663': 'clubs', '\u2666': 'diamonds', '\u2665': 'hearts', '\u2660': 'spades'
    }
    
    def __init__(self, card_str: str):
        """Initialize from card string (e.g., 'Ah' for Ace of hearts)"""
        if len(card_str) >= 2:
            self.rank_str = card_str[0].upper()
            self.suit_str = card_str[1].lower()
            self.rank = self.RANKS.get(self.rank_str, 0)
            self.suit = self.SUITS.get(self.suit_str, 'unknown')
        else:
            self.rank_str = '?'
            self.suit_str = '?'
            self.rank = 0
            self.suit = 'unknown'
    
    def __str__(self):
        return f"{self.rank_str}{self.suit_str}"
    
    def __repr__(self):
        return self.__str__()

class RegionOfInterest:
    """Defines regions of the poker screenshot for targeted extraction"""
    
    def __init__(self, name: str, x1: float, y1: float, x2: float, y2: float):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def extract_from_image(self, image: np.ndarray) -> np.ndarray:
        """Extract this region from the image"""
        h, w = image.shape[:2]
        x1_px = int(self.x1 * w)
        y1_px = int(self.y1 * h)
        x2_px = int(self.x2 * w)
        y2_px = int(self.y2 * h)
        return image[y1_px:y2_px, x1_px:x2_px]

class PokerTableAnalyzer:
    """Advanced poker table analyzer using multiple techniques for card/chip detection"""
    
    # Define UI regions for different poker clients
    # These are approximations and may need calibration
    REGIONS = [
        # Generic layout that fits most poker clients
        RegionOfInterest('pot_area', 0.3, 0.2, 0.7, 0.4),        # Middle upper area for pot info
        RegionOfInterest('community_cards', 0.25, 0.4, 0.75, 0.5), # Middle area for board cards
        RegionOfInterest('hero_cards', 0.4, 0.7, 0.6, 0.9),       # Bottom center for hero cards
        RegionOfInterest('buttons', 0.1, 0.8, 0.9, 1.0),         # Bottom area for action buttons
        RegionOfInterest('hero_stack', 0.4, 0.6, 0.6, 0.7),      # Below hero cards for stack info
        RegionOfInterest('hero_position', 0.25, 0.65, 0.75, 0.75), # Around hero for position info
        RegionOfInterest('player_area', 0.0, 0.0, 1.0, 1.0),     # Full image for player detection
    ]
    
    # More specific regions for ACR (America's Card Room)
    ACR_REGIONS = [
        RegionOfInterest('pot_area', 0.45, 0.25, 0.55, 0.35),
        RegionOfInterest('community_cards', 0.35, 0.35, 0.65, 0.45),
        RegionOfInterest('hero_cards', 0.45, 0.7, 0.55, 0.8),
        RegionOfInterest('buttons', 0.3, 0.85, 0.7, 0.95),
        RegionOfInterest('hero_stack', 0.45, 0.65, 0.55, 0.7),
        RegionOfInterest('hero_position', 0.4, 0.7, 0.6, 0.75),
    ]
    
    # More regions for other clients can be added
    
    def __init__(self, hero_username="rondaygo", client_type="acr"):
        self.hero_username = hero_username.lower()
        self.client_type = client_type.lower()
        
        # Choose region set based on client type
        if client_type == "acr":
            self.regions = self.ACR_REGIONS
        else:
            self.regions = self.REGIONS
        
        # Configure Tesseract with special poker settings
        self.config = '--oem 1 --psm 6 -l eng'
        
        # Precompile regex patterns for speed
        self.patterns = {
            'pot': re.compile(r'(?:pot|total)[:\s]*(\d+\.?\d*)(?:\s*BB)?', re.IGNORECASE),
            'card': re.compile(r'[2-9TJQKA][cdhs\u2663\u2666\u2665\u2660]'),
            'stack': re.compile(r'(\d+\.?\d*)\s*(?:BB|bb)'),
            'action': re.compile(r'(fold|check|call|raise|bet|all[\s-]*in)', re.IGNORECASE),
            'position': re.compile(r'(SB|BB|BTN|CO|MP|UTG)', re.IGNORECASE),
            'username': re.compile(rf'{re.escape(hero_username)}', re.IGNORECASE)
        }
        
        # Card detection thresholds
        self.card_similarity_threshold = 0.7
        
        # Initialize card template matcher
        self.initialize_card_templates()
    
    def initialize_card_templates(self):
        """Initialize template matchers for card detection (backup for OCR)"""
        # This would load pre-made card templates from disk
        self.card_templates = {}
        # In a real implementation, we'd load actual card template images
    
    def enhance_image_for_cards(self, image: np.ndarray) -> np.ndarray:
        """Apply specific image processing to enhance card visibility"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for red and black (card colors)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_red, mask_black)
        
        # Enhance contrast in the card areas
        enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        
        return enhanced
    
    def enhance_image_for_text(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better text recognition"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold to handle varying lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up text
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return opening
    
    def detect_cards(self, image: np.ndarray) -> List[str]:
        """Detect cards using multiple techniques including OCR and template matching"""
        cards = []
        
        # Method 1: OCR
        enhanced = self.enhance_image_for_text(image)
        text = pytesseract.image_to_string(enhanced, config=self.config)
        ocr_cards = self.patterns['card'].findall(text)
        cards.extend(ocr_cards)
        
        # Method 2: Use card-specific color and shape detection (simplified version)
        enhanced_cards = self.enhance_image_for_cards(image)
        gray = cv2.cvtColor(enhanced_cards, cv2.COLOR_BGR2GRAY)
        
        # Find potential card contours (simplified)
        _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # In a complete implementation, we would:
        # 1. Filter contours by size and shape to find card-like rectangles
        # 2. Extract each card region
        # 3. Detect rank and suit using template matching or neural networks
        # 4. Combine with OCR results
        
        # For now, we'll rely primarily on OCR results
        return cards
    
    def detect_pot_size(self, image: np.ndarray) -> float:
        """Detect pot size from the pot area"""
        # Enhance for text
        enhanced = self.enhance_image_for_text(image)
        
        # Get text
        text = pytesseract.image_to_string(enhanced, config=self.config)
        
        # Find pot mention
        pot_match = self.patterns['pot'].search(text)
        if pot_match:
            try:
                return float(pot_match.group(1))
            except ValueError:
                pass
        
        # If no direct pot mention, look for numerical values
        numbers = re.findall(r'(\d+\.?\d*)', text)
        if numbers:
            try:
                # Take the largest value as it's likely the pot
                return max(float(n) for n in numbers)
            except ValueError:
                pass
        
        return 1.5  # Default small blind + big blind
    
    def detect_available_actions(self, image: np.ndarray) -> List[str]:
        """Detect available action buttons"""
        # Enhance for text
        enhanced = self.enhance_image_for_text(image)
        
        # Get text
        text = pytesseract.image_to_string(enhanced, config=self.config)
        
        # Default actions
        actions = []
        
        # Search for action keywords
        text_lower = text.lower()
        if 'fold' in text_lower:
            actions.append('FOLD')
        if 'check' in text_lower:
            actions.append('CHECK')
        if 'call' in text_lower:
            actions.append('CALL')
        if 'bet' in text_lower or 'raise' in text_lower:
            actions.append('RAISE')
        
        # If no actions detected, provide defaults
        if not actions:
            actions = ['FOLD', 'CALL', 'RAISE']
        
        return actions
    
    def detect_position(self, image: np.ndarray) -> str:
        """Detect hero's position"""
        # Enhance for text
        enhanced = self.enhance_image_for_text(image)
        
        # Get text
        text = pytesseract.image_to_string(enhanced, config=self.config)
        
        # Look for position markers
        position_match = self.patterns['position'].search(text)
        if position_match:
            return position_match.group(1).upper()
        
        # Check for dealer button, SB, BB indicators
        text_lower = text.lower()
        if 'dealer' in text_lower or 'btn' in text_lower or 'button' in text_lower:
            return 'BTN'
        if 'small blind' in text_lower or 'sb' in text_lower:
            return 'SB'
        if 'big blind' in text_lower or 'bb' in text_lower:
            return 'BB'
        
        # Default to button
        return 'BTN'
    
    def detect_hero_username(self, image: np.ndarray) -> bool:
        """Check if hero username is visible in the image"""
        # Enhance for text
        enhanced = self.enhance_image_for_text(image)
        
        # Get text
        text = pytesseract.image_to_string(enhanced, config=self.config)
        
        # Check if hero username is present
        return self.hero_username.lower() in text.lower()
    
    def detect_hero_stack(self, image: np.ndarray) -> float:
        """Detect hero's stack size"""
        # Enhance for text
        enhanced = self.enhance_image_for_text(image)
        
        # Get text
        text = pytesseract.image_to_string(enhanced, config=self.config)
        
        # Find stack mentions
        stack_match = self.patterns['stack'].search(text)
        if stack_match:
            try:
                return float(stack_match.group(1))
            except ValueError:
                pass
        
        return 100.0  # Default stack size
    
    def detect_street(self, community_cards: List[str]) -> str:
        """Determine street based on number of community cards"""
        if not community_cards:
            return 'preflop'
        elif len(community_cards) == 3:
            return 'flop'
        elif len(community_cards) == 4:
            return 'turn'
        elif len(community_cards) == 5:
            return 'river'
        else:
            return 'preflop'  # Default
    
    def analyze_table(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the poker table image and extract comprehensive game state
        """
        # Original image copy for display/debug
        original = image.copy()
        
        # Extract regions
        regions_data = {}
        for region in self.regions:
            regions_data[region.name] = region.extract_from_image(image)
        
        # Detect key elements
        hero_cards = self.detect_cards(regions_data['hero_cards'])
        community_cards = self.detect_cards(regions_data['community_cards'])
        pot_size = self.detect_pot_size(regions_data['pot_area'])
        available_actions = self.detect_available_actions(regions_data['buttons'])
        position = self.detect_position(regions_data['hero_position'])
        hero_stack = self.detect_hero_stack(regions_data['hero_stack'])
        street = self.detect_street(community_cards)
        
        # Detect if hero is present
        hero_present = self.detect_hero_username(regions_data['player_area'])
        
        # Build comprehensive game state
        game_state = {
            "hero_username": self.hero_username,
            "hero_present": hero_present,
            "position": position,
            "street": street,
            "pot": pot_size,
            "hero_cards": hero_cards,
            "community_cards": community_cards,
            "hero_stack": hero_stack,
            "available_actions": available_actions
        }
        
        # Debug info
        logger.info(f"Analyzed table: position={position}, street={street}, pot={pot_size}BB")
        logger.info(f"Hero cards: {hero_cards}, Community cards: {community_cards}")
        logger.info(f"Available actions: {available_actions}")
        
        return game_state

class PokerDecisionEngine:
    """
    High-performance poker decision engine with comprehensive strategy module
    """
    
    def __init__(self, hero_username="rondaygo"):
        self.analyzer = PokerTableAnalyzer(hero_username)
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
        """Load comprehensive GTO-based precomputed decisions"""
        self.precomputed_decisions = {
            # Format: "position:street:pot_size_range:hand_type": decision
            # Preflop decisions
            "BTN:preflop:1-4:premium": "RAISE to 2.5BB",
            "BTN:preflop:1-4:strong": "RAISE to 2.5BB",
            "BTN:preflop:1-4:medium": "RAISE to 2.5BB",
            "BTN:preflop:1-4:weak": "FOLD",
            
            "SB:preflop:1-4:premium": "RAISE to 3BB",
            "SB:preflop:1-4:strong": "RAISE to 3BB",
            "SB:preflop:1-4:medium": "CALL",
            "SB:preflop:1-4:weak": "FOLD",
            
            "BB:preflop:1-4:premium": "RAISE to 4BB",
            "BB:preflop:1-4:strong": "RAISE to 3BB",
            "BB:preflop:1-4:medium": "CHECK",
            "BB:preflop:1-4:weak": "CHECK",
            
            # Flop decisions
            "BTN:flop:3-10:strong": "BET 1/2 pot",
            "BTN:flop:3-10:medium": "CHECK",
            "BTN:flop:3-10:weak": "CHECK",
            
            "SB:flop:3-10:strong": "BET 1/2 pot",
            "SB:flop:3-10:medium": "CHECK",
            "SB:flop:3-10:weak": "CHECK",
            
            "BB:flop:3-10:strong": "BET 1/2 pot",
            "BB:flop:3-10:medium": "CHECK",
            "BB:flop:3-10:weak": "CHECK",
            
            # Turn decisions
            "BTN:turn:5-15:strong": "BET 2/3 pot",
            "BTN:turn:5-15:medium": "CHECK",
            "BTN:turn:5-15:weak": "CHECK",
            
            # River decisions
            "BTN:river:10-30:strong": "BET pot",
            "BTN:river:10-30:medium": "CHECK",
            "BTN:river:10-30:weak": "CHECK"
        }
    
    def categorize_hand(self, hero_cards: List[str], community_cards: List[str]) -> str:
        """
        Categorize hand strength based on hole cards and community cards
        
        Returns: "premium", "strong", "medium", or "weak"
        """
        if not hero_cards:
            return "unknown"  # Can't determine hand strength without cards
        
        # Convert string representations to PokerCard objects
        hole_cards = [PokerCard(card) for card in hero_cards]
        
        # Preflop hand strength determination
        if not community_cards:  # Preflop
            # Sort hole cards by rank, highest first
            sorted_cards = sorted(hole_cards, key=lambda card: card.rank, reverse=True)
            
            # Premium hands: AA, KK, QQ, AKs
            if len(sorted_cards) >= 2:
                if sorted_cards[0].rank == sorted_cards[1].rank and sorted_cards[0].rank >= 12:  # Pair of QQ+
                    return "premium"
                if sorted_cards[0].rank == 14 and sorted_cards[1].rank == 13:  # AK
                    return "premium"
            
            # Strong hands: JJ, TT, AQ, AJs, KQs
            if len(sorted_cards) >= 2:
                if sorted_cards[0].rank == sorted_cards[1].rank and sorted_cards[0].rank >= 10:  # Pair of TT+
                    return "strong"
                if sorted_cards[0].rank == 14 and sorted_cards[1].rank >= 11:  # AJ+
                    return "strong"
                if sorted_cards[0].rank == 13 and sorted_cards[1].rank == 12:  # KQ
                    return "strong"
            
            # Medium hands: 99, 88, 77, AT, KJ, QJ, JTs
            if len(sorted_cards) >= 2:
                if sorted_cards[0].rank == sorted_cards[1].rank and sorted_cards[0].rank >= 7:  # Pair of 77+
                    return "medium"
                if sorted_cards[0].rank == 14 and sorted_cards[1].rank == 10:  # AT
                    return "medium"
                if sorted_cards[0].rank == 13 and sorted_cards[1].rank == 11:  # KJ
                    return "medium"
                if sorted_cards[0].rank == 12 and sorted_cards[1].rank == 11:  # QJ
                    return "medium"
            
            # All other hands: weak
            return "weak"
        else:
            # Postflop hand strength would be determined by evaluating full hand
            # This would require a full hand evaluator which is beyond this example
            # For now, we'll use a simplified approach
            if len(hole_cards) >= 2 and hole_cards[0].rank >= 10 and hole_cards[1].rank >= 10:
                return "strong"
            elif len(hole_cards) >= 2 and (hole_cards[0].rank >= 10 or hole_cards[1].rank >= 10):
                return "medium"
            return "weak"
    
    def select_assistant(self, game_state: Dict, tournament_mode: bool) -> int:
        """Select the appropriate assistant based on the game state."""
        street = str(game_state.get("street", "")).lower()
        action = str(game_state.get("action", "")).lower()
        position = str(game_state.get("position", "")).lower()
        
        # ASSISTANT_1: Preflop Position Assistant
        if street == "preflop":
            return 1
            
        # ASSISTANT_2: Stack-Based ICM Assistant
        if tournament_mode or game_state.get("hero_stack", 100) < 20:
            return 2
            
        # ASSISTANT_4: Board Texture Assistant
        if street in ["flop", "turn", "river"] and game_state.get("community_cards", []):
            return 4
            
        # ASSISTANT_5: Pot Odds Assistant
        if game_state.get("pot", 0) > 0:
            return 5
            
        # Default to preflop assistant
        return 1
    
    def format_prompt(self, game_state: Dict) -> str:
        """Format game state into natural language prompt according to required format."""
        street = str(game_state.get("street", "unknown street")).capitalize()
        position = str(game_state.get("position", "unknown position"))
        hero_stack = game_state.get("hero_stack", 100.0)
        pot = game_state.get("pot", "unknown")
        available_actions = game_state.get("available_actions", ["FOLD", "CALL", "RAISE"])
        
        prompt = f"{street}. You are in the {position} with {hero_stack}BB. "
        
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
    
    def get_model_response(self, assistant_num: int, user_input: str) -> str:
        """Get response using the Chat Completions API with optimized parameters."""
        model_config = self.models.get(assistant_num)
        
        if not model_config:
            return "Error: Assistant not found"
        
        try:
            # Create messages for the API call - simplified for speed
            messages = [
                {"role": "system", "content": model_config["system_message"]},
                {"role": "user", "content": user_input}
            ]
            
            # Make API call with optimized parameters for speed
            response = client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=50,  # Reduced for faster response
                temperature=0.2,  # Lower for more consistency
                stream=False  # Disabled streaming for faster response
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return f"Error: {str(e)}"
    
    def normalize_response(self, response: str, available_actions: List[str]) -> str:
        """Normalize assistant response to standard format based on available actions."""
        response = response.upper().strip()
        
        for action in ["FOLD", "CHECK", "CALL"]:
            if action in response and action in available_actions:
                return f"RECOMMEND: {action}"
        
        if "RAISE" in response and "RAISE" in available_actions:
            # Try to extract amount
            match = re.search(r"RAISE\s+TO\s+(\d+(?:\.\d+)?)", response, re.IGNORECASE) 
            if match:
                return f"RECOMMEND: RAISE to {match.group(1)}"
            return "RECOMMEND: RAISE"
        
        # If we matched nothing but have available actions, use the first one
        if available_actions:
            return f"RECOMMEND: {available_actions[0]}"
        else:
            return f"RECOMMEND: {response}"
    
    def get_quick_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """
        Get instant decision from precomputed tables without API call
        """
        # Extract key game state elements
        position = game_state['position']
        street = game_state['street']
        pot = game_state['pot']
        
        # Determine hand strength category
        hand_category = self.categorize_hand(
            game_state.get('hero_cards', []), 
            game_state.get('community_cards', [])
        )
        
        # Create lookup key
        pot_ranges = [
            (1, 4), (4, 8), (8, 15), (15, 30), (30, 60), (60, 200)
        ]
        
        pot_range = "1-4"  # Default
        for low, high in pot_ranges:
            if low <= pot <= high:
                pot_range = f"{low}-{high}"
                break
        
        lookup_key = f"{position}:{street}:{pot_range}:{hand_category}"
        
        # Check cache first
        if lookup_key in self.decision_cache:
            logger.info(f"Cache hit for {lookup_key}")
            return f"RECOMMEND: {self.decision_cache[lookup_key]}"
        
        # Then check precomputed decisions
        if lookup_key in self.precomputed_decisions:
            decision = self.precomputed_decisions[lookup_key]
            logger.info(f"Found precomputed decision for {lookup_key}: {decision}")
            
            # Cache for future use
            self.decision_cache[lookup_key] = decision
            
            return f"RECOMMEND: {decision}"
        
        # Try more generic lookup (without hand category)
        generic_key = f"{position}:{street}:{pot_range}"
        if generic_key in self.precomputed_decisions:
            decision = self.precomputed_decisions[generic_key]
            logger.info(f"Found generic precomputed decision for {generic_key}: {decision}")
            
            # Cache for future use
            self.decision_cache[lookup_key] = decision
            
            return f"RECOMMEND: {decision}"
        
        return None  # No precomputed decision found
    
    def make_decision(self, image: np.ndarray, tournament_mode: bool = False) -> Dict[str, Any]:
        """
        Make a poker decision based on the current table image
        
        Args:
            image: Screenshot of poker table
            tournament_mode: Whether in tournament mode
            
        Returns:
            Dict with decision and analysis
        """
        start_time = time.time()
        
        # Analyze table image
        game_state = self.analyzer.analyze_table(image)
        
        # Try to get quick decision from precomputed tables
        quick_decision = self.get_quick_decision(game_state)
        if quick_decision:
            processing_time = time.time() - start_time
            logger.info(f"Quick decision made in {processing_time:.3f}s: {quick_decision}")
            
            return {
                "suggestion": quick_decision,
                "game_state": game_state,
                "processing_time": processing_time,
                "method": "precomputed"
            }
        
        # If no quick decision, use API
        assistant_num = self.select_assistant(game_state, tournament_mode)
        prompt = self.format_prompt(game_state)
        prompt = f"{assistant_num}. {prompt}"
        
        raw_response = self.get_model_response(assistant_num, prompt)
        normalized_response = self.normalize_response(raw_response, game_state.get("available_actions", ["FOLD", "CALL", "RAISE"]))
        
        # Cache this decision for future use
        lookup_key = f"{game_state['position']}:{game_state['street']}:{game_state['pot']}"
        self.decision_cache[lookup_key] = normalized_response.replace("RECOMMEND: ", "")
        
        processing_time = time.time() - start_time
        logger.info(f"API decision made in {processing_time:.3f}s: {normalized_response}")
        
        return {
            "suggestion": normalized_response,
            "game_state": game_state,
            "processing_time": processing_time,
            "method": "api"
        }

class AsyncPokerProcessor:
    """
    Asynchronous poker decision processor with background thread processing
    """
    
    def __init__(self):
        self.decision_engine = PokerDecisionEngine("rondaygo")
        self.last_decision = None
        self.decision_queue = deque(maxlen=3)  # Keep last 3 decisions for analysis
        self.processing_thread = None
        self.processing_lock = threading.Lock()
        
    def process_image_async(self, image_file):
        """Process poker image asynchronously for quick response"""
        # Convert image file to numpy array
        image = Image.open(image_file)
        image_np = np.array(image)
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(
            target=self._process_image_thread,
            args=(image_np,)
        )
        self.processing_thread.start()
        
        # If we have a previous decision, return it immediately
        # This ensures the UI is always responsive
        if self.last_decision:
            return self.last_decision
            
        # Otherwise, wait briefly for the new decision
        self.processing_thread.join(timeout=0.5)
        
        # Return whatever we have now
        with self.processing_lock:
            if self.last_decision:
                return self.last_decision
            else:
                # If still no decision, return a default
                return {
                    "suggestion": "RECOMMEND: CALL",
                    "game_state": {"position": "unknown", "street": "unknown"},
                    "processing_time": 0.0,
                    "method": "default",
                    "note": "Fast response while analyzing"
                }
    
    def _process_image_thread(self, image_np):
        """Process image in background thread"""
        try:
            # Get decision
            decision = self.decision_engine.make_decision(image_np)
            
            # Store in decision history queue
            self.decision_queue.append(decision)
            
            # Store as last decision
            with self.processing_lock:
                self.last_decision = decision
                
        except Exception as e:
            logger.error(f"Error in background processing: {str(e)}")

def create_app():
    app = Flask(__name__)
    
    # Initialize the async processor
    processor = AsyncPokerProcessor()
    
    @app.route("/api/advice", methods=["POST"])
    def api_advice():
        logger.info(f"Received API request from {request.remote_addr}")
        
        try:
            if request.files:
                # Get the image file
                file_key = next(iter(request.files))
                image_file = request.files[file_key]
                
                # Process asynchronously for speed
                result = processor.process_image_async(image_file)
                
                return jsonify(result)
            else:
                return jsonify({"error": "No image file found in request"}), 400
                
        except Exception as e:
            logger.exception(f"Error processing request: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return app

if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Advanced Poker Analysis Server")
    app.run(debug=True, host="0.0.0.0", port=5000)
