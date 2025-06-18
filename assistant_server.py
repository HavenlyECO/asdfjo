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
        '♣': 'clubs', '♦': 'diamonds', '♥': 'hearts', '♠': 'spades'
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
        
        # Ensure coordinates are within image bounds
        x1_px = max(0, min(x1_px, w-1))
        y1_px = max(0, min(y1_px, h-1))
        x2_px = max(0, min(x2_px, w-1))
        y2_px = max(0, min(y2_px, h-1))
        
        # Ensure x2 > x1 and y2 > y1
        if x2_px <= x1_px: x2_px = x1_px + 1
        if y2_px <= y1_px: y2_px = y1_px + 1
        
        return image[y1_px:y2_px, x1_px:x2_px]

class PokerAnalyzer:
    """Core poker analysis engine"""
    
    # Define comprehensive set of regions for all poker clients
    REGIONS = {
        # General regions
        'pot_area': RegionOfInterest('pot_area', 0.35, 0.25, 0.65, 0.35),
        'community_cards': RegionOfInterest('community_cards', 0.25, 0.35, 0.75, 0.45),
        'hero_cards': RegionOfInterest('hero_cards', 0.35, 0.65, 0.65, 0.8),
        'buttons': RegionOfInterest('buttons', 0.1, 0.8, 0.9, 1.0),
        'hero_stack': RegionOfInterest('hero_stack', 0.35, 0.6, 0.65, 0.7),
        'hero_position': RegionOfInterest('hero_position', 0.2, 0.6, 0.8, 0.7),
        'full_table': RegionOfInterest('full_table', 0.0, 0.0, 1.0, 1.0),
    }
    
    def __init__(self, hero_username="rondaygo"):
        self.hero_username = hero_username.lower()
        
        # Configure Tesseract for optimal poker reading
        self.config = '--oem 1 --psm 6 -l eng'
        
        # Compile regex patterns for performance
        self.patterns = {
            'pot': re.compile(r'(?:pot|total)[:\s]*(\d+\.?\d*)(?:\s*BB)?', re.IGNORECASE),
            'card': re.compile(r'[2-9TJQKA][cdhs♣♦♥♠]'),
            'stack': re.compile(r'(\d+\.?\d*)\s*(?:BB|bb)'),
            'action': re.compile(r'(fold|check|call|raise|bet|all[\s-]*in)', re.IGNORECASE),
            'position': re.compile(r'(SB|BB|BTN|CO|MP|UTG)', re.IGNORECASE),
            'username': re.compile(rf'{re.escape(hero_username)}', re.IGNORECASE)
        }
    
    def process_region(self, region_name: str, image: np.ndarray) -> Dict[str, Any]:
        """Process a specific region of the poker table image"""
        # Get the region definition
        if region_name not in self.REGIONS:
            logger.warning(f"Unknown region: {region_name}")
            return {"error": f"Unknown region: {region_name}"}
            
        region = self.REGIONS[region_name]
        
        # Extract the region from the image
        try:
            region_image = region.extract_from_image(image)
        except Exception as e:
            logger.error(f"Error extracting region {region_name}: {e}")
            return {"error": f"Error extracting region: {str(e)}"}
        
        # Enhance for OCR based on region type
        enhanced = self.enhance_for_region(region_name, region_image)
        
        # Perform OCR
        try:
            text = pytesseract.image_to_string(enhanced, config=self.config)
        except Exception as e:
            logger.error(f"OCR error in region {region_name}: {e}")
            text = ""
        
        # Process based on region type
        result = {"region": region_name, "text": text}
        
        if region_name == 'pot_area':
            pot_match = self.patterns['pot'].search(text)
            if pot_match:
                try:
                    result["pot"] = float(pot_match.group(1))
                except ValueError:
                    pass
            
            # Backup method: find any number in the text
            if "pot" not in result:
                numbers = re.findall(r'(\d+\.?\d*)', text)
                if numbers:
                    try:
                        result["pot"] = max(float(n) for n in numbers)
                    except ValueError:
                        pass
        
        elif region_name in ('community_cards', 'hero_cards'):
            result["cards"] = self.patterns['card'].findall(text)
            
            # If OCR fails to find cards, try image-based card detection
            if not result["cards"] and region_name == 'community_cards':
                result["cards"] = self.detect_cards_by_color(region_image)
        
        elif region_name == 'buttons':
            result["available_actions"] = []
            for action in ('fold', 'check', 'call', 'raise', 'bet', 'all-in'):
                if action in text.lower():
                    result["available_actions"].append(action.upper())
        
        elif region_name == 'hero_stack':
            stack_match = self.patterns['stack'].search(text)
            if stack_match:
                try:
                    result["hero_stack"] = float(stack_match.group(1))
                except ValueError:
                    pass
        
        elif region_name == 'hero_position':
            position_match = self.patterns['position'].search(text)
            if position_match:
                result["position"] = position_match.group(1).upper()
            elif 'dealer' in text.lower() or 'btn' in text.lower():
                result["position"] = 'BTN'
            elif 'small' in text.lower() or 'sb' in text.lower():
                result["position"] = 'SB'
            elif 'big' in text.lower() or 'bb' in text.lower():
                result["position"] = 'BB'
        
        elif region_name == 'full_table':
            if self.hero_username.lower() in text.lower():
                result["hero_found"] = True
            else:
                result["hero_found"] = False
        
        return result
    
    def enhance_for_region(self, region_name: str, image: np.ndarray) -> np.ndarray:
        """Apply specific image enhancements based on region type"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Region-specific enhancements
        if region_name in ('community_cards', 'hero_cards'):
            # Strong contrast for card detection
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, enhanced = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        elif region_name in ('pot_area', 'hero_stack'):
            # Optimize for number detection
            enhanced = cv2.GaussianBlur(gray, (3, 3), 0)
            enhanced = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        
        elif region_name == 'buttons':
            # Enhance button text
            enhanced = cv2.equalizeHist(gray)
            
        else:
            # Default enhancement for other regions
            enhanced = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
        return enhanced
    
    def detect_cards_by_color(self, image: np.ndarray) -> List[str]:
        """Detect cards using color segmentation (backup method)"""
        # This is a simplified implementation
        # In a real system, this would use computer vision to identify cards
        # when OCR fails
        return []
    
    def detect_street(self, community_cards: List[str]) -> str:
        """Determine street based on number of community cards"""
        if not community_cards:
            return 'preflop'
        elif len(community_cards) == 3:
            return 'flop'
        elif len(community_cards) == 4:
            return 'turn'
        elif len(community_cards) >= 5:
            return 'river'
        else:
            return 'preflop'  # Default
    
    def analyze_table(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the poker table image and extract game state
        """
        # Initialize game state with default values
        game_state = {
            "position": "BTN",
            "street": "preflop",
            "pot": 1.5,
            "hero_stack": 100.0,
            "hero_cards": [],
            "community_cards": [],
            "available_actions": ["FOLD", "CALL", "RAISE"],
            "hero_username": self.hero_username
        }
        
        # Process each region independently
        results = {}
        for region_name in self.REGIONS:
            results[region_name] = self.process_region(region_name, image)
        
        # Update game state with extracted information
        if "pot" in results.get('pot_area', {}):
            game_state["pot"] = results['pot_area']["pot"]
            
        if "cards" in results.get('hero_cards', {}):
            game_state["hero_cards"] = results['hero_cards']["cards"]
            
        if "cards" in results.get('community_cards', {}):
            game_state["community_cards"] = results['community_cards']["cards"]
            game_state["street"] = self.detect_street(results['community_cards']["cards"])
            
        if "hero_stack" in results.get('hero_stack', {}):
            game_state["hero_stack"] = results['hero_stack']["hero_stack"]
            
        if "position" in results.get('hero_position', {}):
            game_state["position"] = results['hero_position']["position"]
            
        if "available_actions" in results.get('buttons', {}):
            actions = results['buttons']["available_actions"]
            if actions:  # Only update if we found actions
                game_state["available_actions"] = actions
        
        # Log analysis results
        logger.info(f"Table analysis: {game_state['position']} on {game_state['street']}, pot={game_state['pot']}BB")
        logger.info(f"Hero cards: {game_state['hero_cards']}, Board: {game_state['community_cards']}")
        logger.info(f"Available actions: {game_state['available_actions']}")
        
        return game_state

class HandEvaluator:
    """Evaluates poker hand strength"""
    
    def categorize_hand(self, hero_cards: List[str], community_cards: List[str]) -> str:
        """
        Categorize hand strength
        
        Returns: "premium", "strong", "medium", or "weak"
        """
        if not hero_cards:
            return "unknown"
            
        # Convert string representations to PokerCard objects
        try:
            hole_cards = [PokerCard(card) for card in hero_cards]
        except:
            return "unknown"
        
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
            # Simplified approach for example
            if len(hole_cards) >= 2 and hole_cards[0].rank >= 10 and hole_cards[1].rank >= 10:
                return "strong"
            elif len(hole_cards) >= 2 and (hole_cards[0].rank >= 10 or hole_cards[1].rank >= 10):
                return "medium"
            return "weak"

class PokerDecisionEngine:
    """Poker decision engine"""
    
    def __init__(self, hero_username="rondaygo"):
        self.analyzer = PokerAnalyzer(hero_username)
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
    
    def select_assistant(self, game_state: Dict) -> int:
        """Select the appropriate assistant based on the game state."""
        street = str(game_state.get("street", "")).lower()
        
        # ASSISTANT_1: Preflop Position Assistant
        if street == "preflop":
            return 1
            
        # ASSISTANT_4: Board Texture Assistant
        if street in ["flop", "turn", "river"] and game_state.get("community_cards", []):
            return 4
            
        # ASSISTANT_5: Pot Odds Assistant
        if street in ["flop", "turn", "river"] and game_state.get("pot", 0) > 0:
            return 5
            
        # Default to preflop assistant
        return 1
    
    def format_prompt(self, game_state: Dict) -> str:
        """Format game state into natural language prompt."""
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
    
    def normalize_response(self, response: str, available_actions: List[str]) -> str:
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
            return "RECOMMEND: RAISE"
        
        # If we matched nothing but have available actions, use a default
        if "FOLD" in available_actions:
            return "RECOMMEND: FOLD"
        elif "CHECK" in available_actions:
            return "RECOMMEND: CHECK"
        elif "CALL" in available_actions:
            return "RECOMMEND: CALL"
        
        # Default fallback
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
        hand_category = self.evaluator.categorize_hand(
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
            
            # Ensure decision is compatible with available actions
            available_actions = [a.upper() for a in game_state.get('available_actions', ['FOLD', 'CALL', 'RAISE'])]
            
            decision_action = decision.split()[0].upper()
            
            if decision_action in available_actions:
                return f"RECOMMEND: {decision}"
            else:
                # Try to find a compatible action
                if "FOLD" in available_actions:
                    return "RECOMMEND: FOLD"
                elif "CHECK" in available_actions:
                    return "RECOMMEND: CHECK"
                elif "CALL" in available_actions:
                    return "RECOMMEND: CALL"
        
        return None
    
    def make_decision(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Make a poker decision based on the current table image
        """
        start_time = time.time()
        
        # Analyze table image
        game_state = self.analyzer.analyze_table(image_np)
        
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
        assistant_num = self.select_assistant(game_state)
        prompt = self.format_prompt(game_state)
        prompt = f"{assistant_num}. {prompt}"
        
        raw_response = self.get_model_response(assistant_num, prompt)
        normalized_response = self.normalize_response(raw_response, game_state.get("available_actions", ["FOLD", "CALL", "RAISE"]))
        
        processing_time = time.time() - start_time
        logger.info(f"API decision made in {processing_time:.3f}s: {normalized_response}")
        
        return {
            "suggestion": normalized_response,
            "game_state": game_state,
            "processing_time": processing_time,
            "method": "api"
        }

class PokerProcessor:
    """Main poker processing class"""
    
    def __init__(self):
        self.decision_engine = PokerDecisionEngine("rondaygo")
        self.last_result = None
        self.last_time = time.time()
        self.processing = False
        self.lock = threading.Lock()
        
    def process_image(self, image_file):
        """Process a poker image and make a decision"""
        with self.lock:
            self.processing = True
        
        try:
            # Convert image file to numpy array
            image = Image.open(image_file)
            image_np = np.array(image)
            
            # Make decision
            result = self.decision_engine.make_decision(image_np)
            
            # Update last result and time
            with self.lock:
                self.last_result = result
                self.last_time = time.time()
                self.processing = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            with self.lock:
                self.processing = False
            raise

def create_app():
    app = Flask(__name__)
    
    # Initialize processor
    processor = PokerProcessor()
    
    @app.route("/api/advice", methods=["POST"])
    def api_advice():
        logger.info(f"Received API request from {request.remote_addr}")
        
        try:
            if request.files:
                # Get the image file
                file_key = next(iter(request.files))
                image_file = request.files[file_key]
                
                # Process image
                result = processor.process_image(image_file)
                
                return jsonify(result)
            else:
                return jsonify({"error": "No image file found in request"}), 400
                
        except Exception as e:
            logger.exception(f"Error processing request: {str(e)}")
            return jsonify({"error": str(e), "trace": str(traceback.format_exc())}), 500
    
    return app

if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Poker Analysis Server")
    app.run(debug=True, host="0.0.0.0", port=5000)
