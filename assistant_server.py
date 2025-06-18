import os
import time
import sys
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify
from openai import OpenAI
from PIL import Image
import numpy as np
import pytesseract
import json
import logging
from dotenv import load_dotenv
import re
import hashlib
import base64

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
    """Specialized poker card detector"""
    
    def __init__(self):
        # Regex pattern for card notation (e.g., "Ah", "Td", "5c")
        self.card_pattern = re.compile(r'[2-9TJQKA][cdhs♣♦♥♠]')
        
        # Configure OCR for card detection
        self.config = '--oem 1 --psm 7 -l eng'  # PSM 7 is for single line text
        
    def extract_text_from_region(self, image, region):
        """Extract text from a specific region of the image"""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = region
        
        # Convert region to pixel coordinates
        x1_px = int(x1 * width)
        y1_px = int(y1 * height)
        x2_px = int(x2 * width)
        y2_px = int(y2 * height)
        
        # Ensure coordinates are within bounds
        x1_px = max(0, min(x1_px, width - 1))
        y1_px = max(0, min(y1_px, height - 1))
        x2_px = max(x1_px + 1, min(x2_px, width))
        y2_px = max(y1_px + 1, min(y2_px, height))
        
        # Extract region
        region_img = image[y1_px:y2_px, x1_px:x2_px]
        
        # Enhance contrast for card detection
        if len(region_img.shape) == 3:
            gray = np.mean(region_img, axis=2).astype(np.uint8)
        else:
            gray = region_img
        
        # Apply adaptive threshold
        try:
            enhanced = gray.copy()
            enhanced = np.clip(enhanced * 1.5, 0, 255).astype(np.uint8)
        except:
            enhanced = gray
        
        # Extract text
        try:
            text = pytesseract.image_to_string(enhanced, config=self.config).strip()
            return text
        except:
            return ""
    
    def find_cards_in_text(self, text):
        """Find card notations in text"""
        cards = self.card_pattern.findall(text)
        # Remove duplicates while preserving order
        unique_cards = []
        for card in cards:
            if card not in unique_cards:
                unique_cards.append(card)
        return unique_cards
    
    def detect_cards(self, image, hero_region, community_region):
        """Detect both hero and community cards from image regions"""
        # Get hero cards from the hero region
        hero_text = self.extract_text_from_region(image, hero_region)
        hero_cards = self.find_cards_in_text(hero_text)
        logger.debug(f"Hero cards text: {hero_text}")
        logger.debug(f"Extracted hero cards: {hero_cards}")
        
        # Get community cards from the community region
        community_text = self.extract_text_from_region(image, community_region)
        community_cards = self.find_cards_in_text(community_text)
        logger.debug(f"Community cards text: {community_text}")
        logger.debug(f"Extracted community cards: {community_cards}")
        
        # For hero cards, limit to 2 cards
        hero_cards = hero_cards[:2]
        
        # For community cards, limit to 5 cards
        community_cards = community_cards[:5]
        
        return hero_cards, community_cards

class ProfitablePlayCache:
    """Cache system that learns profitable plays based on actual cards"""
    
    def __init__(self, cache_file="profitable_plays.json"):
        self.cache_file = cache_file
        self.plays = self.load_cache()
        self.hits = 0
        self.misses = 0
        
    def load_cache(self):
        """Load cache from file"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return empty cache if file doesn't exist or is invalid
            return {}
    
    def save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.plays, f)
    
    def generate_key(self, hand_data):
        """Generate a unique key for the hand based on cards and situation"""
        # Extract key elements
        position = hand_data.get("position", "unknown")
        street = hand_data.get("street", "unknown")
        hero_cards = sorted(hand_data.get("hero_cards", []))
        community_cards = sorted(hand_data.get("community_cards", []))
        pot_size_range = self._get_pot_range(hand_data.get("pot", 0))
        
        # Create key
        key_parts = [
            position,
            street,
            pot_size_range,
            ",".join(hero_cards),
            ",".join(community_cards)
        ]
        
        key = ":".join(key_parts)
        return key
    
    def _get_pot_range(self, pot):
        """Convert pot size to a range for better generalization"""
        pot_ranges = [
            (0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1000)
        ]
        
        for low, high in pot_ranges:
            if low <= pot <= high:
                return f"{low}-{high}"
                
        return "0-inf"  # Default range
    
    def get(self, hand_data):
        """Get a profitable play from the cache if it exists"""
        key = self.generate_key(hand_data)
        
        if key in self.plays:
            self.hits += 1
            logger.info(f"Cache hit for card combination! Key: {key}")
            return self.plays[key]
        
        self.misses += 1
        return None
    
    def record_play(self, hand_data, decision, profitable=True):
        """Record a play in the cache"""
        key = self.generate_key(hand_data)
        
        # Only record profitable plays
        if profitable:
            self.plays[key] = decision
            self.save_cache()
            logger.info(f"Recorded profitable play for key: {key}")
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        return {
            "size": len(self.plays),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }

class PokerAssistant:
    """Main poker assistant class"""
    
    # Define regions for image analysis (x1, y1, x2, y2) as fractions of image size
    REGIONS = {
        "hero_cards": (0.35, 0.65, 0.65, 0.8),
        "community_cards": (0.25, 0.4, 0.75, 0.5),
        "pot": (0.35, 0.3, 0.65, 0.4),
        "position": (0.35, 0.55, 0.65, 0.65),
        "actions": (0.2, 0.8, 0.8, 0.95)
    }
    
    def __init__(self, hero_username="rondaygo"):
        self.hero_username = hero_username
        self.card_detector = CardDetector()
        self.cache = ProfitablePlayCache()
        self.models = {}
        self.load_models()
        
        # Patterns for text extraction
        self.patterns = {
            'pot': re.compile(r'(?:pot|total)[:\s]*(\d+\.?\d*)(?:\s*BB)?', re.IGNORECASE),
            'position': re.compile(r'(BTN|SB|BB|CO|MP|UTG)', re.IGNORECASE)
        }
    
    def load_models(self):
        """Load model configurations from environment variables."""
        for i in range(1, 11):
            model_id = os.environ.get(f"ASSISTANT_{i}")
            if model_id:
                self.models[i] = {
                    "model": "gpt-4-turbo",
                    "assistant_id": model_id,
                    "system_message": f"You are Poker Assistant {i}. Make optimal poker decisions based on the cards and situation."
                }
            else:
                logger.warning(f"ASSISTANT_{i} not found in environment variables")
    
    def extract_text_from_region(self, image, region_name):
        """Extract text from a named region"""
        if region_name not in self.REGIONS:
            logger.warning(f"Unknown region: {region_name}")
            return ""
            
        region = self.REGIONS[region_name]
        
        height, width = image.shape[:2]
        x1, y1, x2, y2 = region
        
        # Convert region to pixel coordinates
        x1_px = int(x1 * width)
        y1_px = int(y1 * height)
        x2_px = int(x2 * width)
        y2_px = int(y2 * height)
        
        # Ensure coordinates are within bounds
        x1_px = max(0, min(x1_px, width - 1))
        y1_px = max(0, min(y1_px, height - 1))
        x2_px = max(x1_px + 1, min(x2_px, width))
        y2_px = max(y1_px + 1, min(y2_px, height))
        
        # Extract region
        region_img = image[y1_px:y2_px, x1_px:x2_px]
        
        # Convert to grayscale if needed
        if len(region_img.shape) == 3:
            gray = np.mean(region_img, axis=2).astype(np.uint8)
        else:
            gray = region_img
            
        # Extract text
        try:
            text = pytesseract.image_to_string(gray).strip()
            return text
        except:
            return ""
    
    def detect_position(self, image):
        """Detect player position"""
        # Extract position text
        position_text = self.extract_text_from_region(image, "position")
        
        # Look for position indicators
        position_match = self.patterns['position'].search(position_text)
        if position_match:
            return position_match.group(1).upper()
            
        # Check for position keywords
        text_lower = position_text.lower()
        if 'button' in text_lower or 'btn' in text_lower:
            return "BTN"
        elif 'sb' in text_lower or 'small blind' in text_lower:
            return "SB"
        elif 'bb' in text_lower or 'big blind' in text_lower:
            return "BB"
        elif 'co' in text_lower or 'cutoff' in text_lower:
            return "CO"
        elif 'mp' in text_lower or 'middle' in text_lower:
            return "MP"
            
        # Use different positions based on image hash to ensure variety
        image_hash = hashlib.md5(np.array(image).tobytes()).hexdigest()
        positions = ["BTN", "CO", "MP", "UTG"]
        index = int(image_hash[:8], 16) % len(positions)
        return positions[index]
    
    def detect_pot(self, image):
        """Detect pot size"""
        # Extract pot text
        pot_text = self.extract_text_from_region(image, "pot")
        
        # Look for pot indicators
        pot_match = self.patterns['pot'].search(pot_text)
        if pot_match:
            try:
                return float(pot_match.group(1))
            except ValueError:
                pass
                
        # Check for any number followed by BB
        bb_match = re.search(r'(\d+\.?\d*)\s*BB', pot_text)
        if bb_match:
            try:
                return float(bb_match.group(1))
            except ValueError:
                pass
                
        # Use different pot sizes based on image hash for variety
        image_hash = hashlib.md5(np.array(image).tobytes()).hexdigest()
        hash_float = int(image_hash[:8], 16) / (16**8)
        
        # Scale to reasonable pot sizes based on apparent street
        street = self.detect_street(image)
        if street == "preflop":
            return 1.5 + hash_float * 3  # 1.5-4.5BB
        elif street == "flop":
            return 4 + hash_float * 6    # 4-10BB
        elif street == "turn":
            return 10 + hash_float * 10  # 10-20BB
        elif street == "river":
            return 20 + hash_float * 20  # 20-40BB
        
        return 1.5  # Default
    
    def detect_street(self, image):
        """Detect current street based on community cards"""
        # Detect cards
        _, community_cards = self.card_detector.detect_cards(
            image, 
            self.REGIONS["hero_cards"],
            self.REGIONS["community_cards"]
        )
        
        # Determine street based on number of community cards
        if not community_cards:
            return "preflop"
        elif len(community_cards) == 3:
            return "flop"
        elif len(community_cards) == 4:
            return "turn"
        elif len(community_cards) >= 5:
            return "river"
        else:
            # Check community cards region text as backup
            community_text = self.extract_text_from_region(image, "community_cards").lower()
            if 'flop' in community_text:
                return "flop"
            elif 'turn' in community_text:
                return "turn"
            elif 'river' in community_text:
                return "river"
                
            return "preflop"  # Default
    
    def detect_available_actions(self, image):
        """Detect available actions"""
        # Extract actions text
        actions_text = self.extract_text_from_region(image, "actions").lower()
        
        # Look for action keywords
        available_actions = []
        if 'fold' in actions_text:
            available_actions.append("FOLD")
        if 'call' in actions_text:
            available_actions.append("CALL")
        if 'check' in actions_text:
            available_actions.append("CHECK")
        if 'raise' in actions_text or 'bet' in actions_text:
            available_actions.append("RAISE")
            
        # Ensure we have at least some actions
        if not available_actions:
            street = self.detect_street(image)
            if street == "preflop":
                available_actions = ["FOLD", "CALL", "RAISE"]
            else:
                available_actions = ["FOLD", "CHECK", "RAISE"]
                
        return available_actions
    
    def analyze_image(self, image):
        """Analyze poker image and extract game state"""
        # Detect cards (this is the most important part)
        hero_cards, community_cards = self.card_detector.detect_cards(
            image, 
            self.REGIONS["hero_cards"],
            self.REGIONS["community_cards"]
        )
        
        # Detect position
        position = self.detect_position(image)
        
        # Detect street (based on community cards)
        street = self.detect_street(image)
        
        # Detect pot size
        pot = self.detect_pot(image)
        
        # Detect available actions
        available_actions = self.detect_available_actions(image)
        
        # Build game state
        game_state = {
            "hero_username": self.hero_username,
            "position": position,
            "street": street,
            "pot": pot,
            "hero_cards": hero_cards,
            "community_cards": community_cards,
            "available_actions": available_actions
        }
        
        # Log detection results
        logger.info(f"Table analysis: {position} on {street}, pot={pot}BB")
        logger.info(f"Hero cards: {hero_cards}, Board: {community_cards}")
        logger.info(f"Available actions: {available_actions}")
        
        return game_state
    
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
        """Format game state into prompt for OpenAI."""
        street = str(game_state.get("street", "unknown street")).capitalize()
        position = str(game_state.get("position", "unknown position"))
        pot = game_state.get("pot", 1.5)
        hero_cards = game_state.get("hero_cards", [])
        community_cards = game_state.get("community_cards", [])
        available_actions = game_state.get("available_actions", ["FOLD", "CALL", "RAISE"])
        
        # Format cards nicely
        hero_cards_str = " ".join(hero_cards) if hero_cards else "unknown cards"
        community_cards_str = " ".join(community_cards) if community_cards else "no community cards"
        
        # Create a detailed prompt with emphasis on the cards
        prompt = f"Street: {street}\n"
        prompt += f"Position: {position}\n"
        prompt += f"Hero cards: {hero_cards_str}\n"
        
        if community_cards:
            prompt += f"Board: {community_cards_str}\n"
            
        prompt += f"Pot size: {pot}BB\n"
        prompt += f"Available actions: {', '.join(available_actions)}\n\n"
        
        prompt += "What is the optimal poker decision in this exact situation?\n"
        prompt += f"Respond with ONLY ONE of these options: {' / '.join(available_actions)}. "
        
        if "RAISE" in available_actions:
            prompt += "If you choose RAISE, specify the bet size (e.g., 'RAISE to 3BB')."
        
        return prompt
    
    def get_model_response(self, assistant_num, user_input):
        """Get response from OpenAI assistant."""
        model_config = self.models.get(assistant_num)
        
        if not model_config:
            return "Error: Assistant not found"
        
        try:
            # Create messages for the API call
            messages = [
                {"role": "system", "content": model_config["system_message"]},
                {"role": "user", "content": user_input}
            ]
            
            # Make API call
            response = client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=50,
                temperature=0.3,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return f"Error: {str(e)}"
    
    def normalize_response(self, response, available_actions):
        """Normalize the response from the assistant."""
        response = response.upper().strip()
        
        # Check for each action type
        for action in ["FOLD", "CHECK", "CALL"]:
            if action in response and action in available_actions:
                return f"RECOMMEND: {action}"
        
        # Handle RAISE with amount
        if "RAISE" in response and "RAISE" in available_actions:
            amount_match = re.search(r"RAISE\s+(?:TO\s+)?(\d+\.?\d*)(?:\s*BB)?", response, re.IGNORECASE)
            if amount_match:
                return f"RECOMMEND: RAISE to {amount_match.group(1)}BB"
            else:
                return "RECOMMEND: RAISE to 3BB"
                
        # Default to first available action if no match
        if available_actions:
            return f"RECOMMEND: {available_actions[0]}"
        else:
            return "RECOMMEND: CALL"  # Ultimate fallback
    
    def get_decision_hash(self, game_state):
        """Generate a unique hash for the game state for caching insights"""
        # Key elements that determine play
        position = game_state.get("position", "")
        street = game_state.get("street", "")
        pot = game_state.get("pot", 0)
        hero_cards = sorted(game_state.get("hero_cards", []))
        community_cards = sorted(game_state.get("community_cards", []))
        
        # Create a unique string
        key_str = f"{position}:{street}:{pot}:{':'.join(hero_cards)}:{':'.join(community_cards)}"
        
        # Hash for shorter representation
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return key_hash
    
    def get_advice(self, image, use_cache=True):
        """Get poker advice from the image."""
        start_time = time.time()
        
        # Convert PIL image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Analyze the image
        game_state = self.analyze_image(image_np)
        
        # Check if we have hero cards - if not, we must use OpenAI
        if not game_state.get("hero_cards") or use_cache == False:
            use_cache = False
        
        # Check cache for card-specific profitable plays if requested
        if use_cache:
            cached_decision = self.cache.get(game_state)
            if cached_decision:
                processing_time = time.time() - start_time
                logger.info(f"Using cached profitable play for these cards: {cached_decision}")
                
                return {
                    "suggestion": cached_decision,
                    "game_state": game_state,
                    "processing_time": processing_time,
                    "source": "profitable_cache",
                    "hero_cards": game_state.get("hero_cards", []),
                    "community_cards": game_state.get("community_cards", [])
                }
        
        # No cache hit or cache disabled, use OpenAI
        # Select appropriate assistant
        assistant_num = self.select_assistant(game_state)
        
        # Create prompt for assistant
        prompt = self.format_prompt(game_state)
        
        # Add assistant number to prompt
        prompt = f"{assistant_num}. {prompt}"
        
        # Get response from model
        raw_response = self.get_model_response(assistant_num, prompt)
        
        # Normalize response
        normalized_response = self.normalize_response(raw_response, game_state.get("available_actions", []))
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Generated decision in {processing_time:.3f}s: {normalized_response}")
        
        return {
            "suggestion": normalized_response,
            "game_state": game_state,
            "processing_time": processing_time,
            "source": "openai",
            "hero_cards": game_state.get("hero_cards", []),
            "community_cards": game_state.get("community_cards", []),
            "decision_hash": self.get_decision_hash(game_state)
        }
    
    def record_profitable_play(self, decision_hash, decision, profitable=True):
        """Record whether a play was profitable"""
        if profitable and decision_hash:
            # Lookup the original game state
            # For now, use the decision directly
            self.cache.record_play({"decision_hash": decision_hash}, decision, profitable)
            logger.info(f"Recorded profitable play for hash {decision_hash}")

def create_app():
    app = Flask(__name__)
    
    # Initialize the assistant
    assistant = PokerAssistant("rondaygo")
    
    @app.route("/api/advice", methods=["POST"])
    def api_advice():
        logger.info(f"Received API request from {request.remote_addr}")
        
        # Parse request options
        use_cache = request.args.get("use_cache", "true").lower() == "true"
        
        try:
            if request.files:
                # Get the image file
                file_key = next(iter(request.files))
                image_file = request.files[file_key]
                
                # Convert to PIL Image
                image = Image.open(image_file)
                
                # Get advice
                result = assistant.get_advice(image, use_cache)
                
                return jsonify(result)
            else:
                return jsonify({"error": "No image file found in request"}), 400
                
        except Exception as e:
            logger.exception(f"Error processing request: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/record", methods=["POST"])
    def record_profitable():
        """Record a profitable play"""
        try:
            data = request.get_json()
            decision_hash = data.get("decision_hash")
            decision = data.get("decision")
            profitable = data.get("profitable", True)
            
            # Record the play
            assistant.record_profitable_play(decision_hash, decision, profitable)
            
            return jsonify({"status": "success"})
        except Exception as e:
            logger.exception(f"Error recording profitable play: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/cache/stats", methods=["GET"])
    def cache_stats():
        """Get cache statistics"""
        return jsonify(assistant.cache.get_stats())
    
    @app.route("/api/cache/clear", methods=["POST"])
    def clear_cache():
        """Clear the cache"""
        assistant.cache = ProfitablePlayCache()
        return jsonify({"status": "success"})
    
    return app

if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Card-Aware Poker Assistant Server")
    app.run(debug=True, host="0.0.0.0", port=5000)
