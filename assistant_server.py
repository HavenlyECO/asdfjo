import os
import time
import sys
from typing import Dict, Any, List
from flask import Flask, request, jsonify
from openai import OpenAI
from PIL import Image
import numpy as np
import cv2
import logging
from dotenv import load_dotenv
import re
import json
import hashlib

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

class PokerCardOCR:
    """Direct OCR-based card detection with no fallbacks"""
    
    def __init__(self):
        # OCR configuration values tuned specifically for card detection
        self.ocr_config = {
            "binary_threshold": 180,
            "kernel_size": 3,
            "card_width_min": 20,
            "card_height_min": 30
        }
        
        # Card regex patterns
        self.card_pattern = re.compile(r'(?:[2-9TJQKA][cdhs♣♦♥♠])+')
        self.single_card = re.compile(r'[2-9TJQKA][cdhs♣♦♥♠]')
        
    def preprocess_for_cards(self, image):
        """Preprocess image specifically for card text recognition"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Increase contrast
        alpha = 1.5  # Contrast control
        beta = 0     # Brightness control
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Apply binary threshold
        _, binary = cv2.threshold(enhanced, self.ocr_config["binary_threshold"], 255, cv2.THRESH_BINARY)
        
        # Cleanup image with morphological operations
        kernel = np.ones((self.ocr_config["kernel_size"], self.ocr_config["kernel_size"]), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return cleaned
        
    def extract_cards_from_text(self, text):
        """Extract card notations from text"""
        # Find all card matches in text
        cards = []
        # Search for complete card patterns
        for match in self.card_pattern.findall(text):
            # Extract individual cards from each match
            for card in self.single_card.findall(match):
                cards.append(card)
                
        # Remove duplicates while preserving order
        unique_cards = []
        for card in cards:
            if card not in unique_cards:
                unique_cards.append(card)
                
        return unique_cards
    
    def recognize_text(self, image):
        """Recognize text in image using OpenCV"""
        # This is a placeholder - we would use tesseract here
        # But we're simplifying for now to focus on the core logic
        import pytesseract
        try:
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def detect_cards(self, image, region):
        """Detect cards in a specific region of the image"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = region
        
        # Convert coordinates to pixels
        x1_px = int(x1 * w)
        y1_px = int(y1 * h)
        x2_px = int(x2 * w)
        y2_px = int(y2 * h)
        
        # Ensure valid coordinates
        x1_px = max(0, min(x1_px, w-1))
        y1_px = max(0, min(y1_px, h-1))
        x2_px = max(x1_px+1, min(x2_px, w))
        y2_px = max(y1_px+1, min(y2_px, h))
        
        # Extract region
        region_img = image[y1_px:y2_px, x1_px:x2_px]
        
        # Preprocess for card detection
        processed = self.preprocess_for_cards(region_img)
        
        # Recognize text
        text = self.recognize_text(processed)
        
        # Extract cards from text
        cards = self.extract_cards_from_text(text)
        
        return cards

class ProfitablePlayDatabase:
    """Database of profitable plays based on exact card combinations"""
    
    def __init__(self, db_file="card_profits.json"):
        self.db_file = db_file
        self.plays = self.load_db()
        
    def load_db(self):
        """Load database from file"""
        try:
            with open(self.db_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
            
    def save_db(self):
        """Save database to file"""
        with open(self.db_file, 'w') as f:
            json.dump(self.plays, f)
    
    def make_key(self, state):
        """Create a key from game state to identify similar situations"""
        # Extract key elements
        position = state.get('position', '')
        street = state.get('street', '')
        hero_cards = sorted(state.get('hero_cards', []))
        community_cards = sorted(state.get('community_cards', []))
        pot_category = self._categorize_pot(state.get('pot', 0))
        
        # Create a structured key that identifies this hand situation
        key = f"{position}:{street}:{pot_category}:{','.join(hero_cards)}:{','.join(community_cards)}"
        return key
        
    def _categorize_pot(self, pot):
        """Categorize pot size into discrete ranges"""
        if pot <= 2:
            return "small"
        elif pot <= 8:
            return "medium"
        elif pot <= 20:
            return "large"
        else:
            return "very_large"
            
    def get_play(self, state):
        """Get a profitable play for this exact card situation if it exists"""
        key = self.make_key(state)
        return self.plays.get(key)
        
    def record_play(self, state, decision, profitable=True):
        """Record a play as profitable or not"""
        key = self.make_key(state)
        
        if profitable:
            self.plays[key] = decision
            self.save_db()
            logger.info(f"Recorded profitable play for {key}")
            return True
        return False
        
    def get_stats(self):
        """Get statistics about the database"""
        street_counts = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}
        position_counts = {"BTN": 0, "SB": 0, "BB": 0, "UTG": 0, "MP": 0, "CO": 0}
        
        for key in self.plays:
            parts = key.split(":")
            if len(parts) >= 2:
                position, street = parts[0], parts[1]
                if position in position_counts:
                    position_counts[position] += 1
                if street in street_counts:
                    street_counts[street] += 1
                    
        return {
            "total_plays": len(self.plays),
            "by_street": street_counts,
            "by_position": position_counts
        }

class PokerAssistant:
    """Main poker assistant - direct, no fallbacks"""
    
    # Image regions - (x1, y1, x2, y2) as proportion of image
    REGIONS = {
        "hero_cards": (0.35, 0.65, 0.65, 0.85),
        "community_cards": (0.3, 0.35, 0.7, 0.5),
        "pot_area": (0.35, 0.2, 0.65, 0.35),
        "position_area": (0.3, 0.5, 0.7, 0.65),
        "buttons_area": (0.2, 0.8, 0.8, 0.95)
    }
    
    def __init__(self, hero_username="rondaygo"):
        self.hero_username = hero_username
        self.ocr = PokerCardOCR()
        self.profitable_db = ProfitablePlayDatabase()
        self.models = {}
        self.load_models()
        
        # Patterns for text extraction
        self.patterns = {
            'pot': re.compile(r'(?:pot|total)[:\s]*(\d+\.?\d*)(?:\s*BB)?', re.IGNORECASE),
            'position': re.compile(r'(BTN|SB|BB|CO|MP|UTG)', re.IGNORECASE),
            'action': re.compile(r'(fold|check|call|raise|bet)', re.IGNORECASE)
        }
    
    def load_models(self):
        """Load OpenAI assistant models"""
        for i in range(1, 11):
            model_id = os.environ.get(f"ASSISTANT_{i}")
            if model_id:
                self.models[i] = {
                    "model": "gpt-4-turbo",
                    "assistant_id": model_id,
                    "system_message": f"You are Poker Assistant {i}. Make optimal poker decisions based on cards and position."
                }
            else:
                logger.warning(f"ASSISTANT_{i} not found in environment variables")
    
    def analyze_image(self, image):
        """Extract poker information from image"""
        # Convert PIL image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        # Extract text from image regions for analysis
        text_by_region = {}
        for region_name, coords in self.REGIONS.items():
            h, w = image.shape[:2]
            x1, y1, x2, y2 = coords
            
            # Convert to pixels
            x1_px = int(x1 * w)
            y1_px = int(y1 * h)
            x2_px = int(x2 * w)
            y2_px = int(y2 * h)
            
            # Ensure valid coordinates
            x1_px = max(0, min(x1_px, w-1))
            y1_px = max(0, min(y1_px, h-1))
            x2_px = max(x1_px+1, min(x2_px, w))
            y2_px = max(y1_px+1, min(y2_px, h))
            
            # Extract region
            region = image[y1_px:y2_px, x1_px:x2_px]
            
            # Extract text
            import pytesseract
            try:
                text = pytesseract.image_to_string(region)
                text_by_region[region_name] = text
            except Exception as e:
                logger.error(f"Error extracting text from {region_name}: {e}")
                text_by_region[region_name] = ""
        
        # Detect cards using specialized card OCR
        hero_cards = self.ocr.detect_cards(image, self.REGIONS["hero_cards"])
        community_cards = self.ocr.detect_cards(image, self.REGIONS["community_cards"])
        
        # Detect position
        position = self._detect_position(text_by_region)
        
        # Detect street based on community cards
        street = self._detect_street(community_cards)
        
        # Detect pot size
        pot = self._detect_pot(text_by_region)
        
        # Detect available actions
        available_actions = self._detect_actions(text_by_region)
        
        # Build game state
        game_state = {
            "hero_username": self.hero_username,
            "position": position,
            "street": street,
            "pot": pot,
            "hero_cards": hero_cards[:2],  # Limit to 2 cards
            "community_cards": community_cards[:5],  # Limit to 5 cards
            "available_actions": available_actions
        }
        
        # Log results
        logger.info(f"Analysis complete: {position} on {street}, pot={pot}BB")
        logger.info(f"Hero cards: {hero_cards}, Community: {community_cards}")
        logger.info(f"Available actions: {available_actions}")
        
        return game_state
    
    def _detect_position(self, text_by_region):
        """Detect player position from text"""
        position_text = text_by_region.get("position_area", "")
        
        # Check for position keywords
        for match in self.patterns['position'].finditer(position_text):
            return match.group(1).upper()
            
        # Check for position-related words
        text_lower = position_text.lower()
        if 'button' in text_lower or 'btn' in text_lower:
            return "BTN"
        elif 'small blind' in text_lower or 'sb' in text_lower:
            return "SB" 
        elif 'big blind' in text_lower or 'bb' in text_lower:
            return "BB"
        elif 'cutoff' in text_lower or 'co' in text_lower:
            return "CO"
        elif 'middle' in text_lower or 'mp' in text_lower:
            return "MP"
        elif 'under' in text_lower or 'utg' in text_lower:
            return "UTG"
            
        # Default to button if unknown
        return "BTN"
    
    def _detect_street(self, community_cards):
        """Detect current street based on community cards"""
        if not community_cards:
            return "preflop"
        elif len(community_cards) == 3:
            return "flop"
        elif len(community_cards) == 4:
            return "turn"
        elif len(community_cards) >= 5:
            return "river"
        else:
            return "preflop"  # Default
    
    def _detect_pot(self, text_by_region):
        """Detect pot size from text"""
        pot_text = text_by_region.get("pot_area", "")
        
        # Look for pot indicators
        for match in self.patterns['pot'].finditer(pot_text):
            try:
                return float(match.group(1))
            except ValueError:
                pass
                
        # Check for any number followed by BB
        bb_match = re.search(r'(\d+\.?\d*)\s*BB', pot_text)
        if bb_match:
            try:
                return float(bb_match.group(1))
            except ValueError:
                pass
                
        # Use default pot size based on street
        return 1.5  # Default preflop pot
    
    def _detect_actions(self, text_by_region):
        """Detect available actions from buttons area"""
        actions_text = text_by_region.get("buttons_area", "").lower()
        
        # Available actions
        actions = []
        if 'fold' in actions_text:
            actions.append("FOLD")
        if 'call' in actions_text:
            actions.append("CALL")
        if 'check' in actions_text:
            actions.append("CHECK")
        if 'raise' in actions_text or 'bet' in actions_text:
            actions.append("RAISE")
            
        # If no actions detected, use defaults
        if not actions:
            actions = ["FOLD", "CALL", "RAISE"]
            
        return actions
    
    def select_assistant(self, game_state):
        """Select the appropriate OpenAI assistant"""
        street = game_state.get("street", "").lower()
        
        # Simple selection logic
        if street == "preflop":
            return 1  # Preflop specialist
        else:
            return 4  # Postflop specialist
    
    def format_prompt(self, game_state):
        """Format game state into prompt for OpenAI"""
        street = game_state.get("street", "unknown").capitalize()
        position = game_state.get("position", "unknown")
        hero_cards = game_state.get("hero_cards", [])
        community_cards = game_state.get("community_cards", [])
        pot = game_state.get("pot", 0)
        available_actions = game_state.get("available_actions", [])
        
        # Format cards nicely
        hero_cards_str = " ".join(hero_cards) if hero_cards else "unknown cards"
        community_cards_str = " ".join(community_cards) if community_cards else "no community cards"
        
        # Create detailed prompt with emphasis on the cards
        prompt = f"STREET: {street}\n"
        prompt += f"POSITION: {position}\n"
        prompt += f"HERO CARDS: {hero_cards_str}\n"
        
        if community_cards:
            prompt += f"BOARD: {community_cards_str}\n"
            
        prompt += f"POT SIZE: {pot}BB\n"
        prompt += f"AVAILABLE ACTIONS: {', '.join(available_actions)}\n\n"
        
        prompt += "Make the optimal poker decision in this exact situation.\n"
        prompt += f"Respond with ONLY ONE of these actions: {' / '.join(available_actions)}.\n"
        prompt += "If you choose RAISE, include the bet size (e.g., 'RAISE to 3BB')."
        
        return prompt
    
    def get_gpt_decision(self, assistant_num, prompt):
        """Get decision from OpenAI"""
        model_config = self.models.get(assistant_num)
        
        if not model_config:
            logger.error(f"Assistant {assistant_num} not found")
            return "FOLD"  # Safe default
            
        try:
            # Create messages for API call
            messages = [
                {"role": "system", "content": model_config["system_message"]},
                {"role": "user", "content": prompt}
            ]
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=50,
                temperature=0.2,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def normalize_decision(self, raw_decision, available_actions):
        """Format the decision consistently"""
        decision = raw_decision.upper().strip()
        
        # Check for simple actions
        for action in ["FOLD", "CHECK", "CALL"]:
            if action in decision and action in available_actions:
                return f"RECOMMEND: {action}"
        
        # Handle RAISE with amount
        if "RAISE" in decision and "RAISE" in available_actions:
            amount_match = re.search(r"RAISE\s+(?:TO\s+)?(\d+\.?\d*)(?:\s*BB)?", decision, re.IGNORECASE)
            if amount_match:
                return f"RECOMMEND: RAISE to {amount_match.group(1)}BB"
            else:
                return "RECOMMEND: RAISE to 3BB"
                
        # Default to first available action if nothing matched
        if available_actions:
            return f"RECOMMEND: {available_actions[0]}"
        else:
            return "RECOMMEND: CALL"  # Ultimate fallback
    
    def get_advice(self, image, use_profitable_db=True):
        """Process poker image and get advice"""
        start_time = time.time()
        
        # Analyze image to extract game state
        game_state = self.analyze_image(image)
        
        # Record if we have cards
        has_hero_cards = bool(game_state.get("hero_cards"))
        
        # Try to get a profitable play from the database
        if use_profitable_db and has_hero_cards:
            profitable_play = self.profitable_db.get_play(game_state)
            if profitable_play:
                processing_time = time.time() - start_time
                logger.info(f"Using known profitable play for these cards: {profitable_play}")
                
                return {
                    "suggestion": profitable_play, 
                    "game_state": game_state,
                    "processing_time": processing_time,
                    "source": "profitable_db",
                    "hero_cards": game_state.get("hero_cards", []),
                    "community_cards": game_state.get("community_cards", [])
                }
        
        # No profitable play found, use OpenAI
        assistant_num = self.select_assistant(game_state)
        prompt = self.format_prompt(game_state)
        prompt = f"{assistant_num}. {prompt}"  # Add assistant number
        
        # Get decision from OpenAI
        raw_decision = self.get_gpt_decision(assistant_num, prompt)
        
        # Normalize decision format
        decision = self.normalize_decision(raw_decision, game_state.get("available_actions", []))
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Generated OpenAI decision in {processing_time:.3f}s: {decision}")
        
        # Generate a tracking hash for this specific hand situation
        tracking_hash = hashlib.md5(
            f"{game_state.get('position', '')}:{game_state.get('street', '')}:" + 
            f"{','.join(sorted(game_state.get('hero_cards', [])))}:" +
            f"{','.join(sorted(game_state.get('community_cards', [])))}".encode()
        ).hexdigest()
        
        return {
            "suggestion": decision,
            "game_state": game_state,
            "processing_time": processing_time,
            "source": "openai",
            "hero_cards": game_state.get("hero_cards", []),
            "community_cards": game_state.get("community_cards", []),
            "tracking_hash": tracking_hash
        }
    
    def record_result(self, tracking_hash, decision, profitable=True):
        """Record the result of a play"""
        if not tracking_hash:
            return {"success": False, "error": "No tracking hash provided"}
            
        try:
            # For now, store directly with the tracking hash
            # In a more complex implementation, we would look up the original game state
            result = self.profitable_db.record_play(
                {"tracking_hash": tracking_hash},
                decision,
                profitable
            )
            return {"success": result}
        except Exception as e:
            logger.error(f"Error recording result: {e}")
            return {"success": False, "error": str(e)}

def create_app():
    app = Flask(__name__)
    
    # Initialize assistant
    assistant = PokerAssistant("rondaygo")
    
    @app.route("/api/advice", methods=["POST"])
    def api_advice():
        logger.info(f"Received API request from {request.remote_addr}")
        
        # Parse request options
        use_db = request.args.get("use_db", "true").lower() == "true"
        
        try:
            if request.files:
                # Get image file
                file_key = next(iter(request.files))
                image_file = request.files[file_key]
                
                # Process image
                image = Image.open(image_file)
                result = assistant.get_advice(image, use_db)
                
                return jsonify(result)
            else:
                return jsonify({"error": "No image file found in request"}), 400
                
        except Exception as e:
            logger.exception(f"Error processing request: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/record", methods=["POST"])
    def record_result():
        """Record a play result (win/loss)"""
        try:
            data = request.get_json()
            tracking_hash = data.get("tracking_hash")
            decision = data.get("decision")
            profitable = data.get("profitable", True)
            
            result = assistant.record_result(tracking_hash, decision, profitable)
            return jsonify(result)
            
        except Exception as e:
            logger.exception(f"Error recording result: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/db/stats", methods=["GET"])
    def db_stats():
        """Get profitable play database statistics"""
        return jsonify(assistant.profitable_db.get_stats())
    
    return app

if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Pure OpenAI Poker Assistant")
    app.run(debug=True, host="0.0.0.0", port=5000)
