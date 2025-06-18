import os
import time
import sys
import threading
from typing import Dict, Any, List
from flask import Flask, request, jsonify
from openai import OpenAI
from PIL import Image
import numpy as np
import pytesseract
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

class PokerImageProcessor:
    """Simple processor to extract key information from poker image"""

    def __init__(self, hero_username="rondaygo"):
        self.hero_username = hero_username
        
        # Configure OCR
        self.config = '--oem 1 --psm 6 -l eng'
        
        # Common patterns
        self.patterns = {
            'pot': re.compile(r'(?:pot|total)[:\s]*(\d+\.?\d*)(?:\s*BB)?', re.IGNORECASE),
            'card': re.compile(r'[2-9TJQKA][cdhs♣♦♥♠]'),
            'position': re.compile(r'(SB|BB|BTN|CO|MP|UTG)', re.IGNORECASE),
            'username': re.compile(rf'{re.escape(hero_username)}', re.IGNORECASE)
        }

    def process_image(self, image):
        """Process poker image to extract key information"""
        # Convert to grayscale for OCR
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        # Use OCR to extract text
        text = pytesseract.image_to_string(gray, config=self.config)
        
        # Initialize game state
        game_state = {
            "hero_username": self.hero_username,
            "position": self._detect_position(text),
            "street": self._detect_street(text),
            "pot": self._detect_pot(text),
            "hero_cards": [],  # Cards aren't reliably detectable with OCR
            "community_cards": [],  # Cards aren't reliably detectable with OCR
            "available_actions": self._detect_actions(text)
        }

        logger.info(f"Extracted state: {game_state['position']} on {game_state['street']}, pot={game_state['pot']}BB")
        return game_state

    def _detect_position(self, text):
        """Detect player position from text"""
        # Look for position indicators in text
        position_match = self.patterns['position'].search(text)
        if position_match:
            return position_match.group(1).upper()

        # Keywords search
        text_lower = text.lower()
        if 'button' in text_lower or 'btn' in text_lower or 'dealer' in text_lower:
            return "BTN"
        elif 'small' in text_lower and 'blind' in text_lower:
            return "SB"
        elif 'big' in text_lower and 'blind' in text_lower:
            return "BB"
        elif 'cutoff' in text_lower or 'co' in text_lower:
            return "CO"
        elif 'middle' in text_lower or 'mp' in text_lower:
            return "MP"

        # Rotate positions randomly to provide variety
        positions = ["BTN", "CO", "MP", "UTG"]
        import random
        return random.choice(positions)

    def _detect_street(self, text):
        """Detect current street from text"""
        text_lower = text.lower()

        # Check for street indicators
        if 'flop' in text_lower or '3 card' in text_lower:
            return "flop"
        elif 'turn' in text_lower or '4th' in text_lower or '4 card' in text_lower:
            return "turn"
        elif 'river' in text_lower or '5th' in text_lower or '5 card' in text_lower:
            return "river"
        else:
            return "preflop"  # Default to preflop

    def _detect_pot(self, text):
        """Detect pot size from text"""
        pot_match = self.patterns['pot'].search(text)
        if pot_match:
            try:
                return float(pot_match.group(1))
            except ValueError:
                pass

        # If not found, try to find any number followed by BB
        bb_match = re.search(r'(\d+\.?\d*)\s*BB', text)
        if bb_match:
            try:
                return float(bb_match.group(1))
            except ValueError:
                pass

        # Default pot size based on street
        street = self._detect_street(text)
        if street == "preflop":
            return 1.5  # Default preflop pot
        elif street == "flop":
            return 4.0  # Typical flop pot
        elif street == "turn":
            return 8.0  # Typical turn pot
        elif street == "river":
            return 15.0  # Typical river pot

    def _detect_actions(self, text):
        """Detect available actions from text"""
        text_lower = text.lower()
        actions = []

        if 'fold' in text_lower:
            actions.append("FOLD")
        if 'call' in text_lower:
            actions.append("CALL")
        if 'check' in text_lower:
            actions.append("CHECK")
        if 'raise' in text_lower or 'bet' in text_lower:
            actions.append("RAISE")

        # Ensure we have at least some actions
        if not actions:
            actions = ["FOLD", "CALL", "RAISE"]

        return actions

class PokerDecisionCache:
    """Cache for poker decisions to speed up responses"""

    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """Get a cached decision if available"""
        if key in self.cache:
            self.hits += 1
            logger.info(f"Cache hit! Key: {key}")
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def set(self, key, value):
        """Cache a decision"""
        self.cache[key] = value

        # Trim cache if too large
        if len(self.cache) > self.max_size:
            # Remove oldest entries
            keys_to_remove = list(self.cache.keys())[:len(self.cache) - self.max_size]
            for k in keys_to_remove:
                del self.cache[k]

    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }

class PokerAdvisor:
    """Main poker advisor class using GPT assistants"""

    def __init__(self, hero_username="rondaygo"):
        self.processor = PokerImageProcessor(hero_username)
        self.cache = PokerDecisionCache()
        self.models = {}
        self.load_models()

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

    def select_assistant(self, game_state):
        """Select the appropriate assistant based on the game state."""
        street = game_state.get("street", "").lower()

        # ASSISTANT_1: Preflop Position Assistant
        if street == "preflop":
            return 1

        # ASSISTANT_4: Board Texture Assistant  
        if street in ["flop", "turn", "river"]:
            return 4

        # Default to preflop assistant
        return 1

    def format_prompt(self, game_state):
        """Format game state into prompt for GPT."""
        street = str(game_state.get("street", "unknown street")).capitalize()
        position = str(game_state.get("position", "unknown position"))
        pot = game_state.get("pot", 1.5)
        available_actions = game_state.get("available_actions", ["FOLD", "CALL", "RAISE"])

        # Create a clear, structured prompt
        prompt = f"{street} decision. You are in the {position} position with pot size {pot}BB.\n"
        prompt += f"Available actions: {', '.join(available_actions)}.\n\n"
        prompt += f"What's the optimal poker decision in this situation?\n"
        prompt += f"Reply with ONLY ONE of these options: {' / '.join(available_actions)}."
        prompt += " If you choose RAISE, specify an amount (e.g., 'RAISE to 3BB')."

        return prompt

    def get_cache_key(self, game_state):
        """Generate cache key from game state."""
        position = game_state.get("position", "")
        street = game_state.get("street", "")
        pot = game_state.get("pot", 0)

        # Round pot to nearest 0.5BB for better cache hits
        pot_rounded = round(pot * 2) / 2

        return f"{position}:{street}:{pot_rounded}"

    def get_model_response(self, assistant_num, user_input):
        """Get response from GPT assistant."""
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
        """Format the assistant response consistently."""
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
                # Default raise amount if not specified
                return "RECOMMEND: RAISE to 3BB"

        # Default to first available action if no match
        if available_actions:
            return f"RECOMMEND: {available_actions[0]}"
        else:
            return "RECOMMEND: CALL"  # Ultimate fallback

    def get_advice(self, image):
        """Process poker image and get advice."""
        start_time = time.time()

        # Convert PIL image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image

        # Process the image
        game_state = self.processor.process_image(image_np)

        # Generate cache key
        cache_key = self.get_cache_key(game_state)

        # Check cache for existing decision
        cached_decision = self.cache.get(cache_key)
        if cached_decision:
            processing_time = time.time() - start_time
            logger.info(f"Using cached decision: {cached_decision}")

            return {
                "suggestion": cached_decision,
                "game_state": game_state,
                "processing_time": processing_time,
                "source": "cache"
            }

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

        # Cache the response
        self.cache.set(cache_key, normalized_response)

        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Generated decision in {processing_time:.3f}s: {normalized_response}")

        return {
            "suggestion": normalized_response,
            "game_state": game_state,
            "processing_time": processing_time,
            "source": "assistant"
        }

def create_app():
    app = Flask(__name__)

    # Initialize the advisor
    advisor = PokerAdvisor("rondaygo")

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

                # Get advice
                result = advisor.get_advice(image)

                return jsonify(result)
            else:
                return jsonify({"error": "No image file found in request"}), 400

        except Exception as e:
            logger.exception(f"Error processing request: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/cache/stats", methods=["GET"])
    def cache_stats():
        """Get cache statistics"""
        return jsonify(advisor.cache.get_stats())

    return app

if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Poker Assistant Server")
    app.run(debug=True, host="0.0.0.0", port=5000)
