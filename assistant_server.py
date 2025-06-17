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
import pytesseract  # Faster than EasyOCR
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

class RegionOfInterest:
    """Defines regions of the poker screenshot for targeted extraction"""
    
    def __init__(self, name: str, x1: float, y1: float, x2: float, y2: float):
        """
        Define a region using normalized coordinates (0-1)
        
        Args:
            name: Region name (e.g., 'pot', 'hero_cards')
            x1, y1: Top-left coordinates (normalized 0-1)
            x2, y2: Bottom-right coordinates (normalized 0-1)
        """
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

class ParallelPokerExtractor:
    """
    Fast parallel extractor that processes multiple image regions simultaneously
    for maximum speed.
    """
    
    # Define common poker UI regions - these would be calibrated for your specific poker site
    REGIONS = [
        RegionOfInterest('pot', 0.4, 0.3, 0.6, 0.4),
        RegionOfInterest('hero_cards', 0.4, 0.7, 0.6, 0.8),
        RegionOfInterest('board_cards', 0.3, 0.4, 0.7, 0.5),
        RegionOfInterest('hero_stack', 0.4, 0.8, 0.6, 0.85),
        RegionOfInterest('action_buttons', 0.2, 0.85, 0.8, 0.95),
        RegionOfInterest('player_info', 0.0, 0.6, 0.3, 0.9),
    ]
    
    def __init__(self, hero_username="rondaygo", num_threads=4):
        """Initialize the parallel extractor"""
        self.hero_username = hero_username.lower()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
        
        # Configure Tesseract for speed
        self.config = '--oem 1 --psm 6 -l eng'
        
        # Precompile regex patterns for speed
        self.patterns = {
            'pot': re.compile(r'(?:pot|total)[:\s]*(\d+\.?\d*)(?:\s*BB)?', re.IGNORECASE),
            'cards': re.compile(r'[2-9TJQKA][cdhs\u2663\u2666\u2665\u2660]'),
            'stack': re.compile(r'(\d+\.?\d*)\s*(?:BB|bb)'),
            'action': re.compile(r'(fold|check|call|raise|bet|all[\s-]*in)', re.IGNORECASE),
            'username': re.compile(rf'{re.escape(hero_username)}', re.IGNORECASE)
        }
        
        # Decision cache - avoids repeated API calls for similar situations
        self.decision_cache = {}
        
        # Precomputed decisions for common scenarios
        self.load_precomputed_decisions()
        
    def load_precomputed_decisions(self):
        """Load fast precomputed decisions for common poker scenarios"""
        # This would be expanded with a proper decision table based on GTO principles
        self.precomputed_decisions = {
            # Format: "position:street:pot_size_range:stack_depth_range": decision
            "BTN:preflop:1-4:50-100": "RAISE to 3BB",
            "SB:preflop:1-4:50-100": "RAISE to 3BB",
            "BB:preflop:1-4:50-100": "CHECK",
            # Add more precomputed scenarios for instant decisions
            "BTN:preflop:4-8:50-100": "RAISE to 5BB",
            "SB:preflop:4-8:50-100": "CALL",
            "BB:preflop:4-8:50-100": "FOLD",
            "BTN:flop:5-10:50-100": "CALL",
            "SB:flop:5-10:50-100": "CHECK",
            "BB:flop:5-10:50-100": "CHECK",
            "BTN:turn:10-20:50-100": "CALL",
            "SB:turn:10-20:50-100": "CHECK",
            "BB:turn:10-20:50-100": "CHECK",
            "BTN:river:15-30:50-100": "CALL",
            "SB:river:15-30:50-100": "CHECK",
            "BB:river:15-30:50-100": "CHECK",
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Optimize image for OCR speed and accuracy"""
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to enhance text
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        return thresh
    
    def process_region(self, region: RegionOfInterest, preprocessed_image: np.ndarray) -> Dict[str, Any]:
        """Process a single region of interest"""
        # Extract region
        roi = region.extract_from_image(preprocessed_image)
        
        # Apply region-specific processing if needed
        if region.name == 'action_buttons':
            # Enhance contrast for button text
            roi = cv2.equalizeHist(roi)
        
        # Extract text from region
        text = pytesseract.image_to_string(roi, config=self.config)
        
        # Parse based on region type
        result = {'raw_text': text, 'region': region.name}
        
        if region.name == 'pot':
            pot_match = self.patterns['pot'].search(text)
            if pot_match:
                result['pot'] = float(pot_match.group(1))
                
        elif region.name == 'hero_cards' or region.name == 'board_cards':
            card_matches = self.patterns['cards'].findall(text)
            result['cards'] = card_matches
            
        elif region.name == 'hero_stack':
            stack_match = self.patterns['stack'].search(text)
            if stack_match:
                result['stack'] = float(stack_match.group(1))
                
        elif region.name == 'action_buttons':
            result['available_actions'] = []
            for action in ['fold', 'check', 'call', 'raise', 'bet', 'all-in']:
                if action in text.lower():
                    result['available_actions'].append(action.upper())
        
        elif region.name == 'player_info':
            # Look for hero username
            if self.patterns['username'].search(text):
                result['hero_found'] = True
                # Try to determine position based on text
                if 'btn' in text.lower() or 'button' in text.lower():
                    result['position'] = 'BTN'
                elif 'sb' in text.lower() or 'small blind' in text.lower():
                    result['position'] = 'SB'
                elif 'bb' in text.lower() or 'big blind' in text.lower():
                    result['position'] = 'BB'
                else:
                    result['position'] = 'BTN'  # Default
        
        return result
    
    def extract_game_state(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Rapidly extract poker game state from screenshot using parallel processing
        """
        # Preprocess the image
        preprocessed = self.preprocess_image(image)
        
        # Process all regions in parallel
        future_results = {}
        for region in self.REGIONS:
            future_results[region.name] = self.executor.submit(self.process_region, region, preprocessed)
        
        # Collect results
        results = {name: future.result() for name, future in future_results.items()}
        
        # Combine into comprehensive game state
        game_state = {
            "position": "BTN",  # Default
            "street": "preflop",  # Default
            "pot": 1.5,  # Default
            "stack_btn": 100.0,  # Default
            "stack_bb": 100.0,  # Default
            "hero_username": self.hero_username,
            "available_actions": ["FOLD", "CALL", "RAISE"]
        }
        
        # Update with extracted values
        if 'pot' in results.get('pot', {}):
            game_state['pot'] = results['pot']['pot']
            
        if 'stack' in results.get('hero_stack', {}):
            game_state['hero_stack'] = results['hero_stack']['stack']
            
        if 'cards' in results.get('hero_cards', {}):
            game_state['hero_cards'] = results['hero_cards']['cards']
            
        if 'cards' in results.get('board_cards', {}):
            game_state['board_cards'] = results['board_cards']['cards']
            # Determine street based on board cards
            if len(game_state['board_cards']) == 3:
                game_state['street'] = 'flop'
            elif len(game_state['board_cards']) == 4:
                game_state['street'] = 'turn'
            elif len(game_state['board_cards']) == 5:
                game_state['street'] = 'river'
        
        if 'available_actions' in results.get('action_buttons', {}):
            game_state['available_actions'] = results['action_buttons']['available_actions']
            
        if 'position' in results.get('player_info', {}):
            game_state['position'] = results['player_info']['position']
        
        return game_state
    
    def get_quick_decision(self, game_state: Dict[str, Any]) -> Optional[str]:
        """
        Get an immediate decision without API call using precomputed tables
        for maximum speed
        """
        # Create a cache key
        position = game_state['position']
        street = game_state['street']
        pot = game_state['pot']
        stack = game_state.get('hero_stack', 100.0)
        
        # Check exact cache match
        cache_key = f"{position}:{street}:{pot}:{stack}"
        if cache_key in self.decision_cache:
            logger.info(f"Cache hit! Returning cached decision for {cache_key}")
            return self.decision_cache[cache_key]
        
        # Check precomputed decision match
        for key, decision in self.precomputed_decisions.items():
            parts = key.split(':')
            if len(parts) != 4:
                continue
                
            pos, st, pot_range, stack_range = parts
            
            if pos != position or st != street:
                continue
                
            pot_min, pot_max = map(float, pot_range.split('-'))
            stack_min, stack_max = map(float, stack_range.split('-'))
            
            if pot_min <= pot <= pot_max and stack_min <= stack <= stack_max:
                logger.info(f"Found precomputed decision for {key}")
                return f"RECOMMEND: {decision}"
        
        return None  # No quick decision available

class TieredDecisionEngine:
    """
    High-performance poker decision engine that uses a tiered approach:
    1. Fast local rules for obvious decisions
    2. Pre-computed decision tables
    3. API calls for complex situations
    """
    
    def __init__(self, hero_username="rondaygo"):
        self.extractor = ParallelPokerExtractor(hero_username)
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
                    "system_message": f"You are Poker Assistant {i}. Provide fast, optimal poker decisions."
                }
            else:
                logger.warning(f"ASSISTANT_{i} not found in environment variables")
    
    def select_assistant(self, game_state: Dict, tournament_mode: bool) -> int:
        """Select the appropriate assistant based on the updated assistant architecture."""
        street = str(game_state.get("street", "")).lower()
        action = str(game_state.get("action", "")).lower()
        position = str(game_state.get("position", "")).lower()
        
        # ASSISTANT_1: Preflop Position Assistant
        if street == "preflop" and "3bet" not in action and "3-bet" not in action:
            return 1
            
        # ASSISTANT_2: Stack-Based ICM Assistant
        if tournament_mode or game_state.get("stack_btn", 100) < 20 or game_state.get("stack_bb", 100) < 20:
            return 2
            
        # ASSISTANT_3: Villain Exploit Assistant
        if any(key in game_state for key in ["vpip", "pfr", "3bet_freq", "fold_to_3bet", "aggression_factor"]):
            return 3
            
        # ASSISTANT_4: Board Texture Assistant
        if street in ["flop", "turn", "river"] and game_state.get("board_cards", []):
            return 4
            
        # ASSISTANT_5: Pot Odds Assistant
        if game_state.get("pot", 0) > 0 and game_state.get("bet_to_call", 0) > 0:
            return 5
            
        # ASSISTANT_6: Future Street Pressure Assistant
        stack_btn = float(game_state.get("stack_btn", 1))
        stack_bb = float(game_state.get("stack_bb", 1))
        pot = float(game_state.get("pot", 0))
        if street in ["flop", "turn"] and pot > 0 and pot / max(stack_btn, stack_bb) > 0.3:
            return 6
            
        # ASSISTANT_7: Bluff Catcher Evaluator
        if street == "river" and "check" in action and "raise" in action:
            return 7
            
        # ASSISTANT_8: EV Delta Comparator Assistant
        if "expected_value" in game_state or "ev_" in str(game_state):
            return 8
            
        # ASSISTANT_9: Meta-Image Shift Assistant
        if "image" in game_state or "table_image" in game_state or "history" in game_state:
            return 9
            
        # ASSISTANT_10: Overbet Detection Assistant
        bet_size = float(game_state.get("bet_size", 0))
        if street in ["turn", "river"] and "bet" in action and pot > 0 and bet_size > pot * 0.8:
            return 10
            
        # Default to EV Delta Comparator if no clear match
        return 8
    
    def format_prompt(self, game_state: Dict) -> str:
        """Format game state into natural language prompt according to required format."""
        street = str(game_state.get("street", "unknown street")).capitalize()
        position = str(game_state.get("position", "unknown position"))
        stack_btn = game_state.get("stack_btn", "unknown")
        stack_bb = game_state.get("stack_bb", "unknown")
        pot = game_state.get("pot", "unknown")
        action = str(game_state.get("action", ""))
        available_actions = game_state.get("available_actions", ["FOLD", "CALL", "RAISE"])
        
        prompt = f"{street}. You are in the {position} with {stack_btn}BB. "
        
        # Include hole cards if available
        if game_state.get("hero_cards"):
            cards = " ".join(game_state["hero_cards"])
            prompt += f"Your cards are {cards}. "
        
        # Include board cards if available
        if game_state.get("board_cards") and street != "preflop":
            cards = " ".join(game_state["board_cards"])
            prompt += f"The board shows {cards}. "
        
        if "bb" in position.lower():
            prompt += f"The button has {stack_btn}BB"
        else:
            prompt += f"The big blind has {stack_bb}BB"
        
        if action:
            prompt += f" and has {action}"
        
        prompt += f". The pot is {pot}BB. "
        
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
    
    def get_decision(self, image_np: np.ndarray, tournament_mode: bool = False) -> Dict[str, Any]:
        """
        Fast poker decision making pipeline
        
        Args:
            image_np: Numpy array of screenshot
            tournament_mode: Whether tournament mode is active
            
        Returns:
            Dict with decision and game state
        """
        start_time = time.time()
        
        # Extract game state using parallel processing
        game_state = self.extractor.extract_game_state(image_np)
        
        # Try to get quick decision from cache or precomputed tables
        quick_decision = self.extractor.get_quick_decision(game_state)
        if quick_decision:
            processing_time = time.time() - start_time
            logger.info(f"Quick decision made in {processing_time:.3f}s: {quick_decision}")
            return {
                "suggestion": quick_decision,
                "game_state": game_state,
                "processing_time": processing_time
            }
            
        # If no quick decision, use API
        assistant_num = self.select_assistant(game_state, tournament_mode)
        prompt = self.format_prompt(game_state)
        prompt = f"{assistant_num}. {prompt}"
        
        raw_response = self.get_model_response(assistant_num, prompt)
        normalized_response = self.normalize_response(raw_response, game_state.get("available_actions", ["FOLD", "CALL", "RAISE"]))
        
        # Cache this decision for future use
        cache_key = f"{game_state['position']}:{game_state['street']}:{game_state['pot']}:{game_state.get('hero_stack', 100.0)}"
        self.extractor.decision_cache[cache_key] = normalized_response
        
        processing_time = time.time() - start_time
        logger.info(f"API decision made in {processing_time:.3f}s: {normalized_response}")
        
        return {
            "suggestion": normalized_response,
            "game_state": game_state,
            "processing_time": processing_time
        }

class AsyncDecisionProcessor:
    """
    Processes poker decisions asynchronously to ensure decisions are ready
    before the countdown timer expires
    """
    
    def __init__(self):
        self.decision_engine = TieredDecisionEngine("rondaygo")
        self.last_decision = None
        self.decision_queue = deque(maxlen=3)  # Keep last 3 decisions for analysis
        self.processing_thread = None
        self.processing_lock = threading.Lock()
        
    def process_image_async(self, image_file):
        """Start asynchronous processing of an image"""
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
                    "note": "Fast response while analyzing"
                }
    
    def _process_image_thread(self, image_np):
        """Process image in background thread"""
        try:
            # Get decision
            decision = self.decision_engine.get_decision(image_np)
            
            # Store in decision history queue
            self.decision_queue.append(decision)
            
            # Store as last decision
            with self.processing_lock:
                self.last_decision = decision
                
        except Exception as e:
            logger.error(f"Error in background processing: {str(e)}")

# Flask application setup
def create_app():
    app = Flask(__name__)
    
    # Initialize the async processor
    processor = AsyncDecisionProcessor()
    
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
    logger.info("Starting High-Performance Poker Assistant Server")
    app.run(debug=True, host="0.0.0.0", port=5000)
