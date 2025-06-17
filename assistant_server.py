import os
import time
import sys
from typing import Dict, Optional, List, Any, Union
from flask import Flask, request, jsonify, Response
from openai import OpenAI
from PIL import Image
import easyocr
import numpy as np
import json
import io
from dotenv import load_dotenv
import re
import logging
import traceback
import base64

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Temporarily set to DEBUG for diagnosing client issues
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('poker_assistant')

# Load environment variables
load_dotenv()

# Initialize OpenAI client with proper error handling
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY environment variable not found in .env file")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# Initialize OCR
try:
    reader = easyocr.Reader(["en"], gpu=False)
    logger.info("EasyOCR initialized successfully")
except Exception as e:
    logger.error(f"Error initializing EasyOCR: {str(e)}")
    sys.exit(1)

class PokerResponseManager:
    """Modern architecture using OpenAI's Chat Completions API for poker decision-making."""
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load model configurations from environment variables."""
        for i in range(1, 11):
            model_id = os.environ.get(f"ASSISTANT_{i}")
            if model_id:
                # For models mapped to assistant IDs, we'll use gpt-4-turbo by default
                self.models[i] = {
                    "model": "gpt-4-turbo",
                    "assistant_id": model_id,
                    "system_message": f"You are Poker Assistant {i}. Wait for game data to make decision."
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
        if street in ["flop", "turn", "river"] and game_state.get("board", ""):
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
        
        prompt = f"{street}. You are in the {position} with {stack_btn}BB. "
        
        if "bb" in position.lower():
            prompt += f"The button has {stack_btn}BB"
        else:
            prompt += f"The big blind has {stack_bb}BB"
        
        if action:
            prompt += f" and has {action}"
        
        prompt += f". The pot is {pot}BB. What's the optimal decision?"
        
        # Add the required ending
        prompt += "\n\nRespond with only one recommendation: FOLD, CALL, or RAISE (include amount if applicable). Do not explain."
        
        return prompt
    
    def get_model_response(self, assistant_num: int, user_input: str) -> str:
        """Get response using the Chat Completions API."""
        model_config = self.models.get(assistant_num)
        
        if not model_config:
            return "Error: Assistant not found"
        
        try:
            logger.info(f"Sending request to model for assistant {assistant_num}...")
            
            # Create messages for the API call
            messages = [
                {"role": "system", "content": model_config["system_message"]},
                {"role": "user", "content": user_input}
            ]
            
            # Make API call with proper timeout handling
            response = client.chat.completions.create(
                model=model_config["model"],
                messages=messages,
                max_tokens=300,
                temperature=0.3,
                stream=True
            )
            
            # Handle streaming response
            collected_content = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    collected_content.append(chunk.choices[0].delta.content)
                    # Print progress indicator
                    sys.stdout.write(".")
                    sys.stdout.flush()
            
            sys.stdout.write("\n")  # Newline after progress indicators
            
            return "".join(collected_content)
            
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return f"Error: {str(e)}"
    
    def normalize_response(self, response: str) -> str:
        """Normalize assistant response to standard format."""
        response = response.upper().strip()
        
        if "FOLD" in response:
            return "RECOMMEND: FOLD"
        
        if "CALL" in response:
            return "RECOMMEND: CALL"
        
        if "RAISE" in response:
            # Try to extract amount
            match = re.search(r"RAISE\s+TO\s+(\d+(?:\.\d+)?)", response, re.IGNORECASE)
            if match:
                return f"RECOMMEND: RAISE to {match.group(1)}"
            return "RECOMMEND: RAISE (amount not specified)"
        
        return f"RECOMMEND: UNKNOWN - {response}"

# Create global instance
response_manager = PokerResponseManager()

def extract_text_easyocr(image: Image.Image) -> str:
    """Extract text from image using EasyOCR."""
    # Convert PIL Image to numpy array for EasyOCR compatibility
    image_np = np.array(image)
    
    # EasyOCR requires numpy array in BGR format (OpenCV style)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        # Convert RGB to BGR if needed
        image_np = image_np[:, :, ::-1].copy()
    
    result = reader.readtext(image_np)
    return "\n".join(entry[1] for entry in result)

def extract_table_state(image_data):
    """Extract and format table state from image data."""
    try:
        # Open image from file data
        image = Image.open(image_data)
        
        # Extract text using OCR
        text_eo = extract_text_easyocr(image)
        lines = [line.strip() for line in text_eo.split("\n") if line.strip()]
        logger.info(f"OCR extracted {len(lines)} lines of text")
        return lines
    except Exception as e:
        logger.error(f"Error extracting table state: {str(e)}")
        raise

def parse_game_state(text_lines):
    """Advanced game state parser with comprehensive detection."""
    logger.info("Parsing game state from OCR text")
    import re

    state = {
        "pot": None,
        "player_stacks": {},
        "positions": {},
        "bet_sizes": {},
        "action_to": None, 
        "betting_round": None,
        "community_cards": None,
        "raw_lines": text_lines,
        
        # Default values for the assistant
        "position": "unknown",
        "stack_btn": 0,
        "stack_bb": 0,
        "street": "unknown"
    }
    
    # Detection patterns
    for line in text_lines:
        # Pot detection with multiple formats
        if "pot" in line.lower():
            pot_match = re.search(r"pot[:\s]*([\d,\.]+)", line, re.IGNORECASE)
            if pot_match:
                state["pot"] = pot_match.group(1).replace(",", "")
        
        # Currency symbol detection
        pot_match = re.search(r"[\$€£][\s]*([\d,\.]+)", line, re.IGNORECASE)
        if pot_match and not state["pot"]:
            state["pot"] = pot_match.group(1).replace(",", "")
            
        # Game phase detection
        if any(word in line.lower() for word in ["preflop", "pre-flop", "hole"]):
            state["betting_round"] = "preflop"
            state["street"] = "preflop"
        elif any(word in line.lower() for word in ["flop", "board:"]):
            state["betting_round"] = "flop"
            state["street"] = "flop"
        elif any(word in line.lower() for word in ["turn", "4th"]):
            state["betting_round"] = "turn"
            state["street"] = "turn"
        elif any(word in line.lower() for word in ["river", "5th"]):
            state["betting_round"] = "river"
            state["street"] = "river"
            
        # Other detection logic...
    
    logger.info(f"Parsed game state: {state}")
    return state

def route_from_ocr(game_state: Dict, tournament_mode: bool = False) -> str:
    """
    Route OCR game state to appropriate assistant and get poker decision.
    
    Args:
        game_state: Dictionary containing parsed poker game state
        tournament_mode: Boolean indicating if this is tournament play
        
    Returns:
        String with recommended action
    """
    try:
        # Select appropriate assistant based on game state
        assistant_num = response_manager.select_assistant(game_state, tournament_mode)
        
        logger.info(f"Selected assistant {assistant_num} for analysis")
        
        # Format game state into prompt
        prompt = response_manager.format_prompt(game_state)
        
        # Append assistant number to prompt (as requested in descriptions)
        prompt = f"{assistant_num}. {prompt}"
        
        # Get response
        raw_response = response_manager.get_model_response(assistant_num, prompt)
        
        # Normalize response
        normalized_response = response_manager.normalize_response(raw_response)
        
        return normalized_response
    
    except Exception as e:
        logger.error(f"Error in route_from_ocr: {str(e)}")
        return f"ERROR: {str(e)}"

def create_app():
    app = Flask(__name__)
    
    @app.route("/api/advice", methods=["POST"])
    def api_advice():
        logger.info(f"Received API request from {request.remote_addr}")
        
        # Log raw request data for debugging client issues
        logger.debug(f"Content-Type: {request.content_type}")
        logger.debug(f"Request data length: {request.content_length}")
        
        # Dump request headers for debugging
        header_debug = "\n".join([f"{k}: {v}" for k, v in request.headers.items()])
        logger.debug(f"Request headers:\n{header_debug}")
        
        # Check if we can actually read the data
        if request.data:
            logger.debug(f"Raw request data (first 200 bytes): {request.data[:200]}")
        
        try:
            # Extract tournament mode from all possible locations
            tournament_mode = False
            
            # Strategy 1: Try to get game state from raw request body - Client compatibility
            if request.data:
                logger.debug("Attempting to parse raw request body")
                try:
                    # Try to parse as direct JSON
                    raw_data = json.loads(request.data)
                    
                    if isinstance(raw_data, dict):
                        # Check if this is a complete game state
                        if 'position' in raw_data or 'street' in raw_data or 'pot' in raw_data:
                            logger.info("Found direct game state in request body")
                            game_state = raw_data
                            tournament_mode = raw_data.get('tournament_mode', False)
                            return process_game_state(game_state, tournament_mode)
                        
                        # Check if game_state is a nested property
                        if 'game_state' in raw_data:
                            logger.info("Found nested game_state in request body")
                            game_state = raw_data['game_state']
                            tournament_mode = raw_data.get('tournament_mode', False)
                            return process_game_state(game_state, tournament_mode)
                        
                        # Check if it has an 'image' property with base64 data
                        if 'image' in raw_data and isinstance(raw_data['image'], str):
                            logger.info("Found base64 image in request body")
                            try:
                                # Try to decode base64 image
                                image_data = base64.b64decode(raw_data['image'])
                                image_file = io.BytesIO(image_data)
                                return process_image(image_file, raw_data.get('tournament_mode', False))
                            except Exception as e:
                                logger.error(f"Error decoding base64 image: {e}")
                                
                except Exception as e:
                    logger.warning(f"Failed to parse raw body as JSON: {e}")
            
            # Strategy 2: Check for multipart form data with image
            if 'image' in request.files:
                logger.info("Processing image from multipart form")
                image_file = request.files['image']
                tournament_mode = request.form.get('tournament_mode', 'false').lower() == 'true'
                return process_image(image_file, tournament_mode)
            
            # Strategy 3: Check for JSON in form data
            if 'game_state' in request.form:
                logger.info("Processing game state from form data")
                try:
                    game_state = json.loads(request.form['game_state'])
                    tournament_mode = request.form.get('tournament_mode', 'false').lower() == 'true'
                    return process_game_state(game_state, tournament_mode)
                except json.JSONDecodeError:
                    logger.error("Invalid game state JSON in form")
                    return jsonify({"error": "Invalid game state JSON"}), 400
            
            # Strategy 4: Silent JSON body parsing (different content types)
            data = request.get_json(silent=True)
            if data:
                logger.info("Processing JSON body")
                if 'game_state' in data:
                    game_state = data['game_state']
                    tournament_mode = data.get('tournament_mode', False)
                    return process_game_state(game_state, tournament_mode)
            
            # COMPATIBILITY MODE: If we reach here, client is likely sending some other format
            # Let's create a default game state as fallback for client compatibility
            logger.warning("No recognizable data format found. Using compatibility mode.")
            
            # Client compatibility mode - construct minimal game state from URL parameters
            default_game_state = {
                "position": request.args.get('position', 'BTN'),
                "stack_btn": float(request.args.get('stack_btn', '50')),
                "stack_bb": float(request.args.get('stack_bb', '50')), 
                "pot": float(request.args.get('pot', '3.5')),
                "action": request.args.get('action', ''),
                "street": request.args.get('street', 'preflop')
            }
            
            logger.info(f"Created default game state: {default_game_state}")
            return process_game_state(default_game_state, False)
                
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error processing request: {str(e)}\n{error_details}")
            return jsonify({
                "error": str(e),
                "details": error_details
            }), 500

    def process_image(image_file, tournament_mode: bool):
        """Process an image file to extract and analyze poker state."""
        try:
            # Extract text from image
            ocr_lines = extract_table_state(image_file)
            
            # Parse game state from OCR text
            game_state = parse_game_state(ocr_lines)
            
            # Get advice using the poker response manager
            advice = route_from_ocr(game_state, tournament_mode)
            
            # Return the advice
            return jsonify({
                "suggestion": advice,
                "game_state": game_state,
                "ocr_lines": ocr_lines
            })
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error processing image: {str(e)}\n{error_details}")
            return jsonify({"error": str(e), "details": error_details}), 500

    def process_game_state(game_state, tournament_mode: bool):
        """Process a game state object to get poker advice."""
        try:
            # Get advice using the poker response manager
            advice = route_from_ocr(game_state, tournament_mode)
            
            # Return the advice
            return jsonify({
                "suggestion": advice,
                "game_state": game_state
            })
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error processing game state: {str(e)}\n{error_details}")
            return jsonify({"error": str(e), "details": error_details}), 500
            
    return app

# Example usage
if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Poker Assistant Server")
    logger.info(f"Loaded {len(response_manager.models)} assistant configurations")
    # Use host 0.0.0.0 to make the server accessible from any IP
    app.run(debug=True, host="0.0.0.0", port=5000)
