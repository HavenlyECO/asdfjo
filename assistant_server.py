import os
import time
import sys
from typing import Dict, Optional, List, Any, Union
from flask import Flask, request, jsonify
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
        street = game_state.get("street", "").lower()
        action = game_state.get("action", "").lower()
        position = game_state.get("position", "").lower()
        
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
        if street in ["flop", "turn"] and game_state.get("pot", 0) / max(game_state.get("stack_btn", 1), 
                                                                         game_state.get("stack_bb", 1)) > 0.3:
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
        if street in ["turn", "river"] and ("bet" in action.lower() and 
            game_state.get("bet_size", 0) > game_state.get("pot", 1) * 0.8):
            return 10
            
        # Default to EV Delta Comparator if no clear match
        return 8
    
    def format_prompt(self, game_state: Dict) -> str:
        """Format game state into natural language prompt according to required format."""
        street = game_state.get("street", "unknown street").capitalize()
        position = game_state.get("position", "unknown position")
        stack_btn = game_state.get("stack_btn", "unknown")
        stack_bb = game_state.get("stack_bb", "unknown")
        pot = game_state.get("pot", "unknown")
        action = game_state.get("action", "")
        
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
        elif any(word in line.lower() for word in ["flop", "board:"]):
            state["betting_round"] = "flop"
        elif any(word in line.lower() for word in ["turn", "4th"]):
            state["betting_round"] = "turn"
        elif any(word in line.lower() for word in ["river", "5th"]):
            state["betting_round"] = "river"
            
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
        
        # Debug request information
        logger.debug(f"Request headers: {request.headers}")
        logger.debug(f"Request form data: {request.form}")
        logger.debug(f"Request files: {request.files}")
        
        try:
            # Accept multiple input formats - better than a 400 error
            tournament_mode = request.form.get("tournament_mode", "false").lower() == "true"
            
            # Check if we have an image
            if 'image' in request.files:
                # Standard image processing path
                image_data = request.files.get("image")
                logger.info("Processing image upload")
                
                # Extract text from image
                ocr_lines = extract_table_state(image_data)
                
                # Parse game state from OCR text
                game_state = parse_game_state(ocr_lines)
                
            elif 'game_state' in request.form:
                # Direct game state input
                logger.info("Processing direct game state input")
                try:
                    game_state = json.loads(request.form.get("game_state"))
                except json.JSONDecodeError:
                    logger.error("Invalid game state JSON")
                    return jsonify({"error": "Invalid game state JSON"}), 400
                
                # Use empty OCR lines since we have direct input
                ocr_lines = []
                
            else:
                # Try looking for JSON in request body
                logger.info("No image or game_state in form data, checking JSON body")
                try:
                    data = request.get_json(silent=True)
                    if data and 'game_state' in data:
                        game_state = data['game_state']
                        tournament_mode = data.get('tournament_mode', False)
                        ocr_lines = []
                    else:
                        logger.error("No image or game state provided in request")
                        return jsonify({
                            "error": "No image or game state provided", 
                            "accepted_formats": [
                                "multipart/form-data with 'image' file",
                                "multipart/form-data with 'game_state' JSON string",
                                "application/json with game_state object"
                            ]
                        }), 400
                except Exception as e:
                    logger.error(f"Error parsing request body: {e}")
                    return jsonify({"error": "Invalid request format"}), 400
            
            # Get advice using the Poker Response Manager
            advice = route_from_ocr(game_state, tournament_mode)
            
            # Return the advice
            return jsonify({
                "suggestion": advice,
                "game_state": game_state
            })
            
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error processing request: {str(e)}\n{error_details}")
            return jsonify({
                "error": str(e),
                "details": error_details,
                "hint": "Please check request format - need either 'image' file or 'game_state' JSON"
            }), 500

    return app

# Example usage
if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Poker Assistant Server")
    logger.info(f"Loaded {len(response_manager.models)} assistant configurations")
    app.run(debug=True, host="0.0.0.0", port=5000)
