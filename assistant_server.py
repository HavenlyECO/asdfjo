import os
import time
import sys
import re
from typing import Dict, Optional, List, Any, Union, Tuple
from flask import Flask, request, jsonify
from openai import OpenAI
from PIL import Image
import easyocr
import numpy as np
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('poker_assistant')

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY environment variable not found")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# Initialize OCR
try:
    reader = easyocr.Reader(["en"], gpu=False)
    logger.info("EasyOCR initialized successfully")
except Exception as e:
    logger.error(f"Error initializing EasyOCR: {str(e)}")
    sys.exit(1)

class AdvancedPokerParser:
    """
    Advanced poker screenshot parser that uses spatial, contextual, and keyword analysis
    to extract accurate game state from OCR text.
    """
    
    # Poker position abbreviations and variations
    POSITION_KEYWORDS = {
        'BTN': ['btn', 'button', 'dealer'],
        'SB': ['sb', 'small blind', 'smallblind'],
        'BB': ['bb', 'big blind', 'bigblind'],
        'UTG': ['utg', 'under the gun'],
        'MP': ['mp', 'middle position'],
        'CO': ['co', 'cutoff']
    }
    
    # Betting round keywords
    STREET_KEYWORDS = {
        'preflop': ['preflop', 'pre-flop', 'pre flop', 'hole cards'],
        'flop': ['flop', 'on the flop', '3 cards'],
        'turn': ['turn', 'on the turn', '4th card', '4 cards', 'fourth street'],
        'river': ['river', 'on the river', '5th card', '5 cards', 'fifth street']
    }
    
    # Action keywords
    ACTION_KEYWORDS = {
        'fold': ['fold', 'folded'],
        'check': ['check', 'checked'],
        'call': ['call', 'called', 'limp', 'limped'],
        'bet': ['bet', 'bets'],
        'raise': ['raise', 'raised', '3-bet', '3bet', '4-bet', '4bet'],
        'all-in': ['all-in', 'all in', 'allin', 'shove', 'shoved', 'jam', 'jammed']
    }
    
    def __init__(self):
        """Initialize the advanced poker parser with pattern recognition capabilities."""
        self.card_pattern = re.compile(r'[2-9TJQKA][cdhs♣♦♥♠]')
        self.stack_pattern = re.compile(r'(\d+\.?\d*)\s*(?:BB|bb)')
        self.pot_pattern = re.compile(r'(?:pot|total)[:\s]*(\d+\.?\d*)', re.IGNORECASE)
        self.bet_pattern = re.compile(r'(?:bet|raise)[:\s]*(?:to)?\s*(\d+\.?\d*)', re.IGNORECASE)
        self.player_pattern = re.compile(r'([A-Za-z0-9_]+)\s+(\d+\.?\d*)\s*BB')
    
    def parse_ocr_results(self, ocr_lines: List[str]) -> Dict[str, Any]:
        """
        Parse OCR text lines into a comprehensive poker game state using multi-faceted analysis.
        
        Args:
            ocr_lines: List of text lines from OCR
            
        Returns:
            Dict containing parsed poker game state
        """
        # Initialize with default game state
        game_state = {
            "position": "unknown",
            "stack_btn": 100.0,  # Default
            "stack_bb": 100.0,   # Default
            "pot": 1.5,          # Default (1BB + 0.5BB)
            "action": "",
            "street": "preflop", # Default
            "hero_cards": [],
            "board_cards": [],
            "players": {},
            "raw_lines": ocr_lines
        }
        
        # Use a series of specialized extractors
        self._extract_pot(game_state, ocr_lines)
        self._extract_positions_and_stacks(game_state, ocr_lines)
        self._extract_street(game_state, ocr_lines)
        self._extract_cards(game_state, ocr_lines)
        self._extract_actions(game_state, ocr_lines)
        
        # Post-processing to normalize data
        self._normalize_game_state(game_state)
        
        return game_state
    
    def _extract_pot(self, game_state: Dict[str, Any], ocr_lines: List[str]) -> None:
        """Extract pot size from OCR lines."""
        for line in ocr_lines:
            # Look for explicit pot mentions
            pot_match = re.search(r'pot[:\s]*(\d+\.?\d*)(?:\s*BB)?', line, re.IGNORECASE)
            if pot_match:
                try:
                    game_state["pot"] = float(pot_match.group(1))
                    logger.info(f"Extracted pot: {game_state['pot']}BB")
                    return
                except ValueError:
                    continue
            
            # Try to find "total" which often refers to pot
            total_match = re.search(r'total[:\s]*(\d+\.?\d*)(?:\s*BB)?', line, re.IGNORECASE)
            if total_match:
                try:
                    game_state["pot"] = float(total_match.group(1))
                    logger.info(f"Extracted pot from 'total': {game_state['pot']}BB")
                    return
                except ValueError:
                    continue
    
    def _extract_positions_and_stacks(self, game_state: Dict[str, Any], ocr_lines: List[str]) -> None:
        """Extract positions and stack sizes."""
        players = {}
        
        # First pass: find player names and stack sizes
        for i, line in enumerate(ocr_lines):
            # Look for patterns like "username 45.5 BB"
            player_match = self.player_pattern.search(line)
            if player_match:
                name = player_match.group(1)
                stack = float(player_match.group(2))
                players[name] = {"stack": stack}
                
                # Check for position indicators in nearby lines
                for pos, keywords in self.POSITION_KEYWORDS.items():
                    for offset in range(-2, 3):  # Look at nearby lines
                        if i+offset >= 0 and i+offset < len(ocr_lines):
                            if any(kw in ocr_lines[i+offset].lower() for kw in keywords):
                                players[name]["position"] = pos
                                if pos == "BTN":
                                    game_state["stack_btn"] = stack
                                elif pos == "BB":
                                    game_state["stack_bb"] = stack
                                break
        
        # Set hero position - often the BB in most poker software layouts
        if players:
            # Try to detect which player is the hero based on context
            for name, player_info in players.items():
                if "position" in player_info:
                    if player_info["position"] == "BB":
                        game_state["position"] = "BB"
                        break
            
            # If no hero position found yet, use the first player with a stack
            if game_state["position"] == "unknown" and players:
                game_state["position"] = next(iter(players.values())).get("position", "BTN")
        
        # Store all players
        game_state["players"] = players
    
    def _extract_street(self, game_state: Dict[str, Any], ocr_lines: List[str]) -> None:
        """Extract the betting round/street."""
        for line in ocr_lines:
            line_lower = line.lower()
            
            # Go through each street and its keywords
            for street, keywords in self.STREET_KEYWORDS.items():
                if any(kw in line_lower for kw in keywords):
                    game_state["street"] = street
                    logger.info(f"Detected street: {street}")
                    return
            
            # Look for board cards as another way to determine street
            card_matches = self.card_pattern.findall(line)
            if card_matches:
                if len(card_matches) == 3:
                    game_state["street"] = "flop"
                elif len(card_matches) == 4:
                    game_state["street"] = "turn"
                elif len(card_matches) == 5:
                    game_state["street"] = "river"
    
    def _extract_cards(self, game_state: Dict[str, Any], ocr_lines: List[str]) -> None:
        """Extract hole cards and board cards."""
        for line in ocr_lines:
            card_matches = self.card_pattern.findall(line)
            if card_matches:
                if "hole" in line.lower() or "hero" in line.lower():
                    game_state["hero_cards"] = card_matches[:2]
                elif "board" in line.lower() or "community" in line.lower():
                    game_state["board_cards"] = card_matches
                elif len(game_state["board_cards"]) == 0 and len(card_matches) >= 3:
                    # If we find 3+ cards, they're likely board cards
                    game_state["board_cards"] = card_matches
    
    def _extract_actions(self, game_state: Dict[str, Any], ocr_lines: List[str]) -> None:
        """Extract betting actions."""
        actions = []
        
        for line in ocr_lines:
            line_lower = line.lower()
            
            # Look for action keywords
            for action_type, keywords in self.ACTION_KEYWORDS.items():
                if any(kw in line_lower for kw in keywords):
                    # Try to extract bet size if relevant
                    if action_type in ["bet", "raise", "all-in"]:
                        bet_match = self.bet_pattern.search(line)
                        if bet_match:
                            bet_size = bet_match.group(1)
                            actions.append(f"{action_type} to {bet_size}BB")
                        else:
                            actions.append(action_type)
                    else:
                        actions.append(action_type)
        
        if actions:
            game_state["action"] = " ".join(actions)
    
    def _normalize_game_state(self, game_state: Dict[str, Any]) -> None:
        """Apply data normalization and validation to ensure usable game state."""
        # Ensure position is valid
        if game_state["position"] == "unknown":
            game_state["position"] = "BTN"  # Default to button if unknown
        
        # Ensure pot makes sense
        if game_state["pot"] < 0.5:
            game_state["pot"] = 1.5  # Standard preflop pot (SB + BB)
        
        # Ensure stack sizes are reasonable
        if game_state["stack_btn"] <= 0:
            game_state["stack_btn"] = 100.0
        if game_state["stack_bb"] <= 0:
            game_state["stack_bb"] = 100.0
        
        # Make sure street is valid 
        if game_state["street"] not in ["preflop", "flop", "turn", "river"]:
            game_state["street"] = "preflop"  # Default

class PokerResponseManager:
    """Modern architecture using OpenAI's Chat Completions API for poker decision-making."""
    
    def __init__(self):
        self.models = {}
        self.load_models()
        self.parser = AdvancedPokerParser()
    
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
    
    def process_poker_image(self, image_file) -> Dict[str, Any]:
        """
        Process a poker table screenshot to extract information and generate advice.
        
        Args:
            image_file: Image file object
        
        Returns:
            Dict with extraction results and poker advice
        """
        # Extract text from image
        ocr_lines = extract_table_state(image_file)
        
        # Parse game state with advanced parser
        game_state = self.parser.parse_ocr_results(ocr_lines)
        
        # Get advice using the poker response manager
        advice = self.get_poker_advice(game_state, False)
        
        return {
            "suggestion": advice,
            "game_state": game_state,
            "ocr_lines": ocr_lines
        }
    
    def get_poker_advice(self, game_state: Dict, tournament_mode: bool = False) -> str:
        """Get poker advice based on game state."""
        try:
            # Select appropriate assistant based on game state
            assistant_num = self.select_assistant(game_state, tournament_mode)
            
            logger.info(f"Selected assistant {assistant_num} for analysis")
            
            # Format game state into prompt
            prompt = self.format_prompt(game_state)
            
            # Append assistant number to prompt
            prompt = f"{assistant_num}. {prompt}"
            
            # Get response
            raw_response = self.get_model_response(assistant_num, prompt)
            
            # Normalize response
            normalized_response = self.normalize_response(raw_response)
            
            return normalized_response
        
        except Exception as e:
            logger.error(f"Error getting poker advice: {str(e)}")
            return f"ERROR: {str(e)}"

# Create global instance
response_manager = PokerResponseManager()

def extract_table_state(image_file):
    """Extract and format table state from image file."""
    try:
        # Ensure the image file pointer is at the start
        if hasattr(image_file, 'seek'):
            image_file.seek(0)
            
        # Open image from file data
        image = Image.open(image_file)
        
        # Extract text using OCR
        image_np = np.array(image)
        
        # EasyOCR requires numpy array in BGR format (OpenCV style)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            # Convert RGB to BGR if needed
            image_np = image_np[:, :, ::-1].copy()
        
        result = reader.readtext(image_np)
        lines = [entry[1].strip() for entry in result if entry[1].strip()]
        logger.info(f"OCR extracted {len(lines)} lines of text")
        return lines
    except Exception as e:
        logger.error(f"Error extracting table state: {str(e)}")
        raise

def create_app():
    app = Flask(__name__)
    
    @app.route("/api/advice", methods=["POST"])
    def api_advice():
        logger.info(f"Received API request from {request.remote_addr}")
        
        # Debug request information
        logger.debug(f"Content-Type: {request.content_type}")
        logger.debug(f"Request files keys: {list(request.files.keys())}")
        
        try:
            # Check if we have files in the request
            if request.files:
                # Get the first file (regardless of key name)
                file_key = next(iter(request.files))
                image_file = request.files[file_key]
                
                logger.info(f"Processing image from multipart form with key: {file_key}")
                
                # Process the poker image
                result = response_manager.process_poker_image(image_file)
                
                # Return the result
                return jsonify(result)
            else:
                return jsonify({
                    "error": "No image file found in request",
                    "request_files_keys": list(request.files.keys()),
                    "content_type": request.content_type
                }), 400
                
        except Exception as e:
            logger.exception(f"Error processing request: {str(e)}")
            return jsonify({
                "error": str(e)
            }), 500
    
    return app

if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Poker Assistant Server")
    logger.info(f"Loaded {len(response_manager.models)} assistant configurations")
    app.run(debug=True, host="0.0.0.0", port=5000)
