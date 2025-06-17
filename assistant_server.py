import os
import time
from typing import Dict, Optional, List
from flask import Flask, request, jsonify
from openai import OpenAI
from PIL import Image
import easyocr
import numpy as np
import json
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Core configuration
API_KEY = os.environ.get("POKER_ASSISTANT_API_KEY", "make-this-value-secure")
ALLOWED_IPS = os.environ.get("POKER_ASSISTANT_ALLOWED_IPS", "127.0.0.1,69.110.58.72").split(",")

# Initialize OCR
reader = easyocr.Reader(["en"], gpu=False)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Assistant Management - Architectural pattern for multiple assistants
class AssistantManager:
    """Manages multiple OpenAI assistants for poker decision-making."""

    def __init__(self):
        self.assistants = {}
        self.load_assistants()
        # Initialize all assistant threads with priming message
        self.threads = {}
        self.prime_all_assistants()

    def load_assistants(self):
        """Load assistant IDs from environment variables."""
        for i in range(1, 11):
            assistant_id = os.environ.get(f"ASSISTANT_{i}")
            if assistant_id:
                self.assistants[i] = assistant_id
            else:
                print(f"Warning: ASSISTANT_{i} not found in environment variables")

    def prime_all_assistants(self):
        """Initialize all assistant threads with priming message."""
        for assistant_num, assistant_id in self.assistants.items():
            thread = client.beta.threads.create()
            self.threads[assistant_num] = thread.id

            # Send priming message
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content="wait for game data to make decision"
            )

            # Run assistant to process priming message
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id,
                stream=True
            )

            # Wait for completion
            for event in run:
                if event.event == "thread.run.completed":
                    break

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

    def get_assistant_response(self, assistant_num: int, user_input: str) -> str:
        """Get response from a specific assistant."""
        assistant_id = self.assistants.get(assistant_num)
        thread_id = self.threads.get(assistant_num)

        if not assistant_id:
            return "Error: Assistant not found"

        if not thread_id:
            # Create thread if not exists (shouldn't happen with priming)
            thread = client.beta.threads.create()
            thread_id = thread.id
            self.threads[assistant_num] = thread_id

        # Add user message to thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input
        )

        # Run assistant
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            stream=True
        )

        # Wait for completion
        for event in run:
            if event.event == "thread.run.completed":
                break

        # Get the response
        messages = client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc"  # Get newest messages first
        )

        # Parse assistant response
        for message in messages.data:
            if message.role == "assistant":
                for content_part in message.content:
                    if content_part.type == "text":
                        return content_part.text.value

        return "No response received from assistant."

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

# Create assistant manager
assistant_manager = AssistantManager()

def route_from_ocr(game_state: Dict, tournament_mode: bool = False) -> str:
    """Route OCR game state to appropriate assistant and get poker decision."""
    try:
        assistant_num = assistant_manager.select_assistant(game_state, tournament_mode)
        prompt = assistant_manager.format_prompt(game_state)
        prompt = f"{assistant_num}. {prompt}"
        raw_response = assistant_manager.get_assistant_response(assistant_num, prompt)
        normalized_response = assistant_manager.normalize_response(raw_response)
        return normalized_response
    except Exception as e:
        return f"ERROR: {str(e)}"

def create_app():
    app = Flask(__name__)
    
    
    @app.route("/api/advice", methods=["POST"])
    def api_advice():
        if not check_auth(request):
            return jsonify({"error": "Unauthorized"}), 403
            
        try:
            image_data = request.files.get("image")
            if not image_data:
                return jsonify({"error": "No image provided"}), 400

            image = Image.open(image_data)

            ocr_lines = extract_table_state(image)
            game_state = parse_game_state(ocr_lines)

            advice = route_from_ocr(game_state)

            return jsonify({"suggestion": advice})
        except Exception as e:
            app.logger.error(f"Error processing request: {e}")
            return jsonify({"error": str(e)}), 500
    
    return app

def check_auth(req):
    client_ip = req.remote_addr
    api_key = req.headers.get("X-API-Key", "")
    if client_ip not in ALLOWED_IPS and api_key != API_KEY:
        return False
    return True

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

def extract_table_state(image):
    """Extract and format table state from screen capture."""
    text_eo = extract_text_easyocr(image)
    lines = [line.strip() for line in text_eo.split("\n") if line.strip()]
    return lines

def parse_game_state(text_lines):
    """Advanced game state parser with comprehensive detection."""
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
    
    return state

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
