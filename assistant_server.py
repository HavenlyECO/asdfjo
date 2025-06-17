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
        """Select the appropriate assistant based on game state."""
        street = game_state.get("street", "").lower()
        action = game_state.get("action", "").lower()

        # ICM Tournament logic takes priority if in tournament mode
        if tournament_mode:
            return 10  # Tournament ICM Assistant

        # Check for preflop scenarios
        if street == "preflop":
            if "3bet" in action or "3-bet" in action:
                return 3  # 3-Bet Situations Assistant
            return 1  # Preflop Position Strategy Assistant

        # Check for stack-based scenarios
        stack_btn = game_state.get("stack_btn", 0)
        stack_bb = game_state.get("stack_bb", 0)
        if stack_btn < 20 or stack_bb < 20 or stack_btn > 100 or stack_bb > 100:
            return 2  # Stack vs Stack Matchups Assistant

        # Check for postflop scenarios
        if street in ["turn", "river"]:
            return 4  # Bluff Catching Postflop Assistant

        # Check for polarized ranges
        if "check-raise" in action or "all-in" in action:
            return 7  # Polarization Assistant

        # Check for heads-up scenarios
        player_count = game_state.get("player_count", 0)
        if player_count == 2:
            return 9  # Heads-Up Assistant

        # Check for SPR-based decisions (stack-to-pot ratio) in postflop
        if street in ["flop", "turn", "river"] and game_state.get("pot", 0) > 0:
            return 8  # SPR/Depth Postflop Assistant

        # Default to GTO Assistant
        return 6

    def format_prompt(self, game_state: Dict) -> str:
        """Format game state into natural language prompt."""
        street = game_state.get("street", "unknown street")
        position = game_state.get("position", "unknown position")
        stack_btn = game_state.get("stack_btn", "unknown")
        stack_bb = game_state.get("stack_bb", "unknown")
        pot = game_state.get("pot", "unknown")
        action = game_state.get("action", "unknown")

        prompt = f"{street.capitalize()}. You are on the {position} with {stack_btn}BB. "

        if "bb" in position.lower():
            prompt += f"The button has {stack_btn}BB. "
        else:
            prompt += f"The big blind has {stack_bb}BB. "

        if action != "unknown":
            prompt += f"{action}. "

        prompt += f"Pot is {pot}BB. What is the optimal move?"

        # Add the required ending
        prompt += "\n\nRespond with only one recommendation: FOLD, CALL, or RAISE (include amount). Do not explain or elaborate."

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
