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

# Load environment variables
load_dotenv()

# Core configuration
API_KEY = os.environ.get("POKER_ASSISTANT_API_KEY", "make-this-value-secure")
ALLOWED_IPS = os.environ.get("POKER_ASSISTANT_ALLOWED_IPS", "127.0.0.1,69.110.58.72").split(",")

# Initialize OCR
reader = easyocr.Reader(["en"], gpu=False)

# Assistant Management - Architectural pattern for multiple assistants
class AssistantsManager:
    """Manages multiple OpenAI assistants with thread initialization and interaction."""
    
    def __init__(self):
        self.client = OpenAI()
        self.assistants: Dict[str, str] = {}
        self.active_threads: Dict[str, str] = {}
        self.load_assistant_ids()
        
    def load_assistant_ids(self):
        """Load assistant IDs from environment variables."""
        for i in range(1, 11):  # Assistants 1-10
            assistant_id = os.environ.get(f"POKER_ASSISTANT_ID_{i}")
            if assistant_id:
                self.assistants[f"assistant_{i}"] = assistant_id
    
    def initialize_threads(self):
        """Initialize all assistant threads with priming message."""
        for assistant_name, assistant_id in self.assistants.items():
            thread_id = self._create_and_prime_thread(assistant_id)
            self.active_threads[assistant_name] = thread_id
            print(f"Initialized {assistant_name} with thread {thread_id}")
    
    def _create_and_prime_thread(self, assistant_id: str) -> str:
        """Create a new thread and prime it with the standard waiting message."""
        thread = self.client.beta.threads.create()
        
        # Add the standard priming message
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="wait for game data to make decision"
        )
        
        # Run the assistant to process the priming message
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
            stream=True
        )
        
        # Wait for completion
        for event in run:
            if event.event == "thread.run.completed":
                break
        
        return thread.id
    
    def ask_assistant(self, assistant_number: int, user_input: str) -> str:
        """Send a message to a specific assistant and get its response."""
        assistant_name = f"assistant_{assistant_number}"
        
        if assistant_name not in self.assistants:
            return f"Error: Assistant {assistant_number} not found"
            
        assistant_id = self.assistants[assistant_name]
        thread_id = self.active_threads.get(assistant_name)
        
        if not thread_id:
            # Initialize thread if not yet done
            thread_id = self._create_and_prime_thread(assistant_id)
            self.active_threads[assistant_name] = thread_id
        
        # Add the user's message to the thread
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input
        )
        
        # Run the assistant on the thread
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            stream=True
        )
        
        # Monitor the run through streaming updates
        for event in run:
            if event.event == "thread.run.completed":
                break
        
        # Get the assistant's messages from the thread
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc"  # Get newest messages first
        )
        
        # Return the latest assistant message
        for message in messages.data:
            if message.role == "assistant":
                for content_part in message.content:
                    if content_part.type == "text":
                        return content_part.text.value
        
        return "No response received from assistant."

# Create assistant manager
assistants_manager = AssistantsManager()

def create_app():
    app = Flask(__name__)
    
    # Initialize all assistant threads on startup
    assistants_manager.initialize_threads()
    
    @app.route("/api/advice", methods=["POST"])
    def api_advice():
        if not check_auth(request):
            return jsonify({"error": "Unauthorized"}), 403
            
        try:
            # Extract image from request
            image_data = request.files.get("image")
            if not image_data:
                return jsonify({"error": "No image provided"}), 400
                
            image = Image.open(image_data)
            
            # Extract text from image using optimized OCR
            ocr_lines = extract_table_state(image)
            
            # Parse game state from OCR text
            game_state = parse_game_state(ocr_lines)
            
            # Use assistant 2 for poker advice (as specified in requirements)
            context = {
                "game_state": game_state,
                "ocr_text": "\n".join(ocr_lines),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            }
            
            # Get advice from assistant 2
            prompt = f"""
            Analyze this poker situation and provide specific, actionable advice:
            
            OCR TEXT:
            {context['ocr_text']}
            
            GAME STATE:
            Pot: {game_state.get('pot', 'Unknown')}
            Betting round: {game_state.get('betting_round', 'Unknown')}
            Community cards: {game_state.get('community_cards', 'Unknown')}
            
            Provide specific advice with ACTIONS (FOLD/CALL/RAISE) and reasoning.
            """
            
            advice = assistants_manager.ask_assistant(2, prompt)
            
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
