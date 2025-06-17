import os
from flask import Flask, request, jsonify
from openai import OpenAI
from PIL import Image
import easyocr
import numpy as np
import json

# Core configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ASSISTANT_ID = os.environ.get("POKER_ASSISTANT_ID", "")  # Create this in OpenAI dashboard
API_KEY = os.environ.get("POKER_ASSISTANT_API_KEY", "make-this-value-secure")
ALLOWED_IPS = os.environ.get("POKER_ASSISTANT_ALLOWED_IPS", "127.0.0.1,69.110.58.72").split(",")

# Initialize components
client = OpenAI(api_key=OPENAI_API_KEY)
reader = easyocr.Reader(["en"], gpu=False)

def create_app():
    app = Flask(__name__)
    
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
            
            # Get LLM-powered GTO advice
            advice = get_llm_advice(game_state, ocr_lines)
            
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
            
        # Other detection logic continues...
    
    return state

def get_llm_advice(game_state, ocr_text):
    """Get poker advice using OpenAI Assistant - properly architected LLM integration."""
    # Create a thread for this analysis
    thread = client.beta.threads.create()
    
    # Prepare context for the Assistant
    context = {
        "game_state": game_state,
        "ocr_text": "\n".join(ocr_text),
        "timestamp": "2025-06-17 11:36:50"  # Use real timestamp in production
    }
    
    # Add the context as a message to the thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"""
        Analyze this poker situation and provide specific, actionable advice:
        
        OCR TEXT:
        {context['ocr_text']}
        
        GAME STATE:
        Pot: {game_state.get('pot', 'Unknown')}
        Betting round: {game_state.get('betting_round', 'Unknown')}
        Community cards: {game_state.get('community_cards', 'Unknown')}
        
        Provide specific advice with ACTIONS (FOLD/CALL/RAISE) and reasoning.
        """
    )
    
    # Run the Assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )
    
    # Wait for completion
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run_status.status == "completed":
            break
        time.sleep(0.5)
    
    # Get the response
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    
    # Parse the response to extract advice
    assistant_message = next((msg for msg in messages.data if msg.role == "assistant"), None)
    if assistant_message:
        advice = assistant_message.content[0].text.value
    else:
        advice = "Unable to generate advice for the current situation."
    
    return advice

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
