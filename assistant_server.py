"""
Poker Assistant Cloud Server (Flask API)
----------------------------------------

Requirements:
- Python 3.8+
- Flask
- pytesseract
- Pillow
- easyocr

Setup Instructions:
1. Install system dependencies for Tesseract:
   - Ubuntu: sudo apt-get install tesseract-ocr
   - Windows: Download and install from https://github.com/tesseract-ocr/tesseract
2. Install required Python packages:
   pip install -r requirements.txt
3. Place your GTO chart in gto_chart.json (sample format included).
4. (Recommended) Run behind HTTPS reverse proxy or use gunicorn+certbot for production.
5. Set your API key and allowed IPs in the configuration section below.

Run:
    python assistant_server.py

Author: <Your Name>
"""

import os
import io
import json
from flask import Flask, request, jsonify, abort
from PIL import Image
import pytesseract
import easyocr
import numpy as np  # Add this import at the top of the file

# --- CONFIGURATION ---
API_KEY = os.environ.get("POKER_ASSISTANT_API_KEY", "changeme")
ALLOWED_IPS = os.environ.get("POKER_ASSISTANT_ALLOWED_IPS", "127.0.0.1").split(",")
GTO_CHART_PATH = "gto_chart.json"


def load_gto_chart(path: str):
    with open(path, "r") as f:
        return json.load(f)


# --- Security Helper ---

def check_auth(req):
    client_ip = req.remote_addr
    api_key = req.headers.get("X-API-Key", "")
    if client_ip not in ALLOWED_IPS and api_key != API_KEY:
        return False
    return True


# --- OCR Functions ---

def extract_text_pytesseract(image: Image.Image) -> str:
    """Extract text from image using pytesseract."""
    return pytesseract.image_to_string(image)


def extract_text_easyocr(image: Image.Image) -> str:
    """Extract text from image using EasyOCR."""
    reader = easyocr.Reader(["en"], gpu=False)

    # Convert PIL Image to numpy array for EasyOCR compatibility
    image_np = np.array(image)

    # EasyOCR requires numpy array in BGR format (OpenCV style)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        # Convert RGB to BGR if needed
        image_np = image_np[:, :, ::-1].copy()

    result = reader.readtext(image_np)
    return "\n".join(entry[1] for entry in result)


def extract_table_state(image: Image.Image):
    text_pt = extract_text_pytesseract(image)
    text_eo = extract_text_easyocr(image)
    lines = set(text_pt.splitlines() + text_eo.splitlines())
    return [l.strip() for l in lines if l.strip()]


# --- Poker Table State Parser ---

def parse_game_state(text_lines):
    """Advanced game state parser that captures more poker context."""
    import re

    state = {
        "pot": None,
        "player_stacks": {},
        "positions": {},
        "bet_sizes": {},
        "action": None,
        "betting_round": None,
        "cards_on_board": None,
        "raw_lines": text_lines,
    }

    betting_round_indicators = {
        "preflop": ["preflop", "hole cards", "dealt"],
        "flop": ["flop", "board:", "community:"],
        "turn": ["turn", "4th street"],
        "river": ["river", "5th street"],
    }

    for line in text_lines:
        # Identify betting round
        for round_name, indicators in betting_round_indicators.items():
            if any(indicator in line.lower() for indicator in indicators):
                state["betting_round"] = round_name
                break

        # Detect community cards
        cards_match = re.search(
            r"board:?\s*([2-9TJQKA][cdhs]\s*[2-9TJQKA][cdhs]\s*[2-9TJQKA][cdhs](\s*[2-9TJQKA][cdhs])?(\s*[2-9TJQKA][cdhs])?)",
            line,
            re.IGNORECASE,
        )
        if cards_match:
            state["cards_on_board"] = cards_match.group(1)

        # Current action detection
        action_match = re.search(
            r"action:?\s*(call|raise|bet|fold|check)",
            line,
            re.IGNORECASE,
        )
        if action_match:
            state["action"] = action_match.group(1).upper()

        # Pot detection logic
        if "pot" in line.lower():
            pot_match = re.search(r"pot[:\s]*([\d,\.]+)", line, re.IGNORECASE)
            if pot_match:
                state["pot"] = pot_match.group(1).replace(",", "")

        # ACR-specific formats
        pot_match = re.search(r"[\$€£][\s]*([\d,\.]+)", line, re.IGNORECASE)
        if pot_match and not state["pot"]:
            state["pot"] = pot_match.group(1).replace(",", "")

        # Continue with existing stack and bet parsing
        stack_match = re.search(
            r"(SB|BB|UTG|MP|CO|BU|BTN|Player\d?)[:\s]*([\d,\.]+)",
            line,
            re.IGNORECASE,
        )
        if stack_match:
            pos = stack_match.group(1).upper()
            size = stack_match.group(2).replace(",", "")
            state["player_stacks"][pos] = size
        bet_match = re.search(r"bet[:\s]*([\d,\.]+)", line, re.IGNORECASE)
        if bet_match:
            state["bet_sizes"][line] = bet_match.group(1).replace(",", "")

    return state


# --- GTO Chart Logic ---

def get_gto_advice(game_state, gto_chart):
    """Action-oriented GTO advice engine that recommends specific actions."""
    pot = game_state.get("pot")
    if not pot:
        return "Unable to determine pot size. Try adjusting the screen capture region."

    try:
        pot_value = float(pot.replace(",", ""))
    except Exception:
        return f"Invalid pot size detected: {pot}"

    # Dynamic BB estimation based on pot size and game state
    estimated_bb = 2.0  # Default assumption
    if pot_value < 10:
        estimated_bb = 0.1  # Micro stakes
    elif pot_value < 25:
        estimated_bb = 0.25  # Small stakes

    # Convert to BB units
    pot_in_bb = pot_value / estimated_bb

    # Determine game phase (preflop, flop, etc.)
    betting_round = game_state.get("betting_round", "preflop")

    # Find best matching situation
    best_match = None
    min_distance = float('inf')

    for situation in gto_chart.get("situations", []):
        if situation.get("situation_type") == betting_round:
            distance = abs(situation.get("pot_in_bb", 0) - pot_in_bb)
            tolerance = situation.get("tolerance_in_bb", 5)

            if distance <= tolerance and distance < min_distance:
                min_distance = distance
                best_match = situation

    if best_match:
        # Construct detailed advice with specific action
        hand_strength = best_match.get("hand_strength", {})

        advice = [
            f"Current situation: {betting_round.upper()} with pot ~{pot_value} ({int(pot_in_bb)}BB)",
            f"\nRECOMMENDED ACTIONS:",
            f"- PREMIUM HANDS: {hand_strength.get('premium', {}).get('action', 'CONTINUE')} " +
            f"{hand_strength.get('premium', {}).get('sizing', '')}",
            f"- STRONG HANDS: {hand_strength.get('strong', {}).get('action', 'CALL')} " +
            f"{hand_strength.get('strong', {}).get('sizing', '')}",
            f"- MEDIUM HANDS: {hand_strength.get('medium', {}).get('action', 'CHECK-CALL')}",
            f"- WEAK HANDS: {hand_strength.get('weak', {}).get('action', 'FOLD')}"
        ]

        detailed_explanation = hand_strength.get(
            'strong', {}).get('explanation', '') + " " + hand_strength.get(
            'medium', {}).get('explanation', '')

        advice.append(f"\nEXPLANATION: {detailed_explanation}")

        return "\n".join(advice)

    # Even the "no match" case provides concrete actions
    return ("Pot appears to be non-standard size. General advice:\n"
            "- PREMIUM HANDS: RAISE 3x BB\n"
            "- STRONG HANDS: CALL\n"
            "- WEAK HANDS: FOLD\n"
            "Adjust based on player tendencies and position.")


# --- Flask App Factory ---

def create_app():
    app = Flask(__name__)
    gto_chart = load_gto_chart(GTO_CHART_PATH)

    @app.route("/api/advice", methods=["POST"])
    def api_advice():
        if not check_auth(request):
            abort(403, "Unauthorized: Check API key or IP whitelist.")
        if "screenshot" not in request.files:
            return jsonify({"error": "No screenshot provided"}), 400

        screenshot = request.files["screenshot"]
        image = Image.open(screenshot.stream).convert("RGB")
        ocr_lines = extract_table_state(image)
        parsed_state = parse_game_state(ocr_lines)
        suggestion = get_gto_advice(parsed_state, gto_chart)
        return jsonify({"suggestion": suggestion, "raw_data": parsed_state})

    return app


app = create_app()

"""
Security/Deployment Advice:
- Use a reverse proxy (nginx) and Certbot for Let's Encrypt certificates.
- Or use gunicorn with --certfile and --keyfile for Flask SSL.
- Always change API_KEY and don't expose the server to the public internet without authentication.
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
