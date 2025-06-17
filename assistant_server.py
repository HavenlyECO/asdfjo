"""
Poker Assistant Cloud Server (Flask API)
----------------------------------------

Requirements:
- Python 3.8+
- Flask
- pytesseract
- Pillow
- easyocr
- (optionally: gunicorn for production)

Setup Instructions:
1. Install system dependencies for Tesseract:
   - Ubuntu: sudo apt-get install tesseract-ocr
   - Windows: Download and install from https://github.com/tesseract-ocr/tesseract
2. Install required Python packages:
   pip install -r requirements.txt
3. Place your GTO chart in gto_chart.json (sample format included).
4. (Recommended) Run behind HTTPS reverse proxy or use gunicorn+certbot for production.
5. Set your API key and allowed IPs in the config section below.

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

# --- CONFIGURATION ---
API_KEY = os.environ.get("POKER_ASSISTANT_API_KEY", "changeme")  # Set securely in prod!
ALLOWED_IPS = os.environ.get("POKER_ASSISTANT_ALLOWED_IPS", "127.0.0.1").split(",")
GTO_CHART_PATH = "gto_chart.json"

# --- Flask App Setup ---
app = Flask(__name__)

# --- Load GTO Chart on Startup ---

def load_gto_chart(path):
    with open(path, "r") as f:
        return json.load(f)

GTO_CHART = load_gto_chart(GTO_CHART_PATH)

# --- Security Helper ---

def check_auth(request):
    # Simple API key or IP whitelist check
    client_ip = request.remote_addr
    api_key = request.headers.get("X-API-Key", "")
    if client_ip not in ALLOWED_IPS and api_key != API_KEY:
        return False
    return True

# --- OCR Functions ---

def extract_text_pytesseract(image):
    """Extract text from image using pytesseract."""
    return pytesseract.image_to_string(image)


def extract_text_easyocr(image):
    """Extract text from image using EasyOCR."""
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image)
    # Concatenate detected texts
    return "\n".join([entry[1] for entry in result])


def extract_table_state(image):
    """
    Try both OCR engines and return lines of visible text.
    """
    text_pt = extract_text_pytesseract(image)
    text_eo = extract_text_easyocr(image)
    # Combine, deduplicate, clean
    lines = set(text_pt.splitlines() + text_eo.splitlines())
    lines = [l.strip() for l in lines if l.strip()]
    return lines

# --- Poker Table State Parser ---

def parse_game_state(text_lines):
    """
    Parse the OCR text lines to extract pot, stacks, positions, bets.
    (This is a minimal stub; for real use, improve regexes and logic.)
    """
    import re

    state = {
        "pot": None,
        "player_stacks": {},
        "positions": {},
        "bet_sizes": {},
        "raw_lines": text_lines,
    }
    # Example regex for pot size: "Pot: 1500" or "Pot 1,500"
    for line in text_lines:
        if "pot" in line.lower():
            pot_match = re.search(r"pot[:\s]*([\d,\.]+)", line, re.IGNORECASE)
            if pot_match:
                state["pot"] = pot_match.group(1).replace(",", "")
        # Example for stack sizes: "Player1 12000", "SB: 3500"
        stack_match = re.search(
            r"(SB|BB|UTG|MP|CO|BU|BTN|Player\d?)[:\s]*([\d,\.]+)",
            line,
            re.IGNORECASE,
        )
        if stack_match:
            pos = stack_match.group(1).upper()
            size = stack_match.group(2).replace(",", "")
            state["player_stacks"][pos] = size
        # Example for bet sizes: "Bet: 500"
        bet_match = re.search(r"bet[:\s]*([\d,\.]+)", line, re.IGNORECASE)
        if bet_match:
            state["bet_sizes"][line] = bet_match.group(1).replace(",", "")
    return state

# --- GTO Chart Logic ---

def get_gto_advice(game_state, gto_chart):
    """
    Match parsed state against GTO chart and return suggestion.
    This is a simple lookup; real logic can be plugged in later.
    """
    # Example: Match by effective stack and pot size (very basic)
    pot = game_state.get("pot")
    # Find closest match in chart
    if not pot:
        return "Unable to determine pot size for advice."

    try:
        pot = int(float(pot))
    except Exception:
        return "Invalid pot size detected."

    # Find a matching situation in the chart (this is just a stub)
    for entry in gto_chart.get("situations", []):
        if abs(entry.get("pot", 0) - pot) < entry.get("pot_tolerance", 200):
            return entry.get("suggestion", "No advice found.")
    return "No advice for current situation."

# --- Flask Endpoint ---

@app.route("/api/advice", methods=["POST"])

def api_advice():
    if not check_auth(request):
        abort(403, "Unauthorized: Check API key or IP whitelist.")

    if "screenshot" not in request.files:
        return jsonify({"error": "No screenshot provided"}), 400

    screenshot = request.files["screenshot"]
    image = Image.open(screenshot.stream).convert("RGB")

    # Step 1: Extract OCR text
    ocr_lines = extract_table_state(image)

    # Step 2: Parse game state
    parsed_state = parse_game_state(ocr_lines)

    # Step 3: GTO suggestion
    suggestion = get_gto_advice(parsed_state, GTO_CHART)

    # Step 4: Respond
    return jsonify({"suggestion": suggestion, "raw_data": parsed_state})

# --- Security/Deployment Advice ---
"""
To enable HTTPS on DigitalOcean:
- Use a reverse proxy (nginx) and Certbot for Let's Encrypt certificates.
- Or use gunicorn with --certfile and --keyfile for Flask SSL.
- Always change API_KEY and don't expose the server to the public internet without authentication.
"""

# --- Main Entry Point ---

if __name__ == "__main__":
    # Debug only, use gunicorn or similar for production
    app.run(host="0.0.0.0", port=5000, debug=True)
