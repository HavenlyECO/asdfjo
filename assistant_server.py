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
    """Parse OCR text lines to extract pot, stacks, and bets with improved ACR detection."""
    import re

    state = {
        "pot": None,
        "player_stacks": {},
        "positions": {},
        "bet_sizes": {},
        "raw_lines": text_lines,
    }
    
    # Enhanced pot detection for ACR format
    for line in text_lines:
        # Standard pot format
        if "pot" in line.lower():
            pot_match = re.search(r"pot[:\s]*([\d,\.]+)", line, re.IGNORECASE)
            if pot_match:
                state["pot"] = pot_match.group(1).replace(",", "")
        
        # ACR specific formats - look for currency symbols near numbers
        pot_match = re.search(r"[\$€£][\s]*([\d,\.]+)", line, re.IGNORECASE)
        if pot_match and not state["pot"]:
            state["pot"] = pot_match.group(1).replace(",", "")
        
        # Look for isolated numbers that might be pot sizes
        pot_match = re.search(r"^[\s]*([\d,\.]+)[\s]*$", line)
        if pot_match and not state["pot"] and float(pot_match.group(1).replace(",", "")) > 10:
            state["pot"] = pot_match.group(1).replace(",", "")
            
        # Continue with other parsing
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
    
    # If we still couldn't find pot, set a default for debugging
    if not state["pot"]:
        state["pot"] = "100"  # Default fallback for testing - will be removed after calibration
        
    return state


# --- GTO Chart Logic ---

def get_gto_advice(game_state, gto_chart):
    pot = game_state.get("pot")
    if not pot:
        return "Unable to determine pot size for advice."
    try:
        pot = int(float(pot))
    except Exception:
        return "Invalid pot size detected."
    for entry in gto_chart.get("situations", []):
        if abs(entry.get("pot", 0) - pot) < entry.get("pot_tolerance", 200):
            return entry.get("suggestion", "No advice found.")
    return "No advice for current situation."


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
