"""Flask server for saving uploaded screenshots."""

import os
import time
import logging
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("screenshot_server")

SCREENSHOT_DIR = Path("data_screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

@app.route("/api/save_screenshot", methods=["POST"])
def save_screenshot():
    if 'screenshot' not in request.files:
        return jsonify({"error": "screenshot file missing"}), 400
    img_file = request.files['screenshot']
    img_bytes = img_file.read()
    ts = time.strftime("%Y-%m-%d_%H-%M-%S_%f")
    fname = SCREENSHOT_DIR / f"shot_{ts}.png"
    with open(fname, "wb") as f:
        f.write(img_bytes)
    logger.info("Saved screenshot to %s", fname)
    return jsonify({"status": "ok", "path": str(fname)})

if __name__ == "__main__":
    port = int(os.environ.get("SCREENSHOT_SERVER_PORT", "5001"))
    app.run("0.0.0.0", port, debug=True)
