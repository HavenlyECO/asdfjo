"""Flask server to capture screenshots via local_client."""

import os
import time
import logging
import io
from pathlib import Path

from flask import Flask, send_file, jsonify
import mss

from local_client import capture_frame

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("screenshot_capture_server")

# Directory to store captured screenshots
SCREENSHOT_DIR = Path("captured_screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

@app.route("/api/capture", methods=["GET"])
def capture():
    """Capture the current screen region and return the PNG image."""
    img = capture_frame()
    ts = time.strftime("%Y-%m-%d_%H-%M-%S_%f")
    fname = SCREENSHOT_DIR / f"capture_{ts}.png"
    mss.tools.to_png(img.rgb, img.size, output=str(fname))
    logger.info("Saved screenshot to %s", fname)
    return send_file(fname, mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("CAPTURE_SERVER_PORT", "5002"))
    app.run("0.0.0.0", port, debug=True)
