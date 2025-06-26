"""
Screenshot capture server for remote clients
Provides API endpoints for receiving and serving screenshots
"""

import os
import time
import logging
import io
from pathlib import Path

from flask import Flask, request, send_file, jsonify, Response
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("screenshot_server")

# Application configuration
APP_HOST = os.environ.get("CAPTURE_SERVER_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("CAPTURE_SERVER_PORT", "5002"))
APP_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")

# Screenshot storage settings
SAVE_SCREENSHOTS = os.environ.get("SAVE_SCREENSHOTS", "true").lower() in ("true", "1", "yes")
SCREENSHOT_DIR = Path(os.environ.get("SCREENSHOT_DIR", "captured_screenshots"))

# Create screenshot directory
if SAVE_SCREENSHOTS:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Screenshots will be saved to: {SCREENSHOT_DIR}")
else:
    logger.info("Screenshot saving is disabled")

# Initialize Flask application
app = Flask(__name__)

# In-memory storage for latest screenshot
latest_screenshot = None
latest_screenshot_time = 0

# --- Helper Functions ---
def add_cors_headers(response):
    """Add CORS headers to enable cross-origin requests"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key'
    return response

# --- API Endpoints ---
@app.route("/api/capture", methods=["GET", "POST", "OPTIONS"])
def capture():
    """Upload or retrieve screenshots"""
    global latest_screenshot, latest_screenshot_time
    
    # Handle preflight CORS requests
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        return add_cors_headers(response)
    
    # Handle screenshot upload (POST)
    if request.method == "POST":
        try:
            # Check if the request has the file part
            if 'screenshot' not in request.files:
                logger.warning("No screenshot file in request")
                return add_cors_headers(jsonify({"error": "No screenshot file in request"})), 400
            
            file = request.files['screenshot']
            
            # Check if the file is empty
            if file.filename == '':
                logger.warning("Empty file submitted")
                return add_cors_headers(jsonify({"error": "Empty file submitted"})), 400
            
            # Read and store the screenshot
            screenshot_data = file.read()
            
            # Update the latest screenshot in memory
            latest_screenshot = screenshot_data
            latest_screenshot_time = time.time()
            
            # Save to disk if enabled
            if SAVE_SCREENSHOTS:
                timestamp = time.strftime("%Y%m%d-%H%M%S-%f")
                filename = f"capture_{timestamp}.png"
                filepath = SCREENSHOT_DIR / filename
                
                with open(filepath, 'wb') as f:
                    f.write(screenshot_data)
                    
                logger.info(f"Saved uploaded screenshot to {filepath}")
            else:
                logger.info("Received screenshot (not saved to disk)")
            
            return add_cors_headers(jsonify({"status": "success", "timestamp": latest_screenshot_time}))
            
        except Exception as e:
            logger.error(f"Error processing screenshot upload: {e}")
            return add_cors_headers(jsonify({"error": str(e)})), 500
    
    # Handle screenshot retrieval (GET)
    elif request.method == "GET":
        try:
            # Check if we have a screenshot available
            if latest_screenshot is None:
                logger.warning("No screenshot available")
                return add_cors_headers(jsonify({"error": "No screenshot available"})), 404
            
            # Return the screenshot
            response = Response(latest_screenshot, mimetype="image/png")
            return add_cors_headers(response)
            
        except Exception as e:
            logger.error(f"Error retrieving screenshot: {e}")
            return add_cors_headers(jsonify({"error": str(e)})), 500

@app.route("/api/health", methods=["GET", "OPTIONS"])
def health_check():
    """Server health check endpoint"""
    # Handle preflight CORS requests
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        return add_cors_headers(response)
        
    response = jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "service": "screenshot-server",
        "save_screenshots": SAVE_SCREENSHOTS,
        "latest_screenshot_time": latest_screenshot_time if latest_screenshot else None
    })
    
    return add_cors_headers(response)

@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API information"""
    html = """
    <html>
        <head>
            <title>Screenshot Server</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .endpoint { background: #f4f4f4; padding: 10px; margin: 10px 0; border-radius: 4px; }
                code { background: #eee; padding: 2px 4px; }
            </style>
        </head>
        <body>
            <h1>Screenshot Server</h1>
            <p>This server allows clients to upload screenshots and serves them via API.</p>
            
            <h2>Available endpoints:</h2>
            
            <div class="endpoint">
                <h3>/api/capture</h3>
                <p><strong>GET</strong>: Returns the latest screenshot</p>
                <p><strong>POST</strong>: Upload a new screenshot (multipart/form-data with 'screenshot' field)</p>
            </div>
            
            <div class="endpoint">
                <h3>/api/health</h3>
                <p>Server health check and status</p>
            </div>
        </body>
    </html>
    """
    
    return html

# --- Main Entry Point ---
if __name__ == "__main__":
    logger.info(f"Starting screenshot server on {APP_HOST}:{APP_PORT}")
    logger.info(f"Debug mode: {APP_DEBUG}")
    app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG)