"""
Screenshot capture server for headless Ubuntu environments
Uses scrot or imagemagick to capture X11 screen
"""

import os
import time
import logging
import subprocess
import tempfile
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

# Screenshot settings
SAVE_SCREENSHOTS = os.environ.get("SAVE_SCREENSHOTS", "true").lower() in ("true", "1", "yes")
SCREENSHOT_DIR = Path(os.environ.get("SCREENSHOT_DIR", "captured_screenshots"))
SCREENSHOT_TOOL = os.environ.get("SCREENSHOT_TOOL", "import").lower()  # "import" (imagemagick) or "scrot"

# X11 display
DISPLAY = os.environ.get("DISPLAY", ":0")
os.environ["DISPLAY"] = DISPLAY

# Create screenshot directory
if SAVE_SCREENSHOTS:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Screenshots will be saved to: {SCREENSHOT_DIR}")
else:
    logger.info("Screenshots will only be kept in memory")

# Initialize Flask application
app = Flask(__name__)

# In-memory cache for latest screenshot
latest_screenshot = None
latest_screenshot_time = 0

# --- Screenshot Functionality ---
def check_tool_exists(tool_name):
    """Check if a command-line tool exists"""
    try:
        result = subprocess.run(
            ["which", tool_name], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=False
        )
        exists = result.returncode == 0
        if exists:
            logger.info(f"Found {tool_name} at {result.stdout.decode('utf-8').strip()}")
        else:
            logger.warning(f"{tool_name} not found")
        return exists
    except Exception as e:
        logger.error(f"Error checking for {tool_name}: {e}")
        return False

def check_x11_running():
    """Check if X11 is running"""
    try:
        result = subprocess.run(
            ["xdpyinfo"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            env={"DISPLAY": DISPLAY},
            check=False
        )
        running = result.returncode == 0
        if running:
            logger.info(f"X11 is running on {DISPLAY}")
        else:
            logger.warning(f"X11 is not running on {DISPLAY}: {result.stderr.decode('utf-8')}")
        return running
    except Exception as e:
        logger.error(f"Error checking X11: {e}")
        return False

def capture_screenshot():
    """Capture a screenshot using the selected tool"""
    global latest_screenshot, latest_screenshot_time
    
    timestamp = time.strftime("%Y%m%d-%H%M%S-%f")
    
    if SAVE_SCREENSHOTS:
        filename = f"capture_{timestamp}.png"
        output_path = SCREENSHOT_DIR / filename
    else:
        # Create a temporary file if not saving
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        output_path = Path(temp_file.name)
        temp_file.close()
    
    try:
        if SCREENSHOT_TOOL == "scrot":
            # Use scrot for screenshot
            cmd = ["scrot", "-z", str(output_path)]
        else:
            # Default to imagemagick's import
            cmd = ["import", "-window", "root", str(output_path)]
        
        # Capture the screenshot
        env = os.environ.copy()
        env["DISPLAY"] = DISPLAY
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=True
        )
        
        # Read the file
        with open(output_path, "rb") as f:
            latest_screenshot = f.read()
        
        latest_screenshot_time = time.time()
        
        # Delete temp file if not saving
        if not SAVE_SCREENSHOTS:
            output_path.unlink()
        
        logger.info(f"Screenshot captured at {timestamp}")
        return latest_screenshot
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Screenshot capture failed: {e.stderr.decode('utf-8')}")
        if not SAVE_SCREENSHOTS and output_path.exists():
            output_path.unlink()
        return None
    
    except Exception as e:
        logger.error(f"Error capturing screenshot: {e}")
        if not SAVE_SCREENSHOTS and output_path.exists():
            output_path.unlink()
        return None

# --- System Setup Checks ---
def setup_system():
    """Set up system prerequisites"""
    logger.info("Checking system requirements...")
    
    # Check if X11 is running
    x11_running = check_x11_running()
    if not x11_running:
        logger.warning(f"X11 is not running on {DISPLAY}")
    
    # Check for screenshot tools
    scrot_exists = check_tool_exists("scrot")
    import_exists = check_tool_exists("import")
    
    if not scrot_exists and not import_exists:
        logger.error("No screenshot tools available. Please install imagemagick or scrot.")
        return False
    
    # Set the tool to use
    global SCREENSHOT_TOOL
    if SCREENSHOT_TOOL == "scrot" and not scrot_exists:
        if import_exists:
            logger.warning("scrot not found, falling back to imagemagick import")
            SCREENSHOT_TOOL = "import"
        else:
            logger.error("No screenshot tools available")
            return False
    elif SCREENSHOT_TOOL == "import" and not import_exists:
        if scrot_exists:
            logger.warning("imagemagick import not found, falling back to scrot")
            SCREENSHOT_TOOL = "scrot"
        else:
            logger.error("No screenshot tools available")
            return False
    
    logger.info(f"Using {SCREENSHOT_TOOL} for screenshots")
    return True

# Run setup
setup_ok = setup_system()
if not setup_ok:
    logger.warning("System setup incomplete. Screenshot functionality may not work.")

# --- Helper Functions ---
def add_cors_headers(response):
    """Add CORS headers to enable cross-origin requests"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key'
    return response

# --- API Endpoints ---
@app.route("/api/capture", methods=["GET", "OPTIONS"])
def capture():
    """Get the latest screenshot"""
    # Handle preflight CORS requests
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        return add_cors_headers(response)
    
    # Force capture new screenshot
    screenshot_data = capture_screenshot()
    
    # Check if capture failed
    if screenshot_data is None:
        return add_cors_headers(jsonify({"error": "Failed to capture screenshot"})), 500
    
    # Return the screenshot
    response = Response(screenshot_data, mimetype="image/png")
    return add_cors_headers(response)

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
        "display": DISPLAY,
        "screenshot_tool": SCREENSHOT_TOOL,
        "latest_screenshot_time": latest_screenshot_time,
        "save_screenshots": SAVE_SCREENSHOTS
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
            <p>This server captures screenshots and serves them via API.</p>
            
            <h2>Available endpoints:</h2>
            
            <div class="endpoint">
                <h3>/api/capture</h3>
                <p><strong>GET</strong>: Returns the latest screenshot</p>
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
