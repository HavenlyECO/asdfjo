"""
Clean screenshot capture server implementation
Uses system tools for X11 screen capture
"""

import os
import time
import logging
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, Response
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
SAVE_SCREENSHOTS = os.environ.get("SAVE_SCREENSHOTS", "false").lower() in ("true", "1", "yes")
SCREENSHOT_DIR = Path(os.environ.get("SCREENSHOT_DIR", "captured_screenshots"))

# X11 display
DISPLAY = os.environ.get("DISPLAY", ":0")
os.environ["DISPLAY"] = DISPLAY

# Create screenshot directory
if SAVE_SCREENSHOTS:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Screenshots will be saved to: {SCREENSHOT_DIR}")
else:
    logger.info("Screenshot saving is disabled")

# Initialize Flask application
app = Flask(__name__)

# --- System tool checks ---
def check_command(cmd):
    """Check if a command exists on the system"""
    try:
        result = subprocess.run(
            ["which", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking for command {cmd}: {e}")
        return False

# Check for available tools
IMPORT_AVAILABLE = check_command("import")
SCROT_AVAILABLE = check_command("scrot")
XFCE4_SCREENSHOOTER_AVAILABLE = check_command("xfce4-screenshooter")
GNOME_SCREENSHOT_AVAILABLE = check_command("gnome-screenshot")

logger.info(f"Available screenshot tools:")
logger.info(f"  ImageMagick import: {'✓' if IMPORT_AVAILABLE else '✗'}")
logger.info(f"  scrot: {'✓' if SCROT_AVAILABLE else '✗'}")
logger.info(f"  xfce4-screenshooter: {'✓' if XFCE4_SCREENSHOOTER_AVAILABLE else '✗'}")
logger.info(f"  gnome-screenshot: {'✓' if GNOME_SCREENSHOT_AVAILABLE else '✗'}")

# --- Screenshot Capture Function ---
def capture_screenshot():
    """
    Capture a screenshot using available system tools
    
    Returns:
        Tuple of (success, image_data or error_message)
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Set the environment with the correct display
    env = os.environ.copy()
    env["DISPLAY"] = DISPLAY
    
    result = None
    error = None
    command_used = None
    
    try:
        # Try ImageMagick first
        if IMPORT_AVAILABLE:
            command_used = "import"
            result = subprocess.run(
                ["import", "-window", "root", temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=5
            )
        
        # If import failed or not available, try scrot
        elif SCROT_AVAILABLE:
            command_used = "scrot"
            result = subprocess.run(
                ["scrot", "-z", temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=5
            )
        
        # Try xfce4-screenshooter
        elif XFCE4_SCREENSHOOTER_AVAILABLE:
            command_used = "xfce4-screenshooter"
            result = subprocess.run(
                ["xfce4-screenshooter", "-f", "-s", temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=5
            )
        
        # Try gnome-screenshot
        elif GNOME_SCREENSHOT_AVAILABLE:
            command_used = "gnome-screenshot"
            result = subprocess.run(
                ["gnome-screenshot", "-f", temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=5
            )
        else:
            return False, "No screenshot tools available"
        
        # Check if the command was successful
        if result.returncode != 0:
            error_msg = f"{command_used} failed: {result.stderr.decode('utf-8')}"
            logger.error(error_msg)
            return False, error_msg
        
        # Check if the file was created
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            error_msg = f"Screenshot file was not created with {command_used}"
            logger.error(error_msg)
            return False, error_msg
        
        # Read the file
        with open(temp_path, "rb") as f:
            image_data = f.read()
        
        # Save to permanent location if enabled
        if SAVE_SCREENSHOTS:
            timestamp = time.strftime("%Y%m%d-%H%M%S-%f")
            save_path = SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
            
            with open(save_path, "wb") as f:
                f.write(image_data)
            
            logger.info(f"Screenshot saved to {save_path}")
        
        logger.info(f"Screenshot captured successfully with {command_used}")
        return True, image_data
    
    except subprocess.TimeoutExpired:
        error_msg = f"{command_used} timed out"
        logger.error(error_msg)
        return False, error_msg
    
    except Exception as e:
        error_msg = f"Error capturing screenshot: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
    
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {temp_path}: {e}")

# --- Helper Functions ---
def add_cors_headers(response):
    """Add CORS headers to the response"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key'
    return response

# --- API Endpoints ---
@app.route("/api/capture", methods=["GET", "OPTIONS"])
def capture_endpoint():
    """Endpoint for capturing screenshots"""
    # Handle preflight CORS requests
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        return add_cors_headers(response)
    
    # Capture screenshot
    success, result = capture_screenshot()
    
    if success:
        # Return the image data
        response = Response(result, mimetype="image/png")
        return add_cors_headers(response)
    else:
        # Return the error message
        error_response = jsonify({"error": result})
        error_response.status_code = 500
        return add_cors_headers(error_response)

@app.route("/api/health", methods=["GET", "OPTIONS"])
def health_check():
    """Health check endpoint"""
    # Handle preflight CORS requests
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        return add_cors_headers(response)
    
    # Check available tools
    tools = []
    if IMPORT_AVAILABLE:
        tools.append("import")
    if SCROT_AVAILABLE:
        tools.append("scrot")
    if XFCE4_SCREENSHOOTER_AVAILABLE:
        tools.append("xfce4-screenshooter")
    if GNOME_SCREENSHOT_AVAILABLE:
        tools.append("gnome-screenshot")
    
    # Return server status
    response = jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "display": DISPLAY,
        "available_tools": tools,
        "save_screenshots": SAVE_SCREENSHOTS
    })
    
    return add_cors_headers(response)

@app.route("/", methods=["GET"])
def index():
    """Root endpoint"""
    return """
    <html>
        <head>
            <title>Screenshot Server</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .endpoint { background: #f4f4f4; padding: 10px; margin: 10px 0; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>Screenshot Capture Server</h1>
            <p>This server captures screenshots from the X11 display and serves them via API.</p>
            
            <h2>Available endpoints:</h2>
            
            <div class="endpoint">
                <h3>/api/capture</h3>
                <p>Captures a screenshot and returns it as a PNG image</p>
            </div>
            
            <div class="endpoint">
                <h3>/api/health</h3>
                <p>Returns server health status information</p>
            </div>
        </body>
    </html>
    """

# --- Main Entry Point ---
if __name__ == "__main__":
    logger.info(f"Starting screenshot server on {APP_HOST}:{APP_PORT}")
    logger.info(f"Using display: {DISPLAY}")
    logger.info(f"Debug mode: {APP_DEBUG}")
    app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG)
