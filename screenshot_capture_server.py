"""
X11 Screenshot Server with X Authority handling
Pure implementation with no fallbacks or bandage fixes
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
SAVE_SCREENSHOTS = True  # Always save screenshots
SCREENSHOT_DIR = Path(os.environ.get("SCREENSHOT_DIR", "captured_screenshots"))

# X11 display settings
DISPLAY = os.environ.get("DISPLAY", ":0")
X_USER = os.environ.get("X_USER", "root")  # User that owns the X session

# Create screenshot directory
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Screenshots will be saved to: {SCREENSHOT_DIR}")

# Initialize Flask application
app = Flask(__name__)

# --- X11 Environment Setup ---
def setup_x_environment():
    """Set up X environment variables properly"""
    # Set DISPLAY environment variable
    os.environ["DISPLAY"] = DISPLAY
    
    # Find xauth file
    xauth_file = None
    potential_paths = [
        f"/home/{X_USER}/.Xauthority",
        f"/var/run/lightdm/{X_USER}/.Xauthority",
        f"/run/user/1000/gdm/Xauthority",
        "/var/lib/gdm3/.Xauthority"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            xauth_file = path
            break
    
    # Set XAUTHORITY if found
    if xauth_file:
        os.environ["XAUTHORITY"] = xauth_file
        logger.info(f"Using X authority file: {xauth_file}")
    else:
        logger.warning("No .Xauthority file found")

# Set up X environment
setup_x_environment()

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

logger.info(f"Available screenshot tools:")
logger.info(f"  ImageMagick import: {'✓' if IMPORT_AVAILABLE else '✗'}")
logger.info(f"  scrot: {'✓' if SCROT_AVAILABLE else '✗'}")

# --- Screenshot Capture Function ---
def capture_screenshot():
    """
    Capture a screenshot as the X server user to avoid permission issues
    
    Returns:
        Tuple of (success, image_data or error_message)
    """
    # Generate a timestamp-based filename
    timestamp = time.strftime("%Y%m%d-%H%M%S-%f")
    output_path = SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
    
    # Determine which tool to use
    command_used = None
    cmd = None
    
    if IMPORT_AVAILABLE:
        command_used = "import"
        cmd = ["import", "-window", "root", str(output_path)]
    elif SCROT_AVAILABLE:
        command_used = "scrot"
        cmd = ["scrot", "-z", str(output_path)]
    else:
        return False, "No screenshot tools available"
    
    try:
        # Run as the X session owner if needed
        if os.geteuid() == 0 and X_USER != "root":
            # Run as the X user if we're root
            sudo_cmd = ["sudo", "-u", X_USER, "env", f"DISPLAY={DISPLAY}"]
            
            # Add XAUTHORITY if we found it
            if "XAUTHORITY" in os.environ:
                sudo_cmd.extend([f"XAUTHORITY={os.environ['XAUTHORITY']}"])
            
            # Add the screenshot command
            sudo_cmd.extend(cmd)
            
            # Run the command
            result = subprocess.run(
                sudo_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
        else:
            # Run directly as current user
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
                timeout=5
            )
        
        # Check if successful
        if result.returncode != 0:
            error_msg = f"{command_used} failed: {result.stderr.decode('utf-8')}"
            logger.error(error_msg)
            return False, error_msg
        
        # Check if file exists
        if not output_path.exists() or output_path.stat().st_size == 0:
            error_msg = f"Screenshot file not created or empty"
            logger.error(error_msg)
            return False, error_msg
        
        # Read file contents
        with open(output_path, "rb") as f:
            image_data = f.read()
        
        # Log success 
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

# --- API Implementation ---
def add_cors_headers(response):
    """Add CORS headers to enable cross-origin requests"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key'
    return response

@app.route("/api/capture", methods=["GET", "OPTIONS"])
def capture_endpoint():
    """Screenshot capture endpoint"""
    # Handle CORS preflight
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        return add_cors_headers(response)
    
    # Capture screenshot
    success, result = capture_screenshot()
    
    if success:
        response = Response(result, mimetype="image/png")
        return add_cors_headers(response)
    else:
        response = jsonify({"error": result})
        response.status_code = 500
        return add_cors_headers(response)

@app.route("/api/health", methods=["GET", "OPTIONS"])
def health_check():
    """Health check endpoint"""
    # Handle CORS preflight
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        return add_cors_headers(response)
    
    # Check display access
    display_status = "unknown"
    try:
        # Simple command to check display access
        result = subprocess.run(
            ["xdpyinfo"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy()
        )
        display_status = "ok" if result.returncode == 0 else "error"
    except Exception:
        display_status = "error"
    
    response = jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "display": DISPLAY,
        "display_status": display_status,
        "x_user": X_USER,
        "tools": {
            "import": IMPORT_AVAILABLE,
            "scrot": SCROT_AVAILABLE
        }
    })
    
    return add_cors_headers(response)

@app.route("/", methods=["GET"])
def index():
    """Information page"""
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
