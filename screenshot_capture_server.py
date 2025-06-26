"""
Screenshot capture server with headless implementation
Captures screenshots on the server side and provides them via API
"""

import os
import time
import logging
import io
import subprocess
import base64
from pathlib import Path
from PIL import Image
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
SAVE_SCREENSHOTS = os.environ.get("SAVE_SCREENSHOTS", "false").lower() in ("true", "1", "yes")
SCREENSHOT_DIR = Path(os.environ.get("SCREENSHOT_DIR", "captured_screenshots"))
SCREENSHOT_INTERVAL = float(os.environ.get("SCREENSHOT_INTERVAL", "0.033"))  # ~30 FPS

# X11 settings for headless capture
DISPLAY = os.environ.get("DISPLAY", ":0")
XVFB_RESOLUTION = os.environ.get("XVFB_RESOLUTION", "1920x1080x24")

# Create screenshot directory if saving is enabled
if SAVE_SCREENSHOTS:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Screenshots will be saved to: {SCREENSHOT_DIR}")
else:
    logger.info("Screenshot saving is disabled")

# Initialize Flask application
app = Flask(__name__)

# Initialize X11 display for headless operation if needed
def setup_virtual_display():
    """Set up virtual display if needed"""
    if os.environ.get("USE_XVFB", "false").lower() in ("true", "1", "yes"):
        try:
            # Check if Xvfb is already running
            result = subprocess.run(
                ["pgrep", "Xvfb"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            if result.returncode != 0:
                # Start Xvfb
                subprocess.Popen(
                    ["Xvfb", ":1", "-screen", "0", XVFB_RESOLUTION],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                os.environ["DISPLAY"] = ":1"
                logger.info("Started Xvfb virtual display")
            else:
                logger.info("Xvfb is already running")
        except Exception as e:
            logger.error(f"Failed to start Xvfb: {e}")
    else:
        # Use the specified DISPLAY
        os.environ["DISPLAY"] = DISPLAY
        logger.info(f"Using display: {DISPLAY}")

# Set up virtual display
setup_virtual_display()

# --- Screenshot Capture ---
class ScreenCapture:
    """Screen capture implementation for headless server"""
    
    def __init__(self):
        """Initialize screen capture"""
        # Import here to avoid errors if X is not available during module load
        import gi
        gi.require_version('Gdk', '3.0')
        from gi.repository import Gdk
        
        self.Gdk = Gdk
        
        # Get the display
        self.display = Gdk.Display.get_default()
        if not self.display:
            logger.error("No display found")
            raise RuntimeError("No display found")
            
        # Get the default screen
        self.screen = self.display.get_default_screen()
        if not self.screen:
            logger.error("No screen found")
            raise RuntimeError("No screen found")
            
        # Log screen information
        logger.info(f"Display: {self.display.get_name()}")
        width = self.screen.get_width()
        height = self.screen.get_height()
        logger.info(f"Screen dimensions: {width}x{height}")
        
        # Latest screenshot data
        self.latest_screenshot = None
        self.latest_timestamp = 0
        
        # Background capture thread
        self.capture_thread = None
        self.running = False
        self.shutdown = False
        
    def start_capture_thread(self):
        """Start background capture thread"""
        import threading
        
        if self.running:
            return
            
        self.running = True
        self.shutdown = False
        self.capture_thread = threading.Thread(target=self._capture_worker)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info("Started capture thread")
        
    def _capture_worker(self):
        """Background worker for continuous capture"""
        while not self.shutdown:
            try:
                # Capture the screen
                start_time = time.time()
                self.capture_screenshot()
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                delay = max(0, SCREENSHOT_INTERVAL - elapsed)
                if delay > 0:
                    time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error in capture worker: {e}")
                time.sleep(1.0)  # Error backoff
        
        self.running = False
        
    def capture_screenshot(self):
        """Capture a screenshot"""
        try:
            # Take screenshot of the root window
            root_window = self.Gdk.get_default_root_window()
            if not root_window:
                logger.error("No root window")
                return None
                
            # Get window dimensions
            width = root_window.get_width()
            height = root_window.get_height()
            
            # Create pixbuf
            pixbuf = self.Gdk.pixbuf_get_from_window(root_window, 0, 0, width, height)
            if not pixbuf:
                logger.error("Failed to create pixbuf")
                return None
                
            # Convert to PNG bytes
            png_data = None
            success, png_data = pixbuf.save_to_bufferv("png", [], [])
            
            if not success or not png_data:
                logger.error("Failed to convert to PNG")
                return None
                
            # Update the latest screenshot
            self.latest_screenshot = png_data
            self.latest_timestamp = time.time()
            
            # Save to disk if enabled
            if SAVE_SCREENSHOTS:
                timestamp = time.strftime("%Y%m%d-%H%M%S-%f")
                filename = f"capture_{timestamp}.png"
                filepath = SCREENSHOT_DIR / filename
                
                # Save directly from pixbuf
                pixbuf.savev(str(filepath), "png", [], [])
                logger.info(f"Saved screenshot to {filepath}")
            
            return png_data
            
        except Exception as e:
            logger.error(f"Screenshot capture error: {e}")
            return None
    
    def get_latest_screenshot(self):
        """Get the latest screenshot"""
        return self.latest_screenshot, self.latest_timestamp
    
    def stop(self):
        """Stop the capture thread"""
        self.shutdown = True

# Initialize screen capture with some delay to ensure display is ready
screen_capture = None

def init_screen_capture():
    """Initialize screen capture with retry"""
    global screen_capture
    
    if screen_capture:
        return True
        
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            screen_capture = ScreenCapture()
            screen_capture.start_capture_thread()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize screen capture (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(2)
    
    return False

# Try to initialize screen capture
init_success = init_screen_capture()
if not init_success:
    logger.warning("Failed to initialize screen capture, will retry on first request")

# --- Helper Functions ---
def add_cors_headers(response):
    """Add CORS headers to enable cross-origin requests"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key'
    return response

# --- API Endpoints ---
@app.route("/api/capture", methods=["GET", "OPTIONS"])
def capture():
    """Get the latest screenshot"""
    global screen_capture
    
    # Handle preflight CORS requests
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        return add_cors_headers(response)
    
    # Initialize screen capture if needed
    if not screen_capture:
        if not init_screen_capture():
            return add_cors_headers(jsonify({"error": "Failed to initialize screen capture"})), 500
    
    # Get the latest screenshot
    screenshot_data, timestamp = screen_capture.get_latest_screenshot()
    
    # If no screenshot available yet, try to capture one directly
    if not screenshot_data:
        screenshot_data = screen_capture.capture_screenshot()
        
    if not screenshot_data:
        return add_cors_headers(jsonify({"error": "No screenshot available"})), 404
    
    # Return the screenshot
    response = Response(screenshot_data, mimetype="image/png")
    return add_cors_headers(response)

@app.route("/api/health", methods=["GET", "OPTIONS"])
def health_check():
    """Server health check endpoint"""
    global screen_capture
    
    # Handle preflight CORS requests
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        return add_cors_headers(response)
    
    # Check screen capture status
    capture_status = "ok" if screen_capture else "not_initialized"
    
    if screen_capture:
        _, timestamp = screen_capture.get_latest_screenshot()
        latest_time = timestamp
    else:
        latest_time = None
        
    response = jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "service": "screenshot-server",
        "capture_status": capture_status,
        "latest_screenshot_time": latest_time,
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

# --- Cleanup on exit ---
import atexit

@atexit.register
def cleanup():
    """Clean up resources on exit"""
    global screen_capture
    
    if screen_capture:
        logger.info("Stopping screen capture thread")
        screen_capture.stop()

# --- Main Entry Point ---
if __name__ == "__main__":
    logger.info(f"Starting screenshot server on {APP_HOST}:{APP_PORT}")
    logger.info(f"Debug mode: {APP_DEBUG}")
    app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG)
