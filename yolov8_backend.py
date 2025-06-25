#!/usr/bin/env python3
"""
YOLOv8 Label Studio ML Backend - Main Entry Point
=================================================
This is the main script that sets up and runs the backend server.
All functionality is separated into modules for better organization.

How to use:
1. pip install flask ultralytics torch pillow waitress
2. Place YOLOv8 weights at C:/PokerData/best.pt
3. python yolov8_backend.py
4. In Label Studio, set ML backend URL to: http://host.docker.internal:9090 (Windows/Mac)
"""

import os
import sys
import logging
from pathlib import Path
import threading
import traceback

# Import modules
from modules.config import initialize_directories, REPORTS_DIR, BASE_DIR, WEIGHTS_PATH, TRAIN_DIR, VAL_DIR
from modules.utils import print_debug_banner, diagnose_network
from modules.api import create_app
from modules.data_handler import find_annotated_pairs

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("yolo_backend_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("yolov8_backend")

def verify_training_paths():
    """Verify and log training directory information for architectural correctness"""
    # Create absolute paths for proper architecture
    train_path = os.path.abspath(TRAIN_DIR)
    val_path = os.path.abspath(VAL_DIR)
    
    # Check if directories exist
    if not os.path.exists(train_path):
        log.error(f"Training directory not found at: {train_path}")
        return False
    if not os.path.exists(val_path):
        log.error(f"Validation directory not found at: {val_path}")
        return False
        
    # Check if directories contain images
    train_images = len([f for f in os.listdir(train_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    val_images = len([f for f in os.listdir(val_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    log.info(f"Training path: {train_path} contains {train_images} images")
    log.info(f"Validation path: {val_path} contains {val_images} images")
    
    return train_images > 0 and val_images > 0

def main():
    """Main entry point for the backend server"""
    print_debug_banner("STARTING SERVER")
    
    # Initialize directories
    initialize_directories()
    
    # Log system info
    log.info(f"Python version: {sys.version}")
    log.info(f"Operating system: {os.name} - {sys.platform}")
    log.info(f"Current working directory: {os.getcwd()}")
    
    # Check if model exists
    if os.path.exists(WEIGHTS_PATH):
        log.info(f"Found YOLOv8 weights at: {WEIGHTS_PATH}")
    else:
        log.error(f"WEIGHTS NOT FOUND at {WEIGHTS_PATH}!")
        
    # Check for existing YOLO format annotations
    annotation_pairs = find_annotated_pairs()
    log.info(f"Found {len(annotation_pairs)} images with corresponding YOLO annotations")
    
    # Verify training paths for proper architecture
    verify_training_paths()
    
    # Run network diagnostics in a thread so server starts quickly
    threading.Thread(target=diagnose_network).start()
    
    # Create and run the Flask app
    app = create_app()
    
    # Print testing instructions
    print("\nTest these urls from your host machine:")
    print("  http://localhost:9090/health - Standard health check")

    try:
        from waitress import serve
        log.info("Starting server with Waitress on 0.0.0.0:9090")
        serve(app, host="0.0.0.0", port=9090, threads=10)
    except ImportError:
        log.warning("Waitress not installed, using Flask dev server")
        app.run(host="0.0.0.0", port=9090, debug=True)
    except Exception as e:
        log.error(f"Error starting server: {str(e)}")
        log.error(traceback.format_exc())

if __name__ == "__main__":
    main()