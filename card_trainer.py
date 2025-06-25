"""
Card training module with clean architecture
Handles training of card detection models
"""

import os
import logging
import subprocess
import sys
import torch
from typing import Optional
from pathlib import Path

log = logging.getLogger("card_trainer")

class CardTrainer:
    """Card model trainer with clean architecture"""
    
    def __init__(self, yaml_path: str, output_dir: str):
        """Initialize the trainer"""
        self.yaml_path = yaml_path
        self.output_dir = output_dir
        
        # Training parameters with reasonable defaults
        self.params = {
            'epochs': 30,
            'batch': 16,
            'imgsz': 640,
            'patience': 10
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train(self) -> Optional[str]:
        """Train the model using YOLO architecture"""
        try:
            log.info(f"Starting training with {self.yaml_path}")
            
            # Add PyTorch serialization fix for weights_only issue
            self._add_torch_serialization_safeguard()
            
            # Build the training command with proper parameters
            command = [
                "yolo", "task=detect", "mode=train",
                f"data={self.yaml_path}",
                f"epochs={self.params['epochs']}",
                f"batch={self.params['batch']}",
                f"imgsz={self.params['imgsz']}",
                f"patience={self.params['patience']}",
                f"project={self.output_dir}",
                "name=card_model"
            ]
            
            # Execute the training command
            log.info(f"Running training command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)
            
            # Check for errors
            if result.returncode != 0:
                log.error(f"Training failed with code {result.returncode}")
                log.error(f"Error output: {result.stderr}")
                return None
                
            # Extract best weights path
            output_dir = Path(self.output_dir) / "card_model"
            weights_path = output_dir / "weights/best.pt"
            
            if weights_path.exists():
                log.info(f"Training completed successfully. Best weights: {weights_path}")
                return str(weights_path)
            else:
                log.error(f"Training completed but no weights found at {weights_path}")
                return None
                
        except Exception as e:
            log.error(f"Training error: {e}")
            return None
            
    def _add_torch_serialization_safeguard(self):
        """Add PyTorch serialization safeguards to fix weights loading issue"""
        try:
            # Fix for the PyTorch 2.6+ weights_only issue
            import torch.serialization
            
            # Explicitly add ultralytics classes to the safe globals
            torch.serialization.add_safe_globals([
                'ultralytics.nn.tasks.DetectionModel',
                'ultralytics.nn.modules',
                'ultralytics.nn.tasks',
                'ultralytics.yolo.utils',
                'ultralytics.yolo.v8',
                'ultralytics.engine.model'
            ])
            
            # Set environment variable to use old loading behavior if needed
            os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
            
            log.info("Added PyTorch serialization safeguards")
        except Exception as e:
            log.warning(f"Could not add PyTorch serialization fixes: {e}")
            log.warning("Training might fail due to weights_only issue")