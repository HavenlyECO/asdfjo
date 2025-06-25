"""
Card dataset processing and management with clean architecture
"""

import os
import cv2
import uuid
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Import card definitions
from deliberate_card_detector import CARD_CLASSES

# Configure module logging
log = logging.getLogger("card_dataset")

class CardAnnotation:
    """Clean architecture for card annotations"""
    
    def __init__(self, image_path: str, card_class: str, bbox: Tuple[float, float, float, float],
                confidence: float = 1.0, source: str = "detection"):
        """Initialize card annotation"""
        self.image_path = image_path
        self.card_class = card_class 
        self.bbox = bbox  # (x, y, w, h)
        self.confidence = confidence
        self.source = source
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "image_path": self.image_path,
            "card_class": self.card_class,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "source": self.source
        }
    
    def to_yolo_format(self, img_width: int, img_height: int) -> str:
        """Convert to YOLO format with validation"""
        # Get class index
        class_idx = CARD_CLASSES.index(self.card_class) if self.card_class in CARD_CLASSES else 0
        
        # Extract bbox coordinates
        x, y, w, h = self.bbox
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))
        
        # Convert bbox (x, y, w, h) to YOLO format (x_center, y_center, width, height)
        x_center = (x + w/2) / img_width
        y_center = (y + h/2) / img_height
        width = w / img_width
        height = h / img_height
        
        # Ensure all values are properly normalized (0.0-1.0)
        x_center = max(0.0, min(x_center, 0.999))
        y_center = max(0.0, min(y_center, 0.999))
        width = max(0.001, min(width, 0.999))
        height = max(0.001, min(height, 0.999))
        
        return f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


class CardDataset:
    """Card dataset manager with clean architecture"""
    
    def __init__(self, output_dir: str):
        """Initialize dataset manager"""
        self.output_dir = Path(output_dir)
        self.train_img_dir = self.output_dir / "train" / "images"
        self.train_lbl_dir = self.output_dir / "train" / "labels"
        self.val_img_dir = self.output_dir / "val" / "images"
        self.val_lbl_dir = self.output_dir / "val" / "labels"
        
        # Create directories
        self.train_img_dir.mkdir(parents=True, exist_ok=True)
        self.train_lbl_dir.mkdir(parents=True, exist_ok=True)
        self.val_img_dir.mkdir(parents=True, exist_ok=True)
        self.val_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking metrics
        self.train_count = 0
        self.val_count = 0
    
    def add_image(self, image_path: str, annotations: List[CardAnnotation], is_validation: bool = False) -> bool:
        """Add an image to the dataset with its annotations"""
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                log.warning(f"Could not read image: {image_path}")
                return False
                
            img_height, img_width = img.shape[:2]
            
            # Skip if no valid dimensions
            if img_height <= 0 or img_width <= 0:
                log.warning(f"Invalid image dimensions: {image_path}")
                return False
                
            # Create unique filename
            img_name = f"{Path(image_path).stem}_{uuid.uuid4().hex[:8]}.jpg"
            
            # Determine directories
            if is_validation:
                img_dir = self.val_img_dir
                lbl_dir = self.val_lbl_dir
                self.val_count += 1
            else:
                img_dir = self.train_img_dir
                lbl_dir = self.train_lbl_dir
                self.train_count += 1
            
            # Save image
            out_img_path = img_dir / img_name
            cv2.imwrite(str(out_img_path), img)
            
            # Create label file
            out_lbl_path = lbl_dir / f"{Path(img_name).stem}.txt"
            
            # Write YOLO format labels with validation
            valid_annotations = False
            with open(out_lbl_path, "w") as f:
                for ann in annotations:
                    # Generate YOLO format string with validation
                    yolo_line = ann.to_yolo_format(img_width, img_height)
                    
                    # Validate format before writing
                    parts = yolo_line.split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Additional validation
                            if 0 <= class_id < len(CARD_CLASSES) and \
                               0 < x_center < 1 and 0 < y_center < 1 and \
                               0 < width < 1 and 0 < height < 1:
                                # Write validated annotation
                                f.write(f"{yolo_line}\n")
                                valid_annotations = True
                        except (ValueError, IndexError):
                            continue
            
            # If no valid annotations were written, delete the files
            if not valid_annotations:
                try:
                    # Clean up files if no valid annotations
                    if out_lbl_path.exists():
                        os.remove(out_lbl_path)
                    if out_img_path.exists():
                        os.remove(out_img_path)
                    
                    # Adjust counts
                    if is_validation:
                        self.val_count -= 1
                    else:
                        self.train_count -= 1
                        
                    return False
                except Exception as e:
                    log.warning(f"Error cleaning up invalid files: {e}")
                    return False
            
            return True
            
        except Exception as e:
            log.error(f"Error adding image to dataset: {e}")
            return False
    
    def create_yaml(self) -> str:
        """Create dataset YAML file"""
        yaml_path = self.output_dir / "data.yaml"
        
        try:
            data = {
                "path": str(self.output_dir.absolute()),
                "train": "train/images",
                "val": "val/images",
                "names": {i: name for i, name in enumerate(CARD_CLASSES)}
            }
            
            with open(yaml_path, "w") as f:
                yaml.dump(data, f, sort_keys=False)
                
            return str(yaml_path)
            
        except Exception as e:
            log.error(f"Error creating dataset YAML: {e}")
            return ""
    
    def get_stats(self) -> Dict[str, int]:
        """Get dataset statistics"""
        return {
            "train_count": self.train_count,
            "val_count": self.val_count,
            "total": self.train_count + self.val_count
        }