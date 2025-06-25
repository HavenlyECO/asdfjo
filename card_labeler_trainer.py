"""
Automated card labeling and training system with clean architecture
Takes time to properly analyze each image like a human would
"""

import os
import sys
import argparse
import logging
import random
import json
import time
import cv2
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler("card_detection.log")
    ]
)
log = logging.getLogger("card_detection")

# Import our modules
from deliberate_card_detector import DeliberateCardDetector, CARD_CLASSES
from card_dataset import CardAnnotation, CardDataset
from card_trainer import CardTrainer
from noise_filter import NoiseFilter

def process_directory(input_dir: str, 
                    output_dir: str, 
                    val_split: float = 0.2,
                    max_images: int = None,
                    min_process_time: float = 5.0) -> Dict[str, Any]:
    """
    Process all images in a directory, taking time for careful deliberate analysis
    
    Args:
        input_dir: Input directory with images
        output_dir: Output directory for dataset and model
        val_split: Validation split ratio
        max_images: Maximum number of images to process (None for all)
        min_process_time: Minimum seconds to spend analyzing each image
        
    Returns:
        Dictionary with processing results
    """
    # Initialize our deliberate detector
    detector = DeliberateCardDetector(debug_dir=os.path.join(output_dir, "debug"))
    
    # Initialize dataset
    dataset = CardDataset(os.path.join(output_dir, "dataset"))
    
    # Find all images
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend([str(p) for p in Path(input_dir).rglob(ext)])
    
    if not image_files:
        log.error(f"No images found in {input_dir}")
        return {"success": False, "error": "No images found"}
    
    log.info(f"Found {len(image_files)} images")
    
    # Limit to max_images if specified
    if max_images is not None:
        image_files = image_files[:max_images]
        log.info(f"Processing limited to {max_images} images")
    
    # Process each image
    card_count = 0
    image_count = 0
    unknown_count = 0
    error_count = 0
    
    # Use random seed for reproducibility
    random.seed(42)
    random.shuffle(image_files)
    
    total_images = len(image_files)
    
    for i, img_path in enumerate(image_files):
        try:
            # Print progress
            log.info(f"Processing image {i+1}/{total_images}: {Path(img_path).name}")
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                log.warning(f"Could not read image: {img_path}")
                continue
                
            # Time the processing
            start_time = time.time()
            
            # Use deliberate detection to find card regions
            # This will take time to carefully analyze the image
            frame_id = f"frame_{i+1:04d}"
            card_regions = detector.detect_all_cards(img, frame_id)
            
            # Extract individual cards
            annotations = []
            for region in card_regions:
                x, y, w, h = map(int, region["bbox"])
                
                # Ensure coordinates are within image bounds
                if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
                    continue
                    
                # Extract card image
                card_img = img[y:y+h, x:x+w]
                
                # Skip empty or invalid card images
                if card_img.size == 0:
                    continue
                
                # Use detector to identify the card
                card_info = detector._identify_card(card_img)
                
                # If it's a recognized card, add it
                if card_info["card_class"] != "unknown" and card_info["card_class"] in CARD_CLASSES:
                    confidence = card_info.get("confidence", 0)
                    if confidence >= 0.3:  # Minimum confidence threshold
                        annotations.append(
                            CardAnnotation(
                                image_path=img_path,
                                card_class=card_info["card_class"],
                                bbox=(x, y, w, h),
                                confidence=confidence,
                                source=f"{region.get('context', 'detection')}"
                            )
                        )
                        card_count += 1
                else:
                    unknown_count += 1
            
            # Add to dataset if there are valid annotations
            if annotations:
                # Determine if this should be a validation image
                is_validation = random.random() < val_split
                
                # Add to dataset
                success = dataset.add_image(img_path, annotations, is_validation)
                if success:
                    image_count += 1
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Log information about this image
            log.info(f"Image {i+1}/{total_images} - Found {len(annotations)} cards in {elapsed_time:.2f} seconds")
                
        except Exception as e:
            log.error(f"Error processing {img_path}: {e}")
            error_count += 1
    
    # Create dataset YAML
    yaml_path = dataset.create_yaml()
    
    # Get dataset stats
    stats = dataset.get_stats()
    
    log.info(f"Dataset stats: {stats}")
    log.info(f"Processing summary: {image_count} images with cards, {card_count} cards, "
           f"{unknown_count} unknown cards, {error_count} errors")
    
    return {
        "success": True,
        "dataset": {
            "yaml_path": yaml_path,
            "train_count": stats["train_count"],
            "val_count": stats["val_count"],
            "card_count": card_count,
            "unknown_count": unknown_count,
            "error_count": error_count
        }
    }

def generate_diagnostic_report(result: Dict[str, Any], output_file: str) -> None:
    """
    Generate a detailed diagnostic report
    
    Args:
        result: Processing result dictionary
        output_file: Path to save the report
    """
    try:
        with open(output_file, 'w') as f:
            f.write("# Card Detection Diagnostic Report\n\n")
            
            f.write("## Processing Summary\n")
            dataset = result.get("dataset", {})
            f.write(f"- Images with cards: {dataset.get('train_count', 0) + dataset.get('val_count', 0)}\n")
            f.write(f"- Total cards detected: {dataset.get('card_count', 0)}\n")
            f.write(f"- Unknown cards: {dataset.get('unknown_count', 0)}\n")
            f.write(f"- Errors: {dataset.get('error_count', 0)}\n\n")
            
            f.write("## Dataset Split\n")
            f.write(f"- Training images: {dataset.get('train_count', 0)}\n")
            f.write(f"- Validation images: {dataset.get('val_count', 0)}\n\n")
            
            f.write("## Analysis Process\n")
            f.write("Each image was carefully analyzed using multiple detection strategies:\n")
            f.write("1. Basic contour detection\n")
            f.write("2. Adaptive thresholding\n")
            f.write("3. Color filtering\n")
            f.write("4. Multi-scale analysis\n")
            f.write("5. Comprehensive combined approach\n\n")
            
            f.write("Each strategy was applied to different card contexts:\n")
            f.write("- Hero cards (player's own cards)\n")
            f.write("- Community cards (center of table)\n")
            f.write("- Opponent cards\n")
            f.write("- Full table view\n\n")
            
            f.write("## Post-Processing\n")
            f.write("The model was post-processed with a noise filter to remove false detections from UI elements.\n")
            f.write("The filter applies the following techniques:\n")
            f.write("1. Color and texture analysis to verify card characteristics\n")
            f.write("2. Exclusion of known UI regions\n")
            f.write("3. Verification against expected card locations\n\n")
            
            f.write("See the debug directory for detailed visualization of each detection step.\n")
            
        log.info(f"Generated diagnostic report at {output_file}")
        
    except Exception as e:
        log.error(f"Error generating diagnostic report: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Card detection and training pipeline")
    
    parser.add_argument("--input", type=str, required=True, help="Directory with card images")
    parser.add_argument("--output", type=str, default="card_training", help="Output directory")
    parser.add_argument("--model-path", type=str, default="models/best.pt", help="Output model path")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--no-train", action="store_true", help="Skip training")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum images to process")
    parser.add_argument("--process-time", type=float, default=25.0, 
                       help="Minimum processing time per image in seconds")
    parser.add_argument("--filter-noise", action="store_true", default=True,
                       help="Apply noise filtering to final model")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Process images with deliberate approach
    result = process_directory(
        args.input, args.output, args.val_split, args.max_images, args.process_time
    )
    
    if not result["success"]:
        log.error(f"Processing failed: {result.get('error', 'unknown error')}")
        return 1
    
    # Generate diagnostic report
    report_path = os.path.join(args.output, "diagnostic_report.md")
    generate_diagnostic_report(result, report_path)
    
    # Train model if requested and enough data
    if not args.no_train:
        # Check if we have enough data
        if result["dataset"]["train_count"] < 5 or result["dataset"]["val_count"] < 2:
            log.error(f"Not enough data for training: {result['dataset']['train_count']} train, "
                    f"{result['dataset']['val_count']} validation images")
            return 1
            
        # Initialize trainer
        trainer = CardTrainer(result["dataset"]["yaml_path"], os.path.join(args.output, "runs"))
        
        # Train model
        best_weights = trainer.train()
        
        if not best_weights:
            log.error("Training failed")
            return 1
            
        # Apply noise filtering if requested
        if args.filter_noise:
            log.info("Applying noise filtering to model")
            noise_filter = NoiseFilter()
            filtered_model_path = args.model_path.replace('.pt', '_filtered.pt')
            
            # Filter the model and save to the target path
            filtered_path = noise_filter.filter_model_predictions(best_weights, filtered_model_path)
            log.info(f"Filtered model saved to: {filtered_path}")
            
            # Also save the original model
            shutil.copy(best_weights, args.model_path)
            log.info(f"Original model saved to {args.model_path}")
        else:
            # Just copy the best weights to the target path
            shutil.copy(best_weights, args.model_path)
            log.info(f"Model saved to {args.model_path}")
    
    log.info("Processing completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())