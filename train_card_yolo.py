#!/usr/bin/env python3
"""Simple CLI for training the YOLO-based card detector."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 card detector")
    parser.add_argument("--data", required=True,
                        help="dataset YAML with class names and paths")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="base model or checkpoint")
    parser.add_argument("--out", default="card_yolov8.pt",
                        help="where to save the trained weights")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="image size for training")
    parser.add_argument("--batch", type=int, default=16,
                        help="batch size")
    parser.add_argument("--device", default=None,
                        help="CUDA device or 'cpu'")
    parser.add_argument("--name", default="card_yolo",
                        help="run name inside the project directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name=args.name,
    )

    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, out_path)
    print(f"Best weights saved to {out_path}")


if __name__ == "__main__":
    main()
