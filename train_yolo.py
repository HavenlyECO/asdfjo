from ultralytics import YOLO

# Load a pre-trained model (YOLOv8n = nano version)
model = YOLO("yolov8n.pt")

# Start training
model.train(
    data="C:/PokerData/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="C:/PokerData/yolo_runs",  # where results go
    name="poker_yolo"
)
