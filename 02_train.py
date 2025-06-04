# train.py
from ultralytics import YOLO

# Load a pretrained YOLOv8 classification model (ImageNet pretrained)
model = YOLO(
    "yolov8n-cls.pt"
)  # small pretrained model:contentReference[oaicite:9]{index=9}

# Train on our custom dataset
results = model.train(
    data="dataset",  # path to dataset root (with train/ and val/ folders)
    epochs=30,  # number of epochs (tunable)
    batch=8,  # batch size (tunable)
    imgsz=512,  # image size
    lr0=0.001,  # initial learning rate
    workers=2,  # DataLoader workers
    patience=10,  # early stopping
)
print(f"Training complete. Best model saved at: {results.best_model}")
