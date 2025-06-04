# evaluate.py
import os
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, f1_score

# Load best trained model
model = YOLO("runs/classify/train/weights/best.pt")

y_true, y_pred = [], []
for cls in ["eve_teasing", "normal"]:
    cls_index = 0 if cls == "eve_teasing" else 1
    folder = f"dataset/test/{cls}"
    for img in os.listdir(folder):
        if not img.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        res = model.predict(f"{folder}/{img}", save=False)[0]  # perform classification
        # res.probs is a tensor of class probabilities
        pred_index = int(res.probs.top1)
        confidence = float(res.probs.top1conf)
        y_true.append(cls_index)
        y_pred.append(pred_index)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="binary")
print(f"Test Accuracy: {acc:.3f}, F1 Score: {f1:.3f}")
