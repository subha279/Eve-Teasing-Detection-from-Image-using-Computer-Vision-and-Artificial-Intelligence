# inference.py
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/classify/train/weights/best.pt")

# Class names must match folder names
class_names = ["eve_teasing", "normal"]


def classify_image(image_path):
    results = model.predict(image_path, save=False)[0]
    label = class_names[results.probs.top1]
    conf = float(results.probs.top1conf)
    return label, conf


# Example usage:
# img_path = "some_new_image.jpg"
# label, conf = classify_image(img_path)
# print(f"Image {img_path} classified as {label} (conf={conf:.2f})")
