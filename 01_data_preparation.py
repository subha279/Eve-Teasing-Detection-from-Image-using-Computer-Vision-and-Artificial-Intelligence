# data_preparation.py
import os
import shutil
import random

random.seed(42)
data_dir = "dataset"  # Contains 'eve_teasing/' and 'normal/' subfolders
classes = ["eve_teasing", "normal"]
split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

# Create split directories
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(f"dataset/{split}/{cls}", exist_ok=True)

# Split images
for cls in classes:
    images = [
        f
        for f in os.listdir(f"{data_dir}/{cls}")
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    random.shuffle(images)
    n = len(images)
    train_end = int(split_ratios["train"] * n)
    val_end = train_end + int(split_ratios["val"] * n)
    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }
    for split, imgs in splits.items():
        for img in imgs:
            src = f"{data_dir}/{cls}/{img}"
            dst = f"dataset/{split}/{cls}/{img}"
            shutil.copy(src, dst)
