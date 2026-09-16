import os
import glob
import shutil
import random

RAW_DIR = "data/sessions/EXP-Combined/raw"
DEST_DIR = "data/dataset"

# Create directories
for split in ["train", "val"]:
    os.makedirs(os.path.join(DEST_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "labels", split), exist_ok=True)

# Get all images
images = glob.glob(os.path.join(RAW_DIR, "*.jpg"))
random.shuffle(images)

split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def copy_files(img_paths, split):
    for img_path in img_paths:
        base_name = os.path.basename(img_path)
        txt_path = img_path.replace(".jpg", ".txt")
        
        # Copy image
        shutil.copy(img_path, os.path.join(DEST_DIR, "images", split, base_name))
        
        # Copy label if exists
        if os.path.exists(txt_path):
            shutil.copy(txt_path, os.path.join(DEST_DIR, "labels", split, base_name.replace(".jpg", ".txt")))

print(f"Copying {len(train_imgs)} to train...")
copy_files(train_imgs, "train")

print(f"Copying {len(val_imgs)} to val...")
copy_files(val_imgs, "val")

print("Done splitting data.")
