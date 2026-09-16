import os
import glob
import shutil
import zipfile

# Define the target classes for the public showcase
TARGET_CLASSES = {
    0: "laptop",
    1: "smartphone",
    2: "bottle",
    3: "coffee_mug",
    4: "remote",
    5: "book",
    6: "chair",
    7: "potted_plant"
}

RAW_DIR = "data/raw"
DEST_DIR = "data/dataset"

def extract_and_merge():
    # Find all zips dropped in data/raw
    zips = glob.glob(f"{RAW_DIR}/*.zip")
    if not zips:
        print("⚠️ No ZIP files found in data/raw/. Please download some from Roboflow first!")
        return

    for z in zips:
        name = os.path.basename(z).replace(".zip", "")
        extract_path = os.path.join(RAW_DIR, name)
        
        print(f"📦 Unpacking {name}...")
        with zipfile.ZipFile(z, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        # We assume the user will manually map the IDs in this simple script 
        # or just use it as a base. For a robust merge, you check data.yaml.
        # To keep it simple, we will just copy the raw images and let the user know.
        print(f"✅ Extracted {name}. Please ensure class IDs match before merging labels.")
        
    # Generate the Kaggle YAML
    yaml_content = f"""path: /kaggle/working/dataset
train: images/train
val: images/val

nc: {len(TARGET_CLASSES)}
names: {list(TARGET_CLASSES.values())}
"""
    with open("indoor_objects.yaml", "w") as f:
        f.write(yaml_content)
    
    print("\n📝 Generated indoor_objects.yaml!")
    print("Next step: Map the YOLO label text files to the correct integer IDs and copy them to data/dataset/")

if __name__ == "__main__":
    extract_and_merge()
