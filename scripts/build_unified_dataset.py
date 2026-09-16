import os
import glob
import shutil
import yaml
import random

TARGET_CLASSES = [
    'smartphone', 'laptop', 'bottle', 'keys', 'watch', 'remote', 
    'charging_cable', 'book', 'glasses', 'cup', 'coffee_mug', 'pen', 
    'earbuds', 'earbuds_case', 'headphone', 'shoe', 'keyboard', 
    'umbrella', 'plant', 'container'
]
CLASS_TO_ID = {name: i for i, name in enumerate(TARGET_CLASSES)}

CLASS_MAPPING = {
    # EXP-P-001 / 002 / 003
    'plant': 'plant', 'cup': 'cup', 'container': 'container',
    'earbuds_case': 'earbuds_case', 'coffee_mug': 'coffee_mug',
    # ds1, ds2, ds3
    'AirPods-Detection-Model': 'earbuds', 'earphone': 'earbuds', 'headphone': 'headphone',
    # EXP-P-005
    'Charging-cable': 'charging_cable', 'Computer1': 'laptop', 'Keyboard': 'keyboard',
    'Normal-Glasses': 'glasses', 'Safety-Glasses': 'glasses', 'Umbrella': 'umbrella',
    'book': 'book', 'cups': 'cup', 'keys': 'keys', 'pen': 'pen', 'shoe': 'shoe',
    'shoes': 'shoe', 'wrist watch': 'watch', 'Notebook': 'book',
    'Bowl': None, '1': None, '2': None, '3': None, '5': None,
    'Expiration': None, 'Name': None, 'University': None
}

# Tuple of (Dataset Dir, Path to its data.yaml for mapping)
DATASETS = [
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-001/dataset", "/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-001/dataset/data.yaml"),
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-002/dataset", "/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-001/dataset/data.yaml"),
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-003/dataset", "/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-001/dataset/data.yaml"),
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-004/raw/ds1", "/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-004/raw/ds1/data.yaml"),
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-004/raw/ds2", "/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-004/raw/ds2/data.yaml"),
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-004/raw/ds3", "/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-004/raw/ds3/data.yaml"),
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-005/everyday object detection.v1i.yolov8", "/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-005/everyday object detection.v1i.yolov8/data.yaml")
]

OUTPUT_DIR = "/Users/bcs-mac001/learning/AI Research Learning Hub/indoor_object_detection/data/dataset_v3"

def process_datasets():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"))
    os.makedirs(os.path.join(OUTPUT_DIR, "images", "val"))
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", "train"))
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", "val"))
    
    all_images = []
    
    for ds_path, yaml_path in DATASETS:
        with open(yaml_path, 'r') as f:
            ds_config = yaml.safe_load(f)
            
        local_names = ds_config.get('names', [])
        
        local_to_global = {}
        for local_id, local_name in enumerate(local_names):
            mapped_name = CLASS_MAPPING.get(local_name)
            if mapped_name and mapped_name in CLASS_TO_ID:
                local_to_global[local_id] = CLASS_TO_ID[mapped_name]
                
        images = []
        for split in ['train', 'valid', 'test', 'val']:
            img_dir = os.path.join(ds_path, split, 'images')
            if not os.path.exists(img_dir):
                img_dir = os.path.join(ds_path, 'images', split)
            if os.path.exists(img_dir):
                images.extend(glob.glob(os.path.join(img_dir, '*.jpg')))
                
        for img_path in images:
            txt_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
            if not os.path.exists(txt_path):
                txt_path = img_path.replace('.jpg', '.txt')
                if not os.path.exists(txt_path):
                    continue
            
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                local_id = int(parts[0])
                if local_id in local_to_global:
                    global_id = local_to_global[local_id]
                    new_lines.append(f"{global_id} {' '.join(parts[1:])}\n")
                    
            if not new_lines:
                continue
                
            all_images.append({
                'img_path': img_path,
                'new_labels': new_lines,
                'base_name': f"{len(all_images)}_{os.path.basename(img_path)}"
            })
            
    print(f"\nTotal valid labeled images extracted: {len(all_images)}")
    
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    
    def save_split(data_split, split_name):
        print(f"Saving {len(data_split)} to {split_name}...")
        for item in data_split:
            shutil.copy(item['img_path'], os.path.join(OUTPUT_DIR, 'images', split_name, item['base_name']))
            with open(os.path.join(OUTPUT_DIR, 'labels', split_name, item['base_name'].replace('.jpg', '.txt')), 'w') as f:
                f.writelines(item['new_labels'])
                
    save_split(all_images[:split_idx], 'train')
    save_split(all_images[split_idx:], 'val')
    
    unified_yaml = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(unified_yaml, 'w') as f:
        f.write("path: /kaggle/working/dataset_v3\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(TARGET_CLASSES)}\n")
        f.write(f"names: {TARGET_CLASSES}\n")
        
    print(f"\n✅ Success! New dataset compiled at: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_datasets()
