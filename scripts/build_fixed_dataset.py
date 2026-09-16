import os
import glob
import shutil
import random
import yaml

TARGET_CLASSES = [
    'smartphone', 'laptop', 'bottle', 'keys', 'watch', 'remote', 
    'charging_cable', 'book', 'glasses', 'cup', 'coffee_mug', 'pen', 
    'earbuds', 'earbuds_case', 'headphone', 'shoe', 'keyboard', 
    'umbrella', 'plant', 'container'
]
CLASS_TO_ID = {name: i for i, name in enumerate(TARGET_CLASSES)}

# We must use the pristine raw datasets because the current merged dataset has corrupted class IDs.
DATASETS = [
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-001/dataset", {0: 'plant', 1: 'cup', 2: 'container', 3: 'earbuds_case', 4: 'coffee_mug'}),
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-004/raw/ds1", {0: 'earbuds'}),
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-004/raw/ds2", {0: 'earbuds', 1: 'headphone'}),
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-004/raw/ds3", {0: 'earbuds'}),
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-005/everyday object detection.v1i.yolov8", 
        {5: 'charging_cable', 6: 'laptop', 8: 'keyboard', 10: 'glasses', 12: 'glasses', 13: 'umbrella', 15: 'book', 16: 'cup', 17: 'earbuds', 18: 'headphone', 19: 'keys', 20: 'pen', 21: 'shoe', 22: 'shoe', 23: 'watch', 11: 'book'}),
    ("/Users/bcs-mac001/learning/AI Research Learning Hub/robot-vision-yolo/data/sessions/EXP-P-006/dataset", 
        {6: 'laptop', 7: 'smartphone', 12: 'watch', 16: 'cup', 17: 'bottle', 18: 'remote', 19: 'keys'})
]

OUTPUT_DIR = "/Users/bcs-mac001/learning/AI Research Learning Hub/indoor_object_detection/data/dataset_v4"

def process_datasets():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"))
    os.makedirs(os.path.join(OUTPUT_DIR, "images", "val"))
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", "train"))
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", "val"))
    
    all_images = []
    
    for ds_path, local_mapping in DATASETS:
        # Convert local mapping to global IDs
        local_to_global = {}
        for local_id, target_name in local_mapping.items():
            if target_name in CLASS_TO_ID:
                local_to_global[local_id] = CLASS_TO_ID[target_name]
                
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
            
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    
    def save_split(data_split, split_name):
        for item in data_split:
            shutil.copy(item['img_path'], os.path.join(OUTPUT_DIR, 'images', split_name, item['base_name']))
            with open(os.path.join(OUTPUT_DIR, 'labels', split_name, item['base_name'].replace('.jpg', '.txt')), 'w') as f:
                f.writelines(item['new_labels'])
                
    save_split(all_images[:split_idx], 'train')
    save_split(all_images[split_idx:], 'val')
    
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        f.write("path: /kaggle/working/dataset_v4\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(TARGET_CLASSES)}\n")
        f.write(f"names: {TARGET_CLASSES}\n")
        
if __name__ == "__main__":
    process_datasets()
