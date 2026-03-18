import os
import cv2
import shutil

def prepare_staging_dataset():
    base_dir = "dataset"
    staging_dir = "dataset_staged"
    
    # Clean output directories (except the folders themselves)
    for split in ["train", "val"]:
        for i in range(1, 5):
            stage_dir = os.path.join(staging_dir, split, f"stage{i}")
            if os.path.exists(stage_dir):
                shutil.rmtree(stage_dir)
            os.makedirs(stage_dir, exist_ok=True)
    
    for split in ["train", "val"]:
        images_dir = os.path.join(base_dir, split, "images")
        labels_dir = os.path.join(base_dir, split, "labels")
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Skipping {split} - missing directory")
            continue
            
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue
                
            img_name = label_file.replace(".txt", ".jpg")
            img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(img_path):
                img_name = label_file.replace(".txt", ".png")
                img_path = os.path.join(images_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"Missing image: {img_path}")
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w, _ = img.shape
            
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for idx, line in enumerate(f):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                        
                    cls = int(parts[0])
                    # Classes in datasetNew are 0,1,2,3 for stage 1-4
                    stage_id = cls + 1
                    if stage_id < 1 or stage_id > 4:
                        continue
                        
                    cx, cy, cw, ch = map(float, parts[1:5])
                    
                    x1 = int((cx - cw/2) * w)
                    y1 = int((cy - ch/2) * h)
                    x2 = int((cx + cw/2) * w)
                    y2 = int((cy + ch/2) * h)
                    
                    # Ensure positive indices
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                        
                    target_file = f"{os.path.basename(img_path)}_{idx}.jpg"
                    target_path = os.path.join(staging_dir, split, f"stage{stage_id}", target_file)
                    
                    cv2.imwrite(target_path, crop)
                    
if __name__ == "__main__":
    prepare_staging_dataset()
