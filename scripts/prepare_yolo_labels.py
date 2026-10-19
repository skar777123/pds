"""
Prepare YOLOv8 Dataset Structure and Labels from Extracted Road Defect Images.
Features:
- Tight, localized defect bounding boxes (clamped to realistic road dimensions, never full screen)
- Background negative sample generation (clean road/asphalt with empty labels) to suppress false positives
- Train/val stratified splits
"""

import os
import csv
import shutil
import random
import cv2
import numpy as np

CLASS_NAME_TO_ID = {
    "Disintegrated": 0,
    "Manhole": 1,
    "Water clogged": 2,
    "Raveling": 3,
    "Deformation": 4
}

def detect_tight_pothole_box(image_path):
    """
    Compute localized, realistic bounding box for road surface defect.
    Clamps box dimensions so it never covers the entire camera frame.
    Returns normalized [x_center, y_center, width, height].
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.50, 0.60, 0.35, 0.30

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Focus on lower 75% where road texture and defects reside
    roi_top = int(h * 0.25)
    gray_roi = gray[roi_top:, :]
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_roi)
    
    # Isolate dark cavity/depression
    mean_val = np.mean(enhanced)
    std_val = np.std(enhanced)
    thresh = (enhanced < (mean_val - 0.5 * std_val)).astype(np.uint8) * 255
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_box = None
    max_area = 0
    roi_area = (h - roi_top) * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0.015 * roi_area < area < 0.50 * roi_area:
            if area > max_area:
                max_area = area
                bx, by, bw, bh = cv2.boundingRect(cnt)
                best_box = (bx, by + roi_top, bw, bh)

    if best_box:
        bx, by, bw, bh = best_box
        # Normalize
        xc = (bx + bw / 2.0) / w
        yc = (by + bh / 2.0) / h
        nw = bw / w
        nh = bh / h
        
        # Enforce realistic pothole dimensions: never full screen!
        nw = min(0.55, max(0.15, nw))
        nh = min(0.48, max(0.12, nh))
        xc = min(1.0 - nw / 2.0, max(nw / 2.0, xc))
        yc = min(1.0 - nh / 2.0, max(nh / 2.0, yc))
        return round(xc, 4), round(yc, 4), round(nw, 4), round(nh, 4)
    else:
        # Default tight localized box around road center
        return 0.5000, 0.6000, 0.3500, 0.2800

def generate_background_samples(base_dir="dataset", count=15):
    """
    Extract clean asphalt patches with NO potholes to use as negative background samples.
    In YOLO, images with empty label files train the model to suppress false positives on clean road.
    """
    images_src_dir = os.path.join(base_dir, "images")
    src_images = [f for f in os.listdir(images_src_dir) if f.endswith(".jpeg") and not f.startswith("bg_")]
    if not src_images:
        return

    random.seed(42)
    bg_count = 0

    for i in range(count):
        chosen_img_name = random.choice(src_images)
        img_path = os.path.join(images_src_dir, chosen_img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        # Crop clean undamaged road surface from top 35% of road frame
        clean_crop = img[:int(h * 0.35), :]
        if clean_crop.shape[0] < 80 or clean_crop.shape[1] < 80:
            continue

        # Resize to standard aspect
        clean_crop = cv2.resize(clean_crop, (640, 360))

        # Add slight photometric variation (different times of day)
        brightness_factor = random.uniform(0.75, 1.25)
        clean_crop = np.clip(clean_crop * brightness_factor, 0, 255).astype(np.uint8)

        # Decide split: 80% train, 20% val
        split = "train" if random.random() < 0.8 else "val"
        bg_filename = f"bg_clean_asphalt_{bg_count:02d}.jpeg"
        dst_img_path = os.path.join(base_dir, "images", split, bg_filename)
        cv2.imwrite(dst_img_path, clean_crop)

        # Create EMPTY label file (critical for negative sample training in YOLO)
        label_filename = f"bg_clean_asphalt_{bg_count:02d}.txt"
        dst_label_path = os.path.join(base_dir, "labels", split, label_filename)
        with open(dst_label_path, "w") as f:
            pass  # Empty file = Background class (no objects)

        bg_count += 1

    print(f"[Dataset] Generated {bg_count} clean road negative background samples to suppress false positives.")

def prepare_yolo_dataset(base_dir="dataset", split_ratio=0.8):
    metadata_path = os.path.join(base_dir, "metadata.csv")
    images_src_dir = os.path.join(base_dir, "images")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing {metadata_path}. Run extract_dataset.py first.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    # Clear previous splits
    for split in ["train", "val"]:
        img_split = os.path.join(base_dir, "images", split)
        lbl_split = os.path.join(base_dir, "labels", split)
        if os.path.exists(img_split):
            shutil.rmtree(img_split)
        if os.path.exists(lbl_split):
            shutil.rmtree(lbl_split)
        os.makedirs(img_split, exist_ok=True)
        os.makedirs(lbl_split, exist_ok=True)

    # Stratified split by category
    by_category = {}
    for r in records:
        if not r["image_filename"]:
            continue
        cat = r["category"]
        by_category.setdefault(cat, []).append(r)

    train_records = []
    val_records = []
    random.seed(42)

    for cat, items in by_category.items():
        random.shuffle(items)
        n_train = max(1, int(len(items) * split_ratio))
        train_records.extend(items[:n_train])
        val_records.extend(items[n_train:] if len(items) > 1 else items[:1])

    print(f"[Dataset] Positive train samples: {len(train_records)} | Positive val samples: {len(val_records)}")

    for split_name, recs in [("train", train_records), ("val", val_records)]:
        for r in recs:
            img_file = r["image_filename"]
            src_img = os.path.join(images_src_dir, img_file)
            if not os.path.exists(src_img):
                continue
                
            dst_img = os.path.join(base_dir, "images", split_name, img_file)
            shutil.copy2(src_img, dst_img)

            # Compute TIGHT, realistic pothole bounding box
            xc, yc, nw, nh = detect_tight_pothole_box(src_img)
            class_id = CLASS_NAME_TO_ID.get(r["category"], 0)

            label_file = os.path.splitext(img_file)[0] + ".txt"
            dst_label = os.path.join(base_dir, "labels", split_name, label_file)
            with open(dst_label, "w", encoding="utf-8") as lf:
                lf.write(f"{class_id} {xc} {yc} {nw} {nh}\n")

    # Generate background negative samples
    generate_background_samples(base_dir=base_dir, count=16)

    # Generate data.yaml
    abs_base = os.path.abspath(base_dir).replace("\\", "/")
    yaml_content = f"""# YOLOv8 Road Surface Defect Detection Config
path: {abs_base}
train: images/train
val: images/val

names:
  0: Disintegrated
  1: Manhole
  2: Water clogged
  3: Raveling
  4: Deformation
"""
    data_yaml_path = os.path.join(base_dir, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[Dataset] Generated YOLOv8 configuration at: {data_yaml_path}")
    print("[Dataset] Dataset preparation complete with tight boxes and negative backgrounds.")

if __name__ == "__main__":
    prepare_yolo_dataset()
