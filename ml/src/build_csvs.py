# build_csvs.py

import csv
import random
from pathlib import Path

# ==== SET THIS TO YOUR REAL LOCATION ====
ROOT_DIR = Path("/Users/juan/Desktop/Acne Project/acne-classification-app/ml/data/raw/SkinDisease")
PROCESSED_DIR = Path("/Users/juan/Desktop/Acne Project/acne-classification-app/ml/data/processed")
# =========================================

VAL_FRACTION = 0.1
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def is_image_file(path: Path):
    return path.suffix.lower() in IMAGE_EXTS

def collect_images(split_dir):
    samples = []
    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            label_name = class_dir.name
            for img_path in class_dir.rglob("*"):
                if img_path.is_file() and is_image_file(img_path):
                    samples.append((str(img_path), label_name))
    return samples

def build_label_map(labels):
    unique = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique)}

def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "split", "label_idx"])
        writer.writeheader()
        writer.writerows(rows)

def main():
    train_dir = ROOT_DIR / "Train"
    test_dir = ROOT_DIR / "Test"

    train_data = collect_images(train_dir)
    test_data = collect_images(test_dir)

    all_labels = [lbl for _, lbl in train_data + test_data]
    label_map = build_label_map(all_labels)

    # Split train → train + val manually
    from collections import defaultdict
    grouped = defaultdict(list)
    for fp, lbl in train_data:
        grouped[lbl].append(fp)

    train_rows = []
    val_rows = []

    for lbl, files in grouped.items():
        random.shuffle(files)
        n_val = max(1, int(len(files) * VAL_FRACTION))
        val_files = files[:n_val]
        train_files = files[n_val:]
        lbl_idx = label_map[lbl]

        for fp in train_files:
            train_rows.append({"filepath": fp, "label": lbl, "split": "train", "label_idx": lbl_idx})
        for fp in val_files:
            val_rows.append({"filepath": fp, "label": lbl, "split": "val", "label_idx": lbl_idx})

    test_rows = [
        {"filepath": fp, "label": lbl, "split": "test", "label_idx": label_map[lbl]}
        for fp, lbl in test_data
    ]

    write_csv(train_rows, PROCESSED_DIR / "processed_train.csv")
    write_csv(val_rows, PROCESSED_DIR / "processed_val.csv")
    write_csv(test_rows, PROCESSED_DIR / "processed_test.csv")

    print("CSV generation complete!")
    print("Train:", len(train_rows), "Val:", len(val_rows), "Test:", len(test_rows))

if __name__ == "__main__":
    main()
