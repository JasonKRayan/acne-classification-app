"""
build_csvs.py - Generate train/val/test CSV files from image directories
"""
import csv
import random
from pathlib import Path
from collections import defaultdict
import logging

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, VAL_FRACTION,
    IMAGE_EXTENSIONS, RANDOM_SEED
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(RANDOM_SEED)


def is_image_file(path: Path) -> bool:
    """Check if file has valid image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def collect_images(split_dir: Path) -> list:
    """
    Collect all images from a directory structure.
    
    Expected structure:
        split_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
    
    Returns:
        List of (filepath, label_name) tuples
    """
    if not split_dir.exists():
        logger.error(f"Directory not found: {split_dir}")
        return []
    
    samples = []
    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    
    if not class_dirs:
        logger.warning(f"No class directories found in {split_dir}")
        return []
    
    for class_dir in sorted(class_dirs):
        label_name = class_dir.name
        images = [
            img for img in class_dir.rglob("*") 
            if img.is_file() and is_image_file(img)
        ]
        
        if not images:
            logger.warning(f"No images found in {class_dir}")
            continue
        
        for img_path in images:
            samples.append((str(img_path), label_name))
        
        logger.info(f"Found {len(images)} images for class '{label_name}'")
    
    return samples


def build_label_map(labels: list) -> dict:
    """Create mapping from label names to indices."""
    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    logger.info(f"Created label map with {len(label_map)} classes:")
    for label, idx in label_map.items():
        logger.info(f"  {idx}: {label}")
    
    return label_map


def write_csv(rows: list, path: Path) -> None:
    """Write dataset rows to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["filepath", "label", "split", "label_idx"]
        )
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"Wrote {len(rows)} samples to {path.name}")


def stratified_split(grouped_data: dict, val_fraction: float, min_val_per_class: int = 1):
    """
    Perform stratified train/val split ensuring each class has representation.
    
    Args:
        grouped_data: Dict mapping label -> list of filepaths
        val_fraction: Fraction of data to use for validation
        min_val_per_class: Minimum validation samples per class
    
    Returns:
        (train_files, val_files) as dict mapping label -> list of filepaths
    """
    train_grouped = defaultdict(list)
    val_grouped = defaultdict(list)
    
    for label, files in grouped_data.items():
        random.shuffle(files)
        
        # Calculate validation size (at least min_val_per_class)
        n_val = max(min_val_per_class, int(len(files) * val_fraction))
        
        # Ensure we don't take more validation samples than we have
        n_val = min(n_val, len(files) - 1) if len(files) > 1 else len(files)
        
        val_files = files[:n_val]
        train_files = files[n_val:]
        
        val_grouped[label] = val_files
        train_grouped[label] = train_files
        
        logger.info(
            f"Class '{label}': {len(train_files)} train, "
            f"{len(val_files)} val (total: {len(files)})"
        )
    
    return train_grouped, val_grouped


def main():
    """Generate processed CSV files from raw image directories."""
    logger.info("=" * 60)
    logger.info("Starting CSV generation")
    logger.info("=" * 60)
    
    # Define directories
    train_dir = RAW_DATA_DIR / "Train"
    test_dir = RAW_DATA_DIR / "Test"
    
    # Collect images
    logger.info("\nCollecting training images...")
    train_data = collect_images(train_dir)
    
    logger.info("\nCollecting test images...")
    test_data = collect_images(test_dir)
    
    if not train_data:
        raise ValueError(f"No training images found in {train_dir}")
    
    if not test_data:
        logger.warning(f"No test images found in {test_dir}")
    
    # Build label map from all data
    all_labels = [lbl for _, lbl in train_data + test_data]
    label_map = build_label_map(all_labels)
    
    # Group training data by label for stratified split
    logger.info("\nPerforming stratified train/val split...")
    grouped_train = defaultdict(list)
    for filepath, label in train_data:
        grouped_train[label].append(filepath)
    
    train_grouped, val_grouped = stratified_split(
        grouped_train, 
        VAL_FRACTION
    )
    
    # Convert to row format
    train_rows = []
    val_rows = []
    
    for label, files in train_grouped.items():
        label_idx = label_map[label]
        for fp in files:
            train_rows.append({
                "filepath": fp,
                "label": label,
                "split": "train",
                "label_idx": label_idx
            })
    
    for label, files in val_grouped.items():
        label_idx = label_map[label]
        for fp in files:
            val_rows.append({
                "filepath": fp,
                "label": label,
                "split": "val",
                "label_idx": label_idx
            })
    
    test_rows = [
        {
            "filepath": fp,
            "label": label,
            "split": "test",
            "label_idx": label_map[label]
        }
        for fp, label in test_data
    ]
    
    # Write CSVs
    logger.info("\nWriting CSV files...")
    write_csv(train_rows, PROCESSED_DATA_DIR / "processed_train.csv")
    write_csv(val_rows, PROCESSED_DATA_DIR / "processed_val.csv")
    write_csv(test_rows, PROCESSED_DATA_DIR / "processed_test.csv")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CSV Generation Complete!")
    logger.info("=" * 60)
    logger.info(f"Train samples: {len(train_rows)}")
    logger.info(f"Val samples:   {len(val_rows)}")
    logger.info(f"Test samples:  {len(test_rows)}")
    logger.info(f"Total classes: {len(label_map)}")
    logger.info(f"Output directory: {PROCESSED_DATA_DIR}")
    
    # Class distribution
    logger.info("\nClass distribution:")
    for split_name, rows in [("Train", train_rows), ("Val", val_rows), ("Test", test_rows)]:
        if rows:
            class_counts = defaultdict(int)
            for row in rows:
                class_counts[row['label']] += 1
            logger.info(f"\n{split_name}:")
            for label in sorted(class_counts.keys()):
                logger.info(f"  {label}: {class_counts[label]}")


if __name__ == "__main__":
    main()