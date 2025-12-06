"""
dataset.py - Unified data loading with proper error handling
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import tensorflow as tf

from config import (
    IMG_SIZE, NORMALIZATION_MODE, PROCESSED_DATA_DIR,
    SHUFFLE_BUFFER_SIZE
)
from augmentation import get_acne_augmentation_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUTOTUNE = tf.data.AUTOTUNE


class DatasetLoadError(Exception):
    """Raised when dataset loading fails"""
    pass


def _load_and_validate_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV and validate required columns."""
    if not csv_path.exists():
        raise DatasetLoadError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    required_cols = ["filepath", "label_idx"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise DatasetLoadError(
            f"CSV missing required columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Validate filepaths exist
    df['filepath'] = df['filepath'].astype(str)
    missing_files = []
    for idx, row in df.iterrows():
        if not Path(row['filepath']).exists():
            missing_files.append(row['filepath'])
            if len(missing_files) >= 5:  # Only report first 5
                break
    
    if missing_files:
        logger.warning(
            f"Found {len(missing_files)} missing files. "
            f"First few: {missing_files[:3]}"
        )
    
    logger.info(f"Loaded {len(df)} samples from {csv_path.name}")
    return df


def _decode_and_preprocess_image(
    filepath: tf.Tensor, 
    label: tf.Tensor,
    root_dir: Optional[str] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Decode and preprocess a single image.
    
    Args:
        filepath: Path to image (can be relative if root_dir is provided)
        label: Label index
        root_dir: Optional root directory to prepend to relative paths
    
    Returns image in [0, 1] range (float32).
    Model-specific normalization happens in the model itself.
    """
    # Resolve path if root_dir is provided
    if root_dir is not None:
        filepath = tf.strings.join([root_dir, filepath], separator='/')
    
    # Read image file
    img_bytes = tf.io.read_file(filepath)
    
    # Decode (handles JPEG, PNG, BMP, GIF)
    img = tf.image.decode_image(
        img_bytes, 
        channels=3, 
        expand_animations=False
    )
    
    # Ensure shape is set (required for some operations)
    img.set_shape([None, None, 3])
    
    # Resize to target size
    img = tf.image.resize(img, IMG_SIZE, method='bilinear')
    
    # Convert to float32 in [0, 1] range
    img = tf.cast(img, tf.float32) / 255.0
    
    return img, label


def _safe_decode_image(filepath: tf.Tensor, label: tf.Tensor):
    """Wrapper that catches decode errors and logs them."""
    try:
        return _decode_and_preprocess_image(filepath, label)
    except tf.errors.InvalidArgumentError:
        logger.error(f"Failed to decode image: {filepath}")
        # Return a black image as placeholder
        img = tf.zeros((*IMG_SIZE, 3), dtype=tf.float32)
        return img, label


def _build_dataset(
    csv_path: Path,
    batch_size: int,
    training: bool,
    cache: bool = False,
    root_dir: Optional[str] = None
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from CSV file.
    
    Args:
        csv_path: Path to CSV file
        batch_size: Batch size
        training: Whether this is training data (enables augmentation)
        cache: Whether to cache the dataset in memory
        root_dir: Optional root directory for relative paths in CSV
    """
    
    df = _load_and_validate_csv(csv_path)
    
    filepaths = df["filepath"].tolist()
    labels = df["label_idx"].astype('int32').tolist()
    
    # Create dataset from tensor slices
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    # Shuffle BEFORE map for better randomness (training only)
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(filepaths), SHUFFLE_BUFFER_SIZE),
            reshuffle_each_iteration=True
        )
    
    # Decode and preprocess images
    if root_dir is not None:
        ds = ds.map(
            lambda fp, lbl: _decode_and_preprocess_image(fp, lbl, root_dir),
            num_parallel_calls=AUTOTUNE,
            deterministic=not training
        )
    else:
        ds = ds.map(
            _decode_and_preprocess_image,
            num_parallel_calls=AUTOTUNE,
            deterministic=not training
        )
    
    # Apply augmentation (training only)
    if training:
        aug_model = get_acne_augmentation_model()
        ds = ds.map(
            lambda img, lbl: (aug_model(img, training=True), lbl),
            num_parallel_calls=AUTOTUNE
        )
    
    # Optional caching (useful for small datasets that fit in memory)
    if cache:
        ds = ds.cache()
    
    # Batch
    ds = ds.batch(batch_size, drop_remainder=training)
    
    # Prefetch (LAST operation for performance)
    ds = ds.prefetch(AUTOTUNE)
    
    return ds


def get_datasets(
    processed_dir: Optional[Path] = None,
    batch_size: int = 32,
    cache_train: bool = False,
    cache_val: bool = True,
    root_dir: Optional[str] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load train, validation, and test datasets.
    
    Args:
        processed_dir: Directory containing processed CSV files
        batch_size: Batch size for all datasets
        cache_train: Whether to cache training data (use for small datasets)
        cache_val: Whether to cache validation data (recommended)
        root_dir: Optional root directory if CSV contains relative paths
                 (e.g., if CSV has "Train/Acne/img1.jpg" and root_dir is "/data")
    
    Returns:
        (train_ds, val_ds, test_ds)
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR
    else:
        processed_dir = Path(processed_dir)
    
    csv_files = {
        'train': processed_dir / "processed_train.csv",
        'val': processed_dir / "processed_val.csv",
        'test': processed_dir / "processed_test.csv"
    }
    
    # Verify all files exist
    missing = [name for name, path in csv_files.items() if not path.exists()]
    if missing:
        raise DatasetLoadError(
            f"Missing CSV files: {missing}. "
            f"Run build_csvs.py first."
        )
    
    train_ds = _build_dataset(
        csv_files['train'], 
        batch_size, 
        training=True,
        cache=cache_train,
        root_dir=root_dir
    )
    
    val_ds = _build_dataset(
        csv_files['val'], 
        batch_size, 
        training=False,
        cache=cache_val,
        root_dir=root_dir
    )
    
    test_ds = _build_dataset(
        csv_files['test'], 
        batch_size, 
        training=False,
        cache=False,
        root_dir=root_dir
    )
    
    return train_ds, val_ds, test_ds


def get_dataset_info(processed_dir: Optional[Path] = None) -> dict:
    """
    Get information about the dataset without loading images.
    
    Returns:
        dict with keys: num_train, num_val, num_test, num_classes, class_names
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR
    else:
        processed_dir = Path(processed_dir)
    
    info = {}
    
    for split in ['train', 'val', 'test']:
        csv_path = processed_dir / f"processed_{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            info[f'num_{split}'] = len(df)
            
            if split == 'train':
                info['num_classes'] = df['label_idx'].nunique()
                if 'label' in df.columns:
                    info['class_names'] = sorted(df['label'].unique().tolist())
    
    return info