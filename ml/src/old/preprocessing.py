# preprocessing.py

from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------
# Global config (keep in sync with your model / EfficientNet-Lite4)
# ---------------------------------------------------------

# For EfficientNet-Lite4, many repos use 300x300 or 380x380.
# Pick ONE and use it consistently across preprocessing + model.
IMG_HEIGHT: int = 300
IMG_WIDTH: int = 300
NUM_CHANNELS: int = 3

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------
# DataFrame loading
# ---------------------------------------------------------

def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV containing at least:
      - 'filepath': str, path to the image file
      - 'label_idx': int, numeric class index (0..num_classes-1)

    You can add more columns (e.g., patient_id, split, etc.) if needed.
    """
    df = pd.read_csv(csv_path)
    if "filepath" not in df.columns:
        raise ValueError("CSV must contain a 'filepath' column.")
    if "label_idx" not in df.columns:
        raise ValueError("CSV must contain a 'label_idx' column.")
    return df


# ---------------------------------------------------------
# Core image preprocessing
# ---------------------------------------------------------

def _resolve_path(path: tf.Tensor, root_dir: Optional[str]) -> tf.Tensor:
    """
    Join a relative path with root_dir if provided.
    """
    if root_dir is None:
        return path
    # tf.strings.join expects a list of strings
    return tf.strings.join([root_dir, path], separator=os.sep)


def _load_and_preprocess_image(
    path: tf.Tensor,
    label: tf.Tensor,
    root_dir: Optional[str] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Low-level preprocessing for a single (path, label) example.

    Steps:
      1. Resolve full path (if root_dir is given).
      2. Read the image from disk.
      3. Decode JPEG/PNG (3 channels).
      4. Resize to (IMG_HEIGHT, IMG_WIDTH).
      5. Cast to float32, keeping pixel values in [0, 255].

    IMPORTANT:
      We DO NOT normalize here, so that the EfficientNet-Lite4
      preprocessing layer (in the model) can apply its canonical
      normalization (e.g., mapping to [-1, 1]).
    """
    if root_dir is not None:
        path = _resolve_path(path, root_dir)

    # 1) Read bytes
    image_bytes = tf.io.read_file(path)

    # 2) Decode image (handles JPEG/PNG)
    image = tf.io.decode_image(
        image_bytes,
        channels=NUM_CHANNELS,
        expand_animations=False,
    )

    # 3) Resize
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

    # 4) Cast to float32 [0, 255]
    image = tf.cast(image, tf.float32)

    return image, label


def _make_base_dataset(
    df: pd.DataFrame,
    root_dir: Optional[str] = None,
) -> tf.data.Dataset:
    """
    Create an unbatched tf.data.Dataset from a DataFrame.

    Yields (image_path, label_idx) pairs. Actual image decoding/resizing
    happens later in a .map() for better performance.
    """
    paths = df["filepath"].astype(str).values
    labels = df["label_idx"].astype("int32").values

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    # Dataset of (path, label)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    # Attach root_dir via a closure in the map function later
    if root_dir is not None:
        # Store root_dir as a tf.constant so it can be captured in the map fn
        root_dir_tensor = tf.constant(root_dir, dtype=tf.string)

        def _with_root(path, label):
            # We'll still resolve in the main map, but passing the dir here
            # keeps the signature (path, label) and avoids adding extra fields.
            return path, label, root_dir_tensor

        ds = ds.map(
            lambda p, y: (p, y, root_dir_tensor),
            num_parallel_calls=AUTOTUNE,
        )
    return ds


# ---------------------------------------------------------
# Public helpers to build train/val/test datasets
# ---------------------------------------------------------

def build_dataset_from_df(
    df: pd.DataFrame,
    *,
    root_dir: Optional[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    cache: bool = False,
    shuffle_buffer: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Build a batched, prefetched tf.data.Dataset from a pandas DataFrame.

    Typical usage:
        train_df = load_dataframe("processed_train.csv")
        val_df   = load_dataframe("processed_val.csv")
        test_df  = load_dataframe("processed_test.csv")

        train_ds = build_dataset_from_df(train_df, root_dir="/data/skin", batch_size=32, shuffle=True, cache=True)
        val_ds   = build_dataset_from_df(val_df,   root_dir="/data/skin", batch_size=32, shuffle=False, cache=True)
        test_ds  = build_dataset_from_df(test_df,  root_dir="/data/skin", batch_size=32, shuffle=False, cache=False)

    Best practices it follows:
      - tf.data for IO & transformation, as recommended by TF guides.
      - shuffle → map → (cache) → batch → prefetch for performance.
      - Prefetch(AUTOTUNE) to overlap input pipeline with model compute.
    """

    # Convert paths + labels into a tf.data.Dataset
    paths = df["filepath"].astype(str).values
    labels = df["label_idx"].astype("int32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Shuffle only when requested (usually train only).
    if shuffle:
        if shuffle_buffer is None:
            # Common pattern: buffer_size ~ dataset size but capped
            shuffle_buffer = min(len(df), 1000)
        ds = ds.shuffle(
            buffer_size=shuffle_buffer,
            reshuffle_each_iteration=True,
        )

    # Map: decode + resize + cast.
    # Use AUTOTUNE for num_parallel_calls for performance.
    def _process(path, label):
        return _load_and_preprocess_image(path, label, root_dir=root_dir)

    ds = ds.map(_process, num_parallel_calls=AUTOTUNE)

    # Optional: cache after the expensive map if dataset fits in RAM.
    if cache:
        ds = ds.cache()

    # Batch
    ds = ds.batch(batch_size, drop_remainder=False)

    # Prefetch as the LAST step (official best practice).
    ds = ds.prefetch(AUTOTUNE)

    return ds


def build_train_val_test_datasets(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
    *,
    root_dir: Optional[str] = None,
    batch_size: int = 32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset]]:
    """
    Convenience wrapper if you already have separate CSVs for train/val/test.

    Returns (train_ds, val_ds, test_ds).
    test_ds may be None if test_csv is not provided.
    """
    train_df = load_dataframe(train_csv)
    val_df = load_dataframe(val_csv)
    test_df = load_dataframe(test_csv) if test_csv is not None else None

    train_ds = build_dataset_from_df(
        train_df,
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle=True,
        cache=True,  # often good for smaller medical datasets
    )

    val_ds = build_dataset_from_df(
        val_df,
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle=False,
        cache=True,
    )

    test_ds = None
    if test_df is not None:
        test_ds = build_dataset_from_df(
            test_df,
            root_dir=root_dir,
            batch_size=batch_size,
            shuffle=False,
            cache=False,  # caching optional for test
        )

    return train_ds, val_ds, test_ds
