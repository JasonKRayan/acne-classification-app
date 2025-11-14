"""
Preprocessing utilities for the skin disease classification project.

Responsibilities:
- Resize shortest side to SHORT_SIDE while preserving aspect ratio.
- Center crop to IMG_SIZE (default 224x224).
- Scale pixel values to [0, 1].
- Build tf.data.Dataset objects from a DataFrame with columns:
    - 'filepath': string path to image on disk
    - 'label_idx': integer class index
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
import pandas as pd

# ---------------------------------------------------------------------
# Global config (can be overridden by caller if needed)
# ---------------------------------------------------------------------

IMG_SIZE: Tuple[int, int] = (224, 224)   # (height, width)
SHORT_SIDE: int = 256
BATCH_SIZE: int = 32
AUTOTUNE = tf.data.AUTOTUNE
SEED: int = 42


# ---------------------------------------------------------------------
# Core image preprocessing functions
# ---------------------------------------------------------------------

def resize_shortest_side(
    image: tf.Tensor,
    short_side: int = SHORT_SIDE
) -> tf.Tensor:
    """
    Resize so that the shortest side of the image equals `short_side`,
    while preserving aspect ratio.

    Args:
        image: Tensor of shape (H, W, 3), float or uint8.
        short_side: Desired size of the shortest side in pixels.

    Returns:
        Resized image tensor with shape (H', W', 3),
        where min(H', W') == short_side.
    """
    shape = tf.cast(tf.shape(image)[:2], tf.float32)  # (H, W)
    height, width = shape[0], shape[1]
    scale = short_side / tf.minimum(height, width)
    new_height = tf.cast(height * scale, tf.int32)
    new_width = tf.cast(width * scale, tf.int32)
    image = tf.image.resize(image, [new_height, new_width])
    return image


def center_crop(
    image: tf.Tensor,
    target_height: int = IMG_SIZE[0],
    target_width: int = IMG_SIZE[1]
) -> tf.Tensor:
    """
    Center-crop the image to (target_height, target_width).

    Assumes image height and width are >= target dimensions.

    Args:
        image: Tensor of shape (H, W, 3).
        target_height: Height of the crop.
        target_width: Width of the crop.

    Returns:
        Cropped image tensor of shape (target_height, target_width, 3).
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    offset_height = tf.cast((height - target_height) / 2, tf.int32)
    offset_width = tf.cast((width - target_width) / 2, tf.int32)

    image = tf.image.crop_to_bounding_box(
        image,
        offset_height,
        offset_width,
        target_height,
        target_width,
    )
    return image


def load_and_preprocess_image(path: tf.Tensor) -> tf.Tensor:
    """
    Load an image from disk, decode as RGB, convert to float in [0, 1],
    resize shortest side, and center crop to IMG_SIZE.

    Args:
        path: Scalar tf.string tensor containing the file path.

    Returns:
        Tensor of shape (IMG_SIZE[0], IMG_SIZE[1], 3), dtype float32,
        with values in [0, 1].
    """
    # Read the raw bytes
    image_bytes = tf.io.read_file(path)

    # Decode image (supports JPEG/PNG/etc.), force 3 channels (RGB)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    # Convert to float32 in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize shortest side then center crop
    image = resize_shortest_side(image, short_side=SHORT_SIDE)
    image = center_crop(image, target_height=IMG_SIZE[0], target_width=IMG_SIZE[1])

    # Ensure static shape for downstream models
    image.set_shape((*IMG_SIZE, 3))
    return image


def preprocess_example(
    path: tf.Tensor,
    label_idx: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Preprocess a single (path, label_idx) pair into (image_tensor, label).

    Args:
        path: Scalar tf.string tensor (image path).
        label_idx: Scalar int tensor (integer class index).

    Returns:
        (image, label) where:
            image: Tensor (IMG_SIZE[0], IMG_SIZE[1], 3), float32 in [0, 1].
            label: Scalar int32 tensor.
    """
    image = load_and_preprocess_image(path)
    label = tf.cast(label_idx, tf.int32)
    return image, label


# ---------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------

def df_to_dataset(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    seed: int = SEED,
) -> tf.data.Dataset:
    """
    Convert a DataFrame with 'filepath' and 'label_idx' columns
    into a tf.data.Dataset of (image, label) batches.

    Args:
        df: Pandas DataFrame with at least 'filepath' and 'label_idx' columns.
        batch_size: Batch size for the dataset.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed for shuffling.

    Returns:
        A tf.data.Dataset yielding batches of (images, labels), where:
            images: (batch_size, IMG_SIZE[0], IMG_SIZE[1], 3)
            labels: (batch_size,)
    """
    if "filepath" not in df.columns or "label_idx" not in df.columns:
        raise ValueError("DataFrame must contain 'filepath' and 'label_idx' columns.")

    paths = df["filepath"].astype(str).values
    labels = df["label_idx"].astype(int).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(preprocess_example, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds
