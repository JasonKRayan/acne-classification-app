from pathlib import Path
import pandas as pd
import tensorflow as tf

from augmentation import get_acne_augmentation_model


AUTOTUNE = tf.data.AUTOTUNE

# Default CSV locations (you can override these when calling get_datasets)
DEFAULT_PROCESSED_DIR = Path(
    "/Users/juan/Desktop/Acne Project/acne-classification-app/ml/data/processed"
)


def _load_csv(csv_path: Path):
    """Load CSV and return filepaths (str) and label_idxs (int)."""
    df = pd.read_csv(csv_path)
    filepaths = df["filepath"].astype(str).tolist()
    labels = df["label_idx"].astype("int32").tolist()
    return filepaths, labels


def _decode_and_preprocess_image(filepath, label, img_size=(224, 224)):
    """
    Given a filepath (tf.string), read, decode, resize and normalize image.
    """
    # Read image from disk
    img = tf.io.read_file(filepath)

    # Try both JPEG and PNG gracefully
    img = tf.image.decode_image(img, channels=3, expand_animations=False)

    # Convert to float32 [0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Resize
    img = tf.image.resize(img, img_size)

    return img, label


def _add_augmentation(dataset):
    """Apply acne-safe augmentation model to the dataset."""
    aug_model = get_acne_augmentation_model()

    def _augment(img, label):
        img = aug_model(img, training=True)
        return img, label

    return dataset.map(_augment, num_parallel_calls=AUTOTUNE)


def _build_dataset_from_csv(csv_path: Path, batch_size: int, training: bool):
    filepaths, labels = _load_csv(csv_path)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    # Decode + basic preprocess
    ds = ds.map(_decode_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    if training:
        # Shuffle first, then augment, then batch
        ds = ds.shuffle(buffer_size=len(filepaths))
        ds = _add_augmentation(ds)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_datasets(
    processed_dir: Path | str | None = None,
    batch_size: int = 32,
):
    """
    Returns (train_ds, val_ds, test_ds) built from processed_*.csv files.

    processed_dir:
        Folder containing processed_train.csv, processed_val.csv, processed_test.csv.
        If None, uses DEFAULT_PROCESSED_DIR.
    """
    if processed_dir is None:
        processed_dir = DEFAULT_PROCESSED_DIR
    else:
        processed_dir = Path(processed_dir)

    train_csv = processed_dir / "processed_train.csv"
    val_csv   = processed_dir / "processed_val.csv"
    test_csv  = processed_dir / "processed_test.csv"

    train_ds = _build_dataset_from_csv(train_csv, batch_size=batch_size, training=True)
    val_ds   = _build_dataset_from_csv(val_csv,   batch_size=batch_size, training=False)
    test_ds  = _build_dataset_from_csv(test_csv,  batch_size=batch_size, training=False)

    return train_ds, val_ds, test_ds
