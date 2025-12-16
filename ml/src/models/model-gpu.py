"""
model.py
GPU-accelerated training for acne / skin disease classification from processed CSVs.

Requirements:
- TensorFlow 2.16+ with CUDA (WSL)
- CSVs with columns:
    filepath   (Linux path, e.g. /mnt/c/...)
    label_idx  (int 0..K-1)

Key features:
- HARD GPU REQUIREMENT (fails fast if GPU not detected)
- Mixed precision (Tensor Cores)
- XLA JIT
- EfficientNet backbone
- Two-stage training (frozen → fine-tune)
- Class weights for imbalance
"""

import os
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ----------------------------
# Config imports
# ----------------------------
from config import (
    PROCESSED_DATA_DIR,
    BEST_MODEL_PATH,
    TENSORBOARD_LOG_DIR,
    CHECKPOINT_DIR,
    IMG_SIZE,
    IMG_CHANNELS,
    BATCH_SIZE,
    EPOCHS,
)

MODEL_NAME = getattr(__import__("config"), "MODEL_NAME", "efficientnet_v2s")
SEED = getattr(__import__("config"), "SEED", 1337)

AUTOTUNE = tf.data.AUTOTUNE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model")

# ----------------------------
# TensorFlow / GPU setup
# ----------------------------
def setup_tf():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "❌ No GPU detected by TensorFlow. "
            "Check WSL + PyCharm LD_LIBRARY_PATH configuration."
        )

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    logger.info(f"✅ GPUs detected: {gpus}")

    # Mixed precision (huge speedup on RTX GPUs)
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    logger.info("✅ Mixed precision enabled (mixed_float16)")

    # XLA JIT
    tf.config.optimizer.set_jit(True)
    logger.info("✅ XLA JIT enabled")


# ----------------------------
# Backbone selection
# ----------------------------
def get_backbone_and_preprocess(name: str):
    name = name.lower()
    if name in ("efficientnet_v2s", "v2s"):
        return (
            tf.keras.applications.EfficientNetV2S,
            tf.keras.applications.efficientnet_v2.preprocess_input,
        )
    if name in ("efficientnet_b4", "b4"):
        return (
            tf.keras.applications.EfficientNetB4,
            tf.keras.applications.efficientnet.preprocess_input,
        )

    # Fallback
    return (
        tf.keras.applications.EfficientNetV2S,
        tf.keras.applications.efficientnet_v2.preprocess_input,
    )


# ----------------------------
# CSV loading
# ----------------------------
def load_csvs():
    train_df = pd.read_csv(Path(PROCESSED_DATA_DIR) / "processed_train.csv")
    val_df   = pd.read_csv(Path(PROCESSED_DATA_DIR) / "processed_val.csv")
    test_df  = pd.read_csv(Path(PROCESSED_DATA_DIR) / "processed_test.csv")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if not {"filepath", "label_idx"} <= set(df.columns):
            raise ValueError(f"{name} CSV missing required columns")
        if df["filepath"].isna().any():
            raise ValueError(f"{name} CSV has missing filepaths")

    logger.info(
        f"Loaded CSVs: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df


def infer_num_classes(df: pd.DataFrame) -> int:
    return int(df["label_idx"].nunique())


def compute_class_weights(df: pd.DataFrame, num_classes: int):
    counts = df["label_idx"].value_counts().to_dict()
    total = len(df)
    return {
        c: total / (num_classes * counts.get(c, 1))
        for c in range(num_classes)
    }


# ----------------------------
# tf.data pipeline
# ----------------------------
def decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=IMG_CHANNELS, expand_animations=False)
    img.set_shape([None, None, IMG_CHANNELS])
    img = tf.image.resize(img, IMG_SIZE, antialias=True)
    img = tf.cast(img, tf.float32)
    return img


def make_dataset(df, training, preprocess_fn):
    paths = df["filepath"].astype(str).values
    labels = df["label_idx"].astype(np.int32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(min(len(df), 10_000), seed=SEED)

    def _map(p, y):
        x = decode_and_resize(p)
        x = preprocess_fn(x)
        return x, y

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ----------------------------
# Model
# ----------------------------
def build_model(num_classes):
    Backbone, _ = get_backbone_and_preprocess(MODEL_NAME)

    inputs = keras.Input(shape=(*IMG_SIZE, IMG_CHANNELS))
    base = Backbone(include_top=False, weights="imagenet", input_tensor=inputs)
    base.trainable = False

    x = base.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)

    outputs = keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32"
    )(x)

    model = keras.Model(inputs, outputs)
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )


def make_callbacks():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    return [
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(Path(TENSORBOARD_LOG_DIR) / run_id)
        ),
    ]


# ----------------------------
# Training
# ----------------------------
def train():
    setup_tf()

    train_df, val_df, test_df = load_csvs()
    num_classes = infer_num_classes(train_df)

    Backbone, preprocess_fn = get_backbone_and_preprocess(MODEL_NAME)
    logger.info(f"Backbone: {Backbone.__name__} | Classes: {num_classes}")

    train_ds = make_dataset(train_df, True, preprocess_fn)
    val_ds   = make_dataset(val_df, False, preprocess_fn)
    test_ds  = make_dataset(test_df, False, preprocess_fn)

    class_weights = compute_class_weights(train_df, num_classes)

    model, base = build_model(num_classes)
    callbacks = make_callbacks()

    # -------- Stage 1 --------
    logger.info("Stage 1: training classifier head")
    compile_model(model, lr=3e-4)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(5, EPOCHS // 3),
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # -------- Stage 2 --------
    logger.info("Stage 2: fine-tuning backbone")
    base.trainable = True

    for layer in base.layers[: int(len(base.layers) * 0.75)]:
        layer.trainable = False

    compile_model(model, lr=1e-5)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=model.optimizer.iterations.numpy(),
        class_weight=class_weights,
        callbacks=callbacks,
    )

    logger.info("Evaluating...")
    logger.info(dict(zip(model.metrics_names, model.evaluate(test_ds))))

    final_path = Path(BEST_MODEL_PATH).with_name("final_model.keras")
    model.save(final_path)
    logger.info(f"Saved final model → {final_path}")


if __name__ == "__main__":
    train()