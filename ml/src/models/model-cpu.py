"""
model.py - Improved training for acne/skin disease classification from processed CSVs.

CSV columns expected:
  filepath   : absolute image path
  label_idx  : integer class id
  label      : optional readable name

Key upgrades vs basic version:
  - Uses correct keras.applications preprocessing for the chosen backbone
  - Two-stage training: (1) train head with frozen backbone, (2) fine-tune
  - Class weights to fight imbalance
  - Better callbacks (ReduceLROnPlateau + EarlyStopping + checkpoints)
  - Optional label smoothing
  - Robust decoding + explicit shape for TF graph performance
"""

import os
import math
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config import (
    PROCESSED_DATA_DIR,
    BEST_MODEL_PATH,
    TENSORBOARD_LOG_DIR,
    CHECKPOINT_DIR,
    IMG_SIZE,
    IMG_CHANNELS,
    BATCH_SIZE,
    EPOCHS,  # used for total epochs; we split into stage1+stage2 below
)

# If your config.py doesn't define these, we provide safe defaults here.
MODEL_NAME = getattr(__import__("config"), "MODEL_NAME", "efficientnet_v2s")
SEED = getattr(__import__("config"), "SEED", 1337)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model")

AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------
# Performance options (GPU)
# ----------------------------
def setup_tf():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPUs detected: {gpus}")
        except RuntimeError as e:
            logger.warning(f"Could not set memory growth: {e}")
    else:
        logger.warning("No GPU detected by TensorFlow; training will run on CPU.")

    # Mixed precision can speed up training on many NVIDIA GPUs (Tensor Cores)
    # If you see instability or NaNs, comment this out.
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        logger.info("Enabled mixed precision (mixed_float16).")
    except Exception as e:
        logger.warning(f"Mixed precision not enabled: {e}")


# ----------------------------
# Backbone selection
# ----------------------------
def get_backbone_and_preprocess(model_name: str):
    """
    Returns (BackboneClass, preprocess_fn, default_input_size_guess)
    """
    model_name = (model_name or "").lower()

    if model_name in ("efficientnet_v2s", "effnetv2s", "v2s"):
        return tf.keras.applications.EfficientNetV2S, tf.keras.applications.efficientnet_v2.preprocess_input
    if model_name in ("efficientnet_b0", "b0"):
        return tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input
    if model_name in ("efficientnet_b4", "b4", "lite4", "efficientnet_lite4"):
        return tf.keras.applications.EfficientNetB4, tf.keras.applications.efficientnet.preprocess_input

    # sensible fallback
    return tf.keras.applications.EfficientNetV2S, tf.keras.applications.efficientnet_v2.preprocess_input


# ----------------------------
# CSV loading
# ----------------------------
def load_csvs(processed_dir: Path):
    processed_dir = Path(processed_dir)
    train_df = pd.read_csv(processed_dir / "processed_train.csv")
    val_df   = pd.read_csv(processed_dir / "processed_val.csv")
    test_df  = pd.read_csv(processed_dir / "processed_test.csv")

    # Basic sanity checks
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "filepath" not in df.columns or "label_idx" not in df.columns:
            raise ValueError(f"{name} CSV must contain columns: filepath, label_idx")
        if df["filepath"].isna().any():
            raise ValueError(f"{name} CSV has missing filepaths.")
        if df["label_idx"].isna().any():
            raise ValueError(f"{name} CSV has missing label_idx.")

    logger.info(f"Loaded rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def infer_num_classes(train_df: pd.DataFrame) -> int:
    # assumes label_idx is 0..K-1; safe if it's not contiguous:
    return int(train_df["label_idx"].nunique())


def compute_class_weights(train_df: pd.DataFrame, num_classes: int) -> dict:
    """
    Balanced weights: total/(K*count_c)
    """
    counts = train_df["label_idx"].value_counts().to_dict()
    total = len(train_df)

    weights = {}
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        if cnt == 0:
            # If a class is missing in train, training can't learn it; still define weight.
            weights[c] = 1.0
        else:
            weights[c] = total / (num_classes * cnt)

    logger.info(f"Class weights (sample): {dict(list(weights.items())[:5])} ...")
    return weights


# ----------------------------
# tf.data pipeline
# ----------------------------
def decode_and_resize(path: tf.Tensor):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=IMG_CHANNELS, expand_animations=False)
    img.set_shape([None, None, IMG_CHANNELS])  # important for TF shape inference
    img = tf.image.resize(img, IMG_SIZE, antialias=True)
    img = tf.cast(img, tf.float32)  # keep float32; preprocess_input handles scaling
    return img


def make_dataset(df: pd.DataFrame, batch_size: int, training: bool, preprocess_fn):
    paths = df["filepath"].astype(str).values
    labels = df["label_idx"].astype(np.int32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(min(len(df), 10_000), seed=SEED, reshuffle_each_iteration=True)

    def _map(path, y):
        x = decode_and_resize(path)
        # EfficientNet preprocess expects float input; returns float
        x = preprocess_fn(x)
        return x, y

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ----------------------------
# Model (stronger head + fine-tuning)
# ----------------------------
def build_model(num_classes: int, backbone_name: str):
    BackboneCls, _ = get_backbone_and_preprocess(backbone_name)

    inputs = keras.Input(shape=(*IMG_SIZE, IMG_CHANNELS), name="image")

    base = BackboneCls(include_top=False, weights="imagenet", input_tensor=inputs, pooling=None)
    base.trainable = False  # stage 1

    x = base.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)

    # IMPORTANT with mixed precision: force float32 output for numeric stability in loss/metrics
    outputs = keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name=f"{backbone_name}_classifier")
    return model, base


def compile_model(model: keras.Model, lr: float):
    loss = keras.losses.SparseCategoricalCrossentropy()  # ✅ no label_smoothing
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )


def make_callbacks():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(TENSORBOARD_LOG_DIR) / f"run_{timestamp}"

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]


# ----------------------------
# Train (two-stage)
# ----------------------------
def train():
    setup_tf()

    train_df, val_df, test_df = load_csvs(PROCESSED_DATA_DIR)
    num_classes = infer_num_classes(train_df)
    logger.info(f"Num classes: {num_classes}")

    BackboneCls, preprocess_fn = get_backbone_and_preprocess(MODEL_NAME)
    logger.info(f"Backbone: {BackboneCls.__name__} (MODEL_NAME={MODEL_NAME})")

    train_ds = make_dataset(train_df, BATCH_SIZE, training=True, preprocess_fn=preprocess_fn)
    val_ds   = make_dataset(val_df,   BATCH_SIZE, training=False, preprocess_fn=preprocess_fn)
    test_ds  = make_dataset(test_df,  BATCH_SIZE, training=False, preprocess_fn=preprocess_fn)

    class_weights = compute_class_weights(train_df, num_classes)

    model, base = build_model(num_classes, MODEL_NAME)

    callbacks = make_callbacks()

    # ---- Stage 1: train head (frozen backbone) ----
    stage1_epochs = max(5, min(12, EPOCHS // 3))
    logger.info(f"Stage 1: training head for {stage1_epochs} epochs (backbone frozen).")

    compile_model(model, lr=3e-4)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # ---- Stage 2: fine-tune backbone ----
    # Unfreeze last ~25% of layers (common sweet spot)
    base.trainable = True
    layers = base.layers
    cutoff = int(len(layers) * 0.75)
    for l in layers[:cutoff]:
        l.trainable = False

    # Lower LR for fine-tuning
    stage2_epochs = max(10, EPOCHS - stage1_epochs)
    logger.info(
        f"Stage 2: fine-tuning for {stage2_epochs} epochs "
        f"(unfroze {len(layers) - cutoff}/{len(layers)} backbone layers)."
    )

    compile_model(model, lr=1e-5)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs + stage2_epochs,
        initial_epoch=stage1_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Evaluate
    logger.info("Evaluating on validation set...")
    val_metrics = model.evaluate(val_ds, verbose=1)
    logger.info(f"Val metrics: {dict(zip(model.metrics_names, val_metrics))}")

    logger.info("Evaluating on test set...")
    test_metrics = model.evaluate(test_ds, verbose=1)
    logger.info(f"Test metrics: {dict(zip(model.metrics_names, test_metrics))}")

    # Save final
    final_path = Path(BEST_MODEL_PATH).with_name("final_model.keras")
    model.save(final_path)
    logger.info(f"Saved final model: {final_path}")
    logger.info(f"Best checkpoint path: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()
