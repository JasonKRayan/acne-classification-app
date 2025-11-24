import tensorflow as tf

# ----------------------------
# Acne-Safe Image Augmentations
# ----------------------------

def get_acne_augmentation_model():
    """
    Returns a Keras Sequential model that applies augmentation
    safe for facial acne / dermatology images.
    """
    return tf.keras.Sequential([
        # 1. Horizontal flip — safe for faces
        tf.keras.layers.RandomFlip("horizontal"),

        # 2. Small rotations: ±10 degrees (≈0.17 radians)
        tf.keras.layers.RandomRotation(0.17),  # 10 degrees

        # 3. Mild zoom in OR out (±10%)
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1)
        ),

        # 4. Small translations (±10%)
        tf.keras.layers.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1
        ),

        # 5. Mild brightness & contrast adjustments
        tf.keras.layers.RandomContrast(0.1),      # 10%
        tf.keras.layers.RandomBrightness(0.1),    # 10%
    ], name="acne_augmentation")
    


def apply_augmentations(dataset, batch_size):
    """
    Takes a dataset of (image, label) pairs and applies augmentation
    ONLY to the training data.
    """
    aug = get_acne_augmentation_model()

    return (dataset
            .shuffle(4096)
            .map(lambda x, y: (aug(x, training=True), y),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
