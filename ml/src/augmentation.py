"""
augmentation.py - Medical-safe image augmentations for acne/dermatology
"""
import tensorflow as tf
from config import AUGMENTATION_CONFIG


def get_acne_augmentation_model():
    """
    Returns a Keras Sequential model with augmentations safe for 
    facial acne and dermatology images.
    
    Key principles:
    - No vertical flips (faces have natural orientation)
    - Conservative rotations (±10°)
    - Mild zoom/translation to simulate different camera distances
    - Brightness/contrast for lighting variations
    - NO color jittering (could alter diagnostic features)
    - NO extreme distortions
    """
    layers = []
    
    # Horizontal flip - safe for symmetric faces
    if AUGMENTATION_CONFIG.get('horizontal_flip', True):
        layers.append(
            tf.keras.layers.RandomFlip("horizontal")
        )
    
    # Small rotations - faces aren't always perfectly aligned
    rotation_factor = AUGMENTATION_CONFIG.get('rotation_factor', 0.17)
    if rotation_factor > 0:
        layers.append(
            tf.keras.layers.RandomRotation(
                rotation_factor,
                fill_mode='reflect'  # Better than zeros at edges
            )
        )
    
    # Mild zoom - simulates different camera distances
    zoom_range = AUGMENTATION_CONFIG.get('zoom_range', 0.1)
    if zoom_range > 0:
        layers.append(
            tf.keras.layers.RandomZoom(
                height_factor=(-zoom_range, zoom_range),
                width_factor=(-zoom_range, zoom_range),
                fill_mode='reflect'
            )
        )
    
    # Small translations - simulates slightly off-center framing
    translation_range = AUGMENTATION_CONFIG.get('translation_range', 0.1)
    if translation_range > 0:
        layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=translation_range,
                width_factor=translation_range,
                fill_mode='reflect'
            )
        )
    
    # Contrast adjustment - different lighting conditions
    contrast_range = AUGMENTATION_CONFIG.get('contrast_range', 0.1)
    if contrast_range > 0:
        layers.append(
            tf.keras.layers.RandomContrast(contrast_range)
        )
    
    # Brightness adjustment - different exposure settings
    brightness_range = AUGMENTATION_CONFIG.get('brightness_range', 0.1)
    if brightness_range > 0:
        layers.append(
            tf.keras.layers.RandomBrightness(
                brightness_range,
                value_range=(0.0, 1.0)  # Important: specify input range
            )
        )
    
    return tf.keras.Sequential(layers, name="acne_augmentation")


def visualize_augmentations(image, num_examples=9):
    """
    Helper function to visualize augmentations on a single image.
    Useful for debugging and tuning augmentation parameters.
    
    Args:
        image: A single image tensor [H, W, C] in range [0, 1]
        num_examples: Number of augmented versions to generate
    
    Returns:
        List of augmented images
    """
    aug_model = get_acne_augmentation_model()
    
    # Add batch dimension
    image_batch = tf.expand_dims(image, 0)
    
    augmented_images = []
    for _ in range(num_examples):
        aug_img = aug_model(image_batch, training=True)
        augmented_images.append(tf.squeeze(aug_img, 0))
    
    return augmented_images


# For backward compatibility - but this shouldn't be used
# The dataset.py handles augmentation properly
def apply_augmentations(dataset, batch_size):
    """
    DEPRECATED: Use dataset.py's get_datasets() instead.
    
    This function is kept for backward compatibility but should not be used.
    The dataset.py module handles augmentation in the correct order.
    """
    import warnings
    warnings.warn(
        "apply_augmentations() is deprecated. "
        "Use dataset.get_datasets() which handles augmentation correctly.",
        DeprecationWarning
    )
    
    aug = get_acne_augmentation_model()
    
    return (dataset
            .shuffle(4096)
            .map(lambda x, y: (aug(x, training=True), y),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))