"""
config.py - Centralized configuration for the acne classification project
"""
from pathlib import Path

# ============================================
# PROJECT PATHS 
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent  # Points to ml/ directory
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "SkinDisease"  # Contains Train/ and Test/
PROCESSED_DATA_DIR = DATA_DIR / "processed"      # Will contain CSVs
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# ============================================
# IMAGE PREPROCESSING
# ============================================
IMG_HEIGHT = 300
IMG_WIDTH = 300
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Normalization strategy for EfficientNet-Lite4
# Options: 'tf' ([-1, 1]), 'torch' (ImageNet), 'caffe', or None
NORMALIZATION_MODE = 'tf'  # EfficientNet typically uses [-1, 1]

# ============================================
# DATA AUGMENTATION
# ============================================
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'rotation_factor': 0.17,  # ~10 degrees in radians
    'zoom_range': 0.1,  # ±10%
    'translation_range': 0.1,  # ±10%
    'contrast_range': 0.1,  # ±10%
    'brightness_range': 0.1,  # ±10%
}

# ============================================
# DATASET CONFIGURATION
# ============================================
VAL_FRACTION = 0.1
SHUFFLE_BUFFER_SIZE = 1000
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

# ============================================
# TRAINING HYPERPARAMETERS
# ============================================
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# Early stopping patience
EARLY_STOPPING_PATIENCE = 10

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_NAME = 'efficientnet_lite4'
NUM_CLASSES = None  
FREEZE_BASE_MODEL = True  # Start with frozen base, then unfreeze

# ============================================
# REPRODUCIBILITY
# ============================================
RANDOM_SEED = 42

# ============================================
# LOGGING & CHECKPOINTING
# ============================================
TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
BEST_MODEL_PATH = MODEL_DIR / "best_model.keras"

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, MODEL_DIR, LOGS_DIR, 
                  TENSORBOARD_LOG_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)