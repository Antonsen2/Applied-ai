"""Configuration settings for the CNN model"""

from pathlib import Path

BATCH_SIZE = 32
IMG_SIZE = (250, 250)
RANDOM_SEED = 42

DATA_DIR = ''  # Local or Drive path to the images
IMG_DIR = Path(DATA_DIR)
CHECKPOINT_PATH = 'fire_classification_model_checkpoint'

COLOR_MODE = 'rgb'
CLASS_MODE = 'categorical'
