"""Central configuration: paths, constants, and hyperparameter defaults.

DR_DATA_DIR / DR_OUTPUT_DIR environment variables override the default
locations — used by the smoke test, and useful on Colab to point at Drive.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("DR_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"                 # unzipped Kaggle download
PROCESSED_DIR = DATA_DIR / "processed"     # cached preprocessed images
SPLIT_CSV = DATA_DIR / "splits.csv"        # frozen train/val/test assignment
OUTPUT_DIR = Path(os.environ.get("DR_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

KAGGLE_COMPETITION = "aptos2019-blindness-detection"

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
NUM_CLASSES = 5
# Grades >= 2 are "referable DR": the binary decision screening programmes act on.
REFERABLE_THRESHOLD = 2

SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Images are cached once at this size; per-model transforms resize from here.
CACHE_SIZE = 320
# Ben Graham Gaussian-blur subtraction strength.
BEN_SIGMA = 10

# Native input resolution per architecture.
IMG_SIZE = {
    "resnet18_scratch": 224,
    "resnet18": 224,
    "efficientnet_b0": 224,
    "efficientnet_b3": 300,
}

# ImageNet normalisation — used for all models so the ablation only varies
# the fundus preprocessing, not the tensor statistics.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULTS = {
    "epochs": 30,
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "early_stop_patience": 8,   # epochs without val QWK improvement
    "num_workers": 2,
}
