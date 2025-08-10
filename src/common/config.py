# src/common/config.py

import os

# General project config
PROJECT_NAME = "OptionsLab"
DATA_DIR = os.getenv("DATA_DIR", "data/processed")
MODEL_DIR = os.getenv("MODEL_DIR", "models/saved_models")

# Training defaults
DEFAULT_RANDOM_SEED = 42
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 200

# Logging config
LOG_LEVEL = "INFO"

# Model hyperparams can go here or better yet, per model separately
