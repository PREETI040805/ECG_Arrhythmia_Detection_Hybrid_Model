"""
Utility functions: configuration, reproducibility, logging.
"""

import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================
# Default Configuration
# ============================================================
CONFIG = {
    # Data
    "sampling_rate": 125,          # Hz — Kaggle heartbeat dataset is resampled to 125 Hz
    "num_classes": 5,              # AAMI: N(0), S(1), V(2), F(3), Q(4)
    "signal_length": 186,          # Time-domain features per beat (187 cols - 1 label)
    "class_names": ["N", "S", "V", "F", "Q"],
    "class_full_names": [
        "Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"
    ],

    # Paths
    "data_dir": Path("data"),
    "results_dir": Path("results"),
    "models_dir": Path("models"),

    # Training
    "batch_size": 128,
    "epochs": 50,
    "learning_rate": 1e-3,
    "val_split": 0.2,
    "random_seed": 42,

    # Model
    "cnn_filters": [32, 64, 128],
    "cnn_kernel_size": 5,
    "transformer_heads": 4,
    "transformer_head_size": 32,
    "transformer_ff_dim": 128,
    "transformer_layers": 2,
    "dropout_rate": 0.3,
    "dense_units": 128,

    # FFT
    "fft_bins": 93,  # N//2 for signal_length=186

    # Callbacks
    "early_stopping_patience": 10,
    "lr_reduce_patience": 4,
    "lr_reduce_factor": 0.5,
    "min_lr": 1e-6,
}


def set_seed(seed: int = None):
    """Set random seeds for reproducibility across Python, NumPy, and TensorFlow."""
    if seed is None:
        seed = CONFIG["random_seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpu():
    """Detect and configure GPU. Returns device summary string."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        summary = f"GPU(s) available: {[g.name for g in gpus]}"
    else:
        summary = "No GPU detected — training on CPU"
    print(summary)
    return summary


def get_callbacks(monitor: str = "val_loss"):
    """Standard training callbacks: early stopping, LR reduction, model checkpoint."""
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=CONFIG["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=CONFIG["lr_reduce_factor"],
            patience=CONFIG["lr_reduce_patience"],
            min_lr=CONFIG["min_lr"],
            verbose=1,
        ),
    ]

    # Model checkpoint
    ckpt_dir = CONFIG["models_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        ModelCheckpoint(
            filepath=str(ckpt_dir / "best_model.keras"),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        )
    )

    return callbacks
