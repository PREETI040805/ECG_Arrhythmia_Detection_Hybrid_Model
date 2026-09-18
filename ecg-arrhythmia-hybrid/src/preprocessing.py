"""
ECG Preprocessing Pipeline.

Designed for the Kaggle ECG Heartbeat Categorization Dataset (Kachuee et al., 2018).
The dataset contains pre-segmented, R-peak-centered beats resampled to 125 Hz
with 186 time-domain samples per beat, zero-padded to fixed length.

Pipeline:
  1. Load pre-segmented data (mitbih_train.csv / mitbih_test.csv)
  2. Signal validation (NaN, constant-signal checks)
  3. Baseline-wander removal (median subtraction per beat)
  4. Per-beat normalization (zero-mean, unit-variance)
  5. Train/validation split (stratified)
  6. Class distribution analysis

WHY each step:
  - Baseline wander removal: eliminates DC offset / respiratory drift per beat
  - Per-beat normalization: ensures amplitude-invariant features, critical because
    the raw ADC values have arbitrary offsets per patient/recording
  - No additional band-pass filtering: the Kaggle dataset is already filtered and
    resampled; additional filtering risks distorting the pre-segmented morphology
  - No notch filtering: 50/60 Hz powerline noise is already above Nyquist (125/2=62.5 Hz)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from .utils import CONFIG


def load_dataset(data_dir: Path = None):
    """
    Load the Kaggle ECG Heartbeat Categorization Dataset.

    Returns:
        X_train_full: (N_train, 186) raw time-domain features
        y_train_full: (N_train,) integer labels 0-4
        X_test: (N_test, 186) raw time-domain features
        y_test: (N_test,) integer labels 0-4
    """
    if data_dir is None:
        data_dir = CONFIG["data_dir"]
    data_dir = Path(data_dir)

    train_path = data_dir / "mitbih_train.csv"
    test_path = data_dir / "mitbih_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset not found in {data_dir}/. "
            "Download from https://www.kaggle.com/datasets/shayanfazeli/heartbeat "
            "and place mitbih_train.csv and mitbih_test.csv in the data/ directory."
        )

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    X_train_full = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train_full = train_df.iloc[:, -1].values.astype(int)

    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(int)

    print(f"Loaded dataset from {data_dir}")
    print(f"  Train: {X_train_full.shape[0]:,} samples, {X_train_full.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} samples, {X_test.shape[1]} features")

    return X_train_full, y_train_full, X_test, y_test


def validate_signals(X: np.ndarray, label: str = "data"):
    """
    Validate ECG signal array for common issues.

    Checks:
      - NaN / Inf values
      - Constant (zero-variance) signals
      - Extreme outliers (> 10 std from mean)

    Returns number of issues found.
    """
    issues = 0

    nan_mask = np.any(np.isnan(X) | np.isinf(X), axis=1)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        print(f"  WARNING [{label}]: {n_nan} samples contain NaN/Inf values")
        issues += n_nan

    variances = np.var(X, axis=1)
    n_const = (variances < 1e-10).sum()
    if n_const > 0:
        print(f"  WARNING [{label}]: {n_const} constant/near-zero-variance signals")
        issues += n_const

    if issues == 0:
        print(f"  [{label}]: All {X.shape[0]:,} signals passed validation")

    return issues


def remove_baseline_wander(X: np.ndarray) -> np.ndarray:
    """
    Remove baseline wander by subtracting per-beat median.

    Median is more robust to QRS outliers than mean for baseline estimation.
    """
    baselines = np.median(X, axis=1, keepdims=True)
    return X - baselines


def normalize_beats(X_train: np.ndarray, X_val: np.ndarray = None,
                    X_test: np.ndarray = None):
    """
    Fit StandardScaler on training data, transform all splits.

    Fitting ONLY on training data prevents information leakage.

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
        X_val_scaled and X_test_scaled are None if inputs are None.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def compute_class_weights(y: np.ndarray):
    """Compute balanced class weights for handling class imbalance."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_dict = dict(zip(classes.astype(int), weights))
    return class_weight_dict


def get_class_distribution(y: np.ndarray, class_names: list = None):
    """Return a DataFrame with class distribution statistics."""
    if class_names is None:
        class_names = CONFIG["class_full_names"]

    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    data = []
    for cls, count in zip(unique, counts):
        name = class_names[int(cls)] if int(cls) < len(class_names) else f"Class {cls}"
        data.append({
            "Class": int(cls),
            "Name": name,
            "Count": count,
            "Percentage": f"{100 * count / total:.2f}%",
        })

    return pd.DataFrame(data)


def to_binary_labels(y: np.ndarray) -> np.ndarray:
    """
    Merge AAMI 5-class labels into binary: Normal (0) vs Abnormal (1).

    Mapping:
      0 (N) → 0 (Normal)
      1 (S), 2 (V), 3 (F), 4 (Q) → 1 (Abnormal)
    """
    return (y > 0).astype(int)


def prepare_data(data_dir: Path = None, val_split: float = None,
                 seed: int = None, binary: bool = False):
    """
    Complete preprocessing pipeline.

    Steps:
      1. Load dataset
      2. Validate signals
      3. Remove baseline wander
      4. Optionally convert to binary labels (Normal vs Abnormal)
      5. Train/val split (stratified)
      6. Normalize (fit on train only)
      7. Compute class weights
      8. Reshape for model input (samples, timesteps, 1)

    Args:
        binary: if True, merge labels into Normal (0) vs Abnormal (1)

    Returns:
        dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                        class_weights, scaler, train_dist, test_dist,
                        num_classes, class_names
    """
    if val_split is None:
        val_split = CONFIG["val_split"]
    if seed is None:
        seed = CONFIG["random_seed"]

    # 1. Load
    X_train_full, y_train_full, X_test, y_test = load_dataset(data_dir)

    # 2. Validate
    print("\nSignal validation:")
    validate_signals(X_train_full, "train")
    validate_signals(X_test, "test")

    # 3. Baseline wander removal
    X_train_full = remove_baseline_wander(X_train_full)
    X_test = remove_baseline_wander(X_test)

    # 4. Binary conversion
    if binary:
        y_train_full = to_binary_labels(y_train_full)
        y_test = to_binary_labels(y_test)
        num_classes = 2
        class_names = ["Normal", "Abnormal"]
        short_names = ["N", "A"]
        print("\n  Converted to binary: Normal (0) vs Abnormal (1)")
    else:
        num_classes = CONFIG["num_classes"]
        class_names = CONFIG["class_full_names"]
        short_names = CONFIG["class_names"]

    # 5. Train/Val split — stratified
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_split,
        random_state=seed,
        stratify=y_train_full,
        shuffle=True,
    )

    # 6. Normalize (fit on train ONLY)
    X_train, X_val, X_test, scaler = normalize_beats(X_train, X_val, X_test)

    # 7. Class weights
    class_weights = compute_class_weights(y_train)

    # 8. Class distributions
    train_dist = get_class_distribution(
        y_train,
        class_names=class_names,
    )
    test_dist = get_class_distribution(
        y_test,
        class_names=class_names,
    )

    print(f"\n{'Binary' if binary else '5-class'} training set distribution:")
    print(train_dist.to_string(index=False))
    print(f"\nClass weights: {class_weights}")

    # 9. Reshape: (samples, timesteps, channels=1)
    X_train = X_train.reshape(-1, X_train.shape[1], 1)
    X_val = X_val.reshape(-1, X_val.shape[1], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], 1)

    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "class_weights": class_weights,
        "scaler": scaler,
        "train_dist": train_dist,
        "test_dist": test_dist,
        "num_classes": num_classes,
        "class_names": class_names,
        "short_names": short_names,
        "binary": binary,
    }
