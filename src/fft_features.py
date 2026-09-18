"""
Fourier Transform Feature Extraction for ECG Signals.

This module computes frequency-domain representations of ECG beats
to provide complementary features to the time-domain signal.

Design choices:
  - Uses np.fft.rfft (real FFT) since ECG signals are real-valued,
    avoiding redundant negative-frequency bins
  - Log-magnitude spectrum: compresses dynamic range, making low-amplitude
    frequency components (often clinically relevant) more distinguishable
  - Normalization: per-beat L2-normalization of the FFT magnitude ensures
    scale invariance across different recording amplitudes
  - The frequency resolution is fs/N = 125/186 ≈ 0.67 Hz per bin
  - Nyquist frequency = 62.5 Hz (sufficient for ECG: clinical content < 40 Hz)

Integration with the model:
  FFT features are computed as a separate input branch and fused with
  time-domain CNN features before the Transformer encoder.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def compute_fft_features(X: np.ndarray, fs: int = 125, log_scale: bool = True):
    """
    Compute FFT magnitude spectrum for a batch of ECG beats.

    Args:
        X: (N, T) or (N, T, 1) array of ECG beats
        fs: Sampling frequency in Hz
        log_scale: If True, return log(1 + |FFT|) for dynamic range compression

    Returns:
        fft_mag: (N, num_bins) magnitude spectrum (positive frequencies only)
        freqs: (num_bins,) frequency values in Hz
    """
    # Handle 3D input
    if X.ndim == 3:
        X = X.squeeze(axis=-1)

    N = X.shape[1]

    # Real FFT — only positive frequencies
    fft_vals = np.fft.rfft(X, axis=1)

    # Magnitude spectrum
    fft_mag = np.abs(fft_vals)

    if log_scale:
        # log(1 + x) avoids log(0) and compresses dynamic range
        fft_mag = np.log1p(fft_mag)

    # Frequency axis
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    return fft_mag.astype(np.float32), freqs


def normalize_fft(fft_train: np.ndarray, fft_val: np.ndarray = None,
                  fft_test: np.ndarray = None):
    """
    Normalize FFT features using training statistics.
    Fit on train only to prevent leakage.

    Returns:
        Tuple of normalized arrays and (mean, std) statistics.
    """
    mean = fft_train.mean(axis=0, keepdims=True)
    std = fft_train.std(axis=0, keepdims=True) + 1e-8

    fft_train_norm = (fft_train - mean) / std
    fft_val_norm = (fft_val - mean) / std if fft_val is not None else None
    fft_test_norm = (fft_test - mean) / std if fft_test is not None else None

    return fft_train_norm, fft_val_norm, fft_test_norm, (mean, std)


def prepare_fft_inputs(X_train, X_val, X_test, fs: int = 125):
    """
    Compute and normalize FFT features for all splits.

    Args:
        X_train, X_val, X_test: (N, T, 1) arrays
        fs: Sampling frequency

    Returns:
        dict with fft_train, fft_val, fft_test (each shaped (N, num_bins, 1))
        and freqs array
    """
    fft_train, freqs = compute_fft_features(X_train, fs=fs)
    fft_val, _ = compute_fft_features(X_val, fs=fs)
    fft_test, _ = compute_fft_features(X_test, fs=fs)

    # Normalize (fit on train only)
    fft_train, fft_val, fft_test, stats = normalize_fft(fft_train, fft_val, fft_test)

    # Reshape for model: (N, num_bins, 1)
    fft_train = fft_train.reshape(-1, fft_train.shape[1], 1)
    fft_val = fft_val.reshape(-1, fft_val.shape[1], 1)
    fft_test = fft_test.reshape(-1, fft_test.shape[1], 1)

    print(f"FFT features computed:")
    print(f"  Frequency resolution: {freqs[1] - freqs[0]:.2f} Hz/bin")
    print(f"  Number of frequency bins: {len(freqs)}")
    print(f"  Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
    print(f"  FFT shape: {fft_train.shape}")

    return {
        "fft_train": fft_train,
        "fft_val": fft_val,
        "fft_test": fft_test,
        "freqs": freqs,
        "stats": stats,
    }


class FFTLayer(layers.Layer):
    """
    Keras layer that computes log-magnitude FFT of input signals.
    Can be used as an in-model transformation for end-to-end training.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs: (batch, timesteps, 1)
        x = tf.squeeze(inputs, axis=-1)  # (batch, timesteps)
        fft_vals = tf.signal.rfft(x)
        magnitude = tf.abs(fft_vals)
        log_mag = tf.math.log1p(magnitude)
        return tf.expand_dims(log_mag, axis=-1)  # (batch, num_bins, 1)

    def compute_output_shape(self, input_shape):
        n_bins = input_shape[1] // 2 + 1
        return (input_shape[0], n_bins, 1)
