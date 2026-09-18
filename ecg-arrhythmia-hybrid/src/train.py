"""
Training pipeline for ECG arrhythmia classification models.

Supports four model types for ablation study:
  1. baseline_cnn — time-domain only, no Transformer
  2. cnn_transformer — time-domain with Transformer encoder
  3. fourier_hybrid — dual-branch time+frequency with Transformer
  4. fourier_vit_hybrid — dual-branch time+frequency with genuine ViT backbone

Uses focal loss by default to handle severe class imbalance in the
MIT-BIH dataset (Normal beats dominate at ~82%).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from pathlib import Path

from .utils import CONFIG, set_seed, setup_gpu, get_callbacks
from .preprocessing import prepare_data
from .fft_features import prepare_fft_inputs
from .models import build_model


# ============================================================
# Focal Loss — handles class imbalance better than CE
# ============================================================

class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss (Lin et al., 2017).

    Focuses training on hard, misclassified examples by down-weighting
    easy examples. Particularly effective for class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha  # Per-class weights
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        num_classes = tf.shape(y_pred)[-1]

        # One-hot encode sparse labels
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true = tf.one_hot(y_true, num_classes)

        # Label smoothing
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + \
                     self.label_smoothing / num_classes

        # Focal modulation
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Cross-entropy
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        loss = focal_weight * ce

        # Class-wise alpha weighting
        if self.alpha is not None:
            alpha_t = tf.reduce_sum(y_true * self.alpha, axis=-1)
            loss = alpha_t * loss

        return tf.reduce_mean(loss)


def compile_model(model, learning_rate=None, class_weights=None,
                  use_focal_loss=True, focal_gamma=1.0,
                  label_smoothing=0.05, max_alpha=5.0):
    """Compile model with optimizer and loss function.

    Args:
        max_alpha: cap per-class alpha weights to prevent over-predicting
            minority classes.  Balanced class weights can exceed 20× for
            rare classes (e.g. Fusion at 0.7%), which combined with focal
            modulation causes the model to sacrifice majority-class recall
            and tank overall accuracy on the patient-level DS2 test split.
    """
    if learning_rate is None:
        learning_rate = CONFIG["learning_rate"]

    optimizer = optimizers.Adam(learning_rate=learning_rate)

    if use_focal_loss:
        # Convert class weights to alpha tensor for focal loss
        alpha = None
        if class_weights is not None:
            num_classes = max(class_weights.keys()) + 1
            raw = [class_weights.get(i, 1.0) for i in range(num_classes)]
            # Cap alpha values to prevent over-weighting rare classes
            capped = [min(w, max_alpha) for w in raw]
            alpha = tf.constant(capped, dtype=tf.float32)
        loss = FocalLoss(gamma=focal_gamma, alpha=alpha,
                         label_smoothing=label_smoothing)
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    return model


def train_model(model, model_type, data, fft_data=None,
                epochs=None, batch_size=None, class_weights=None):
    """
    Train a model with proper callbacks.

    Args:
        model: compiled Keras model
        model_type: str — determines whether FFT inputs are needed
        data: dict from prepare_data()
        fft_data: dict from prepare_fft_inputs() (needed for fourier_hybrid)
        epochs: number of training epochs
        batch_size: training batch size
        class_weights: dict of class weights

    Returns:
        history: Keras training history
    """
    if epochs is None:
        epochs = CONFIG["epochs"]
    if batch_size is None:
        batch_size = CONFIG["batch_size"]
    if class_weights is None:
        class_weights = data.get("class_weights")

    # Prepare inputs based on model type
    if model_type in ("fourier_hybrid", "fourier_vit_hybrid"):
        assert fft_data is not None, f"FFT data required for {model_type} model"
        train_inputs = [data["X_train"], fft_data["fft_train"]]
        val_inputs = [data["X_val"], fft_data["fft_val"]]
    else:
        train_inputs = data["X_train"]
        val_inputs = data["X_val"]

    callbacks = get_callbacks()

    # Note: class_weight is NOT passed to fit() because focal loss
    # already handles class imbalance via the alpha parameter.
    # Passing both would double-weight minority classes.
    history = model.fit(
        train_inputs, data["y_train"],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_inputs, data["y_val"]),
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    return history


def run_ablation(data_dir=None, epochs=None, batch_size=None):
    """
    Run the complete ablation study: train all three models.

    Returns:
        dict mapping model_type -> (model, history)
    """
    set_seed()
    setup_gpu()

    # Prepare data
    data = prepare_data(data_dir=data_dir)

    # Prepare FFT features
    fft_data = prepare_fft_inputs(
        data["X_train"], data["X_val"], data["X_test"],
        fs=CONFIG["sampling_rate"],
    )

    signal_length = data["X_train"].shape[1]
    fft_length = fft_data["fft_train"].shape[1]
    num_classes = CONFIG["num_classes"]

    results = {}
    model_types = ["baseline_cnn", "cnn_transformer", "fourier_hybrid",
                    "fourier_vit_hybrid"]

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training: {model_type}")
        print(f"{'='*60}\n")

        set_seed()  # Reset seed for fair comparison

        model = build_model(
            model_type,
            signal_length=signal_length,
            fft_length=fft_length,
            num_classes=num_classes,
            dropout_rate=CONFIG["dropout_rate"],
        )

        model = compile_model(
            model,
            class_weights=data["class_weights"],
        )

        model.summary()

        history = train_model(
            model, model_type, data,
            fft_data=fft_data,
            epochs=epochs,
            batch_size=batch_size,
        )

        results[model_type] = {
            "model": model,
            "history": history,
        }

    return results, data, fft_data
