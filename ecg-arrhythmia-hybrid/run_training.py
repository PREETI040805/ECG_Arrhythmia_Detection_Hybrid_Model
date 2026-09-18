#!/usr/bin/env python3
"""
Full training and evaluation — optimized for CPU execution.
Trains 3 models with early stopping, generates all results.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from src.utils import CONFIG, set_seed, setup_gpu
from src.preprocessing import prepare_data
from src.fft_features import prepare_fft_inputs
from src.models import build_model
from src.train import compile_model, train_model
from src.evaluate import run_full_evaluation

def main():
    print("=" * 60)
    print("ECG Arrhythmia Detection — Training Pipeline")
    print("=" * 60)

    set_seed()
    setup_gpu()

    # 1. Data
    print("\n[1/5] Loading data...")
    data = prepare_data(data_dir="data")

    # 2. FFT
    print("\n[2/5] Computing FFT features...")
    fft_data = prepare_fft_inputs(
        data["X_train"], data["X_val"], data["X_test"],
        fs=CONFIG["sampling_rate"],
    )

    signal_length = data["X_train"].shape[1]
    fft_length = fft_data["fft_train"].shape[1]
    num_classes = CONFIG["num_classes"]

    # 3. Train
    print("\n[3/5] Training models...")
    EPOCHS = 30  # Early stopping will kick in well before this
    BATCH_SIZE = 256  # Larger batch for faster CPU training

    trained_results = {}
    model_types = ["baseline_cnn", "cnn_transformer", "fourier_hybrid"]

    for model_type in model_types:
        print(f"\n{'=' * 60}")
        print(f"Training: {model_type}")
        print(f"{'=' * 60}")

        set_seed()

        model = build_model(
            model_type,
            signal_length=signal_length,
            fft_length=fft_length,
            num_classes=num_classes,
            dropout_rate=CONFIG["dropout_rate"],
        )

        model = compile_model(model, class_weights=data["class_weights"])

        # Print param count
        total_params = model.count_params()
        print(f"Parameters: {total_params:,}")

        history = train_model(
            model, model_type, data,
            fft_data=fft_data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )

        trained_results[model_type] = {
            "model": model,
            "history": history,
        }

        # Quick interim metric
        best_val_loss = min(history.history['val_loss'])
        best_val_acc = max(history.history['val_accuracy'])
        print(f"\nBest val loss: {best_val_loss:.4f}, Best val acc: {best_val_acc:.4f}")

    # 4. Evaluate
    print("\n[4/5] Evaluating all models...")
    all_eval, comparison_df = run_full_evaluation(
        trained_results, data, fft_data,
        output_dir=CONFIG["results_dir"],
    )

    # 5. Summary
    print("\n[5/5] Complete!")
    print(f"\nResults saved to: {CONFIG['results_dir']}/")
    print(f"\nFinal Comparison:")
    print(comparison_df.to_string(index=False))

    return all_eval, comparison_df


if __name__ == "__main__":
    main()
