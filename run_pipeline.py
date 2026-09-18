#!/usr/bin/env python3
"""
ECG Arrhythmia Detection — Full Training & Evaluation Pipeline

Usage:
    python run_pipeline.py [--data_dir data] [--epochs 50] [--batch_size 128]
    python run_pipeline.py --binary          # Binary mode only (Normal vs Abnormal)
    python run_pipeline.py --both            # Run both 5-class and binary

This script:
  1. Loads and preprocesses the ECG heartbeat dataset
  2. Computes FFT frequency-domain features
  3. Trains four models for ablation study:
       a. Baseline CNN (time-domain only)
       b. CNN + Transformer (time-domain only)
       c. Fourier Hybrid CNN-Transformer (time + frequency domain)
       d. Fourier ViT Hybrid (time + frequency, genuine ViT backbone)
  4. Evaluates all models on the held-out test set
  5. Generates comparison tables, plots, bootstrap CIs, and saliency maps
  6. Saves all results to results/

NOTE: The dataset must be downloaded manually from Kaggle.
      See data/README.md for instructions.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import CONFIG, set_seed, setup_gpu
from src.preprocessing import prepare_data
from src.fft_features import prepare_fft_inputs
from src.models import build_model
from src.train import compile_model, train_model, FocalLoss
from src.evaluate import (
    run_full_evaluation,
    evaluate_model,
    build_comparison_table,
)


def run_mode(mode_name, binary, model_types, args, output_dir):
    """
    Run a single training+evaluation pass (either 5-class or binary).

    Args:
        mode_name: display name for this mode
        binary: whether to use binary labels
        model_types: list of model type strings to train
        args: parsed CLI arguments
        output_dir: Path for saving results

    Returns:
        (all_eval, comparison_df)
    """
    print(f"\n{'#'*60}")
    print(f"  {mode_name}")
    print(f"{'#'*60}")

    # 1. Prepare data
    print(f"\n[1/4] Preparing data ({mode_name})...")
    data = prepare_data(data_dir=args.data_dir, binary=binary)

    # 2. Compute FFT features
    print(f"\n[2/4] Computing FFT features...")
    fft_data = prepare_fft_inputs(
        data["X_train"], data["X_val"], data["X_test"],
        fs=CONFIG["sampling_rate"],
    )

    signal_length = data["X_train"].shape[1]
    fft_length = fft_data["fft_train"].shape[1]
    num_classes = data["num_classes"]

    # 3. Train all models
    print(f"\n[3/4] Training {len(model_types)} models...")
    trained_results = {}

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training: {model_type} ({mode_name})")
        print(f"{'='*60}\n")

        set_seed()  # Reset for fair comparison

        model = build_model(
            model_type,
            signal_length=signal_length,
            fft_length=fft_length,
            num_classes=num_classes,
            dropout_rate=CONFIG["dropout_rate"],
        )

        model = compile_model(model, class_weights=data["class_weights"])
        model.summary()

        history = train_model(
            model, model_type, data,
            fft_data=fft_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        trained_results[model_type] = {
            "model": model,
            "history": history,
        }

    # 4. Evaluate all models
    print(f"\n[4/4] Evaluating models ({mode_name})...")
    all_eval, comparison_df = run_full_evaluation(
        trained_results, data, fft_data,
        output_dir=output_dir,
    )

    print(f"\n{'='*50}")
    print(f"{mode_name} — Final Comparison:")
    print(f"{'='*50}")
    print(comparison_df.to_string(index=False))

    return all_eval, comparison_df


def main():
    parser = argparse.ArgumentParser(
        description="ECG Arrhythmia Classification Pipeline"
    )
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing mitbih_train.csv and mitbih_test.csv")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"],
                        help="Training batch size")
    parser.add_argument("--seed", type=int, default=CONFIG["random_seed"],
                        help="Random seed for reproducibility")
    parser.add_argument("--binary", action="store_true",
                        help="Run binary classification only (Normal vs Abnormal)")
    parser.add_argument("--both", action="store_true",
                        help="Run both 5-class and binary classification")
    args = parser.parse_args()

    CONFIG["data_dir"] = Path(args.data_dir)
    CONFIG["random_seed"] = args.seed

    print("=" * 60)
    print("ECG Arrhythmia Detection — Hybrid CNN-ViT Pipeline")
    print("=" * 60)

    # Setup
    set_seed()
    setup_gpu()

    # All 4 model types for the ablation study
    model_types = [
        "baseline_cnn",
        "cnn_transformer",
        "fourier_hybrid",
        "fourier_vit_hybrid",
    ]

    results_base = Path(CONFIG["results_dir"])

    if args.both:
        # Run both modes
        print("\nRunning BOTH 5-class and binary classification modes.\n")

        run_mode(
            "5-Class AAMI Classification", binary=False,
            model_types=model_types, args=args,
            output_dir=results_base / "5class",
        )
        run_mode(
            "Binary Classification (Normal vs Abnormal)", binary=True,
            model_types=model_types, args=args,
            output_dir=results_base / "binary",
        )

        print(f"\n{'='*60}")
        print("ALL DONE — Both modes complete!")
        print(f"  5-class results: {results_base / '5class'}/")
        print(f"  Binary results:  {results_base / 'binary'}/")
        print(f"{'='*60}")

    elif args.binary:
        run_mode(
            "Binary Classification (Normal vs Abnormal)", binary=True,
            model_types=model_types, args=args,
            output_dir=results_base,
        )
    else:
        run_mode(
            "5-Class AAMI Classification", binary=False,
            model_types=model_types, args=args,
            output_dir=results_base,
        )


if __name__ == "__main__":
    main()
