"""
Evaluation, visualization, and model comparison for ECG classification.

Generates:
  - Classification report (per-class precision, recall, F1)
  - Confusion matrix (raw and normalized)
  - ROC curves (one-vs-rest for multiclass)
  - Precision-Recall curves
  - Training/validation curves
  - Bootstrap confidence intervals
  - Model comparison table
  - Attention / saliency visualization
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score,
)
import tensorflow as tf

from .utils import CONFIG


# ============================================================
# Plot style
# ============================================================

def _setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
    })

_setup_style()

CLASS_NAMES = CONFIG["class_full_names"]
SHORT_NAMES = CONFIG["class_names"]


# ============================================================
# Core evaluation
# ============================================================

def evaluate_model(model, model_type, data, fft_data=None,
                   class_names=None):
    """
    Evaluate a trained model on the test set.

    Returns a dict with predictions, probabilities, and all metrics.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Prepare test inputs
    if model_type in ("fourier_hybrid", "fourier_vit_hybrid"):
        test_inputs = [data["X_test"], fft_data["fft_test"]]
    else:
        test_inputs = data["X_test"]

    y_test = data["y_test"]

    # Predict
    y_prob = model.predict(test_inputs, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ROC-AUC
    try:
        n_classes = y_prob.shape[1]
        if n_classes == 2:
            # Binary: use probability of positive class
            roc_auc_macro = roc_auc_score(y_test, y_prob[:, 1])
            roc_auc_weighted = roc_auc_macro  # same for binary
        else:
            # Multiclass: one-vs-rest
            roc_auc_macro = roc_auc_score(
                y_test, y_prob, multi_class="ovr", average="macro"
            )
            roc_auc_weighted = roc_auc_score(
                y_test, y_prob, multi_class="ovr", average="weighted"
            )
    except ValueError:
        roc_auc_macro = roc_auc_weighted = float("nan")

    report_str = classification_report(
        y_test, y_pred,
        target_names=class_names[:len(np.unique(y_test))],
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "weighted_precision": weighted_precision,
        "macro_recall": macro_recall,
        "weighted_recall": weighted_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
    }

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "report_str": report_str,
    }


# ============================================================
# Visualization functions
# ============================================================

def plot_confusion_matrix(y_true, y_pred, class_names=None,
                          normalize=False, save_path=None):
    """Plot confusion matrix with optional normalization."""
    if class_names is None:
        class_names = SHORT_NAMES

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names[:cm.shape[0]],
                yticklabels=class_names[:cm.shape[0]], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_roc_curves(y_true, y_prob, class_names=None, save_path=None):
    """Plot one-vs-rest ROC curves for all classes."""
    if class_names is None:
        class_names = SHORT_NAMES

    n_classes = y_prob.shape[1]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        label = class_names[i] if i < len(class_names) else f"Class {i}"
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_pr_curves(y_true, y_prob, class_names=None, save_path=None):
    """Plot precision-recall curves for all classes."""
    if class_names is None:
        class_names = SHORT_NAMES

    n_classes = y_prob.shape[1]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin, y_prob[:, i])
        ap = average_precision_score(y_bin, y_prob[:, i])
        label = class_names[i] if i < len(class_names) else f"Class {i}"
        ax.plot(rec, prec, label=f"{label} (AP={ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_training_history(history, save_path=None):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(history.history["accuracy"], label="Train Accuracy")
    ax2.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training vs Validation Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_ecg_examples(X, y, class_names=None, fs=125, save_path=None):
    """Plot example ECG beats from each class."""
    if class_names is None:
        class_names = CLASS_NAMES

    if X.ndim == 3:
        X = X.squeeze(axis=-1)

    classes = np.unique(y)
    n_classes = len(classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 3))
    if n_classes == 1:
        axes = [axes]

    t = np.arange(X.shape[1]) / fs * 1000  # time in ms

    for ax, cls in zip(axes, classes):
        idx = np.where(y == cls)[0]
        if len(idx) > 0:
            sample = X[idx[0]]
            label = class_names[int(cls)] if int(cls) < len(class_names) else f"Class {cls}"
            ax.plot(t, sample, linewidth=0.8)
            ax.set_title(label)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude")
            ax.grid(alpha=0.3)

    plt.suptitle("ECG Beat Examples by Class", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_fft_comparison(X, y, freqs, class_names=None, save_path=None):
    """Plot FFT magnitude for example beats from each class."""
    if class_names is None:
        class_names = CLASS_NAMES

    if X.ndim == 3:
        X = X.squeeze(axis=-1)

    classes = np.unique(y)
    n_classes = len(classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 3))
    if n_classes == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        idx = np.where(y == cls)[0]
        if len(idx) > 0:
            fft_mag = np.abs(np.fft.rfft(X[idx[0]]))
            log_mag = np.log1p(fft_mag)
            label = class_names[int(cls)] if int(cls) < len(class_names) else f"Class {cls}"
            ax.plot(freqs[:len(log_mag)], log_mag, linewidth=0.8)
            ax.set_title(f"{label} — FFT")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Log Magnitude")
            ax.grid(alpha=0.3)

    plt.suptitle("FFT Magnitude Spectrum by Class", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================
# Bootstrap Confidence Intervals
# ============================================================

def bootstrap_metrics(y_true, y_pred, y_prob=None, n_bootstrap=1000,
                      confidence=0.95, seed=42):
    """
    Compute bootstrap confidence intervals for classification metrics.

    Returns dict of {metric_name: (point_estimate, ci_lower, ci_upper)}.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    boot_acc, boot_macro_f1, boot_weighted_f1 = [], [], []
    boot_macro_prec, boot_macro_rec = [], []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_pred[idx]

        # Skip if bootstrap sample has only one class
        if len(np.unique(y_t)) < 2:
            continue

        boot_acc.append(accuracy_score(y_t, y_p))
        boot_macro_f1.append(f1_score(y_t, y_p, average="macro", zero_division=0))
        boot_weighted_f1.append(f1_score(y_t, y_p, average="weighted", zero_division=0))
        boot_macro_prec.append(precision_score(y_t, y_p, average="macro", zero_division=0))
        boot_macro_rec.append(recall_score(y_t, y_p, average="macro", zero_division=0))

    alpha = (1 - confidence) / 2

    def ci(values):
        values = np.array(values)
        return (
            np.mean(values),
            np.percentile(values, 100 * alpha),
            np.percentile(values, 100 * (1 - alpha)),
        )

    results = {
        "Accuracy": ci(boot_acc),
        "Macro Precision": ci(boot_macro_prec),
        "Macro Recall": ci(boot_macro_rec),
        "Macro F1": ci(boot_macro_f1),
        "Weighted F1": ci(boot_weighted_f1),
    }

    return results


def plot_bootstrap_ci(boot_results, model_name="Model", save_path=None):
    """Plot bootstrap confidence intervals as a bar chart with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = list(boot_results.keys())
    means = [boot_results[m][0] for m in metrics]
    ci_low = [boot_results[m][0] - boot_results[m][1] for m in metrics]
    ci_high = [boot_results[m][2] - boot_results[m][0] for m in metrics]

    x = np.arange(len(metrics))
    ax.bar(x, means, yerr=[ci_low, ci_high], capsize=5,
           color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} — Bootstrap 95% Confidence Intervals")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for i, m in enumerate(metrics):
        point, lo, hi = boot_results[m]
        ax.text(i, means[i] + ci_high[i] + 0.01,
                f"{point:.4f}\n[{lo:.4f}, {hi:.4f}]",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================
# Saliency Map (Gradient-based interpretation)
# ============================================================

def compute_saliency(model, model_type, x_sample, fft_sample=None):
    """
    Compute input-gradient saliency for a single sample.

    Shows which time points most influence the prediction.
    """
    if model_type in ("fourier_hybrid", "fourier_vit_hybrid"):
        x_tensor = tf.Variable(tf.expand_dims(x_sample, 0), dtype=tf.float32)
        f_tensor = tf.Variable(tf.expand_dims(fft_sample, 0), dtype=tf.float32)
        with tf.GradientTape() as tape:
            pred = model([x_tensor, f_tensor], training=False)
            class_idx = tf.argmax(pred[0])
            loss = pred[0, class_idx]
        grads = tape.gradient(loss, x_tensor)
        saliency = tf.abs(grads).numpy().squeeze()
    else:
        x_tensor = tf.Variable(tf.expand_dims(x_sample, 0), dtype=tf.float32)
        with tf.GradientTape() as tape:
            pred = model(x_tensor, training=False)
            class_idx = tf.argmax(pred[0])
            loss = pred[0, class_idx]
        grads = tape.gradient(loss, x_tensor)
        saliency = tf.abs(grads).numpy().squeeze()

    return saliency


def plot_saliency(x_sample, saliency, true_label, pred_label,
                  class_names=None, fs=125, save_path=None):
    """Plot ECG signal overlaid with saliency heatmap."""
    if class_names is None:
        class_names = CLASS_NAMES

    if x_sample.ndim > 1:
        x_sample = x_sample.squeeze()
    if saliency.ndim > 1:
        saliency = saliency.squeeze()

    t = np.arange(len(x_sample)) / fs * 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    ax1.plot(t, x_sample, "k-", linewidth=0.8)
    ax1.set_ylabel("Amplitude")
    true_name = class_names[int(true_label)] if int(true_label) < len(class_names) else f"Class {true_label}"
    pred_name = class_names[int(pred_label)] if int(pred_label) < len(class_names) else f"Class {pred_label}"
    ax1.set_title(f"True: {true_name} | Predicted: {pred_name}")
    ax1.grid(alpha=0.3)

    # Saliency as heatmap
    norm_sal = saliency / (saliency.max() + 1e-8)
    ax2.fill_between(t, 0, norm_sal, alpha=0.6, color="red")
    ax2.plot(t, norm_sal, "r-", linewidth=0.5)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Saliency")
    ax2.set_title("Input Gradient Saliency")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================
# Model Comparison Table
# ============================================================

def build_comparison_table(all_results):
    """
    Build a comparison DataFrame from evaluation results of multiple models.

    Args:
        all_results: dict of {model_name: eval_result_dict}

    Returns:
        pd.DataFrame with one row per model and columns for each metric
    """
    rows = []
    for name, res in all_results.items():
        m = res["metrics"]
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Macro Precision": m["macro_precision"],
            "Weighted Precision": m["weighted_precision"],
            "Macro Recall": m["macro_recall"],
            "Weighted Recall": m["weighted_recall"],
            "Macro F1": m["macro_f1"],
            "Weighted F1": m["weighted_f1"],
            "ROC-AUC (Macro)": m["roc_auc_macro"],
            "ROC-AUC (Weighted)": m["roc_auc_weighted"],
        })

    df = pd.DataFrame(rows)
    return df


def plot_model_comparison(comparison_df, save_path=None):
    """Bar chart comparing key metrics across models."""
    metrics_to_plot = ["Accuracy", "Macro Precision", "Macro Recall",
                       "Macro F1", "ROC-AUC (Macro)"]
    plot_df = comparison_df.set_index("Model")[
        [m for m in metrics_to_plot if m in comparison_df.columns]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Ablation Study")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ============================================================
# Full evaluation pipeline
# ============================================================

def run_full_evaluation(trained_results, data, fft_data,
                        output_dir=None, class_names=None,
                        short_names=None):
    """
    Run complete evaluation for all trained models.

    Generates all plots, metrics, and comparison tables.
    Saves everything to output_dir.

    Args:
        trained_results: dict from run_ablation()
        data: dict from prepare_data()
        fft_data: dict from prepare_fft_inputs()
        output_dir: Path for saving results

    Returns:
        dict of all evaluation results and comparison table
    """
    if output_dir is None:
        output_dir = CONFIG["results_dir"]
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    metric_dir = output_dir / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    # Use class names from data dict if available, else from args or config
    if class_names is None:
        class_names = data.get("class_names", CLASS_NAMES)
    if short_names is None:
        short_names = data.get("short_names", SHORT_NAMES)
    num_classes = data.get("num_classes", CONFIG["num_classes"])

    # Plot ECG examples
    plot_ecg_examples(
        data["X_test"], data["y_test"],
        class_names=class_names,
        fs=CONFIG["sampling_rate"],
        save_path=fig_dir / "ecg_examples.png",
    )

    # Plot FFT examples
    freqs = np.fft.rfftfreq(data["X_test"].shape[1], d=1.0 / CONFIG["sampling_rate"])
    plot_fft_comparison(
        data["X_test"], data["y_test"], freqs,
        class_names=class_names,
        save_path=fig_dir / "fft_examples.png",
    )

    all_eval = {}
    model_name_map = {
        "baseline_cnn": "Baseline CNN",
        "cnn_transformer": "CNN + Transformer",
        "fourier_hybrid": "Fourier + CNN + Transformer",
        "fourier_vit_hybrid": "Fourier + CNN-ViT Hybrid",
    }

    for model_type, res in trained_results.items():
        display_name = model_name_map.get(model_type, model_type)
        print(f"\n{'='*50}")
        print(f"Evaluating: {display_name}")
        print(f"{'='*50}")

        model = res["model"]
        history = res["history"]

        # Evaluate
        eval_result = evaluate_model(
            model, model_type, data, fft_data,
            class_names=class_names,
        )
        all_eval[display_name] = eval_result

        prefix = model_type.replace(" ", "_")

        # Classification report
        print(f"\n{eval_result['report_str']}")
        with open(metric_dir / f"{prefix}_classification_report.txt", "w") as f:
            f.write(f"Model: {display_name}\n\n")
            f.write(eval_result["report_str"])

        # Confusion matrices
        plot_confusion_matrix(
            eval_result["y_test"], eval_result["y_pred"],
            class_names=short_names,
            save_path=fig_dir / f"{prefix}_confusion_matrix.png",
        )
        plot_confusion_matrix(
            eval_result["y_test"], eval_result["y_pred"],
            class_names=short_names,
            normalize=True,
            save_path=fig_dir / f"{prefix}_confusion_matrix_normalized.png",
        )

        # ROC curves
        plot_roc_curves(
            eval_result["y_test"], eval_result["y_prob"],
            class_names=short_names,
            save_path=fig_dir / f"{prefix}_roc_curves.png",
        )

        # PR curves
        plot_pr_curves(
            eval_result["y_test"], eval_result["y_prob"],
            class_names=short_names,
            save_path=fig_dir / f"{prefix}_pr_curves.png",
        )

        # Training history
        plot_training_history(
            history,
            save_path=fig_dir / f"{prefix}_training_history.png",
        )

        # Bootstrap CI
        boot = bootstrap_metrics(
            eval_result["y_test"], eval_result["y_pred"],
            eval_result["y_prob"],
        )
        plot_bootstrap_ci(
            boot, model_name=display_name,
            save_path=fig_dir / f"{prefix}_bootstrap_ci.png",
        )

        # Print bootstrap results
        print(f"\nBootstrap 95% CI:")
        for metric, (mean, lo, hi) in boot.items():
            print(f"  {metric}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

        eval_result["bootstrap"] = boot

    # Saliency for the best model (prefer fourier_vit_hybrid > fourier_hybrid)
    best_fourier = None
    for mt in ("fourier_vit_hybrid", "fourier_hybrid"):
        if mt in trained_results:
            best_fourier = mt
            break

    if best_fourier is not None:
        print(f"\nGenerating saliency maps ({best_fourier})...")
        model = trained_results[best_fourier]["model"]
        for cls in range(num_classes):
            idx = np.where(data["y_test"] == cls)[0]
            if len(idx) > 0:
                i = idx[0]
                x_s = data["X_test"][i]
                f_s = fft_data["fft_test"][i]
                sal = compute_saliency(model, best_fourier, x_s, f_s)
                y_prob = model.predict(
                    [x_s[np.newaxis], f_s[np.newaxis]], verbose=0
                )
                pred = np.argmax(y_prob)
                plot_saliency(
                    x_s, sal, cls, pred,
                    class_names=class_names,
                    save_path=fig_dir / f"saliency_class_{cls}.png",
                )

    # Comparison table
    comparison_df = build_comparison_table(all_eval)
    comparison_df.to_csv(metric_dir / "model_comparison.csv", index=False)
    print(f"\n{'='*50}")
    print("Model Comparison:")
    print(f"{'='*50}")
    print(comparison_df.to_string(index=False))

    plot_model_comparison(
        comparison_df,
        save_path=fig_dir / "model_comparison.png",
    )

    # Save all metrics
    metrics_rows = []
    for name, res in all_eval.items():
        row = {"Model": name}
        row.update(res["metrics"])
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(metric_dir / "metrics.csv", index=False)

    return all_eval, comparison_df
