#!/usr/bin/env python3
"""
Build Kaggle-compatible MIT-BIH dataset from raw PhysioNet CSV files.

Follows the methodology of:
  Kachuee et al., "ECG Heartbeat Classification: A Deep Transferable
  Representation," IEEE ICHI 2018.

Steps:
  1. Load raw ECG signals (360 Hz) and beat annotations
  2. Segment beats around R-peak annotations (±window)
  3. Map PhysioNet beat labels to AAMI 5-class standard
  4. Resample each beat segment from 360 Hz to 125 Hz
  5. Zero-pad/truncate to 186 samples
  6. Split into train/test by record (DS1/DS2 per AAMI recommendation)
  7. Save as mitbih_train.csv and mitbih_test.csv

Usage:
    python scripts/build_dataset.py --raw_dir "path/to/Training MIT dataset]" --out_dir data/
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample


# ─── AAMI beat-type mapping ─────────────────────────────────────────
# PhysioNet annotation symbol → AAMI class
# Reference: ANSI/AAMI EC57:1998, Table 3
AAMI_MAP = {
    # N (Normal): Normal beat, Left/Right bundle branch block, Atrial/Nodal escape
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    # S (Supraventricular ectopic): Atrial premature, Aberrated atrial premature,
    #   Nodal premature, Supraventricular premature
    "A": 1, "a": 1, "J": 1, "S": 1,
    # V (Ventricular ectopic): Premature ventricular, Ventricular escape
    "V": 2, "E": 2,
    # F (Fusion): Fusion of ventricular and normal
    "F": 3,
    # Q (Unknown/Paced): Paced, Fusion of paced and normal, Unclassifiable
    "/": 4, "f": 4, "Q": 4,
}

# Non-beat annotations to skip
SKIP_TYPES = {"+", "~", "|", "!", '"', "x", "[", "]", "#"}

# ─── DS1/DS2 split (AAMI recommendation) ─────────────────────────────
# DS1 = training, DS2 = testing
# Standard split used by de Chazal et al. (2004), Kachuee et al. (2018)
DS1_RECORDS = {
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
    122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
    223, 230,
}
DS2_RECORDS = {
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
    210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
    233, 234,
}

# Constants
FS_ORIGINAL = 360  # Original sampling rate (Hz)
FS_TARGET = 125    # Target sampling rate (Hz)
BEAT_LENGTH = 186  # Samples per beat at target fs (excluding label column)
WINDOW_LEFT = 90   # Samples before R-peak at 360 Hz (~250ms)
WINDOW_RIGHT = 180  # Samples after R-peak at 360 Hz (~500ms)


def load_signal(csv_path):
    """Load ECG signal from raw CSV. Returns the first available lead."""
    samples = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Use MLII if available, otherwise first signal column
        lead_idx = 1  # Default to first signal column
        for i, col in enumerate(header):
            if "MLII" in col.upper():
                lead_idx = i
                break
        for row in reader:
            samples.append(int(row[lead_idx]))
    return np.array(samples, dtype=np.float64)


def load_annotations(ann_path):
    """Load beat annotations. Returns list of (sample_idx, beat_type)."""
    beats = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Time"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                sample_idx = int(parts[1])
                beat_type = parts[2]
            except (ValueError, IndexError):
                continue
            if beat_type not in SKIP_TYPES:
                beats.append((sample_idx, beat_type))
    return beats


def segment_beat(signal, r_peak, left=WINDOW_LEFT, right=WINDOW_RIGHT):
    """Extract a beat segment around the R-peak with zero-padding if needed."""
    start = r_peak - left
    end = r_peak + right

    segment = np.zeros(left + right)

    # Handle boundaries
    sig_start = max(0, start)
    sig_end = min(len(signal), end)
    seg_start = max(0, -start)
    seg_end = seg_start + (sig_end - sig_start)

    segment[seg_start:seg_end] = signal[sig_start:sig_end]
    return segment


def process_record(record_num, raw_dir):
    """Process one MIT-BIH record into beat segments with AAMI labels."""
    csv_path = os.path.join(raw_dir, f"{record_num}.csv")
    ann_path = os.path.join(raw_dir, f"{record_num}annotations.txt")

    if not os.path.exists(csv_path) or not os.path.exists(ann_path):
        print(f"  Skipping record {record_num}: files not found")
        return [], []

    signal = load_signal(csv_path)
    annotations = load_annotations(ann_path)

    beats = []
    labels = []
    skipped = 0

    for sample_idx, beat_type in annotations:
        if beat_type not in AAMI_MAP:
            skipped += 1
            continue

        aami_label = AAMI_MAP[beat_type]

        # Extract segment at original sampling rate
        segment = segment_beat(signal, sample_idx)

        # Resample from 360 Hz to 125 Hz
        n_target = int(len(segment) * FS_TARGET / FS_ORIGINAL)
        resampled = resample(segment, n_target)

        # Pad or truncate to BEAT_LENGTH
        beat = np.zeros(BEAT_LENGTH)
        copy_len = min(len(resampled), BEAT_LENGTH)
        beat[:copy_len] = resampled[:copy_len]

        # Normalize to [0, 1] range (per-beat min-max)
        bmin, bmax = beat.min(), beat.max()
        if bmax - bmin > 1e-8:
            beat = (beat - bmin) / (bmax - bmin)

        beats.append(beat)
        labels.append(aami_label)

    if skipped > 0:
        print(f"  Record {record_num}: {len(beats)} beats extracted, {skipped} non-AAMI skipped")
    else:
        print(f"  Record {record_num}: {len(beats)} beats extracted")

    return beats, labels


def build_dataset(raw_dir, out_dir):
    """Build the full dataset from raw MIT-BIH records."""
    raw_dir = str(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find available records
    available = set()
    for f in os.listdir(raw_dir):
        if f.endswith(".csv") and f[:-4].isdigit():
            available.add(int(f[:-4]))

    print(f"Found {len(available)} records: {sorted(available)}")

    train_records = sorted(DS1_RECORDS & available)
    test_records = sorted(DS2_RECORDS & available)

    # Records not in either split — add to training
    extra = available - DS1_RECORDS - DS2_RECORDS
    if extra:
        print(f"Records not in DS1/DS2 standard split: {sorted(extra)} — adding to training")
        train_records = sorted(set(train_records) | extra)

    print(f"\nTraining records ({len(train_records)}): {train_records}")
    print(f"Test records ({len(test_records)}): {test_records}")

    # Process training records
    print("\n--- Processing training set ---")
    train_beats, train_labels = [], []
    for rec in train_records:
        b, l = process_record(rec, raw_dir)
        train_beats.extend(b)
        train_labels.extend(l)

    # Process test records
    print("\n--- Processing test set ---")
    test_beats, test_labels = [], []
    for rec in test_records:
        b, l = process_record(rec, raw_dir)
        test_beats.extend(b)
        test_labels.extend(l)

    # Convert to arrays
    X_train = np.array(train_beats)
    y_train = np.array(train_labels)
    X_test = np.array(test_beats)
    y_test = np.array(test_labels)

    print(f"\n--- Dataset Summary ---")
    print(f"Training: {X_train.shape[0]} beats, shape {X_train.shape}")
    print(f"Test:     {X_test.shape[0]} beats, shape {X_test.shape}")

    class_names = ["N (Normal)", "S (Supraventricular)", "V (Ventricular)",
                   "F (Fusion)", "Q (Unknown/Paced)"]
    for split_name, labels in [("Train", y_train), ("Test", y_test)]:
        print(f"\n{split_name} class distribution:")
        for c in range(5):
            count = np.sum(labels == c)
            pct = 100 * count / len(labels)
            print(f"  {c} {class_names[c]}: {count:>6d} ({pct:.1f}%)")

    # Save as CSV (same format as Kaggle: 186 features + 1 label column = 187 cols)
    train_data = np.column_stack([X_train, y_train])
    test_data = np.column_stack([X_test, y_test])

    train_path = out_dir / "mitbih_train.csv"
    test_path = out_dir / "mitbih_test.csv"

    np.savetxt(train_path, train_data, delimiter=",", fmt="%.6f")
    np.savetxt(test_path, test_data, delimiter=",", fmt="%.6f")

    print(f"\nSaved: {train_path} ({train_data.shape})")
    print(f"Saved: {test_path} ({test_data.shape})")

    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(description="Build MIT-BIH dataset from raw PhysioNet files")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Directory containing raw MIT-BIH CSV and annotation files")
    parser.add_argument("--out_dir", type=str, default="data",
                        help="Output directory for mitbih_train.csv and mitbih_test.csv")
    args = parser.parse_args()

    build_dataset(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
