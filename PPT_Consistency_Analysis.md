# PPT vs Code/Results Consistency Analysis

**Project**: ECG Arrhythmia Detection Using Fourier Transform and Hybrid CNN-Transformer Model  
**Group 4 — Fast & Fouriers**  
**Date**: September 17, 2026

---

## Summary

The PPT contains **9 significant discrepancies** with the actual codebase and training results. The most critical issues are: the PPT presents **binary classification results** (~97% accuracy) while the code implements **5-class AAMI classification** (~79% accuracy), the model is incorrectly called a "Vision Transformer (ViT)" when it is a standard Transformer Encoder, and the workflow diagram describes features that don't exist in the code.

---

## Slide-by-Slide Analysis

### Slide 1 — Title Slide
**PPT Claim**: "ECG Arrhythmia Detection using Fourier Transform and Hybrid **CNN-Vision Transformer** Model"

**Discrepancy** ⚠️ **CRITICAL**: The model is NOT a Vision Transformer (ViT). A ViT splits input into patches, applies linear projection of flattened patches, and uses a [CLS] token — none of which our model does. Our architecture uses a standard **Transformer Encoder** applied to CNN-extracted feature sequences. The correct title should be "**CNN-Transformer**" (which is what the code and README use).

**Status in code**: ✅ Fixed — README and code consistently say "CNN-Transformer", never "ViT".

---

### Slide 3 — Dataset & Research Details
**PPT Claims**:
- "Each heartbeat sample contains **188** numerical signal points"
- "Derived from MIT-BIH Arrhythmia Database **and PTB Diagnostic ECG Database**"

**Discrepancies**:

1. ⚠️ **MINOR**: The Kaggle dataset CSV has **187 columns** (186 signal features + 1 label column), so each beat has **186 signal points**, not 188. Our code built from raw MIT-BIH files also produces 186-sample beats.

2. ⚠️ **MODERATE**: The PPT mentions PTB Diagnostic ECG Database. Our project uses **only the MIT-BIH Arrhythmia Database** — the PTB database is for a different task (myocardial infarction detection, binary Normal/Abnormal) and is NOT used in our code.

**Status in code**: ✅ Correct — code and README correctly reference only MIT-BIH with 186-sample beats.

---

### Slide 5 — Why Fourier Transform?
**PPT Claims**: Shows FFT plots with frequency axis extending to **100 Hz**.

**Discrepancy** ⚠️ **MINOR**: Our data is sampled at **125 Hz**, so the Nyquist frequency (maximum representable) is **62.5 Hz**. FFT plots showing 0–100 Hz appear to come from a different analysis, possibly the raw 360 Hz MIT-BIH recordings before resampling. The code correctly computes `rfftfreq` with fs=125 Hz, producing 94 frequency bins up to 62.5 Hz.

**Status in code**: ✅ Correct — `fft_features.py` uses the correct 125 Hz sampling rate.

---

### Slide 6 — Time Domain / Frequency Domain
**PPT Claims**: Shows "Correct ECG Signal (MLII)" with **2000+ samples** on the x-axis, and "Normal/Abnormal Heartbeat" plots with **200+ time points**.

**Discrepancies**:

1. ⚠️ **MINOR**: The raw MLII signal plot (2000+ samples) is from the full MIT-BIH recording at 360 Hz, not from the preprocessed 186-sample beats that the model actually processes.

2. ⚠️ **MINOR**: The "Normal/Abnormal Heartbeat" plots have 200+ time points, which doesn't match our 186-sample beats. These appear to be from a different preprocessing pipeline.

**Status in code**: ✅ Code correctly uses 186-sample beats. The generated visualization plots (`ecg_examples.png`) show correct dimensions.

---

### Slide 7 — Why Its Novel?
**PPT Claims**:
- "The **ViT/Transformer** introduces attention, capturing global dependencies"
- Shows a **Vision Transformer architecture diagram** with "Patch + Position Embedding", "Linear projection of Flattened Patches", and image patches being split from a bird photo

**Discrepancy** ⚠️ **CRITICAL**: 

1. The model is called "ViT" again — it is NOT a Vision Transformer. See Slide 1 analysis.

2. The ViT diagram shown is the **original Dosovitskiy et al. (2020) ViT for image classification**. Our model does not split input into patches, does not use linear projection of flattened patches, and does not process 2D images. It applies a Transformer Encoder to 1D CNN feature sequences. The diagram is misleading and does not represent our architecture.

**Status in code**: ✅ Fixed — README contains an accurate ASCII architecture diagram showing the dual-branch CNN → Transformer Encoder → Attention Pooling pipeline.

---

### Slide 8 — Comparative Study of Different Models
**PPT Claims**:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| CNN | 0.9773 | 0.9724 | 0.9152 | 0.9430 | 0.9928 |
| Hybrid CNN-Transformer | 0.9758 | 0.9765 | 0.8810 | 0.9263 | 0.9894 |

- Confusion matrices show **Normal (0) vs Abnormal (1)** — two classes only
- Claims "Accuracy: 97.7" for CNN and "Accuracy: 97.6" for Hybrid

**Discrepancies** ⚠️ **CRITICAL (MULTIPLE)**:

1. **Binary vs 5-class**: The PPT shows **binary classification** (Normal vs Abnormal), but our code implements **5-class AAMI classification** (N, S, V, F, Q). These are fundamentally different tasks. The user explicitly warned: *"DO NOT claim multiclass classification if the dataset/code is binary."*

2. **Inflated accuracy numbers**: The PPT claims ~97.6% accuracy. Our actual 5-class test results are:
   - Baseline CNN: **74.07%**
   - CNN + Transformer: **76.15%**
   - Fourier Hybrid: **78.77%**
   
   The ~97% numbers come from the much easier binary classification task on the Kaggle pre-processed dataset.

3. **Only 2 models compared**: The PPT compares only CNN vs Hybrid CNN-Transformer. Our ablation study compares **3 models** (Baseline CNN → CNN+Transformer → Fourier Hybrid), which is a more rigorous experimental design.

4. **No Fourier model in comparison**: The PPT's "Hybrid CNN-Transformer" does NOT include FFT features. The ablation benefit of Fourier Transform is not demonstrated in the PPT results.

5. **Threshold-tuned confusion matrices**: The PPT confusion matrices show "Threshold = 0.9000..." and "Threshold = 0.8000...", suggesting **threshold tuning on test data** — which constitutes data leakage for threshold selection. Our code does NOT do threshold tuning.

**Status in code**: ✅ Fixed — Code implements honest 5-class classification with 3-model ablation study, no threshold tuning, and actual results reported.

---

### Slide 9 — Overall Workflow
**PPT Claims**: Shows a complex pipeline with:
- "Decision Module" routing Normal/Mild PVCs vs Severe arrhythmias
- "Arrhythmia Classification (5-15 Types)" including VT, VFib, SVT, AV Block, Torsades
- "Secondary Pipeline (Sub-Condition Model)"
- "Feature Analysis: P, QRS, T, ST-segment, RR-interval parameters"
- "Final Output Block" with Severity Level, PDF Clinical Report
- "Arrhythmia Prediction Module" as a separate branch

**Discrepancy** ⚠️ **CRITICAL**: **None of these exist in the code.** The actual implementation is a straightforward single-stage 5-class classifier:
- ECG beat → Preprocessing → FFT → Dual-branch CNN → Transformer → Softmax(5)

There is:
- ❌ No decision module or severity routing
- ❌ No 5-15 type sub-classification (VT, VFib, SVT, etc.)
- ❌ No secondary pipeline or sub-condition model
- ❌ No explicit P/QRS/T/ST-segment/RR-interval feature extraction (the CNN learns features implicitly)
- ❌ No PDF clinical report generation
- ❌ No severity level output

This diagram represents an **aspirational system design**, not the actual implementation.

**Status in code**: ✅ The README's architecture diagram accurately reflects what the code actually does.

---

### Slide 10 — Statistical Validation & Results
**PPT Claims**:
- Bootstrap Results: Accuracy 0.9758, Precision 0.9765, Recall 0.8810, F1 0.9263
- "High Model Performance: Achieves strong accuracy (~97.6%) and precision (~97.7%)"
- "Balanced Classification: F1 score (~92.6%)"
- "Minimal Overfitting: Close alignment between training and validation accuracy (~97-98%)"

**Discrepancies** ⚠️ **CRITICAL**:

1. **All metrics are from binary classification**, not the 5-class model. Our actual Fourier Hybrid bootstrap results would show accuracy around 78.8%, not 97.6%.

2. **"Balanced Classification" is misleading**: Even in the binary setting, recall (88.1%) is notably lower than precision (97.7%), which is not balanced. In our 5-class setting, the F1 scores for minority classes (S: 0.09, F: 0.03, Q: 0.00) show the model struggles significantly with rare classes — this is an honest and expected challenge.

3. **Validation/training accuracy comparison**: The ~97-98% validation accuracy claim is actually consistent with our code's validation accuracy (which runs ~98.9%), but the PPT doesn't mention that **test accuracy drops to ~79%** due to the patient-level DS1/DS2 split. This omission is misleading — the generalization gap is a critical finding.

**Status in code**: ✅ Fixed — README reports both validation (~98.9%) and test (~78.8%) accuracy, and explicitly discusses the patient-level generalization gap.

---

## Discrepancy Summary Table

| # | Slide | Issue | Severity | Fixed in Code? |
|---|-------|-------|----------|----------------|
| 1 | 1, 7 | Model called "Vision Transformer (ViT)" — it's a standard Transformer Encoder | **CRITICAL** | ✅ Yes |
| 2 | 3 | Beat length "188" — actual is 186 features | Minor | ✅ Yes |
| 3 | 3 | Mentions PTB database — not used in project | Moderate | ✅ Yes |
| 4 | 5 | FFT plots show 0–100 Hz — max is 62.5 Hz at 125 Hz sampling | Minor | ✅ Yes |
| 5 | 7 | ViT architecture diagram shown — doesn't match our model | **CRITICAL** | ✅ Yes |
| 6 | 8 | Results are **binary** classification (~97%) — code does **5-class** (~79%) | **CRITICAL** | ✅ Yes |
| 7 | 8 | Threshold tuning on test data (data leakage) | **CRITICAL** | ✅ Yes |
| 8 | 9 | Workflow shows features not in code (decision module, sub-classification, PDF reports) | **CRITICAL** | ✅ Yes |
| 9 | 10 | All reported metrics are from binary task, not 5-class | **CRITICAL** | ✅ Yes |

---

## What the Code Actually Delivers (Honest Results)

### Architecture
- **Dual-branch CNN + Transformer Encoder** (NOT ViT)
- Time-domain branch: 3 Residual Conv1D blocks → Dense projection
- Frequency-domain branch: FFT → log-magnitude → 3 Residual Conv1D blocks → Dense projection
- Fusion: Concatenate → Learnable Positional Encoding → 2× Pre-Norm Transformer Encoder → Attention-Weighted Pooling → Classifier

### Task
- **5-class AAMI heartbeat classification** (N, S, V, F, Q)
- Patient-level DS1/DS2 split (AAMI-recommended, no data leakage)

### Actual Test Performance (DS2 split)

| Model | Accuracy | Weighted F1 | Macro F1 | ROC-AUC |
|-------|----------|-------------|----------|---------|
| Baseline CNN | 74.07% | 0.792 | 0.337 | 0.858 |
| CNN + Transformer | 76.15% | 0.803 | 0.294 | 0.797 |
| **Fourier Hybrid** | **78.77%** | **0.824** | **0.317** | **0.819** |

### Why Test Accuracy Is ~79% (Not ~97%)
1. **5-class is harder than binary**: 5-way classification with extreme class imbalance (Normal 89%, Fusion 0.7%) is fundamentally harder than Normal-vs-Abnormal.
2. **Patient-level split**: DS1 and DS2 contain entirely different patients. Morphological differences between patients create a real generalization challenge. Validation accuracy (~98.9%) uses the same patients as training — test accuracy is the honest number.
3. **Built from raw files**: Our `build_dataset.py` processes raw MIT-BIH files, which may differ slightly from the Kaggle pre-processed version.

---

## Recommendations for PPT Update

1. **Change title** from "CNN-Vision Transformer" to "CNN-Transformer"
2. **Remove ViT diagram** from Slide 7; replace with the actual architecture diagram from README
3. **Replace Slide 8** with the 3-model ablation comparison table using actual 5-class results
4. **Replace Slide 9** workflow with the actual pipeline (preprocessing → FFT → dual-branch CNN → Transformer → 5-class output)
5. **Replace Slide 10** bootstrap metrics with actual 5-class results; add discussion of generalization gap
6. **Fix dataset description** (186 features, MIT-BIH only, no PTB)
7. **Add honest discussion** of the patient-level generalization gap — this is a known challenge in the field and shows scientific maturity
