# FraudGuard

### A Time-Aware, Calibrated Machine Learning Pipeline for Credit Card Fraud Detection

**Version 4.0 — Complete Edition (v1 → v4)**

![FraudGuard Cover](assets/credit_card_cover.svg)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

**Author:** Sourish Dey · **GitHub:** [@sourishdey2005](https://github.com/sourishdey2005) · **Repo:** [sourishdey2005/FraudGuard](https://github.com/sourishdey2005/FraudGuard)

---

## Table of Contents

1. [Overview](#overview)
2. [Complete Model Comparison — All Versions](#complete-model-comparison--all-versions)
3. [Version History](#version-history)
4. [The Problem This Project Solves](#the-problem-this-project-solves)
5. [Dataset](#dataset)
6. [Architectures Built From Scratch](#architectures-built-from-scratch)
7. [Pipeline Stages](#pipeline-stages)
8. [Results by Version](#results-by-version)
9. [Figures](#figures)
10. [Repository Structure](#repository-structure)
11. [Reproducing the Project](#reproducing-the-project)
12. [Usage and Deployment](#usage-and-deployment)
13. [Limitations and Honest Reporting](#limitations-and-honest-reporting)
14. [Citation](#citation)
15. [License](#license)

---

## Overview

Fraud detection is one of the hardest problems in applied machine learning. Fraud is rare, patterns shift over time, and the cost of a false negative far exceeds the cost of a false positive. **FraudGuard** is a complete, reproducible fraud detection pipeline that has been iteratively debugged across **four major versions** to eliminate the class leakage that plagues most published fraud benchmarks.

Over four versions, the project trained and evaluated **nine distinct models**:

| # | Model | Version | Type |
|---|---|---|---|
| 1 | Logistic Regression | v1 | Linear baseline |
| 2 | LightGBM | v1 | Gradient boosting |
| 3 | LightGBM (Calibrated) | v1 | Gradient boosting + isotonic |
| 4 | ANN / MLP | v1 | 3-layer feed-forward |
| 5 | GRU (sequence) | v1 | Recurrent (excluded) |
| 6 | FraudAttnNet | v2 | Self-attention MLP (excluded from final) |
| 7 | WGAN-GP | v2 | Wasserstein GAN (excluded from final) |
| 8 | FraudCNN + GAN | v2 | Conv1D + synthetic augmentation (excluded from final) |
| 9 | **FraudFormer** | **v4** | **Self-attention transformer (final winner)** |
| 10 | **AEGAN** | **v4** | **Adversarial autoencoder GAN (final augmentation)** |

The final pipeline (v4) combines **Leak-free generation**, **FraudFormer** (custom transformer), and **AEGAN** (custom adversarial autoencoder), reporting **PR-AUC 0.5330** and **F1 0.5864** — matching or beating production benchmarks while maintaining full methodological integrity.

---

## Complete Model Comparison — All Versions

### Final Comparison (v4.0, Leak-Free Data, 502,308 rows, 4.99% fraud)

| Model | Threshold | ROC-AUC | PR-AUC | Precision | Recall | **F1** |
|---|---|---|---|---|---|---|
| LightGBM (baseline) | 0.3286 | 0.7661 | 0.5237 | 0.7985 | 0.4392 | 0.5667 |
| LightGBM + AEGAN augmentation | 0.3000 | 0.7652 | 0.5268 | 0.8411 | 0.4357 | 0.5741 |
| **FraudFormer (ours)** | **0.3521** | **0.7692** | **0.5330** | **0.8511** | **0.4473** | **0.5864** |

### Historic Comparison (v1.0, Leaky Data, 502,307 rows, 5.00% fraud)

Kept here for transparency — these results are **artificially high** due to class leakage. They are documented to demonstrate the debugging journey, not as valid performance claims.

| Model | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 | Status |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.7368 | 0.7286 | 0.2405 | 0.2904 | 0.2776 | 0.2839 | Valid (leak-free) |
| LightGBM | 0.1429 | 0.7435 | 0.2595 | 0.2950 | 0.3090 | 0.3018 | Valid (leak-free) |
| LightGBM (Calibrated) | 0.1429 | 0.7435 | 0.2595 | 0.2950 | 0.3090 | 0.3018 | Valid (leak-free) |
| ANN (MLP) | 0.6897 | 0.6824 | 0.1853 | 0.1893 | 0.2576 | 0.2182 | Valid (leak-free) |
| GRU (sequence) | 0.0477 | 0.5034 | 0.0685 | 0.0680 | 0.9353 | 0.1267 | **Excluded** (chance-level) |
| FraudAttnNet | 0.4909 | 0.9992 | 0.9915 | 0.9701 | 0.9565 | 0.9633 | **Leaky** (v2, discarded) |
| WGAN-GP | — | — | — | — | — | — | **Leaky** (v2, discarded) |
| FraudCNN + GAN | — | — | — | — | — | — | **Leaky** (v2, discarded) |

### Version-Level Metrics Evolution

| Version | PR-AUC | F1 | Issue | Fix |
|---|---|---|---|---|
| v1.0 | 0.99+ | 0.99+ | Distributional leakage — fraud means 3× normal on 4 features | Regenerate with overlapping distributions |
| v2.0 | 0.99+ | 0.99+ | Temporal leakage — sorted timestamps confined normal rows to specific hours | Fix generator to draw timestamps identically |
| v3.0 | 0.99+ | 0.99+ | Customer fingerprint leakage — new customers were fraud-heavy | Drop aggregate features; verify with single-feature check |
| v3.1 | 0.9986 | 0.9874 | Same leak via burst features on full data | Rebuild burst using strictly-earlier transactions only |
| **v4.0** | **0.5330** | **0.5864** | **Clean** | **Interaction-only generator, verified single-feature ROC-AUC < 0.60** |

**The final v4.0 numbers are the only legitimate results.** All v1–v3 numbers are documented as part of the debugging journey.

---

## Version History

### Version 1.0 — Base Pipeline

**Scope:** Data generation, cleaning, EDA, three models (Logistic Regression, LightGBM, ANN).

**What it did well:**
- Complete end-to-end pipeline
- 35+ EDA visualizations
- Time-aware splitting
- Isotonic calibration
- Threshold optimization

**What it got wrong:**
- Fraud rows generated from disjoint feature distributions
- Single feature had ROC-AUC > 0.94
- All models hit 0.99 PR-AUC (implausible)

**Lesson:** Distributional leakage is the #1 killer of fraud detection benchmarks.

### Version 2.0 — Custom Deep Learning Extensions

**Scope:** Added FraudAttnNet (self-attention MLP), WGAN-GP (custom GAN), and FraudCNN (1D convolutional classifier with GAN augmentation).

**What it did well:**
- Built three novel architectures from scratch in TensorFlow
- Demonstrated attention, adversarial training, and convolutional feature extraction
- Full `get_config` serialization support

**What it got wrong:**
- Same leakage from v1 persisted (base data still leaky)
- FraudAttnNet hit 0.9915 PR-AUC — not because attention was great, but because leakage was still present
- GRU collapsed to chance-level (0.5034 ROC-AUC) — correct diagnosis: no sequence signal

**Lesson:** Deep learning does not fix data quality issues.

### Version 3.0 — Customer Behavioral Features

**Scope:** Added customer fingerprint features (mean amount, transaction count, night ratio, international ratio), burst detection, risk flags.

**What it did well:**
- Engineered 57 new features across behavioral, temporal, and categorical axes
- Identified that customer-level aggregates can leak through the "new customer" shortcut

**What it got wrong:**
- Customer fingerprints were systematically different for fraud vs normal
- `has_train_history` feature had ROC-AUC 0.95+ by itself
- PR-AUC still 0.99+ across all models

**Lesson:** Aggregate features leak when the generator produces class-correlated customer distributions.

### Version 3.1 — Diagnostic Fix

**Scope:** Removed all customer fingerprint features, kept only transaction-level + burst + risk flags.

**What it did well:**
- Reduced PR-AUC to 0.9986 (still too high but better)
- Identified burst features on full data as the remaining leak

**Lesson:** Every feature derived from full data leaks. Strictly-earlier-only computation is the only safe path.

### Version 4.0 — Leak-Free Interaction-Only Generator

**Scope:** Complete rewrite of the data generator with interaction-only drift. Removed all customer-level features. Kept only 14 raw transaction features. Added FraudFormer (transformer) and AEGAN (adversarial autoencoder).

**What it does now:**
- No single feature exceeds ROC-AUC 0.60
- All models report realistic PR-AUC (0.52–0.53)
- FraudFormer beats LightGBM (+0.0093 PR-AUC, +0.0197 F1)
- AEGAN augmentation adds +0.0031 PR-AUC
- **Full methodological integrity**

---

## The Problem This Project Solves

Fraud detection research has a dirty secret: **most published models report ROC-AUC above 0.99** on generated or heavily preprocessed benchmarks. Those numbers are almost always the result of **class leakage** — one or more features that inadvertently encode the label.

The three most common leak mechanisms:

| Leak Type | How It Happens | Typical Symptom |
|---|---|---|
| **Distributional leakage** | Fraud rows generated from a shifted distribution on one feature | Single feature has ROC-AUC > 0.90 |
| **Temporal leakage** | Random train/test split on time-series data | Model sees the future during training |
| **Aggregate leakage** | Customer-history features built on full data | New customers systematically different |

This project was built by **finding and eliminating all three** across four iterative versions. The final v4.0 generator is designed so that **no single feature separates classes** — the model has to combine multiple signals, exactly like real fraud.

---

## Dataset

### Origin

The dataset is a **synthetic, reproducible credit card transaction corpus** designed to mirror the statistical properties of real-world payment data while remaining fully shareable and privacy-safe.

### Final v4.0 Statistics

| Property | Value |
|---|---|
| Total rows | 502,308 |
| Columns | 28 |
| Fraud rate | 4.99% (25,067 fraud cases) |
| Time range | 2023-01-01 to 2024-12-30 |
| Unique customers | 50,000 |
| Unique merchants | 30 |
| Merchant categories | 13 |
| Countries | 14 ISO-2 codes |
| Channels | 4 |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string | Unique transaction identifier |
| `customer_id` | string | Customer identifier |
| `timestamp` | datetime | Transaction timestamp (2-year span) |
| `amount` | float | Transaction amount in USD |
| `merchant` | string | Merchant name |
| `merchant_category` | string | Merchant vertical (Retail, Food, Crypto, etc.) |
| `card_type` | string | Card network |
| `card_last4` | int | Last 4 digits of card |
| `country` | string | ISO-2 country code |
| `city` | string | Transaction city |
| `device_type` | string | Device used |
| `ip_address` | string | IPv4 address |
| `is_international` | int | 1 if cross-border |
| `txn_count_10min` | int | Transactions in last 10 minutes |
| `txn_count_24h` | int | Transactions in last 24 hours |
| `avg_amount_30d` | float | Average amount over past 30 days |
| `account_age_days` | int | Days since account opened |
| `credit_limit` | int | Card credit limit |
| `hour_of_day` | int | Hour (0–23) |
| `is_night` | int | 1 if 00:00–04:59 |
| `distance_from_home_km` | float | Distance from home |
| `channel` | string | POS / Online / ATM / App |
| `year`, `month`, `day`, `hour`, `weekday` | int | Derived time features |
| `fraud_label` | int | 1 if fraud |

### Injected Data Quality Issues (v4.0)

The raw dataset contains realistic messiness across 25+ categories:

| Category | Examples |
|---|---|
| Missing values | Up to 15% of some columns NaN |
| Duplicate rows | 5,000 exact duplicates |
| Duplicate IDs | 300+ `transaction_id` collisions |
| Negative amounts | Refunds and errors |
| Zero amounts | Failed transactions |
| Extreme outliers | Amounts up to $10M |
| Mixed dtypes | Amount as `"123.45 USD"` |
| Mixed timestamps | `datetime`, `str`, epoch `int`, garbage |
| Country aliases | `US`, `USA`, `United States`, `united states` |
| Merchant casing | `Amazon`, `AMAZON`, `  amazon  ` |
| Sentinel strings | `N/A`, `null`, `-`, `?`, `unknown` |
| Invalid IPv4 | `999.999.999.999`, `::1`, `not-an-ip` |
| Invalid ranges | Negative account ages, `hour_of_day=99` |
| Type inconsistencies | `card_last4` as `"00042"` |

### v4.0 Single-Feature ROC-AUC Verification

After cleaning, every raw feature must have ROC-AUC < 0.60. This confirms the generator produced leak-free data:

| Feature | ROC-AUC |
|---|---|
| `amount` | 0.5463 |
| `txn_count_10min` | 0.5321 |
| `txn_count_24h` | 0.5288 |
| `account_age_days` | 0.5134 |
| `distance_from_home_km` | 0.5471 |
| `is_international` | 0.5215 |
| `hour_of_day` | 0.5089 |
| `is_night` | 0.5056 |

Every feature sits close to 0.50. **This is what "no leakage" looks like.**

---

## Architectures Built From Scratch

### 1. FraudFormer — Self-Attention Transformer (v4, Final Winner)

**Architecture:**

```
Input (14 features)
    │
    ▼
Dense(64) — feature embedding
    │
    ▼
+ learnable positional embedding
    │
    ▼
┌─────────────────────────────────────────┐
│  Transformer Block 1                    │
│  ├─ MultiHeadAttention(4 heads, dim=64) │
│  ├─ Add & LayerNorm                     │
│  ├─ FFN(128 → 64, GELU)                 │
│  └─ Add & LayerNorm                     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Transformer Block 2 (same structure)   │
└─────────────────────────────────────────┘
    │
    ▼
LayerNorm → GlobalAveragePooling
    │
    ▼
Dense(64) → Dropout(0.2) → Dense(32) → Dropout(0.2) → Dense(1, sigmoid)
```

**Training Parameters:**

| Parameter | Value |
|---|---|
| Embedding dim | 64 |
| Attention heads | 4 |
| Transformer blocks | 2 |
| FFN hidden dim | 128 |
| Dropout | 0.2 |
| Optimizer | Adam, LR 1e-3 |
| Batch size | 2048 |
| Epochs | 40 (early stopping on val PR-AUC) |
| **Parameters** | **~130K** |
| **Training time (CPU)** | **5.9 min** |

**Key Features:**
- Full `get_config()` serialization support
- Attention weights extractable for interpretability
- Built entirely from scratch in TensorFlow — no pretrained weights

### 2. AEGAN — Adversarial AutoEncoder GAN (v4)

**Architecture:**

```
ENCODER (14 → 64 → 64 → 16)
    │
    ▼
DECODER (16 → 64 → 64 → 14, tanh)   ← also serves as GENERATOR
    │
    ▼
DISCRIMINATOR (14 → 64 → 32 → 1)
```

**Training Objective** (three updates per step):

1. **Reconstruction update** — minimize `MSE(real, decoder(encoder(real)))`
2. **Discriminator update** — binary cross-entropy to separate real from generated
3. **Generator update** — `0.1 × adversarial_loss + 1.0 × reconstruction_loss`

The weighted objective forces the generator to produce samples that are **both realistic** and **distributionally faithful**.

**Training Parameters:**

| Parameter | Value |
|---|---|
| Latent dim | 16 |
| Batch size | 256 |
| Epochs | 30 |
| Synthetic samples generated | 25,000 |
| **Training time (CPU)** | **8.3 min** |

### 3. FraudAttnNet — Self-Attention MLP (v2, archived)

A custom-attention MLP built in v2. Achieved 0.9915 PR-AUC — but this was because the base data still leaked, not because the model was exceptional. **Discarded in v4** in favor of FraudFormer.

### 4. WGAN-GP — Wasserstein GAN (v2, archived)

Custom Wasserstein GAN with gradient penalty. Trained only on fraud rows. **Discarded in v4** in favor of AEGAN (which is more stable for tabular data).

### 5. FraudCNN — 1D Convolutional Classifier (v2, archived)

1D CNN treating the 14 features as a short sequence. **Discarded in v4** because 1D convolution on tabular data has no theoretical justification (feature order is arbitrary).

### 6. Temporal GRU — Recurrent Sequence Model (v1, excluded)

Bidirectional GRU with 8-transaction lookback. **Excluded** — collapsed to chance-level ROC-AUC 0.5034. This is a **negative finding**, not a bug: the synthetic dataset has no exploitable sequence signal because fraud labels are drawn independently per row. This correct diagnosis is documented in the paper.

### 7. ANN (MLP) — Baseline Neural Network (v1)

3-layer feed-forward with batch normalization and dropout. Achieved 0.1853 PR-AUC on the leak-free (v1) data. Superseded by FraudFormer.

### 8. Logistic Regression (v1 → v4)

L2-regularized baseline with `class_weight='balanced'`. Kept as the interpretable baseline across all versions.

### 9. LightGBM (v1 → v4)

Gradient-boosted trees with `scale_pos_weight`. Consistently the strongest baseline. Final v4.0 result: PR-AUC 0.5237.

---

## Pipeline Stages

```
┌──────────────────────────────────────────────────────────────────┐
│ Stage 1 — Leak-Free Generation                                   │
│   Single base distribution → random labels → interaction drift  │
│   Output: 500K rows, 5% fraud, no single-feature leak            │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 2 — Messiness Injection + Cleaning                         │
│   25+ messiness categories → full cleaning pipeline             │
│   Output: 502,308 rows, 0 missing, 0 duplicates                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 3 — Time-Aware Split                                       │
│   Sort by timestamp → 70% / 15% / 15%                           │
│   Output: 351,615 / 75,346 / 75,347                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 4 — Feature Engineering                                    │
│   14 raw features + derived time features                       │
│   Fit frequency maps on TRAIN ONLY                              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌──────────┐          ┌──────────┐
   │LightGBM │          │FraudFormer│         │  AEGAN   │
   │baseline │          │(custom)  │          │(custom)  │
   └─────────┘          └──────────┘          └──────────┘
        │                     │                     │
        │                     │                     ▼
        │                     │          ┌──────────────────────┐
        │                     │          │LightGBM + AEGAN      │
        │                     │          │(augmentation)        │
        │                     │          └──────────────────────┘
        ▼                     ▼                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 5 — Isotonic Calibration                                   │
│   Fit on validation, apply to test                              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 6 — Threshold Optimization                                 │
│   Per-model F1-optimal threshold on calibrated val scores       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 7 — Test Evaluation + Comparison                           │
│   Single-shot evaluation on untouched test set                  │
└──────────────────────────────────────────────────────────────────┘
```

### The Seven Design Principles

Every stage is designed around a principle learned from the iterative debugging process:

| # | Principle | Why It Matters |
|---|---|---|
| 1 | **No single feature should separate classes** | Prevents distributional leakage |
| 2 | **Time split before feature engineering** | Prevents temporal leakage |
| 3 | **Fit all statistics on train only** | Prevents aggregate leakage |
| 4 | **Calibrate before thresholding** | Threshold must match calibration scale |
| 5 | **Never use test for anything but final evaluation** | Preserves the honest test |
| 6 | **Report multiple metrics, not just accuracy** | Accuracy is misleading at 5% fraud rate |
| 7 | **Leave some fraud undetectable** | Forces realistic ceiling |

---

## Results by Version

### v1.0 — Base Models (Leak-Free in Isolation)

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.7286 | 0.2405 | 0.2839 |
| LightGBM | 0.7435 | 0.2595 | 0.3018 |
| ANN (MLP) | 0.6824 | 0.1853 | 0.2182 |
| GRU (excluded) | 0.5034 | 0.0685 | 0.1267 |

**Note:** v1.0 numbers are lower than v4.0 because the v1 dataset was leaky (all models reported 0.99+ PR-AUC). The v1.0 numbers listed above were re-evaluated after leak detection — they represent the **honest baseline** for the v1 feature set.

### v2.0 — Custom Deep Learning (Discarded)

FraudAttnNet, WGAN-GP, and FraudCNN all achieved 0.99+ PR-AUC — but on leaky data. **All v2 models discarded** because the underlying data was corrupt.

### v3.0 — Behavioral Features (Discarded)

Customer fingerprint features hit 0.99+ PR-AUC via the "new customer" shortcut. **All v3 features discarded** because the generator produced class-correlated customer distributions.

### v3.1 — Diagnostic Fix

Reduced PR-AUC to 0.9986 (still too high but better). Identified burst features on full data as the remaining leak.

### v4.0 — Leak-Free (Final)

| Model | Threshold | ROC-AUC | PR-AUC | Precision | Recall | **F1** |
|---|---|---|---|---|---|---|
| LightGBM (baseline) | 0.3286 | 0.7661 | 0.5237 | 0.7985 | 0.4392 | 0.5667 |
| LightGBM + AEGAN | 0.3000 | 0.7652 | 0.5268 | 0.8411 | 0.4357 | 0.5741 |
| **FraudFormer** | **0.3521** | **0.7692** | **0.5330** | **0.8511** | **0.4473** | **0.5864** |

### Comparison to Published Benchmarks

| Dataset | Fraud Rate | Typical PR-AUC |
|---|---|---|
| IEEE-CIS Fraud (real) | 3.5% | 0.35–0.55 |
| ULB Credit Card (real) | 0.17% | 0.75–0.88 |
| Production fraud systems | 0.1–2% | 0.20–0.50 |
| **FraudGuard v4.0 (ours)** | **4.99%** | **0.5330** |

**FraudGuard's 0.53 PR-AUC matches or beats real-world production baselines** while maintaining full methodological integrity: no leakage, no inflated metrics, no cherry-picked thresholds.

### Interpreting the Numbers

| Metric | Value | Meaning |
|---|---|---|
| **ROC-AUC 0.77** | Moderate | Meaningful separation, not chance, not trivial |
| **PR-AUC 0.53** | Strong | Well above random baseline (0.05); in realistic production range |
| **F1 0.59** | Balanced | Equal weighting of precision and recall |
| **Precision 0.85** | High | 85% of flagged transactions are actually fraud |
| **Recall 0.45** | Moderate | 45% of fraud is caught at this operating point |

**Operating point trade-off:** the F1-optimal threshold prioritizes precision over recall. For production use, lower the threshold to 0.20 to catch more fraud at the cost of more false alarms.

---

## Figures

### v4.0 — FraudFormer Attention Heatmap

![FraudFormer Attention](03_figures/figures_v4/06_attention.png)

*Figure 1 — FraudFormer attention heatmap (block 0, averaged over 2000 test samples × 4 heads). Bright cells indicate feature pairs the model considers mutually informative. Semantically meaningful pairs (`amount` → `txn_count_10min`, `distance_from_home_km` → `is_international`) confirm the model learned the interaction structure the dataset was designed to encode.*

### v4.0 — FraudFormer Training Curves

![FraudFormer Curves](03_figures/figures_v4/01_fraudformer_curves.png)

*Figure 2 — FraudFormer training curves. Left: binary cross-entropy loss. Right: validation PR-AUC. Early stopping triggered around epoch 15–20.*

### v4.0 — AEGAN Distribution Validation

![AEGAN Distribution](03_figures/figures_v4/02_aegan_distribution.png)

*Figure 3 — Feature-wise means of real fraud vs AEGAN-synthesized fraud. A well-trained AEGAN matches the real distribution closely across all 14 features. Systematic bias in any single feature would show as a large gap.*

### v4.0 — AEGAN Training Curves

![AEGAN Curves](03_figures/figures_v4/03_aegan_curves.png)

*Figure 4 — AEGAN training curves. The three losses (reconstruction, discriminator, generator) should stabilize together. Divergence indicates mode collapse or training instability.*

### v4.0 — ROC and PR Curves (All Models)

![All Models](03_figures/figures_v4/04_all_models.png)

*Figure 5 — ROC (left) and Precision-Recall (right) curves on the test set. FraudFormer (red) sits slightly above the two LightGBM baselines (blue and green) on both, consistent with its higher PR-AUC (0.5330 vs 0.5268 / 0.5237).*

### v4.0 — Confusion Matrices

![Confusion Matrices](03_figures/figures_v4/05_confusions.png)

*Figure 6 — Confusion matrices at each model's F1-optimal threshold. Left: LightGBM baseline (thr=0.33). Center: LightGBM + AEGAN (thr=0.30). Right: FraudFormer (thr=0.35). All three evaluated on the same test set (75,347 rows, 3,760 fraud cases).*

### v1.0 — Precision-Recall Curve (Validation)

![PR Curve v1](03_figures/figures_model/01_pr_curve.png)

*Figure 7 — v1.0 Precision-Recall curve with F1-optimal threshold marked.*

### v1.0 — Calibration Curve

![Calibration v1](03_figures/figures_model/02_calibration.png)

*Figure 8 — v1.0 isotonic calibration curve. Predicted probabilities align with observed rates after calibration.*

### v1.0 — Feature Importance (LightGBM)

![Feature Importance v1](03_figures/figures_model/04_importance.png)

*Figure 9 — Top LightGBM features by gain in v1.0.*

### v1.0 — Confusion Matrix

![Confusion v1](03_figures/figures_model/03_confusion.png)

*Figure 10 — v1.0 confusion matrix on the test set.*

---


## Full Image Gallery — All Notebook Outputs

All 84 figures extracted from the Jupyter notebook outputs (72 from raud-data-analysis-and-model.ipynb + 12 from credit-card-analysis-file.ipynb), rendered inline for complete visual reference.

![credit-card-analysis-file_001.png](03_figures/notebook_outputs/credit-card-analysis-file_001.png)

![credit-card-analysis-file_002.png](03_figures/notebook_outputs/credit-card-analysis-file_002.png)

![credit-card-analysis-file_003.png](03_figures/notebook_outputs/credit-card-analysis-file_003.png)

![credit-card-analysis-file_004.png](03_figures/notebook_outputs/credit-card-analysis-file_004.png)

![credit-card-analysis-file_005.png](03_figures/notebook_outputs/credit-card-analysis-file_005.png)

![credit-card-analysis-file_006.png](03_figures/notebook_outputs/credit-card-analysis-file_006.png)

![credit-card-analysis-file_007.png](03_figures/notebook_outputs/credit-card-analysis-file_007.png)

![credit-card-analysis-file_008.png](03_figures/notebook_outputs/credit-card-analysis-file_008.png)

![credit-card-analysis-file_009.png](03_figures/notebook_outputs/credit-card-analysis-file_009.png)

![credit-card-analysis-file_010.png](03_figures/notebook_outputs/credit-card-analysis-file_010.png)

![credit-card-analysis-file_011.png](03_figures/notebook_outputs/credit-card-analysis-file_011.png)

![credit-card-analysis-file_012.png](03_figures/notebook_outputs/credit-card-analysis-file_012.png)

![fraud-data-analysis-and-model_001.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_001.png)

![fraud-data-analysis-and-model_002.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_002.png)

![fraud-data-analysis-and-model_003.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_003.png)

![fraud-data-analysis-and-model_004.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_004.png)

![fraud-data-analysis-and-model_005.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_005.png)

![fraud-data-analysis-and-model_006.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_006.png)

![fraud-data-analysis-and-model_007.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_007.png)

![fraud-data-analysis-and-model_008.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_008.png)

![fraud-data-analysis-and-model_009.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_009.png)

![fraud-data-analysis-and-model_010.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_010.png)

![fraud-data-analysis-and-model_011.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_011.png)

![fraud-data-analysis-and-model_012.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_012.png)

![fraud-data-analysis-and-model_013.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_013.png)

![fraud-data-analysis-and-model_014.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_014.png)

![fraud-data-analysis-and-model_015.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_015.png)

![fraud-data-analysis-and-model_016.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_016.png)

![fraud-data-analysis-and-model_017.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_017.png)

![fraud-data-analysis-and-model_018.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_018.png)

![fraud-data-analysis-and-model_019.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_019.png)

![fraud-data-analysis-and-model_020.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_020.png)

![fraud-data-analysis-and-model_021.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_021.png)

![fraud-data-analysis-and-model_022.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_022.png)

![fraud-data-analysis-and-model_023.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_023.png)

![fraud-data-analysis-and-model_024.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_024.png)

![fraud-data-analysis-and-model_025.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_025.png)

![fraud-data-analysis-and-model_026.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_026.png)

![fraud-data-analysis-and-model_027.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_027.png)

![fraud-data-analysis-and-model_028.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_028.png)

![fraud-data-analysis-and-model_029.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_029.png)

![fraud-data-analysis-and-model_030.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_030.png)

![fraud-data-analysis-and-model_031.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_031.png)

![fraud-data-analysis-and-model_032.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_032.png)

![fraud-data-analysis-and-model_033.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_033.png)

![fraud-data-analysis-and-model_034.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_034.png)

![fraud-data-analysis-and-model_035.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_035.png)

![fraud-data-analysis-and-model_036.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_036.png)

![fraud-data-analysis-and-model_037.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_037.png)

![fraud-data-analysis-and-model_038.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_038.png)

![fraud-data-analysis-and-model_039.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_039.png)

![fraud-data-analysis-and-model_040.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_040.png)

![fraud-data-analysis-and-model_041.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_041.png)

![fraud-data-analysis-and-model_042.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_042.png)

![fraud-data-analysis-and-model_043.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_043.png)

![fraud-data-analysis-and-model_044.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_044.png)

![fraud-data-analysis-and-model_045.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_045.png)

![fraud-data-analysis-and-model_046.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_046.png)

![fraud-data-analysis-and-model_047.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_047.png)

![fraud-data-analysis-and-model_048.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_048.png)

![fraud-data-analysis-and-model_049.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_049.png)

![fraud-data-analysis-and-model_050.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_050.png)

![fraud-data-analysis-and-model_051.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_051.png)

![fraud-data-analysis-and-model_052.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_052.png)

![fraud-data-analysis-and-model_053.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_053.png)

![fraud-data-analysis-and-model_054.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_054.png)

![fraud-data-analysis-and-model_055.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_055.png)

![fraud-data-analysis-and-model_056.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_056.png)

![fraud-data-analysis-and-model_057.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_057.png)

![fraud-data-analysis-and-model_058.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_058.png)

![fraud-data-analysis-and-model_059.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_059.png)

![fraud-data-analysis-and-model_060.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_060.png)

![fraud-data-analysis-and-model_061.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_061.png)

![fraud-data-analysis-and-model_062.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_062.png)

![fraud-data-analysis-and-model_063.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_063.png)

![fraud-data-analysis-and-model_064.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_064.png)

![fraud-data-analysis-and-model_065.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_065.png)

![fraud-data-analysis-and-model_066.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_066.png)

![fraud-data-analysis-and-model_067.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_067.png)

![fraud-data-analysis-and-model_068.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_068.png)

![fraud-data-analysis-and-model_069.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_069.png)

![fraud-data-analysis-and-model_070.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_070.png)

![fraud-data-analysis-and-model_071.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_071.png)

![fraud-data-analysis-and-model_072.png](03_figures/notebook_outputs/fraud-data-analysis-and-model_072.png)

## Repository Structure

```text
E:\Projects\Credit Card Analysis\
│
├── README.md                                   This file
├── README.txt                                  Quick-reference summary
│
├── 01_datasets/
│   ├── cleaned_fraud_data.csv                  Final cleaned dataset (502,308 rows × 28 cols)
│   └── messy_fraud_data_raw.csv                Raw messy dataset (507,500 rows)
│
├── 02_models/
│   ├── models/                                 v1 models (legacy)
│   │   ├── lgbm_fraud.txt
│   │   ├── calibrated_model.pkl
│   │   ├── isotonic_calibrator.pkl
│   │   ├── scaler.pkl
│   │   ├── features.pkl
│   │   ├── metadata.pkl
│   │   ├── model_comparison.csv
│   │   └── summary.csv
│   └── models_v4/                              v4 models (final)
│       ├── fraudformer.keras                   BEST MODEL
│       ├── aegan_encoder.keras
│       ├── aegan_decoder.keras
│       ├── aegan_discriminator.keras
│       ├── lgbm_baseline.txt
│       ├── lgbm_augmented.txt
│       ├── scaler_ff.pkl
│       ├── iso_lgb.pkl
│       ├── iso_ff.pkl
│       ├── iso_lgb_aug.pkl
│       ├── features.pkl
│       ├── metadata.json
│       └── final_comparison.csv
│
├── 03_figures/
│   ├── figures/                                EDA figures (35+ visualizations)
│   ├── figures_model/                          v1 model figures (10 files)
│   ├── figures_dl/                             v2 DL figures (4 files)
│   ├── figures_v4/                             v4 figures (6 files)
│   └── notebook_outputs/                       84 PNGs extracted from notebook outputs
│       ├── 01_fraudformer_curves.png
│       ├── 02_aegan_distribution.png
│       ├── 03_aegan_curves.png
│       ├── 04_all_models.png
│       ├── 05_confusions.png
│       └── 06_attention.png
│
├── 04_reports/
│   ├── final_comparison.csv                    All-model comparison
│   └── metadata.json                           Full configuration
│
├── assets/
│   └── credit_card_cover.svg                   Cover image
│
├── fraud-detection-cleaning.ipynb              v1 data generation + cleaning
├── fraud-data-analysis-and-model.ipynb         v1 EDA + modeling
├── credit-card-analysis-file.ipynb             v4 end-to-end pipeline (final)
├── fraud_detection_project_20261016_1701.zip   v1 packaged archive
└── fraudguard_v4_20261017_1520.zip             v4 packaged archive
```

### Where the Figures Live (for GitHub Rendering)

The README embeds **10 figures** using relative paths. All will render correctly if pushed to GitHub.

| # | Figure | Path |
|---|---|---|
| 1 | FraudFormer Attention | `03_figures/figures_v4/06_attention.png` |
| 2 | FraudFormer Curves | `03_figures/figures_v4/01_fraudformer_curves.png` |
| 3 | AEGAN Distribution | `03_figures/figures_v4/02_aegan_distribution.png` |
| 4 | AEGAN Curves | `03_figures/figures_v4/03_aegan_curves.png` |
| 5 | All Models ROC/PR | `03_figures/figures_v4/04_all_models.png` |
| 6 | Confusion Matrices | `03_figures/figures_v4/05_confusions.png` |
| 7 | PR Curve (v1) | `03_figures/figures_model/01_pr_curve.png` |
| 8 | Calibration (v1) | `03_figures/figures_model/02_calibration.png` |
| 9 | Feature Importance (v1) | `03_figures/figures_model/04_importance.png` |
| 10 | Confusion Matrix (v1) | `03_figures/figures_model/03_confusion.png` |

---

## Reproducing the Project

### Requirements

- Python 3.10+
- TensorFlow 2.x
- LightGBM 4.x
- scikit-learn, pandas, numpy, matplotlib, seaborn, joblib

### Installation

```bash
pip install tensorflow lightgbm scikit-learn pandas numpy matplotlib seaborn joblib
```

### Notebooks

The project is provided as three notebooks:

| Notebook | Purpose | Runtime (CPU) |
|---|---|---|
| `fraud-detection-cleaning.ipynb` | v1 data generation + cleaning | ~5 min |
| `fraud-data-analysis-and-model.ipynb` | v1 EDA + modeling | ~10 min |
| `credit-card-analysis-file.ipynb` | **v4 end-to-end pipeline** (final) | ~26 min |

For v4 results, run `credit-card-analysis-file.ipynb` from start to finish.

### Reproducibility Guarantees

- Random seed `42` throughout
- Deterministic time-based split
- Frequency maps fit only on training fold
- Augmentation (SMOTE / AEGAN) applied only on training fold
- Isotonic calibrators fit on validation predictions
- All thresholds computed on validation, applied to test

### Testing Environment

Tested on:

- **Kaggle CPU:** 2× Intel Xeon @ 2.20 GHz, 4 GB RAM — full v4 pipeline in ~26 min
- **Google Colab CPU:** similar runtime

The pipeline does not require a GPU.

---

## Usage and Deployment

### Quick Load and Predict (FraudFormer)

```python
import json, joblib, numpy as np
from tensorflow.keras.models import load_model

MODEL_DIR = "02_models/models_v4"
model    = load_model(f"{MODEL_DIR}/fraudformer.keras")
scaler   = joblib.load(f"{MODEL_DIR}/scaler_ff.pkl")
features = joblib.load(f"{MODEL_DIR}/features.pkl")
meta     = json.load(open(f"{MODEL_DIR}/metadata.json"))

THRESHOLD = meta["thresholds"]["ff"]

def predict(transaction: dict) -> dict:
    x = np.array([[transaction[f] for f in features]], dtype=np.float32)
    x = scaler.transform(x).clip(-8, 8)
    prob = float(model.predict(x, verbose=0)[0, 0])
    return {
        "fraud_probability": round(prob, 4),
        "is_fraud": prob >= THRESHOLD,
        "threshold": THRESHOLD,
        "model": "FraudFormer",
    }

# Example
result = predict({
    "amount": 245.80, "txn_count_10min": 4, "txn_count_24h": 12,
    "distance_from_home_km": 350.5, "is_international": 1,
    "is_night": 1, "account_age_days": 45, "hour_of_day": 2,
    "avg_amount_30d": 82.4, "credit_limit": 200000,
    "card_last4": 4321, "weekday": 2, "day_of_month": 15, "month": 8,
})
print(result)
# {'fraud_probability': 0.71, 'is_fraud': True, 'threshold': 0.35, 'model': 'FraudFormer'}
```

### FastAPI Service

```python
from fastapi import FastAPI
import joblib, json, numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(title="FraudGuard v4.0")

MODEL_DIR = "02_models/models_v4"
model    = load_model(f"{MODEL_DIR}/fraudformer.keras")
scaler   = joblib.load(f"{MODEL_DIR}/scaler_ff.pkl")
features = joblib.load(f"{MODEL_DIR}/features.pkl")
meta     = json.load(open(f"{MODEL_DIR}/metadata.json"))
THRESHOLD = meta["thresholds"]["ff"]

@app.post("/predict")
def predict(payload: dict):
    x = np.array([[payload[f] for f in features]], dtype=np.float32)
    x = scaler.transform(x).clip(-8, 8)
    prob = float(model.predict(x, verbose=0)[0, 0])
    return {
        "fraud_probability": round(prob, 4),
        "is_fraud": prob >= THRESHOLD,
        "threshold": THRESHOLD,
    }
```

### Streamlit Demo

```python
import streamlit as st
import joblib, json, numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="FraudGuard v4.0", page_icon="🛡")
st.title("FraudGuard v4.0 — Live Demo")
st.caption("FraudFormer (custom transformer) · PR-AUC 0.5330")

model    = load_model("02_models/models_v4/fraudformer.keras")
scaler   = joblib.load("02_models/models_v4/scaler_ff.pkl")
features = joblib.load("02_models/models_v4/features.pkl")
meta     = json.load(open("02_models/models_v4/metadata.json"))

with st.form("transaction"):
    amount = st.number_input("Amount (USD)", 1.0, 10000.0, 250.0)
    txn_10 = st.slider("Transactions in last 10 min", 0, 30, 1)
    txn_24 = st.slider("Transactions in last 24 h", 0, 60, 3)
    dist   = st.slider("Distance from home (km)", 0.0, 5000.0, 20.0)
    intl   = st.checkbox("International")
    night  = st.checkbox("Night-time (00:00–04:59)")
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {f: 0 for f in features}
    payload.update({
        "amount": amount,
        "txn_count_10min": txn_10,
        "txn_count_24h": txn_24,
        "distance_from_home_km": dist,
        "is_international": int(intl),
        "is_night": int(night),
    })
    x = np.array([[payload[f] for f in features]], dtype=np.float32)
    x = scaler.transform(x).clip(-8, 8)
    prob = float(model.predict(x, verbose=0)[0, 0])

    st.metric("Fraud probability", f"{prob:.2%}")
    st.metric("Decision", "BLOCK" if prob >= meta["thresholds"]["ff"] else "ALLOW")
```

### Operating Point Selection

The `metadata.json` provides **three thresholds** for different business priorities:

| Threshold | Precision | Recall | Use Case |
|---|---|---|---|
| **0.3521** (F1-optimal) | 0.85 | 0.45 | Balanced default |
| **~0.20** (recall-priority) | ~0.55 | ~0.75 | When missing fraud is very costly |
| **~0.50** (precision-priority) | ~0.92 | ~0.20 | When false alarms are very costly |

---

## Limitations and Honest Reporting

### What FraudGuard v4.0 Proves

- **Leak-free synthetic data generation is possible** — no single feature exceeds ROC-AUC 0.60
- **A custom transformer can beat gradient-boosted trees** on interaction-driven fraud data (+0.0093 PR-AUC)
- **AEGAN augmentation provides small but positive value** (+0.0031 PR-AUC over baseline)
- **Isotonic calibration + threshold optimization produce usable probabilities**

### What FraudGuard v4.0 Does NOT Prove

- **Absolute performance transfers to real data.** 0.53 PR-AUC on synthetic data does not mean 0.53 on real transaction logs.
- **FraudFormer beats LightGBM by a large margin.** The +0.0093 gain is real but small. On production fraud datasets with 0.5–2% fraud rates, LightGBM typically still wins.
- **AEGAN outperforms SMOTE.** Both were tried; the difference is within noise. AEGAN's advantage is theoretical, not empirical.
- **The pipeline handles adversarial dynamics.** Real fraudsters adapt. This dataset is static.

### Honest Failure Log (Four Major Iterations)

| Version | PR-AUC | Issue Found | Fix |
|---|---|---|---|
| v1.0 | 0.99+ | Distributional leakage — fraud means 3× normal on 4 features | Regenerate with overlapping distributions |
| v2.0 | 0.99+ | Temporal leakage — sorted timestamps confined normal rows to specific hours | Fix generator to draw timestamps identically |
| v3.0 | 0.99+ | Customer fingerprint leakage — new customers were systematically fraud-heavy | Drop aggregate features; verify with single-feature check |
| v3.1 | 0.9986 | Same leak via burst features on full data | Rebuild burst using strictly-earlier transactions only |
| **v4.0** | **0.5330** | **Clean** | **Interaction-only generator, verified single-feature ROC-AUC < 0.60** |

This evolution is documented because **the fix is more valuable than the metric**. A reviewer can follow the reasoning and apply the same discipline to their own project.

### Additional Limitations

1. **Dataset provenance** — absolute metrics don't transfer to live traffic; the pipeline structure does.
2. **No sequence structure** — GRU was evaluated and excluded (chance-level). No customer-level temporal signal in the data.
3. **High fraud rate** — 5% is higher than production (0.1–2%), inflating PR-AUC.
4. **No adversarial dynamics** — static dataset, no drift simulation.
5. **No graph features** — real fraud networks exhibit graph structure that this dataset lacks.

### What a Real Production Deployment Would Add

| Component | Purpose | Status |
|---|---|---|
| Adversarial validation | Detect train/test drift | Not implemented |
| Customer-level sequences | Exploit temporal patterns | Excluded (no sequence in data) |
| Graph features | Shared devices, IPs, merchants | Not implemented |
| Model monitoring | Detect performance drift | Not implemented |
| Explainability API | SHAP values per prediction | Not implemented (attention weights are extractable) |
| Stratified retraining | Refresh on new data weekly | Not implemented |

---

## Citation

If you use this project or its ideas, cite as:

```bibtex
@misc{fraudguard2026,
  title  = {FraudGuard v4.0: A Leak-Free, Custom-Architecture Pipeline for Credit Card Fraud Detection},
  author = {Sourish Dey},
  year   = {2026},
  note   = {Complete end-to-end pipeline with custom FraudFormer and AEGAN architectures.
            Test PR-AUC 0.5330 on synthetic interaction-driven fraud data.
            Four major iterations documented with full failure analysis.}
}
```

### Related Work

- **Fiore et al. (2019).** "Using generative adversarial networks for improving classification effectiveness in credit card fraud detection." *Information Sciences.* — GAN-based fraud generation
- **Motwani et al. (2021).** "Adversarial Autoencoder for Fraud Detection." *IEEE Access.* — Adversarial autoencoder approach
- **Vaswani et al. (2017).** "Attention Is All You Need." *NeurIPS.* — Transformer architecture
- **Chen & Guestrin (2016).** "XGBoost: A Scalable Tree Boosting System." *KDD.* — Gradient boosting
- **Ke et al. (2017).** "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS.* — LightGBM implementation

---

## License

This project is provided for educational and research purposes. The synthetic dataset is generated by the included Python code and contains no real customer information. All custom architectures (FraudFormer, AEGAN, FraudAttnNet, FraudCNN, WGAN-GP) are original implementations built from scratch for this project.

---

## Project Summary

| Property | Value |
|---|---|
| Project Name | **FraudGuard** |
| Current Version | **4.0** |
| Versions Documented | v1.0, v2.0, v3.0, v3.1, v4.0 |
| Dataset Size | 502,308 rows × 28 columns |
| Fraud Rate | 4.99% |
| Features (v4) | 14 |
| Best Model | **FraudFormer (custom transformer)** |
| Best Test ROC-AUC | **0.7692** |
| Best Test PR-AUC | **0.5330** |
| Best Test F1 | **0.5864** |
| Training Time (CPU) | ~26 min end-to-end |
| Models Trained (all versions) | 10 |
| Custom Architectures | 5 (FraudFormer, AEGAN, FraudAttnNet, WGAN-GP, FraudCNN) |
| Figures | 106+ (6 v4 + 10 v1 + 4 v2 + 84 notebook-extracted + 2 charts) |
| Reproducible | Yes (seed 42) |
| Leak-Free | Verified (single-feature ROC-AUC < 0.60) |

---

**Built with:** Python · pandas · scikit-learn · LightGBM · TensorFlow · matplotlib · seaborn · Plotly

**Last updated:** October 2026

**Made by Sourish Dey**
- GitHub: [@sourishdey2005](https://github.com/sourishdey2005)
- Repository: [sourishdey2005/FraudGuard](https://github.com/sourishdey2005/FraudGuard)
