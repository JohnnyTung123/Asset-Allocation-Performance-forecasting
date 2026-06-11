# Asset-Allocation-Performance-forecasting

# Trust or Short? Predicting Asset Allocation Performance

This project is a solution to the QRT systematic trading challenge: **Trust or Short? Predicting the Performance of Daily Asset Allocations**.

The goal is to predict whether a given asset allocation will generate a **positive or negative next-day return**, using its recent historical behavior.

---

## 📌 Problem Overview

Each sample represents an **asset allocation strategy** applied on a specific day.  
We are given the past **20-day history** of:

- Daily returns of the allocation
- Signed volume exposure
- Turnover behavior
- Median turnover
- Allocation group information

### Objective

Predict:

- `1` → Allocation will have **positive next-day return** (Trust / Long)
- `0` → Allocation will have **negative next-day return** (Short)

Only **directional accuracy** is evaluated.

---

## 📊 Evaluation Metric

Accuracy of sign prediction:

\[
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{sign}(\hat{r}_i) = \text{sign}(r_i)]
\]

Where:

- \( \hat{r}_i \): predicted return
- \( r_i \): true next-day return

---

## 📁 Dataset Description

Each row corresponds to `(date, allocation)` and includes:

### Time Series Features (20-day history)
- `RET_1 ... RET_20` → past allocation returns
- `SIGNED_VOLUME_1 ... SIGNED_VOLUME_20` → volume exposure proxy

### Static / Aggregated Features
- `MEDIAN_DAILY_TURNOVER`
- `GROUP` (anonymized strategy cluster)

### Target
- `TARGET` → next-day return (used only for training)

---

## ⚙️ Feature Engineering

We enrich raw signals with statistical and time-series features:

### 1. Return-based features
- Mean / std of returns over multiple windows (3, 5, 10, 20 days)
- Momentum indicators
- Trend consistency (sign stability)

### 2. Volatility features
- Rolling volatility of returns
- Volatility of signed volume
- Stability score of allocation performance

### 3. Liquidity & turnover features
- Average signed volume
- Turnover level and changes
- Turnover volatility (strategy instability proxy)

### 4. Cross-sectional normalization features
- Group-level z-scores
- Relative performance vs group average allocation

---

## 🧠 Models

We use a **two-model ensemble**:

### 1. LightGBM (Main Model)
- Gradient boosting decision trees
- Handles nonlinear interactions well
- Robust to noisy financial signals

### 2. Ridge Regression (Baseline)
- Linear benchmark model
- Helps stabilize predictions in noisy regimes

### Final Prediction
Weighted average / blending of:
- `LightGBM probability`
- `Ridge output`

---

## 🔄 Training Strategy

- Time-agnostic shuffled dataset (no strict time continuity)
- Cross-validation using grouped splits (by allocation or time proxy)
- Early stopping for LightGBM
- Regularization tuned to avoid overfitting

---

## 📈 Benchmark

The provided baseline uses:
- Simple rolling features
- Ridge + LightGBM ensemble

Public leaderboard score:
- **0.5079 accuracy**

Our goal is to improve signal extraction beyond naive historical averaging.

---

## 🚀 Key Insights

- Allocation performance is **regime-dependent**
- Signed volume contains predictive liquidity pressure signals
- Turnover instability often precedes reversals
- Cross-sectional normalization improves robustness

---

## 📦 Repository Structure
