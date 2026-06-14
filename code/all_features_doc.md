# all_features.py — Documentation

## Overview

`all_features.py` is the complete feature engineering pipeline for the Asset Allocation Performance Forecasting competition. It can be used in two ways:

- **Run directly** — executes the full pipeline (feature engineering → 8-fold CV → final model → submission file):
  ```bash
  python all_features.py
  ```
- **Imported as a module** — use `build_features` and `apply_target_encoding` inside your own notebook or script:
  ```python
  from all_features import build_features, apply_target_encoding
  ```

---

## Public API

### `build_features(df, ret_cols, vol_cols)`

Adds all engineered features to a dataframe. Call on both `X_train` and `X_test` before the CV loop.

| Parameter  | Type            | Description                                              |
|------------|-----------------|----------------------------------------------------------|
| `df`       | `pd.DataFrame`  | Input dataframe (`X_train` or `X_test`)                 |
| `ret_cols` | `list[str]`     | Return column names, e.g. `['RET_1', ..., 'RET_20']`   |
| `vol_cols` | `list[str]`     | Volume column names, e.g. `['SIGNED_VOLUME_1', ...]`    |

**Returns:** `(df_with_features, feature_names)` — the enriched dataframe and the list of all feature column names.

> **Note:** `ALLOC_ENC` (target encoding for `ALLOCATION`) is **not** included here. It must be computed inside each CV fold via `apply_target_encoding()` to avoid label leakage.

**Example:**
```python
RET_cols = [f'RET_{i}'           for i in range(1, 21)]
VOL_cols = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

X_train, features = build_features(X_train, RET_cols, VOL_cols)
X_test,  _        = build_features(X_test,  RET_cols, VOL_cols)
```

---

### `apply_target_encoding(df_fit, y_fit, df_transform, smoothing=10)`

Computes a smoothed mean target encoding for the `ALLOCATION` column and adds it as `ALLOC_ENC`.

| Parameter      | Type           | Description                                                           |
|----------------|----------------|-----------------------------------------------------------------------|
| `df_fit`       | `pd.DataFrame` | Training split used to compute encoding statistics                   |
| `y_fit`        | `pd.Series`    | Target values aligned with `df_fit`                                  |
| `df_transform` | `pd.DataFrame` | Split to apply the encoding to (can be train or val/test)            |
| `smoothing`    | `int`          | Regularisation strength (default 10); higher = shrink more to global mean |

**Returns:** A copy of `df_transform` with the `ALLOC_ENC` column added.

**Formula:**
```
ALLOC_ENC = (count × strategy_mean + smoothing × global_mean) / (count + smoothing)
```

**Strategies not seen in `df_fit`** are assigned the global mean.

**Example — inside a CV fold:**
```python
X_tr  = apply_target_encoding(X_train.loc[tr_mask], y_tr, X_train.loc[tr_mask])
X_val = apply_target_encoding(X_train.loc[tr_mask], y_tr, X_train.loc[val_mask])
```

**Example — final model:**
```python
X_train_enc = apply_target_encoding(X_train, y_train['target'], X_train)
X_test_enc  = apply_target_encoding(X_train, y_train['target'], X_test)
```

---

## Feature Groups

`build_features()` runs 17 internal functions in the order shown below. The ordering matters — later groups depend on columns created by earlier ones (e.g. `AVERAGE_PERF_*` and `STD_PERF_20` must exist before cross-sectional or interaction features are computed).

### 1. Benchmark (`_benchmark`)

The baseline features from the competition benchmark notebook.

| Feature | Description |
|---------|-------------|
| `RET_1` … `RET_20` | Raw 20-day return history (most recent = `RET_1`) |
| `SIGNED_VOLUME_1` … `SIGNED_VOLUME_20` | Raw signed volume history |
| `MEDIAN_DAILY_TURNOVER` | Static liquidity proxy |
| `AVERAGE_PERF_{3,5,10,15,20}` | Rolling mean return over each window |
| `ALLOCATIONS_AVERAGE_PERF_{3,5,10,15,20}` | Cross-allocation mean of the above per date |
| `STD_PERF_20` | 20-day realised volatility (row-wise std of returns) |
| `ALLOCATIONS_STD_PERF_20` | Cross-allocation mean of vol per date |

---

### 2. Cross-Sectional Z-Score (`_xs_zscore`)

Strips out the common market factor by normalising every return and volume column within its date's cross-section.

| Feature | Description |
|---------|-------------|
| `XS_Z_RET_1` … `XS_Z_RET_20` | Within-date z-score of each return lag |
| `XS_Z_SIGNED_VOLUME_1` … `XS_Z_SIGNED_VOLUME_20` | Within-date z-score of each volume lag |

---

### 3. Cross-Sectional Ranks (`_xs_ranks`)

Percentile rank of each strategy within its date's cross-section — scale-free and regime-robust.

| Feature | Description |
|---------|-------------|
| `XS_RANK_RET_{1,3,5,10,20}` | Rank of momentum at each horizon |
| `XS_RANK_SVOL_{1,5}` | Rank of signed-volume mean at horizons 1 and 5 |

---

### 4. Date-Level Context (`_date_context`)

Tells the model how unusual today's cross-section is.

| Feature | Description |
|---------|-------------|
| `XS_DISP_RET1` | Spread (std) of `RET_1` across all strategies on this date |
| `XS_DISP_RET5` | Spread of 5-day mean returns across strategies |
| `XS_MEAN_RET1` | Common market move on the most recent day |
| `XS_PCT_RET1` | This strategy's percentile in today's cross-section |
| `XS_DISP_SVOL1` | Spread of signed volume across strategies |

---

### 5. Momentum Sums (`_momentum`)

Return sums at four horizons — sum (not mean) preserves scale across window lengths.

| Feature | Description |
|---------|-------------|
| `MOM_SUM_SHORT` | 3-day sum |
| `MOM_SUM_MED5` | 5-day sum |
| `MOM_SUM_MED10` | 10-day sum |
| `MOM_SUM_LONG` | 20-day sum |

---

### 6. Momentum Acceleration (`_acceleration`)

Differences between momentum horizons — captures whether recent returns are strengthening or fading.

| Feature | Description |
|---------|-------------|
| `ACCEL_S_M5` | Short vs medium-5 momentum |
| `ACCEL_M5_M10` | Medium-5 vs medium-10 momentum |
| `ACCEL_M10_L` | Medium-10 vs long momentum |
| `ACCEL_S_L` | Short vs long momentum |

---

### 7. EWMA Returns (`_ewma`)

Exponentially weighted mean of returns at 7 half-lives (1, 2, 3, 5, 7, 10, 15). Half-life 1 gives near-full weight to `RET_1`; half-life 15 approaches an equal-weight 20-day mean.

| Feature | Description |
|---------|-------------|
| `EWMA_HL1` … `EWMA_HL15` | EWMA at each half-life |

---

### 8. Standardised Reversal (`_reversal`)

Measures how extreme the most recent move was relative to this strategy's normal range.

| Feature | Description |
|---------|-------------|
| `STD_REVERSAL` | `RET_1` divided by 20-day vol |
| `ABS_STD_REVERSAL` | Absolute value of the above |

---

### 9. Return Distribution Shape (`_distribution`)

Shape statistics of the 20-day return distribution.

| Feature | Description |
|---------|-------------|
| `DOWNSIDE_VOL` | Std of negative-return days only |
| `RET_SKEW` | Skewness of the 20-day return series |
| `FRAC_POS_20` | Fraction of positive days over 20 days |
| `COUNT_POS_5` | Count of positive days in the most recent 5 days |

---

### 10. Streaks (`_streaks`)

Consecutive up/down runs capture directional persistence that simple means miss.

| Feature | Description |
|---------|-------------|
| `LONGEST_UP_STREAK` | Longest consecutive positive-return run |
| `LONGEST_DOWN_STREAK` | Longest consecutive negative-return run |
| `POS_LAST_3` | Count of positive days in the most recent 3 days |

---

### 11. Max Drawdown (`_drawdown`)

Path-dependent risk indicators computed on the chronological (oldest-first) return sequence.

| Feature | Description |
|---------|-------------|
| `MAX_DRAWDOWN` | Worst trough below the running high-water mark |
| `DIST_FROM_PEAK` | Current distance below the 20-day peak |

---

### 12. Vol-Adjusted Momentum (`_vol_interactions`)

Sharpe-like ratios: momentum features divided by 20-day realised vol. Lets the model distinguish a strong move in a low-vol regime from the same move in high vol.

| Feature | Description |
|---------|-------------|
| `AVERAGE_PERF_{3,5,10,20}_DIV_VOL` | Mean return at each horizon ÷ vol |
| `MOM_SUM_SHORT_DIV_VOL` | Short momentum ÷ vol |
| `MOM_SUM_MED5_DIV_VOL` | Medium-5 momentum ÷ vol |
| `MOM_SUM_LONG_DIV_VOL` | Long momentum ÷ vol |
| `EWMA_HL2_DIV_VOL` | EWMA (HL=2) ÷ vol |
| `EWMA_HL5_DIV_VOL` | EWMA (HL=5) ÷ vol |

---

### 13. Signed-Volume Momentum (`_svol_momentum`)

Signed-volume sums at multiple horizons reveal conviction behind price moves.

| Feature | Description |
|---------|-------------|
| `SVOL_SUM_{SHORT,MED5,MED10,LONG}` | Net signed-volume at each horizon |
| `XS_RANK_SVOL_SUM_{SHORT,MED5,MED10,LONG}` | Cross-sectional rank of each sum |

---

### 14. Return–Volume Interaction (`_ret_vol_interaction`)

Captures whether price moved *with* or *against* order flow.

| Feature | Description |
|---------|-------------|
| `RET_VOL_DOT` | Dot product of returns and volumes over 20 days; >0 = continuation, <0 = divergence |
| `RET_VOL_DOT_5` | Same but over the most recent 5 days |
| `RET_VOL_CORR` | Pearson correlation between returns and volumes (scale-free) |

---

### 15. Volume Spike (`_vol_spike`)

Detects unusually high volume days — a reversal on a spike is a stronger signal than one on quiet tape.

| Feature | Description |
|---------|-------------|
| `SVOL_SPIKE_Z` | Z-score of today's absolute volume vs its 20-day norm |
| `SVOL_ABS_MEAN_20` | 20-day mean absolute volume |
| `SVOL_ABS_RECENT` | Today's absolute volume |
| `RET1_X_SPIKE` | `SIGNED_VOLUME_1 × SVOL_SPIKE_Z` (interaction term) |

---

### 16. Within-GROUP Cross-Sectional Features (`_group_xs`)

The four strategy families (`GROUP`) have different return dynamics. Ranking within a family is more informative than ranking against the full panel.

| Feature | Description |
|---------|-------------|
| `GROUP_ENC` | Integer label encoding of `GROUP` (for LightGBM) |
| `GROUP_XS_DEMEAN_{RET1,RET5,RET20}` | Return minus within-GROUP mean |
| `GROUP_XS_RANK_{RET1,RET5,RET20}` | Percentile rank within GROUP |
| `GROUP_MEAN_RET1` | GROUP-level mean of `RET_1` (regime context) |
| `GROUP_STD_RET1` | GROUP-level std of `RET_1` |

---

### 17. Turnover Interactions (`_turnover`)

`MEDIAN_DAILY_TURNOVER` is a static liquidity proxy. Multiplying/dividing by vol or volume lets the model learn how liquidity modulates flow signals.

| Feature | Description |
|---------|-------------|
| `TURN_X_VOL` | Turnover × 20-day vol |
| `TURN_DIV_VOL` | Turnover ÷ 20-day vol |
| `TURN_X_SVOL1` | Turnover × absolute signed volume on day 1 |
| `TURN_X_SPIKE` | Turnover × `SVOL_SPIKE_Z` *(added only if `_vol_spike` has run)* |

---

## Full Usage Example (notebook)

```python
import pandas as pd
from all_features import build_features, apply_target_encoding
import lightgbm as lgbm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

# Load data
X_train = pd.read_csv('../data/X_train.csv', index_col='ROW_ID')
X_test  = pd.read_csv('../data/X_test.csv',  index_col='ROW_ID')
y_train = pd.read_csv('../data/y_train.csv', index_col='ROW_ID')

RET_cols = [f'RET_{i}'           for i in range(1, 21)]
VOL_cols = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

# Build features (benchmark + all engineered)
X_train, features = build_features(X_train, RET_cols, VOL_cols)
X_test,  _        = build_features(X_test,  RET_cols, VOL_cols)
features          = features + ['ALLOC_ENC']   # placeholder; filled per fold

# 8-fold CV (always split on TS, not rows)
train_dates = X_train['TS'].unique()
scores = []

for fold, (tr_idx, val_idx) in enumerate(
        KFold(n_splits=8, shuffle=True, random_state=0).split(train_dates)):

    tr_mask  = X_train['TS'].isin(train_dates[tr_idx])
    val_mask = X_train['TS'].isin(train_dates[val_idx])
    y_tr     = y_train.loc[tr_mask,  'target']
    y_val    = y_train.loc[val_mask, 'target']

    # Fit target encoding on training split only
    X_tr  = apply_target_encoding(X_train.loc[tr_mask],  y_tr, X_train.loc[tr_mask])[features].fillna(0)
    X_val = apply_target_encoding(X_train.loc[tr_mask],  y_tr, X_train.loc[val_mask])[features].fillna(0)

    model = lgbm.train(
        {'objective': 'mse', 'metric': 'mse', 'learning_rate': 0.01,
         'max_depth': 3, 'seed': 42, 'verbosity': -1},
        lgbm.Dataset(X_tr, label=y_tr.values),
        num_boost_round=500
    )
    preds = model.predict(X_val.values)
    acc   = accuracy_score((y_val > 0).astype(int), (preds > 0).astype(int))
    scores.append(acc)
    print(f"Fold {fold+1}: {acc*100:.2f}%")

print(f"CV Accuracy: {np.mean(scores)*100:.2f}% ± {np.std(scores)*100:.2f}%")
```

---

## Important Constraints

| Rule | Reason |
|------|--------|
| Always split CV on `TS` (dates), never on rows | Multiple strategies share the same date; row-based splits leak cross-allocation information |
| Call `apply_target_encoding()` inside each CV fold | Fitting on the full training set before folding leaks target information into the validation split |
| Call `build_features()` on `X_test` as well | Features are computed per-row; test set needs the same columns |
| `_benchmark()` always runs first inside `build_features()` | Several downstream functions depend on `AVERAGE_PERF_*` and `STD_PERF_20` already existing |
