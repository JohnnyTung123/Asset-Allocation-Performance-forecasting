# Trust or Short? — Predicting Daily Asset Allocation Performance

## Overview

This is a **binary classification** competition in systematic trading.

Each day, multiple asset allocation strategies (portfolios) are rebalanced. Given a 20-day history of each strategy's behaviour, the goal is to predict whether it will make or lose money the **next day**.

- Predict **1** → trust the allocation (go long)
- Predict **0** → short the allocation (do the opposite)

Only the **direction** matters — not the magnitude of the return.

---

## The Core Question

> "Looking at how a strategy behaved over the last 20 days — will it go up or down tomorrow?"

---

## Key Concepts

### Allocation
A trading strategy / recipe that assigns weights to a set of assets every day. Weights can be positive (long) or negative (short). All absolute weights sum to 1.

### Group
An anonymized category label for allocations built using similar logic. Allocations within the same group tend to behave similarly.

### Turnover
How much the strategy reshuffles its portfolio from one day to the next. Low turnover = stable positions. High turnover = completely different portfolio each day.

### Signed Volume
A weighted combination of market volumes, respecting the direction of each position. Positive signed volume means the strategy is riding heavily-traded long positions.

---

## Dataset Structure

Each row = one **(date, allocation)** pair.

| Column | Description |
|--------|-------------|
| `DATE` | Anonymized, shuffled timestamp |
| `ALLOCATION` | Strategy identifier |
| `RET_1` … `RET_20` | Daily returns over the last 20 days (most recent = RET_1) |
| `SIGNED_VOLUME_1` … `SIGNED_VOLUME_20` | Signed volume over the last 20 days |
| `MEDIAN_DAILY_TURNOVER` | Median turnover over the last 20 days |
| `GROUP` | Anonymized allocation family |
| `TARGET` | Next-day return (training only — you predict its sign) |

### File Summary

| File | Description |
|------|-------------|
| `X_train.csv` | Training features (527,073 rows) |
| `y_train.csv` | Training targets |
| `X_test.csv` | Test features (31,870 rows) |
| `sample_submission.csv` | Correct submission format |
| `benchmark_submission.ipynb` | Benchmark notebook |

---

## Evaluation Metric

**Accuracy** — the fraction of rows where the predicted sign matches the true sign.

```
Accuracy = (number of correct sign predictions) / (total rows)
```

Where:
- `sign(x) = 1` if x > 0, else `0`
- Only the sign is evaluated — not the magnitude

### Example

| Row | True Return | Predicted Return | Correct? |
|-----|-------------|-----------------|----------|
| 1 | +0.5% | +0.3% | ✅ |
| 2 | -0.2% | -0.1% | ✅ |
| 3 | +0.1% | -0.4% | ❌ |
| 4 | -0.3% | +0.2% | ❌ |
| 5 | +0.8% | +0.1% | ✅ |

**Accuracy = 3/5 = 0.60**

---

## Benchmark

The provided benchmark uses **LightGBM** with engineered features including:

- Rolling average returns (multiple windows)
- Cross-allocation average returns
- Rolling volatility per allocation
- Cross-allocation average volatility

**Benchmark public leaderboard score: 0.5079** (barely above random)

---

## Submission Format

Each row must contain a `ROW_ID` and a prediction of `1` (positive return) or `0` (negative return).

```
ROW_ID,TARGET
1,1
2,0
3,1
...
```

---

## Modelling Notes

- Dates are **anonymized and shuffled** — no guaranteed continuity between DATE_0001 and DATE_0002
- The signal is extremely noisy — even 51–52% accuracy is meaningful
- Feature engineering is likely more impactful than model complexity
- The `GROUP` column may warrant training separate models per group
- Use **grouped cross-validation** to avoid leakage across allocations

---

## Suggested Feature Ideas

- Momentum: average of recent returns (are they trending up?)
- Mean reversion: are returns reversing after a streak?
- Volatility: standard deviation of the 20-day return window
- Volume/return divergence: is price moving against volume?
- Per-group statistics: normalise features within each GROUP
