# Data Documentation

## Task

Binary classification: given 20 days of history for an asset allocation strategy, predict whether it will make or lose money **the next day**.

- **Label 1** — positive return (profit)
- **Label 0** — negative or zero return (loss)
- **Evaluation metric** — accuracy (sign prediction only)

---

## Files

| File | Rows | Description |
|---|---|---|
| `X_train.csv` | 527,073 | Training features |
| `y_train.csv` | 527,073 | Training labels (raw returns) |
| `X_test.csv` | 31,870 | Test features (no labels) |
| `sample_submission.csv` | 31,870 | Submission template |

All files use `ROW_ID` as the index column.

```python
X_train = pd.read_csv('X_train.csv', index_col='ROW_ID')
X_test  = pd.read_csv('X_test.csv',  index_col='ROW_ID')
y_train = pd.read_csv('y_train.csv', index_col='ROW_ID')
```

---

## Columns

### Identifiers

| Column | Type | Description |
|---|---|---|
| `TS` | string | Anonymized date (e.g. `DATE_0001`). 2,522 unique dates in train. |
| `ALLOCATION` | string | Strategy ID (e.g. `ALLOCATION_01`). 278 unique strategies. |
| `GROUP` | int | Strategy family: one of {1, 2, 3, 4}. |

> Dates are anonymized and **not guaranteed to be temporally contiguous** — there is no meaningful ordering between `DATE_0001` and `DATE_0002`.

---

### Return History — `RET_1` to `RET_20`

20 columns of daily returns for the strategy, looking **backwards** from the prediction date.

| Column | Meaning |
|---|---|
| `RET_1` | Return **1 day ago** (most recent) |
| `RET_2` | Return 2 days ago |
| ... | ... |
| `RET_20` | Return 20 days ago (oldest) |

- Units: raw return (e.g. `0.002` = +0.2%)
- Can contain missing values (~404k NaNs across all float columns combined)

---

### Signed Volume History — `SIGNED_VOLUME_1` to `SIGNED_VOLUME_20`

20 columns of signed trading volume, with the same lag convention as returns.

| Column | Meaning |
|---|---|
| `SIGNED_VOLUME_1` | Signed volume **1 day ago** (most recent) |
| ... | ... |
| `SIGNED_VOLUME_20` | Signed volume 20 days ago (oldest) |

- Positive value = net buying pressure; negative = net selling pressure

---

### Other Features

| Column | Type | Description |
|---|---|---|
| `MEDIAN_DAILY_TURNOVER` | float | Median daily turnover of the strategy (normalised) |

---

## Target (`y_train`)

| Property | Value |
|---|---|
| Column | `target` |
| Type | Raw continuous return (float) |
| Range | −0.052 to +0.052 |
| Mean | ~0.000035 (near-zero, unbiased) |
| Std | 0.003106 |
| % Positive | 50.71% (267,323 / 527,073) |

The target is a **raw return**, not a binary label. To get the classification label, threshold at 0:

```python
y_label = (y_train['target'] > 0).astype(int)
```

---

## Dataset Structure

Each row represents one **strategy × date** observation. On a given date, many strategies are observed simultaneously — this panel structure is important for cross-validation.

```
DATE_0001 × ALLOCATION_01  →  row
DATE_0001 × ALLOCATION_02  →  row
DATE_0001 × ALLOCATION_03  →  row
...
DATE_0002 × ALLOCATION_01  →  row
...
```

- 2,522 unique dates × ~209 strategies per date on average = 527,073 rows

---

## Cross-Validation Rule

**Always split on `TS` (dates), never on rows.** Splitting on rows leaks future information because multiple strategies share the same timestamp.

```python
train_dates = X_train['TS'].unique()
splits = KFold(n_splits=8, shuffle=True, random_state=0).split(train_dates)

for train_date_ids, val_date_ids in splits:
    local_train_mask = X_train['TS'].isin(train_dates[train_date_ids])
    local_val_mask   = X_train['TS'].isin(train_dates[val_date_ids])
```

---

## Submission Format

The submission file must have columns `ROW_ID` and `prediction`, where `prediction ∈ {0, 1}`.

```
ROW_ID,prediction
527073,1
527074,0
...
```

---

## Key Modelling Notes

- Signal is **extremely noisy** — 51–52% CV accuracy is meaningful above the 50% baseline
- The `GROUP` column groups strategies into 4 families; consider group-level normalisation or separate models per group
- `MEDIAN_DAILY_TURNOVER` is a static strategy property (same across dates for a given strategy)
- Missing values should be filled (e.g. `fillna(0)`) before model training