# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Binary classification competition in systematic trading. Given a 20-day history of asset allocation strategy behaviour, predict whether it will make or lose money **next day** (1 = positive return, 0 = negative return). Evaluation metric is **accuracy** (sign prediction only).

## Running Notebooks

Notebooks are run with Jupyter. From the project root:

```bash
# Start Jupyter
jupyter notebook

# Or run a notebook headlessly
jupyter nbconvert --to notebook --execute data/benchmark_submission.ipynb
```

The benchmark notebook at `data/benchmark_submission.ipynb` reads CSVs relative to its own directory (`data/`). The research notebook at `code/researcher.ipynb` should also load data from `../data/`.

## Data

All data lives in `data/`. Files must be loaded with `ROW_ID` as index:

```python
X_train = pd.read_csv('X_train.csv', index_col='ROW_ID')
X_test  = pd.read_csv('X_test.csv',  index_col='ROW_ID')
y_train = pd.read_csv('y_train.csv', index_col='ROW_ID')
```

Key columns: `TS` (anonymized timestamp), `ALLOCATION` (strategy ID), `RET_1`–`RET_20` (returns, most recent = RET_1), `SIGNED_VOLUME_1`–`SIGNED_VOLUME_20`, `MEDIAN_DAILY_TURNOVER`, `GROUP` (allocation family). Training target is `y_train['target']` (raw return; threshold at 0 for label).

## Cross-Validation Rule

**Always split on `TS` (dates), never on rows.** Splitting on rows leaks future cross-allocation information because multiple allocations share the same timestamp.

```python
train_dates = X_train['TS'].unique()
splits = KFold(n_splits=8, shuffle=True, random_state=0).split(train_dates)

for train_date_ids, val_date_ids in splits:
    local_train_mask = X_train['TS'].isin(train_dates[train_date_ids])
    local_val_mask   = X_train['TS'].isin(train_dates[val_date_ids])
```

## Benchmark Approach

The benchmark (`data/benchmark_submission.ipynb`) uses LightGBM with `objective: mse` (regression, then threshold at 0) and achieves ~52.22% CV accuracy. Key engineered features: rolling mean returns over windows [3,5,10,15,20], cross-allocation mean returns per TS, rolling return std, cross-allocation std.

## Modelling Notes

- Dates are anonymized and shuffled — no guaranteed temporal continuity between TS values
- Signal is extremely noisy; 51–52% is meaningful
- `GROUP` column: consider training separate models per group or adding group-level normalisation
- Submission format: `ROW_ID, TARGET` where TARGET ∈ {0, 1}