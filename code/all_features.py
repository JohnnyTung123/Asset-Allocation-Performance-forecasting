"""
all_features.py
---------------
Complete feature engineering pipeline + model training.

Can be used two ways:
  1. Run directly:  python all_features.py
  2. Import into a notebook:
       from all_features import build_features, apply_target_encoding

Public API
----------
build_features(df, ret_cols, vol_cols) -> (df, feature_names)
    Adds benchmark + all engineered features (excluding ALLOC_ENC).
    Call on both X_train and X_test before CV.

apply_target_encoding(df_fit, y_fit, df_transform) -> df_transform
    Smoothed ALLOC target encoding. Must be called inside each CV fold
    to avoid leakage, and on full training data for the final model.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import lightgbm as lgbm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Benchmark features
# ---------------------------------------------------------------------------

def _benchmark(df, ret_cols, vol_cols):
    """Core rolling features used as the baseline.

    - AVERAGE_PERF_{3,5,10,15,20}: row-wise rolling mean return over each window
    - ALLOCATIONS_AVERAGE_PERF_*: cross-allocation mean of the above per date,
      giving the model a sense of how the whole panel is trending
    - STD_PERF_20: row-wise 20-day realised vol (used by downstream functions)
    - ALLOCATIONS_STD_PERF_20: cross-allocation mean of vol per date

    NOTE: several downstream functions (_xs_ranks, _vol_interactions, etc.)
    depend on AVERAGE_PERF_* and STD_PERF_20 already existing, so this must
    always run first inside build_features().
    """
    features = ret_cols + vol_cols + ['MEDIAN_DAILY_TURNOVER']
    for i in [3, 5, 10, 15, 20]:
        df[f'AVERAGE_PERF_{i}'] = df[ret_cols[:i]].mean(axis=1)
        df[f'ALLOCATIONS_AVERAGE_PERF_{i}'] = (
            df.groupby('TS')[f'AVERAGE_PERF_{i}'].transform('mean')
        )
        features += [f'AVERAGE_PERF_{i}', f'ALLOCATIONS_AVERAGE_PERF_{i}']
    df['STD_PERF_20'] = df[ret_cols].std(axis=1)
    df['ALLOCATIONS_STD_PERF_20'] = df.groupby('TS')['STD_PERF_20'].transform('mean')
    features += ['STD_PERF_20', 'ALLOCATIONS_STD_PERF_20']
    return df, features


# ---------------------------------------------------------------------------
# Cross-sectional normalisation
# ---------------------------------------------------------------------------

def _xs_zscore(df, ret_cols, vol_cols):
    """Within-date z-score of every return and volume lag.

    Subtracts the cross-sectional mean and divides by std on each date,
    stripping out the common market factor shared by all strategies that day.
    The residual is strategy-specific signal.
    """
    features = []
    for col in ret_cols + vol_cols:
        grp  = df.groupby('TS')[col]
        mean = grp.transform('mean')
        std  = grp.transform('std').replace(0, np.nan)
        name = f'XS_Z_{col}'
        df[name] = (df[col] - mean) / std
        features.append(name)
    return df, features


def _xs_ranks(df, ret_cols, vol_cols):
    """Percentile rank of each strategy's momentum within its date's cross-section.

    Ranks are scale-free and regime-robust — the model sees where a strategy
    sits relative to its peers today rather than its raw return level.
    Computed for return horizons 1/3/5/10/20 and signed-volume horizons 1/5.
    """
    features = []
    horizon_col = {
        1: ret_cols[0],
        3: 'AVERAGE_PERF_3',
        5: 'AVERAGE_PERF_5',
        10: 'AVERAGE_PERF_10',
        20: 'AVERAGE_PERF_20',
    }
    for i, col in horizon_col.items():
        name = f'XS_RANK_RET_{i}'
        df[name] = df.groupby('TS')[col].rank(pct=True)
        features.append(name)
    for i in [1, 5]:
        src  = f'SVOL_MEAN_{i}'
        df[src] = df[vol_cols[:i]].mean(axis=1)
        name = f'XS_RANK_SVOL_{i}'
        df[name] = df.groupby('TS')[src].rank(pct=True)
        features.append(name)
    return df, features


def _date_context(df, ret_cols, vol_cols):
    """Date-level aggregate context features.

    Tells the model how unusual today's cross-section is:
    - XS_DISP_*: spread (std) of returns/volume across all strategies on that date
    - XS_MEAN_RET1: the common market move on the most recent day
    - XS_PCT_RET1: this strategy's percentile within today's cross-section
    """
    df['XS_DISP_RET1']  = df.groupby('TS')[ret_cols[0]].transform('std')
    df['XS_DISP_RET5']  = df.groupby('TS')['AVERAGE_PERF_5'].transform('std')
    df['XS_MEAN_RET1']  = df.groupby('TS')[ret_cols[0]].transform('mean')
    df['XS_PCT_RET1']   = df.groupby('TS')[ret_cols[0]].rank(pct=True)
    df['XS_DISP_SVOL1'] = df.groupby('TS')[vol_cols[0]].transform('std')
    features = ['XS_DISP_RET1', 'XS_DISP_RET5', 'XS_MEAN_RET1',
                'XS_PCT_RET1', 'XS_DISP_SVOL1']
    return df, features


# ---------------------------------------------------------------------------
# Reversal vs. momentum separated by horizon
# ---------------------------------------------------------------------------

def _ewma_weights(half_life, n=20):
    alpha = 1 - np.exp(-np.log(2) / half_life)
    w = np.array([(1 - alpha) ** (k - 1) for k in range(1, n + 1)])
    return w / w.sum()


def _momentum(df, ret_cols):
    """Return sums at short (3), medium (5, 10), and long (20) horizons.

    Kept separate so the model can learn that short and long momentum
    behave differently — short-term often reverses, longer-term can trend.
    Sum (not mean) preserves the scale signal across window lengths.
    """
    features = []
    for horizon, name in [(3, 'SHORT'), (5, 'MED5'), (10, 'MED10'), (20, 'LONG')]:
        col = f'MOM_SUM_{name}'
        df[col] = df[ret_cols[:horizon]].sum(axis=1)
        features.append(col)
    return df, features


def _acceleration(df):
    """Differences between momentum horizons (momentum acceleration).

    Positive value = recent returns stronger than longer-term average (surging).
    Negative value = recent returns weaker than longer-term average (fading/reversing).
    Gives the model explicit information about the direction of change in momentum.
    """
    features = []
    for col_a, col_b, name in [
        ('MOM_SUM_SHORT', 'MOM_SUM_MED5',  'ACCEL_S_M5'),
        ('MOM_SUM_MED5',  'MOM_SUM_MED10', 'ACCEL_M5_M10'),
        ('MOM_SUM_MED10', 'MOM_SUM_LONG',  'ACCEL_M10_L'),
        ('MOM_SUM_SHORT', 'MOM_SUM_LONG',  'ACCEL_S_L'),
    ]:
        df[name] = df[col_a] - df[col_b]
        features.append(name)
    return df, features


def _ewma(df, ret_cols, half_lives=(1, 2, 3, 5, 7, 10, 15)):
    """Exponentially weighted mean of returns at multiple decay rates.

    Recent days are weighted more than older days. Seven half-lives give the
    model a dense spectrum from 'almost only RET_1' (HL=1) to 'near equal-weight
    20-day mean' (HL=15), letting it pick the decay rate that best fits the signal.
    """
    features = []
    for hl in half_lives:
        w   = _ewma_weights(hl, n=len(ret_cols))
        col = f'EWMA_HL{hl}'
        df[col] = df[ret_cols].values @ w
        features.append(col)
    return df, features


def _reversal(df, ret_cols):
    """Standardised reversal: RET_1 divided by the strategy's own 20-day vol.

    Measures how extreme the most recent move was relative to this strategy's
    normal range. A large absolute value signals an unusual move that is more
    likely to mean-revert. More informative than raw RET_1 alone.
    """
    vol20 = df[ret_cols].std(axis=1).replace(0, np.nan)
    df['STD_REVERSAL']     = df[ret_cols[0]] / vol20
    df['ABS_STD_REVERSAL'] = df['STD_REVERSAL'].abs()
    return df, ['STD_REVERSAL', 'ABS_STD_REVERSAL']


# ---------------------------------------------------------------------------
# Distribution-shape features
# ---------------------------------------------------------------------------

def _distribution(df, ret_cols):
    """Shape statistics of the 20-day return distribution.

    - DOWNSIDE_VOL: std of negative return days only (asymmetric risk measure)
    - RET_SKEW: tail direction — positive skew means rare large gains
    - FRAC_POS_20: hit rate over the full window (consistency of direction)
    - COUNT_POS_5: hit rate in the most recent 5 days (near-term consistency)
    """
    rets = df[ret_cols].values
    neg  = np.where(rets < 0, rets, np.nan)
    df['DOWNSIDE_VOL'] = np.nanstd(neg, axis=1)
    df['RET_SKEW']     = df[ret_cols].skew(axis=1)
    df['FRAC_POS_20']  = (rets > 0).mean(axis=1)
    df['COUNT_POS_5']  = (rets[:, :5] > 0).sum(axis=1)
    return df, ['DOWNSIDE_VOL', 'RET_SKEW', 'FRAC_POS_20', 'COUNT_POS_5']


def _streaks(df, ret_cols):
    """Longest consecutive up/down streak and recent sign summary.

    Streaks capture persistence of direction which simple means miss.
    A long up streak suggests momentum; a long down streak suggests distress.
    POS_LAST_3 is a quick read on whether the very recent tape is positive.
    """
    rets = df[ret_cols].values

    def _max_consec(arr):
        max_s = cur = 0
        for v in arr:
            cur = cur + 1 if v else 0
            if cur > max_s:
                max_s = cur
        return max_s

    up   = (rets > 0).astype(np.uint8)
    down = (rets < 0).astype(np.uint8)
    df['LONGEST_UP_STREAK']   = np.apply_along_axis(_max_consec, 1, up)
    df['LONGEST_DOWN_STREAK'] = np.apply_along_axis(_max_consec, 1, down)
    df['POS_LAST_3']          = (rets[:, :3] > 0).sum(axis=1)
    return df, ['LONGEST_UP_STREAK', 'LONGEST_DOWN_STREAK', 'POS_LAST_3']


def _drawdown(df, ret_cols):
    """Max drawdown and current distance from the 20-day running peak.

    Returns are reversed to chronological order (oldest first) before
    computing cumulative sums. MAX_DRAWDOWN captures the worst trough over
    the window; DIST_FROM_PEAK is how far below the high-water mark the
    strategy currently sits — a path-dependent risk indicator.
    """
    rets_chron = df[ret_cols[::-1]].values
    cum        = np.cumsum(rets_chron, axis=1)
    peak       = np.maximum.accumulate(cum, axis=1)
    drawdown   = peak - cum
    df['MAX_DRAWDOWN']   = drawdown.max(axis=1)
    df['DIST_FROM_PEAK'] = drawdown[:, -1]
    return df, ['MAX_DRAWDOWN', 'DIST_FROM_PEAK']


def _vol_interactions(df):
    """Vol-adjusted momentum (Sharpe-like ratio per horizon).

    Divides momentum features by 20-day realised vol so the model can
    distinguish a strong move in a low-vol regime from the same move in a
    high-vol regime — the former is more informative. Also applied to EWMA
    features if they exist (i.e. _ewma has already been called).
    """
    features = []
    vol = df['STD_PERF_20'].replace(0, np.nan)
    for i in [3, 5, 10, 20]:
        name = f'AVERAGE_PERF_{i}_DIV_VOL'
        df[name] = df[f'AVERAGE_PERF_{i}'] / vol
        features.append(name)
    for col in ['MOM_SUM_SHORT', 'MOM_SUM_MED5', 'MOM_SUM_LONG', 'EWMA_HL2', 'EWMA_HL5']:
        if col in df.columns:
            name = f'{col}_DIV_VOL'
            df[name] = df[col] / vol
            features.append(name)
    return df, features


# ---------------------------------------------------------------------------
# Volume / order-flow features
# ---------------------------------------------------------------------------

def _svol_momentum(df, vol_cols):
    """Signed-volume sums at multiple horizons plus their cross-sectional ranks.

    Returns are symmetric noise; signed volume reveals conviction behind each move.
    Positive sum = net buying pressure; negative = net selling pressure.
    Cross-sectional rank normalises across strategies on the same date.
    """
    features = []
    for horizon, name in [(3, 'SHORT'), (5, 'MED5'), (10, 'MED10'), (20, 'LONG')]:
        col      = f'SVOL_SUM_{name}'
        rank_col = f'XS_RANK_SVOL_SUM_{name}'
        df[col]      = df[vol_cols[:horizon]].sum(axis=1)
        df[rank_col] = df.groupby('TS')[col].rank(pct=True)
        features += [col, rank_col]
    return df, features


def _ret_vol_interaction(df, ret_cols, vol_cols):
    """Return–volume dot product and correlation over the 20-day window.

    Captures whether price moved *with* or *against* order flow:
    - RET_VOL_DOT > 0: price moved with flow (conviction, signals continuation)
    - RET_VOL_DOT < 0: price moved against flow (divergence, signals reversal)
    - RET_VOL_CORR: Pearson correlation version, scale-free
    - RET_VOL_DOT_5: same as dot product but on the most recent 5 days only
    """
    rets = df[ret_cols].values
    vols = df[vol_cols].values
    df['RET_VOL_DOT']   = (rets * vols).sum(axis=1)
    df['RET_VOL_DOT_5'] = (rets[:, :5] * vols[:, :5]).sum(axis=1)
    ret_dm  = rets - rets.mean(axis=1, keepdims=True)
    vol_dm  = vols - vols.mean(axis=1, keepdims=True)
    denom   = rets.std(axis=1, keepdims=True) * vols.std(axis=1, keepdims=True) * rets.shape[1]
    denom   = np.where(denom == 0, np.nan, denom)
    df['RET_VOL_CORR'] = (ret_dm * vol_dm).sum(axis=1) / denom.squeeze()
    return df, ['RET_VOL_DOT', 'RET_VOL_DOT_5', 'RET_VOL_CORR']


def _vol_spike(df, vol_cols):
    """Volume spike detection: z-score of recent absolute volume vs its 20-day norm.

    A large SVOL_SPIKE_Z means today's volume is unusually high for this strategy.
    A reversal on a spike is a stronger signal than one on quiet tape — the
    RET1_X_SPIKE interaction term captures exactly that combination.
    """
    abs_vols = df[vol_cols].abs().values
    mean_abs = abs_vols.mean(axis=1)
    std_abs  = np.where(abs_vols.std(axis=1) == 0, np.nan, abs_vols.std(axis=1))
    df['SVOL_SPIKE_Z']     = (abs_vols[:, 0] - mean_abs) / std_abs
    df['SVOL_ABS_MEAN_20'] = mean_abs
    df['SVOL_ABS_RECENT']  = abs_vols[:, 0]
    df['RET1_X_SPIKE']     = (df[vol_cols[0]].abs()
                               * np.sign(df[vol_cols[0]])
                               * df['SVOL_SPIKE_Z'].fillna(0))
    return df, ['SVOL_SPIKE_Z', 'SVOL_ABS_MEAN_20', 'SVOL_ABS_RECENT', 'RET1_X_SPIKE']


# ---------------------------------------------------------------------------
# Group and strategy structure
# ---------------------------------------------------------------------------

def _group_xs(df, ret_cols):
    """Within-GROUP cross-sectional features and GROUP label encoding.

    The four strategy families likely have different return dynamics.
    Ranking a strategy within its own family (not the full panel) captures
    relative performance more meaningfully than a panel-wide rank.
    GROUP_ENC is an integer label for the GROUP categorical passed to LightGBM.
    """
    features = []
    df['GROUP_ENC'] = df['GROUP'].astype('category').cat.codes
    features.append('GROUP_ENC')
    for col, tag in [('RET_1', 'RET1'), ('AVERAGE_PERF_5', 'RET5'), ('AVERAGE_PERF_20', 'RET20')]:
        grp = df.groupby('GROUP')[col]
        df[f'GROUP_XS_DEMEAN_{tag}'] = df[col] - grp.transform('mean')
        df[f'GROUP_XS_RANK_{tag}']   = grp.rank(pct=True)
        features += [f'GROUP_XS_DEMEAN_{tag}', f'GROUP_XS_RANK_{tag}']
    df['GROUP_MEAN_RET1'] = df.groupby('GROUP')['RET_1'].transform('mean')
    df['GROUP_STD_RET1']  = df.groupby('GROUP')['RET_1'].transform('std')
    features += ['GROUP_MEAN_RET1', 'GROUP_STD_RET1']
    return df, features


def _turnover(df):
    """MEDIAN_DAILY_TURNOVER interactions with vol and volume features.

    Turnover is a static proxy for strategy liquidity and capacity.
    High-turnover strategies may react differently to vol or flow signals,
    so multiplying/dividing by turnover lets the model learn those interactions.
    TURN_X_SPIKE is added only if _vol_spike() has already run.
    """
    features = []
    turn = df['MEDIAN_DAILY_TURNOVER']
    vol  = df['STD_PERF_20'].replace(0, np.nan)
    df['TURN_X_VOL']   = turn * vol
    df['TURN_DIV_VOL'] = turn / vol
    df['TURN_X_SVOL1'] = turn * df['SIGNED_VOLUME_1'].abs()
    features += ['TURN_X_VOL', 'TURN_DIV_VOL', 'TURN_X_SVOL1']
    if 'SVOL_SPIKE_Z' in df.columns:
        df['TURN_X_SPIKE'] = turn * df['SVOL_SPIKE_Z'].fillna(0)
        features.append('TURN_X_SPIKE')
    return df, features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(df, ret_cols, vol_cols):
    """Build benchmark + all engineered features (excluding ALLOC_ENC).

    Returns (df_with_features, feature_names).
    Call on X_train and X_test before the CV loop.
    ALLOC_ENC is excluded here — use apply_target_encoding() inside each fold.
    """
    all_features = []

    # Benchmark (must run first — downstream functions depend on AVERAGE_PERF_* and STD_PERF_20)
    df, f = _benchmark(df, ret_cols, vol_cols)
    all_features += f

    # Cross-sectional normalisation
    df, f = _xs_zscore(df, ret_cols, vol_cols)
    all_features += f
    df, f = _xs_ranks(df, ret_cols, vol_cols)
    all_features += f
    df, f = _date_context(df, ret_cols, vol_cols)
    all_features += f

    # Reversal vs. momentum
    df, f = _momentum(df, ret_cols)
    all_features += f
    df, f = _acceleration(df)
    all_features += f
    df, f = _ewma(df, ret_cols)
    all_features += f
    df, f = _reversal(df, ret_cols)
    all_features += f

    # Distribution shape
    df, f = _distribution(df, ret_cols)
    all_features += f
    df, f = _streaks(df, ret_cols)
    all_features += f
    df, f = _drawdown(df, ret_cols)
    all_features += f
    df, f = _vol_interactions(df)
    all_features += f

    # Volume / order-flow
    df, f = _svol_momentum(df, vol_cols)
    all_features += f
    df, f = _ret_vol_interaction(df, ret_cols, vol_cols)
    all_features += f
    df, f = _vol_spike(df, vol_cols)
    all_features += f

    # Group and strategy structure
    df, f = _group_xs(df, ret_cols)
    all_features += f
    df, f = _turnover(df)
    all_features += f

    return df, all_features


def apply_target_encoding(df_fit, y_fit, df_transform, smoothing=10):
    """Smoothed ALLOC target encoding.

    Fit on df_fit/y_fit, apply to df_transform.
    Returns df_transform with 'ALLOC_ENC' column added.

    Usage in CV loop:
        X_tr  = apply_target_encoding(X_train[tr_mask], y_tr, X_train[tr_mask])
        X_val = apply_target_encoding(X_train[tr_mask], y_tr, X_train[val_mask])
    Usage for final model:
        X_train_enc = apply_target_encoding(X_train, y_train['target'], X_train)
        X_test_enc  = apply_target_encoding(X_train, y_train['target'], X_test)
    """
    global_mean = float(y_fit.mean())
    stats       = y_fit.groupby(df_fit['ALLOCATION']).agg(['mean', 'count'])
    encoded     = (
        (stats['count'] * stats['mean'] + smoothing * global_mean)
        / (stats['count'] + smoothing)
    )
    out = df_transform.copy()
    out['ALLOC_ENC'] = out['ALLOCATION'].map(encoded).fillna(global_mean)
    return out


# ---------------------------------------------------------------------------
# Full pipeline (run directly: python all_features.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    sns.set()

    # --- Config ---
    USE_TARGET_ENCODING = False
    LGBM_PARAMS = {
        'objective':     'binary',
        'metric':        'binary_logloss',
        'num_threads':   50,
        'seed':          42,
        'verbosity':     -1,
        'learning_rate': 1e-2,
        'max_depth':     3,
    }
    NUM_BOOST_ROUND = 2000
    N_SPLITS        = 8

    # --- Load data ---
    X_train = pd.read_csv('../data/X_train.csv', index_col='ROW_ID')
    X_test  = pd.read_csv('../data/X_test.csv',  index_col='ROW_ID')
    y_train = pd.read_csv('../data/y_train.csv', index_col='ROW_ID')
    sample_submission = pd.read_csv('../submission/sample_submission.csv', index_col='ROW_ID')

    RET_cols = [f'RET_{i}'           for i in range(1, 21)]
    VOL_cols = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

    # --- Feature engineering ---
    X_train, features = build_features(X_train, RET_cols, VOL_cols)
    X_test,  _        = build_features(X_test,  RET_cols, VOL_cols)

    if USE_TARGET_ENCODING:
        features = features + ['ALLOC_ENC']

    print(f"Total features: {len(features)}")

    # --- 8-fold CV (split on dates, not rows) ---
    train_dates = X_train['TS'].unique()
    scores, models = [], []

    splits = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0).split(train_dates)

    for fold, (tr_idx, val_idx) in enumerate(splits):
        tr_mask  = X_train['TS'].isin(train_dates[tr_idx])
        val_mask = X_train['TS'].isin(train_dates[val_idx])

        y_tr  = y_train.loc[tr_mask,  'target']
        y_val = y_train.loc[val_mask, 'target']

        if USE_TARGET_ENCODING:
            # Fit encoding on training split only to avoid leakage
            X_tr  = apply_target_encoding(X_train.loc[tr_mask], y_tr,
                                          X_train.loc[tr_mask])[features].fillna(0)
            X_val = apply_target_encoding(X_train.loc[tr_mask], y_tr,
                                          X_train.loc[val_mask])[features].fillna(0)
        else:
            X_tr  = X_train.loc[tr_mask,  features].fillna(0)
            X_val = X_train.loc[val_mask, features].fillna(0)

        model = lgbm.train(LGBM_PARAMS,
                           lgbm.Dataset(X_tr, label=y_tr.values),
                           num_boost_round=NUM_BOOST_ROUND)

        preds = model.predict(X_val.values, num_threads=LGBM_PARAMS['num_threads'])
        acc   = accuracy_score((y_val > 0).astype(int), (preds > 0).astype(int))

        models.append(model)
        scores.append(acc)
        print(f"Fold {fold+1} — Accuracy: {acc*100:.2f}%")

    mean = np.mean(scores) * 100
    std  = np.std(scores)  * 100
    print(f"\nAccuracy: {mean:.2f}% ± {std:.2f}%  [{mean-std:.2f} ; {mean+std:.2f}]")

    # --- Feature importance (top 30 by gain) ---
    importances = pd.DataFrame(
        [m.feature_importance(importance_type='gain') for m in models],
        columns=features
    )
    top30 = importances.mean().sort_values(ascending=False).head(30).index

    plt.figure(figsize=(10, 9))
    sns.barplot(data=importances[top30], orient='h',
                order=importances[top30].mean().sort_values(ascending=False).index)
    plt.title("Top-30 features by mean gain (8-fold)")
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()

    # --- Final model + submission ---
    if USE_TARGET_ENCODING:
        X_train_final = apply_target_encoding(X_train, y_train['target'], X_train)
        X_test_final  = apply_target_encoding(X_train, y_train['target'], X_test)
    else:
        X_train_final = X_train
        X_test_final  = X_test

    final_model = lgbm.train(
        LGBM_PARAMS,
        lgbm.Dataset(X_train_final[features].fillna(0), label=y_train['target'].values),
        num_boost_round=NUM_BOOST_ROUND
    )

    test_preds = final_model.predict(X_test_final[features].fillna(0).values)
    submission = pd.DataFrame(
        (test_preds > 0).astype(int),
        index=sample_submission.index,
        columns=['TARGET']
    )
    submission.to_csv('preds_all_features.csv')
    print("Saved preds_all_features.csv")
    print(submission['TARGET'].value_counts())
