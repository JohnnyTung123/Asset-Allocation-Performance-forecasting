import numpy as np


def add_distribution_features(df, ret_cols):
    """Downside vol, skew, fraction/count of positive days."""
    features = []
    rets = df[ret_cols].values  # col 0 = RET_1 (most recent)

    # Downside vol: std of returns on negative days only
    neg = np.where(rets < 0, rets, np.nan)
    df['DOWNSIDE_VOL'] = np.nanstd(neg, axis=1)
    features.append('DOWNSIDE_VOL')

    # Skew of the 20-day return distribution
    df['RET_SKEW'] = df[ret_cols].skew(axis=1)
    features.append('RET_SKEW')

    # Fraction of positive days across full window
    df['FRAC_POS_20'] = (rets > 0).mean(axis=1)
    features.append('FRAC_POS_20')

    # Count of positive days in most recent 5
    df['COUNT_POS_5'] = (rets[:, :5] > 0).sum(axis=1)
    features.append('COUNT_POS_5')

    return df, features


def add_streak_features(df, ret_cols):
    """Longest up/down streak and sign summary of last 3 days."""
    features = []
    rets = df[ret_cols].values  # col 0 = RET_1 (most recent)

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
    features += ['LONGEST_UP_STREAK', 'LONGEST_DOWN_STREAK']

    # Count positives in most recent 3 days
    df['POS_LAST_3'] = (rets[:, :3] > 0).sum(axis=1)
    features.append('POS_LAST_3')

    return df, features


def add_drawdown_features(df, ret_cols):
    """Max drawdown and current distance from running peak over the window."""
    features = []

    # Reverse to chronological order: RET_20 (oldest) → RET_1 (most recent)
    rets_chron = df[ret_cols[::-1]].values
    cum        = np.cumsum(rets_chron, axis=1)
    peak       = np.maximum.accumulate(cum, axis=1)
    drawdown   = peak - cum

    df['MAX_DRAWDOWN']   = drawdown.max(axis=1)
    df['DIST_FROM_PEAK'] = drawdown[:, -1]   # distance at end of window (most recent)
    features += ['MAX_DRAWDOWN', 'DIST_FROM_PEAK']

    return df, features


def add_vol_interaction_features(df, ret_cols):
    """Vol-adjusted momentum. Uses benchmark AVERAGE_PERF_* always;
    also interacts with Tip-2 features if present in df."""
    features = []
    vol = df['STD_PERF_20'].replace(0, np.nan)

    for i in [3, 5, 10, 20]:
        name = f'AVERAGE_PERF_{i}_DIV_VOL'
        df[name] = df[f'AVERAGE_PERF_{i}'] / vol
        features.append(name)

    for col in ['MOM_SUM_SHORT', 'MOM_SUM_MED5', 'MOM_SUM_LONG',
                'EWMA_HL2', 'EWMA_HL5']:
        if col in df.columns:
            name = f'{col}_DIV_VOL'
            df[name] = df[col] / vol
            features.append(name)

    return df, features


def add_features(df, ret_cols, vol_cols):
    all_features = []
    df, f = add_distribution_features(df, ret_cols)
    all_features += f
    df, f = add_streak_features(df, ret_cols)
    all_features += f
    df, f = add_drawdown_features(df, ret_cols)
    all_features += f
    df, f = add_vol_interaction_features(df, ret_cols)
    all_features += f
    return df, all_features
