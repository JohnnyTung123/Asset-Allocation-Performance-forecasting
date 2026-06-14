import numpy as np


def add_svol_momentum_features(df, vol_cols):
    """Signed-volume sums at multiple horizons + cross-sectional ranks."""
    features = []
    for horizon, name in [(3, 'SHORT'), (5, 'MED5'), (10, 'MED10'), (20, 'LONG')]:
        col = f'SVOL_SUM_{name}'
        df[col] = df[vol_cols[:horizon]].sum(axis=1)
        features.append(col)
        rank_col = f'XS_RANK_SVOL_SUM_{name}'
        df[rank_col] = df.groupby('TS')[col].rank(pct=True)
        features.append(rank_col)
    return df, features


def add_ret_vol_interaction_features(df, ret_cols, vol_cols):
    """Return–volume correlation and dot product over the 20-day window.

    Positive dot product = price moved with flow (conviction/continuation).
    Negative dot product = price moved against flow (divergence/reversal signal).
    """
    features = []
    rets = df[ret_cols].values   # shape (n, 20)
    vols = df[vol_cols].values

    # Row-wise dot product: sum(RET_k * SIGNED_VOLUME_k)
    df['RET_VOL_DOT'] = (rets * vols).sum(axis=1)
    features.append('RET_VOL_DOT')

    # Row-wise Pearson correlation between RET_k and SIGNED_VOLUME_k
    ret_dm  = rets - rets.mean(axis=1, keepdims=True)
    vol_dm  = vols - vols.mean(axis=1, keepdims=True)
    ret_std = rets.std(axis=1, keepdims=True)
    vol_std = vols.std(axis=1, keepdims=True)
    denom   = (ret_std * vol_std * rets.shape[1])
    denom   = np.where(denom == 0, np.nan, denom)
    df['RET_VOL_CORR'] = (ret_dm * vol_dm).sum(axis=1) / denom.squeeze()
    features.append('RET_VOL_CORR')

    # Short-window (5-day) version — more sensitive to recent flow
    df['RET_VOL_DOT_5'] = (rets[:, :5] * vols[:, :5]).sum(axis=1)
    features.append('RET_VOL_DOT_5')

    return df, features


def add_vol_spike_features(df, vol_cols):
    """Recent volume magnitude vs 20-day norm (z-score spike detection)."""
    features = []
    abs_vols = df[vol_cols].abs().values   # shape (n, 20)

    mean_abs = abs_vols.mean(axis=1)
    std_abs  = abs_vols.std(axis=1)
    std_abs  = np.where(std_abs == 0, np.nan, std_abs)

    # Z-score of most recent absolute volume vs its own 20-day window
    df['SVOL_SPIKE_Z']    = (abs_vols[:, 0] - mean_abs) / std_abs
    df['SVOL_ABS_MEAN_20'] = mean_abs
    df['SVOL_ABS_RECENT']  = abs_vols[:, 0]
    features += ['SVOL_SPIKE_Z', 'SVOL_ABS_MEAN_20', 'SVOL_ABS_RECENT']

    # Interaction: reversal on a spike vs quiet tape
    # STD_REVERSAL comes from Tip 2; use RET_1 directly here so Tip 4 is self-contained
    df['RET1_X_SPIKE'] = df[vol_cols[0]].abs() * df[vol_cols[0]].apply(np.sign) * df['SVOL_SPIKE_Z'].fillna(0)
    features.append('RET1_X_SPIKE')

    return df, features


def add_features(df, ret_cols, vol_cols):
    all_features = []
    df, f = add_svol_momentum_features(df, vol_cols)
    all_features += f
    df, f = add_ret_vol_interaction_features(df, ret_cols, vol_cols)
    all_features += f
    df, f = add_vol_spike_features(df, vol_cols)
    all_features += f
    return df, all_features
