import numpy as np
import pandas as pd


def add_group_xs_features(df, ret_cols):
    """Within-GROUP cross-sectional demean, rank, and group-level context.
    Each GROUP family may have different return dynamics, so ranking within
    the family is more informative than ranking against the full panel."""
    features = []

    # GROUP label-encoded as integer (for LightGBM categorical_feature)
    df['GROUP_ENC'] = df['GROUP'].astype('category').cat.codes
    features.append('GROUP_ENC')

    for col, tag in [('RET_1', 'RET1'), ('AVERAGE_PERF_5', 'RET5'), ('AVERAGE_PERF_20', 'RET20')]:
        grp = df.groupby('GROUP')[col]
        df[f'GROUP_XS_DEMEAN_{tag}'] = df[col] - grp.transform('mean')
        df[f'GROUP_XS_RANK_{tag}']   = grp.rank(pct=True)
        features += [f'GROUP_XS_DEMEAN_{tag}', f'GROUP_XS_RANK_{tag}']

    # Group-level mean and std of RET_1 (regime context per family)
    df['GROUP_MEAN_RET1'] = df.groupby('GROUP')['RET_1'].transform('mean')
    df['GROUP_STD_RET1']  = df.groupby('GROUP')['RET_1'].transform('std')
    features += ['GROUP_MEAN_RET1', 'GROUP_STD_RET1']

    return df, features


def add_turnover_interaction_features(df):
    """MEDIAN_DAILY_TURNOVER interactions with vol and volume features.
    Turnover is a static liquidity/capacity proxy — it may modulate how
    much the flow signal matters."""
    features = []
    turn = df['MEDIAN_DAILY_TURNOVER']
    vol  = df['STD_PERF_20'].replace(0, np.nan)

    df['TURN_X_VOL']   = turn * vol
    df['TURN_DIV_VOL'] = turn / vol
    df['TURN_X_SVOL1'] = turn * df['SIGNED_VOLUME_1'].abs()
    features += ['TURN_X_VOL', 'TURN_DIV_VOL', 'TURN_X_SVOL1']

    # Also interact with spike z-score if Tip 4 was applied
    if 'SVOL_SPIKE_Z' in df.columns:
        df['TURN_X_SPIKE'] = turn * df['SVOL_SPIKE_Z'].fillna(0)
        features.append('TURN_X_SPIKE')

    return df, features


def add_allocation_encoding(df_fit, y_fit, df_transform, smoothing=10):
    """Smoothed target encoding for ALLOCATION. Must be called inside each CV
    fold (fit on training split, transform on val/test) to avoid leakage.

    Returns the transformed dataframe with 'ALLOC_ENC' added."""
    global_mean  = float(y_fit.mean())
    stats        = y_fit.groupby(df_fit['ALLOCATION']).agg(['mean', 'count'])
    encoded      = (
        (stats['count'] * stats['mean'] + smoothing * global_mean)
        / (stats['count'] + smoothing)
    )
    out = df_transform.copy()
    out['ALLOC_ENC'] = out['ALLOCATION'].map(encoded).fillna(global_mean)
    return out


def add_features(df, ret_cols, vol_cols):
    """GROUP xs + TURNOVER interactions (no labels needed).
    Call add_allocation_encoding() separately inside each CV fold."""
    all_features = []
    df, f = add_group_xs_features(df, ret_cols)
    all_features += f
    df, f = add_turnover_interaction_features(df)
    all_features += f
    return df, all_features
