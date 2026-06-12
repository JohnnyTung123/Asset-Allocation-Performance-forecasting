import numpy as np


def add_xs_zscore_features(df, ret_cols, vol_cols):
    features = []
    for col in ret_cols + vol_cols:
        grp   = df.groupby('TS')[col]
        mean_ = grp.transform('mean')
        std_  = grp.transform('std').replace(0, np.nan)
        name  = f'XS_Z_{col}'
        df[name] = (df[col] - mean_) / std_
        features.append(name)
    return df, features


def add_xs_rank_features(df, ret_cols, vol_cols):
    features = []
    horizon_col = {
        1: ret_cols[0],                  # RET_1
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


def add_date_context_features(df, ret_cols, vol_cols):
    features = []
    df['XS_DISP_RET1']  = df.groupby('TS')[ret_cols[0]].transform('std')
    df['XS_DISP_RET5']  = df.groupby('TS')['AVERAGE_PERF_5'].transform('std')
    df['XS_MEAN_RET1']  = df.groupby('TS')[ret_cols[0]].transform('mean')
    df['XS_PCT_RET1']   = df.groupby('TS')[ret_cols[0]].rank(pct=True)
    df['XS_DISP_SVOL1'] = df.groupby('TS')[vol_cols[0]].transform('std')
    features += ['XS_DISP_RET1', 'XS_DISP_RET5', 'XS_MEAN_RET1',
                 'XS_PCT_RET1', 'XS_DISP_SVOL1']
    return df, features


def add_features(df, ret_cols, vol_cols):
    all_features = []
    df, f = add_xs_zscore_features(df, ret_cols, vol_cols)
    all_features += f
    df, f = add_xs_rank_features(df, ret_cols, vol_cols)
    all_features += f
    df, f = add_date_context_features(df, ret_cols, vol_cols)
    all_features += f
    return df, all_features