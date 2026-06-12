import numpy as np


def _ewma_weights(half_life, n=20):
    alpha = 1 - np.exp(-np.log(2) / half_life)
    w = np.array([(1 - alpha) ** (k - 1) for k in range(1, n + 1)])
    return w / w.sum()


def add_momentum_features(df, ret_cols):
    features = []
    for horizon, name in [(3, 'SHORT'), (5, 'MED5'), (10, 'MED10'), (20, 'LONG')]:
        col = f'MOM_SUM_{name}'
        df[col] = df[ret_cols[:horizon]].sum(axis=1)
        features.append(col)
    return df, features


def add_acceleration_features(df):
    features = []
    pairs = [
        ('MOM_SUM_SHORT', 'MOM_SUM_MED5',  'ACCEL_S_M5'),
        ('MOM_SUM_MED5',  'MOM_SUM_MED10', 'ACCEL_M5_M10'),
        ('MOM_SUM_MED10', 'MOM_SUM_LONG',  'ACCEL_M10_L'),
        ('MOM_SUM_SHORT', 'MOM_SUM_LONG',  'ACCEL_S_L'),
    ]
    for col_a, col_b, name in pairs:
        df[name] = df[col_a] - df[col_b]
        features.append(name)
    return df, features


def add_ewma_features(df, ret_cols, half_lives=(1, 2, 3, 5, 7, 10, 15)):
    features = []
    for hl in half_lives:
        w   = _ewma_weights(hl, n=len(ret_cols))
        col = f'EWMA_HL{hl}'
        df[col] = df[ret_cols].values @ w
        features.append(col)
    return df, features


def add_reversal_features(df, ret_cols):
    vol20 = df[ret_cols].std(axis=1).replace(0, np.nan)
    df['STD_REVERSAL']     = df[ret_cols[0]] / vol20
    df['ABS_STD_REVERSAL'] = df['STD_REVERSAL'].abs()
    return df, ['STD_REVERSAL', 'ABS_STD_REVERSAL']


def add_features(df, ret_cols, vol_cols):
    all_features = []
    df, f = add_momentum_features(df, ret_cols)
    all_features += f
    df, f = add_acceleration_features(df)
    all_features += f
    df, f = add_ewma_features(df, ret_cols)
    all_features += f
    df, f = add_reversal_features(df, ret_cols)
    all_features += f
    return df, all_features