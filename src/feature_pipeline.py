
import numpy as np
import pandas as pd

def compute_features_from_candles(candles: pd.DataFrame) -> np.ndarray:
    df = candles.copy()
    df['ret'] = df['close'].pct_change()
    df['sma5'] = df['close'].rolling(window=5).mean()
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['vol20'] = df['ret'].rolling(window=20).std()
    df['mom5'] = df['close'] - df['close'].shift(5)
    df = df.dropna()
    if df.empty:
        return np.zeros((1,5))
    feat = df[['ret','sma5','sma20','vol20','mom5']].iloc[-1].values
    return feat.reshape(1, -1)
