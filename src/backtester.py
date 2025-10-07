
import pandas as pd, numpy as np
from src.feature_pipeline import compute_features_from_candles

class Backtester:
    def __init__(self, initial_cash=1000.0):
        self.initial_cash = initial_cash

    def run_simple_strategy(self, candles_df: pd.DataFrame, threshold=0.001):
        # strategy: buy if last return > threshold, sell if < -threshold
        cash = self.initial_cash
        position = 0.0
        trades = []
        df = candles_df.copy()
        df['ret'] = df['close'].pct_change()
        for i in range(1, len(df)):
            r = df['ret'].iloc[i]
            price = df['close'].iloc[i]
            if r > threshold and cash > 0:
                # buy with half cash
                qty = (cash * 0.5) / price
                position += qty
                cash -= qty * price
                trades.append(('buy', price, qty))
            elif r < -threshold and position > 0:
                # sell all
                cash += position * price
                trades.append(('sell', price, position))
                position = 0
        # final portfolio
        final_value = cash + position * df['close'].iloc[-1]
        return {'initial_cash': self.initial_cash, 'final_value': final_value, 'trades': trades}
