"""The Bull and Bear trades of simulate_bull_bear_recent.py as (entry time, net return)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import simulate_bull_bear_recent as sbr
from research_recent_legs import sim_leg
from research_walkforward_patterns import build_features


def trades_from_simulation(df, s0):
    close = df["close"].to_numpy(np.float64)
    atr1h = df["atr_1h"].to_numpy(np.float64)
    X = build_features(df)
    fwd = np.log(df["close"].shift(-96) / df["close"])
    target = (fwd / (df["atr_15m"] / df["close"] * np.sqrt(96))).astype("float32")
    tests = pd.date_range(df.index[s0] + pd.Timedelta(days=270), df.index[-1], freq="QS")
    pred = np.full(len(df), np.nan)
    top = np.full(len(df), np.inf)
    for q_start, q_end in zip(tests, list(tests[1:]) + [df.index[-1] + pd.Timedelta(minutes=15)]):
        train = (np.arange(len(df)) >= s0) & (df.index < q_start - pd.Timedelta(hours=24)) & target.notna().to_numpy()
        idx = np.flatnonzero(train)[::2]
        model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=200,
                                              l2_regularization=1.0, random_state=11).fit(X.iloc[idx], target.iloc[idx].clip(-5, 5))
        sel = (df.index >= q_start) & (df.index < q_end)
        pred[sel] = model.predict(X[sel])
        top[sel] = np.quantile(model.predict(X.iloc[idx[-20000:]]), 0.9)
    first = int(np.searchsorted(df.index, tests[0]))
    exit_low = df["low"].rolling(32).min().shift(1).fillna(-np.inf).to_numpy()
    median = np.full(len(df), -np.inf)
    bull = sbr.sim_signal(np.nan_to_num(pred, nan=-np.inf) >= top, 0, exit_low, np.nan_to_num(pred), median, atr1h, close, first, 96, 0.0, 0.0, 0)
    bear_entry = (close < df["low"].rolling(96).min().shift(1).to_numpy()) & (df["ema_trend_4h"].to_numpy() < 0)
    bear = sim_leg(-1, bear_entry, df["high"].rolling(32).max().shift(1).fillna(np.inf).to_numpy(), atr1h, 3.0, 2.5, 192, close, first)
    out = {}
    for name, pos in (("bull", bull), ("bear", bear)):
        rows, i = [], first
        while i < len(pos):
            if pos[i] != 0 and pos[i - 1] == 0:
                j = i
                while j < len(pos) and pos[j] != 0:
                    j += 1
                gross = pos[i] * (close[min(j, len(close) - 1)] / close[i] - 1)
                rows.append((df.index[i], gross - 2 * sbr.COST))
                i = j
            i += 1
        out[name] = rows
    return out
