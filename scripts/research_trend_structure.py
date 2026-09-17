"""Ride whole trends through their pullbacks: which trend-state detector works, long and short?

A trend is not one leg. Price stacks candles upward (higher highs, higher lows)
with small pullbacks in between; the trade should survive the pullbacks and end
only when the structure breaks. The mirror holds for falls. Three causal
detectors of "trend on" with an explicit pullback tolerance:

  zigzag     swing structure: a swing high/low is confirmed once price retraces
             k ATR from it; up-trend while the last confirmed swing low is
             higher than the one before and the last swing high was broken.
             Exit when a confirmed swing low is broken (mirror for down).
  supertrend ATR band that trails the trend and flips only when crossed
             (period 10, multiplier m on the chosen timeframe's ATR).
  ema_stack  EMA20 > EMA50 > EMA200 with a rising EMA50 (mirror for down);
             exit when the stack breaks.

Optional order-flow confirmation at entry (16-bar CVD in the trend's direction).
15m bars of the live-built 2020-2026 history, 0.05% per side, funding, next-bar
execution. Choice on TRAIN only; validation and holdout reported by year.
"Capture" = strategy return over the trend's move, on bars the detector was in.

    python scripts/research_trend_structure.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COST = 0.0005


@numba.njit(cache=True)
def zigzag_state(high, low, close, atr, k):
    """+1 up-structure, -1 down-structure, 0 undecided; causal swing confirmation."""
    n = close.shape[0]
    state = np.zeros(n)
    direction = 0            # current leg being tracked: 1 up, -1 down
    ext_hi, ext_lo = high[0], low[0]
    last_hi = np.nan
    prev_hi = np.nan
    last_lo = np.nan
    prev_lo = np.nan
    trend = 0
    for i in range(1, n):
        if direction >= 0:
            if high[i] > ext_hi:
                ext_hi = high[i]
            if close[i] < ext_hi - k * atr[i]:      # swing high confirmed
                prev_hi, last_hi = last_hi, ext_hi
                direction = -1
                ext_lo = low[i]
        if direction <= 0:
            if low[i] < ext_lo:
                ext_lo = low[i]
            if close[i] > ext_lo + k * atr[i]:      # swing low confirmed
                prev_lo, last_lo = last_lo, ext_lo
                direction = 1
                ext_hi = high[i]
        # Structure: higher low confirmed and price above the last swing high -> up.
        if not np.isnan(prev_lo) and last_lo > prev_lo and not np.isnan(last_hi) and close[i] > last_hi:
            trend = 1
        elif not np.isnan(prev_hi) and last_hi < prev_hi and not np.isnan(last_lo) and close[i] < last_lo:
            trend = -1
        # A confirmed swing broken against the trend ends it.
        if trend == 1 and not np.isnan(last_lo) and close[i] < last_lo:
            trend = 0
        if trend == -1 and not np.isnan(last_hi) and close[i] > last_hi:
            trend = 0
        state[i] = trend
    return state


@numba.njit(cache=True)
def supertrend_state(high, low, close, atr, m):
    n = close.shape[0]
    state = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    trend = 1
    for i in range(n):
        mid = (high[i] + low[i]) / 2.0
        up = mid + m * atr[i]
        dn = mid - m * atr[i]
        if i == 0:
            upper[i], lower[i] = up, dn
            continue
        upper[i] = up if (up < upper[i - 1] or close[i - 1] > upper[i - 1]) else upper[i - 1]
        lower[i] = dn if (dn > lower[i - 1] or close[i - 1] < lower[i - 1]) else lower[i - 1]
        if trend == 1 and close[i] < lower[i]:
            trend = -1
        elif trend == -1 and close[i] > upper[i]:
            trend = 1
        state[i] = trend
    return state


@numba.njit(cache=True)
def positions(side, state, confirm, min_bars):
    """Enter when the detector turns to `side` (and flow confirms), hold while it stays."""
    n = state.shape[0]
    pos = np.zeros(n)
    held = 0
    run = 0
    for i in range(n):
        run = run + 1 if state[i] == side else 0
        if held != 0 and state[i] != side:
            held = 0
        if held == 0 and state[i] == side and run >= min_bars and confirm[i]:
            held = side
        pos[i] = held
    return pos


def main() -> int:
    cols = ["high", "low", "close", "atr_15m", "atr_1h", "atr_4h", "funding_rate", "cz_cvd_z_16", "cz_aggression"]
    df = pd.read_parquet(ROOT / "data" / "history_causal.parquet", columns=cols).sort_index().ffill().fillna(0.0)
    n = len(df)
    train_end = int(n * 0.70)
    val_start = train_end + 768
    val_end = val_start + int(n * 0.15)
    hold_start = val_end + 768
    blocks = {"treino": (0, train_end), "validacao": (val_start, val_end), "holdout": (hold_start, n)}
    high, low, close = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    ret = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    funding = df["funding_rate"].to_numpy(np.float64) * ((df.index.hour % 8 == 0) & (df.index.minute == 0))
    cvd = df["cz_cvd_z_16"].to_numpy()
    close_s = df["close"]
    ema20, ema50, ema200 = (close_s.ewm(span=s, adjust=False).mean().to_numpy() for s in (80, 200, 800))  # on 15m ~ 20/50/200 of 1h
    slope50 = np.concatenate([[0.0] * 16, ema50[16:] - ema50[:-16]])

    detectors = []
    for tf, k in itertools.product(("1h", "4h"), (2.0, 3.0, 4.0)):
        detectors.append(("zigzag atr%s k=%.0f" % (tf, k), zigzag_state(high, low, close, df["atr_" + tf].to_numpy(np.float64), k)))
    for tf, m in itertools.product(("1h", "4h"), (2.0, 3.0, 4.0)):
        detectors.append(("supertrend atr%s m=%.0f" % (tf, m), supertrend_state(high, low, close, df["atr_" + tf].to_numpy(np.float64), m)))
    stack = np.where((ema20 > ema50) & (ema50 > ema200) & (slope50 > 0), 1, np.where((ema20 < ema50) & (ema50 < ema200) & (slope50 < 0), -1, 0))
    detectors.append(("ema_stack 1h", stack.astype(np.float64)))

    def evaluate(pos):
        held = np.concatenate([[0.0], pos[:-1]])
        strat = held * ret - held * funding - COST * np.abs(np.diff(np.concatenate([[0.0], held])))
        out = {}
        for name, (a, b) in blocks.items():
            s = pd.Series(strat[a:b], index=df.index[a:b])
            eq = (1 + s).cumprod()
            daily = eq.resample("1D").last().pct_change().dropna()
            h = held[a:b]
            trades = int((np.abs(np.diff(np.concatenate([[0.0], h]))) > 0).sum()) // 2
            out[name] = {"ret": float(eq.iloc[-1] - 1), "sharpe": float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0,
                         "dd": float((1 - eq / eq.cummax()).max()), "trades": trades, "exposure": float(np.abs(h).mean())}
        y = pd.Series(strat, index=df.index)
        out["anos"] = {int(k): float(v) for k, v in ((1 + y).groupby(y.index.year).prod() - 1).items()}
        return out

    began = time.time()
    report = {}
    for side, name in ((1, "BULL"), (-1, "BEAR")):
        rows = []
        for (label, state), flow, min_bars in itertools.product(detectors, (False, True), (1, 4)):
            confirm = (side * cvd > 1.0) if flow else np.ones(n, dtype=bool)
            r = evaluate(positions(float(side), state, confirm, min_bars))
            rows.append(("%s fluxo=%s min=%d" % (label, flow, min_bars), r))
        rows = [x for x in rows if x[1]["treino"]["trades"] >= 50]
        rows.sort(key=lambda x: x[1]["treino"]["sharpe"], reverse=True)
        print("\n%s (%d detectores, %.0fs) - top pelo Sharpe de TREINO:" % (name, len(rows), time.time() - began))
        for label, r in rows[:8]:
            t, v, h = r["treino"], r["validacao"], r["holdout"]
            print("  %-36s | TR %+6.0f%% sh %.2f dd %2.0f%% n=%4d exp %2.0f%% | VAL %+4.0f%% sh %.2f dd %2.0f%% n=%3d | HO %+4.0f%% sh %.2f dd %2.0f%% n=%3d | %s" % (
                label, 100 * t["ret"], t["sharpe"], 100 * t["dd"], t["trades"], 100 * t["exposure"],
                100 * v["ret"], v["sharpe"], 100 * v["dd"], v["trades"], 100 * h["ret"], h["sharpe"], 100 * h["dd"], h["trades"],
                " ".join("%d:%+.0f" % (y, 100 * val) for y, val in r["anos"].items())))
        report[name] = [{"detector": label, **r} for label, r in rows[:20]]
    out = ROOT / "reports" / "trend_structure.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
