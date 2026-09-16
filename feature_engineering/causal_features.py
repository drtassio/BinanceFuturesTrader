"""Single source of truth for causal feature construction.

Both the offline dataset builder and the live pipeline import from here, so a
model can never be trained on a feature the bot cannot reproduce in real time.

Three things happen in this module.

1. shift_higher_timeframes removes a lookahead present in every dataset built
   so far. The pipeline merges 1h/4h indicators onto the 15m grid with
   merge_asof(direction='backward') keyed on the higher timeframe bar's *open*
   time. A 4h bar labelled 00:00 spans [00:00, 04:00) and only exists once it
   closes, yet the merge exposed it from 00:00 onward, so the agent read that
   bar's own close up to 3h45m early. Measured on the shipped dataset:
   corr(log_return_4h, return of the current 4h bar) = +0.95, against -0.01
   for the previous, genuinely known bar.

   A decision on the 15m bar labelled t is taken at its close, wall clock
   t + 15m. A higher timeframe bar labelled B of period P is known by then iff
   B + P <= t + 15m, which is a shift of exactly P - 15m. 1m and 5m bars close
   inside the 15m bar, so a backward merge already leaves them causal and they
   are not touched.

2. add_trend_structure describes a trend while it forms: distance from the mean
   and slope in volatility units, range position, how fresh the extreme is, and
   how efficiently price is travelling. All of it from closed bars.

3. add_tape_features reconstructs the order-flow view historically. The live
   order book cannot be backfilled, but the aggressor side of every trade can:
   Binance klines carry taker buy volume, quote volume and trade count, so
   aggression, participant size and price improvement are all recoverable and
   comparable between training and production.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BAR = pd.Timedelta(minutes=15)
HIGHER_TF: Dict[str, pd.Timedelta] = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}

# Prefix for every feature created here. The specialist feature filter keeps
# these unconditionally, and it makes an audit of the observation trivial.
PREFIX = "cz_"


def _clean(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).astype("float32")


def shift_higher_timeframes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Expose a higher timeframe bar only once it has closed.

    The shift is expressed in time rather than in rows so it stays correct on
    an irregular index, such as a live window containing a gap.
    """
    out = df.copy()
    report: Dict[str, Dict[str, int]] = {}
    for suffix, period in HIGHER_TF.items():
        cols = [c for c in out.columns if isinstance(c, str) and c.endswith("_" + suffix)]
        if not cols:
            continue
        delay = period - BAR
        moved = out[cols].copy()
        # Re-stamp each observation with the moment it becomes knowable, then
        # read it back on the original grid.
        moved.index = moved.index + delay
        out[cols] = moved.reindex(out.index, method="ffill")
        report[suffix] = {
            "columns": len(cols),
            "delay_minutes": int(delay.total_seconds() // 60),
        }
    return out, report


def add_trend_structure(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Causal description of a forming trend, from closed 15m bars only."""
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    if "atr_15m" in out.columns:
        atr = out["atr_15m"].astype(float)
    else:
        atr = pd.Series(np.nan, index=out.index)
    atr = atr.replace(0.0, np.nan).fillna(close * 0.005)
    added: List[str] = []

    def put(name: str, series: pd.Series) -> None:
        out[PREFIX + name] = _clean(series)
        added.append(PREFIX + name)

    for span in (20, 50, 100, 200):
        ema = close.ewm(span=span, adjust=False).mean()
        # Scale free across price level and volatility regime.
        put("ema_dist_%d" % span, (close - ema) / atr)
        put("ema_slope_%d" % span, (ema - ema.shift(10)) / atr)

    for win in (20, 50, 100):
        hh = high.rolling(win).max()
        ll = low.rolling(win).min()
        rng = (hh - ll).replace(0.0, np.nan)
        put("donchian_pos_%d" % win, (close - ll) / rng)
        put("donchian_width_%d" % win, rng / atr)
        # A freshly set extreme marks a young trend rather than an exhausted one.
        put("bars_since_high_%d" % win,
            (win - high.rolling(win).apply(np.argmax, raw=True) - 1) / win)
        put("bars_since_low_%d" % win,
            (win - low.rolling(win).apply(np.argmin, raw=True) - 1) / win)

    logret = np.log(close).diff()
    base_vol = logret.rolling(384).std()
    for win in (24, 96, 384):
        # Net move over path travelled: near 1 the market trends cleanly,
        # near 0 it chops.
        net = close - close.shift(win)
        path = close.diff().abs().rolling(win).sum().replace(0.0, np.nan)
        put("efficiency_%d" % win, net / path)
        rv = logret.rolling(win).std()
        put("rvol_%d" % win, rv * np.sqrt(96))
        put("rvol_ratio_%d" % win, rv / base_vol)

    # Volatility expansion tends to accompany and sustain a directional move.
    put("atr_expansion", atr.rolling(12).mean() / atr.rolling(96).mean().replace(0.0, np.nan))
    put("atr_pct", atr / close)
    return out, added


def add_tape_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Historical reconstruction of the order-flow (tape) view.

    Whatever kline flow column is missing is skipped, so a frame without them
    still builds.
    """
    out = df.copy()
    added: List[str] = []
    close = out["close"].astype(float)
    volume = out["volume"].astype(float).replace(0.0, np.nan)

    def put(name: str, series: pd.Series) -> None:
        out[PREFIX + name] = _clean(series)
        added.append(PREFIX + name)

    def zscore(series: pd.Series, win: int) -> pd.Series:
        mean = series.rolling(win, min_periods=win // 4).mean()
        std = series.rolling(win, min_periods=win // 4).std().replace(0.0, np.nan)
        return ((series - mean) / std).clip(-6, 6)

    imb_col = None
    for candidate in ("aggressor_imbalance_15m", "aggressor_imbalance"):
        if candidate in out.columns:
            imb_col = candidate
            break

    if imb_col is not None:
        # Signed aggression: above zero means buyers lifted more than sellers hit.
        imb = out[imb_col].astype(float)
        delta = imb * volume
        put("aggression", imb)
        for win in (16, 64, 256):
            # Cumulative volume delta, standardised so pressure building over
            # hours is visible instead of one noisy bar.
            cvd = delta.rolling(win).sum()
            scale = delta.rolling(win).std().replace(0.0, np.nan) * np.sqrt(win)
            put("cvd_z_%d" % win, (cvd / scale).clip(-6, 6))
        price_move = np.sign(close - close.shift(16))
        flow_dir = np.sign(delta.rolling(16).sum())
        # Absorption: price refuses to follow aggressive flow, the signature of
        # passive size standing on the other side.
        put("absorption",
            (flow_dir - price_move).abs() / 2.0
            * delta.rolling(16).sum().abs() / (volume.rolling(16).sum() + 1e-9))
        put("flow_price_div", price_move * -flow_dir)
        # VPIN-like toxicity: the share of volume that is one-sided.
        put("vpin_64",
            (delta.rolling(64).sum().abs() / (volume.rolling(64).sum() + 1e-9)).clip(0, 1))

    if "trade_count" in out.columns:
        tc = out["trade_count"].astype(float)
        # Average trade size separates many small orders from few large ones,
        # the practical proxy for institutional participation.
        avg_size = volume / tc.replace(0.0, np.nan)
        put("avg_trade_size_z", zscore(avg_size, 96))
        put("trade_intensity_z", zscore(tc, 96))

    qv = next((c for c in ("quote_volume", "quote_volume_15m") if c in out.columns), None)
    tbq = next((c for c in ("taker_buy_quote_volume", "taker_buy_quote_volume_15m") if c in out.columns), None)
    tbb = next((c for c in ("taker_buy_base_volume", "taker_buy_base_volume_15m") if c in out.columns), None)
    if qv and tbq and tbb:
        # Average price paid by aggressive buyers versus aggressive sellers,
        # relative to the bar's own VWAP: who is paying up.
        buy_vwap = out[tbq].astype(float) / out[tbb].astype(float).replace(0.0, np.nan)
        sell_base = (volume - out[tbb].astype(float)).replace(0.0, np.nan)
        sell_vwap = (out[qv].astype(float) - out[tbq].astype(float)) / sell_base
        bar_vwap = out[qv].astype(float) / volume
        put("buy_price_improve", (buy_vwap - bar_vwap) / (bar_vwap * 1e-4 + 1e-12))
        put("sell_price_improve", (sell_vwap - bar_vwap) / (bar_vwap * 1e-4 + 1e-12))
        put("vwap_dist", (close - bar_vwap) / (bar_vwap + 1e-9))

    fr_col = next((c for c in ("funding_rate_15m", "funding_rate") if c in out.columns), None)
    if fr_col is not None:
        fr = out[fr_col].astype(float)
        # Sustained funding is crowd positioning; extremes precede squeezes.
        put("funding_cum_96", fr.rolling(96).sum())
        put("funding_extreme", np.tanh(fr.rolling(32).mean() / 1e-4))
        put("funding_z_96", zscore(fr, 96))
    return out, added


def build_causal_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Apply the full causal treatment. Used by training and by the live bot."""
    out, shift_report = shift_higher_timeframes(df)
    out, trend_cols = add_trend_structure(out)
    out, tape_cols = add_tape_features(out)
    meta: Dict[str, object] = {
        "higher_timeframe_shift": shift_report,
        "trend_features": trend_cols,
        "tape_features": tape_cols,
    }
    return out, meta
