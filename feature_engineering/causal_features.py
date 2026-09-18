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


# Raw order-flow columns under the one name everything downstream reads. The
# same quantity arrives under different names depending on where it came from:
#   training: merged from the Binance flow parquet   -> trade_count, quote_volume
#   live:     kline API, then suffixed by create_features -> number_of_trades_15m
# Without this, a specialist trained on "funding_rate" and "aggressor_imbalance"
# receives zeros live, where only the _15m versions exist.
FLOW_ALIASES: Dict[str, Tuple[str, ...]] = {
    "trade_count": ("trade_count", "trade_count_15m", "number_of_trades", "number_of_trades_15m"),
    "quote_volume": ("quote_volume", "quote_volume_15m", "quote_asset_volume", "quote_asset_volume_15m"),
    "taker_buy_base_volume": ("taker_buy_base_volume", "taker_buy_base_volume_15m",
                              "taker_buy_base_asset_volume", "taker_buy_base_asset_volume_15m"),
    "taker_buy_quote_volume": ("taker_buy_quote_volume", "taker_buy_quote_volume_15m",
                               "taker_buy_quote_asset_volume", "taker_buy_quote_asset_volume_15m"),
    "aggressor_imbalance": ("aggressor_imbalance", "aggressor_imbalance_15m"),
    "taker_buy_ratio": ("taker_buy_ratio", "taker_buy_ratio_15m"),
    "funding_rate": ("funding_rate", "funding_rate_15m"),
}


def normalize_flow_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Make every raw flow column available under its canonical name."""
    out = df.copy()
    created: List[str] = []
    for canonical, candidates in FLOW_ALIASES.items():
        if canonical in out.columns:
            continue
        source = next((c for c in candidates if c in out.columns), None)
        if source is not None:
            out[canonical] = out[source]
            created.append(canonical)
    return out, created


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


# A staircase: price sits in a tight box, then two or more large candles in the
# same direction close out of it. The move has started; nothing is predicted.
# Values from scripts/label_staircase_examples.py, chosen on its training block.
STEP_BODY_ATR = 1.5
STAIRCASE_BOX_BARS = 48      # 12h box
STAIRCASE_BOX_GAP = 3        # the box ends before the steps themselves
STRUCTURE_4H_BUFFER_ATR = 0.5
BAR_4H = pd.Timedelta(hours=4)


def add_staircase_structure(df: pd.DataFrame, atr: pd.Series) -> Dict[str, pd.Series]:
    """Steps, the box they leave and the 4h structure that carries the trend.

    step_up_count / step_down_count: consecutive candles whose body is at least
    STEP_BODY_ATR x ATR and whose close beats the previous close.
    box_width: height of the 12h box before the steps, in ATR (tight = small).
    box_break_up / box_break_down: close beyond that box, in ATR.
    struct_4h_long / struct_4h_short: distance from the close to the 4h
    structure stop (lowest low / highest high of the last two CLOSED 4h candles,
    widened by half a 4h ATR), in 15m ATR. Negative: the structure broke.
    """
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    body = (close - open_) / atr
    step_up = (body >= STEP_BODY_ATR) & (close > close.shift(1))
    step_down = (body <= -STEP_BODY_ATR) & (close < close.shift(1))
    up_count = step_up.astype(int).groupby((~step_up).cumsum()).cumsum()
    down_count = step_down.astype(int).groupby((~step_down).cumsum()).cumsum()

    box_high = high.rolling(STAIRCASE_BOX_BARS).max().shift(STAIRCASE_BOX_GAP)
    box_low = low.rolling(STAIRCASE_BOX_BARS).min().shift(STAIRCASE_BOX_GAP)

    # 4h candles from the 15m bars, each stamped on the 15m bar that closes it.
    bars = pd.DataFrame({"high": high, "low": low, "close": close})
    four = bars.resample(BAR_4H, closed="left", label="left").agg(
        {"high": "max", "low": "min", "close": "last"}).dropna()
    four.index = four.index + BAR_4H - BAR
    tr = np.maximum(four["high"] - four["low"],
                    np.maximum((four["high"] - four["close"].shift()).abs(),
                               (four["low"] - four["close"].shift()).abs()))
    atr_4h = tr.rolling(14, min_periods=4).mean()
    support = (four["low"].rolling(2).min() - STRUCTURE_4H_BUFFER_ATR * atr_4h).reindex(df.index, method="ffill")
    resistance = (four["high"].rolling(2).max() + STRUCTURE_4H_BUFFER_ATR * atr_4h).reindex(df.index, method="ffill")

    return {
        "step_body_atr": body,
        "step_up_count": up_count.astype(float),
        "step_down_count": down_count.astype(float),
        "box_width": (box_high - box_low) / atr,
        "box_break_up": (close - box_high) / atr,
        "box_break_down": (close - box_low) / atr,
        "struct_4h_long": (close - support) / atr,
        "struct_4h_short": (resistance - close) / atr,
    }


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

    # Distance past the prior N-bar extreme, in ATR. On 15m bars 480 is about a
    # 30-bar Donchian on 4h, the horizon on which trend following on BTC
    # perpetuals held up from 2020 to 2026 (Sharpe ~1 long-only, costs and
    # funding included); the 100-bar channel above is a 25-hour trend and chops.
    # Positive breakout_up: the close cleared the prior highs; negative
    # breakout_down: it lost the prior lows. shift(1) keeps the current bar out
    # of its own reference. 32 and 192 serve the fast rally teacher (a 48-hour
    # high on expanding volatility with buyers in control, out on an 8-hour low).
    for win in (32, 192, 240, 480):
        prior_high = high.rolling(win).max().shift(1)
        prior_low = low.rolling(win).min().shift(1)
        put("breakout_up_%d" % win, (close - prior_high) / atr)
        put("breakout_down_%d" % win, (close - prior_low) / atr)

    staircase = add_staircase_structure(out, atr)
    for name, series in staircase.items():
        put(name, series)

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

    # Market "physics": Hurst and Shannon entropy over the last 30 returns,
    # the same definition utils.physics_sensors uses, precomputed here.
    #
    # The environment recomputed these on every step, twice, which was 72% of
    # its CPU time, and then discarded the result because the caller read keys
    # the function did not return. Precomputing makes the training loop several
    # times faster and guarantees training and live see the same number.
    returns = close.pct_change()

    def _hurst(window: np.ndarray) -> float:
        n = len(window)
        if n < 20:
            return 0.5
        lags = sorted({max(4, n // 8), max(8, n // 4), max(12, n // 2)})
        rs_values, valid = [], []
        for lag in lags:
            chunks = [window[i:i + lag] for i in range(0, n - lag + 1, lag)]
            if len(chunks) < 2:
                continue
            ratios = []
            for chunk in chunks:
                deviation = np.cumsum(chunk - chunk.mean())
                spread = deviation.max() - deviation.min()
                sigma = chunk.std(ddof=1)
                if sigma > 1e-9:
                    ratios.append(spread / sigma)
            if ratios:
                rs_values.append(float(np.mean(ratios)))
                valid.append(lag)
        if len(valid) < 2:
            return 0.5
        slope = np.polyfit(np.log(valid), np.log(rs_values), 1)[0]
        return float(np.clip(slope, 0.1, 0.9))

    def _entropy(window: np.ndarray, bins: int = 10) -> float:
        counts, _ = np.histogram(window, bins=bins)
        probability = counts / (len(window) + 1e-12)
        probability = probability[probability > 0]
        return float(np.clip(-np.sum(probability * np.log2(probability)), 0.0, 5.0))

    put("hurst", returns.rolling(30).apply(_hurst, raw=True))
    put("entropy", returns.rolling(30).apply(_entropy, raw=True))
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

    imb_col = "aggressor_imbalance" if "aggressor_imbalance" in out.columns else None

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

    qv = "quote_volume" if "quote_volume" in out.columns else None
    tbq = "taker_buy_quote_volume" if "taker_buy_quote_volume" in out.columns else None
    tbb = "taker_buy_base_volume" if "taker_buy_base_volume" in out.columns else None
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

    fr_col = "funding_rate" if "funding_rate" in out.columns else None
    if fr_col is not None:
        fr = out[fr_col].astype(float)
        # Sustained funding is crowd positioning; extremes precede squeezes.
        put("funding_cum_96", fr.rolling(96).sum())
        put("funding_extreme", np.tanh(fr.rolling(32).mean() / 1e-4))
        put("funding_z_96", zscore(fr, 96))
    return out, added


def add_regime_priors(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Derive the tp_* prior columns the environment and the live bot expect.

    These were the single largest train/production mismatch in the bot. The
    live controller builds tp_prior_conf, tp_prior_dir and the tp_regime_*
    one-hots from the regime detector before calling a specialist, but the
    training parquet never contained them, so every `row.get('tp_prior_dir',
    0.0)` in the environment resolved to the default. Training therefore ran
    with prior_dir pinned at 0.0, prior_conf at 0.5 and every regime one-hot at
    zero, which silently disabled the entry gates, made `regime_neutral`
    permanently true, and pinned position size at its floor. The agent then met
    real values the moment it went live.

    The mapping below is the one FeatureEngineeringPipeline.apply_hidden_features
    applies on the live decision path, so both sides see identical inputs. Regime labels themselves are causal: measured against
    forward returns their rank correlation is 0.01 to 0.02 across horizons.
    """
    out = df.copy()
    added: List[str] = []
    regime_col = next((c for c in ("regime_val", "regime") if c in out.columns), None)
    if regime_col is None or "regime_confidence" not in out.columns:
        return out, added

    regime = out[regime_col].astype(float)
    confidence = out["regime_confidence"].astype(float).clip(0.0, 1.0)

    # 0 = Bull, 1 = Bear, 2+ = Ranger.
    out["tp_prior_conf"] = confidence.astype("float32")
    out["tp_prior_dir"] = np.select(
        [regime == 0, regime == 1],
        [confidence, -confidence],
        default=0.0,
    ).astype("float32")
    # Ponderados pela confianca, exatamente como FeatureEngineeringPipeline.
    # apply_hidden_features, que e o caminho que monta a observacao ao vivo
    # (ai_controller.py, geracao de sinal). Um one-hot 0/1 aqui divergiria dos
    # ~0.88 que o bot entrega ao especialista na mesma situacao.
    out["tp_regime_up"] = np.where(regime == 0, confidence, 0.0).astype("float32")
    out["tp_regime_down"] = np.where(regime == 1, confidence, 0.0).astype("float32")
    out["tp_regime_sideways"] = np.where(regime >= 2, confidence, 0.0).astype("float32")
    added = ["tp_prior_conf", "tp_prior_dir", "tp_regime_up",
             "tp_regime_down", "tp_regime_sideways"]
    return out, added


def build_causal_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Apply the full causal treatment. Used by training and by the live bot."""
    out, flow_aliases = normalize_flow_columns(df)
    out, shift_report = shift_higher_timeframes(out)
    out, trend_cols = add_trend_structure(out)
    out, tape_cols = add_tape_features(out)
    out, prior_cols = add_regime_priors(out)
    meta: Dict[str, object] = {
        "flow_aliases": flow_aliases,
        "higher_timeframe_shift": shift_report,
        "trend_features": trend_cols,
        "tape_features": tape_cols,
        "regime_priors": prior_cols,
    }
    return out, meta
