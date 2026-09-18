"""Where are the patterns that precede and sustain up-legs and down-legs?

Two questions, answered out of sample year by year on BTCUSDT 15m (2020-2026):

1. Univariate: which indicators, alone, rank the next 4h / 24h move the same
   way every year? (Spearman IC per calendar year; stable sign required.)
2. Multivariate: can a model refit every quarter on the past only predict the
   next 4h / 24h move? Out-of-sample IC per year, and the forward return of the
   top and bottom deciles of its predictions (bottom = where a Bear would short),
   after 0.10% round-trip cost.

Feature library, all causal (rolling on closed bars up to and including t):
classic indicators at 15m, ~1h and ~4h scales (RSI, MACD, stochastic, CCI, MFI,
ADX/DMI, Aroon, Bollinger %B and width, Keltner position, Ichimoku distances,
CMF, OBV slope), Heikin-Ashi streaks, higher-high/lower-low counts, distance to
24h and 7d extremes, multi-horizon returns and volatility ratios, volume z,
return skew and kurtosis, time of day and weekday, funding, the order-flow
columns already in the dataset. BTC only, by the owner's decision: no other
asset enters the features. Hyperparameters fixed before any result was seen.

    python scripts/research_walkforward_patterns.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
# Training window for each quarter: 0 = everything since 2020 (expanding);
# N = only the last N days before the quarter (the recent, more mature market).
WINDOW_DAYS = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--window-days=")), "0"))
HORIZONS = (16, 96)
COST_ROUND_TRIP = 0.0010


def rsi(close, n):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def true_range(h, l, c):
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)


def build_features(btc: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = (btc[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    f = pd.DataFrame(index=btc.index)
    tr = true_range(h, l, c)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    logret = np.log(c).diff()
    for s, tag in ((1, "15m"), (4, "1h"), (16, "4h")):
        atr_s = tr.ewm(alpha=1 / (14 * s), adjust=False).mean()
        f["rsi_" + tag] = rsi(c, 14 * s)
        macd = c.ewm(span=12 * s, adjust=False).mean() - c.ewm(span=26 * s, adjust=False).mean()
        f["macd_hist_" + tag] = (macd - macd.ewm(span=9 * s, adjust=False).mean()) / atr_s
        hh, ll = h.rolling(14 * s).max(), l.rolling(14 * s).min()
        f["stoch_" + tag] = (c - ll) / (hh - ll).replace(0, np.nan)
        tp = (h + l + c) / 3
        f["cci_" + tag] = (tp - tp.rolling(20 * s).mean()) / (0.015 * (tp - tp.rolling(20 * s).mean()).abs().rolling(20 * s).mean())
        mf = tp * v
        pos_mf = mf.where(tp > tp.shift(), 0).rolling(14 * s).sum()
        neg_mf = mf.where(tp < tp.shift(), 0).rolling(14 * s).sum()
        f["mfi_" + tag] = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))
        up_move, down_move = h.diff(), -l.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        pdi = 100 * plus_dm.ewm(alpha=1 / (14 * s), adjust=False).mean() / atr_s
        mdi = 100 * minus_dm.ewm(alpha=1 / (14 * s), adjust=False).mean() / atr_s
        f["dmi_diff_" + tag] = pdi - mdi
        f["adx_" + tag] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).ewm(alpha=1 / (14 * s), adjust=False).mean()
        win = 25 * s
        f["aroon_" + tag] = (h.rolling(win).apply(np.argmax, raw=True) - l.rolling(win).apply(np.argmin, raw=True)) / win
        mid, sd = c.rolling(20 * s).mean(), c.rolling(20 * s).std()
        f["bb_pctb_" + tag] = (c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)
        f["bb_width_" + tag] = 4 * sd / mid
        ema = c.ewm(span=20 * s, adjust=False).mean()
        f["keltner_" + tag] = (c - ema) / (2 * atr_s)
        tenkan = (h.rolling(9 * s).max() + l.rolling(9 * s).min()) / 2
        kijun = (h.rolling(26 * s).max() + l.rolling(26 * s).min()) / 2
        f["ichi_kijun_" + tag] = (c - kijun) / atr_s
        f["ichi_tk_" + tag] = (tenkan - kijun) / atr_s
        clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        f["cmf_" + tag] = (clv * v).rolling(20 * s).sum() / v.rolling(20 * s).sum()
        signed = np.sign(c.diff()) * v
        f["obv_z_" + tag] = signed.rolling(20 * s).sum() / (v.rolling(20 * s).sum() + 1e-9)
        f["hh_ll_" + tag] = ((h > h.shift()).astype(float) - (l < l.shift()).astype(float)).rolling(8 * s).sum() / (8 * s)
    # Heikin-Ashi streak: candles stacking one after another.
    ha_close = (o + h + l + c) / 4
    ha_open = ha_close.copy()
    values_open, values_close = ha_open.to_numpy(), ha_close.to_numpy()
    for i in range(1, len(values_open)):
        values_open[i] = (values_open[i - 1] + values_close[i - 1]) / 2
    ha_dir = np.sign(values_close - values_open)
    streak = np.zeros(len(ha_dir))
    for i in range(1, len(ha_dir)):
        streak[i] = streak[i - 1] + ha_dir[i] if np.sign(streak[i - 1]) == ha_dir[i] else ha_dir[i]
    f["ha_streak"] = np.clip(streak, -20, 20)
    for n, tag in ((96, "24h"), (672, "7d")):
        f["dist_high_" + tag] = (c - h.rolling(n).max()) / atr
        f["dist_low_" + tag] = (c - l.rolling(n).min()) / atr
    vol96 = logret.rolling(96).std()
    for n in (4, 16, 96, 672):
        f["ret_%d" % n] = logret.rolling(n).sum() / (vol96 * np.sqrt(n))
    f["rv_ratio_16_96"] = logret.rolling(16).std() / vol96
    f["rv_ratio_96_672"] = vol96 / logret.rolling(672).std()
    f["vol_z_96"] = (v - v.rolling(96).mean()) / v.rolling(96).std()
    f["skew_96"] = logret.rolling(96).skew()
    f["kurt_96"] = logret.rolling(96).kurt()
    hour = btc.index.hour + btc.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * btc.index.dayofweek / 7), np.cos(2 * np.pi * btc.index.dayofweek / 7)
    for col in ("funding_rate", "cz_funding_z_96", "cz_funding_cum_96", "cz_aggression", "cz_cvd_z_16", "cz_cvd_z_64",
                "cz_cvd_z_256", "cz_absorption", "cz_flow_price_div", "cz_vpin_64", "cz_avg_trade_size_z",
                "cz_trade_intensity_z", "cz_vwap_dist", "cz_efficiency_24", "cz_efficiency_96", "cz_efficiency_384"):
        f[col] = btc[col]
    return f.replace([np.inf, -np.inf], np.nan).astype("float32")


def main() -> int:
    began = time.time()
    btc = pd.read_parquet(ROOT / "data" / "history_causal.parquet").sort_index()
    X = build_features(btc)
    close = btc["close"].astype(float)
    atr_pct = (true_range(btc["high"], btc["low"], close).ewm(alpha=1 / 14, adjust=False).mean() / close)
    fwd = {H: np.log(close.shift(-H) / close) for H in HORIZONS}
    print("%d features, %d barras (%.0fs)" % (X.shape[1], len(X), time.time() - began), flush=True)

    # 1) Univariate stability by calendar year, 24h horizon.
    years = sorted(set(X.index.year))
    uni = {}
    step = 16  # one sample per 4h keeps overlapping labels from inflating counts
    for col in X.columns:
        ics = []
        for y in years:
            sel = (X.index.year == y)
            a = X[col][sel].iloc[::step]
            b = fwd[96][sel].iloc[::step]
            ok = a.notna() & b.notna()
            ics.append(spearmanr(a[ok], b[ok]).correlation if ok.sum() > 200 else np.nan)
        ics = np.array(ics)
        same_sign = np.nanmin(np.sign(ics)) == np.nanmax(np.sign(ics))
        uni[col] = {"ic_by_year": dict(zip(map(int, years), map(float, ics))), "mean": float(np.nanmean(ics)),
                    "min_abs": float(np.nanmin(np.abs(ics))), "stable_sign": bool(same_sign)}
    stable = sorted([k for k, v in uni.items() if v["stable_sign"]], key=lambda k: -abs(uni[k]["mean"]))
    print("\n1) INDICADORES COM O MESMO SINAL DE IC EM TODOS OS ANOS (retorno 24h a frente):")
    for k in stable[:20]:
        print("   %-22s IC medio %+.3f | %s" % (k, uni[k]["mean"], " ".join("%d:%+.3f" % kv for kv in uni[k]["ic_by_year"].items())))
    print("   (%d de %d features com sinal estavel)" % (len(stable), X.shape[1]))

    # 2) Walk-forward model, refit quarterly on the past only.
    quarters = pd.date_range(pd.Timestamp("2021-01-01", tz=X.index.tz), X.index[-1], freq="QS")
    report = {"univariate": uni, "walk_forward": {}}
    for H in HORIZONS:
        target = (fwd[H] / (atr_pct * np.sqrt(H / 4))).astype("float32")
        preds = pd.Series(np.nan, index=X.index, dtype="float32")
        thresholds = {}
        for q_start, q_end in zip(quarters, list(quarters[1:]) + [X.index[-1] + pd.Timedelta(minutes=15)]):
            train_mask = (X.index < q_start - pd.Timedelta(minutes=15 * H)) & target.notna()
            if WINDOW_DAYS:
                train_mask &= X.index >= q_start - pd.Timedelta(days=WINDOW_DAYS)
            train_idx = np.flatnonzero(train_mask)[::4]
            model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=200,
                                                  l2_regularization=1.0, random_state=11)
            model.fit(X.iloc[train_idx], target.iloc[train_idx].clip(-5, 5))
            test = (X.index >= q_start) & (X.index < q_end)
            preds[test] = model.predict(X[test])
            in_sample = model.predict(X.iloc[train_idx[-20000:]])
            thresholds[q_start] = (np.quantile(in_sample, 0.10), np.quantile(in_sample, 0.90))
        lo = pd.Series(np.nan, index=X.index)
        hi = pd.Series(np.nan, index=X.index)
        for q_start, (a, b) in thresholds.items():
            sel = X.index >= q_start
            lo[sel], hi[sel] = a, b
        rows = {}
        print("\n2) MODELO WALK-FORWARD, horizonte %s:" % ("4h" if H == 16 else "24h"))
        for y in sorted(set(preds.dropna().index.year)):
            sel = (preds.index.year == y) & preds.notna() & fwd[H].notna()
            p, r = preds[sel].iloc[::step], fwd[H][sel].iloc[::step]
            ic = spearmanr(p, r).correlation
            base = r.mean()
            top = r[p >= hi[sel].iloc[::step]]
            bottom = r[p <= lo[sel].iloc[::step]]
            long_net = top.mean() - COST_ROUND_TRIP
            short_net = -bottom.mean() - COST_ROUND_TRIP
            rows[int(y)] = {"ic": float(ic), "base": float(base), "long_net": float(long_net), "short_net": float(short_net),
                            "n_long": int(len(top)), "n_short": int(len(bottom))}
            print("   %d  IC %+.3f | todos %+.2f%% | compra no topo 10%%: %+.2f%% liq (n=%d) | venda no fundo 10%%: %+.2f%% liq (n=%d)" % (
                y, ic, 100 * base, 100 * long_net, len(top), 100 * short_net, len(bottom)))
        report["walk_forward"][str(H)] = rows
    out = ROOT / "reports" / ("walkforward_patterns%s.json" % ("_janela%d" % WINDOW_DAYS if WINDOW_DAYS else ""))
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("\nrelatorio: %s (%.0fs)" % (out, time.time() - began))
    return 0


if __name__ == "__main__":
    sys.exit(main())
