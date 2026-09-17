"""Which sharp moves continue (or, sideways, revert)? Event-conditioned meta-labelling.

The old meta-model scored every bar and had no out-of-sample information (AUC
~0.50). A move that matters is rare, so the question is narrower: GIVEN that a
sharp leg has just started, will it carry far enough to pay before it stops?
That is meta-labelling in Lopez de Prado's sense: a simple primary event
proposes a trade, a classifier decides which proposals to take.

Events (primary, deliberately loose):
  bear   close breaks the prior N-bar low, or an k-bar drop beyond z sigma
  bull   the mirror
  ranger in a low-efficiency, low-ADX market, close beyond z sigma of its
         20-bar mean (fade it back to the mean)
Label: triple barrier from the event close, in 1h ATR: target first -> win.
The realized return (target, stop or horizon) is net of 0.05% per side.
Features: order flow, trend, volatility and higher-timeframe context, all
causal, all present in the live-built history.

Protocol: classifier and threshold chosen inside the TRAIN block only
(expanding-window CV, events purged at fold boundaries). Validation and holdout
are reported per year, never used to choose.

    python scripts/research_event_edges.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
COST = 0.0005
FEATURES = [
    "cz_aggression", "cz_cvd_z_16", "cz_cvd_z_64", "cz_cvd_z_256", "cz_absorption", "cz_flow_price_div", "cz_vpin_64",
    "cz_avg_trade_size_z", "cz_trade_intensity_z", "cz_buy_price_improve", "cz_sell_price_improve", "cz_vwap_dist",
    "cz_funding_cum_96", "cz_funding_extreme", "cz_funding_z_96", "funding_rate",
    "cz_ema_dist_20", "cz_ema_slope_20", "cz_ema_dist_50", "cz_ema_slope_50", "cz_ema_dist_100", "cz_ema_slope_100",
    "cz_ema_dist_200", "cz_ema_slope_200", "cz_donchian_pos_20", "cz_donchian_pos_50", "cz_donchian_pos_100",
    "cz_donchian_width_100", "cz_efficiency_24", "cz_efficiency_96", "cz_efficiency_384", "cz_rvol_24", "cz_rvol_ratio_24",
    "cz_rvol_ratio_96", "cz_atr_expansion", "cz_atr_pct", "cz_hurst", "cz_entropy",
    "cz_breakout_up_32", "cz_breakout_down_32", "cz_breakout_up_192", "cz_breakout_down_192",
    "cz_breakout_up_480", "cz_breakout_down_480",
    "rsi_15m", "adx_15m", "rsi_1h", "adx_1h", "ema_trend_1h", "ema_trend_4h", "bb_width_1h", "atr_percentage_4h",
]


@numba.njit(cache=True)
def barrier(side, idx, close, high, low, atr, target_k, stop_k, horizon, cost):
    out = np.zeros(idx.shape[0])
    hold = np.zeros(idx.shape[0])
    n = close.shape[0]
    for e in range(idx.shape[0]):
        i = idx[e]
        entry = close[i]
        tgt = target_k * atr[i]
        stp = stop_k * atr[i]
        result = np.nan
        j = i + 1
        last = min(i + horizon, n - 1)
        while j <= last:
            if side > 0:
                hit_stop = low[j] <= entry - stp
                hit_tgt = high[j] >= entry + tgt
            else:
                hit_stop = high[j] >= entry + stp
                hit_tgt = low[j] <= entry - tgt
            if hit_stop:  # both in one bar: assume the stop came first
                result = -stp / entry
                break
            if hit_tgt:
                result = tgt / entry
                break
            j += 1
        if np.isnan(result):
            j = last
            result = side * (close[last] - entry) / entry
        out[e] = result - 2 * cost
        hold[e] = j - i
    return out, hold


def events(df: pd.DataFrame, kind: str) -> np.ndarray:
    close, high, low = df["close"], df["high"], df["low"]
    logret = np.log(close).diff()
    mask = pd.Series(False, index=df.index)
    if kind in ("bear", "bull"):
        side = -1 if kind == "bear" else 1
        for N in (48, 96, 192):
            ref = low.rolling(N).min().shift(1) if side < 0 else high.rolling(N).max().shift(1)
            mask |= (close < ref) if side < 0 else (close > ref)
        for k in (4, 8, 16):
            move = logret.rolling(k).sum()
            scale = move.rolling(2880).std().shift(1)
            mask |= (side * move > 2.5 * scale)
        # One event per leg: ignore repeats within the next 16 bars.
        raw = np.flatnonzero(mask.fillna(False).to_numpy())
    else:
        eff = df["cz_efficiency_96"].abs()
        mean20 = close.rolling(20).mean()
        z = (close - mean20) / close.rolling(20).std()
        ranging = (eff < 0.30) & (df["adx_15m"] < 25)
        raw = np.flatnonzero((ranging & (z.abs() > 2.0)).fillna(False).to_numpy())
    kept, last = [], -10**9
    for i in raw:
        if i - last >= 16:
            kept.append(i)
            last = i
    return np.asarray(kept, dtype=np.int64)


def ranger_side(df, idx):
    close = df["close"].to_numpy()
    mean20 = df["close"].rolling(20).mean().to_numpy()
    return np.where(close[idx] > mean20[idx], -1, 1)


def main() -> int:
    data = ROOT / "data" / "history_causal.parquet"
    df = pd.read_parquet(data).sort_index().ffill().fillna(0.0)
    n = len(df)
    train_end = int(n * 0.70)
    val_start = train_end + 768
    val_end = val_start + int(n * 0.15)
    hold_start = val_end + 768
    blocks = {"treino": (0, train_end), "validacao": (val_start, val_end), "holdout": (hold_start, n)}
    close, high, low = (df[c].to_numpy(np.float64) for c in ("close", "high", "low"))
    atr1h = df["atr_1h"].to_numpy(np.float64)
    X_all = df[FEATURES].to_numpy(np.float32)
    configs = {
        "bear": dict(target=3.0, stop=2.0, horizon=96),
        "bull": dict(target=3.0, stop=2.0, horizon=96),
        "ranger": dict(target=1.0, stop=1.5, horizon=16),
    }
    report = {}
    for kind, cfg in configs.items():
        idx = events(df, kind)
        if kind == "ranger":
            sides = ranger_side(df, idx)
            net = np.zeros(len(idx))
            hold = np.zeros(len(idx))
            for s in (-1, 1):
                sel = sides == s
                r, h = barrier(s, idx[sel], close, high, low, atr1h, cfg["target"], cfg["stop"], cfg["horizon"], COST)
                net[sel], hold[sel] = r, h
        else:
            s = -1 if kind == "bear" else 1
            net, hold = barrier(s, idx, close, high, low, atr1h, cfg["target"], cfg["stop"], cfg["horizon"], COST)
        y = (net > 0).astype(int)
        X = X_all[idx]
        when = df.index[idx]

        def block_of(i):
            for name, (a, b) in blocks.items():
                if a <= i < b - cfg["horizon"]:
                    return name
            return None
        which = np.array([block_of(i) for i in idx])
        tr = which == "treino"

        # Expanding-window CV inside train, purged by the label horizon.
        tr_pos = np.flatnonzero(tr)
        folds = np.array_split(tr_pos, 5)
        oof = np.full(len(idx), np.nan)
        for f in range(1, 5):
            fit_idx = np.concatenate(folds[:f])
            test_idx = folds[f]
            fit_idx = fit_idx[idx[fit_idx] < idx[test_idx[0]] - cfg["horizon"]]
            model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                   min_samples_leaf=40, l2_regularization=1.0, random_state=7)
            model.fit(X[fit_idx], y[fit_idx])
            oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        cv = np.isfinite(oof)
        # Threshold: best mean net return per trade on the CV predictions, >= 25% of events kept.
        best_thr, best_mean = 0.5, -np.inf
        for thr in np.quantile(oof[cv], np.linspace(0.0, 0.75, 16)):
            take = cv & (oof >= thr)
            if take.sum() >= 0.25 * cv.sum() and net[take].mean() > best_mean:
                best_thr, best_mean = float(thr), float(net[take].mean())
        final = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                               min_samples_leaf=40, l2_regularization=1.0, random_state=7)
        final.fit(X[tr], y[tr])
        prob = final.predict_proba(X)[:, 1]

        lines = {}
        print("\n%s: %d eventos (%s) | alvo %.1f ATR1h, stop %.1f, horizonte %d barras | limiar %.3f (CV treino: liq/trade %+.3f%%, AUC %.3f)" % (
            kind.upper(), len(idx), ", ".join("%s=%d" % (b, (which == b).sum()) for b in blocks), cfg["target"], cfg["stop"], cfg["horizon"],
            best_thr, 100 * best_mean, roc_auc_score(y[cv], oof[cv])))
        for name in ("treino", "validacao", "holdout"):
            # Train is scored only with out-of-fold predictions: the final model
            # has seen those events and would report its own memory.
            score = oof if name == "treino" else prob
            sel = (which == name) & np.isfinite(score)
            take = sel & (score >= best_thr)
            auc = roc_auc_score(y[sel], score[sel]) if y[sel].std() > 0 else float("nan")
            yearly = pd.Series(net[take], index=when[take]).groupby(when[take].year).agg(["count", "sum", "mean"])
            lines[name] = {"events": int(sel.sum()), "taken": int(take.sum()), "base_mean": float(net[sel].mean()),
                           "taken_mean": float(net[take].mean()) if take.any() else 0.0, "taken_sum": float(net[take].sum()),
                           "auc": auc, "win": float(y[take].mean()) if take.any() else 0.0,
                           "by_year": {int(k): [int(v["count"]), float(v["sum"]), float(v["mean"])] for k, v in yearly.iterrows()}}
            print("  %-9s eventos=%4d todos liq/trade %+.3f%% | filtrados=%4d liq/trade %+.3f%% soma %+.1f%% acerto %.0f%% %s | por ano: %s" % (
                name, sel.sum(), 100 * net[sel].mean(), take.sum(), 100 * lines[name]["taken_mean"], 100 * lines[name]["taken_sum"],
                100 * lines[name]["win"], ("AUC %.3f" % auc) if np.isfinite(auc) else "",
                " ".join("%d:%d/%+.0f%%" % (k, v[0], 100 * v[1]) for k, v in lines[name]["by_year"].items())))
        report[kind] = {"config": cfg, "threshold": best_thr, "blocks": lines}
    out = ROOT / "reports" / "event_edges.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
