"""Can the turns be recognised early? Bull enters at the bottom, Bear at the top.

Hindsight marks the legs; only the past is used to recognise them.

Labels (answer key, never a feature): a ZigZag on 1h ATR finds every swing low
and high of 2020-2026. A leg counts when its amplitude pays well beyond costs
(>= min_leg ATR). The first `zone` bars after a swing low that starts such an
up-leg are "bottom zone" (Bull entry, Bear exit); the first `zone` bars after a
swing high that starts a down-leg are "top zone" (Bear entry, Bull exit).

Recognition: a classifier per side refit every quarter on the past only, over
the causal feature library of research_walkforward_patterns (indicators at
three scales, Heikin-Ashi streaks, extremes, flow, funding). BTC only.

Trading the recognition, out of sample: enter long when P(bottom zone) crosses
its in-sample 95th percentile, exit when P(top zone) does (or after max_hold
bars / an ATR stop); the mirror for shorts. 0.05% per side, funding.

Hyperparameters fixed before any result was seen.

    python scripts/research_turning_points.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
# Training window for each quarter: 0 = everything since 2020 (expanding);
# N = only the last N days before the quarter (the recent, more mature market).
WINDOW_DAYS = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--window-days=")), "0"))
sys.path.insert(0, str(ROOT / "scripts"))
from research_walkforward_patterns import build_features, true_range  # noqa: E402

K_ATR = 3.0        # swing confirmation distance, 1h ATR
MIN_LEG = 6.0      # leg amplitude that counts, 1h ATR
ZONE = 8           # bars after the swing point that are an entry zone
MAX_HOLD = 384
STOP_ATR = 3.0
COST = 0.0005


@numba.njit(cache=True)
def zigzag_points(high, low, close, atr, k):
    """Hindsight swing points: +1 swing low, -1 swing high, 0 otherwise (at the extreme's bar)."""
    n = close.shape[0]
    marks = np.zeros(n)
    direction = 0
    ext_i = 0
    ext = close[0]
    for i in range(1, n):
        if direction >= 0:
            if high[i] >= ext or direction == 0:
                if direction == 0 or high[i] >= ext:
                    ext, ext_i = high[i], i
                    direction = 1
            if close[i] < ext - k * atr[i]:
                marks[ext_i] = -1
                direction, ext, ext_i = -1, low[i], i
                continue
        if direction < 0:
            if low[i] <= ext:
                ext, ext_i = low[i], i
            if close[i] > ext + k * atr[i]:
                marks[ext_i] = 1
                direction, ext, ext_i = 1, high[i], i
    return marks


@numba.njit(cache=True)
def trade(side, p_enter, thr_enter, p_exit, thr_exit, atr, close, max_hold, stop_k):
    pos = np.zeros(close.shape[0])
    state, entry, age = 0, 0.0, 0
    for i in range(close.shape[0]):
        c = close[i]
        if state != 0:
            age += 1
            stopped = (side > 0 and c < entry - stop_k * atr[i]) or (side < 0 and c > entry + stop_k * atr[i])
            if p_exit[i] >= thr_exit[i] or stopped or age >= max_hold:
                state = 0
        if state == 0 and p_enter[i] >= thr_enter[i]:
            state, entry, age = side, c, 0
        pos[i] = state
    return pos


def main() -> int:
    began = time.time()
    btc = pd.read_parquet(ROOT / "data" / "history_causal.parquet").sort_index()
    X = build_features(btc)
    high, low, close = (btc[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    atr1h = btc["atr_1h"].astype(float).to_numpy()
    marks = zigzag_points(high, low, close, atr1h, K_ATR)
    points = np.flatnonzero(marks != 0)
    bottom = np.zeros(len(close), dtype=np.int8)
    top = np.zeros(len(close), dtype=np.int8)
    legs = 0
    for a, b in zip(points[:-1], points[1:]):
        amplitude = abs(close[b] - close[a]) / max(atr1h[a], 1e-9)
        if amplitude < MIN_LEG:
            continue
        legs += 1
        if marks[a] > 0:
            bottom[a:min(a + ZONE, b)] = 1
        else:
            top[a:min(a + ZONE, b)] = 1
    print("%d pernas >= %.0f ATR1h | zona de fundo %.2f%% das barras, zona de topo %.2f%% (%.0fs)" % (
        legs, MIN_LEG, 100 * bottom.mean(), 100 * top.mean(), time.time() - began), flush=True)
    # The label at t looks ahead by construction; it is only a target. A training row
    # may be used only once its leg is known, i.e. well before the fold starts.
    purge = pd.Timedelta(days=10)

    quarters = pd.date_range(pd.Timestamp("2021-01-01", tz=X.index.tz), X.index[-1], freq="QS")
    probs = {name: pd.Series(np.nan, index=X.index) for name in ("bottom", "top")}
    thr = {name: pd.Series(np.nan, index=X.index) for name in ("bottom", "top")}
    for q_start, q_end in zip(quarters, list(quarters[1:]) + [X.index[-1] + pd.Timedelta(minutes=15)]):
        recent = X.index >= q_start - pd.Timedelta(days=WINDOW_DAYS) if WINDOW_DAYS else np.ones(len(X), dtype=bool)
        train_idx = np.flatnonzero((X.index < q_start - purge) & recent)[::2]
        test = (X.index >= q_start) & (X.index < q_end)
        for name, y in (("bottom", bottom), ("top", top)):
            model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=300,
                                                   l2_regularization=1.0, class_weight="balanced", random_state=13)
            model.fit(X.iloc[train_idx], y[train_idx])
            probs[name][test] = model.predict_proba(X[test])[:, 1]
            recent = model.predict_proba(X.iloc[train_idx[-40000:]])[:, 1]
            thr[name][test] = np.quantile(recent, 0.95)
        print("  trimestre %s ajustado (%.0fs)" % (q_start.date(), time.time() - began), flush=True)

    oos = probs["bottom"].notna().to_numpy()
    ret = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    funding = btc["funding_rate"].astype(float).to_numpy() * ((btc.index.hour % 8 == 0) & (btc.index.minute == 0))
    report = {"legs": legs, "by_year": {}}
    pb, pt = probs["bottom"].fillna(0).to_numpy(), probs["top"].fillna(0).to_numpy()
    tb, tt = thr["bottom"].fillna(np.inf).to_numpy(), thr["top"].fillna(np.inf).to_numpy()
    positions = {"BULL": trade(1, pb, tb, pt, tt, atr1h, close, MAX_HOLD, STOP_ATR),
                 "BEAR": trade(-1, pt, tt, pb, tb, atr1h, close, MAX_HOLD, STOP_ATR)}
    print()
    for y in sorted(set(X.index[oos].year)):
        sel = oos & (X.index.year == y)
        line = {"auc_bottom": float(roc_auc_score(bottom[sel], pb[sel])) if bottom[sel].any() else float("nan"),
                "auc_top": float(roc_auc_score(top[sel], pt[sel])) if top[sel].any() else float("nan")}
        text = "%d AUC fundo %.3f topo %.3f" % (y, line["auc_bottom"], line["auc_top"])
        for side, pos in positions.items():
            held = np.concatenate([[0.0], pos[:-1]])
            strat = held * ret - held * funding - COST * np.abs(np.diff(np.concatenate([[0.0], held])))
            s = strat[sel]
            trades = int((np.abs(np.diff(np.concatenate([[0.0], held[sel]]))) > 0).sum()) // 2
            growth = float(np.prod(1 + s) - 1)
            line[side] = {"ret": growth, "trades": trades}
            text += " | %s %+.1f%% (%d trades)" % (side, 100 * growth, trades)
        close_y = close[sel]
        text += " | BTC %+.0f%%" % (100 * (close_y[-1] / close_y[0] - 1))
        report["by_year"][int(y)] = line
        print(text)
    out = ROOT / "reports" / ("turning_points%s.json" % ("_janela%d" % WINDOW_DAYS if WINDOW_DAYS else ""))
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("relatorio: %s (%.0fs)" % (out, time.time() - began))
    return 0


if __name__ == "__main__":
    sys.exit(main())
