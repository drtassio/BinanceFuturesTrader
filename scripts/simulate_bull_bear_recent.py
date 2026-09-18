"""Trade the two recent candidates for real: Bull, Bear, and both on one account.

Bull: a gradient-boosting model of the next 24h return, refit every quarter on
      the last two years' past only (research_walkforward_patterns features).
      Entry when its prediction is in the top 10% of its in-sample predictions.
      Exit variant chosen on the TRAIN part only: fixed 24h hold, leg exit
      (8h low or trailing 2.5 ATR1h), or signal fade (prediction below its
      in-sample median, stop 3 ATR1h).
Bear: close below the prior 24h low while the 4h trend is down (EMA50 < EMA200),
      exit on the prior 8h high, trailing 2.5 ATR1h, max 48h (chosen on train in
      research_recent_legs.py).
Together: one position at a time. A new position opens only on the bar its agent
      signals; if both signal on the same bar, or the account is busy, nothing
      opens. No signal = no trade: the account stays flat.

Model predictions exist from nine months after the start of the universe, so
the simulation runs from there; its blocks are the research blocks cut to that
range. 0.05% per side, funding every 8 hours, 1x notional.

    python scripts/simulate_bull_bear_recent.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_recent_legs import sim_leg  # noqa: E402
from research_walkforward_patterns import build_features  # noqa: E402

COST = 0.0005
EMBARGO = 768


@numba.njit(cache=True)
def sim_signal(entry, mode, exit_low, pred, median_cut, atr, close, active_from, hold_bars, trail_k, stop_k, max_hold):
    pos = np.zeros(close.shape[0])
    state, extreme, entry_px, entry_atr, age = 0, 0.0, 0.0, 0.0, 0
    for i in range(active_from, close.shape[0]):
        c = close[i]
        if state != 0:
            age += 1
            extreme = max(extreme, c)
            if mode == 0:
                out = age >= hold_bars
            elif mode == 1:
                out = c < exit_low[i] or c < extreme - trail_k * entry_atr or age >= max_hold
            else:
                out = pred[i] < median_cut[i] or c < entry_px - stop_k * entry_atr or age >= max_hold
            if out:
                state = 0
        if state == 0 and entry[i]:
            state, extreme, entry_px, entry_atr, age = 1, c, c, atr[i], 0
        pos[i] = state
    return pos


@numba.njit(cache=True)
def merge(bull, bear, active_from):
    pos = np.zeros(bull.shape[0])
    active = 0
    for i in range(max(active_from, 1), bull.shape[0]):
        if active == 1 and bull[i] == 0:
            active = 0
        elif active == -1 and bear[i] == 0:
            active = 0
        if active == 0:
            bull_start = bull[i] != 0 and bull[i - 1] == 0
            bear_start = bear[i] != 0 and bear[i - 1] == 0
            if bull_start and not bear_start:
                active = 1
            elif bear_start and not bull_start:
                active = -1
        if active == 1:
            pos[i] = bull[i]
        elif active == -1:
            pos[i] = bear[i]
    return pos


def main() -> int:
    began = time.time()
    full = pd.read_parquet(ROOT / "data" / "history_causal.parquet").sort_index()
    start = full.index[-1] - pd.Timedelta(days=730)
    df = full.loc[full.index >= start - pd.Timedelta(days=45)].ffill().fillna(0.0)
    s0 = int(np.searchsorted(df.index, start))
    m = len(df) - s0
    train_end = s0 + int(m * 0.70)
    val_start = train_end + EMBARGO
    val_end = val_start + int(m * 0.15)
    hold_start = val_end + EMBARGO
    close = df["close"].to_numpy(np.float64)
    ret = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    funding = df["funding_rate"].to_numpy(np.float64) * ((df.index.hour % 8 == 0) & (df.index.minute == 0))
    atr1h = df["atr_1h"].to_numpy(np.float64)

    # Bull model, walk-forward on the last two years only.
    X = build_features(df)
    fwd = np.log(df["close"].shift(-96) / df["close"])
    target = (fwd / (df["atr_15m"] / df["close"] * np.sqrt(96))).astype("float32")
    tests = pd.date_range(df.index[s0] + pd.Timedelta(days=270), df.index[-1], freq="QS")
    pred = np.full(len(df), np.nan)
    top_cut = np.full(len(df), np.inf)
    median_cut = np.full(len(df), -np.inf)
    for q_start, q_end in zip(tests, list(tests[1:]) + [df.index[-1] + pd.Timedelta(minutes=15)]):
        train = (np.arange(len(df)) >= s0) & (df.index < q_start - pd.Timedelta(hours=24)) & target.notna().to_numpy()
        idx = np.flatnonzero(train)[::2]
        model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=200,
                                              l2_regularization=1.0, random_state=11).fit(X.iloc[idx], target.iloc[idx].clip(-5, 5))
        sel = (df.index >= q_start) & (df.index < q_end)
        pred[sel] = model.predict(X[sel])
        insample = model.predict(X.iloc[idx[-20000:]])
        top_cut[sel], median_cut[sel] = np.quantile(insample, 0.90), np.quantile(insample, 0.50)
    first = int(np.searchsorted(df.index, tests[0]))
    print("modelo pronto (%.0fs); simulacao de %s a %s" % (time.time() - began, df.index[first].date(), df.index[-1].date()), flush=True)

    blocks = {"treino": (first, train_end), "validacao": (val_start, val_end), "holdout": (hold_start, len(df))}

    def evaluate(pos):
        held = np.concatenate([[0.0], pos[:-1]])
        strat = held * ret - held * funding - COST * np.abs(np.diff(np.concatenate([[0.0], held])))
        out = {}
        for name, (a, b) in blocks.items():
            s = pd.Series(strat[a:b], index=df.index[a:b])
            eq = (1 + s).cumprod()
            daily = eq.resample("1D").last().pct_change().dropna()
            h = held[a:b]
            starts = np.flatnonzero((h != 0) & (np.concatenate([[0.0], h[:-1]]) == 0))
            ends = np.flatnonzero((h == 0) & (np.concatenate([[0.0], h[:-1]]) != 0))
            eqv = eq.to_numpy()
            trade_returns = []
            for e in starts:
                later = ends[ends > e]
                x = later[0] if len(later) else len(h) - 1
                trade_returns.append(eqv[x] / (eqv[e - 1] if e > 0 else 1.0) - 1)
            tr = np.array(trade_returns)
            gains, losses = tr[tr > 0].sum(), -tr[tr < 0].sum()
            out[name] = {"ret": float(eq.iloc[-1] - 1), "sharpe": float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0,
                         "dd": float((1 - eq / eq.cummax()).max()), "trades": int(len(tr)), "win": float((tr > 0).mean()) if len(tr) else 0.0,
                         "pf": float(gains / losses) if losses > 0 else 0.0, "exposure": float(np.abs(h).mean()),
                         "long_share": float((h > 0).sum() / max((h != 0).sum(), 1))}
        q = pd.Series(strat[first:], index=df.index[first:])
        out["trimestres"] = {str(k): float(v) for k, v in ((1 + q).groupby(q.index.to_period("Q")).prod() - 1).items()}
        return out

    def show(label, r):
        parts = []
        for name in ("treino", "validacao", "holdout"):
            x = r[name]
            parts.append("%s %+5.1f%% sh %5.2f dd %4.1f%% n=%3d acerto %2.0f%% PF %.2f exp %2.0f%%" % (
                name[:3].upper(), 100 * x["ret"], x["sharpe"], 100 * x["dd"], x["trades"], 100 * x["win"], x["pf"], 100 * x["exposure"]))
        return "%-26s | %s | %s" % (label, " | ".join(parts), " ".join("%s:%+.1f" % (k[-2:], 100 * v) for k, v in r["trimestres"].items()))

    bull_entry = np.nan_to_num(pred, nan=-np.inf) >= top_cut
    exit_low = df["low"].rolling(32).min().shift(1).fillna(-np.inf).to_numpy()
    variants = {
        "Bull segura 24h": sim_signal(bull_entry, 0, exit_low, np.nan_to_num(pred), median_cut, atr1h, close, first, 96, 0.0, 0.0, 0),
        "Bull saida de perna": sim_signal(bull_entry, 1, exit_low, np.nan_to_num(pred), median_cut, atr1h, close, first, 0, 2.5, 0.0, 192),
        "Bull sinal enfraquece": sim_signal(bull_entry, 2, exit_low, np.nan_to_num(pred), median_cut, atr1h, close, first, 0, 0.0, 3.0, 192),
    }
    results = {name: evaluate(pos) for name, pos in variants.items()}
    print("\nBULL (modelo 24h) — variantes de saida; escolhida pelo Sharpe no TREINO:")
    for name, r in results.items():
        print("  " + show(name, r))
    chosen = max(results, key=lambda k: results[k]["treino"]["sharpe"])
    print("  -> escolhida: %s" % chosen)

    trend_down = df["ema_trend_4h"].to_numpy() < 0
    bear_entry = (close < df["low"].rolling(96).min().shift(1).to_numpy()) & trend_down
    exit_high = df["high"].rolling(32).max().shift(1).fillna(np.inf).to_numpy()
    bear = sim_leg(-1, bear_entry, exit_high, atr1h, 3.0, 2.5, 192, close, first)
    together = merge(variants[chosen], bear, first)
    final = {"bull": results[chosen], "bear": evaluate(bear), "conjunto": evaluate(together)}
    close_bh = {k: float(close[b - 1] / close[a] - 1) for k, (a, b) in blocks.items()}
    print("\nSIMULACAO FINAL (buy&hold: %s):" % " | ".join("%s %+.1f%%" % (k, 100 * v) for k, v in close_bh.items()))
    print("  " + show(chosen, final["bull"]))
    print("  " + show("Bear tendencia 4h", final["bear"]))
    print("  " + show("Conjunto (1 conta)", final["conjunto"]))
    print("  conjunto: tempo comprado %.0f%% das posicoes no holdout; fora do mercado %.0f%% do tempo" % (
        100 * final["conjunto"]["holdout"]["long_share"], 100 * (1 - final["conjunto"]["holdout"]["exposure"])))
    report = {"blocks": {k: [str(df.index[a]), str(df.index[b - 1])] for k, (a, b) in blocks.items()},
              "buy_and_hold": close_bh, "bull_variants": results, "bull_chosen": chosen, "final": final}
    out = ROOT / "reports" / "simulation_bull_bear_recent.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nrelatorio: %s (%.0fs)" % (out, time.time() - began))
    return 0


if __name__ == "__main__":
    sys.exit(main())
