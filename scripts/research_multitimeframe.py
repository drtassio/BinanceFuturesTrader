"""Multi-timeframe research on BTC, last two years: horizon, timeframe contribution, hierarchy.

The bot decides every 15 minutes. Two things are separate:
  * the timeframes it LOOKS at (5m, 15m, 1h, 4h, 1d candles), and
  * the horizon it JUDGES a decision by (the move over the next 1h ... 3d).

A. Horizon: which forward horizon is most predictable and most tradable?
B. Timeframe contribution: the same walk-forward model fed one timeframe group at
   a time and in combinations, to see where the information lives.
C. Hierarchy, as a trader reads a chart: the higher timeframe gives the direction
   (4h or daily trend), the lower timeframe times the entry at the end of a
   pullback (15m or 5m RSI turn, Heikin-Ashi colour flip, order-flow turn) and
   the exit at the end of the leg. Bull and Bear mirror each other.

All features are causal. 5m features at a 15m decision are those of the 5m
candles closed by that 15m close. BTC only. Last two years (+45 days warm-up
for indicator look-back only). Walk-forward models refit every quarter on the
universe's past; rule grids chosen on train only. 0.05% per side, funding.

    python scripts/research_multitimeframe.py
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
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
COST = 0.0005
EMBARGO = 768


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def heikin_streak(o, h, l, c):
    hc = ((o + h + l + c) / 4).to_numpy()
    ho = hc.copy()
    for i in range(1, len(ho)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    direction = np.sign(hc - ho)
    streak = np.zeros(len(direction))
    for i in range(1, len(direction)):
        streak[i] = streak[i - 1] + direction[i] if np.sign(streak[i - 1]) == direction[i] else direction[i]
    return pd.Series(np.clip(streak, -20, 20), index=c.index)


def scale_block(o, h, l, c, v, s, tag):
    """One timeframe's view built from 15m candles with windows scaled by s (4=1h, 16=4h, 96=1d)."""
    f = pd.DataFrame(index=c.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / (14 * s), adjust=False).mean()
    f["rsi_" + tag] = rsi(c, 14 * s)
    macd = c.ewm(span=12 * s, adjust=False).mean() - c.ewm(span=26 * s, adjust=False).mean()
    f["macd_" + tag] = (macd - macd.ewm(span=9 * s, adjust=False).mean()) / atr
    hh, ll = h.rolling(14 * s).max(), l.rolling(14 * s).min()
    f["stoch_" + tag] = (c - ll) / (hh - ll).replace(0, np.nan)
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0).ewm(alpha=1 / (14 * s), adjust=False).mean()
    mdm = dn.where((dn > up) & (dn > 0), 0.0).ewm(alpha=1 / (14 * s), adjust=False).mean()
    f["dmi_" + tag] = 100 * (pdm - mdm) / atr
    f["adx_" + tag] = (100 * (pdm - mdm).abs() / (pdm + mdm).replace(0, np.nan)).ewm(alpha=1 / (14 * s), adjust=False).mean()
    mid, sd = c.rolling(20 * s).mean(), c.rolling(20 * s).std()
    f["bbpos_" + tag] = (c - mid) / (2 * sd).replace(0, np.nan)
    f["bbwidth_" + tag] = 4 * sd / mid
    e20, e50 = c.ewm(span=20 * s, adjust=False).mean(), c.ewm(span=50 * s, adjust=False).mean()
    f["ema_dist_" + tag] = (c - e20) / atr
    f["ema_stack_" + tag] = np.sign(e20 - e50)
    f["ema_slope_" + tag] = (e50 - e50.shift(4 * s)) / atr
    kijun = (h.rolling(26 * s).max() + l.rolling(26 * s).min()) / 2
    f["kijun_" + tag] = (c - kijun) / atr
    f["ret_" + tag] = np.log(c / c.shift(4 * s)) / (np.log(c).diff().rolling(96).std() * np.sqrt(4 * s))
    f["hhll_" + tag] = ((h > h.shift(s)).astype(float) - (l < l.shift(s)).astype(float)).rolling(8 * s).mean()
    f["dist_high_" + tag] = (c - h.rolling(24 * s).max()) / atr
    f["dist_low_" + tag] = (c - l.rolling(24 * s).min()) / atr
    f["vol_ratio_" + tag] = tr.rolling(4 * s).mean() / tr.rolling(24 * s).mean()
    return f


def five_minute_block(path, index15):
    k = pd.read_parquet(path).sort_index()
    o, h, l, c, v = (k[x] for x in ("open", "high", "low", "close", "volume"))
    f = pd.DataFrame(index=k.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi_5m"] = rsi(c, 14)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_5m"] = (macd - macd.ewm(span=9, adjust=False).mean()) / atr
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    f["stoch_5m"] = (c - ll) / (hh - ll).replace(0, np.nan)
    f["ema_dist_5m"] = (c - c.ewm(span=20, adjust=False).mean()) / atr
    f["ret_5m"] = np.log(c / c.shift(3)) / (np.log(c).diff().rolling(288).std() * np.sqrt(3))
    delta = 2 * k["taker_buy_base"] - v
    f["cvd_5m"] = delta.rolling(36).sum() / v.rolling(36).sum().replace(0, np.nan)
    f["aggr_5m"] = (delta / v.replace(0, np.nan)).rolling(3).mean()
    f["ha_5m"] = heikin_streak(o, h, l, c)
    f["dist_high_5m"] = (c - h.rolling(12).max()) / atr
    f["dist_low_5m"] = (c - l.rolling(12).min()) / atr
    f["rsi_min_5m"] = f["rsi_5m"].rolling(3).min()
    f["rsi_max_5m"] = f["rsi_5m"].rolling(3).max()
    # A 5m candle opened at t closes at t+5m; a 15m bar opened at T closes at T+15m.
    f.index = f.index + pd.Timedelta(minutes=5)
    aligned = f.reindex(index15 + pd.Timedelta(minutes=15), method="ffill")
    aligned.index = index15
    return aligned


def build_groups(df, five_path):
    o, h, l, c, v = (df[x].astype(float) for x in ("open", "high", "low", "close", "volume"))
    groups = {"5m": five_minute_block(five_path, df.index)}
    g15 = scale_block(o, h, l, c, v, 1, "15m")
    g15["ha_15m"] = heikin_streak(o, h, l, c)
    for col in ("cz_aggression", "cz_cvd_z_16", "cz_cvd_z_64", "cz_absorption", "cz_vpin_64", "cz_efficiency_96",
                "funding_rate", "cz_funding_z_96"):
        g15[col] = df[col]
    groups["15m"] = g15
    groups["1h"] = scale_block(o, h, l, c, v, 4, "1h")
    groups["4h"] = scale_block(o, h, l, c, v, 16, "4h")
    groups["1d"] = scale_block(o, h, l, c, v, 96, "1d")
    return {k: g.replace([np.inf, -np.inf], np.nan).astype("float32") for k, g in groups.items()}


@numba.njit(cache=True)
def sim_mtf(side, entry, exit_signal, atr, trail_k, max_hold, close, active_from):
    pos = np.zeros(close.shape[0])
    state, extreme, entry_atr, age = 0, 0.0, 0.0, 0
    for i in range(active_from, close.shape[0]):
        c = close[i]
        if state != 0:
            age += 1
            extreme = max(extreme, c) if side > 0 else min(extreme, c)
            trailed = (side > 0 and c < extreme - trail_k * entry_atr) or (side < 0 and c > extreme + trail_k * entry_atr)
            if exit_signal[i] or trailed or age >= max_hold:
                state = 0
        if state == 0 and entry[i]:
            state, extreme, entry_atr, age = side, c, atr[i], 0
        pos[i] = state
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
    blocks = {"treino": (s0, train_end), "validacao": (val_start, val_end), "holdout": (hold_start, len(df))}
    groups = build_groups(df, ROOT / "data" / "external" / "binance_usdm_btcusdt_5m.parquet")
    print("grupos: %s (%.0fs)" % ({k: g.shape[1] for k, g in groups.items()}, time.time() - began), flush=True)
    close = df["close"].astype(float)
    closev = close.to_numpy()
    atr_pct = df["atr_15m"] / close
    tests = pd.date_range(df.index[s0] + pd.Timedelta(days=270), df.index[-1], freq="QS")
    report = {"horizons": {}, "ablation": {}, "hierarchy": {}}

    def walk_forward(features, H):
        target = (np.log(close.shift(-H) / close) / (atr_pct * np.sqrt(H))).astype("float32")
        fwd = np.log(close.shift(-H) / close)
        preds = pd.Series(np.nan, index=df.index)
        hi = pd.Series(np.nan, index=df.index)
        lo = pd.Series(np.nan, index=df.index)
        for q_start, q_end in zip(tests, list(tests[1:]) + [df.index[-1] + pd.Timedelta(minutes=15)]):
            train = (np.arange(len(df)) >= s0) & (df.index < q_start - pd.Timedelta(minutes=15 * H)) & target.notna().to_numpy()
            idx = np.flatnonzero(train)[::2]
            model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, min_samples_leaf=200,
                                                  l2_regularization=1.0, random_state=5).fit(features.iloc[idx], target.iloc[idx].clip(-5, 5))
            sel = (df.index >= q_start) & (df.index < q_end)
            preds[sel] = model.predict(features[sel])
            ins = model.predict(features.iloc[idx[-15000:]])
            hi[sel], lo[sel] = np.quantile(ins, 0.9), np.quantile(ins, 0.1)
        step = max(H, 4)
        rows = []
        for q in preds.dropna().index.to_period("Q").unique():
            sel = (preds.index.to_period("Q") == q) & preds.notna() & fwd.notna()
            p, r = preds[sel].iloc[::step], fwd[sel].iloc[::step]
            rows.append((float(spearmanr(p, r).correlation), float(r[p >= hi[sel].iloc[::step]].mean() - 2 * COST),
                         float(-r[p <= lo[sel].iloc[::step]].mean() - 2 * COST)))
        arr = np.array(rows)
        return {"ic_by_quarter": arr[:, 0].tolist(), "long_by_quarter": arr[:, 1].tolist(), "short_by_quarter": arr[:, 2].tolist(),
                "ic_mean": float(np.nanmean(arr[:, 0])), "ic_pos": int((arr[:, 0] > 0).sum()),
                "long_mean": float(np.nanmean(arr[:, 1])), "long_pos": int((arr[:, 1] > 0).sum()),
                "short_mean": float(np.nanmean(arr[:, 2])), "short_pos": int((arr[:, 2] > 0).sum()), "quarters": len(arr)}

    def line(label, r):
        return "  %-26s IC medio %+.3f (%d/%d tri +) | compra 10%% topo %+.2f%% liq/trade (%d/%d tri +) | venda 10%% fundo %+.2f%% (%d/%d tri +)" % (
            label, r["ic_mean"], r["ic_pos"], r["quarters"], 100 * r["long_mean"], r["long_pos"], r["quarters"],
            100 * r["short_mean"], r["short_pos"], r["quarters"])

    everything = pd.concat(groups.values(), axis=1)
    print("\nA) HORIZONTE DA PREVISAO (todos os tempos graficos):")
    for H, label in ((4, "1h a frente"), (16, "4h a frente"), (48, "12h a frente"), (96, "24h a frente"), (288, "3 dias a frente")):
        r = walk_forward(everything, H)
        report["horizons"][label] = r
        print(line(label, r), flush=True)

    print("\nB) DE ONDE VEM A INFORMACAO (tempos graficos usados pelo modelo):")
    sets = [("so 5m", ["5m"]), ("so 15m", ["15m"]), ("so 1h", ["1h"]), ("so 4h", ["4h"]), ("so diario", ["1d"]),
            ("5m + 15m", ["5m", "15m"]), ("4h + diario", ["4h", "1d"]), ("5m+15m+4h+diario", ["5m", "15m", "4h", "1d"]),
            ("todos", ["5m", "15m", "1h", "4h", "1d"])]
    for H, label in ((16, "4h"), (96, "24h")):
        report["ablation"][label] = {}
        print(" horizonte %s:" % label)
        for name, keys in sets:
            r = walk_forward(pd.concat([groups[k] for k in keys], axis=1), H)
            report["ablation"][label][name] = r
            print(line(name, r), flush=True)

    ret = np.concatenate([[0.0], closev[1:] / closev[:-1] - 1.0])
    funding = df["funding_rate"].to_numpy(np.float64) * ((df.index.hour % 8 == 0) & (df.index.minute == 0))
    atr1h = df["atr_1h"].to_numpy(np.float64)

    def evaluate(pos):
        held = np.concatenate([[0.0], pos[:-1]])
        strat = held * ret - held * funding - COST * np.abs(np.diff(np.concatenate([[0.0], held])))
        out = {}
        for name, (a, b) in blocks.items():
            s = pd.Series(strat[a:b], index=df.index[a:b])
            eq = (1 + s).cumprod()
            daily = eq.resample("1D").last().pct_change().dropna()
            trades = int((np.abs(np.diff(np.concatenate([[0.0], held[a:b]]))) > 0).sum()) // 2
            out[name] = [float(eq.iloc[-1] - 1), float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0,
                         float((1 - eq / eq.cummax()).max()), trades]
        q = pd.Series(strat[s0:], index=df.index[s0:])
        out["trimestres"] = {str(k): float(v) for k, v in ((1 + q).groupby(q.index.to_period("Q")).prod() - 1).items()}
        return out

    g5, g15, g4h, g1d = groups["5m"], groups["15m"], groups["4h"], groups["1d"]
    print("\nC) HIERARQUIA: tendencia no tempo maior + gatilho no tempo menor (escolha pelo Sharpe de TREINO):")
    for side, name in ((1, "BULL"), (-1, "BEAR")):
        filters = {
            "4h EMAs": (g4h["ema_stack_4h"] * side > 0) & (g4h["ema_slope_4h"] * side > 0),
            "diario EMAs": (g1d["ema_stack_1d"] * side > 0) & (g1d["ema_slope_1d"] * side > 0),
            "4h DMI+ADX": (g4h["dmi_4h"] * side > 0) & (g4h["adx_4h"] > 20),
        }
        r15, r5 = g15["rsi_15m"], g5["rsi_5m"]
        if side > 0:
            triggers = {
                "RSI15m sai de sobrevenda": (r15 > 40) & (r15.shift(1) <= 40) & (r15.rolling(8).min().shift(1) < 35),
                "RSI5m sai de sobrevenda": (r5 > 40) & (g5["rsi_min_5m"].rolling(4).min().shift(1) < 30),
                "HA15m vira verde": (g15["ha_15m"] == 1) & (g15["ha_15m"].shift(1) <= -3),
                "fluxo 15m vira comprador": (g15["cz_cvd_z_16"] > 0) & (g15["cz_cvd_z_16"].rolling(8).min().shift(1) < -1),
            }
            exits = {"topo 15m": (r15 < 65) & (r15.rolling(8).max().shift(1) > 70)}
        else:
            triggers = {
                "RSI15m sai de sobrecompra": (r15 < 60) & (r15.shift(1) >= 60) & (r15.rolling(8).max().shift(1) > 65),
                "RSI5m sai de sobrecompra": (r5 < 60) & (g5["rsi_max_5m"].rolling(4).max().shift(1) > 70),
                "HA15m vira vermelho": (g15["ha_15m"] == -1) & (g15["ha_15m"].shift(1) >= 3),
                "fluxo 15m vira vendedor": (g15["cz_cvd_z_16"] < 0) & (g15["cz_cvd_z_16"].rolling(8).max().shift(1) > 1),
            }
            exits = {"fundo 15m": (r15 > 35) & (r15.rolling(8).min().shift(1) < 30)}
        rows = []
        for (fname, filt), (tname, trig), exit_mode, trail, hold in itertools.product(
                filters.items(), triggers.items(), ("virada menor", "tendencia maior acaba", "qualquer das duas"), (2.0, 3.0), (192, 384)):
            entry = (filt & trig).fillna(False).to_numpy()
            lower = list(exits.values())[0].fillna(False).to_numpy()
            upper_end = (~filt).fillna(True).to_numpy()
            exit_signal = lower if exit_mode == "virada menor" else upper_end if exit_mode == "tendencia maior acaba" else (lower | upper_end)
            pos = sim_mtf(side, entry, exit_signal, atr1h, trail, hold, closev, s0)
            rows.append(("%s | %s | sai: %s trail=%.0f max=%d" % (fname, tname, exit_mode, trail, hold), evaluate(pos)))
        rows = [x for x in rows if x[1]["treino"][3] >= 30]
        rows.sort(key=lambda x: x[1]["treino"][1], reverse=True)
        print(" %s:" % name)
        for label, r in rows[:6]:
            t, v, h = r["treino"], r["validacao"], r["holdout"]
            print("  %-78s | TR %+5.1f%% sh %5.2f dd %4.1f%% n=%3d | VAL %+5.1f%% sh %5.2f n=%3d | HO %+5.1f%% sh %5.2f n=%3d | %s" % (
                label, 100 * t[0], t[1], 100 * t[2], t[3], 100 * v[0], v[1], v[3], 100 * h[0], h[1], h[3],
                " ".join("%s:%+.0f" % (k[-2:], 100 * x) for k, x in r["trimestres"].items())))
        report["hierarchy"][name] = [{"rule": label, **r} for label, r in rows[:15]]

    out = ROOT / "reports" / "multitimeframe_research.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nrelatorio: %s (%.0fs)" % (out, time.time() - began))
    return 0


if __name__ == "__main__":
    sys.exit(main())
