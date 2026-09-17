"""Where is price going? Each timeframe's view of the day, verified on BTC.

The bot operates inside the day. It does not need weeks of daily history: it
reads the day it is in (today's candle so far, yesterday's candle, today's
buyer flow) and the intraday timeframes, and has to know where price is going:
the next 15 minutes, hour, 4 hours, the rest of the day, the next 24 hours.

1. Each timeframe's state on its own (5m, 15m, 1h, 4h, day): when it says up,
   how much more often is price higher at each horizon than when it says down,
   and what following it returns, quarter by quarter.
2. A learned probability of direction from each timeframe's day-view features
   alone and from all of them, refit every quarter on the past only: AUC by
   quarter, and the net return of acting only when it is confident (in-sample
   15%/85% cuts, 0.10% round trip).
3. The Bull and Bear trades of the recent simulation split by whether they went
   with or against the all-timeframe probability at entry.

Built from BTCUSDT perpetual 5m candles; at every 15m decision only candles
closed by then are used. Last two years. Nothing is selected here.

    python scripts/research_day_view.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FIVE = ROOT / "data" / "external" / "binance_usdm_btcusdt_5m.parquet"
HORIZONS = {"15m": pd.Timedelta("15min"), "1h": pd.Timedelta("1h"), "4h": pd.Timedelta("4h"),
            "resto do dia": None, "24h": pd.Timedelta("24h")}
COST_BPS = 10.0


def candles(five, rule):
    step = pd.Timedelta(rule)
    agg = five.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "taker_buy_base": "sum"})
    agg = agg.dropna(subset=["close"])
    agg = agg[agg.index + step <= five.index[-1] + pd.Timedelta("5min")]
    agg.index = agg.index + step  # indexed by close time: usable from then on
    return agg


def build(five):
    b5, b15, b1h, b4h, b1d = (candles(five, r) for r in ("5min", "15min", "1h", "4h", "1D"))
    T = b15.index
    price = b15["close"]
    c = price.to_numpy()

    def at(series):
        return series.reindex(T, method="ffill").to_numpy()

    tr = pd.concat([b1h.high - b1h.low, (b1h.high - b1h.close.shift()).abs(), (b1h.low - b1h.close.shift()).abs()], axis=1).max(axis=1)
    atr = at(tr.rolling(24).mean())
    day = (T - pd.Timedelta("15min")).floor("1D")
    key = np.asarray(day)
    day_open = five["open"].reindex(day).to_numpy()
    hi = b15["high"].groupby(key).cummax().to_numpy()
    lo = b15["low"].groupby(key).cummin().to_numpy()
    day_flow = (b15["taker_buy_base"].groupby(key).cumsum() / b15["volume"].groupby(key).cumsum()).to_numpy()
    prev = b1d.reindex(day)
    prev_range = (prev.high - prev.low).to_numpy()
    ema = lambda s, n: s.ewm(span=n, adjust=False).mean()  # noqa: E731
    flow = lambda b, n: b.taker_buy_base.rolling(n).sum() / b.volume.rolling(n).sum()  # noqa: E731
    back = lambda h: price.reindex(T - pd.Timedelta(h)).to_numpy()  # noqa: E731
    ema4, ema1h = at(ema(b4h.close, 6)), at(ema(b1h.close, 24))
    e15 = at(ema(b15.close, 20)) - at(ema(b15.close, 50))
    e5 = at(ema(b5.close, 20)) - at(ema(b5.close, 50))
    last4 = at(b4h.close - b4h.open)
    last1 = at(b1h.close - b1h.open)
    last15 = at(b15.close - b15.open)
    last5 = at(b5.close - b5.open)

    groups = {
        "dia": pd.DataFrame({
            "desde_abertura": (c - day_open) / atr, "posicao_range_hoje": np.nan_to_num((c - lo) / (hi - lo), nan=0.5),
            "range_hoje_vs_ontem": (hi - lo) / prev_range, "ontem": (prev.close - prev.open).to_numpy() / prev_range,
            "ontem_fechou_em": (prev.close - prev.low).to_numpy() / prev_range, "vs_max_ontem": (c - prev.high.to_numpy()) / atr,
            "vs_min_ontem": (c - prev.low.to_numpy()) / atr, "fluxo_dia": day_flow - 0.5, "hora": T.hour + T.minute / 60}, index=T),
        "4h": pd.DataFrame({"ultimo_4h": last4 / atr, "vs_ema6_4h": (c - ema4) / atr, "fluxo_4h": at(flow(b4h, 1)) - 0.5,
                            "ultimas_12h": (c - back("12h")) / atr}, index=T),
        "1h": pd.DataFrame({"ultimo_1h": last1 / atr, "vs_ema24_1h": (c - ema1h) / atr, "ultimas_4h": (c - back("4h")) / atr,
                            "fluxo_1h": at(flow(b1h, 1)) - 0.5}, index=T),
        "15m": pd.DataFrame({"ema20_50_15m": e15 / atr, "ultimo_15m": last15 / atr, "ultima_1h": (c - back("1h")) / atr,
                             "fluxo_1h_15m": at(flow(b15, 4)) - 0.5}, index=T),
        "5m": pd.DataFrame({"ema20_50_5m": e5 / atr, "ultimo_5m": last5 / atr, "fluxo_15m_5m": at(flow(b5, 3)) - 0.5,
                            "ultimos_15m": (c - at(b5.close.shift(3))) / atr}, index=T),
    }
    s = np.sign
    breakout = np.where(c > prev.high.to_numpy(), 1, np.where(c < prev.low.to_numpy(), -1, 0))
    aligned = s(c - day_open) + s(c - ema4) + s(c - ema1h) + s(e15)
    states = {
        "dia": {"preco vs abertura do dia": s(c - day_open), "ontem verde/vermelho": s((prev.close - prev.open).to_numpy()),
                "rompeu maxima/minima de ontem": breakout, "metade de cima/baixo do range de hoje": s((c - lo) / (hi - lo) - 0.5),
                "fluxo comprador/vendedor do dia": s(day_flow - 0.5)},
        "4h": {"ultimo candle 4h": s(last4), "preco vs EMA6 do 4h (1 dia)": s(c - ema4)},
        "1h": {"ultimo candle 1h": s(last1), "ultimas 4h subindo/caindo": s(c - back("4h")), "preco vs EMA24 do 1h (1 dia)": s(c - ema1h)},
        "15m": {"ultimo candle 15m": s(last15), "EMA20 vs EMA50 15m": s(e15), "fluxo da ultima 1h": s(at(flow(b15, 4)) - 0.5)},
        "5m": {"ultimo candle 5m": s(last5), "EMA20 vs EMA50 5m": s(e5)},
        "todos": {"dia+4h+1h+15m alinhados": np.where(aligned == 4, 1, np.where(aligned == -4, -1, 0))},
    }
    fwd = {}
    for name, h in HORIZONS.items():
        end = day + pd.Timedelta("1D") if h is None else T + h
        r = np.log(price.reindex(end).to_numpy() / c)
        if h is None:
            r[np.asarray(end == T)] = np.nan
        fwd[name] = r
    return T, c, groups, states, fwd


def main() -> int:
    began = time.time()
    five = pd.read_parquet(FIVE).sort_index()
    T, c, groups, states, fwd = build(five)
    start = T[-1] - pd.Timedelta(days=730)
    univ = np.asarray(T >= start)
    quarter = T.tz_localize(None).to_period("Q")
    quarters = [q for q in pd.unique(quarter[univ]) if (univ & (quarter == q)).sum() >= 2000]
    report = {"states": {}, "model": {}, "filter": {}}
    print("universo %s -> %s, %d trimestres (%.0fs)" % (start.date(), T[-1].date(), len(quarters), time.time() - began), flush=True)

    print("\n1) CADA TEMPO GRAFICO SOZINHO: P(alta | sinal de alta) - P(alta | sinal de queda), e retorno de seguir o sinal")
    for tf, items in states.items():
        for sname, st in items.items():
            for hname, r in fwd.items():
                gaps, follows = [], []
                for q in quarters:
                    sel = univ & (quarter == q) & ~np.isnan(r) & (st != 0)
                    up, down = sel & (st > 0), sel & (st < 0)
                    if up.sum() < 200 or down.sum() < 200:
                        continue
                    gaps.append((r[up] > 0).mean() - (r[down] > 0).mean())
                    follows.append(1e4 * np.mean(st[sel] * r[sel]))
                if not gaps:
                    continue
                active = float(np.mean(st[univ] != 0))
                report["states"]["%s | %s | %s" % (tf, sname, hname)] = {"gap_by_quarter": gaps, "follow_bps_by_quarter": follows, "active": active}
                print("  %-5s %-38s %-12s dif %+5.1f pp (%d/%d tri +) | seguir %+6.1f bps (%d/%d tri +)" % (
                    tf, sname, hname, 100 * np.mean(gaps), sum(g > 0 for g in gaps), len(gaps),
                    np.mean(follows), sum(f > 0 for f in follows), len(follows)), flush=True)

    print("\n2) PROBABILIDADE APRENDIDA (fora da amostra, re-treino por trimestre so com o passado):")
    tests = pd.date_range((start + pd.Timedelta(days=270)).normalize(), T[-1], freq="QS")
    feature_sets = dict(groups)
    feature_sets["todos"] = pd.concat(list(groups.values()), axis=1)
    probs = {}
    for hname, r in fwd.items():
        horizon = HORIZONS[hname] or pd.Timedelta("1D")
        step = max(int(horizon / pd.Timedelta("15min")), 4)
        y = pd.Series(r > 0, index=T).where(~np.isnan(r))
        for fname, X in feature_sets.items():
            prob = np.full(len(T), np.nan)
            lo_cut = np.full(len(T), np.nan)
            hi_cut = np.full(len(T), np.nan)
            for q_start, q_end in zip(tests, list(tests[1:]) + [T[-1] + pd.Timedelta("15min")]):
                train = univ & np.asarray(T < q_start - horizon) & y.notna().to_numpy() & X.notna().all(axis=1).to_numpy()
                idx = np.flatnonzero(train)[::3]
                model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=300,
                                                       l2_regularization=1.0, random_state=5).fit(X.iloc[idx], y.iloc[idx].astype(int))
                sel = np.asarray((T >= q_start) & (T < q_end))
                prob[sel] = model.predict_proba(X[sel])[:, 1]
                ins = model.predict_proba(X.iloc[idx])[:, 1]
                lo_cut[sel], hi_cut[sel] = np.quantile(ins, 0.15), np.quantile(ins, 0.85)
            probs[(hname, fname)] = prob
            aucs, longs, shorts = [], [], []
            for q in pd.unique(quarter[~np.isnan(prob)]):
                sel = (quarter == q) & ~np.isnan(prob) & ~np.isnan(r)
                if sel.sum() < 2000:
                    continue
                pick = np.flatnonzero(sel)[::step]
                aucs.append(roc_auc_score(r[pick] > 0, prob[pick]))
                long_sel, short_sel = sel & (prob >= hi_cut), sel & (prob <= lo_cut)
                longs.append(1e4 * r[long_sel].mean() - COST_BPS if long_sel.sum() >= 50 else np.nan)
                shorts.append(-1e4 * r[short_sel].mean() - COST_BPS if short_sel.sum() >= 50 else np.nan)
            report["model"]["%s | %s" % (hname, fname)] = {"auc": aucs, "long_net_bps": longs, "short_net_bps": shorts}
            print("  %-12s %-6s AUC %.3f (%d/%d tri >0.5) | compra confiante %+6.1f bps liq (%d tri +) | venda confiante %+6.1f bps liq (%d tri +)" % (
                hname, fname, np.mean(aucs), sum(a > 0.5 for a in aucs), len(aucs), np.nanmean(longs), sum(x > 0 for x in longs),
                np.nanmean(shorts), sum(x > 0 for x in shorts)), flush=True)

    print("\n3) TRADES DA SIMULACAO (Bull modelo 24h, Bear tendencia 4h) a favor / contra a probabilidade de todos os tempos:")
    from simulation_trades import trades_from_simulation
    history = pd.read_parquet(ROOT / "data" / "history_causal.parquet").sort_index()
    hstart = history.index[-1] - pd.Timedelta(days=730)
    hdf = history.loc[history.index >= hstart - pd.Timedelta(days=45)].ffill().fillna(0.0)
    trades = trades_from_simulation(hdf, int(np.searchsorted(hdf.index, hstart)))
    for hname in ("1h", "4h", "resto do dia", "24h"):
        p = pd.Series(probs[(hname, "todos")], index=T)
        for agent, rows in trades.items():
            frame = pd.DataFrame(rows, columns=["entry", "ret"])
            frame["p"] = p.reindex(pd.DatetimeIndex(frame["entry"])).to_numpy()
            frame = frame.dropna()
            favor = frame["p"] > 0.5 if agent == "bull" else frame["p"] < 0.5
            w, a = frame[favor], frame[~favor]
            report["filter"]["%s | %s" % (hname, agent)] = {"with": [len(w), float(w.ret.mean()), float((w.ret > 0).mean())],
                                                            "against": [len(a), float(a.ret.mean()), float((a.ret > 0).mean())]}
            print("  prob %-12s %-4s a favor: %3d trades %+.2f%% medio acerto %2.0f%% | contra: %3d trades %+.2f%% medio acerto %2.0f%%" % (
                hname, agent.upper(), len(w), 100 * w.ret.mean(), 100 * (w.ret > 0).mean(), len(a), 100 * a.ret.mean(), 100 * (a.ret > 0).mean()))

    out = ROOT / "reports" / "day_view_research.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print("\nrelatorio: %s (%.0fs)" % (out, time.time() - began))
    return 0


if __name__ == "__main__":
    sys.exit(main())
