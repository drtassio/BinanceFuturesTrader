"""
research_5m_patterns.py
Procura padrões operáveis com o gráfico de 5 minutos como principal e 15m, 1h e
4h como contexto (só candles FECHADOS de cada timeframe maior).

(a) Previsibilidade: em cada início de onda do 5m (confirmação de CONFIRM a
    partir do último fundo/topo), os dados daquele instante separam as ondas que
    viram perna (andam TARGET) das que falham (voltam SWING)? AUC walk-forward.
(b) Regras: confirmação por degraus no 5m + rompimento + fluxo + filtro de
    timeframe maior, saída na estrutura de 15m/1h/4h. Escolha só no TREINO;
    validação e holdout são conferência.

Custos: 0,05% taker + 0,02% slippage por lado, funding real.

    python scripts/research_5m_patterns.py
"""
import itertools
import json
import sys
import warnings
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COST = 0.0014
TRAIN_END = pd.Timestamp("2025-12-31 23:55")
VAL_END = pd.Timestamp("2026-04-30 23:55")
BAR = pd.Timedelta("5min")


def load():
    p = pd.read_parquet(ROOT / "data/external_5m/binance_usdm_btcusdt_5m_flow.parquet").sort_index()
    p = p[~p.index.duplicated(keep="last")]
    f = pd.read_parquet(ROOT / "data/external_5m/binance_usdm_btcusdt_funding.parquet").sort_index()
    p["funding"] = f["funding_rate"].reindex(p.index).fillna(0.0)
    return p


def atr(d, n=14):
    tr = np.maximum(d.high - d.low, np.maximum((d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()))
    return tr.rolling(n).mean()


def htf(d, rule):
    """Candle de timeframe maior carimbado no candle de 5m que o fecha."""
    h = d.resample(rule, closed="left", label="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
         "taker_buy_base_volume": "sum"}).dropna()
    h["atr"] = atr(h)
    h["trend"] = np.sign(h.close.ewm(span=20).mean() - h.close.ewm(span=50).mean())
    h["struct_lo"] = h.low.rolling(2).min() - 0.5 * h.atr
    h["struct_hi"] = h.high.rolling(2).max() + 0.5 * h.atr
    h["flow"] = (2 * h.taker_buy_base_volume - h.volume) / h.volume
    h["mom"] = np.log(h.close).diff(3) / (h.atr / h.close)
    h["expansion"] = h.atr / h.atr.rolling(50).mean()
    h.index = h.index + pd.Timedelta(rule) - BAR
    return h


def resample_15m(d5):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
           "taker_buy_base_volume": "sum", "funding": "sum"}
    return d5.resample("15min", closed="left", label="left").agg(agg).dropna()


def inside_5m(d5, index15):
    """O que os três candles de 5m de dentro de cada candle de 15m mostram (causal)."""
    a5 = atr(d5)
    body5 = (d5.close - d5.open) / a5
    delta5 = 2 * d5.taker_buy_base_volume - d5.volume
    g = pd.DataFrame(index=d5.index)
    g["steps5_up_6"] = (body5 >= 1.0).astype(int).rolling(6).sum()
    g["steps5_dn_6"] = (body5 <= -1.0).astype(int).rolling(6).sum()
    g["body5_max_3"] = body5.rolling(3).max()
    g["body5_min_3"] = body5.rolling(3).min()
    g["flow5_agree_3"] = np.sign(delta5).rolling(3).sum()
    g["flow5_z_12"] = delta5.rolling(12).sum() / (delta5.rolling(48).std() * np.sqrt(12))
    g["eff5_6"] = (d5.close - d5.close.shift(6)) / d5.close.diff().abs().rolling(6).sum()
    g["last5_ret"] = np.log(d5.close).diff() / (a5 / d5.close)
    # o candle de 15m rotulado t fecha junto com o 5m rotulado t+10min
    g.index = g.index - pd.Timedelta("10min")
    return g.reindex(index15)


def features(d, higher=("15min", "1h", "4h")):
    f = pd.DataFrame(index=d.index)
    a = atr(d)
    f["atr"] = a
    f["body"] = (d.close - d.open) / a
    delta = 2 * d.taker_buy_base_volume - d.volume
    f["flow"] = delta / d.volume
    for w in (12, 48, 144, 288):
        f["cvd_%d" % w] = delta.rolling(w).sum() / (delta.rolling(w).std() * np.sqrt(w))
        f["eff_%d" % w] = (d.close - d.close.shift(w)) / d.close.diff().abs().rolling(w).sum()
        f["brk_up_%d" % w] = (d.close - d.high.rolling(w).max().shift(1)) / a
        f["brk_dn_%d" % w] = (d.close - d.low.rolling(w).min().shift(1)) / a
        f["box_%d" % w] = (d.high.rolling(w).max() - d.low.rolling(w).min()).shift(1) / a
    f["vol_exp"] = d.volume / d.volume.rolling(48).mean()
    for span in (20, 100, 288):
        f["ema_dist_%d" % span] = (d.close - d.close.ewm(span=span).mean()) / a
    f["hour_sin"] = np.sin(2 * np.pi * d.index.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * d.index.hour / 24)
    step = d.index[1] - d.index[0]
    for rule in higher:
        h = htf(d, rule)
        h.index = h.index + BAR - step      # carimbo no candle principal que fecha o maior
        h = h.reindex(d.index, method="ffill")
        tag = {"15min": "15m", "1h": "1h", "4h": "4h"}[rule]
        for c in ("trend", "flow", "mom", "expansion"):
            f["%s_%s" % (c, tag)] = h[c]
        f["struct_long_%s" % tag] = (d.close - h.struct_lo) / a
        f["struct_short_%s" % tag] = (h.struct_hi - d.close) / a
    return f


def wave_starts(close, swing, confirm, target):
    """Inícios de onda confirmados (causal) e se viraram perna (hindsight, só rótulo)."""
    n = len(close)
    out = []
    direction, ext, armed = 1, 0, True
    for i in range(1, n):
        if direction == 1:
            if close[i] > close[ext]:
                ext, armed = i, True
            if close[i] <= close[ext] * (1 - swing):
                direction, ext, armed = -1, i, True
        else:
            if close[i] < close[ext]:
                ext, armed = i, True
            if close[i] >= close[ext] * (1 + swing):
                direction, ext, armed = 1, i, True
        side = 0
        if armed and direction == -1 and close[i] >= close[ext] * (1 + confirm):
            side = 1
        elif armed and direction == 1 and close[i] <= close[ext] * (1 - confirm):
            side = -1
        if side:
            armed = False
            pivot = close[ext]
            y = None
            for j in range(i + 1, min(n, i + 1500)):
                if side * (close[j] / pivot - 1) >= target:
                    y = 1
                    break
                if side * (close[j] / close[i] - 1) <= -swing:
                    y = 0
                    break
            if y is not None:
                out.append((i, side, y))
    return pd.DataFrame(out, columns=["i", "side", "y"])


def predictability(d, f, waves):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    ev = wave_starts(d.close.to_numpy(float), *waves)
    ev["t"] = d.index[ev.i]
    ev = ev[ev.t >= "2024-10-01"].reset_index(drop=True)
    signed = [c for c in f.columns if not c.startswith(("hour", "atr", "box", "vol_exp", "expansion"))]
    X = f.iloc[ev.i.values].copy()
    X[signed] = X[signed].to_numpy() * ev.side.values[:, None]
    X["side"] = ev.side.values
    X = X.drop(columns=["atr"]).to_numpy(float)
    y = ev.y.to_numpy()
    q = ev.t.dt.to_period("Q")
    quarters = sorted(q.unique())
    p = np.full(len(y), np.nan)
    for qq in quarters[2:]:
        tr, te = (q < qq).values, (q == qq).values
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.04, min_samples_leaf=80)
        p[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    ok = ~np.isnan(p)
    top = p[ok] >= np.quantile(p[ok], 0.8)
    return {"events": int(len(ev)), "base_rate": float(y.mean()), "auc_out_of_sample": float(roc_auc_score(y[ok], p[ok])),
            "hit_rate_top20pct": float(y[ok][top].mean()), "hit_rate_all": float(y[ok].mean())}


def run_rule(d, f, body_k, steps, window, brk_w, htf_filter, exit_tf, use_flow=True, confirm5="none"):
    C, H, L = d.close.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float)
    fund = d.funding.to_numpy(float)
    body = f.body.to_numpy()
    up = pd.Series((body >= body_k).astype(int)).rolling(window).sum().to_numpy()
    dn = pd.Series((body <= -body_k).astype(int)).rolling(window).sum().to_numpy()
    bu, bd = f["brk_up_%d" % brk_w].to_numpy(), f["brk_dn_%d" % brk_w].to_numpy()
    cvd = f.cvd_12.to_numpy()
    trend = {tf: f["trend_%s" % tf].to_numpy() for tf in ("15m", "1h", "4h") if "trend_%s" % tf in f}
    s_long, s_short = f["struct_long_%s" % exit_tf].to_numpy(), f["struct_short_%s" % exit_tf].to_numpy()
    a = f.atr.to_numpy()
    idx = d.index
    if confirm5 != "none":
        s5u, s5d = f.steps5_up_6.to_numpy(), f.steps5_dn_6.to_numpy()
        fl5 = f.flow5_agree_3.to_numpy()
    trades, pos = [], 0
    for i in range(300, len(C)):
        if pos:
            fa += fund[i] * pos
            stop_now = C[i] - s_long[i] * a[i] if pos > 0 else C[i] + s_short[i] * a[i]
            stop = max(stop, stop_now) if pos > 0 else min(stop, stop_now)
            if (L[i] <= stop) if pos > 0 else (H[i] >= stop):
                px = min(stop, C[i - 1]) if pos > 0 else max(stop, C[i - 1])
                trades.append((idx[e0], pos, pos * (px / ep - 1) - COST - fa))
                pos = 0
            continue
        for side in (1, -1):
            cnt = up[i] if side > 0 else dn[i]
            brk = bu[i] > 0 if side > 0 else bd[i] < 0
            if cnt < steps or not brk or np.sign(body[i]) != side:
                continue
            if use_flow and np.sign(cvd[i]) != side:
                continue
            if htf_filter != "none" and any(trend[tf][i] != side for tf in htf_filter.split("+")):
                continue
            if confirm5 == "degraus5" and (s5u[i] if side > 0 else s5d[i]) < 2:
                continue
            if confirm5 == "fluxo5" and side * fl5[i] < 3:
                continue
            pos, ep, e0, fa = side, C[i], i, 0.0
            stop = C[i] - s_long[i] * a[i] if side > 0 else C[i] + s_short[i] * a[i]
            break
    return pd.DataFrame(trades, columns=["t", "side", "net"])


def stats(t):
    if t.empty:
        return {"n": 0, "pf": 0.0, "ret": 0.0, "long": 0, "short": 0}
    w, l = t.net[t.net > 0].sum(), -t.net[t.net <= 0].sum()
    return {"n": int(len(t)), "pf": round(float(w / l) if l else 99.0, 2),
            "ret": round(float(((1 + t.net).prod() - 1) * 100), 1),
            "long": int((t.side > 0).sum()), "short": int((t.side < 0).sum())}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", choices=("5m", "15m"), default="15m")
    primary = ap.parse_args().primary
    d5 = load()
    if primary == "15m":
        d = resample_15m(d5)
        f = features(d, higher=("1h", "4h"))
        f = f.join(inside_5m(d5, d.index))
        print("principal 15m: %d candles de %s a %s | contexto: 5m por dentro, 1h e 4h por cima" % (len(d), d.index.min(), d.index.max()))
        waves, windows, breaks, exits, confirms = (0.006, 0.0025, 0.009), [4, 12], [48, 144], ["1h", "4h"], ["none", "degraus5", "fluxo5"]
    else:
        d = d5
        f = features(d, higher=("15min", "1h", "4h"))
        print("principal 5m: %d candles de %s a %s | contexto: 15m, 1h e 4h por cima" % (len(d), d.index.min(), d.index.max()))
        waves, windows, breaks, exits, confirms = (0.005, 0.002, 0.008), [6, 12], [144, 288], ["15m", "1h", "4h"], ["none"]

    print("\n(a) previsibilidade do início das ondas no %s (onda %.2f%%, confirma %.2f%%, perna %.2f%%)"
          % (primary, waves[0] * 100, waves[1] * 100, waves[2] * 100))
    pred = predictability(d, f, waves)
    print("    %d inícios | viram perna %.0f%% | AUC fora da amostra %.3f | 20%% mais confiantes viram perna %.0f%% (todos %.0f%%)"
          % (pred["events"], pred["base_rate"] * 100, pred["auc_out_of_sample"], pred["hit_rate_top20pct"] * 100, pred["hit_rate_all"] * 100))

    print("\n(b) regras de confirmação no %s + contexto multi-timeframe" % primary)
    rows = []
    grid = itertools.product([1.0, 1.5], [2, 3], windows, breaks,
                             ["none", "1h", "4h", "1h+4h"], exits, confirms)
    for body_k, steps, window, brk_w, htf_filter, exit_tf, confirm5 in grid:
        if steps > window:
            continue
        t = run_rule(d, f, body_k, steps, window, brk_w, htf_filter, exit_tf, confirm5=confirm5)
        blocks = {"treino": t[t.t <= TRAIN_END], "validacao": t[(t.t > TRAIN_END) & (t.t <= VAL_END)], "holdout": t[t.t > VAL_END]}
        row = {"degrau_atr": body_k, "degraus": steps, "janela": window, "rompe": brk_w, "filtro_htf": htf_filter,
               "saida": exit_tf, "confirma_5m": confirm5}
        for name, sub in blocks.items():
            s = stats(sub)
            row.update({"%s_n" % name: s["n"], "%s_pf" % name: s["pf"], "%s_ret%%" % name: s["ret"]})
        rows.append(row)
    g = pd.DataFrame(rows)
    g = g[g["treino_n"] >= 30]
    pd.set_option("display.width", 250)
    print(g.sort_values("treino_pf", ascending=False).head(15).to_string(index=False))
    good = g[g.treino_pf > 1.2]
    print("\n    regras com PF_treino > 1,2: %d de %d | destas com PF > 1 na validação: %d | e também no holdout: %d"
          % (len(good), len(g), (good.validacao_pf > 1).sum(), ((good.validacao_pf > 1) & (good.holdout_pf > 1)).sum()))
    out = ROOT / ("reports/research_patterns_primary_%s.json" % primary)
    out.write_text(json.dumps({"predictability": pred, "rules": g.to_dict("records")}, indent=1, default=float), encoding="utf-8")
    print("    detalhes: %s" % out)


if __name__ == "__main__":
    main()
