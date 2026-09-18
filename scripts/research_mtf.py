"""
research_mtf.py
Operar no 15m com o 5m como gatilho rápido e o 1h/4h como filtro lento.

Três famílias de entrada, todas causais (só candles FECHADOS de cada timeframe):
  A  rompimento confirmado: degraus no 15m + rompimento de 12h + aceleração no 5m
  B  correção dentro da tendência: 1h e 4h a favor, o 15m corrigiu contra, o 5m
     acelera de volta a favor (retomada; nada é previsto)
  C  alinhamento de momento: momento do 5m, 15m e 1h no sentido da tendência do 4h
Saídas: estrutura de 1h, estrutura de 4h ou trailing em ATR de 15m.

Escolha só no TREINO (>= 40 trades); validação e holdout são conferência.
Custos 0,14% por trade + funding real.

    python scripts/research_mtf.py
"""
import itertools
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_5m_patterns as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_END, VAL_END = R.TRAIN_END, R.VAL_END


def frame():
    d5 = R.load()
    d = R.resample_15m(d5)
    f = R.features(d, higher=("1h", "4h")).join(R.inside_5m(d5, d.index))
    # momento do 15m e distância à média de 1h (correção dentro da tendência)
    f["mom_15m"] = np.log(d.close).diff(4) / (f.atr / d.close)
    h1 = d.close.resample("1h", closed="left", label="left").last().dropna()
    ema1h = h1.ewm(span=20).mean()
    ema1h.index = ema1h.index + pd.Timedelta("1h") - pd.Timedelta("15min")
    f["dist_ema20_1h"] = (d.close - ema1h.reindex(d.index, method="ffill")) / f.atr
    return d, f


def entries(f, family, p):
    """Máscaras booleanas (long, short) de entrada por candle de 15m."""
    t1, t4 = f.trend_1h.values, f.trend_4h.values
    s5u, s5d = f.steps5_up_6.values, f.steps5_dn_6.values
    body = f.body.values
    cvd = f.cvd_12.values
    if family == "A":
        up = pd.Series((f.body >= 1.0).astype(int)).rolling(12).sum().values
        dn = pd.Series((f.body <= -1.0).astype(int)).rolling(12).sum().values
        lg = (up >= 3) & (f.brk_up_48.values > 0) & (body > 0) & (cvd > 0) & (s5u >= p["s5"])
        sh = (dn >= 3) & (f.brk_dn_48.values < 0) & (body < 0) & (cvd < 0) & (s5d >= p["s5"])
    elif family == "B":
        dist = f.dist_ema20_1h.values
        slow_up = (t1 > 0) & (t4 > 0) if p["slow"] == "1h+4h" else (t4 > 0)
        slow_dn = (t1 < 0) & (t4 < 0) if p["slow"] == "1h+4h" else (t4 < 0)
        # houve correção recente: nas últimas 8 barras o preço esteve abaixo da média de 1h
        was_below = pd.Series(dist < -p["pull"]).rolling(8).max().values > 0
        was_above = pd.Series(dist > p["pull"]).rolling(8).max().values > 0
        lg = slow_up & was_below & (s5u >= p["s5"]) & (body > 0) & (f.flow5_agree_3.values >= 2)
        sh = slow_dn & was_above & (s5d >= p["s5"]) & (body < 0) & (f.flow5_agree_3.values <= -2)
    else:  # C
        m5, m15, m1 = f.last5_ret.values, f.mom_15m.values, f.mom_1h.values
        k = p["mom"]
        lg = (t4 > 0) & (m5 > 0) & (m15 > k) & (m1 > 0) & (cvd > 0) & (s5u >= p["s5"])
        sh = (t4 < 0) & (m5 < 0) & (m15 < -k) & (m1 < 0) & (cvd < 0) & (s5d >= p["s5"])
    return np.nan_to_num(lg).astype(bool), np.nan_to_num(sh).astype(bool)


def simulate(d, f, lg, sh, exit_kind):
    C, H, L = d.close.values, d.high.values, d.low.values
    fund, a, idx = d.funding.values, f.atr.values, d.index
    if exit_kind.startswith("struct"):
        tf = exit_kind.split("_")[1]
        sl, ss = f["struct_long_%s" % tf].values, f["struct_short_%s" % tf].values
    else:
        mult = float(exit_kind.split("_")[1])
    trades, pos = [], 0
    for i in range(300, len(C)):
        if pos:
            fa += fund[i] * pos
            if exit_kind.startswith("struct"):
                cand = C[i] - sl[i] * a[i] if pos > 0 else C[i] + ss[i] * a[i]
            else:
                best = max(best, H[i]) if pos > 0 else min(best, L[i])
                cand = best - mult * a[i] if pos > 0 else best + mult * a[i]
            stop = max(stop, cand) if pos > 0 else min(stop, cand)
            if (L[i] <= stop) if pos > 0 else (H[i] >= stop):
                px = min(stop, C[i - 1]) if pos > 0 else max(stop, C[i - 1])
                trades.append((idx[e0], pos, pos * (px / ep - 1) - R.COST - fa))
                pos = 0
            continue
        side = 1 if lg[i] else (-1 if sh[i] else 0)
        if side:
            pos, ep, e0, fa, best = side, C[i], i, 0.0, C[i]
            if exit_kind.startswith("struct"):
                stop = C[i] - sl[i] * a[i] if side > 0 else C[i] + ss[i] * a[i]
            else:
                stop = C[i] - mult * a[i] if side > 0 else C[i] + mult * a[i]
    return pd.DataFrame(trades, columns=["t", "side", "net"])


def main():
    d, f = frame()
    print("principal 15m | gatilho 5m | filtro 1h/4h | %s a %s" % (d.index.min().date(), d.index.max().date()))
    grids = {
        "A": [{"s5": s} for s in (1, 2, 3)],
        "B": [{"slow": sl, "pull": pl, "s5": s} for sl in ("4h", "1h+4h") for pl in (0.5, 1.0, 2.0) for s in (1, 2)],
        "C": [{"mom": m, "s5": s} for m in (0.5, 1.0, 2.0) for s in (0, 1, 2)],
    }
    exits = ["struct_1h", "struct_4h", "atr_3", "atr_6"]
    rows = []
    for fam, params in grids.items():
        for p, ex in itertools.product(params, exits):
            lg, sh = entries(f, fam, p)
            t = simulate(d, f, lg, sh, ex)
            row = {"familia": fam, "params": json.dumps(p), "saida": ex}
            for name, sub in (("treino", t[t.t <= TRAIN_END]), ("validacao", t[(t.t > TRAIN_END) & (t.t <= VAL_END)]),
                              ("holdout", t[t.t > VAL_END])):
                s = R.stats(sub)
                row.update({"%s_n" % name: s["n"], "%s_pf" % name: s["pf"], "%s_ret" % name: s["ret"]})
            rows.append(row)
    g = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    ok = g[g.treino_n >= 40]
    for fam in ("A", "B", "C"):
        sub = ok[ok.familia == fam].sort_values("treino_pf", ascending=False).head(6)
        print("\nFamília %s (melhores pelo PF de TREINO):" % fam)
        print(sub.drop(columns="familia").to_string(index=False))
    good = ok[ok.treino_pf > 1.2]
    robust = good[(good.validacao_pf > 1) & (good.holdout_pf > 1)]
    print("\nPF_treino > 1,2: %d de %d | e PF > 1 na validação e no holdout: %d" % (len(good), len(ok), len(robust)))
    if len(robust):
        print(robust.sort_values("treino_pf", ascending=False).to_string(index=False))
    out = ROOT / "reports/research_mtf.json"
    out.write_text(g.to_json(orient="records", indent=1), encoding="utf-8")
    print("detalhes: %s" % out)


if __name__ == "__main__":
    main()
