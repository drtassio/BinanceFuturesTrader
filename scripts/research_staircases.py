"""Staircases on BTC 15m: every candle is a step.

Find every rise and fall made of steps, and measure what happens when an agent
does not forecast anything: it enters once the staircase has started and formed
(k steps) and leaves when the staircase breaks.

Detector (causal, closed candles only). An up staircase starts when a candle
closes above the highest high of the previous L candles: step 1. Every later
close above the staircase's highest close is another step and lifts the support
to the lowest low of the last L candles. Pullbacks are allowed while closes stay
at or above the support; the first close below it ends the staircase. Down
staircases are the mirror image. L sets how deep a correction may be: 2 candles
(30 min), 4 (1 hour), 8 (2 hours).

1. Map: how many staircases, how many steps, how far they travel. Small ones
   (< 1%) are the Ranger's, big ones (>= 3%) the Bull's and the Bear's.
2. Entry on formation: enter at the close of step k, exit at the close that
   breaks the support; 0.05% per side plus funding. For each L and k, on the
   train block: trades, hit rate, average net, and how often a staircase formed
   to k steps went on to travel 3% or more. (L, k) is chosen on train only.
3. High-probability filter: a classifier on what was visible at the entry (the
   staircase so far, the day view of every timeframe, 15m/1h/4h indicators)
   learns which formed staircases pay. The cut is chosen out-of-fold on train;
   validation and holdout are read once.

BTC only, last two years.

    python scripts/research_staircases.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_day_view import FIVE, build as build_day_view  # noqa: E402
from research_walkforward_patterns import build_features  # noqa: E402

COST = 0.0005
EMBARGO = 288
LOOKBACKS = (2, 4, 8)
STEPS = (1, 2, 3, 4, 5, 6, 8)
MAX_STEPS = 64
BIG_MOVE = 0.03
SMALL_MOVE = 0.01


@numba.njit(cache=True)
def detect(hi, lo, cl, L, a, b):
    """Up staircases on (hi, lo, cl); pass negated, swapped prices for down staircases."""
    cap = (b - a) // 2 + 1
    starts = np.full(cap, -1, np.int64)
    ends = np.full(cap, -1, np.int64)
    nsteps = np.zeros(cap, np.int64)
    peak = np.zeros(cap)
    bottom = np.zeros(cap)
    step_bar = np.full((cap, MAX_STEPS), -1, np.int64)
    step_support = np.zeros((cap, MAX_STEPS))
    count = 0
    active = False
    steps, best, support, top = 0, 0.0, 0.0, 0.0
    for i in range(a + L, b):
        c = cl[i]
        if active:
            if c < support:
                ends[count] = i
                nsteps[count] = steps
                peak[count] = top
                count += 1
                active = False
                continue
            if hi[i] > top:
                top = hi[i]
            if c > best:
                best = c
                m = lo[i]
                for t in range(i - L + 1, i):
                    if lo[t] < m:
                        m = lo[t]
                if m > support:
                    support = m
                if steps < MAX_STEPS:
                    step_bar[count, steps] = i
                    step_support[count, steps] = support
                steps += 1
        else:
            prev_high = hi[i - L]
            for t in range(i - L + 1, i):
                if hi[t] > prev_high:
                    prev_high = hi[t]
            if c > prev_high:
                active = True
                steps, best, top = 1, c, hi[i]
                support = lo[i]
                bot = lo[i - L]
                for t in range(i - L + 1, i):
                    if lo[t] < support:
                        support = lo[t]
                    if lo[t] < bot:
                        bot = lo[t]
                if support < bot:
                    bot = support
                starts[count] = i
                bottom[count] = bot
                step_bar[count, 0] = i
                step_support[count, 0] = support
    if active:
        nsteps[count] = steps
        peak[count] = top
        count += 1
    return starts[:count], ends[:count], nsteps[:count], peak[:count], bottom[:count], step_bar[:count], step_support[:count]


def main() -> int:
    began = time.time()
    full = pd.read_parquet(ROOT / "data" / "history_causal.parquet").sort_index()
    start = full.index[-1] - pd.Timedelta(days=730)
    df = full.loc[full.index >= start - pd.Timedelta(days=45)].ffill().fillna(0.0)
    s0, n = int(np.searchsorted(df.index, start)), len(df)
    o, h, l, c, v = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "volume"))
    atr = df["atr_1h"].to_numpy(np.float64)
    fcum = np.cumsum(df["funding_rate"].to_numpy(np.float64) * ((df.index.hour % 8 == 0) & (df.index.minute == 0)))
    m = n - s0
    train_end = s0 + int(0.70 * m)
    val_start, val_end = train_end + EMBARGO, train_end + EMBARGO + int(0.15 * m)
    blocks = {"treino": (s0, train_end), "validacao": (val_start, val_end), "holdout": (val_end + EMBARGO, n)}
    days = m / 96
    quarter = df.index.tz_localize(None).to_period("Q")
    report = {"map": {}, "formation": {}, "chosen": {}, "filter": {}}

    stairs = {}
    for side, name in ((1, "alta"), (-1, "queda")):
        hi, lo, cl = (h, l, c) if side > 0 else (-l, -h, -c)
        for L in LOOKBACKS:
            st, en, ns, pk, bt, sb, ss = detect(hi, lo, cl, L, s0, n)
            move = (pk - bt) / np.abs(bt)
            stairs[(side, L)] = {"start": st, "end": en, "steps": ns, "move": move, "step_bar": sb, "step_support": side * ss, "bottom": side * bt}

    print("1) MAPA DAS ESCADAS %s -> %s (cada candle de 15m e um degrau)" % (df.index[s0].date(), df.index[-1].date()))
    for L in LOOKBACKS:
        for side, name in ((1, "alta"), (-1, "queda")):
            s = stairs[(side, L)]
            mv, ns = s["move"], s["steps"]
            dur = np.where(s["end"] >= 0, s["end"], n - 1) - s["start"]
            small, big = mv < SMALL_MOVE, mv >= BIG_MOVE
            mid = ~small & ~big
            row = {"count": int(len(mv)), "per_day": float(len(mv) / days), "median_steps": float(np.median(ns)),
                   "median_move": float(np.median(mv)), "small": int(small.sum()), "medium": int(mid.sum()), "big": int(big.sum()),
                   "big_median_steps": float(np.median(ns[big])) if big.any() else 0.0,
                   "big_median_hours": float(np.median(dur[big]) / 4) if big.any() else 0.0}
            report["map"]["L%d %s" % (L, name)] = row
            print("  quebra em %d candles | escadas de %-5s %5d (%4.1f/dia) | degraus mediana %2.0f | percurso mediano %.2f%% | pequenas <1%%: %4d (Ranger) | medias 1-3%%: %4d | grandes >=3%%: %3d (%.0f degraus, %.0fh medianos)" % (
                L, name, row["count"], row["per_day"], row["median_steps"], 100 * row["median_move"], row["small"], row["medium"], row["big"],
                row["big_median_steps"], row["big_median_hours"]))

    def trades_for(side, L, k):
        s = stairs[(side, L)]
        rows = []
        for idx in range(len(s["start"])):
            if s["steps"][idx] < k or k > MAX_STEPS:
                continue
            e = s["step_bar"][idx, k - 1]
            x = s["end"][idx] if s["end"][idx] >= 0 else n - 1
            net = side * (c[x] / c[e] - 1) - 2 * COST - side * (fcum[x] - fcum[e])
            rows.append((e, x, net, s["move"][idx] >= BIG_MOVE, idx))
        return rows

    def stats(rows):
        if not rows:
            return {"n": 0, "hit": 0.0, "mean": 0.0, "compound": 0.0, "t": 0.0, "p_big": 0.0, "hours": 0.0}
        net = np.array([r[2] for r in rows])
        return {"n": len(rows), "hit": float((net > 0).mean()), "mean": float(net.mean()), "compound": float(np.prod(1 + net) - 1),
                "t": float(net.mean() / net.std() * np.sqrt(len(net))) if net.std() > 0 else 0.0,
                "p_big": float(np.mean([r[3] for r in rows])), "hours": float(np.median([(r[1] - r[0]) / 4 for r in rows]))}

    def in_block(rows, name):
        a, b = blocks[name]
        return [r for r in rows if a <= r[0] < b]

    print("\n2) ENTRAR QUANDO A ESCADA SE FORMOU (degrau k), SAIR QUANDO QUEBRAR — bloco de TREINO (%s -> %s):" % (
        df.index[blocks["treino"][0]].date(), df.index[blocks["treino"][1] - 1].date()))
    chosen = {}
    for side, agent in ((1, "BULL"), (-1, "BEAR")):
        candidates = {}
        for L in LOOKBACKS:
            for k in STEPS:
                rows = trades_for(side, L, k)
                tr = stats(in_block(rows, "treino"))
                candidates[(L, k)] = (rows, tr)
                report["formation"]["%s L%d k%d" % (agent, L, k)] = tr
                print("  %s quebra %d | entra no degrau %d: %5d trades | acerto %2.0f%% | liquido medio %+.2f%% | composto %+8.1f%% | segura %4.1fh | virou escada grande (>=3%%) %2.0f%%" % (
                    agent, L, k, tr["n"], 100 * tr["hit"], 100 * tr["mean"], 100 * tr["compound"], tr["hours"], 100 * tr["p_big"]))
        eligible = {key: val for key, val in candidates.items() if val[1]["n"] >= 100}
        best = max(eligible, key=lambda key: eligible[key][1]["t"])
        rows = candidates[best][0]
        chosen[agent] = (side, best, rows)
        positives = {name: sum(1 for key in eligible if stats(in_block(candidates[key][0], name))["mean"] > 0) for name in ("validacao", "holdout")}
        report["chosen"][agent] = {"L": best[0], "k": best[1], **{name: stats(in_block(rows, name)) for name in blocks},
                                   "configs_positive": positives, "configs": len(eligible)}
        print("  -> %s escolhido no treino: quebra %d, entra no degrau %d" % (agent, *best))
        for name in blocks:
            x = report["chosen"][agent][name]
            a, b = blocks[name]
            print("     %-9s %4d trades | acerto %2.0f%% | liquido medio %+.2f%% | composto %+7.1f%% | buy&hold %+.1f%%" % (
                name, x["n"], 100 * x["hit"], 100 * x["mean"], 100 * x["compound"], 100 * (c[b - 1] / c[a] - 1)))
        print("     robustez: das %d configuracoes, positivas na validacao %d, no holdout %d" % (len(eligible), positives["validacao"], positives["holdout"]))
        net = pd.Series([r[2] for r in rows], index=df.index[[r[0] for r in rows]])
        by_q = net.groupby(quarter[[r[0] for r in rows]]).mean()
        print("     por trimestre (liquido medio por trade): " + " ".join("%s:%+.2f%%" % (str(q)[-4:], 100 * val) for q, val in by_q.items()))

    print("\n3) FILTRO DE ALTA PROBABILIDADE (o que era visivel na entrada) (%.0fs)" % (time.time() - began), flush=True)
    base = build_features(df)
    base = base.drop(columns=[col for col in base.columns if col.endswith("7d") or col in ("ret_672", "rv_ratio_96_672")])
    _, _, groups, _, _ = build_day_view(pd.read_parquet(FIVE).sort_index())
    day_view = pd.concat(list(groups.values()), axis=1).reindex(df.index + pd.Timedelta("15min"))
    day_view.index = df.index
    base = pd.concat([base, day_view], axis=1).replace([np.inf, -np.inf], np.nan).astype("float32")
    vol_mean = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    for agent, (side, (L, k), rows) in chosen.items():
        s = stairs[(side, L)]
        entries = np.array([r[0] for r in rows])
        net = np.array([r[2] for r in rows])
        stair = np.array([r[4] for r in rows])
        X = base.iloc[entries].reset_index(drop=True)
        X["subida_ate_agora_atr"] = side * (c[entries] - s["bottom"][stair]) / atr[entries]
        X["candles_desde_inicio"] = entries - s["start"][stair]
        X["corpo_degrau_atr"] = side * (c[entries] - o[entries]) / atr[entries]
        X["volume_degrau"] = v[entries] / vol_mean[entries]
        X["distancia_suporte_atr"] = side * (c[entries] - s["step_support"][stair, k - 1]) / atr[entries]
        y = (net > 0).astype(int)
        (ta, tb) = blocks["treino"]
        train = np.flatnonzero((entries >= ta) & (entries < tb))
        oof = np.full(len(rows), np.nan)

        def model():
            return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                                  l2_regularization=1.0, early_stopping=False, random_state=3)
        for fold in np.array_split(train, 5):
            lo_bar, hi_bar = entries[fold[0]] - EMBARGO, entries[fold[-1]] + EMBARGO
            keep = train[(entries[train] < lo_bar) | (entries[train] > hi_bar)]
            oof[fold] = model().fit(X.iloc[keep], y[keep]).predict_proba(X.iloc[fold])[:, 1]
        final = model().fit(X.iloc[train], y[train])
        score = final.predict_proba(X)[:, 1]
        options = {}
        for keep_share in (1.0, 0.5, 0.35, 0.25, 0.15):
            cut = np.quantile(oof[train], 1 - keep_share) if keep_share < 1 else -np.inf
            sel = net[train][oof[train] >= cut]
            if len(sel) >= 40:
                options[keep_share] = (cut, sel.mean() / sel.std() * np.sqrt(len(sel)), sel)
        share = max(options, key=lambda key: options[key][1])
        cut = options[share][0]
        print("  %s (quebra %d, degrau %d): filtro escolhido no treino fora da dobra = fica com os %.0f%% mais provaveis" % (agent, L, k, 100 * share))
        out = {"keep_share": share}
        for name, (a, b) in blocks.items():
            inside = (entries >= a) & (entries < b)
            use = oof if name == "treino" else score
            picked = inside & (use >= cut)
            everything = net[inside]
            chosen_net = net[picked]
            memorized = net[inside & (score >= cut)] if name == "treino" else None
            out[name] = {"all": [int(inside.sum()), float((everything > 0).mean()) if len(everything) else 0.0, float(np.prod(1 + everything) - 1)],
                         "filtered": [int(picked.sum()), float((chosen_net > 0).mean()) if len(chosen_net) else 0.0, float(np.prod(1 + chosen_net) - 1)]}
            label = name + (" (fora da dobra)" if name == "treino" else "")
            print("     %-24s todas: %4d trades acerto %2.0f%% composto %+7.1f%% | filtradas: %4d trades acerto %2.0f%% medio %+.2f%% composto %+7.1f%%" % (
                label, inside.sum(), 100 * out[name]["all"][1], 100 * out[name]["all"][2], picked.sum(), 100 * out[name]["filtered"][1],
                100 * (chosen_net.mean() if len(chosen_net) else 0.0), 100 * out[name]["filtered"][2]))
            if memorized is not None and len(memorized):
                print("     %-24s (decorado, so para comparar: %d trades acerto %2.0f%% composto %+.1f%%)" % (
                    "", len(memorized), 100 * (memorized > 0).mean(), 100 * (np.prod(1 + memorized) - 1)))
        report["filter"][agent] = out

    path = ROOT / "reports" / "staircases_research.json"
    path.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print("\nrelatorio: %s (%.0fs)" % (path, time.time() - began))
    return 0


if __name__ == "__main__":
    sys.exit(main())
