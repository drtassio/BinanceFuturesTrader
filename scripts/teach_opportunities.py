"""Map every buy and sell opportunity of the last two years of BTC and teach a
student to see them, then look at where and why it errs.

Opportunity map (hindsight, with costs): a perfect trader that may be long,
short or flat on every 15m bar, pays 0.05% per side, an extra penalty for
every new position (1%) and for every hour it stays positioned (0.05%), so it
only takes legs where price really travels. It holds through corrections
smaller than about 1% and rides the leg: up legs are the Bull's (in at the
bottom, out at the top), down legs are the Bear's. Where price does not travel
(lateral), the same trader with a 0.2% penalty and no hourly cost takes the
small swings: those are the Ranger's.

Students see only what was known at each bar's close (the day view of every
timeframe plus 15m/1h/4h indicators, nothing older than a few days):
  v1 "perna":  is this bar inside a Bull leg, a Bear leg, or neither?
  v2 "fases":  start of a Bull leg (the bottom), its middle, its end (the top
               and the half hour after it), the same three for Bear, or lateral.
               Enters on "start", holds while "start"+"middle", leaves otherwise.
Labels are computed separately inside each block (train, validation, holdout),
so no block's labels use another block's prices. Entry/exit thresholds are
chosen on out-of-fold predictions of the train block only; validation and
holdout are read once.

Error analysis: legs caught or missed, entry lag, share of each leg captured,
time held after the leg ended, false trades and where they happened.

    python scripts/teach_opportunities.py
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "scripts"))
from research_day_view import FIVE, build as build_day_view  # noqa: E402
from research_walkforward_patterns import build_features  # noqa: E402

COST = 0.0005
EMBARGO = 288
BIG, SMALL = 0.01, 0.002
HOLD = 0.0005 / 4  # per 15m bar held in a leg


@numba.njit(cache=True)
def oracle(r, cost, penalty, allowed, hold):
    """Best flat(0)/long(1)/short(2) path: held returns minus costs, entry penalties and a cost per bar held."""
    n = r.shape[0]
    pos = np.array([0.0, 1.0, -1.0])
    best = np.empty((n, 3))
    back = np.zeros((n, 3), np.int8)
    for s in range(3):
        best[0, s] = -1e18 if (s != 0 and not allowed[0]) else -cost * abs(pos[s]) - (penalty if s != 0 else 0.0) + pos[s] * r[0] - (hold if s != 0 else 0.0)
    for i in range(1, n):
        for s in range(3):
            if s != 0 and not allowed[i]:
                best[i, s] = -1e18
                continue
            top, arg = -1e30, 0
            for p in range(3):
                v = best[i - 1, p] - cost * abs(pos[s] - pos[p]) - (penalty if (s != 0 and s != p) else 0.0)
                if v > top:
                    top, arg = v, p
            best[i, s] = top + pos[s] * r[i] - (hold if s != 0 else 0.0)
            back[i, s] = arg
    out = np.zeros(n, np.int8)
    s, top = 0, -1e30
    for k in range(3):
        v = best[n - 1, k] - cost * abs(pos[k])
        if v > top:
            top, s = v, k
    for i in range(n - 1, -1, -1):
        out[i] = s
        s = back[i, s]
    return out


@numba.njit(cache=True)
def policy(enter_long, hold_long, enter_short, hold_short, enter, leave, allow_long, allow_short, a, b):
    pos = np.zeros(enter_long.shape[0])
    state = 0
    for i in range(a, b):
        if state == 1 and hold_long[i] < leave:
            state = 0
        elif state == -1 and hold_short[i] < leave:
            state = 0
        if state == 0:
            if allow_long and enter_long[i] >= enter and (not allow_short or enter_long[i] >= enter_short[i]):
                state = 1
            elif allow_short and enter_short[i] >= enter and (not allow_long or enter_short[i] > enter_long[i]):
                state = -1
        pos[i] = state
    return pos


def runs(states, a, b):
    """(start, end_exclusive, state) for each run of a nonzero state inside [a, b)."""
    out, i = [], a
    while i < b:
        if states[i] != 0:
            j = i
            while j < b and states[j] == states[i]:
                j += 1
            out.append((i, j, int(states[i])))
            i = j
        else:
            i += 1
    return out


def phases(y, a, b):
    """0 lateral, 1/2/3 Bull start/middle/end, 4/5/6 Bear start/middle/end."""
    z = np.zeros(len(y), np.int8)
    for i, j, s in runs(y, a, b):
        k = max(1, min(8, (j - i) // 4))
        base = 1 if s == 1 else 4
        z[i:j] = base + 1
        z[i:i + k] = base
        z[j - k:j] = base + 2
        for t in range(j, min(j + 4, b)):
            if y[t] == 0:
                z[t] = base + 2
    return z


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-penalty", type=float, default=BIG, help="custo extra por nova escada (0.03 = 3%%)")
    parser.add_argument("--hold-per-hour", type=float, default=0.0005, help="custo por hora posicionado (0.0002 = 0.02%%/h)")
    args = parser.parse_args()
    big_penalty, hold = args.entry_penalty, args.hold_per_hour / 4
    tag = "escada_%g_%g" % (100 * big_penalty, 100 * args.hold_per_hour)
    began = time.time()
    full = pd.read_parquet(ROOT / "data" / "history_causal.parquet").sort_index()
    start = full.index[-1] - pd.Timedelta(days=730)
    df = full.loc[full.index >= start - pd.Timedelta(days=45)].ffill().fillna(0.0)
    s0, n = int(np.searchsorted(df.index, start)), len(df)
    close = df["close"].to_numpy(np.float64)
    r = np.concatenate([np.log(close[1:] / close[:-1]), [0.0]])
    simple = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    funding = df["funding_rate"].to_numpy(np.float64) * ((df.index.hour % 8 == 0) & (df.index.minute == 0))
    report = {}

    # ---- 1) Opportunity map over the whole two years ----
    everywhere = np.ones(n - s0, np.bool_)
    big = np.zeros(n, np.int8)
    big[s0:] = oracle(r[s0:], COST, big_penalty, everywhere, hold)
    small = np.zeros(n, np.int8)
    small[s0:] = oracle(r[s0:], COST, SMALL, big[s0:] == 0, 0.0)
    print("1) MAPA DE OPORTUNIDADES %s -> %s (BTC 15m, custo 0.05%%/lado)" % (df.index[s0].date(), df.index[-1].date()))
    report["map"] = {}
    days = (n - s0) / 96
    for label, states, sides in (("escada (%g%% + %g%%/h)" % (100 * big_penalty, 100 * args.hold_per_hour), big, ((1, "BULL"), (2, "BEAR"))), ("oscilacao lateral (0.2%)", small, ((1, "RANGER compra"), (2, "RANGER venda")))):
        for code, name in sides:
            legs = [(i, j) for i, j, s in runs(states, s0, n) if s == code]
            gains = np.array([(1 if code == 1 else -1) * np.log(close[min(j, n - 1)] / close[i]) for i, j in legs])
            hours = np.array([(j - i) / 4 for i, j in legs])
            report["map"][name] = {"legs": len(legs), "per_day": len(legs) / days, "median_hours": float(np.median(hours)),
                                   "median_gain": float(np.median(gains)), "mean_gain": float(gains.mean()),
                                   "time_share": float(np.mean(states[s0:] == code))}
            print("  %-14s %-26s %4d oportunidades (%.1f/dia) | duracao mediana %5.1fh | ganho mediano %+.2f%% medio %+.2f%% | %2.0f%% do tempo" % (
                name, label, len(legs), len(legs) / days, np.median(hours), 100 * np.median(gains), 100 * gains.mean(), 100 * np.mean(states[s0:] == code)))
    print("  sem oportunidade nenhuma: %.0f%% do tempo" % (100 * np.mean((big[s0:] == 0) & (small[s0:] == 0))))

    # ---- 2) Features known at each close; labels per block ----
    X = build_features(df)
    X = X.drop(columns=[c for c in X.columns if c.endswith("7d") or c in ("ret_672", "rv_ratio_96_672")])
    _, _, groups, _, _ = build_day_view(pd.read_parquet(FIVE).sort_index())
    day_view = pd.concat(list(groups.values()), axis=1).reindex(df.index + pd.Timedelta("15min"))
    day_view.index = df.index
    X = pd.concat([X, day_view], axis=1).replace([np.inf, -np.inf], np.nan).astype("float32")
    m = n - s0
    train_end = s0 + int(0.70 * m)
    val_start, val_end = train_end + EMBARGO, train_end + EMBARGO + int(0.15 * m)
    hold_start = val_end + EMBARGO
    blocks = {"treino": (s0, train_end), "validacao": (val_start, val_end), "holdout": (hold_start, n)}
    y = np.zeros(n, np.int8)
    z = np.zeros(n, np.int8)
    for a, b in blocks.values():
        rb = r[a:b].copy()
        rb[-1] = 0.0
        y[a:b] = oracle(rb, COST, big_penalty, np.ones(b - a, np.bool_), hold)
        z[a:b] = phases(y, a, b)[a:b]
    print("\n2) ALUNOS: %d indicadores (dia, 4h, 1h, 15m, 5m)" % X.shape[1])
    for name, (a, b) in blocks.items():
        print("  %-9s %s -> %s | pernas: bull %2.0f%% bear %2.0f%% lateral %2.0f%% do tempo | buy&hold %+.1f%%" % (
            name, df.index[a].date(), df.index[b - 1].date(), 100 * np.mean(y[a:b] == 1), 100 * np.mean(y[a:b] == 2),
            100 * np.mean(y[a:b] == 0), 100 * (close[b - 1] / close[a] - 1)))

    def fit_predict(train_rows, target, predict_slices, classes):
        mdl = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250, min_samples_leaf=200,
                                             l2_regularization=1.0, early_stopping=False, random_state=7)
        mdl.fit(X.iloc[train_rows], target[train_rows])
        outs = []
        for sl in predict_slices:
            p = np.zeros((len(sl), classes))
            p[:, mdl.classes_] = mdl.predict_proba(X.iloc[sl])
            outs.append(p)
        return outs

    def teach(target, classes):
        prob = {k: np.full((n, classes), np.nan) for k in ("oof", "decorado", "fora")}
        a, b = blocks["treino"]
        for fold in np.array_split(np.arange(a, b), 5):
            keep = np.arange(a, b)
            keep = keep[(keep < fold[0] - EMBARGO) | (keep >= fold[-1] + EMBARGO)][::2]
            prob["oof"][fold] = fit_predict(keep, target, [fold], classes)[0]
        (va, vb), (ha, hb) = blocks["validacao"], blocks["holdout"]
        inside, val, hold = fit_predict(np.arange(a, b, 2), target, [np.arange(a, b), np.arange(va, vb), np.arange(ha, hb)], classes)
        prob["decorado"][a:b], prob["fora"][va:vb], prob["fora"][ha:hb] = inside, val, hold
        print("  aluno pronto (%.0fs)" % (time.time() - began), flush=True)
        return prob

    def views():
        yield "treino (decorado)", "decorado", blocks["treino"]
        yield "treino (fora da dobra)", "oof", blocks["treino"]
        yield "validacao", "fora", blocks["validacao"]
        yield "holdout", "fora", blocks["holdout"]

    def evaluate(pos, va, vb):
        held = np.concatenate([[0.0], pos[:-1]])[va:vb]
        strat = held * simple[va:vb] - held * funding[va:vb] - COST * np.abs(np.diff(np.concatenate([[0.0], held])))
        eq = pd.Series(np.cumprod(1 + strat), index=df.index[va:vb])
        daily = eq.resample("1D").last().pct_change().dropna()
        t = np.array([np.prod(1 + strat[i + 1 - va:min(j + 1, vb) - va]) - 1 for i, j, _ in runs(pos, va, vb)])
        gains, losses = t[t > 0].sum(), -t[t < 0].sum()
        return {"ret": float(eq.iloc[-1] - 1), "sharpe": float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0,
                "dd": float((1 - eq / eq.cummax()).max()), "trades": len(t), "win": float((t > 0).mean()) if len(t) else 0.0,
                "pf": float(gains / losses) if losses > 0 else 0.0, "exposure": float(np.mean(held != 0))}

    def diagnose(agent, side_code, sign, positions):
        lines = {}
        for label, pos, (va, vb) in positions:
            legs = [(i, j) for i, j, s in runs(y, va, vb) if s == side_code]
            lags, captured, missed, caught_gain = [], [], [], []
            for i, j in legs:
                gain = sign * np.log(close[min(j, n - 1)] / close[i])
                inside = pos[i:j] == sign
                if inside.any():
                    lags.append(np.argmax(inside) / 4)
                    captured.append(np.sum(inside * sign * r[i:j]) / gain)
                    caught_gain.append(gain)
                else:
                    missed.append(gain)
            good, bad, kind, late, given = 0, 0, {"lateral": 0, "contra": 0}, [], []
            for i, j, s in runs(pos, va, vb):
                if s != sign:
                    continue
                if np.mean(y[i:j] == side_code) >= 0.5:
                    good += 1
                else:
                    bad += 1
                    kind["contra" if np.mean(y[i:j] == 3 - side_code) > np.mean(y[i:j] == 0) else "lateral"] += 1
                tail = j - 1
                while tail > i and y[tail] != side_code:
                    tail -= 1
                if y[tail] == side_code and tail < j - 1:
                    late.append((j - 1 - tail) / 4)
                    given.append(sign * np.log(close[min(j, n - 1)] / close[tail + 1]))
            x = {"legs": len(legs), "caught": len(lags), "entry_lag_h": float(np.median(lags)) if lags else None,
                 "captured": float(np.median(captured)) if captured else None, "missed_gain": float(np.median(missed)) if missed else None,
                 "caught_gain": float(np.median(caught_gain)) if caught_gain else None, "trades_ok": good, "trades_false": bad,
                 "false_kind": kind, "late_exit_h": float(np.median(late)) if late else None, "given_back": float(np.median(given)) if given else None}
            lines[label] = x
            fmt = lambda v, f: "-" if v is None else f % v  # noqa: E731
            print("    %s %-22s pegou %3d/%3d pernas (%2.0f%%) | entra %s apos o %s | captura %s da perna | sai %s apos o %s devolvendo %s | trades certos %3d falsos %3d (lateral %d, contra %d)" % (
                agent, label, len(lags), len(legs), 100 * len(lags) / max(len(legs), 1), fmt(x["entry_lag_h"], "%.1fh"),
                "fundo" if sign > 0 else "topo", fmt(x["captured"] and 100 * x["captured"], "%.0f%%"), fmt(x["late_exit_h"], "%.1fh"),
                "topo" if sign > 0 else "fundo", fmt(x["given_back"] and 100 * x["given_back"], "%+.2f%%"), good, bad, kind["lateral"], kind["contra"]))
        return lines

    students = {
        "v1 perna": (y, 3, lambda p: (p[:, 1], p[:, 1], p[:, 2], p[:, 2]),
                     [(e, l) for e in (0.35, 0.45, 0.55, 0.65, 0.75) for l in (0.15, 0.25, 0.35, 0.45, 0.55) if l < e]),
        "v2 fases": (z, 7, lambda p: (p[:, 1], p[:, 1] + p[:, 2], p[:, 4], p[:, 4] + p[:, 5]),
                     [(e, l) for e in (0.15, 0.2, 0.3, 0.4, 0.5) for l in (0.2, 0.3, 0.4, 0.5, 0.6)]),
    }
    report["students"] = {}
    for sname, (target, classes, signals, grid) in students.items():
        print("\n=== ALUNO %s ===" % sname, flush=True)
        prob = teach(target, classes)
        rec = {}
        for label, key, (va, vb) in views():
            pick = np.arange(va, vb)[::16]
            p = np.nan_to_num(prob[key][pick])
            rec[label] = [roc_auc_score(y[pick] == 1, p[:, 1] + (p[:, 2] + p[:, 3] if classes == 7 else 0)),
                          roc_auc_score(y[pick] == 2, p[:, 2] if classes == 3 else p[:, 4] + p[:, 5] + p[:, 6])]
        print("  reconhece a perna (AUC; 0.5 chute, 1.0 perfeito): " + " | ".join("%s Bull %.2f Bear %.2f" % (k, *v) for k, v in rec.items()))
        entry = {"recognition": rec, "agents": {}}
        for agent, allow_long, allow_short in (("BULL", True, False), ("BEAR", False, True), ("BULL+BEAR", True, True)):
            ta, tb = blocks["treino"]
            s_oof = [np.nan_to_num(v) for v in signals(prob["oof"])]
            scores = {g: evaluate(policy(*s_oof, g[0], g[1], allow_long, allow_short, ta, tb), ta, tb) for g in grid}
            enter, leave = max(grid, key=lambda g: scores[g]["sharpe"])
            print("  %s (entra com prob >= %.2f, sai com prob < %.2f; escolhido no treino fora da dobra):" % (agent, enter, leave))
            res_agent, positions = {"enter": enter, "leave": leave}, []
            for label, key, (va, vb) in views():
                s = [np.nan_to_num(v) for v in signals(prob[key])]
                pos = policy(*s, enter, leave, allow_long, allow_short, va, vb)
                res = evaluate(pos, va, vb)
                res["buy_hold"] = float(close[vb - 1] / close[va] - 1)
                res_agent[label] = res
                if key != "decorado":
                    positions.append((label, pos, (va, vb)))
                print("    %-24s %+8.1f%% | sharpe %5.2f | dd %4.1f%% | %3d trades | acerto %2.0f%% | PF %.2f | exposto %2.0f%% | buy&hold %+.1f%%" % (
                    label, 100 * res["ret"], res["sharpe"], 100 * res["dd"], res["trades"], 100 * res["win"], res["pf"], 100 * res["exposure"], 100 * res["buy_hold"]))
            if agent != "BULL+BEAR":
                res_agent["errors"] = diagnose(agent, 1 if agent == "BULL" else 2, 1 if agent == "BULL" else -1, positions)
            entry["agents"][agent] = res_agent
        report["students"][sname] = entry
        if sname == "v2 fases":
            pd.DataFrame({"perna_2anos": big, "ranger_2anos": small, "perna_bloco": y, "fase_bloco": z}, index=df.index).iloc[s0:].assign(
                **{"p_fase_%d_oof" % k: prob["oof"][s0:, k] for k in range(classes)},
                **{"p_fase_%d_fora" % k: prob["fora"][s0:, k] for k in range(classes)}).to_parquet(ROOT / "data" / ("opportunity_labels_%s.parquet" % tag))

    out = ROOT / "reports" / ("teach_opportunities_%s.json" % tag)
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print("\nrelatorio: %s (%.0fs)" % (out, time.time() - began))
    return 0


if __name__ == "__main__":
    sys.exit(main())
