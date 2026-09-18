"""Full edge research on the last two years of BTC only (the owner's scope).

The market changed: the average daily range fell from 4-6% (2020-2022) to ~3%
(2023-2026), and patterns that paid in 2020-2021 stopped paying. Everything is
redone on the recent, more mature market.

Data: data/history_causal.parquet (live-built, BTCUSDT 15m). Rows from START
(default two years before the last bar) are the research universe; 45 days
before START are loaded only so that indicators have their look-back filled.
Nothing before START is used to fit, choose or evaluate.

Split of the universe: train 70% / embargo 8 days / validation 15% / embargo /
holdout. Rules and parameters are chosen on train only; results are reported
for validation and holdout and by calendar quarter. Costs: 0.05% per side
(taker + slippage), maker variant 0.02% where noted; funding every 8 hours.

Parts
  1. Rule grids for each agent:
     BULL  up-leg: close above the prior N-bar high, optional volatility
           expansion and buyer flow, exit on an M-bar low / trailing / time
     BEAR  down-leg: the mirror, optional larger-trend filter; and the fade of
           an exhausted sharp rally
     RANGER sideways reversion to the 20-bar mean in low-efficiency markets
  2. Event-conditioned meta-labelling (research_event_edges), train-only CV
  3. Indicator stability by quarter, a walk-forward model of the next 4h/24h,
     and early recognition of bottoms and tops (research_turning_points),
     refit every quarter on the universe's past only

    python scripts/research_recent_legs.py [--start=2024-09-17]
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_event_edges import FEATURES as EVENT_FEATURES, barrier, events as event_index, ranger_side  # noqa: E402
from research_turning_points import zigzag_points  # noqa: E402
from research_walkforward_patterns import build_features  # noqa: E402

TAKER, MAKER = 0.0005, 0.0002
WARMUP = pd.Timedelta(days=45)
EMBARGO = 768


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--%s=" % name)), default)


# ── simulators ────────────────────────────────────────────────────────────────
@numba.njit(cache=True)
def sim_leg(side, entry, exit_ref, atr, stop_k, trail_k, max_hold, close, active_from):
    pos = np.zeros(close.shape[0])
    state, extreme, entry_px, entry_atr, age = 0, 0.0, 0.0, 0.0, 0
    for i in range(active_from, close.shape[0]):
        c = close[i]
        if state != 0:
            age += 1
            if side > 0:
                extreme = max(extreme, c)
                out = c < exit_ref[i] or c < entry_px - stop_k * entry_atr or c < extreme - trail_k * entry_atr
            else:
                extreme = min(extreme, c)
                out = c > exit_ref[i] or c > entry_px + stop_k * entry_atr or c > extreme + trail_k * entry_atr
            if out or age >= max_hold:
                state = 0
        if state == 0 and entry[i]:
            state, extreme, entry_px, entry_atr, age = side, c, c, atr[i], 0
        pos[i] = state
    return pos


@numba.njit(cache=True)
def sim_target(side, entry, target_k, stop_k, trail_k, max_hold, atr, close, active_from):
    pos = np.zeros(close.shape[0])
    state, extreme, entry_px, entry_atr, age = 0, 0.0, 0.0, 0.0, 0
    for i in range(active_from, close.shape[0]):
        c = close[i]
        if state != 0:
            age += 1
            if side < 0:
                extreme = min(extreme, c)
                out = (c <= entry_px - target_k * entry_atr or c > entry_px + stop_k * entry_atr
                       or (trail_k > 0 and c > extreme + trail_k * entry_atr))
            else:
                extreme = max(extreme, c)
                out = (c >= entry_px + target_k * entry_atr or c < entry_px - stop_k * entry_atr
                       or (trail_k > 0 and c < extreme - trail_k * entry_atr))
            if out or age >= max_hold:
                state = 0
        if state == 0 and entry[i]:
            state, extreme, entry_px, entry_atr, age = side, c, c, atr[i], 0
        pos[i] = state
    return pos


@numba.njit(cache=True)
def sim_revert(entry_long, entry_short, target, atr, stop_k, max_hold, close, active_from):
    pos = np.zeros(close.shape[0])
    state, entry_px, age = 0, 0.0, 0
    for i in range(active_from, close.shape[0]):
        c = close[i]
        if state != 0:
            age += 1
            out = (state > 0 and (c >= target[i] or c < entry_px - stop_k * atr[i])) or \
                  (state < 0 and (c <= target[i] or c > entry_px + stop_k * atr[i]))
            if out or age >= max_hold:
                state = 0
        if state == 0:
            if entry_long[i]:
                state, entry_px, age = 1, c, 0
            elif entry_short[i]:
                state, entry_px, age = -1, c, 0
        pos[i] = state
    return pos


def main() -> int:
    began = time.time()
    full = pd.read_parquet(ROOT / "data" / "history_causal.parquet").sort_index()
    start = pd.Timestamp(arg("start") or (full.index[-1] - pd.Timedelta(days=730)).strftime("%Y-%m-%d"), tz=full.index.tz)
    df = full.loc[full.index >= start - WARMUP].ffill().fillna(0.0)
    s0 = int(np.searchsorted(df.index, start))
    m = len(df) - s0
    train_end = s0 + int(m * 0.70)
    val_start = train_end + EMBARGO
    val_end = val_start + int(m * 0.15)
    hold_start = val_end + EMBARGO
    blocks = {"treino": (s0, train_end), "validacao": (val_start, val_end), "holdout": (hold_start, len(df))}
    print("universo %s -> %s (%d barras) | treino ate %s | validacao %s -> %s | holdout %s -> %s" % (
        df.index[s0].date(), df.index[-1].date(), m, df.index[train_end - 1].date(), df.index[val_start].date(),
        df.index[val_end - 1].date(), df.index[hold_start].date(), df.index[-1].date()), flush=True)
    close, high, low = (df[c].to_numpy(np.float64) for c in ("close", "high", "low"))
    ret = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
    funding = df["funding_rate"].to_numpy(np.float64) * ((df.index.hour % 8 == 0) & (df.index.minute == 0))
    atr15, atr1h = df["atr_15m"].to_numpy(np.float64), df["atr_1h"].to_numpy(np.float64)
    bh = {k: float(close[b - 1] / close[a] - 1) for k, (a, b) in blocks.items()}
    print("buy&hold: " + " | ".join("%s %+.1f%%" % (k, 100 * v) for k, v in bh.items()))

    def evaluate(pos, cost):
        held = np.concatenate([[0.0], pos[:-1]])
        strat = held * ret - held * funding - cost * np.abs(np.diff(np.concatenate([[0.0], held])))
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

    def show(label, r):
        t, v, h = r["treino"], r["validacao"], r["holdout"]
        return "%-58s | TR %+5.0f%% sh %5.2f dd %2.0f%% n=%4d | VAL %+4.0f%% sh %5.2f dd %2.0f%% n=%3d | HO %+4.0f%% sh %5.2f dd %2.0f%% n=%3d | %s" % (
            label, 100 * t[0], t[1], 100 * t[2], t[3], 100 * v[0], v[1], 100 * v[2], v[3], 100 * h[0], h[1], 100 * h[2], h[3],
            " ".join("%s:%+.0f" % (k[-2:], 100 * x) for k, x in r["trimestres"].items()))

    report = {"universe": [str(df.index[s0]), str(df.index[-1])], "blocks": {k: [str(df.index[a]), str(df.index[b - 1])] for k, (a, b) in blocks.items()},
              "buy_and_hold": bh, "grids": {}}
    aggr, cvd, expansion = (df[c].to_numpy() for c in ("cz_aggression", "cz_cvd_z_16", "cz_atr_expansion"))
    trend4h = df["ema_trend_4h"].to_numpy()

    def run_grid(title, rows, min_trades):
        rows = [x for x in rows if x[1]["treino"][3] >= min_trades]
        rows.sort(key=lambda x: x[1]["treino"][1], reverse=True)
        print("\n%s — top pelo Sharpe de TREINO (>= %d trades):" % (title, min_trades))
        for label, r in rows[:6]:
            print("  " + show(label, r))
        report["grids"][title] = [{"rule": label, **r} for label, r in rows[:20]]

    # ── Part 1: rule grids ───────────────────────────────────────────────────
    for side, title in ((1, "BULL perna de alta"), (-1, "BEAR perna de queda")):
        rows = []
        exit_refs = {M: (df["low"].rolling(M).min() if side > 0 else df["high"].rolling(M).max()).shift(1)
                     .fillna(-np.inf if side > 0 else np.inf).to_numpy() for M in (8, 16, 32)}
        for N, vol, flow, trend in itertools.product((48, 96, 192), (0.0, 1.2), (False, True), (False, True)):
            if side > 0 and trend:
                continue
            ref = (df["high"].rolling(N).max() if side > 0 else df["low"].rolling(N).min()).shift(1).to_numpy()
            entry = (close > ref) if side > 0 else (close < ref)
            if vol:
                entry &= expansion > vol
            if flow:
                entry &= (side * aggr > 0) & (side * cvd > 1.0)
            if trend:
                entry &= trend4h < 0
            for M, trail, hold in itertools.product((8, 16, 32), (1.5, 2.5, 4.0), (96, 192)):
                pos = sim_leg(side, entry, exit_refs[M], atr1h, 3.0, trail, hold, close, s0)
                rows.append(("N=%d exp>%.1f fluxo=%s tend4h=%s sai%d trail=%.1f max=%d" % (N, vol, flow, trend, M, trail, hold),
                             evaluate(pos, TAKER)))
        run_grid(title, rows, 40)
        print("  (%.0fs)" % (time.time() - began), flush=True)

    logret = np.log(df["close"]).diff()
    absorption, rsi15 = df["cz_absorption"].to_numpy(), df["rsi_15m"].to_numpy()
    rows = []
    for k, z, filt in itertools.product((4, 8, 16), (2.5, 3.5), ("nenhum", "exaustao", "rsi")):
        move = logret.rolling(k).sum()
        scale = move.rolling(2880, min_periods=500).std().shift(1)
        shock = np.nan_to_num((move > z * scale).to_numpy()).astype(bool)
        if filt == "exaustao":
            shock &= (absorption > 0) & (cvd > 1.0)
        elif filt == "rsi":
            shock &= rsi15 > 75
        for tf, target, stop, trail, hold in itertools.product(("15m", "1h"), (1.5, 3.0), (1.5, 3.0), (0.0, 2.0), (16, 64)):
            pos = sim_target(-1, shock, target, stop, trail, hold, atr15 if tf == "15m" else atr1h, close, s0)
            rows.append(("choque%dx%.1f %s atr%s alvo=%.1f stop=%.1f trail=%.1f max=%d" % (k, z, filt, tf, target, stop, trail, hold),
                         evaluate(pos, TAKER)))
    run_grid("BEAR venda da exaustao de alta forte", rows, 40)

    eff96, adx15 = df["cz_efficiency_96"].abs().to_numpy(), df["adx_15m"].to_numpy()
    mean20 = df["close"].rolling(20).mean().to_numpy()
    std20 = df["close"].rolling(20).std().to_numpy()
    zscore = np.nan_to_num((close - mean20) / np.where(std20 > 0, std20, np.nan))
    for cost, title in ((TAKER, "RANGER custo a mercado"), (MAKER, "RANGER custo ordem limitada")):
        rows = []
        for eff_max, adx_max, zk, stop, hold in itertools.product((0.15, 0.30), (20, 30), (2.0, 2.5, 3.0), (1.5, 3.0), (8, 16, 32)):
            ranging = (eff96 < eff_max) & (adx15 < adx_max)
            pos = sim_revert(ranging & (zscore < -zk), ranging & (zscore > zk), np.nan_to_num(mean20, nan=close.mean()),
                             atr15, stop, hold, close, s0)
            rows.append(("eff<%.2f adx<%d z>%.1f stop=%.1f max=%d" % (eff_max, adx_max, zk, stop, hold), evaluate(pos, cost)))
        run_grid(title, rows, 100)
    print("  (%.0fs)" % (time.time() - began), flush=True)

    # ── Part 2: event-conditioned meta-labelling ──────────────────────────────
    X_ev = df[EVENT_FEATURES].to_numpy(np.float32)
    report["events"] = {}
    configs = {"bear": dict(target=3.0, stop=2.0, horizon=96), "bull": dict(target=3.0, stop=2.0, horizon=96),
               "ranger": dict(target=1.0, stop=1.5, horizon=16)}
    print("\nMETA-ROTULAGEM POR EVENTOS (CV so no treino):")
    for kind, cfg in configs.items():
        idx = event_index(df, kind)
        idx = idx[idx >= s0]
        if kind == "ranger":
            sides = ranger_side(df, idx)
            net = np.zeros(len(idx))
            for sd in (-1, 1):
                sel = sides == sd
                net[sel] = barrier(sd, idx[sel], close, high, low, atr1h, cfg["target"], cfg["stop"], cfg["horizon"], TAKER)[0]
        else:
            net = barrier(-1 if kind == "bear" else 1, idx, close, high, low, atr1h, cfg["target"], cfg["stop"], cfg["horizon"], TAKER)[0]
        y = (net > 0).astype(int)
        which = np.array([next((n for n, (a, b) in blocks.items() if a <= i < b - cfg["horizon"]), None) for i in idx])
        tr_pos = np.flatnonzero(which == "treino")
        folds = np.array_split(tr_pos, 5)
        oof = np.full(len(idx), np.nan)
        for f in range(1, 5):
            fit = np.concatenate(folds[:f])
            fit = fit[idx[fit] < idx[folds[f][0]] - cfg["horizon"]]
            if len(set(y[fit])) < 2:
                continue
            model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                                   l2_regularization=1.0, random_state=7).fit(X_ev[idx[fit]], y[fit])
            oof[folds[f]] = model.predict_proba(X_ev[idx[folds[f]]])[:, 1]
        cv = np.isfinite(oof)
        thr, best = 0.5, -np.inf
        for q in np.quantile(oof[cv], np.linspace(0, 0.75, 16)) if cv.any() else []:
            take = cv & (oof >= q)
            if take.sum() >= 0.25 * cv.sum() and net[take].mean() > best:
                thr, best = float(q), float(net[take].mean())
        final = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                               l2_regularization=1.0, random_state=7).fit(X_ev[idx[tr_pos]], y[tr_pos])
        prob = final.predict_proba(X_ev[idx])[:, 1]
        line = {}
        text = "  %-6s eventos=%d" % (kind.upper(), len(idx))
        for name in blocks:
            score = oof if name == "treino" else prob
            sel = (which == name) & np.isfinite(score)
            take = sel & (score >= thr)
            auc = roc_auc_score(y[sel], score[sel]) if sel.any() and y[sel].std() > 0 else float("nan")
            line[name] = {"auc": float(auc), "all_mean": float(net[sel].mean()) if sel.any() else 0.0,
                          "taken": int(take.sum()), "taken_mean": float(net[take].mean()) if take.any() else 0.0}
            text += " | %s AUC %.3f todos %+.3f%% filtrados %d %+.3f%%/trade" % (
                name, auc, 100 * line[name]["all_mean"], take.sum(), 100 * line[name]["taken_mean"])
        print(text)
        report["events"][kind] = line
    print("  (%.0fs)" % (time.time() - began), flush=True)

    # ── Part 3: indicators by quarter, walk-forward model, turning points ────
    X = build_features(df)
    fwd = {H: np.log(df["close"].shift(-H) / df["close"]) for H in (16, 96)}
    quarters_all = pd.period_range(df.index[s0], df.index[-1], freq="Q")
    uni = {}
    for col in X.columns:
        ics = []
        for q in quarters_all:
            sel = (X.index >= q.start_time.tz_localize(X.index.tz)) & (X.index <= q.end_time.tz_localize(X.index.tz)) & (np.arange(len(X)) >= s0)
            a, b = X[col][sel].iloc[::16], fwd[96][sel].iloc[::16]
            ok = a.notna() & b.notna()
            ics.append(spearmanr(a[ok], b[ok]).correlation if ok.sum() > 100 else np.nan)
        ics = np.array(ics, dtype=float)
        uni[col] = {"ics": [float(x) for x in ics], "mean": float(np.nanmean(ics)),
                    "stable": bool(np.nanmin(np.sign(ics)) == np.nanmax(np.sign(ics)))}
    stable = sorted([k for k, v in uni.items() if v["stable"]], key=lambda k: -abs(uni[k]["mean"]))
    print("\nINDICADORES COM O MESMO SINAL DE IC EM TODOS OS %d TRIMESTRES (retorno 24h):" % len(quarters_all))
    for k in stable[:15]:
        print("  %-20s IC medio %+.3f | %s" % (k, uni[k]["mean"], " ".join("%+.3f" % x for x in uni[k]["ics"])))
    print("  (%d de %d)" % (len(stable), X.shape[1]))
    report["indicators"] = uni

    tests = pd.date_range(df.index[s0] + pd.Timedelta(days=270), df.index[-1], freq="QS")
    atr_pct = df["atr_15m"] / df["close"]
    report["walk_forward"] = {}
    print("\nMODELO WALK-FORWARD (treino so no universo, retreino trimestral):")
    for H in (16, 96):
        target = (fwd[H] / (atr_pct * np.sqrt(H))).astype("float32")
        preds = pd.Series(np.nan, index=X.index)
        cut_lo = pd.Series(np.nan, index=X.index)
        cut_hi = pd.Series(np.nan, index=X.index)
        for q_start, q_end in zip(tests, list(tests[1:]) + [X.index[-1] + pd.Timedelta(minutes=15)]):
            train = (np.arange(len(X)) >= s0) & (X.index < q_start - pd.Timedelta(minutes=15 * H)) & target.notna().to_numpy()
            tr_idx = np.flatnonzero(train)[::2]
            model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=200,
                                                  l2_regularization=1.0, random_state=11).fit(X.iloc[tr_idx], target.iloc[tr_idx].clip(-5, 5))
            sel = (X.index >= q_start) & (X.index < q_end)
            preds[sel] = model.predict(X[sel])
            insample = model.predict(X.iloc[tr_idx[-20000:]])
            cut_lo[sel], cut_hi[sel] = np.quantile(insample, 0.10), np.quantile(insample, 0.90)
        rows = {}
        for q in preds.dropna().index.to_period("Q").unique():
            sel = (preds.index.to_period("Q") == q) & preds.notna() & fwd[H].notna()
            p, r = preds[sel].iloc[::16], fwd[H][sel].iloc[::16]
            long_r = r[p >= cut_hi[sel].iloc[::16]].mean() - 2 * TAKER
            short_r = -r[p <= cut_lo[sel].iloc[::16]].mean() - 2 * TAKER
            rows[str(q)] = {"ic": float(spearmanr(p, r).correlation), "long_net": float(long_r), "short_net": float(short_r)}
            print("  %s %s IC %+.3f | compra topo 10%%: %+.2f%% liq | venda fundo 10%%: %+.2f%% liq" % (
                "4h " if H == 16 else "24h", q, rows[str(q)]["ic"], 100 * long_r, 100 * short_r))
        report["walk_forward"][str(H)] = rows

    marks = zigzag_points(high, low, close, atr1h, 3.0)
    points = np.flatnonzero(marks != 0)
    bottom = np.zeros(len(close), dtype=np.int8)
    top = np.zeros(len(close), dtype=np.int8)
    for a, b in zip(points[:-1], points[1:]):
        if abs(close[b] - close[a]) / max(atr1h[a], 1e-9) >= 6.0:
            (bottom if marks[a] > 0 else top)[a:min(a + 8, b)] = 1
    probs = {k: np.zeros(len(close)) for k in ("bottom", "top")}
    thr = {k: np.full(len(close), np.inf) for k in ("bottom", "top")}
    for q_start, q_end in zip(tests, list(tests[1:]) + [X.index[-1] + pd.Timedelta(minutes=15)]):
        tr_idx = np.flatnonzero((np.arange(len(X)) >= s0) & (X.index < q_start - pd.Timedelta(days=10)))[::2]
        sel = (X.index >= q_start) & (X.index < q_end)
        for name, y in (("bottom", bottom), ("top", top)):
            model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=300,
                                                   l2_regularization=1.0, class_weight="balanced", random_state=13).fit(X.iloc[tr_idx], y[tr_idx])
            probs[name][sel] = model.predict_proba(X[sel])[:, 1]
            thr[name][sel] = np.quantile(model.predict_proba(X.iloc[tr_idx[-40000:]])[:, 1], 0.95)
    from research_turning_points import trade
    pos_bull = trade(1, probs["bottom"], thr["bottom"], probs["top"], thr["top"], atr1h, close, 384, 3.0)
    pos_bear = trade(-1, probs["top"], thr["top"], probs["bottom"], thr["bottom"], atr1h, close, 384, 3.0)
    tested = X.index >= tests[0]
    print("\nRECONHECIMENTO DE FUNDOS E TOPOS (walk-forward):")
    report["turning_points"] = {}
    for q in X.index[tested].to_period("Q").unique():
        sel = tested & (X.index.to_period("Q") == q)
        text = "  %s AUC fundo %.3f topo %.3f" % (q, roc_auc_score(bottom[sel], probs["bottom"][sel]) if bottom[sel].any() else np.nan,
                                                 roc_auc_score(top[sel], probs["top"][sel]) if top[sel].any() else np.nan)
        line = {}
        for label, pos in (("BULL", pos_bull), ("BEAR", pos_bear)):
            held = np.concatenate([[0.0], pos[:-1]])
            strat = held * ret - held * funding - TAKER * np.abs(np.diff(np.concatenate([[0.0], held])))
            growth = float(np.prod(1 + strat[sel]) - 1)
            trades = int((np.abs(np.diff(np.concatenate([[0.0], held[sel]]))) > 0).sum()) // 2
            line[label] = [growth, trades]
            text += " | %s %+.1f%% (%d)" % (label, 100 * growth, trades)
        text += " | BTC %+.0f%%" % (100 * (close[sel][-1] / close[sel][0] - 1))
        report["turning_points"][str(q)] = line
        print(text)

    out = ROOT / "reports" / "recent_research.json"
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("\nrelatorio: %s (%.0fs)" % (out, time.time() - began))
    return 0


if __name__ == "__main__":
    sys.exit(main())
