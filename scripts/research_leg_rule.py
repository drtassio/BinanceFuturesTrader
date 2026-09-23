"""Choose a specialist's leg-confirm teacher again, in the corrected environment.

Same protocol as scripts/research_bear_rule.py, for either side:

1. Train: 64 structural variants (steps on 15m and 5m, 4h trend, exit on the 1h
   trend, 4h ADX, stop in ATR), ranked by return per drawdown with >= 40 trades
   and return > 0.
2. Validation: the best of the ten best on train (>= 10 trades).
3. Bear only, validation: the learned entry filter (models_ai/bear_filter.joblib,
   fitted on train) off or at a few thresholds. It is judged on validation
   because on train it is in-sample.

The holdout is not read. Every variant runs through the real training
environment: fees, funding, slippage, stops in ATR of ENV_STOP_ATR_TIMEFRAME,
sizing, with the per-trade results net of costs (bug C1).

    set ENV_STOP_ATR_TIMEFRAME=1h
    py scripts/research_leg_rule.py --agent bull
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")
DATA = ROOT / "data" / "featured_data_causal_leg5m.parquet"
FILTER = "models_ai/bear_filter.joblib"
KEYS = ("min_steps", "min_steps_5m", "require_trend_4h", "exit_trend_1h", "min_adx_4h", "sl_mult",
        "filter_threshold")
_SPLITS = {}


def _split(which):
    if which not in _SPLITS:
        from cloud.train_agent import load_dataset, split_chronological
        train, validation, _ = split_chronological(load_dataset(DATA))
        _SPLITS.update({"train": train, "validation": validation})
    return _SPLITS[which]


def _worker(job):
    agent, rule_dict, which = job
    import logging
    logging.disable(logging.WARNING)
    from learning.edge_policy import LegConfirmRule
    from tune_edge_rule import run_rule
    return rule_dict, which, run_rule(_split(which), agent, LegConfirmRule(**rule_dict))


def score(r, min_trades):
    if r["trades"] < min_trades or r["net_return"] <= 0:
        return -np.inf
    return r["net_return"] / max(r["max_drawdown"], 0.02)


def line(r):
    return "trades=%3d ret=%+7.2f%% PF=%5.2f DD=%5.1f%% acerto=%4.1f%%" % (
        r["trades"], 100 * r["net_return"], r["profit_factor"], 100 * r["max_drawdown"], r["win_rate"])


def short(d):
    return {k: d.get(k) for k in KEYS if k in d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=("bull", "bear"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "10")))
    ap.add_argument("--output", type=Path, help="padrao: models_ai/<agente>_leg_rule_b1b3.json")
    args = ap.parse_args()
    agent = args.agent
    stop_tf = os.environ.get("ENV_STOP_ATR_TIMEFRAME")
    if not stop_tf:
        raise SystemExit("defina ENV_STOP_ATR_TIMEFRAME (os especialistas aprovados usam 1h)")

    grid = [dict(kind="leg_confirm", min_steps=s, min_steps_5m=s5, require_trend_4h=t4,
                 exit_trend_1h=x1, min_adx_4h=adx, sl_mult=sl, leverage=3.0, vote=0.8)
            for s, s5, t4, x1, adx, sl in itertools.product(
                (2, 3), (1, 2), (False, True), (False, True), (0.0, 25.0), (2.0, 3.0))]
    print("%s | stops em ATR de %s | %d variantes" % (agent, stop_tf, len(grid)), flush=True)
    with Pool(args.workers) as pool:
        train = [(d, r) for d, _, r in pool.imap_unordered(_worker, [(agent, d, "train") for d in grid])]
        train.sort(key=lambda x: score(x[1], 40), reverse=True)
        print("\nTREINO (10 melhores):")
        for d, r in train[:10]:
            print("  %s | %s" % (line(r), short(d)))
        top = [(d, r) for d, r in train[:10] if np.isfinite(score(r, 40))]
        if not top:
            print("\nNenhuma variante lucra no treino com >= 40 trades.")
            return 1
        val = list(pool.imap_unordered(_worker, [(agent, d, "validation") for d, _ in top]))
        by_key = {json.dumps(d, sort_keys=True): r for d, r in top}
        print("\nVALIDACAO dos 10 melhores do treino:")
        best = (-np.inf, None, None, None)
        for d, _, r in val:
            s = score(r, 10)
            print("  %s | %s" % (line(r), short(d)))
            if s > best[0]:
                best = (s, d, by_key[json.dumps(d, sort_keys=True)], r)
        if best[1] is None:
            print("\nNenhuma variante lucra no treino e na validacao.")
            return 1

        filter_note = "sem filtro"
        if agent == "bear" and (ROOT / FILTER).exists():
            variants = [dict(best[1], filter_path=FILTER, filter_threshold=t) for t in (0.40, 0.43, 0.462, 0.50, 0.53)]
            print("\nFILTRO DO BEAR na validacao (estrutura escolhida; sem filtro: %s):" % line(best[3]))
            for d, _, r in pool.imap_unordered(_worker, [(agent, d, "validation") for d in variants]):
                s = score(r, 10)
                print("  limiar %.3f: %s" % (d["filter_threshold"], line(r)))
                if s > best[0]:
                    best = (s, d, None, r)
                    filter_note = "filtro com limiar %.3f" % d["filter_threshold"]
            if best[2] is None:
                best = (best[0], best[1], _worker((agent, best[1], "train"))[2], best[3])

    out = {"agent": agent, "rule": best[1], "train": best[2], "validation": best[3],
           "selected_on": "scripts/research_leg_rule.py: top 10 on train (return per drawdown, >= 40 trades), "
                          "best on validation (>= 10 trades); bear filter off/threshold on validation (%s); "
                          "holdout not read; per-trade results net of costs" % filter_note,
           "stop_atr_timeframe": stop_tf}
    path = args.output or ROOT / "models_ai" / ("%s_leg_rule_b1b3.json" % agent)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nESCOLHIDA: %s\n  treino    %s\n  validacao %s\n  -> %s" % (
        short(best[1]), line(best[2]), line(best[3]), path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
