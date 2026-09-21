"""Search the Bear's own confirmation rule, train and validation only.

The Bear used the Bull's rule mirrored to the short side. Falls behave
differently (fast drops, sharp squeezes), so this grid adds short-side options:
leave when the 1h trend turns up, enter only in a 4h downtrend, and only while
the 4h trend is strong (ADX). Every variant runs through the real training
environment (costs, 1h-ATR stops, sizing, funding).

Selection: rank on train (>= 40 trades, return > 0, return per drawdown), then
among the ten best on train take the best on validation. The holdout is not
read here.

    set ENV_STOP_ATR_TIMEFRAME=1h
    py scripts/research_bear_rule.py
"""
from __future__ import annotations

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
_SPLITS = {}


def _split(which):
    if which not in _SPLITS:
        from cloud.train_agent import load_dataset, split_chronological
        train, validation, _ = split_chronological(load_dataset(DATA))
        _SPLITS.update({"train": train, "validation": validation})
    return _SPLITS[which]


def _worker(job):
    rule_dict, which = job
    from learning.edge_policy import LegConfirmRule
    from tune_edge_rule import run_rule
    return rule_dict, which, run_rule(_split(which), "bear", LegConfirmRule(**rule_dict))


def score(r, min_trades):
    if r["trades"] < min_trades or r["net_return"] <= 0:
        return -np.inf
    return r["net_return"] / max(r["max_drawdown"], 0.02)


def line(r):
    return "trades=%3d ret=%+7.2f%% PF=%5.2f DD=%5.1f%% acerto=%4.1f%%" % (
        r["trades"], 100 * r["net_return"], r["profit_factor"], 100 * r["max_drawdown"], r["win_rate"])


def main() -> int:
    grid = [dict(kind="leg_confirm", min_steps=s, min_steps_5m=s5, require_trend_4h=t4,
                 exit_trend_1h=x1, min_adx_4h=adx, sl_mult=sl, leverage=3.0, vote=0.8)
            for s, s5, t4, x1, adx, sl in itertools.product(
                (2, 3), (1, 2), (False, True), (False, True), (0.0, 25.0), (2.0, 3.0))]
    print("stops em ATR de %s | %d variantes do Bear" % (os.environ.get("ENV_STOP_ATR_TIMEFRAME"), len(grid)),
          flush=True)
    with Pool(int(os.environ.get("WORKERS", "6"))) as pool:
        train = [(d, r) for d, _, r in pool.imap_unordered(_worker, [(d, "train") for d in grid])]
        train.sort(key=lambda x: score(x[1], 40), reverse=True)
        print("\nTREINO (10 melhores):")
        for d, r in train[:10]:
            print("  %s | %s" % (line(r), {k: d[k] for k in ("min_steps", "min_steps_5m", "require_trend_4h",
                                                           "exit_trend_1h", "min_adx_4h", "sl_mult")}))
        top = [(d, r) for d, r in train[:10] if np.isfinite(score(r, 40))]
        val = list(pool.imap_unordered(_worker, [(d, "validation") for d, _ in top]))
    by_key = {json.dumps(d, sort_keys=True): r for d, r in top}
    print("\nVALIDACAO dos 10 melhores do treino:")
    best = (-np.inf, None, None, None)
    for d, _, r in val:
        s = score(r, 10)
        print("  %s | %s" % (line(r), {k: d[k] for k in ("min_steps", "min_steps_5m", "require_trend_4h",
                                                       "exit_trend_1h", "min_adx_4h", "sl_mult")}))
        if s > best[0]:
            best = (s, d, by_key[json.dumps(d, sort_keys=True)], r)
    if best[1] is None:
        print("\nNenhuma variante do Bear lucra no treino e na validacao.")
        return 1
    out = {"agent": "bear", "rule": best[1], "train": best[2], "validation": best[3],
           "selected_on": "scripts/research_bear_rule.py: top 10 on train (return per drawdown, >= 40 trades), "
                          "best on validation (>= 10 trades); holdout not read",
           "stop_atr_timeframe": os.environ.get("ENV_STOP_ATR_TIMEFRAME")}
    path = ROOT / "models_ai" / "bear_exclusive_rule.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nESCOLHIDA: %s\n  treino    %s\n  validacao %s\n  -> %s" % (best[1], line(best[2]), line(best[3]), path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
