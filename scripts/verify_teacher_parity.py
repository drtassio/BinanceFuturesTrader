"""Does the live replay hold the same position the backtest held?

The live bot replays only the last few hundred closed bars through the
environment (trading/teacher_policy.py). This script runs the full backtest on
a block of the dataset, then, at many bar ends inside it, replays just the
window the live bot would have had and compares the position, whether it was
entered on that bar and the stop level.

With --live-end it also rebuilds the frame the running bot builds from Binance
(read-only GET requests) ending at a past moment, and compares the teacher's
inputs and replayed position with the dataset. Sends no orders.

    python scripts/verify_teacher_parity.py
    python scripts/verify_teacher_parity.py --live-end "2026-03-10 00:00"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")
os.chdir(ROOT)

from trading.teacher_policy import AGENTS, BAR, artifact_hashes, load_rules, replay, trajectory  # noqa: E402

TEACHER_INPUTS = ("ml_p_long", "ml_p_short", "ml_edge", "tp_prior_dir")


def load_block(which: str) -> pd.DataFrame:
    from cloud.train_agent import load_dataset, split_chronological
    train, validation, holdout = split_chronological(load_dataset(ROOT / "data" / "featured_data_causal.parquet"))
    return {"train": train, "validation": validation, "holdout": holdout}[which]


def offline(block: pd.DataFrame, window: int, checks: int, seed: int) -> bool:
    rules = load_rules(ROOT / "models_ai")
    rng = np.random.default_rng(seed)
    ends = np.sort(rng.choice(np.arange(window, len(block) - 1), size=checks, replace=False))
    ok = True
    for agent in AGENTS:
        full = trajectory(block, agent, rules[agent])
        mismatches = 0
        for end in ends:
            frame = block.iloc[end - window + 1:end + 1]
            state = replay(frame, agent, rules[agent])
            truth = full.loc[frame.index[-1]]
            same = (state.side == int(truth["side"]) and state.entered_on_last_bar == bool(truth["entered"]))
            if same and state.side and not np.isnan(truth["stop_price"]):
                same = abs((state.stop_price or 0.0) - truth["stop_price"]) <= 1e-6 * truth["stop_price"]
            if not same:
                mismatches += 1
                print("  %s %s: replay side=%d entrou=%s stop=%s | backtest side=%d entrou=%s stop=%.2f" % (
                    agent, frame.index[-1], state.side, state.entered_on_last_bar, state.stop_price,
                    int(truth["side"]), bool(truth["entered"]), truth["stop_price"]))
        in_trade = int((full.loc[block.index[ends]]["side"] != 0).sum())
        print("%s: %d/%d finais de janela iguais ao backtest (%d deles com posicao aberta)" % (
            agent, checks - mismatches, checks, in_trade))
        ok = ok and mismatches == 0
    return ok


def live(end: pd.Timestamp, block: pd.DataFrame) -> bool:
    from scripts.verify_live_parity import build_live_frame
    from trading.teacher_policy import closed_bars

    frame = asyncio.run(build_live_frame(end))
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    frame = closed_bars(frame, end)
    common = frame.index.intersection(block.index)
    print("janela ao vivo: %d barras fechadas, %d em comum com o dataset" % (len(frame), len(common)))
    ok = True
    for column in TEACHER_INPUTS:
        diff = (frame.loc[common, column].astype(float) - block.loc[common, column].astype(float)).abs()
        worst = float(diff.max())
        print("  %-12s diferenca maxima %.6f (ultimas 100 barras: %.6f)" % (column, worst, float(diff.iloc[-100:].max())))
        ok = ok and float(diff.iloc[-100:].max()) < 1e-3
    rules = load_rules(ROOT / "models_ai")
    for agent in AGENTS:
        state = replay(frame, agent, rules[agent])
        full = trajectory(block.loc[:frame.index[-1] + BAR], agent, rules[agent])
        truth = full.loc[frame.index[-1]]
        same = state.side == int(truth["side"]) and state.entered_on_last_bar == bool(truth["entered"])
        print("  %s em %s: ao vivo side=%d | backtest side=%d -> %s" % (
            agent, frame.index[-1], state.side, int(truth["side"]), "OK" if same else "DIFERENTE"))
        ok = ok and same
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", default="holdout", choices=("train", "validation", "holdout"))
    parser.add_argument("--window", type=int, default=700, help="barras fechadas que o bot ao vivo replaya")
    parser.add_argument("--checks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--live-end", default=None)
    args = parser.parse_args()

    block = load_block(args.block)
    if block.index.tz is None:
        block.index = block.index.tz_localize("UTC")
    ok = offline(block, args.window, args.checks, args.seed)
    if args.live_end:
        ok = live(pd.Timestamp(args.live_end, tz="UTC"), block) and ok
    report = {"all_passed": bool(ok), "block": args.block, "window": args.window,
              "checks": args.checks, "seed": args.seed, "live_end": args.live_end,
              "artifact_hashes": artifact_hashes(ROOT / "models_ai")}
    (ROOT / "models_ai" / "teacher_parity_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("PARIDADE %s" % ("OK" if ok else "FALHOU"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
