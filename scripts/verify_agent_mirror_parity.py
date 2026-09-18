"""Does the live replay hold the position the backtest held?

The bot replays the promoted specialist over its last 3000 closed bars, with the
trend structure recomputed on that window (trading/agent_mirror.py). This runs
the specialist once over a long continuous stretch of the training dataset —
the backtest — and then, at random bar ends, replays only the window the bot
would have, comparing position, whether it was entered on that bar and the
environment's stop. Writes models_ai/agent_mirror_parity.json; the approval
requires it.

    python scripts/verify_agent_mirror_parity.py --agents bull
"""
from __future__ import annotations

import argparse
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

from trading import agent_mirror as mirror  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="bull")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models_ai")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "history_causal.parquet")
    parser.add_argument("--span-bars", type=int, default=12000, help="trecho continuo do backtest")
    parser.add_argument("--checks", type=int, default=30)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    history = pd.read_parquet(args.data).sort_index()
    span = history.iloc[-args.span_bars:]
    rng = np.random.default_rng(args.seed)
    ends = np.sort(rng.choice(np.arange(mirror.REPLAY_BARS + 500, len(span) - 1), size=args.checks, replace=False))
    results, ok = {}, True
    for agent_name in [a.strip() for a in args.agents.split(",") if a.strip()]:
        agent, contract = mirror.load_specialist(agent_name, args.model_dir)
        truth_frame = mirror.prepare_frame(span, contract)
        padding = [truth_frame.iloc[[-1]].set_axis([truth_frame.index[-1] + mirror.BAR * k]) for k in (1, 2, 3)]
        truth = mirror.agent_trajectory(agent, pd.concat([truth_frame, *padding]), agent_name)
        mismatches, in_trade = [], 0
        for end in ends:
            window = span.iloc[:end + 1]
            state = mirror.replay(agent, contract, window, agent_name)
            expected = truth.loc[state.bar]
            in_trade += int(expected["side"] != 0)
            same = state.side == int(expected["side"]) and state.entered_on_last_bar == bool(expected["entered"])
            if same and state.side and not np.isnan(expected["stop_price"]):
                same = abs((state.stop_price or 0.0) - expected["stop_price"]) <= 1e-6 * expected["stop_price"]
            if not same:
                mismatches.append({"bar": str(state.bar), "replay_side": state.side, "backtest_side": int(expected["side"]),
                                   "replay_stop": state.stop_price, "backtest_stop": float(expected["stop_price"])})
        print("%s: %d/%d finais de janela iguais ao backtest (%d com posicao aberta)" % (
            agent_name, len(ends) - len(mismatches), len(ends), in_trade))
        for m in mismatches[:5]:
            print("   diverge: %s" % m)
        results[agent_name] = {"checks": len(ends), "mismatches": mismatches, "in_trade": in_trade}
        ok = ok and not mismatches
    report = {"all_passed": ok, "replay_bars": mirror.REPLAY_BARS, "data": str(args.data), "results": results,
              "artifact_hashes": mirror.artifact_hashes(args.model_dir, list(results))}
    (args.model_dir / "agent_mirror_parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PARIDADE DO ESPELHO %s" % ("OK" if ok else "FALHOU"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
