"""Shadow mode: run a specialist on live bars without sending any order.

The bot keeps data/live_history.parquet up to date (closed 15m bars, exactly as
the training dataset). On each new bar this replays the shadow specialist
through the training environment, the same way the live mirror does, and logs
what it would hold. It never touches the exchange, so a specialist that failed
approval can be judged on bars no model has seen.

    py scripts/shadow_agent.py --agent bear --model-dir models_ai/shadow

Log: logs/shadow/<agent>_decisions.jsonl, one line per closed bar. Trades are
closed when the shadow position changes; the summary is in
logs/shadow/<agent>_trades.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading import agent_mirror as mirror  # noqa: E402

COST = 0.0014  # taker 0.05% + slippage 0.02% per side, as in the research


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=("bull", "bear"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models_ai" / "shadow")
    parser.add_argument("--poll", type=int, default=30, help="segundos entre leituras do historico")
    args = parser.parse_args()

    out = ROOT / "logs" / "shadow"
    out.mkdir(parents=True, exist_ok=True)
    decisions, trades = out / ("%s_decisions.jsonl" % args.agent), out / ("%s_trades.jsonl" % args.agent)
    agent, contract = mirror.load_specialist(args.agent, args.model_dir)
    print("[SOMBRA] %s carregado de %s; nenhuma ordem sera enviada" % (args.agent, args.model_dir), flush=True)

    last_bar, open_trade = None, None
    while True:
        try:
            history = pd.read_parquet(mirror.HISTORY_PATH)
        except Exception as exc:  # the bot may be writing it
            print("[SOMBRA] historico indisponivel: %s" % exc, flush=True)
            time.sleep(args.poll)
            continue
        newest = history.index[-1]
        if newest != last_bar and len(history) >= mirror.REPLAY_BARS:
            state = mirror.replay(agent, contract, history, args.agent)
            close = float(history["close"].iloc[-1])
            record = {"bar": str(state.bar), "side": state.side, "entered": state.entered_on_last_bar,
                      "entry_price": state.entry_price, "stop_price": state.stop_price,
                      "notional_fraction": state.notional_fraction, "close": close,
                      "logged_at": datetime.now(timezone.utc).isoformat()}
            with decisions.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if open_trade and (state.side != open_trade["side"] or state.entered_on_last_bar):
                gross = open_trade["side"] * (close / open_trade["entry_price"] - 1.0)
                done = {**open_trade, "exit_bar": str(state.bar), "exit_price": close,
                        "return_pct": 100 * (gross - 2 * COST)}
                with trades.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(done) + "\n")
                print("[SOMBRA] %s saiu em %s: %+.2f%%" % (args.agent, state.bar, done["return_pct"]), flush=True)
                open_trade = None
            if state.side and open_trade is None:
                open_trade = {"side": state.side, "entry_bar": str(state.bar),
                              "entry_price": state.entry_price or close, "stop_price": state.stop_price}
                print("[SOMBRA] %s entrou %s em %s a %.1f (stop %s)" % (
                    args.agent, "short" if state.side < 0 else "long", state.bar,
                    open_trade["entry_price"], state.stop_price), flush=True)
            print("[SOMBRA] barra %s | %s=%+d%s | preco %.1f" % (
                state.bar, args.agent, state.side, "*" if state.entered_on_last_bar else "", close), flush=True)
            last_bar = newest
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
