"""Print the mirror chart image in this terminal (Sixel), without the bot.

Replays both agents over the bot's candle history, redraws
logs/charts/espelho.png and prints it here. Needs a terminal with Sixel
(Windows Terminal 1.22 or later).

    py scripts/show_chart.py            # redesenha e imprime
    py scripts/show_chart.py --cached   # so imprime a ultima imagem salva
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    from trading import mirror_chart, sixel

    if not args.cached:
        import pandas as pd
        from trading import agent_mirror as mirror

        history = pd.read_parquet(mirror.HISTORY_PATH)
        paths, shadows = {}, []
        for name, folder in (("bull", ROOT / "models_ai"), ("bear", ROOT / "models_ai" / "shadow")):
            agent, contract = mirror.load_specialist(name, folder)
            state, path = mirror.replay_with_path(agent, contract, history, name)
            paths[name], shadows = path, shadows + [state]
        mirror_chart.render_png(history, paths, {"close": float(history["close"].iloc[-1]), "shadows": shadows})
    print("\n" + sixel.encode(mirror_chart.CHART_PATH), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
