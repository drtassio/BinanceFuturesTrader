"""Holdout of every saved checkpoint of a guided run, for information only.

The model that trades is still chosen on validation (scripts/reselect_checkpoint.py);
picking it by these numbers would make the holdout another validation set.

    python scripts/holdout_all_checkpoints.py --agent bull --run cloud/artifacts/leg_confirm_1h/bull_guided_<stamp>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")

from cloud.train_agent import AGENTS, buy_and_hold_return, evaluate, load_dataset, split_chronological  # noqa: E402
from config.settings import AIConfig, TradingConfig  # noqa: E402
from reselect_checkpoint import line  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=("bull", "bear"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "featured_data_causal_leg5m.parquet")
    args = ap.parse_args()
    from specialists.trend_specialist import ClippedSAC

    models = args.run.resolve() / "models"
    _, _, holdout_df = split_chronological(load_dataset(args.data))
    config = AIConfig()
    config.ECONOMIC_REWARD_ONLY = True
    config.MODEL_DIR = str(models)
    agent = AGENTS[args.agent](config=config, trading_config=TradingConfig(),
                               input_dim=len(holdout_df.select_dtypes(include="number").columns))
    agent.load_model()
    names = ["best_validation"] + sorted(p.stem for p in models.glob("dagger_*_validation.zip")) + sorted(
        (p.stem for p in models.glob("finetune_*_validation.zip")), key=lambda s: int(s.split("_")[1]))
    print("HOLDOUT de todos os checkpoints (%s a %s, buy&hold %+.1f%%) - so informacao, a escolha e pela validacao"
          % (holdout_df.index.min().date(), holdout_df.index.max().date(), 100 * buy_and_hold_return(holdout_df)))
    for name in names:
        agent.model = ClippedSAC.load(str(models / (name + ".zip")), device=agent.model.device)
        print("  %-28s holdout: %s" % (name, line(evaluate(agent, holdout_df, args.agent, deterministic=True))), flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.exit(main())
