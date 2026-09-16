"""Train one regime specialist on Colab, Kaggle, or a local GPU.

The script intentionally never imports exchange credentials. It consumes only
historical parquet data and writes a model plus a JSON training report.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from config.settings import AIConfig, TradingConfig
from specialists.bull_specialist import BullSpecialist
from specialists.bear_specialist import BearSpecialist
from specialists.ranger_specialist import RangerSpecialist


AGENTS = {
    "bull": BullSpecialist,
    "bear": BearSpecialist,
    "ranger": RangerSpecialist,
}


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("dataset index must be a DatetimeIndex")
    # A causal pipeline never backfills features. Remaining warm-up NaNs are
    # neutralized only after the chronological split.
    frame = frame.ffill().fillna(0.0)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "featured_data_flow_funding.parquet")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=ROOT / "cloud" / "artifacts")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data)
    cutoff = int(len(df) * 0.8)
    train_df, eval_df = df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()
    config = AIConfig()
    config.ECONOMIC_REWARD_ONLY = True
    trading_config = TradingConfig()
    agent_cls = AGENTS[args.agent]
    input_dim = len(train_df.select_dtypes(include="number").columns)
    # Cloud runs must start from a model whose action space was built from the
    # current environment. Reusing old local checkpoints can silently fail when
    # action bounds changed between revisions. The trained artifact is still
    # written to the requested output/model directory afterward.
    model_path = Path(config.MODEL_DIR) / f"{args.agent}_specialist_sac.zip"
    if model_path.exists() and not args.resume:
        model_path.unlink()
    agent = agent_cls(config=config, trading_config=trading_config, input_dim=input_dim)

    result = agent.train_model(train_df, total_timesteps=args.timesteps)
    if isinstance(result, dict) and result.get("success") is False:
        raise RuntimeError(f"training failed: {result}")
    report = {
        "agent": args.agent,
        "timesteps": args.timesteps,
        "train_rows": len(train_df),
        "holdout_rows": len(eval_df),
        "train_start": str(train_df.index.min()),
        "train_end": str(train_df.index.max()),
        "holdout_start": str(eval_df.index.min()),
        "holdout_end": str(eval_df.index.max()),
        "result": result,
        "economic_reward_only": True,
    }
    report_path = args.output / f"{args.agent}_training_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
