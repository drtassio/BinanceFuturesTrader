"""Train one regime specialist on Colab, Kaggle, or a local GPU.

The script intentionally never imports exchange credentials. It consumes only
historical parquet data and writes a model plus a JSON training report.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
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


def evaluate_holdout(agent, frame, agent_name):
    """One complete chronological episode, with the training scaler frozen."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
    from specialists.bull_specialist import BullTradingEnv
    from specialists.bear_specialist import BearTradingEnv
    from specialists.trend_specialist import TrendFollowingEnv

    env_class = {'bull': BullTradingEnv, 'bear': BearTradingEnv}.get(agent_name, TrendFollowingEnv)
    raw = agent._make_trend_env(frame, mode='training', env_class=env_class,
                               feature_columns=agent.feature_columns)
    # Keep identical trading rules, but evaluate the entire unseen sequence.
    raw.max_steps = len(frame) - 1
    env = VecNormalize(VecFrameStack(DummyVecEnv([lambda: raw]), n_stack=4),
                       norm_obs=False, norm_reward=False, training=False)
    try:
        observation = env.reset()
        for _ in range(len(frame)):
            action, _ = agent.model.predict(observation, deterministic=True)
            observation, _, done, _ = env.step(action)
            if bool(done[0]):
                summaries = env.env_method('consume_episode_summaries')[0]
                if not summaries:
                    raise RuntimeError('Holdout ended without financial metrics')
                return summaries[-1]
        raise RuntimeError('Holdout did not terminate')
    finally:
        env.close()


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
    parser.add_argument("--run-directory", type=Path, help="Existing run to resume")
    args = parser.parse_args()
    if args.resume and not args.run_directory:
        parser.error('--resume requires --run-directory for an explicit existing run')

    args.output.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data)
    cutoff = int(len(df) * 0.8)
    train_df, eval_df = df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()
    config = AIConfig()
    # Keep net equity as the primary reward, but do not short-circuit the
    # opportunity-cost and trend-alignment terms: otherwise a flat policy gets
    # exactly zero forever and deterministic SAC collapses to 100% HOLD.
    config.ECONOMIC_REWARD_ONLY = False
    os.environ.setdefault("TREND_SKIP_OPTUNA", "1")
    # Isolate both models AND validation checkpoints; never silently resume a
    # checkpoint distributed in the repository during a fresh cloud run.
    run_name = args.agent if args.resume else f"{args.agent}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
    run_dir = args.run_directory.resolve() if args.resume else args.output.resolve() / run_name
    config.MODEL_DIR = str(run_dir / 'models')
    config.CHECKPOINT_DIR = str(run_dir / 'checkpoints')
    Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    trading_config = TradingConfig()
    agent_cls = AGENTS[args.agent]
    input_dim = len(train_df.select_dtypes(include="number").columns)
    # Cloud runs must start from a model whose action space was built from the
    # current environment. Reusing old local checkpoints can silently fail when
    # action bounds changed between revisions. The trained artifact is still
    # written to the requested output/model directory afterward.
    agent = agent_cls(config=config, trading_config=trading_config, input_dim=input_dim)

    result = agent.train_model(train_df, total_timesteps=args.timesteps)
    if isinstance(result, dict) and result.get("success") is False:
        raise RuntimeError(f"training failed: {result}")
    holdout_metrics = evaluate_holdout(agent, eval_df, args.agent)
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
        "economic_reward_only": False,
        "run_directory": str(run_dir),
        "holdout_evaluated": True,
        "holdout_metrics": holdout_metrics,
        "approved_for_live_trading": False,
    }
    report_path = args.output / f"{args.agent}_training_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
