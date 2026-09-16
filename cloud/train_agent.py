"""Train one regime specialist on Colab, Kaggle, or a local GPU.

The script never imports exchange credentials. It consumes historical parquet
data and writes a model, a scaler, the observation contract and a JSON report.

Data is split three ways, chronologically, with an embargo between the blocks:

    train (70%)   the agent learns here
    validation    early stopping and model selection
    holdout       touched exactly once, at the end

Using the holdout for early stopping would turn it into a second validation set
and every number produced from it would be optimistic. It is read once.

A model is only marked fit for live trading when the *deterministic* policy
trades on the holdout and makes money net of costs. That check exists because
of a concrete failure: a previous run reported a healthy profit factor while
its deterministic policy never opened a single position — the exploration noise
was doing all the trading, and none of it would survive contact with a live
bot, which always acts deterministically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import AIConfig, TradingConfig  # noqa: E402
from specialists.bull_specialist import BullSpecialist  # noqa: E402
from specialists.bear_specialist import BearSpecialist  # noqa: E402
from specialists.ranger_specialist import RangerSpecialist  # noqa: E402

AGENTS = {
    "bull": BullSpecialist,
    "bear": BearSpecialist,
    "ranger": RangerSpecialist,
}

# Bars held out between blocks so no label or indicator window straddles a
# boundary. 768 bars is eight days on a 15m grid, wider than any window used.
EMBARGO_BARS = 768

# A deterministic policy that opens fewer than this many positions over the
# holdout has not learned to trade, whatever its reported reward says.
MIN_HOLDOUT_TRADES = 10


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("dataset must use a DatetimeIndex")
    # A causal pipeline never backfills. Warm-up NaNs are neutralised only
    # forward in time.
    return frame.ffill().fillna(0.0)


def split_chronological(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_start = train_end + EMBARGO_BARS
    val_end = val_start + int(n * val_frac)
    holdout_start = val_end + EMBARGO_BARS
    if holdout_start >= n - 1000:
        raise ValueError("dataset too short for a train/validation/holdout split")
    return (
        df.iloc[:train_end].copy(),
        df.iloc[val_start:val_end].copy(),
        df.iloc[holdout_start:].copy(),
    )


def _episode_environment(agent, frame: pd.DataFrame, agent_name: str):
    """One full chronological episode with the training rules unchanged."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
    from specialists.bull_specialist import BullTradingEnv
    from specialists.bear_specialist import BearTradingEnv
    from specialists.trend_specialist import TrendFollowingEnv

    env_class = {"bull": BullTradingEnv, "bear": BearTradingEnv}.get(agent_name, TrendFollowingEnv)
    raw = agent._make_trend_env(
        frame, mode="training", env_class=env_class, feature_columns=agent.feature_columns
    )
    raw.max_steps = len(frame) - 1
    return VecNormalize(
        VecFrameStack(DummyVecEnv([lambda: raw]), n_stack=4),
        norm_obs=False,
        norm_reward=False,
        training=False,
    )


def evaluate(agent, frame: pd.DataFrame, agent_name: str, deterministic: bool = True) -> dict:
    """Run the policy over a whole block and report what it actually did."""
    env = _episode_environment(agent, frame, agent_name)
    actions = []
    try:
        observation = env.reset()
        summary = None
        for _ in range(len(frame)):
            action, _ = agent.model.predict(observation, deterministic=deterministic)
            actions.append(float(np.asarray(action).reshape(-1)[0]))
            observation, _, done, _ = env.step(action)
            if bool(done[0]):
                summaries = env.env_method("consume_episode_summaries")[0]
                summary = summaries[-1] if summaries else None
                break
        if summary is None:
            summaries = env.env_method("consume_episode_summaries")[0]
            summary = summaries[-1] if summaries else {}
    finally:
        env.close()

    votes = np.asarray(actions, dtype=float)
    summary = dict(summary or {})
    summary["deterministic"] = deterministic
    # The distribution of the position vote is the direct evidence of whether
    # the policy collapsed: a constant vote means one decision, forever.
    summary["vote_mean"] = float(votes.mean()) if votes.size else 0.0
    summary["vote_std"] = float(votes.std()) if votes.size else 0.0
    summary["vote_abs_mean"] = float(np.abs(votes).mean()) if votes.size else 0.0
    return summary


def buy_and_hold_return(frame: pd.DataFrame) -> float:
    close = frame["close"].astype(float)
    return float(close.iloc[-1] / close.iloc[0] - 1.0)


def judge(holdout: dict, benchmark: float) -> dict:
    """Decide whether this model may trade real money, and say why."""
    # Os nomes vêm de TrendFollowingEnv._build_financial_snapshot. Ler uma chave
    # inexistente devolveria o default silenciosamente e o veredito aprovaria um
    # modelo com base em zeros.
    trades = int(holdout.get("num_trades", 0) or 0)
    net = float(holdout.get("total_return_pct", 0.0) or 0.0)
    drawdown = float(holdout.get("max_drawdown_pct", 0.0) or 0.0)
    checks = {
        "deterministic_policy_trades": trades >= MIN_HOLDOUT_TRADES,
        "net_return_positive": net > 0.0,
        "beats_buy_and_hold": net > benchmark,
        "drawdown_under_35pct": drawdown < 0.35,
        "policy_not_constant": float(holdout.get("vote_std", 0.0)) > 0.01,
    }
    return {
        "checks": checks,
        "approved_for_live_trading": all(checks.values()),
        "holdout_trades": trades,
        "holdout_net_return": net,
        "holdout_max_drawdown": drawdown,
        "buy_and_hold_return": benchmark,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "featured_data_causal.parquet")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--output", type=Path, default=ROOT / "cloud" / "artifacts")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-directory", type=Path, help="Existing run to resume")
    args = parser.parse_args()
    if args.resume and not args.run_directory:
        parser.error("--resume requires --run-directory")

    args.output.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data)
    train_df, val_df, holdout_df = split_chronological(df)

    config = AIConfig()
    # Net equity, plus the opportunity cost of standing aside, and nothing else.
    #
    # The hand-tuned shaping layer was measured against scripted policies on the
    # strongest uptrend and the worst drawdown in the data. It preferred doing
    # nothing in all six agent/regime combinations, including the two where the
    # agent finished +80% and +76% in equity: the Bull scored -913.8 for making
    # 80% against -195.2 for sitting still. Dozens of bonuses and penalties,
    # each individually reasonable, summed to an objective that was the opposite
    # of profit.
    #
    # The economic reward agrees with equity in all six cases. It is also
    # auditable: one term, the change in net worth, with fees, funding and
    # slippage already inside it.
    config.ECONOMIC_REWARD_ONLY = True
    os.environ.setdefault("TREND_SKIP_OPTUNA", "1")

    run_name = args.agent if args.resume else "%s_%s" % (
        args.agent, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    run_dir = args.run_directory.resolve() if args.resume else args.output.resolve() / run_name
    config.MODEL_DIR = str(run_dir / "models")
    config.CHECKPOINT_DIR = str(run_dir / "checkpoints")
    Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    input_dim = len(train_df.select_dtypes(include="number").columns)
    # Always build the agent from the current environment. Reusing a checkpoint
    # shipped in the repository fails silently when the action bounds or the
    # observation contract changed between revisions.
    agent = AGENTS[args.agent](
        config=config, trading_config=TradingConfig(), input_dim=input_dim
    )

    result = agent.train_model(train_df, total_timesteps=args.timesteps)
    if isinstance(result, dict) and result.get("success") is False:
        raise RuntimeError("training failed: %s" % result)

    validation = evaluate(agent, val_df, args.agent, deterministic=True)
    holdout = evaluate(agent, holdout_df, args.agent, deterministic=True)
    # Same block under exploration noise. If the stochastic policy trades and
    # the deterministic one does not, the agent learned a vote that only crosses
    # the entry threshold by accident, which is the collapse this bot hit before.
    holdout_stochastic = evaluate(agent, holdout_df, args.agent, deterministic=False)

    verdict = judge(holdout, buy_and_hold_return(holdout_df))
    report = {
        "agent": args.agent,
        "timesteps": args.timesteps,
        "dataset": str(args.data),
        "rows": {"train": len(train_df), "validation": len(val_df), "holdout": len(holdout_df)},
        "periods": {
            "train": [str(train_df.index.min()), str(train_df.index.max())],
            "validation": [str(val_df.index.min()), str(val_df.index.max())],
            "holdout": [str(holdout_df.index.min()), str(holdout_df.index.max())],
        },
        "embargo_bars": EMBARGO_BARS,
        "train_result": result,
        "validation_metrics": validation,
        "holdout_metrics": holdout,
        "holdout_metrics_stochastic": holdout_stochastic,
        "verdict": verdict,
        "run_directory": str(run_dir),
        "feature_count": len(getattr(agent, "feature_columns", []) or []),
    }
    (run_dir / "feature_contract.json").write_text(
        json.dumps({"feature_columns": list(getattr(agent, "feature_columns", []) or [])}, indent=2),
        encoding="utf-8",
    )
    report_path = args.output / ("%s_training_report.json" % args.agent)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    if not verdict["approved_for_live_trading"]:
        failed = [k for k, v in verdict["checks"].items() if not v]
        print("\nNAO APROVADO PARA OPERAR. Criterios reprovados: %s" % ", ".join(failed))


if __name__ == "__main__":
    main()
