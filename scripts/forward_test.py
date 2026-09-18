"""Forward test: a trained specialist on bars no training step has seen.

Runs the saved deterministic policy through the same evaluation path as the
training holdout (cloud.train_agent.evaluate: one chronological episode, the
phase-3 runtime rules, frame stack of 4) over data/forward_live.parquet, which
scripts/build_forward_dataset.py builds from the live pipeline. The teacher
rule runs over the same bars for reference, and results are broken down by
month next to BTC so a long-only or short-only specialist is judged against
the regime it actually faced.

The candidate must be fixed before running this: picking whichever of several
checkpoints looks best here would turn the forward block into a validation set.

    python scripts/forward_test.py --agent bull --run-dir cloud/artifacts/position_aware/bull_guided_<stamp>
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
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")
os.chdir(ROOT)

from cloud.train_agent import AGENTS, _episode_environment, buy_and_hold_return, judge  # noqa: E402
from config.settings import AIConfig, TradingConfig  # noqa: E402


def load_agent(name: str, run_dir: Path, model_file: Path = None):
    from specialists.trend_specialist import ClippedSAC

    config = AIConfig()
    config.ECONOMIC_REWARD_ONLY = True
    config.MODEL_DIR = str(run_dir / "models")
    config.CHECKPOINT_DIR = str(run_dir / "checkpoints")
    contract = json.loads((run_dir / "feature_contract.json").read_text(encoding="utf-8"))["feature_columns"]
    agent = AGENTS[name](config=config, trading_config=TradingConfig(), input_dim=len(contract))
    agent.load_model()
    if model_file is not None:
        agent.model = ClippedSAC.load(str(model_file), device="cpu")
    if agent.model is None:
        raise RuntimeError("modelo nao carregou de %s" % run_dir)
    if list(agent.feature_columns) != list(contract):
        raise RuntimeError("scaler carregado nao corresponde ao feature_contract.json da execucao")
    return agent


def run_policy(agent, frame: pd.DataFrame, name: str):
    """cloud.train_agent.evaluate, keeping the equity curve for the breakdown."""
    env = _episode_environment(agent, frame, name)
    votes, equity = [], [float(env.get_attr("initial_balance")[0])]
    summary = None
    try:
        observation = env.reset()
        for _ in range(len(frame)):
            action, _ = agent.model.predict(observation, deterministic=True)
            votes.append(float(np.asarray(action).reshape(-1)[0]))
            observation, _, done, _ = env.step(action)
            equity.append(float(env.get_attr("net_worth")[0]))
            if bool(done[0]):
                summaries = env.env_method("consume_episode_summaries")[0]
                summary = summaries[-1] if summaries else None
                if summary:
                    equity[-1] = float(summary.get("final_net_worth", equity[-1]))
                break
        if summary is None:
            summaries = env.env_method("consume_episode_summaries")[0]
            summary = summaries[-1] if summaries else {}
    finally:
        env.close()
    index = pd.DatetimeIndex([frame.index[0] - pd.Timedelta(minutes=15), *frame.index[:len(equity) - 1]])
    curve = pd.Series(equity, index=index)
    daily = curve.resample("1D").last().pct_change().dropna()
    summary = dict(summary or {})
    summary.update(deterministic=True, vote_mean=float(np.mean(votes)), vote_std=float(np.std(votes)),
                   vote_abs_mean=float(np.mean(np.abs(votes))),
                   sharpe_ratio=float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0)
    return summary, curve


def line(m: dict) -> str:
    return "trades=%d retorno=%+.2f%% PF=%.2f maxDD=%.1f%% sharpe=%.2f acerto=%.1f%%" % (
        int(m.get("num_trades", 0) or 0), float(m.get("total_return_pct", 0) or 0) * 100,
        float(m.get("profit_factor", 0) or 0), float(m.get("max_drawdown_pct", 0) or 0) * 100,
        float(m.get("sharpe_ratio", 0) or 0), float(m.get("win_rate_pct", 0) or 0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None, help="padrao: <agente>_specialist_sac.zip da execucao")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "forward_live.parquet")
    args = parser.parse_args()

    from feature_engineering.training_schema import align_to_training_frame
    contract = json.loads((args.run_dir / "feature_contract.json").read_text(encoding="utf-8"))
    # The model is only valid under the stop scale and the frame it trained on.
    AIConfig.ENV_STOP_ATR_TIMEFRAME = contract.get("stop_atr_timeframe", "15m")
    frame = align_to_training_frame(pd.read_parquet(args.data).sort_index().ffill().fillna(0.0),
                                    contract.get("training_frame_columns"))
    meta = json.loads(Path(str(args.data) + ".meta.json").read_text(encoding="utf-8"))
    if not meta.get("stable") or meta.get("gaps"):
        print("BLOQUEADO: bloco futuro instavel ou com lacunas: %s" % meta)
        return 1
    agent = load_agent(args.agent, args.run_dir.resolve(), args.model)
    missing = [c for c in agent.feature_columns if c not in frame.columns]
    if missing:
        print("BLOQUEADO: %d colunas do contrato ausentes no bloco futuro: %s" % (len(missing), missing[:10]))
        return 1

    print("stops em ATR de %s | bloco %s -> %s (%d barras) | buy&hold %+.2f%%" % (AIConfig.ENV_STOP_ATR_TIMEFRAME,
        frame.index[0], frame.index[-1], len(frame), buy_and_hold_return(frame) * 100))
    metrics, curve = run_policy(agent, frame, args.agent)
    verdict = judge(metrics, buy_and_hold_return(frame))
    print("AGENTE     : %s" % line(metrics))
    exits = {k: v for k, v in metrics.items() if "exit" in k.lower()}
    if exits:
        print("saidas     : %s" % exits)

    teacher = {}
    try:
        from dataclasses import replace
        from learning.edge_policy import load_rule
        from tune_edge_rule import run_rule
        report_path = args.run_dir / ("%s_guided_report.json" % args.agent)
        saved = json.loads(report_path.read_text(encoding="utf-8"))["teacher_rule"]
        low, high = agent.model.action_space.low, agent.model.action_space.high
        rule = load_rule(saved)
        rule = replace(rule, sl_mult=float(np.clip(rule.sl_mult, low[1], high[1])))
        teacher = run_rule(frame, args.agent, rule)
        print("PROFESSORA : trades=%d retorno=%+.2f%% PF=%.2f maxDD=%.1f%%" % (
            teacher["trades"], teacher["net_return"] * 100, teacher["profit_factor"], teacher["max_drawdown"] * 100))
    except Exception as exc:
        print("aviso: professora nao medida: %s" % exc)

    close = frame["close"].astype(float)
    monthly = pd.DataFrame({
        "agente": curve.resample("MS").last().pct_change().fillna(curve.resample("MS").last().iloc[0] / curve.iloc[0] - 1),
        "btc": close.resample("MS").last() / close.resample("MS").first() - 1,
    })
    print("\nmes        agente      btc")
    for month, row in monthly.iterrows():
        print("%s  %+7.2f%%  %+7.2f%%" % (month.strftime("%Y-%m"), row["agente"] * 100, row["btc"] * 100))

    print()
    for name, passed in verdict["checks"].items():
        print("  [%s] %s" % ("OK " if passed else "NAO", name))
    print("APROVADO NO TESTE FUTURO: %s" % verdict["approved_for_live_trading"])

    report = {"agent": args.agent, "run_dir": str(args.run_dir), "model": str(args.model or ""),
              "generated_at": datetime.now(timezone.utc).isoformat(), "block": meta,
              "metrics": metrics, "verdict": verdict, "teacher": teacher,
              "monthly": {m.strftime("%Y-%m"): {k: float(v) for k, v in r.items()} for m, r in monthly.iterrows()}}
    out = args.run_dir / ("%s_forward_report.json" % args.agent)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("relatorio: %s" % out)
    return 0 if verdict["approved_for_live_trading"] else 1


if __name__ == "__main__":
    sys.exit(main())
