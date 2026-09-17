"""Out-of-sample approval for the edge teacher that operates live.

The unit approved is what trades: Bull and Bear teachers on one account. Each
is run through its environment on the holdout block; their per-bar returns add
(positions never overlap: Bull holds while edge > 0, Bear enters at
edge <= -0.1), and the combined equity curve is held to the same thresholds as
the SAC specialists (config OOS_*). The report binds the approval to the
SHA-256 of the meta-model, the rule files and the rule code, so replacing any
of them invalidates it.

    python scripts/approve_teacher_oos.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")
os.chdir(ROOT)

from config.settings import AIConfig  # noqa: E402
from trading.teacher_policy import (AGENTS, approval_is_valid, approval_path,  # noqa: E402
                                    artifact_hashes, load_rules, parity_is_valid, trajectory)


def trade_returns(path: pd.DataFrame) -> list:
    """Net return of each trade, from the equity before entry to the bar it closed."""
    returns, start = [], None
    worth = path["net_worth"].to_numpy()
    sides = path["side"].to_numpy()
    before = np.concatenate([[worth[0]], worth[:-1]])
    for i, side in enumerate(sides):
        if side != 0 and (i == 0 or sides[i - 1] != side):
            if start is not None:
                returns.append(worth[i - 1] / before[start] - 1.0)
            start = i
        elif side == 0 and start is not None:
            returns.append(worth[i] / before[start] - 1.0)
            start = None
    if start is not None:
        returns.append(worth[-1] / before[start] - 1.0)
    return returns


def metrics(equity: pd.Series, trades: list) -> dict:
    daily = equity.resample("1D").last().dropna().pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0
    peak = equity.cummax()
    gains = sum(r for r in trades if r > 0)
    losses = -sum(r for r in trades if r < 0)
    return {
        "net_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "max_drawdown": float(((peak - equity) / peak).max()),
        "sharpe_ratio": sharpe,
        "profit_factor": float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0,
        "num_trades": len(trades),
        "win_rate": float(np.mean([r > 0 for r in trades])) if trades else 0.0,
    }


def main() -> int:
    from cloud.train_agent import load_dataset, split_chronological

    config = AIConfig()
    model_dir = Path(str(config.MODEL_DIR))
    if not parity_is_valid(model_dir):
        print("BLOQUEADO: paridade do replay ausente, divergente ou desatualizada. Rode verify_teacher_parity.py.")
        return 1
    _, _, holdout = split_chronological(load_dataset(ROOT / "data" / "featured_data_causal.parquet"))
    rules = load_rules(model_dir)
    thresholds = {
        "min_sharpe": float(getattr(config, "OOS_MIN_SHARPE", 0.5)),
        "min_profit_factor": float(getattr(config, "OOS_MIN_PROFIT_FACTOR", 1.1)),
        "max_drawdown": float(getattr(config, "OOS_MAX_DRAWDOWN", 0.15)),
        "min_net_return": float(getattr(config, "OOS_MIN_NET_RETURN", 0.0)),
        "min_trades": int(getattr(config, "OOS_MIN_TRADES", 20)),
    }

    per_agent, bar_returns, all_trades, positions = {}, [], [], []
    for agent in AGENTS:
        path = trajectory(holdout, agent, rules[agent], AIConfig())
        trades = trade_returns(path)
        per_agent[agent] = metrics(path["net_worth"], trades)
        bar_returns.append(path["net_worth"].pct_change().fillna(0.0))
        positions.append(path["side"])
        all_trades.extend(trades)
    combined_returns = pd.concat(bar_returns, axis=1).fillna(0.0).sum(axis=1)
    equity = (1.0 + combined_returns).cumprod()
    combined = metrics(equity, all_trades)

    checks = {
        # Summing independent account returns is not a shared-account backtest
        # when both specialists carry exposure on the same bar.
        "no_overlapping_positions": bool((pd.concat(positions, axis=1).fillna(0).ne(0).sum(axis=1) <= 1).all()),
        "sharpe": combined["sharpe_ratio"] >= thresholds["min_sharpe"],
        "profit_factor": combined["profit_factor"] >= thresholds["min_profit_factor"],
        "drawdown": combined["max_drawdown"] <= thresholds["max_drawdown"],
        "net_return": combined["net_return"] >= thresholds["min_net_return"],
        "trades": combined["num_trades"] >= thresholds["min_trades"],
    }
    hashes = artifact_hashes(model_dir)
    close = holdout["close"].astype(float)
    report = {
        "policy": "edge_teacher",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holdout": [str(holdout.index[0]), str(holdout.index[-1])],
        "buy_and_hold": float(close.iloc[-1] / close.iloc[0] - 1.0),
        "agents": list(AGENTS),
        "rules": {agent: rules[agent].as_dict() for agent in AGENTS},
        "thresholds": thresholds,
        "per_agent": per_agent,
        "combined": combined,
        "checks": checks,
        "all_passed": all(checks.values()) and all(hashes.values()),
        "artifact_hashes": hashes,
    }
    approval_path(model_dir).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def line(m):
        return "retorno=%+.2f%% maxDD=%.1f%% sharpe=%.2f PF=%.2f trades=%d acerto=%.0f%%" % (
            m["net_return"] * 100, m["max_drawdown"] * 100, m["sharpe_ratio"], m["profit_factor"],
            m["num_trades"], m["win_rate"] * 100)

    print("holdout %s -> %s | buy&hold %+.2f%%" % (report["holdout"][0][:16], report["holdout"][1][:16],
                                                   report["buy_and_hold"] * 100))
    for agent in AGENTS:
        print("  %-9s %s" % (agent, line(per_agent[agent])))
    print("  %-9s %s" % ("combinado", line(combined)))
    for name, passed in checks.items():
        print("  [%s] %s" % ("OK " if passed else "NAO", name))
    print("APROVADO: %s | aprovacao valida para os arquivos em disco: %s" % (
        report["all_passed"], approval_is_valid(model_dir)))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
