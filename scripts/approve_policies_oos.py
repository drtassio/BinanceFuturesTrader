"""Run the out-of-sample gate against the trained policies and record the verdict.

This is the bridge between training and live trading. cloud/train_agent.py
reports how one agent did on its own holdout; this script asks the same
question of all three at once, in the shape the bot itself checks before it is
allowed to send an order, and writes an approval bound to the SHA-256 of each
policy file.

Retraining any specialist changes that file's hash and silently invalidates the
approval, which is the point: an approval should vouch for the model that was
measured, not for whatever happens to sit at that path later.

    python scripts/approve_policies_oos.py
    python scripts/approve_policies_oos.py --holdout-fraction 0.15
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")

from config.settings import AIConfig, TradingConfig  # noqa: E402
from trading.ai_controller import AIController  # noqa: E402
from specialists.bull_specialist import BullSpecialist  # noqa: E402
from specialists.bear_specialist import BearSpecialist  # noqa: E402
from specialists.ranger_specialist import RangerSpecialist  # noqa: E402

AGENTS = {"bull": BullSpecialist, "bear": BearSpecialist, "ranger": RangerSpecialist}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "featured_data_causal.parquet")
    parser.add_argument("--holdout-fraction", type=float, default=None,
                        help="legacy option: only 0.15 is supported; uses the trainer's embargoed split")
    args = parser.parse_args()
    if args.holdout_fraction is not None and args.holdout_fraction != 0.15:
        parser.error('use the canonical embargoed training/validation/holdout split')

    frame = pd.read_parquet(args.data).sort_index().ffill().fillna(0.0)
    from cloud.train_agent import split_chronological
    _, _, holdout = split_chronological(frame)
    print("holdout: %s -> %s (%d barras)" % (holdout.index.min(), holdout.index.max(), len(holdout)))
    close = holdout["close"].astype(float)
    print("buy & hold no periodo: %+.2f%%\n" % ((close.iloc[-1] / close.iloc[0] - 1.0) * 100))

    config = AIConfig()
    trading_config = TradingConfig()
    input_dim = len(frame.select_dtypes(include="number").columns)

    controller = object.__new__(AIController)
    controller.config_ai = config
    controller.policy_validation_path = str(Path(config.MODEL_DIR) / "policy_oos_validation.json")
    controller.policy_oos_approved = False
    controller.last_oos_validation = {}
    controller.specialists = {}

    for name, agent_class in AGENTS.items():
        try:
            agent = agent_class(config=config, trading_config=trading_config, input_dim=input_dim)
            agent.load_model()
            controller.specialists[name] = agent
            print("  %-7s modelo carregado" % name)
        except Exception as exc:
            print("  %-7s NAO carregado: %s" % (name, exc))

    report = controller._validate_specialists_oos(holdout)

    print()
    print("=" * 96)
    print("%-8s %8s %9s %9s %8s %9s  %s" % ("agente", "trades", "retorno", "sharpe", "PF", "maxDD", "veredito"))
    print("=" * 96)
    for name, result in report["specialists"].items():
        metrics = result.get("metrics", {})
        print("%-8s %8d %8.2f%% %9.2f %8.2f %8.1f%%  %s" % (
            name,
            int(metrics.get("num_trades", 0) or 0),
            float(metrics.get("net_return", 0.0) or 0.0) * 100,
            float(metrics.get("sharpe_ratio", 0.0) or 0.0),
            float(metrics.get("profit_factor", 0.0) or 0.0),
            float(metrics.get("max_drawdown", 0.0) or 0.0) * 100,
            result.get("reason", ""),
        ))
    print()
    thresholds = report["thresholds"]
    print("limiares: sharpe>=%.2f  PF>=%.2f  maxDD<=%.0f%%  retorno>=%.0f%%  trades>=%d" % (
        thresholds["min_sharpe"], thresholds["min_profit_factor"],
        thresholds["max_drawdown"] * 100, thresholds["min_net_return"] * 100,
        thresholds["min_trades"]))
    print()
    if report["all_passed"]:
        print("APROVADO: as tres politicas podem operar.")
    else:
        print("NAO APROVADO. O bot recusara enviar ordens enquanto isto nao passar.")
    print("relatorio: %s" % controller.policy_validation_path)
    print("aprovacao valida agora: %s" % controller._load_policy_oos_approval())
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
