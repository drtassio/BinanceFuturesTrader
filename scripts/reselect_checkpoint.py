"""Pick a guided run's checkpoint with the trade minimum registered for its teacher.

cloud/train_guided.py only scores a validation checkpoint that makes at least
AIConfig.OOS_MIN_TRADES (20) trades. A teacher registered with another minimum
before training (reports/leg_confirm_preregistration.json: 10, since it trades
about once a week per side) would otherwise keep a checkpoint chosen under a
rule it was never held to.

Selection reads VALIDATION only. Train and holdout are measured once, for the
chosen checkpoint, after the choice is made. The run's own model and report are
kept next to the new ones (suffix .train_guided).

    python scripts/reselect_checkpoint.py --agent bull --run cloud/artifacts/leg_confirm_run/bull_guided_<stamp>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")

from cloud.train_agent import AGENTS, buy_and_hold_return, evaluate, judge, load_dataset, split_chronological  # noqa: E402
from config.settings import AIConfig, TradingConfig  # noqa: E402


def score(m: dict, min_trades: int, min_pf: float, max_dd: float) -> float:
    trades = int(m.get("num_trades", 0) or 0)
    net = float(m.get("total_return_pct", 0.0) or 0.0)
    pf = float(m.get("profit_factor", 0.0) or 0.0)
    dd = float(m.get("max_drawdown_pct", 1.0) or 0.0)
    if trades < min_trades or net <= 0 or pf < min_pf or not 0.0 <= dd <= max_dd:
        return -np.inf
    return net / max(dd, 0.02)


def line(m: dict) -> str:
    return "trades=%d retorno=%+.2f%% PF=%.2f DD=%.1f%% acerto=%.1f%%" % (
        int(m.get("num_trades", 0) or 0), 100 * float(m.get("total_return_pct", 0.0) or 0.0),
        float(m.get("profit_factor", 0.0) or 0.0), 100 * float(m.get("max_drawdown_pct", 0.0) or 0.0),
        float(m.get("win_rate_pct", 0.0) or 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=("bull", "bear"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "featured_data_causal_leg5m.parquet")
    ap.add_argument("--min-trades", type=int, default=10)
    ap.add_argument("--min-pf", type=float, default=1.2)
    ap.add_argument("--max-dd", type=float, default=0.15)
    args = ap.parse_args()

    from specialists.trend_specialist import ClippedSAC

    run = args.run.resolve()
    contract_path = run / "feature_contract.json"
    
    # [FIX C5] Previne re-seleção cruzada que vazaria holdouts entre datasets
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        trained_dataset = contract.get("dataset")
        if trained_dataset and Path(trained_dataset).name != args.data.name:
            print(f"⚠️ [C5 ALERTA] Agente treinado em {Path(trained_dataset).name}, mas reselect tentou usar {args.data.name}.")
            print("Forçando uso do dataset original para preservar a integridade do holdout.")
            args.data = Path(trained_dataset)

    models = run / "models"
    train_df, val_df, holdout_df = split_chronological(load_dataset(args.data))
    config = AIConfig()
    config.ECONOMIC_REWARD_ONLY = True
    config.MODEL_DIR = str(models)
    agent = AGENTS[args.agent](config=config, trading_config=TradingConfig(),
                               input_dim=len(train_df.select_dtypes(include="number").columns))
    agent.load_model()
    if agent.model is None:
        raise SystemExit("modelo da execucao nao carregou")

    # best_validation.zip holds the best policy train_guided had seen, which is
    # the cloned one when no DAgger round or fine-tuning step beat it.
    candidates = ([models / "best_validation.zip"] if (models / "best_validation.zip").exists() else []) \
        + sorted(models.glob("dagger_*_validation.zip")) + sorted(
            models.glob("finetune_*_validation.zip"), key=lambda p: int(p.stem.split("_")[1]))
    print("stops em ATR de %s | %d checkpoints | criterio: >= %d trades, retorno > 0, PF >= %.1f, DD <= %.0f%%"
          % (AIConfig.ENV_STOP_ATR_TIMEFRAME, len(candidates), args.min_trades, args.min_pf, 100 * args.max_dd))
    best = (-np.inf, None, None)
    for path in candidates:
        agent.model = ClippedSAC.load(str(path), device=agent.model.device)
        m = evaluate(agent, val_df, args.agent, deterministic=True)
        s = score(m, args.min_trades, args.min_pf, args.max_dd)
        print("  %-30s validacao: %s%s" % (path.stem, line(m), "  -> apto" if np.isfinite(s) else ""))
        if s > best[0]:
            best = (s, path, m)
    if best[1] is None:
        print("NENHUM checkpoint atende ao criterio na validacao; nada foi alterado.")
        return 1

    chosen, val_metrics = best[1], best[2]
    agent.model = ClippedSAC.load(str(chosen), device=agent.model.device)
    train_metrics = evaluate(agent, train_df, args.agent, deterministic=True)
    holdout_metrics = evaluate(agent, holdout_df, args.agent, deterministic=True)
    config.OOS_MIN_TRADES = args.min_trades
    verdict = judge(holdout_metrics, buy_and_hold_return(holdout_df), config)
    print("\nescolhido na validacao: %s (%s)" % (chosen.stem, line(val_metrics)))
    print("TREINO : %s" % line(train_metrics))
    print("HOLDOUT: %s | buy&hold %+.2f%%" % (line(holdout_metrics), 100 * verdict["buy_and_hold_return"]))

    final = models / ("%s_specialist_sac.zip" % args.agent)
    report_path = run / ("%s_guided_report.json" % args.agent)
    for path in (final, report_path):
        keep = path.with_name(path.stem + ".train_guided" + path.suffix)
        if path.exists() and not keep.exists():
            shutil.copy2(path, keep)
    shutil.copy2(chosen, final)
    contract_path = run / "feature_contract.json"
    if not contract_path.exists():
        # A run stopped before its end never wrote the contract the promotion
        # and the live mirror read; it is the same one train_guided writes.
        full = load_dataset(args.data)
        contract_path.write_text(json.dumps({
            "feature_columns": list(agent.feature_columns),
            "stop_atr_timeframe": AIConfig.ENV_STOP_ATR_TIMEFRAME,
            "leverage_bounds": [float(TradingConfig.MIN_LEVERAGE_PER_TRADE),
                                float(min(TradingConfig.MAX_LEVERAGE_PER_TRADE, AIConfig.TRAINING_LEVERAGE_CAP))],
            "training_frame_columns": list(full.columns),
            "dataset": str(args.data),
        }, indent=2), encoding="utf-8")
        print("contrato gravado: %s" % contract_path)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report.update({
        "periods": {"train": [str(train_df.index.min()), str(train_df.index.max())],
                    "validation": [str(val_df.index.min()), str(val_df.index.max())],
                    "holdout": [str(holdout_df.index.min()), str(holdout_df.index.max())]},
        "agent": args.agent, "selected": chosen.stem, "selected_validation": val_metrics,
        "train_metrics": train_metrics, "holdout_metrics": holdout_metrics, "verdict": verdict,
        "reselection": {"script": "scripts/reselect_checkpoint.py", "min_trades": args.min_trades,
                        "min_pf": args.min_pf, "max_dd": args.max_dd,
                        "registered_in": "reports/leg_confirm_preregistration.json"},
    })
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("relatorio: %s" % report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
