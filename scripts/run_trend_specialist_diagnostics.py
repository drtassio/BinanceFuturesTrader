#!/usr/bin/env python3
"""
Execução orientada de diagnósticos para o TrendSpecialist.

O script permite:
  • Gerar um dataset sintético controlado (útil para smoke tests)
  • Rodar uma mini otimização Optuna (n_trials reduzido) com logging dedicado
  • Coletar estatísticas de gating diretamente do ambiente (armazenadas em user_attrs)

Exemplo:
    python scripts/run_trend_specialist_diagnostics.py --synthetic --n-trials 2 --timesteps 12000 --relaxed-gates
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from config.settings import AIConfig  # noqa: E402
from specialists.trend_specialist import (  # noqa: E402
    ANTI_SCALPING_REQUIRED_KEYS,
    TrendSpecialist,
)
from utils.logger import get_logger, setup_phase_logger  # noqa: E402

DEFAULT_STUDY_NAME = "trend_specialist_sac_optimization_v3"


def build_synthetic_dataset(rows: int = 12000, freq: str = "5T") -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=rows, freq=freq)
    rng = np.random.default_rng(42)
    drift = rng.normal(loc=0.0005, scale=0.0008, size=rows)
    noise = rng.normal(loc=0.0, scale=0.002, size=rows)
    close = 100 + np.cumsum(drift + noise)
    close = np.maximum(close, 1.0)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    high = np.maximum(open_price, close) + np.abs(rng.normal(scale=0.003, size=rows))
    low = np.minimum(open_price, close) - np.abs(rng.normal(scale=0.003, size=rows))
    volume = rng.lognormal(mean=12, sigma=0.4, size=rows)

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )

    df["ema_trend_15m"] = df["close"].ewm(span=55, adjust=False).mean() - df["close"].ewm(span=144, adjust=False).mean()
    true_range = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    true_range[0] = high[0] - low[0]
    df["atr_15m"] = pd.Series(true_range, index=idx).rolling(window=14, min_periods=1).mean()

    slope = np.gradient(df["close"])
    slope_norm = np.tanh(slope / np.std(slope))
    df["tp_prior_dir"] = slope_norm
    df["tp_prior_conf"] = np.clip(0.45 + np.abs(slope_norm) * 0.35, 0.1, 0.95)
    df["tp_uncertainty"] = 1.0 - df["tp_prior_conf"]
    df["tp_duration_median"] = np.clip(20 + rng.normal(scale=5, size=rows), 5, 80)
    df["sdae_recon_error"] = np.abs(rng.normal(scale=0.05, size=rows))

    up_score = np.exp(slope_norm * 2.0)
    down_score = np.exp(-slope_norm * 2.0)
    side_score = np.clip(1.0 + rng.normal(scale=0.1, size=rows), 0.1, None)
    denom = up_score + down_score + side_score
    df["tp_regime_up"] = up_score / denom
    df["tp_regime_down"] = down_score / denom
    df["tp_regime_sideways"] = side_score / denom

    for key in ("qpnl_24_p50", "qpnl_24_p90", "qpnl_48_p50", "qpnl_48_p90"):
        df[key] = rng.normal(loc=0.0, scale=0.01, size=rows)

    return df.fillna(method="bfill").fillna(method="ffill")


def ensure_anti_scalping(agent: TrendSpecialist) -> None:
    if getattr(agent, "anti_scalping_config", None):
        return
    default_config: Dict[str, float] = {
        "min_trade_duration": 15,
        "scalp_penalty": 2.5,
        "opening_bonus_mult": 1.0,
        "long_trade_bonus_threshold": 24,
        "long_trade_bonus_max": 2.5,
        "post_trade_cooldown": 40,
        "max_trades_per_episode": 120,
        "max_trade_duration": 300,
    }
    for key in ANTI_SCALPING_REQUIRED_KEYS:
        default_config.setdefault(key, 0.0)
    agent.anti_scalping_config = default_config


def _archive_optuna_db(db_path: Path) -> None:
    if not db_path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.with_name(f"{db_path.stem}_{timestamp}{db_path.suffix}.bak")
    try:
        db_path.rename(backup_path)
    except OSError:
        pass


def run_smoke_optuna(
    df: pd.DataFrame,
    n_trials: int,
    n_timesteps: int,
    n_eval_episodes: int,
    relaxed_gates: bool,
    reset_db: bool,
    no_pruner: bool,
) -> None:
    ai_cfg = AIConfig()
    ai_cfg.RELAXED_GATES_TREND = bool(relaxed_gates)
    if relaxed_gates:
        os.environ["RELAXED_GATES_TREND"] = "1"
    if no_pruner:
        os.environ["TREND_OPTUNA_NOPRUNE"] = "1"
        os.environ["TREND_DISABLE_FINANCIAL_PRUNE"] = "1"
    else:
        os.environ.pop("TREND_OPTUNA_NOPRUNE", None)
        os.environ.pop("TREND_DISABLE_FINANCIAL_PRUNE", None)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    input_dim = len(numeric_cols)
    agent = TrendSpecialist(config=ai_cfg, input_dim=input_dim)
    ensure_anti_scalping(agent)
    if reset_db:
        _archive_optuna_db(Path(agent.optuna_db_path))

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    eval_df = df.iloc[split_idx:].copy()

    logger = get_logger("TrendSpecialistDiagnostics")
    logger.info(
        "[DIAG] Iniciando smoke-test Optuna | n_trials=%s | timesteps=%s | relaxed_gates=%s",
        n_trials,
        n_timesteps,
        relaxed_gates,
    )
    if n_trials > 0:
        agent.optimize_hyperparameters(
            train_df=train_df,
            eval_df=eval_df,
            n_trials=n_trials,
            n_timesteps=n_timesteps,
            n_eval_episodes=n_eval_episodes,
        )
    else:
        logger.info("[DIAG] n_trials=0 -> apenas coleta de métricas existentes.")

    storage_url = f"sqlite:///{agent.optuna_db_path}"
    try:
        study = optuna.load_study(study_name=DEFAULT_STUDY_NAME, storage=storage_url)
        trials = study.get_trials(deepcopy=False)
        if not trials:
            logger.info("[DIAG] Study carregado, mas sem trials registrados.")
            return
        latest_trial = max(trials, key=lambda t: t.number)
        logger.info("[DIAG] Último trial: #%s | state=%s | value=%s", latest_trial.number, latest_trial.state, latest_trial.value)
        gate_stats = latest_trial.user_attrs.get("gate_stats")
        if gate_stats:
            logger.info("[DIAG] Gate stats (último trial): %s", gate_stats)
        reason = latest_trial.user_attrs.get("forced_prune_reason")
        if reason:
            logger.info("[DIAG] Motivo da poda: %s (steps=%s)", reason, latest_trial.user_attrs.get("pruned_timesteps"))
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if completed:
            best_trial = max(completed, key=lambda t: t.value)
            logger.info("[DIAG] Melhor trial completo: #%s | value=%.4f", best_trial.number, best_trial.value)
    except Exception as exc:
        logger.warning("[DIAG] Não foi possível carregar o estudo Optuna para resumo: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa diagnóstico automatizado do TrendSpecialist.")
    parser.add_argument("--synthetic", action="store_true", help="Usa dataset sintético em vez de carregar features reais.")
    parser.add_argument("--n-trials", type=int, default=3, help="Número de trials Optuna para o smoke-test.")
    parser.add_argument("--timesteps", type=int, default=20000, help="Timesteps por trial no smoke-test.")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Episódios de avaliação por trial.")
    parser.add_argument("--relaxed-gates", action="store_true", help="Ativa modo relaxado dos gates durante o teste.")
    parser.add_argument("--reset-db", action="store_true", help="Arquiva o banco Optuna antes de rodar os trials.")
    parser.add_argument("--no-pruner", action="store_true", help="Desativa pruning do Optuna para coletar trials completos.")
    parser.add_argument("--export-dataset", type=Path, help="Se definido, salva o dataset sintético gerado em CSV.")
    parser.add_argument("--last-days", type=int, default=0, help="Se > 0, usa apenas os últimos N dias do dataset real.")
    parser.add_argument("--last-years", type=int, default=0, help="Se > 0, usa apenas os últimos N anos do dataset real.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = setup_phase_logger("TrendSpecialist_Diagnostics", "TrendSpecialist")
    logger = get_logger("TrendSpecialistDiagnostics")
    logger.info("[DIAG] Logging detalhado em %s", log_file)

    if args.synthetic:
        df = build_synthetic_dataset()
        if args.export_dataset:
            args.export_dataset.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.export_dataset)
            logger.info("[DIAG] Dataset sintético exportado para %s", args.export_dataset)
    else:
        from scripts.train_trend_specialist import load_or_build_features  # type: ignore

        df = load_or_build_features()
        df = df.sort_index()
        # Recorte temporal opcional (dados mais recentes)
        try:
            if args.last_days and args.last_days > 0 and not df.empty:
                cutoff = df.index.max() - pd.Timedelta(days=int(args.last_days))
                df = df[df.index >= cutoff]
            if args.last_years and args.last_years > 0 and not df.empty:
                cutoff = df.index.max() - pd.DateOffset(years=int(args.last_years))
                df = df[df.index >= cutoff]
        except Exception:
            pass
        logger.info("[DIAG] Dataset real carregado (%s linhas).", len(df))

    run_smoke_optuna(
        df=df,
        n_trials=args.n_trials,
        n_timesteps=args.timesteps,
        n_eval_episodes=args.eval_episodes,
        relaxed_gates=args.relaxed_gates,
        reset_db=args.reset_db,
        no_pruner=args.no_pruner,
    )

    logger.info("[DIAG] Diagnóstico concluído.")


if __name__ == "__main__":
    main()
