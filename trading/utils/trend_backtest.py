from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from specialists.trend_specialist import TrendSpecialist, TrendFollowingEnv

DEFAULT_EVAL_MONTHS = 6
DEFAULT_EVAL_MIN_ROWS = 2000
DEFAULT_HOLDOUT_MONTHS = 2


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    result = df.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index, utc=True, errors="coerce")
    else:
        if result.index.tz is None:
            result.index = result.index.tz_localize("UTC")
        else:
            result.index = result.index.tz_convert("UTC")
    result = result[~result.index.isna()].sort_index()
    return result


def check_data_freshness(
    df: pd.DataFrame, max_staleness_hours: int = 24
) -> Tuple[bool, Optional[pd.Timestamp], float]:
    if df is None or df.empty:
        return False, None, math.inf
    latest_ts = df.index.max()
    if isinstance(latest_ts, pd.Timestamp):
        if latest_ts.tz is None:
            latest_ts = latest_ts.tz_localize("UTC")
        else:
            latest_ts = latest_ts.tz_convert("UTC")
    # Usa now com timezone UTC diretamente para evitar ambiguidade de tz
    now_utc = pd.Timestamp.now(tz="UTC")
    staleness_hours = float((now_utc - latest_ts).total_seconds() / 3600.0)
    return staleness_hours <= max_staleness_hours, latest_ts, staleness_hours


def split_train_holdout(
    featured: pd.DataFrame,
    holdout_months: int = DEFAULT_HOLDOUT_MONTHS,
    min_holdout_rows: int = DEFAULT_EVAL_MIN_ROWS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = ensure_datetime_index(featured)
    if df.empty:
        return df, pd.DataFrame()
    latest_ts = df.index.max()
    holdout_start = latest_ts - pd.DateOffset(months=holdout_months)
    holdout_df = df.loc[df.index >= holdout_start].copy()
    train_df = df.loc[df.index < holdout_start].copy()
    if holdout_df.empty or len(holdout_df) < max(min_holdout_rows, 1):
        return df, pd.DataFrame()
    if train_df.empty:
        return df, pd.DataFrame()
    return train_df, holdout_df


def build_unseen_eval_frame(
    featured: pd.DataFrame,
    train_cutoff_date: Optional[str] = None,
    eval_months: int = 3,
) -> pd.DataFrame:
    df = ensure_datetime_index(featured)
    if df.empty:
        return pd.DataFrame()

    if train_cutoff_date:
        cutoff_ts = pd.to_datetime(train_cutoff_date, utc=True)
    else:
        cutoff_idx = int(len(df) * 0.8)
        cutoff_ts = df.index[min(cutoff_idx, len(df) - 1)]

    eval_start = cutoff_ts + pd.DateOffset(days=1)
    eval_end = eval_start + pd.DateOffset(months=eval_months)
    eval_df = df.loc[(df.index >= eval_start) & (df.index <= eval_end)].copy()

    if eval_df.empty:
        end_ts = df.index.max()
        start_ts = end_ts - pd.DateOffset(months=eval_months)
        eval_df = df.loc[df.index >= start_ts].copy()
    return eval_df


def build_recent_eval_frame(
    featured: pd.DataFrame, months: int, min_rows: int
) -> pd.DataFrame:
    df = ensure_datetime_index(featured)
    if df.empty:
        return pd.DataFrame()

    months = max(int(months or 1), 1)
    min_rows = max(int(min_rows or 0), 0)

    end_ts = df.index.max()
    start_ts = end_ts - pd.DateOffset(months=months)
    eval_df = df.loc[df.index >= start_ts].copy()

    if eval_df.empty and min_rows:
        eval_df = df.tail(min_rows).copy()
    elif len(eval_df) < min_rows:
        eval_df = df.tail(max(min_rows, len(eval_df))).copy()
    return eval_df


def _log_message(
    logger: Optional[Any], level: str, message: str, *, default_printer: Callable[[str], None] = print
) -> None:
    if logger is not None:
        log_fn = getattr(logger, level, None)
        if callable(log_fn):
            log_fn(message)
            return
    default_printer(message)


def run_manual_backtest(
    agent: TrendSpecialist,
    eval_df: pd.DataFrame,
    log_dir: Path,
    label: str = "manual_backtest",
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    if eval_df is None or eval_df.empty:
        _log_message(logger, "warning", "[BACKTEST] DataFrame de avaliação está vazio. Backtest não executado.")
        return {}

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_env = getattr(agent, "_env", None)
    model = getattr(base_env, "model", None)
    if model is None:
        try:
            agent.load_model()
            base_env = getattr(agent, "_env", None)
            model = getattr(base_env, "model", None)
        except Exception as exc:
            _log_message(
                logger,
                "warning",
                f"[BACKTEST] Não foi possível carregar o modelo SAC salvo: {exc}",
            )
            return {}

    if model is None:
        _log_message(logger, "warning", "[BACKTEST] Modelo SAC indisponível após o treinamento. Backtest não executado.")
        return {}

    agent_cfg = getattr(agent, "config", None) or getattr(agent, "config_ai", None)
    if agent_cfg is None:
        _log_message(
            logger, "warning", "[BACKTEST] Configuração do agente não encontrada. Backtest não executado."
        )
        return {}

    try:
        eval_env = TrendFollowingEnv(
            df=eval_df,
            config=agent_cfg,
            profitability_predictor=getattr(agent, "profitability_predictor", None),
            mode="evaluation",
        )
    except TypeError:
        eval_env = TrendFollowingEnv(eval_df, agent_cfg)

    obs, _ = eval_env.reset()
    cumulative_reward = 0.0
    trades: List[Dict[str, Any]] = []
    nav_records: List[Dict[str, Any]] = []

    def _fmt_ts(idx: int) -> str:
        if idx < 0 or idx >= len(eval_env.df.index):
            idx = len(eval_env.df.index) - 1
        ts = eval_env.df.index[idx]
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is None:
                try:
                    ts = ts.tz_localize("UTC")
                except (TypeError, ValueError):
                    pass
            return ts.isoformat()
        if isinstance(ts, datetime):
            return ts.isoformat()
        return str(ts)

    nav_records.append(
        {
            "step": 0,
            "timestamp": _fmt_ts(0),
            "net_worth": float(eval_env.net_worth),
            "reward": 0.0,
        }
    )

    done = False
    step_idx = 0
    safety_cap = len(eval_env.df) * 2

    while not done and step_idx < safety_cap:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        cumulative_reward += float(reward)
        done = bool(terminated or truncated)
        step_idx += 1

        current_step = min(eval_env.current_step, len(eval_env.df) - 1)
        nav_records.append(
            {
                "step": step_idx,
                "timestamp": _fmt_ts(current_step),
                "net_worth": float(eval_env.net_worth),
                "reward": float(reward),
            }
        )

        trade = info.get("trade")
        if trade:
            trades.append(dict(trade))

    # Métricas agregadas
    total_trades = len(trades)
    profits = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    win_rate_pct = (len(profits) / total_trades * 100.0) if total_trades else 0.0
    avg_profit = float(np.mean(profits)) if profits else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    total_profit = float(np.sum(profits)) if profits else 0.0
    total_loss_abs = float(abs(np.sum(losses))) if losses else 0.0
    profit_factor = float(total_profit / total_loss_abs) if total_loss_abs > 0 else float("inf")

    durations = [t.get("duration_steps", 0) for t in trades]
    avg_duration = float(np.mean(durations)) if durations else 0.0

    final_net_worth = float(eval_env.net_worth)
    initial_capital = float(eval_env.initial_balance)
    net_profit = final_net_worth - initial_capital
    net_return_pct = (net_profit / initial_capital * 100.0) if initial_capital else 0.0

    # Confusion matrix (direção vs resultado)
    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for trade in trades:
        side = trade.get("side")
        pnl = trade.get("pnl", 0.0)
        if side == "long":
            if pnl > 0:
                confusion["TP"] += 1
            else:
                confusion["FN"] += 1
        elif side == "short":
            if pnl > 0:
                confusion["TN"] += 1
            else:
                confusion["FP"] += 1

    confusion_total = sum(confusion.values())
    accuracy = float((confusion["TP"] + confusion["TN"]) / confusion_total) if confusion_total else 0.0
    precision = float(confusion["TP"] / (confusion["TP"] + confusion["FP"])) if (confusion["TP"] + confusion["FP"]) else 0.0
    recall = float(confusion["TP"] / (confusion["TP"] + confusion["FN"])) if (confusion["TP"] + confusion["FN"]) else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    confusion["accuracy"] = accuracy
    confusion["precision"] = precision
    confusion["recall"] = recall
    confusion["f1"] = f1

    ts_label = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    nav_path = log_dir / f"{label}_nav_{ts_label}.csv"
    trades_path = log_dir / f"{label}_trades_{ts_label}.csv"
    summary_path = log_dir / f"{label}_summary_{ts_label}.json"

    nav_df = pd.DataFrame(nav_records)
    nav_df.to_csv(nav_path, index=False)
    if total_trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(trades_path, index=False)
    else:
        trades_path = None

    summary: Dict[str, Any] = {
        "window_start": str(eval_df.index.min()),
        "window_end": str(eval_df.index.max()),
        "rows_evaluated": int(len(eval_df)),
        "final_net_worth": final_net_worth,
        "net_profit": float(net_profit),
        "net_return_pct": float(net_return_pct),
        "total_reward": float(cumulative_reward),
        "total_trades": total_trades,
        "win_rate_pct": float(win_rate_pct),
        "profit_factor": float(profit_factor),
        "avg_profit": float(avg_profit),
        "avg_loss": float(avg_loss),
        "avg_trade_duration_steps": float(avg_duration),
        "max_drawdown_pct": float(eval_env.episode_max_drawdown if hasattr(eval_env, "episode_max_drawdown") else 0.0),
        "confusion_matrix": confusion,
        "nav_path": str(nav_path),
        "trades_path": str(trades_path) if trades_path else "",
        "log_dir": str(log_dir),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary["summary_path"] = str(summary_path)

    message = (
        f"[BACKTEST] Execução concluída. NetWorth final: {final_net_worth:.2f} | "
        f"Lucro: {net_profit:.2f} ({net_return_pct:.2f}%) | Trades: {total_trades}"
    )
    _log_message(logger, "info", message)
    if total_trades and confusion_total > 0:
        _log_message(logger, "info", "[BACKTEST] Matriz de Confusão (direção prevista vs real):")
        _log_message(logger, "info", "         Actual Up   Actual Down")
        _log_message(
            logger,
            "info",
            f"Pred Up     {confusion['TP']:>6}        {confusion['FP']:>6}",
        )
        _log_message(
            logger,
            "info",
            f"Pred Down   {confusion['FN']:>6}        {confusion['TN']:>6}",
        )
        _log_message(
            logger,
            "info",
            (
                f"         Acc={confusion['accuracy']:.3f} | Prec={confusion['precision']:.3f} | "
                f"Recall={confusion['recall']:.3f} | F1={confusion['f1']:.3f}"
            ),
        )

    return summary

