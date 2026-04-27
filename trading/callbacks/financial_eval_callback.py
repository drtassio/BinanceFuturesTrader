from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger, LOG_LEVEL_DEBUG
from stable_baselines3.common.callbacks import EvalCallback


logger = get_logger("FinancialEvalMetrics", LOG_LEVEL_DEBUG)


class FinancialEvalCallback(EvalCallback):
    """Callback de avaliação que coleta métricas financeiras detalhadas a cada avaliação."""

    def __init__(
        self,
        eval_env,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 0,
        metrics_log_path: Optional[str] = None,
        log_name: str = "FinancialEvalMetrics",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            **kwargs,
        )
        self.log_name = log_name
        self.metrics_log_path = Path(metrics_log_path) if metrics_log_path else None
        self.financial_history: List[Dict[str, Any]] = []
        self._last_eval_count = 0

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if not continue_training:
            return False

        eval_timesteps = getattr(self, "evaluations_timesteps", None)
        mean_reward = getattr(self, "last_mean_reward", None)

        if eval_timesteps is not None and isinstance(eval_timesteps, list):
            current_eval_count = len(eval_timesteps)
            if current_eval_count > self._last_eval_count and mean_reward is not None:
                self._last_eval_count = current_eval_count
                step_value = int(getattr(self.model, "num_timesteps", self.n_calls))
                episode_summaries = self._collect_episode_metrics()
                if episode_summaries:
                    record: Dict[str, Any] = {
                        "timesteps": step_value,
                        "mean_reward": float(mean_reward),
                        "episodes": episode_summaries,
                    }
                    aggregated = self._aggregate_episode_metrics(episode_summaries)
                    if aggregated:
                        record.update(aggregated)
                    self.financial_history.append(record)
                    log_payload: Dict[str, Any] = {
                        "steps": step_value,
                        "mean_reward": float(mean_reward),
                        "num_episodes": len(episode_summaries),
                    }
                    log_payload.update({k: v for k, v in record.items() if k not in {"timesteps", "mean_reward", "episodes"}})
                    get_logger(self.log_name, LOG_LEVEL_DEBUG).info("[Eval] %s", log_payload)
                    self._persist_metrics()
                else:
                    get_logger(self.log_name, LOG_LEVEL_DEBUG).info("[Eval] steps=%s mean_reward=%.2f (sem metricas financeiras - nenhum episodio finalizado)", step_value, float(mean_reward))

        return True

    def _collect_episode_metrics(self) -> List[Dict[str, Any]]:
        if not hasattr(self.eval_env, "env_method"):
            return []
        try:
            batches = self.eval_env.env_method("consume_episode_summaries")
        except AttributeError:
            return []
        summaries: List[Dict[str, Any]] = []
        for batch in batches:
            if not isinstance(batch, list):
                continue
            for item in batch:
                if isinstance(item, dict):
                    normalized: Dict[str, Any] = {}
                    for key, value in item.items():
                        if isinstance(value, (np.floating, float, int)):
                            normalized[key] = float(value)
                        else:
                            normalized[key] = value
                    summaries.append(normalized)
        return summaries

    def _aggregate_episode_metrics(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not summaries:
            return {}

        def _collect(key: str) -> List[float]:
            return [float(item[key]) for item in summaries if key in item]

        final_net_worth = _collect("final_net_worth")
        total_return_pct = _collect("total_return_pct")
        max_drawdown_pct = _collect("max_drawdown_pct")
        trade_sharpe = _collect("trade_sharpe")

        aggregated: Dict[str, Any] = {"num_episodes": len(summaries)}
        if final_net_worth:
            aggregated["mean_final_net_worth"] = float(np.mean(final_net_worth))
        if total_return_pct:
            aggregated["mean_total_return_pct"] = float(np.mean(total_return_pct))
            aggregated["best_total_return_pct"] = float(np.max(total_return_pct))
            aggregated["worst_total_return_pct"] = float(np.min(total_return_pct))
        if max_drawdown_pct:
            aggregated["mean_max_drawdown_pct"] = float(np.mean(max_drawdown_pct))
            aggregated["worst_max_drawdown_pct"] = float(np.max(max_drawdown_pct))
        if trade_sharpe:
            aggregated["mean_trade_sharpe"] = float(np.mean(trade_sharpe))
        return aggregated
    def _persist_metrics(self) -> None:
        if not self.metrics_log_path:
            return
        try:
            self.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_log_path.write_text(json.dumps(self.financial_history, indent=2), encoding="utf-8")
        except Exception:
            # Não interrompe o treino caso a escrita falhe
            pass
