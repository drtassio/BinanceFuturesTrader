from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from scipy import stats
from stable_baselines3.common.callbacks import EvalCallback

from utils.logger import LOG_LEVEL_DEBUG, get_logger

# Caminho absoluto do projeto (independente de os.getcwd())
_CALLBACKS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CALLBACKS_DIR))
_HPO_TB_ROOT = os.path.join(_PROJECT_ROOT, "logs", "tensorboard")


class TrialEvalCallback(EvalCallback):
    """
    Callback de avaliação integrado ao Optuna.

    - Executa avaliações periódicas durante o treino (via EvalCallback)
    - Reporta a recompensa média ao `trial` do Optuna
    - Solicita pruning quando `trial.should_prune()` retornar True

    Isso evita dependência de exemplos externos e corrige o NameError quando
    `TrialEvalCallback` não está disponível no ambiente.
    """

    def __init__(
        self,
        eval_env,
        trial: optuna.trial.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        log_name: str = "OptunaFinancialMetrics",
        log_path: str | None = None,
        total_timesteps: int = 0,
        learning_monitor=None,
    ) -> None:
        # ⚠️ DESABILITADO: Não gravar arquivos em disco - apenas logs
        # base_log_dir = log_path or os.path.join(os.getcwd(), "logs", "optuna_eval", log_name)
        # os.makedirs(base_log_dir, exist_ok=True)
        # log_path_full = os.path.join(base_log_dir, f"trial_{trial.number}")
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            render=False,
            verbose=verbose,
            log_path=None,  # Não grava em disco
        )
        self.trial = trial
        self.log_name = log_name
        self.is_pruned = False
        self.best_mean_reward = -np.inf
        # Controla reportes para o Optuna de forma independente de n_calls
        self._last_eval_count = 0
        self._total_evals = (
            max(1, total_timesteps // eval_freq) if total_timesteps > 0 else 0
        )
        self._learning_monitor = learning_monitor
        self.financial_history: List[Dict[str, Any]] = []
        self._last_report_step: float = float("-inf")

        # 🚀 HPO ULTRA RÁPIDO: Configuração de pruning para 10k timesteps
        # Baseado em best practices de RL HPO para mercados financeiros
        self.trade_prune_threshold = 5  # Poucos trades esperados em 10k
        self.min_force_prune_step = 8500  # ~85% do trial de 10k
        self.early_negative_prune_step = (
            15000  # Dá mais tempo para o SAC sair do buraco inicial
        )
        self.mean_reward_floor = (
            -10000.0
        )  # Praticamente desativa poda por valor negativo absoluto
        self.episode_reward_floor = -50000.0  # Muito tolerante
        self.min_episode_count_for_prune = 2  # Menos episódios em 10k steps
        self.negative_streak_patience = 6  # Paciência razoável

        self._negative_mean_reward_streak = 0
        self.eval_history: List[Tuple[int, float]] = []
        self.last_eval_step: int = -1

        # [REMOVIDO] SummaryWriter por trial causava poluição no TensorBoard:
        # 100 trials HPO = 100 runs separados no TB, tornando a interface inutilizável.
        # Métricas de HPO ficam disponíveis via Optuna DB (logs/*.db) e console logs.
        self._tb_writer = None

    def _safe_report(self, value: float, step: float) -> None:
        if step <= self._last_report_step:
            step = self._last_report_step + 1.0
        self.trial.report(float(value), step=step)
        self._last_report_step = step

    def _raise_prune(self, *, reason: str, step: float, log_message: str) -> None:
        self.trial.set_user_attr("forced_prune_reason", reason)
        self.trial.set_user_attr("pruned_timesteps", int(step))
        get_logger(self.log_name, LOG_LEVEL_DEBUG).info(log_message)
        self.is_pruned = True
        raise optuna.exceptions.TrialPruned(f"{reason} at {int(step)} steps")

    def _on_step(self) -> bool:
        # Executa a lógica padrão de avaliação do EvalCallback conforme a frequência
        continue_training = super()._on_step()
        if not continue_training:
            return False

        # Detecta eval pelo n_calls % eval_freq porque evaluations_timesteps só é
        # preenchido pelo SB3 quando log_path != None (passamos None para não gravar em disco).
        mean_reward = getattr(self, "last_mean_reward", None)
        eval_ocorreu = (
            self.eval_freq > 0
            and self.n_calls % self.eval_freq == 0
            and mean_reward is not None
        )
        if eval_ocorreu:
            self._last_eval_count += 1
            mean_reward_value = float(mean_reward)
            self.best_mean_reward = max(self.best_mean_reward, mean_reward_value)
            # Usa num_timesteps do modelo como passo para coerência com o treinamento
            step_value = int(getattr(self.model, "num_timesteps", self.n_calls))
            self.last_eval_step = step_value
            self.eval_history.append((step_value, mean_reward_value))

            episode_summaries = self._collect_episode_metrics()
            aggregated: Dict[str, Any] = {}
            if episode_summaries:
                record = {
                    "timesteps": step_value,
                    "mean_reward": mean_reward_value,
                    "episodes": episode_summaries,
                }
                aggregated = self._aggregate_episode_metrics(episode_summaries)
                if aggregated:
                    record.update(aggregated)
                self.financial_history.append(record)
                self.trial.set_user_attr("financial_history", self.financial_history)
                self.trial.set_user_attr(f"financial_metrics_step_{step_value}", record)

                log_payload = {
                    "steps": step_value,
                    "mean_reward": mean_reward_value,
                    "num_episodes": len(episode_summaries),
                }
                log_payload.update(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in {"timesteps", "mean_reward", "episodes"}
                    }
                )
                get_logger(self.log_name, LOG_LEVEL_DEBUG).info(
                    "[Optuna Eval] %s", log_payload
                )
            else:
                get_logger(self.log_name, LOG_LEVEL_DEBUG).info(
                    "[Optuna Eval] steps=%s mean_reward=%.2f (sem metricas financeiras - nenhum episodio finalizado)",
                    step_value,
                    mean_reward_value,
                )

            # Logs financeiros peridicos (PF, Trades, Score) durante o treino
            fin_stats: Dict[str, Any] = {}
            profit_factor = (
                float(aggregated.get("mean_profit_factor", 0.0)) if aggregated else 0.0
            )
            trade_sharpe = (
                float(aggregated.get("mean_trade_sharpe", 0.0)) if aggregated else 0.0
            )
            avg_return_pct = (
                float(aggregated.get("mean_avg_return_pct", 0.0)) if aggregated else 0.0
            )
            max_drawdown_pct = (
                float(aggregated.get("mean_max_drawdown_pct", 0.0))
                if aggregated
                else 0.0
            )
            num_trades = float(aggregated.get("num_trades", 0.0)) if aggregated else 0.0
            if not aggregated:
                try:
                    fin_stats_list = self.eval_env.env_method("get_financial_stats")
                    fin_stats = (
                        fin_stats_list[0]
                        if isinstance(fin_stats_list, list) and fin_stats_list
                        else {}
                    )
                    profit_factor = float(fin_stats.get("profit_factor", profit_factor))
                    trade_sharpe = float(fin_stats.get("trade_sharpe", trade_sharpe))
                    avg_return_pct = float(
                        fin_stats.get("avg_return_pct", avg_return_pct)
                    )
                    max_drawdown_pct = float(
                        fin_stats.get("max_drawdown_pct", max_drawdown_pct)
                    )
                    num_trades = float(fin_stats.get("num_trades", num_trades))
                except Exception:
                    pass
            profit_factor = min(profit_factor, 10.0)
            trade_sharpe = float(np.clip(trade_sharpe, -5.0, 5.0))
            avg_return_pct = float(np.clip(avg_return_pct, -1.0, 1.0))
            max_drawdown_pct = max(
                0.001, float(max_drawdown_pct)
            )  # Evita divisão por zero

            # 🔬 SCORE CIENTÍFICO PARA TRADING RL (Papers 2023-2024)
            # Baseado em: "DRL for Cryptocurrency Trading" (MDPI 2023)
            # e "Deep Reinforcement Learning for Stock Trading" (2025)
            #
            # Componentes:
            # 1. PSR (Probabilistic Sharpe Ratio): probabilidade de Sharpe > 0 (peso 0.4)
            #    Ref: Bailey & Lopez de Prado (2012) - corrige non-normality
            # 2. Calmar Ratio: retorno/drawdown - foco em drawdown (peso 0.3)
            # 3. Profit Factor: lucros/perdas - qualidade dos trades (peso 0.2)
            # 4. Retorno médio: ganho absoluto (peso 0.1)

            # Calmar Ratio aproximado (retorno / max_drawdown)
            calmar_ratio = float(
                np.clip(avg_return_pct / max_drawdown_pct, -10.0, 10.0)
            )

            # 🔬 PSR: Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012)
            # PSR = Prob[Sharpe verdadeiro > 0] usando Edgeworth expansion.
            # Mais robusto que Sharpe bruto — penaliza distribuições não-normais.
            # Se trade_returns disponível, usa fórmula completa c/ skew/kurtosis.
            # Caso contrário, aproximação gaussiana: PSR ≈ Φ(Sharpe × √N_trades).
            psr_value = 0.0
            try:
                trade_returns_raw = fin_stats.get("trade_returns", None)
                if trade_returns_raw is None and hasattr(self.eval_env, "env_method"):
                    tr_list = self.eval_env.env_method("get_trade_returns")
                    if tr_list and tr_list[0] is not None:
                        trade_returns_raw = tr_list[0]
                if trade_returns_raw is not None and len(trade_returns_raw) >= 10:
                    from utils.sharpe_ratio import probabilistic_sharpe_ratio

                    psr_value = probabilistic_sharpe_ratio(
                        float(trade_sharpe),
                        np.asarray(trade_returns_raw, dtype=np.float64),
                    )
                else:
                    # Aproximação normal: SE ≈ 1/√n para Sharpe
                    psr_value = float(
                        stats.norm.cdf(
                            float(trade_sharpe) * np.sqrt(max(num_trades, 1))
                        )
                    )
            except Exception:
                psr_value = float(
                    stats.norm.cdf(float(trade_sharpe) * np.sqrt(max(num_trades, 1)))
                )
            psr_value = float(np.clip(psr_value, 0.0, 1.0))

            # Score combinado (pesos baseados em literatura)
            score = (
                0.40 * psr_value  # PSR: probabilidade de Sharpe positivo
                + 0.30 * calmar_ratio  # Calmar: drawdown-focado
                + 0.20 * max(0.0, profit_factor - 1.0)  # PF > 1 = lucro
                + 0.10 * (avg_return_pct * 100)  # Retorno em %
            )

            # Penalidade suave por poucos trades (mínimo 5, não 20)
            # Papers indicam que qualidade > quantidade
            if num_trades < 5:
                score = score * 0.5  # Reduz 50%, não zera

            get_logger(self.log_name, LOG_LEVEL_DEBUG).info(
                "[Optuna Eval Financials] steps=%s | PF=%.3f | Sharpe(trade)=%.3f | AvgRet=%.4f | MaxDD=%.4f | Trades=%.0f | Score=%.6f",
                step_value,
                profit_factor,
                trade_sharpe,
                avg_return_pct,
                max_drawdown_pct,
                num_trades,
                score,
            )
            # Log de duração dos trades (Scalping vs Surfing)
            avg_duration = 0.0
            min_duration = 0.0
            max_duration = 0.0
            _eval_misalign_rate: float | None = None
            _eval_scalp_rate: float | None = None
            try:
                fin_stats_list = self.eval_env.env_method("get_financial_stats")
                fin_stats = (
                    fin_stats_list[0]
                    if isinstance(fin_stats_list, list) and fin_stats_list
                    else {}
                )
                avg_duration = float(fin_stats.get("avg_trade_duration", 0.0))
                min_duration = float(fin_stats.get("min_trade_duration", 0.0))
                max_duration = float(fin_stats.get("max_trade_duration", 0.0))
                # Tenta obter misalign_rate e scalp_rate se o env os expuser
                _mr = fin_stats.get(
                    "vote_misalign_rate", fin_stats.get("misalign_rate", None)
                )
                if _mr is not None:
                    _eval_misalign_rate = float(_mr)
                _sr = fin_stats.get(
                    "scalp_rate", fin_stats.get("scalp_rate_duration1", None)
                )
                if _sr is not None:
                    _eval_scalp_rate = float(_sr)
            except Exception:
                pass

            if avg_duration > 0:
                trade_style = (
                    "🏄 SURFING"
                    if avg_duration >= 20
                    else ("⚡ SCALPING" if avg_duration < 10 else "📊 MODERATE")
                )
                get_logger(self.log_name, LOG_LEVEL_DEBUG).info(
                    "[Trade Duration] steps=%s | Avg=%.1f | Min=%.0f | Max=%.0f | Style=%s",
                    step_value,
                    avg_duration,
                    min_duration,
                    max_duration,
                    trade_style,
                )

            # Log exit reasons for diagnostics
            try:
                exit_counts_list = (
                    self.eval_env.env_method("get_exit_reason_counts")
                    if hasattr(self.eval_env, "env_method")
                    else [{}]
                )
                exit_counts = {}
                for env_counts in exit_counts_list:
                    if isinstance(env_counts, dict):
                        for reason, count in env_counts.items():
                            exit_counts[reason] = exit_counts.get(reason, 0) + count
                if any(exit_counts.values()):
                    total_exits = sum(exit_counts.values())
                    agent_pct = (
                        (exit_counts.get("agent", 0) / total_exits * 100)
                        if total_exits > 0
                        else 0
                    )
                    sl_pct = (
                        (exit_counts.get("stop_loss", 0) / total_exits * 100)
                        if total_exits > 0
                        else 0
                    )
                    ts_pct = (
                        (exit_counts.get("time_stop", 0) / total_exits * 100)
                        if total_exits > 0
                        else 0
                    )
                    get_logger(self.log_name, LOG_LEVEL_DEBUG).info(
                        "[Exit Reasons] Agent=%.0f%% | SL=%.0f%% | TimeStop=%.0f%% | Other=%.0f%%",
                        agent_pct,
                        sl_pct,
                        ts_pct,
                        100 - agent_pct - sl_pct - ts_pct,
                    )
            except Exception:
                pass

            # --- Exibe métricas do eval na tela (tqdm-safe) ---
            _eval_num = self._last_eval_count
            _total_evals_str = f"/{self._total_evals}" if self._total_evals > 0 else ""
            # Usa atributos cacheados no ClippedSAC.train() — name_to_value é limpo apos dump()
            _actor_loss = float(getattr(self.model, "_last_actor_loss", float("nan")))
            _ent_coef = float(getattr(self.model, "_last_ent_coef", float("nan")))
            _critic_loss = float(getattr(self.model, "_last_critic_loss", float("nan")))
            _pf_str = f"{profit_factor:.3f}" if profit_factor else "n/a"
            _sharpe_str = f"{trade_sharpe:.3f}" if trade_sharpe else "n/a"
            _trades_str = f"{int(num_trades)}"
            _actor_str = f"{_actor_loss:.4f}" if not np.isnan(_actor_loss) else "n/a"
            _ent_str = f"{_ent_coef:.4f}" if not np.isnan(_ent_coef) else "n/a"
            _eval_line = (
                f"  [Trial {self.trial.number} | Eval {_eval_num}{_total_evals_str} | step {step_value}] "
                f"reward={mean_reward_value:.2f} (best={self.best_mean_reward:.2f}) | "
                f"PF={_pf_str} | Sharpe={_sharpe_str} | Trades={_trades_str} | "
                f"actor_loss={_actor_str} | ent_coef={_ent_str}"
            )
            try:
                import tqdm as _tqdm_mod

                _tqdm_mod.tqdm.write(_eval_line)
            except Exception:
                print(_eval_line, flush=True)

            # --- Registra no LearningMonitor e exibe dashboard ---
            if self._learning_monitor is not None:
                # ⚠️ Computa std em bloco isolado — log_std pode ser nn.Linear (use_sde=False)
                # ou nn.Parameter (use_sde=True), e falhar não deve impedir o monitor de rodar.
                _std_val: float | None = None
                try:
                    if (
                        self.model
                        and hasattr(self.model, "actor")
                        and self.model.actor is not None
                    ):
                        import torch as _torch

                        _log_std_attr = getattr(self.model.actor, "log_std", None)
                        if _log_std_attr is not None:
                            with _torch.no_grad():
                                if hasattr(_log_std_attr, "data"):
                                    # nn.Parameter (use_sde=True)
                                    _std_val = float(
                                        _torch.exp(_log_std_attr.data).mean().item()
                                    )
                                elif hasattr(_log_std_attr, "weight"):
                                    # nn.Linear (use_sde=False)
                                    _std_val = float(
                                        _torch.exp(_log_std_attr.weight.data)
                                        .mean()
                                        .item()
                                    )
                except Exception:
                    _std_val = None

                # record_trial_eval e print_health_dashboard sempre executam,
                # independente de _std_val ter falhado.
                try:
                    # Apenas passa métricas financeiras quando há trades (evita poluir deques com zeros)
                    _has_trades = num_trades > 0
                    self._learning_monitor.record_trial_eval(
                        trial_number=self.trial.number,
                        mean_reward=mean_reward_value,
                        timesteps=step_value,
                        actor_loss=_actor_loss if not np.isnan(_actor_loss) else None,
                        critic_loss=_critic_loss
                        if not np.isnan(_critic_loss)
                        else None,
                        ent_coef=_ent_coef if not np.isnan(_ent_coef) else None,
                        std=_std_val,
                        sharpe=float(trade_sharpe) if _has_trades else None,
                        pnl_per_trade=float(avg_return_pct) if _has_trades else None,
                        avg_duration=float(avg_duration)
                        if _has_trades and avg_duration > 0
                        else None,
                        n_trades=int(num_trades) if _has_trades else None,
                        misalign_rate=_eval_misalign_rate,
                        scalp_rate=_eval_scalp_rate,
                    )
                    self._learning_monitor.print_health_dashboard(
                        trial_number=self.trial.number
                    )
                except Exception as _mon_err:
                    import traceback as _tb

                    try:
                        import tqdm as _tqdm_mod

                        _tqdm_mod.tqdm.write(
                            f"  [LearningMonitor] Erro: {_mon_err}\n{_tb.format_exc()}"
                        )
                    except Exception:
                        print(f"  [LearningMonitor] Erro: {_mon_err}", flush=True)

            # ── REPORT AO OPTUNA PRUNER ──
            # Usa o `score` composto (Sharpe+Calmar+PF+Retorno) quando disponível —
            # é a MESMA métrica que o objetivo final do trial retorna. Isso mantém
            # coerência: o pruner descarta trials pelo mesmo critério da seleção final.
            # Se métricas financeiras não estiverem disponíveis ainda, usa mean_reward
            # normalizado como proxy (Henderson et al., 2018).
            if aggregated:
                _pf_rep = float(
                    np.clip(aggregated.get("mean_profit_factor", 0.0), 0.0, 10.0)
                )
                _sh_rep = float(
                    np.clip(aggregated.get("mean_trade_sharpe", 0.0), -5.0, 5.0)
                )
                _ret_rep = float(
                    np.clip(aggregated.get("mean_avg_return_pct", 0.0), -1.0, 1.0)
                )
                _dd_rep = max(
                    0.001, float(aggregated.get("mean_max_drawdown_pct", 0.001))
                )
                _calmar = float(np.clip(_ret_rep / _dd_rep, -10.0, 10.0))
                _nt_rep = float(aggregated.get("num_trades", 0.0))
                _score_rep = (
                    0.40 * _sh_rep
                    + 0.30 * _calmar
                    + 0.20 * max(0.0, _pf_rep - 1.0)
                    + 0.10 * (_ret_rep * 100)
                )
                if _nt_rep < 5:
                    _score_rep *= 0.5
                report_value = float(_score_rep)
            else:
                # Fallback: normaliza mean_reward por step (escala comparável ao score)
                report_value = float(np.clip(mean_reward_value / 7062.0, -3.0, 3.0))
            self._safe_report(report_value, float(step_value))
            if mean_reward_value <= self.mean_reward_floor:
                self._negative_mean_reward_streak += 1
            else:
                self._negative_mean_reward_streak = 0
            if (
                step_value >= self.early_negative_prune_step
                and mean_reward_value <= self.mean_reward_floor
            ):
                self._raise_prune(
                    reason="early_negative_prune",
                    step=float(step_value),
                    log_message="[Optuna Prune] steps=%s pruning imediato por mean_reward negativo (%.4f) apos %s timesteps"
                    % (step_value, mean_reward_value, self.early_negative_prune_step),
                )
            elif (
                step_value >= self.min_force_prune_step
                and self._negative_mean_reward_streak >= self.negative_streak_patience
            ):
                self._raise_prune(
                    reason="mean_reward_floor",
                    step=float(step_value),
                    log_message="[Optuna Prune] steps=%s pruning por mean_reward=%.4f abaixo do piso %.4f (%d avaliacoes negativas consecutivas)"
                    % (
                        step_value,
                        mean_reward_value,
                        self.mean_reward_floor,
                        self._negative_mean_reward_streak,
                    ),
                )
            elif self.trial.should_prune():
                self._raise_prune(
                    reason="optuna_pruner",
                    step=float(step_value),
                    log_message="[Optuna Prune] steps=%s pruning solicitado por Optuna pruner padrao"
                    % step_value,
                )

            total_trades = (
                float(
                    aggregated.get(
                        "total_num_trades", aggregated.get("num_trades", num_trades)
                    )
                )
                if aggregated
                else float(num_trades)
            )
            total_episode_reward = (
                float(aggregated.get("total_episode_reward", 0.0))
                if aggregated
                else 0.0
            )
            episode_count = int(aggregated.get("num_episodes", 0)) if aggregated else 0
            if (
                total_trades >= self.trade_prune_threshold
                and episode_count >= self.min_episode_count_for_prune
            ):
                self._safe_report(total_episode_reward, float(step_value) + 0.5)
                if total_episode_reward <= self.episode_reward_floor:
                    self._raise_prune(
                        reason="episode_reward_floor",
                        step=float(step_value),
                        log_message="[Optuna Prune] steps=%s pruning por recompensa acumulada=%.4f com %s trades (piso %.4f)"
                        % (
                            step_value,
                            total_episode_reward,
                            total_trades,
                            self.episode_reward_floor,
                        ),
                    )
                elif self.trial.should_prune():
                    self._raise_prune(
                        reason="optuna_pruner",
                        step=float(step_value),
                        log_message="[Optuna Prune] steps=%s pruning solicitado por Optuna pruner padrao"
                        % step_value,
                    )

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

    def _aggregate_episode_metrics(
        self, summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not summaries:
            return {}

        def _collect(key: str) -> List[float]:
            return [float(item[key]) for item in summaries if key in item]

        final_net_worth = _collect("final_net_worth")
        total_return_pct = _collect("total_return_pct")
        max_drawdown_pct = _collect("max_drawdown_pct")
        trade_sharpe = _collect("trade_sharpe")
        num_trades = _collect("num_trades")
        profit_factors = _collect("profit_factor")
        avg_returns = _collect("avg_return_pct")
        episode_rewards = _collect("episode_reward")

        aggregated: Dict[str, Any] = {}
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
        if num_trades:
            total_trades = float(np.sum(num_trades))
            aggregated["num_trades"] = total_trades
            aggregated["total_num_trades"] = total_trades
            aggregated["mean_num_trades"] = float(np.mean(num_trades))
        if profit_factors:
            aggregated["mean_profit_factor"] = float(np.mean(profit_factors))
        if avg_returns:
            aggregated["mean_avg_return_pct"] = float(np.mean(avg_returns))
            # [D-A1 FIX] Sortino com semi-desvio canônico: RMSE de min(r, 0) sobre todos retornos.
            # np.std(neg_ret) usava desvio em torno da média dos negativos (não de 0), inflando.
            _all_ret = np.array(avg_returns)
            _downside = np.minimum(_all_ret, 0.0)
            ds = float(np.sqrt(np.mean(_downside**2)))
            if ds > 1e-9:
                aggregated["sortino_ratio"] = float(
                    np.clip(np.mean(avg_returns) / ds, -10, 10)
                )
        if episode_rewards:
            aggregated["mean_episode_reward"] = float(np.mean(episode_rewards))
            aggregated["total_episode_reward"] = float(np.sum(episode_rewards))
        # [CIENTÍFICO] CALMAR RATIO
        if total_return_pct and max_drawdown_pct:
            mret = float(np.mean(total_return_pct))
            wdd = float(np.max(max_drawdown_pct))
            if wdd > 1e-9:
                # [H2 FIX] Removido ×12 arbitrário que assumia episódios mensais.
                # Episódios de trading têm duração variável — anualização deve ser
                # feita no nível agregado com a duração real do episódio, não aqui.
                aggregated["calmar_ratio"] = float(np.clip(mret / wdd, -100, 100))
        aggregated["num_episodes"] = len(summaries)
        return aggregated

    def _on_training_end(self) -> None:
        super()._on_training_end()
