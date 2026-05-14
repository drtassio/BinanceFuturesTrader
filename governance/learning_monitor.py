# -----------------------------------------------------------------------------
# ARQUIVO: governance/learning_monitor.py
# -----------------------------------------------------------------------------

"""
Monitor de Aprendizado (Learning Monitor) — v2.0
=================================================
Rastreia em tempo real se o agente RL esta aprendendo, convergindo
e progredindo para politicas lucrativas.

Inclui CIRCUIT BREAKER: para o bot/trial automaticamente quando detecta
que o agente NAO esta aprendendo, indicando o motivo exato.

Diagnosticos implementados:
  [D1] Policy Collapse      : Detecta queda de reward apos pico (> COLLAPSE_THRESHOLD)
  [D2] Q-Overestimation     : Actor loss < ACTOR_LOSS_CRITICAL (-50) = critic inflado
  [D3] Entropy Collapse     : std < ENTROPY_MIN_STD (0.05) = convergencia prematura
  [D4] Vote Misalignment    : Taxa de VOTE_MISALIGNED > MISALIGN_RATE_MAX (30%)
  [D5] Scalping Regime      : Duracao media < SCALP_DURATION_MIN (3 steps)
  [D6] Convergencia         : Sharpe slope > CONVERGENCE_MIN_SLOPE por N episodios
  [D7] Plateau              : Sharpe slope ~ 0 por PLATEAU_PATIENCE episodios consecutivos
  [D8] Lucratividade        : PnL medio dos ultimos K trades > PROFITABLE_THRESHOLD
  [D9] Trial Degradation    : Reward medio de trial PIOR que threshold minimo
  [D10] Scalping Duration   : Taxa de trades duration=1 > SCALP_RATE_MAX (50%)

Circuit Breaker:
  - Para o trial Optuna se D1+D2 ambos CRITICAL (colapso confirmado)
  - Para o treino final se D8 CRITICAL por mais de ABORT_PATIENCE episodios
  - Loga motivo exato com instrucoes de correcao

Uso:
    monitor = LearningMonitor(config)
    monitor.record_train_step(actor_loss, critic_loss, ent_coef, std)
    monitor.record_episode(mean_reward, sharpe, pnl_per_trade, duration_avg,
                           vote_misalign_rate, n_trades)
    monitor.record_trial_eval(trial_number, mean_reward, timesteps)
    if monitor.should_abort_trial():
        raise optuna.TrialPruned(monitor.get_abort_reason())
    report = monitor.get_report()
    monitor.log_report(logger)
"""

import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger("LearningMonitor")


# ---------------------------------------------------------------------------
# Thresholds diagnosticos
# ---------------------------------------------------------------------------
_COLLAPSE_THRESHOLD      = 0.40   # Queda relativa de reward > 40% do pico → colapso
_ACTOR_LOSS_CRITICAL     = -50.0  # Actor loss abaixo deste valor → Q-overestimation critica
_ACTOR_LOSS_WARNING      = -20.0  # Actor loss abaixo deste valor → aviso
_ENTROPY_MIN_STD         = 0.05   # std abaixo → convergencia prematura
_MISALIGN_RATE_MAX       = 0.30   # 30% misaligned → politica adversarial
_SCALP_DURATION_MIN      = 3.0    # Duracao media < 3 → scalping
_SCALP_RATE_MAX          = 0.50   # > 50% trades duration=1 → scalping sistemico
_CONVERGENCE_MIN_SLOPE   = 0.005  # Inclinacao minima de Sharpe para considerar convergencia
_PLATEAU_PATIENCE        = 20     # Episodios com slope ~ 0 para declarar plateau
_PROFITABLE_MIN_PNL      = 0.0    # PnL medio > 0 para trade lucrativo
_LOOKBACK_EPISODES       = 50     # Janela de episodios para calculos de tendencia
_LOOKBACK_STEPS          = 1000   # Janela de steps de treino para metricas de gradiente
_ABORT_PATIENCE          = 10     # Episodios consecutivos CRITICAL antes de abortar treino final
_TRIAL_MIN_REWARD        = -1500.0  # Reward minimo esperado de um trial (abaixo = abort)


class LearningStopSignal(Exception):
    """Levantada pelo circuit breaker para parar o treinamento imediatamente."""
    def __init__(self, reason: str, diagnostics: Dict[str, Any]):
        self.reason = reason
        self.diagnostics = diagnostics
        super().__init__(f"[LearningMonitor] CIRCUIT BREAKER ATIVADO: {reason}")


class DiagnosticStatus:
    OK       = "OK"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class LearningMonitor:
    """
    Monitor de convergencia e saude do aprendizado para agentes SAC.
    Inclui circuit breaker que para o bot quando o agente nao esta aprendendo.
    Thread-safe. Pode ser usado por multiplos ambientes paralelos.
    """

    def __init__(
        self,
        config=None,
        specialist_name: str = "Agent",
        log_dir: str = "logs/learning_monitor",
        save_json: bool = True,
    ):
        self.specialist_name = specialist_name
        self.log_dir = log_dir
        self.save_json = save_json
        self._lock = threading.Lock()

        # Config overrides
        cfg = config or {}
        self._collapse_threshold    = float(getattr(cfg, 'COLLAPSE_THRESHOLD',    _COLLAPSE_THRESHOLD))
        self._actor_loss_critical   = float(getattr(cfg, 'ACTOR_LOSS_CRITICAL',   _ACTOR_LOSS_CRITICAL))
        self._actor_loss_warning    = float(getattr(cfg, 'ACTOR_LOSS_WARNING',    _ACTOR_LOSS_WARNING))
        self._entropy_min_std       = float(getattr(cfg, 'ENTROPY_MIN_STD',       _ENTROPY_MIN_STD))
        self._misalign_rate_max     = float(getattr(cfg, 'MISALIGN_RATE_MAX',     _MISALIGN_RATE_MAX))
        self._scalp_duration_min    = float(getattr(cfg, 'SCALP_DURATION_MIN',    _SCALP_DURATION_MIN))
        self._scalp_rate_max        = float(getattr(cfg, 'SCALP_RATE_MAX',        _SCALP_RATE_MAX))
        self._convergence_min_slope = float(getattr(cfg, 'CONVERGENCE_MIN_SLOPE', _CONVERGENCE_MIN_SLOPE))
        self._plateau_patience      = int(getattr(cfg,   'PLATEAU_PATIENCE',      _PLATEAU_PATIENCE))
        self._profitable_min_pnl    = float(getattr(cfg, 'PROFITABLE_MIN_PNL',    _PROFITABLE_MIN_PNL))
        self._lookback_episodes     = int(getattr(cfg,   'LOOKBACK_EPISODES',     _LOOKBACK_EPISODES))
        self._lookback_steps        = int(getattr(cfg,   'LOOKBACK_STEPS',        _LOOKBACK_STEPS))
        self._abort_patience        = int(getattr(cfg,   'ABORT_PATIENCE',        _ABORT_PATIENCE))
        self._trial_min_reward      = float(getattr(cfg, 'TRIAL_MIN_REWARD',      _TRIAL_MIN_REWARD))

        # Historico de treino (por gradient step)
        self._actor_losses:  Deque[float] = deque(maxlen=self._lookback_steps)
        self._critic_losses: Deque[float] = deque(maxlen=self._lookback_steps)
        self._ent_coefs:     Deque[float] = deque(maxlen=self._lookback_steps)
        self._stds:          Deque[float] = deque(maxlen=self._lookback_steps)

        # Historico de episodios
        self._rewards:           Deque[float] = deque(maxlen=self._lookback_episodes)
        self._sharpes:           Deque[float] = deque(maxlen=self._lookback_episodes)
        self._pnl_per_trades:    Deque[float] = deque(maxlen=self._lookback_episodes)
        self._duration_avgs:     Deque[float] = deque(maxlen=self._lookback_episodes)
        self._misalign_rates:    Deque[float] = deque(maxlen=self._lookback_episodes)
        self._n_trades_history:  Deque[int]   = deque(maxlen=self._lookback_episodes)
        self._scalp_rates:       Deque[float] = deque(maxlen=self._lookback_episodes)

        # Historico de trials HPO
        self._trial_rewards:     Deque[float] = deque(maxlen=50)
        self._trial_numbers:     Deque[int]   = deque(maxlen=50)
        self._trial_timesteps:   Deque[int]   = deque(maxlen=50)

        # ── [D11] SAC Convergence Speed ────────────────────────────────────
        # Rastreia em quantos episódios o reward supera threshold pela 1ª vez.
        # Convergência rápida com melhores features indica embeddings úteis.
        self._convergence_reward_threshold: float = float(
            getattr(cfg, 'CONVERGENCE_REWARD_THRESHOLD', 50.0)
        )
        self._convergence_episode: Optional[int] = None   # episódio em que convergiu
        self._reward_slope_window: Deque[float] = deque(maxlen=20)

        # ── [D12] Critic Loss Variance ─────────────────────────────────────
        # Alta variância do critic loss = observation space ruidoso / embeddings instáveis.
        # Thresholds: WARNING > 50x média, CRITICAL > 200x média (heurística empírica).
        self._critic_loss_var_warning  = float(getattr(cfg, 'CRITIC_VAR_WARNING',  50.0))
        self._critic_loss_var_critical = float(getattr(cfg, 'CRITIC_VAR_CRITICAL', 200.0))
        self._critic_losses_recent: Deque[float] = deque(maxlen=200)

        # ── [D13] Policy Entropy Evolution ─────────────────────────────────
        # Rastreia trajetória de entropia ao longo do treino.
        # Declínio muito rápido = convergência prematura (exploração insuficiente).
        # Declínio muito lento = política não está especializando.
        self._entropy_history: Deque[Tuple[int, float]] = deque(maxlen=500)
        self._entropy_decay_rate: float = 0.0   # estimado por regressão linear

        # Estado
        self._peak_reward: float = -np.inf
        self._total_episodes: int = 0
        self._total_steps: int = 0
        self._plateau_counter: int = 0
        self._convergence_counter: int = 0
        self._consecutive_critical: int = 0   # para abort patience
        self._start_time: float = time.time()
        self._current_phase: str = "init"     # "hpo" | "training" | "eval"

        # Circuit breaker
        self._circuit_breaker_active: bool = False
        self._abort_reason: str = ""
        self._abort_diagnostics: Dict[str, Any] = {}

        # Ultimo diagnostico gerado
        self._last_report: Optional[Dict[str, Any]] = None

        if self.save_json:
            os.makedirs(self.log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._json_path = os.path.join(
                self.log_dir, f"{specialist_name}_learning_{ts}.jsonl"
            )
        else:
            self._json_path = None

        logger.info(
            "[LearningMonitor] Iniciado para '%s'. JSON: %s",
            self.specialist_name,
            self._json_path or "desabilitado",
        )

    # ------------------------------------------------------------------
    # Pickle support (threading.Lock não é serializável — recria após unpickle)
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove o lock antes de serializar (não é picklável)
        state.pop('_lock', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recria o lock após deserialização
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registro de dados
    # ------------------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        """Define a fase atual: 'hpo', 'training', 'eval'."""
        with self._lock:
            self._current_phase = phase
        logger.info("[LearningMonitor:%s] Fase: %s", self.specialist_name, phase.upper())

    def record_train_step(
        self,
        actor_loss: float,
        critic_loss: float,
        ent_coef: float,
        std: Optional[float] = None,
    ) -> None:
        """Registra metricas de um gradient step do ClippedSAC."""
        with self._lock:
            self._total_steps += 1
            if np.isfinite(actor_loss):
                self._actor_losses.append(actor_loss)
            if np.isfinite(critic_loss):
                self._critic_losses.append(critic_loss)
                self._critic_losses_recent.append(critic_loss)  # [D12]
            if np.isfinite(ent_coef):
                self._ent_coefs.append(ent_coef)
                self._entropy_history.append((self._total_steps, ent_coef))  # [D13]
            if std is not None and np.isfinite(std):
                self._stds.append(std)

    def record_episode(
        self,
        mean_reward: float,
        sharpe: float = 0.0,
        pnl_per_trade: float = 0.0,
        duration_avg: float = 1.0,
        vote_misalign_rate: float = 0.0,
        n_trades: int = 0,
        scalp_rate: float = 0.0,   # fracao de trades com duration=1
    ) -> None:
        """Registra metricas de um episodio completo."""
        with self._lock:
            self._total_episodes += 1
            if np.isfinite(mean_reward):
                self._rewards.append(mean_reward)
                self._peak_reward = max(self._peak_reward, mean_reward)
            if np.isfinite(sharpe):
                self._sharpes.append(sharpe)
            if np.isfinite(pnl_per_trade):
                self._pnl_per_trades.append(pnl_per_trade)
            if np.isfinite(duration_avg):
                self._duration_avgs.append(duration_avg)
            if np.isfinite(vote_misalign_rate):
                self._misalign_rates.append(vote_misalign_rate)
            self._n_trades_history.append(n_trades)
            if np.isfinite(scalp_rate):
                self._scalp_rates.append(scalp_rate)

            # [D11] SAC Convergence Speed — registra 1º episódio acima do threshold
            if (
                self._convergence_episode is None
                and np.isfinite(mean_reward)
                and mean_reward >= self._convergence_reward_threshold
            ):
                self._convergence_episode = self._total_episodes
                logger.info(
                    "[D11] SAC convergiu no episódio %d (reward=%.2f >= threshold=%.2f)",
                    self._convergence_episode, mean_reward,
                    self._convergence_reward_threshold,
                )

            # [D11] Slope de reward nos últimos 20 episódios (velocidade de aprendizado)
            if np.isfinite(mean_reward):
                self._reward_slope_window.append(mean_reward)

        # Avalia circuit breaker apos cada episodio
        self._evaluate_circuit_breaker()

        # Gera e persiste relatorio apos cada episodio
        report = self.get_report()
        if self._json_path:
            self._append_json(report)

    def record_trial_eval(
        self,
        trial_number: int,
        mean_reward: float,
        timesteps: int,
        actor_loss: Optional[float] = None,
        critic_loss: Optional[float] = None,
        ent_coef: Optional[float] = None,
        std: Optional[float] = None,
        # Métricas financeiras do eval (alimentam diagnósticos D5/D6/D8/D10 durante HPO)
        sharpe: Optional[float] = None,
        pnl_per_trade: Optional[float] = None,
        avg_duration: Optional[float] = None,
        n_trades: Optional[int] = None,
        misalign_rate: Optional[float] = None,
        scalp_rate: Optional[float] = None,
    ) -> None:
        """
        Registra resultado de avaliacao de um trial Optuna.
        Chamado apos cada eval_callback dentro do trial.
        """
        with self._lock:
            if np.isfinite(mean_reward):
                self._trial_rewards.append(mean_reward)
                self._trial_numbers.append(trial_number)
                self._trial_timesteps.append(timesteps)
                self._peak_reward = max(self._peak_reward, mean_reward)
            # Atualiza contadores de steps/episodios para o dashboard
            # (durante HPO, cada eval conta como um pseudo-episodio)
            self._total_steps = timesteps
            self._total_episodes += 1
            if actor_loss is not None and np.isfinite(actor_loss):
                self._actor_losses.append(actor_loss)
            if critic_loss is not None and np.isfinite(critic_loss):
                self._critic_losses.append(critic_loss)
            if ent_coef is not None and np.isfinite(ent_coef):
                self._ent_coefs.append(ent_coef)
            if std is not None and np.isfinite(std):
                self._stds.append(std)
            # Métricas financeiras do eval — preenchem deques normalmente vazios no HPO
            if sharpe is not None and np.isfinite(sharpe):
                self._sharpes.append(sharpe)
            if pnl_per_trade is not None and np.isfinite(pnl_per_trade):
                self._pnl_per_trades.append(pnl_per_trade)
            if avg_duration is not None and np.isfinite(avg_duration) and avg_duration > 0:
                self._duration_avgs.append(avg_duration)
            if n_trades is not None and n_trades >= 0:
                self._n_trades_history.append(int(n_trades))
            if misalign_rate is not None and np.isfinite(misalign_rate):
                self._misalign_rates.append(misalign_rate)
            if scalp_rate is not None and np.isfinite(scalp_rate):
                self._scalp_rates.append(scalp_rate)

        logger.info(
            "[LearningMonitor:%s] Trial #%d | steps=%d | reward=%.4f%s",
            self.specialist_name,
            trial_number,
            timesteps,
            mean_reward,
            f" | actor_loss={actor_loss:.2f}" if actor_loss is not None else "",
        )

        # Avalia circuit breaker
        self._evaluate_circuit_breaker()

    # ------------------------------------------------------------------
    # Circuit Breaker
    # ------------------------------------------------------------------

    def _evaluate_circuit_breaker(self) -> None:
        """
        Avalia se o circuit breaker deve ser ativado.
        Chamado automaticamente apos cada episodio e avaliacao de trial.
        """
        if self._circuit_breaker_active:
            return  # ja ativado

        report = self.get_report()
        diags = report["diagnostics"]

        d1 = diags["D1_policy_collapse"]["status"]
        d2 = diags["D2_q_overestimation"]["status"]
        d8 = diags["D8_profitability"]["status"]
        d10 = diags.get("D10_scalping_rate", {}).get("status", DiagnosticStatus.OK)

        # Colapso confirmado: D1 + D2 ambos CRITICAL → abort trial imediatamente
        if d1 == DiagnosticStatus.CRITICAL and d2 == DiagnosticStatus.CRITICAL:
            self._trigger_circuit_breaker(
                reason=(
                    f"POLICY COLLAPSE + Q-OVERESTIMATION simultaneos: "
                    f"{diags['D1_policy_collapse']['message']} | "
                    f"{diags['D2_q_overestimation']['message']}"
                ),
                action="ABORT_TRIAL",
                diagnostics=diags,
            )
            return

        # Lucratividade negativa persistente → abortar treino final
        if d8 == DiagnosticStatus.CRITICAL:
            with self._lock:
                self._consecutive_critical += 1
        else:
            with self._lock:
                self._consecutive_critical = 0

        if self._consecutive_critical >= self._abort_patience:
            self._trigger_circuit_breaker(
                reason=(
                    f"LUCRATIVIDADE NEGATIVA por {self._consecutive_critical} episodios consecutivos: "
                    f"{diags['D8_profitability']['message']}"
                ),
                action="ABORT_TRAINING",
                diagnostics=diags,
            )

    def _trigger_circuit_breaker(
        self, reason: str, action: str, diagnostics: Dict[str, Any]
    ) -> None:
        """Ativa o circuit breaker e loga o motivo detalhado."""
        with self._lock:
            if self._circuit_breaker_active:
                return
            self._circuit_breaker_active = True
            self._abort_reason = reason
            self._abort_diagnostics = diagnostics

        logger.critical(
            "🚨 [LearningMonitor:%s] CIRCUIT BREAKER ATIVADO!\n"
            "   Acao: %s\n"
            "   Motivo: %s\n"
            "   Como corrigir:\n"
            "     → D1 Policy Collapse: aumente timesteps por trial (min 50k), adicione entropy floor\n"
            "     → D2 Q-Overestimation: ative CQL critic clipping (max_grad_norm * 0.5)\n"
            "     → D8 Lucratividade: verifique reward shaping, anti-scalping e VOTE_MISALIGNED\n"
            "     → D10 Scalping: verifique _grace_floor e HARD_MIN_HOLD no trend_specialist.py",
            self.specialist_name,
            action,
            reason,
        )

        # Persiste estado do circuit breaker
        if self._json_path:
            cb_record = {
                "timestamp": datetime.now().isoformat(),
                "event": "CIRCUIT_BREAKER",
                "specialist": self.specialist_name,
                "action": action,
                "reason": reason,
                "diagnostics": {k: v for k, v in diagnostics.items()},
            }
            self._append_json(cb_record)

    @property
    def circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_active

    def should_abort_trial(self) -> bool:
        """
        Retorna True se o trial Optuna deve ser interrompido.
        Verificar ANTES de iniciar cada trial novo.
        """
        if self._circuit_breaker_active:
            return True
        # Verifica se trial atual esta em colapso
        if len(self._actor_losses) >= 20:
            recent_actor = float(np.mean(list(self._actor_losses)[-20:]))
            if recent_actor < self._actor_loss_critical:
                return True
        return False

    def get_abort_reason(self) -> str:
        """Retorna o motivo do abort para ser usado em TrialPruned."""
        return self._abort_reason or "LearningMonitor: circuit breaker ativado sem motivo registrado"

    def raise_if_circuit_breaker(self) -> None:
        """
        Levanta LearningStopSignal se o circuit breaker estiver ativo.
        Usar no loop de treinamento final para parar graciosamente.
        """
        if self._circuit_breaker_active:
            raise LearningStopSignal(
                reason=self._abort_reason,
                diagnostics=self._abort_diagnostics,
            )

    def reset_circuit_breaker(self) -> None:
        """Reseta o circuit breaker (usar com cuidado, apenas após correcao do problema)."""
        with self._lock:
            self._circuit_breaker_active = False
            self._abort_reason = ""
            self._abort_diagnostics = {}
            self._consecutive_critical = 0
        logger.warning("[LearningMonitor:%s] Circuit breaker RESETADO manualmente.", self.specialist_name)

    # ------------------------------------------------------------------
    # Diagnosticos
    # ------------------------------------------------------------------

    def _diag_policy_collapse(self) -> Tuple[str, str]:
        """D1: Detecta policy collapse (reward caiu > COLLAPSE_THRESHOLD do pico)."""
        # Usa rewards de episodios OU de trials (o que houver)
        data = list(self._rewards) if self._rewards else list(self._trial_rewards)
        if len(data) < 5 or self._peak_reward <= -np.inf:
            return DiagnosticStatus.OK, "Dados insuficientes"
        recent_reward = float(np.mean(data[-3:]))
        # Se o pico está dentro dos últimos 3 evals, o agente ainda está subindo — não é colapso
        if max(data[-3:]) >= self._peak_reward * 0.98:
            return DiagnosticStatus.OK, f"Reward recente={recent_reward:.2f} | pico={self._peak_reward:.2f} (em alta)"
        # [D-C5 FIX] Com peak_reward ≤ 0, o denominador original forçava drop_ratio = 0.0,
        # tornando o circuit breaker matematicamente impossível de disparar durante toda a
        # fase inicial do treino (quando rewards são negativos). Exemplo: pico -100, recente -190
        # → queda de 90% mas drop_ratio = 0.0 → sem alarme.
        # Correção: medir piora relativa ao pico usando |peak_reward| como denominador.
        if abs(self._peak_reward) > 1e-9:
            drop_ratio = (self._peak_reward - recent_reward) / abs(self._peak_reward)
        else:
            drop_ratio = 0.0
        if drop_ratio > self._collapse_threshold:
            pct = drop_ratio * 100
            return (
                DiagnosticStatus.CRITICAL,
                f"COLLAPSE: reward caiu {pct:.1f}% do pico "
                f"(pico={self._peak_reward:.2f}, recente={recent_reward:.2f})",
            )
        return DiagnosticStatus.OK, f"Reward recente={recent_reward:.2f} | pico={self._peak_reward:.2f}"

    def _diag_q_overestimation(self) -> Tuple[str, str]:
        """D2: Detecta Q-overestimation (actor_loss muito negativo) ou
        divergência positiva (actor_loss crescendo acima de zero)."""
        if len(self._actor_losses) < 2:
            return DiagnosticStatus.OK, "Dados insuficientes"
        recent_actor = float(np.mean(list(self._actor_losses)[-20:]))
        # --- lado negativo: Q-values inflados (critic superestima) ---
        if recent_actor < self._actor_loss_critical:
            return (
                DiagnosticStatus.CRITICAL,
                f"Q-OVERESTIMATION: actor_loss={recent_actor:.2f} < {self._actor_loss_critical} "
                "(critico: critic inflado, politica otimiza recompensas fantasma)",
            )
        if recent_actor < self._actor_loss_warning:
            return (
                DiagnosticStatus.WARNING,
                f"Actor loss elevado (negativo): {recent_actor:.2f} (warning={self._actor_loss_warning})",
            )
        # --- lado positivo: gradiente divergindo (policy instável) ---
        if recent_actor > 5.0:
            return (
                DiagnosticStatus.CRITICAL,
                f"DIVERGÊNCIA POSITIVA: actor_loss={recent_actor:.2f} > 5.0 "
                "(critico: gradiente explodindo, política instável)",
            )
        if recent_actor > 2.0:
            return (
                DiagnosticStatus.WARNING,
                f"Actor loss positivo alto: {recent_actor:.2f} > 2.0 "
                "(policy gradient divergindo, monitorar)",
            )
        return DiagnosticStatus.OK, f"Actor loss={recent_actor:.2f}"

    def _diag_entropy_collapse(self) -> Tuple[str, str]:
        """D3: Detecta convergencia prematura (std muito baixo)."""
        if len(self._stds) < 2:
            return DiagnosticStatus.OK, "Dados insuficientes"
        recent_std = float(np.mean(list(self._stds)[-20:]))
        if recent_std < self._entropy_min_std:
            return (
                DiagnosticStatus.WARNING,
                f"ENTROPY COLLAPSE: std={recent_std:.4f} < {self._entropy_min_std} "
                "(politica prematuramente deterministica — verificar entropy floor)",
            )
        return DiagnosticStatus.OK, f"std={recent_std:.4f}"

    def _diag_vote_misalignment(self) -> Tuple[str, str]:
        """D4: Detecta politica adversarial ao predictor de tendencia."""
        if len(self._misalign_rates) < 3:
            return DiagnosticStatus.OK, "Dados insuficientes"
        recent_rate = float(np.mean(list(self._misalign_rates)[-5:]))
        if recent_rate > self._misalign_rate_max:
            return (
                DiagnosticStatus.CRITICAL,
                f"VOTE MISALIGNED: {recent_rate*100:.1f}% de trades contra o predictor "
                f"(max={self._misalign_rate_max*100:.0f}%)",
            )
        if recent_rate > self._misalign_rate_max * 0.5:
            return (
                DiagnosticStatus.WARNING,
                f"Misalignment moderado: {recent_rate*100:.1f}%",
            )
        return DiagnosticStatus.OK, f"Misalignment={recent_rate*100:.1f}%"

    def _diag_scalping(self) -> Tuple[str, str]:
        """D5: Detecta comportamento de scalping (duracao media muito curta)."""
        if len(self._duration_avgs) < 1:
            return DiagnosticStatus.OK, "Dados insuficientes"
        recent_dur = float(np.mean(list(self._duration_avgs)[-5:]))
        if recent_dur < self._scalp_duration_min:
            return (
                DiagnosticStatus.WARNING,
                f"SCALPING: duracao media={recent_dur:.1f} steps < {self._scalp_duration_min}",
            )
        return DiagnosticStatus.OK, f"Duracao media={recent_dur:.1f} steps"

    def _diag_convergence(self) -> Tuple[str, str]:
        """D6/D7: Analisa tendencia do Sharpe para detectar convergencia ou plateau."""
        if len(self._sharpes) < 5:
            return DiagnosticStatus.OK, "Dados insuficientes para analise de tendencia"
        arr = np.array(list(self._sharpes))
        x = np.arange(len(arr))
        try:
            slope, _ = np.polyfit(x, arr, 1)
        except Exception:
            return DiagnosticStatus.OK, "Falha no polyfit"

        # Plateau: slope perto de zero
        if abs(slope) < self._convergence_min_slope:
            self._plateau_counter += 1
        else:
            self._plateau_counter = 0

        # Convergencia: slope positivo sustentado
        if slope > self._convergence_min_slope:
            self._convergence_counter += 1
        else:
            self._convergence_counter = 0

        if self._plateau_counter >= self._plateau_patience:
            return (
                DiagnosticStatus.WARNING,
                f"PLATEAU: Sharpe sem progresso por {self._plateau_counter} episodios "
                f"(slope={slope:.5f}) — considerar curriculum stage avancado",
            )
        if self._convergence_counter >= 5:
            return (
                DiagnosticStatus.OK,
                f"CONVERGINDO: slope Sharpe={slope:.5f} positivo por {self._convergence_counter} ep",
            )
        return (
            DiagnosticStatus.OK,
            f"Sharpe slope={slope:.5f} | plateau_counter={self._plateau_counter}",
        )

    def _diag_profitability(self) -> Tuple[str, str]:
        """D8: Verifica lucratividade media dos trades recentes."""
        if len(self._pnl_per_trades) < 1:
            return DiagnosticStatus.OK, "Dados insuficientes"
        recent_pnl = float(np.mean(list(self._pnl_per_trades)[-10:]))
        if recent_pnl > self._profitable_min_pnl:
            return DiagnosticStatus.OK, f"PnL medio por trade={recent_pnl:.4f} (LUCRATIVO)"
        if recent_pnl > self._profitable_min_pnl - 0.005:
            return DiagnosticStatus.WARNING, f"PnL medio marginal={recent_pnl:.4f}"
        return (
            DiagnosticStatus.CRITICAL,
            f"PnL medio NEGATIVO={recent_pnl:.4f} — agente perde dinheiro consistentemente",
        )

    def _diag_trial_degradation(self) -> Tuple[str, str]:
        """D9: Verifica se os trials estao degradando (reward cada vez pior)."""
        if len(self._trial_rewards) < 3:
            return DiagnosticStatus.OK, "Dados insuficientes de trials"
        recent = float(np.mean(list(self._trial_rewards)[-5:]))
        if recent < self._trial_min_reward:
            return (
                DiagnosticStatus.CRITICAL,
                f"TRIAL DEGRADATION: reward medio={recent:.1f} < minimo={self._trial_min_reward}",
            )
        # Verifica tendencia negativa nos trials
        arr = np.array(list(self._trial_rewards))
        x = np.arange(len(arr))
        try:
            slope, _ = np.polyfit(x, arr, 1)
            if slope < -10.0 and len(arr) >= 10:
                return (
                    DiagnosticStatus.WARNING,
                    f"Trials degradando: slope={slope:.2f} (reward deteriora a cada trial)",
                )
        except Exception:
            pass
        return DiagnosticStatus.OK, f"Reward medio trials={recent:.2f}"

    def _diag_scalping_rate(self) -> Tuple[str, str]:
        """D10: Detecta scalping sistemico via taxa de duration=1."""
        if len(self._scalp_rates) < 3:
            return DiagnosticStatus.OK, "Dados insuficientes"
        recent_rate = float(np.mean(list(self._scalp_rates)[-5:]))
        if recent_rate > self._scalp_rate_max:
            return (
                DiagnosticStatus.CRITICAL,
                f"SCALPING SISTEMICO: {recent_rate*100:.1f}% dos trades com duration=1 "
                f"(max={self._scalp_rate_max*100:.0f}%) — HARD_MIN_HOLD nao esta funcionando",
            )
        if recent_rate > self._scalp_rate_max * 0.5:
            return (
                DiagnosticStatus.WARNING,
                f"Scalping parcial: {recent_rate*100:.1f}% trades duration=1",
            )
        return DiagnosticStatus.OK, f"Taxa scalp={recent_rate*100:.1f}%"

    # ------------------------------------------------------------------
    # D11 — SAC Convergence Speed
    # ------------------------------------------------------------------

    def _diag_convergence_speed(self) -> Tuple[str, str, dict]:
        """
        [D11] Velocidade de convergência do SAC.
        Mede slope de reward nos últimos 20 episódios.
        Convergência rápida com bons embeddings = slope alto e estável.
        """
        data: dict = {
            "convergence_episode": self._convergence_episode,
            "reward_slope": None,
        }
        window = list(self._reward_slope_window)
        if len(window) < 5:
            return DiagnosticStatus.OK, "Dados insuficientes para slope", data

        x = np.arange(len(window), dtype=float)
        slope = float(np.polyfit(x, window, 1)[0])
        data["reward_slope"] = round(slope, 4)

        if slope < -2.0:
            return (
                DiagnosticStatus.CRITICAL,
                f"[D11] Reward em queda acelerada (slope={slope:.3f}) — possível degradação pós-mudança de features",
                data,
            )
        if slope < 0.0 and self._total_episodes > 30:
            return (
                DiagnosticStatus.WARNING,
                f"[D11] Reward com tendência negativa (slope={slope:.3f}) nos últimos {len(window)} ep",
                data,
            )
        return DiagnosticStatus.OK, f"[D11] Slope={slope:.3f} ({len(window)} ep)", data

    # ------------------------------------------------------------------
    # D12 — Critic Loss Variance
    # ------------------------------------------------------------------

    def _diag_critic_variance(self) -> Tuple[str, str, dict]:
        """
        [D12] Variância do critic loss.
        Alta variância = observation space ruidoso ou embeddings instáveis.
        Razão var/mean² (coef. de variação²) > threshold = sinal de ruído.
        """
        data: dict = {"critic_cv2": None, "critic_std": None}
        losses = list(self._critic_losses_recent)
        if len(losses) < 20:
            return DiagnosticStatus.OK, "Dados insuficientes para variance", data

        arr = np.array(losses)
        mean_l = float(arr.mean())
        std_l  = float(arr.std())
        if mean_l == 0:
            return DiagnosticStatus.OK, "Critic mean=0", data

        cv2 = (std_l / abs(mean_l)) ** 2   # coeficiente de variação ao quadrado
        data["critic_cv2"] = round(cv2, 4)
        data["critic_std"] = round(std_l, 4)

        if cv2 > self._critic_loss_var_critical:
            return (
                DiagnosticStatus.CRITICAL,
                f"[D12] Critic CV²={cv2:.1f} >> {self._critic_loss_var_critical} — "
                f"observation space muito ruidoso (verificar embeddings CVAE)",
                data,
            )
        if cv2 > self._critic_loss_var_warning:
            return (
                DiagnosticStatus.WARNING,
                f"[D12] Critic CV²={cv2:.1f} > {self._critic_loss_var_warning} — "
                f"variância elevada (monitorar temporal_smoothness do CVAE)",
                data,
            )
        return DiagnosticStatus.OK, f"[D12] Critic CV²={cv2:.3f} (OK)", data

    # ------------------------------------------------------------------
    # D13 — Policy Entropy Evolution
    # ------------------------------------------------------------------

    def _diag_entropy_evolution(self) -> Tuple[str, str, dict]:
        """
        [D13] Evolução da entropia da política ao longo do treino.
        Declínio muito rápido = convergência prematura (exploração insuficiente).
        Declínio muito lento = política não especializando.
        Estimado via slope linear de ent_coef nos últimos 200 steps.
        """
        data: dict = {"entropy_slope_per_1k": None, "current_entropy": None}
        history = list(self._entropy_history)  # lista de (step, ent_coef)
        if len(history) < 50:
            return DiagnosticStatus.OK, "Dados insuficientes para entropy trend", data

        steps   = np.array([h[0] for h in history], dtype=float)
        entropies = np.array([h[1] for h in history], dtype=float)

        # Normaliza steps para escala 1k para slope legível
        slope_per_1k = float(np.polyfit(steps / 1000.0, entropies, 1)[0])
        current_ent  = float(entropies[-1])

        data["entropy_slope_per_1k"] = round(slope_per_1k, 6)
        data["current_entropy"] = round(current_ent, 6)

        # Declínio rápido demais: slope < -0.05 por 1k steps no início do treino
        if slope_per_1k < -0.10 and self._total_steps < 50_000:
            return (
                DiagnosticStatus.WARNING,
                f"[D13] Entropia caindo muito rápido (slope={slope_per_1k:.4f}/1k) — "
                f"risco de convergência prematura (considerar aumento de ent_coef inicial)",
                data,
            )
        # Entropia estagnada após treino longo: política não especializa
        if abs(slope_per_1k) < 1e-5 and self._total_steps > 100_000 and current_ent > 0.5:
            return (
                DiagnosticStatus.WARNING,
                f"[D13] Entropia estagnada em {current_ent:.4f} após {self._total_steps} steps — "
                f"política pode não estar especializando (verificar reward signal)",
                data,
            )
        return (
            DiagnosticStatus.OK,
            f"[D13] Entropy={current_ent:.4f} slope={slope_per_1k:.5f}/1k",
            data,
        )

    # ------------------------------------------------------------------
    # Relatorio
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        """Gera relatorio completo de saude do aprendizado."""
        with self._lock:
            elapsed = time.time() - self._start_time
            fps = self._total_steps / max(elapsed, 1)

            # Executa todos os diagnosticos
            d1_status,  d1_msg  = self._diag_policy_collapse()
            d2_status,  d2_msg  = self._diag_q_overestimation()
            d3_status,  d3_msg  = self._diag_entropy_collapse()
            d4_status,  d4_msg  = self._diag_vote_misalignment()
            d5_status,  d5_msg  = self._diag_scalping()
            d6_status,  d6_msg  = self._diag_convergence()
            d8_status,  d8_msg  = self._diag_profitability()
            d9_status,  d9_msg  = self._diag_trial_degradation()
            d10_status, d10_msg = self._diag_scalping_rate()
            d11_status, d11_msg, d11_data = self._diag_convergence_speed()
            d12_status, d12_msg, d12_data = self._diag_critic_variance()
            d13_status, d13_msg, d13_data = self._diag_entropy_evolution()

            # Status global = pior status entre os diagnosticos
            all_statuses = [d1_status, d2_status, d3_status, d4_status,
                            d5_status, d6_status, d8_status, d9_status, d10_status,
                            d11_status, d12_status, d13_status]
            if DiagnosticStatus.CRITICAL in all_statuses:
                global_status = DiagnosticStatus.CRITICAL
            elif DiagnosticStatus.WARNING in all_statuses:
                global_status = DiagnosticStatus.WARNING
            else:
                global_status = DiagnosticStatus.OK

            # Metricas resumidas
            # Metricas resumidas
            # Fallback para trial rewards durante HPO (record_episode() nunca é chamado no HPO)
            recent_rewards  = list(self._rewards)[-10:]      if self._rewards      else list(self._trial_rewards)[-10:]
            recent_trials   = list(self._trial_rewards)[-5:] if self._trial_rewards else []
            recent_sharpes  = list(self._sharpes)[-10:]      if self._sharpes      else []
            recent_actor    = list(self._actor_losses)[-20:] if self._actor_losses  else []
            recent_critic   = list(self._critic_losses)[-20:] if self._critic_losses else []
            recent_ent      = list(self._ent_coefs)[-20:]    if self._ent_coefs    else []
            recent_stds     = list(self._stds)[-20:]         if self._stds         else []

            report = {
                "timestamp": datetime.now().isoformat(),
                "specialist": self.specialist_name,
                "phase": self._current_phase,
                "global_status": global_status,
                "circuit_breaker": {
                    "active": self._circuit_breaker_active,
                    "reason": self._abort_reason if self._circuit_breaker_active else "",
                },
                "totals": {
                    "episodes": self._total_episodes,
                    "steps": self._total_steps,
                    "elapsed_s": round(elapsed, 1),
                    "fps": round(fps, 1),
                    "trials_recorded": len(self._trial_numbers),
                },
                "metrics": {
                    "mean_reward_recent":   round(float(np.mean(recent_rewards)), 4) if recent_rewards else None,
                    "mean_reward_trials":   round(float(np.mean(recent_trials)), 4)  if recent_trials  else None,
                    "peak_reward":          round(self._peak_reward, 4) if np.isfinite(self._peak_reward) else None,
                    "mean_sharpe_recent":   round(float(np.mean(recent_sharpes)), 4) if recent_sharpes else None,
                    "mean_actor_loss":      round(float(np.mean(recent_actor)), 4)   if recent_actor   else None,
                    "mean_critic_loss":     round(float(np.mean(recent_critic)), 4)  if recent_critic  else None,
                    "mean_ent_coef":        round(float(np.mean(recent_ent)), 4)     if recent_ent     else None,
                    "mean_std":             round(float(np.mean(recent_stds)), 4)    if recent_stds    else None,
                    "mean_duration_avg":    round(float(np.mean(list(self._duration_avgs)[-5:])), 2)
                                            if self._duration_avgs else None,
                    "mean_pnl_per_trade":   round(float(np.mean(list(self._pnl_per_trades)[-10:])), 5)
                                            if self._pnl_per_trades else None,
                    "misalign_rate":        round(float(np.mean(list(self._misalign_rates)[-5:])), 4)
                                            if self._misalign_rates else None,
                    "scalp_rate_recent":    round(float(np.mean(list(self._scalp_rates)[-5:])), 4)
                                            if self._scalp_rates else None,
                    "plateau_counter":      self._plateau_counter,
                    "convergence_counter":  self._convergence_counter,
                    "consecutive_critical": self._consecutive_critical,
                },
                "diagnostics": {
                    "D1_policy_collapse":   {"status": d1_status,  "message": d1_msg},
                    "D2_q_overestimation":  {"status": d2_status,  "message": d2_msg},
                    "D3_entropy_collapse":  {"status": d3_status,  "message": d3_msg},
                    "D4_vote_misalignment": {"status": d4_status,  "message": d4_msg},
                    "D5_scalping":          {"status": d5_status,  "message": d5_msg},
                    "D6_convergence":       {"status": d6_status,  "message": d6_msg},
                    "D8_profitability":     {"status": d8_status,  "message": d8_msg},
                    "D9_trial_degradation": {"status": d9_status,  "message": d9_msg},
                    "D10_scalping_rate":    {"status": d10_status, "message": d10_msg},
                    "D11_convergence_speed": {"status": d11_status, "message": d11_msg, **d11_data},
                    "D12_critic_variance":   {"status": d12_status, "message": d12_msg, **d12_data},
                    "D13_entropy_evolution": {"status": d13_status, "message": d13_msg, **d13_data},
                },
            }

            self._last_report = report
            return report

    def log_report(self, log=None, force: bool = False) -> None:
        """
        Loga o ultimo relatorio de forma legivel.
        Se force=True, loga mesmo que nao haja alertas.
        """
        report = self.get_report()
        _log = log or logger
        status = report["global_status"]
        emoji = {"OK": "✅", "WARNING": "⚠️", "CRITICAL": "❌"}.get(status, "🔵")
        cb_flag = " 🚨CIRCUIT_BREAKER_ATIVO" if report["circuit_breaker"]["active"] else ""

        _log.info(
            "%s [LearningMonitor:%s] Fase=%s | Ep=%d | Steps=%d | FPS=%.1f | "
            "Reward=%.2f | Sharpe=%.3f | ActorLoss=%.2f | std=%.4f | Status=%s%s",
            emoji,
            self.specialist_name,
            report["phase"].upper(),
            report["totals"]["episodes"],
            report["totals"]["steps"],
            report["totals"]["fps"],
            report["metrics"]["mean_reward_recent"] or 0.0,
            report["metrics"]["mean_sharpe_recent"] or 0.0,
            report["metrics"]["mean_actor_loss"] or 0.0,
            report["metrics"]["mean_std"] or 0.0,
            status,
            cb_flag,
        )

        # Loga diagnosticos que nao sao OK
        for diag_key, diag_val in report["diagnostics"].items():
            if diag_val["status"] != DiagnosticStatus.OK:
                level = "critical" if diag_val["status"] == DiagnosticStatus.CRITICAL else "warning"
                getattr(_log, level)(
                    "  [%s] %s — %s", diag_key, diag_val["status"], diag_val["message"]
                )

        if report["circuit_breaker"]["active"]:
            _log.critical(
                "  🚨 CIRCUIT BREAKER: %s", report["circuit_breaker"]["reason"]
            )

    def print_health_dashboard(self, trial_number: Optional[int] = None) -> None:
        """
        Imprime dashboard de saúde formatado diretamente no console (stdout).
        Mostra visualmente se o agente está aprendendo de forma saudável.
        """
        try:
            report = self.get_report()
            status = report["global_status"]
            cb = report["circuit_breaker"]
            metrics = report["metrics"]
            diags = report["diagnostics"]

            _STATUS_ICON = {"OK": "✅ SAUDÁVEL", "WARNING": "⚠️  ATENÇÃO", "CRITICAL": "❌ CRÍTICO"}
            _DIAG_ICON   = {"OK": "✅", "WARNING": "⚠️ ", "CRITICAL": "❌"}

            reward   = metrics["mean_reward_recent"] if metrics.get("mean_reward_recent") is not None else (metrics.get("mean_reward_trials") or 0.0)
            sharpe   = metrics.get("mean_sharpe_recent")   # None quando sem trades
            aloss    = metrics["mean_actor_loss"]     if metrics.get("mean_actor_loss")     is not None else 0.0
            closs    = metrics["mean_critic_loss"]    if metrics.get("mean_critic_loss")    is not None else 0.0
            ent_coef = metrics["mean_ent_coef"]       if metrics.get("mean_ent_coef")       is not None else 0.0
            std_val  = metrics["mean_std"]            if metrics.get("mean_std")            is not None else 0.0
            eps      = report["totals"].get("episodes", 0)
            steps    = report["totals"].get("steps", 0)
            phase    = report.get("phase", "?").upper()
            name     = self.specialist_name

            _sharpe_str = f"{sharpe:+.4f}   {'📈' if sharpe > 0 else '📉'}" if sharpe is not None else "n/a   (sem trades)"
            trial_str = f" | Trial {trial_number}" if trial_number is not None else ""
            sep = "═" * 70

            lines = [
                f"\n{sep}",
                f"  🤖 LEARNING MONITOR — {name} | Fase: {phase}{trial_str}",
                f"  Status: {_STATUS_ICON.get(status, status)}{'  🚨 CIRCUIT BREAKER ATIVO!' if cb['active'] else ''}",
                sep,
                f"  📊 MÉTRICAS",
                f"     Reward  (recente): {reward:+.4f}   {'📈' if reward > 0 else '📉'}",
                f"     Sharpe  (recente): {_sharpe_str}",
                f"     Actor Loss       : {aloss:+.4f}   {'✅' if -5 < aloss < 0 else ('⚠️' if aloss <= 2.0 else '🔴')}",
                f"     Critic Loss      : {closs:+.4f}   {'✅' if closs < 2.0 else '⚠️'}",
                f"     Ent Coef         : {ent_coef:.4f}   {'✅' if ent_coef > 0.05 else '⚠️ baixo'}",
                f"     Std/Entropia     : {std_val:.4f}   {'✅' if std_val > 0.05 else '❌ colapso!'}",
                f"     Episódios        : {eps}   |   Steps: {steps:,}",
                sep,
                f"  🩺 DIAGNÓSTICOS",
            ]

            _DIAG_LABELS = {
                "D1_policy_collapse":   "D1 Colapso de Política",
                "D2_q_overestimation":  "D2 Q Overestimation  ",
                "D3_entropy_collapse":  "D3 Colapso Entropia  ",
                "D4_vote_misalignment": "D4 Vote Misalignment ",
                "D5_scalping":          "D5 Scalping Regime   ",
                "D6_convergence":       "D6 Convergência      ",
                "D7_plateau":           "D7 Plateau           ",
                "D8_profitability":     "D8 Lucratividade     ",
                "D9_trial_degradation": "D9 Trial Degradation ",
                "D10_scalping_rate":    "D10 Taxa Scalping    ",
            }

            for key, label in _DIAG_LABELS.items():
                if key in diags:
                    d = diags[key]
                    icon = _DIAG_ICON.get(d.get("status", "OK"), "❓")
                    msg  = d.get("message", "")
                    lines.append(f"     {icon} {label}: {msg}")

            if cb["active"]:
                lines += [
                    sep,
                    f"  🚨 CIRCUIT BREAKER ATIVADO!",
                    f"     Motivo : {cb.get('reason', 'N/A')}",
                    f"     Ação   : {cb.get('action', 'N/A')}",
                ]

            lines.append(sep + "\n")
            output = "\n".join(lines)
            try:
                import tqdm as _tqdm_mod
                _tqdm_mod.tqdm.write(output)
            except Exception:
                print(output, flush=True)

        except Exception as _dash_err:
            import traceback as _tb
            _err_msg = f"  [LearningMonitor] Erro no dashboard: {_dash_err}\n{_tb.format_exc()}"
            try:
                import tqdm as _tqdm_mod
                _tqdm_mod.tqdm.write(_err_msg)
            except Exception:
                print(_err_msg, flush=True)

    def is_healthy(self) -> bool:
        """Retorna True se nao ha diagnosticos CRITICAL ativos e circuit breaker inativo."""
        if self._circuit_breaker_active:
            return False
        report = self.get_report()
        return report["global_status"] != DiagnosticStatus.CRITICAL

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def _append_json(self, record: Dict[str, Any]) -> None:
        """Adiciona um registro ao arquivo JSONL de forma segura."""
        if not self._json_path:
            return
        try:
            with open(self._json_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[LearningMonitor] Falha ao salvar JSONL: %s", e)

    def save_summary(self) -> Optional[str]:
        """Salva relatorio resumido em arquivo JSON para inspecao manual."""
        report = self.get_report()
        if not self._json_path:
            return None
        summary_path = self._json_path.replace(".jsonl", "_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info("[LearningMonitor] Resumo salvo em: %s", summary_path)
            return summary_path
        except Exception as e:
            logger.warning("[LearningMonitor] Falha ao salvar resumo: %s", e)
            return None

    def reset_trial(self) -> None:
        """Reseta estado intra-trial (rewards, peak, circuit breaker, contadores).
        Preserva métricas financeiras cross-trial (_sharpes, _pnl_per_trades, etc.)."""
        with self._lock:
            self._rewards.clear()
            self._peak_reward = -np.inf
            self._actor_losses.clear()
            self._critic_losses.clear()
            self._ent_coefs.clear()
            self._stds.clear()
            # [FIX] Reseta circuit breaker entre trials — cada trial começa limpo
            self._circuit_breaker_active = False
            self._abort_reason = ""
            self._abort_diagnostics = {}
            self._consecutive_critical = 0
            # [FIX] Reseta contadores de episódio/steps para D6 medir convergência por trial
            self._total_episodes = 0
            self._total_steps = 0
            self._plateau_counter = 0
            self._convergence_counter = 0
            # Preserva: _sharpes, _pnl_per_trades, _duration_avgs, _misalign_rates,
            #           _n_trades_history, _scalp_rates, _trial_rewards — acumulam cross-trial
        logger.info("[LearningMonitor:%s] Reset de trial: rewards/peak/CB limpos, financeiros preservados.", self.specialist_name)

    def reset(self) -> None:
        """Reinicia o monitor para um novo trial/sessao de treino (nao reseta circuit breaker)."""
        with self._lock:
            self._actor_losses.clear()
            self._critic_losses.clear()
            self._ent_coefs.clear()
            self._stds.clear()
            self._rewards.clear()
            self._sharpes.clear()
            self._pnl_per_trades.clear()
            self._duration_avgs.clear()
            self._misalign_rates.clear()
            self._n_trades_history.clear()
            self._scalp_rates.clear()
            self._peak_reward = -np.inf
            self._total_episodes = 0
            self._total_steps = 0
            self._plateau_counter = 0
            self._convergence_counter = 0
            self._start_time = time.time()
            self._last_report = None
            # NÃO reseta: _trial_rewards, _circuit_breaker_active, _consecutive_critical
        logger.info("[LearningMonitor:%s] Monitor de episodios reiniciado.", self.specialist_name)
