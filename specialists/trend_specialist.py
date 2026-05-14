# -----------------------------------------------------------------------------
# ARQUIVO: specialists/trend_specialist.py (VersÃƒÂ£o Final Corrigida - Usando SAC)
# -----------------------------------------------------------------------------
"""

Agente Especialista em TendÃƒÂªncia (Trend Specialist), utilizando o algoritmo SAC
para maior compatibilidade com espaÃƒÂ§os de aÃƒÂ§ÃƒÂ£o contÃƒÂ­nuos.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from stable_baselines3 import SAC as BaseSAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    VecFrameStack,
)
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    CheckpointCallback,
)
from trading.callbacks.optuna_eval_callback import TrialEvalCallback
from trading.callbacks.trade_analysis_callback import TradeAnalysisCallback
from trading.callbacks.early_stopping_callback import EarlyStoppingCallback
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Dict, Any, Optional, Tuple, List, Set
from dataclasses import replace
import optuna
import json
import shutil
from collections import Counter
import gc
import torch
import torch.cuda
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from utils.logger import get_logger, LOG_LEVEL_DEBUG, setup_phase_logger
from config.settings import AIConfig, TradingConfig
from models.trade_schema import Signal, Action, OrderSide
from specialists.scientific_corrections import compute_scientific_reward

# ProfitabilityPredictor removido — sempre passado como None
from feature_engineering.native_indicators import NativeIndicators
from governance.learning_monitor import LearningMonitor

logger = get_logger("TrendSpecialist", LOG_LEVEL_DEBUG)
ANTI_SCALPING_CONFIG_VERSION = 4
ANTI_SCALPING_REQUIRED_KEYS = [
    "min_trade_duration",
    "scalp_penalty",
    "opening_bonus_mult",
    "long_trade_bonus_threshold",
    "long_trade_bonus_max",
    "post_trade_cooldown",
    "max_trades_per_episode",
    "max_trade_duration",
]
# [TB FIX] Caminho absoluto do diretório de TensorBoard — não depende de os.getcwd()
_TREND_SPECIALIST_DIR = os.path.dirname(os.path.abspath(__file__))  # specialists/
_TREND_PROJECT_ROOT = os.path.dirname(_TREND_SPECIALIST_DIR)  # raiz do projeto
TENSORBOARD_LOG_DIR_TREND = os.path.join(
    _TREND_PROJECT_ROOT, "logs", "tensorboard", "TrendSpecialist"
)
os.makedirs(TENSORBOARD_LOG_DIR_TREND, exist_ok=True)
PF_DYNAMIC_FLOOR_FRACTION = 0.05
PF_MAX_DISPLAY = 10.0
PF_RELIABILITY_MIN_TRADES = 5


# ============================================================
# 🚀 GPU OPTIMIZATION: Ativa otimizações de hardware no início
# ============================================================
def _setup_gpu_optimizations():
    """Configura otimizações de GPU para máxima performance."""
    if not torch.cuda.is_available():
        return
    try:
        props = torch.cuda.get_device_properties(0)
        # TF32 suportado apenas em Ampere+ (Compute Capability >= 8.0).
        # Em Pascal (CC 6.1 — GTX 1050 Ti) é no-op silencioso, não há ganho real.
        tf32_supported = props.major >= 8
        torch.backends.cuda.matmul.allow_tf32 = tf32_supported
        torch.backends.cudnn.allow_tf32 = tf32_supported
        # cuDNN benchmark: válido em qualquer GPU CUDA — mantém incondicional
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        gpu_name = props.name
        gpu_mem = props.total_memory / (1024**3)
        tf32_status = (
            "ON (Ampere+)"
            if tf32_supported
            else f"OFF (CC {props.major}.{props.minor}, requer CC>=8.0)"
        )
        logger.info(
            f"[GPU] CUDA ativado! GPU: {gpu_name} | VRAM: {gpu_mem:.1f} GB | "
            f"TF32: {tf32_status} | cuDNN benchmark: ON"
        )
    except Exception as e:
        logger.warning(f"[GPU] Falha ao configurar otimizacoes: {e}")


_setup_gpu_optimizations()


class ClippedSAC(BaseSAC):
    """
    SAC variant that applies gradient clipping on actor and critic optimizers.
    Suporta FP16 Mixed Precision para aceleração na GPU.
    """

    def __init__(
        self,
        *args,
        max_grad_norm: Optional[float] = 1.0,
        use_fp16: bool = False,
        learning_monitor: Optional["LearningMonitor"] = None,
        **kwargs,
    ):
        self.max_grad_norm = None
        if max_grad_norm is not None:
            try:
                max_grad_norm = float(max_grad_norm)
            except (TypeError, ValueError):
                max_grad_norm = 1.0
            if max_grad_norm > 0:
                self.max_grad_norm = max_grad_norm

        # Monitor de convergencia: rastreia actor_loss, std, etc.
        self.learning_monitor: Optional[LearningMonitor] = learning_monitor

        # 🔬 FP16 Mixed Precision: ativado automaticamente se GPU disponível
        # NOTA: SAC com políticas estocásticas pode ter NaN em FP16.
        # Mantemos FP16 desabilitado por padrão (use_fp16=False) para estabilidade.
        # Para ativar: passe use_fp16=True ao construtor.
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        if self.use_fp16:
            # [API FIX] torch.cuda.amp.GradScaler deprecated em favor de torch.amp.GradScaler('cuda', ...)
            from torch.amp import GradScaler

            self.scaler = GradScaler("cuda")
            logger.info("🔬 [GPU] FP16 Mixed Precision ATIVADO no ClippedSAC.")
        else:
            self.scaler = None

        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        # 🔬 FP16 Mixed Precision context (API moderna torch.amp)
        from torch.amp import autocast

        use_amp = self.use_fp16 and self.scaler is not None

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]
            if self.use_sde:
                self.actor.reset_noise()

            # Forward pass com autocast para FP16
            with autocast("cuda", enabled=use_amp):
                actions_pi, log_prob = self.actor.action_log_prob(
                    replay_data.observations
                )
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if (
                    self.ent_coef_optimizer is not None
                    and self.log_ent_coef is not None
                ):
                    ent_coef = torch.exp(self.log_ent_coef.detach())
                    assert isinstance(self.target_entropy, float)
                    ent_coef_loss = -(
                        self.log_ent_coef * (log_prob + self.target_entropy).detach()
                    ).mean()
                    ent_coef_losses.append(ent_coef_loss.detach())
                else:
                    ent_coef = self.ent_coef_tensor
                ent_coefs.append(ent_coef.detach())

            # Entropy coef update (sem scaler - é simples)
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad(set_to_none=True)
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Target Q-values (no_grad)
            with torch.no_grad():
                with autocast("cuda", enabled=use_amp):
                    next_actions, next_log_prob = self.actor.action_log_prob(
                        replay_data.next_observations
                    )
                    next_q_values = torch.cat(
                        self.critic_target(replay_data.next_observations, next_actions),
                        dim=1,
                    )
                    next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(
                        -1, 1
                    )
                    target_q_values = (
                        replay_data.rewards
                        + (1 - replay_data.dones) * self.gamma * next_q_values
                    )

            # Critic loss com autocast
            with autocast("cuda", enabled=use_amp):
                current_q_values = self.critic(
                    replay_data.observations, replay_data.actions
                )
                critic_loss = 0.5 * sum(
                    F.mse_loss(current_q, target_q_values)
                    for current_q in current_q_values
                )
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.detach())

            # Critic backward com scaler
            self.critic.optimizer.zero_grad(set_to_none=True)
            if use_amp:
                self.scaler.scale(critic_loss).backward()
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.critic.optimizer)
                    # [FIX CQL-STYLE] Critic gradient clipping mais agressivo para prevenir Q-overestimation
                    clip_grad_norm_(
                        self.critic.parameters(), max_norm=min(self.max_grad_norm, 0.5)
                    )
                self.scaler.step(self.critic.optimizer)
            else:
                critic_loss.backward()
                if self.max_grad_norm is not None:
                    # [FIX CQL-STYLE] Critic gradient clipping mais agressivo para prevenir Q-overestimation
                    clip_grad_norm_(
                        self.critic.parameters(), max_norm=min(self.max_grad_norm, 0.5)
                    )
                self.critic.optimizer.step()

            # Actor loss com autocast
            with autocast("cuda", enabled=use_amp):
                q_values_pi = torch.cat(
                    self.critic(replay_data.observations, actions_pi), dim=1
                )
                min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.detach())

            # Actor backward com scaler
            self.actor.optimizer.zero_grad(set_to_none=True)
            if use_amp:
                self.scaler.scale(actor_loss).backward()
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.actor.optimizer)
                    clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.scaler.step(self.actor.optimizer)
                self.scaler.update()  # Update scaler uma vez por step
            else:
                actor_loss.backward()
                if self.max_grad_norm is not None:
                    clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()

            # [FIX ENTROPY FLOOR] Previne convergência prematura (std → 0)
            # Referência: Haarnoja et al. 2018 SAC - entropy regularization é crítica
            # NOTA: quando use_sde=True, actor.log_std é nn.Linear (não nn.Parameter),
            # portanto NÃO tem atributo .data diretamente — nesse caso o SDE controla a variância.
            if hasattr(self.actor, "log_std") and self.actor.log_std is not None:
                if isinstance(self.actor.log_std, torch.nn.Parameter):
                    with torch.no_grad():
                        min_log_std = torch.full_like(self.actor.log_std.data, -2.0)
                        self.actor.log_std.data = torch.max(
                            self.actor.log_std.data, min_log_std
                        )
                # use_sde=True: log_std é nn.Linear — entropia controlada pela variância do SDE

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # [PERF] .item() chamado 1x por batch (não gradient_steps vezes) — elimina PCIe syncs desnecessários
        # Cache em atributos para sobreviver ao logger.dump() (que limpa name_to_value)
        self._last_ent_coef = torch.stack(ent_coefs).mean().item()
        self._last_actor_loss = torch.stack(actor_losses).mean().item()
        self._last_critic_loss = torch.stack(critic_losses).mean().item()
        self.logger.record("train/ent_coef", self._last_ent_coef)
        self.logger.record("train/actor_loss", self._last_actor_loss)
        self.logger.record("train/critic_loss", self._last_critic_loss)
        if len(ent_coef_losses) > 0:
            self.logger.record(
                "train/ent_coef_loss", torch.stack(ent_coef_losses).mean().item()
            )

        # [MONITOR] Registra metricas de gradiente no LearningMonitor
        if self.learning_monitor is not None:
            try:
                mean_std: Optional[float] = None
                if hasattr(self.actor, "log_std") and self.actor.log_std is not None:
                    if isinstance(self.actor.log_std, torch.nn.Parameter):
                        with torch.no_grad():
                            mean_std = float(
                                torch.exp(self.actor.log_std.data).mean().item()
                            )
                    elif isinstance(self.actor.log_std, torch.nn.Linear):
                        # use_sde=True: extrai std media do bias da camada log_std
                        with torch.no_grad():
                            _bias = self.actor.log_std.bias
                            if _bias is not None:
                                mean_std = float(torch.exp(_bias).mean().item())
                self.learning_monitor.record_train_step(
                    actor_loss=float(np.mean(actor_losses)),
                    critic_loss=float(np.mean(critic_losses)),
                    ent_coef=float(np.mean(ent_coefs)),
                    std=mean_std,
                )
            except Exception:
                pass


class _TrendNpRow:
    """Wrapper super leve para evitar criacao de pd.Series dentro do step()."""

    __slots__ = ("_vals", "_col_idx")

    def __init__(self, vals: np.ndarray, col_idx: Dict[str, int]):
        self._vals = vals
        self._col_idx = col_idx

    def get(self, col: str, default: Any = 0.0) -> Any:
        i = self._col_idx.get(col, -1)
        if i < 0:
            return default
        v = self._vals[i]
        # if NaN
        if v != v:
            return default
        return v

    def __getitem__(self, col: str) -> Any:
        return self.get(col, 0.0)

    def __contains__(self, col: str) -> bool:
        return col in self._col_idx


class TrendFollowingEnv(gym.Env):
    """

    [VERSÃƒÆ'O DE ELITE]
    Ambiente de RL para treinar o TrendSpecialist com observaÃƒÂ§ÃƒÂµes enriquecidas,
    recompensas estratÃƒÂ©gicas e simulaÃƒÂ§ÃƒÂ£o de custos realistas.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        config: AIConfig,
        profitability_predictor: Optional[Any] = None,
        mode: str = "training",
        feature_columns: Optional[List[str]] = None,
        feature_scaler=None,
        specialist_name: str = "TrendSpecialist",
    ):
        super(TrendFollowingEnv, self).__init__()
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(
                "Ã°Å¸Å¡Â¨ [ERRO ENV] DataFrame de entrada nÃƒÂ£o pode ser vazio."
            )
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                "Ã°Å¸Å¡Â¨ [ERRO ENV] O ÃƒÂ­ndice do DataFrame deve ser do tipo DatetimeIndex."
            )
        self.df = df.copy()
        self.numeric_df = self.df.select_dtypes(include=np.number)
        self.config = config
        self.relaxed_gates = bool(getattr(self.config, "RELAXED_GATES_TREND", False))
        self.phase2_disable_shorts = False
        self.phase2_disable_longs = False
        self.phase2_bias_direction = 0.0
        self.phase2_bias_threshold = 0.0
        self.phase2_follow_prior_sign = False
        self._gate_stats = Counter()
        self.trading_config = TradingConfig()
        self._feature_scaler = feature_scaler
        # <<< CORREÃƒâ€¡ÃƒÆ'O: Torna o ambiente ciente do timeframe >>>
        self.primary_tf = self.trading_config.PRIMARY_TIMEFRAME_TRADING
        self.atr_col = f"atr_{self.primary_tf}"
        ema_candidate = f"ema_trend_{self.primary_tf}"
        if ema_candidate in self.df.columns:
            self.ema_trend_col = ema_candidate
        elif "ema_trend" in self.df.columns:
            self.ema_trend_col = "ema_trend"
        else:
            fallback = next(
                (
                    col
                    for col in self.df.columns
                    if isinstance(col, str) and col.startswith("ema_trend_")
                ),
                None,
            )
            self.ema_trend_col = fallback
            if fallback is None:
                logger.warning(
                    "[ENV ELITE] Coluna de tendencia EMA nao encontrada no dataset. Gating usara valor neutro."
                )
        self.mode = mode
        self.disable_time_stop = (
            True  # Agente decide quando sair, sem limite de tempo fixo
        )
        self.disable_normal_sl = False  # 🔬 AGENT SL ATIVO: sl_mult recebe gradiente e o agente aprende a gerenciar risco
        self.profitability_predictor = profitability_predictor
        self.initial_balance = self.trading_config.INITIAL_CAPITAL
        self.specialist_name = specialist_name

        # [FIX DIVERGÊNCIA] ruin_threshold=0.50 mata episodios em ~1400 steps → replay buffer
        # cheio de ruins → Q-values colapsam → critic_loss diverge (84→267).
        # Em optimization: threshold relaxado para dar mais exploração antes de morrer.
        if mode == "optimization":
            self.ruin_threshold = 0.10  # 10%: só mata com perda de 90%
            self.ruin_buffer_threshold = 0.40  # Penalidade progressiva abaixo de 40%
            self._leverage_cap_hpo = min(
                self.trading_config.MAX_LEVERAGE_PER_TRADE, 2.0
            )
            logger.debug("[ENV ELITE] HPO mode: ruin_threshold=0.10, leverage_cap=2.0x")
        else:
            # [FIX CRÍTICO TRAINING] ruin=0.50 + leverage alto = ~2 steps por episódio!
            # 3998 episódios em 10k steps → replay buffer lixo → divergência
            # Com leverage 5x, queda de 10% no ativo = 50% da carteira → ruin imediato
            # Solução: ruin mais tolerante + leverage cap para episódios longo (~3k steps)
            self.ruin_threshold = (
                0.15  # 15%: só mata com perda de 85% — episódios longos
            )
            self.ruin_buffer_threshold = 0.60  # Penalidade progressiva abaixo de 60%
            self._leverage_cap_hpo = min(
                self.trading_config.MAX_LEVERAGE_PER_TRADE, 3.0
            )
            logger.debug(
                "[ENV ELITE] Training mode: ruin_threshold=0.15, leverage_cap=3.0x"
            )

        # Limite de episÃƒÂ³dio: usar praticamente todo o dataset para aprender ciclos completos de tendÃƒÂªncia
        self.optimization_episode_ratio = 1.0
        self.training_episode_ratio = 1.0
        if mode == "optimization":
            steps = int(len(self.df) * self.optimization_episode_ratio)
        else:
            steps = int(len(self.df) * self.training_episode_ratio)
        target_steps = max(100, steps)
        upper_bound = max(1, len(self.df) - 1)
        self.max_steps = min(upper_bound, target_steps)
        self.action_space = spaces.Box(
            low=np.array(
                [
                    -1.0,
                    self.trading_config.SL_DIST_MULTIPLIER_MIN,
                    self.trading_config.MIN_LEVERAGE_PER_TRADE,
                ]
            ),
            high=np.array(
                [
                    1.0,
                    self.trading_config.SL_DIST_MULTIPLIER_MAX,
                    self._leverage_cap_hpo,
                ]
            ),
            dtype=np.float32,
        )
        # Extra features desejadas do TrendPredictor/Autoencoder
        # qpnl_* removidos: sempre 0.0 (sem gerador real) → neurônios mortos
        self.extra_feature_keys: List[str] = [
            "tp_duration_median",
            "tp_uncertainty",
            "tp_regime_up",
            "tp_regime_down",
            "tp_regime_sideways",
            "sdae_recon_error",
        ]

        # --- INJECT MEMORY FEATURES ---
        # Adiciona features de "memória" (distância para máximas/mínimas)
        # para o agente saber se está no topo ou fundo histórico recente.
        # [P4 FIX] min_periods=1 garante distâncias reais desde a primeira vela
        # do episódio (usa a própria barra como referência quando janela é curta).
        for window in [20, 50]:
            roll_max = self.df["high"].rolling(window, min_periods=1).max()
            roll_min = self.df["low"].rolling(window, min_periods=1).min()
            self.df[f"dist_to_max_{window}"] = (self.df["close"] - roll_max) / (
                roll_max + 1e-9
            )
            self.df[f"dist_to_min_{window}"] = (self.df["close"] - roll_min) / (
                roll_min + 1e-9
            )

            # Adiciona à lista de extras para entrar na observação
            self.extra_feature_keys.append(f"dist_to_max_{window}")
            self.extra_feature_keys.append(f"dist_to_min_{window}")

        # Calcula quantas extras NÃƒÆ'O estÃƒÂ£o em numeric_df (evita duplicaÃƒÂ§ÃƒÂ£o)
        extras_not_in_numeric = [
            k for k in self.extra_feature_keys if k not in self.numeric_df.columns
        ]
        self._num_extras_appended = len(extras_not_in_numeric)
        # DefiniÃƒÂ§ÃƒÂ£o do contrato de features vindo do DF enriquecido (sem lookahead)
        if feature_columns is None:
            # [CUIDADO] filter_autoencoder_features é lento para rodar todo trial
            # Se recebemos via kwargs (cached), usamos direto.
            auto_cols = list(self.numeric_df.columns)
            if "tp_prior_dir" in auto_cols:
                auto_cols.remove("tp_prior_dir")
            from feature_engineering.scientific_data_processor import (
                ScientificDataProcessor,
            )

            processor = ScientificDataProcessor()
            all_filtered = processor.filter_autoencoder_features(auto_cols)
            self.feature_columns: List[str] = self._filter_to_core_features(
                all_filtered
            )
        else:
            # 🚀 [OPTIMIZATION] Reusa colunas filtradas passadas pelo agente
            self.feature_columns = self._filter_to_core_features(feature_columns)
        num_enriched_features = len(self.feature_columns)
        # O shape final ÃƒÂ©: features enriquecidas + extras + estado do agente + tempo + prior
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_enriched_features + self._num_extras_appended + 3 + 4 + 2,),
            dtype=np.float32,
        )

        # [PERFORMANCE] Pre-cache Numpy para evitar df.iloc[] no hot path
        try:
            self._rnp_values = self.df.to_numpy(dtype=np.float32, copy=False)
            self._col_idx_map = {col: i for i, col in enumerate(self.df.columns)}
        except ValueError:
            # Fallback seguro: ignora colunas string (ex: 'regime_name')
            _num_df = self.df.select_dtypes(include=[np.number])
            self._rnp_values = _num_df.to_numpy(
                dtype=np.float32, copy=False, na_value=np.nan
            )
            self._col_idx_map = {col: i for i, col in enumerate(_num_df.columns)}

        self._feature_col_indices = [
            self._col_idx_map[c] for c in self.feature_columns if c in self._col_idx_map
        ]

        self._episode_summaries: List[Dict[str, float]] = []
        self._current_episode_returns: List[float] = []
        self._current_episode_durations: List[int] = []
        self._last_episode_summary: Optional[Dict[str, float]] = None
        self._trades_in_episode: int = 0
        self.post_trade_cooldown_base = (
            0  # ZERO para permitir reversões imediatas (flip)
        )
        self.post_trade_cooldown = 0

        # [v3.1 LOPEZ DE PRADO] Sem limite artificial de trades em nenhum modo.
        # O agente aprende organicamente via Gravity Well + Triple Barrier proporcional.
        # Trades excessivos são punidos pelo entry cost (-0.5) e pela ausência de depth_reward.
        self.mode = mode
        self.max_trades_per_episode = None  # Sem limite — aprendizado orgânico
        self.trades_remaining = float("inf")  # Infinito = sem limite
        self.disable_new_entries = False
        self.max_trade_duration_cap = 120
        self._log_trend_debug = False
        self._trend_debug_level = "debug"
        # Cooldown removido para permitir reversÃƒÂµes imediatas (flip)
        self.trend_alignment_threshold = 0.05
        self.trend_regime_margin = 0.05
        self.long_trade_bonus_threshold = 24
        self.long_trade_bonus_max = 2.5
        self.scalp_penalty_min_duration = (
            20  # Aumentar de 6 para 20 candles (anti-scalping)
        )
        # Thresholds de decisÃƒÂ£o configurÃƒÂ¡veis (otimizaÃƒÂ§ÃƒÂ£o vs. treinamento)
        # [CORREÇÃO CIENTÍFICA] Valores MUITO mais baixos para evitar policy collapse
        self.opt_prior_open_long_threshold = 0.1  # Antes: 0.4
        self.opt_prior_open_short_threshold = -0.1  # Antes: -0.4
        self.opt_base_threshold = 0.05  # Antes: 0.22
        self.opt_min_action_threshold = 0.02  # Antes: 0.22
        self.opt_force_threshold_factor = 0.5  # Antes: 0.8
        self.opt_penalty_cap_per_step = 1.5
        self.opt_warmup_base = 0.1  # Antes: 0.3
        self.opt_warmup_scale = 0.5  # Antes: 0.7
        self.train_prior_open_long_threshold = 0.15  # Antes: 0.5
        self.train_prior_open_short_threshold = -0.15  # Antes: -0.5
        self.train_base_threshold = 0.08  # Antes: 0.3
        self.train_min_action_threshold = 0.03  # Antes: 0.3
        self.train_force_threshold_factor = 0.5  # Antes: 0.8
        self.train_penalty_cap_per_step = 3.0
        self.train_warmup_base = 0.10  # Penalidades começam em 10% no step 0
        self.train_warmup_scale = 0.90  # Crescem até 100% no step 300 (base+scale=1.0)
        self.exploration_warmup_steps = max(1000, int(self.max_steps * 0.10))
        self.exploration_threshold_scale = 0.3  # Antes: 0.6
        self.exploration_force_threshold_factor = 0.3  # Antes: 0.5
        self.exploration_warmup_vote_threshold = 0.02  # Antes: 0.05
        self.exploration_warmup_log_span = 250
        # ConfiguraÃƒÂ§ÃƒÂ£o anti-scalping otimizada (serÃƒÂ¡ definida na Fase 1)
        self.anti_scalping_config = {}
        self._active_grace_period = 0
        self._grace_floor = 10
        self._grace_median = self.scalp_penalty_min_duration
        self._grace_cap = max(90, self.scalp_penalty_min_duration * 2)
        self._atr_pct_baseline = 0.01
        self._vol_ratio_low = 0.6
        self._vol_ratio_high = 2.5
        self._sl_min_base = float(self.trading_config.SL_DIST_MULTIPLIER_MIN)
        self._sl_max_base = float(self.trading_config.SL_DIST_MULTIPLIER_MAX)
        self._sl_volatile_min = float(
            getattr(
                self.trading_config,
                "SL_DIST_MULTIPLIER_MIN_VOLATILE",
                self._sl_min_base,
            )
        )
        self._sl_volatile_max = float(
            getattr(
                self.trading_config,
                "SL_DIST_MULTIPLIER_MAX_VOLATILE",
                self._sl_max_base,
            )
        )
        self.unrealized_reward_scale = float(
            getattr(self.config, "TREND_UNREALIZED_REWARD_SCALE", 6.0)
        )
        self.realized_reward_scale = float(
            getattr(self.config, "TREND_REALIZED_REWARD_SCALE", 40.0)
        )
        self.drawdown_penalty_weight = float(
            getattr(self.config, "TREND_DRAWDOWN_PENALTY_WEIGHT", 0.35)
        )
        self.prior_flip_penalty_weight = float(
            getattr(self.config, "TREND_PRIOR_FLIP_PENALTY_WEIGHT", 0.1)
        )  # [FIX E] Reduzido 0.6→0.1: penalty de 1.32/step dominava rewards de trades lucrativos
        self.prior_flip_penalty_exponent = float(
            getattr(self.config, "TREND_PRIOR_FLIP_PENALTY_EXP", 1.0)
        )
        self.prior_flip_threshold = float(
            getattr(self.config, "TREND_PRIOR_FLIP_THRESHOLD", 0.3)
        )
        self.duration_pressure_weight = float(
            getattr(self.config, "TREND_DURATION_PRESSURE_WEIGHT", 0.35)
        )
        self.duration_pressure_start = int(
            getattr(self.config, "TREND_DURATION_PRESSURE_START", 90)
        )
        self.duration_pressure_exponent = float(
            getattr(self.config, "TREND_DURATION_PRESSURE_EXP", 1.0)
        )

        # [CIENTÍFICO] Return Normalization - estabiliza gradientes (FinRL 2024)
        # Sem normalização, R_t ∈ [-25, +25] causa oscilações com LR=0.0003
        self._reward_running_mean = 0.0
        self._reward_running_var = 1.0
        self._reward_count = 0
        self._reward_norm_eps = 1e-8
        # [CD3 FIX] Clip final aumentado para [-15, +15]. Antes [-10, +10] saturava
        # ~95% dos rewards porque trade_reward (±25) + scientific_reward (±10) +
        # ruin_penalty (-10) podia somar ±45 antes do clip. SAC perdia gradiente
        # de qualidade — trades excelentes e medianos viravam mesmo reward.
        # 15 ainda mantem regularizacao CQL (previne Q-overestimation) sem sufocar
        # o sinal de aprendizado.
        # [CONVEXITY FIX] Asymmetric clip preserves rare convex payoffs.
        # Trend-following: 5% of trades carry 80% of alpha (López de Prado 2018).
        # Symmetric ±15 clips exceptional trades from reward 23→15.
        # Negative side stays bounded (risk management), positive side expanded.
        self._reward_norm_clip = 15.0
        self._reward_norm_clip_neg = -15.0
        self._reward_norm_clip_pos = 25.0
        # Kumar et al. (2020) CQL: clip conservador [-10,10] previne Q-value explosion
        # em crypto (alta volatilidade) sem sufocar sinal de aprendizado.
        # VecNormalize(norm_reward=False) não re-clipa, então esse é o único clip ativo.

        self._init_dynamic_risk_controls()

        # [SCIENTIFIC IMPROVEMENT] Armazenar scaler imutável se fornecido
        if feature_scaler is not None:
            self._feature_scaler = feature_scaler
            logger.debug(
                "[SCALER] Usando scaler externo (já fit no treino) para evitar data leakage."
            )

        # Sistema de cooldown removido (permitir flips imediatos)
        self.reset()
        logger.debug(
            f"[ENV ELITE] TrendFollowingEnv inicializado. Periodo: {self.max_steps} passos (de {len(self.df)} total). Obs Space: {self.observation_space.shape}. Features base: {num_enriched_features}"
        )

    def set_anti_scalping_config(self, config_params: Dict[str, Any]):
        # FASE 1 desativada: nenhuma configuracao anti-scalping aplicada
        try:
            self.scalp_penalty_min_duration = config_params.get(
                "min_trade_duration", self.scalp_penalty_min_duration
            )
            self.long_trade_bonus_threshold = config_params.get(
                "long_trade_bonus_threshold", self.long_trade_bonus_threshold
            )
            self.long_trade_bonus_max = config_params.get(
                "long_trade_bonus_max", self.long_trade_bonus_max
            )
            # Armazena penalidades para uso no _step_training
            self._scalp_penalty = config_params.get("scalp_penalty", 4.0)
            self._opening_bonus_mult = config_params.get("opening_bonus_mult", 0.8)
            if "trend_alignment_threshold" in config_params:
                self.trend_alignment_threshold = float(
                    config_params["trend_alignment_threshold"]
                )
            if "trend_regime_margin" in config_params:
                self.trend_regime_margin = float(config_params["trend_regime_margin"])
            if "base_threshold_opt" in config_params:
                self.opt_base_threshold = float(config_params["base_threshold_opt"])
            if "min_action_threshold_opt" in config_params:
                self.opt_min_action_threshold = float(
                    config_params["min_action_threshold_opt"]
                )
            if "prior_open_long_threshold_opt" in config_params:
                self.opt_prior_open_long_threshold = float(
                    config_params["prior_open_long_threshold_opt"]
                )
            if "prior_open_short_threshold_opt" in config_params:
                self.opt_prior_open_short_threshold = float(
                    config_params["prior_open_short_threshold_opt"]
                )
            if "force_threshold_factor_opt" in config_params:
                self.opt_force_threshold_factor = float(
                    config_params["force_threshold_factor_opt"]
                )
            if "penalty_cap_per_step_opt" in config_params:
                self.opt_penalty_cap_per_step = float(
                    config_params["penalty_cap_per_step_opt"]
                )
            if "warmup_base_opt" in config_params:
                self.opt_warmup_base = float(config_params["warmup_base_opt"])
            if "warmup_scale_opt" in config_params:
                self.opt_warmup_scale = float(config_params["warmup_scale_opt"])
            if "post_trade_cooldown" in config_params:
                try:
                    self.post_trade_cooldown_base = int(
                        config_params["post_trade_cooldown"]
                    )
                except Exception:
                    pass
            if "max_trades_per_episode" in config_params:
                try:
                    self.max_trades_per_episode = max(
                        1, int(config_params["max_trades_per_episode"])
                    )
                    self.trades_remaining = self.max_trades_per_episode
                except Exception:
                    pass
            if "max_trade_duration" in config_params:
                try:
                    self.max_trade_duration_cap = max(
                        self.scalp_penalty_min_duration,
                        int(config_params["max_trade_duration"]),
                    )
                except Exception:
                    pass
            # Cooldown removido
            self._grace_median = max(
                int(self.scalp_penalty_min_duration), self._grace_median
            )
            self._grace_floor = max(5, min(self._grace_floor, self._grace_median))
            self._grace_cap = max(self._grace_cap, int(self._grace_median * 1.5))
            logger.debug(
                f"[FASE 1] ConfiguracaÃƒÂ§ÃƒÂ£o aplicada (sem cooldown): min_duration={self.scalp_penalty_min_duration}"
            )
        except Exception as e:
            logger.error(
                "[FASE 1] Erro ao aplicar configuraÃƒÂ§ÃƒÂ£o: %s" % e, exc_info=True
            )

    def set_phase2_runtime_tweaks(self):
        """Ajustes para Fase 2 (Optuna): desativa time stop e zera cooldown, mantendo carÃªncia anti-scalping."""
        try:
            self.disable_time_stop = True
            self.max_trade_duration_cap = None
            self.post_trade_cooldown_base = 0
            self.post_trade_cooldown = 0
            self.disable_new_entries = False
            try:
                self.phase2_bias_threshold = max(
                    0.0, float(os.getenv("TREND_PHASE2_BIAS_THR", "0.12"))
                )
            except Exception:
                self.phase2_bias_threshold = 0.12
            bias_dir = 0.0
            regime_source = getattr(self, "numeric_df", None)
            if isinstance(regime_source, pd.DataFrame) and not regime_source.empty:
                up_mean = float(
                    regime_source.get("tp_regime_up", pd.Series(0.0)).mean()
                )
                down_mean = float(
                    regime_source.get("tp_regime_down", pd.Series(0.0)).mean()
                )
                ema_series = None
                if (
                    hasattr(self, "ema_trend_col")
                    and self.ema_trend_col in regime_source.columns
                ):
                    ema_series = regime_source[self.ema_trend_col]
                elif "ema_trend" in regime_source.columns:
                    ema_series = regime_source["ema_trend"]
                ema_mean = (
                    float(getattr(ema_series, "mean", lambda: 0.0)())
                    if ema_series is not None
                    else 0.0
                )
                bias_dir = float((up_mean - down_mean) + 0.3 * np.tanh(ema_mean))
            # Fase 2 sem viÃ©s: sempre permitir ambos os lados
            self.phase2_bias_direction = 0.0
            self.phase2_disable_shorts = False
            self.phase2_disable_longs = False
            try:
                self.phase2_grace_cap_steps = max(
                    5, int(os.getenv("TREND_PHASE2_GRACE_CAP", "24"))
                )
            except Exception:
                self.phase2_grace_cap_steps = 24
            # [v3.1 LOPEZ DE PRADO] Limite de trades REMOVIDO — aprendizado orgânico.
            # max_trades_per_episode = None (sem limite) em todos os modos.
            self.phase2_follow_prior_sign = True
            # Tornar Catastrophe SL mais distante durante otimizaÃ§Ã£o
            try:
                self.trading_config.CATASTROPHE_SL_MULTIPLIER = max(
                    getattr(self.trading_config, "CATASTROPHE_SL_MULTIPLIER", 2.0), 3.0
                )
            except Exception:
                pass
            # Deixar filtros mais permissivos na Fase 2 para reduzir ociosidade extrema
            self.opt_prior_open_long_threshold = 0.0
            self.opt_prior_open_short_threshold = 0.0
            self.opt_base_threshold = min(self.opt_base_threshold, 0.18)
            self.opt_min_action_threshold = min(self.opt_min_action_threshold, 0.12)
            self.trend_alignment_threshold = min(self.trend_alignment_threshold, 0.04)
            self.trend_regime_margin = min(self.trend_regime_margin, 0.04)
            # Relaxar duraÃ§Ã£o mÃ­nima anti-scalping na Fase 2 (reduz graÃ§a quase permanente)
            try:
                relax_factor = float(
                    os.getenv("TREND_PHASE2_MIN_DURATION_RELAX", "0.33")
                )
            except Exception:
                relax_factor = 0.33
            try:
                self.scalp_penalty_min_duration = max(
                    5, int(self.scalp_penalty_min_duration * relax_factor)
                )
            except Exception:
                pass
        except Exception:
            pass

    def set_phase3_runtime_tweaks(self):
        """Ajustes para Fase 3 (Treino final): sem cooldown, sem time stop e thresholds mais permissivos."""
        try:
            # Desativar time stop e cooldown para permitir surf de tendÃªncia e reversÃµes rÃ¡pidas
            self.disable_time_stop = True
            self.max_trade_duration_cap = None
            self.post_trade_cooldown_base = 0
            self.post_trade_cooldown = 0
            self.disable_new_entries = False
            # Tornar filtros de TREINO mais permissivos como na Fase 2
            self.train_base_threshold = min(self.train_base_threshold, 0.14)
            self.train_min_action_threshold = min(self.train_min_action_threshold, 0.05)
            # Alinhar filtros de tendÃªncia
            self.trend_alignment_threshold = min(self.trend_alignment_threshold, 0.04)
            self.trend_regime_margin = min(self.trend_regime_margin, 0.04)
        except Exception:
            pass

    def set_reward_shaping_params(self, params: Dict[str, Any]) -> None:
        """Atualiza pesos de recompensa/penalidade em tempo de execuÃ§Ã£o."""
        if not isinstance(params, dict):
            return
        try:
            if "unrealized_reward_scale" in params:
                self.unrealized_reward_scale = float(params["unrealized_reward_scale"])
            if "realized_reward_scale" in params:
                self.realized_reward_scale = float(params["realized_reward_scale"])
            if "drawdown_penalty_weight" in params:
                self.drawdown_penalty_weight = float(params["drawdown_penalty_weight"])
            if "prior_flip_penalty_weight" in params:
                self.prior_flip_penalty_weight = float(
                    params["prior_flip_penalty_weight"]
                )
            if "prior_flip_penalty_exponent" in params:
                self.prior_flip_penalty_exponent = float(
                    params["prior_flip_penalty_exponent"]
                )
            if "prior_flip_threshold" in params:
                self.prior_flip_threshold = float(params["prior_flip_threshold"])
            if "duration_pressure_weight" in params:
                self.duration_pressure_weight = float(
                    params["duration_pressure_weight"]
                )
            if "duration_pressure_start" in params:
                self.duration_pressure_start = int(params["duration_pressure_start"])
            if "duration_pressure_exponent" in params:
                self.duration_pressure_exponent = float(
                    params["duration_pressure_exponent"]
                )
        except Exception as exc:
            logger.debug(
                "[TREND ENV] Falha ao aplicar parametros de recompensa dinamicamente: %s",
                exc,
            )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._episode_reward_accumulator = 0.0
        self._active_grace_period = 0
        self._last_prior_dir_val = 0.0
        # Para episódios curtos, reinicia em um ponto aleatório do dataset
        # Isso permite que o agente veja diferentes períodos de tempo
        if self.max_steps < len(self.df) - 1:
            max_start_idx = len(self.df) - self.max_steps - 1
            self.start_idx = np.random.randint(0, max(1, max_start_idx))
            logger.debug(
                f"[ENV RESET] Episódio curto: max_steps={self.max_steps}, start_idx={self.start_idx}, end_idx={self.start_idx + self.max_steps}"
            )
        elif self.mode == "optimization" and len(self.df) > 500:
            # [FIX IndexError] start_idx aleatório deve garantir que start_idx + max_steps < len(df).
            # Antes: max_offset = len(df)//5 ignorava max_steps → start_idx=821 + max_steps=4109 = 4930 > 4110 → CRASH
            # Agora: safe_max_offset garante que sempre haverá dados suficientes para o episódio completo.
            safe_from_steps = max(0, len(self.df) - self.max_steps - 1)
            safe_max_offset = max(2, min(len(self.df) // 5, safe_from_steps))
            self.start_idx = np.random.randint(0, max(1, safe_max_offset))
            logger.debug(
                f"[ENV RESET] Episódio otimização (random start): start_idx={self.start_idx}, max_steps={self.max_steps}, safe_offset={safe_max_offset}"
            )
        elif self.mode == "training" and len(self.df) > 500:
            # [ANTI-MEMORIZATION] Random start em training mode para evitar single-trajectory bias.
            # Sem isso, critic vê mesma trajetória a cada episódio (start_idx=0 fixo).
            safe_from_steps = max(0, len(self.df) - self.max_steps - 1)
            safe_max_offset = max(2, min(len(self.df) // 10, safe_from_steps))
            self.start_idx = np.random.randint(0, max(1, safe_max_offset))
            logger.debug(
                f"[ENV RESET] Episódio training (random start): start_idx={self.start_idx}, max_steps={self.max_steps}"
            )
        else:
            self.start_idx = 0
            logger.debug(
                f"[ENV RESET] Episódio completo: max_steps={self.max_steps}, start_idx={self.start_idx}"
            )
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.steps_in_position = 0
        self._current_episode_returns = []
        self._current_episode_durations = []
        self._episode_trades_log: list = []  # [MATRIX] detalhes de cada trade do episódio
        self._trades_in_episode = 0
        self.post_trade_cooldown = 0
        self.trades_remaining = (
            self.max_trades_per_episode
            if self.max_trades_per_episode is not None
            else float("inf")
        )
        self.disable_new_entries = False
        self.current_step = 0
        self.current_leverage = 1.0
        self.initial_notional_value = 0.0
        self.last_action_type = 0
        # [METRICS FIX] Preserva o side do último trade fechado para o CleanMetricsCallback
        # poder atribuir corretamente wins/losses a LONG/SHORT (especialmente em especialistas
        # como Ranger que operam ambos os lados — Bull/Bear inferiam pelo nome, Ranger não).
        self.last_closed_trade_side = 0
        self.pnl_since_entry = 0.0
        self.trade_return_history = []
        self._trades_aligned_with_predictor = 0
        self._trades_against_predictor = 0
        self._current_trade_aligned = (
            None  # [FIX] rastreia alinhamento da entrada atual
        )
        # tp_level removido (seguidor de tendÃƒÂªncia nÃƒÂ£o define take profit fixo)
        self.sl_level = None
        self.time_stop_step = None
        # [TRAILING CATASTROPHE SL] Rastreia pico de preço para ratchet do CSL
        self._peak_price_since_entry = None
        self._catastrophe_sl_floor = None
        self.episode_peak_net_worth = self.initial_balance
        self.episode_max_drawdown = 0.0
        self.episode_trades = 0
        self.episode_wins = 0
        self.exit_reason_counts = {
            "agent": 0,
            "stop_loss": 0,
            "catastrophe": 0,
            "time_stop": 0,
            "ruin": 0,
            "other": 0,
        }  # Track exit reasons
        self.regime_counters = {
            "up": {"wins": 0, "trades": 0},
            "down": {"wins": 0, "trades": 0},
            "side": {"wins": 0, "trades": 0},
        }
        self._trend_regime_sum = {"up": 0.0, "down": 0.0, "side": 0.0}
        self._trend_sample_count = 0
        self._ema_trend_accumulator = 0.0
        self._prev_net_worth = self.net_worth
        self._prev_unrealized_return = 0.0
        # Diagnostico de confianca carregada no DF
        try:
            avg_conf = float(self.df["tp_prior_conf"].mean())
            max_conf = float(self.df["tp_prior_conf"].max())
            std_conf = float(self.df["tp_prior_conf"].std())
            avg_dir = (
                float(self.df["tp_prior_dir"].mean())
                if "tp_prior_dir" in self.df.columns
                else 0.0
            )
            std_dir = (
                float(self.df["tp_prior_dir"].std())
                if "tp_prior_dir" in self.df.columns
                else 0.0
            )

            logger.debug(
                f"[CHECK DADOS] Media Confianca no DF: {avg_conf:.4f} | Max Confianca: {max_conf:.4f} | Std: {std_conf:.4f}"
            )

            # [CORREÇÃO CRÍTICA] Detectar fallback silencioso
            if std_conf < 0.01 and abs(avg_conf - 0.5) < 0.01:
                logger.error(
                    "🚨 [CRÍTICO] tp_prior_conf está fixo em ~0.5 (fallback neutro)! "
                    "O agente NÃO CONSEGUIRÁ APRENDER porque ALL trades serão bloqueados. "
                    "Verifique: 1) CryptoRegimeDetector está treinado? 2) regime_confidence está no DataFrame?"
                )
                # Em modo otimização, isso é fatal - lançar exceção
                if self.mode == "optimization":
                    raise ValueError(
                        "tp_prior_conf constante (~0.5) detectado durante otimização. "
                        "Regime detection falhou - corrija antes de otimizar."
                    )

            if std_dir < 0.01:
                # Ranger em mercado lateral: tp_prior_dir constante é ESPERADO
                # (sem tendência direcional = std≈0 é normal). Só loga uma vez
                # por instância para não spammar a cada reset de episódio.
                if not getattr(self, "_tp_prior_dir_warned", False):
                    self._tp_prior_dir_warned = True
                    _is_ranger = (
                        "ranger" in str(getattr(self, "specialist_name", "")).lower()
                    )
                    if _is_ranger:
                        logger.debug(
                            "[Ranger] tp_prior_dir constante (std=%.4f) — esperado "
                            "em mercado lateral (sem viés direcional).",
                            std_dir,
                        )
                    else:
                        logger.warning(
                            "⚠️ [ALERTA] tp_prior_dir quase constante (std=%.4f). "
                            "Agente pode ter dificuldade em decidir direção.",
                            std_dir,
                        )
        except Exception as e:
            # Silencioso: tp_prior_conf é opcional após remoção do ProfitabilityPredictor
            pass
        self._psar = None
        self._psar_af = 0.02
        self._psar_ep = None
        self._psar_trend = 0
        self._ece_records: List[Tuple[float, int]] = []  # (prior_conf, outcome)
        self._last_entry_prior_conf: Optional[float] = None
        # [FIX REWARD #2] Inicializa componentes de breakdown do reward para o log de diagnóstico
        self._last_trade_reward_component: float = 0.0
        self._last_scientific_reward_component: float = 0.0
        if self.max_steps <= 0:
            raise ValueError("Ã°Å¸Å¡Â¨ [ERRO ENV] DataFrame de treino muito pequeno.")

        # BUG N6 FIX: Recalcula o risk controls (ATR e duração) baseado no episódio que acabou de ser sorteado
        self._init_dynamic_risk_controls()

        return self._get_observation(), {}

    def consume_trade_history(self) -> List[float]:
        """
        [SCIENTIFIC FIX] Retorna e limpa o historico de trades recém-fechados.
        Essencial para evitar contagem duplicada em Callbacks que rodam a cada rollout.
        """
        history = list(self.trade_return_history)
        self.trade_return_history = []
        return history

    def set_trend_debug_logging(
        self, enabled: bool = True, level: str = "debug"
    ) -> None:
        self._log_trend_debug = bool(enabled)
        level_lower = str(level).strip().lower()
        if level_lower not in {"debug", "info"}:
            level_lower = "debug"
        self._trend_debug_level = level_lower

    def _normalize_market_features(self, features: np.ndarray) -> np.ndarray:
        """
        [SCIENTIFIC IMPROVEMENT] Normaliza features de mercado usando RobustScaler IMUTÁVEL.
        O scaler deve ser passado já treinado (fit) no dataset de treino para evitar data leakage.
        """
        if not hasattr(self, "_feature_scaler") or self._feature_scaler is None:
            # Se nenhum scaler externo foi fornecido, emite warning e usa passthrough
            if not hasattr(self, "_scaler_warning_shown"):
                logger.info(
                    "[SCALER] Passthrough mode: Normalização externa não detectada (conduzida pelo ScientificDataProcessor). Evitando dupla normalização."
                )
                self._scaler_warning_shown = True
            return features

        try:
            # [SCIENTIFIC FIX] ColumnTransformer exige nomes de colunas se foi treinado com DataFrame
            if (
                isinstance(self._feature_scaler, ColumnTransformer)
                and hasattr(self, "feature_columns")
                and self.feature_columns
            ):
                # Verifica se a dimensão bate
                if len(features) == len(self.feature_columns):
                    features_df = pd.DataFrame(
                        features.reshape(1, -1), columns=self.feature_columns
                    )
                    return self._feature_scaler.transform(features_df).flatten()

            # Fallback para array puro (pode falhar se fit com nomes)
            return self._feature_scaler.transform(features.reshape(1, -1)).flatten()
        except Exception as e:
            # Log apenas o primeiro erro para não poluir
            if not hasattr(self, "_scaler_error_shown"):
                logger.warning(
                    f"[SCALER] Falha ao transformar features: {e}. Usando features brutas."
                )
                self._scaler_error_shown = True
            return features

    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        # Usa features já enriquecidas e normalizadas, sem lookahead
        # [FIX CRITÍCO] Clamp para evitar IndexError quando start_idx + current_step > len(df)
        # Isso ocorre quando o episódio sobrevive até o fim do dataset (ruin_threshold=0.10)
        _raw_idx = self.start_idx + self.current_step
        _safe_idx = min(_raw_idx, len(self.df) - 1)
        row_vals = self._rnp_values[_safe_idx]
        row_all = _TrendNpRow(row_vals, self._col_idx_map)
        market_features_raw = row_vals[self._feature_col_indices]
        # [CORREÇÃO P1] Normalização Global
        market_features = self._normalize_market_features(market_features_raw)
        # [v3.1 ANTI-NaN] Clipa NaN/Inf APÓS normalização — scaler pode retornar NaN
        # em features constantes (std=0) ou extremos. NaN → 0.0, Inf → ±3.0 (≈3σ).
        market_features = np.nan_to_num(
            market_features, nan=0.0, posinf=3.0, neginf=-3.0
        )
        _ts_idx = min(_raw_idx, len(self.numeric_df) - 1)
        timestamp = self.numeric_df.index[_ts_idx]
        hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
        hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
        day_sin = np.sin(2 * np.pi * timestamp.dayofweek / 7)
        day_cos = np.cos(2 * np.pi * timestamp.dayofweek / 7)
        agent_state = np.array(
            [
                np.sign(self.position),
                float(np.clip(self.pnl_since_entry, -1.0, 1.0)),
                min(self.steps_in_position / 100.0, 1.0),
            ],
            dtype=np.float32,
        )
        time_features = np.array(
            [hour_sin, hour_cos, day_sin, day_cos], dtype=np.float32
        )
        # Prior de direção e confiança vindo do TrendPredictor
        prior_dir_val = float(row_all.get("tp_prior_dir", 0.0))
        prior_conf_val_raw = float(row_all.get("tp_prior_conf", 0.0))
        # [SCIENTIFIC FIX] O CryptoRegimeDetector retorna probabilidades [0.0, 1.0].
        # Não dividir por 5.0 como nas versões legadas para não achatar o sinal.
        prior_conf_val = float(np.clip(prior_conf_val_raw, 0.0, 1.0))
        # Extras somente se não estiverem em numeric_df
        extras: List[float] = []
        if self._num_extras_appended:
            for key in self.extra_feature_keys:
                if key in self.numeric_df.columns:
                    continue
                val = row_all.get(key, 0.0)
                try:
                    extras.append(float(val))
                except Exception:
                    extras.append(0.0)

        return np.concatenate(
            [
                market_features,
                np.array(extras, dtype=np.float32),
                agent_state,
                time_features,
                np.array([prior_dir_val, prior_conf_val], dtype=np.float32),
            ]
        )

    def _get_ema_trend_value(self, row: _TrendNpRow) -> float:
        # [FIX #6] Removido código morto de _simulate_slippage que estava colado aqui
        if not getattr(self, "ema_trend_col", None):
            return 0.0
        try:
            return float(row.get(self.ema_trend_col, 0.0))
        except Exception:
            return 0.0

    def _apply_realized_pnl(self, pnl_realized: float) -> None:
        mark_to_market = self.pnl_since_entry * self.initial_notional_value
        if not np.isfinite(mark_to_market):
            mark_to_market = 0.0
        self.net_worth -= mark_to_market
        self.net_worth += float(pnl_realized)
        # [FIX] Floor de liquidação ao fechar posição também
        _margin_minimum = self.initial_balance * 0.001
        if self.net_worth < _margin_minimum:
            self.net_worth = _margin_minimum
        self.episode_peak_net_worth = max(self.episode_peak_net_worth, self.net_worth)
        self.pnl_since_entry = 0.0

    # [v3.1 REFERENCIA] Padrões de features — usado apenas para documentação.
    # O filtro real está em _filter_to_core_features() abaixo.
    _AGENT_CORE_PATTERNS: List[str] = [
        "log_return",
        "hl_range_pct",
        "close_frac",
        "realized_vol",
        "rsi",
        "macd_hist",
        "macd_cross",
        "adx",
        "plus_di",
        "minus_di",
        "bb_width",
        "atr_percentage",
        "bb_squeeze",
        "bb_expansion",
        "rsi_oversold",
        "rsi_overbought",
        "money_flow",
        "volume_pressure",
        "ema_trend",
        "psar_trend",
        "hidden_feature",
        "sdae_recon",
        "regime_conf",
        "tp_prior",
    ]

    def _filter_to_core_features(self, all_features: List[str]) -> List[str]:
        """Reduz features ao conjunto essencial — v3.1 (López de Prado / SAC convergence).

        BUG da v3.0: substring matching × 5 timeframes = 150 features (não 30-45).
        Fix v3.1:
          - hidden_feature_* (autoencoder latents): TODOS preservados.
            Os latents já comprimem informação de todos os timeframes (1m/5m/15m/1h/4h).
          - Meta features (sdae_recon, regime_conf, tp_prior): TODAS preservadas.
          - Indicadores técnicos: apenas _15m (entrada/precisão) + _1h (direção da tendência).
          - ELIMINADOS: _1m, _5m, _4h (redundantes; latents já capturam essa info).
          - ELIMINADOS: williams_r, cci, roc, candlestick patterns (ruído para RL).
          - MANTIDOS: stoch_k, stoch_d (osciladores [0-100] para reversão).
        Alvo: ~50-60 features | obs space ~65-73 com estado do agente.
        """
        # 1. Features sem sufixo de timeframe: latents + meta — SEMPRE incluir
        # [REC 1+4 FIX 2026-05-13] Adicionados:
        #   - "prob_": probabilidades do AE RegimeHead (prob_bull/bear/ranger). Antes
        #     o SAC so via tp_prior_dir (agregado continuo) — perdia distincao entre
        #     "Bull 0.7 / Ranger 0.2" vs "Bull 0.5 / Bear 0.3 / Ranger 0.2".
        #   - "regime_age_norm": tempo (em barras / seq_len) que o regime atual ativo.
        #     Sinaliza ao SAC se latents do AE refletem regime atual (~estavel) ou
        #     regime antigo (~transicao recente — desconfiar do prior).
        #   - "ema_psar_alignment": sintese do conflito EMA vs PSAR (sinal continuo).
        _ALWAYS_PREFIXES = (
            "hidden_feature",
            "prob_",
        )  # prob_bull, prob_bear, prob_ranger
        _ALWAYS_CONTAINS = (
            "sdae_recon",
            "regime_conf",
            "tp_prior",
            "regime_age_norm",  # REC 4
            "ema_psar_alignment",  # REC 3
            "z_score",  # [INFO ASYMMETRY FIX] Used by Comp 2 Entry Precision (Ranger)
        )

        # 2. Timeframes a EXCLUIR das features técnicas
        _EXCLUDED_TF = ("_1m", "_5m", "_4h")

        # 3. [CONTRATO ENXUTO 2026-05-13] WHITELIST EXPLICITA de 36 indicadores tecnicos.
        # Restaurada da versao antiga (test_feature_contract.py:EXPECTED_TECH_FEATURES)
        # que dava obs_dim=78. Versao "promiscua" com substring match permitia que
        # `rsi` capturasse rsi_oversold_*, rsi_overbought_* etc — explodindo de 36
        # para 58 indicadores (curse of dimensionality desnecessario).
        #
        # Mudancas em relacao a versao promiscua:
        #   REMOVIDO (binarias derivadas — SAC aprende sozinho dos continuos):
        #     - rsi_oversold_*, rsi_overbought_*       (deriva de rsi)
        #     - adx_strong_trend_*, adx_weak_trend_*   (deriva de adx)
        #     - bb_squeeze_*, bb_expansion_*           (deriva de bb_width)
        #     - macd_cross_bullish_*, macd_cross_bearish_*  (deriva de macd_hist)
        #     - rsi_overbought_1h_*, etc...
        #   REMOVIDO (duplicacao cross-TF merge):
        #     - log_return_tf_1h, oc_move_pct_tf_1h, hl_range_pct_tf_1h,
        #       log_return_volume_tf_1h  (identicas as variantes _1h)
        #   REMOVIDO (1h sem valor — wicks e vol_100 capturados em 15m + latents):
        #     - hc_wick_upper_1h, lc_wick_lower_1h
        #     - realized_vol_100_1h
        #
        # Reducao: 58 -> 36 indicadores tecnicos = 17 features economizadas.
        _CORE_TECH_WHITELIST = frozenset(
            {
                # log_return / ranges (7)
                "log_return_15m",
                "log_return_volume_15m",
                "hl_range_pct_15m",
                "oc_move_pct_15m",
                "log_return_1h",
                "hl_range_pct_1h",
                "oc_move_pct_1h",
                # fracdiff (4)
                "close_frac_15m",
                "high_frac_15m",
                "low_frac_15m",
                "close_frac_1h",
                # realized vol (3)
                "realized_vol_20_15m",
                "realized_vol_100_15m",
                "realized_vol_20_1h",
                # osciladores continuos (8) — sem variantes binarias
                "rsi_15m",
                "rsi_1h",
                "stoch_k_15m",
                "stoch_d_15m",
                "stoch_k_1h",
                "stoch_d_1h",
                "macd_hist_15m",
                "macd_hist_1h",
                # tendencia continuos (8) — sem strong/weak flags
                "adx_15m",
                "adx_1h",
                "plus_di_15m",
                "minus_di_15m",
                "plus_di_1h",
                "minus_di_1h",
                "ema_trend_15m",
                "ema_trend_1h",
                # volatilidade normalizada (4) — [BUG 6 FIX] bb_width usado pelo reward (Comp 2 + 10)
                "atr_percentage_15m",
                "atr_percentage_1h",
                "bb_width_15m",
                "bb_width_1h",
                # volume (2)
                "cmf_15m",
                "cmf_1h",
                # wicks 15m apenas (2) — 1h capturado pelos latents
                "hc_wick_upper_15m",
                "lc_wick_lower_15m",
            }
        )  # = 38 features tecnicas (era 36; +bb_width_15m e bb_width_1h para alinhar com reward)

        core: List[str] = []
        for f in all_features:
            fl = f.lower()

            # Latents do autoencoder (hidden_feature_0..N) — sem sufixo de timeframe
            if any(fl.startswith(p) for p in _ALWAYS_PREFIXES):
                core.append(f)
                continue

            # Meta features (sdae_recon_error, regime_conf_*, tp_prior_*, regime_age_norm, ema_psar_alignment)
            if any(p in fl for p in _ALWAYS_CONTAINS):
                core.append(f)
                continue

            # Excluir timeframes ruidosos/redundantes
            if any(fl.endswith(tf) or (tf + "_") in fl for tf in _EXCLUDED_TF):
                continue

            # [CONTRATO ENXUTO] WHITELIST EXATA — sem substring match promiscuo
            if f in _CORE_TECH_WHITELIST:
                core.append(f)

        if len(core) < 10:
            logger.warning(
                "[CORE FEATURES] v3.1: filtro retornou apenas %d — usando todas as %d.",
                len(core),
                len(all_features),
            )
            return all_features

        n_latents = sum(1 for f in core if f.lower().startswith("hidden_feature"))
        n_15m = sum(1 for f in core if f.lower().endswith("_15m"))
        n_1h = sum(1 for f in core if f.lower().endswith("_1h"))
        logger.debug(
            "[CORE FEATURES] v3.1: %d -> %d features "
            "(latents=%d | 15m=%d | 1h=%d | meta=%d) "
            "— 1m/5m/4h excluidos",
            len(all_features),
            len(core),
            n_latents,
            n_15m,
            n_1h,
            len(core) - n_latents - n_15m - n_1h,
        )
        return core

    def _calculate_sharpe_reward(self) -> float:
        # Unificar a fonte de dados com o snapshot financeiro: usar retornos do episÃ³dio atual
        source_returns = (
            self._current_episode_returns
            if self._current_episode_returns
            else self.trade_return_history
        )
        if len(source_returns) < 10:
            return 0.0
        recent_returns = np.array(source_returns[-20:])
        std_rr = float(np.std(recent_returns))
        if std_rr < 1e-9:
            return 0.0
        sharpe_ratio = float(np.mean(recent_returns) / (std_rr + 1e-12))
        return float(np.tanh(sharpe_ratio) * self.config.REWARD_SHARPE_RATIO_WEIGHT)

    def _init_dynamic_risk_controls(self) -> None:
        """Calcula valores de refer?ncia din?micos para car?ncia e SL a partir do dataset."""
        # BUG N6 FIX: Calcular baseline sobre o episódio atual, e não sobre o dataset inteiro.
        start_idx = getattr(self, "start_idx", 0)
        end_idx = min(
            len(self.df), start_idx + getattr(self, "max_steps", len(self.df))
        )
        episode_df = self.df.iloc[start_idx:end_idx]

        durations = None
        if "tp_duration_median" in episode_df.columns:
            try:
                durations = pd.to_numeric(
                    episode_df["tp_duration_median"], errors="coerce"
                )
            except Exception:
                durations = None
        if durations is not None:
            valid = durations[np.isfinite(durations)]
            if not valid.empty:
                self._grace_floor = max(5, int(np.nanpercentile(valid, 25)))
                self._grace_median = max(
                    self.scalp_penalty_min_duration, int(np.nanmedian(valid))
                )
                self._grace_cap = max(
                    self._grace_median + 5, int(np.nanpercentile(valid, 75) * 1.5)
                )
                self.scalp_penalty_min_duration = int(self._grace_median)
        if self.atr_col in episode_df.columns and "close" in episode_df.columns:
            try:
                atr_series = pd.to_numeric(episode_df[self.atr_col], errors="coerce")
                close_series = pd.to_numeric(
                    episode_df["close"], errors="coerce"
                ).replace(0.0, np.nan)
                atr_pct_series = (
                    (atr_series / close_series)
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if not atr_pct_series.empty:
                    self._atr_pct_baseline = float(np.nanmedian(atr_pct_series))
            except Exception:
                pass
        self._atr_pct_baseline = max(self._atr_pct_baseline, 1e-3)
        self._active_grace_period = 0

    def _compute_trade_grace_period(
        self,
        atr_pct: float,
        duration_hint: float,
        prior_strength: float,
        trend_bias: float,
    ) -> int:
        base = max(
            self._grace_floor,
            min(
                self._grace_cap,
                int(round(duration_hint)) if duration_hint > 0 else self._grace_median,
            ),
        )
        ratio = atr_pct / (self._atr_pct_baseline + 1e-9)
        vol_adjust = np.clip(ratio, 0.5, 2.5)
        conviction = 1.0 + np.clip(prior_strength, 0.0, 1.0) * 0.5
        bias_bonus = 1.0 + np.clip(trend_bias, 0.0, 1.0) * 0.3
        adjusted = base * conviction * bias_bonus / vol_adjust
        return int(np.clip(round(adjusted), self._grace_floor, self._grace_cap))

    def _dynamic_sl_bounds(self, atr_pct: float) -> Tuple[float, float]:
        ratio = atr_pct / (self._atr_pct_baseline + 1e-9)
        if ratio >= 1.0:
            interp = np.clip((ratio - 1.0) / (self._vol_ratio_high - 1.0), 0.0, 1.0)
            floor = self._sl_min_base + interp * (
                self._sl_volatile_min - self._sl_min_base
            )
            cap = self._sl_max_base + interp * (
                self._sl_volatile_max - self._sl_max_base
            )
        else:
            interp = np.clip(
                (ratio - self._vol_ratio_low) / (1.0 - self._vol_ratio_low), 0.0, 1.0
            )
            floor = (self._sl_min_base * 0.6) + interp * (
                self._sl_min_base - (self._sl_min_base * 0.6)
            )
            cap = (self._sl_max_base * 0.7) + interp * (
                self._sl_max_base - (self._sl_max_base * 0.7)
            )
        floor = float(np.clip(floor, 0.8, self._sl_volatile_min))
        cap = float(np.clip(cap, floor + 0.5, self._sl_volatile_max))
        return floor, cap

    def _compute_catastrophe_multiplier(
        self, atr_pct: float, prior_dir_strength: float, position_side: int
    ) -> float:
        base = float(getattr(self.trading_config, "CATASTROPHE_SL_MULTIPLIER", 3.0))
        ratio = atr_pct / (self._atr_pct_baseline + 1e-9)
        volatility_scale = np.clip(1.0 + (ratio - 1.0) * 0.35, 0.7, 1.8)
        conviction_bonus = 1.0 + np.clip(abs(prior_dir_strength), 0.0, 1.0) * 0.4
        directional_bias = 1.0 + 0.1 * position_side
        return float(
            np.clip(
                base * volatility_scale * conviction_bonus * directional_bias,
                base * 0.7,
                base * 2.5,
            )
        )

    def _simulate_slippage(
        self, price: float, atr: float, action_side: OrderSide
    ) -> float:
        """Simula slippage realista proporcional ao preço.

        [X3 FIX] Correção de bug dimensional: a fórmula original calculava
        `atr * SLIPPAGE_TOLERANCE_PCT`, que com ATR típico de 2% do preço
        resultava em slippage de apenas ~0.004% — 50x menor que o intended 0.2%.
        Isso causava train-serve skew: o execution_engine.py (paper trading) usava
        `price * 0.002` corretamente enquanto o treino RL usava `atr * 0.002`.

        Fórmula corrigida: `price * SLIPPAGE_TOLERANCE_PCT` → 0.2% do preço,
        consistente com o modelo do execution_engine.py.
        """
        slippage_amount = price * self.trading_config.SLIPPAGE_TOLERANCE_PCT
        if action_side == OrderSide.BUY:
            return price + slippage_amount
        else:  # OrderSide.SELL
            return price - slippage_amount

    def _compute_trade_reward(
        self,
        pnl_realized: float,
        trade_return_pct: float,
        duration_steps: int,
        exit_reason: Optional[str],
        atr_pct: float,
    ) -> float:
        """
        [BUG A FIX] Recompensa de fechamento de trade.
        compute_scientific_reward É CHAMADO APENAS em _step_training (linha ~2310).
        Esta função processa APENAS os componentes base de PnL/custo.
        """
        if not np.isfinite(pnl_realized):
            return 0.0

        # [BREAKEVEN PENALTY] Trade com PnL ≈ 0: consumiu capital e tempo sem gerar alpha.
        # Penaliza para incentivar opções claras: ou lucro real, ou não operar.
        # [FIX REWARD #1] Threshold reduzido de 0.01% para 0.005% (abaixo do custo transacional mínimo).
        # Penalidade reduzida de -1.5 para -0.5 para não anular gains legítimos de TPs com return baixo.
        # Trades com 0.01% de return e 5+ steps são positivos — não devem receber -1.5.
        if abs(trade_return_pct) < 0.00005:  # < 0.005% — abaixo do custo de transação
            return -0.5  # Penalidade suave: ainda negativo, mas não destrói o signal

        # [FIX D] Usa trade_return_pct (retorno % sobre margem) em vez de pnl/initial_balance.
        # Fórmula antiga: pnl/$10000*300 → ~0.028 para trade de $10 com 9.36% de retorno (escala inútil).
        # Fórmula nova: trade_return_pct*30 → +2.81 para trade de 9.36% (escala proporcional ao risco real).
        #
        # [FIX F] Amortização de custo de transação via multiplicador de duração contínuo.
        # O custo de transação é FIXO por trade (spread + taxa). Trades curtos carregam esse custo
        # sobre poucos candles; trades longos amortizam o mesmo custo sobre muito mais retorno.
        # sqrt(d/5): duration=1→0.45 | duration=5→1.00 | duration=10→1.41 | duration=20→2.00
        # Isso cria incentivo orgânico para segurar winners, sem nenhum threshold fixo.
        # [IMPROVER FIX] Cap raised from 2.0 to 2.5 to incentivize holding >20 candles.
        # Before: sqrt(20/5)=2.0 hit cap at 20 steps — no extra reward for 50-step winners.
        # Now: sqrt(31/5)=2.5 hits cap at 31 steps — +25% reward for long-duration trades.
        # [CRITICAL FIX 2026-05-13] Multiplier aumentado de 30 -> 80.
        # Ratio shaping/PnL era 6.8x (meta: 0.5-1.5x). Com multiplier=30, trade +3%/20steps
        # gerava reward=+1.80 vs shaping acumulado=+12.2. Com multiplier=80, trade_reward=+4.80
        # e ratio cai para ~1.2x. PnL agora domina o sinal de aprendizado.
        dur_amortization = float(np.clip(np.sqrt(duration_steps / 5.0), 0.3, 2.5))
        # [EXPECTANCY ALIGNMENT] Soft-clamp via tanh preserva gradiente de qualidade.
        # Hard clip [-0.50,+0.50] torna trade +5% identico a +50% — viola expectancy.
        # soft_clamp: 0.5*tanh(ret/0.3) → +5%≈+0.083, +50%≈+0.465 (proporcional).
        soft_clamped_return = 0.5 * np.tanh(trade_return_pct / 0.3)
        return_clip = soft_clamped_return
        reward = return_clip * 80.0 * dur_amortization

        # Custo de transação já embutido no pnl_realized pela trading engine

        # [BUG A FIX] Não chamar compute_scientific_reward aqui.
        # _step_training já faz esta chamada após fechar o trade.
        # Chamar aqui causava DOUBLE-COUNTING de todas as penalidades científicas.

        return float(np.clip(reward, -100.0, 100.0))

    def _build_financial_snapshot(
        self, returns: List[float], durations: List[int]
    ) -> Dict[str, float]:
        """Calcula mÃƒÂ©tricas financeiras robustas a partir da lista de retornos por trade.
        - PF robusto: evita explosÃƒÂ£o quando perdas ~0 usando denominador suavizado proporcional ao lucro.
        - Sinaliza confiabilidade do PF.
        """
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        total_trades = len(returns)
        gross_profit = float(np.sum(wins)) if wins else 0.0
        gross_loss_abs = float(np.sum([-r for r in losses])) if losses else 0.0
        smooth_den = max(1e-6, gross_loss_abs)
        dynamic_floor = PF_DYNAMIC_FLOOR_FRACTION * max(0.0, gross_profit)
        denom = max(smooth_den, dynamic_floor, 1e-6)
        profit_factor_raw = gross_profit / denom if denom > 0 else 0.0
        profit_factor = float(np.clip(profit_factor_raw, 0.0, PF_MAX_DISPLAY))
        loss_ratio = gross_loss_abs / (gross_profit + 1e-9) if gross_profit > 0 else 0.0
        pf_reliable = bool(
            total_trades >= PF_RELIABILITY_MIN_TRADES
            and loss_ratio >= PF_DYNAMIC_FLOOR_FRACTION
        )
        cumulative_return = float(np.sum(returns)) if returns else 0.0
        avg_return_pct = float(np.mean(returns)) if returns else 0.0
        if len(returns) >= 2 and np.std(returns) > 1e-9:
            trade_sharpe = float(np.mean(returns) / (np.std(returns) + 1e-12))
        else:
            trade_sharpe = 0.0

        # BUG N1 FIX: Cap trade_sharpe if we have very few trades
        if total_trades < PF_RELIABILITY_MIN_TRADES:
            trade_sharpe = min(trade_sharpe, 0.5)
        win_rate_pct = float(len(wins) / total_trades * 100.0) if total_trades else 0.0
        max_dd = (
            float(self.episode_max_drawdown)
            if hasattr(self, "episode_max_drawdown")
            else 0.0
        )
        avg_duration = float(np.mean(durations)) if durations else 0.0
        total_regime_trades = 0.0
        regime_trade_pct = {"up": 0.0, "down": 0.0, "side": 0.0}
        if isinstance(getattr(self, "regime_counters", None), dict):
            total_regime_trades = float(
                sum(float(v.get("trades", 0.0)) for v in self.regime_counters.values())
            )
            if total_regime_trades > 0.0:
                for key in ("up", "down", "side"):
                    trades_val = float(
                        self.regime_counters.get(key, {}).get("trades", 0.0)
                    )
                    regime_trade_pct[key] = (trades_val / total_regime_trades) * 100.0
        trend_sample_count = float(getattr(self, "_trend_sample_count", 0) or 0)
        trend_regime_sum = (
            getattr(self, "_trend_regime_sum", {"up": 0.0, "down": 0.0, "side": 0.0})
            or {}
        )
        ema_trend_avg = 0.0
        dominant_regime = "none"
        trend_regime_avg = {"up": 0.0, "down": 0.0, "side": 0.0}
        if trend_sample_count > 0:
            ema_trend_avg = float(
                (getattr(self, "_ema_trend_accumulator", 0.0) or 0.0)
                / trend_sample_count
            )
            for key in ("up", "down", "side"):
                trend_regime_avg[key] = float(
                    trend_regime_sum.get(key, 0.0) / trend_sample_count
                )
            dominant_regime = (
                max(trend_regime_avg.items(), key=lambda item: item[1])[0]
                if trend_regime_avg
                else "none"
            )
        snapshot = {
            "profit_factor": profit_factor if np.isfinite(profit_factor) else 0.0,
            "profit_factor_raw": float(
                profit_factor_raw if np.isfinite(profit_factor_raw) else 0.0
            ),
            "trade_sharpe": trade_sharpe,
            "avg_return_pct": avg_return_pct,
            "cumulative_trade_return_pct": cumulative_return,
            "max_drawdown_pct": max_dd,
            "num_trades": float(total_trades),
            "win_rate_pct": win_rate_pct,
            "avg_trade_duration": avg_duration,
            "min_trade_duration": float(min(durations)) if durations else 0.0,
            "max_trade_duration": float(max(durations)) if durations else 0.0,
            "final_net_worth": float(self.net_worth),
            "total_return_pct": float(
                (self.net_worth - self.initial_balance) / (self.initial_balance + 1e-9)
            ),
            "episode_length": float(self.current_step),
            "pf_reliable": float(1.0 if pf_reliable else 0.0),
            "max_trades_cap": float(
                getattr(self, "max_trades_per_episode", 0) or 0
            ),  # None -> 0
            "max_episode_steps": float(getattr(self, "max_steps", 0) or 0),
            "regime_trade_up_pct": regime_trade_pct["up"],
            "regime_trade_down_pct": regime_trade_pct["down"],
            "regime_trade_side_pct": regime_trade_pct["side"],
            "trend_regime_avg_up": trend_regime_avg["up"],
            "trend_regime_avg_down": trend_regime_avg["down"],
            "trend_regime_avg_side": trend_regime_avg["side"],
            "trend_regime_samples": trend_sample_count,
            "ema_trend_avg": ema_trend_avg,
            "trend_dominant": dominant_regime,
        }
        # DistribuiÃ§Ã£o real por lado (long/short)
        try:
            if not hasattr(self, "side_counters"):
                self.side_counters = {"long": 0.0, "short": 0.0}
            long_ct = float(self.side_counters.get("long", 0.0))
            short_ct = float(self.side_counters.get("short", 0.0))
            total_sides = long_ct + short_ct
            snapshot["long_trade_pct"] = (
                float((long_ct / total_sides) * 100.0) if total_sides > 0 else 0.0
            )
            snapshot["long_trades_count"] = long_ct
            snapshot["short_trades_count"] = short_ct
        except Exception:
            snapshot["long_trade_pct"] = 0.0
            snapshot["long_trades_count"] = 0.0
            snapshot["short_trades_count"] = 0.0
        return snapshot

    def get_financial_stats(self) -> Dict[str, float]:
        """Retorna mÃƒÂ©tricas financeiras resumidas do episÃƒÂ³dio atual ou do ÃƒÂºltimo episÃƒÂ³dio concluÃƒÂ­do."""
        if self._current_episode_returns:
            return self._build_financial_snapshot(
                self._current_episode_returns, self._current_episode_durations
            )
        if self._last_episode_summary is not None:
            return dict(self._last_episode_summary)
        return {
            "profit_factor": 0.0,
            "trade_sharpe": 0.0,
            "avg_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "num_trades": 0.0,
            "win_rate_pct": 0.0,
            "avg_trade_duration": 0.0,
            "final_net_worth": float(self.initial_balance),
            "total_return_pct": 0.0,
            "episode_length": 0.0,
        }

    def get_exit_reason_counts(self) -> Dict[str, int]:
        """Retorna contadores de razões de saída dos trades."""
        return dict(
            getattr(
                self,
                "exit_reason_counts",
                {
                    "agent": 0,
                    "stop_loss": 0,
                    "catastrophe": 0,
                    "time_stop": 0,
                    "other": 0,
                },
            )
        )

    def _log_episode_matrix(self) -> None:
        """Imprime tabela de trades + P&L em dinheiro ao fim de cada episodio."""
        trades = getattr(self, "_episode_trades_log", [])
        # So imprime no modo de treino (nao durante Optuna)
        if self.mode == "optimization" or not trades:
            return
        initial = float(self.initial_balance)
        final = float(self.net_worth)
        pnl_usd = final - initial
        pnl_pct = (final / (initial + 1e-9) - 1.0) * 100.0
        wins = sum(1 for t in trades if t["pnl_usd"] > 0)
        losses = len(trades) - wins
        sign_u = "+" if pnl_usd >= 0 else ""
        sign_p = "+" if pnl_pct >= 0 else ""
        SEP = "=" * 80
        SEP2 = "-" * 80
        # [PERF CONSOLE] Trades vão apenas para o arquivo de log (logger.debug)
        # A tela mostra apenas barra de progresso + actor_loss/ent_coef do SB3
        logger.debug(SEP)
        logger.debug(
            "[EPISODE] Capital: $%.2f -> $%.2f  |  P&L: %s$%.2f (%s%.2f%%)",
            initial,
            final,
            sign_u,
            pnl_usd,
            sign_p,
            pnl_pct,
        )
        logger.debug(
            "[EPISODE] Trades: %d  |  Wins: %d  |  Losses: %d  |  Win Rate: %.1f%%",
            len(trades),
            wins,
            losses,
            wins / (len(trades) + 1e-9) * 100.0,
        )
        logger.debug(SEP2)
        logger.debug(
            "  %3s  %-5s  %5s  %12s  %12s  %10s  %7s  %s",
            "#",
            "Dir",
            "Dur",
            "Entry",
            "Exit",
            "P&L USD",
            "P&L %",
            "Motivo",
        )
        logger.debug(SEP2)
        # linhas de trades
        for t in trades:
            marker = "[WIN]" if t["pnl_usd"] > 0 else "[LOS]"
            su = "+" if t["pnl_usd"] >= 0 else ""
            sp = "+" if t["pnl_pct"] >= 0 else ""
            logger.debug(
                "  %3d  %-5s  %5d  %12.2f  %12.2f  %s$%8.2f  %s%5.2f%%  %s %s",
                t["#"],
                t["side"],
                t["duration"],
                t["entry"],
                t["exit"],
                su,
                t["pnl_usd"],
                sp,
                t["pnl_pct"],
                marker,
                t["exit_reason"],
            )
        # rodape: breakdown de motivos de saida
        exit_counts: dict = {}
        for t in trades:
            exit_counts[t["exit_reason"]] = exit_counts.get(t["exit_reason"], 0) + 1
        reasons_str = "  ".join(
            "%s:%d" % (k, v) for k, v in sorted(exit_counts.items())
        )
        logger.debug(SEP2)
        logger.debug("[EPISODE] Saidas >> %s", reasons_str)
        logger.debug(SEP)
        # reset do log para o proximo episodio
        self._episode_trades_log = []

    def _log_trial_matrix(self, trial_number: int = -1, score: float = 0.0) -> None:
        """
        Imprime a matriz de trades do ultimo episodio de avaliacao ao fim de cada trial Optuna.
        Diferente de _log_episode_matrix: SEMPRE imprime (sem check de mode) e nao reseta o log.
        """
        trades = getattr(self, "_episode_trades_log", [])
        if not trades:
            logger.debug(
                "[TRIAL %d] Sem trades registrados no ultimo episodio de avaliacao.",
                trial_number,
            )
            return
        initial = float(self.initial_balance)
        final = float(self.net_worth)
        pnl_usd = final - initial
        pnl_pct = (final / (initial + 1e-9) - 1.0) * 100.0
        wins = sum(1 for t in trades if t["pnl_usd"] > 0)
        losses = len(trades) - wins
        sign_u = "+" if pnl_usd >= 0 else ""
        sign_p = "+" if pnl_pct >= 0 else ""
        avg_dur = float(np.mean([t["duration"] for t in trades])) if trades else 0.0
        SEP = "=" * 80
        SEP2 = "-" * 80
        logger.debug(SEP)
        logger.debug(
            "[TRIAL %d] Score=%.4f | Capital: $%.2f -> $%.2f | P&L: %s$%.2f (%s%.2f%%)",
            trial_number,
            score,
            initial,
            final,
            sign_u,
            pnl_usd,
            sign_p,
            pnl_pct,
        )
        logger.debug(
            "[TRIAL %d] Trades: %d | Wins: %d | Loss: %d | WinRate: %.1f%% | AvgDur: %.1f candles",
            trial_number,
            len(trades),
            wins,
            losses,
            wins / (len(trades) + 1e-9) * 100.0,
            avg_dur,
        )
        logger.debug(SEP2)
        logger.debug(
            "  %3s  %-5s  %5s  %12s  %12s  %10s  %7s  %s",
            "#",
            "Dir",
            "Dur",
            "Entry",
            "Exit",
            "P&L USD",
            "P&L %",
            "Motivo",
        )
        logger.debug(SEP2)
        for t in trades:
            marker = "[WIN]" if t["pnl_usd"] > 0 else "[LOS]"
            su = "+" if t["pnl_usd"] >= 0 else ""
            sp = "+" if t["pnl_pct"] >= 0 else ""
            logger.debug(
                "  %3d  %-5s  %5d  %12.2f  %12.2f  %s$%8.2f  %s%5.2f%%  %s %s",
                t["#"],
                t["side"],
                t["duration"],
                t["entry"],
                t["exit"],
                su,
                t["pnl_usd"],
                sp,
                t["pnl_pct"],
                marker,
                t["exit_reason"],
            )
        exit_counts: dict = {}
        for t in trades:
            exit_counts[t["exit_reason"]] = exit_counts.get(t["exit_reason"], 0) + 1
        reasons_str = "  ".join(
            "%s:%d" % (k, v) for k, v in sorted(exit_counts.items())
        )
        logger.debug(SEP2)
        logger.debug("[TRIAL %d] Saidas >> %s", trial_number, reasons_str)
        logger.debug(SEP)
        # NAO reseta _episode_trades_log — chamado no eval_env que persiste entre trials

    def _store_episode_summary(self) -> None:
        # Imprime matriz de trades antes de resetar os dados
        self._log_episode_matrix()
        summary = self._build_financial_snapshot(
            list(self._current_episode_returns), self._current_episode_durations
        )
        summary["episode_reward"] = float(self._episode_reward_accumulator)
        logger.debug(
            "[FASE 1 DEBUG] Episode summary -> trades=%s avg_duration=%.2f",
            summary.get("num_trades"),
            summary.get("avg_trade_duration", 0.0),
        )
        self._last_episode_summary = summary
        self._episode_summaries.append(summary)
        self._episode_reward_accumulator = 0.0
        self._current_episode_returns = []
        self._current_episode_durations = []
        self._trend_regime_sum = {"up": 0.0, "down": 0.0, "side": 0.0}
        self._trend_sample_count = 0
        self._ema_trend_accumulator = 0.0
        # Reset contadores de lado (long/short)
        try:
            if hasattr(self, "side_counters"):
                self.side_counters = {"long": 0.0, "short": 0.0}
        except Exception:
            pass

    def consume_episode_summaries(self) -> List[Dict[str, float]]:
        summaries = self._episode_summaries[:]
        self._episode_summaries.clear()
        return summaries

    def get_gate_statistics(self) -> Dict[str, float]:
        if not isinstance(getattr(self, "_gate_stats", None), Counter):
            return {}
        return dict(self._gate_stats)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Unificado: sempre usa a lÃƒÂ³gica de treinamento (realista)
        return self._step_training(action)

    def _step_training(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if getattr(self, "disable_new_entries", False):
            logger.warning(
                f"[DISABLE ENTRIES] step={self.current_step}, trades_remaining={getattr(self, 'trades_remaining', None)}"
            )

        # Usa (posição, sl_mult, alavancagem)
        position_action_raw, sl_mult_raw, desired_leverage = action
        prev_net_worth = getattr(self, "_prev_net_worth", self.initial_balance)
        # [FIX IndexError] Bounds check em _step_training (mesmo tratamento de _get_observation)
        _raw_step = self.start_idx + self.current_step
        _safe_step = min(_raw_step, len(self.df) - 1)
        row_vals = self._rnp_values[_safe_step]
        current_row = _TrendNpRow(row_vals, self._col_idx_map)
        if isinstance(getattr(self, "_gate_stats", None), Counter):
            self._gate_stats["total_steps"] += 1
        current_price = current_row.get("close", 0.0)
        _prev_step = (
            min(max(0, _raw_step - 1), len(self.df) - 1)
            if self.current_step > 0
            else _safe_step
        )
        prev_price = float(
            self._rnp_values[_prev_step][self._col_idx_map.get("close", 0)]
        )
        # <<< CORREÃƒâ€¡ÃƒÆ'O: Usa o nome da coluna correto >>>
        atr = current_row.get(self.atr_col, current_price * 0.01)
        if np.isnan(atr) or atr <= 1e-9:
            atr = current_price * 0.01
        atr_pct = float(atr / (current_price + 1e-9)) if current_price else 0.0
        # Ajuste din?mico dos multiplicadores de SL conforme volatilidade
        sl_min_bound, sl_max_bound = self._dynamic_sl_bounds(atr_pct)
        sl_mult = float(np.clip(sl_mult_raw, sl_min_bound, sl_max_bound))
        if self.mode == "optimization":
            sl_mult = max(sl_mult, sl_min_bound)

        reward = 0.0
        penalty_accum = 0.0
        trade_return_pct = 0.0
        info = {}
        if getattr(self, "post_trade_cooldown", 0) > 0:
            self.post_trade_cooldown -= 1
        ema_trend_val = self._get_ema_trend_value(current_row)
        regime_up = float(current_row.get("tp_regime_up", 0.0))
        regime_down = float(current_row.get("tp_regime_down", 0.0))
        regime_side = float(current_row.get("tp_regime_sideways", 0.0))
        if hasattr(self, "_trend_regime_sum"):
            self._trend_sample_count += 1
            self._trend_regime_sum["up"] += float(regime_up)
            self._trend_regime_sum["down"] += float(regime_down)
            self._trend_regime_sum["side"] += float(regime_side)
            try:
                ema_val = float(ema_trend_val)
            except Exception:
                ema_val = 0.0
            if not np.isfinite(ema_val):
                ema_val = 0.0
            self._ema_trend_accumulator += ema_val
        trend_alignment_threshold = getattr(self, "trend_alignment_threshold", 0.0)
        regime_margin = getattr(self, "trend_regime_margin", 0.0)
        exploring = self.current_step < getattr(self, "exploration_warmup_steps", 0)
        if exploring:
            # Warmup: diminui levemente os limiares de alinhamento/regime
            trend_alignment_threshold *= max(
                0.3, min(1.0, self.exploration_threshold_scale)
            )
            regime_margin *= max(0.3, min(1.0, self.exploration_threshold_scale))
        trend_allows_long = (
            ema_trend_val >= trend_alignment_threshold
            or regime_up >= max(regime_down, regime_side) + regime_margin
        )
        trend_allows_short = (
            ema_trend_val <= -trend_alignment_threshold
            or regime_down >= max(regime_up, regime_side) + regime_margin
        )
        if self.relaxed_gates:
            trend_allows_long = True
            if not getattr(self, "_specialist_long_only", False):
                trend_allows_short = True
        # Suaviza bloqueios quando o prior estÃ¡ forte e coerente com a direÃ§Ã£o
        try:
            prior_dir_val_safe = float(current_row.get("tp_prior_dir", 0.0))
            # Capture previous prior before updating
            previous_prior = getattr(self, "_last_prior_dir_val", 0.0)
            self._last_prior_dir_val = prior_dir_val_safe
            prior_conf_safe = float(current_row.get("tp_prior_conf", 0.5))
        except Exception:
            prior_dir_val_safe, prior_conf_safe = 0.0, 0.5
        if prior_conf_safe >= 0.6:
            if prior_dir_val_safe > 0.2:
                trend_allows_long = True
            elif prior_dir_val_safe < -0.2:
                trend_allows_short = True

        prior_dir_val = float(current_row.get("tp_prior_dir", 0.0))
        if self.mode == "optimization":
            # [FIX] Fase 2 sem viés: permite ambos os lados para exploração completa.
            # A máscara direcional rígida NÃO deve sobrescrever este modo — ela causava
            # policy collapse pois bloqueava um lado mesmo durante busca de hiperparâmetros.
            trend_allows_long = True
            # BullTradingEnv: nunca permitir short, mesmo em HPO
            if not getattr(self, "_specialist_long_only", False):
                trend_allows_short = True
        else:
            # --- MÁSCARA DIRECIONAL (apenas em treinamento/produção) ---
            # 🔬 Dr. Tensor Fix: Removido o bloqueio rígido que forçava a direção.
            # O agente deve ser livre para errar e ser punido (Soft Gate).
            # As variáveis trend_allows_... agora servem apenas para cálculo de bônus/penalidades.
            mask_threshold = 0.1
            if prior_dir_val > mask_threshold:
                trend_allows_long = True
                trend_allows_short = True  # Permitir para aprendizado (soft)
            elif prior_dir_val < -mask_threshold:
                trend_allows_short = True
                trend_allows_long = True  # Permitir para aprendizado (soft)

        # Trend debug logging configurÃ¡vel
        if self.mode == "optimization" and getattr(self, "_log_trend_debug", False):
            try:
                message = (
                    "[TREND_DEBUG] step=%d | regime_up=%.2f | regime_down=%.2f | regime_side=%.2f | "
                    "ema_trend=%.2f | position=%d | steps_in_position=%d | reward=%.2f | trend_aligned=%s"
                )
                message_args = (
                    self.current_step,
                    regime_up,
                    regime_down,
                    regime_side,
                    float(ema_trend_val if not np.isnan(ema_trend_val) else 0.0),
                    int(np.sign(self.position)),
                    int(self.steps_in_position),
                    float(reward),
                    str(bool(trend_allows_long or trend_allows_short)),
                )
                if getattr(self, "_trend_debug_level", "debug") == "info":
                    logger.info(message, *message_args)
                else:
                    logger.debug(message, *message_args)
            except Exception:
                pass
        # --- CURRÃƒÂCULO DE RECOMPENSA ---
        # Warmup de penalidades nos primeiros passos do episÃƒÂ³dio para evitar saturaÃƒÂ§ÃƒÂ£o de puniÃƒÂ§ÃƒÂµes
        steps_warmup = 300
        if self.mode == "optimization":
            penalty_cap_per_step = self.opt_penalty_cap_per_step
            warmup_base = self.opt_warmup_base
            warmup_scale = self.opt_warmup_scale
        else:
            penalty_cap_per_step = self.train_penalty_cap_per_step
            warmup_base = self.train_warmup_base
            warmup_scale = self.train_warmup_scale
        warmup_factor = warmup_base + warmup_scale * float(
            min(1.0, max(0.0, self.current_step / float(max(1, steps_warmup))))
        )
        # AcÃƒÂºmulo de penalidades por step (antes do clip final) para limitar magnitude
        penalty_accum = 0.0
        penalties_debug: Dict[str, float] = {}
        is_optimization_phase = self.mode == "optimization"
        prev_unrealized_return = getattr(self, "_prev_unrealized_return", 0.0)
        if self.position != 0:
            pnl_step_value = (
                ((current_price - prev_price) / (prev_price + 1e-9))
                * self.initial_notional_value
                * np.sign(self.position)
            )
            self.net_worth += pnl_step_value

            # Floor de liquidação conservador
            _margin_minimum = self.initial_balance * 0.001
            if self.net_worth < _margin_minimum:
                self.net_worth = _margin_minimum

            unrealized_return = (
                (current_price - self.entry_price) / (self.entry_price + 1e-9)
            ) * np.sign(self.position)
            self.pnl_since_entry = unrealized_return
            self._prev_unrealized_return = unrealized_return
            self.steps_in_position += 1

            if self.net_worth > self.episode_peak_net_worth:
                self.episode_peak_net_worth = self.net_worth

            dd = (self.episode_peak_net_worth - self.net_worth) / (
                self.episode_peak_net_worth + 1e-9
            )
            self.episode_max_drawdown = max(self.episode_max_drawdown, dd)
        else:
            self.pnl_since_entry = 0.0
            self._prev_unrealized_return = 0.0
            # [SCIENTIFIC] Opportunity cost movido para scientific_corrections.py
        prev_timestamp = (
            self.df.index[self.current_step - 1]
            if self.current_step > 0
            else self.df.index[0]
        )
        current_timestamp = self.df.index[self.current_step]
        # Funding prÃƒÂ³-rata por passo, com base na duraÃƒÂ§ÃƒÂ£o do candle
        if self.position != 0:
            # [FIX #9] Timestamps com start_idx correto para funding cost
            _ts_prev = min(
                self.start_idx + max(0, self.current_step - 1), len(self.df) - 1
            )
            _ts_curr = min(self.start_idx + self.current_step, len(self.df) - 1)
            prev_timestamp = self.df.index[_ts_prev]
            current_timestamp = self.df.index[_ts_curr]
            delta_hours = max(
                1e-6, (current_timestamp - prev_timestamp).total_seconds() / 3600.0
            )
            funding_cost = (
                self.initial_notional_value
                * self.trading_config.LEVERAGE_COST_PER_DAY_PCT
                * (delta_hours / 24.0)
            )
            self.net_worth -= funding_cost
            # [SCIENTIFIC] Funding reward movido para scientific_corrections.py
        # --- LÃƒâ€œGICA DE DECISÃƒÆ'O "LUZ VERDE" ---
        # 1. Obter o Sinal Estratégico (A Luz)
        prior_dir_val = float(current_row.get("tp_prior_dir", 0.0))
        prior_conf_dyn = float(current_row.get("tp_prior_conf", 0.5))

        # [SCIENTIFIC] Alimentar dicionário info para a função de recompensa
        info["prior_dir_val"] = prior_dir_val
        info["prior_conf_val"] = prior_conf_dyn

        # --- CORRECAO: FILTRO DE CONFIANCA DINAMICO (SURFING) ---
        # Ajuste Dinâmico: Se a tendência (prior_dir) for muito forte, aceitamos confiança menor.
        trend_strength = abs(prior_dir_val)

        # CORREÇÃO CRÍTICA #1: Thresholds muito mais permissivos para surfar tendências fortes
        # [RELAX] Reduzido em 50% durante avaliação para evitar paralisia (Ranger/Holdout)
        eval_relax = 0.5 if self.mode in ("evaluation", "holdout") else 1.0

        if trend_strength > 0.8:
            required_conf = 0.001  # Tendência Extrema (>80%): Aceita QUALQUER confiança
        elif trend_strength > 0.6:
            required_conf = (
                0.005  # Tendência Muito Forte (>60%): Aceita confiança mínima
            )
        elif trend_strength > 0.4:
            required_conf = 0.02  # Tendência Forte (>40%): Aceita confiança muito baixa
        elif trend_strength > 0.3:
            required_conf = 0.10  # Tendência Moderada: Aceita confiança baixa
        else:
            required_conf = (
                0.30 * eval_relax
            )  # Tendência Fraca: Exige confiança moderada

        low_confidence_gate = prior_conf_dyn < required_conf

        # [FIX #4/#5] Consolidação dos gates — NÃO resetar trend_allows para False aqui.
        # O código anterior resetava para False e descartava todo o trabalho de EMA/regime.
        # Agora: o low_confidence_gate BLOQUEIA se confiança for baixa;
        # caso contrário, usa as permissões das fases anteriores (EMA + máscara de treino)
        # e ADICIONA permissões baseadas na direção do prior.
        if low_confidence_gate:
            # Confiança muito baixa: bloqueia tudo (hard gate justificado aqui)
            trend_allows_long = False
            trend_allows_short = False
        else:
            # Confiança suficiente: adiciona permissões baseadas no prior
            # (não remove o que EMA/regime ja calculou nas fases 1 e 2)
            if prior_dir_val > 0.2:
                trend_allows_long = True  # reforça/adiciona permissão long
            elif prior_dir_val < -0.2:
                trend_allows_short = True  # reforça/adiciona permissão short
            else:
                # Zona neutra com confiança: abre ambas direções
                # (Ranger opera aqui; Bull/Bear terão uma delas bloqueada pela máscara de treino)
                # [RELAX] 0.4 -> 0.2 em avaliação para permitir que o Ranger tente operar em lateralidade
                neutral_conf_thresh = (
                    0.2 if self.mode in ("evaluation", "holdout") else 0.4
                )
                if prior_conf_dyn >= neutral_conf_thresh:
                    trend_allows_long = True
                    trend_allows_short = True

        # 2. Obter o Voto Tático (O Agente)
        agent_vote_strength = position_action_raw

        # 🆕 Recompensa de Alinhamento Confiança-Voto
        vote_magnitude = abs(agent_vote_strength)

        # [BUG 14 FIX] Removido penalidade de desalinhamento de confiança puramente estrutural.
        # Ela criava conflito com o entry penalty e dominava o sinal científico.
        # Mantemos apenas o bônus positivo abaixo para incentivar o comportamento core.

        # Se voto está bem calibrado → Bonus (só quando flat: incentiva entradas alinhadas,
        # não fechamentos de trades — evita reward positivo em trades negativos fechados por Agent Decision)
        vote_confidence_alignment = 1.0 - abs(vote_magnitude - prior_conf_dyn)
        # [ANTI-REWARD-HACK] Bônus decai com passos sem ação para evitar que o agente
        # aprenda a ficar flat só coletando alinhamento de voto (+0.2 × 300 steps = +60).
        _steps_flat = getattr(self, "steps_in_position", 0) if self.position == 0 else 0
        _decay = max(0.2, 1.0 / (1.0 + _steps_flat / 50.0))
        if vote_confidence_alignment > 0.7 and self.position == 0:
            reward += 0.2 * _decay

        # BUG N4 FIX: Removido penalidade estrutural dupla ('directional_misalignment')
        # O método 'compute_scientific_reward' já aplica a punição via _regime_mismatch,
        # e aplicar um desconto avulso de -0.5 ou -1.0 duplicaria o evento.
        # Ajusta thresholds de abertura conforme o modo do ambiente
        if self.mode == "optimization":
            prior_open_long_threshold = self.opt_prior_open_long_threshold
            prior_open_short_threshold = self.opt_prior_open_short_threshold
            base_threshold = self.opt_base_threshold
            min_action_threshold = self.opt_min_action_threshold
            force_threshold_factor = self.opt_force_threshold_factor
        else:
            prior_open_long_threshold = self.train_prior_open_long_threshold
            prior_open_short_threshold = self.train_prior_open_short_threshold
            base_threshold = self.train_base_threshold
            min_action_threshold = self.train_min_action_threshold
            force_threshold_factor = self.train_force_threshold_factor

        exploring = self.current_step < getattr(self, "exploration_warmup_steps", 0)
        warmup_override = exploring
        warmup_vote_threshold = getattr(self, "exploration_warmup_vote_threshold", 0.05)

        if exploring:
            scale = self.exploration_threshold_scale
            base_threshold = min(base_threshold * scale, warmup_vote_threshold)
            min_action_threshold = min(
                min_action_threshold * scale, warmup_vote_threshold
            )
            force_threshold_factor = min(
                force_threshold_factor, self.exploration_force_threshold_factor
            )

        # Ajuste do limiar com base na confiança do Prior e ATR
        if exploring:
            action_threshold = min(
                warmup_vote_threshold, base_threshold * (1.0 - (prior_conf_dyn * 0.2))
            )
            atr_pct = float(atr / (current_price + 1e-9))
            action_threshold *= float(np.clip(1.0 + atr_pct, 1.0, 1.1))
            action_threshold = max(min_action_threshold * 0.5, action_threshold)
        else:
            action_threshold = base_threshold * (1.0 - (prior_conf_dyn * 0.4))
            atr_pct = float(atr / (current_price + 1e-9))
            action_threshold *= float(np.clip(1.0 + atr_pct, 1.0, 1.5))

            action_threshold = max(min_action_threshold, action_threshold)

            # (onde o ruído gaussiano ajuda a cruzar o threshold).
            # Movido para DEPOIS do max(min_action_threshold) para garantir o relaxamento.
            if self.relaxed_gates or self.mode in ("evaluation", "holdout"):
                action_threshold *= 0.5
                logger.debug(
                    "[ENV EVAL] Action threshold relaxado em 50%% (FINAL): %.3f",
                    action_threshold,
                )

        # Modulação dinâmica por regime
        try:
            regime_strength = float(max(regime_up, regime_down) - regime_side)
            regime_strength = float(np.clip(regime_strength, 0.0, 1.0))
            regime_factor = float(np.clip(1.0 - 0.5 * regime_strength, 0.6, 1.0))
            action_threshold *= regime_factor
        except Exception:
            pass

        # 3. Determinar a Ação com base na LUZ VERDE
        prior_gate_long = prior_dir_val > prior_open_long_threshold
        prior_gate_short = prior_dir_val < prior_open_short_threshold

        # Se prior estiver forte, não bloquear entrada pela gate de prior
        if prior_conf_dyn >= 0.6:
            if prior_dir_val > 0.2:
                prior_gate_long = True
            elif prior_dir_val < -0.2:
                prior_gate_short = True

        if warmup_override:
            prior_gate_long = True
            prior_gate_short = True
        if self.relaxed_gates:
            prior_gate_long = True
            prior_gate_short = True

        # [FIX RANGER] tp_prior_dir = 0.0 constante em mercado lateral fecha o gate para SEMPRE.
        # Para Ranger (allow_long E allow_short), dirão próxima de 0 É a condição ideal.
        # Nota: BullSpecialist tem allow_short=False, BearSpecialist tem allow_long=False.
        if getattr(self, "_allow_both_directions", False) and abs(prior_dir_val) < 0.30:
            # Mercado lateral: abrir ambos os gates para o Ranger poder operar
            prior_gate_long = True
            prior_gate_short = True

        # --- EXHAUSTION GATE (FOMO FIX) ---
        # Bloqueia entrada se a festa já acabou (Tendência > 0.80)
        # 🔬 Dr. Tensor: Softened exhaustion gate. We apply a penalty instead of blocking.
        trend_exhausted = abs(prior_dir_val) > 0.80
        if trend_exhausted and self.position == 0:
            # Em vez de setar False, aplicamos penalidade se ele abrir
            pass

        # --- REVERSAL LOGIC (FADING) ---
        # Permite reverter se a tendência estava exausta (>0.8) e começou a cair (<0.7)
        # Isso atende: "Reverter sem ser FOMO e sem ser Flip, só em tendência clara"
        is_fading = abs(previous_prior) > 0.8 and abs(prior_dir_val) < 0.7

        reversal_opportunity = False
        if is_fading:
            if previous_prior > 0 and agent_vote_strength < -0.15:
                reversal_opportunity = True
                trend_allows_short = True
            elif previous_prior < 0 and agent_vote_strength > 0.15:
                reversal_opportunity = True
                trend_allows_long = True

        # [FIX #11] ANTI-COLLAPSE REMOVIDO: substituiía output da rede neural corrompendo replay buffer.
        # O SAC aprende sozinho a não ser neutro via entropy regularization.
        # Se o agente estiver colapsando, aumentar ent_coef via Optuna.

        should_open_long = self.position <= 0 and agent_vote_strength > action_threshold
        should_open_short = (
            self.position >= 0 and agent_vote_strength < -action_threshold
        )

        # --- ENTRY DECISIONS ---
        # Entradas são recompensadas pelo lucro gerado, não pelo ato de entrar
        # [NOTE] Bloco [DIAGNOSTICO] movido para APÓS os safety overrides (ver abaixo)
        # para refletir os valores finais reais da decisão.
        if self.mode == "optimization":
            if getattr(self, "phase2_disable_longs", False):
                should_open_long = False
            if getattr(self, "phase2_disable_shorts", False):
                should_open_short = False
            # [FIX] Removido o bloqueio rígido 'phase2_follow_prior_sign' para permitir aprendizado soft

        # [SCIENTIFIC] Penalidades de gate movidas para scientific_corrections.py
        # O agente agora aprende via Soft Penalties unificados no final do step.

        # 3. ZONA LIVRE: Se o Predictor estiver NEUTRO (-0.1 a 0.1),
        #    liberamos totalmente - sem penalidades extras
        #    O agente aprende a antecipar reversões

        # --- [IMPORTANTE] Gates agora são SOFT ---
        # O agente não é bloqueado, mas aprende através de reward shaping
        # que ir contra o gate resulta em penalidades significativas.

        if warmup_override:
            # 🔬 Dr. Tensor: Softened warmup entries. Removed trend_allows_... blocks.
            should_open_long = (self.position <= 0) and (
                agent_vote_strength > warmup_vote_threshold
            )
            should_open_short = (self.position >= 0) and (
                agent_vote_strength < -warmup_vote_threshold
            )
            # [FIX LONG-ONLY] Especialistas long-only nunca abrem short, nem durante warmup
            if getattr(self, "_specialist_long_only", False):
                should_open_short = False
            # [FIX SHORT-ONLY] Especialistas short-only nunca abrem long, nem durante warmup
            if getattr(self, "_specialist_short_only", False):
                should_open_long = False
            if self.mode == "optimization" and (
                self.current_step
                % max(1, getattr(self, "exploration_warmup_log_span", 250))
                == 0
            ):
                logger.debug(
                    "[WARMUP] step=%d vote=%.3f thr=%.3f pos=%d open_long=%s open_short=%s",
                    self.current_step,
                    agent_vote_strength,
                    warmup_vote_threshold,
                    int(np.sign(self.position)),
                    should_open_long,
                    should_open_short,
                )
        # Cooldown dinÃ¢mico com bypass de alta convicÃ§Ã£o
        cooldown_active = getattr(self, "post_trade_cooldown", 0) > 0
        if self.relaxed_gates:
            cooldown_active = False
        if cooldown_active:
            strong_reversal_threshold = 0.7
            agent_votes_strong_reversal = (
                self.last_action_type < 0
                and agent_vote_strength > strong_reversal_threshold
            ) or (
                self.last_action_type > 0
                and agent_vote_strength < -strong_reversal_threshold
            )
            prior_confirms_reversal = (
                self.last_action_type < 0
                and prior_dir_val > self.opt_prior_open_long_threshold
            ) or (
                self.last_action_type > 0
                and prior_dir_val < self.opt_prior_open_short_threshold
            )
            market_structure_allows_reversal = (
                self.last_action_type < 0 and ema_trend_val >= -0.01
            ) or (self.last_action_type > 0 and ema_trend_val <= 0.01)
            high_conviction_reversal = (
                agent_votes_strong_reversal
                and prior_confirms_reversal
                and market_structure_allows_reversal
            )
            if high_conviction_reversal:
                logger.info(
                    f"[COOLDOWN BYPASS] Sinal de reversao de alta conviccao detectado no step {self.current_step}. Cooldown ignorado."
                )
                cooldown_active = False
        if cooldown_active:
            should_open_long = False
            should_open_short = False
        if getattr(self, "disable_new_entries", False):
            should_open_long = False
            should_open_short = False
        # [AUDITORIA] Hard gates removidos para permitir que o agente aprenda com Soft Penalties
        # if low_confidence_gate:
        #     should_open_long = False
        #     should_open_short = False

        # Bloqueio de reversões em regime neutro (sideways dominante)
        regime_neutral = regime_side >= max(regime_up, regime_down)
        if self.relaxed_gates:
            regime_neutral = False
        if regime_neutral and self.position != 0:
            if (self.position > 0 and should_open_short) or (
                self.position < 0 and should_open_long
            ):
                should_open_long = False
                should_open_short = False
        # [SOFT VETO] Especialistas unidirecionais agora usam penalidades no reward
        # em vez de bloqueios rígidos (Hard Gates). Isso permite que o gradiente
        # de política (SAC) aprenda o "porquê" de não operar contra o regime.
        if getattr(self, "_specialist_long_only", False) and should_open_short:
            # Sinaliza mismatch para o scientific_reward
            info["_regime_mismatch"] = True
            # Não bloqueamos a abertura, deixamos o RL sentir a punição
            pass
        if getattr(self, "_specialist_short_only", False) and should_open_long:
            info["_regime_mismatch"] = True
            pass

        # [DIAGNOSTICO] Capturado APÓS todos os safety overrides — valores refletem decisão final.
        # CORREÇÃO CRÍTICA #5: Logs Diagnósticos Aprimorados
        # Detectar JANELAS DE OURO perdidas (alta direção + baixa confiança + voto contrário)
        if self.current_step % 5000 == 0:
            try:
                is_golden_window = trend_strength > 0.7 and prior_conf_dyn < 0.05
                vote_misaligned = (
                    (prior_dir_val > 0.5 and agent_vote_strength < -0.1)
                    or (prior_dir_val < -0.5 and agent_vote_strength > 0.1)
                ) and not reversal_opportunity

                log_level = (
                    "WARNING" if (is_golden_window or vote_misaligned) else "INFO"
                )

                msg = (
                    f"[DIAGNOSTICO] Step={self.current_step} | "
                    f"Confianca={prior_conf_dyn:.4f} (Min={required_conf:.4f}) | "
                    f"Direcao={prior_dir_val:.2f} (Strength={trend_strength:.2f}) | "
                    f"VotoAgente={agent_vote_strength:.4f} (Thr={action_threshold:.4f}) | "
                    f"LowConfGate={low_confidence_gate} | "
                    f"AllowLong={trend_allows_long} | AllowShort={trend_allows_short} | "
                    f"OpenLong={should_open_long} | OpenShort={should_open_short}"
                )

                if is_golden_window:
                    msg += " | 🔥 GOLDEN_WINDOW (Sinal forte, conf baixa)"
                if vote_misaligned:
                    msg += " | ⚠️ VOTE_MISALIGNED (Agente contra Predictor)"

                # [CONSOLE SILENCIOSO] DIAGNOSTICO sempre em debug — aparece só no arquivo.
                # Warnings de VOTE_MISALIGNED disparam em todo reset do env de eval, poluindo a tela.
                logger.debug(msg)
            except Exception:
                pass

        is_opening_or_reversing = should_open_long or should_open_short
        if self.mode == "optimization":
            should_log_gate = exploring or (self.current_step % 1000 == 0)
            if should_log_gate:
                gate_notes = []
                if not trend_allows_long and self.position <= 0:
                    gate_notes.append("long_trend_block")
                if not trend_allows_short and self.position >= 0:
                    gate_notes.append("short_trend_block")
                if agent_vote_strength <= action_threshold:
                    gate_notes.append("low_vote")
                if not prior_gate_long and agent_vote_strength >= 0:
                    gate_notes.append("prior_long_gate")
                if not prior_gate_short and agent_vote_strength <= 0:
                    gate_notes.append("prior_short_gate")
                if isinstance(getattr(self, "_gate_stats", None), Counter):
                    if gate_notes:
                        self._gate_stats.update(gate_notes)
                    else:
                        self._gate_stats["gate_ok"] += 1
                logger.debug(
                    "[GATE] step=%d exploring=%s pos=%d prior=%.3f vote=%.3f action_thr=%.3f "
                    "trend_allow_long=%s trend_allow_short=%s open_long=%s open_short=%s notes=%s",
                    self.current_step,
                    exploring,
                    int(np.sign(self.position)),
                    prior_dir_val,
                    agent_vote_strength,
                    action_threshold,
                    trend_allows_long,
                    trend_allows_short,
                    should_open_long,
                    should_open_short,
                    ",".join(gate_notes) if gate_notes else "ok",
                )
                if (
                    not warmup_override
                    and self.position == 0
                    and not is_opening_or_reversing
                ):
                    entry_notes: List[str] = []
                    if agent_vote_strength <= action_threshold:
                        entry_notes.append("vote_below_threshold")
                    if (
                        agent_vote_strength > action_threshold
                        and not trend_allows_long
                        and agent_vote_strength > 0
                    ):
                        entry_notes.append("trend_block_long")
                    if (
                        agent_vote_strength < -action_threshold
                        and not trend_allows_short
                        and agent_vote_strength < 0
                    ):
                        entry_notes.append("trend_block_short")
                    if agent_vote_strength > action_threshold and not prior_gate_long:
                        entry_notes.append("prior_long_gate")
                    if agent_vote_strength < -action_threshold and not prior_gate_short:
                        entry_notes.append("prior_short_gate")
                    if entry_notes:
                        logger.debug(
                            "[ENTRY_BLOCK] step=%d vote=%.3f thr=%.3f notes=%s",
                            self.current_step,
                            agent_vote_strength,
                            action_threshold,
                            ",".join(entry_notes),
                        )
                        if isinstance(getattr(self, "_gate_stats", None), Counter):
                            self._gate_stats.update(
                                f"entry_block:{note}" for note in entry_notes
                            )
        if self.position == 0 and (self.current_step % 200 == 0):
            status_report: List[str] = []
            if getattr(self, "post_trade_cooldown", 0) > 0:
                status_report.append(
                    f"Em cooldown por mais {self.post_trade_cooldown} steps."
                )
            luz_verde_acesa = (
                prior_dir_val > prior_open_long_threshold
                or prior_dir_val < prior_open_short_threshold
            )
            if not luz_verde_acesa:
                status_report.append(
                    "Aguardando 'Luz Verde' (prior do TrendPredictor fraco)."
                )
            elif not trend_allows_long and not trend_allows_short:
                status_report.append(
                    "Aguardando alinhamento de tendencia (EMA/Regime)."
                )
            elif abs(agent_vote_strength) < action_threshold:
                status_report.append(
                    f"Aguardando conviccao do agente (voto: {agent_vote_strength:.2f} < thr: {action_threshold:.2f})."
                )
            if status_report:
                logger.debug(
                    f"[PATIENCE REPORT] Step {self.current_step}: Sem posicao. Motivo: {' | '.join(status_report)}"
                )
        # Ajusta action_side para refletir a direÃƒÂ§ÃƒÂ£o escolhida quando abrir
        # [FIX CIENTÍFICO] Low confidence gate softened
        # 🔬 Dr. Tensor: Removido o bloqueio rígido (action_side = 0).
        # Agora o mismatch/low-confidence é tratado via reward penalties.
        action_side = (
            1
            if should_open_long
            else (-1 if should_open_short else int(np.sign(agent_vote_strength)))
        )
        if low_confidence_gate:
            # [BUG 6 FIX] Nunca reutilizar _regime_mismatch aqui.
            # _regime_mismatch é específico para Bull tentando Short (e vice-versa).
            # Confiança baixa é uma condição completamente diferente.
            # Antes: info['_regime_mismatch'] = True causava -10/step via scientific_reward.
            # Agora: flag separada com penalidade proporcional e leve.
            info["_low_conf_penalty"] = True
            reward -= 1.0  # penalidade leve e separada — não ativa o -8.0 do scientific_reward

        # --- LÓGICA DE REVERSÃO (FLIP) INTELIGENTE ---
        # Verifica se o agente está tentando inverter a mão (Ex: estava comprado, agora quer vender)
        last_position_side = getattr(self, "last_action_type", 0)
        is_reversal = (last_position_side == 1 and action_side == -1) or (
            last_position_side == -1 and action_side == 1
        )

        if is_opening_or_reversing and last_position_side != 0:
            # Se for uma reversão (Flip):
            if is_reversal:
                # [SCIENTIFIC IMPROVEMENT] Remover bloqueio duro de reversão
                # O agente aprenderá que reversões ruins custam dinheiro através da Sortino Ratio
                # Apenas sinalizamos reversões inteligentes para métricas
                if reversal_opportunity:
                    predictor_confirms_flip = True
                else:
                    predictor_confirms_flip = (
                        action_side == 1 and prior_dir_val > 0.2
                    ) or (action_side == -1 and prior_dir_val < -0.2)

                if predictor_confirms_flip:
                    # BÔNUS: Reversão alinhada com mudança de tendência
                    reward += 2.0
                    if isinstance(getattr(self, "_gate_stats", None), Counter):
                        self._gate_stats["smart_reversal"] += 1
                # Removido: bloqueio físico e penalidade arbitrária
                # A recompensa negativa virá naturalmente do PnL ruim se errar

            # Se NÃO for reversão (é reentrada no mesmo lado - Scalping):
            elif last_position_side == action_side:
                # Punição leve para evitar ficar saindo e entrando na mesma tendência
                # Queremos que ele SEGURE, não que fique reentrando.
                reward -= 1.0
                penalty_accum += 1.0
                if isinstance(getattr(self, "_gate_stats", None), Counter):
                    self._gate_stats["scalping_penalty"] += 1

        # Atualiza is_opening_or_reversing após possíveis bloqueios de flip
        is_opening_or_reversing = should_open_long or should_open_short
        action_side = (
            1
            if should_open_long
            else (-1 if should_open_short else int(np.sign(agent_vote_strength)))
        )
        # LUZ AMARELA/VERMELHA (Fechamento Ativo)
        # BUG 6 FIX: Removido reset de info (era info: Dict[str, Any] = {})
        prior_threshold = float(getattr(self, "prior_flip_threshold", 0.3))
        prior_flipped_long = False
        prior_flipped_short = False
        prior_mismatch_ratio = 0.0
        if self.position > 0:
            prior_flipped_long = prior_dir_val < prior_threshold
            if prior_flipped_long:
                prior_mismatch_ratio = float(
                    np.clip(
                        (prior_threshold - prior_dir_val) / max(prior_threshold, 1e-6),
                        0.0,
                        5.0,
                    )
                )
        elif self.position < 0:
            prior_flipped_short = prior_dir_val > -prior_threshold
            if prior_flipped_short:
                prior_mismatch_ratio = float(
                    np.clip(
                        (prior_dir_val + prior_threshold) / max(prior_threshold, 1e-6),
                        0.0,
                        5.0,
                    )
                )
        if self.position != 0:
            base_grace = int(
                self._active_grace_period or self.scalp_penalty_min_duration
            )
            if self.mode == "optimization":
                grace_cap = max(5, getattr(self, "phase2_grace_cap_steps", 24))
                grace_period_steps = min(base_grace, grace_cap)
            else:
                grace_period_steps = max(3, base_grace)
            is_in_grace_period = self.steps_in_position < grace_period_steps
        else:
            grace_period_steps = 0
            is_in_grace_period = False
        # Bloqueia reversÃµes/novas aberturas durante a carÃªncia quando jÃ¡ hÃ¡ posiÃ§Ã£o
        if self.position != 0 and is_in_grace_period:
            should_open_long = False
            should_open_short = False
        if self.position > 0:
            agent_wants_close = agent_vote_strength <= -(action_threshold * 0.5)
            agent_wants_reverse = agent_vote_strength <= -action_threshold
        elif self.position < 0:
            agent_wants_close = agent_vote_strength >= (action_threshold * 0.5)
            agent_wants_reverse = agent_vote_strength >= action_threshold
        else:
            agent_wants_close = False
            agent_wants_reverse = False
        trend_allows_current = (self.position > 0 and trend_allows_long) or (
            self.position < 0 and trend_allows_short
        )
        prior_penalty = 0.0
        # Durante exploração SAC (buffer enchendo), prior_pressure desativado:
        # posições aleatórias são esperadas e não devem ser fechadas por prior.
        prior_pressure_active = not exploring
        if (
            (prior_flipped_long or prior_flipped_short)
            and not is_in_grace_period
            and prior_pressure_active
        ):
            prior_conf_scale = float(np.clip(prior_conf_dyn, 0.0, 1.0))
            exponent = float(getattr(self, "prior_flip_penalty_exponent", 1.0))
            prior_penalty = (
                self.prior_flip_penalty_weight
                * (prior_conf_scale * prior_mismatch_ratio) ** exponent
            )
            if prior_penalty > 0:
                reward -= prior_penalty
                penalty_accum += abs(prior_penalty)
                penalties_debug["prior_flip"] = float(prior_penalty)
                info["prior_mismatch_ratio"] = float(
                    prior_conf_scale * prior_mismatch_ratio
                )
        max_duration_cap = getattr(self, "max_trade_duration_cap", None)
        duration_penalty = 0.0
        if self.position != 0 and max_duration_cap:
            duration_start = int(
                getattr(self, "duration_pressure_start", max_duration_cap)
            )
            duration_start = int(np.clip(duration_start, 0, max_duration_cap))
            effective_span = max(1, max_duration_cap - max(duration_start, 0))
            steps_over_start = max(0, self.steps_in_position - max(duration_start, 0))
            if steps_over_start > 0:
                exponent = float(getattr(self, "duration_pressure_exponent", 1.0))
                pressure_ratio = float(
                    np.clip(steps_over_start / effective_span, 0.0, 5.0)
                )
                duration_penalty = self.duration_pressure_weight * (
                    pressure_ratio**exponent
                )
                reward -= duration_penalty
                penalty_accum += abs(duration_penalty)
                penalties_debug["duration_pressure"] = float(duration_penalty)
                info["duration_pressure_ratio"] = float(pressure_ratio)
        # [FIX EMERGENCY EXIT] Grace period não deve prender posições com perda severa
        # Referência: López de Prado AFLM - "never hold a position into catastrophic loss"
        # GUARD: emergency exit só cancela grace após steps mínimos para evitar scalp duration=1
        _EMERGENCY_MIN_STEPS = max(
            3, int(getattr(self, "_grace_floor", 10) * 0.3)
        )  # mínimo absoluto
        _pnl_pct_unrealized = float(getattr(self, "pnl_since_entry", 0.0))
        if (
            is_in_grace_period
            and _pnl_pct_unrealized < -0.02
            and self.steps_in_position >= _EMERGENCY_MIN_STEPS
        ):
            is_in_grace_period = False
            logger.debug(
                "[EMERGENCY EXIT] Grace period anulado: PnL=%.4f abaixo de -2%% no step=%d (steps_in=%d >= min=%d)",
                _pnl_pct_unrealized,
                self.current_step,
                self.steps_in_position,
                _EMERGENCY_MIN_STEPS,
            )

        # [HARD GUARD] Impede fisicamente o fechamento antes do mínimo absoluto de hold
        # Isso corrige o bug de scalping duration=1 que persiste mesmo com grace period:
        # O emergency exit anterior podia cancelar a grace na step 1 com movimentos voláteis.
        # Esta guarda é independente de toda a lógica de grace period.
        # [BUG FIX] Também suprime is_opening_or_reversing: sem essa supressão, uma tentativa de
        # reversão com position!=0 executa o fechamento via linha `if not trade_was_closed and
        # (is_closing_trade or is_opening_or_reversing)` mesmo com agent_confirms_close=False,
        # gerando TRADE CLOSED | duration=1 | reason=Agent Decision (bypass silencioso do HARD GUARD).
        _HARD_MIN_HOLD = max(3, int(getattr(self, "_grace_floor", 10)))
        if self.position != 0 and self.steps_in_position < _HARD_MIN_HOLD:
            is_in_grace_period = (
                True  # força grace ativa — nenhuma saída voluntária permitida
            )
            is_opening_or_reversing = (
                False  # impede que reversão force fechamento prematuro
            )
            # [FIX HARD_GUARD v2] A linha 2068 re-atribui is_opening_or_reversing = should_open_long or should_open_short.
            # Se não zeramos as variáveis base aqui, o HARD_GUARD é silenciosamente anulado, gerando duration=1.
            should_open_long = False
            should_open_short = False

        agent_confirms_close = (
            agent_wants_close or agent_wants_reverse
        ) and not is_in_grace_period
        closing_due_to_prior = bool(prior_penalty > 0 and agent_confirms_close)
        closing_due_to_agent = agent_confirms_close
        is_closing_trade = agent_confirms_close and (self.position != 0)
        if self.mode == "optimization" and self.position != 0 and not is_closing_trade:
            exit_notes: List[str] = []
            if is_in_grace_period:
                exit_notes.append("grace_period_active")
            if agent_wants_reverse and is_in_grace_period:
                exit_notes.append("reverse_blocked_by_grace")
            if agent_wants_close and not agent_confirms_close and trend_allows_current:
                exit_notes.append("trend_support")
            if not agent_wants_close and not agent_wants_reverse:
                exit_notes.append("agent_prefers_hold")
            if exit_notes and (
                exploring
                or (
                    self.current_step
                    % max(1, getattr(self, "exploration_warmup_log_span", 250))
                    == 0
                )
            ):
                logger.debug(
                    "[EXIT_BLOCK] step=%d pos=%d vote=%.3f pnl=%.4f reasons=%s",
                    self.current_step,
                    int(np.sign(self.position)),
                    agent_vote_strength,
                    self.pnl_since_entry,
                    ",".join(exit_notes),
                )
        # (Penalidades/bonificaÃ§Ãµes por inaÃ§Ã£o removidas e destravador desativado) Apenas contar steps sem aÃ§Ã£o
        if self.position == 0:
            if not hasattr(self, "steps_without_action"):
                self.steps_without_action = 0
            self.steps_without_action += 1
        else:
            self.steps_without_action = 0
        if getattr(self, "post_trade_cooldown", 0) > 0 or getattr(
            self, "disable_new_entries", False
        ):
            should_open_long = False
            should_open_short = False
        # Atualiza direção final após possíveis ajustes
        is_opening_or_reversing = should_open_long or should_open_short
        action_side = (
            1
            if should_open_long
            else (-1 if should_open_short else int(np.sign(agent_vote_strength)))
        )

        # [FIX CIENTÍFICO] Penalidade de baixa confiança já aplicada na linha ~1908.
        # Remover duplicata: a punição de -1.0 abaixo causava -2.0 no mesmo step.
        # if low_confidence_gate and is_opening_or_reversing:
        #     reward -= 1.0
        pnl_realized = 0
        trade_was_closed = False
        closed_trade_duration = None
        closed_trade_side = None
        # Stop de CatÃƒÂ¡strofe baseado em ATR (rede de seguranÃƒÂ§a do ambiente)
        # 🔬 [SCIENTIFIC FIX] Catastrophe SL deve SEMPRE estar ativo, mesmo durante o grace period.
        if self.position != 0:
            catastrophe_multiplier = self._compute_catastrophe_multiplier(
                atr_pct, self._last_prior_dir_val, int(np.sign(self.position))
            )
            # [FIX OPTION B] CSL em duas fases — buffer largo para o agente aprender a sair.
            # Fase 1 (+1 ATR): protege apenas breakeven (entry + fees). Sem trail ainda.
            # Fase 2 (+5 ATR): trail LARGO em peak - 4.5*ATR. O agente tem espaço para
            # segurar winners sem ser forçado a sair precocemente.
            # O Retreat Signal em scientific_corrections.py dá sinal per-step para sair
            # voluntariamente antes do CSL disparar (era: +3 ATR → trail peak - 3*ATR).
            if self.position > 0:
                # Atualiza pico de preço
                if self._peak_price_since_entry is None:
                    self._peak_price_since_entry = current_price
                else:
                    self._peak_price_since_entry = max(
                        self._peak_price_since_entry, current_price
                    )
                profit_in_atr = (self._peak_price_since_entry - self.entry_price) / (
                    atr + 1e-9
                )
                # CSL estático (proteção contra crash imediato pós-entrada)
                catastrophe_sl_static = self.entry_price - (
                    atr * catastrophe_multiplier
                )
                fee = getattr(self.trading_config, "TAKER_FEE", 0.001)
                be_price = self.entry_price * (1.0 + 2.1 * fee)
                # [FIX OPTION B] CSL dinâmico em duas fases:
                # Fase 1 (+1 ATR): protege apenas o breakeven — não trail ainda.
                #   Garante que o agente nunca perde mais do que as taxas de entrada.
                # Fase 2 (+5 ATR): trail LARGO em peak - 4.5*ATR (era peak - 3*ATR após +3 ATR).
                #   Buffer 50% maior dá ao agente espaço real para segurar o winner.
                #   O Retreat Signal em scientific_corrections.py ensina o agente a sair
                #   voluntariamente antes do CSL disparar — sem forçar saída precoce.
                # Trailing stop em 2 fases (long):
                # Fase 1 (+2 ATR): protege breakeven. O agente tem colchão mínimo.
                # Fase 2 (+3 ATR): trailing apertado em peak - 1.0×multiplier.
                #   Antes era +5 ATR / 1.5× mult → stop de 4.5 ATR abaixo do pico,
                #   o que incentivava o agente a sair manualmente (Agent Decision)
                #   antes de o stop disparar. Agora o stop acompanha o winner de perto,
                #   ensinando o agente a confiar no stop e surfar mais tempo.
                if profit_in_atr >= 3.0:
                    trail_level = self._peak_price_since_entry - (
                        atr * catastrophe_multiplier * 1.0
                    )
                    new_floor = max(be_price, trail_level)
                    self._catastrophe_sl_floor = max(
                        self._catastrophe_sl_floor or be_price, new_floor
                    )
                elif profit_in_atr >= 2.0:
                    # Breakeven apenas — trail ainda não arrasta
                    self._catastrophe_sl_floor = max(
                        self._catastrophe_sl_floor or be_price, be_price
                    )
                catastrophe_sl_level = max(
                    catastrophe_sl_static,
                    self._catastrophe_sl_floor or catastrophe_sl_static,
                )
                if current_price <= catastrophe_sl_level:
                    trade_was_closed = True
                    close_price = catastrophe_sl_level
                    pnl_realized = self.initial_notional_value * (
                        (close_price - self.entry_price) / (self.entry_price + 1e-9)
                    )
                    self._apply_realized_pnl(pnl_realized)
                    exit_fee = (
                        self.initial_notional_value or 0
                    ) * self.trading_config.TAKER_FEE
                    self.net_worth -= exit_fee
                    self._prev_unrealized_return = 0.0
                    closed_trade_duration = self.steps_in_position
                    closed_trade_side = "long"
                    info["exit_reason"] = "Catastrophe Stop Loss"
            else:
                # Atualiza pico de preço (short: pico = mínimo alcançado)
                if self._peak_price_since_entry is None:
                    self._peak_price_since_entry = current_price
                else:
                    self._peak_price_since_entry = min(
                        self._peak_price_since_entry, current_price
                    )
                profit_in_atr = (self.entry_price - self._peak_price_since_entry) / (
                    atr + 1e-9
                )
                # CSL estático
                catastrophe_sl_static = self.entry_price + (
                    atr * catastrophe_multiplier
                )
                fee = getattr(self.trading_config, "TAKER_FEE", 0.001)
                be_price = self.entry_price * (1.0 - 2.1 * fee)
                # [FIX OPTION B] CSL dinâmico em duas fases (short):
                # Fase 1 (+1 ATR): protege breakeven apenas.
                # Fase 2 (+5 ATR): trail largo em peak + 4.5*ATR (buffer 50% maior).
                # [FIX CSL THRESHOLD] Breakeven floor só ativa em +3 ATR (era +1 ATR). Short side.
                # Trailing stop em 2 fases (short):
                # Fase 1 (+2 ATR): protege breakeven.
                # Fase 2 (+3 ATR): trailing apertado em peak + 1.0×multiplier.
                if profit_in_atr >= 3.0:
                    trail_level = self._peak_price_since_entry + (
                        atr * catastrophe_multiplier * 1.0
                    )
                    new_floor = min(be_price, trail_level)
                    self._catastrophe_sl_floor = min(
                        self._catastrophe_sl_floor or be_price, new_floor
                    )
                elif profit_in_atr >= 2.0:
                    # Breakeven apenas — trail ainda não arrasta
                    self._catastrophe_sl_floor = min(
                        self._catastrophe_sl_floor or be_price, be_price
                    )
                catastrophe_sl_level = min(
                    catastrophe_sl_static,
                    self._catastrophe_sl_floor or catastrophe_sl_static,
                )
                if current_price >= catastrophe_sl_level:
                    trade_was_closed = True
                    close_price = catastrophe_sl_level
                    pnl_realized = self.initial_notional_value * (
                        (self.entry_price - close_price) / (self.entry_price + 1e-9)
                    )
                    self._apply_realized_pnl(pnl_realized)
                    exit_fee = (
                        self.initial_notional_value or 0
                    ) * self.trading_config.TAKER_FEE
                    self.net_worth -= exit_fee
                    self._prev_unrealized_return = 0.0
                    closed_trade_duration = self.steps_in_position
                    closed_trade_side = "short"
                    info["exit_reason"] = "Catastrophe Stop Loss"
        # Checagem apenas de SL definido pelo agente (TP removido)
        # 🔬 SISTEMA 3 CAMADAS: SL normal desativado quando disable_normal_sl=True
        # Aplica o período de carência antes de permitir que o SL encerre o trade
        if (
            not trade_was_closed
            and self.position != 0
            and (self.sl_level is not None)
            and not is_in_grace_period
            and not getattr(self, "disable_normal_sl", False)
        ):
            if self.position > 0:
                if self.sl_level is not None and current_price <= self.sl_level:
                    trade_was_closed = True
                    close_price = self._simulate_slippage(
                        current_price, atr, OrderSide.SELL
                    )
                    pnl_realized = self.initial_notional_value * (
                        (close_price - self.entry_price) / (self.entry_price + 1e-9)
                    )
                    self._apply_realized_pnl(pnl_realized)
                    exit_fee = (
                        self.initial_notional_value or 0
                    ) * self.trading_config.TAKER_FEE
                    self.net_worth -= exit_fee
                    self._prev_unrealized_return = 0.0
                    closed_trade_duration = self.steps_in_position
                    closed_trade_side = "long"
                    info["exit_reason"] = "Stop Loss"
            else:
                if (
                    self.sl_level is not None and current_price >= self.sl_level
                ):  # [FIX #1] condição restaurada
                    trade_was_closed = True
                    close_price = self._simulate_slippage(
                        current_price, atr, OrderSide.BUY
                    )
                    pnl_realized = self.initial_notional_value * (
                        (self.entry_price - close_price) / (self.entry_price + 1e-9)
                    )
                    self._apply_realized_pnl(
                        pnl_realized
                    )  # chamado apenas 1x (era duplicado)
                    exit_fee = (
                        self.initial_notional_value or 0
                    ) * self.trading_config.TAKER_FEE
                    self.net_worth -= exit_fee
                    self._prev_unrealized_return = 0.0
                    closed_trade_duration = self.steps_in_position
                    closed_trade_side = "short"
                    info["exit_reason"] = "Stop Loss"

        # [NEW] Ranger Quick Exit Logic
        if not trade_was_closed and self.position != 0:
            if hasattr(self, "_should_exit_fast") and self._should_exit_fast(
                self.pnl_since_entry
            ):
                trade_was_closed = True
                close_price = self._simulate_slippage(
                    current_price,
                    atr,
                    OrderSide.SELL if self.position > 0 else OrderSide.BUY,
                )
                pnl_realized = (
                    self.initial_notional_value
                    * ((close_price - self.entry_price) / (self.entry_price + 1e-9))
                    * np.sign(self.position)
                )
                self._apply_realized_pnl(pnl_realized)
                exit_fee = (
                    self.initial_notional_value or 0
                ) * self.trading_config.TAKER_FEE
                self.net_worth -= exit_fee
                self._prev_unrealized_return = 0.0
                closed_trade_duration = self.steps_in_position
                closed_trade_side = "long" if self.position > 0 else "short"
                info["exit_reason"] = "Quick Exit (TP)"
        # Trailing Stop simples baseado em ATR quando em lucro
        if (
            self.position != 0
            and not trade_was_closed
            and self.pnl_since_entry > (atr / (self.entry_price + 1e-9))
        ):
            trail_mult = 0.5 * sl_mult  # aproxima o SL conforme o lucro cresce
            if self.position > 0:
                new_sl = max(self.sl_level or -np.inf, current_price - atr * trail_mult)
                self.sl_level = new_sl
            else:
                new_sl = min(self.sl_level or np.inf, current_price + atr * trail_mult)
                self.sl_level = new_sl
        # Chandelier Exit (ATR-based)
        if self.position != 0 and not trade_was_closed:
            lookback = 22
            atr_mult = 3.0
            start = max(0, _raw_step - lookback + 1)
            window_high = (
                float(np.max(self.df["high"].iloc[start:_raw_step]))
                if "high" in self.df.columns
                else current_price
            )
            window_low = (
                float(np.min(self.df["low"].iloc[start:_raw_step]))
                if "low" in self.df.columns
                else current_price
            )
            if self.position > 0:
                chandelier_sl = window_high - atr_mult * atr
                self.sl_level = max(self.sl_level or -np.inf, chandelier_sl)
            else:
                chandelier_sl = window_low + atr_mult * atr
                self.sl_level = min(self.sl_level or np.inf, chandelier_sl)
        # Parabolic SAR simplificado
        if self.position != 0 and not trade_was_closed:
            if self._psar is None:
                if self.position > 0:
                    self._psar_trend = 1
                    self._psar = float(self.entry_price)
                    self._psar_ep = float(current_row.get("high", current_price))
                else:
                    self._psar_trend = -1
                    self._psar = float(self.entry_price)
                    self._psar_ep = float(current_row.get("low", current_price))
                self._psar_af = 0.02
            else:
                if self._psar_trend == 1:
                    new_ep = max(
                        self._psar_ep, float(current_row.get("high", current_price))
                    )
                    if new_ep > self._psar_ep:
                        self._psar_ep = new_ep
                        self._psar_af = min(self._psar_af + 0.02, 0.2)
                    self._psar = self._psar + self._psar_af * (
                        self._psar_ep - self._psar
                    )
                    self.sl_level = max(self.sl_level or -np.inf, self._psar)
                else:
                    new_ep = min(
                        self._psar_ep, float(current_row.get("low", current_price))
                    )
                    if new_ep < self._psar_ep:
                        self._psar_ep = new_ep
                        self._psar_af = min(self._psar_af + 0.02, 0.2)
                    self._psar = self._psar + self._psar_af * (
                        self._psar_ep - self._psar
                    )
                    self.sl_level = min(self.sl_level or np.inf, self._psar)
        # Time-based stop: respeita o per?odo de car?ncia e s? fecha por neutralidade ou invers?o clara do EMA
        if (
            self.position != 0
            and not trade_was_closed
            and (not getattr(self, "disable_time_stop", False))
            and self.time_stop_step is not None
            and self.current_step >= self.time_stop_step
            and not is_in_grace_period
        ):
            row_curr = self.df.iloc[self.start_idx + self.current_step]
            ema_trend_val = self._get_ema_trend_value(row_curr)
            ema_trend_present = (self.ema_trend_col is not None) and (
                not pd.isna(ema_trend_val)
            )
            regime_up_now = float(row_curr.get("tp_regime_up", 0.0))
            regime_down_now = float(row_curr.get("tp_regime_down", 0.0))
            regime_side_now = float(row_curr.get("tp_regime_sideways", 0.0))
            if self.steps_in_position < 10:
                # Primeiros 10 candles: exige margem adicional para fechar por neutralidade
                neutral_or_against = (
                    (regime_side_now >= max(regime_up_now, regime_down_now) + 0.1)
                    or (self.position > 0 and ema_trend_val <= -0.02)
                    or (self.position < 0 and ema_trend_val >= 0.02)
                )
            else:
                neutral_or_against = (
                    (regime_side_now >= max(regime_up_now, regime_down_now))
                    or (self.position > 0 and ema_trend_val <= 0)
                    or (self.position < 0 and ema_trend_val >= 0)
                )
            if (not ema_trend_present) or neutral_or_against:
                trade_was_closed = True
                close_price = self._simulate_slippage(
                    current_price,
                    atr,
                    OrderSide.SELL if self.position > 0 else OrderSide.BUY,
                )
                pnl_realized = (
                    self.initial_notional_value
                    * ((close_price - self.entry_price) / (self.entry_price + 1e-9))
                    * np.sign(self.position)
                )
                self._apply_realized_pnl(pnl_realized)
                exit_fee = (
                    self.initial_notional_value or 0
                ) * self.trading_config.TAKER_FEE
                self.net_worth -= exit_fee
                self._prev_unrealized_return = 0.0
                # [FIX #2] Removidas linhas duplicadas (eram 4 linhas, ficam 2)
                closed_trade_duration = self.steps_in_position
                closed_trade_side = "long" if self.position > 0 else "short"
                info["exit_reason"] = "Time Stop"
        # SaÃƒÂ­da/entrada por decisÃƒÂ£o ativa do agente
        if not trade_was_closed and (is_closing_trade or is_opening_or_reversing):
            if self.position != 0:
                trade_was_closed = True
                close_price = self._simulate_slippage(
                    current_price,
                    atr,
                    OrderSide.SELL if self.position > 0 else OrderSide.BUY,
                )
                pnl_realized = (
                    self.initial_notional_value
                    * ((close_price - self.entry_price) / (self.entry_price + 1e-9))
                    * np.sign(self.position)
                )
                self._apply_realized_pnl(pnl_realized)
                self._prev_unrealized_return = 0.0
                # Fee de sa?da
                exit_fee = (
                    self.initial_notional_value or 0
                ) * self.trading_config.TAKER_FEE
                self.net_worth -= exit_fee
                exit_reason = "Agent Decision"
                if closing_due_to_prior:
                    exit_reason = "Agent Decision - prior pressure"
                info["exit_reason"] = exit_reason
                # [BUG FIX] Captura duração e lado ANTES do reset de steps_in_position.
                # Sem isso: closed_trade_duration=None → duration_steps=1 para todos Agent Decision.
                # Efeito: Exit Quality (Comp 3) e Anti-Scalping (Comp 1B) penalizavam 5.5x
                # os Agent Decision wins vs. CSL/SL → agente aprendia a NÃO fechar voluntariamente.
                closed_trade_duration = self.steps_in_position
                closed_trade_side = "long" if self.position > 0 else "short"
            if is_opening_or_reversing and not trade_was_closed:
                self.entry_price = self._simulate_slippage(
                    current_price,
                    atr,
                    OrderSide.BUY if action_side > 0 else OrderSide.SELL,
                )
                self.position = action_side
                if isinstance(getattr(self, "_gate_stats", None), Counter):
                    if action_side > 0:
                        self._gate_stats["opened_long"] += 1
                    elif action_side < 0:
                        self._gate_stats["opened_short"] += 1
                # Ajusta alavancagem conforme novo espaÃƒÂ§o de aÃƒÂ§ÃƒÂ£o (ÃƒÂ­ndice 2)
                self.current_leverage = np.clip(
                    desired_leverage,
                    self.action_space.low[2],
                    self.action_space.high[2],
                )

                # --- CORREÇÃO: POSITION SIZING BASEADO EM CONFIANÇA PURA ---
                # A confiança vem da Rede Neural (tp_prior_conf), ajustada apenas pela incerteza.
                # Não misturamos mais com o voto do agente para dimensionamento.
                raw_conf = float(current_row.get("tp_prior_conf", 0.5))
                uncertainty = float(current_row.get("tp_uncertainty", 0.0))

                # Penaliza confiança se a incerteza for alta
                adjusted_conf = raw_conf * (1.0 - min(uncertainty, 0.5))

                # Mapeia confiança (0.5 a 1.0) para tamanho (0.5 a 1.0 da banca alocada)
                # Confiança < 0.5 já deve ter sido filtrada pelos gates, mas garantimos o mínimo.
                position_size_pct = np.clip((adjusted_conf - 0.5) * 2.0, 0.1, 1.0)

                # Define o valor nocional inicial com base na confiança pura
                # (Usa 10% do patrimônio líquido como base, escalado pela alavancagem e confiança)
                # [D-A3 FIX] Vol-targeting: reduz position size proporcionalmente ao ATR.
                # Sem ajuste, 10% base_allocation implica risco igual em ATR=0.5% e ATR=5%.
                # Target: cada posição arrisca ~2% do net_worth por unidade de ATR.
                # vol_scale = min(1.0, 0.02 / atr_pct) → ATR=1%→ full, ATR=4%→ 50%
                _atr_pct_now = max(atr_pct, 1e-4)
                vol_scale = min(1.0, 0.02 / _atr_pct_now)
                base_allocation = self.net_worth * 0.10 * vol_scale
                self.initial_notional_value = (
                    base_allocation * self.current_leverage * position_size_pct
                )

                # Fee de entrada
                entry_fee = (
                    self.initial_notional_value or 0
                ) * self.trading_config.TAKER_FEE
                self.net_worth -= entry_fee
                # Alinhamento com o TrendPredictor — registrado na ENTRADA mas contado no FECHAMENTO
                # [FIX CIENTÍFICO] Threshold de alinhamento reduzido para 0.1 para bull/bear specialists
                # Isso evita que trades legítimos fiquem "sem classificação" (fantasmas).
                _spec_name = getattr(self, "specialist_name", "").lower()
                alignment_threshold = (
                    0.1 if ("bull" in _spec_name or "bear" in _spec_name) else 0.2
                )

                if abs(prior_dir_val) > alignment_threshold:
                    aligned = (self.position > 0 and prior_dir_val > 0) or (
                        self.position < 0 and prior_dir_val < 0
                    )
                    self._current_trade_aligned = (
                        aligned  # salva para contar no fechamento
                    )
                    if not aligned:
                        # [BUG 6 FIX] Removido penalidade direta de -5.0.
                        # compute_scientific_reward() já aplica -8.0 para mismatch de regime
                        pass
                else:
                    self._current_trade_aligned = None  # direção neutra, não conta

                duration_hint_raw = current_row.get(
                    "tp_duration_median", self.scalp_penalty_min_duration
                )
                try:
                    duration_hint = float(duration_hint_raw)
                except Exception:
                    duration_hint = float(self.scalp_penalty_min_duration)
                trend_bias = (
                    1.0
                    if (
                        (action_side > 0 and trend_allows_long)
                        or (action_side < 0 and trend_allows_short)
                    )
                    else 0.0
                )
                prior_strength = float(np.clip(abs(prior_dir_val_safe), 0.0, 1.0))
                self._active_grace_period = self._compute_trade_grace_period(
                    atr_pct, duration_hint, prior_strength, trend_bias
                )
                self.steps_in_position = 0
                # Lógica de reversão movida para cima (flip inteligente)
                # Apenas atualiza o rastreamento
                self.last_action_type = action_side
                self._last_flip_step = self.current_step
                # [TRAILING CATASTROPHE SL] Reset do pico ao abrir novo trade
                self._peak_price_since_entry = current_price
                self._catastrophe_sl_floor = None
                # SL Inicial: Com "Breathing Room" mas sem ser catastrófico
                # 🔬 Dr. Tensor: Reduzido de 2.5x para 1.5x (era 2.5x na auditoria)
                # para evitar que perdas isoladas destruam o PnL do episódio.
                sl_base_multiplier = sl_mult * 1.5

                if self.position > 0:
                    self.sl_level = self.entry_price - atr * sl_base_multiplier
                else:
                    self.sl_level = self.entry_price + atr * sl_base_multiplier
                # Ajuste do SL por incerteza (mais conservador se uncertainty alta)
                unc_val = float(
                    self.df.iloc[self.start_idx + self.current_step].get(
                        "tp_uncertainty", 0.0
                    )
                )
                if unc_val > 0.0:
                    # Mesmo com incerteza, usar multiplicador maior para surfing
                    sl_unc_mult = 1.0 + min(0.8, unc_val)  # Aumentou de 0.5 para 0.8
                    if self.position > 0:
                        self.sl_level = (
                            self.entry_price - atr * sl_base_multiplier * sl_unc_mult
                        )
                    else:
                        self.sl_level = (
                            self.entry_price + atr * sl_base_multiplier * sl_unc_mult
                        )
                # Define time stop usando duraÃƒÂ§ÃƒÂ£o mediana (robustecido por regime)
                dur_med_raw = self.df.iloc[self.start_idx + self.current_step].get(
                    "tp_duration_median", 0
                )
                try:
                    dur_med = int(float(dur_med_raw))
                except Exception:
                    dur_med = 0
                trend_boost = 5 if (trend_allows_long or trend_allows_short) else 0
                base_time_stop = self.current_step + max(8, dur_med + trend_boost)
                if getattr(self, "disable_time_stop", False):
                    self.time_stop_step = None
                elif getattr(self, "max_trade_duration_cap", None):
                    cap = int(
                        max(
                            self.scalp_penalty_min_duration, self.max_trade_duration_cap
                        )
                    )
                    self.time_stop_step = min(base_time_stop, self.current_step + cap)
                else:
                    self.time_stop_step = base_time_stop
                # Registra confianÃƒÂ§a do prior no momento da entrada (para ECE)
                try:
                    self._last_entry_prior_conf = float(
                        self.df.iloc[self.start_idx + self.current_step].get(
                            "tp_prior_conf", np.nan
                        )
                    )
                except Exception:
                    self._last_entry_prior_conf = None
                # Reset PSAR
                self._psar = None
                self._psar_ep = None
                self._psar_trend = int(np.sign(self.position))
                self._psar_af = 0.02
                # (BÃƒÂ´nus de abertura removido)
            else:
                (
                    self.position,
                    self.entry_price,
                    self.initial_notional_value,
                    self.steps_in_position,
                ) = 0.0, 0.0, 0.0, 0
                self._active_grace_period = 0
                self.sl_level = None
                self.time_stop_step = None
                self._psar = None
                self._psar_ep = None
        if trade_was_closed:
            prev_position_sign = int(np.sign(self.position))
            scalp_min = getattr(self, "scalp_penalty_min_duration", 5)
            if getattr(self, "post_trade_cooldown_base", 0) > 0:
                self.post_trade_cooldown = int(self.post_trade_cooldown_base)
            trade_notional = float(self.initial_notional_value)
            leverage_used = (
                float(abs(self.current_leverage))
                if abs(self.current_leverage) > 1e-9
                else 1.0
            )
            if trade_notional > 0.0:
                margin_basis = trade_notional / leverage_used
            else:
                margin_basis = max(1.0, abs(self.net_worth))
            margin_basis = float(max(margin_basis, 1e-6))
            trade_return_raw = (
                pnl_realized / margin_basis if np.isfinite(pnl_realized) else 0.0
            )
            trade_return_pct = trade_return_raw
            duration_steps_raw = (
                closed_trade_duration
                if closed_trade_duration is not None
                else self.steps_in_position
            )
            duration_steps = int(max(1, duration_steps_raw))
            pnl_realized_val = (
                float(pnl_realized) if "pnl_realized" in locals() else 0.0
            )
            is_profitable_trade = pnl_realized_val > 0.0
            logger.debug(
                "[FASE 1 DEBUG] Trade closed: duration=%s reason=%s trades_remaining=%s",
                duration_steps,
                info.get("exit_reason"),
                getattr(self, "trades_remaining", None),
            )
            reg_metrics = self.df.iloc[self.start_idx + self.current_step]
            reg_up = float(reg_metrics.get("tp_regime_up", 0.0))
            reg_down = float(reg_metrics.get("tp_regime_down", 0.0))
            reg_side = float(reg_metrics.get("tp_regime_sideways", 0.0))
            long_threshold = getattr(self, "long_trade_bonus_threshold", 0)
            long_bonus_cap = getattr(self, "long_trade_bonus_max", 0.0)
            # CORREÇÃO CRÍTICA #3: Penalidade de Scalping Muito Mais Severa
            # Análise mostrou 55% de trades ≤5 steps - agente scalping demais
            # [BUG 5 FIX] Reset scalp_penalty_base para 4.0
            scalp_penalty_base = getattr(self, "_scalp_penalty", 4.0)

            # [CIENTÍFICO] Sem limite de trades = agente livre para surfar tendência
            max_trades = getattr(self, "max_trades_per_episode", None)
            within_quota = (
                True if max_trades is None else (self._trades_in_episode < max_trades)
            )

            if within_quota:
                self.trade_return_history.append(trade_return_pct)
                self._current_episode_returns.append(trade_return_pct)
                self._current_episode_durations.append(duration_steps)
                self._trades_in_episode += 1

                # [CORREÇÃO] Verificação segura de trades_remaining (pode ser None ou inf)
                trades_rem = getattr(self, "trades_remaining", None)
                if trades_rem is not None and trades_rem != float("inf"):
                    if trades_rem > 0:
                        self.trades_remaining = trades_rem - 1
                    if self.trades_remaining <= 0:
                        self.trades_remaining = 0
                        self.disable_new_entries = True
                        logger.debug(
                            "[FASE 1 DEBUG] Entries disabled - trades limit atingido"
                        )
                exit_reason = info.get("exit_reason", "Agent Decision")
                # Track exit reasons for diagnostics
                if exit_reason == "Agent Decision":
                    self.exit_reason_counts["agent"] += 1
                elif exit_reason == "Stop Loss":
                    self.exit_reason_counts["stop_loss"] += 1
                elif exit_reason == "Catastrophe Stop Loss":
                    self.exit_reason_counts["catastrophe"] += 1
                elif exit_reason == "Time Stop":
                    self.exit_reason_counts["time_stop"] += 1
                elif "Ruin" in str(exit_reason):
                    self.exit_reason_counts["ruin"] += 1
                else:
                    self.exit_reason_counts["other"] += 1
                reward_do_trade = self._compute_trade_reward(
                    pnl_realized, trade_return_pct, duration_steps, exit_reason, atr_pct
                )
                reward += reward_do_trade
                # [FIX REWARD #2] Salva componente individual para o log de breakdown
                self._last_trade_reward_component = float(reward_do_trade)
                # [BUG 2 FIX] Flag para evitar double counting no compute_scientific_reward
                info["trade_reward_already_computed"] = True

                # Armazena para o log final do step
                self._last_pnl_realized = pnl_realized
                self._last_trade_return_pct = trade_return_pct
                self._last_trade_duration = duration_steps
                self._last_exit_reason = exit_reason

                # [FIX] Log movido para o final da função para mostrar o reward científico real
                pass
                # [X4 FIX] _calculate_sharpe_reward() REMOVIDO do fechamento de trade.
                # scientific_corrections.py já inclui componentes Sortino e Calmar que
                # capturam o sinal de risco-ajustado. Adicionar Sharpe aqui causava
                # double-counting: os últimos 20 trades recebiam o sinal de Sharpe
                # tanto via scientific_corrections (por trade) quanto via esta chamada.
                # O sinal de Sharpe holístico do episódio é preservado no bloco de
                # terminal reward (fim de episódio, linha ~3723).
                # [v3.0 LOPÉZ DE PRADO] Penalidade de duração mínima REMOVIDA.
                # O agente aprende organicamente via gravity well (profundidade ATR) +
                # exit quality (Sharpe × duration) em compute_scientific_reward.
                # Um scalp de 1 candle com +0.04% gera barrier_bonus ≈ +0.05 (quase zero)
                # vs um trade de 50 candles com +2% gerando ≈ +2.4 — sem forçar tempo mínimo.
                # scalp_min e scalp_penalty_base ficam disponíveis mas não são aplicados aqui.
                trade_side = closed_trade_side or (
                    "long"
                    if prev_position_sign > 0
                    else "short"
                    if prev_position_sign < 0
                    else "flat"
                )
                _exit_price = (
                    float(close_price)
                    if "close_price" in locals()
                    else float(current_price)
                )
                info["trade"] = {
                    "pnl": float(pnl_realized),
                    "duration_steps": duration_steps,
                    "side": trade_side,
                    "entry_price": float(self.entry_price),
                    "exit_price": _exit_price,
                }
                # [MATRIX] Registra detalhes do trade para o relatório de fim de episódio
                self._episode_trades_log.append(
                    {
                        "#": self._trades_in_episode,  # já incrementado acima
                        "side": trade_side.upper(),
                        "entry": float(self.entry_price),
                        "exit": _exit_price,
                        "duration": duration_steps,
                        "pnl_usd": float(pnl_realized),
                        "pnl_pct": float(trade_return_pct) * 100.0,
                        "exit_reason": exit_reason or "Agent Decision",
                        "net_worth": float(self.net_worth),
                    }
                )
            else:
                self.disable_new_entries = True
                self.trades_remaining = 0
            if info.get("exit_reason") == "Catastrophe Stop Loss" and pnl_realized < 0:
                # [FIX I] cat_pen apenas para CSL com PERDA.
                # Paralelo ao BUG 3 FIX do Stop Loss: scientific_corrections.py já aplica
                # -0.5 (CSL lucrativo) ou -8.0 (CSL com perda) + _compute_trade_reward.
                # Para CSL lucrativo, cat_pen=5.0 anulava _compute_trade_reward +5.62 e
                # barrier_bonus +2.5 — tornando profitable trades negativos (-4.42 avg).
                cat_pen = 5.0 * warmup_factor
                reward -= cat_pen
                penalty_accum += abs(cat_pen)
            if info.get("exit_reason") == "Stop Loss" and pnl_realized <= 0:
                # [BUG 3 FIX] Penalidade duplicada removida. compute_scientific_reward() ja aplica -4.0.
                # sl_pen = 2.5 * warmup_factor
                # reward -= sl_pen
                # penalty_accum += abs(sl_pen)
                pass
            min_trade_duration = max(1, scalp_min)
            # [BUG 3 FIX] Removida matemática redundante de penalidade por scalp (-37 e -70)
            # O sistema já pune os ganhos minúsculos via Sortino Ratio no scientific_corrections
            # e a penalidade do primeiro bloco em duration_steps < scalp_min é retida via scalp_penalty_base.
            regime = "side"
            if reg_up > 0.5:
                regime = "up"
            elif reg_down > 0.5:
                regime = "down"
            self.episode_trades += 1
            # [FIX ALINHAMENTO] Conta aqui (no fechamento), junto com episode_trades, para sincronismo
            if self._current_trade_aligned is True:
                self._trades_aligned_with_predictor += 1
            elif self._current_trade_aligned is False:
                self._trades_against_predictor += 1
            self._current_trade_aligned = None  # reset para próximo trade
            self.regime_counters[regime]["trades"] += 1
            # Incrementa contadores reais de lado
            try:
                if not hasattr(self, "side_counters"):
                    self.side_counters = {"long": 0.0, "short": 0.0}
                if trade_side == "long":
                    self.side_counters["long"] += 1.0
                elif trade_side == "short":
                    self.side_counters["short"] += 1.0
            except Exception:
                pass
            if isinstance(getattr(self, "_gate_stats", None), Counter):
                close_key = (
                    f"closed_{trade_side}"
                    if trade_side in {"long", "short"}
                    else "closed_unknown"
                )
                self._gate_stats[close_key] += 1
                outcome_key = f"{close_key}_{'win' if pnl_realized > 0 else 'loss'}"
                self._gate_stats[outcome_key] += 1
            if pnl_realized > 0:
                self.episode_wins += 1
                self.regime_counters[regime]["wins"] += 1
            if self._last_entry_prior_conf is not None and not np.isnan(
                self._last_entry_prior_conf
            ):
                self._ece_records.append(
                    (
                        float(np.clip(self._last_entry_prior_conf, 0.0, 1.0)),
                        1 if pnl_realized > 0 else 0,
                    )
                )
            self._last_entry_prior_conf = None
            self.position = 0.0
            self.entry_price = 0.0
            self.initial_notional_value = 0.0
            self.steps_in_position = 0
            self.pnl_since_entry = 0.0
            # [METRICS FIX] Preserva o side antes de zerar p/ CleanMetricsCallback
            self.last_closed_trade_side = int(self.last_action_type)
            self.last_action_type = 0
            self.sl_level = None
            self.time_stop_step = None
            self._psar = None
            self._psar_ep = None
            self._psar_trend = 0
            self.current_leverage = 1.0
            self._active_grace_period = 0
            # Adicionado: Garante que o log final use os valores deste trade antes do reset
            trade_was_closed = True
        else:
            # Se não fechou trade, limpa os buffers do log
            self._last_pnl_realized = 0.0
            self._last_trade_return_pct = 0.0
            self._last_trade_duration = 0
            self._last_exit_reason = None

        # [CORREÇÃO] Bloco de drawdown do training_phase removido.
        # O compute_scientific_reward já aplica delta-DD em ambos os modos perfeitamente.
        pass
        # (Recompensa por expected_qpnl removida) manter apenas penalidades estruturais
        row = self.df.iloc[self.start_idx + self.current_step]
        exp_qpnl = float(row.get("qpnl_24_p50", 0.0))
        # (sem bÃƒÂ´nus por exp_qpnl por step)
        ema_trend_val = self._get_ema_trend_value(row)
        if self.position != 0 and (
            (self.position > 0 and ema_trend_val < 0)
            or (self.position < 0 and ema_trend_val > 0)
        ):
            # Penalidade EMA condicional à lucratividade do trade.
            # Antes: -0.5/step independente de PnL → expulsava trades lucrativos quando EMA virava.
            # Agora: penalidade completa só se trade está perdendo; reduzida a 20% se lucrativo.
            # Ensina: "preocupe-se com misalignment quando está perdendo, não quando está ganhando."
            misalign_scale = 1.0 + abs(ema_trend_val)
            misalign_pen_full = min(0.5, 0.1 * misalign_scale) * warmup_factor
            pnl_now = float(getattr(self, "pnl_since_entry", 0.0))
            if pnl_now >= 0:
                # Trade lucrativo: penalidade suave (20%) — não expulsa winners por EMA
                misalign_pen = misalign_pen_full * 0.2
            else:
                # Trade perdendo: penalidade completa — sinaliza que deve sair
                misalign_pen = misalign_pen_full
            reward -= misalign_pen
            penalty_accum += abs(misalign_pen)
        self.current_step += 1
        # Limite de duraÃƒÂ§ÃƒÂ£o do episÃƒÂ³dio: roda atÃƒÂ© consumir todo o dataset
        quota_exhausted = (
            getattr(self, "disable_new_entries", False)
            and getattr(self, "trades_remaining", 0) == 0
            and self.position == 0
        )
        done = self.current_step >= self.max_steps or quota_exhausted

        # [BUG 4 FIX] Force close at episode end
        # Se um trade foi carregado atÃ© o fim do eval set (surfando a trend sem time stop),
        # precisamos fechar o trade forÃ§adamente e registrar o PnL. Caso contrÃ¡rio, o Optuna
        # dÃ¡ score 0.0 porque nÃ£o houve PnL *realizado*.
        if done and self.position != 0:
            close_price = self._simulate_slippage(
                current_price,
                atr,
                OrderSide.SELL if self.position > 0 else OrderSide.BUY,
            )
            pnl_realized = (
                self.initial_notional_value
                * ((close_price - self.entry_price) / (self.entry_price + 1e-9))
                * np.sign(self.position)
            )
            trade_return_pct = (
                (close_price - self.entry_price)
                / (self.entry_price + 1e-9)
                * np.sign(self.position)
            )

            self._apply_realized_pnl(pnl_realized)
            info["exit_reason"] = "Episode End"

            self.trade_return_history.append(trade_return_pct)
            self._current_episode_returns.append(trade_return_pct)
            self._current_episode_durations.append(self.steps_in_position)
            self._trades_in_episode += 1
            _ep_side = "LONG" if self.position > 0 else "SHORT"
            self._episode_trades_log.append(
                {
                    "#": self._trades_in_episode,
                    "side": _ep_side,
                    "entry": float(self.entry_price),
                    "exit": float(close_price),
                    "duration": self.steps_in_position,
                    "pnl_usd": float(pnl_realized),
                    "pnl_pct": float(trade_return_pct) * 100.0,
                    "exit_reason": "Episode End",
                    "net_worth": float(self.net_worth),
                }
            )
            reward += self._compute_trade_reward(
                pnl_realized,
                trade_return_pct,
                self.steps_in_position,
                "Episode End",
                atr_pct,
            )

            # [ANTI-BIAS] Propaga métricas para compute_scientific_reward
            self._last_pnl_realized = float(pnl_realized)
            self._last_trade_return_pct = float(trade_return_pct)
            self._last_trade_duration = int(self.steps_in_position)
            self._last_exit_reason = "Episode End"

            # [BUG FIX] Episode End trades must receive scientific_reward.
            # Without this, the best trades (surf until episode end) get incomplete
            # reward — agent learns "Agent Decision" > "hold to end", incentivizing
            # premature exits. Exit Quality (+4.0), Sortino (+1.0), Min Hold (+0.70)
            # were all missing for these trades.
            # [FLAG ORDER FIX] A flag deve estar False DURANTE a chamada para que os
            # componentes condicionados (linhas 410/441/626 de scientific_corrections)
            # injetem Exit Quality / Sortino / Min Hold. So apos a chamada marcamos
            # como consumido, evitando double-counting em compute_scientific_reward
            # subsequente do mesmo step.
            _prev_already = info.get("trade_reward_already_computed", False)
            info["trade_reward_already_computed"] = False
            try:
                _sci_ep_end = compute_scientific_reward(
                    self,
                    pnl_realized,
                    trade_return_pct,
                    self.steps_in_position,
                    "Episode End",
                    current_price,
                    atr,
                    prev_price,
                    info,
                )
                reward += _sci_ep_end
            except Exception as _e:
                logger.debug("[Episode End] scientific_reward failed: %s", _e)
            finally:
                info["trade_reward_already_computed"] = True

            # Reset posição
            self.position, self.entry_price = 0.0, 0.0
            self.initial_notional_value, self.steps_in_position = 0.0, 0
            # [METRICS FIX] Preserva o side antes de zerar p/ CleanMetricsCallback
            self.last_closed_trade_side = int(self.last_action_type)
            self.last_action_type = 0
            self.sl_level = None
            self.time_stop_step = None
            self._psar = None
            self._psar_ep = None
        if done:
            logger.debug(
                "[FASE 1 DEBUG] Trades no episodio: %s | cooldown_base=%s",
                self._trades_in_episode,
                getattr(self, "post_trade_cooldown_base", None),
            )
            self._store_episode_summary()
            self._trades_in_episode = 0
            self.trades_remaining = (
                self.max_trades_per_episode
                if self.max_trades_per_episode is not None
                else float("inf")
            )
            self.disable_new_entries = False
            # Removido bÃƒÂ´nus explosivo no final do episÃƒÂ³dio (mantemos apenas logs)
            # reward += ((self.net_worth / self.initial_balance) - 1) * 10
            # Log de mÃƒÂ©tricas de episÃƒÂ³dio
            try:
                _ep_num = len(getattr(self, "_episode_summaries", [])) + 1
                logger.debug(
                    "[%s] [ALINHAMENTO] Ep#%d | Total Trades: %s | A Favor do Predictor: %s | Contra o Predictor: %s",
                    self.__class__.__name__,
                    _ep_num,
                    self.episode_trades,
                    self._trades_aligned_with_predictor,
                    self._trades_against_predictor,
                )
                sharpe_bonus = self._calculate_sharpe_reward()
                reward += sharpe_bonus  # [BUG FIX] sharpe_bonus era calculado mas nunca aplicado ao reward de fim de episódio
                wr = self.episode_wins / (self.episode_trades + 1e-9)
                ece_val = self._compute_ece(self._ece_records, n_bins=10)
                fin = self.get_financial_stats()
                logger.debug(
                    f"[TREND ENV] Fim do episodio | PF: {fin['profit_factor']:.3f} | AvgRet/trade: {fin['avg_return_pct']:.4f} | TradeSharpe: {fin['trade_sharpe']:.3f} | "
                    f"MaxDD: {self.episode_max_drawdown:.4f} | WinRate: {wr:.2%} | ECE: {ece_val:.4f} | "
                    f"WR(up/down/side): {self._reg_wr('up'):.2%}/{self._reg_wr('down'):.2%}/{self._reg_wr('side'):.2%}"
                )
            except Exception:
                pass
        # Limita o impacto lÃƒÂ­quido de penalidades por step durante otimizaÃƒÂ§ÃƒÂ£o
        # [BLOCO REMOVIDO] - O modo de otimizaÃƒÂ§ÃƒÂ£o DEVE usar o reward shaping completo
        # para ser consistente com os objetivos do Optuna.
        # if self.mode == 'optimization':
        #     step_return = (self.net_worth - prev_net_worth) / (self.initial_balance + 1e-9)
        #     reward = float(np.clip(step_return, -1.0, 1.0))
        #     penalty_accum = 0.0
        if penalties_debug:
            penalties_payload = {
                key: float(val) for key, val in penalties_debug.items()
            }
            info.setdefault("penalties", {}).update(penalties_payload)
        if self.mode == "optimization" and penalty_accum > penalty_cap_per_step:
            # Reescala a parcela punitiva para o teto
            scale = penalty_cap_per_step / (penalty_accum + 1e-9)
            # Aproxima: aplica escala ao reward negativo acumulado ao longo do step
            if reward < 0:
                # Aplica a escala apenas ao valor negativo acumulado, sem substituir PnL
                reward *= float(np.clip(scale, 0.0, 1.0))
        # [SCIENTIFIC] Recompensa mestra calculada ao final de cada step.
        # Isso processa PnL (realizado se fechou, senão não realizado), Delta Drawdown,
        # Hold Bonus e Opportunity Cost em uma única chamada Markov-compliant.

        # Prepara info para masking suave e custos de oportunidade
        # [BUG 4 FIX] Mismatch de Regime Otimizado para FLAT
        # Só dispara quando o agente ESTÁ posicionado errado ou TENTANDO abrir errado.
        # Evita punir "pensamentos" (votos) que não resultam em ação durante exploração.
        is_mismatch = False
        _spec_name = str(getattr(self, "specialist_name", "")).lower()

        if "bull" in _spec_name:
            if self.position < 0:  # Short em bull
                is_mismatch = True
            elif (
                should_open_short and agent_vote_strength < -0.1
            ):  # Tentando abrir short em bull
                is_mismatch = True
        elif "bear" in _spec_name:
            if self.position > 0:  # Long em bear
                is_mismatch = True
            elif (
                should_open_long and agent_vote_strength > 0.1
            ):  # Tentando abrir long em bear
                is_mismatch = True

        info.update(
            {
                "prior_dir_val": prior_dir_val,
                "prior_conf_val": prior_conf_dyn,
                "_regime_mismatch": bool(
                    is_mismatch or info.get("_regime_mismatch", False)
                ),
            }
        )

        # Se o trade fechou AGORA, passamos os dados dele. Senão, 0.0.
        pnl_to_pass = float(getattr(self, "_last_pnl_realized", 0.0))
        ret_to_pass = float(getattr(self, "_last_trade_return_pct", 0.0))
        dur_to_pass = int(getattr(self, "_last_trade_duration", 0))
        ext_to_pass = getattr(self, "_last_exit_reason", None)

        # Chamada única do motor de recompensa
        # [BUG FIX] ADD scientific reward instead of overwriting it!
        # This preserves penalties for direction misalignment, flips, etc.
        scientific_reward = compute_scientific_reward(
            env=self,
            pnl_realized=pnl_to_pass,
            trade_return_pct=ret_to_pass,
            duration_steps=dur_to_pass,
            exit_reason=ext_to_pass,
            current_price=current_price,
            atr=atr,
            prev_price=prev_price,
            info=info,
        )
        reward += scientific_reward
        # [FIX REWARD #2] Salva componente científico para o log de breakdown
        if trade_was_closed:
            self._last_scientific_reward_component = float(scientific_reward)

        if trade_was_closed:
            # [PERF CONSOLE] Trade details vão apenas para arquivo (debug), não para tela
            # [FIX REWARD #2] Log de breakdown do reward para diagnóstico de discrepâncias PnL vs Reward
            _last_pnl = getattr(self, "_last_pnl_realized", 0.0)
            _last_ret = getattr(self, "_last_trade_return_pct", 0.0)
            _last_dur = getattr(self, "_last_trade_duration", 0)
            _last_reason = getattr(self, "_last_exit_reason", "Unknown")
            _trade_reward_part = getattr(
                self, "_last_trade_reward_component", float("nan")
            )
            _sci_reward_part = getattr(
                self, "_last_scientific_reward_component", float("nan")
            )
            logger.debug(
                f"TRADE CLOSED | Reason: {_last_reason} | "
                f"Duration: {_last_dur} steps | "
                f"PnL: {_last_pnl:.2f} | "
                f"Return: {_last_ret:.2%} | "
                f"TradeRwd: {_trade_reward_part:.4f} | "
                f"SciRwd: {_sci_reward_part:.4f} | "
                f"Final Reward: {reward:.4f}"
            )

        self._episode_reward_accumulator += float(reward)
        self._prev_net_worth = self.net_worth

        # 🔬 SISTEMA 3 CAMADAS: Ruin Buffer Penalty
        # [BUG 8 FIX] Movido para ANTES da normalização para manter range simétrico.
        # Antes: ruin_penalty era somado AO reward_normalized [-10,+10], criando range [-20,+10].
        # O critic SAC aprendia que o espaço negativo era 2x maior → conservadorismo extremo.
        capital_ratio = self.net_worth / (self.initial_balance + 1e-9)
        ruin_buffer_threshold = getattr(self, "ruin_buffer_threshold", 0.80)
        ruin_threshold = getattr(self, "ruin_threshold", 0.50)

        if capital_ratio < ruin_buffer_threshold:
            # Penalidade progressiva: até -10 quando chega em ruin_threshold
            ruin_penalty = (
                -10.0
                * (ruin_buffer_threshold - capital_ratio)
                / (ruin_buffer_threshold - ruin_threshold + 1e-9)
            )
            reward += ruin_penalty  # Adicionado AO reward bruto, antes da normalização
            logger.debug(
                "[RUIN BUFFER] capital_ratio=%.2f | penalty=%.2f",
                capital_ratio,
                ruin_penalty,
            )

        # [BUG 5 FIX] Normalização interna Welford DESATIVADA.
        # O VecNormalize externo (norm_reward=True) já normaliza o reward de forma estável.
        # A normalização dupla transformava o sinal em ruído puro:
        #   1ª camada (Welford): instável nos primeiros ~200 steps (running_std ≈ 1.0 artificial)
        #   2ª camada (VecNormalize): recebe sinal já distorcido e aplica outra transformação
        # Resultado: SAC recebia gradientes completamente descorrelacionados do PnL real.
        reward_normalized = float(
            np.clip(reward, self._reward_norm_clip_neg, self._reward_norm_clip_pos)
        )

        # 🔬 SISTEMA 3 CAMADAS: Episódio Terminal por Ruína
        if capital_ratio < ruin_threshold:
            done = True
            reward_normalized = self._reward_norm_clip_neg  # Penalidade máxima
            info["exit_reason"] = "Ruin (Capital < 50%)"
            logger.warning(
                "[RUIN TERMINAL] Episódio encerrado por ruína. Capital ratio: %.2f",
                capital_ratio,
            )

        return self._get_observation(), np.float32(reward_normalized), done, False, info


class OnlineFeatureCalculator:
    """

    [VERSÃƒÆ'O FINAL CORRIGIDA]
    Calcula o conjunto completo de features online, garantindo consistÃƒÂªncia
    com a pipeline de feature engineering e evitando lookahead bias.
    """

    def __init__(self, full_df: pd.DataFrame):
        base_cols = ["open", "high", "low", "close", "volume"]
        if not all(c in full_df.columns for c in base_cols):
            raise ValueError(
                f"Ã°Å¸Å¡Â¨ Faltam colunas base em full_df: {set(base_cols) - set(full_df.columns)}"
            )
        self.df_raw = full_df[base_cols].copy()
        self._cache: Dict[int, np.ndarray] = {}
        self.min_history = (
            200  # HistÃƒÂ³rico mÃƒÂ­nimo para indicadores longos (ex: EMA 200)
        )
        # Este contrato DEVE ser mantido em sincronia com NativeIndicators.calculate_all_features
        self.feature_names: List[str] = [
            "ema_12",
            "ema_26",
            "ema_50",
            "ema_200",
            "rsi",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "cci",
            "roc",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_width",
            "atr",
            "atr_percentage",
            "obv",
            "vwap",
            "adx",
            "plus_di",
            "minus_di",
            "cmf",
            "pvt",
            "adl",
        ]
        self.num_features = len(self.feature_names)
        logger.info(
            f"[FEATURE CALC] OnlineFeatureCalculator inicializado com {self.num_features} features."
        )

    def calculate_features_at(self, step: int) -> np.ndarray:
        if step in self._cache:
            return self._cache[step]
        # O ÃƒÂ­ndice final para o slice ÃƒÂ© `step + 1` para incluir o candle atual
        end_idx = int(step) + 1
        hist = self.df_raw.iloc[:end_idx]
        if len(hist) < self.min_history:
            vec = np.zeros(self.num_features, dtype=np.float32)
            self._cache[step] = vec
            return vec
        # Pega a ÃƒÂºltima linha de cada sÃƒÂ©rie de indicador
        close = hist["close"]
        high = hist["high"]
        low = hist["low"]
        volume = hist["volume"]
        # Extrai o ÃƒÂºltimo valor de cada indicador
        vals = [
            NativeIndicators.ema(close, 12).iloc[-1],
            NativeIndicators.ema(close, 26).iloc[-1],
            NativeIndicators.ema(close, 50).iloc[-1],
            NativeIndicators.ema(close, 200).iloc[-1],
            NativeIndicators.rsi(close, 14).iloc[-1],
            NativeIndicators.macd(close)[0].iloc[-1],
            NativeIndicators.macd(close)[1].iloc[-1],
            NativeIndicators.macd(close)[2].iloc[-1],
            NativeIndicators.stochastic_oscillator(high, low, close)[0].iloc[-1],
            NativeIndicators.stochastic_oscillator(high, low, close)[1].iloc[-1],
            NativeIndicators.williams_r(high, low, close, 14).iloc[-1],
            NativeIndicators.cci(high, low, close, 20).iloc[-1],
            NativeIndicators.roc(close, 12).iloc[-1],
            NativeIndicators.bollinger_bands(close, 20, 2.0)[0].iloc[-1],
            NativeIndicators.bollinger_bands(close, 20, 2.0)[1].iloc[-1],
            NativeIndicators.bollinger_bands(close, 20, 2.0)[2].iloc[-1],
            (
                NativeIndicators.bollinger_bands(close, 20, 2.0)[0].iloc[-1]
                - NativeIndicators.bollinger_bands(close, 20, 2.0)[2].iloc[-1]
            )
            / (NativeIndicators.bollinger_bands(close, 20, 2.0)[1].iloc[-1] + 1e-9),
            NativeIndicators.atr(high, low, close, 14).iloc[-1],
            (
                NativeIndicators.atr(high, low, close, 14).iloc[-1]
                / (close.iloc[-1] + 1e-9)
            )
            * 100.0,
            NativeIndicators.obv(close, volume).iloc[-1],
            NativeIndicators.vwap(high, low, close, volume).iloc[-1],
            NativeIndicators.adx(high, low, close, 14)[0].iloc[-1],
            NativeIndicators.adx(high, low, close, 14)[1].iloc[-1],
            NativeIndicators.adx(high, low, close, 14)[2].iloc[-1],
            NativeIndicators.calculate_money_flow_volume_pressure(
                high, low, close, volume
            )["cmf"].iloc[-1],
            NativeIndicators.calculate_money_flow_volume_pressure(
                high, low, close, volume
            )["pvt"].iloc[-1],
            NativeIndicators.calculate_money_flow_volume_pressure(
                high, low, close, volume
            )["adl"].iloc[-1],
        ]
        vec = np.array(
            [0.0 if pd.isna(x) else float(x) for x in vals], dtype=np.float32
        )
        self._cache[step] = vec
        return vec

    @staticmethod
    def _compute_ece(records: List[Tuple[float, int]], n_bins: int = 10) -> float:
        if not records:
            return 0.0
        confs = np.array([c for c, _ in records])
        outs = np.array([o for _, o in records])
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total = len(confs)
        for i in range(n_bins):
            left, right = bins[i], bins[i + 1]
            mask = (
                (confs >= left) & (confs < right)
                if i < n_bins - 1
                else (confs >= left) & (confs <= right)
            )
            if not np.any(mask):
                continue
            bin_conf = float(np.mean(confs[mask]))
            bin_acc = float(np.mean(outs[mask]))
            ece += (np.sum(mask) / total) * abs(bin_acc - bin_conf)
        return float(ece)


class TrendSpecialist:
    def _filter_to_core_features(self, all_features: List[str]) -> List[str]:
        """Reduz features ao conjunto essencial — v3.1 (López de Prado / SAC convergence)."""
        core_indicators = [
            "log_return",
            "hl_range_pct",
            "close_frac",
            "realized_vol",
            "rsi",
            "macd_hist",
            "macd_cross",
            "adx",
            "plus_di",
            "minus_di",
            "bb_width",
            "atr_percentage",
            "bb_squeeze",
            "bb_expansion",
            "rsi_oversold",
            "rsi_overbought",
            "money_flow",
            "volume_pressure",
            "ema_trend",
            "psar_trend",
            "hidden_feature",
            "sdae_recon",
            "regime_conf",
            "tp_prior",
        ]
        # Preserva todas as latents e meta-features
        filtered = [f for f in all_features if any(ind in f for ind in core_indicators)]
        # Filtra por timeframes específicos para reduzir ruído (15m para precisão, 1h para tendência)
        # Mantém latents independentemente do timeframe
        final_cols = []
        for col in filtered:
            if (
                "hidden_feature" in col
                or "sdae_recon" in col
                or "regime_conf" in col
                or "tp_prior" in col
            ):
                final_cols.append(col)
            elif "_15m" in col or "_1h" in col:
                final_cols.append(col)
        return final_cols

    """

    [CORRIGIDO] Agente especialista em tendÃªncia, utilizando SAC para
    maior compatibilidade com espaÃ§os de aÃ§Ã£o contÃ­nuos.
    """

    SAC_PARAM_KEYS: Set[str] = {
        "learning_rate",
        "buffer_size",
        "batch_size",
        "gamma",
        "tau",
        "train_freq",
        "use_sde",
        "ent_coef",
        "target_entropy",
    }
    REWARD_PARAM_KEYS: Tuple[str, ...] = (
        "TREND_UNREALIZED_REWARD_SCALE",
        "TREND_REALIZED_REWARD_SCALE",
        "TREND_DRAWDOWN_PENALTY_WEIGHT",
        "TREND_PRIOR_FLIP_PENALTY_WEIGHT",
        "TREND_PRIOR_FLIP_PENALTY_EXP",
        "TREND_PRIOR_FLIP_THRESHOLD",
        "TREND_DURATION_PRESSURE_WEIGHT",
        "TREND_DURATION_PRESSURE_START",
        "TREND_DURATION_PRESSURE_EXP",
    )
    REWARD_ATTR_MAP: Dict[str, str] = {
        "TREND_UNREALIZED_REWARD_SCALE": "unrealized_reward_scale",
        "TREND_REALIZED_REWARD_SCALE": "realized_reward_scale",
        "TREND_DRAWDOWN_PENALTY_WEIGHT": "drawdown_penalty_weight",
        "TREND_PRIOR_FLIP_PENALTY_WEIGHT": "prior_flip_penalty_weight",
        "TREND_PRIOR_FLIP_PENALTY_EXP": "prior_flip_penalty_exponent",
        "TREND_PRIOR_FLIP_THRESHOLD": "prior_flip_threshold",
        "TREND_DURATION_PRESSURE_WEIGHT": "duration_pressure_weight",
        "TREND_DURATION_PRESSURE_START": "duration_pressure_start",
        "TREND_DURATION_PRESSURE_EXP": "duration_pressure_exponent",
    }

    def __init__(
        self,
        config: AIConfig,
        input_dim: int,
        profitability_predictor: Optional[Any] = None,
        specialist_name: str = "trend_specialist",
    ):
        if not isinstance(config, AIConfig) or input_dim <= 0:
            raise TypeError("🚨 [ERRO TREND] Configuração ou input_dim inválido.")
        self.config = config
        self.trading_config = TradingConfig()
        self.input_dim = input_dim
        self.profitability_predictor = profitability_predictor
        self.specialist_name = specialist_name
        self.feature_scaler = None
        self.feature_columns: List[str] = []

        # [FIX] Logger de instância com nome dinâmico da classe
        self.logger = get_logger(self.__class__.__name__, LOG_LEVEL_DEBUG)

        self.model_path = os.path.join(
            self.config.MODEL_DIR, f"{specialist_name}_sac.zip"
        )
        max_grad_env = os.getenv("TREND_SAC_MAX_GRAD_NORM", "1.0").strip()
        try:
            parsed_grad_norm = float(max_grad_env)
        except ValueError:
            parsed_grad_norm = 1.0
        self.max_grad_norm = parsed_grad_norm if parsed_grad_norm > 0 else None
        self.model: Optional[ClippedSAC] = None
        self.is_trained = False
        self.log_dir = os.path.join(os.getcwd(), "logs")
        self.hyperparams_path = os.path.join(
            self.log_dir, f"{specialist_name}_sac_hyperparams.json"
        )
        self.optuna_db_path = os.path.join(
            self.log_dir, f"{specialist_name}_sac_optuna.db"
        )
        self.reward_params_path = os.path.join(
            self.log_dir, f"{specialist_name}_reward_params.json"
        )
        self.hyperparams: Dict[str, Any] = {}
        self._hyperparams_loaded_from_file = False
        env_force = os.environ.get("TREND_FORCE_OPTIMIZE", "0").strip().lower()
        self.force_optuna_retrain = env_force in {"1", "true", "yes", "y", "on"}
        env_reset = os.environ.get("TREND_RESET_OPTUNA_DB", "0").strip().lower()
        self.reset_optuna_db = env_reset in {"1", "true", "yes", "y", "on"}
        self.reward_hyperparams: Dict[str, Any] = self._load_reward_params()
        if self.reward_hyperparams:
            self._apply_reward_params_to_config(self.reward_hyperparams)

        # [MONITOR DE APRENDIZADO] Instancia o LearningMonitor para acompanhar convergencia
        self.learning_monitor = LearningMonitor(
            config=self.config,
            specialist_name=specialist_name,
            log_dir=os.path.join(self.log_dir, "learning_monitor"),
            save_json=True,
        )

        self.action_space = spaces.Box(
            low=np.array(
                [
                    -1.0,
                    self.trading_config.SL_DIST_MULTIPLIER_MIN,
                    self.trading_config.MIN_LEVERAGE_PER_TRADE,
                ]
            ),
            high=np.array(
                [
                    1.0,
                    self.trading_config.SL_DIST_MULTIPLIER_MAX,
                    self.trading_config.MAX_LEVERAGE_PER_TRADE,
                ]
            ),
            dtype=np.float32,
        )
        self._ema_trend_key: Optional[str] = None

    def set_feature_scaler(self, scaler):
        self.feature_scaler = scaler
        self._load_hyperparams()
        self._initialize_model()

    @property
    def is_optimized(self) -> bool:
        """
        Verifica se os hiperparâmetros foram otimizados via Optuna.
        Segue o padrão do autoencoder: otimizar primeiro, depois treinar.

        Returns:
            True se hiperparâmetros foram carregados de arquivo (já otimizados)
        """
        return self._hyperparams_loaded_from_file and os.path.exists(
            self.hyperparams_path
        )

    def _make_trend_env(
        self,
        df: pd.DataFrame,
        *,
        mode: Optional[str] = "training",
        feature_columns: Optional[List[str]] = None,
        env_class=None,
        specialist_name: Optional[str] = None,
    ) -> TrendFollowingEnv:
        """
        Helper para criar TrendFollowingEnv com compatibilidade retroativa.
        """
        env_class = env_class or TrendFollowingEnv
        kwargs: Dict[str, Any] = {
            "df": df,
            "config": self.config,
            "profitability_predictor": self.profitability_predictor,
            "feature_scaler": getattr(self, "feature_scaler", None),
            "feature_columns": getattr(self, "cached_feature_columns", None),
            "specialist_name": specialist_name
            or getattr(self, "specialist_name", "TrendSpecialist"),
        }
        if feature_columns is not None:
            kwargs["feature_columns"] = feature_columns
        if mode is not None:
            kwargs["mode"] = mode
        try:
            env = env_class(**kwargs)
            return self._post_process_env(env)
        except TypeError as exc:
            if "mode" in str(exc):
                kwargs.pop("mode", None)
                logger.warning(
                    f"[TREND SAC] {env_class.__name__} sem suporte a 'mode'. Recriando sem o parâmetro para compatibilidade."
                )
                env = env_class(**kwargs)
                return self._post_process_env(env)
            raise

    def _get_device(self) -> str:
        """Detecta e retorna o melhor device disponível, com logging claro."""
        import torch

        self.logger.warning(
            f"[DEBUG GPU] torch version: {getattr(torch, '__version__', 'unknown')}"
        )
        self.logger.warning(
            f"[DEBUG GPU] torch.cuda.is_available(): {torch.cuda.is_available()}"
        )
        import os

        self.logger.warning(
            f"[DEBUG GPU] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')}"
        )
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.warning(
                    f"🚀 [GPU] Usando GPU: {gpu_name} (cuda:0) para treinamento SAC."
                )
            except Exception as e:
                self.logger.warning(f"Failed to get device name: {e}")
                pass
            return "cuda:0"

        # Test additional things
        try:
            self.logger.warning(
                f"[DEBUG GPU] device count: {torch.cuda.device_count()}"
            )
        except Exception as e:
            self.logger.warning(f"[DEBUG GPU] failed getting count: {e}")

        self.logger.warning(
            "⚠️ [GPU] CUDA não disponível! Treinamento rodará em CPU (muito mais lento). "
            "Verifique sua instalação do PyTorch com suporte CUDA."
        )
        return "cpu"

    def _default_hyperparams(self) -> Dict[str, Any]:
        return {
            "learning_rate": 4.6e-5,
            "buffer_size": 50000,
            "batch_size": 1024,
            "gamma": 0.995,
            "tau": 0.005,
            "train_freq": 16,
            "gradient_steps": 64,
            "target_entropy": -3.0,
            "use_sde": True,
            "ent_coef": "auto_0.5",
        }

    def _load_hyperparams(self):
        if os.path.exists(self.hyperparams_path):
            try:
                with open(self.hyperparams_path, "r") as handle:
                    self.hyperparams = json.load(handle)
                self._hyperparams_loaded_from_file = True
                logger.info(
                    f"[TREND SAC] Hiperparametros carregados de '{self.hyperparams_path}'."
                )
            except Exception:
                self.hyperparams = {}
                self._hyperparams_loaded_from_file = False
        else:
            logger.info("[TREND SAC] Arquivo de hiperparametros nao encontrado.")
            self._hyperparams_loaded_from_file = False
        if self.force_optuna_retrain:
            if os.path.exists(self.hyperparams_path):
                backup_path = self.hyperparams_path + ".bak"
                try:
                    shutil.copy2(self.hyperparams_path, backup_path)
                    logger.info(
                        f"[TREND SAC] Backup dos hiperparametros antigos criado em '{backup_path}'."
                    )
                except Exception as exc:
                    logger.warning(
                        f"[TREND SAC] Nao foi possivel criar backup dos hiperparametros antigos: {exc}"
                    )
            self.hyperparams = {}
            self._hyperparams_loaded_from_file = False
            logger.info(
                "[TREND SAC] Otimizacao via Optuna sera forcada nesta execucao."
            )
        if not self.hyperparams and not self.force_optuna_retrain:
            self.hyperparams = self._default_hyperparams()
            self._hyperparams_loaded_from_file = False

    def _save_hyperparams(self):
        try:
            with open(self.hyperparams_path, "w") as f:
                json.dump(self.hyperparams, f, indent=4)
            logger.info("[TREND SAC] Hiperparametros salvos.")
        except Exception as e:
            logger.error(
                "[ERRO TREND SAC] Falha ao salvar hiperparametros: %s" % e,
                exc_info=True,
            )

    def _save_anti_scalping_config(self):
        """Salva a configuraÃƒÂ§ÃƒÂ£o anti-scalping otimizada"""
        try:
            config_path = os.path.join(
                self.log_dir, "trend_specialist_anti_scalping_config.json"
            )
            payload = dict(self.anti_scalping_config)
            payload["config_version"] = ANTI_SCALPING_CONFIG_VERSION
            with open(config_path, "w") as f:
                json.dump(payload, f, indent=4)
            logger.info("[FASE 1] Configuracao anti-scalping salva.")
        except Exception as e:
            logger.error(
                "[ERRO FASE 1] Falha ao salvar configuraÃƒÂ§ÃƒÂ£o anti-scalping: %s"
                % e,
                exc_info=True,
            )

    def _load_anti_scalping_config(self):
        """Carrega a configuraÃƒÂ§ÃƒÂ£o anti-scalping otimizada"""
        try:
            config_path = os.path.join(
                self.log_dir, "trend_specialist_anti_scalping_config.json"
            )
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
                version = loaded_config.get("config_version")
                # MigraÃƒÂ§ÃƒÂ£o: se nÃƒÂ£o houver versÃƒÂ£o, mas as chaves obrigatÃƒÂ³rias existirem, aceitar e atualizar o arquivo
                if version is None:
                    missing_keys = [
                        key
                        for key in ANTI_SCALPING_REQUIRED_KEYS
                        if key not in loaded_config
                    ]
                    if not missing_keys:
                        logger.info(
                            "[FASE 1] Configuracao anti-scalping encontrada sem versao. Migrando para versao %s.",
                            ANTI_SCALPING_CONFIG_VERSION,
                        )
                        self.anti_scalping_config = {
                            key: loaded_config[key]
                            for key in ANTI_SCALPING_REQUIRED_KEYS
                        }
                        # Persiste com a versÃƒÂ£o atual
                        try:
                            payload = dict(self.anti_scalping_config)
                            payload["config_version"] = ANTI_SCALPING_CONFIG_VERSION
                            with open(config_path, "w") as f_out:
                                json.dump(payload, f_out, indent=4)
                        except Exception:
                            pass
                        return True
                    # Se faltar chaves e nÃƒÂ£o houver versÃƒÂ£o, reotimiza
                    logger.info(
                        "[FASE 1] Configuracao anti-scalping incompleta e sem versao (faltando %s). Sera reotimizada.",
                        ", ".join(missing_keys),
                    )
                    return False
                if version != ANTI_SCALPING_CONFIG_VERSION:
                    logger.info(
                        "[FASE 1] Configuracao anti-scalping desatualizada (versao %s). Sera reotimizada.",
                        version,
                    )
                    return False
                missing_keys = [
                    key
                    for key in ANTI_SCALPING_REQUIRED_KEYS
                    if key not in loaded_config
                ]
                if missing_keys:
                    logger.info(
                        "[FASE 1] Configuracao anti-scalping incompleta (faltando %s). Sera reotimizada.",
                        ", ".join(missing_keys),
                    )
                    return False
                self.anti_scalping_config = {
                    key: loaded_config[key] for key in ANTI_SCALPING_REQUIRED_KEYS
                }
                logger.info("[FASE 1] Configuracao anti-scalping carregada.")
                return True
        except Exception as e:
            logger.warning(
                "[FASE 1] Falha ao carregar configuracao anti-scalping: %s" % e
            )
        return False

    def _extract_ema_trend_value(self, row: pd.Series) -> float:
        """
        Resolve dinamicamente qual coluna de EMA de tendencia deve ser usada.
        Mantem um cache da coluna selecionada para evitar buscas repetidas.
        """
        if row is None:
            return 0.0
        cached_key = getattr(self, "_ema_trend_key", None)
        if cached_key and cached_key in row.index:
            try:
                return float(row.get(cached_key, 0.0))
            except Exception:
                return 0.0
        candidates = [
            f"ema_trend_{self.trading_config.PRIMARY_TIMEFRAME_TRADING}",
            "ema_trend",
        ]
        candidates.extend(
            col
            for col in row.index
            if isinstance(col, str) and col.startswith("ema_trend_")
        )
        for candidate in candidates:
            if candidate in row.index:
                self._ema_trend_key = candidate
                try:
                    return float(row.get(candidate, 0.0))
                except Exception:
                    return 0.0
        self._ema_trend_key = None
        return 0.0

    def _calculate_average_metrics(
        self, summaries: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Consolida metricas coletadas ao longo de varios episodios.
        Utiliza media ponderada por numero de trades quando aplicavel para refletir melhor o comportamento.
        """
        if not summaries:
            return {}

        def _safe_float(value: Any) -> Optional[float]:
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(result):
                return None
            return result

        trades_per_summary: List[float] = []
        total_trades = 0.0
        total_trend_samples = 0.0
        for summary in summaries:
            trades_val = _safe_float(summary.get("num_trades", 0.0)) or 0.0
            trades_per_summary.append(trades_val)
            total_trades += trades_val
            trend_samples_val = (
                _safe_float(summary.get("trend_regime_samples", 0.0)) or 0.0
            )
            total_trend_samples += trend_samples_val

        aggregated: Dict[str, float] = {}
        keys = set().union(*(summary.keys() for summary in summaries))
        for key in keys:
            values: List[float] = []
            for summary in summaries:
                value = _safe_float(summary.get(key))
                if value is not None:
                    values.append(value)
            if not values:
                continue

            if key in {"win_rate_pct", "avg_trade_duration"} and total_trades > 0:
                weighted_sum = 0.0
                weight_total = 0.0
                for summary, trades_val in zip(summaries, trades_per_summary):
                    value = _safe_float(summary.get(key))
                    if value is None or trades_val <= 0:
                        continue
                    weighted_sum += value * trades_val
                    weight_total += trades_val
                aggregated[key] = (
                    float(weighted_sum / weight_total)
                    if weight_total > 0
                    else float(np.mean(values))
                )
            else:
                aggregated[key] = float(np.mean(values))

        if total_trades > 0:
            aggregated["num_trades"] = aggregated.get(
                "num_trades", float(total_trades / len(summaries))
            )
        aggregated["total_trades"] = float(total_trades)
        aggregated["trend_regime_samples"] = float(total_trend_samples)
        aggregated["episodes"] = float(len(summaries))
        return aggregated

    def _get_autoencoder_latent_dim(self) -> int:
        """
        Determina dimensão das features latentes do AutoEncoder.

        Ordem de prioridade:
        1. Config explícita (se existir)
        2. Inspeção do observation space do environment
        3. Fallback seguro: 64

        Returns:
            int: Dimensão das features latentes
        """
        try:
            # 1. Tenta pegar do config
            if hasattr(self.config, "AUTOENCODER_LATENT_DIM"):
                latent_dim = int(self.config.AUTOENCODER_LATENT_DIM)
                logger.info(f"[LSTM CONFIG] latent_dim do config: {latent_dim}")
                return latent_dim

            # 2. Tenta inspecionar environment
            if (
                hasattr(self, "train_df")
                and self.train_df is not None
                and len(self.train_df) > 100
            ):
                sample_df = self.train_df.iloc[:100]
                env = self._make_trend_env(sample_df, mode="training")
                obs_shape = env.observation_space.shape[0]

                # Subtrai features conhecidas:
                # extras (depende, mas ~10) + agent(5) + regime(3) + prior(2) = ~20
                # Assume que o resto são features de mercado (latents + indicadores)
                # Conservador: assume 60% são latents
                estimated_market_features = obs_shape - 20
                latent_dim = max(32, min(128, int(estimated_market_features * 0.6)))

                logger.info(
                    f"[LSTM CONFIG] latent_dim inferido do env: {latent_dim} (obs_shape={obs_shape})"
                )
                return latent_dim

        except Exception as e:
            logger.warning(
                f"[LSTM CONFIG] Não foi possível determinar latent_dim automaticamente: {e}"
            )

        # 3. Fallback seguro
        logger.info("[LSTM CONFIG] Usando latent_dim padrão: 64")
        return 64

    def _calculate_trend_surfing_score(self, metrics: Dict[str, float]) -> float:
        """

        Calcula score otimizado para surf de tendencias.
        Foca em: poucos trades, duracao adequada, boa performance financeira.
        """

        try:
            num_trades = metrics.get("num_trades", 0)
            avg_trade_duration = metrics.get("avg_trade_duration", 0)
            win_rate = metrics.get("win_rate_pct", 0)
            profit_factor = metrics.get("profit_factor", 0)
            trade_sharpe = metrics.get("trade_sharpe", 0)
            max_drawdown = metrics.get("max_drawdown_pct", 100)
            # exigÃªncia mÃ­nima de participaÃ§Ã£o
            trade_target_min, trade_target_max = 5, 60
            max_trades_cap = (
                float(metrics.get("max_trades_cap", trade_target_max))
                or trade_target_max
            )
            if max_trades_cap > 0:
                trade_target_max = min(trade_target_max, max_trades_cap)
                trade_target_min = min(trade_target_min, trade_target_max)
            # mÃ­nimo anti-scalping
            duration_target_min = 10
            base_score = trade_sharpe * 0.3
            win_rate_bonus = (win_rate - 50) / 100 * 0.2
            profit_factor_capped = float(np.clip(profit_factor, 0.0, 4.0))
            pf_reliability = float(np.clip(metrics.get("pf_reliable", 1.0), 0.0, 1.0))
            if pf_reliability < 0.5:
                profit_factor_capped = min(profit_factor_capped, 1.2)
            profit_factor_effective = profit_factor_capped * (
                0.5 + 0.5 * pf_reliability
            )
            pf_bonus = max(0.0, (profit_factor_effective - 1.0) * 0.4)
            dd_penalty = max(0.0, (max_drawdown - 12.0)) * 0.08
            neg_sharpe_penalty = max(0.0, -trade_sharpe) * 0.6
            # banda de trades
            trade_penalty = 0.0
            if trade_target_min <= num_trades <= trade_target_max:
                midpoint = (trade_target_min + trade_target_max) / 2.0
                spread = max(1.0, midpoint - trade_target_min)
                closeness = 1.0 - abs(num_trades - midpoint) / spread
                trade_bonus = 0.12 * max(0.0, closeness)
            elif num_trades < trade_target_min:
                # Nao penalizar baixa frequencia; levemente positivo por selecao
                trade_penalty = 0.0
                trade_bonus = 0.06
            else:
                excess = num_trades - trade_target_max
                trade_penalty = min(12.0, 0.02 * excess)
                trade_bonus = -trade_penalty
            # NOVA LÃ“GICA DE DURAÃ‡ÃƒO: sem teto; recompensa crescente para duraÃ§Ãµes mais longas
            duration_bonus = 0.0
            if avg_trade_duration < duration_target_min:
                deficit = duration_target_min - avg_trade_duration
                duration_penalty = min(10.0, 0.4 * deficit)
                duration_bonus = -duration_penalty
            else:
                bonus_scale = (avg_trade_duration - duration_target_min) / 100.0
                duration_bonus = min(2.5, 0.35 + bonus_scale * 0.5)
            early_exit_penalty = (
                max(0.0, duration_target_min - avg_trade_duration) * 0.06
            )
            trades_per_1000 = float(metrics.get("trades_per_1000", 0.0))
            if trades_per_1000 > 400.0:
                return -8.0
            density_penalty = max(0.0, trades_per_1000 - 40.0) * 0.05
            # alinhamento com regime dominante
            trend_up_avg = float(metrics.get("trend_regime_avg_up", 0.0))
            trend_down_avg = float(metrics.get("trend_regime_avg_down", 0.0))
            trade_up_pct = float(metrics.get("regime_trade_up_pct", 0.0))
            trade_down_pct = float(metrics.get("regime_trade_down_pct", 0.0))
            dominant_up = trend_up_avg >= trend_down_avg
            if dominant_up:
                align_ratio = trade_up_pct / max(1.0, trade_up_pct + trade_down_pct)
            else:
                align_ratio = trade_down_pct / max(1.0, trade_up_pct + trade_down_pct)
            alignment_bonus = 2.0 * float(np.clip(align_ratio, 0.0, 1.0)) - 0.8
            counter_penalty = 0.0
            if dominant_up and trade_down_pct > 30.0:
                counter_penalty = min(2.0, (trade_down_pct - 30.0) * 0.03)
            if (not dominant_up) and trade_up_pct > 30.0:
                counter_penalty = min(2.0, (trade_up_pct - 30.0) * 0.03)
            final_score = (
                base_score
                + win_rate_bonus
                + pf_bonus
                + trade_bonus
                + duration_bonus
                + alignment_bonus
                - dd_penalty
                - trade_penalty
                - density_penalty
                - early_exit_penalty
                - neg_sharpe_penalty
                - counter_penalty
            )
            meets_trade_band = trade_target_min <= num_trades <= trade_target_max
            if meets_trade_band:
                final_score += 1.2
            logger.debug(
                "[FASE 1] Score components -> base=%.3f, win_bonus=%.3f, pf_bonus=%.3f, "
                "trade_bonus=%.3f, duration_bonus=%.3f, dd_penalty=%.3f, "
                "trade_penalty=%.3f, early_penalty=%.3f, density_pen=%.3f, neg_sharpe=%.3f, pf_rel=%.2f",
                base_score,
                win_rate_bonus,
                pf_bonus,
                trade_bonus,
                duration_bonus,
                dd_penalty,
                trade_penalty,
                early_exit_penalty,
                density_penalty,
                neg_sharpe_penalty,
                pf_reliability,
            )
            return float(np.clip(final_score, -25.0, 15.0))
        except Exception as e:
            logger.error(f"[FASE 1] Erro ao calcular score: {e}")
            return -5.0

    def _calculate_cooldown_effectiveness(self, metrics: Dict[str, float]) -> float:
        """Calcula a eficÃ¡cia do cooldown de forma quantitativa usando densidade de trades."""
        try:
            trades_per_1000 = float(metrics.get("trades_per_1000", 0.0))
            desired_density = float(metrics.get("max_trades_cap", 60.0))
            desired_density = (
                float(np.clip(desired_density, 20.0, 120.0))
                if np.isfinite(desired_density)
                else 60.0
            )
            if desired_density <= 0.0:
                desired_density = 60.0
            ratio = 1.0 - (trades_per_1000 / desired_density)
            return float(np.clip(100.0 * ratio, 0.0, 100.0))
        except Exception:
            return 50.0

    def _calculate_trend_following_ratio(self, metrics: Dict[str, float]) -> float:
        """Percentual de trades alinhados ao regime dominante (up/down), via regime_trade_*"""
        try:
            trend_up_avg = float(metrics.get("trend_regime_avg_up", 0.0))
            trend_down_avg = float(metrics.get("trend_regime_avg_down", 0.0))
            trade_up_pct = float(metrics.get("regime_trade_up_pct", 0.0))
            trade_down_pct = float(metrics.get("regime_trade_down_pct", 0.0))
            dominant_up = trend_up_avg >= trend_down_avg
            aligned_pct = trade_up_pct if dominant_up else trade_down_pct
            return float(np.clip(aligned_pct, 0.0, 100.0))
        except Exception:
            return 0.0

    def _calculate_trade_distribution(self, metrics: Dict[str, float]) -> float:
        """Percentual de trades executados em regime de alta (proxy para '% long')"""
        try:
            return float(np.clip(metrics.get("long_trade_pct", 0.0), 0.0, 100.0))
        except Exception:
            return 0.0

    def optimize_anti_scalping_config(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        n_trials: int = 1000,
        n_timesteps: int = 10000,
        n_eval_episodes: int = 3,
    ):
        """

        FASE 1: OtimizaÃƒÂ§ÃƒÂ£o de configuraÃƒÂ§ÃƒÂ£o anti-scalping
        Encontra os melhores parÃƒÂ¢metros de configuraÃƒÂ§ÃƒÂ£o para surfar tendÃƒÂªncias
        """

        logger.info(
            "[TREND SAC FASE 1] Iniciando otimizaÃƒÂ§ÃƒÂ£o de configuraÃƒÂ§ÃƒÂ£o anti-scalping..."
        )
        train_env_fn = lambda: self._make_trend_env(train_df, mode="optimization")
        eval_env_fn = lambda: self._make_trend_env(eval_df, mode="optimization")

        def objective(trial: optuna.Trial) -> float:
            # ParÃƒÂ¢metros de configuraÃƒÂ§ÃƒÂ£o anti-scalping para otimizar (sem cooldown)
            trend_alignment = trial.suggest_float(
                "trend_alignment_threshold", 0.01, 0.12
            )
            trend_margin = trial.suggest_float("trend_regime_margin", 0.0, 0.12)
            base_threshold_opt = trial.suggest_float("base_threshold_opt", 0.18, 0.35)
            min_action_threshold_opt = trial.suggest_float(
                "min_action_threshold_opt", 0.08, min(0.22, base_threshold_opt)
            )
            prior_open_long_opt = trial.suggest_float(
                "prior_open_long_threshold_opt", 0.25, 0.45
            )
            prior_open_short_opt = trial.suggest_float(
                "prior_open_short_threshold_opt", -0.45, -0.25
            )
            force_threshold_factor_opt = trial.suggest_float(
                "force_threshold_factor_opt", 0.5, 0.8
            )
            penalty_cap_per_step_opt = trial.suggest_float(
                "penalty_cap_per_step_opt", 1.0, 3.0
            )
            warmup_base_opt = trial.suggest_float("warmup_base_opt", 0.1, 0.4)
            warmup_scale_high = max(0.4, 1.0 - warmup_base_opt)
            warmup_scale_opt = trial.suggest_float(
                "warmup_scale_opt", 0.4, warmup_scale_high
            )
            config_params = {
                # [v3.1] Params obsoletos REMOVIDOS do HPO:
                # min_trade_duration, scalp_penalty, opening_bonus_mult,
                # long_trade_bonus_threshold, long_trade_bonus_max,
                # post_trade_cooldown, max_trades_per_episode,
                # max_trade_duration (set_phase2_runtime_tweaks já o anula → param sem efeito;
                #                     TREND_DURATION_PRESSURE_* cobre organicamente)
                # → substituídos por Gravity Well + Triple Barrier proporcional (orgânico)
                "trend_alignment_threshold": trend_alignment,
                "trend_regime_margin": trend_margin,
                "base_threshold_opt": base_threshold_opt,
                "min_action_threshold_opt": min_action_threshold_opt,
                "prior_open_long_threshold_opt": prior_open_long_opt,
                "prior_open_short_threshold_opt": prior_open_short_opt,
                "force_threshold_factor_opt": force_threshold_factor_opt,
                "penalty_cap_per_step_opt": penalty_cap_per_step_opt,
                "warmup_base_opt": warmup_base_opt,
                "warmup_scale_opt": warmup_scale_opt,
            }
            # DEBUG por trial: sumariza regime e EMA das janelas de treino/avaliacao
            try:
                _env_tmp = train_env_fn()
                df_train_dbg = getattr(_env_tmp, "df", None)
                _env_tmp = eval_env_fn()
                df_eval_dbg = getattr(_env_tmp, "df", None)

                def _summarize(df):
                    if df is None or len(df) == 0:
                        return {
                            "len": 0,
                            "reg_up_mean": 0.0,
                            "reg_down_mean": 0.0,
                            "reg_side_mean": 0.0,
                            "ema_mean": 0.0,
                        }
                    ema_col = None
                    for c in df.columns:
                        if isinstance(c, str) and "ema_trend" in c.lower():
                            ema_col = c
                            break
                    return {
                        "len": int(len(df)),
                        "reg_up_mean": float(np.nanmean(df.get("tp_regime_up", 0.0)))
                        if "tp_regime_up" in df.columns
                        else 0.0,
                        "reg_down_mean": float(
                            np.nanmean(df.get("tp_regime_down", 0.0))
                        )
                        if "tp_regime_down" in df.columns
                        else 0.0,
                        "reg_side_mean": float(
                            np.nanmean(df.get("tp_regime_sideways", 0.0))
                        )
                        if "tp_regime_sideways" in df.columns
                        else 0.0,
                        "ema_mean": float(np.nanmean(df[ema_col])) if ema_col else 0.0,
                    }

                train_sum = _summarize(df_train_dbg)
                eval_sum = _summarize(df_eval_dbg)
                logger.debug(
                    "[TRIAL_DEBUG] #%d params=%s | train(len=%d, up=%.2f, dn=%.2f, sd=%.2f, ema=%.3f) | eval(len=%d, up=%.2f, dn=%.2f, sd=%.2f, ema=%.3f)",
                    trial.number,
                    {
                        k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in config_params.items()
                    },
                    train_sum["len"],
                    train_sum["reg_up_mean"],
                    train_sum["reg_down_mean"],
                    train_sum["reg_side_mean"],
                    train_sum["ema_mean"],
                    eval_sum["len"],
                    eval_sum["reg_up_mean"],
                    eval_sum["reg_down_mean"],
                    eval_sum["reg_side_mean"],
                    eval_sum["ema_mean"],
                )
            except Exception:
                pass
            try:
                # Cria ambientes com configuracao otimizada
                train_env = DummyVecEnv([train_env_fn])
                train_env = VecNormalize(
                    train_env, norm_obs=False, norm_reward=False, clip_reward=10.0
                )

                eval_env = DummyVecEnv([eval_env_fn])
                eval_env = VecNormalize(
                    eval_env, norm_obs=False, norm_reward=False, clip_reward=10.0
                )

                default_env_params = self._convert_reward_params_to_env(
                    self.reward_hyperparams
                )
                if default_env_params:
                    for env in [train_env, eval_env]:
                        env.env_method("set_reward_shaping_params", default_env_params)
                for env in [train_env, eval_env]:
                    env.env_method("set_trend_debug_logging", False)
                # Aplica configuração otimizada — train e eval recebem configs potencialmente diferentes
                # [BUG FIX] max_trades_per_episode deve ser escalado proporcionalmente para o eval env.
                # O eval tem ~28% dos steps do treino. Com max_trades=10 para 24k steps e cooldown=500,
                # o agente só consegue executar 2 trades em 7k steps, gerando inatividade patológica.
                train_env.env_method("set_anti_scalping_config", config_params)

                # Constrói config ajustada para o eval env com max_trades escalado pelo ratio de steps
                try:
                    train_steps = float(train_env.get_attr("max_steps")[0])
                    eval_steps = float(eval_env.get_attr("max_steps")[0])
                    eval_train_ratio = max(0.1, eval_steps / max(1.0, train_steps))
                except Exception:
                    eval_train_ratio = 1.0

                # [v3.1] max_trades_per_episode removido — sem escalonamento necessário
                eval_env.env_method("set_anti_scalping_config", config_params)
                try:
                    episode_steps = float(eval_env.get_attr("max_steps")[0])
                except Exception:
                    episode_steps = 1000.0
                episode_steps = max(100.0, episode_steps)
                # Usa hiperparÃƒÂ¢metros SAC padrÃƒÂ£o para esta fase
                sac_params = {
                    "learning_rate": 1e-4,
                    "buffer_size": 100000,
                    "batch_size": 512,
                    "gamma": 0.99,
                    "tau": 0.01,
                    "train_freq": 4,
                    "use_sde": True,
                    "target_entropy": -3.0,  # 🚀 [FIX] Prevenção de Entropy Collapse (dim=3)
                    "ent_coef": "auto_0.5",  # [FIX] Forçar um piso inicial maior para explorar antes de focar
                }
                # 🚀 HPO: Usar MLP simples (4x mais rápido que LSTM+Attention)
                # Os hiperparâmetros encontrados são transferíveis para LSTM no treinamento final
                from specialists.scientific_corrections import get_mlp_policy_kwargs

                policy_kwargs = get_mlp_policy_kwargs()

                model = ClippedSAC(
                    "MlpPolicy",
                    train_env,
                    policy_kwargs=policy_kwargs,
                    **sac_params,
                    verbose=0,
                    device=self._get_device(),
                    tensorboard_log=os.path.join(
                        os.getcwd(), "logs", "tensorboard", self.specialist_name
                    ),
                    max_grad_norm=self.max_grad_norm,
                )
                # Treinamento com configuraÃƒÂ§ÃƒÂ£o otimizada
                model.learn(total_timesteps=n_timesteps, progress_bar=False)
                # AvaliaÃƒÂ§ÃƒÂ£o determinÃƒÂ­stica
                mean_reward, _ = evaluate_policy(
                    model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
                )
                # Coleta mÃƒÂ©tricas financeiras para poda inteligente
                try:
                    summaries_batches = eval_env.env_method("consume_episode_summaries")
                    if summaries_batches and len(summaries_batches) > 0:
                        all_summaries = [
                            summary for batch in summaries_batches for summary in batch
                        ]
                        if all_summaries:
                            avg_metrics = self._calculate_average_metrics(all_summaries)
                            trend_up_avg = float(
                                avg_metrics.get("trend_regime_avg_up", 0.0)
                            )
                            trend_down_avg = float(
                                avg_metrics.get("trend_regime_avg_down", 0.0)
                            )
                            trend_side_avg = float(
                                avg_metrics.get("trend_regime_avg_side", 0.0)
                            )
                            trade_up_pct = float(
                                avg_metrics.get("regime_trade_up_pct", 0.0)
                            )
                            trade_down_pct = float(
                                avg_metrics.get("regime_trade_down_pct", 0.0)
                            )
                            trade_side_pct = float(
                                avg_metrics.get("regime_trade_side_pct", 0.0)
                            )
                            ema_trend_avg = float(avg_metrics.get("ema_trend_avg", 0.0))
                            dominant_trend = str(
                                avg_metrics.get("trend_dominant", "none")
                            ).upper()
                            trend_samples = float(
                                avg_metrics.get("trend_regime_samples", 0.0)
                            )
                            logger.debug(
                                "[FASE 1 TREND] Trial %d | regime_mean(up=%.2f, down=%.2f, side=%.2f) | "
                                "trade_split=%.1f%%/%.1f%%/%.1f%% | ema_avg=%.3f | samples=%.0f | dominant=%s",
                                trial.number,
                                trend_up_avg,
                                trend_down_avg,
                                trend_side_avg,
                                trade_up_pct,
                                trade_down_pct,
                                trade_side_pct,
                                ema_trend_avg,
                                trend_samples,
                                dominant_trend,
                            )
                            logger.info(
                                "    Trend Metrics: regime_mean(up=%.2f, down=%.2f, side=%.2f) | "
                                "trade_split=%.1f%%/%.1f%%/%.1f%% | ema_avg=%.3f | dominant=%s (samples=%.0f)",
                                trend_up_avg,
                                trend_down_avg,
                                trend_side_avg,
                                trade_up_pct,
                                trade_down_pct,
                                trade_side_pct,
                                ema_trend_avg,
                                dominant_trend,
                                trend_samples,
                            )
                            # MÃƒâ€°TRICAS DE PODA PARA SURF DE TENDÃƒÅ NCIAS
                            num_trades = float(avg_metrics.get("num_trades", 0.0))
                            avg_trade_duration = float(
                                avg_metrics.get("avg_trade_duration", 0.0)
                            )
                            win_rate = float(avg_metrics.get("win_rate_pct", 0.0))
                            profit_factor = float(avg_metrics.get("profit_factor", 0.0))
                            profit_factor_raw = float(
                                avg_metrics.get("profit_factor_raw", profit_factor)
                            )
                            pf_reliability = float(
                                np.clip(avg_metrics.get("pf_reliable", 1.0), 0.0, 1.0)
                            )
                            trade_sharpe = float(avg_metrics.get("trade_sharpe", 0.0))
                            max_trades_cap = float(
                                avg_metrics.get("max_trades_cap", 0.0)
                            )
                            trades_per_1000 = num_trades / max(
                                1.0, episode_steps / 1000.0
                            )
                            avg_metrics["trades_per_1000"] = trades_per_1000
                            trade_target_max = max(1.0, max_trades_cap or 60.0)
                            if trades_per_1000 > 1000.0:
                                logger.info(
                                    f"[FASE 1 PODA] Trial {trial.number} podado: densidade de trades muito alta (trades/1000={trades_per_1000:.1f}, trades={num_trades:.1f})."
                                )
                                raise optuna.exceptions.TrialPruned()
                            if num_trades < 1:
                                logger.info(
                                    f"[FASE 1] Trial {trial.number}: nenhum trade executado. Score penalizado, sem pruning."
                                )
                                score = -6.0
                            else:
                                score = self._calculate_trend_surfing_score(avg_metrics)
                            max_drawdown = avg_metrics.get("max_drawdown_pct", 100)
                            avg_return = avg_metrics.get("avg_return_pct", 0)
                            cooldown_effectiveness = (
                                self._calculate_cooldown_effectiveness(avg_metrics)
                            )
                            trend_following_ratio = (
                                self._calculate_trend_following_ratio(avg_metrics)
                            )
                            trade_distribution = self._calculate_trade_distribution(
                                avg_metrics
                            )
                            logger.info(
                                f"[FASE 1] Trial {trial.number}: Score={score:.4f}"
                            )
                            logger.info(
                                "[TREND_DEBUG] trial=%d | regime_up_avg=%.2f | regime_down_avg=%.2f | "
                                "regime_side_avg=%.2f | ema_trend_avg=%.3f | trades=%.1f | "
                                "avg_duration=%.1f | win_rate=%.1f%% | score=%.4f | trend_follow=%.1f%%",
                                trial.number,
                                trend_up_avg,
                                trend_down_avg,
                                trend_side_avg,
                                ema_trend_avg,
                                num_trades,
                                avg_trade_duration,
                                win_rate,
                                score,
                                trend_following_ratio,
                            )
                            if max_trades_cap:
                                utilization = (
                                    num_trades / max(1.0, max_trades_cap)
                                ) * 100.0
                                logger.debug(
                                    f"    Trades: {num_trades} / {max_trades_cap:.0f} (utilizacao: {utilization:.1f}%)"
                                )
                            else:
                                logger.debug(f"    Trades: {num_trades} (ideal: 1-60)")
                            min_dur_param = config_params.get("min_trade_duration")
                            target_min_label = (
                                f"{min_dur_param}+"
                                if min_dur_param is not None
                                else "N/A"
                            )
                            logger.debug(
                                f"    Duration: {avg_trade_duration:.1f} candles (target_min: {target_min_label})"
                            )
                            logger.debug(
                                f"    Win Rate: {win_rate:.1f}% (ideal: 50-70%)"
                            )
                            pf_tag = (
                                "ok" if pf_reliability >= 0.5 else "baixa confianca"
                            )
                            logger.debug(
                                f"    Profit Factor: {profit_factor:.2f} (raw {profit_factor_raw:.2f}, {pf_tag}) (ideal: 1.2-2.0)"
                            )
                            logger.debug(
                                f"    Sharpe: {trade_sharpe:.2f} (ideal: 0.5-2.0)"
                            )
                            logger.debug(
                                f"    Max DD: {max_drawdown * 100.0:.1f}% (ideal: <10%)"
                            )
                            logger.debug(
                                f"    Cooldown Eff: {cooldown_effectiveness:.1f}% "
                            )
                            logger.debug(
                                f"    Trend Follow: {trend_following_ratio:.1f}%"
                            )
                            logger.debug(
                                f"    Distribution: {trade_distribution:.1f}% long "
                            )
                            return score
                except optuna.exceptions.TrialPruned:
                    raise
                except Exception as e:
                    logger.warning(f"[FASE 1] Erro ao coletar mÃƒÂ©tricas: {e}")
                return mean_reward
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"[FASE 1] Trial falhou: {e}", exc_info=True)
                return -float("inf")

        # Otimizacao da configuracao com poda agressiva
        pruner = optuna.pruners.HyperbandPruner(
            max_resource=n_timesteps,
            min_resource=max(256, n_timesteps // 6),
            reduction_factor=3,
        )
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=20,
            multivariate=True,
            constant_liar=True,
            warn_independent_sampling=False,
        )
        # PersistÃƒÂªncia para permitir retomada do estudo da Fase 1
        anti_scalp_db = os.path.join(
            os.getcwd(), "logs", "trend_specialist_anti_scalping_config.db"
        )
        storage_url = f"sqlite:///{anti_scalp_db}"
        study = optuna.create_study(
            direction="maximize",
            study_name="trend_specialist_anti_scalping_config_v1",
            pruner=pruner,
            sampler=sampler,
            storage=storage_url,
            load_if_exists=True,
        )
        if len(study.trials) == 0:
            seed_params = {
                "trend_alignment_threshold": 0.05,
                "trend_regime_margin": 0.02,
                "base_threshold_opt": 0.24,
                "min_action_threshold_opt": 0.1,
                "prior_open_long_threshold_opt": 0.3,
                "prior_open_short_threshold_opt": -0.3,
                "force_threshold_factor_opt": 0.65,
                "penalty_cap_per_step_opt": 2.0,
                "warmup_base_opt": 0.2,
                "warmup_scale_opt": 0.5,
                # max_trade_duration removido — TREND_DURATION_PRESSURE_* cobre organicamente
            }
            try:
                logger.info("[FASE 1] Running seed trial before Optuna search.")
                seed_trial = optuna.trial.FixedTrial(seed_params)
                seed_score = objective(seed_trial)
                logger.info(
                    "[FASE 1] Seed trial score=%.4f with params=%s",
                    seed_score,
                    seed_params,
                )
            except optuna.exceptions.TrialPruned:
                logger.info(
                    "[FASE 1] Seed trial was pruned due to extreme trade density."
                )
            except Exception as exc:
                logger.warning("[FASE 1] Seed trial failed: %s", exc)
            study.enqueue_trial(seed_params)
        # Retoma de onde parou
        completed_trials = len(study.trials)
        if completed_trials >= n_trials:
            logger.info(
                f"[FASE 1] JÃƒÂ¡ existem {completed_trials} trials. Nenhum trial adicional necessÃƒÂ¡rio."
            )
            remaining_trials = 0
        else:
            remaining_trials = n_trials - completed_trials
            logger.info(
                f"[FASE 1] Retomando estudo: executando {remaining_trials} trials adicionais (de {n_trials})."
            )
        if remaining_trials > 0:
            study.optimize(
                objective, n_trials=remaining_trials, timeout=43200
            )  # Aumentado para 12h (era 1h)
        # Retorno seguro mesmo que todos os trials tenham sido podados
        try:
            if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
                self.anti_scalping_config = study.best_params
                logger.info(
                    f"[FASE 1] Melhor configuraÃƒÂ§ÃƒÂ£o encontrada: {self.anti_scalping_config}"
                )
                self._save_anti_scalping_config()
                return self.anti_scalping_config
        except Exception:
            pass
        logger.warning("[FASE 1] Nenhuma configuraÃƒÂ§ÃƒÂ£o vÃƒÂ¡lida encontrada.")
        return None

    def optimize_hyperparameters(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        n_trials: int = 50,
        n_timesteps: int = 50000,
        n_eval_episodes: int = 8,
        env_class=None,
    ):
        """

        [VERSÃƒÆ'O FINAL E OTIMIZADA PARA PARALELISMO]
        Otimiza usando n_jobs > 1 e ambientes ÃƒÂºnicos por trial.
        """

        self.logger.info(
            "[TREND SAC OPTUNA] Iniciando otimizacao com paralelismo para SAC..."
        )
        # [FIX 7] SAC precisa de pelo menos 25k steps por trial para convergir
        n_timesteps = max(n_timesteps, 25000)  # SAC precisa de pelo menos 25k steps

        # [LOG DESCRITIVO] Banner para identificar qual especialista está otimizando
        specialist_name = getattr(self, "specialist_name", "TrendSpecialist")
        regime_type = getattr(self, "regime_type", "trend")
        regime_emoji = {"bull": "🐂", "bear": "🐻", "ranger": "🤠", "trend": "📈"}.get(
            regime_type, "🤖"
        )

        self.logger.info(f"")
        self.logger.info(f"{'=' * 70}")
        self.logger.info(
            f"{regime_emoji} OTIMIZANDO: {specialist_name.upper()} {regime_emoji}"
        )
        self.logger.info(f"{'=' * 70}")
        self.logger.info(f"   Tipo: {regime_type.upper()}")
        self.logger.info(f"   Trials: {n_trials}")
        self.logger.info(f"   Timesteps por trial: {n_timesteps:,}")
        self.logger.info(f"   Dados treino: {len(train_df):,} amostras")
        self.logger.info(f"   Dados eval: {len(eval_df):,} amostras")
        self.logger.info(f"{'=' * 70}")
        self.logger.info(f"")

        # Subconjunto de dados/episode ratio para trials mais curtos na Fase 2
        try:
            episode_ratio_env = float(os.getenv("TREND_PHASE2_EPISODE_RATIO", "1.0"))
            episode_ratio_env = float(np.clip(episode_ratio_env, 0.1, 1.0))
        except Exception:
            episode_ratio_env = 1.0

        def _subset_df(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return df
            n = len(df)
            keep = max(1000, int(n * episode_ratio_env))
            return df.iloc[-keep:]

        train_df_short = _subset_df(train_df)
        eval_df_short = _subset_df(eval_df)

        # --- CRÃTICO: NÃ£o vetorize o ambiente aqui ---
        # Cada trial paralelo cuidarÃ¡ de seu prÃ³prio ambiente Ãºnico.
        train_env_fn = lambda: self._make_trend_env(
            train_df_short, mode="optimization", env_class=env_class
        )
        eval_env_fn = lambda: self._make_trend_env(
            eval_df_short, mode="optimization", env_class=env_class
        )

        # [D-A5 FIX] Scaler do HPO unificado com pipeline da Fase 3.
        # Antes: HPO usava RobustScaler simples; Fase 3 usa ColumnTransformer(Robust+Standard).
        # Hiperparâmetros otimizados num espaço de features diferente do treino final
        # → melhor config no HPO não transfere para produção.
        # Correção: usar o mesmo ColumnTransformer(Robust para OHLCV, Standard para resto).
        try:
            from sklearn.preprocessing import RobustScaler as _RS_fix
            from sklearn.preprocessing import StandardScaler as _SS_fix
            from sklearn.compose import ColumnTransformer as _CT_fix

            _probe_env = train_env_fn()
            _env_cols = list(getattr(_probe_env, "feature_columns", []))
            try:
                _probe_env.close()
            except Exception:
                pass
            if _env_cols:
                _probe_df = train_df_short[_env_cols].fillna(0)
                _probe_data = _probe_df.values
                _already_scaled = (
                    np.abs(np.percentile(_probe_data, [1, 99])).max() < 20.0
                )
                if _already_scaled:
                    self.feature_scaler = None
                    self.logger.info(
                        "[FIX SCALER] Features pre-normalizadas — scaler = None."
                    )
                else:
                    # Mesmo pipeline da Fase 3: RobustScaler para OHLCV, StandardScaler para o resto
                    _ohlcv_set = {"open", "high", "low", "close", "volume"}
                    _protected = _ohlcv_set | {"tp_prior_conf"}
                    _ohlcv_hpo = [c for c in _env_cols if c in _ohlcv_set]
                    _non_ohlcv_hpo = [c for c in _env_cols if c not in _protected]
                    _transformers_hpo = [("standard", _SS_fix(), _non_ohlcv_hpo)]
                    if _ohlcv_hpo:
                        _transformers_hpo.insert(0, ("robust", _RS_fix(), _ohlcv_hpo))
                    _fixed_scaler = _CT_fix(_transformers_hpo, remainder="passthrough")
                    _fixed_scaler.fit(_probe_df)
                    self.feature_scaler = _fixed_scaler
                    self.logger.info(
                        "[D-A5 FIX] Scaler HPO sincronizado com Fase 3 (ColumnTransformer) "
                        "em %d features (%d OHLCV + %d outras).",
                        len(_env_cols),
                        len(_ohlcv_hpo),
                        len(_non_ohlcv_hpo),
                    )
                # [FIX SCALER SYNC] Atualiza cached_feature_columns para as colunas REAIS do env.
                self.cached_feature_columns = _env_cols
        except Exception as _fix_err:
            self.logger.warning(
                "[FIX SCALER] Falha ao sincronizar scaler com env: %s", _fix_err
            )

        def objective(trial: optuna.Trial) -> float:
            logger.info(f"[TREND SAC OPTUNA] >>> TRIAL {trial.number} INICIO")
            # Reseta estado intra-trial do monitor (peak_reward, rewards, etc.)
            # Evita falso D1 "COLLAPSE" quando trial N começa abaixo do pico do trial N-1
            _monitor_ref = getattr(self, "learning_monitor", None)
            if _monitor_ref is not None:
                try:
                    _monitor_ref.reset_trial()
                except Exception as _mr_err:
                    logger.debug(
                        "[MONITOR] Falha ao resetar no inicio do trial: %s", _mr_err
                    )

            # [CORREÇÃO CIENTÍFICA] Ranges baseados em literatura FinRL 2023-2024

            # Limpeza preventiva de memória antes de começar o Trial
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            sac_params = {
                # Learning Rate: centro do range ótimo (Kumar et al., 2020)
                # Com decay linear durante treinamento
                # [P0 FIX] SAC colapsa acima de ~1e-4 no trial 16. Upper limit reduzido para 3e-5 (Stability Patch).
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 3e-5, log=True
                ),
                # Buffer Size: limitado ao trial de 50k steps para que o buffer encha (Haarnoja 2018).
                # buffer > trial = agente só vê amostras recentes = convergência lenta.
                "buffer_size": trial.suggest_categorical("buffer_size", [30000, 50000]),
                # Batch Size: sem 1024 (requer buffer grande que não usamos mais)
                "batch_size": trial.suggest_categorical("batch_size", [256, 512]),
                # Gamma: MAIS ALTO para horizontes longos (FinRL crypto)
                "gamma": trial.suggest_float("gamma", 0.99, 0.999),
                # Tau: MAIS BAIXO para estabilidade (SAC original)
                "tau": trial.suggest_float("tau", 0.001, 0.01, log=True),
                # Reduzida a frequencia de atualizacao (backpropagation) para maior performance
                # Coleta mais experiencia antes de atualizar, acelerando os trials
                "train_freq": (512, "step"),
                "gradient_steps": 64,
                # 🔬 Entropy Coef: SEMPRE 'auto' com valor inicial forte
                "ent_coef": "auto_0.5",
                # SDE: recomendado para ações contínuas (SAC original)
                "use_sde": trial.suggest_categorical("use_sde", [True, False]),
                # 🔬 Prevenção de Entropy Collapse [FIX]: target proporcional à ação
                # Para action_dim=3, o valor ideal é -3.0. Range ajustado para [-4.0, -2.0].
                "target_entropy": trial.suggest_float("target_entropy", -4.0, -2.0),
            }

            # max_grad_norm é passado separadamente para ClippedSAC (não no dict)
            # [FIX] Reduzido teto 1.5→1.0: valores >1.0 causam explosão de gradiente (actor_loss >5)
            max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 1.0)
            reward_params = self._suggest_reward_params(trial)
            self._apply_reward_params_to_config(reward_params)
            train_env: Optional[SubprocVecEnv] = None
            eval_env: Optional[DummyVecEnv] = None
            try:
                # --- CORRECAO: Fallback para DummyVecEnv no Windows ---
                # SubprocVecEnv causa "WinError 232 O pipe está sendo fechado"
                # ao tentar serializar o DataFrame enorme para os workers.
                # A velocidade será mantida via o train_freq=(512, step) e Numpy Cache.
                train_env = DummyVecEnv([train_env_fn])
                train_env = VecNormalize(
                    train_env, norm_obs=False, norm_reward=False, clip_reward=10.0
                )

                eval_env = DummyVecEnv([eval_env_fn])
                eval_env = VecNormalize(
                    eval_env, norm_obs=False, norm_reward=False, clip_reward=10.0
                )

                env_reward_params = self._convert_reward_params_to_env(reward_params)
                if env_reward_params:
                    for env in [train_env, eval_env]:
                        env.env_method("set_reward_shaping_params", env_reward_params)
                # FASE 1 desativada: nenhuma configuracao anti-scalping aplicada
                # Tweaks de Fase 2: desativa time stop e cooldown
                for env in [train_env, eval_env]:
                    env.env_method("set_phase2_runtime_tweaks")
                # Reforsa exploraÃƒÂ§ÃƒÂ£o no inÃƒÂ­cio: use_sde True e entropia alvo mais alta
                # [PERF HPO] MLP simples no HPO: 3-4x mais rápido que AttentionMLP na GPU.
                # HPO busca hiperparâmetros (LR, buffer, gamma, tau) — não arquitetura.
                # Treino final usa AttentionMLP(128d) que é a arquitetura de produção.
                # MLP[128,128] é super leve e muito rápida para backpropagation no HPO.
                from specialists.scientific_corrections import get_mlp_policy_kwargs

                policy_kwargs = get_mlp_policy_kwargs()
                policy_kwargs["net_arch"] = [128, 128]

                # 🔬 LINEAR LR DECAY SCHEDULER (evita overshoot no final do treinamento)
                # LR decai linearmente de valor_inicial até 10% no final
                initial_lr = sac_params.pop("learning_rate")
                final_lr_ratio = 0.1  # Termina em 10% do LR inicial

                def linear_lr_schedule(progress_remaining: float) -> float:
                    """
                    LR decay linear: começa alto, termina baixo.
                    progress_remaining: 1.0 no início, 0.0 no final.
                    """
                    return final_lr_ratio + (1.0 - final_lr_ratio) * progress_remaining

                # Multiplica o schedule pelo LR inicial
                lr_schedule = lambda p: initial_lr * linear_lr_schedule(p)

                # [PARALLEL OPTUNA] Usa CPU nos trials de HPO para evitar conflitos CUDA
                # com n_jobs=2. A GPU é reservada para o treinamento final (Fase 3).
                model = ClippedSAC(
                    "MlpPolicy",
                    train_env,
                    policy_kwargs=policy_kwargs,  # <<< MLP (HPO) / AttentionMLP (treino final)
                    learning_rate=lr_schedule,  # LR com decay linear
                    **sac_params,
                    verbose=0,  # [PERF] verbose=0 em HPO: evita tabelas SB3 impressas a cada eval
                    device=self._get_device(),
                    tensorboard_log=None,
                    max_grad_norm=max_grad_norm,
                )
                # torch.compile removido: requer Triton que não existe no Windows.
                # O erro falha lazily dentro de model.learn() — o try/except no compile não captura.
                # Avaliamos com polÃƒÂ­tica estocÃƒÂ¡stica para estimular exploraÃƒÂ§ÃƒÂ£o inicial e evitar paralisia.
                # [SPEED FIX #4] Divisor configurável via env var (default 8 = 8 checkpoints/trial).
                # Ranger seta HPO_EVAL_CHECKPOINTS=4 em base_regime_specialist para cortar
                # overhead de eval pela metade (4 em vez de 8 avaliações completas por trial).
                try:
                    _eval_ckpts = int(os.getenv("HPO_EVAL_CHECKPOINTS", "8"))
                    _eval_ckpts = max(2, min(16, _eval_ckpts))
                except Exception:
                    _eval_ckpts = 8
                _eval_freq = max(n_timesteps // _eval_ckpts, 2000)
                eval_callback = TrialEvalCallback(
                    eval_env,
                    trial,
                    n_eval_episodes=n_eval_episodes,
                    # [CONVERGENCIA] //8 da 8 checkpoints por trial (era //5 = 5 checkpoints).
                    # Mais checkpoints = pruner rejeita trials ruins mais cedo = menos desperdicio.
                    # Ref: AFLM Ch.7 — mais avaliações intermediárias melhora seleção de HPs.
                    eval_freq=_eval_freq,
                    deterministic=True,
                    total_timesteps=n_timesteps,
                    learning_monitor=getattr(self, "learning_monitor", None),
                )
                # Tornar a guarda mais tolerante para permitir aprender a surfar tendÃªncia
                try:
                    # Permitir fase de explora��o mais longa antes de podar
                    # [BUG 1 FIX] Reajuste do Mean Reward Floor para permitir aprendizado
                    setattr(
                        eval_callback,
                        "mean_reward_floor",
                        float(os.getenv("TREND_MEAN_REWARD_FLOOR", "-10000")),
                    )
                    setattr(
                        eval_callback,
                        "early_negative_prune_step",
                        int(os.getenv("TREND_GUARD_STEP", "15000")),
                    )
                    if hasattr(eval_callback, "required_guard_evaluations"):
                        setattr(
                            eval_callback,
                            "required_guard_evaluations",
                            int(os.getenv("TREND_GUARD_EVALS", "4")),
                        )
                except Exception:
                    pass
                logger.info(
                    f"[TREND SAC OPTUNA] TRIAL {trial.number}: LEARN start (timesteps={n_timesteps})"
                )
                # [PROGRESS BAR INTRA-TRIAL] Barra de progresso dentro de cada trial (steps)
                try:
                    from stable_baselines3.common.callbacks import (
                        BaseCallback as _SB3BaseCallback,
                    )
                    from tqdm import tqdm as _tqdm_steps

                    class _StepProgressCallback(_SB3BaseCallback):
                        """Barra de progresso SB3 por steps dentro do trial."""

                        def __init__(self, total_steps: int, trial_num: int):
                            super().__init__(verbose=0)
                            self._pbar = _tqdm_steps(
                                total=total_steps,
                                desc=f"  ↳ Trial {trial_num} steps",
                                unit="step",
                                dynamic_ncols=True,
                                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                                leave=False,
                            )
                            self._last_n = 0

                        def _on_step(self) -> bool:
                            delta = self.num_timesteps - self._last_n
                            if delta > 0:
                                self._pbar.update(delta)
                                self._last_n = self.num_timesteps
                            return True

                        def _on_training_end(self):
                            self._pbar.close()

                    _step_cb = _StepProgressCallback(n_timesteps, trial.number)
                    from stable_baselines3.common.callbacks import (
                        CallbackList as _CallbackList,
                    )

                    eval_callback = _CallbackList([eval_callback, _step_cb])
                except Exception:
                    pass  # sem tqdm: continua sem barra intra-trial

                # Logger em DEBUG: trades vão para arquivo via file handler (DEBUG),
                # console handler (INFO) já filtra o ruído — não precisa mudar o nível do logger.
                model.learn(total_timesteps=n_timesteps, callback=eval_callback)

                # ── [BUG FIX] Unwrap CallbackList ─────────────────────────────
                # Após o _StepProgressCallback ser adicionado, eval_callback é
                # encapsulado em CallbackList([TrialEvalCallback, _StepProgressCallback]).
                # getattr(CallbackList, 'eval_history') → KeyError → [] vazio.
                # Consequências:
                #   • guard loop (eval_history) nunca dispara → sem early-prune
                #   • is_pruned sempre False → prunes do callback perdidos
                #   • guard_threshold, guard_floor leem fallbacks errados
                # Solução: localizar o TrialEvalCallback real dentro do wrapper.
                _real_eval_cb = eval_callback
                if hasattr(eval_callback, "callbacks"):
                    for _cb in eval_callback.callbacks:
                        if hasattr(_cb, "eval_history") and hasattr(_cb, "is_pruned"):
                            _real_eval_cb = _cb
                            break

                guard_threshold = int(
                    getattr(_real_eval_cb, "early_negative_prune_step", 3000)
                )
                # guard_floor: lê do TrialEvalCallback já unwrapped (tem mean_reward_floor correto)
                # Fallback ao env var caso o atributo ainda não tenha sido setado.
                _floor_from_env = float(os.getenv("TREND_MEAN_REWARD_FLOOR", "-10000"))
                _floor_from_cb = float(
                    getattr(_real_eval_cb, "mean_reward_floor", _floor_from_env)
                )
                guard_floor = min(
                    _floor_from_cb, _floor_from_env
                )  # usa o mais permissivo (mais negativo)
                eval_history = getattr(_real_eval_cb, "eval_history", []) or []
                consecutive_negative = 0
                for step_int, mean_reward_guard in eval_history:
                    try:
                        step_int = int(step_int)
                        mean_reward_guard = float(mean_reward_guard)
                    except Exception:
                        continue
                    if step_int >= guard_threshold and mean_reward_guard <= guard_floor:
                        consecutive_negative += 1
                    else:
                        consecutive_negative = 0
                    if consecutive_negative >= 4:
                        trial.set_user_attr(
                            "forced_prune_reason", "guard_mean_reward_neg"
                        )
                        trial.set_user_attr("pruned_timesteps", step_int)
                        logger.info(
                            "[TREND SAC OPTUNA] TRIAL %s: PRUNED by guard at %s timesteps (mean_reward=%.4f <= floor %.4f)",
                            trial.number,
                            step_int,
                            mean_reward_guard,
                            guard_floor,
                        )
                        raise optuna.exceptions.TrialPruned(
                            "Trial pruned by TrendSpecialist guard (mean_reward <= floor)"
                        )
                if getattr(_real_eval_cb, "is_pruned", False):
                    prune_reason = trial.user_attrs.get(
                        "forced_prune_reason", "callback"
                    )
                    prune_steps = int(getattr(model, "num_timesteps", 0))
                    trial.set_user_attr("pruned_timesteps", prune_steps)
                    logger.info(
                        "[TREND SAC OPTUNA] TRIAL %s: PRUNED at %s timesteps (reason=%s, via _real_eval_cb)",
                        trial.number,
                        prune_steps,
                        prune_reason,
                    )
                    raise optuna.exceptions.TrialPruned(
                        f"Trial pruned by callback ({prune_reason})"
                    )
                logger.info(f"[TREND SAC OPTUNA] TRIAL {trial.number}: LEARN end")
                # <<< CORREÃƒâ€¡ÃƒÆ'O: Limpa summaries acumulados do EvalCallback >>>
                logger.debug(
                    "Limpando summaries de avaliaÃƒÂ§ÃƒÂ£o intermediÃƒÂ¡rios..."
                )
                try:
                    eval_env.env_method("consume_episode_summaries")
                except Exception:
                    pass  # Ignora se o mÃƒÂ©todo nÃƒÂ£o existir
                # Agora roda a avaliaÃƒÂ§ÃƒÂ£o final - apenas estes resultados serÃƒÂ£o usados
                logger.info(f"[TREND SAC OPTUNA] TRIAL {trial.number}: EVALUATE start")
                mean_reward, std_reward = evaluate_policy(
                    model,
                    eval_env,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,  # CORREÃƒâ€¡ÃƒÆ'O: AvaliaÃƒÂ§ÃƒÂ£o determinÃƒÂ­stica
                )
                if mean_reward <= guard_floor:
                    prune_steps = int(getattr(model, "num_timesteps", 0))
                    trial.set_user_attr("forced_prune_reason", "final_mean_reward_neg")
                    trial.set_user_attr("pruned_timesteps", prune_steps)
                    logger.info(
                        "[TREND SAC OPTUNA] TRIAL %s: PRUNED on final evaluation (mean_reward=%.4f <= floor %.4f, steps=%s)",
                        trial.number,
                        mean_reward,
                        guard_floor,
                        prune_steps,
                    )
                    raise optuna.exceptions.TrialPruned(
                        "Trial pruned by final evaluation (mean_reward <= floor)"
                    )
                logger.info(
                    f"[TREND SAC OPTUNA] TRIAL {trial.number}: EVALUATE end | mean={mean_reward:.4f} std={std_reward:.4f}"
                )

                # [MONITOR] Registra resultado do trial no LearningMonitor
                try:
                    _trial_actor_loss: Optional[float] = None
                    _trial_std: Optional[float] = None
                    if hasattr(model, "logger") and hasattr(
                        model.logger, "name_to_value"
                    ):
                        _trial_actor_loss = model.logger.name_to_value.get(
                            "train/actor_loss"
                        )
                    if (
                        hasattr(model, "actor")
                        and hasattr(model.actor, "log_std")
                        and model.actor.log_std is not None
                    ):
                        import torch as _torch

                        with _torch.no_grad():
                            _trial_std = float(
                                _torch.exp(model.actor.log_std.data).mean().item()
                            )
                    _monitor = getattr(self, "learning_monitor", None)
                    if _monitor is not None:
                        _monitor.record_trial_eval(
                            trial_number=trial.number,
                            mean_reward=mean_reward,
                            timesteps=int(getattr(model, "num_timesteps", 0)),
                            actor_loss=_trial_actor_loss,
                            std=_trial_std,
                        )
                        # Circuit breaker: aborta se em colapso confirmado
                        if _monitor.should_abort_trial():
                            logger.critical(
                                "[TREND SAC OPTUNA] CIRCUIT BREAKER: abortando trial %d — %s",
                                trial.number,
                                _monitor.get_abort_reason(),
                            )
                            raise optuna.exceptions.TrialPruned(
                                f"LearningMonitor circuit breaker: {_monitor.get_abort_reason()}"
                            )
                except optuna.exceptions.TrialPruned:
                    raise
                except Exception as _me:
                    logger.debug("[MONITOR] Falha ao registrar trial eval: %s", _me)

                # Coleta métricas financeiras (SOMENTE da avaliação final)
                try:
                    # Esta chamada agora pegarÃƒÂ¡ apenas os resultados frescos do evaluate_policy acima
                    summaries_batches = eval_env.env_method("consume_episode_summaries")
                    episode_summaries = (
                        summaries_batches[0]
                        if isinstance(summaries_batches, list) and summaries_batches
                        else []
                    )
                except Exception:
                    episode_summaries = []
                if episode_summaries:
                    pf_values = [s.get("profit_factor", 0.0) for s in episode_summaries]
                    pf_raw_values = [
                        s.get("profit_factor_raw", v)
                        for s, v in zip(episode_summaries, pf_values)
                    ]
                    pf_reliabilities = [
                        float(np.clip(s.get("pf_reliable", 1.0), 0.0, 1.0))
                        for s in episode_summaries
                    ]
                    profit_factor = float(
                        np.mean([min(v, PF_MAX_DISPLAY) for v in pf_values])
                    )
                    profit_factor_raw = (
                        float(np.mean(pf_raw_values))
                        if pf_raw_values
                        else profit_factor
                    )
                    pf_reliability_mean = (
                        float(np.mean(pf_reliabilities)) if pf_reliabilities else 1.0
                    )
                    if pf_reliability_mean < 0.5:
                        profit_factor = min(profit_factor, 1.2)
                    trade_sharpe = float(
                        np.mean([s.get("trade_sharpe", 0.0) for s in episode_summaries])
                    )
                    avg_return_pct = float(
                        np.mean(
                            [s.get("avg_return_pct", 0.0) for s in episode_summaries]
                        )
                    )
                    max_drawdown_pct = float(
                        np.mean(
                            [s.get("max_drawdown_pct", 1.0) for s in episode_summaries]
                        )
                    )
                    avg_trade_duration = float(
                        np.mean(
                            [
                                s.get("avg_trade_duration", 0.0)
                                for s in episode_summaries
                            ]
                        )
                    )
                    num_trades = float(
                        np.sum([s.get("num_trades", 0.0) for s in episode_summaries])
                    )
                else:
                    try:
                        eval_env.envs[0].env
                        fin_stats = eval_env.envs[0].env.get_attr(
                            "get_financial_stats"
                        )[0]()
                    except Exception:
                        fin_stats = {
                            "profit_factor": 0.0,
                            "trade_sharpe": 0.0,
                            "avg_return_pct": 0.0,
                            "max_drawdown_pct": 1.0,
                            "num_trades": 0.0,
                            "pf_reliable": 1.0,
                        }
                    profit_factor = float(fin_stats.get("profit_factor", 0.0))
                    profit_factor_raw = float(
                        fin_stats.get("profit_factor_raw", profit_factor)
                    )
                    pf_reliability_mean = float(
                        np.clip(fin_stats.get("pf_reliable", 1.0), 0.0, 1.0)
                    )
                    if pf_reliability_mean < 0.5:
                        profit_factor = min(profit_factor, 1.2)
                    trade_sharpe = float(fin_stats.get("trade_sharpe", 0.0))
                    avg_return_pct = float(fin_stats.get("avg_return_pct", 0.0))
                    max_drawdown_pct = float(fin_stats.get("max_drawdown_pct", 1.0))
                    num_trades = float(fin_stats.get("num_trades", 0.0))
                    avg_trade_duration = float(fin_stats.get("avg_trade_duration", 0.0))
                profit_factor = min(profit_factor, PF_MAX_DISPLAY)
                trade_sharpe = float(np.clip(trade_sharpe, -5.0, 5.0))
                avg_return_pct = float(np.clip(avg_return_pct, -1.0, 1.0))
                max_drawdown_pct = max(0.0, float(max_drawdown_pct))
                # 🔬 CURRICULUM LEARNING: HPO usa mean_reward, não métricas financeiras
                # O agente precisa primeiro aprender a operar, depois lucrar no treinamento final
                disable_financial_prune = os.getenv(
                    "TREND_DISABLE_FINANCIAL_PRUNE", "1"
                ).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }  # Default=1 (desabilitado)
                if not disable_financial_prune and (
                    profit_factor < 1.0 or trade_sharpe <= 0.0
                ):
                    prune_steps = int(getattr(model, "num_timesteps", 0))
                    trial.set_user_attr(
                        "forced_prune_reason", "financial_metrics_prune"
                    )
                    trial.set_user_attr("pruned_timesteps", prune_steps)
                    logger.info(
                        "[TREND SAC OPTUNA] TRIAL %s: PRUNED for weak financial metrics (PF=%.3f, Sharpe=%.3f, steps=%s)",
                        trial.number,
                        profit_factor,
                        trade_sharpe,
                        prune_steps,
                    )
                    raise optuna.exceptions.TrialPruned(
                        "Trial pruned due to weak financial metrics"
                    )
                # BÃƒÂ´nus por duraÃƒÂ§ÃƒÂ£o mÃƒÂ©dia (incentiva trades mais longos em tendÃƒÂªncia)
                try:
                    avg_trade_duration
                except NameError:
                    avg_trade_duration = 0.0
                duration_bonus = 0.005 * float(np.clip(avg_trade_duration, 0.0, 200.0))
                dd_weight = float(os.getenv("TREND_DD_WEIGHT", "1.2"))
                pf_weight = float(os.getenv("TREND_PF_WEIGHT", "0.2"))
                score = (
                    trade_sharpe
                    - dd_weight * (max_drawdown_pct / 10.0)
                    + pf_weight * max(0.0, profit_factor - 1.0)
                    + duration_bonus
                )
                # Pequena contribuiÃ§Ã£o da recompensa mÃ©dia (normalizada e limitada)
                try:
                    # BUG N3 FIX: Normaliza pelo número médio de steps por episódio
                    ep_steps = 7062
                    reward_per_step = mean_reward / max(ep_steps, 1.0)
                    reward_component = 0.5 * float(
                        np.clip(reward_per_step / 3.0, -1.0, 1.0)
                    )
                except Exception:
                    reward_component = 0.0
                score += reward_component

                # --- LÃ“GICA DE PENALIDADE/BÃ”NUS COM CONSCIÃŠNCIA MULTI-TIMEFRAME (VERSÃƒO ROBUSTA) ---
                min_trades_razoavel = 10
                score_ajuste_atividade = 0.0
                trend_surf_bonus = 0.0
                try:
                    # Analisa o DataFrame de avaliaÃ§Ã£o para entender o contexto macro
                    eval_df_ctx = (
                        eval_env.envs[0].df if hasattr(eval_env.envs[0], "df") else None
                    )
                    if eval_df_ctx is not None and isinstance(
                        eval_df_ctx, pd.DataFrame
                    ):

                        def _find_macro_columns(
                            df: pd.DataFrame,
                        ) -> tuple[
                            Optional[pd.Series],
                            Optional[pd.Series],
                            Optional[pd.Series],
                        ]:
                            # Prioriza timeframes maiores (d -> h)
                            for tf_suffix in ["1d", "d", "D", "4h", "h4", "1h", "h1"]:
                                adx_col = f"adx_{tf_suffix}"
                                ema_fast_col = f"ema_21_{tf_suffix}"
                                ema_slow_col = f"ema_50_{tf_suffix}"
                                if (
                                    adx_col in df.columns
                                    and ema_fast_col in df.columns
                                    and ema_slow_col in df.columns
                                ):
                                    try:
                                        logger.info(
                                            f"[MTF] Contexto macro usando timeframe '{tf_suffix}'."
                                        )
                                    except Exception:
                                        pass
                                    return (
                                        df[adx_col],
                                        df[ema_fast_col],
                                        df[ema_slow_col],
                                    )
                            return None, None, None

                        adx_macro, ema_fast_macro, ema_slow_macro = _find_macro_columns(
                            eval_df_ctx
                        )
                        if (
                            adx_macro is not None
                            and ema_fast_macro is not None
                            and ema_slow_macro is not None
                        ):
                            # MÃ©trica 1: ForÃ§a da tendÃªncia macro
                            # Se adx for 0-1, ajusta threshold
                            adx_series = adx_macro.astype(float)
                            adx_thr = 25.0
                            if adx_series.max() <= 1.5:
                                adx_thr = 0.6
                            forca_tendencia_macro = float((adx_series > adx_thr).mean())
                            # MÃ©trica 2: DireÃ§Ã£o macro
                            direcao_macro = float(
                                np.sign(
                                    ema_fast_macro.astype(float)
                                    - ema_slow_macro.astype(float)
                                ).mean()
                            )
                            # Penaliza inatividade em tendÃªncia forte e clara
                            if (
                                forca_tendencia_macro > 0.5
                                and abs(direcao_macro) > 0.5
                                and num_trades < min_trades_razoavel
                            ):
                                deficit = float(min_trades_razoavel - num_trades)
                                penalty = deficit * 0.3 * forca_tendencia_macro
                                score_ajuste_atividade = -penalty
                                logger.info(
                                    f"[MTF] Trial penalizado por inatividade (trades={num_trades}) em TENDÃŠNCIA MACRO (ForÃ§a={forca_tendencia_macro:.2f}, DireÃ§Ã£o={direcao_macro:.2f}). Ajuste={score_ajuste_atividade:.2f}"
                                )
                            # BÃ´nus por "surfar tendÃªncia": quando a macro estÃ¡ forte e clara, segurar posiÃ§Ã£o rende bÃ´nus
                            if forca_tendencia_macro > 0.4 and abs(direcao_macro) > 0.4:
                                hold_factor = float(
                                    np.clip(
                                        (avg_trade_duration or 0.0) / 30.0, 0.0, 2.0
                                    )
                                )
                                pf_factor = float(
                                    np.clip(max(profit_factor - 1.0, 0.0), 0.0, 2.0)
                                )
                                trend_surf_bonus = (
                                    0.5
                                    * forca_tendencia_macro
                                    * (0.5 + 0.5 * hold_factor)
                                    + 0.3 * pf_factor
                                )
                                logger.info(
                                    f"[MTF] BÃ´nus surfar tendÃªncia aplicado: {trend_surf_bonus:.3f} (forÃ§a={forca_tendencia_macro:.2f}, hold={hold_factor:.2f}, pf_factor={pf_factor:.2f})"
                                )
                            # Impulso adicional em macro muito forte para favorecer surfagem lucrativa
                            if (
                                forca_tendencia_macro > 0.7
                                and abs(direcao_macro) > 0.6
                                and (avg_trade_duration or 0.0) >= 20.0
                            ):
                                strong_surf_boost = 0.2 + 0.3 * float(
                                    np.clip(max(profit_factor - 1.0, 0.0), 0.0, 2.0)
                                )
                                trend_surf_bonus += strong_surf_boost
                                try:
                                    logger.info(
                                        f"[MTF] Boost de surf forte aplicado: +{strong_surf_boost:.3f}"
                                    )
                                except Exception:
                                    pass
                            # Recompensa disciplina em macro sem forÃ§a
                            elif forca_tendencia_macro < 0.25 and num_trades < 5:
                                discipline_bonus = 1.0 * (1.0 - forca_tendencia_macro)
                                score_ajuste_atividade = discipline_bonus
                                logger.info(
                                    f"[MTF] Trial recompensado por disciplina (trades={num_trades}) em MACRO SEM FORÃ‡A (ForÃ§a={forca_tendencia_macro:.2f}). Ajuste={score_ajuste_atividade:.2f}"
                                )
                    else:
                        logger.warning(
                            "[MTF] Colunas/DF de Multi-Timeframe nÃ£o disponÃ­veis para ajuste de atividade."
                        )
                except Exception as _e:
                    try:
                        logger.debug(
                            f"[MTF] Ajuste de atividade desativado por erro: {_e}"
                        )
                    except Exception:
                        pass
                score += score_ajuste_atividade
                score += trend_surf_bonus
                logger.info(
                    f"[TREND SAC OPTUNA] <<< TRIAL {trial.number} BASE | score_base={score:.4f} trades={num_trades} pf={profit_factor:.3f} sharpe={trade_sharpe:.3f}"
                )

                # GUARDA: Inatividade severa invalida o trial independente do contexto macro
                n_episodes = max(1, len(episode_summaries))
                MIN_TRADES_PER_EPISODE = 8
                min_total_trades = MIN_TRADES_PER_EPISODE * n_episodes  # 8 × 5 = 40
                if num_trades < min_total_trades:
                    inactivity_penalty = (
                        (min_total_trades - num_trades) * 0.4 / n_episodes
                    )
                    score -= inactivity_penalty
                    logger.info(
                        f"[GUARDA INATIVIDADE] Trial {trial.number}: {num_trades:.0f} trades totais "
                        f"({num_trades / n_episodes:.1f}/episódio) < mínimo {MIN_TRADES_PER_EPISODE}/episódio. "
                        f"Penalidade: -{inactivity_penalty:.2f}"
                    )
                    if (num_trades / n_episodes) <= 3:
                        raise optuna.exceptions.TrialPruned(
                            f"Inatividade patológica: {num_trades / n_episodes:.1f} trades/episódio"
                        )

                # [REMOVIDO] Regra rÃ­gida antiga de nÃ£o operar
                # if num_trades < 20:
                #     score = -10.0

                # [FIX SCALP LIMIT] Limite baseado em duração real do episódio, não valor fixo.
                # BUG anterior: scalp_limit=44/ep era 4-5x menor que o volume real de trades
                # (ep=5506 steps, avg_dur=10 → ~183 trades/ep esperados).
                # Resultado: penalty=(665-220)*0.02=8.9 destruía score mesmo com Sharpe>0 e PF>3.
                # Fix: usa avg_trade_duration para derivar volume esperado. Se o agente está
                # segurando trades longos, o volume alto é saudável, não scalping.
                n_eval_eps = max(1, len(episode_summaries))
                ep_steps_eval = 7062  # comprimento médio do episódio de avaliação
                if avg_trade_duration > 1.0:
                    # Volume esperado: ep_steps / avg_duration * fator_flat (50% do tempo em trade)
                    expected_trades_per_ep = (
                        ep_steps_eval / max(avg_trade_duration, 1.0) * 0.5
                    )
                else:
                    expected_trades_per_ep = 200.0
                scalp_limit = (
                    int(expected_trades_per_ep * 1.5) * n_eval_eps
                )  # 50% de folga
                if num_trades > scalp_limit:
                    scalp_penalty_weight = float(
                        os.getenv("TREND_SCALP_WEIGHT", "0.005")
                    )  # 0.02→0.005
                    score -= (num_trades - scalp_limit) * scalp_penalty_weight
                # Penaliza duração média MUITO curta (real scalping: avg < 5 steps)
                if num_trades >= 50 and avg_trade_duration < 5.0:
                    score -= 0.15 * float(5.0 - avg_trade_duration)

                # Pruning adicional: muitos trades e duraÃƒÂ§ÃƒÂ£o mÃƒÂ©dia muito baixa
                if num_trades > 500 and avg_trade_duration < 5.0:
                    raise optuna.exceptions.TrialPruned()
                # Aplica deslocamento positivo e piso em zero AO FINAL
                trend_shift = float(os.getenv("TREND_SCORE_SHIFT", "0.8"))
                score = float(max(0.0, score + trend_shift))
                logger.info(
                    "Recompensa MÃƒÂ©dia: %.4f +/- %.4f | PF: %.3f | Sharpe(trade): %.3f | AvgRet: %.4f | MaxDD: %.4f | Trades: %.0f | ScoreFinal: %.6f",
                    mean_reward,
                    std_reward,
                    profit_factor,
                    trade_sharpe,
                    avg_return_pct,
                    max_drawdown_pct,
                    num_trades,
                    score,
                )
                logger.info("---------------------------------")
                # [TRIAL MATRIX] Imprime trades do ultimo episodio de avaliacao
                try:
                    eval_env.env_method("_log_trial_matrix", trial.number, score)
                except Exception as _mat_err:
                    logger.warning(
                        "[TRIAL MATRIX] Falha ao imprimir matriz: %s", _mat_err
                    )
                # Anexa mÃ©tricas ao trial para auditoria/automaÃ§Ã£o
                try:
                    gate_stats_list = train_env.env_method("get_gate_statistics")
                    if isinstance(gate_stats_list, list) and gate_stats_list:
                        trial.set_user_attr("gate_stats", gate_stats_list[0])
                except Exception:
                    pass
                try:
                    # Marca se houve comportamento de surf de tendÃªncia
                    is_trend_surf = False
                    try:
                        forca_tendencia_macro
                        direcao_macro
                    except NameError:
                        forca_tendencia_macro = 0.0
                        direcao_macro = 0.0
                    if (
                        forca_tendencia_macro > 0.4
                        and abs(direcao_macro) > 0.4
                        and float(avg_trade_duration or 0.0) >= 20.0
                        and float(profit_factor or 0.0) >= 1.0
                    ):
                        is_trend_surf = True
                    trial.set_user_attr("trend_surf_bonus", float(trend_surf_bonus))
                    trial.set_user_attr(
                        "forca_tendencia_macro", float(forca_tendencia_macro)
                    )
                    trial.set_user_attr(
                        "avg_trade_duration", float(avg_trade_duration or 0.0)
                    )
                    trial.set_user_attr("profit_factor", float(profit_factor))
                    trial.set_user_attr("trade_sharpe", float(trade_sharpe))
                    trial.set_user_attr("max_drawdown_pct", float(max_drawdown_pct))
                    trial.set_user_attr("num_trades", float(num_trades))
                    trial.set_user_attr("is_trend_surf", bool(is_trend_surf))
                    trial.set_user_attr("score_final", float(score))
                except Exception:
                    pass
                return float(score)
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"[TREND SAC OPTUNA] Trial falhou: {e}", exc_info=True)
                return -float("inf")
            finally:
                try:
                    if train_env is not None:
                        train_env.close()
                    if eval_env is not None:
                        eval_env.close()
                except Exception:
                    pass

                # 🧹 LIMPEZA PROFUNDA (Evita CUDA error: unknown error)
                try:
                    if "model" in locals():
                        del model
                    if "train_env" in locals():
                        del train_env
                    if "eval_env" in locals():
                        del eval_env
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    if eval_env is not None:
                        eval_env.close()
                except Exception:
                    pass
                try:
                    if self.reward_hyperparams:
                        self._apply_reward_params_to_config(self.reward_hyperparams)
                except Exception:
                    pass

        if self.reset_optuna_db and os.path.exists(self.optuna_db_path):
            try:
                os.remove(self.optuna_db_path)
                logger.info("[TREND SAC] Banco de optuna reiniciado por configuracao.")
            except Exception as exc:
                logger.warning(
                    f"[TREND SAC] Falha ao remover banco de optuna existente: {exc}"
                )
        storage_url = f"sqlite:///{self.optuna_db_path}"
        # Pruner paciente para dar tempo ao aprendizado
        disable_pruner = os.getenv("TREND_OPTUNA_NOPRUNE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if disable_pruner:
            pruner = optuna.pruners.NopPruner()
            logger.info(
                "[TREND SAC OPTUNA] Pruner desabilitado por variavel de ambiente."
            )
        else:
            # [v3.1] MedianPruner calibrado para 100k steps × 28 hyperparâmetros.
            # n_startup_trials=10: TPE precisa de base antes de prunar — era 3 (insuficiente).
            # n_warmup_steps=15000: 15% de 100k — deixa o agente se estabilizar antes de avaliar.
            # interval_steps=2000: checkpoints frequentes após warmup.
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=10, n_warmup_steps=15000, interval_steps=2000
            )
            logger.info(
                "[TREND SAC OPTUNA] MedianPruner v3.1: startup=10, warmup=15000, interval=2000"
            )
        try:
            pruner_name = os.getenv("TREND_OPTUNA_PRUNER", "median").strip().lower()
            if pruner_name == "sh":
                pruner = optuna.pruners.SuccessiveHalvingPruner(
                    min_resource=6000, reduction_factor=3, min_early_stopping_rate=0
                )
            elif pruner_name == "nop":
                pruner = optuna.pruners.NopPruner()
        except Exception:
            pass
        study = optuna.create_study(
            direction="maximize",
            storage=storage_url,
            study_name=f"{self.specialist_name}_sac_optimization_v3",
            load_if_exists=True,
            pruner=pruner,
        )
        # Verifica quantos trials jÃƒÂ¡ foram executados
        completed_trials = len(study.trials)
        logger.info(
            f"[TREND SAC OPTUNA] Trials jÃƒÂ¡ executados: {completed_trials}, Objetivo: {n_trials}"
        )
        # Se jÃƒÂ¡ temos trials suficientes, nÃƒÂ£o executa mais
        if completed_trials >= n_trials:
            logger.info(
                f"[TREND SAC OPTUNA] JÃƒÂ¡ temos {completed_trials} trials completados. Usando resultados existentes."
            )
            remaining_trials = 0
        else:
            remaining_trials = n_trials - completed_trials
            logger.info(
                f"[TREND SAC OPTUNA] Executando {remaining_trials} trials adicionais..."
            )
        # [GPU MODE] Trials sequenciais na GPU (n_jobs=1).
        # n_jobs=2 com CUDA causa conflito de memória entre threads no mesmo processo.
        # GPU compensa com throughput superior por trial vs 2 trials paralelos em CPU.
        try:
            max_workers_env = int(os.getenv("TREND_OPTUNA_JOBS", "1"))  # default 2→1
            max_workers = max(1, min(4, max_workers_env))
        except Exception:
            max_workers = 1
        if remaining_trials > 0:
            # [PROGRESS BAR] Barra de progresso visual para HPO dos especialistas
            try:
                from tqdm import tqdm as _tqdm

                class _HpoProgressCallback:
                    """Callback Optuna que mostra barra de progresso + health summary."""

                    def __init__(self, n_total: int, specialist_name: str, monitor):
                        import sys as _sys
                        _is_interactive = hasattr(_sys.stderr, "fileno") and _sys.stderr.isatty()
                        self._pbar = _tqdm(
                            total=n_total,
                            desc=f"🎯 {specialist_name} HPO",
                            unit="trial",
                            dynamic_ncols=True,
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                            disable=not _is_interactive,
                        )
                        self._best = float("-inf")
                        self._monitor = monitor

                    def __call__(self, study, trial):
                        val = trial.value if trial.value is not None else float("nan")
                        if val > self._best:
                            self._best = val
                        state_icon = "✅" if trial.state.name == "COMPLETE" else "⚠️"
                        self._pbar.update(1)
                        self._pbar.set_postfix(
                            {
                                "best": f"{self._best:.3f}",
                                "last": f"{val:.3f}"
                                if trial.value is not None
                                else "pruned",
                                "st": state_icon,
                            },
                            refresh=True,
                        )
                        # Log health dashboard a cada 10 trials
                        if self._monitor is not None and (trial.number + 1) % 10 == 0:
                            try:
                                self._monitor.print_health_dashboard(
                                    trial_number=trial.number + 1
                                )
                            except Exception:
                                pass

                    def close(self):
                        self._pbar.close()

                _monitor_ref = getattr(self, "learning_monitor", None)
                _hpo_cb = _HpoProgressCallback(
                    remaining_trials, self.specialist_name, _monitor_ref
                )
                study.optimize(
                    objective,
                    n_trials=remaining_trials,
                    n_jobs=max_workers,
                    callbacks=[_hpo_cb],
                )
                _hpo_cb.close()
            except ImportError:
                # tqdm não instalado — roda sem barra
                study.optimize(objective, n_trials=remaining_trials, n_jobs=max_workers)
        try:
            best_trial = study.best_trial
        except ValueError:
            best_trial = None
        if best_trial and best_trial.value is not None:
            best_params = dict(best_trial.params)
            sac_params = {
                key: best_params[key]
                for key in self.SAC_PARAM_KEYS
                if key in best_params
            }
            if sac_params:
                self.hyperparams.update(sac_params)
                self._save_hyperparams()
            reward_params = {
                key: best_params[key]
                for key in self.REWARD_PARAM_KEYS
                if key in best_params
            }
            if reward_params:
                self.reward_hyperparams = reward_params
                self._apply_reward_params_to_config(self.reward_hyperparams)
                self._save_reward_params(self.reward_hyperparams)

            # [Y2 FIX] Avaliação no hold-out após seleção do best trial.
            # López de Prado (MLAM Ch.7): o best trial foi selecionado pelo val set —
            # sem avaliação no hold-out, inflação de performance por data-snooping não é detectada.
            # Esta etapa NÃO muda o best trial selecionado (seria necessário re-treinar N trails),
            # mas emite WARNING se a degradação val→holdout for > 30%, sinalizando overfitting.
            _holdout_df = getattr(self, "holdout_df", None)
            if _holdout_df is not None and len(_holdout_df) > 200:
                try:
                    _holdout_env_fn = lambda: self._make_trend_env(
                        _holdout_df, mode="optimization", env_class=env_class
                    )
                    _val_score = float(best_trial.value)
                    # Usar os user_attrs do best trial para comparar métricas financeiras
                    _val_sharpe = float(best_trial.user_attrs.get("trade_sharpe", 0.0))
                    _val_pf = float(best_trial.user_attrs.get("profit_factor", 0.0))
                    logger.info(
                        "[Y2 HOLDOUT] Best trial %d: val_score=%.4f | val_sharpe=%.3f | val_PF=%.3f",
                        best_trial.number,
                        _val_score,
                        _val_sharpe,
                        _val_pf,
                    )
                    logger.info(
                        "[Y2 HOLDOUT] AVISO: avaliação completa no hold-out requer re-treino "
                        "do best trial — use Fase 3 com holdout_validation_callback para "
                        "verificar generalização. val_score=%.4f é estimativa otimista.",
                        _val_score,
                    )
                    # Verificar se degradação estimada é crítica (heurística simples)
                    # Se o val_score dos top-3 trials tem alta variância, HPO pode estar sofrendo
                    # de alta instabilidade — melhor indicador disponível sem re-treino.
                    _top_trials = sorted(
                        [t for t in study.trials if t.value is not None],
                        key=lambda t: t.value,
                        reverse=True,
                    )[:5]
                    if len(_top_trials) >= 3:
                        _top_scores = [t.value for t in _top_trials]
                        _top_cv = float(
                            np.std(_top_scores) / (abs(np.mean(_top_scores)) + 1e-9)
                        )
                        if _top_cv > 0.5:
                            logger.warning(
                                "[Y2 HOLDOUT] Alta variância nos top-5 trials (CV=%.2f) — "
                                "best trial pode ser um outlier estatístico. "
                                "Considere aumentar n_timesteps ou reduzir n_trials.",
                                _top_cv,
                            )
                except Exception as _y2_err:
                    logger.debug("[Y2 HOLDOUT] Avaliação skipped: %s", _y2_err)
        else:
            logger.warning("Otimizacao concluida sem nenhum trial completo ou valido.")

    def _initialize_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = ClippedSAC.load(
                    self.model_path,
                    device=self._get_device(),
                    max_grad_norm=self.max_grad_norm,
                )
                self.is_trained = True
                logger.info(
                    "[TREND SAC] TrendSpecialist (SAC) carregado de '%s'.",
                    self.model_path,
                )
            except ModuleNotFoundError as exc:
                logger.warning(
                    "[TREND SAC] Checkpoint legado incompatÃ­vel com a versÃ£o atual de dependÃªncias (%s). Recriando o modelo do zero.",
                    exc,
                )
                self.model = None
                self.is_trained = False
            except Exception as e:
                logger.error(f"[ERRO TREND SAC] Falha ao carregar: {e}", exc_info=True)
                self.is_trained = False
        else:
            logger.info("[TREND SAC] Nenhum TrendSpecialist (SAC) treinado encontrado.")

    def train(
        self,
        featured_df: pd.DataFrame,
        timesteps_to_train: int,
        env_class=None,
        specialist_name=None,
    ) -> int:
        """
        [VERSÃO DE NÍVEL INSTITUCIONAL]
        """
        # [BUG 1 FIX] specialist_env movido para após definição de feature_columns

        # [REGIME FILTER] Filtra o dataset para o regime específico do specialist.
        # Problema raiz: BullSpecialist recebia 45.3% bear + 22.4% ranger = 67.7% de dados não-bull,
        # fazendo o agente aprender a shortar em vez de comprar (viés bearish confirmado nos logs).
        # Bear specialist recebe o complemento. Ranger recebe tudo (opera em ambas direções).
        _sname = (specialist_name or self.specialist_name or "").lower()
        if featured_df is not None and len(featured_df) > 0:
            _regime_col = None
            for _col in ("regime_name", "tp_regime_name", "market_regime", "regime"):
                if _col in featured_df.columns:
                    _regime_col = _col
                    break
            if _regime_col is not None:
                _orig_len = len(featured_df)
                if "bull" in _sname and "bear" not in _sname:
                    _mask = (
                        featured_df[_regime_col]
                        .str.lower()
                        .str.contains("bull", na=False)
                    )
                    _filtered = featured_df[_mask]
                    if len(_filtered) >= 5000:
                        featured_df = _filtered.copy()

                        # [TEMPORAL WEIGHTING] Mitigar viés temporal: 2025-2026 é bull regime mas
                        # com estrutura de queda. Pesar amostras antigas (2023-2024) mais que recentes.
                        # Problema: mesmo em regime bull, timing de entrada pode ser ruim se mercado
                        # estruturalmente em alta mas com pullbacks recentes.
                        try:
                            if isinstance(featured_df.index, pd.DatetimeIndex):
                                _dates = featured_df.index
                                _cutoff_recent = _dates[int(len(_dates) * 0.70)]
                                _recent_mask = _dates >= _cutoff_recent

                                # Amostras antigas (2023-2024): weight=1.0
                                # Amostras recentes (2025-2026): weight=0.4 (downweight 60%)
                                _sample_weights = np.ones(len(featured_df))
                                _sample_weights[_recent_mask] = 0.4

                                # [LEAKAGE FIX 2026-05-13] replace=False evita duplicacao temporal
                                # silenciosa que envenena o replay buffer do SAC (P(s,a,r,s')
                                # duplicado dezenas de vezes -> critic overconfidence + entropy
                                # collapse). Antes: replace=True permitia mesmo candle aparecer
                                # 5-10x no buffer durante 25k+ samples.
                                _target_size = min(len(featured_df), 25000)
                                if _target_size >= len(featured_df):
                                    # sem reducao — mantem df original com pesos como coluna
                                    featured_df = featured_df.copy()
                                    featured_df["_temporal_weight"] = _sample_weights
                                else:
                                    featured_df = featured_df.sample(
                                        n=_target_size,
                                        weights=_sample_weights,
                                        replace=False,
                                        random_state=int(
                                            len(featured_df) + _target_size
                                        )
                                        % (2**31),
                                    ).sort_index()

                                _recent_pct = (
                                    100.0 * _recent_mask.sum() / len(_filtered)
                                )
                                logger.info(
                                    "[REGIME FILTER + TEMPORAL WEIGHT] BullSpecialist: %d → %d linhas (%.1f%% bull regime). "
                                    "Downweight 2025-2026 (%.1f%% das amostras) para mitigar viés temporal.",
                                    _orig_len,
                                    len(featured_df),
                                    100.0 * len(featured_df) / _orig_len,
                                    _recent_pct,
                                )
                            else:
                                featured_df = _filtered
                                logger.info(
                                    "[REGIME FILTER] BullSpecialist: %d → %d linhas (%.1f%% bull regime retido). "
                                    "[TEMPORAL WEIGHT] Não aplicado — index não é DatetimeIndex.",
                                    _orig_len,
                                    len(featured_df),
                                    100.0 * len(featured_df) / _orig_len,
                                )
                        except Exception as _tw_err:
                            featured_df = _filtered
                            logger.warning(
                                "[TEMPORAL WEIGHT] Erro ao aplicar downweight temporal: %s. Usando bull-filtered dataset sem temporal weighting.",
                                _tw_err,
                            )
                    else:
                        logger.warning(
                            "[REGIME FILTER] BullSpecialist: apenas %d linhas bull (< 5000) — usando dataset completo para evitar underfitting.",
                            len(_filtered),
                        )
                elif "bear" in _sname and "bull" not in _sname:
                    _mask = (
                        featured_df[_regime_col]
                        .str.lower()
                        .str.contains("bear", na=False)
                    )
                    _filtered = featured_df[_mask]
                    if len(_filtered) >= 5000:
                        featured_df = _filtered.copy()

                        # [TEMPORAL WEIGHTING] Para BearSpecialist, 2024-2026 teve muitos movimentos
                        # estruturalmente bearish, mas recentes podem ter pullbacks em uptrend maior.
                        # Pesar dados antigos (2023-2024) mais para evitar overfitting em viés recente.
                        try:
                            if isinstance(featured_df.index, pd.DatetimeIndex):
                                _dates = featured_df.index
                                _cutoff_recent = pd.Timestamp("2025-01-01")
                                _recent_mask = _dates >= _cutoff_recent

                                # Amostras antigas: weight=1.0, recentes: weight=0.4
                                _sample_weights = np.ones(len(featured_df))
                                _sample_weights[_recent_mask] = 0.4

                                # [LEAKAGE FIX 2026-05-13] replace=False — vide nota no Bull acima
                                _target_size = min(len(featured_df), 25000)
                                if _target_size >= len(featured_df):
                                    featured_df = featured_df.copy()
                                    featured_df["_temporal_weight"] = _sample_weights
                                else:
                                    featured_df = featured_df.sample(
                                        n=_target_size,
                                        weights=_sample_weights,
                                        replace=False,
                                        random_state=int(
                                            len(featured_df) + _target_size
                                        )
                                        % (2**31),
                                    ).sort_index()

                                _recent_pct = (
                                    100.0 * _recent_mask.sum() / len(_filtered)
                                )
                                logger.info(
                                    "[REGIME FILTER + TEMPORAL WEIGHT] BearSpecialist: %d → %d linhas (%.1f%% bear regime). "
                                    "Downweight 2025-2026 (%.1f%% das amostras).",
                                    _orig_len,
                                    len(featured_df),
                                    100.0 * len(featured_df) / _orig_len,
                                    _recent_pct,
                                )
                            else:
                                featured_df = _filtered
                                logger.info(
                                    "[REGIME FILTER] BearSpecialist: %d → %d linhas (%.1f%% bear regime retido). "
                                    "[TEMPORAL WEIGHT] Não aplicado — index não é DatetimeIndex.",
                                    _orig_len,
                                    len(featured_df),
                                    100.0 * len(featured_df) / _orig_len,
                                )
                        except Exception as _tw_err:
                            featured_df = _filtered
                            logger.warning(
                                "[TEMPORAL WEIGHT] Erro ao aplicar downweight temporal: %s. Usando bear-filtered dataset sem temporal weighting.",
                                _tw_err,
                            )
                    else:
                        logger.warning(
                            "[REGIME FILTER] BearSpecialist: apenas %d linhas bear (< 5000) — usando dataset completo.",
                            len(_filtered),
                        )
            else:
                logger.warning(
                    "[REGIME FILTER] Coluna de regime não encontrada no dataset (procurou: regime_name, tp_regime_name, market_regime, regime). "
                    "BullSpecialist receberá dataset completo — viés bearish pode persistir."
                )

        if timesteps_to_train <= 0:
            logger.info(
                "[TREND SAC] Nenhum timestep adicional necessario. Objetivo de treino ja alcancado. Pulando."
            )
            return self.model.num_timesteps if self.model else 0
        logger.info(
            f"[TREND SAC] Iniciando ou continuando treinamento por {timesteps_to_train} timesteps..."
        )
        try:
            granular_mode = bool(getattr(self.config, "GRANULAR_TRAINING_TREND", False))
            # FASE 2: Otimização de hiperparâmetros SAC via Optuna
            # 🔧 FIX: Carregar hiperparâmetros se não foram carregados ainda
            if not self._hyperparams_loaded_from_file and os.path.exists(
                self.hyperparams_path
            ):
                self._load_hyperparams()
                logger.info(
                    f"[FASE 2] Hiperparâmetros carregados de '{self.hyperparams_path}'. Pulando otimização."
                )

            # Só roda otimização se: modo granular OU forçado OU não tem hyperparams
            should_run_phase2 = self.force_optuna_retrain or (
                not self._hyperparams_loaded_from_file
                and not os.path.exists(self.hyperparams_path)
            )
            if should_run_phase2:
                setup_phase_logger("TrendSpecialist_Phase2_Hyperparams", logger.name)
                logger.info(
                    "[FASE 2] Iniciando otimizaÃƒÂ§ÃƒÂ£o de hiperparÃƒÂ¢metros SAC..."
                )
                # [MONITOR] Marca fase HPO
                try:
                    self.learning_monitor.set_phase("hpo")
                    self.learning_monitor.reset()
                except Exception:
                    pass
                if self.force_optuna_retrain:
                    logger.info(
                        "[FASE 2] Reexecutando otimizacao do Optuna conforme configuracao."
                    )
                # [FIX CIENTÍFICO] Bypass de escala in-place para Fase 2
                from sklearn.preprocessing import RobustScaler, StandardScaler
                from sklearn.compose import ColumnTransformer

                # [v3.1 HOLD-OUT TEMPORAL] Reserva os últimos 10% do dataset como
                # conjunto de avaliação final completamente isolado do treinamento.
                # López de Prado (MLAM Ch.7): nunca usar dados futuros para otimização.
                _holdout_ratio = float(os.environ.get("TREND_HOLDOUT_RATIO", "0.10"))
                _holdout_idx = int(len(featured_df) * (1.0 - _holdout_ratio))
                _train_pool = featured_df.iloc[:_holdout_idx]
                self.holdout_df = featured_df.iloc[_holdout_idx:].copy()
                self.logger.info(
                    "[HOLD-OUT] Últimos %.0f%% reservados para avaliação final: %d amostras (de %d total).",
                    _holdout_ratio * 100,
                    len(self.holdout_df),
                    len(featured_df),
                )
                train_df_opt, eval_df_opt = train_test_split(
                    _train_pool, test_size=0.2, shuffle=False
                )
                train_df_opt = train_df_opt.copy()
                eval_df_opt = eval_df_opt.copy()

                ohlcv_cols = ["open", "high", "low", "close", "volume"]
                protected_cols = set(ohlcv_cols + ["tp_prior_dir", "tp_prior_conf"])
                num_cols_all = [
                    c
                    for c in train_df_opt.select_dtypes(include=np.number).columns
                    if c not in protected_cols
                ]

                # Limpeza (in-place é ok no subset_opt que é cópia)
                # [ANTI-LEAKAGE] bfill() removido — propaga valores futuros. Apenas ffill().
                for df_tmp in (train_df_opt, eval_df_opt):
                    df_tmp.loc[:, num_cols_all] = df_tmp[num_cols_all].ffill().fillna(0)

                scaler_opt_pipeline = ColumnTransformer(
                    [
                        ("robust", RobustScaler(), ohlcv_cols),
                        ("standard", StandardScaler(), num_cols_all),
                    ],
                    remainder="passthrough",
                )

                scaler_opt_pipeline.fit(train_df_opt)
                self.feature_scaler = scaler_opt_pipeline

                # Timesteps por trial: 100k para RTX 3500 Ada (~18min/trial).
                # 50k era insuficiente — política aprendia no treino mas não generalizava
                # para o eval (0 trades no eval por falta de gradient updates suficientes).
                self.optimize_hyperparameters(
                    train_df_opt, eval_df_opt, n_trials=50, n_timesteps=100000
                )
                self.force_optuna_retrain = False
                self._hyperparams_loaded_from_file = True
            setup_phase_logger("TrendSpecialist_Phase3_FinalTrain", logger.name)
            logger.info("[FASE 3] Iniciando treinamento final do modelo SAC...")
            # [MONITOR] Marca fase treinamento final e verifica circuit breaker
            try:
                self.learning_monitor.set_phase("training")
                self.learning_monitor.reset()
                # Se circuit breaker ativo da HPO, reseta para dar chance ao treino final
                if self.learning_monitor.circuit_breaker_active:
                    logger.warning(
                        "[MONITOR] Circuit breaker da fase HPO resetado para treino final."
                    )
                    self.learning_monitor.reset_circuit_breaker()
            except Exception:
                pass
            # Reutiliza o hold-out separado na Fase 2 (se disponível), caso contrário cria novo
            _feat_for_train = getattr(self, "holdout_df", None)
            if _feat_for_train is None:
                _holdout_ratio = float(os.environ.get("TREND_HOLDOUT_RATIO", "0.10"))
                _holdout_idx = int(len(featured_df) * (1.0 - _holdout_ratio))
                _feat_for_train_pool = featured_df.iloc[:_holdout_idx]
                self.holdout_df = featured_df.iloc[_holdout_idx:].copy()
            else:
                _holdout_idx = int(len(featured_df) * 0.90)
                _feat_for_train_pool = featured_df.iloc[:_holdout_idx]
            train_df, eval_df = train_test_split(
                _feat_for_train_pool, test_size=0.15, shuffle=False
            )

            # [RESTAURAÇÃO] Limpeza de NaNs (essencial para o RL)
            for df_tmp in (train_df, eval_df):
                num_cols_tmp = df_tmp.select_dtypes(include=np.number).columns
                df_tmp.loc[:, num_cols_tmp] = df_tmp[num_cols_tmp].ffill().fillna(0)

            # [FIX CIENTÍFICO] Define contrato de features ANTES do fit do scaler
            # Isso garante que o scaler seja treinado apenas nas colunas que o ambiente enviará
            feature_columns = [
                c
                for c in train_df.select_dtypes(include=np.number).columns
                if c != "tp_prior_dir"
            ]
            self.feature_columns = (
                feature_columns  # Persiste na instância para inferência
            )

            # [FIX SCALER PHASE2→PHASE3] Limpa o scaler antigo da Fase 2 (HPO) ANTES de
            # criar o env da Fase 3. O scaler da HPO foi fit em um conjunto de colunas
            # diferente do que o env da Fase 3 usará (env aplica _filter_to_core_features
            # internamente). Sem esse reset, a criação do env dispara _normalize_market_features
            # que chama o scaler velho com o nº de cols novo → warning "X has 65 features,
            # expecting 63" + fallback para features brutas na inicialização.
            # O scaler correto será ajustado e atribuído logo abaixo (L4990).
            self._feature_scaler = None
            self.feature_scaler = None
            # Limpa flag de warning para que, se houver outro problema depois, ele apareça.
            if hasattr(self, "_scaler_error_shown"):
                delattr(self, "_scaler_error_shown")
            if hasattr(self, "_scaler_warning_shown"):
                delattr(self, "_scaler_warning_shown")

            # [BUG 1 FIX] Criar env de avaliação com contrato de features completo
            self.specialist_env = self._make_trend_env(
                featured_df,
                mode="training",
                feature_columns=self.feature_columns,
                env_class=env_class,
                specialist_name=specialist_name,
            )
            # [FIX SCALER MISMATCH] Lê feature_columns do env APÓS criação, pois o env aplica
            # _filter_to_core_features() internamente (183→150). Sem isso, o scaler é fit em 183
            # colunas mas o env usa 150 → erro "X has 150 features, RobustScaler expecting 183".
            feature_columns = list(self.specialist_env.feature_columns)
            self.feature_columns = feature_columns
            logger.info(
                f"[TRAIN] Contrato de features sincronizado com env: {len(feature_columns)} colunas core."
            )

            import joblib

            # [FIX CIENTÍFICO] Criar um Pipeline de Escalonamento Robusto sem transformar o DF original
            # Isso garante que o PnL seja calculado sobre os preços reais
            from sklearn.compose import ColumnTransformer

            # [FIX DYNAMIC OHLCV] Apenas inclui OHLCV no RobustScaler se estiverem em feature_columns.
            # Após _filter_to_core_features, OHLCV pode não estar no contrato de features.
            _all_ohlcv = ["open", "high", "low", "close", "volume"]
            ohlcv_cols_main = [c for c in _all_ohlcv if c in feature_columns]
            protected_cols_main = set(
                _all_ohlcv + ["tp_prior_conf"]
            )  # 'tp_prior_dir' já foi removido de feature_columns
            # num_cols_all_main deve ser um subset de feature_columns
            num_cols_all_main = [
                c for c in feature_columns if c not in protected_cols_main
            ]

            # Construtor do Scaler para observações (não altera o DataFrame bruto)
            from sklearn.preprocessing import RobustScaler, StandardScaler

            _transformers = [("standard", StandardScaler(), num_cols_all_main)]
            if ohlcv_cols_main:
                _transformers.insert(0, ("robust", RobustScaler(), ohlcv_cols_main))
            scaler_pipeline = ColumnTransformer(_transformers, remainder="passthrough")

            # [SCIENTIFIC FIX] Treina o scaler APENAS nas colunas que o ambiente usará
            # Isso resolve o erro "X has 52 features, but ColumnTransformer is expecting 54"
            scaler_pipeline.fit(train_df[feature_columns])
            self.set_feature_scaler(scaler_pipeline)

            # Exporta para uso em produção
            scaler = scaler_pipeline  # Para manter compatibilidade com as linhas de dump abaixo
            std_scaler_main = scaler_pipeline  # Para manter compatibilidade com as linhas de dump abaixo (embora repetitivo)
            # Persiste o contrato de features e o scaler para uso consistente em produÃ§Ã£o/inferÃªncia
            models_dir = os.path.join(os.getcwd(), "models_ai")
            os.makedirs(models_dir, exist_ok=True)
            scaler_path = os.path.join(
                models_dir, f"{self.specialist_name}_scaler.joblib"
            )
            std_scaler_path = (
                scaler_path  # Usa o mesmo path, pois estão no mesmo ColumnTransformer
            )
            joblib.dump(scaler, scaler_path)
            # Legado (para fallback):
            joblib.dump(
                scaler, os.path.join(models_dir, "feature_scaler_robust.joblib")
            )
            # feature_columns já definido acima
            metadata_path = os.path.join(models_dir, "training_metadata.json")
            try:
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                else:
                    meta = {}
            except Exception:
                meta = {}
            meta.update(
                {
                    "base_feature_columns": feature_columns,
                    "final_feature_count": len(feature_columns),
                    "scaler_type": "Robust+Standard",
                    "robust_scaler_path": scaler_path,
                    "standard_scaler_path": std_scaler_path,
                    "standard_scaler_columns": num_cols_all_main,
                    "protected_columns": sorted(list(protected_cols_main)),
                    "specialist": self.specialist_name,
                    "last_full_training_date": datetime.utcnow().isoformat(),
                }
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            # [FRAME STACKING] Número de candles empilhados como sequência temporal real para o LSTM.
            # VecFrameStack(n_stack=N) concatena as últimas N observações em um vetor flat
            # (obs_dim * N), que o HybridLSTMAttentionExtractor faz reshape para (N, obs_dim)
            # antes de passar ao LSTM — dando ao agente memória temporal real de N candles.
            N_STACK_FRAMES = 4

            num_envs = int(os.getenv("TREND_TRAIN_NUM_ENVS", "2"))
            train_env = DummyVecEnv(
                [
                    lambda: self._make_trend_env(
                        train_df,
                        mode="training",
                        feature_columns=feature_columns,
                        env_class=env_class,
                        specialist_name=specialist_name,
                    )
                    for _ in range(num_envs)
                ]
            )
            train_env = VecFrameStack(
                train_env, n_stack=N_STACK_FRAMES
            )  # ← sequência temporal real
            train_env = VecNormalize(
                train_env, norm_obs=False, norm_reward=False, clip_reward=10.0
            )

            eval_env = DummyVecEnv(
                [
                    lambda: self._make_trend_env(
                        eval_df,
                        mode="training",
                        feature_columns=feature_columns,
                        env_class=env_class,
                        specialist_name=specialist_name,
                    )
                ]
            )
            eval_env = VecFrameStack(
                eval_env, n_stack=N_STACK_FRAMES
            )  # ← sequência temporal real
            eval_env = VecNormalize(
                eval_env, norm_obs=False, norm_reward=False, clip_reward=10.0
            )

            env_reward_params = self._convert_reward_params_to_env(
                self.reward_hyperparams
            )
            if env_reward_params:
                train_env.env_method("set_reward_shaping_params", env_reward_params)
                eval_env.env_method("set_reward_shaping_params", env_reward_params)
            # Aplicar tweaks de Fase 3 nos ambientes
            try:
                train_env.env_method("set_phase3_runtime_tweaks")
                eval_env.env_method("set_phase3_runtime_tweaks")
            except Exception:
                pass
            # 💾 [FIX] Cada specialist tem seu próprio diretório de checkpoints
            checkpoint_dir = os.path.join(
                self.config.CHECKPOINT_DIR,
                f"{self.specialist_name.lower()}_checkpoints",
            )
            best_model_path = os.path.join(checkpoint_dir, "best_model.zip")
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.info(f"💾 [CHECKPOINT] Diretório de checkpoints: {checkpoint_dir}")
            # --- LÃƒâ€œGICA DE NÃƒÂVEL INSTITUCIONAL ---
            early_stopping_callback = EarlyStoppingCallback(
                patience=self.config.PATIENCE_EARLY_STOPPING, verbose=1
            )
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=checkpoint_dir,
                log_path=checkpoint_dir,
                eval_freq=self.config.CHECKPOINT_FREQ,
                deterministic=True,
                render=False,
                n_eval_episodes=10,  # 🔬 Fix: era 5
                callback_after_eval=early_stopping_callback,
            )
            trade_analysis_callback = TradeAnalysisCallback(
                eval_env=eval_env, n_eval_episodes=10
            )  # 🔬 Fix: era 5

            # 🔬 [SCIENTIFIC] Callbacks para métricas avançadas e prevenção de policy collapse
            try:
                from trading.callbacks.scientific_rl_callback import (
                    ScientificRLCallback,
                    EntropyRegularizationCallback,
                    ActionDiversityCallback,
                )

                scientific_callback = ScientificRLCallback(
                    eval_freq=1000, log_freq=100, verbose=1
                )
                entropy_callback = EntropyRegularizationCallback(
                    min_entropy_threshold=-10.0, check_freq=1000, verbose=1
                )
                action_diversity_callback = ActionDiversityCallback(
                    check_freq=500, min_action_std=0.1, exploration_noise=0.1, verbose=1
                )

                # 📊 [CLEAN LOGGING] Callback para métricas limpas com matriz de confusão
                from trading.callbacks.clean_metrics_callback import (
                    CleanMetricsCallback,
                )

                clean_metrics_callback = CleanMetricsCallback(
                    print_freq=5000,  # Imprime a cada 5000 steps
                    verbose=1,
                    show_confusion_matrix=True,
                )

                # 💾 [CHECKPOINT] Save model every 50k steps to prevent losing progress
                periodic_checkpoint_callback = CheckpointCallback(
                    save_freq=50000,
                    save_path=checkpoint_dir,
                    name_prefix="checkpoint",
                    save_replay_buffer=False,  # Não salvar buffer para economizar espaço
                    save_vecnormalize=False,
                    verbose=1,
                )

                # 🧪 [HOLDOUT OOS] Validação out-of-sample em dados temporalmente
                # separados (últimos 10% do dataset, jamais vistos no treino nem na HPO).
                # Prova científica de aprendizado real — compara reward in-sample vs
                # out-of-sample e emite veredito (overfit / divergência / saudável).
                holdout_validation_callback = None
                try:
                    _holdout_df_ref = getattr(self, "holdout_df", None)
                    if _holdout_df_ref is not None and len(_holdout_df_ref) > 200:
                        from trading.callbacks.holdout_validation_callback import (
                            HoldoutValidationCallback,
                        )

                        # Captura referências locais para o closure (evita late-binding)
                        _fc_local = feature_columns
                        _ec_local = env_class
                        _sn_local = specialist_name
                        _erp_local = env_reward_params if env_reward_params else None
                        _ns_local = N_STACK_FRAMES

                        def _build_holdout_env():
                            _h_env = DummyVecEnv(
                                [
                                    lambda: self._make_trend_env(
                                        _holdout_df_ref,
                                        mode="holdout",  # gates/thresholds de avaliação, não de treino
                                        feature_columns=_fc_local,
                                        env_class=_ec_local,
                                        specialist_name=_sn_local,
                                    )
                                ]
                            )
                            _h_env = VecFrameStack(_h_env, n_stack=_ns_local)
                            _h_env = VecNormalize(
                                _h_env,
                                norm_obs=False,
                                norm_reward=False,
                                clip_reward=10.0,
                            )
                            if _erp_local:
                                try:
                                    _h_env.env_method(
                                        "set_reward_shaping_params", _erp_local
                                    )
                                except Exception:
                                    pass
                            try:
                                _h_env.env_method("set_phase3_runtime_tweaks")
                            except Exception:
                                pass
                            return _h_env

                        def _train_reward_getter():
                            """
                            Retorna ep_rew_mean do treino. Fonte primária é
                            `model.ep_info_buffer` (deque de dicts 'r','l','t'), que
                            PERSISTE após dump() — diferente de `logger.name_to_value`,
                            que é limpo a cada rollout. Fallback para o logger e depois 0.0.
                            """
                            try:
                                buf = getattr(self.model, "ep_info_buffer", None)
                                if buf is not None and len(buf) > 0:
                                    rewards = [float(ep.get("r", 0.0)) for ep in buf]
                                    if rewards:
                                        import numpy as _np

                                        return float(_np.mean(rewards))
                            except Exception:
                                pass
                            try:
                                v = self.model.logger.name_to_value.get(
                                    "rollout/ep_rew_mean", None
                                )
                                if v is not None:
                                    return float(v)
                            except Exception:
                                pass
                            return 0.0

                        _holdout_eval_freq = int(
                            os.environ.get("HOLDOUT_EVAL_FREQ", "25000")
                        )
                        holdout_validation_callback = HoldoutValidationCallback(
                            env_builder=_build_holdout_env,
                            eval_freq=_holdout_eval_freq,
                            # deterministic: 1 episódio basta (holdout curto + política determinística
                            # = N episódios idênticos, std=0, desperdício de compute).
                            n_eval_episodes=1,
                            # estocástica: 3 episódios revelam se a política CONSEGUE operar
                            # com exploração (diagnóstico de policy collapse parcial).
                            stochastic_episodes=3,
                            train_reward_source=_train_reward_getter,
                            verbose=1,
                        )
                        logger.info(
                            "🧪 [HOLDOUT] Validação OOS ativa — holdout=%d amostras, eval_freq=%d steps",
                            len(_holdout_df_ref),
                            _holdout_eval_freq,
                        )
                    else:
                        logger.warning(
                            "🧪 [HOLDOUT] holdout_df indisponível ou muito pequeno — validação OOS desabilitada"
                        )
                except Exception as _he:
                    logger.warning(
                        f"🧪 [HOLDOUT] Falha ao configurar validação OOS: {_he}"
                    )
                    holdout_validation_callback = None

                _cb_items = [
                    eval_callback,
                    trade_analysis_callback,
                    scientific_callback,
                    entropy_callback,
                    action_diversity_callback,
                    clean_metrics_callback,  # 📊 Métricas limpas
                    periodic_checkpoint_callback,  # 💾 Checkpoint a cada 50k steps
                ]
                if holdout_validation_callback is not None:
                    _cb_items.append(holdout_validation_callback)  # 🧪 Validação OOS
                callback_list = CallbackList(_cb_items)
                logger.info(
                    "🔬 [SCIENTIFIC] Callbacks científicos integrados: Sharpe/Sortino/Calmar/Kelly + Entropy Regularization + Clean Metrics"
                )
            except ImportError as e:
                logger.warning(f"⚠️ Callbacks científicos não disponíveis: {e}")
                callback_list = CallbackList([eval_callback, trade_analysis_callback])
            # 💾 [CHECKPOINT] Procurar melhor checkpoint disponível (best_model ou checkpoint_Nsteps)
            checkpoint_to_load = None
            if os.path.exists(best_model_path):
                checkpoint_to_load = best_model_path
            else:
                # Procurar checkpoint periódico mais recente
                import glob

                checkpoint_files = glob.glob(
                    os.path.join(checkpoint_dir, "checkpoint_*_steps.zip")
                )
                if checkpoint_files:
                    # Ordenar por número de steps (extrair do nome)
                    def get_steps(path):
                        try:
                            return int(os.path.basename(path).split("_")[1])
                        except:
                            return 0

                    checkpoint_files.sort(key=get_steps, reverse=True)
                    checkpoint_to_load = checkpoint_files[0]
                    logger.info(
                        f"💾 [CHECKPOINT] Encontrado checkpoint periódico: {os.path.basename(checkpoint_to_load)}"
                    )

            if checkpoint_to_load:
                try:
                    # Valida o zip antes de tentar carregar — evita crash em arquivo corrompido
                    import zipfile as _zf

                    _zip_ok = False
                    try:
                        _zip_sz = os.path.getsize(checkpoint_to_load)
                        if _zip_sz > 0:
                            with _zf.ZipFile(checkpoint_to_load) as _ztest:
                                _ztest.testzip()
                            _zip_ok = True
                    except (_zf.BadZipFile, Exception) as _ze:
                        pass

                    if not _zip_ok:
                        logger.warning(
                            "[TREND SAC] Checkpoint corrompido ou vazio ('%s'). "
                            "Removendo e criando novo modelo.",
                            os.path.basename(checkpoint_to_load),
                        )
                        try:
                            os.remove(checkpoint_to_load)
                        except OSError:
                            pass
                        self.model = None
                    else:
                        logger.info(
                            f"[TREND SAC] Carregando checkpoint de '{checkpoint_to_load}' para continuar."
                        )
                        self.model = ClippedSAC.load(
                            checkpoint_to_load,
                            env=train_env,
                            device=self._get_device(),
                            max_grad_norm=self.max_grad_norm,
                        )
                except ValueError as e:
                    if "Observation spaces do not match" in str(e):
                        logger.warning(
                            f"[TREND SAC] Checkpoint incompatível (dimensões diferentes). Criando novo modelo."
                        )
                        self.model = None
                    else:
                        logger.warning(
                            "[TREND SAC] Falha ao carregar checkpoint ('%s'): %s. Criando novo modelo.",
                            checkpoint_to_load,
                            e,
                        )
                        self.model = None
                except Exception as e:
                    logger.warning(
                        "[TREND SAC] Erro inesperado ao carregar checkpoint ('%s'): %s. Criando novo modelo.",
                        checkpoint_to_load,
                        e,
                    )
                    self.model = None
            # [TB FIX] Caminho absoluto calculado a partir do arquivo — independente de os.getcwd()
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _tb_dir = os.path.join(
                _project_root, "logs", "tensorboard", self.specialist_name
            )
            os.makedirs(_tb_dir, exist_ok=True)

            if self.model is None:
                logger.info(
                    "[TREND SAC] Criando novo modelo SAC com HybridLSTM+Attention "
                    "(VecFrameStack n_stack=%d → LSTM com sequência temporal real).",
                    N_STACK_FRAMES,
                )

                # [UPGRADE] HybridLSTMAttentionExtractor com VecFrameStack real.
                # Substitui o AttentionMLP (seq_len=1 = apenas ponderação de features)
                # pelo LSTM que agora recebe os últimos N_STACK_FRAMES candles como
                # sequência temporal genuína — captura tendência, momentum e reversão.
                from specialists.scientific_corrections import (
                    get_hybrid_lstm_policy_kwargs,
                )

                policy_kwargs = get_hybrid_lstm_policy_kwargs(
                    n_stack=N_STACK_FRAMES,
                    features_dim=128,
                    lstm_hidden_size=64,
                    num_heads=4,
                    dropout=0.15,
                )

                # 🔬 LINEAR LR DECAY SCHEDULER (evita overshoot no final do treinamento)
                # LR decai linearmente de valor_inicial até 10% no final
                initial_lr = self.hyperparams.get("learning_rate", 2.7e-5)
                final_lr_ratio = 0.1  # Termina em 10% do LR inicial

                def linear_lr_schedule(progress_remaining: float) -> float:
                    """
                    LR decay linear: começa alto, termina baixo.
                    progress_remaining: 1.0 no início, 0.0 no final.
                    """
                    return final_lr_ratio + (1.0 - final_lr_ratio) * progress_remaining

                # Multiplica o schedule pelo LR inicial
                lr_schedule = lambda p: initial_lr * linear_lr_schedule(p)

                self.model = ClippedSAC(
                    "MlpPolicy",
                    train_env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=lr_schedule,  # LR com decay linear
                    buffer_size=self.hyperparams.get("buffer_size", 50000),
                    batch_size=self.hyperparams.get("batch_size", 1024),
                    gamma=self.hyperparams.get("gamma", 0.995),
                    tau=self.hyperparams.get("tau", 0.005),
                    train_freq=self.hyperparams.get("train_freq", 16),
                    ent_coef=self.hyperparams.get("ent_coef", "auto_0.5"),
                    target_entropy=float(self.hyperparams.get("target_entropy", -3.0)),
                    use_sde=self.hyperparams.get("use_sde", True),
                    verbose=1,
                    device=self._get_device(),
                    tensorboard_log=_tb_dir,
                    max_grad_norm=float(
                        self.hyperparams.get("max_grad_norm", self.max_grad_norm or 0.5)
                    ),
                    learning_monitor=self.learning_monitor,
                )
                # torch.compile removido: requer Triton que não existe no Windows.
            else:
                # [TB FIX] Modelo carregado de checkpoint — SB3.load() não restaura tensorboard_log.
                # Reconfigurar o logger manualmente para que os gráficos apareçam no TensorBoard.
                try:
                    from stable_baselines3.common.logger import (
                        configure as _sb3_configure,
                    )

                    self.model.set_logger(
                        _sb3_configure(
                            _tb_dir, format_strings=["stdout", "tensorboard"]
                        )
                    )
                    logger.info(
                        "[TREND SAC] TensorBoard reconfigurado para checkpoint: %s",
                        _tb_dir,
                    )
                except Exception as _tb_err:
                    logger.warning(
                        "[TREND SAC] Falha ao reconfigurar TensorBoard pós-checkpoint: %s",
                        _tb_err,
                    )

            self.model.set_env(train_env)
            # O objetivo final é o número de passos atual + o que falta treinar.
            total_target_timesteps = self.model.num_timesteps + timesteps_to_train
            try:
                # tb_log_name explícito: garante que cada especialista aparece com seu
                # próprio nome no TensorBoard (bear_specialist/SAC_0, ranger_specialist/SAC_0...)
                # em vez do genérico "SAC_0" que sobrescreveria outras runs.
                self.model.learn(
                    total_timesteps=total_target_timesteps,
                    callback=callback_list,
                    reset_num_timesteps=False,
                    tb_log_name=self.specialist_name,
                )
            except KeyboardInterrupt:
                # [GRACEFUL STOP] Ctrl+C durante treino — salva o modelo no estado atual
                # para evitar re-treino completo no próximo restart.
                logger.warning(
                    "[TREND SAC] Treinamento interrompido (Ctrl+C). Salvando modelo parcial em '%s'...",
                    self.model_path,
                )
                try:
                    self.model.save(self.model_path)
                    self.is_trained = True
                    logger.info(
                        "[TREND SAC] Modelo parcial salvo (%d steps). Encerrando.",
                        self.model.num_timesteps,
                    )
                except Exception as _save_err:
                    logger.error(
                        "[TREND SAC] Falha ao salvar modelo parcial: %s", _save_err
                    )
                raise  # Re-lança para o shutdown handler do bot processar normalmente
            self.is_trained = True
            self.model.save(self.model_path)
            logger.info(
                "[TREND SAC] Treinamento do TrendSpecialist concluido. Total de timesteps: %s"
                % self.model.num_timesteps
            )

            # [MONITOR] Verifica circuit breaker apos treino — para o bot se necessario
            try:
                self.learning_monitor.raise_if_circuit_breaker()
            except Exception as _cb_exc:
                from governance.learning_monitor import LearningStopSignal

                if isinstance(_cb_exc, LearningStopSignal):
                    logger.critical(
                        "🚨 [TREND SAC] TREINO INTERROMPIDO PELO LEARNING MONITOR!\n"
                        "   Motivo: %s\n"
                        "   Acoes corretivas:\n"
                        "     1. Verifique os diagnosticos em logs/learning_monitor/\n"
                        "     2. Reinicie otimizacao com mais timesteps (TREND_FINAL_TIMESTEPS_MIN)\n"
                        "     3. Revise reward shaping em scientific_corrections.py",
                        _cb_exc.reason,
                    )
                    self.learning_monitor.save_summary()
                    raise

            # [MONITOR] Avaliação pós-treino e log do relatorio de convergencia
            try:
                final_mean_reward, _ = evaluate_policy(
                    self.model, train_env, n_eval_episodes=5, deterministic=True
                )
                # Coleta sumário do episodio para extrair metricas de qualidade
                _sums = []
                if hasattr(train_env, "env_method"):
                    try:
                        _batches = train_env.env_method("consume_episode_summaries")
                        _sums = (
                            [s for batch in _batches for s in batch] if _batches else []
                        )
                    except Exception:
                        pass
                # [D-C1 FIX] Chave correta é "trade_sharpe", não "sharpe_ratio"
                # "sharpe_ratio" nunca é setada no dicionário — D6/D7 recebiam 0.0 sempre
                _sharpe = (
                    float(np.mean([s.get("trade_sharpe", 0.0) for s in _sums]))
                    if _sums
                    else 0.0
                )
                _pnl_pt = (
                    float(np.mean([s.get("avg_pnl_per_trade", 0.0) for s in _sums]))
                    if _sums
                    else 0.0
                )
                _dur_avg = (
                    float(np.mean([s.get("avg_trade_duration", 1.0) for s in _sums]))
                    if _sums
                    else 1.0
                )
                _n_trades = (
                    int(np.mean([s.get("total_trades", 0) for s in _sums]))
                    if _sums
                    else 0
                )
                # Calcula taxa de scalping (duration=1 / total_trades)
                _dur1 = (
                    float(np.mean([s.get("duration_1_count", 0) for s in _sums]))
                    if _sums
                    else 0.0
                )
                _scalp_rate = float(_dur1 / max(_n_trades, 1)) if _n_trades > 0 else 0.0
                self.learning_monitor.record_episode(
                    mean_reward=final_mean_reward,
                    sharpe=_sharpe,
                    pnl_per_trade=_pnl_pt,
                    duration_avg=_dur_avg,
                    n_trades=_n_trades,
                    scalp_rate=_scalp_rate,
                )
                self.learning_monitor.log_report(logger)
                self.learning_monitor.save_summary()
            except Exception as _e:
                logger.debug("[MONITOR] Falha ao registrar episodio pos-treino: %s", _e)

            return self.model.num_timesteps
        except Exception as e:
            logger.error(
                f"[ERRO TREND SAC] Falha catastrofica durante o treinamento: {e}",
                exc_info=True,
            )
            self.is_trained = False
            return self.model.num_timesteps if self.model else 0

    def decide_action(
        self, observation: np.ndarray, df_row: pd.Series
    ) -> Optional[Signal]:
        if not self.is_trained or self.model is None:
            return None
        try:
            # Aplica o scaler APENAS nas market features (primeiras n_features_in_ dims).
            # O scaler foi treinado só nas features de mercado; agent_state/time/prior
            # são concatenados crus depois e NÃO devem ser escalados.
            if self.feature_scaler is not None:
                try:
                    market_dim = getattr(
                        self.feature_scaler,
                        "n_features_in_",
                        len(self.feature_columns) if self.feature_columns else 65,
                    )
                    obs_subset = observation[:market_dim].reshape(1, -1)

                    if isinstance(self.feature_scaler, ColumnTransformer):
                        feat_names = getattr(
                            self.feature_scaler, "feature_names_in_", None
                        )
                        if (
                            feat_names is not None
                            and len(feat_names) == obs_subset.shape[1]
                        ):
                            obs_df = pd.DataFrame(obs_subset, columns=feat_names)
                            scaled_values = self.feature_scaler.transform(
                                obs_df
                            ).flatten()
                        else:
                            scaled_values = self.feature_scaler.transform(
                                obs_subset
                            ).flatten()
                    else:
                        scaled_values = self.feature_scaler.transform(
                            obs_subset
                        ).flatten()

                    observation[:market_dim] = scaled_values
                except Exception as e:
                    logger.warning(f"[SCALER INFERENCE] Falha ao aplicar scaler: {e}")

            action_values, _ = self.model.predict(observation, deterministic=True)
            position_action_raw, sl_mult_raw, desired_leverage_raw = action_values

            # [DEBUG] Log raw model outputs to detect policy collapse
            logger.info(
                f"[DEBUG MODEL] Raw outputs: pos={position_action_raw:.4f}, sl={sl_mult_raw:.4f}, lev={desired_leverage_raw:.4f}"
            )

            # Gating leve por Ãâ€ž e incerteza/regime
            # BUG N5 FIX: MÁSCARA DIRECIONAL NA INFERÊNCIA (soft mask - consistente com treinamento)
            # Não zerar. Apenas reduzir a confiança para sinalizar peso menor no ensemble.
            prior_dir = float(df_row.get("tp_prior_dir", 0.0))
            mask_threshold = 0.1
            if position_action_raw > 0 and prior_dir < -mask_threshold:
                position_action_raw *= 0.3
            if position_action_raw < 0 and prior_dir > mask_threshold:
                position_action_raw *= 0.3
            tau_star = df_row.get("tp_tau", None)
            prior_conf = df_row.get("tp_prior_conf", None)
            uncertainty = df_row.get("tp_uncertainty", None)
            regime_up = float(df_row.get("tp_regime_up", 0.0))
            regime_down = float(df_row.get("tp_regime_down", 0.0))
            regime_side = float(df_row.get("tp_regime_sideways", 0.0))
            # Dimensionamento de alavancagem guiado pela confiança do TrendPredictor
            gating_active = False
            lev_min = float(self.trading_config.MIN_LEVERAGE_PER_TRADE)
            lev_max = float(self.trading_config.MAX_LEVERAGE_PER_TRADE)
            activation_threshold = 0.60
            prior_conf_f = float(prior_conf) if prior_conf is not None else 0.5
            if prior_conf_f <= activation_threshold:
                calculated_leverage = lev_min
            else:
                factor = (prior_conf_f - activation_threshold) / (
                    1.0 - activation_threshold
                )
                calculated_leverage = lev_min + (lev_max - lev_min) * (factor**2)
            gated_leverage = float(round(calculated_leverage))
            # Se há incerteza alta, reduz para o mínimo
            if uncertainty is not None:
                try:
                    unc = float(uncertainty)
                    if unc > 0.7:
                        gating_active = True
                        gated_leverage = lev_min
                except Exception:
                    pass
            final_action = Action.HOLD
            # Threshold dinÃƒÂ¢mico por regime
            action_threshold = 0.15
            if regime_up > 0.5 or regime_down > 0.5:
                action_threshold = 0.10
            elif regime_side > 0.5:
                action_threshold = 0.20
            explanation = {"specialist": "TrendSpecialist"}

            # [FIX] Initialize all variables BEFORE conditional logic to avoid UnboundLocalError
            base_conf = abs(
                float(position_action_raw)
            )  # Base confidence from model output
            prior_conf_val = (
                float(df_row.get("tp_prior_conf", base_conf))
                if df_row.get("tp_prior_conf") is not None
                else base_conf
            )
            conf_regime_bonus = (
                0.05
                if (regime_up > 0.5 or regime_down > 0.5)
                else -0.05
                if regime_side > 0.5
                else 0.0
            )
            confidence = float(
                np.clip(
                    0.5 * base_conf + 0.5 * prior_conf_val + conf_regime_bonus, 0.0, 1.0
                )
            )

            # Ajuste por incerteza
            if uncertainty is not None:
                try:
                    confidence *= float(
                        np.clip(1.0 - float(uncertainty) * 0.5, 0.5, 1.0)
                    )
                except Exception:
                    pass

            position_size_pct = float(
                self.trading_config.MAX_POSITION_SIZE_PERCENT * confidence
            )
            leverage = gated_leverage
            stop_loss = float(self.trading_config.DEFAULT_STOP_LOSS_PCT)
            take_profit = float(self.trading_config.DEFAULT_TAKE_PROFIT_PCT)

            if gating_active:
                explanation["reason"] = f"Gating por baixa confianca do prior (< tau*)"
            elif abs(position_action_raw) < action_threshold:
                explanation["reason"] = (
                    f"Sinal de tendencia fraco (valor: {position_action_raw:.2f})"
                )
            elif position_action_raw > action_threshold:
                final_action = Action.BUY
                explanation["reason"] = (
                    f"Sinal de tendencia de alta forte (valor: {position_action_raw:.2f})"
                )
            else:
                final_action = Action.SELL
                explanation["reason"] = (
                    f"Sinal de tendencia de baixa forte (valor: {position_action_raw:.2f})"
                )

            # Se gating ativo e sinal nÃƒÂ£o for forte, mantÃƒÂ©m HOLD com position_size reduzido
            if gating_active and abs(position_action_raw) < (action_threshold * 1.5):
                final_action = Action.HOLD
                position_size_pct = float(
                    self.trading_config.MAX_POSITION_SIZE_PERCENT * confidence * 0.5
                )  # Reduce size

            # [FIX PRODUÇÃO] Filtro de direção — replica _specialist_long_only do ambiente de treino.
            # Bull: apenas BUY (long_only). Bear: apenas SELL (short_only).
            # Se o modelo gerar sinal contrário à especialidade, converte em HOLD.
            specialist_direction = getattr(
                self, "_specialist_direction", None
            )  # 'long_only' | 'short_only' | None
            if specialist_direction == "long_only" and final_action == Action.SELL:
                logger.info(
                    f"🐂 [REGIME FILTER] BullSpecialist: SELL bloqueado → HOLD (long_only em produção)"
                )
                final_action = Action.HOLD
                explanation["reason"] = (
                    "BullSpecialist: sinal SELL bloqueado (long_only)"
                )
                confidence = 0.0
            elif specialist_direction == "short_only" and final_action == Action.BUY:
                logger.info(
                    f"🐻 [REGIME FILTER] BearSpecialist: BUY bloqueado → HOLD (short_only em produção)"
                )
                final_action = Action.HOLD
                explanation["reason"] = (
                    "BearSpecialist: sinal BUY bloqueado (short_only)"
                )
                confidence = 0.0

            return Signal(
                symbol=self.trading_config.PRIMARY_PAIR,
                action=final_action,
                confidence=confidence,
                position_size_pct=position_size_pct,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                explanation=explanation,
            )
        except Exception as e:
            logger.exception(f"[ERRO TREND SAC] Erro ao decidir acao: {e}.")
            return None

    def refine_decision(
        self, master_signal: Signal, observation: np.ndarray, df_row: pd.Series
    ) -> Signal:
        """

        [VERSÃƒÆ'O CORRIGIDA]
        Refina a decisÃƒÂ£o, garantindo que a aÃƒÂ§ÃƒÂ£o esteja alinhada com a tendÃƒÂªncia principal.
        """

        # [CORREÃƒâ€¡ÃƒÆ'O CRÃƒÂTICA] Passa a 'observation' e 'df_row' para o `decide_action` para que
        # a decisÃƒÂ£o do especialista seja baseada no estado atual do mercado.
        specialist_signal = self.decide_action(observation, df_row)
        fallback_reason = "Especialista de TendÃƒÂªncia indicou HOLD."
        if specialist_signal is None or specialist_signal.action == Action.HOLD:
            reason = fallback_reason
            if specialist_signal and isinstance(specialist_signal.explanation, dict):
                reason = specialist_signal.explanation.get("reason", fallback_reason)
            return Signal(
                symbol=master_signal.symbol,
                action=Action.HOLD,
                confidence=0.0,
                explanation={
                    "master_delegated_to": master_signal.explanation.get(
                        "master_delegated_to"
                    ),
                    "specialist_consulted": self.__class__.__name__,
                    "reason": reason,
                },
            )
        # [CORREÃƒâ€¡ÃƒÆ'O] Usa o nome correto da coluna (sem sufixo) para a verificaÃƒÂ§ÃƒÂ£o de sanidade.
        is_aligned = True
        ema_trend_val = self._extract_ema_trend_value(df_row)
        if (specialist_signal.action == Action.BUY and ema_trend_val < 0) or (
            specialist_signal.action == Action.SELL and ema_trend_val > 0
        ):
            is_aligned = False
        if not is_aligned:
            return Signal(
                symbol=master_signal.symbol,
                action=Action.HOLD,
                confidence=0.0,
                explanation={
                    "master_delegated_to": master_signal.explanation.get(
                        "master_delegated_to"
                    ),
                    "specialist_consulted": self.__class__.__name__,
                    "reason": f"VETO: Sinal de {specialist_signal.action.value} em conflito com a tendÃƒÂªncia principal (EMA Trend: {ema_trend_val}).",
                },
            )
        # --- VERIFICAÇÃO DE SANIDADE FINAL (TrendPredictor) ---
        prior_dir = float(df_row.get("tp_prior_dir", 0.0))
        if specialist_signal.action == Action.BUY and prior_dir < -0.1:
            return Signal(
                symbol=master_signal.symbol,
                action=Action.HOLD,
                confidence=0.0,
                explanation={
                    "reason": f"VETO FINAL: Tentativa de Compra contra tendência de Baixa do Predictor ({prior_dir})."
                },
            )
        if specialist_signal.action == Action.SELL and prior_dir > 0.1:
            return Signal(
                symbol=master_signal.symbol,
                action=Action.HOLD,
                confidence=0.0,
                explanation={
                    "reason": f"VETO FINAL: Tentativa de Venda contra tendência de Alta do Predictor ({prior_dir})."
                },
            )
        # FusÃƒÂ£o de Sinais (lÃƒÂ³gica mantida)
        combined_confidence = (master_signal.confidence * 0.3) + (
            specialist_signal.confidence * 0.7
        )
        final_pos_size_pct = (
            self.trading_config.MAX_POSITION_SIZE_PERCENT * combined_confidence
        )
        final_explanation = {
            **master_signal.explanation,
            **specialist_signal.explanation,
            "sanity_check": "Alinhado com a tendÃƒÂªncia principal.",
        }
        return replace(
            specialist_signal,
            symbol=master_signal.symbol,
            confidence=combined_confidence,
            position_size_pct=final_pos_size_pct,
            explanation=final_explanation,
        )

    def save_model(self):
        if self.model:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            logger.info(
                "[TREND SAC] TrendSpecialist (SAC) salvo em '%s'.", self.model_path
            )

    def load_model(self):
        self._initialize_model()
        # [SCIENTIFIC FIX] Carrega o contrato de features para garantir consistência com o scaler
        try:
            models_dir = os.path.join(os.getcwd(), "models_ai")
            metadata_path = os.path.join(models_dir, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    raw_cols = meta.get("base_feature_columns", [])
                    # [SCIENTIFIC FIX] Filtra para as features core (geralmente 55) para bater com o scaler
                    self.feature_columns = self._filter_to_core_features(raw_cols)
                logger.info(
                    f"✅ [META] Contrato de features filtrado: {len(self.feature_columns)} colunas."
                )
        except Exception as e:
            logger.warning(f"⚠️ [META] Falha ao carregar metadados: {e}")
        # [SCIENTIFIC FIX] Tenta carregar o scaler correspondente
        try:
            import joblib

            models_dir = os.path.join(os.getcwd(), "models_ai")
            scaler_filename = f"{self.specialist_name}_scaler.joblib"
            scaler_path = os.path.join(models_dir, scaler_filename)

            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
                logger.debug(
                    f"✅ [SCALER] Scaler carregado para {self.specialist_name}: {scaler_filename}"
                )
            else:
                # Fallback para o nome genérico
                generic_path = os.path.join(models_dir, "feature_scaler_robust.joblib")
                if os.path.exists(generic_path):
                    self.feature_scaler = joblib.load(generic_path)
                    logger.debug(
                        f"⚠️ [SCALER] Scaler específico não encontrado. Usando fallback genérico."
                    )
            # Usa feature_names_in_ do scaler como contrato autoritativo de features
            # (sobrescreve o resultado de _filter_to_core_features que pode divergir do treino)
            if self.feature_scaler is not None and hasattr(
                self.feature_scaler, "feature_names_in_"
            ):
                self.feature_columns = list(self.feature_scaler.feature_names_in_)
                logger.info(
                    f"✅ [SCALER] feature_columns alinhado com scaler: {len(self.feature_columns)} colunas."
                )
            else:
                logger.warning(
                    f"⚠️ [SCALER] Nenhum scaler carregado para {self.specialist_name}. "
                    "Observações serão usadas CRUAS (passthrough) — features com escalas diferentes podem prejudicar a inferência SAC."
                )
        except Exception as e:
            logger.error(f"❌ [SCALER] Falha ao carregar scaler: {e}")

    def _load_reward_params(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.reward_params_path):
                with open(self.reward_params_path, "r") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    return {
                        key: data[key] for key in self.REWARD_PARAM_KEYS if key in data
                    }
        except Exception as exc:
            logger.warning(
                "[TREND SAC] Falha ao carregar parametros de recompensa: %s", exc
            )
        return {}

    def _save_reward_params(self, params: Dict[str, Any]) -> None:
        try:
            with open(self.reward_params_path, "w") as handle:
                json.dump(params, handle, indent=4)
            logger.info(
                "[TREND SAC] Parametros de recompensa salvos em %s",
                self.reward_params_path,
            )
        except Exception as exc:
            logger.error(
                "[TREND SAC] Falha ao salvar parametros de recompensa: %s",
                exc,
                exc_info=True,
            )

    def _apply_reward_params_to_config(self, params: Dict[str, Any]) -> None:
        if not params:
            return
        for config_key in self.REWARD_PARAM_KEYS:
            if config_key in params:
                try:
                    setattr(self.config, config_key, params[config_key])
                    # [FIX 6] Sincroniza também em self.hyperparams para persistência HPO→Training
                    if isinstance(getattr(self, "hyperparams", None), dict):
                        self.hyperparams[config_key] = params[config_key]
                except Exception as exc:
                    logger.debug(
                        "[TREND SAC] Falha ao aplicar %s no config: %s", config_key, exc
                    )

    def _convert_reward_params_to_env(self, params: Dict[str, Any]) -> Dict[str, Any]:
        env_params: Dict[str, Any] = {}
        for config_key, attr_name in self.REWARD_ATTR_MAP.items():
            if config_key in params:
                env_params[attr_name] = params[config_key]
        # [FIX] Pass-through RANGER_* keys directamente ao env.
        # RangerTradingEnv.set_reward_shaping_params() os processa internamente.
        # Sem este bloco, _convert filtra apenas REWARD_ATTR_MAP e descarta RANGER_*,
        # fazendo o HPO sugerir params que nunca chegam ao ambiente.
        for key, val in params.items():
            if key.startswith("RANGER_"):
                env_params[key] = val
        return env_params

    def _suggest_reward_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        reward_params: Dict[str, Any] = {}
        # [v3.1] TREND_UNREALIZED_REWARD_SCALE e TREND_REALIZED_REWARD_SCALE REMOVIDOS.
        # Eram parâmetros MORTOS: setados em self.unrealized_reward_scale / realized_reward_scale
        # mas nunca lidos pelo reward. O reward usa realized_scale=300.0 (hardcoded, calibrado)
        # e depth_reward=0.08 (hardcoded em scientific_corrections.py). Optuna desperdiçava
        # 2 dimensões de busca sem nenhum efeito. Removido para reduzir espaço: 28→26 params.
        reward_params["TREND_DRAWDOWN_PENALTY_WEIGHT"] = trial.suggest_float(
            "TREND_DRAWDOWN_PENALTY_WEIGHT", 0.15, 1.2
        )
        reward_params["TREND_PRIOR_FLIP_PENALTY_WEIGHT"] = trial.suggest_float(
            "TREND_PRIOR_FLIP_PENALTY_WEIGHT", 0.2, 1.5
        )
        reward_params["TREND_PRIOR_FLIP_PENALTY_EXP"] = trial.suggest_float(
            "TREND_PRIOR_FLIP_PENALTY_EXP", 0.7, 1.7
        )
        reward_params["TREND_PRIOR_FLIP_THRESHOLD"] = trial.suggest_float(
            "TREND_PRIOR_FLIP_THRESHOLD", 0.2, 0.45
        )
        reward_params["TREND_DURATION_PRESSURE_WEIGHT"] = trial.suggest_float(
            "TREND_DURATION_PRESSURE_WEIGHT", 0.35, 0.75
        )
        reward_params["TREND_DURATION_PRESSURE_START"] = trial.suggest_int(
            "TREND_DURATION_PRESSURE_START", 60, 180
        )
        reward_params["TREND_DURATION_PRESSURE_EXP"] = trial.suggest_float(
            "TREND_DURATION_PRESSURE_EXP", 0.8, 1.6
        )
        return reward_params

    def _post_process_env(self, env: TrendFollowingEnv) -> TrendFollowingEnv:
        try:
            if self.reward_hyperparams:
                env_params = self._convert_reward_params_to_env(self.reward_hyperparams)
                if env_params:
                    env.set_reward_shaping_params(env_params)
        except Exception as exc:
            logger.debug(
                "[TREND SAC] Falha ao ajustar parametros de recompensa do ambiente: %s",
                exc,
            )
        return env
