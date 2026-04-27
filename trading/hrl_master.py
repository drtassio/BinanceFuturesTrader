# -----------------------------------------------------------------------------
# ARQUIVO: trading/hrl_master.py (Versão Corrigida - Usando PPO Padrão)
# -----------------------------------------------------------------------------

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import torch
import json
import optuna
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from trading.callbacks.optuna_eval_callback import TrialEvalCallback
from trading.callbacks.financial_eval_callback import FinancialEvalCallback
from trading.callbacks.early_stopping_callback import EarlyStoppingCallback
from optuna.pruners import MedianPruner

# <<< CORREÇÃO: Importar PPO da biblioteca principal em vez de RecurrentPPO da contrib >>>
from stable_baselines3 import PPO

from utils.logger import get_logger, LOG_LEVEL_DEBUG
from config.settings import AIConfig, TradingConfig
from models.trade_schema import Signal, Action
# from learning.regime_switch import RegimeSwitch # Removido pois o arquivo não existe mais

# A importação das classes dos especialistas é necessária para a tipagem do ambiente
from specialists.trend_specialist import TrendSpecialist
from specialists.bull_specialist import BullSpecialist
from specialists.bear_specialist import BearSpecialist
from specialists.ranger_specialist import RangerSpecialist
from specialists.scientific_corrections import apply_temperature_scaling

logger = get_logger("HRLMaster", LOG_LEVEL_DEBUG)

# Definição local de regimes já que RegimeSwitch foi removido
REGIMES = ['bull', 'bear', 'ranger']

# [FIX] Usa path absoluto baseado em __file__ — independente de os.getcwd()
# os.getcwd() é instável no Windows/OneDrive e pode apontar para pasta errada.
_HRL_MASTER_DIR  = os.path.dirname(os.path.abspath(__file__))   # trading/
_HRL_PROJECT_ROOT = os.path.dirname(_HRL_MASTER_DIR)             # raiz do projeto
TENSORBOARD_LOG_DIR_HRL = os.path.join(_HRL_PROJECT_ROOT, "logs", "tensorboard", "HRLMaster")
os.makedirs(TENSORBOARD_LOG_DIR_HRL, exist_ok=True)

class MasterEnv(gym.Env):
    """
    🔬 VERSÃO CIENTÍFICA - Baseado em:
    - Barto & Mahadevan (2003): Hierarchical RL
    - Hamilton (1989): Regime Switching Models
    - Shazeer et al. (2017): Gating Networks
    
    Ambiente para treinar o HRLMaster com:
    1. Regime features (6 dimensões) para melhor seleção de especialistas
    2. Sistema de recompensa científica baseado em coerência de regime
    3. Confidence calibration para decisões mais confiáveis
    """
    metadata = {'render_modes': ['human']}
    
    # 🔬 Pesos do sistema de recompensa (Sutton, Precup & Singh, 1999)
    REWARD_WEIGHT_COHERENCE = 0.40   # Coerência com regime detectado
    REWARD_WEIGHT_PERFORMANCE = 0.50  # Performance do especialista
    REWARD_WEIGHT_CALIBRATION = 0.10  # Calibração de confiança

    def __init__(self, df: pd.DataFrame, specialists: Dict[str, Any], config: AIConfig, mode: str = 'training'):
        super(MasterEnv, self).__init__()
        if df.empty or not isinstance(df.index, pd.DatetimeIndex) or not specialists:
            raise ValueError("🚨 [ERRO ENV] Argumentos inválidos para MasterEnv (df, index ou specialists).")

        self.df = df.copy()
        self.specialists = specialists
        self.config = config
        self.max_steps = len(self.df) - 1
        self.mode = mode
        
        # Configuração de timeframe
        self.trading_config = TradingConfig()
        self.primary_tf = self.trading_config.PRIMARY_TIMEFRAME_TRADING
        self.atr_col = f'atr_{self.primary_tf}'

        # Regimes disponíveis
        self.regimes = REGIMES
        self.regime_to_idx = {r: i for i, r in enumerate(REGIMES)}
        self.action_space = spaces.Discrete(len(self.regimes))
        
        # 🔬 Features numéricas do mercado
        self.numeric_df = self.df.select_dtypes(include=np.number)
        self.market_feature_dim = self.numeric_df.shape[1]
        
        # 🔬 Regime features (6 dimensões) - Guidolin & Timmermann (2007)
        self.regime_feature_dim = 6
        
        # 🔬 Observation space expandido: market + regime features
        total_dim = self.market_feature_dim + self.regime_feature_dim
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_dim,),
            dtype=np.float32
        )
        
        # 🔬 Tracking de regime persistence (Hamilton, 1989)
        self._last_regime = None
        self._regime_persistence = 0
        
        logger.info(f"🔬 [MasterEnv] Inicializado com {self.market_feature_dim} market features + {self.regime_feature_dim} regime features = {total_dim} total")
        self.reset()

    def _extract_regime_features(self) -> np.ndarray:
        """
        🔬 Extrai features de regime científicas baseadas em:
        - Hamilton (1989): Regime persistence
        - Ang & Bekaert (2002): Probability smoothing
        - Guidolin & Timmermann (2007): Multi-regime allocation
        
        Returns:
            np.ndarray com 6 features de regime
        """
        if self.current_step >= len(self.df):
            return np.zeros(self.regime_feature_dim, dtype=np.float32)
        
        row = self.df.iloc[self.current_step]
        
        # Feature 1: Regime confidence (do CryptoRegimeDetector)
        regime_confidence = float(row.get('regime_confidence', 0.5))
        
        # Features 2-4: Probabilidades de cada regime (softmax do detector)
        regime_bull_prob = float(row.get('prob_bull', 0.33))
        regime_bear_prob = float(row.get('prob_bear', 0.33))
        regime_ranger_prob = float(row.get('prob_ranger', 0.33))
        
        # Se probabilidades não existem, inferir do regime detectado
        if 'regime' in row and regime_bull_prob == 0.33:
            detected_regime = int(row.get('regime', 2))
            if detected_regime == 0:  # Bull
                regime_bull_prob, regime_bear_prob, regime_ranger_prob = 0.7, 0.15, 0.15
            elif detected_regime == 1:  # Bear
                regime_bull_prob, regime_bear_prob, regime_ranger_prob = 0.15, 0.7, 0.15
            else:  # Ranger
                regime_bull_prob, regime_bear_prob, regime_ranger_prob = 0.15, 0.15, 0.7
        
        # Feature 5: Regime persistence normalizado (quantity bars no mesmo regime)
        regime_persistence_norm = min(self._regime_persistence / 100.0, 1.0)
        
        # Feature 6: Trend strength (ADX normalizado)
        adx_raw = float(row.get('adx_15m', row.get('adx', 25.0)))
        trend_strength = np.clip(adx_raw / 50.0, 0.0, 1.0)  # ADX > 50 = trend muito forte
        
        return np.array([
            regime_confidence,
            regime_bull_prob,
            regime_bear_prob,
            regime_ranger_prob,
            regime_persistence_norm,
            trend_strength
        ], dtype=np.float32)

    def _get_actual_regime(self) -> int:
        """
        🔬 Obtém o regime real detectado pelo CryptoRegimeDetector.
        Usado como ground truth para calcular coerência.
        
        Returns:
            int: 0=Bull, 1=Bear, 2=Ranger
        """
        if self.current_step >= len(self.df):
            return 2  # Default: Ranger (neutro)
        
        row = self.df.iloc[self.current_step]
        return int(row.get('regime', 2))

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self._last_regime = None
        self._regime_persistence = 0
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """
        🔬 Observação expandida com regime features.
        Combina features de mercado com features de regime para melhor seleção.
        """
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Features de mercado
        market_features = self.numeric_df.iloc[self.current_step].values.astype(np.float32)
        
        # Features de regime (6 dim)
        regime_features = self._extract_regime_features()
        
        # Concatena
        observation = np.concatenate([market_features, regime_features])
        return observation

    def _compute_scientific_reward(
        self, 
        chosen_regime_idx: int, 
        actual_regime_idx: int,
        specialist_pnl: float,
        atr: float
    ) -> float:
        """
        🔬 Sistema de Recompensa Científica baseado em:
        - Barto & Mahadevan (2003): Hierarchical reward decomposition
        - Gneiting & Raftery (2007): Proper scoring rules para calibração
        
        Componentes:
        1. Regime Coherence (40%): Escolheu o especialista certo para o regime?
        2. Specialist Performance (50%): O especialista escolhido lucrou?
        3. Confidence Calibration (10%): Bonus por coerência de certeza
        
        Returns:
            float: Recompensa total normalizada
        """
        # 1. 🔬 Regime Coherence Reward (Shazeer et al., 2017)
        # Match perfeito = +1.0, Errado = -0.3 (penalidade suave)
        regime_match = 1.0 if chosen_regime_idx == actual_regime_idx else -0.3
        coherence_reward = regime_match * self.REWARD_WEIGHT_COHERENCE
        
        # 2. 🔬 Specialist Performance Reward (Sutton et al., 1999)
        # ATR-normalized PnL para comparabilidade entre volatilidades
        normalized_pnl = np.tanh(specialist_pnl / (atr + 1e-9))  # Range [-1, 1]
        performance_reward = normalized_pnl * self.REWARD_WEIGHT_PERFORMANCE
        
        # 3. 🔬 Confidence Calibration Reward (Brier Score invertido)
        # Bonus quando regime confidence alta + acerto, ou baixa + erro
        row = self.df.iloc[self.current_step] if self.current_step < len(self.df) else {}
        regime_confidence = float(row.get('regime_confidence', 0.5)) if isinstance(row, pd.Series) else 0.5
        
        if regime_match > 0:
            # Acertou - bonus por alta confiança
            calibration_reward = regime_confidence * self.REWARD_WEIGHT_CALIBRATION
        else:
            # Errou - bonus por baixa confiança (estava incerto, ok errar)
            calibration_reward = (1 - regime_confidence) * self.REWARD_WEIGHT_CALIBRATION * 0.5
        
        total_reward = coherence_reward + performance_reward + calibration_reward
        return float(total_reward)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        🔬 Step com sistema de recompensa científica.
        """
        chosen_regime = self.regimes[int(action)]
        chosen_regime_idx = int(action)
        
        # Mapeamento para os nomes das classes dos especialistas
        regime_to_specialist = {
            'bull': 'BullSpecialist',
            'bear': 'BearSpecialist',
            'ranger': 'RangerSpecialist'
        }
        spec_name = regime_to_specialist.get(chosen_regime)
        chosen_specialist = self.specialists.get(spec_name)
        
        # 🔬 Obtém regime real para cálculo de coerência
        actual_regime_idx = self._get_actual_regime()
        actual_regime = REGIMES[actual_regime_idx]
        
        # Atualiza regime persistence (Hamilton, 1989)
        if actual_regime == self._last_regime:
            self._regime_persistence += 1
        else:
            self._regime_persistence = 1
            self._last_regime = actual_regime
        
        reward = 0.0
        specialist_pnl = 0.0
        lookahead_period = 5

        if chosen_specialist and chosen_specialist.is_trained and self.current_step < self.max_steps - lookahead_period:
            current_row = self.df.iloc[self.current_step]
            
            # Prepara observação para o especialista
            market_features = self.numeric_df.iloc[self.current_step].values.astype(np.float32)
            timestamp = self.numeric_df.index[self.current_step]
            
            agent_state = np.array([0.0, 0.0, 0.0], dtype=np.float32) 
            
            hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
            hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
            day_sin = np.sin(2 * np.pi * timestamp.dayofweek / 7)
            day_cos = np.cos(2 * np.pi * timestamp.dayofweek / 7)
            time_features = np.array([hour_sin, hour_cos, day_sin, day_cos], dtype=np.float32)
            
            observation_for_specialist = np.concatenate([market_features, agent_state, time_features])

            specialist_signal = chosen_specialist.decide_action(observation_for_specialist, current_row)
            
            if specialist_signal and specialist_signal.action != Action.HOLD:
                entry_price = current_row['close']
                future_price = self.df['close'].iloc[self.current_step + lookahead_period]
                
                atr = current_row.get(self.atr_col, entry_price * 0.01)
                
                if specialist_signal.action == Action.BUY:
                    specialist_pnl = future_price - entry_price
                elif specialist_signal.action == Action.SELL:
                    specialist_pnl = entry_price - future_price
                
                # 🔬 Calcula recompensa científica
                reward = self._compute_scientific_reward(
                    chosen_regime_idx=chosen_regime_idx,
                    actual_regime_idx=actual_regime_idx,
                    specialist_pnl=specialist_pnl,
                    atr=atr
                )
            else:
                # HOLD - penalidade mínima baseada em coerência
                reward = self._compute_scientific_reward(
                    chosen_regime_idx=chosen_regime_idx,
                    actual_regime_idx=actual_regime_idx,
                    specialist_pnl=0.0,
                    atr=1.0
                ) * 0.1  # 10% da recompensa normal
        else:
            # Especialista não treinado - penalidade por coerência apenas
            reward = self._compute_scientific_reward(
                chosen_regime_idx=chosen_regime_idx,
                actual_regime_idx=actual_regime_idx,
                specialist_pnl=0.0,
                atr=1.0
            ) * 0.5  # 50% da recompensa (sem performance)

        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            'chosen_regime': chosen_regime,
            'actual_regime': actual_regime,
            'regime_match': chosen_regime_idx == actual_regime_idx,
            'specialist_pnl': specialist_pnl
        }
        
        return self._get_observation(), np.float32(reward), done, False, info


class HRLMaster:
    """
    [VERSÃO CORRIGIDA]
    Gerencia o agente Mestre, usando PPO padrão para maior estabilidade e
    compatibilidade, delegando tarefas aos especialistas.
    """
    def __init__(self, config: AIConfig, input_dim: Optional[int], specialists: Dict[str, Any]):
        # [FIX] Allow input_dim to be None during fast startup
        if not isinstance(config, AIConfig):
            raise TypeError("🚨 [ERRO HRL] Configuração inválida.")
        
        # [FIX] Allow specialists to be None during fast startup
        self.config = config
        self.input_dim = input_dim if input_dim is not None else 0
        self.specialists = specialists if specialists else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # <<< CORREÇÃO: Nomes de arquivos atualizados para PPO >>>
        self.model_path = os.path.join(self.config.MODEL_DIR, "hrl_master_ppo.zip")
        self.log_dir = os.path.join(os.getcwd(), "logs")
        self.hyperparams_path = os.path.join(self.log_dir, "hrl_master_ppo_hyperparams.json")
        self.optuna_db_path = os.path.join(self.log_dir, "hrl_master_ppo_optuna.db")
        
        self.hyperparams: Dict[str, Any] = {}
        # <<< CORREÇÃO: Tipo do agente e remoção do estado LSTM >>>
        self.ppo_agent: Optional[PPO] = None
        self.is_trained = False

        self._load_hyperparams()
        self._initialize_model()
        logger.info(f"🧠 [HRL ELITE] HRLMaster (PPO) inicializado com input_dim={input_dim}.")

    @property
    def is_optimized(self) -> bool:
        """
        Verifica se os hiperparâmetros foram otimizados via Optuna.
        Segue o padrão do autoencoder: otimizar primeiro, depois treinar.
        
        Returns:
            True se hiperparâmetros foram carregados de arquivo (já otimizados)
        """
        return bool(self.hyperparams) and os.path.exists(self.hyperparams_path)

    def _load_hyperparams(self):
        if os.path.exists(self.hyperparams_path):
            with open(self.hyperparams_path, 'r') as f: self.hyperparams = json.load(f)
            logger.info("✅ [HRL PPO] Hiperparâmetros carregados.")
        else:
            logger.info("ℹ️ [HRL PPO] Arquivo de hiperparâmetros não encontrado.")

    def _save_hyperparams(self):
        with open(self.hyperparams_path, 'w') as f: json.dump(self.hyperparams, f, indent=4)
        logger.info("✅ [HRL PPO] Hiperparâmetros salvos.")

    def optimize_hyperparameters(self, train_df: pd.DataFrame, eval_df: pd.DataFrame, n_trials: int = 100, n_timesteps: int = 50000, n_eval_episodes: int = 5):
        """
        [VERSÃO FINAL E OTIMIZADA PARA PARALELISMO]
        Otimiza usando n_jobs > 1 e ambientes únicos por trial.
        """
        logger.info(f"🧠 [HRL MASTER OPTUNA] Iniciando otimização com Pruning e Paralelismo para PPO.")
        
        # --- CORREÇÃO CRÍTICA: Não vetorize o ambiente aqui ---
        # Cada trial paralelo cuidará de seu próprio ambiente único.
        train_env_fn = lambda: MasterEnv(train_df, self.specialists, self.config, mode='optimization')
        eval_env_fn = lambda: MasterEnv(eval_df, self.specialists, self.config, mode='optimization')

        def objective(trial: optuna.Trial) -> float:
            ppo_params = {
                "learning_rate": trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
                "batch_size": trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
                "n_steps": trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
                "gamma": trial.suggest_uniform('gamma', 0.95, 0.999),
                "gae_lambda": trial.suggest_uniform('gae_lambda', 0.9, 0.99),
                "clip_range": trial.suggest_uniform('clip_range', 0.1, 0.3),
                "ent_coef": trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
                "vf_coef": trial.suggest_uniform('vf_coef', 0.25, 1.0),
                "max_grad_norm": trial.suggest_uniform('max_grad_norm', 0.3, 2.0),
            }
            try:
                # --- CORREÇÃO: Cria um ambiente novo para cada trial ---
                train_env = DummyVecEnv([train_env_fn])
                train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
                eval_env = DummyVecEnv([eval_env_fn])
                eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, clip_reward=10.0)
                
                model = PPO("MlpPolicy", train_env, **ppo_params, verbose=0, device=self.device)
                eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=n_eval_episodes, eval_freq=max(n_timesteps // 10, 1000), deterministic=True, log_name="HRLMaster")
                model.learn(total_timesteps=n_timesteps, callback=eval_callback)
                mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
                return mean_reward
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"❌ [HRL MASTER OPTUNA] Trial falhou: {e}", exc_info=True)
                return -float('inf')

        storage_url = f"sqlite:///{self.optuna_db_path}"
        pruner = MedianPruner()
        
        study = optuna.create_study(direction="maximize", storage=storage_url,
                                    study_name="hrl_master_ppo_optimization_v2",
                                    pruner=pruner, load_if_exists=True)
        
        # 🔧 FIX: Calcular trials restantes para retomar de onde parou
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = max(0, n_trials - completed_trials)
        
        if remaining_trials > 0:
            logger.info(f"📊 [HRL] Optuna: {completed_trials} trials completos. Executando mais {remaining_trials}...")
            # BUG FIX: Optuna com n_jobs=20 no SQLite causa error 'database is locked'. Reduzido para 2.
            study.optimize(objective, n_trials=remaining_trials, n_jobs=2, timeout=7200)
        else:
            logger.info(f"✅ [HRL] Optuna: {completed_trials} trials completos. Nenhum trial adicional necessário.")

        if study.best_trial:
            self.hyperparams = study.best_params
            self._save_hyperparams()
        else:
            logger.warning("Otimização concluída sem nenhum trial bem-sucedido.")

    def _initialize_model(self):
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.ppo_agent = PPO.load(self.model_path, device=self.device)
                self.is_trained = True
                logger.info(f"✅ [HRL] HRLMaster (PPO) carregado de '{self.model_path}'.")
            except Exception as e:
                logger.error(f"❌ [ERRO HRL] Falha ao carregar HRLMaster PPO: {e}.")
                self.is_trained = False
    
    # FIX: Alias to satisfy AIController interface which calls .load()
    load = load_model
    
    def save_model(self):
        if self.ppo_agent and self.is_trained:
            self.ppo_agent.save(self.model_path)
            logger.info(f"✅ [HRL] HRLMaster (PPO) salvo em '{self.model_path}'.")

    def decide_action(self, observation: np.ndarray, row: pd.Series) -> Dict[str, Any]:
        """
        Interface compatível com AIController.
        Retorna um dicionário com 'specialist_name' e 'confidence'.
        """
        signal = self.get_high_level_decision(observation)
        regime = signal.explanation.get('master_delegated_to')
        
        regime_to_specialist = {
            'bull': 'BullSpecialist',
            'bear': 'BearSpecialist',
            'ranger': 'RangerSpecialist'
        }
        spec_name = regime_to_specialist.get(regime)
        
        return {
            'specialist_name': spec_name,
            'confidence': signal.confidence,
            'regime': regime
        }

    def get_high_level_decision(self, observation: np.ndarray) -> Signal:
        """
        🔬 VERSÃO CIENTÍFICA - Baseado em:
        - Guo et al. (2017): On Calibration of Modern Neural Networks
        - Schulman et al. (2017): PPO action probabilities
        
        Recebe uma observação e retorna decisão com confiança calibrada
        baseada nas probabilidades de ação do modelo PPO.
        """
        if not self.is_trained or self.ppo_agent is None:
            return Signal(action=Action.HOLD, confidence=0.0, symbol="N/A", explanation={"reason": "HRLMaster not trained."})
        
        try:
            expected_shape = self.ppo_agent.observation_space.shape
            if observation.shape != expected_shape:
                logger.error(f"❌ [ERRO HRL] Shape Mismatch CRÍTICO! Esperado: {expected_shape}, Recebido: {observation.shape}.")
                raise ValueError(f"Shape Mismatch: Expected {expected_shape}, Got {observation.shape}")

            # 🔬 Predição determinística (argmax)
            action_idx, _ = self.ppo_agent.predict(observation, deterministic=True)
            action_idx = int(action_idx)
            
            # 🔬 Calcular confiança real a partir das probabilidades de ação
            confidence = self._compute_action_confidence(observation, action_idx)
            
            chosen_regime = REGIMES[action_idx]
            
            return Signal(
                symbol="DELEGATE",
                action=Action.DELEGATE, 
                confidence=float(confidence),
                explanation={
                    "master_delegated_to": chosen_regime,
                    "action_probabilities": self._get_action_probabilities(observation)
                }
            )
        except Exception as e:
            logger.exception(f"❌ [ERRO HRL] Erro em get_high_level_decision: {e}.")
            return Signal(action=Action.HOLD, confidence=0.0, symbol="N/A", explanation={"reason": f"Error in HRLMaster: {e}"})

    def _compute_action_confidence(self, observation: np.ndarray, chosen_action: int) -> float:
        """
        🔬 Calcula a confiança da ação escolhida baseado nas probabilidades do PPO.
        
        Baseado em:
        - Murphy (1973): Hedging bets
        - Gneiting & Raftery (2007): Probabilistic forecasting
        
        Returns:
            float: Confiança entre 0 e 1
        """
        try:
            probs = self._get_action_probabilities(observation)
            if probs is None:
                return 0.5  # Default incerto
            
            # A confiança é a probabilidade da ação escolhida
            confidence = probs[chosen_action]
            
            # Ajuste de calibração: penaliza incerteza baixa
            # Se todas as probabilidades são muito próximas (entropia alta), reduz confiança
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(REGIMES))  # Entropia máxima (distribuição uniforme)
            
            # Fator de calibração: 1.0 se determinístico, menor se incerto
            calibration_factor = 1.0 - (entropy / max_entropy) * 0.3
            
            return float(np.clip(confidence * calibration_factor, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"⚠️ [HRL] Erro ao calcular confiança: {e}")
            return 0.5
    
    def _get_action_probabilities(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """
        🔬 Extrai as probabilidades de ação do modelo PPO.
        
        Para PPO, as probabilidades são obtidas passando a observação
        pela rede de política e aplicando softmax às logits.
        """
        try:
            if self.ppo_agent is None:
                return None
            
            # Prepara observação no formato correto
            obs_tensor = torch.as_tensor(observation.reshape(1, -1)).float().to(self.device)
            
            # Obtém distribuição de ação do PPO
            with torch.no_grad():
                distribution = self.ppo_agent.policy.get_distribution(obs_tensor)
                
                # 🔬 [SCIENTIFIC CALIBRATION] 
                # T > 1.0 reduz overconfidence (evita delegar errado com 99% de certeza falsa)
                temperature = 1.5 
                
                if hasattr(distribution.distribution, 'probs'):
                    raw_probs = distribution.distribution.probs.cpu().numpy().flatten()
                    probs = apply_temperature_scaling(raw_probs, temperature=temperature)
                elif hasattr(distribution.distribution, 'logits'):
                    logits = distribution.distribution.logits.cpu().numpy().flatten()
                    probs = apply_temperature_scaling(logits, temperature=temperature)
                else:
                    probs = np.ones(len(REGIMES)) / len(REGIMES)
                
            return probs
        except Exception as e:
            logger.warning(f"⚠️ [HRL] Erro ao obter probabilidades de ação: {e}")
            return None

    def train(self, featured_df: pd.DataFrame, specialists: Dict, timesteps_to_train: int) -> int:
        """
        [VERSÃO DE NÍVEL INSTITUCIONAL - CORREÇÃO FINAL]
        Treinamento robusto com checkpointing e frequência de avaliação dinâmica.
        """
        if timesteps_to_train <= 0:
            logger.info(f"✅ [HRL PPO] Nenhum timestep adicional necessário. O objetivo de treino já foi alcançado. Pulando.")
            return self.ppo_agent.num_timesteps if self.ppo_agent else 0

        if not self.hyperparams:
            logger.warning("⚡ [HRL PPO] Pulando otimização. Usando hiperparâmetros mínimos para teste rápido.")
            self.hyperparams = { "learning_rate": 3e-4, "n_steps": 32, "batch_size": 16, "gamma": 0.99, "ent_coef": 0.01, "clip_range": 0.2 }

        min_batch_size = self.hyperparams.get('n_steps', 2048)
        
        logger.info(f"💪 [HRL PPO] Recebido pedido para treinar o HRLMaster por mais {timesteps_to_train} timesteps.")
        
        if timesteps_to_train < min_batch_size:
            logger.warning(f"⚠️ [HRL PPO] O treino solicitado de {timesteps_to_train} steps é menor que o lote mínimo do PPO ({min_batch_size}). O treino será executado pelo lote mínimo de {min_batch_size} steps.")
        
        try:
            # [FRAME STACKING] Sequência temporal real para o LSTM do HRL Master.
            # O HRL Master usa PPO para orquestrar especialistas — ter contexto temporal
            # dos últimos N candles de regime/mercado melhora a transição entre especialistas.
            N_STACK_FRAMES = 4

            train_df, eval_df = train_test_split(featured_df, test_size=0.15, shuffle=False)
            train_env = DummyVecEnv([lambda: MasterEnv(train_df, specialists, self.config, mode='training')])
            train_env = VecFrameStack(train_env, n_stack=N_STACK_FRAMES)   # ← sequência temporal real
            train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

            eval_env = DummyVecEnv([lambda: MasterEnv(eval_df, specialists, self.config, mode='training')])
            eval_env = VecFrameStack(eval_env, n_stack=N_STACK_FRAMES)     # ← sequência temporal real
            eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, clip_reward=10.0)

            checkpoint_dir = os.path.join(self.config.CHECKPOINT_DIR, "hrl_master_checkpoints")
            best_model_path = os.path.join(checkpoint_dir, "best_model.zip")
            os.makedirs(checkpoint_dir, exist_ok=True)

            early_stopping_callback = EarlyStoppingCallback(patience=self.config.PATIENCE_EARLY_STOPPING, verbose=1)

            # --- INÍCIO DA CORREÇÃO DA LÓGICA DE FREQUÊNCIA ---
            
            # Começamos com a frequência da configuração (ex: 100)
            eval_freq_final = self.config.CHECKPOINT_FREQ
            
            # Se o treino pedido for menor que a frequência, ajustamos a frequência
            # para ser igual ao lote mínimo, garantindo uma única avaliação no final do lote.
            if timesteps_to_train < eval_freq_final:
                eval_freq_final = min_batch_size

            logger.info(f"⚙️ [HRL PPO] Frequência de avaliação final definida para: {eval_freq_final} steps.")
            
            # Usamos a variável final e corrigida aqui.
            financial_metrics_path = os.path.join(checkpoint_dir, "financial_eval_metrics.json")
            eval_callback = FinancialEvalCallback(
                eval_env,
                best_model_save_path=checkpoint_dir,
                log_path=checkpoint_dir,
                eval_freq=eval_freq_final,  # <<< USA A FREQUÊNCIA FINAL CORRIGIDA
                deterministic=True,
                render=False,
                n_eval_episodes=1,  # Reduzido para 1 para testes mais rápidos
                callback_after_eval=early_stopping_callback,
                metrics_log_path=financial_metrics_path,
                log_name="HRLMaster",
            )
            
            # --- FIM DA CORREÇÃO ---

            # Lógica robusta de carregamento e validação de shape
            if os.path.exists(best_model_path):
                logger.info(f"🔄 [HRL PPO] Checkpoint encontrado. Carregando de '{best_model_path}' para continuar.")
                try:
                    self.ppo_agent = PPO.load(best_model_path, env=train_env, device=self.device)
                except ValueError as e:
                    if "Observation spaces do not match" in str(e):
                        logger.warning(f"⚠️ [HRL PPO] Arquitetura incompatível no Checkpoint (Features alteradas). Iniciando treino DO ZERO. Erro: {e}")
                        self.ppo_agent = None
                    else:
                        raise e
            
            # Validação adicional para agente carregado via __init__ ou checkpoint
            if self.ppo_agent is not None:
                # Verifica compatibilidade de shape (mesmo se loaded sem erro, o set_env pode falhar depois)
                curr_shape = self.ppo_agent.observation_space.shape
                env_shape = train_env.observation_space.shape
                if curr_shape != env_shape:
                    logger.warning(f"⚠️ [HRL PPO] Mismatch detectado: Agente {curr_shape} vs Ambiente {env_shape}. Reiniciando modelo.")
                    self.ppo_agent = None
                else:
                    self.ppo_agent.set_env(train_env)

            if self.ppo_agent is None:
                logger.info(
                    "🛠️ [HRL PPO] Criando nova instância PPO com HybridLSTM+Attention "
                    "(VecFrameStack n_stack=%d → LSTM com sequência temporal real).",
                    N_STACK_FRAMES,
                )
                ppo_params = { "learning_rate": self.hyperparams.get('learning_rate', 3e-4), "n_steps": self.hyperparams.get('n_steps', 2048), "batch_size": self.hyperparams.get('batch_size', 128), "gamma": self.hyperparams.get('gamma', 0.99), "ent_coef": self.hyperparams.get('ent_coef', 0.01), "clip_range": self.hyperparams.get('clip_range', 0.1) }
                # [UPGRADE] HybridLSTMAttentionExtractor com VecFrameStack real.
                # O HRL Master orquestra especialistas — contexto temporal de N candles
                # de regime melhora decisões de transição entre Bull/Bear/Ranger.
                from specialists.scientific_corrections import get_hybrid_lstm_policy_kwargs
                hrl_policy_kwargs = get_hybrid_lstm_policy_kwargs(
                    n_stack=N_STACK_FRAMES,
                    features_dim=128,
                    lstm_hidden_size=64,
                    num_heads=4,
                    dropout=0.15,
                )
                self.ppo_agent = PPO("MlpPolicy", train_env, **ppo_params, policy_kwargs=hrl_policy_kwargs, verbose=1, device=self.device, tensorboard_log=TENSORBOARD_LOG_DIR_HRL)

            # BUG FIX: PPO.learn recebe timesteps_to_train para ESTA SESSÃO.
            # Se fosse .num_timesteps + timesteps_to_train, dobraria o tempo de treino a cada iteração!
            total_target_timesteps = timesteps_to_train

            self.ppo_agent.learn(
                total_timesteps=total_target_timesteps,
                reset_num_timesteps=False,
                tb_log_name="HRLMaster",   # aparece como HRLMaster/PPO_0 no TensorBoard
                callback=eval_callback,
            )
            
            self.is_trained = True
            self.ppo_agent.save(self.model_path)
            logger.info(f"✅ [HRL PPO] Treinamento do HRLMaster concluído. Total de timesteps: {self.ppo_agent.num_timesteps}")
            return self.ppo_agent.num_timesteps

        except Exception as e:
            logger.exception(f"❌ [ERRO HRL PPO] Erro catastrófico durante o treinamento: {e}")
            self.is_trained = False
            return self.ppo_agent.num_timesteps if self.ppo_agent else 0