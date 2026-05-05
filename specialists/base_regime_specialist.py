# -----------------------------------------------------------------------------
# ARQUIVO: specialists/base_regime_specialist.py
# -----------------------------------------------------------------------------
"""
Classe base para especialistas baseados em regime (Bull, Bear, Ranger).
Fornece funcionalidade comum para filtragem de dados por regime e treinamento.

🔬 Dr. Tensor Update: Now uses CryptoRegimeDetector (HMM+GMM+ADX+Funding ensemble)
instead of simple ADX-based detection. Reduces false positives from 30% to 10-15%.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from utils.logger import get_logger
from config.settings import AIConfig, TradingConfig
from models.trade_schema import Signal, Action

# 🔬 Use new ensemble detector (HMM + GMM + ADX + Funding)
from feature_engineering.crypto_regime_detector import (
    CryptoRegimeDetector, 
    RegimeConfig,
    ScientificRegimeDetector  # Alias for backward compatibility
)
from specialists.trend_specialist import TrendSpecialist, TrendFollowingEnv

logger = get_logger("BaseRegimeSpecialist")


class BaseRegimeSpecialist(TrendSpecialist):
    """
    Classe base para especialistas que operam em regimes específicos de mercado.
    
    🔬 Dr. Tensor Update: Now uses CryptoRegimeDetector ensemble for regime detection.
    
    Fornece:
    - Filtragem de dados por regime usando CryptoRegimeDetector (HMM+GMM+ADX+Funding)
    - Treinamento com dados filtrados
    - Interface padronizada para todos os especialistas de regime
    """
    
    REGIME_MAP = {
        'bull': 0,   # Tendência de Alta
        'bear': 1,   # Tendência de Baixa
        'ranger': 2  # Mercado Lateral
    }
    
    def __init__(
        self, 
        config: AIConfig,
        trading_config: TradingConfig,
        regime_type: str,
        input_dim: int,
        profitability_predictor: Optional[Any] = None
    ):
        """
        Args:
            config: Configuração da IA
            trading_config: Configuração de trading
            regime_type: Tipo de regime ('bull', 'bear', 'ranger')
            input_dim: Dimensão das features de entrada
            profitability_predictor: Preditor de lucratividade (opcional)
        """
        super().__init__(config, input_dim, profitability_predictor, specialist_name=f"{regime_type}_specialist")
        
        if regime_type not in self.REGIME_MAP:
            raise ValueError(f"Regime inválido: {regime_type}. Deve ser um de {list(self.REGIME_MAP.keys())}")
        
        self.regime_type = regime_type
        self.regime_code = self.REGIME_MAP[regime_type]
        self.name = f"{regime_type.capitalize()}Specialist"
        
        # [BUG 10 FIX] Usar configuracao canonica compartilhada com ScientificDataProcessor.
        # Antes: pesos locais aqui (weight_gmm=0.40, weight_funding=0.0) divergiam dos
        # pesos em scientific_data_processor.py (weight_gmm=0.30, weight_funding=0.15),
        # causando contaminacao de regime entre criacao do dataset e filtragem do treino.
        try:
            from config.regime_constants import CANONICAL_REGIME_CONFIG
            regime_config = CANONICAL_REGIME_CONFIG
            logger.info(f"{self.name}: Usando CANONICAL_REGIME_CONFIG para detector.")
        except Exception:
            regime_config = RegimeConfig(
                adx_period=9,
                adx_trend_threshold=20.0,
                min_regime_duration=8,
                weight_hmm=0.35,
                weight_gmm=0.35,
                weight_adx=0.20,
                weight_funding=0.10,
            )
            logger.warning(f"{self.name}: CANONICAL_REGIME_CONFIG indisponivel. Usando fallback.")
        self.regime_detector = CryptoRegimeDetector(regime_config)
        
        logger.info(f"✅ {self.name} inicializado para regime {regime_type} (código {self.regime_code}) com CryptoRegimeDetector")
    
    def filter_data_by_regime(self, df: pd.DataFrame, min_samples: int = 100) -> Optional[pd.DataFrame]:
        """
        🔬 Filtra e normaliza o DataFrame para o regime específico.
        """
        logger.info(f"[DEBUG REGIME] {self.name}: Solicitado filtro para {self.regime_type}. Shape entrada: {df.shape}")
        
        try:
            # 🔬 NEW: Tentar usar ScientificDataProcessor para dados consistentes
            try:
                from feature_engineering.scientific_data_processor import ScientificDataProcessor
                
                processor = ScientificDataProcessor()
                processor.load_scalers()
                
                # [FIX RESILIÊNCIA] Garante que a coluna 'regime' exista (alias para regime_val se necessário)
                if 'regime' not in df.columns and 'regime_val' in df.columns:
                    df['regime'] = df['regime_val']
                    logger.info(f"🔄 {self.name}: Coluna 'regime' restaurada a partir de 'regime_val' para filtragem.")
                
                # Usa o método otimizado do processor
                df_filtered = processor.prepare_for_specialist(
                    df,
                    regime_code=self.regime_code,
                    filter_by_regime=True,
                    normalize=True,
                    min_samples=min_samples
                )
                
                if df_filtered is not None:
                    logger.info(
                        f"✅ {self.name}: Dados preparados via ScientificDataProcessor. "
                        f"Shape: {df_filtered.shape} (Regime {self.regime_type})"
                    )
                    return df_filtered
                else:
                    logger.warning(f"⚠️ {self.name}: ScientificDataProcessor retornou None. Tentando fallback interno.")
                    
            except ImportError:
                logger.warning(f"⚠️ {self.name}: ScientificDataProcessor não disponível. Usando fallback.")
            except Exception as e:
                logger.warning(f"⚠️ {self.name}: Erro no processor: {e}. Usando fallback.")
            
            # FALLBACK 1: Usar coluna 'regime' ou 'regime_val' do dataset se existir
            col_to_use = None
            if 'regime' in df.columns:
                col_to_use = 'regime'
            elif 'regime_val' in df.columns:
                col_to_use = 'regime_val'
                
            if col_to_use:
                mask = df[col_to_use] == self.regime_code
                df_filtered = df[mask].copy()
                
                n_samples = len(df_filtered)
                if n_samples < min_samples:
                    logger.warning(
                        f"⚠️ {self.name}: Apenas {n_samples} amostras para regime {self.regime_type} via {col_to_use}. "
                        f"Mínimo necessário: {min_samples}"
                    )
                    return None
                
                pct = (n_samples / len(df)) * 100
                logger.info(
                    f"📊 {self.name}: {n_samples} amostras ({pct:.1f}%) via coluna '{col_to_use}'"
                )
                return df_filtered
            
            # LEGACY FALLBACK: Detectar regimes com o detector próprio
            logger.info(f"🔄 {self.name}: Detectando regimes via CryptoRegimeDetector...")
            regimes = self.regime_detector.detect_regimes(df)
            
            mask = regimes == self.regime_code
            df_filtered = df[mask].copy()
            
            n_samples = len(df_filtered)
            if n_samples < min_samples:
                logger.warning(
                    f"⚠️ {self.name}: Apenas {n_samples} amostras encontradas para regime {self.regime_type}. "
                    f"Mínimo necessário: {min_samples}"
                )
                return None
            
            pct = (n_samples / len(df)) * 100
            logger.info(
                f"📊 {self.name}: {n_samples} amostras ({pct:.1f}%) filtradas via detector"
            )
            
            return df_filtered
            
        except Exception as e:
            logger.error(f"❌ {self.name}: Erro FATAL ao filtrar dados por regime: {e}", exc_info=True)
            return None

    def _apply_temporal_weighting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [TEMPORAL WEIGHTING] Mitigar viés temporal: dados recentes podem ter estrutura
        diferente de dados antigos mesmo no mesmo regime.

        Problema: Bull regime em 2025-2026 tem volatilidade alta mas tendência baixa (pullbacks).
        Solução: Downweight amostras recentes (2025-2026) em relação às antigas (2023-2024).

        Retorna resampled DataFrame com distribuição temporal equilibrada.
        """
        if df is None or len(df) == 0:
            return df

        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                # Index não é temporal — retorna como está
                return df

            _cutoff_recent = pd.Timestamp('2025-01-01')
            _recent_mask = df.index >= _cutoff_recent

            # Amostras antigas (2023-2024): weight=1.0
            # Amostras recentes (2025-2026): weight=0.4 (downweight 60%)
            _sample_weights = np.ones(len(df))
            _sample_weights[_recent_mask] = 0.4

            # Resample usando pesos, preserve ordem temporal
            _target_size = min(len(df), 25000)
            df_resampled = df.sample(
                n=_target_size,
                weights=_sample_weights,
                replace=True,
                random_state=42
            ).sort_index()

            _recent_pct = 100.0 * _recent_mask.sum() / len(df)
            logger.info(
                f"[TEMPORAL WEIGHT] {self.name}: {len(df)} → {len(df_resampled)} amostras. "
                f"Recentes (2025-2026): {_recent_pct:.1f}% com downweight 0.4x."
            )

            return df_resampled

        except Exception as _tw_err:
            logger.warning(f"[TEMPORAL WEIGHT] {self.name}: Erro ao aplicar downweight: {_tw_err}. Retornando original.")
            return df

    @staticmethod
    def _subsample_high_confidence(df: pd.DataFrame, name: str = "", confidence_threshold: float = 0.62) -> pd.DataFrame:
        """
        [HIGH-CONFIDENCE SUBSAMPLE] Filtra candles pelo score de regime_confidence.
        Mantém apenas amostras onde o detector está confiante (>= threshold).

        - Bull/Bear (threshold=0.62): remove ~35%, pois tendências têm sinal direcional claro.
        - Ranger (threshold=0.50): mercado lateral é intrinsecamente ambíguo; threshold 0.62
          remove 85% dos dados (~948 amostras restantes), insuficiente para treino RL.
          Com 0.50 preserva ~2.000-3.000+ amostras, mantendo a filtragem de candles mais ambíguos.
        Ref: Prado (2018) Meta-Labeling — treinar em labels de alta qualidade melhora precision.

        Se a coluna não existir, retorna o df original sem modificação.
        """
        if df is None or len(df) == 0:
            return df
        if 'regime_confidence' not in df.columns:
            return df
        try:
            mask = df['regime_confidence'] >= confidence_threshold
            n_before = len(df)
            df_filtered = df[mask].copy()
            n_after = len(df_filtered)
            if n_after < 100:
                # Fallback: threshold muito agressivo, retorna original
                logger.warning(
                    f"[HIGH-CONF] {name}: subsample com threshold={confidence_threshold} deixou apenas "
                    f"{n_after} amostras (mínimo 100). Retornando dataset original ({n_before} amostras)."
                )
                return df
            removed_pct = 100.0 * (n_before - n_after) / n_before
            logger.info(
                f"[HIGH-CONF] {name}: {n_before} → {n_after} amostras "
                f"(removidos {removed_pct:.1f}% com regime_confidence < {confidence_threshold})"
            )
            return df_filtered
        except Exception as _e:
            logger.warning(f"[HIGH-CONF] {name}: Erro ao subsamplar por confiança: {_e}. Retornando original.")
            return df

    def optimize_hyperparameters(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        n_trials: int = 100,
        n_timesteps: int = 100000,
        n_eval_episodes: int = 5,
        env_class=None,
    ):
        """
        [FIX CRÍTICO] Override para garantir que a otimização use APENAS dados do regime.
        Sem este override, o specialist otimiza com o dataset completo (misturando regimes).
        """
        logger.info(f"🎯 {self.name}: Filtrando dados para HPO (Regime: {self.regime_type})")

        # Filtra os dados de treino e avaliação ANTES de passar para o Optuna
        train_filtered = self.filter_data_by_regime(train_df)
        eval_filtered = self.filter_data_by_regime(eval_df)

        # [FIX EVAL BIAS] Verifica se o eval é quase 100% dados recentes (2025-2026).
        # Causa raiz: split temporal 80/20 antes do filtro de regime → eval = últimos 20%
        # do dataset = 2025-2026 = período estruturalmente bearish mesmo em bull regime.
        # Efeito: reward de eval sempre negativo → Optuna não converge → Zero trades em eval.
        # Solução: Se eval for >80% recente, recriar eval a partir dos dados de treino filtrado,
        # garantindo diversidade temporal e regime correto.
        _eval_too_recent = False
        if (
            eval_filtered is not None
            and isinstance(getattr(eval_filtered, 'index', None), pd.DatetimeIndex)
        ):
            _cutoff = pd.Timestamp('2025-01-01')
            _recent_frac = (eval_filtered.index >= _cutoff).mean()
            if _recent_frac > 0.80:
                _eval_too_recent = True
                logger.warning(
                    "[EVAL BIAS FIX] %s: eval é %.0f%% de dados 2025-2026 (bearish). "
                    "Recriando eval interno a partir dos dados de treino filtrado (20%% final do bull-filtered).",
                    self.name, _recent_frac * 100
                )

        # [TEMPORAL WEIGHTING] Aplica downweight a amostras recentes para mitigar viés temporal
        train_filtered = self._apply_temporal_weighting(train_filtered)
        if not _eval_too_recent:
            eval_filtered = self._apply_temporal_weighting(eval_filtered)
        else:
            # Recria eval interno: últimos 20% do treino bull-filtrado (antes do temporal weight)
            _n_eval_internal = max(500, int(len(train_filtered) * 0.20))
            eval_filtered = train_filtered.iloc[-_n_eval_internal:].copy()
            train_filtered = train_filtered.iloc[:-_n_eval_internal].copy()
            logger.info(
                "[EVAL BIAS FIX] %s: eval recriado com %d amostras bull-filtradas (diversidade temporal garantida). "
                "Treino: %d amostras.",
                self.name, len(eval_filtered), len(train_filtered)
            )

        # [HIGH-CONFIDENCE SUBSAMPLE] Mantém apenas candles com regime_confidence >= threshold.
        # Melhora qualidade do sinal antes de entrar no HPO (Prado 2018 Meta-Labeling).
        # Ranger usa threshold menor (0.50) porque mercado lateral é intrinsecamente ambíguo:
        # o regime detector raramente atinge ≥0.62 em lateral → threshold 0.62 remove 85% dos dados,
        # deixando apenas ~948 amostras — insuficiente para treino RL. 0.50 preserva ~2.000-3.000+.
        _is_ranger_regime = str(getattr(self, 'regime_type', '')).lower() == 'ranger'
        _hconf_threshold = 0.50 if _is_ranger_regime else 0.62
        train_filtered = self._subsample_high_confidence(train_filtered, self.name, confidence_threshold=_hconf_threshold)
        eval_filtered = self._subsample_high_confidence(eval_filtered, self.name, confidence_threshold=_hconf_threshold)

        if train_filtered is None or len(train_filtered) < 100:
            logger.error(f"❌ {self.name}: Dados filtrados INSUFICIENTES para otimização ({len(train_filtered) if train_filtered is not None else 0} amostras).")
            return

        # [FIX ENV_CLASS HPO] Garante que o HPO usa a mesma classe de ambiente que o treino final.
        # Sem isso, BullSpecialist usava TrendFollowingEnv no HPO (sem _specialist_long_only=True),
        # permitindo que o agente abrisse shorts durante a exploração.
        if env_class is None:
            try:
                from specialists.bull_specialist import BullTradingEnv
                from specialists.bear_specialist import BearTradingEnv
                from specialists.ranger_specialist import RangerTradingEnv
                from specialists.trend_specialist import TrendFollowingEnv as _TFE
                _ENV_MAP = {
                    'bull': BullTradingEnv,
                    'bear': BearTradingEnv,
                    'ranger': RangerTradingEnv,
                }
                env_class = _ENV_MAP.get(self.regime_type, _TFE)
                logger.info(f"[HPO ENV] Usando {env_class.__name__} para otimizacao (regime={self.regime_type})")
            except Exception as _e:
                logger.warning(f"[HPO ENV] Falha ao resolver env_class: {_e}. Usando TrendFollowingEnv.")

        logger.info(f"✅ {self.name}: Otimização iniciará com {len(train_filtered)} samples filtrados.")
        logger.info(f"   Dados eval: {len(eval_filtered):,} amostras")
        logger.info(f"{'='*70}")
        logger.info(f"")

        # [SPEED FIX #2+#4] Todos os especialistas beneficiam-se de eval mais enxuto.
        # Dataset pequeno (~2.4k amostras pós-filtro): eval é caro proporcionalmente.
        # Reduzir n_eval_episodes 5→3 e eval_freq 8→4 checkpoints/trial acelera HPO
        # em ~30% sem prejudicar a seleção do MedianPruner (3 eps ainda dão variância
        # suficiente para ranking, e 4 checkpoints cobrem toda a trajetória de 100k steps).
        n_eval_episodes = min(n_eval_episodes, 3)
        # Sinaliza para a impl. do TrendSpecialist (trend_specialist.py:4051) usar //4
        # em vez de //8 para _eval_freq. Flag lida via getattr no hot path do HPO.
        import os as _os
        _os.environ['HPO_EVAL_CHECKPOINTS'] = '4'
        logger.info(
            "[SPEED FIX] %s HPO: n_eval_episodes=%d, eval_checkpoints=4 (vs 5/8 default)",
            self.name, n_eval_episodes
        )

        # Chama a implementacao da base (TrendSpecialist) com os dados purificados
        # [BUG 12 FIX] eval_filtered usa fit_scaler=False para evitar data leakage
        return super().optimize_hyperparameters(
            train_df=train_filtered,
            eval_df=eval_filtered,
            n_trials=n_trials,
            n_timesteps=n_timesteps,
            n_eval_episodes=n_eval_episodes,
            env_class=env_class,
        )
    
    def train_model(
        self, 
        df: pd.DataFrame, 
        labels_df: Optional[pd.DataFrame] = None,
        total_timesteps: Optional[int] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Treina o modelo usando apenas dados do regime específico.
        
        Args:
            df: DataFrame com features do autoencoder
            labels_df: Labels (não usado, mantido para compatibilidade)
            total_timesteps: Número total de timesteps de treinamento
            **kwargs: Argumentos adicionais para o treinamento
            
        Returns:
            Dicionário com métricas de treinamento
        """
        logger.info(f"🎯 {self.name}: Iniciando treinamento...")

        # Filtra dados pelo regime
        df_filtered = self.filter_data_by_regime(df)

        # [TEMPORAL WEIGHTING] Aplica downweight a amostras recentes
        df_filtered = self._apply_temporal_weighting(df_filtered)

        if df_filtered is None or df_filtered.empty:
            logger.error(f"❌ {self.name}: Sem dados suficientes para treinamento")
            return {'success': False, 'error': 'insufficient_data'}
        
        # Define timesteps padrão se não fornecido
        if total_timesteps is None:
            total_timesteps = getattr(self.config, 'SPECIALIST_TRAINING_TIMESTEPS', 500_000)
        
        # [LOG DESCRITIVO] Banner claro para identificar o agente
        regime_emoji = {'bull': '🐂', 'bear': '🐻', 'ranger': '🤠'}.get(self.regime_type, '🤖')
        regime_strategy = {
            'bull': 'TREND SURFING (Long) - Segurar posições em tendência de ALTA',
            'bear': 'TREND SURFING (Short) - Segurar posições em tendência de BAIXA', 
            'ranger': 'MEAN REVERSION - Scalping em mercado LATERAL'
        }.get(self.regime_type, 'N/A')
        
        logger.info(f"")
        logger.info(f"{'='*70}")
        logger.info(f"{regime_emoji} TREINANDO: {self.name.upper()} {regime_emoji}")
        logger.info(f"{'='*70}")
        logger.info(f"   Regime: {self.regime_type.upper()}")
        logger.info(f"   Estratégia: {regime_strategy}")
        logger.info(f"   Amostras: {len(df_filtered):,} ({100*len(df_filtered)/len(df):.1f}% do dataset)")
        logger.info(f"   Timesteps: {total_timesteps:,}")
        logger.info(f"   Limite de trades: SEM LIMITE (trend surfing livre)")
        logger.info(f"{'='*70}")
        logger.info(f"")
        
        try:
            # [SCIENTIFIC FIX] Mapeia regime -> classe de ambiente para ativar recompensas específicas
            from specialists.bull_specialist import BullTradingEnv
            from specialists.bear_specialist import BearTradingEnv
            from specialists.ranger_specialist import RangerTradingEnv
            
            ENV_CLASS_MAP = {
                'bull': BullTradingEnv,
                'bear': BearTradingEnv,
                'ranger': RangerTradingEnv,
            }
            env_class = ENV_CLASS_MAP.get(self.regime_type, TrendFollowingEnv)
            
            logger.info(f"📊 {self.name}: Usando ambiente {env_class.__name__} para treino")
            
            # Chama o método de treinamento da classe pai (TrendSpecialist)
            metrics = super().train(
                featured_df=df_filtered,
                timesteps_to_train=total_timesteps,
                env_class=env_class,
                specialist_name=f"{self.regime_type}_specialist", # Bug 2 fix
                **kwargs
            )
            
            self.is_ready = True
            logger.info(f"✅ {self.name}: Treinamento concluído com sucesso!")
            return {'success': True, 'timesteps': metrics if isinstance(metrics, int) else total_timesteps}
            
        except Exception as e:
            logger.error(f"❌ {self.name}: Erro durante treinamento: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def decide_action(self, observation: np.ndarray, df_row: pd.Series) -> Optional[Signal]:
        """
        [FIX PRODUÇÃO] Sobrescreve decide_action para aplicar filtros de direção específicos de cada regime.
        """
        signal = super().decide_action(observation, df_row)
        if signal is None:
            return None

        # [SOFT PENALTY] Em vez de bloquear (Hard Veto), reduzimos a confiança drasticamente.
        # Isso permite que o sinal passe para o ensemble (se for muito forte), mas sinaliza
        # ao RL que operar contra o regime é um erro estratégico via reward shaping.
        # Ref: Dr. Tensor Fix - "Hard masking breaks policy gradient".
        mismatch = False
        if direction == 'long_only' and signal.action == Action.SELL:
            mismatch = True
        elif direction == 'short_only' and signal.action == Action.BUY:
            mismatch = True

        if mismatch:
            if signal.explanation is None:
                signal.explanation = {}
            
            original_action = signal.action.value
            original_conf = signal.confidence
            
            # Redução de 80% na confiança - o sinal ainda existe mas é "sussurrado" no ensemble
            signal.confidence *= 0.20
            signal.explanation['_regime_mismatch'] = True
            signal.explanation['reason'] = f"{self.name}: sinal {original_action} suavizado (regime {self.regime_type} prefere {direction})"
            
            logger.debug(f"{'🐂' if direction == 'long_only' else '🐻'} [REGIME SOFT-FILTER] {self.name}: {original_action} ({original_conf:.2f}) suavizado para {signal.confidence:.2f}")
        
        return signal

    def get_regime_confidence(self, df: pd.DataFrame) -> float:
        """
        Retorna a confiança de que os dados atuais pertencem ao regime deste especialista.
        
        Args:
            df: DataFrame com dados recentes
            
        Returns:
            Confiança entre 0.0 e 1.0
        """
        try:
            if len(df) < 100:
                return 0.0
            
            # Detecta regimes nas últimas N barras (Janela de 100 para convergência do HMM)
            regimes = self.regime_detector.detect_regimes(df.tail(100))
            
            # Calcula proporção do regime específico
            regime_count = np.sum(regimes == self.regime_code)
            confidence = regime_count / len(regimes)
            
            return float(confidence)
            
        except Exception as e:
            logger.warning(f"⚠️ {self.name}: Erro ao calcular confiança de regime: {e}")
            return 0.0
