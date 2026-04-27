# -----------------------------------------------------------------------------
# ARQUIVO: feature_engineering/scientific_data_processor.py
# -----------------------------------------------------------------------------
"""
🔬 Scientific Data Processor for ML Pipeline

Based on peer-reviewed research:
- Hamilton (1989): Regime-conditional normalization
- Ang & Bekaert (2002): Regime-specific statistics
- de Prado (2018): Purged cross-validation, fractional differentiation

This module provides:
1. Regime-conditional Z-score normalization
2. Feature filtering for autoencoders
3. Consistent data delivery to specialists
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import RobustScaler, StandardScaler
from dataclasses import dataclass
import joblib
import os

from utils.logger import get_logger

logger = get_logger("ScientificDataProcessor")


@dataclass
class RegimeStats:
    """Statistics for a specific regime."""
    regime_code: int
    regime_name: str
    n_samples: int
    percentage: float
    mean_return: float
    volatility: float


class ScientificDataProcessor:
    """
    🔬 Scientific Data Processor for consistent ML pipeline.
    
    Implements:
    1. Regime-conditional normalization (Hamilton 1989)
    2. Feature filtering for autoencoders
    3. Anti-leakage scaling (fit on train only)
    
    Usage:
        processor = ScientificDataProcessor()
        df_normalized = processor.prepare_for_specialists(df, regime_code=0)
    """
    
    # Features to EXCLUDE from autoencoder (absolute values)
    ABSOLUTE_VALUE_PREFIXES = [
        'open', 'high', 'low', 'close',  # Prices
        'volume',  # Volume (not volume_z)
        'ema_',  # EMAs are absolute prices
        'bb_upper', 'bb_middle', 'bb_lower',  # Bollinger absolute
        'vwap', 'psar',  # Absolute prices
        'obv',  # On-Balance Volume (cumulative)
        'pvt', 'adl',  # Cumulative volume indicators
        'macd_line', 'macd_signal',  # Price-scale values
        'atr_abs',  # ATR absolute (prefix for strictly absolute versions)
        'tp_prior_',  # Regime signals from TrendPredictor/Detector (0-1 or -1-1)
        'regime_',  # Confidence and other regime metadata
    ]
    
    ABSOLUTE_VALUE_CONTAINS = [
        '_tf_',  # Timeframe absolutes
        'close_reference',  # Reference prices
    ]
    # As colunas de regime agora são tratadas separadamente por nome exato para evitar falso-positivos
    REGIME_METADATA_COLS = ['regime', 'regime_confidence', 'regime_name', 'regime_val']
    
    # Features to KEEP (normalized indicators)
    NORMALIZED_FEATURE_PATTERNS = [
        '_frac_',  # Fractional diff of prices (STATIONARY but preserves memory)
        'rsi', 'stoch', 'williams', 'cci',  # Oscillators (bounded)
        'adx', 'plus_di', 'minus_di',  # Trend strength (bounded)
        'macd_hist',  # MACD histogram (difference)
        'log_return', 'pct',  # Returns (centered)
        '_z', '_ratio',  # Z-scores and ratios
        'bb_width', 'atr_percentage',  # Normalized measures
        'squeeze', 'expansion', 'trend', 'cross',  # Binary/boolean
        'overbought', 'oversold', 'strong', 'weak',  # Boolean
        'doji', 'marubozu', 'engulfing', 'hammer',  # Candlestick patterns
    ]
    
    REGIME_NAMES = {0: 'Bull', 1: 'Bear', 2: 'Ranger'}
    
    def __init__(self, model_dir: str = 'models_ai'):
        """
        Initialize the processor.
        
        Args:
            model_dir: Directory for saving scalers
        """
        self.model_dir = model_dir
        self.scalers: Dict[int, RobustScaler] = {}  # Per-regime scalers
        self.global_scaler = RobustScaler()
        self.feature_columns: List[str] = []
        self.autoencoder_columns: List[str] = []
        self.regime_stats: Dict[int, RegimeStats] = {}
        
        os.makedirs(model_dir, exist_ok=True)
        self.scaler_path = os.path.join(model_dir, 'scientific_scalers.joblib')
        
        logger.info("✅ ScientificDataProcessor initialized")
    
    def add_regime_labels(
        self, 
        df: pd.DataFrame, 
        detector: Any = None
    ) -> pd.DataFrame:
        """
        Add regime labels to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            detector: Pre-trained CryptoRegimeDetector (optional)
            
        Returns:
            DataFrame with 'regime' and 'regime_confidence' columns
        """
        from feature_engineering.crypto_regime_detector import (
            CryptoRegimeDetector, RegimeConfig
        )
        
        df = df.copy()
        
        # Load or create detector
        if detector is None:
            detector = CryptoRegimeDetector.load()
            
            if detector is None or not detector.is_trained:
                logger.info("Training new CryptoRegimeDetector...")
                # [BUG 10 FIX] Usar configuracao canonica compartilhada.
                # Antes: pesos locais aqui divergiam dos pesos em base_regime_specialist.py.
                try:
                    from config.regime_constants import CANONICAL_REGIME_CONFIG
                    config = CANONICAL_REGIME_CONFIG
                    logger.info("Usando CANONICAL_REGIME_CONFIG para detector de regime.")
                except Exception:
                    config = RegimeConfig(
                        adx_period=9,
                        adx_trend_threshold=20.0,
                        min_regime_duration=8,
                        weight_hmm=0.35,
                        weight_gmm=0.35,
                        weight_adx=0.20,
                        weight_funding=0.10,
                    )
                    logger.warning("CANONICAL_REGIME_CONFIG nao disponivel. Usando fallback padrao.")
                detector = CryptoRegimeDetector(config)
                # [ANTI-LEAKAGE v3.1 — López de Prado MLAM Ch.7]
                # Regime detector treinado SOMENTE nos primeiros 80% do dataset.
                # Os 20% finais são "futuro invisível" para o detector → sem lookahead.
                # Depois do fit, predict() no dataset completo usando o modelo do passado.
                _fit_split = int(len(df) * 0.80)
                _train_df_regime = df.iloc[:_fit_split]
                logger.info(
                    "[ANTI-LEAKAGE] Regime detector treinado nos primeiros 80%% "
                    "(%d linhas de %d total). 20%% final invisivel ao detector.",
                    _fit_split, len(df)
                )
                detector.fit_predict(_train_df_regime)  # fit only on past 80%
                detector.save()
                result = detector.predict(df)  # predict full dataset with past-only model
            else:
                logger.info("✅ Loaded CryptoRegimeDetector from disk")
                # BUG C3 FIX: Usar predict ao invés de fit_predict para não destruir os parâmetros do HMM
                result = detector.predict(df)
        else:
            # BUG C3 FIX: Mesmo na injeção manual de objeto, checa se está treinado
            if detector.is_trained:
                result = detector.predict(df)
            else:
                result = detector.fit_predict(df)
        
        # Add columns
        df['regime'] = result['regime'].values
        df['regime_confidence'] = result['confidence'].values
        df['regime_name'] = df['regime'].map(self.REGIME_NAMES)
        
        # Compute and log statistics
        self._compute_regime_stats(df)
        
        return df
    
    def _compute_regime_stats(self, df: pd.DataFrame) -> None:
        """Compute and store regime statistics."""
        if 'regime' not in df.columns:
            return
        
        total = len(df)
        
        for code in [0, 1, 2]:
            mask = df['regime'] == code
            regime_df = df[mask]
            n = len(regime_df)
            
            # Compute returns if possible
            if 'close' in df.columns and n > 1:
                returns = regime_df['close'].pct_change().dropna()
                mean_return = returns.mean() if len(returns) > 0 else 0
                volatility = returns.std() if len(returns) > 0 else 0
            else:
                mean_return = 0
                volatility = 0
            
            self.regime_stats[code] = RegimeStats(
                regime_code=code,
                regime_name=self.REGIME_NAMES[code],
                n_samples=n,
                percentage=(n / total * 100) if total > 0 else 0,
                mean_return=mean_return,
                volatility=volatility
            )
        
        logger.info(
            f"📊 Regime Distribution: "
            f"Bull={self.regime_stats[0].percentage:.1f}%, "
            f"Bear={self.regime_stats[1].percentage:.1f}%, "
            f"Ranger={self.regime_stats[2].percentage:.1f}%"
        )
    
    def filter_autoencoder_features(
        self, 
        feature_columns: List[str],
        df: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Filter features suitable for autoencoder training.
        
        Removes absolute-value columns that cause high MSE loss.
        🔬 Dr. Tensor Fix: Added '_frac' to whitelist to preserve stationary price memory.
        """
        filtered = []
        excluded = []
        
        # Combine patterns to keep
        whitelist_patterns = ['_pct', '_percentage', '_z', '_ratio', '_roc', 'log_return', '_frac']
        
        for col in feature_columns:
            col_lower = col.lower()
            
            # [FIX 1] Whitelist explícita para Hidden Features (não importa a escala ou nome)
            if col_lower.startswith('hidden_feature'):
                filtered.append(col)
                continue

            # Check Prefixos e Conteúdo
            is_normalized_pattern = any(pattern in col_lower for pattern in self.NORMALIZED_FEATURE_PATTERNS)
            
            is_absolute_prefix = any(col_lower.startswith(p) for p in self.ABSOLUTE_VALUE_PREFIXES)
            
            # [FIX] Removido match parcial de 'regime' para evitar excluir features úteis
            is_absolute_contains = any(p in col_lower for p in self.ABSOLUTE_VALUE_CONTAINS)
            is_regime_meta = col_lower in self.REGIME_METADATA_COLS
            
            # Decision Logic
            should_exclude = (is_absolute_prefix or is_absolute_contains or is_regime_meta) and not is_normalized_pattern
            
            # Double check for large values
            if not should_exclude and df is not None and col in df.columns:
                try:
                    # [FIX 2] Adicionado fillna(0) antes do cálculo de escala
                    max_abs = df[col].fillna(0).abs().max()
                    if max_abs > 1000 and not is_normalized_pattern:
                        should_exclude = True
                except:
                    pass
            
            if should_exclude:
                excluded.append(col)
            else:
                filtered.append(col)
        
        # [FIX 3] Log explícito para proteção de hidden features
        hidden_excluded = [c for c in excluded if 'hidden' in c.lower()]
        if hidden_excluded:
            logger.warning(f"⚠️ [SCIENTIFIC] Hidden features excluídas pelo filtro (Verificar Escala): {hidden_excluded}")
        
        self.autoencoder_columns = filtered
        
        logger.info(
            f"🔍 Autoencoder features: kept {len(filtered)}, "
            f"excluded {len(excluded)} absolute-value columns"
        )
        
        return filtered
    
    def normalize_for_specialist(
        self,
        df: pd.DataFrame,
        regime_code: int,
        feature_columns: Optional[List[str]] = None,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        🔬 Regime-conditional normalization (Hamilton 1989).
        
        Normalizes features using statistics specific to each regime,
        which produces better results than global normalization.
        
        Args:
            df: DataFrame with features
            regime_code: Target regime (0=Bull, 1=Bear, 2=Ranger)
            feature_columns: Columns to normalize (uses stored if None)
            fit: If True, fit scalers on this data
            
        Returns:
            Normalized DataFrame
        """
        if feature_columns is None:
            # 🔬 Dr. Tensor Fix (Finding #8): RL Env NEEDS raw prices for PnL!
            # If we normalize 'close', 'high', 'low', 'atr_1h', the environment logic breaks.
            # We must filter out absolute non-stationary features before regime normalization.
            all_numeric = list(df.select_dtypes(include=[np.number]).columns)
            feature_columns = self.feature_columns or self.filter_autoencoder_features(all_numeric, df)
        
        df = df.copy()
        
        # Get regime-specific data
        if 'regime' in df.columns:
            regime_mask = df['regime'] == regime_code
            n_regime = regime_mask.sum()
            logger.info(f"📊 Normalizing for {self.REGIME_NAMES[regime_code]}: {n_regime} samples")
        else:
            logger.warning("⚠️ No 'regime' column - using global normalization")
            regime_mask = pd.Series(True, index=df.index)
        
        # Get or create scaler for this regime
        if regime_code not in self.scalers:
            self.scalers[regime_code] = RobustScaler()
        scaler = self.scalers[regime_code]

        # Select numeric columns that exist
        numeric_cols = [c for c in feature_columns if c in df.columns and df[c].dtype in [np.float32, np.float64, np.int64]]
        
        # [SCIENTIFIC DIMENSION CACHE] Check for feature count mismatch (e.g. 266 vs 252)
        if hasattr(scaler, 'n_features_in_'):
            if scaler.n_features_in_ != len(numeric_cols):
                if fit:
                    logger.info(
                        f"🔄 [SDAE] Adaptando scaler de {scaler.n_features_in_} para {len(numeric_cols)} features "
                        f"(Regime: {self.REGIME_NAMES[regime_code]})"
                    )
                else:
                    # Se estamos em inferência, tentamos filtrar para as colunas originais
                    # Mas se as colunas originais sumiram, precisamos avisar.
                    logger.warning(
                        f"⚠️ [SCIENTIFIC] Dimension mismatch in {self.REGIME_NAMES[regime_code]}: "
                        f"Scaler expects {scaler.n_features_in_}, data has {len(numeric_cols)}. "
                        "Isso indica que o Feature Pipeline mudou."
                    )
                    # BUG I1 FIX: Removido 'fit = True' para evitar contaminação (leakage) com estatísticas de inferência.

        if fit:
            # 🔬 [HYBRID REGIME NORMALIZATION] 
            # Fitamos no GLOBAL para a escala (IQR) e no REGIME para o centro (Mediana).
            # Isso impede que o bot ignore sinais fortes de outros regimes.
            global_data = df[numeric_cols].fillna(0).values
            regime_data = df.loc[regime_mask, numeric_cols].fillna(0).values if regime_mask.any() else None
            
            if regime_data is not None and len(regime_data) > 0:
                # 1. Escala Global (Contexto de 101k samples)
                temp_global = RobustScaler()
                temp_global.fit(global_data)
                
                # 2. Centro de Regime (Contexto de 27k samples)
                scaler.fit(regime_data)
                
                # 3. Injeção: Mantém a mediana do regime, mas usa a escala global
                scaler.scale_ = temp_global.scale_
                logger.info(f"✅ Scaler Regime-{regime_code} hibridizado (Estabilidade: Global IQR)")
            else:
                logger.warning(f"⚠️ Dados insuficientes para fit do regime {regime_code}. Usando fit global.")
                scaler.fit(global_data)
        
        # 🔬 [TRANSFORM] Aplicar a normalização híbrida (ou global) ao dataset
        # Agora o df retornado terá escala Z (Robust) adequada ao regime.
        try:
            df[numeric_cols] = scaler.transform(df[numeric_cols].fillna(0).values)
        except Exception as e:
            logger.warning(f"⚠️ [SCIENTIFIC] Transform falhou: {e}. Usando fallback global para evitar crash.")
            fallback = RobustScaler()
            df[numeric_cols] = fallback.fit_transform(df[numeric_cols].fillna(0).values)
        
        self.feature_columns = numeric_cols
        return df
    
    def prepare_for_specialist(
        self,
        df: pd.DataFrame,
        regime_code: int,
        filter_by_regime: bool = True,
        normalize: bool = True,
        min_samples: int = 100,
        fit_scaler: bool = True,  # [BUG 12 FIX] False para eval_df para evitar data leakage
    ) -> Optional[pd.DataFrame]:
        """
        Complete data preparation for specialist training.
        
        1. Filters by regime (if requested)
        2. Applies regime-conditional normalization
        3. Validates sample count
        
        Args:
            df: DataFrame with features and 'regime' column
            regime_code: Target regime (0=Bull, 1=Bear, 2=Ranger)
            filter_by_regime: If True, keeps only regime-specific data
            normalize: If True, applies regime-conditional normalization
            min_samples: Minimum required samples
            fit_scaler: If True, refits the scaler (train only). False for eval to avoid leakage.
            
        Returns:
            Prepared DataFrame or None if insufficient data
        """
        logger.info(f"🎯 Preparing data for {self.REGIME_NAMES[regime_code]} specialist...")
        
        # Validate regime column exists
        if 'regime' not in df.columns:
            logger.error("❌ DataFrame must have 'regime' column. Run add_regime_labels() first.")
            return None
        
        # [BUG 11 FIX] Detectar se as features latentes do autoencoder ja normalizaram os dados.
        # O temporal_autoencoder usa RobustScaler nas features latentes (sdae_latent_*).
        # Se elas estiverem presentes, os dados ja passaram por uma normalizacao robusta
        # e normalizar de novo criaria double-normalization (Z de Z = ruido puro).
        latent_cols_present = any(col.startswith('sdae_latent') or col.startswith('hidden_') 
                                   for col in df.columns)
        if normalize and latent_cols_present:
            logger.info("[BUG 11 FIX] Autoencoder latents detectados. Pulando 2a normalizacao para evitar double-scaling.")
            normalize = False

        # [BUG 12 FIX] fit_scaler=True deve ser usado APENAS no train_df.
        # Para eval_df, sempre False para evitar data leakage das estat. do eval no scaler.
        if normalize:
            df = self.normalize_for_specialist(
                df, 
                regime_code=regime_code,
                fit=fit_scaler
            )
        
        # [Garantir que df tenha regimes após normalização]
        if filter_by_regime:
            mask = df['regime'] == regime_code
            df_prepared = df[mask].copy()
            
            if len(df_prepared) < min_samples:
                logger.warning(
                    f"⚠️ Insufficient samples for {self.REGIME_NAMES[regime_code]}: "
                    f"{len(df_prepared)} < {min_samples}"
                )
                return None
            
            pct = len(df_prepared) / len(df) * 100
            logger.info(f"📊 Filtered: {len(df_prepared)} samples ({pct:.1f}%)")
        else:
            df_prepared = df.copy()
        
        # Validate no NaN
        nan_count = df_prepared.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"⚠️ Found {nan_count} NaN values. Filling with 0.")
            df_prepared = df_prepared.fillna(0)
        
        logger.info(f"✅ Data prepared: {df_prepared.shape}")
        return df_prepared
    
    def prepare_for_autoencoder(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        train_split: float = 0.80
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data for autoencoder training.

        [FIX LEAKAGE] O global_scaler é fit APENAS no split de treino cronológico
        (primeiros `train_split`% dos dados). O split de validação é transformado
        com as estatísticas do treino — sem vazamento de dados futuros.

        Args:
            df: DataFrame com features (index cronológico)
            feature_columns: Todas as colunas de features disponíveis
            train_split: Fração dos dados para fit do scaler (padrão 80%)

        Returns:
            Tuple de (DataFrame normalizado, lista de colunas)
        """
        # Filtrar features absolutas (preços brutos, EMAs, etc.)
        filtered_cols = self.filter_autoencoder_features(feature_columns, df)
        existing_cols = [c for c in filtered_cols if c in df.columns]
        df_filtered = df[existing_cols].copy()

        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # [FIX] Split cronológico: fit apenas nos primeiros 80% dos dados
            split_idx = int(len(df_filtered) * train_split)
            if split_idx < 10:
                # Dataset muito pequeno — fallback para fit global (pouco impacto)
                logger.warning(
                    f"⚠️ Dataset muito pequeno ({len(df_filtered)} linhas) para split. "
                    "Usando fit global (leakage mínimo aceitável)."
                )
                self.global_scaler.fit(df_filtered[numeric_cols].fillna(0).values)
            else:
                train_data = df_filtered.iloc[:split_idx][numeric_cols].fillna(0).values
                self.global_scaler.fit(train_data)
                logger.info(
                    f"✅ [ANTI-LEAKAGE] global_scaler fit em {split_idx} amostras "
                    f"(treino {train_split:.0%}) → transform em {len(df_filtered)} total."
                )
            # Transform no dataset completo usando estatísticas do treino apenas
            df_filtered[numeric_cols] = self.global_scaler.transform(
                df_filtered[numeric_cols].values
            )

        df_filtered = df_filtered.fillna(0)
        logger.info(f"✅ Autoencoder data prepared: {df_filtered.shape}")
        return df_filtered, existing_cols
    
    def save_scalers(self) -> bool:
        """Save all scalers to disk."""
        try:
            state = {
                'regime_scalers': self.scalers,
                'global_scaler': self.global_scaler,
                'feature_columns': self.feature_columns,
                'autoencoder_columns': self.autoencoder_columns,
                'regime_stats': self.regime_stats
            }
            joblib.dump(state, self.scaler_path)
            logger.info(f"✅ Scalers saved to {self.scaler_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save scalers: {e}")
            return False
    
    def load_scalers(self) -> bool:
        """Load scalers from disk."""
        if not os.path.exists(self.scaler_path):
            return False
        
        try:
            state = joblib.load(self.scaler_path)
            self.scalers = state.get('regime_scalers', {})
            self.global_scaler = state.get('global_scaler', RobustScaler())
            self.feature_columns = state.get('feature_columns', [])
            self.autoencoder_columns = state.get('autoencoder_columns', [])
            self.regime_stats = state.get('regime_stats', {})
            logger.info(f"✅ Scalers loaded from {self.scaler_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load scalers: {e}")
            return False


# Convenience function for quick use
def create_data_processor(model_dir: str = 'models_ai') -> ScientificDataProcessor:
    """Factory function to create or load data processor."""
    processor = ScientificDataProcessor(model_dir)
    processor.load_scalers()
    return processor
