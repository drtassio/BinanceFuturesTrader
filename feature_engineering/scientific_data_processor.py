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
        "open",
        "high",
        "low",
        "close",  # Prices
        "volume",  # Volume (not volume_z)
        "ema_",  # EMAs are absolute prices
        "bb_upper",
        "bb_middle",
        "bb_lower",  # Bollinger absolute
        "vwap",
        "psar",  # Absolute prices
        "obv",  # On-Balance Volume (cumulative)
        "pvt",
        "adl",  # Cumulative volume indicators
        "macd_line",
        "macd_signal",  # Price-scale values
        "atr_abs",  # ATR absolute (prefix for strictly absolute versions)
        "atr_1",  # ATR timeframe variants (atr_1m, atr_15m, atr_1h) — absolute price scale
        "atr_4",  # atr_4h
        "atr_5",  # atr_5m
        "tp_prior_",  # Regime signals from TrendPredictor/Detector (0-1 or -1-1)
        "regime_",  # Confidence and other regime metadata
    ]

    ABSOLUTE_VALUE_CONTAINS = [
        "_tf_",  # Timeframe absolutes
        "close_reference",  # Reference prices
    ]
    # As colunas de regime agora são tratadas separadamente por nome exato para evitar falso-positivos
    REGIME_METADATA_COLS = ["regime", "regime_confidence", "regime_name", "regime_val"]

    # Features to KEEP (normalized indicators)
    NORMALIZED_FEATURE_PATTERNS = [
        "_frac_",  # Fractional diff of prices (STATIONARY but preserves memory)
        "rsi",
        "stoch",
        "williams",
        "cci",  # Oscillators (bounded)
        "adx",
        "plus_di",
        "minus_di",  # Trend strength (bounded)
        "macd_hist",  # MACD histogram (difference)
        "log_return",
        "pct",  # Returns (centered)
        "_z",
        "_ratio",  # Z-scores and ratios
        "bb_width",
        "atr_percentage",  # Normalized measures
        "squeeze",
        "expansion",
        "trend",
        "cross",  # Binary/boolean
        "overbought",
        "oversold",
        "strong",
        "weak",  # Boolean
        "doji",
        "marubozu",
        "engulfing",
        "hammer",  # Candlestick patterns
    ]

    REGIME_NAMES = {0: "Bull", 1: "Bear", 2: "Ranger"}

    def __init__(self, model_dir: str = "models_ai"):
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
        self.scaler_path = os.path.join(model_dir, "scientific_scalers.joblib")

        logger.info("✅ ScientificDataProcessor initialized")

    def add_regime_labels(self, df: pd.DataFrame, detector: Any = None) -> pd.DataFrame:
        """
        Add regime labels to DataFrame.

        Args:
            df: DataFrame with OHLCV data
            detector: Pre-trained CryptoRegimeDetector (optional)

        Returns:
            DataFrame with 'regime' and 'regime_confidence' columns
        """
        from feature_engineering.crypto_regime_detector import (
            CryptoRegimeDetector,
            RegimeConfig,
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
                    logger.info(
                        "Usando CANONICAL_REGIME_CONFIG para detector de regime."
                    )
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
                    logger.warning(
                        "CANONICAL_REGIME_CONFIG nao disponivel. Usando fallback padrao."
                    )
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
                    _fit_split,
                    len(df),
                )
                detector.fit_predict(_train_df_regime)  # fit only on past 80%
                detector.save()
                result = detector.predict(
                    df
                )  # predict full dataset with past-only model
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
        df["regime"] = result["regime"].values
        df["regime_confidence"] = result["confidence"].values
        df["regime_name"] = df["regime"].map(self.REGIME_NAMES)

        # Compute and log statistics
        self._compute_regime_stats(df)

        recent_confidence = result["confidence"].tail(1000).mean()
        DRIFT_THRESHOLD = 0.55
        if recent_confidence < DRIFT_THRESHOLD:
            logger.warning(
                f"⚠️ [REGIME DRIFT] Confidence média recente = {recent_confidence:.3f} "
                f"< {DRIFT_THRESHOLD}. Considere re-fit do detector."
            )

        return df

    def _compute_regime_stats(self, df: pd.DataFrame) -> None:
        """Compute and store regime statistics."""
        if "regime" not in df.columns:
            return

        total = len(df)

        for code in [0, 1, 2]:
            mask = df["regime"] == code
            regime_df = df[mask]
            n = len(regime_df)

            # Compute returns if possible
            # FIX: pct_change deve ser calculado sobre o tempo contínuo. Aplicar a máscara
            # ANTES do pct_change faria com que regimes não-contíguos (ex: Bull jan + Bull jul)
            # gerassem retornos cruzando meses como se fossem 1 período, inflando volatilidade.
            if "close" in df.columns and n > 1:
                all_returns = df["close"].pct_change()
                returns = all_returns.loc[mask].dropna()
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
                volatility=volatility,
            )

        logger.info(
            f"📊 Regime Distribution: "
            f"Bull={self.regime_stats[0].percentage:.1f}%, "
            f"Bear={self.regime_stats[1].percentage:.1f}%, "
            f"Ranger={self.regime_stats[2].percentage:.1f}%"
        )

    def filter_autoencoder_features(
        self, feature_columns: List[str], df: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Filter features suitable for autoencoder training.

        Removes absolute-value columns that cause high MSE loss.
        🔬 Dr. Tensor Fix: Added '_frac' to whitelist to preserve stationary price memory.
        """
        filtered = []
        excluded = []

        # Combine patterns to keep
        whitelist_patterns = [
            "_pct",
            "_percentage",
            "_z",
            "_ratio",
            "_roc",
            "log_return",
            "_frac",
        ]

        for col in feature_columns:
            col_lower = col.lower()

            # [FIX 1] Whitelist explícita para Hidden Features (não importa a escala ou nome)
            if col_lower.startswith("hidden_feature"):
                filtered.append(col)
                continue

            # Check Prefixos e Conteúdo
            is_normalized_pattern = any(
                pattern in col_lower for pattern in self.NORMALIZED_FEATURE_PATTERNS
            )

            is_absolute_prefix = any(
                col_lower.startswith(p) for p in self.ABSOLUTE_VALUE_PREFIXES
            )

            # [FIX] Removido match parcial de 'regime' para evitar excluir features úteis
            is_absolute_contains = any(
                p in col_lower for p in self.ABSOLUTE_VALUE_CONTAINS
            )
            is_regime_meta = col_lower in self.REGIME_METADATA_COLS

            # Decision Logic
            should_exclude = (
                is_absolute_prefix or is_absolute_contains or is_regime_meta
            ) and not is_normalized_pattern

            # Double check for large values
            # [M4 FIX] Usa percentil 99.9 ao invés de max absoluto para ser robusto a outliers.
            # Antes: max_abs > 1000 excluía features com 1 outlier (ex: CCI spike) silenciosamente.
            # Agora: p99.9 > 1000 indica que a feature é SISTEMATICAMENTE fora de escala,
            # não apenas um outlier pontual. Também loga a exclusão para rastreabilidade.
            if not should_exclude and df is not None and col in df.columns:
                try:
                    p999 = df[col].dropna().abs().quantile(0.999)
                    if p999 > 1000 and not is_normalized_pattern:
                        should_exclude = True
                        logger.debug(
                            f"[FILTER] Excluindo '{col}': p99.9={p999:.1f} > 1000 (escala absoluta)"
                        )
                except Exception:
                    pass

            if should_exclude:
                excluded.append(col)
            else:
                filtered.append(col)

        # [FIX 3] Log explícito para proteção de hidden features
        hidden_excluded = [c for c in excluded if "hidden" in c.lower()]
        if hidden_excluded:
            logger.warning(
                f"⚠️ [SCIENTIFIC] Hidden features excluídas pelo filtro (Verificar Escala): {hidden_excluded}"
            )

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
        fit: bool = False,
        train_end_idx: Optional[int] = None,
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
            feature_columns = self.feature_columns or self.filter_autoencoder_features(
                all_numeric, df
            )

        df = df.copy()

        # Get regime-specific data
        if "regime" in df.columns:
            regime_mask = df["regime"] == regime_code
            n_regime = regime_mask.sum()
            logger.info(
                f"📊 Normalizing for {self.REGIME_NAMES[regime_code]}: {n_regime} samples"
            )
        else:
            logger.warning("⚠️ No 'regime' column - using global normalization")
            regime_mask = pd.Series(True, index=df.index)

        # Get or create scaler for this regime
        if regime_code not in self.scalers:
            self.scalers[regime_code] = RobustScaler()
        scaler = self.scalers[regime_code]
        # [FIX SDP-1] Se o scaler ainda não foi fitado (joblib vazio ou primeira execução),
        # fitar agora mesmo com fit=False para evitar crash no transform().
        # Não é leakage: scaler novo nunca viu esses dados.
        if not hasattr(scaler, "n_features_in_"):
            fit = True

        # Select numeric columns that exist
        numeric_cols = [
            c
            for c in feature_columns
            if c in df.columns and df[c].dtype in [np.float32, np.float64, np.int64]
        ]

        # [FIX] Jamais normalizar Hidden Features ou Latents (elas já vêm prontas do Autoencoder)
        # Se normalizarmos 2x, perdemos a escala original e geramos ruído.
        cols_to_normalize = [
            c
            for c in numeric_cols
            if not (c.startswith("hidden_") or c.startswith("sdae_latent"))
        ]
        logger.debug(
            f"🧪 [SCIENTIFIC] Normalizando {len(cols_to_normalize)} colunas técnicas. Preservando {len(numeric_cols) - len(cols_to_normalize)} colunas latentes."
        )

        # [SCIENTIFIC DIMENSION CACHE] Check for feature count mismatch
        if hasattr(scaler, "n_features_in_"):
            if scaler.n_features_in_ != len(cols_to_normalize):
                if fit:
                    logger.info(
                        f"🔄 [SDAE] Adaptando scaler de {scaler.n_features_in_} para {len(cols_to_normalize)} features "
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
            # 🔬 [HYBRID REGIME NORMALIZATION] — ANTI-FALSE-SIGNAL v2
            # Fitamos no GLOBAL para a escala (IQR) e no REGIME para o centro (Mediana).
            # [C2 FIX] NUNCA usar fillna(0) em features normalizadas (RSI=0=oversold extremo,
            # z-score=0=média perfeita, ATR=0=vol zero — sinais falsos que contaminam o scaler).
            # Estratégia: imputar com mediana do TREINO (causal, robusta) ANTES do fit.
            # RobustScaler usa quantis (25%/75%) — a mediana do treino é o impute mais neutro.
            # [PERSIST] Salvamos a mediana do treino para reuso em transform() de inferência.
            # [ANTI-LEAKAGE] Se train_end_idx fornecido, fitar APENAS nos dados de treino.
            # Se nao fornecido, usar 80% como default seguro.
            _effective_end = train_end_idx if train_end_idx is not None else int(len(df) * 0.80)
            fit_df = df.iloc[:_effective_end]
            train_subset = fit_df.loc[fit_df.index.isin(df.loc[regime_mask].index), cols_to_normalize] if regime_mask.any() else fit_df[cols_to_normalize]
            self._train_medians = train_subset.median(numeric_only=True)

            def _impute_train_median(frame: pd.DataFrame) -> np.ndarray:
                return frame.fillna(self._train_medians).fillna(0.0).values  # fallback p/ cols 100% NaN

            regime_data = _impute_train_median(fit_df.loc[fit_df.index.isin(df.loc[regime_mask].index), cols_to_normalize]) if regime_mask.any() else None

            if regime_data is not None and len(regime_data) > 0:
                # 1. Escala Global (IQR de toda a série de TREINO) — impute por mediana do treino
                temp_global = RobustScaler()
                temp_global.fit(_impute_train_median(fit_df[cols_to_normalize]))

                # 2. Centro de Regime (Mediana específica do regime) — mesmo impute
                scaler.fit(regime_data)

                # 3. Injeção: Mantém a mediana do regime, usa escala global
                if len(scaler.scale_) == len(temp_global.scale_):
                    scaler.scale_ = temp_global.scale_
                    logger.info(
                        f"✅ Scaler Regime-{regime_code} hibridizado (impute=mediana do treino)"
                    )
                else:
                    logger.error(
                        f"❌ [E1] Shape mismatch no hybrid scaler: "
                        f"regime={len(scaler.scale_)}, global={len(temp_global.scale_)}. "
                        "Mantendo escala do regime (sem hibridização)."
                    )
            else:
                logger.warning(
                    f"⚠️ Dados insuficientes para fit do regime {regime_code}. Usando fit global."
                )
                scaler.fit(_impute_train_median(fit_df[cols_to_normalize]))

            # Persiste o scaler recém-fitado para reutilização em calls subsequentes
            self.save_scalers()

        # 🔬 [TRANSFORM] Aplicar a normalização híbrida (ou global) APENAS às colunas técnicas
        # Impute por mediana persistida do treino (NÃO usar fillna(0) — sinal falso).
        try:
            medians = getattr(self, "_train_medians", None)
            if medians is None:
                # Sem fit prévio — usa mediana dos primeiros 80% (anti-leakage)
                _safe_end = int(len(df) * 0.80)
                medians = df.iloc[:_safe_end][cols_to_normalize].median(numeric_only=True)
            df[cols_to_normalize] = scaler.transform(
                df[cols_to_normalize].fillna(medians).fillna(0.0).values
            )
        except Exception as e:
            logger.warning(
                f"⚠️ [SCIENTIFIC] Transform do regime scaler falhou: {e}. "
                "Tentando fallback com global_scaler pré-fitado."
            )
            # [SCIENTIFIC FIX v3] fit_transform de emergência foi REMOVIDO.
            # Anteriormente mascarava bugs no pipeline com leakage silencioso em produção.
            # Agora: se global_scaler também falhar, falhamos alto (fail-fast).
            if hasattr(self.global_scaler, "n_features_in_"):
                try:
                    # [C2 FIX] Impute por mediana do treino persistida, não fillna(0)
                    fallback_medians = getattr(self, "_train_medians", df[cols_to_normalize].median(numeric_only=True))
                    df[cols_to_normalize] = self.global_scaler.transform(
                        df[cols_to_normalize].fillna(fallback_medians).fillna(0.0).values
                    )
                except Exception as e2:
                    raise RuntimeError(
                        f"[R1] Ambos regime scaler e global_scaler falharam no transform. "
                        f"Regime: {e}. Global: {e2}. "
                        "Pipeline quebrado — NÃO aplicar fit_transform de emergência (= leakage). "
                        "Reinvestigar o feature pipeline antes de continuar."
                    ) from e2
            else:
                raise RuntimeError(
                    f"[R1] global_scaler não fitado e regime scaler falhou: {e}. "
                    "Pipeline quebrado — NÃO aplicar fit_transform de emergência (= leakage). "
                    "Execute prepare_for_autoencoder() primeiro para fitar o global_scaler."
                ) from e

        self.feature_columns = numeric_cols
        return df

    def prepare_for_specialist(
        self,
        df: pd.DataFrame,
        regime_code: int,
        filter_by_regime: bool = True,
        normalize: bool = True,
        min_samples: int = 100,
        fit_scaler: bool = False,  # Só fit em train; eval/inferência deve usar scaler pré-carregado
        train_end_idx: Optional[int] = None,
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
        logger.info(
            f"🎯 Preparing data for {self.REGIME_NAMES[regime_code]} specialist..."
        )

        # Validate regime column exists
        if "regime" not in df.columns:
            logger.error(
                "❌ DataFrame must have 'regime' column. Run add_regime_labels() first."
            )
            return None

        # [REMOVIDO BUG 11 FIX] Não desativamos mais a normalização global.
        # Agora o normalize_for_specialist é inteligente o suficiente para pular APENAS as latents.

        if normalize:
            df = self.normalize_for_specialist(
                df, regime_code=regime_code, fit=fit_scaler, train_end_idx=train_end_idx
            )

        # [Garantir que df tenha regimes após normalização]
        if filter_by_regime:
            mask = df["regime"] == regime_code
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

        # [C3 FIX] Anti-false-signal: NaN residuais sao SINAL DE BUG upstream.
        # Antes: fillna(0) silencioso mascarava o bug e injetava sinal falso (RSI=0=oversold).
        # Agora: imputa por mediana das colunas afetadas (causal, robusta) e LOGA por coluna.
        nan_count = df_prepared.isna().sum().sum()
        if nan_count > 0:
            nan_per_col = df_prepared.isna().sum()
            problematic = nan_per_col[nan_per_col > 0]
            pct = nan_count / max(1, df_prepared.size) * 100
            logger.warning(
                f"⚠️ [SDP] {nan_count} NaN ({pct:.2f}% das celulas) em "
                f"{len(problematic)} colunas. Imputando por MEDIANA (nao zero). "
                f"Top NaN cols: {problematic.nlargest(5).to_dict()}"
            )
            col_medians = df_prepared.median(numeric_only=True)
            df_prepared = df_prepared.fillna(col_medians).fillna(0.0)

        logger.info(f"✅ Data prepared: {df_prepared.shape}")
        return df_prepared

    def prepare_for_autoencoder(
        self, df: pd.DataFrame, feature_columns: List[str], train_split: float = 0.80
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
            # [C4 FIX] Persiste mediana do TREINO para imputar NaN (não fillna(0)).
            if split_idx < 10:
                logger.warning(
                    f"⚠️ Dataset muito pequeno ({len(df_filtered)} linhas) para split. "
                    "Usando fit global (leakage mínimo aceitável)."
                )
                self._train_medians = df_filtered[numeric_cols].median(numeric_only=True)
                self.global_scaler.fit(
                    df_filtered[numeric_cols].fillna(self._train_medians).fillna(0.0).values
                )
            else:
                train_df = df_filtered.iloc[:split_idx][numeric_cols]
                self._train_medians = train_df.median(numeric_only=True)  # causal
                train_data = train_df.fillna(self._train_medians).fillna(0.0).values
                self.global_scaler.fit(train_data)
                logger.info(
                    f"✅ [ANTI-LEAKAGE] global_scaler fit em {split_idx} amostras "
                    f"(treino {train_split:.0%}) → transform em {len(df_filtered)} total."
                )
            # Transform usa mediana do TREINO para imputar (não vaza estatística de val)
            df_filtered[numeric_cols] = self.global_scaler.transform(
                df_filtered[numeric_cols].fillna(self._train_medians).fillna(0.0).values
            )

        logger.info(f"✅ Autoencoder data prepared: {df_filtered.shape}")
        return df_filtered, existing_cols

    def save_scalers(self) -> bool:
        """Save all scalers + train medians (anti-false-signal imputation)."""
        try:
            state = {
                "regime_scalers": self.scalers,
                "global_scaler": self.global_scaler,
                "feature_columns": self.feature_columns,
                "autoencoder_columns": self.autoencoder_columns,
                "regime_stats": self.regime_stats,
                "train_medians": getattr(self, "_train_medians", None),  # C2-C4 FIX
            }
            joblib.dump(state, self.scaler_path)
            logger.info(f"✅ Scalers saved to {self.scaler_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save scalers: {e}")
            return False

    def load_scalers(self) -> bool:
        """Load scalers + train medians from disk."""
        if not os.path.exists(self.scaler_path):
            return False

        try:
            state = joblib.load(self.scaler_path)
            self.scalers = state.get("regime_scalers", {})
            self.global_scaler = state.get("global_scaler", RobustScaler())
            self.feature_columns = state.get("feature_columns", [])
            self.autoencoder_columns = state.get("autoencoder_columns", [])
            self.regime_stats = state.get("regime_stats", {})
            # [C2-C4 FIX] Restaura medianas do treino para imputação consistente
            train_medians = state.get("train_medians", None)
            if train_medians is not None:
                self._train_medians = train_medians
            logger.info(f"✅ Scalers loaded from {self.scaler_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load scalers: {e}")
            return False


# Convenience function for quick use
def create_data_processor(model_dir: str = "models_ai") -> ScientificDataProcessor:
    """Factory function to create or load data processor."""
    processor = ScientificDataProcessor(model_dir)
    processor.load_scalers()
    return processor
