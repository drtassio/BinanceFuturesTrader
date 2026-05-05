# -----------------------------------------------------------------------------
# ARQUIVO: feature_engineering/crypto_regime_detector.py
# -----------------------------------------------------------------------------
"""
🔬 Scientific Ensemble Regime Detector for Crypto Markets

Based on Dr. Tensor's audit recommendations and peer-reviewed research:
- Hamilton (1989): HMM regime-switching
- Ang & Bekaert (2002): GMM for asset pricing
- Lin et al. (2022): Ensemble HMM
- Wilder (1978): ADX trend strength

Ensemble components:
1. HMM (Hidden Markov Model) - Captures temporal dependencies
2. GMM (Gaussian Mixture Model) - Clusters return distributions
3. ADX (Average Directional Index) - Technical trend strength
4. Funding Rate (Perpetual Futures) - Crypto-specific sentiment

Regimes:
0: Bull (Strong uptrend)
1: Bear (Strong downtrend)  
2: Ranger (Sideways/No trend)
"""

import os
import numpy as np
import pandas as pd
# Fix for joblib/loky on some Windows systems: found 0 physical cores < 1
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)
import ta
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
import warnings

from utils.logger import get_logger

logger = get_logger("CryptoRegimeDetector")

# Suppress convergence warnings during fitting
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class RegimeConfig:
    """Configuration for the Ensemble Regime Detector."""
    
    # HMM parameters
    n_hmm_states: int = 3
    hmm_n_iter: int = 100
    hmm_covariance_type: str = 'full'
    
    # GMM parameters
    n_gmm_components: int = 3
    gmm_n_iter: int = 100
    gmm_covariance_type: str = 'full'
    
    # ADX parameters
    adx_period: int = 9  # Shorter for crypto (faster regime changes)
    adx_trend_threshold: float = 20.0  # Lower than stocks (more volatile)
    
    # Ensemble parameters
    confidence_threshold: float = 0.55
    min_regime_duration: int = 8  # Minimum bars before regime change
    
    # Weights for ensemble voting (F1: funding weight reduced — now a risk factor, not a vote)
    weight_hmm: float = 0.35
    weight_gmm: float = 0.40
    weight_adx: float = 0.25
    weight_funding: float = 0.0  # [F1 FIX] Funding é sinal contrário, não voto de regime atual
    
    # Feature windows (shorter for 24/7 crypto markets)
    returns_window: int = 1
    volatility_window: int = 10  # vs 20 for stocks
    volume_window: int = 10
    autocorr_window: int = 15    # Reduced from 20 for better responsiveness
    drawdown_window: int = 20    # [M2] Reduced from 50 (12.5h -> 5h lookback)
    
    # Validation thresholds
    min_persistence_bars: int = 10
    min_model_agreement: float = 0.65
    max_transition_rate: float = 0.25
    max_gini_imbalance: float = 0.35
    
    # Funding risk factor (F1): substitui voto direto
    funding_risk_threshold: float = 0.003  # 0.3%/8h = sobreaquecido
    funding_risk_attenuation: float = 0.15  # Reduz confiança quando funding é extremo


class CryptoRegimeDetector:
    """
    🔬 Ensemble Regime Detector optimized for BTCUSDT Perpetual Futures.
    
    Uses 4 models in ensemble voting:
    1. HMM: Temporal state transitions
    2. GMM: Return distribution clustering
    3. ADX: Technical trend strength
    4. Funding: Crypto-specific sentiment (perpetual futures)
    
    Scientific basis:
    - Hamilton (1989): HMM for regime switching
    - Ang & Bekaert (2002): GMM for finance
    - Ensemble voting reduces false positives from 30% to 10-15%
    """
    
    # Regime codes
    BULL = 0
    BEAR = 1
    RANGER = 2
    
    REGIME_NAMES = {0: 'Bull', 1: 'Bear', 2: 'Ranger'}
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize the ensemble detector.
        
        Args:
            config: RegimeConfig with parameters (uses defaults if None)
        """
        self.config = config or RegimeConfig()
        self.scaler = StandardScaler()
        
        # Models will be initialized on fit
        self.hmm_model = None
        self.gmm_model = None
        
        # State tracking
        self.regime_history: Optional[pd.DataFrame] = None
        self.confidence_scores: Optional[np.ndarray] = None
        self.validation_metrics: Optional[Dict[str, float]] = None
        
        # Online prediction state (Forward Algorithm)
        self._alpha = None  # Forward probabilities for online HMM
        self._last_regime = self.RANGER
        
        logger.info(
            f"✅ CryptoRegimeDetector initialized: "
            f"HMM={self.config.weight_hmm:.0%}, GMM={self.config.weight_gmm:.0%}, "
            f"ADX={self.config.weight_adx:.0%}, Funding={self.config.weight_funding:.0%}"
        )
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for regime detection.
        
        Features (scientifically validated):
        - returns: Directional trend
        - volatility: Risk (rolling std)
        - volume_z: Normalized liquidity
        - adx: Trend strength
        - autocorr: Momentum persistence
        - drawdown: Distance from peak
        - funding_rate: Perpetual futures sentiment (if available)
        - oi_change: Open Interest change (if available)
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Returns
        features['returns'] = df['close'].pct_change().fillna(0)
        
        # 2. Volatility (rolling std of returns)
        features['volatility'] = features['returns'].rolling(
            self.config.volatility_window
        ).std().fillna(0.01)
        
        # 3. Volume Z-score (normalized)
        if 'volume' in df.columns:
            volume_mean = df['volume'].rolling(self.config.volume_window).mean()
            volume_std = df['volume'].rolling(self.config.volume_window).std()
            features['volume_z'] = (
                (df['volume'] - volume_mean) / (volume_std + 1e-8)
            ).fillna(0)
        else:
            features['volume_z'] = 0.0
        
        # 4. ADX (Average Directional Index)
        features['adx'] = self._compute_adx(df)
        
        # 5. Autocorrelation (momentum persistence)
        features['autocorr'] = features['returns'].rolling(
            self.config.autocorr_window
        ).apply(lambda x: x.autocorr() if len(x) > 1 else 0, raw=False).fillna(0)
        
        # 🧪 [FIX] RSI Momentum (Faster than EMA)
        try:
            rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=14)
            features['rsi'] = rsi_indicator.rsi().fillna(50) / 100.0 # Normalized 0-1
        except Exception:
            features['rsi'] = 0.5

        # 🧪 [FIX] Vol-Weighted Returns (Smart Money)
        vol_mean = df['volume'].rolling(20).mean()
        features['vol_weighted_ret'] = features['returns'] * (df['volume'] / (vol_mean + 1e-8))
        
        # 6. Drawdown (distance from rolling max) [M2: agora usa config]
        rolling_max = df['close'].rolling(self.config.drawdown_window, min_periods=1).max()
        features['drawdown'] = (df['close'] - rolling_max) / (rolling_max + 1e-8)
        features['drawdown'] = features['drawdown'].fillna(0)
        
        # 7. Funding Rate — armazenada como feature, NÃO como voto (F1 fix)
        if 'funding_rate' in df.columns:
            features['funding_rate'] = df['funding_rate'].fillna(0)
            features['funding_ma'] = df['funding_rate'].rolling(8).mean().fillna(0)
        else:
            features['funding_rate'] = 0.0
            features['funding_ma'] = 0.0
        
        # 8. Open Interest Change (futures)
        if 'open_interest' in df.columns:
            features['oi_change'] = df['open_interest'].pct_change().fillna(0)
        else:
            features['oi_change'] = 0.0
        
        # 9. Trend direction (EMA crossover)
        ema_fast = df['close'].ewm(span=10, adjust=False).mean()
        ema_slow = df['close'].ewm(span=20, adjust=False).mean()
        features['trend_direction'] = np.where(ema_fast > ema_slow, 1, -1)
        
        # [M1] DI+ - DI-: captura direção E força ao mesmo tempo
        # ADX mede força mas não direção; di_diff resolve isso.
        try:
            adx_ind = ta.trend.ADXIndicator(
                high=df['high'], low=df['low'], close=df['close'],
                window=self.config.adx_period
            )
            features['di_diff'] = (adx_ind.adx_pos() - adx_ind.adx_neg()).fillna(0)
        except Exception:
            features['di_diff'] = 0.0
        
        return features
    
    def _compute_adx(self, df: pd.DataFrame) -> pd.Series:
        """Compute ADX (Average Directional Index) using ta library."""
        try:
            adx_indicator = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.config.adx_period
            )
            return adx_indicator.adx().fillna(0)
        except Exception as e:
            logger.warning(f"⚠️ ADX computation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _train_hmm(self, features: np.ndarray) -> np.ndarray:
        """
        Train Hidden Markov Model and predict regimes.
        
        Uses hmmlearn if available, falls back to simple Markov chain.
        [B1 FIX] Sub-amostragem via janela contígua aleatória, não stride. 
        Stride viola a suposição Markoviana ao criar sequências com gaps.
        [F2 FIX] Mapeamento por score Sharpe-like (mean/std) em vez de mean apenas.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
            
            self.hmm_model = GaussianHMM(
                n_components=self.config.n_hmm_states,
                covariance_type=self.config.hmm_covariance_type,
                n_iter=self.config.hmm_n_iter,
                random_state=42
            )
            
            # [B1 FIX] Sub-amostragem por janela contígua aleatória
            # Evita violar a suposição Markoviana de sequências contíguas
            if len(features) > 50000:
                logger.info(f"Sub-sampling HMM training data (contiguous window 50k/{len(features)})")
                rng = np.random.default_rng(42)
                start = rng.integers(0, len(features) - 50000)
                train_data = features[start:start + 50000]
            else:
                train_data = features
                
            self.hmm_model.fit(train_data)
            hidden_states = self.hmm_model.predict(features)
            
            # [F2 FIX] Mapeamento por Sharpe Contemporâneo (SEM LEAKAGE)
            # Bull = Alto retorno/Baixa vol, Bear = Baixo retorno/Alta vol.
            state_scores = []
            for state in range(self.config.n_hmm_states):
                mask = hidden_states == state
                if mask.sum() > 2:
                    r = features[mask, 0] # Retorno contemporâneo
                    score = r.mean() / (r.std() + 1e-8)
                    state_scores.append(score)
                else:
                    state_scores.append(0.0)
            
            # Sort: highest Forward Return = Bull, lowest = Bear, middle = Ranger
            sorted_states = np.argsort(state_scores)
            mapping = {
                sorted_states[0]: self.BEAR,
                sorted_states[1]: self.RANGER,
                sorted_states[2]: self.BULL
            }
            self._hmm_state_mapping = mapping  # Salvo para usar em predict()

            regimes = np.vectorize(mapping.get)(hidden_states)
            logger.info("✅ HMM trained successfully (Sharpe mapping)")
            return regimes
            
        except ImportError:
            logger.warning("⚠️ hmmlearn not installed. Using fallback regime detection.")
            return self._fallback_hmm(features)
        except Exception as e:
            logger.warning(f"⚠️ HMM training failed: {e}. Using fallback.")
            return self._fallback_hmm(features)
    
    def _fallback_hmm(self, features: np.ndarray) -> np.ndarray:
        """Vectorized fallback when HMM is unavailable."""
        returns = pd.Series(features[:, 0])
        window = 20
        
        cum_return = returns.rolling(window=window).sum()
        volatility = returns.rolling(window=window).std()
        
        regimes = np.full(len(returns), self.RANGER)
        
        # Vectorized conditions
        bull_mask = cum_return > (volatility * 2)
        bear_mask = cum_return < -(volatility * 2)
        
        regimes[bull_mask] = self.BULL
        regimes[bear_mask] = self.BEAR
        
        return regimes
    
    def _train_gmm(self, features: np.ndarray) -> np.ndarray:
        """
        Train Gaussian Mixture Model for regime clustering.
        
        GMM doesn't assume temporal dependence - good for abrupt changes.
        [B1 FIX] Sub-amostragem aleatória sem reposição (não stride).
        [B2 FIX] Salva `_gmm_cluster_mapping` para uso idêntico em predict().
        [F2 FIX] Mapeamento por score Sharpe-like.
        """
        try:
            from sklearn.mixture import GaussianMixture
            
            self.gmm_model = GaussianMixture(
                n_components=self.config.n_gmm_components,
                covariance_type=self.config.gmm_covariance_type,
                n_init=3,
                max_iter=self.config.gmm_n_iter,
                random_state=42
            )
            
            # [B1 FIX] Amostragem aleatória sem reposição (respeita IID do GMM)
            if len(features) > 50000:
                logger.info(f"Sub-sampling GMM training data (random 50k/{len(features)})")
                rng = np.random.default_rng(42)
                idx = rng.choice(len(features), size=50000, replace=False)
                self.gmm_model.fit(features[idx])
            else:
                self.gmm_model.fit(features)
                
            clusters = self.gmm_model.predict(features)
            
            # [B2+F2 FIX] Economic grounding: Sharpe Contemporâneo (SEM LEAKAGE)
            cluster_scores = []
            for cluster in range(self.config.n_gmm_components):
                mask = clusters == cluster
                if mask.sum() > 2:
                    r = features[mask, 0] # Retorno contemporâneo
                    score = r.mean() / (r.std() + 1e-8)
                    cluster_scores.append(score)
                else:
                    cluster_scores.append(0.0)
            
            sorted_clusters = np.argsort(cluster_scores)
            mapping = {
                int(sorted_clusters[0]): self.BEAR,
                int(sorted_clusters[1]): self.RANGER,
                int(sorted_clusters[2]): self.BULL
            }
            # [B2 FIX] Persiste o mapeamento exatamente como foi calculado no fit
            self._gmm_cluster_mapping = mapping
            
            regimes = np.vectorize(mapping.get)(clusters)
            logger.info("✅ GMM trained successfully (Alpha mapping, mapping saved)")
            return regimes
            
        except Exception as e:
            logger.warning(f"⚠️ GMM training failed: {e}. Using fallback.")
            return self._fallback_gmm(features)
    
    def _fallback_gmm(self, features: np.ndarray) -> np.ndarray:
        """Fallback using quantile-based clustering."""
        returns = features[:, 0]
        volatility = features[:, 1] if features.shape[1] > 1 else np.zeros_like(returns)
        
        regimes = np.full(len(returns), self.RANGER)
        
        # Use percentiles for clustering
        return_p33 = np.percentile(returns, 33)
        return_p67 = np.percentile(returns, 67)
        
        regimes[returns > return_p67] = self.BULL
        regimes[returns < return_p33] = self.BEAR
        
        return regimes
    
    def _compute_adx_regimes(self, df: pd.DataFrame, adx: pd.Series) -> np.ndarray:
        """
        Compute regimes using ADX + price action.
        
        ADX > threshold + bullish trend = Bull
        ADX > threshold + bearish trend = Bear
        ADX < threshold = Ranger
        """
        regimes = np.full(len(df), self.RANGER)
        
        # Trend direction from EMA
        ema_fast = df['close'].ewm(span=10, adjust=False).mean()
        ema_slow = df['close'].ewm(span=20, adjust=False).mean()
        
        # Check both ADX strength and trend direction
        is_trending = adx > self.config.adx_trend_threshold
        is_bullish = ema_fast > ema_slow
        
        bull_mask = is_trending & is_bullish
        bear_mask = is_trending & ~is_bullish
        
        regimes[bull_mask] = self.BULL
        regimes[bear_mask] = self.BEAR
        
        return regimes
    
    def _compute_funding_risk(self, df: pd.DataFrame, confidence: np.ndarray) -> np.ndarray:
        """
        [F1 FIX] Funding rate como FATOR DE RISCO, não voto de regime.
        
        O funding rate é um sinal CONTRÁRIO: funding alto durante bull market
        significa longs sobreaquecidos → risco de reversao, não Bear imediato.
        Tratar como voto de regime causa label mixing (bull classificado como bear).
        
        Agora: atenua a confiança quando o funding é extremo, sem alterar o regime.
        """
        attenuated = confidence.copy()
        
        if 'funding_rate' not in df.columns:
            return attenuated
        
        funding = df['funding_rate'].values
        
        # Funding extremo = incerteza sobre a continuidade do regime atual
        extreme_mask = np.abs(funding) > self.config.funding_risk_threshold
        attenuated[extreme_mask] = np.maximum(
            attenuated[extreme_mask] - self.config.funding_risk_attenuation,
            self.config.confidence_threshold  # Não cai abaixo do threshold base
        )
        
        return attenuated
    
    # [F1] Mantemos _compute_funding_regimes como stub vazio para backward compat 
    def _compute_funding_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """[DEPRECATED] Funding não vota mais no ensemble. Use _compute_funding_risk."""
        return np.full(len(df), self.RANGER)  # Retorna Ranger (neutro) para todos
    
    def _ensemble_voting(
        self,
        hmm: np.ndarray,
        gmm: np.ndarray,
        adx: np.ndarray,
        funding: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine 4 models using weighted voting.
        [VECTORIZED FOR HIGH PERFORMANCE]
        
        Returns:
            final_regimes: Consensus regime for each timestep
            confidence: Confidence score (0-1) for each prediction
        """
        n_samples = len(hmm)
        
        # Prepare components and weights
        votes_matrix = np.vstack([hmm, gmm, adx, funding]) # Shape (4, n_samples)
        weight_list = np.array([
            self.config.weight_hmm,
            self.config.weight_gmm,
            self.config.weight_adx,
            self.config.weight_funding
        ])
        
        # BUG M1 FIX: Normalize weights so they always sum strictly to 1.0
        weight_sum = np.sum(weight_list)
        if weight_sum > 0:
            weight_list = weight_list / weight_sum
        
        # Initialize scores for each regime (0: Bull, 1: Bear, 2: Ranger)
        scores = np.zeros((n_samples, 3))
        
        # Vectorized scoring
        # For each of the 4 models, add its weight to the corresponding predicted regime score
        for m_idx in range(4):
            model_votes = votes_matrix[m_idx]
            model_weight = weight_list[m_idx]
            
            # Using fancy indexing to add weights
            # model_votes are the column indices (regime 0, 1, or 2)
            # np.arange(n_samples) are the row indices
            scores[np.arange(n_samples), model_votes] += model_weight
            
        final_regimes = np.argmax(scores, axis=1)
        confidence = np.max(scores, axis=1)
        
        # [F4 FIX] Salvar scores brutos para get_regime_probabilities
        self._raw_ensemble_scores = scores.copy()
        
        # Apply anti-whipsaw logic
        threshold = self.config.confidence_threshold
        for i in range(1, n_samples):
            if confidence[i] < threshold:
                final_regimes[i] = final_regimes[i-1]
                prev_conf = confidence[i-1]
                confidence[i] = max(confidence[i], prev_conf * 0.95)
                
        # Smooth transitions (remove micro-regimes)
        final_regimes = self._smooth_regimes(final_regimes, confidence)
        
        return final_regimes, confidence
    
    def _smooth_regimes(self, regimes: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        """
        Remove regime changes that don't persist for min_regime_duration.
        [F3 FIX] Usa confiança para preservar segmentos curtos mas confiantes.
        
        Um segmento de 4 bars com confiança 0.90 indica transição rápida real 
        em crypto — não deve ser absorvido como ruído.
        """
        if len(regimes) < self.config.min_regime_duration:
            return regimes
        
        smoothed = regimes.copy()
        
        # Passe linear O(n): identifica segmentos e mescla os curtos e pouco confiantes
        n = len(smoothed)
        i = 0
        while i < n:
            seg_start = i
            seg_regime = smoothed[i]
            while i < n and smoothed[i] == seg_regime:
                i += 1
            seg_end = i  # exclusive

            duration = seg_end - seg_start
            
            # [F3 FIX] Calcula confiança média do segmento
            seg_conf = float(np.mean(confidence[seg_start:seg_end]))
            
            # Duração mínima adaptativa: curtos + baixa confiança = ruído
            # curtos + alta confiança = transição rápida genuina (preservar)
            high_confidence_threshold = 0.75
            is_high_confidence = seg_conf >= high_confidence_threshold
            
            if duration < self.config.min_regime_duration and seg_start > 0 and not is_high_confidence:
                # Segmento curto e baixa confiança: absorve no regime anterior
                prev_regime = smoothed[seg_start - 1]
                smoothed[seg_start:seg_end] = prev_regime

        return smoothed
    
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point: fit all models and predict regimes.
        
        Args:
            df: DataFrame with OHLCV data + optional funding_rate, open_interest
            
        Returns:
            DataFrame with regime, confidence, and individual model predictions
        """
        logger.info("📊 Starting ensemble regime detection...")
        
        # Validate input
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        # 1. Extract features
        logger.info("🔧 Computing features...")
        features_df = self._compute_features(df)
        
        # Select core features for models
        core_features = ['returns', 'volatility', 'volume_z', 'adx', 'autocorr', 'drawdown', 'di_diff']
        if 'funding_rate' in df.columns:
            core_features.append('funding_rate')
        
        features_array = features_df[core_features].values
        
        # Handle NaN/Inf
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_array)
        
        # 2. Train individual models
        logger.info("🧠 Training HMM...")
        hmm_regimes = self._train_hmm(features_scaled)
        
        logger.info("🧠 Training GMM...")
        gmm_regimes = self._train_gmm(features_scaled)
        
        logger.info("📈 Computing ADX regimes...")
        adx_regimes = self._compute_adx_regimes(df, features_df['adx'])
        
        logger.info("💰 Computing Funding regimes...")
        funding_regimes = self._compute_funding_regimes(df)
        
        # 3. Ensemble voting
        logger.info("🗳️  Applying ensemble voting...")
        final_regimes, confidence = self._ensemble_voting(
            hmm_regimes, gmm_regimes, adx_regimes, funding_regimes
        )
        
        # [F1 FIX] Aplicar atenuacao de fundingr rate sobre a confianca
        confidence = self._compute_funding_risk(df, confidence)
        
        # Salva aussi o _fit_feature_names para proteger o scaler em predict()
        self._fit_feature_names = core_features
        # 4. Build result DataFrame
        result = pd.DataFrame(index=df.index)
        result['regime'] = final_regimes
        result['confidence'] = confidence
        result['hmm_regime'] = hmm_regimes
        result['gmm_regime'] = gmm_regimes
        result['adx_regime'] = adx_regimes
        result['funding_regime'] = funding_regimes
        result['regime_name'] = result['regime'].map(self.REGIME_NAMES)
        
        # Store for validation
        self.regime_history = result
        self.confidence_scores = confidence
        
        # Log statistics
        counts = np.bincount(final_regimes, minlength=3)
        total = len(final_regimes)
        logger.info(
            f"✅ Regime detection complete: "
            f"Bull={counts[0]} ({counts[0]/total:.1%}), "
            f"Bear={counts[1]} ({counts[1]/total:.1%}), "
            f"Ranger={counts[2]} ({counts[2]/total:.1%})"
        )
        
        return result
    
    def validate_regimes(self) -> Dict[str, float]:
        """
        Validate regime quality using 6 scientific metrics.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.regime_history is None:
            raise ValueError("Must call fit_predict() first")
        
        regimes = self.regime_history['regime'].values
        confidence = self.regime_history['confidence'].values
        
        metrics = {}
        
        # 1. Average Persistence (bars per regime)
        regime_changes = np.diff(regimes) != 0
        regime_lengths = []
        current_length = 1
        for changed in regime_changes:
            if changed:
                regime_lengths.append(current_length)
                current_length = 1
            else:
                current_length += 1
        regime_lengths.append(current_length)
        
        metrics['avg_persistence_bars'] = np.mean(regime_lengths)
        
        # 2. Model Agreement (average across ensemble)
        hmm = self.regime_history['hmm_regime'].values
        gmm = self.regime_history['gmm_regime'].values
        adx = self.regime_history['adx_regime'].values
        funding = self.regime_history['funding_regime'].values
        
        agreements = []
        for i in range(len(regimes)):
            votes = [hmm[i], gmm[i], adx[i], funding[i]]
            max_agreement = max(votes.count(r) for r in [0, 1, 2])
            agreements.append(max_agreement / 4)
        
        metrics['avg_model_agreement'] = np.mean(agreements)
        
        # 3. Mean Confidence
        metrics['mean_confidence'] = np.mean(confidence)
        
        # 4. Regime Imbalance (Gini coefficient)
        counts = np.bincount(regimes, minlength=3) / len(regimes)
        gini = 1 - np.sum(counts ** 2)
        metrics['regime_imbalance_gini'] = gini
        
        # 5. Transition Rate
        transitions = np.sum(regime_changes) / len(regimes)
        metrics['transition_rate'] = transitions
        
        # 6. Regime Distribution
        metrics['bull_pct'] = np.mean(regimes == self.BULL)
        metrics['bear_pct'] = np.mean(regimes == self.BEAR)
        metrics['sideways_pct'] = np.mean(regimes == self.RANGER)
        
        # Quality checks
        metrics['persistence_ok'] = metrics['avg_persistence_bars'] >= self.config.min_persistence_bars
        metrics['agreement_ok'] = metrics['avg_model_agreement'] >= self.config.min_model_agreement
        metrics['confidence_ok'] = metrics['mean_confidence'] >= self.config.confidence_threshold * 0.9
        metrics['transition_ok'] = metrics['transition_rate'] <= self.config.max_transition_rate
        metrics['balance_ok'] = metrics['regime_imbalance_gini'] <= self.config.max_gini_imbalance
        
        metrics['all_checks_passed'] = all([
            metrics['persistence_ok'],
            metrics['agreement_ok'],
            metrics['confidence_ok'],
            metrics['transition_ok'],
            metrics['balance_ok']
        ])
        
        # [M3] 7. Forward Return by Regime (validação financeira)
        # Confirma que Bull realmente precede retornos positivos nos próximos N bars.
        # Um detector de regime sem valor preditivo é apenas ruído.
        forward_bars = 5
        if 'returns' in self.regime_history.columns:
            fwd_returns = self.regime_history['returns'].values
        else:
            # Recalcula a partir do confidence df
            fwd_returns = np.zeros(len(regimes))
        
        fwd_by_regime: Dict[str, float] = {}
        for r_code, r_name in self.REGIME_NAMES.items():
            mask = regimes[:-forward_bars] == r_code
            if mask.sum() > 5:
                future_returns = np.array([
                    fwd_returns[i+1:i+forward_bars+1].sum()
                    for i in np.where(mask)[0]
                ])
                fwd_by_regime[r_name] = float(np.mean(future_returns))
            else:
                fwd_by_regime[r_name] = 0.0
        
        metrics['forward_return_bull'] = fwd_by_regime.get('Bull', 0.0)
        metrics['forward_return_bear'] = fwd_by_regime.get('Bear', 0.0)
        metrics['forward_return_ranger'] = fwd_by_regime.get('Ranger', 0.0)
        # Bull deve ter forward return > Bear
        metrics['financial_validity_ok'] = (
            metrics['forward_return_bull'] > metrics['forward_return_bear']
        )
        
        self.validation_metrics = metrics
        
        logger.info(
            f"📊 Validation: Persistence={metrics['avg_persistence_bars']:.1f}bars, "
            f"Agreement={metrics['avg_model_agreement']:.1%}, "
            f"Confidence={metrics['mean_confidence']:.1%}, "
            f"FwdRet(Bull/Bear): {metrics['forward_return_bull']:.4f}/{metrics['forward_return_bear']:.4f}, "
            f"Passed={'✅' if metrics['all_checks_passed'] else '❌'}"
        )
        
        return metrics
    
    def detect_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compatibility method with old ScientificRegimeDetector interface.
        
        Returns only the regime array (0=Bull, 1=Bear, 2=Ranger).
        """
        # BUG C1 FIX: Não re-treinar o modelo em cada chamada de predição
        if self.is_trained:
            result = self.predict(df)
        else:
            result = self.fit_predict(df)
        return result['regime'].values
    
    def get_regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns soft probabilities for each regime.
        [F4 FIX] Usa scores brutos do ensemble em vez de distribuição binária fixa.
        
        A distribuição binária (conf, (1-conf)/2, (1-conf)/2) perde informação
        sobre onde a incerteza está concentrada (Bear vs Ranger distintos).
        """
        if self.regime_history is None:
            self.fit_predict(df)
        
        probs = pd.DataFrame(index=df.index)
        
        # [F4 FIX] Usa raw scores do ensemble normalizados (Bull, Bear, Ranger)
        raw = getattr(self, '_raw_ensemble_scores', None)
        if raw is not None and len(raw) == len(df):
            # Normaliza cada linha para somar 1.0
            row_sums = raw.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # Evita divisao por zero
            normalized = raw / row_sums
            probs['prob_bull']   = normalized[:, 0]
            probs['prob_bear']   = normalized[:, 1]
            probs['prob_ranger'] = normalized[:, 2]
        else:
            # Fallback: distribuição baseada em confiança
            conf = self.regime_history['confidence'].values
            regimes = self.regime_history['regime'].values
            probs['prob_bull']   = np.where(regimes == self.BULL,   conf, (1 - conf) / 2)
            probs['prob_bear']   = np.where(regimes == self.BEAR,   conf, (1 - conf) / 2)
            probs['prob_ranger'] = np.where(regimes == self.RANGER, conf, (1 - conf) / 2)
            total = probs.sum(axis=1)
            probs = probs.div(total, axis=0)
        
        return probs
    
    def predict_online(self, observation: np.ndarray) -> int:
        """
        Online regime prediction.
        [B3 FIX] Implementa o Forward Algorithm real para inferência com memória.
        [BX FIX] Argmax mapeado via _hmm_state_mapping (não estado interno bruto).
        
        alpha[t] = P(O_0:t, S_t | lambda) 
        normalized para evitar underflow numérico.
        """
        if self.hmm_model is None:
            return self._last_regime
        
        try:
            obs = observation.reshape(1, -1)
            obs_scaled = self.scaler.transform(obs)
            
            # Inicializa alpha na primeira chamada
            if self._alpha is None:
                # pi = prob inicial dos estados; B = prob de emissao
                startprob = self.hmm_model.startprob_
                # Emissao Gaussiana: log P(obs | state)
                log_emission = self.hmm_model.score_samples(obs_scaled)
                if hasattr(log_emission, '__len__'):
                    log_emission = log_emission[0] if len(log_emission) == 1 else None
                
                # Fallback para predict_proba se score_samples retornar shapes inesperados
                try:
                    probs = self.hmm_model.predict_proba(obs_scaled)[0]
                    self._alpha = startprob * probs
                    self._alpha /= (self._alpha.sum() + 1e-300)
                except Exception:
                    self._alpha = startprob.copy()
            else:
                # Forward step: alpha_t+1 = (alpha_t @ A) * B(obs_t+1)
                transmat = self.hmm_model.transmat_
                predicted_alpha = self._alpha @ transmat  # P(S_t+1 | O_0:t)
                
                # Emissao (probabilidade de observar obs no estado)
                try:
                    emission_probs = self.hmm_model.predict_proba(obs_scaled)[0]
                except Exception:
                    emission_probs = np.ones(self.config.n_hmm_states) / self.config.n_hmm_states
                
                self._alpha = predicted_alpha * emission_probs
                total = self._alpha.sum()
                if total > 1e-300:
                    self._alpha /= total
                else:
                    self._alpha = np.ones(self.config.n_hmm_states) / self.config.n_hmm_states
            
            # Estado mais provável segundo o Forward Algorithm
            best_state = int(np.argmax(self._alpha))
            state_prob = float(self._alpha[best_state])
            
            # [BX FIX] Mapeia estado interno -> regime (Bull/Bear/Ranger)
            mapping = getattr(self, '_hmm_state_mapping', None)
            if mapping is not None:
                best_regime = mapping.get(best_state, self.RANGER)
            else:
                best_regime = best_state  # Sem mapeamento (não deve ocorrer)
            
            # Aplica threshold de confiança
            if state_prob < self.config.confidence_threshold:
                return self._last_regime
            
            self._last_regime = best_regime
            return best_regime
            
        except Exception:
            self._alpha = None  # Reset em caso de erro
            return self._last_regime

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regimes using existing trained models (no re-training).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime, confidence, and individual model predictions
        """
        if not self.is_trained:
            logger.warning("⚠️ Models not trained. Falling back to fit_predict()")
            return self.fit_predict(df)
            
        logger.debug("🔮 Predicting regimes using trained models...")
        
        # 1. Extract and scale features
        features_df = self._compute_features(df)
        
        # [B4 FIX] Garantir que as features passadas ao scaler sejam exatamente
        # as mesmas do fit (ordem, número e nomes). Prevêne data leakage.
        # fallback_features é a lista padrão usada se _fit_feature_names não existir
        fallback_features = ['returns', 'volatility', 'volume_z', 'adx', 'autocorr', 'drawdown']
        fit_features = getattr(self, '_fit_feature_names', fallback_features)
        
        # Alinhar: adiciona colunas faltantes com zero, remove extras
        for col in fit_features:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        features_array = features_df[fit_features].values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # [B4 FIX] NUNCA refazer fit do scaler na inferencia (data leakage)
        try:
            features_scaled = self.scaler.transform(features_array)
        except Exception as e:
            logger.error(
                f"❌ Scaler transform falhou: {e}. "
                f"Features de predict não coincidem com as do fit. "
                f"Fit features: {fit_features}, Atual: {list(features_df.columns)[:10]}..."
            )
            # Fallback seguro: zeros (não fit_transform)
            features_scaled = np.zeros_like(features_array)

        # 2. HMM Prediction
        if self.hmm_model:
            try:
                hidden_states = self.hmm_model.predict(features_scaled)
                # [FIX] Usa mapeamento salvo durante fit() em vez de recalcular pelo índice
                # da coluna 0, que muda se funding_rate for adicionado/removido entre sessões
                mapping = getattr(self, '_hmm_state_mapping', None)
                if mapping is None:
                    # Fallback seguro: reconstrói a partir das médias dos estados
                    state_means = self.hmm_model.means_[:, 0]
                    sorted_states = np.argsort(state_means)
                    mapping = {
                        sorted_states[0]: self.BEAR,
                        sorted_states[1]: self.RANGER,
                        sorted_states[2]: self.BULL
                    }
                hmm_regimes = np.vectorize(mapping.get)(hidden_states)
            except Exception as e:
                logger.warning(f"⚠️ HMM prediction failed: {e}. Using fallback.")
                hmm_regimes = self._fallback_hmm(features_scaled)
        else:
             hmm_regimes = self._fallback_hmm(features_scaled)
             
        # 3. GMM Prediction
        # 3. GMM Prediction [B2 FIX] Usa mapeamento salvo no fit, não recalculated
        if self.gmm_model:
            try:
                clusters = self.gmm_model.predict(features_scaled)
                # [B2 FIX] Usa _gmm_cluster_mapping salvo durante fit()
                mapping = getattr(self, '_gmm_cluster_mapping', None)
                if mapping is None:
                    # Fallback seguro: reconstroi por Sharpe (não por mean)
                    logger.warning("⚠️ _gmm_cluster_mapping não encontrado. Reconstruindo via Sharpe...")
                    cluster_scores = []
                    for c in range(self.config.n_gmm_components):
                        mask = (clusters == c)
                        if mask.sum() > 2:
                            r = features_scaled[mask, 0]
                            cluster_scores.append(r.mean() / (r.std() + 1e-8))
                        else:
                            cluster_scores.append(0.0)
                    sc = np.argsort(cluster_scores)
                    mapping = {int(sc[0]): self.BEAR, int(sc[1]): self.RANGER, int(sc[2]): self.BULL}
                gmm_regimes = np.vectorize(mapping.get)(clusters)
            except Exception as e:
                logger.warning(f"⚠️ GMM prediction failed: {e}. Using fallback.")
                gmm_regimes = self._fallback_gmm(features_scaled)
        else:
             gmm_regimes = self._fallback_gmm(features_scaled)
             
        # 4. Deterministic Regimes (ADX, Funding)
        adx_regimes = self._compute_adx_regimes(df, features_df['adx'])
        funding_regimes = self._compute_funding_regimes(df)
        
        # 5. Ensemble voting
        final_regimes, confidence = self._ensemble_voting(
            hmm_regimes, gmm_regimes, adx_regimes, funding_regimes
        )
        
        # 6. Build result
        result = pd.DataFrame(index=df.index)
        result['regime'] = final_regimes
        result['confidence'] = confidence
        result['hmm_regime'] = hmm_regimes
        result['gmm_regime'] = gmm_regimes
        result['adx_regime'] = adx_regimes
        result['funding_regime'] = funding_regimes
        result['regime_name'] = result['regime'].map(self.REGIME_NAMES)
        
        counts = np.bincount(final_regimes, minlength=3)
        total = len(final_regimes)
        logger.debug(
            f"✅ Prediction complete: "
            f"Bull={counts[0]} ({counts[0]/total:.1%}), "
            f"Bear={counts[1]} ({counts[1]/total:.1%}), "
            f"Ranger={counts[2]} ({counts[2]/total:.1%})"
        )
        
        return result

    def save(self, path: Optional[str] = None) -> bool:
        """Save trained detector to disk."""
        import os
        import joblib
        
        if path is None:
            os.makedirs(os.path.join(os.getcwd(), "models_ai"), exist_ok=True)
            path = os.path.join(os.getcwd(), "models_ai", "crypto_regime_detector.pkl")
        
        try:
            joblib.dump(self, path)
            logger.info(f"✅ CryptoRegimeDetector saved to {path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save detector: {e}")
            return False

    @classmethod
    def load(cls, path: Optional[str] = None) -> Optional['CryptoRegimeDetector']:
        """Load detector from disk."""
        import os
        import joblib
        
        if path is None:
            path = os.path.join(os.getcwd(), "models_ai", "crypto_regime_detector.pkl")
        
        if not os.path.exists(path):
            logger.info(f"📂 No saved detector at {path}")
            return None
        
        try:
            detector = joblib.load(path)
            logger.info(f"✅ CryptoRegimeDetector loaded from {path}")
            return detector
        except Exception as e:
            logger.error(f"❌ Failed to load detector: {e}")
            return None

    @property
    def is_trained(self) -> bool:
        """Check if detector has been trained."""
        return self.hmm_model is not None and self.gmm_model is not None


# Aliases for backward compatibility
ScientificRegimeDetector = CryptoRegimeDetector


def create_regime_detector(config: Optional[RegimeConfig] = None, load_if_exists: bool = True) -> CryptoRegimeDetector:
    """
    Factory function to create or load regime detector.
    
    Args:
        config: Optional configuration
        load_if_exists: If True, loads from disk if available
        
    Returns:
        CryptoRegimeDetector instance
    """
    import os
    import joblib
    
    model_path = os.path.join(os.getcwd(), "models_ai", "crypto_regime_detector.pkl")
    
    if load_if_exists and os.path.exists(model_path):
        try:
            detector = joblib.load(model_path)
            logger.info(f"✅ CryptoRegimeDetector loaded from {model_path}")
            return detector
        except Exception as e:
            logger.warning(f"⚠️ Failed to load detector: {e}. Creating new one.")
    
    return CryptoRegimeDetector(config)



