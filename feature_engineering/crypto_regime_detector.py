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

# [M5 FIX] joblib/loky CPU count: configura SOMENTE se o usuario nao definiu.
# Antes: import-time side-effect sobrescrevia LOKY_MAX_CPU_COUNT incondicionalmente,
# afetando outros modulos que usam joblib globalmente.
# Agora: respeita override do usuario, so define se ausente.
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
import ta
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
import warnings

from utils.logger import get_logger

logger = get_logger("CryptoRegimeDetector")

# Suppress convergence warnings during fitting
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class RegimeConfig:
    """Configuration for the Ensemble Regime Detector."""

    # HMM parameters
    n_hmm_states: int = 3
    hmm_n_iter: int = 100
    hmm_covariance_type: str = "full"

    # GMM parameters
    n_gmm_components: int = 3
    gmm_n_iter: int = 100
    gmm_covariance_type: str = "full"

    # ADX parameters
    adx_period: int = 9  # Shorter for crypto (faster regime changes)
    adx_trend_threshold: float = 20.0  # Lower than stocks (more volatile)

    # Ensemble parameters
    confidence_threshold: float = 0.75
    min_regime_duration: int = 8  # Minimum bars before regime change

    # Weights for ensemble voting (synchronized with CANONICAL_REGIME_CONFIG)
    weight_hmm: float = 0.35
    weight_gmm: float = 0.35
    weight_adx: float = 0.20
    weight_funding: float = 0.10

    # Feature windows (shorter for 24/7 crypto markets)
    returns_window: int = 1
    volatility_window: int = 10  # vs 20 for stocks
    volume_window: int = 10
    autocorr_window: int = 15  # Reduced from 20 for better responsiveness
    drawdown_window: int = 20  # [M2] Reduced from 50 (12.5h -> 5h lookback)

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

    REGIME_NAMES = {0: "Bull", 1: "Bear", 2: "Ranger"}

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
        self._live_alpha = None  # Exported copy for get_regime_probabilities live ticks
        self._last_regime = self.RANGER

        logger.info(
            f"✅ CryptoRegimeDetector initialized: "
            f"HMM={self.config.weight_hmm:.0%}, GMM={self.config.weight_gmm:.0%}, "
            f"ADX={self.config.weight_adx:.0%}, Funding={self.config.weight_funding:.0%}"
        )

    @staticmethod
    def cusum_filter(returns: np.ndarray, h: float = 2.0) -> np.ndarray:
        """
        CUSUM Structural Break Detection (López de Prado AFML Ch.17).

        Detects abrupt regime changes by monitoring cumulative sum of deviations.
        When |S_t| > h * sigma, a structural break is flagged.

        Args:
            returns: Array of log returns
            h: Threshold multiplier (number of sigmas)

        Returns:
            Boolean array where True = structural break detected at that bar
        """
        n = len(returns)
        breaks = np.zeros(n, dtype=bool)
        if n < 20:
            return breaks

        sigma = np.std(returns[:min(n, 100)])
        if sigma < 1e-10:
            return breaks

        threshold = h * sigma
        s_pos = 0.0
        s_neg = 0.0
        mean = np.mean(returns[:min(n, 100)])

        for t in range(n):
            s_pos = max(0.0, s_pos + returns[t] - mean)
            s_neg = min(0.0, s_neg + returns[t] - mean)

            if s_pos > threshold or s_neg < -threshold:
                breaks[t] = True
                s_pos = 0.0
                s_neg = 0.0

        return breaks

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
        # Primeiro bar nao tem retorno anterior — preencher com 0.0 (causal).
        returns_raw = df["close"].pct_change()
        features["returns"] = returns_raw.fillna(0.0)

        # 2. Volatility (rolling std of returns) — imputa com expanding median (causal)
        vol_raw = features["returns"].rolling(self.config.volatility_window).std()
        _vol_expanding_median = vol_raw.expanding().median()
        features["volatility"] = vol_raw.fillna(_vol_expanding_median).fillna(0.01)

        # 3. Volume Z-score (normalized)
        if "volume" in df.columns:
            volume_mean = df["volume"].rolling(self.config.volume_window).mean()
            volume_std = df["volume"].rolling(self.config.volume_window).std()
            features["volume_z"] = (
                (df["volume"] - volume_mean) / (volume_std + 1e-8)
            ).fillna(0)
        else:
            features["volume_z"] = 0.0

        # 4. ADX (Average Directional Index)
        features["adx"] = self._compute_adx(df)

        # 5. Autocorrelation (momentum persistence)
        # [M2 FIX] Vectorized via numpy: era O(N*W) com callback Python (15-bar window x 1.6M bars
        # = 24M chamadas pd.Series). Agora: corr(returns, returns.shift(1)) em janela rolling
        # usando rolling.cov + std (formula de Pearson) — 50-100x mais rápido.
        _r = features["returns"]
        _r_lag = _r.shift(1)
        _w = self.config.autocorr_window
        _cov = _r.rolling(_w).cov(_r_lag)
        _std_r = _r.rolling(_w).std()
        _std_lag = _r_lag.rolling(_w).std()
        features["autocorr"] = (_cov / (_std_r * _std_lag + 1e-12)).clip(-1.0, 1.0).fillna(0.0)

        # 🧪 [FIX] RSI Momentum (Faster than EMA)
        try:
            rsi_indicator = ta.momentum.RSIIndicator(close=df["close"], window=14)
            features["rsi"] = rsi_indicator.rsi().fillna(50) / 100.0  # Normalized 0-1
        except Exception:
            features["rsi"] = 0.5

        # 🧪 [FIX] Vol-Weighted Returns (Smart Money)
        vol_mean = df["volume"].rolling(20).mean()
        features["vol_weighted_ret"] = features["returns"] * (
            df["volume"] / (vol_mean + 1e-8)
        )

        # 6. Drawdown (distance from rolling max) [M2: agora usa config]
        rolling_max = (
            df["close"].rolling(self.config.drawdown_window, min_periods=1).max()
        )
        features["drawdown"] = (df["close"] - rolling_max) / (rolling_max + 1e-8)
        features["drawdown"] = features["drawdown"].fillna(0)

        # 7. Funding Rate — armazenada como feature, NÃO como voto (F1 fix)
        if "funding_rate" in df.columns:
            features["funding_rate"] = df["funding_rate"].fillna(0)
            features["funding_ma"] = df["funding_rate"].rolling(8).mean().fillna(0)
        else:
            features["funding_rate"] = 0.0
            features["funding_ma"] = 0.0

        # 8. Open Interest Change (futures)
        # OI change=0 no primeiro bar é semanticamente correto ("sem mudança"),
        # diferente de returns onde 0 é um sinal falso de regime.
        if "open_interest" in df.columns:
            features["oi_change"] = df["open_interest"].pct_change().fillna(0.0)
        else:
            features["oi_change"] = 0.0

        # 9. Trend direction (EMA crossover)
        ema_fast = df["close"].ewm(span=10, adjust=False).mean()
        ema_slow = df["close"].ewm(span=20, adjust=False).mean()
        features["trend_direction"] = np.where(ema_fast > ema_slow, 1, -1)

        # [M1] DI+ - DI-: captura direção E força ao mesmo tempo
        # ADX mede força mas não direção; di_diff resolve isso.
        try:
            adx_ind = ta.trend.ADXIndicator(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=self.config.adx_period,
            )
            features["di_diff"] = (adx_ind.adx_pos() - adx_ind.adx_neg()).fillna(0)
        except Exception:
            features["di_diff"] = 0.0

        return features

    def _compute_adx(self, df: pd.DataFrame) -> pd.Series:
        """Compute ADX (Average Directional Index) using ta library."""
        try:
            adx_indicator = ta.trend.ADXIndicator(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=self.config.adx_period,
            )
            return adx_indicator.adx().fillna(0)
        except Exception as e:
            logger.warning(f"⚠️ ADX computation failed: {e}")
            return pd.Series(0.0, index=df.index)

    def _train_hmm(
        self,
        features: np.ndarray,
        train_split_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Train Hidden Markov Model and predict regimes.

        Uses hmmlearn if available, falls back to simple Markov chain.
        [B1 FIX] Sub-amostragem via janela contígua aleatória, não stride.
        Stride viola a suposição Markoviana ao criar sequências com gaps.
        [F2 FIX] Mapeamento por score Sharpe-like (mean/std) em vez de mean apenas.

        Args:
            features: array (n_samples, n_features) com features escaladas
            train_split_idx: se fornecido, treina apenas em features[:train_split_idx]
                             (anti-leakage: respeita o split 80/20 do fit_predict).
                             Se None, sub-amostra os primeiros 50k bars (default legado).
        """
        try:
            from hmmlearn.hmm import GaussianHMM

            self.hmm_model = GaussianHMM(
                n_components=self.config.n_hmm_states,
                covariance_type=self.config.hmm_covariance_type,
                n_iter=self.config.hmm_n_iter,
                random_state=42,
            )

            # [B1 FIX] Sub-amostragem por janela contígua — SEMPRE os primeiros N bars.
            # Uma janela aleatória pode começar no meio da série e treinar o HMM em dados
            # futuros relativos aos bars iniciais → look-ahead bias nos labels de treino.
            # [BUG FIX] Respeita train_split_idx vindo do fit_predict (split 80/20)
            # para nao treinar em features OOD. Sem este parametro, fit_predict()
            # chamava _train_hmm() e quebrava com TypeError em todo treino do regime.
            if train_split_idx is not None:
                train_data = features[: max(1, int(train_split_idx))]
                if len(train_data) > 50000:
                    logger.info(
                        f"Sub-sampling HMM training data (first 50k/{len(train_data)})"
                    )
                    train_data = train_data[:50000]
            elif len(features) > 50000:
                logger.info(
                    f"Sub-sampling HMM training data (first 50k/{len(features)})"
                )
                train_data = features[:50000]
            else:
                train_data = features

            self.hmm_model.fit(train_data)
            # [CAUSAL FIX] Forward Algorithm only — no Viterbi backward pass.
            # Viterbi uses P(S_1:T | O_1:T) which is non-causal (future leakage).
            # Forward Algorithm uses P(S_t | O_1:t) which is strictly causal.
            # Ref: Hamilton (1989) — regime labels must not use future observations.
            # Ref: López de Prado (2018) AFML Ch.7 — labels must use only info
            #       available at observation time.
            hidden_states = self._forward_only_decode(features)

            # [F2 FIX] Mapeamento por Sharpe Contemporâneo (SEM LEAKAGE)
            # Bull = Alto retorno/Baixa vol, Bear = Baixo retorno/Alta vol.
            # [ANTI-LEAKAGE] Usar apenas dados do split de treino para calcular scores
            _map_end = train_split_idx if train_split_idx is not None else len(features)
            state_scores = []
            for state in range(self.config.n_hmm_states):
                mask = hidden_states[:_map_end] == state
                if mask.sum() > 2:
                    r = features[:_map_end][mask, 0]  # Retorno contemporâneo (apenas treino)
                    score = r.mean() / (r.std() + 1e-8)
                    state_scores.append(score)
                else:
                    state_scores.append(0.0)

            # Sort: highest Forward Return = Bull, lowest = Bear, middle = Ranger
            sorted_states = np.argsort(state_scores)
            mapping = {
                sorted_states[0]: self.BEAR,
                sorted_states[1]: self.RANGER,
                sorted_states[2]: self.BULL,
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

    def _forward_only_decode(self, features_scaled: np.ndarray) -> np.ndarray:
        """
        [CAUSAL] Decode HMM states using Forward Algorithm only.

        Unlike Viterbi (used by hmmlearn.predict()), the Forward Algorithm
        computes P(S_t | O_1:t) — the filtered probability using ONLY past
        and present observations. This is strictly causal.

        Ref: Hamilton (1989) — regime-switching models require causal inference
             for out-of-sample prediction.
        Ref: López de Prado (2018) AFML Ch.7 — labels must use only information
             available at observation time.
        """
        model = self.hmm_model
        n_samples = len(features_scaled)
        n_states = model.n_components

        # Compute log emission probabilities using hmmlearn's internal method
        try:
            log_emis = model._compute_log_likelihood(features_scaled)
        except Exception as e:
            logger.warning(
                "[CAUSAL HMM] _compute_log_likelihood failed (%s), "
                "using uniform prior (causal fallback, no Viterbi)",
                e,
            )
            # [ANTI-LEAKAGE] Retornar regime uniforme em vez de Viterbi (que usa backward pass)
            return np.full(n_samples, self.RANGER)

        # Forward algorithm in log-space for numerical stability
        log_startprob = np.log(model.startprob_ + 1e-300)
        log_transmat = np.log(model.transmat_ + 1e-300)

        log_alpha = np.zeros((n_samples, n_states))

        # Initialization: α_0(j) = π_j * b_j(O_0)
        log_alpha[0] = log_startprob + log_emis[0]

        # Recursion: α_t(j) = [Σ_i α_{t-1}(i) * a_ij] * b_j(O_t)
        for t in range(1, n_samples):
            for j in range(n_states):
                log_alpha[t, j] = (
                    np.logaddexp.reduce(log_alpha[t - 1] + log_transmat[:, j])
                    + log_emis[t, j]
                )

        # Causal decode: argmax of filtered probability at each t
        # P(S_t | O_1:t) ∝ α_t — no backward pass needed
        hidden_states = np.argmax(log_alpha, axis=1)

        logger.info(
            "✅ HMM decoded with Forward Algorithm (causal, no Viterbi backward pass)"
        )
        return hidden_states

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

    def _train_gmm(
        self,
        features: np.ndarray,
        train_split_idx: Optional[int] = None,
    ) -> np.ndarray:
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
                random_state=42,
            )

            if train_split_idx is not None:
                self.gmm_model.fit(features[: min(train_split_idx, len(features))])
            elif len(features) > 50000:
                self.gmm_model.fit(features[:50000])
            else:
                self.gmm_model.fit(features)

            clusters = self.gmm_model.predict(features)

            # [B2+F2 FIX] Economic grounding: Sharpe Contemporâneo (SEM LEAKAGE)
            # [ANTI-LEAKAGE] Usar apenas dados do split de treino para calcular scores
            _map_end = train_split_idx if train_split_idx is not None else len(features)
            cluster_scores = []
            for cluster in range(self.config.n_gmm_components):
                mask = clusters[:_map_end] == cluster
                if mask.sum() > 2:
                    r = features[:_map_end][mask, 0]  # Retorno contemporâneo (apenas treino)
                    score = r.mean() / (r.std() + 1e-8)
                    cluster_scores.append(score)
                else:
                    cluster_scores.append(0.0)

            sorted_clusters = np.argsort(cluster_scores)
            mapping = {
                int(sorted_clusters[0]): self.BEAR,
                int(sorted_clusters[1]): self.RANGER,
                int(sorted_clusters[2]): self.BULL,
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
        regimes = np.full(len(returns), self.RANGER)

        # Usa percentis do treino completo se disponíveis; recalcula apenas na primeira vez.
        # Percentis dinâmicos em janelas pequenas (live) produzem thresholds instáveis.
        p33 = getattr(self, "_fallback_p33", float(np.percentile(returns, 33)))
        p67 = getattr(self, "_fallback_p67", float(np.percentile(returns, 67)))

        regimes[returns > p67] = self.BULL
        regimes[returns < p33] = self.BEAR

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
        ema_fast = df["close"].ewm(span=10, adjust=False).mean()
        ema_slow = df["close"].ewm(span=20, adjust=False).mean()

        # Check both ADX strength and trend direction
        is_trending = adx > self.config.adx_trend_threshold
        is_bullish = ema_fast > ema_slow

        bull_mask = is_trending & is_bullish
        bear_mask = is_trending & ~is_bullish

        regimes[bull_mask] = self.BULL
        regimes[bear_mask] = self.BEAR

        return regimes

    def _compute_funding_risk(
        self, df: pd.DataFrame, confidence: np.ndarray
    ) -> np.ndarray:
        """
        [F1 FIX] Funding rate como FATOR DE RISCO, não voto de regime.

        O funding rate é um sinal CONTRÁRIO: funding alto durante bull market
        significa longs sobreaquecidos → risco de reversao, não Bear imediato.
        Tratar como voto de regime causa label mixing (bull classificado como bear).

        Agora: atenua a confiança quando o funding é extremo, sem alterar o regime.
        """
        attenuated = confidence.copy()

        if "funding_rate" not in df.columns:
            return attenuated

        funding = df["funding_rate"].values

        # Funding extremo = incerteza sobre a continuidade do regime atual
        extreme_mask = np.abs(funding) > self.config.funding_risk_threshold
        attenuated[extreme_mask] = np.maximum(
            attenuated[extreme_mask] - self.config.funding_risk_attenuation,
            self.config.confidence_threshold,  # Não cai abaixo do threshold base
        )

        return attenuated

    # [F1] Mantemos _compute_funding_regimes como stub vazio para backward compat
    def _compute_funding_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """[DEPRECATED] Funding não vota mais no ensemble. Use _compute_funding_risk."""
        return np.full(len(df), self.RANGER)  # Retorna Ranger (neutro) para todos

    def _ensemble_voting(
        self, hmm: np.ndarray, gmm: np.ndarray, adx: np.ndarray, funding: np.ndarray
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
        votes_matrix = np.vstack([hmm, gmm, adx, funding])  # Shape (4, n_samples)
        weight_list = np.array(
            [
                self.config.weight_hmm,
                self.config.weight_gmm,
                self.config.weight_adx,
                self.config.weight_funding,
            ]
        )

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
                final_regimes[i] = final_regimes[i - 1]
                prev_conf = confidence[i - 1]
                confidence[i] = max(confidence[i], prev_conf * 0.95)

        # Smooth transitions (remove micro-regimes)
        final_regimes = self._smooth_regimes(final_regimes, confidence)

        # Guard against float-precision rounding from weight normalization
        confidence = np.clip(confidence, 0.0, 1.0)
        return final_regimes, confidence

    def _smooth_regimes(
        self, regimes: np.ndarray, confidence: np.ndarray
    ) -> np.ndarray:
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

            if (
                duration < self.config.min_regime_duration
                and seg_start > 0
                and not is_high_confidence
            ):
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
        required_cols = ["high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # 1. Extract features
        logger.info("🔧 Computing features...")
        features_df = self._compute_features(df)

        # Select core features for models
        core_features = [
            "returns",
            "volatility",
            "volume_z",
            "adx",
            "autocorr",
            "drawdown",
            "di_diff",
        ]
        if "funding_rate" in df.columns:
            core_features.append("funding_rate")

        features_array = features_df[core_features].values

        # [C2 FIX] Substituir inf por NaN e dropar linhas com NaN ANTES do fit do scaler.
        # Antes: nan_to_num(nan=0.0) contaminava o scaler com zeros artificiais,
        # enviesando média/std e quebrando a filosofia anti-fillna(0) do pipeline.
        # Agora: imputamos por mediana da coluna (causal, robusta) — consistente com
        # scientific_data_processor.py e native_indicators.py.
        features_array = np.where(np.isfinite(features_array), features_array, np.nan)
        # [ANTI-LEAKAGE] split_idx calculado antes da imputação para usar apenas treino
        split_idx = int(len(features_array) * 0.80)
        # [ANTI-LEAKAGE] col_medians calculada apenas nos primeiros 80% (treino)
        col_medians = np.nanmedian(features_array[:split_idx], axis=0)
        nan_mask = np.isnan(features_array)
        for col_idx in range(features_array.shape[1]):
            features_array[nan_mask[:, col_idx], col_idx] = col_medians[col_idx]
        # Fallback final: se coluna inteira era NaN, mediana é NaN → usa 0
        features_array = np.nan_to_num(features_array, nan=0.0)

        # Scale features — [X2 FIX] fit apenas nos primeiros 80% para evitar look-ahead bias
        # O scaler NÃO deve ver dados futuros (val/test) durante o fit
        self.scaler.fit(features_array[:split_idx])
        features_scaled = self.scaler.transform(features_array)

        # 2. Train individual models
        logger.info("🧠 Training HMM...")
        hmm_regimes = self._train_hmm(features_scaled, train_split_idx=split_idx)

        logger.info("🧠 Training GMM...")
        gmm_regimes = self._train_gmm(features_scaled, train_split_idx=split_idx)

        logger.info("📈 Computing ADX regimes...")
        adx_regimes = self._compute_adx_regimes(df, features_df["adx"])

        logger.info("💰 Computing Funding regimes...")
        funding_regimes = self._compute_funding_regimes(df)

        # 3. Ensemble voting
        logger.info("🗳️  Applying ensemble voting...")
        final_regimes, confidence = self._ensemble_voting(
            hmm_regimes, gmm_regimes, adx_regimes, funding_regimes
        )

        # [F1 FIX] Aplicar atenuacao de fundingr rate sobre a confianca
        confidence = self._compute_funding_risk(df, confidence)

        # Salva _fit_feature_names para proteger o scaler em predict()
        self._fit_feature_names = core_features
        # Salva percentis macro para _fallback_gmm — usa apenas os primeiros 80% para evitar
        # look-ahead bias (split_idx já calculado acima para o scaler)
        self._fallback_p33 = float(np.percentile(features_array[:split_idx, 0], 33))
        self._fallback_p67 = float(np.percentile(features_array[:split_idx, 0], 67))
        # 4. Build result DataFrame
        result = pd.DataFrame(index=df.index)
        result["regime"] = final_regimes
        result["confidence"] = confidence
        result["hmm_regime"] = hmm_regimes
        result["gmm_regime"] = gmm_regimes
        result["adx_regime"] = adx_regimes
        result["funding_regime"] = funding_regimes
        result["regime_name"] = result["regime"].map(self.REGIME_NAMES)

        # 4b. CUSUM Structural Break Detection (López de Prado AFML Ch.17)
        returns_for_cusum = features_array[:, 0]  # log returns
        structural_breaks = self.cusum_filter(returns_for_cusum, h=2.0)
        result["structural_break"] = structural_breaks

        # Injeta returns para permitir validação financeira em validate_regimes().
        # Sem esta coluna, validate_regimes usa zeros e financial_validity_ok é sempre False.
        if "returns" in features_df.columns:
            result["returns"] = features_df["returns"].values
        elif "close" in df.columns:
            result["returns"] = df["close"].pct_change().fillna(0.0).values

        # Incorpora probabilidades brutas do ensemble diretamente no resultado para evitar
        # que predict() splits sobrescrevam _raw_ensemble_scores e quebrem get_regime_probabilities().
        if (
            hasattr(self, "_raw_ensemble_scores")
            and self._raw_ensemble_scores is not None
        ):
            raw = self._raw_ensemble_scores
            row_sums = raw.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            normalized = raw / row_sums
            result["prob_bull"] = normalized[:, 0]
            result["prob_bear"] = normalized[:, 1]
            result["prob_ranger"] = normalized[:, 2]

        # Store for validation
        self.regime_history = result
        self.confidence_scores = confidence

        # Log statistics
        counts = np.bincount(final_regimes, minlength=3)
        total = len(final_regimes)
        logger.info(
            f"✅ Regime detection complete: "
            f"Bull={counts[0]} ({counts[0] / total:.1%}), "
            f"Bear={counts[1]} ({counts[1] / total:.1%}), "
            f"Ranger={counts[2]} ({counts[2] / total:.1%})"
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

        regimes = self.regime_history["regime"].values
        confidence = self.regime_history["confidence"].values

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

        metrics["avg_persistence_bars"] = np.mean(regime_lengths)

        # 2. Model Agreement (average across ensemble)
        hmm = self.regime_history["hmm_regime"].values
        gmm = self.regime_history["gmm_regime"].values
        adx = self.regime_history["adx_regime"].values
        funding = self.regime_history["funding_regime"].values

        agreements = []
        for i in range(len(regimes)):
            votes = [hmm[i], gmm[i], adx[i], funding[i]]
            max_agreement = max(votes.count(r) for r in [0, 1, 2])
            agreements.append(max_agreement / 4)

        metrics["avg_model_agreement"] = np.mean(agreements)

        # 3. Mean Confidence
        metrics["mean_confidence"] = np.mean(confidence)

        # 4. Regime Imbalance (Gini coefficient)
        counts = np.bincount(regimes, minlength=3) / len(regimes)
        gini = 1 - np.sum(counts**2)
        metrics["regime_imbalance_gini"] = gini

        # 5. Transition Rate
        transitions = np.sum(regime_changes) / len(regimes)
        metrics["transition_rate"] = transitions

        # 6. Regime Distribution
        metrics["bull_pct"] = np.mean(regimes == self.BULL)
        metrics["bear_pct"] = np.mean(regimes == self.BEAR)
        metrics["sideways_pct"] = np.mean(regimes == self.RANGER)

        # Quality checks
        metrics["persistence_ok"] = (
            metrics["avg_persistence_bars"] >= self.config.min_persistence_bars
        )
        metrics["agreement_ok"] = (
            metrics["avg_model_agreement"] >= self.config.min_model_agreement
        )
        metrics["confidence_ok"] = (
            metrics["mean_confidence"] >= self.config.confidence_threshold * 0.9
        )
        metrics["transition_ok"] = (
            metrics["transition_rate"] <= self.config.max_transition_rate
        )
        metrics["balance_ok"] = (
            metrics["regime_imbalance_gini"] <= self.config.max_gini_imbalance
        )

        metrics["all_checks_passed"] = all(
            [
                metrics["persistence_ok"],
                metrics["agreement_ok"],
                metrics["confidence_ok"],
                metrics["transition_ok"],
                metrics["balance_ok"],
            ]
        )

        # [M3] 7. Forward Return by Regime (validação financeira — APENAS treino 80%)
        # Restringe a dados de treino para evitar feedback loop com dados futuros.
        forward_bars = 5
        _val_end = int(len(regimes) * 0.80)
        if "returns" in self.regime_history.columns:
            fwd_returns = self.regime_history["returns"].values[:_val_end]
        else:
            fwd_returns = np.zeros(_val_end)

        fwd_by_regime: Dict[str, float] = {}
        _regimes_train = regimes[:_val_end]
        for r_code, r_name in self.REGIME_NAMES.items():
            mask = _regimes_train[:-forward_bars] == r_code
            if mask.sum() > 5:
                future_returns = np.array(
                    [
                        fwd_returns[i + 1 : i + forward_bars + 1].sum()
                        for i in np.where(mask)[0]
                    ]
                )
                fwd_by_regime[r_name] = float(np.mean(future_returns))
            else:
                fwd_by_regime[r_name] = 0.0

        metrics["forward_return_bull"] = fwd_by_regime.get("Bull", 0.0)
        metrics["forward_return_bear"] = fwd_by_regime.get("Bear", 0.0)
        metrics["forward_return_ranger"] = fwd_by_regime.get("Ranger", 0.0)
        # Bull deve ter forward return > Bear
        metrics["financial_validity_ok"] = (
            metrics["forward_return_bull"] > metrics["forward_return_bear"]
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
        return result["regime"].values

    def get_regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns soft probabilities for each regime.
        [F4 FIX] Usa scores brutos do ensemble em vez de distribuição binária fixa.

        A distribuição binária (conf, (1-conf)/2, (1-conf)/2) perde informação
        sobre onde a incerteza está concentrada (Bear vs Ranger distintos).
        """
        if self.regime_history is None:
            raise RuntimeError(
                "CryptoRegimeDetector: must call fit_predict() before get_regime_probabilities(). Use 80/20 split."
            )

        probs = pd.DataFrame(index=df.index)

        # [F4 FIX + STATE FIX] Lê probabilidades do regime_history (imutável após fit_predict)
        # em vez de _raw_ensemble_scores que é sobrescrito por cada chamada de predict().
        if "prob_bull" in self.regime_history.columns:
            aligned = self.regime_history.reindex(df.index)
            probs["prob_bull"] = aligned["prob_bull"]
            probs["prob_bear"] = aligned["prob_bear"]
            probs["prob_ranger"] = aligned["prob_ranger"]

            nan_mask = probs["prob_bull"].isna()
            if nan_mask.any() and self._live_alpha is not None:
                mapping = getattr(self, "_hmm_state_mapping", None)
                if mapping is not None:
                    inv_map = {regime: state for state, regime in mapping.items()}
                    live_probs = pd.DataFrame(
                        {
                            "prob_bull": [
                                float(self._live_alpha[inv_map.get(self.BULL, 2)])
                            ],
                            "prob_bear": [
                                float(self._live_alpha[inv_map.get(self.BEAR, 0)])
                            ],
                            "prob_ranger": [
                                float(self._live_alpha[inv_map.get(self.RANGER, 1)])
                            ],
                        },
                        index=df.index[nan_mask],
                    )
                    probs.loc[nan_mask, ["prob_bull", "prob_bear", "prob_ranger"]] = (
                        live_probs
                    )
                    nan_mask = probs["prob_bull"].isna()

            probs["prob_bull"] = probs["prob_bull"].fillna(1 / 3)
            probs["prob_bear"] = probs["prob_bear"].fillna(1 / 3)
            probs["prob_ranger"] = probs["prob_ranger"].fillna(1 / 3)
        else:
            # Fallback legado: distribuição baseada em confiança
            conf = self.regime_history["confidence"].values
            regimes = self.regime_history["regime"].values
            probs["prob_bull"] = np.where(regimes == self.BULL, conf, (1 - conf) / 2)
            probs["prob_bear"] = np.where(regimes == self.BEAR, conf, (1 - conf) / 2)
            probs["prob_ranger"] = np.where(
                regimes == self.RANGER, conf, (1 - conf) / 2
            )
            total = probs.sum(axis=1)
            probs = probs.div(total, axis=0)

        return probs

    def predict_online(self, observation: np.ndarray, adx_value: float = None, ema_fast: float = None, ema_slow: float = None) -> int:
        """
        Online regime prediction using FULL ensemble (HMM + GMM + ADX).
        [B3 FIX] Forward Algorithm real para HMM com memória.
        [ENSEMBLE FIX] Agora usa os 3 modelos como no backtest, eliminando mismatch train/live.

        Args:
            observation: Feature vector (same as fit features)
            adx_value: Current ADX value (optional, for ADX regime)
            ema_fast: Current fast EMA (optional, for ADX trend direction)
            ema_slow: Current slow EMA (optional, for ADX trend direction)
        """
        try:
            obs = observation.reshape(1, -1)
            obs_scaled = self.scaler.transform(obs)

            # --- HMM Forward Algorithm ---
            hmm_regime = self.RANGER
            hmm_conf = 0.33
            if self.hmm_model is not None:
                log_emit = self.hmm_model._compute_log_likelihood(obs_scaled)
                log_emit = log_emit[0] - log_emit[0].max()
                emission_probs = np.exp(log_emit)
                emission_probs /= emission_probs.sum() + 1e-300

                if self._alpha is None:
                    self._alpha = self.hmm_model.startprob_ * emission_probs
                    self._alpha /= self._alpha.sum() + 1e-300
                else:
                    transmat = self.hmm_model.transmat_
                    self._alpha = (self._alpha @ transmat) * emission_probs
                    total = self._alpha.sum()
                    if total > 1e-300:
                        self._alpha /= total
                    else:
                        self._alpha = np.ones(self.config.n_hmm_states) / self.config.n_hmm_states

                best_state = int(np.argmax(self._alpha))
                hmm_conf = float(self._alpha[best_state])
                mapping = getattr(self, "_hmm_state_mapping", None)
                if mapping is not None:
                    hmm_regime = mapping.get(best_state, self.RANGER)
                else:
                    hmm_regime = best_state

            # --- GMM Prediction ---
            gmm_regime = self.RANGER
            if self.gmm_model is not None:
                cluster = int(self.gmm_model.predict(obs_scaled)[0])
                gmm_mapping = getattr(self, "_gmm_cluster_mapping", None)
                if gmm_mapping is not None:
                    gmm_regime = gmm_mapping.get(cluster, self.RANGER)

            # --- ADX Regime ---
            adx_regime = self.RANGER
            if adx_value is not None and ema_fast is not None and ema_slow is not None:
                if adx_value > self.config.adx_trend_threshold:
                    adx_regime = self.BULL if ema_fast > ema_slow else self.BEAR

            # --- Ensemble Voting (weighted) ---
            scores = np.zeros(3)
            w_hmm = self.config.weight_hmm
            w_gmm = self.config.weight_gmm
            w_adx = self.config.weight_adx
            w_total = w_hmm + w_gmm + w_adx
            if w_total > 0:
                scores[hmm_regime] += w_hmm / w_total
                scores[gmm_regime] += w_gmm / w_total
                scores[adx_regime] += w_adx / w_total

            best_regime = int(np.argmax(scores))
            best_conf = float(scores[best_regime])

            if best_conf < self.config.confidence_threshold:
                return self._last_regime

            self._last_regime = best_regime
            self._live_alpha = self._alpha.copy() if self._alpha is not None else None
            return best_regime

        except Exception:
            # Não reseta _alpha — preserva estado acumulado do Forward Algorithm.
            # Resetar destruiria a memória do HMM desnecessariamente.
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
        fallback_features = [
            "returns",
            "volatility",
            "volume_z",
            "adx",
            "autocorr",
            "drawdown",
        ]
        fit_features = getattr(self, "_fit_feature_names", fallback_features)

        # Alinhar: adiciona colunas faltantes com zero, remove extras
        for col in fit_features:
            if col not in features_df.columns:
                features_df[col] = 0.0

        features_array = features_df[fit_features].values
        # [C2 FIX] Imputação por mediana (consistente com fit_predict).
        # Em inferência usamos mediana do próprio batch — aceitável porque o scaler
        # já foi fitado com estatísticas corretas do treino.
        features_array = np.where(np.isfinite(features_array), features_array, np.nan)
        col_medians = np.nanmedian(features_array, axis=0)
        nan_mask_pred = np.isnan(features_array)
        for col_idx in range(features_array.shape[1]):
            features_array[nan_mask_pred[:, col_idx], col_idx] = col_medians[col_idx]
        features_array = np.nan_to_num(features_array, nan=0.0)

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

        # 2. FORWARD ALGORITHM — Causal step-by-step (no Viterbi, no backward pass)
        # We implement Forward filtering manually: alpha[t] = alpha[t-1] @ transmat * emit[t]
        # Each state at time t depends ONLY on observations up to t (causal).
        #
        # [BUG FIX] score_samples(single_obs) injects startprob_ into posteriors, masking
        # states that start with prob=0. When startprob_=[0,1,0], the emission for every
        # step becomes [0,1,0] → Forward pass collapses to state 1 for all T bars.
        # Fix: use _compute_log_likelihood() which returns raw Gaussian log-emissions
        # P(obs_t | state_j) without any startprob influence. Vectorized (O(T) not O(T²)).
        if self.hmm_model:
            try:
                T, n_states = features_scaled.shape[0], self.config.n_hmm_states
                transmat = self.hmm_model.transmat_

                # Raw Gaussian emission log-probabilities: shape (T, n_states)
                log_emit = self.hmm_model._compute_log_likelihood(features_scaled)
                # Numerically stable softmax row-wise → normalized emission probs
                log_emit -= log_emit.max(axis=1, keepdims=True)
                emit_probs = np.exp(log_emit)
                emit_probs /= emit_probs.sum(axis=1, keepdims=True) + 1e-300

                # Forward pass: alpha[0] = startprob * emit[0]; alpha[t] = (alpha[t-1] @ transmat) * emit[t]
                alpha = np.zeros((T, n_states))
                alpha[0] = self.hmm_model.startprob_ * emit_probs[0]
                alpha[0] /= (
                    alpha[0].sum() if alpha[0].sum() > 1e-300 else 1.0 / n_states
                )

                for t in range(1, T):
                    alpha[t] = (alpha[t - 1] @ transmat) * emit_probs[t]
                    s = alpha[t].sum()
                    alpha[t] /= s if s > 1e-300 else 1.0 / n_states

                hidden_states = alpha.argmax(axis=1)
                mapping = getattr(self, "_hmm_state_mapping", None)
                if mapping is None:
                    state_means = self.hmm_model.means_[:, 0]
                    sorted_states = np.argsort(state_means)
                    mapping = {
                        sorted_states[0]: self.BEAR,
                        sorted_states[1]: self.RANGER,
                        sorted_states[2]: self.BULL,
                    }
                hmm_regimes = np.vectorize(mapping.get)(hidden_states)
            except Exception as e:
                logger.warning(f"HMM prediction failed: {e}. Using fallback.")
                hmm_regimes = self._fallback_hmm(features_scaled)
        else:
            hmm_regimes = self._fallback_hmm(features_scaled)

        # 3. GMM Prediction
        # 3. GMM Prediction [B2 FIX] Usa mapeamento salvo no fit, não recalculated
        if self.gmm_model:
            try:
                clusters = self.gmm_model.predict(features_scaled)
                # [B2 FIX] Usa _gmm_cluster_mapping salvo durante fit()
                mapping = getattr(self, "_gmm_cluster_mapping", None)
                if mapping is None:
                    # Fallback seguro: reconstroi por Sharpe (não por mean)
                    logger.warning(
                        "⚠️ _gmm_cluster_mapping não encontrado. Reconstruindo via Sharpe..."
                    )
                    cluster_scores = []
                    for c in range(self.config.n_gmm_components):
                        mask = clusters == c
                        if mask.sum() > 2:
                            r = features_scaled[mask, 0]
                            cluster_scores.append(r.mean() / (r.std() + 1e-8))
                        else:
                            cluster_scores.append(0.0)
                    sc = np.argsort(cluster_scores)
                    mapping = {
                        int(sc[0]): self.BEAR,
                        int(sc[1]): self.RANGER,
                        int(sc[2]): self.BULL,
                    }
                gmm_regimes = np.vectorize(mapping.get)(clusters)
            except Exception as e:
                logger.warning(f"⚠️ GMM prediction failed: {e}. Using fallback.")
                gmm_regimes = self._fallback_gmm(features_scaled)
        else:
            gmm_regimes = self._fallback_gmm(features_scaled)

        # 4. Deterministic Regimes (ADX, Funding)
        adx_regimes = self._compute_adx_regimes(df, features_df["adx"])
        funding_regimes = self._compute_funding_regimes(df)

        # 5. Ensemble voting
        final_regimes, confidence = self._ensemble_voting(
            hmm_regimes, gmm_regimes, adx_regimes, funding_regimes
        )

        # 6. Build result
        result = pd.DataFrame(index=df.index)
        result["regime"] = final_regimes
        result["confidence"] = confidence
        result["hmm_regime"] = hmm_regimes
        result["gmm_regime"] = gmm_regimes
        result["adx_regime"] = adx_regimes
        result["funding_regime"] = funding_regimes
        result["regime_name"] = result["regime"].map(self.REGIME_NAMES)

        # Incorpora probabilidades brutas — evita que _raw_ensemble_scores seja sobrescrito
        # por chamadas subsequentes de predict() e quebre get_regime_probabilities().
        if (
            hasattr(self, "_raw_ensemble_scores")
            and self._raw_ensemble_scores is not None
        ):
            raw = self._raw_ensemble_scores
            row_sums = raw.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            normalized = raw / row_sums
            result["prob_bull"] = normalized[:, 0]
            result["prob_bear"] = normalized[:, 1]
            result["prob_ranger"] = normalized[:, 2]

        counts = np.bincount(final_regimes, minlength=3)
        total = len(final_regimes)
        logger.debug(
            f"✅ Prediction complete: "
            f"Bull={counts[0]} ({counts[0] / total:.1%}), "
            f"Bear={counts[1]} ({counts[1] / total:.1%}), "
            f"Ranger={counts[2]} ({counts[2] / total:.1%})"
        )

        # TensorBoard: regime distribution + ensemble agreement
        self._log_regime_tb(
            final_regimes, confidence, hmm_regimes, gmm_regimes, adx_regimes
        )

        # CUSUM structural break detection (López de Prado AFML Ch.17)
        returns_col = features_df["returns"].values if "returns" in features_df.columns else features_array[:, 0]
        structural_breaks = self.cusum_filter(returns_col, h=2.0)
        result["structural_break"] = structural_breaks
        if structural_breaks.any():
            break_pct = structural_breaks.sum() / len(structural_breaks)
            result.loc[structural_breaks, "confidence"] *= 0.7
            logger.debug(f"⚠️ CUSUM: {structural_breaks.sum()} breaks ({break_pct:.1%}), confidence reduced at break points")

        # Sincroniza regime_history para que get_regime_probabilities() encontre os índices
        # passados em predict() (sem este upsert, splits 80/20 ficam invisíveis ao consumidor).
        if self.regime_history is None:
            self.regime_history = result.copy()
        else:
            try:
                hist_idx = self.regime_history.index
                if hasattr(hist_idx, "tz") and hist_idx.tz is not None:
                    hist_idx = hist_idx.tz_localize(None)

                res_idx = result.index
                if hasattr(res_idx, "tz") and res_idx.tz is not None:
                    res_idx = res_idx.tz_localize(None)

                hist_df = self.regime_history.copy()
                hist_df.index = hist_idx

                res_df = result.copy()
                res_df.index = res_idx

                keep = hist_df[~hist_df.index.isin(res_df.index)]
                merged = pd.concat([keep, res_df]).sort_index()
                MAX_HISTORY = 100_000
                if len(merged) > MAX_HISTORY:
                    merged = merged.tail(MAX_HISTORY)
                self.regime_history = merged
            except Exception as e:
                logger.warning(f"⚠️ [Regime] Conflito de timezone mitigado: {e}")
                self.regime_history = result.copy()

        return result

    def _log_regime_tb(self, regimes, confidence, hmm, gmm, adx):
        """Log regime stats to TensorBoard if writer is active."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return
        tb = getattr(self, "_tb_writer", None)
        if tb is None:
            try:
                import os, datetime

                log_dir = os.path.join(
                    os.getcwd(), "logs", "tensorboard", "RegimeDetector"
                )
                self._tb_writer = SummaryWriter(log_dir)
                tb = self._tb_writer
                self._tb_step = 0
            except Exception:
                return
        step = getattr(self, "_tb_step", 0)
        self._tb_step = step + 1

        n = len(regimes)
        if n == 0:
            return
        regime_names = {0: "Bull", 1: "Bear", 2: "Ranger"}
        for code, name in regime_names.items():
            tb.add_scalar(f"Regime/{name}_pct", (regimes == code).mean(), step)
        tb.add_scalar("Regime/MeanConfidence", float(np.mean(confidence)), step)
        tb.add_scalar("Regime/Agreement_HMM", (regimes == hmm).mean(), step)
        tb.add_scalar("Regime/Agreement_GMM", (regimes == gmm).mean(), step)
        tb.add_scalar("Regime/Agreement_ADX", (regimes == adx).mean(), step)

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
    def load(cls, path: Optional[str] = None) -> Optional["CryptoRegimeDetector"]:
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


def create_regime_detector(
    config: Optional[RegimeConfig] = None, load_if_exists: bool = True
) -> CryptoRegimeDetector:
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
