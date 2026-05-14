"""
Deflated Sharpe Ratio & Probabilistic Sharpe Ratio Utilities.

References:
- Bailey & Lopez de Prado (2012): "The Sharpe Ratio Efficient Frontier"
- Harvey & Liu (2015): "Backtesting"
- Lopez de Prado (2018): "Advances in Financial Machine Learning", Ch.12
"""

import numpy as np
from scipy import stats


def deflated_sharpe_ratio(
    sharpe: float,
    returns: np.ndarray,
    n_trials: int,
    skew: float = None,
    kurtosis: float = None,
) -> float:
    """
    Compute the Deflated Sharpe Ratio (Harvey & Liu 2015, Eq.5).

    DSR corrects for selection bias under multiple testing:
    given N trials, what is the probability that the observed Sharpe
    is statistically significant after accounting for the fact that
    we selected the best among N?

    Args:
        sharpe: Observed Sharpe ratio
        returns: Array of returns used to compute the Sharpe
        n_trials: Number of trials/searches performed
        skew: Return skewness (auto-computed if None)
        kurtosis: Return kurtosis (auto-computed if None)

    Returns:
        DSR value in [0, 1], representing PSR * (1 - selection_bias_correction)
    """
    if len(returns) < 10:
        return 0.0

    if skew is None:
        skew = float(stats.skew(returns, bias=False))
    if kurtosis is None:
        kurtosis = float(stats.kurtosis(returns, bias=False))

    psr = probabilistic_sharpe_ratio(sharpe, returns, skew, kurtosis)
    expected_max_sharpe = _expected_max_sharpe(n_trials, len(returns))

    dsr = psr * max(0.0, 1.0 - expected_max_sharpe / max(abs(sharpe), 1e-6))
    return float(np.clip(dsr, 0.0, 1.0))


def probabilistic_sharpe_ratio(
    sharpe: float,
    returns: np.ndarray,
    skew: float = None,
    kurtosis: float = None,
    benchmark: float = 0.0,
) -> float:
    """
    Compute Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012).

    PSR = Probability that the true Sharpe exceeds a benchmark.

    Uses the Edgeworth expansion to account for non-normality.

    Args:
        sharpe: Observed Sharpe ratio
        returns: Return series
        skew: Return skewness
        kurtosis: Return kurtosis
        benchmark: Reference Sharpe (0.0 = test against zero)

    Returns:
        PSR in [0, 1]
    """
    if len(returns) < 10:
        return 0.0

    n = len(returns)

    if skew is None:
        skew = float(stats.skew(returns, bias=False))
    if kurtosis is None:
        kurtosis = float(stats.kurtosis(returns, bias=False))

    sr_se = np.sqrt(
        (
            1.0
            + 0.5 * sharpe**2
            - kurtosis * sharpe / 3.0
            + (kurtosis - 3.0) * sharpe**2 / 12.0
        )
        / n
    )
    z_stat = (sharpe - benchmark) / max(sr_se, 1e-6)

    psr = float(stats.norm.cdf(z_stat))
    return float(np.clip(psr, 0.0, 1.0))


def _expected_max_sharpe(n_trials: int, n_samples: int) -> float:
    """
    Expected maximum Sharpe under the null hypothesis (no skill).
    Uses extreme value theory approximation from De Prado (2018) Ch.12.

    E[max(Sharpe)] ≈ σ_sharpe × sqrt(2 × ln(n_trials))
    where σ_sharpe ≈ 1 / sqrt(n_samples) under IID normality.
    """
    if n_trials <= 1:
        return 0.0

    sigma_sharpe = 1.0 / np.sqrt(max(n_samples, 1))
    evt_factor = np.sqrt(2.0 * np.log(max(n_trials, 2)))
    kurtosis_adj = _kurtosis_adjustment(n_trials)

    return float(sigma_sharpe * evt_factor * kurtosis_adj)


def _kurtosis_adjustment(n_trials: int) -> float:
    """
    Adjust extreme value theory factor for finite samples.
    For n_trials > 1, adds a small penalty.
    """
    if n_trials <= 1:
        return 1.0
    e = np.euler_gamma
    return float(1.0 - e / (2.0 * np.sqrt(n_trials)))
