"""Supervised edge model that feeds the reinforcement learning specialists.

Why this exists
---------------
A SAC policy searching directly over ~400 noisy columns has to discover a
trading edge and a control policy at the same time, from 90k bars. That is the
regime where model-free RL is weakest. Splitting the problem is the standard
fix: a supervised model estimates *whether an edge exists right now*, and the
agent decides *what to do about it* — size, timing and exit.

Labels
------
Triple barrier. From the close of bar t, a hypothetical position runs until it
touches a profit target, a stop, or a holding limit, whichever comes first. The
barriers are set in ATR units so they mean the same thing across volatility
regimes. Round-trip cost is subtracted, so a label of 1 means the trade made
money *after fees*, which is the only definition that matters.

Longs and shorts are labelled separately rather than assumed symmetric: funding
and the aggressor imbalance are directional, so the two sides genuinely differ.

Leakage control
---------------
Barrier labels overlap: the label for bar t may depend on prices also used by
the label for t+1. Training and validating naively on overlapping labels leaks
future information and inflates every score. Two defences are applied.

* Purged, embargoed splits. Any training sample whose label window touches the
  validation window is dropped, plus an embargo either side.
* Sample weights by uniqueness. A bar whose label window is shared with many
  others counts proportionally less.

The features handed to the agent are produced walk-forward: each block is
predicted by a model fitted only on data that closed before the block began.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised implicitly by the trainer
    from utils.logger import get_logger

    logger = get_logger("MetaLabeler")
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("MetaLabeler")

ROUND_TRIP_COST = 0.0010  # 0.04% taker each side plus ~1bp slippage each side


@dataclass
class BarrierConfig:
    """Geometry of the triple barrier, in ATR units and bars."""

    profit_atr: float = 2.0
    stop_atr: float = 1.5
    max_bars: int = 48  # 12h on a 15m grid
    cost: float = ROUND_TRIP_COST


@dataclass
class MetaLabelResult:
    features: List[str] = field(default_factory=list)
    report: Dict[str, object] = field(default_factory=dict)


def _first_touch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    cfg: BarrierConfig,
    side: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return net return and holding length for a position opened at each bar.

    The scan walks forward bar by bar and stops at the first barrier touched.
    When a bar's range spans both barriers the stop is assumed to hit first;
    assuming the target would make every backtest built on these labels
    optimistic, and being wrong in the pessimistic direction is the only safe
    choice for a trading system.
    """
    n = len(close)
    out_ret = np.full(n, np.nan, dtype=np.float64)
    out_len = np.zeros(n, dtype=np.int32)

    for i in range(n):
        entry = close[i]
        unit = atr[i]
        if not np.isfinite(entry) or not np.isfinite(unit) or unit <= 0.0:
            continue
        if side > 0:
            target = entry + cfg.profit_atr * unit
            stop = entry - cfg.stop_atr * unit
        else:
            target = entry - cfg.profit_atr * unit
            stop = entry + cfg.stop_atr * unit

        last = min(i + cfg.max_bars, n - 1)
        exit_price = close[last]
        exit_len = last - i
        for j in range(i + 1, last + 1):
            if side > 0:
                hit_stop = low[j] <= stop
                hit_target = high[j] >= target
            else:
                hit_stop = high[j] >= stop
                hit_target = low[j] <= target
            if hit_stop:
                exit_price = stop
                exit_len = j - i
                break
            if hit_target:
                exit_price = target
                exit_len = j - i
                break

        gross = side * (exit_price - entry) / entry
        out_ret[i] = gross - cfg.cost
        out_len[i] = max(1, exit_len)
    return out_ret, out_len


def build_labels(df: pd.DataFrame, cfg: Optional[BarrierConfig] = None) -> pd.DataFrame:
    """Triple-barrier outcome for a long and for a short at every bar."""
    cfg = cfg or BarrierConfig()
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    if "atr_15m" in df.columns:
        atr = df["atr_15m"].to_numpy(dtype=np.float64)
    else:
        atr = (close * 0.005).astype(np.float64)
    atr = np.where(np.isfinite(atr) & (atr > 0), atr, close * 0.005)

    long_ret, long_len = _first_touch(high, low, close, atr, cfg, +1)
    short_ret, short_len = _first_touch(high, low, close, atr, cfg, -1)

    out = pd.DataFrame(index=df.index)
    out["long_ret"] = long_ret
    out["short_ret"] = short_ret
    out["long_len"] = long_len
    out["short_len"] = short_len
    out["y_long"] = (long_ret > 0).astype(np.int8)
    out["y_short"] = (short_ret > 0).astype(np.int8)
    # The label window is what purging and uniqueness weighting operate on.
    out["span"] = np.maximum(long_len, short_len)
    return out


def uniqueness_weights(span: np.ndarray) -> np.ndarray:
    """Down-weight bars whose label window is shared with many neighbours.

    Concurrency is counted with a difference array, so the cost is linear in the
    number of bars rather than quadratic.
    """
    n = len(span)
    marks = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        end = min(n, i + int(span[i]) + 1)
        marks[i] += 1.0
        marks[end] -= 1.0
    concurrency = np.cumsum(marks[:-1])
    concurrency = np.maximum(concurrency, 1.0)

    weights = np.zeros(n, dtype=np.float64)
    inverse = 1.0 / concurrency
    cumulative = np.concatenate([[0.0], np.cumsum(inverse)])
    for i in range(n):
        end = min(n, i + int(span[i]) + 1)
        weights[i] = (cumulative[end] - cumulative[i]) / max(1, end - i)
    mean = weights.mean()
    return weights / mean if mean > 0 else np.ones(n)


# Families of predictors the live bot can reproduce exactly from its 852-bar
# window. Selection is by construction, not by a correlation test, because the
# failure to avoid is history dependence and that does not show up as price
# correlation. Measured on this dataset: obv_15m, pvt_1h and adl_15m are
# cumulative series (adl_15m runs from -8.2e4 to +3.6e6 across the file) yet
# are barely correlated with close; ema_200_5m is a dollar price level with
# correlation 0.75, below any sane cut-off. A model trained on either reads a
# value live that depends on where the live window happens to start.
_LIVE_REPRODUCIBLE_PREFIXES = ("cz_", "tp_")
_LIVE_REPRODUCIBLE_EXACT = {"regime", "regime_confidence", "aggressor_imbalance",
                            "taker_buy_ratio", "funding_rate"}
_LIVE_REPRODUCIBLE_FAMILIES = (
    "rsi", "stoch_k", "stoch_d", "williams_r", "cci", "roc", "mfi",
    "adx", "plus_di", "minus_di", "cmf", "bb_width", "atr_percentage",
    "realized_vol", "log_return", "hl_range_pct", "oc_move_pct",
    "hc_wick", "lc_wick", "ema_trend", "aggressor_imbalance",
    "aggressor_delta_z", "taker_buy_ratio", "funding_rate",
)
_HISTORY_DEPENDENT = ("obv", "pvt", "adl", "vwap", "close_reference", "bb_upper",
                      "bb_lower", "bb_middle", "macd_line", "macd_signal", "psar_value",
                      # legacy resample columns with no live counterpart
                      "_tf_")


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Predictors that are stationary AND reproducible live.

    Only bounded oscillators, returns, volatility ratios, flow ratios and the
    causal cz_/tp_ features qualify. Anything expressed as a price level or
    accumulated since the start of the series is excluded, whatever its
    correlation with price.
    """
    numeric = df.select_dtypes(include=[np.number])
    keep: List[str] = []
    for col in numeric.columns:
        name = str(col)
        if name.startswith("ml_"):
            continue
        # cz_/tp_ come from feature_engineering.causal_features, built with
        # bounded windows by the same code live; cz_vwap_dist, for instance, is
        # a per-bar ratio despite its name.
        if name.startswith(_LIVE_REPRODUCIBLE_PREFIXES):
            allowed = True
        elif any(token in name for token in _HISTORY_DEPENDENT):
            continue
        else:
            allowed = (
                name in _LIVE_REPRODUCIBLE_EXACT
                or any(family in name for family in _LIVE_REPRODUCIBLE_FAMILIES)
            )
        if not allowed:
            continue
        values = numeric[col].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0 or float(finite.std()) <= 0.0:
            continue
        keep.append(name)
    return keep


def purged_splits(
    n: int,
    span: np.ndarray,
    n_splits: int = 5,
    embargo: int = 96,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Contiguous validation blocks with overlapping training labels removed."""
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    index = np.arange(n)
    for k in range(n_splits):
        start, stop = bounds[k], bounds[k + 1]
        validation = index[start:stop]
        # A training bar is dropped when its label window reaches into the
        # validation block, or when it sits inside the embargo after it.
        reaches_in = index + span >= start - 1
        before = index < start
        after = index >= stop + embargo
        train = index[(before & ~reaches_in) | after]
        if len(train) > 500 and len(validation) > 200:
            splits.append((train, validation))
    return splits


# Capacity deliberately kept low. Measured on this dataset with the 8/3/384
# barrier, long side, train versus a 2025-09 to 2026-03 holdout:
#
#   300 iters, depth 5, leaf 200,  316 features -> AUC 0.952 train / 0.560 holdout
#   200 iters, depth 3, leaf 2000,  40 features -> AUC 0.725 train / 0.574 holdout
#
# The second model is far worse at explaining the past and better at the only
# thing that matters. The gap between train and holdout AUC is the honest
# read-out of memorisation, and shrinking it is what raised holdout accuracy.
MODEL_PARAMS = {
    "max_iter": 200,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_samples_leaf": 2000,
    "l2_regularization": 20.0,
}
TOP_FEATURES = 40


def _fit_classifier(X, y, weights, seed: int = 7):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        early_stopping=False, random_state=seed, **MODEL_PARAMS
    )
    model.fit(X, y, sample_weight=weights)
    return model


def rank_features(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    embargo: int,
    top_n: int = TOP_FEATURES,
    seed: int = 7,
) -> np.ndarray:
    """Column indices ordered by out-of-sample permutation importance.

    Importance is measured on a held-back tail of the supplied block, never on
    the rows used to fit. In-sample importance rewards exactly the columns the
    model memorised, which is the opposite of what feature selection is for.
    """
    from sklearn.inspection import permutation_importance

    split = int(len(y) * 0.8)
    fit_slice = slice(0, split)
    score_slice = slice(split + embargo, len(y))
    if len(y) - (split + embargo) < 500:
        return np.arange(X.shape[1])[:top_n]

    probe = _fit_classifier(X[fit_slice], y[fit_slice], weights[fit_slice], seed=seed)
    importance = permutation_importance(
        probe, X[score_slice], y[score_slice],
        n_repeats=3, random_state=seed, scoring="roc_auc", n_jobs=-1,
    )
    return np.argsort(-importance.importances_mean)[:top_n]


def cross_validate(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: Sequence[str],
    n_splits: int = 5,
    embargo: int = 96,
) -> Dict[str, object]:
    """Purged cross-validation of both sides, scored the way trading cares.

    AUC is reported for calibration, but the decisive number is the average net
    return of the trades the model actually selects at its operating threshold:
    a model can have a respectable AUC and still lose money after costs.
    """
    from sklearn.metrics import roc_auc_score

    X = df[list(feature_columns)].to_numpy(dtype=np.float32)
    span = labels["span"].to_numpy()
    weights = uniqueness_weights(span)
    splits = purged_splits(len(df), span, n_splits=n_splits, embargo=embargo)

    report: Dict[str, object] = {"folds": [], "n_splits": len(splits)}
    for side in ("long", "short"):
        y = labels["y_%s" % side].to_numpy()
        ret = labels["%s_ret" % side].to_numpy()
        aucs, selected_returns, coverage = [], [], []
        for train_idx, val_idx in splits:
            ok = np.isfinite(ret[train_idx])
            model = _fit_classifier(X[train_idx][ok], y[train_idx][ok], weights[train_idx][ok])
            proba = model.predict_proba(X[val_idx])[:, 1]
            valid = np.isfinite(ret[val_idx])
            if valid.sum() < 50 or len(np.unique(y[val_idx][valid])) < 2:
                continue
            aucs.append(float(roc_auc_score(y[val_idx][valid], proba[valid])))
            picked = valid & (proba >= 0.55)
            coverage.append(float(picked.sum() / max(1, valid.sum())))
            selected_returns.append(float(ret[val_idx][picked].mean()) if picked.sum() else 0.0)
        report[side] = {
            "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "auc_std": float(np.std(aucs)) if aucs else float("nan"),
            "net_return_at_0.55": float(np.mean(selected_returns)) if selected_returns else float("nan"),
            "coverage_at_0.55": float(np.mean(coverage)) if coverage else float("nan"),
            "base_rate": float(np.mean(y[np.isfinite(ret)])),
        }
    return report


def walk_forward_predict(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: Sequence[str],
    warmup: int = 20000,
    step: int = 5000,
    embargo: int = 96,
) -> pd.DataFrame:
    """Out-of-sample probabilities for every bar past the warm-up.

    Each block is scored by a model fitted only on bars whose label window had
    already closed before the block started, so these columns can be fed to the
    RL agent without leaking anything into it.
    """
    X = df[list(feature_columns)].to_numpy(dtype=np.float32)
    span = labels["span"].to_numpy()
    weights = uniqueness_weights(span)
    n = len(df)

    out = pd.DataFrame(
        {"ml_p_long": np.full(len(df), np.nan), "ml_p_short": np.full(len(df), np.nan)},
        index=df.index,
    )

    # Feature selection happens once, on the warm-up block alone, and is then
    # frozen. Re-selecting per block using later data would let the choice of
    # columns carry information from bars the model is about to be scored on.
    warmup_slice = np.arange(0, min(warmup, n))
    resolved = warmup_slice[np.isfinite(labels["long_ret"].to_numpy()[warmup_slice])]
    selected = rank_features(
        X[resolved], labels["y_long"].to_numpy()[resolved], weights[resolved], embargo
    )
    X = X[:, selected]
    logger.info("[META] %d features selecionadas no bloco de aquecimento", len(selected))

    start = warmup
    while start < n:
        stop = min(n, start + step)
        # Only bars whose barrier already resolved before the block opened.
        train_idx = np.arange(0, start)
        resolved = train_idx + span[train_idx] < start - embargo
        train_idx = train_idx[resolved]
        if len(train_idx) < 2000:
            start = stop
            continue
        block = np.arange(start, stop)
        for side in ("long", "short"):
            y = labels["y_%s" % side].to_numpy()
            ret = labels["%s_ret" % side].to_numpy()
            ok = np.isfinite(ret[train_idx])
            idx = train_idx[ok]
            if len(np.unique(y[idx])) < 2:
                continue
            model = _fit_classifier(X[idx], y[idx], weights[idx])
            out.iloc[block, out.columns.get_loc("ml_p_%s" % side)] = model.predict_proba(X[block])[:, 1]
        logger.info("[META] walk-forward %d..%d fitted on %d bars", start, stop, len(train_idx))
        start = stop

    # A bar with no model behind it must look like "no opinion", never like a
    # confident zero.
    out["ml_p_long"] = out["ml_p_long"].astype(float).fillna(0.5)
    out["ml_p_short"] = out["ml_p_short"].astype(float).fillna(0.5)
    out["ml_edge"] = out["ml_p_long"] - out["ml_p_short"]
    out["ml_conf"] = out[["ml_p_long", "ml_p_short"]].max(axis=1)
    out = out.astype(np.float32)
    out.attrs["selected_features"] = [list(feature_columns)[i] for i in selected]
    return out


def fit_final_bundle(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    selected_features: Sequence[str],
    cfg: "BarrierConfig",
    embargo: int,
) -> Dict[str, object]:
    """Models for LIVE use, fitted on every bar whose barrier has resolved.

    Never use this to score historical rows: it has seen them. The training
    columns come from walk_forward_predict; this bundle only exists so the bot
    can produce the same four columns in real time.
    """
    X = df[list(selected_features)].to_numpy(dtype=np.float32)
    span = labels["span"].to_numpy()
    weights = uniqueness_weights(span)
    n = len(df)
    index = np.arange(n)
    resolved = index + span < n - embargo
    bundle: Dict[str, object] = {
        "features": list(selected_features),
        "params": dict(MODEL_PARAMS),
        "barrier": {"profit_atr": cfg.profit_atr, "stop_atr": cfg.stop_atr, "max_bars": cfg.max_bars},
        "trained_until": str(df.index[resolved][-1]) if resolved.any() else None,
        "rows": int(resolved.sum()),
    }
    for side in ("long", "short"):
        ret = labels["%s_ret" % side].to_numpy()
        ok = resolved & np.isfinite(ret)
        y = labels["y_%s" % side].to_numpy()
        bundle[side] = _fit_classifier(X[ok], y[ok], weights[ok])
    return bundle


def predict_bundle(bundle: Dict[str, object], frame: pd.DataFrame) -> pd.DataFrame:
    """The four ml_* columns for live rows. Missing inputs fail loudly."""
    features = list(bundle["features"])
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise KeyError("meta-modelo sem as colunas de entrada: %s" % missing[:10])
    X = frame[features].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    out = pd.DataFrame(index=frame.index)
    out["ml_p_long"] = bundle["long"].predict_proba(X)[:, 1]
    out["ml_p_short"] = bundle["short"].predict_proba(X)[:, 1]
    out["ml_edge"] = out["ml_p_long"] - out["ml_p_short"]
    out["ml_conf"] = out[["ml_p_long", "ml_p_short"]].max(axis=1)
    return out.astype(np.float32)
