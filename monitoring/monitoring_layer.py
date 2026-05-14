"""
Detecção de drift de modelo em produção.
"""

from collections import deque
from typing import Dict, Any, Optional
import numpy as np
from scipy import stats

from utils.logger import get_logger

logger = get_logger("MonitoringLayer")


class DriftMonitor:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.train_distribution: Optional[np.ndarray] = None
        self.live_buffer: deque = deque(maxlen=window_size)
        self.alerts_fired: int = 0

    def set_train_baseline(self, train_features: np.ndarray):
        self.train_distribution = train_features.copy()
        logger.info(
            f"📊 [DRIFT] Baseline salvo: shape={train_features.shape}, "
            f"mean={train_features.mean():.4f}, std={train_features.std():.4f}"
        )

    def add_live_observation(self, features: np.ndarray):
        self.live_buffer.append(features.flatten())

    def detect_drift(self, alpha: float = 0.01) -> Dict[str, Any]:
        if self.train_distribution is None:
            return {"status": "no_baseline"}
        if len(self.live_buffer) < self.window_size // 2:
            return {"status": "insufficient_data", "n_obs": len(self.live_buffer)}

        live_arr = np.array(list(self.live_buffer))
        n_features = live_arr.shape[1]
        drift_per_feature = []

        for i in range(n_features):
            train_col = (
                self.train_distribution[:, i]
                if self.train_distribution.ndim > 1
                else self.train_distribution
            )
            live_col = live_arr[:, i]
            try:
                _stat, pvalue = stats.ks_2samp(train_col, live_col)
                drift_per_feature.append(
                    {
                        "feature_idx": i,
                        "pvalue": float(pvalue),
                        "drifted": pvalue < alpha,
                    }
                )
            except Exception:
                pass

        n_drifted = sum(1 for d in drift_per_feature if d["drifted"])
        drift_pct = n_drifted / max(n_features, 1)

        result = {
            "status": "ok",
            "n_drifted_features": n_drifted,
            "drift_pct": drift_pct,
            "alert": drift_pct > 0.20,
        }

        if result["alert"]:
            self.alerts_fired += 1
            logger.warning(
                f"🚨 [DRIFT] {drift_pct:.1%} das features com KS test p<{alpha}. "
                "Considere retreino do modelo."
            )
        return result
