"""
Meta-Labeling Filter for Trading Decisions.

Based on Lopez de Prado (2018) "Advances in Financial ML", Chapter 3.
The primary model (RL agent) determines SIDE (long/short).
The meta-labeler determines whether to EXECUTE.

Only trades with P(profit) > threshold pass the filter.

Model: LogisticRegression (lightweight, interpretable, no GPU needed).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os


class MetaLabelingFilter:
    """
    Binary classifier that predicts P(trade_profitable).

    Training data:
        X = features at trade ENTRY time
        y = 1 if trade was profitable, 0 otherwise

    Inference:
        P(profit | entry_features) > threshold → EXECUTE trade
    """

    def __init__(
        self,
        profit_threshold: float = 0.55,
        min_samples: int = 50,
        model_dir: str = None,
    ):
        self.profit_threshold = profit_threshold
        self.min_samples = min_samples
        self.model_dir = model_dir or os.path.join(os.getcwd(), "models_ai")
        os.makedirs(self.model_dir, exist_ok=True)

        self.model: LogisticRegression = None
        self.scaler: StandardScaler = None
        self.is_trained = False
        self.n_samples_seen = 0
        self.feature_names = None

    def update(
        self,
        entry_features: np.ndarray,
        trade_outcomes: np.ndarray,
        feature_names: list = None,
    ) -> bool:
        """
        Incrementally train the meta-labeler on trade outcomes.

        Args:
            entry_features: (N, D) features at trade entry
            trade_outcomes: (N,) binary (1 = profit, 0 = loss)
            feature_names: Optional list of feature column names

        Returns:
            True if model was (re)trained successfully
        """
        if len(entry_features) < self.min_samples:
            self.n_samples_seen = len(entry_features)
            return False

        X = np.asarray(entry_features, dtype=np.float64)
        y = np.asarray(trade_outcomes, dtype=np.int64)

        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.1,
            solver="lbfgs",
            random_state=42,
        )
        self.model.fit(X_scaled, y)

        self.is_trained = True
        self.n_samples_seen = len(entry_features)
        self.feature_names = feature_names
        self._save()
        return True

    def should_execute(self, entry_features: np.ndarray) -> tuple:
        """
        Check if a trade should be executed based on predicted profitability.

        Args:
            entry_features: (D,) feature vector at trade entry

        Returns:
            (should_execute: bool, probability: float)
        """
        if not self.is_trained or self.model is None:
            return True, 0.5

        X = np.asarray(entry_features, dtype=np.float64).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        proba = self.model.predict_proba(X)[0, 1]
        proba = float(np.clip(proba, 0.0, 1.0))

        return proba >= self.profit_threshold, proba

    def _save(self):
        """Persist model to disk."""
        path = os.path.join(self.model_dir, "meta_labeling_filter.pkl")
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, model_dir: str = None, **kwargs):
        """Load persisted model from disk."""
        model_dir = model_dir or os.path.join(os.getcwd(), "models_ai")
        path = os.path.join(model_dir, "meta_labeling_filter.pkl")
        obj = cls(model_dir=model_dir, **kwargs)
        if os.path.exists(path):
            data = joblib.load(path)
            obj.model = data.get("model")
            obj.scaler = data.get("scaler")
            obj.feature_names = data.get("feature_names")
            obj.is_trained = obj.model is not None
        return obj
