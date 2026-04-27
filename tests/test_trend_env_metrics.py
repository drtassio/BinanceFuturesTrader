import os
import sys
import unittest

import numpy as np
import pandas as pd

# Ensure project root is importable when tests run standalone
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config.settings import AIConfig
from specialists.trend_specialist import TrendFollowingEnv


def _build_minimal_dataframe(rows: int = 16) -> pd.DataFrame:
    """Create a minimal dataframe accepted by TrendFollowingEnv."""
    idx = pd.date_range("2022-01-01", periods=rows, freq="T")
    base = {
        "open": np.linspace(100, 101, rows),
        "high": np.linspace(101, 102, rows),
        "low": np.linspace(99, 100, rows),
        "close": np.linspace(100, 101, rows),
        "volume": np.ones(rows),
    }
    # Provide neutral placeholders for optional columns accessed in the env
    optional = {
        "tp_uncertainty": np.zeros(rows),
        "tp_duration_median": np.full(rows, 10.0),
        "tp_regime_up": np.zeros(rows),
        "tp_regime_down": np.zeros(rows),
        "tp_regime_sideways": np.ones(rows),
        "tp_prior_conf": np.full(rows, 0.5),
        "ema_trend": np.zeros(rows),
    }
    data = {**base, **optional}
    return pd.DataFrame(data, index=idx)


class TrendEnvMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        df = _build_minimal_dataframe()
        self.env = TrendFollowingEnv(df, AIConfig(), mode="training")

    def test_profit_factor_sanity_three_trades(self) -> None:
        """Ensure PF is computed with realistic values for a simple trade series."""
        history = [10.0, -5.0, 3.0]
        snapshot = self.env._build_financial_snapshot(history, [5, 5, 5])
        self.assertAlmostEqual(snapshot["profit_factor"], 2.6, places=6)
        self.assertAlmostEqual(snapshot["profit_factor_raw"], 2.6, places=6)

    def test_sharpe_reward_consistent_with_episode_returns(self) -> None:
        """Sharpe reward deve refletir o sinal da média de retornos do episódio atual."""
        # Força retornos do episódio com média positiva
        self.env._current_episode_returns = [0.05, -0.02, 0.03, 0.01]
        pos_reward = self.env._calculate_sharpe_reward()
        self.assertGreaterEqual(pos_reward, 0.0)

        # Agora média negativa
        self.env._current_episode_returns = [-0.05, -0.02, 0.01, -0.01]
        neg_reward = self.env._calculate_sharpe_reward()
        self.assertLessEqual(neg_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
