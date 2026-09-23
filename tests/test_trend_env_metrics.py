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
        # O ambiente preca os stops em ATR (ENV_STOP_ATR_TIMEFRAME, 15m ou 1h).
        "atr_15m": np.full(rows, 0.5),
        "atr_1h": np.full(rows, 1.0),
    }
    data = {**base, **optional}
    return pd.DataFrame(data, index=idx)


class TrendEnvMetricsTest(unittest.TestCase):
    def test_stop_intrabar_rebound_and_gap(self):
        fill = TrendFollowingEnv._resting_stop_fill
        self.assertEqual(fill(1, 95, 100, 110, 94), 95)
        self.assertEqual(fill(1, 95, 90, 100, 85), 90)
        self.assertEqual(fill(-1, 105, 100, 106, 90), 105)
        self.assertEqual(fill(-1, 105, 110, 115, 100), 110)
        self.assertIsNone(fill(1, 95, 100, 110, 96))
        self.assertIsNone(fill(0, 95, 100, 110, 94))

    def test_historical_funding_is_signed_and_only_on_boundary(self):
        cost = TrendFollowingEnv._historical_funding_cost
        before = pd.Timestamp('2024-01-01 07:45:00')
        settlement = pd.Timestamp('2024-01-01 08:00:00')
        self.assertAlmostEqual(cost(1000, 1, .001, before, settlement), 1)
        self.assertAlmostEqual(cost(1000, -1, .001, before, settlement), -1)
        self.assertEqual(cost(1000, 1, .001, settlement, settlement + pd.Timedelta(minutes=15)), 0)

    def test_normal_stop_enabled_during_training(self):
        self.assertFalse(self.env.disable_normal_sl)

    def setUp(self) -> None:
        df = _build_minimal_dataframe()
        self.env = TrendFollowingEnv(df, AIConfig(), mode="training")

    def test_profit_factor_sanity_three_trades(self) -> None:
        """Ensure PF is computed with realistic values for a simple trade series."""
        history = [10.0, -5.0, 3.0]
        snapshot = self.env._build_financial_snapshot(history, [5, 5, 5])
        self.assertAlmostEqual(snapshot["profit_factor"], 2.6, places=6)
        self.assertAlmostEqual(snapshot["profit_factor_raw"], 2.6, places=6)

    def test_closed_trade_result_is_net_of_fees_and_funding(self) -> None:
        """C1: the per-trade result behind PF and win rate pays every cost.

        A trade that gains 0.05% on the price but pays two taker fees and
        funding loses money; it must count as a loss, not as a win.
        """
        env = self.env
        fee = env.trading_config.TAKER_FEE
        notional = 3000.0
        env.net_worth = 1000.0
        env._trade_initial_net_worth = 1000.0            # before the entry fee
        env.net_worth -= notional * fee                    # entry fee
        env.net_worth -= 0.4                               # funding while in position
        env.initial_notional_value = notional
        env.pnl_since_entry = 0.0
        gross = notional * 0.0005
        env._apply_realized_pnl(gross)                     # exit fee charged inside
        expected = gross - 2 * notional * fee - 0.4
        self.assertAlmostEqual(env._last_closed_net_pnl, expected, places=9)
        self.assertLess(env._last_closed_net_pnl, 0.0)
        self.assertAlmostEqual(env.net_worth - 1000.0, expected, places=9)

    def test_profit_factor_uses_dollars_when_given(self) -> None:
        """C1: two +10% trades on a small stake and one -10% trade on a large one
        lose money; the PF must say so (dollars), not 2.0 (percent sums)."""
        snapshot = self.env._build_financial_snapshot([0.10, 0.10, -0.10], [5, 5, 5],
                                                      pnl_usd=[10.0, 10.0, -50.0])
        self.assertAlmostEqual(snapshot["profit_factor"], 0.4, places=6)
        self.assertAlmostEqual(snapshot["win_rate_pct"], 100 * 2 / 3, places=6)

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
