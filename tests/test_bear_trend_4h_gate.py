"""A Bear trained with the 4h gate never opens a short while the 4h structure is up."""
import numpy as np
import pandas as pd

from config.settings import AIConfig
from specialists.bear_specialist import BearTradingEnv


def _frame(rows=300, trend_4h=1.0):
    idx = pd.date_range("2025-01-01", periods=rows, freq="15min")
    close = np.linspace(100, 90, rows)
    return pd.DataFrame({"open": close, "high": close + .5, "low": close - .5, "close": close,
                         "volume": np.ones(rows), "atr_15m": np.full(rows, .5), "atr_1h": np.full(rows, 1.0),
                         "ema_trend": np.full(rows, -1.0), "cz_trend_4h": np.full(rows, trend_4h)}, index=idx)


def _shorts_opened(trend_4h, gate):
    AIConfig.BEAR_REQUIRE_TREND_4H = gate
    try:
        env = BearTradingEnv(df=_frame(trend_4h=trend_4h), config=AIConfig(), mode="training")
        env.reset()
        opened = 0
        for _ in range(60):
            before = env.position
            env.step(np.array([-0.9, 3.0, 3.0], dtype=np.float32))   # strong short vote
            opened += int(before == 0 and env.position < 0)
        return opened
    finally:
        AIConfig.BEAR_REQUIRE_TREND_4H = False


def test_gate_blocks_shorts_against_a_4h_uptrend():
    assert _shorts_opened(trend_4h=1.0, gate=True) == 0


def test_gate_allows_shorts_in_a_4h_downtrend():
    assert _shorts_opened(trend_4h=-1.0, gate=True) >= 1


def test_without_the_gate_the_old_bear_is_unchanged():
    assert _shorts_opened(trend_4h=1.0, gate=False) >= 1
