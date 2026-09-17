from types import SimpleNamespace
import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from models.trade_schema import Action, Signal
from specialists.base_regime_specialist import BaseRegimeSpecialist
from specialists.trend_specialist import TrendSpecialist


@pytest.mark.parametrize('side,action,closing', [
    (1, Action.SELL, True), (-1, Action.BUY, True),
    (0, Action.SELL, False), (0, Action.BUY, False),
    (1, Action.BUY, False), (-1, Action.SELL, False),
    (float('nan'), Action.SELL, False),
])
def test_opposite_vote_only_closes_existing_position(side, action, closing):
    obs = np.zeros(100)
    obs[-11] = side
    assert TrendSpecialist._is_position_exit(obs, action) == closing


@pytest.mark.parametrize('side,expected', [(0, Action.HOLD), (1, Action.SELL)])
def test_bull_wrapper_allows_sell_to_close_not_to_enter(monkeypatch, side, expected):
    monkeypatch.setattr(TrendSpecialist, 'decide_action', lambda *args, **kwargs:
                        Signal(symbol='BTCUSDT', action=Action.SELL, confidence=1, position_size_pct=1))
    agent = object.__new__(BaseRegimeSpecialist)
    agent.regime_type = 'bull'
    agent.name = 'bull'
    obs = np.zeros(100)
    obs[-11] = side
    assert agent.decide_action(obs, pd.Series(dtype=float)).action == expected


def test_local_gate_delegates_to_same_cloud_evaluator(monkeypatch):
    from cloud import train_agent
    captured = {}

    def evaluate(agent, frame, name, deterministic):
        captured.update(name=name, deterministic=deterministic)
        return dict(num_trades=25, total_return_pct=.1, max_drawdown_pct=.07,
                    profit_factor=1.7, trade_sharpe=.01, sharpe_ratio=1.3)

    monkeypatch.setattr(train_agent, 'evaluate', evaluate)
    agent = object.__new__(BaseRegimeSpecialist)
    agent.model = object()
    agent.regime_type = 'bear'
    frame = pd.DataFrame(index=pd.date_range('2026-01-01', periods=200, freq='15min'))
    result = agent.evaluate(frame)
    assert captured == dict(name='bear', deterministic=True)
    assert result['sharpe_ratio'] == 1.3
    assert result['net_return'] == .1


@pytest.mark.parametrize('quantity', [0.0, -1.0])
def test_stale_long_exit_never_opens_new_short(quantity):
    from trading.execution_engine import ExecutionEngine

    engine = object.__new__(ExecutionEngine)
    engine.portfolio = SimpleNamespace(positions={'BTCUSDT': SimpleNamespace(quantity=quantity)})
    engine._check_position_limit = AsyncMock()
    signal = Signal(symbol='BTCUSDT', action=Action.SELL, confidence=1,
                    position_size_pct=1, explanation={'position_exit': True})
    asyncio.run(engine.submit_order(signal))
    engine._check_position_limit.assert_not_called()
