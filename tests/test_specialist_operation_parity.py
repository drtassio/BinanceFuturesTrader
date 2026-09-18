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


@pytest.mark.parametrize('side,vote,direction,expected', [
    (1, -.08, 'long_only', Action.SELL),
    (-1, .08, 'short_only', Action.BUY),
    (1, -.01, 'long_only', Action.HOLD),
    (-1, .01, 'short_only', Action.HOLD),
    (0, -.08, 'long_only', Action.HOLD),
    (0, .08, 'short_only', Action.HOLD),
])
def test_actor_exit_survives_entry_prior_gating(side, vote, direction, expected):
    agent = object.__new__(TrendSpecialist)
    agent.is_trained = True
    agent.model = SimpleNamespace(predict=lambda *args, **kwargs: (np.array([vote, 2., 2.]), None))
    agent.feature_scaler = None
    agent.config = SimpleNamespace(INFERENCE_ACTION_THRESHOLD=.064, TRAINING_LEVERAGE_CAP=3.)
    agent.trading_config = SimpleNamespace(MIN_LEVERAGE_PER_TRADE=1., MAX_LEVERAGE_PER_TRADE=3.,
        MAX_POSITION_SIZE_PERCENT=.2, PRIMARY_PAIR='BTCUSDT', DEFAULT_STOP_LOSS_PCT=.02,
        DEFAULT_TAKE_PROFIT_PCT=.04)
    agent._specialist_direction = direction
    obs = np.zeros(100)
    obs[-11] = side
    row = pd.Series(dict(tp_prior_dir=1 if vote < 0 else -1, tp_prior_conf=.01,
                         tp_uncertainty=1., tp_regime_sideways=1., close=100., atr=1.))
    signal = agent.decide_action(obs, row)
    assert signal is not None
    assert signal.action == expected
    if expected != Action.HOLD:
        assert signal.position_size_pct == 1.
        assert signal.explanation['position_exit'] is True


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


@pytest.mark.parametrize('quantity,action,weight,flag,preserved', [
    (1, Action.SELL, .6, True, True),
    (-1, Action.BUY, .6, True, True),
    (0, Action.SELL, .6, True, False),
    (-1, Action.SELL, .6, True, False),
    (1, Action.HOLD, .6, True, False),
    (1, Action.SELL, 0, True, False),
    (1, Action.SELL, .6, False, False),
    (float('nan'), Action.SELL, .6, True, False),
])
def test_ensemble_preserves_only_agreed_live_policy_exit(quantity, action, weight, flag, preserved):
    from trading.policy_exit import preserve_policy_exit
    aggregate = Signal(symbol='BTCUSDT', action=action, confidence=.7, position_size_pct=.2)
    expert = Signal(symbol='BTCUSDT', action=action, confidence=1,
                    explanation={'position_exit': flag})
    result = preserve_policy_exit(aggregate, [(expert, weight, 'Bull', 'bull')], quantity)
    assert bool(result.explanation.get('position_exit')) == preserved
    assert result.position_size_pct == (1. if preserved else .2)


@pytest.mark.parametrize('side,action', [(1., Action.SELL), (-1., Action.BUY)])
def test_learned_close_queue_uses_exact_quantity_and_reduce_only(side, action):
    from trading.execution_engine import ExecutionEngine
    engine = object.__new__(ExecutionEngine)
    engine.portfolio = SimpleNamespace(
        positions={'BTCUSDT': SimpleNamespace(quantity=side * .123)},
        get_total_value=lambda: 10000., get_current_price=lambda symbol: 50000.)
    engine._check_position_limit = AsyncMock(return_value=False)
    engine._choose_smart_strategy = lambda signal: 'MARKET'
    engine._get_cached_symbol_info = AsyncMock(return_value={'quantityPrecision': 3})
    engine.active_orders = {}
    engine.system_state = {}
    engine.order_queue = asyncio.Queue()
    signal = Signal(symbol='BTCUSDT', action=action, confidence=1.,
                    position_size_pct=1., leverage=3., explanation={'position_exit': True})
    asyncio.run(engine.submit_order(signal))
    order = engine.order_queue.get_nowait()
    assert order.total_quantity == .123
    assert order.reduce_only is True


def test_position_disappearing_during_validation_cannot_open_reverse_trade():
    from trading.execution_engine import ExecutionEngine
    engine = object.__new__(ExecutionEngine)
    engine.portfolio = SimpleNamespace(positions={'BTCUSDT': SimpleNamespace(quantity=1.)})
    async def refresh(symbol):
        engine.portfolio.positions.clear()
        return False
    engine._check_position_limit = refresh
    engine._choose_smart_strategy = lambda signal: pytest.fail('stale close reached entry sizing')
    signal = Signal(symbol='BTCUSDT', action=Action.SELL, confidence=1.,
                    position_size_pct=1., explanation={'position_exit': True})
    asyncio.run(engine.submit_order(signal))
