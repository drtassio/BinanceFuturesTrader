import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd

from config.settings import TradingConfig
from data_provider import DataProvider
from models.trade_schema import Action, Signal
from trading.ai_controller import AIController
from trading.binance_connector import BinanceConnector
from trading.risk_manager import RiskManager


def test_production_data_connector_is_read_only():
    connector = object.__new__(BinanceConnector)
    connector.allow_order_execution = False

    result = asyncio.run(
        connector._make_request(
            'POST', '/fapi/v1/order',
            params={'symbol': 'BTCUSDT'}, signed=True,
        )
    )

    assert result is None


class _FakeResponse:
    def __init__(self, status, body, headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def text(self):
        return self._body


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.closed = False
        self.calls = 0

    def request(self, *_args, **_kwargs):
        self.calls += 1
        return self.responses.pop(0)


def test_idempotent_get_retries_after_rate_limit():
    connector = object.__new__(BinanceConnector)
    connector.allow_order_execution = True
    connector.session = _FakeSession([
        _FakeResponse(429, '{"code": -1003}', {'Retry-After': '0'}),
        _FakeResponse(200, '{"serverTime": 123}'),
    ])
    connector.base_url = 'https://example.invalid'
    connector.trading_config = SimpleNamespace(EXECUTION_TIMEOUT_SECONDS=1)
    connector._request_lock = asyncio.Lock()
    connector._last_request_monotonic = 0.0
    connector._minimum_request_interval_seconds = 0.0
    connector._max_get_retries = 2
    connector.api_key = 'test'
    connector.api_secret = 'test'
    connector._timestamp_offset = 0

    with patch('trading.binance_connector.asyncio.sleep', new=AsyncMock()):
        result = asyncio.run(connector._make_request('GET', '/fapi/v1/time'))

    assert result == {'serverTime': 123}
    assert connector.session.calls == 2


def _coverage_provider():
    provider = object.__new__(DataProvider)
    provider.data_config = SimpleNamespace(
        HISTORICAL_MIN_COVERAGE_RATIO=0.995,
        HISTORICAL_MAX_GAP_MULTIPLIER=3.0,
    )
    return provider


def test_historical_coverage_accepts_complete_window():
    provider = _coverage_provider()
    start = pd.Timestamp('2025-01-01', tz='UTC').to_pydatetime()
    end = pd.Timestamp('2025-01-02', tz='UTC').to_pydatetime()
    frame = pd.DataFrame(
        {'close': range(96)},
        index=pd.date_range(start, periods=96, freq='15min'),
    )

    valid, report = provider._validate_historical_coverage(
        frame, '15m', start, end
    )

    assert valid is True
    assert report['coverage_ratio'] == 1.0


def test_historical_coverage_rejects_truncated_window():
    provider = _coverage_provider()
    start = pd.Timestamp('2025-01-01', tz='UTC').to_pydatetime()
    end = pd.Timestamp('2025-01-02', tz='UTC').to_pydatetime()
    frame = pd.DataFrame(
        {'close': range(12)},
        index=pd.date_range(start, periods=12, freq='15min'),
    )

    valid, report = provider._validate_historical_coverage(
        frame, '15m', start, end
    )

    assert valid is False
    assert report['end_ok'] is False
    assert report['coverage_ratio'] < 0.995


def test_profit_probability_gate_is_disabled_by_default():
    manager = object.__new__(RiskManager)
    manager.config = SimpleNamespace(
        MIN_CONFIDENCE_FOR_TRADE=0.60,
        MIN_PROFIT_PROBABILITY=0.90,
        ENABLE_PROFIT_PROBABILITY_GATE=False,
    )
    signal = Signal(
        symbol='BTCUSDT', action=Action.BUY, confidence=0.80,
        position_size_pct=0.01, profit_probability=0.50,
    )

    approved, _ = manager.validate_signal_quality(signal)

    assert approved is True


def test_explicit_probability_gate_still_fails_closed():
    manager = object.__new__(RiskManager)
    manager.config = SimpleNamespace(
        MIN_CONFIDENCE_FOR_TRADE=0.60,
        MIN_PROFIT_PROBABILITY=0.60,
        ENABLE_PROFIT_PROBABILITY_GATE=True,
    )
    signal = Signal(
        symbol='BTCUSDT', action=Action.BUY, confidence=0.80,
        position_size_pct=0.01, profit_probability=0.50,
    )

    approved, _ = manager.validate_signal_quality(signal)

    assert approved is False


class _EvaluatedSpecialist:
    def __init__(self, metrics):
        self.metrics = metrics

    def evaluate(self, _df):
        return dict(self.metrics)


def _controller_for_oos_gate(tmp_path: Path, metrics):
    config = SimpleNamespace(
        MODEL_DIR=str(tmp_path),
        OOS_MIN_SHARPE=0.50,
        OOS_MIN_PROFIT_FACTOR=1.10,
        OOS_MAX_DRAWDOWN=0.15,
        OOS_MIN_NET_RETURN=0.0,
        OOS_MIN_TRADES=20,
        REQUIRE_OOS_POLICY_APPROVAL=True,
    )
    controller = object.__new__(AIController)
    controller.config_ai = config
    controller.policy_validation_path = str(tmp_path / 'policy_oos_validation.json')
    controller.policy_oos_approved = False
    controller.last_oos_validation = {}
    controller.specialists = {
        name: _EvaluatedSpecialist(metrics[name])
        for name in ('bull', 'bear', 'ranger')
    }
    for name in ('bull', 'bear', 'ranger'):
        (tmp_path / f'{name}_specialist_scaler.joblib').write_bytes(b'scaler')
        (tmp_path / f'{name}_feature_contract.json').write_text('{}')
        (tmp_path / f'{name}_specialist_sac.zip').write_bytes(
            (name * 100).encode('ascii')
        )
    return controller


def test_oos_gate_approves_and_binds_report_to_model_hashes(tmp_path):
    passing = {
        name: {
            'sharpe_ratio': 0.80,
            'max_drawdown': 0.10,
            'profit_factor': 1.30,
            'num_trades': 25,
            'net_return': 0.05,
        }
        for name in ('bull', 'bear', 'ranger')
    }
    controller = _controller_for_oos_gate(tmp_path, passing)
    frame = pd.DataFrame(index=pd.date_range('2024-01-01', periods=100, freq='15min'))

    report = controller._validate_specialists_oos(frame)

    assert report['all_passed'] is True
    assert controller._load_policy_oos_approval() is True

    (tmp_path / 'bear_specialist_sac.zip').write_bytes(b'changed-policy')
    assert controller._load_policy_oos_approval() is False


def test_oos_gate_rejects_unprofitable_specialist(tmp_path):
    passing_metrics = {
        'sharpe_ratio': 0.80,
        'max_drawdown': 0.10,
        'profit_factor': 1.30,
        'num_trades': 25,
        'net_return': 0.05,
    }
    metrics = {name: dict(passing_metrics) for name in ('bull', 'bear', 'ranger')}
    metrics['ranger']['net_return'] = -0.01
    controller = _controller_for_oos_gate(tmp_path, metrics)
    frame = pd.DataFrame(index=pd.date_range('2024-01-01', periods=100, freq='15min'))

    report = controller._validate_specialists_oos(frame)

    assert report['all_passed'] is False
    assert report['specialists']['ranger']['checks']['net_return'] is False
