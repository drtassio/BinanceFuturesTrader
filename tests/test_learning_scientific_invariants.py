"""Scientific invariants for the learning pipeline.

These tests protect causal ordering and the economic identities that must hold
before model profitability can be evaluated.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch.nn as nn

from config.settings import AIConfig
from feature_engineering.crypto_regime_detector import CryptoRegimeDetector, RegimeConfig
from feature_engineering.temporal_autoencoder import TemporalAutoencoderPipeline, TemporalSDAE
from learning.curriculum_engine import CurriculumEngine
from models.trade_schema import Action, Signal
from specialists.base_regime_specialist import BaseRegimeSpecialist
from specialists.ranger_specialist import RangerTradingEnv
from specialists.scientific_corrections import compute_scientific_reward
from specialists.trend_specialist import TrendFollowingEnv, TrendSpecialist


def test_sanitizer_never_backfills_initial_nan(tmp_path):
    config = AIConfig()
    config.MODEL_DIR = str(tmp_path)
    pipeline = TemporalAutoencoderPipeline(config)
    frame = pd.DataFrame({'signal': [np.nan, 7.0, 8.0]})

    clean = pipeline._sanitize_data(frame)

    assert clean.iloc[0, 0] == 0.0
    assert clean.iloc[1, 0] == 7.0


def test_decoder_supports_negative_robust_scaled_targets():
    model = TemporalSDAE(
        input_dim=6,
        cnn_out_channels=12,
        kernel_size=3,
        gru_hidden=8,
        gru_layers=1,
        latent_dim=3,
        seq_length=8,
    )
    assert isinstance(model.decoder_relu, nn.Identity)
    assert isinstance(model.decoder_dropout, nn.Identity)


def test_future_targets_do_not_cross_partition_boundary(tmp_path):
    config = AIConfig()
    config.MODEL_DIR = str(tmp_path)
    pipeline = TemporalAutoencoderPipeline(config)
    full = pd.DataFrame({'close': np.exp(np.arange(40) * 0.01)})

    full_targets = pipeline._build_future_return_targets(full, (4,))
    train_targets = pipeline._build_future_return_targets(full.iloc[:30], (4,))

    assert np.isfinite(full_targets[28, 0])
    assert np.isnan(train_targets[28, 0])


def test_regime_smoothing_is_prefix_invariant():
    detector = CryptoRegimeDetector(RegimeConfig(min_regime_duration=4))
    regimes = np.array([0] * 5 + [1] * 5 + [2] * 5)
    confidence = np.full(len(regimes), 0.60)
    full = detector._smooth_regimes(regimes, confidence)

    for end in range(2, len(regimes) + 1):
        prefix = detector._smooth_regimes(regimes[:end], confidence[:end])
        assert np.array_equal(prefix, full[:end])


def test_funding_stub_cannot_bias_ensemble_to_ranger():
    config = RegimeConfig(
        weight_hmm=0.30,
        weight_gmm=0.30,
        weight_adx=0.30,
        weight_funding=0.10,
        min_regime_duration=1,
    )
    detector = CryptoRegimeDetector(config)
    bull = np.zeros(10, dtype=int)
    ranger = np.full(10, detector.RANGER, dtype=int)

    regimes, confidence = detector._ensemble_voting(bull, bull, bull, ranger)

    assert np.all(regimes == detector.BULL)
    assert np.allclose(confidence, 1.0)


def test_mark_to_market_reconciles_with_realized_pnl():
    env = object.__new__(TrendFollowingEnv)
    env.trading_config = SimpleNamespace(TAKER_FEE=0.0004)
    env.initial_balance = 10_000.0
    env.net_worth = 10_000.0
    env.episode_peak_net_worth = 10_000.0
    env.entry_price = 100.0
    env.initial_notional_value = 1_000.0
    env.pnl_since_entry = 0.05

    prices = [100.0, 101.0, 103.0, 105.0]
    for previous, current in zip(prices[:-1], prices[1:]):
        env.net_worth += (
            (current - previous) / env.entry_price
        ) * env.initial_notional_value

    env._apply_realized_pnl(50.0)

    assert env.net_worth == pytest.approx(10_050.0 - 1_000.0 * 0.0004)


def test_reward_tracks_net_equity_log_return():
    env = SimpleNamespace(
        position=0,
        steps_in_position=0,
        episode_max_drawdown=0.0,
        _prev_scientific_dd=0.0,
    )
    common = dict(
        env=env,
        pnl_realized=0.0,
        trade_return_pct=0.0,
        duration_steps=0,
        exit_reason=None,
        current_price=100.0,
        atr=1.0,
    )
    gain = compute_scientific_reward(**common, info={'economic_step_return': 0.001})
    loss = compute_scientific_reward(**common, info={'economic_step_return': -0.001})

    assert gain > 0.0
    assert loss < 0.0
    assert gain == pytest.approx(100.0 * np.log1p(0.001))
    assert loss == pytest.approx(100.0 * np.log1p(-0.001))


def test_funding_is_charged_only_at_settlement_boundary():
    before = pd.Timestamp('2024-01-01 07:45:00', tz='UTC')
    settlement = pd.Timestamp('2024-01-01 08:00:00', tz='UTC')
    after = pd.Timestamp('2024-01-01 08:15:00', tz='UTC')

    long_cost = TrendFollowingEnv._historical_funding_cost(
        10_000.0, 1.0, 0.0001, before, settlement, 8.0
    )
    no_settlement = TrendFollowingEnv._historical_funding_cost(
        10_000.0, 1.0, 0.0001, settlement, after, 8.0
    )
    short_credit = TrendFollowingEnv._historical_funding_cost(
        10_000.0, -1.0, 0.0001, before, settlement, 8.0
    )

    assert np.isclose(long_cost, 1.0)
    assert no_settlement == 0.0
    assert np.isclose(short_credit, -1.0)


def test_curriculum_preserves_full_chronology_with_eligibility_mask(tmp_path):
    config = AIConfig()
    config.CHECKPOINT_DIR = str(tmp_path)
    config.CURRICULUM_STAGES = [{
        'name': 'causal-stage',
        'volatility_max': 0.02,
        'min_training_runs': 3,
        'max_training_runs': 3,
        'success_criteria': {'sharpe_ratio': 100.0},
    }]
    engine = CurriculumEngine(config)
    col = f'atr_percentage_{engine.primary_timeframe}'
    index = pd.date_range('2024-01-01', periods=1_000, freq='15min', tz='UTC')
    frame = pd.DataFrame({col: np.tile([0.01, 0.03], 500)}, index=index)

    prepared = engine.prepare_training_data(frame)

    assert prepared.index.equals(index)
    assert len(prepared) == len(frame)
    assert prepared['curriculum_eligible'].sum() == 500


def test_curriculum_never_advances_without_oos_acceptance(tmp_path):
    config = AIConfig()
    config.CHECKPOINT_DIR = str(tmp_path)
    config.CURRICULUM_STAGES = [
        {
            'name': 'unproven',
            'min_training_runs': 3,
            'max_training_runs': 3,
            'success_criteria': {'sharpe_ratio': 100.0},
        },
        {'name': 'next', 'success_criteria': {}},
    ]
    engine = CurriculumEngine(config)
    for sharpe in (0.0, 1.0, 2.0):
        engine.update_progress({'sharpe_ratio': sharpe})

    assert engine.current_stage_index == 0


def test_production_specialist_blocks_wrong_way_signal(monkeypatch):
    sell = Signal(
        symbol='BTCUSDT', action=Action.SELL, confidence=0.9,
        position_size_pct=0.5,
    )
    monkeypatch.setattr(TrendSpecialist, 'decide_action', lambda *args, **kwargs: sell)
    specialist = object.__new__(BaseRegimeSpecialist)
    specialist.regime_type = 'bull'
    specialist.name = 'BullSpecialist'

    filtered = specialist.decide_action(np.zeros(1), pd.Series(dtype=float))

    assert filtered.action == Action.HOLD
    assert filtered.confidence == 0.0
    assert filtered.position_size_pct == 0.0


def test_uncalibrated_ranger_exit_rules_are_disabled_by_default():
    env = object.__new__(RangerTradingEnv)
    env.config = SimpleNamespace(ENABLE_RANGER_RULE_BASED_EXITS=False)

    assert env._should_exit_fast(0.50) is False
