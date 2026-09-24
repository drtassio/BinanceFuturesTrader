"""Regression tests for the four defects that made the specialists untrainable.

Each test asserts a behaviour, not the presence of a line of code, so a future
refactor is free to reshape the implementation and still be held to the same
contract. Every one of these failed before the fixes.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TREND_SKIP_OPTUNA", "1")

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "featured_data_causal.parquet"

pytestmark = pytest.mark.skipif(
    not DATASET.exists(),
    reason="run scripts/build_causal_dataset.py first",
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    df = pd.read_parquet(DATASET).sort_index()
    return df.iloc[-4000:-2000].copy().ffill().fillna(0.0)


def _environment(frame: pd.DataFrame, economic_only: bool = True, env_class=None,
                 specialist: str = "bull_specialist"):
    from config.settings import AIConfig
    from specialists.bull_specialist import BullTradingEnv

    config = AIConfig()
    config.ECONOMIC_REWARD_ONLY = economic_only
    env = (env_class or BullTradingEnv)(
        df=frame, config=config, mode="training", specialist_name=specialist
    )
    env.max_steps = len(frame) - 2
    env.reset()
    return env


def _drive(env, vote: float, steps: int, sl_mult: float = 3.0, leverage: float = 2.0):
    """Run a scripted policy and collect what the environment did."""
    total_reward = 0.0
    exits, durations = [], []
    for _ in range(min(steps, env.max_steps)):
        _, reward, done, truncated, info = env.step(
            np.array([vote, sl_mult, leverage], dtype=np.float32)
        )
        total_reward += float(reward)
        if info.get("exit_reason"):
            exits.append(str(info["exit_reason"]))
            durations.append(int(getattr(env, "_last_trade_duration", 0)))
        if done or truncated:
            break
    return total_reward, exits, durations


def test_standing_flat_is_not_rewarded(frame):
    """A position-free episode must never accumulate positive reward.

    The environment used to pay +0.2 per step for a "calibrated vote" while
    flat and up to +0.10 more for sitting out an adverse move, while the
    opportunity cost was gated behind a condition that never fired. Doing
    nothing scored roughly +600 over an episode, which made HOLD the optimal
    policy and is why the deterministic policy never traded.

    Note the vote of 0.0: for a long-only specialist a *negative* vote is not
    "flat", it is an attempt to short, which draws a regime-mismatch penalty
    and would measure something else entirely.
    """
    env = _environment(frame)
    total, exits, _ = _drive(env, vote=0.0, steps=800)
    assert not exits, "a zero vote must not open a position"
    assert env.net_worth == pytest.approx(env.initial_balance), "flat must not move equity"
    assert total <= 0.0, "standing aside earned %+.1f reward" % total


@pytest.mark.parametrize('vote', [0.0, 0.9, -0.9])
def test_economic_reward_telescopes_to_net_equity_including_final_step(frame, vote):
    env = _environment(frame)
    total, _, _ = _drive(env, vote=vote, steps=len(frame))
    expected = env.config.ECONOMIC_REWARD_SCALE * np.log(env.net_worth / env.initial_balance)
    assert total == pytest.approx(expected, abs=2e-4)
    summary = env.consume_episode_summaries()[-1]
    assert summary['episode_reward'] == pytest.approx(expected, abs=2e-4)


def test_reward_prefers_trading_when_trading_makes_money(frame):
    """The objective must agree with equity, in both directions.

    This is the property the whole system rests on, and the hand-tuned shaping
    layer failed it in all six agent/regime combinations tested: on the
    strongest uptrend in the data the Bull finished +80% in equity and scored
    -913.8, against -195.2 for never opening a position. An agent maximising
    that objective correctly learns to do nothing.
    """
    from specialists.bull_specialist import BullTradingEnv

    def score(vote: float) -> tuple:
        env = _environment(frame, env_class=BullTradingEnv)
        total, _, _ = _drive(env, vote=vote, steps=len(frame) - 2)
        return total, float(env.net_worth) / float(env.initial_balance) - 1.0

    trading_reward, trading_equity = score(0.9)
    flat_reward, _ = score(0.0)

    # Only clearly signed outcomes are asserted. A marginal window depends on
    # how the opportunity cost is calibrated against trade variance, which is a
    # trade-off, not an invariant.
    if trading_equity > 0.10:
        assert trading_reward > flat_reward, (
            "trading returned %+.1f%% in equity but scored %+.1f against %+.1f for "
            "standing aside" % (trading_equity * 100, trading_reward, flat_reward)
        )
    elif trading_equity < -0.10:
        assert trading_reward < flat_reward, (
            "trading lost %+.1f%% in equity but still outscored standing aside "
            "(%+.1f versus %+.1f)" % (trading_equity * 100, trading_reward, flat_reward)
        )


def test_position_size_follows_conviction(frame, monkeypatch):
    """Notional must respond to the agent, not sit pinned at a floor.

    Sizing used to read tp_prior_conf, a column absent from the dataset, so it
    resolved to the 0.1 clip floor on every trade: notional 2.00 on capital
    100, which scaled the profit signal to nothing.

    Pinned at the default 2% risk per trade: a machine whose .env raises it
    (0.10) sends every trade to the leverage cap, where size cannot follow the
    vote any more.
    """
    from config.settings import TradingConfig
    monkeypatch.setattr(TradingConfig, "MAX_PORTFOLIO_RISK_PERCENT", 0.02)
    sizes = {}
    for vote in (0.2, 0.95):
        env = _environment(frame)
        for _ in range(env.max_steps):
            env.step(np.array([vote, 3.0, 2.0], dtype=np.float32))
            if env.position != 0:
                sizes[vote] = float(env.initial_notional_value)
                break
    assert len(sizes) == 2, "both votes must be able to open a position"
    assert min(sizes.values()) > 10.0, "notional collapsed to the floor: %r" % sizes
    assert sizes[0.95] > sizes[0.2], "higher conviction must size larger: %r" % sizes


def test_stop_does_not_tighten_before_one_r(frame):
    """The stop must give a trade room to become a trend.

    Three ratchets (trailing, Chandelier, PSAR) ran from the first bar and each
    took the tightest level, so the agent's sl_mult had no effect. The
    Chandelier read the highest high of the previous 22 bars, including bars
    from before entry, which could place the stop above the entry price.
    Measured before the fix: mean holding time 3.1 bars, nearly every trade
    closed by Stop Loss.
    """
    env = _environment(frame)
    _, exits, durations = _drive(env, vote=0.9, steps=2000)
    assert exits, "the scripted long policy must produce trades"
    assert np.mean(durations) > 8.0, (
        "mean holding time %.1f bars is too short to ride a trend" % np.mean(durations)
    )


def test_wider_stop_produces_longer_trades(frame):
    """sl_mult is an action; it has to change the outcome."""
    durations = {}
    for sl_mult in (1.0, 5.0):
        env = _environment(frame)
        _, _, lengths = _drive(env, vote=0.9, steps=2000, sl_mult=sl_mult)
        assert lengths, "no trade closed with sl_mult=%.1f" % sl_mult
        durations[sl_mult] = float(np.mean(lengths))
    assert durations[5.0] > durations[1.0], (
        "the stop distance did not reach the environment: %r" % durations
    )


def test_available_edge_is_directional(frame):
    """A directional specialist only sees opportunity on its own side."""
    from specialists.scientific_corrections import _available_edge

    class _Row:
        def __init__(self, data):
            self._data = data

        def get(self, key, default=0.0):
            return self._data.get(key, default)

    class _Env:
        pass

    env = _Env()
    env.current_row = _Row({"ml_edge": 0.4})

    env.specialist_name = "bull_specialist"
    assert _available_edge(env, {}) == pytest.approx(0.4)
    env.specialist_name = "bear_specialist"
    assert _available_edge(env, {}) == pytest.approx(0.0)

    # With no supervised opinion the fallback must stay quiet in chop, so the
    # agent is free to stand aside when there is nothing to follow.
    env.specialist_name = "bull_specialist"
    env.current_row = _Row({"cz_ema_dist_50": 0.1, "cz_efficiency_96": 0.02})
    assert _available_edge(env, {}) < 0.05


def test_training_leverage_cap_matches_the_environment(frame):
    """Production must not be allowed to exceed the leverage used in training.

    Live leverage used to be derived from regime confidence alone, ignoring the
    agent's own leverage action. With tp_prior_conf averaging 0.878 on this
    history the formula returned 8x, and 15x at full confidence, while the
    training action space caps at 3x. The same sequence of trades the agent
    learned on would have been executed with nearly three times the risk.
    """
    from config.settings import AIConfig, TradingConfig

    env = _environment(frame)
    declared = float(getattr(AIConfig(), "TRAINING_LEVERAGE_CAP", 3.0))
    actual = float(env.action_space.high[2])
    assert actual == pytest.approx(declared), (
        "the environment trains up to %.1fx but TRAINING_LEVERAGE_CAP declares "
        "%.1fx; production reads the declared value" % (actual, declared)
    )
    assert declared <= float(TradingConfig().MAX_LEVERAGE_PER_TRADE)
