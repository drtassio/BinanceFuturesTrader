import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from tests.test_reward_and_sizing_invariants import DATASET, _environment
from specialists.trend_specialist import TrendFollowingEnv


def test_shared_liquidation_charges_fee_and_removes_mark_to_market_once():
    env = object.__new__(TrendFollowingEnv)
    env.initial_balance = 100.0
    env.net_worth = 120.0
    env.pnl_since_entry = 0.02
    env.initial_notional_value = 1000.0
    env.episode_peak_net_worth = 120.0
    env.trading_config = SimpleNamespace(TAKER_FEE=0.0004)

    env._apply_realized_pnl(20.0)

    assert env.net_worth == pytest.approx(119.6)
    assert env.pnl_since_entry == 0.0


@pytest.mark.skipif(not DATASET.exists(), reason="causal dataset required")
def test_forced_exit_pays_both_fees_and_uses_margin_returns(monkeypatch):
    frame = pd.read_parquet(DATASET).iloc[-4000:-2000].ffill().fillna(0.0)
    env = _environment(frame)
    env.max_steps = 1
    captured = {}
    original = env._apply_realized_pnl

    def capture(pnl):
        captured.update(notional=env.initial_notional_value,
                        leverage=env.current_leverage, pnl=pnl)
        original(pnl)

    monkeypatch.setattr(env, "_apply_realized_pnl", capture)
    _, _, terminated, truncated, _ = env.step(np.array([1.0, 3.0, 1.0], dtype=np.float32))
    assert terminated or truncated
    assert captured["notional"] > 0
    summary = env.consume_episode_summaries()[-1]
    expected_equity = (env.initial_balance + captured["pnl"]
                       - 2 * captured["notional"] * env.trading_config.TAKER_FEE)
    assert summary["final_net_worth"] == pytest.approx(expected_equity)
    margin = captured["notional"] / abs(captured["leverage"])
    assert summary["avg_return_pct"] == pytest.approx(captured["pnl"] / margin)
