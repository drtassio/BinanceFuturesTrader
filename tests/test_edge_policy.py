import pytest

from learning.edge_policy import EdgeRule, edge_action

RULE = EdgeRule(enter_probability=0.5, enter_edge=0.1, exit_edge=0.0, sl_mult=3.0, leverage=3.0)


def test_entry_needs_probability_edge_and_regime():
    row = {'ml_p_long': 0.6, 'ml_p_short': 0.2, 'ml_edge': 0.2, 'tp_prior_dir': 0.7}
    assert edge_action(row, 0.0, 'bull', RULE)[0] == pytest.approx(0.8)
    for column, value in [('ml_p_long', 0.4), ('ml_edge', 0.05), ('tp_prior_dir', 0.0)]:
        assert edge_action({**row, column: value}, 0.0, 'bull', RULE)[0] == pytest.approx(-0.8)


def test_open_position_rides_until_edge_crosses_exit_bar():
    weak = {'ml_p_long': 0.3, 'ml_p_short': 0.3, 'ml_edge': 0.02, 'tp_prior_dir': -0.5}
    assert edge_action(weak, 1.0, 'bull', RULE)[0] == pytest.approx(0.8)
    assert edge_action({**weak, 'ml_edge': -0.01}, 1.0, 'bull', RULE)[0] == pytest.approx(-0.8)


def test_ranger_uses_short_threshold_on_short_scale_and_stays_neutral_when_flat():
    rule = EdgeRule(enter_probability=0.5, enter_probability_short=0.3, enter_edge=0.1, require_regime=False)
    short = {'ml_p_long': 0.1, 'ml_p_short': 0.32, 'ml_edge': -0.2, 'tp_prior_dir': 0.0}
    assert edge_action(short, 0.0, 'ranger', rule)[0] == pytest.approx(-0.8)
    assert edge_action(short, 0.0, 'ranger', EdgeRule(enter_probability=0.5, require_regime=False))[0] == 0.0


def test_breakout_teacher_rides_until_the_exit_channel_breaks():
    from learning.edge_policy import BreakoutRule, breakout_action, load_rule, teacher_inputs

    rule = load_rule({"kind": "breakout"})
    assert isinstance(rule, BreakoutRule)
    assert teacher_inputs(rule, "bull")["columns"] == ("cz_breakout_up_480", "cz_breakout_down_240")
    calm = {"cz_breakout_up_480": -1.0, "cz_breakout_down_240": 3.0}
    assert breakout_action(calm, 0.0, "bull", rule)[0] == pytest.approx(-0.8)
    assert breakout_action({**calm, "cz_breakout_up_480": 0.2}, 0.0, "bull", rule)[0] == pytest.approx(0.8)
    # Held long: new highs are not required to stay in, only not losing the 240-bar low.
    assert breakout_action(calm, 1.0, "bull", rule)[0] == pytest.approx(0.8)
    assert breakout_action({**calm, "cz_breakout_down_240": -0.1}, 1.0, "bull", rule)[0] == pytest.approx(-0.8)


def test_breakout_features_never_include_the_current_bar_in_their_reference():
    import numpy as np
    import pandas as pd
    from feature_engineering.causal_features import add_trend_structure

    index = pd.date_range("2024-01-01", periods=600, freq="15min")
    close = pd.Series(np.linspace(100.0, 101.0, 600), index=index)
    frame = pd.DataFrame({"close": close, "high": close + 0.1, "low": close - 0.1, "atr_15m": 0.2}, index=index)
    out, _ = add_trend_structure(frame)
    before = out["cz_breakout_up_480"].iloc[:-1].copy()
    frame.iloc[-1, frame.columns.get_loc("close")] = 200.0
    frame.iloc[-1, frame.columns.get_loc("high")] = 200.0
    changed, _ = add_trend_structure(frame)
    assert changed["cz_breakout_up_480"].iloc[:-1].equals(before)
    assert changed["cz_breakout_up_480"].iloc[-1] > 0
