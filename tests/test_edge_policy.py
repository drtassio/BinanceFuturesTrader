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
