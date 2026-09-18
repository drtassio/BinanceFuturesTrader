import pandas as pd
import pytest

from scripts.evaluate_edge_reference import EdgeReference


@pytest.mark.parametrize('side', [1, -1])
def test_reference_requires_probability_edge_and_regime(side):
    long = side == 1
    row = {'ml_p_long': 0.7 if long else 0.3,
           'ml_p_short': 0.3 if long else 0.7,
           'ml_edge': side * 0.4,
           'tp_regime_up': 0.8 if long else 0.1,
           'tp_regime_down': 0.1 if long else 0.8}
    active = EdgeReference(pd.DataFrame([row]), side)
    action, _ = active.predict(None)
    assert action[0, 0] == pytest.approx(side * 0.8)
    assert action[0, 1:].tolist() == [3.0, 1.0]
    for column, value in [('ml_edge', 0.0),
                          ('ml_p_long' if long else 'ml_p_short', 0.5),
                          ('tp_regime_up' if long else 'tp_regime_down', 0.0)]:
        inactive = EdgeReference(pd.DataFrame([{**row, column: value}]), side)
        action, _ = inactive.predict(None)
        assert action[0, 0] == pytest.approx(-side * 0.8)


def test_reference_current_decision_does_not_use_future_rows():
    row = {'ml_p_long': 0.7, 'ml_p_short': 0.3, 'ml_edge': 0.4,
           'tp_regime_up': 0.8, 'tp_regime_down': 0.1}
    first = EdgeReference(pd.DataFrame([row, {**row, 'ml_edge': -1.0}]), 1)
    second = EdgeReference(pd.DataFrame([row, {**row, 'ml_edge': 1.0}]), 1)
    assert first.predict(None)[0].tolist() == second.predict(None)[0].tolist()
