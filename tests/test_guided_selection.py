import numpy as np
import pytest

from cloud.train_guided import score


def test_tiny_high_ratio_cannot_displace_operational_candidate():
    tiny = dict(num_trades=8, total_return_pct=.0792, profit_factor=6.73, max_drawdown_pct=.02)
    usable = dict(num_trades=59, total_return_pct=.1118, profit_factor=1.71, max_drawdown_pct=.128)
    assert score(tiny) == -np.inf
    assert score(usable) > score(tiny)


def test_excessive_drawdown_cannot_win_selection():
    assert score(dict(num_trades=60, total_return_pct=.8,
                      profit_factor=2, max_drawdown_pct=.3)) == -np.inf


@pytest.mark.parametrize('column,value', [('total_return_pct', float('nan')),
                                        ('profit_factor', float('nan')),
                                        ('max_drawdown_pct', float('nan')),
                                        ('max_drawdown_pct', -.1)])
def test_invalid_metrics_are_not_candidates(column, value):
    metrics = dict(num_trades=50, total_return_pct=.3, profit_factor=2, max_drawdown_pct=.1)
    metrics[column] = value
    assert score(metrics) == -np.inf
