import pytest

from cloud.train_agent import judge


def good():
    return dict(num_trades=40, total_return_pct=.2, max_drawdown_pct=.1,
                profit_factor=1.5, sharpe_ratio=1.0, vote_std=.3, deterministic=True)


def test_profitable_deterministic_policy_passes_operational_thresholds():
    assert judge(good(), -.2)['approved_for_live_trading']


@pytest.mark.parametrize('column,value', [
    ('max_drawdown_pct', .159), ('profit_factor', .98), ('sharpe_ratio', .2),
    ('num_trades', 10), ('deterministic', False), ('total_return_pct', 0),
    ('vote_std', .001), ('max_drawdown_pct', float('nan')),
])
def test_cloud_cannot_approve_policy_rejected_by_bot(column, value):
    assert not judge({**good(), column: value}, -.2)['approved_for_live_trading']


@pytest.mark.parametrize('column', ['max_drawdown_pct', 'profit_factor', 'sharpe_ratio', 'deterministic'])
def test_missing_evidence_fails_closed(column):
    metrics = good()
    metrics.pop(column)
    assert not judge(metrics, -.2)['approved_for_live_trading']
