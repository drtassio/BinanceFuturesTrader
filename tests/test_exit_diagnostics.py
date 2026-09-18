import pytest

from learning.exit_diagnostics import summarize_exits


def test_learned_exits_are_separate_from_protection_and_terminal_liquidation():
    result = summarize_exits([
        dict(exit_reason='Agent Decision', pnl_usd=10, duration=20),
        dict(exit_reason='Agent Decision - prior pressure', pnl_usd=-2, duration=10),
        dict(exit_reason='Stop Loss', pnl_usd=3, duration=30),
        dict(exit_reason='Episode End', pnl_usd=4, duration=40),
    ])
    assert result['closed_trades'] == 4
    assert result['learned_exit_count'] == 2
    assert result['learned_exit_fraction'] == .5
    assert result['by_reason']['Stop Loss']['recorded_pnl_usd'] == 3
    assert result['by_reason']['Agent Decision']['avg_duration_steps'] == 20


def test_empty_diagnostics_do_not_claim_learned_exits():
    assert summarize_exits([])['learned_exit_fraction'] == 0


def test_trailing_stop_is_not_counted_as_learned_exit_or_initial_stop():
    result = summarize_exits([
        dict(exit_reason='Stop Loss', stop_protection_kind='initial', pnl_usd=-3, duration=4),
        dict(exit_reason='Stop Loss', stop_protection_kind='trailing', pnl_usd=7, duration=20),
    ])
    assert result['learned_exit_count'] == 0
    assert result['by_reason']['Trailing Stop']['count'] == 1
    assert result['by_reason']['Stop Loss']['count'] == 1


def test_episode_summary_captures_exits_before_logger_clears_trades():
    from specialists.trend_specialist import TrendFollowingEnv
    env = object.__new__(TrendFollowingEnv)
    env._episode_trades_log = [dict(exit_reason='Agent Decision', pnl_usd=5, duration=8)]
    env._log_episode_matrix = lambda: env._episode_trades_log.clear()
    env._build_financial_snapshot = lambda *args: {'num_trades': 1}
    env._current_episode_returns = [.01]
    env._current_episode_durations = [8]
    env._episode_reward_accumulator = .01
    env._episode_summaries = []
    env.policy_exit_requests = {'requested_steps': 5, 'blocked_by_grace_steps': 4,
                                'eligible_steps': 1}
    env._store_episode_summary()
    assert env._episode_summaries[0]['exit_diagnostics']['learned_exit_count'] == 1
    assert env._episode_trades_log == []
    assert env._episode_summaries[0]['policy_exit_requests']['blocked_by_grace_steps'] == 4
    env.policy_exit_requests['blocked_by_grace_steps'] = 0
    assert env._episode_summaries[0]['policy_exit_requests']['blocked_by_grace_steps'] == 4


@pytest.mark.parametrize('pnl,duration', [(float('nan'), 1), (1, -1), (1, float('inf'))])
def test_invalid_trade_diagnostics_fail_closed(pnl, duration):
    with pytest.raises(ValueError):
        summarize_exits([dict(pnl_usd=pnl, duration=duration)])
