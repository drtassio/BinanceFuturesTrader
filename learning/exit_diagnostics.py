"""Describe observed closes without changing rewards or selecting a policy."""
import math


def summarize_exits(trades):
    groups = {}
    for trade in trades:
        reason = str(trade.get('exit_reason') or 'Unknown')
        if reason == 'Stop Loss' and trade.get('stop_protection_kind') == 'trailing':
            reason = 'Trailing Stop'
        group = groups.setdefault(reason, {'count': 0, 'wins': 0,
                                           'recorded_pnl_usd': 0., 'duration_steps': 0.})
        pnl = float(trade['pnl_usd'])
        duration = float(trade['duration'])
        if not math.isfinite(pnl) or not math.isfinite(duration) or duration < 0:
            raise ValueError('Invalid exit diagnostic trade')
        group['count'] += 1
        group['wins'] += int(pnl > 0)
        group['recorded_pnl_usd'] += pnl
        group['duration_steps'] += duration
    total = sum(group['count'] for group in groups.values())
    learned = sum(group['count'] for reason, group in groups.items()
                  if reason.startswith('Agent Decision'))
    for group in groups.values():
        group['win_rate'] = group.pop('wins') / group['count']
        group['avg_duration_steps'] = group.pop('duration_steps') / group['count']
    return {'closed_trades': total, 'learned_exit_count': learned,
            'learned_exit_fraction': learned / total if total else 0.,
            'by_reason': groups,
            'pnl_basis': 'Recorded realized trade PnL; use account equity for all-cost return'}
