"""Preserve a learned close only when the ensemble and live position agree."""
from dataclasses import replace
import math

from models.trade_schema import Action


def preserve_policy_exit(signal, expert_signals, quantity):
    quantity = float(quantity)
    closing = ((quantity > 0 and signal.action == Action.SELL)
               or (quantity < 0 and signal.action == Action.BUY))
    if not math.isfinite(quantity) or not closing:
        return signal
    sources = [name for candidate, weight, name, _ in expert_signals
               if math.isfinite(float(weight)) and weight > 0
               and candidate.action == signal.action
               and (candidate.explanation or {}).get('position_exit')]
    if not sources:
        return signal
    return replace(signal, position_size_pct=1.0,
                   explanation={**(signal.explanation or {}),
                                'position_exit': True, 'exit_sources': sources})
