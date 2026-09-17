"""Rule that turns the supervised edge estimate into specialist actions.

It is the teacher for behaviour cloning, not the product. The SAC specialists
learn to reproduce it from their own observation, then reinforcement learning
may improve on it; a checkpoint is only kept if it beats the teacher on
validation.

The rule is a pure function of one row of causal features plus the current
position, which the specialist also observes (agent_state carries its sign), so
it is learnable from the observation alone.

Entry needs a strong, regime-aligned opinion; exit waits for the opinion to
fade below a lower bar. The hysteresis is what lets a position ride a trend:
exiting on the entry threshold closed trades at the first wobble, which is why
the earlier reference made 18 short trades.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

SIDES = {"bull": (1,), "bear": (-1,), "ranger": (1, -1)}


@dataclass(frozen=True)
class EdgeRule:
    enter_probability: float = 0.60
    enter_edge: float = 0.10
    exit_edge: float = 0.0
    require_regime: bool = True
    sl_mult: float = 3.0
    leverage: float = 3.0
    vote: float = 0.8
    # The two meta-models live on different scales (short base rate 26.5%), so
    # a two-sided specialist needs its own bar for shorts. None: same as long.
    enter_probability_short: Optional[float] = None

    def entry_probability(self, side: int) -> float:
        if side < 0 and self.enter_probability_short is not None:
            return self.enter_probability_short
        return self.enter_probability

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _regime_allows(row, side: int) -> bool:
    # tp_prior_dir is +confidence in a Bull regime, -confidence in a Bear regime
    # and 0 in a Ranger regime, so its sign is exactly "up > down". The
    # specialists observe tp_prior_dir but not tp_regime_up/down, and a rule the
    # agent must imitate can only read what the agent can see.
    direction = row.get("tp_prior_dir", None)
    if direction is not None:
        return float(direction) * side > 0.0
    up = float(row.get("tp_regime_up", 0.0))
    down = float(row.get("tp_regime_down", 0.0))
    return up > down if side > 0 else down > up


def edge_action(row, position: float, agent: str, rule: EdgeRule) -> np.ndarray:
    """[vote, sl_mult, leverage] for one bar."""
    edge = float(row.get("ml_edge", 0.0))
    probability = {1: float(row.get("ml_p_long", 0.5)), -1: float(row.get("ml_p_short", 0.5))}
    held = int(np.sign(position))

    if held != 0:
        # Stay while the opinion still points our way; the vote must actively
        # cross to the other side for the environment to close the trade.
        keep = held * edge > rule.exit_edge
        vote = held * rule.vote if keep else -held * rule.vote
        return np.array([vote, rule.sl_mult, rule.leverage], dtype=np.float32)

    for side in SIDES[agent]:
        if (probability[side] >= rule.entry_probability(side)
                and side * edge >= rule.enter_edge
                and (not rule.require_regime or _regime_allows(row, side))):
            return np.array([side * rule.vote, rule.sl_mult, rule.leverage], dtype=np.float32)

    # Flat and no setup. A directional specialist votes firmly AGAINST its own
    # side: the environment opens a position as soon as the vote crosses about
    # +/-0.064, so a flat vote of 0 leaves no margin, and a cloned actor whose
    # output drifts slightly positive enters on noise. Measured in the smoke
    # test: the teacher made 0 trades on a validation slice, its clone with a
    # zero flat vote made 60. A vote of -0.8 turns the decision into a sign with
    # a wide margin. It cannot open the opposite side (Bull is long-only, Bear
    # short-only) and the economic reward carries no mismatch penalty.
    # The Ranger trades both sides, so for it the only neutral vote is zero.
    sides = SIDES[agent]
    flat_vote = 0.0 if len(sides) > 1 else -sides[0] * rule.vote
    return np.array([flat_vote, rule.sl_mult, rule.leverage], dtype=np.float32)


@dataclass(frozen=True)
class BreakoutRule:
    """Channel breakout: the teacher for multi-day trend following.

    The edge rule above rests on ml_p_long/ml_p_short, and those turned out to
    carry no information out of sample (AUC 0.505-0.523 on validation and
    holdout, 0.487-0.491 on bars after the dataset ends). Its backtest profits
    were the environment's exit mechanics riding whatever the market did.
    Breakout trend following is the pattern with a record on BTC perpetuals:
    long-only Donchian 30/15 on 4h made Sharpe ~1 from 2020 to 2026 after costs
    and funding. On 15m bars that is a 480-bar entry and a 240-bar exit, with
    stops priced in 4h ATR (ENV_STOP_ATR_TIMEFRAME=4h). Nothing here is tuned:
    the windows are the classic ones, fixed before looking at any split.
    """
    kind: str = "breakout"
    entry_window: int = 480
    exit_window: int = 240
    sl_mult: float = 3.0
    leverage: float = 3.0
    vote: float = 0.8

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def breakout_inputs(rule: BreakoutRule, agent: str) -> tuple:
    side = SIDES[agent][0]
    if side > 0:
        return ("cz_breakout_up_%d" % rule.entry_window, "cz_breakout_down_%d" % rule.exit_window)
    return ("cz_breakout_down_%d" % rule.entry_window, "cz_breakout_up_%d" % rule.exit_window)


def breakout_action(row, position: float, agent: str, rule: BreakoutRule) -> np.ndarray:
    """[vote, sl_mult, leverage]: enter past the long channel, leave past the short one."""
    if len(SIDES[agent]) != 1:
        raise ValueError("breakout teacher is directional; %s trades both sides" % agent)
    side = SIDES[agent][0]
    entry_column, exit_column = breakout_inputs(rule, agent)
    entry = float(row.get(entry_column, 0.0))
    exit_ = float(row.get(exit_column, 0.0))
    held = int(np.sign(position))
    if held == side:
        # Long: out once the close loses the prior exit-window low (negative
        # distance). Short: out once it clears the prior exit-window high.
        broken = exit_ < 0.0 if side > 0 else exit_ > 0.0
        vote = -side * rule.vote if broken else side * rule.vote
    else:
        started = entry > 0.0 if side > 0 else entry < 0.0
        vote = side * rule.vote if started else -side * rule.vote
    return np.array([vote, rule.sl_mult, rule.leverage], dtype=np.float32)


def load_rule(saved: Dict[str, object]):
    """Teacher from its saved JSON 'rule' block."""
    if saved.get("kind") == "breakout":
        return BreakoutRule(**saved)
    return EdgeRule(**saved)


def teacher_action(row, position: float, agent: str, rule) -> np.ndarray:
    if isinstance(rule, BreakoutRule):
        return breakout_action(row, position, agent, rule)
    return edge_action(row, position, agent, rule)


def teacher_inputs(rule, agent: str) -> Dict[str, object]:
    """Observation columns the teacher reads, and whether it reads tp_prior_dir."""
    if isinstance(rule, BreakoutRule):
        return {"columns": breakout_inputs(rule, agent), "prior_dir": False}
    return {"columns": ("ml_p_long", "ml_p_short", "ml_edge"), "prior_dir": bool(rule.require_regime)}
