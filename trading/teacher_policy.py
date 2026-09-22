"""The edge teacher operating live: mirror what the backtest environment does.

The policy that trades is the one whose validation and holdout numbers were
measured: meta-model probabilities plus the decision rule
(learning/edge_policy.py), executed by the trading environment. That
environment owns the exits that make the numbers what they are: the stop
distance with its grace period, the catastrophe stop, trailing only after the
trade pays 1R, the Chandelier and PSAR. Writing a second copy of all that in
the live bot would operate an untested strategy under a tested name.

So on each closed 15-minute bar the recent window is replayed through the same
environment class, and the live account mirrors the position the environment
holds after the last closed bar. Replay starts flat; when flat the rule's
decision depends only on that bar's features, so once the replay and a full
backtest are both flat on the same bar their trajectories coincide from there
on. scripts/verify_teacher_parity.py measures that agreement.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from learning.edge_policy import EdgeRule, load_rule, teacher_action

POLICY_NAME = "edge_teacher"
AGENTS = ("bull", "bear")
BAR = pd.Timedelta(minutes=15)
# A median trade lasts about 30 bars; the live frame carries ~850.
MIN_REPLAY_BARS = 300


@dataclass(frozen=True)
class ShadowState:
    agent: str
    bar: pd.Timestamp
    side: int
    entered_on_last_bar: bool
    notional_fraction: float
    entry_price: float
    stop_price: Optional[float]
    entry_bar: Optional[pd.Timestamp] = None  # bar the shadow's current trade opened on


@dataclass(frozen=True)
class MirrorDecision:
    action: str  # "open", "close" or "hold"
    side: int
    shadow: Optional[ShadowState]
    reason: str


def _env_class(agent: str):
    if agent == "bull":
        from specialists.bull_specialist import BullTradingEnv
        return BullTradingEnv
    if agent == "bear":
        from specialists.bear_specialist import BearTradingEnv
        return BearTradingEnv
    raise ValueError("agente sem ambiente: %s (so bull e bear)" % agent)


def make_env(frame: pd.DataFrame, agent: str, config=None):
    """The environment exactly as scripts/tune_edge_rule.run_rule drives it."""
    from config.settings import AIConfig

    config = config or AIConfig()
    config.ECONOMIC_REWARD_ONLY = True
    from feature_engineering.training_schema import align_to_training_frame

    frame = align_to_training_frame(frame)
    env = _env_class(agent)(df=frame, config=config, mode="training", specialist_name="%s_specialist" % agent)
    env.set_phase3_runtime_tweaks()
    env.max_steps = len(frame) - 2
    env.reset()
    return env


def trajectory(frame: pd.DataFrame, agent: str, rule: EdgeRule, config=None) -> pd.DataFrame:
    """Drive the rule over every bar; one row per bar acted on.

    The environment never acts on the final row of its frame, so callers that
    need a decision on their last closed bar pass one padding row after it.
    """
    env = make_env(frame, agent, config)
    start = int(getattr(env, "start_idx", 0) or 0)
    records: List[dict] = []
    previous = 0
    for _ in range(env.max_steps + 1):
        index = min(start + env.current_step, len(frame) - 1)
        action = teacher_action(frame.iloc[index], env.position, agent, rule)
        _, _, done, truncated, _ = env.step(action)
        side = int(np.sign(env.position))
        records.append({
            "bar": frame.index[index],
            "side": side,
            "entered": side != 0 and side != previous,
            "notional_fraction": float(env.initial_notional_value / env.net_worth) if side and env.net_worth > 0 else 0.0,
            "entry_price": float(env.entry_price) if side else 0.0,
            "stop_price": float(env.sl_level) if side and env.sl_level is not None else np.nan,
            "net_worth": float(env.net_worth),
        })
        previous = side
        if done or truncated:
            break
    return pd.DataFrame.from_records(records).set_index("bar")


def closed_bars(frame: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop bars still forming. Training only ever saw closed candles."""
    index = frame.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    return frame.loc[(index + BAR) <= now]


def replay(frame: pd.DataFrame, agent: str, rule: EdgeRule, config=None) -> ShadowState:
    """State of the environment after the last row of a frame of closed bars."""
    if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Replay requires nonempty, ordered, unique closed bars")
    # Leave the observed bar BEFORE episode termination. A single padding row
    # makes the environment terminate on an earlier bar and liquidate its
    # position artificially; live replay must not mirror that liquidation.
    padding = [frame.iloc[[-1]].set_axis([frame.index[-1] + BAR * offset])
               for offset in (1, 2, 3)]
    padded = pd.concat([frame, *padding])
    last = trajectory(padded, agent, rule, config).loc[frame.index[-1]]
    return ShadowState(
        agent=agent,
        bar=frame.index[-1],
        side=int(last["side"]),
        entered_on_last_bar=bool(last["entered"]),
        notional_fraction=float(last["notional_fraction"]),
        entry_price=float(last["entry_price"]),
        stop_price=None if pd.isna(last["stop_price"]) else float(last["stop_price"]),
    )


def mirror(shadows: Iterable[ShadowState], live_side: int) -> MirrorDecision:
    """What the live account must do to hold what the environments hold.

    Only a position opened on the last closed bar is copied. A shadow that has
    been in a trade for a while, while the account is flat (an order failed,
    the catastrophe stop fired, the bot just started), is not chased: entering
    late buys a different trade from the one the backtest measured.
    """
    active = [s for s in shadows if s.side != 0]
    sides = {s.side for s in active}
    if len(sides) > 1:
        # Bull holds while edge > 0 and Bear enters only at edge <= -0.1, so
        # both in opposite trades is impossible by construction; stand aside
        # rather than guess if it ever happens.
        if live_side != 0:
            return MirrorDecision("close", -live_side, None, "sombras em conflito: zerar")
        return MirrorDecision("hold", 0, None, "sombras em conflito: ficar de fora")
    target = active[0] if active else None
    target_side = target.side if target else 0
    if live_side != 0 and live_side != target_side:
        return MirrorDecision("close", -live_side, target, "agente saiu da posicao")
    if live_side == 0 and target is not None and target.entered_on_last_bar:
        return MirrorDecision("open", target.side, target, "agente %s entrou" % target.agent)
    if live_side == 0 and target is not None:
        return MirrorDecision("hold", 0, target, "trade do agente ja em curso: nao persegue")
    return MirrorDecision("hold", live_side, target, "posicao igual a do agente")


def load_rules(model_dir: Path, agents: Iterable[str] = AGENTS) -> Dict[str, EdgeRule]:
    rules = {}
    for agent in agents:
        saved = json.loads((Path(model_dir) / ("%s_edge_rule.json" % agent)).read_text(encoding="utf-8"))
        rules[agent] = load_rule(saved["rule"])
    return rules


def artifact_paths(model_dir: Path, agents: Iterable[str] = AGENTS) -> List[Path]:
    """Files whose exact contents the OOS approval vouches for."""
    root = Path(__file__).resolve().parents[1]
    return ([Path(model_dir) / "meta_labeler.joblib"]
            + [Path(model_dir) / ("%s_edge_rule.json" % agent) for agent in agents]
            + [root / "learning" / "edge_policy.py", Path(__file__),
               root / "config" / "settings.py",
               root / "specialists" / "trend_specialist.py",
               root / "specialists" / "bull_specialist.py",
               root / "specialists" / "bear_specialist.py",
               root / "trading" / "ai_controller.py",
               root / "scripts" / "approve_teacher_oos.py",
               root / "scripts" / "verify_teacher_parity.py"])


def artifact_hashes(model_dir: Path, agents: Iterable[str] = AGENTS) -> Dict[str, Optional[str]]:
    hashes = {}
    for path in artifact_paths(model_dir, agents):
        try:
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            hashes[path.name] = None
    return hashes


def approval_path(model_dir: Path) -> Path:
    return Path(model_dir) / "teacher_oos_validation.json"


def approval_is_valid(model_dir: Path) -> bool:
    """The recorded approval passed and still describes the files on disk."""
    try:
        report = json.loads(approval_path(model_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not report.get("all_passed"):
        return False
    if report.get("policy") != POLICY_NAME or report.get("agents") != list(AGENTS):
        return False
    if not parity_is_valid(model_dir):
        return False
    current = artifact_hashes(model_dir, AGENTS)
    recorded = report.get("artifact_hashes") or {}
    return all(digest and recorded.get(name) == digest for name, digest in current.items())


def parity_is_valid(model_dir: Path) -> bool:
    try:
        report = json.loads((Path(model_dir) / "teacher_parity_validation.json").read_text(encoding="utf-8"))
        if not report.get("all_passed") or int(report.get("checks", 0)) < 40:
            return False
        current = artifact_hashes(model_dir)
        recorded = report.get("artifact_hashes") or {}
        return all(digest and recorded.get(name) == digest for name, digest in current.items())
    except (OSError, ValueError, TypeError):
        return False
