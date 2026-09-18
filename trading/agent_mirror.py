"""A trained specialist operating live: mirror the position its environment holds.

The numbers a specialist is approved on come from the trading environment:
the policy votes, and the environment turns votes into entries, stops priced in
the contract's ATR, trailing after 1R, the catastrophe stop and exits. Sending
the raw vote to the exchange and letting the execution engine apply its own
stops would trade a strategy nobody measured.

So, as for the teacher (trading/teacher_policy.py), the bot replays the policy
through the same environment over recent closed bars and the account mirrors
the environment's position after the last one. Replay starts flat; the policy
is deterministic and its observation depends on the bars and on the position
it holds, so once replay and a full backtest are flat on the same bar they
coincide from there on. Trend trades last ~300 bars; the replay covers 3000.

The bars come from a persistent history. The live pipeline's frame is only
exact on its newest row (earlier rows carry start-up residuals of long EMAs and
channels), so each cycle appends just that row, and the trend structure is
recomputed over the continuous history, exactly as the training dataset was
built (scripts/build_history_dataset.py). A missing stretch, after downtime, is
rebuilt with the same stitched live pipeline the dataset came from.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading.teacher_policy import BAR, ShadowState, closed_bars

ROOT = Path(__file__).resolve().parents[1]
POLICY_NAME = "agent_mirror"
HISTORY_PATH = ROOT / "data" / "live_history.parquet"
REPLAY_BARS = 3000
PADDING = 3


def apply_leverage_bounds(contract: dict) -> None:
    """Replay with the leverage range the specialist was trained in.

    The environment's leverage action spans MIN_LEVERAGE_PER_TRADE up to
    TRAINING_LEVERAGE_CAP. Replaying under different settings would size the
    same votes differently from the backtest that approved them. Contracts
    written before the bounds were recorded were all trained in 1x..3x.
    """
    from config.settings import AIConfig, TradingConfig

    low, high = contract.get("leverage_bounds", [1.0, 3.0])
    TradingConfig.MIN_LEVERAGE_PER_TRADE = float(low)
    TradingConfig.MAX_LEVERAGE_PER_TRADE = max(float(TradingConfig.MAX_LEVERAGE_PER_TRADE), float(high))
    AIConfig.TRAINING_LEVERAGE_CAP = float(high)


def load_specialist(agent_name: str, model_dir: Path):
    """The promoted specialist with the stop scale and frame it was trained on."""
    from cloud.train_agent import AGENTS
    from config.settings import AIConfig, TradingConfig

    model_dir = Path(model_dir)
    contract = json.loads((model_dir / ("%s_feature_contract.json" % agent_name)).read_text(encoding="utf-8"))
    if not contract.get("training_frame_columns"):
        raise ValueError("contrato de %s sem training_frame_columns: modelo anterior ao espelho" % agent_name)
    AIConfig.ENV_STOP_ATR_TIMEFRAME = contract.get("stop_atr_timeframe", "15m")
    apply_leverage_bounds(contract)
    config = AIConfig()
    config.ECONOMIC_REWARD_ONLY = True
    config.MODEL_DIR = str(model_dir)
    agent = AGENTS[agent_name](config=config, trading_config=TradingConfig(),
                               input_dim=len(contract["feature_columns"]))
    agent.load_model()
    if agent.model is None or list(agent.feature_columns) != list(contract["feature_columns"]):
        raise ValueError("modelo/scaler de %s nao correspondem ao contrato" % agent_name)
    return agent, contract


def prepare_frame(history: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """History -> the frame the environment saw in training."""
    from feature_engineering.causal_features import add_trend_structure
    from feature_engineering.training_schema import align_to_training_frame

    frame = history.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame, _ = add_trend_structure(frame)
    frame = frame.ffill().fillna(0.0)
    return align_to_training_frame(frame, contract["training_frame_columns"])


def agent_trajectory(agent, frame: pd.DataFrame, agent_name: str) -> pd.DataFrame:
    """Deterministic policy through the training environment; one row per bar acted on."""
    from cloud.train_agent import _episode_environment

    env = _episode_environment(agent, frame, agent_name)
    inner = env
    while hasattr(inner, "venv"):
        inner = inner.venv
    raw = inner.envs[0]
    records: List[dict] = []
    previous = 0
    try:
        observation = env.reset()
        start = int(getattr(raw, "start_idx", 0) or 0)
        for _ in range(len(frame)):
            index = min(start + raw.current_step, len(frame) - 1)
            action, _ = agent.model.predict(observation, deterministic=True)
            observation, _, done, _ = env.step(action)
            if bool(done[0]):
                # The vector env has already reset the raw environment.
                break
            side = int(np.sign(raw.position))
            records.append({
                "bar": frame.index[index], "side": side, "entered": side != 0 and side != previous,
                "notional_fraction": float(raw.initial_notional_value / raw.net_worth) if side and raw.net_worth > 0 else 0.0,
                "entry_price": float(raw.entry_price) if side else 0.0,
                "stop_price": float(raw.sl_level) if side and raw.sl_level is not None else np.nan,
                "leverage": float(raw.current_leverage) if side else 0.0,
                "net_worth": float(raw.net_worth),
            })
            previous = side
    finally:
        env.close()
    return pd.DataFrame.from_records(records).set_index("bar")


def replay(agent, contract: dict, history: pd.DataFrame, agent_name: str) -> ShadowState:
    frame = prepare_frame(history.tail(REPLAY_BARS), contract)
    last_bar = frame.index[-1]
    padding = [frame.iloc[[-1]].set_axis([last_bar + BAR * k]) for k in range(1, PADDING + 1)]
    path = agent_trajectory(agent, pd.concat([frame, *padding]), agent_name)
    if last_bar not in path.index:
        raise RuntimeError("replay terminou antes do ultimo candle fechado")
    row = path.loc[last_bar]
    return ShadowState(agent=agent_name, bar=last_bar, side=int(row["side"]),
                       entered_on_last_bar=bool(row["entered"]),
                       notional_fraction=float(row["notional_fraction"]),
                       entry_price=float(row["entry_price"]),
                       stop_price=None if pd.isna(row["stop_price"]) else float(row["stop_price"]))


class LiveHistory:
    """Closed-bar rows exactly as the dataset holds them, persisted across restarts."""

    def __init__(self, path: Path = HISTORY_PATH, keep: int = REPLAY_BARS + 600):
        self.path = Path(path)
        self.keep = keep
        self.frame = pd.read_parquet(self.path) if self.path.exists() else pd.DataFrame()

    def missing_since(self, newest: pd.Timestamp) -> Optional[pd.Timestamp]:
        """First bar not in the history that a replay ending on 'newest' needs."""
        wanted_start = newest - BAR * (REPLAY_BARS - 1)
        if self.frame.empty or self.frame.index[-1] < wanted_start - BAR:
            return wanted_start
        index = self.frame.index[self.frame.index >= wanted_start]
        if len(index) and index[0] > wanted_start:
            return wanted_start
        gaps = index.to_series().diff().dropna()
        if (gaps > BAR).any():
            return wanted_start
        if self.frame.index[-1] < newest:
            return self.frame.index[-1] + BAR
        return None

    def extend(self, rows: pd.DataFrame) -> None:
        merged = pd.concat([self.frame, rows]) if not self.frame.empty else rows.copy()
        merged = merged.sort_index()
        merged = merged.loc[~merged.index.duplicated(keep="last")].tail(self.keep)
        self.frame = merged
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(self.path)

    def append_newest(self, live_frame: pd.DataFrame, now: pd.Timestamp) -> pd.Timestamp:
        closed = closed_bars(live_frame, now)
        newest = closed.index[-1]
        if self.frame.empty or newest > self.frame.index[-1]:
            self.extend(closed.iloc[[-1]])
        return newest

    def is_complete(self, newest: pd.Timestamp) -> bool:
        return self.missing_since(newest) is None and self.frame.index[-1] >= newest


async def rebuild(history: LiveHistory, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Fill [start, end] with the stitched live pipeline (read-only requests)."""
    from scripts.build_forward_dataset import build, checked_columns

    stitched, _, _ = await build(start - BAR * 610, end, 150, 50, 0.1, checked_columns([]))
    history.extend(stitched.loc[stitched.index >= start - BAR * 610])


def artifact_paths(model_dir: Path, agents) -> List[Path]:
    paths = []
    for agent in agents:
        paths += [Path(model_dir) / ("%s_specialist_sac.zip" % agent),
                  Path(model_dir) / ("%s_specialist_scaler.joblib" % agent),
                  Path(model_dir) / ("%s_feature_contract.json" % agent)]
    return paths + [ROOT / "trading" / "agent_mirror.py", ROOT / "specialists" / "trend_specialist.py",
                    ROOT / "feature_engineering" / "causal_features.py"]


def artifact_hashes(model_dir: Path, agents) -> Dict[str, Optional[str]]:
    hashes = {}
    for path in artifact_paths(model_dir, agents):
        try:
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            hashes[path.name] = None
    return hashes


def approval_path(model_dir: Path) -> Path:
    return Path(model_dir) / "agent_mirror_approval.json"


def approval_is_valid(model_dir: Path) -> Tuple[bool, str]:
    try:
        report = json.loads(approval_path(model_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "sem aprovacao registrada"
    if not report.get("approved"):
        return False, "aprovacao registrada como reprovada"
    current = artifact_hashes(model_dir, report.get("agents", []))
    changed = [name for name, digest in current.items() if not digest or report["artifact_hashes"].get(name) != digest]
    if changed:
        return False, "artefatos alterados desde a aprovacao: %s" % changed
    return True, "ok"
