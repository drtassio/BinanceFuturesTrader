import json

import pandas as pd

from models.trade_schema import Action, Signal
from trading import agent_mirror as mirror
from trading.execution_engine import ExecutionEngine


def _history(tmp_path, start, bars):
    index = pd.date_range(start, periods=bars, freq="15min", tz="UTC")
    history = mirror.LiveHistory(path=tmp_path / "history.parquet")
    history.extend(pd.DataFrame({"close": range(bars)}, index=index).astype(float))
    return history, index


def test_history_is_complete_only_without_gaps_over_the_replay_window(tmp_path):
    history, index = _history(tmp_path, "2026-01-01", mirror.REPLAY_BARS + 10)
    assert history.is_complete(index[-1])
    # A bar that has closed since the last append is missing from the tail.
    assert history.missing_since(index[-1] + mirror.BAR) == index[-1] + mirror.BAR
    gappy = history.frame.drop(index[-500])
    history.frame = gappy
    assert history.missing_since(index[-1]) is not None


def test_only_the_newest_closed_row_of_a_live_frame_is_appended(tmp_path):
    history, index = _history(tmp_path, "2026-01-01", 20)
    live_index = pd.date_range(index[-1] + mirror.BAR, periods=3, freq="15min", tz="UTC")
    live = pd.DataFrame({"close": [-1.0, -2.0, -3.0]}, index=live_index)
    now = live_index[1] + mirror.BAR  # live_index[2] is still forming
    newest = history.append_newest(live, now)
    assert newest == live_index[1]
    assert history.frame.index[-1] == live_index[1]
    assert live_index[0] not in history.frame.index  # earlier rows of a live window carry start-up residuals


def test_mirror_signals_execute_at_market_and_others_keep_smart_routing():
    engine = object.__new__(ExecutionEngine)
    signal = Signal(symbol="BTCUSDT", action=Action.BUY, confidence=1.0, position_size_pct=0.9,
                    leverage=10.0, explanation={"policy": "agent_mirror"})
    assert engine._is_mirror_signal(signal)
    assert engine._choose_smart_strategy(signal) == "MARKET"
    other = Signal(symbol="BTCUSDT", action=Action.BUY, confidence=1.0, explanation={"policy": "sac"})
    assert not engine._is_mirror_signal(other)


def test_approval_is_void_once_an_artifact_changes(tmp_path):
    for name in ("bull_specialist_sac.zip", "bull_specialist_scaler.joblib", "bull_feature_contract.json"):
        (tmp_path / name).write_bytes(name.encode())
    report = {"approved": True, "agents": ["bull"], "artifact_hashes": mirror.artifact_hashes(tmp_path, ["bull"])}
    mirror.approval_path(tmp_path).write_text(json.dumps(report))
    assert mirror.approval_is_valid(tmp_path)[0]
    (tmp_path / "bull_specialist_sac.zip").write_bytes(b"retrained")
    ok, detail = mirror.approval_is_valid(tmp_path)
    assert not ok and "bull_specialist_sac.zip" in detail
