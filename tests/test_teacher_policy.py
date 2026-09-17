import json

import pandas as pd
import pytest

from learning.edge_policy import EdgeRule
from trading import teacher_policy as teacher


def test_replay_selects_last_closed_bar_before_artificial_liquidation(monkeypatch):
    index = pd.date_range("2026-01-01", periods=4, freq="15min")
    frame = pd.DataFrame({"close": [100.0] * 4}, index=index)

    def fake_trajectory(padded, *args):
        assert len(padded) == len(frame) + 3
        # The last recorded row is a terminal liquidation, not live state.
        result = pd.DataFrame({"side": 1, "entered": False,
                               "notional_fraction": 0.2, "entry_price": 100.0,
                               "stop_price": 95.0}, index=padded.index[:-2])
        result.loc[index[-1], "entered"] = True
        result.loc[result.index[-1], "side"] = 0
        return result

    monkeypatch.setattr(teacher, "trajectory", fake_trajectory)
    state = teacher.replay(frame, "bull", EdgeRule())
    assert state.bar == index[-1]
    assert state.side == 1
    assert state.entered_on_last_bar


def test_replay_rejects_empty_frame():
    with pytest.raises(ValueError):
        teacher.replay(pd.DataFrame(), "bull", EdgeRule())


@pytest.mark.parametrize("agents", [[], ["bull"], ["bear"], ["bull", "bear", "ranger"]])
def test_approval_cannot_skip_required_specialists(tmp_path, monkeypatch, agents):
    monkeypatch.setattr(teacher, "artifact_hashes", lambda *args: {"model": "digest"})
    teacher.approval_path(tmp_path).write_text(json.dumps({
        "all_passed": True, "policy": teacher.POLICY_NAME, "agents": agents,
        "artifact_hashes": {"model": "digest"}}), encoding="utf-8")
    assert not teacher.approval_is_valid(tmp_path)


def test_mirror_does_not_chase_existing_shadow_trade():
    state = teacher.ShadowState("bull", pd.Timestamp("2026-01-01"), 1, False, .2, 100, 95)
    assert teacher.mirror([state], 0).action == "hold"
    assert teacher.mirror([], 1).action == "close"


def test_approval_requires_parity_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(teacher, "artifact_hashes", lambda *args: {"model": "digest"})
    report = {"all_passed": True, "policy": teacher.POLICY_NAME,
              "agents": list(teacher.AGENTS), "artifact_hashes": {"model": "digest"}}
    teacher.approval_path(tmp_path).write_text(json.dumps(report), encoding="utf-8")
    assert not teacher.approval_is_valid(tmp_path)
    parity = tmp_path / "teacher_parity_validation.json"
    parity.write_text(json.dumps({**report, "checks": 40}), encoding="utf-8")
    assert teacher.approval_is_valid(tmp_path)
    parity.write_text(json.dumps({**report, "checks": 4}), encoding="utf-8")
    assert not teacher.approval_is_valid(tmp_path)
