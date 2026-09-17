from types import SimpleNamespace

import pandas as pd
import pytest


def controls(frame, step):
    from specialists.trend_specialist import TrendFollowingEnv

    state = SimpleNamespace(df=frame, start_idx=0, current_step=step,
                            scalp_penalty_min_duration=20, atr_col="atr_15m",
                            _active_grace_period=17)
    TrendFollowingEnv._init_dynamic_risk_controls(state)
    assert state._active_grace_period == 17
    assert state.scalp_penalty_min_duration == 20
    return state._atr_pct_baseline, state._grace_floor, state._grace_median, state._grace_cap


def test_future_rows_cannot_change_current_risk():
    frame = pd.DataFrame({"close": [100.] * 600, "atr_15m": [1.] * 600,
                          "tp_duration_median": [30.] * 600})
    changed = frame.copy()
    changed.loc[401:, ["atr_15m", "tp_duration_median"]] = 9999.
    assert controls(frame, 400) == controls(changed, 400)


def test_recent_window_and_full_history_use_same_risk():
    frame = pd.DataFrame({"close": [100.] * 600,
                          "atr_15m": [1. + i / 1000 for i in range(600)],
                          "tp_duration_median": [20. + i % 30 for i in range(600)]})
    assert controls(frame, 599) == pytest.approx(controls(frame.tail(300), 299))
