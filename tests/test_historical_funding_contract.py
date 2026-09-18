import pandas as pd
import pytest

from specialists.trend_specialist import TrendFollowingEnv


@pytest.mark.parametrize("row,expected", [
    ({"funding_rate": 0.001}, 0.001),
    ({"funding_rate_15m": -0.001}, -0.001),
    ({"funding_rate": 0.0, "funding_rate_15m": 0.1}, 0.0),
    ({"close": 100.0}, None),
])
def test_rate_column_contract(row, expected):
    env = object.__new__(TrendFollowingEnv)
    env.df = pd.DataFrame([row])
    assert env._funding_rate_from_row(env.df.iloc[0]) == expected


@pytest.mark.parametrize("side,expected", [(1, 1.0), (-1, -1.0)])
def test_funding_is_settled_at_boundary_with_correct_side(side, expected):
    previous = pd.Timestamp("2025-01-01 07:45", tz="UTC")
    current = pd.Timestamp("2025-01-01 08:00", tz="UTC")
    assert TrendFollowingEnv._historical_funding_cost(
        1000, side, 0.001, previous, current
    ) == pytest.approx(expected)
    assert TrendFollowingEnv._historical_funding_cost(
        1000, side, 0.001, current, current + pd.Timedelta(minutes=15)
    ) == 0.0


def test_nonfinite_historical_rate_is_not_silently_replaced():
    env = object.__new__(TrendFollowingEnv)
    env.df = pd.DataFrame([{"funding_rate": float("nan")}])
    with pytest.raises(ValueError, match="Non-finite"):
        env._funding_rate_from_row(env.df.iloc[0])
