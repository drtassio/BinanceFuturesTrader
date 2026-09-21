"""The exchange stop of a mirrored position follows the environment's stop and only tightens."""
import asyncio

from trading.execution_engine import ExecutionEngine


class FakeConnector:
    def __init__(self):
        self.placed = []
        self.cancelled = 0

    async def place_stop_loss_order(self, symbol, side, quantity, stop_price):
        self.placed.append((side, round(stop_price, 2)))
        return {"orderId": len(self.placed)}

    async def get_open_orders(self, symbol):
        self.cancelled += 1
        return []


def _engine():
    engine = object.__new__(ExecutionEngine)
    engine.connector = FakeConnector()
    engine.system_state = {"live_trading_enabled": True}
    engine._mirror_stops = {}
    engine.portfolio = type("P", (), {"positions": {}})()
    engine.config = type("C", (), {"MIRROR_EMERGENCY_STOP_PCT": 0.10})()
    return engine


def test_long_stop_moves_up_but_never_down():
    engine = _engine()
    run = asyncio.run
    run(engine.sync_mirror_stop("BTCUSDT", 1, 0.01, 100.0))
    run(engine.sync_mirror_stop("BTCUSDT", 1, 0.01, 99.0))    # looser: ignored
    run(engine.sync_mirror_stop("BTCUSDT", 1, 0.01, 100.0))   # same: ignored
    run(engine.sync_mirror_stop("BTCUSDT", 1, 0.01, 101.5))   # tighter: moved
    assert engine.connector.placed == [("SELL", 100.0), ("SELL", 101.5)]
    assert engine._mirror_stops["BTCUSDT"] == 101.5


def test_short_stop_moves_down_but_never_up():
    engine = _engine()
    run = asyncio.run
    run(engine.sync_mirror_stop("BTCUSDT", -1, 0.01, 110.0))
    run(engine.sync_mirror_stop("BTCUSDT", -1, 0.01, 111.0))  # looser: ignored
    run(engine.sync_mirror_stop("BTCUSDT", -1, 0.01, 108.0))  # tighter: moved
    assert engine.connector.placed == [("BUY", 110.0), ("BUY", 108.0)]


def test_nothing_is_sent_when_not_live():
    engine = _engine()
    engine.system_state["live_trading_enabled"] = False
    asyncio.run(engine.sync_mirror_stop("BTCUSDT", 1, 0.01, 100.0))
    assert engine.connector.placed == []
