"""The exchange stop of a mirrored position follows the environment's stop and only tightens."""
import asyncio

from trading.execution_engine import ExecutionEngine


class FakeConnector:
    """Keeps the open stops like Binance does, including its rule of a single
    closePosition stop per direction (-4130)."""

    def __init__(self, open_stops=None):
        self.placed = []
        self.open = list(open_stops or [])
        self.cancelled = []

    async def place_stop_loss_order(self, symbol, side, quantity, stop_price, close_position=True):
        if close_position and any(o["side"] == side and o["closePosition"] for o in self.open):
            return None   # -4130: an open closePosition stop in this direction exists
        self.placed.append((side, round(stop_price, 2)))
        oid = 100 + len(self.placed)
        self.open.append({"orderId": oid, "side": side, "type": "STOP_MARKET", "stopPrice": stop_price,
                          "closePosition": close_position, "is_algo": True})
        return {"algoId": oid}

    async def get_open_orders(self, symbol):
        return [dict(o) for o in self.open]

    async def _make_request(self, method, path, params=None, signed=False):
        if method == "DELETE":
            self.cancelled.append(params["algoId"])
            self.open = [o for o in self.open if o["orderId"] != params["algoId"]]
        return {}


def _engine(open_stops=None):
    engine = object.__new__(ExecutionEngine)
    engine.connector = FakeConnector(open_stops)
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
    # exactly one stop left on the exchange: the new one
    assert [o["stopPrice"] for o in engine.connector.open] == [101.5]


def test_short_stop_moves_down_but_never_up():
    engine = _engine()
    run = asyncio.run
    run(engine.sync_mirror_stop("BTCUSDT", -1, 0.01, 110.0))
    run(engine.sync_mirror_stop("BTCUSDT", -1, 0.01, 111.0))  # looser: ignored
    run(engine.sync_mirror_stop("BTCUSDT", -1, 0.01, 108.0))  # tighter: moved
    assert engine.connector.placed == [("BUY", 110.0), ("BUY", 108.0)]
    assert [o["stopPrice"] for o in engine.connector.open] == [108.0]


def test_tightening_an_existing_close_position_stop_does_not_hit_4130():
    # The stop placed at the fill is closePosition; the trailing one must coexist
    # with it until it is cancelled, so it goes reduceOnly with the quantity.
    engine = _engine([{"orderId": 7, "side": "BUY", "type": "STOP_MARKET", "stopPrice": 86843.6,
                       "closePosition": True, "is_algo": True}])
    engine._mirror_stops["BTCUSDT"] = 86843.6
    asyncio.run(engine.sync_mirror_stop("BTCUSDT", -1, 0.0285, 86000.0))
    assert engine.connector.placed == [("BUY", 86000.0)]
    assert engine.connector.cancelled == [7]
    assert [(o["stopPrice"], o["closePosition"]) for o in engine.connector.open] == [(86000.0, False)]


def test_restart_adopts_the_stop_already_on_the_exchange():
    # After a restart _mirror_stops is empty but the exchange still holds the stop.
    engine = _engine([{"orderId": 7, "side": "BUY", "type": "STOP_MARKET", "stopPrice": 86843.6,
                       "closePosition": True, "is_algo": True}])
    asyncio.run(engine.sync_mirror_stop("BTCUSDT", -1, 0.0285, 86843.61))
    assert engine.connector.placed == []
    assert engine.connector.cancelled == []
    assert engine._mirror_stops["BTCUSDT"] == 86843.6


def test_nothing_is_sent_when_not_live():
    engine = _engine()
    engine.system_state["live_trading_enabled"] = False
    asyncio.run(engine.sync_mirror_stop("BTCUSDT", 1, 0.01, 100.0))
    assert engine.connector.placed == []
