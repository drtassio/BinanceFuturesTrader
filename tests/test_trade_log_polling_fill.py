"""A market order that only fills on the status poll (NEW -> FILLED) is still
written to the trade log, exactly once."""
import asyncio

from models.trade_schema import Action, Signal
from trading.execution_engine import ExecutionEngine, ExecutionOrder


class FakeConnector:
    def __init__(self, first_status):
        self.first_status = first_status

    async def place_order(self, params):
        if self.first_status == "FILLED":
            return {"orderId": 1, "clientOrderId": "c1", "status": "FILLED", "executedQty": "0.0285",
                    "cumQuote": str(0.0285 * 85093.6), "updateTime": 1790172939000}
        return {"orderId": 1, "clientOrderId": "c1", "status": "NEW", "executedQty": "0", "cumQuote": "0"}

    async def get_order(self, symbol, order_id=None):
        return {"orderId": 1, "status": "FILLED", "executedQty": "0.0285",
                "cumQuote": str(0.0285 * 85093.6), "updateTime": 1790172940000}


def _run(first_status):
    logged = []
    engine = object.__new__(ExecutionEngine)
    engine.connector = FakeConnector(first_status)
    engine.system_state = {"recent_trades": []}
    engine.trade_log_callback = logged.append
    engine.portfolio = type("P", (), {"update_from_trade": lambda self, trade: None})()
    engine.config = type("C", (), {"DEFAULT_STOP_LOSS_PCT": 0.02, "DEFAULT_TAKE_PROFIT_PCT": 0.0})()
    engine._mirror_stops = {}

    async def symbol_info(symbol):
        return {"quantityPrecision": 3}

    async def no_op(*args, **kwargs):
        return None

    engine._get_cached_symbol_info = symbol_info
    engine._cancel_protective_orders = no_op
    signal = Signal(symbol="BTCUSDT", action=Action.SELL, confidence=1.0, position_size_pct=1.0)
    order = ExecutionOrder(signal, "MARKET", "exec_test")
    order.total_quantity = 0.0285
    order.reduce_only = True          # a close: no protections to arm in this test
    asyncio.run(engine._execute_market_real(order))
    return logged


def test_fill_found_by_polling_is_logged_once():
    logged = _run("NEW")
    assert len(logged) == 1
    assert logged[0]["quantity"] == 0.0285
    assert abs(logged[0]["price"] - 85093.6) < 1e-6


def test_immediate_fill_is_logged_once():
    assert len(_run("FILLED")) == 1
