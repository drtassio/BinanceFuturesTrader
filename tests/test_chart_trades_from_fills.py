"""The chart's trade marks come from the account's fills: a trade closed while
this bot was off (or by another machine) is drawn when and where it closed."""
import pandas as pd

from trading.mirror_chart import trades_from_fills

MS = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)
CLOSE = pd.Series(1.0, index=pd.date_range("2026-09-23", periods=200, freq="15min", tz="UTC"))


def test_short_opened_and_closed_elsewhere():
    fills = [
        {"id": 1, "time": MS("2026-09-23 14:15:39"), "side": "SELL", "qty": "0.0285", "price": "85093.6",
         "realizedPnl": "0", "commission": "0.97"},
        {"id": 2, "time": MS("2026-09-23 20:36:45"), "side": "BUY", "qty": "0.0285", "price": "84300.0",
         "realizedPnl": "22.62", "commission": "0.96"},
    ]
    (t,) = trades_from_fills(fills, CLOSE)
    assert t["side"] == -1
    assert t["entry_bar"] == pd.Timestamp("2026-09-23 14:15", tz="UTC")
    assert t["exit_bar"] == pd.Timestamp("2026-09-23 20:30", tz="UTC")
    assert abs(t["exit_price"] - 84300.0) < 1e-9
    assert abs(t["result"] - (-1) * (84300.0 / 85093.6 - 1) * 100) < 1e-9
    assert abs(t["pnl_usd"] - (22.62 - 0.97 - 0.96)) < 1e-9


def test_partial_fills_average_and_open_trade_has_no_exit():
    fills = [
        {"id": 1, "time": MS("2026-09-23 10:00:05"), "side": "BUY", "qty": "0.01", "price": "100", "realizedPnl": "0", "commission": "0"},
        {"id": 2, "time": MS("2026-09-23 10:00:06"), "side": "BUY", "qty": "0.03", "price": "104", "realizedPnl": "0", "commission": "0"},
        {"id": 3, "time": MS("2026-09-23 12:00:00"), "side": "SELL", "qty": "0.04", "price": "110", "realizedPnl": "0.3", "commission": "0"},
        {"id": 4, "time": MS("2026-09-23 13:00:00"), "side": "SELL", "qty": "0.02", "price": "111", "realizedPnl": "0", "commission": "0"},
    ]
    closed, still_open = trades_from_fills(fills, CLOSE)
    assert abs(closed["entry_price"] - 103.0) < 1e-9          # (0.01*100 + 0.03*104) / 0.04
    assert closed["exit_price"] == 110.0 and closed["side"] == 1
    assert still_open["side"] == -1 and "exit_bar" not in still_open
