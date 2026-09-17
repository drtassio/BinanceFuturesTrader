"""Public BTCUSDT perpetual klines at another timeframe (BTC only), read-only.

    python scripts/fetch_btc_klines.py --interval 5m --start 2024-08-01
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=%s&limit=1500&startTime=%d"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--start", default="2024-08-01")
    args = parser.parse_args()
    t = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    now = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    rows = []
    while t < now:
        for attempt in range(6):
            try:
                batch = json.loads(urllib.request.urlopen(URL % (args.interval, t), timeout=30).read())
                break
            except Exception:
                time.sleep(2 + 3 * attempt)
        else:
            raise RuntimeError("falha ao baixar a partir de %d" % t)
        if not batch:
            break
        rows += batch
        t = batch[-1][0] + 1
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
                                     "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[~df.index.duplicated()].sort_index()
    df = df[["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]].astype(float)
    out = ROOT / "data" / "external" / ("binance_usdm_btcusdt_%s.parquet" % args.interval)
    df.to_parquet(out)
    print("BTCUSDT %s: %d candles de %s a %s -> %s" % (args.interval, len(df), df.index[0], df.index[-1], out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
