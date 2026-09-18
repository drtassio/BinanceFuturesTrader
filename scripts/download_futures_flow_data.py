"""Download causal Binance USD-M Futures flow and funding history.

The Kline endpoint aggregates taker-buy volume at the candle level.  The
derived seller volume, delta and imbalance are therefore available historically
without pretending to reconstruct a full order book.  This script deliberately
does not alter ``featured_data.parquet``; merging must happen in a separately
validated feature-build step.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = f"{BASE_URL}/fapi/v1/klines"
FUNDING_ENDPOINT = f"{BASE_URL}/fapi/v1/fundingRate"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_base_volume",
    "taker_buy_quote_volume", "ignore",
]


def request_json(url: str, params: dict) -> list:
    for attempt in range(5):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Binance rate limit persisted for {url}")


def download_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[list] = []
    cursor = start_ms
    while cursor <= end_ms:
        batch = request_json(KLINES_ENDPOINT, {
            "symbol": symbol, "interval": interval, "startTime": cursor,
            "endTime": end_ms, "limit": 1500,
        })
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 1
        if len(batch) < 1500:
            break
        time.sleep(0.08)

    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS).drop_duplicates("open_time")
    numeric = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
    frame[numeric] = frame[numeric].astype(float)
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    frame = frame.set_index("timestamp").sort_index()
    frame["taker_sell_base_volume"] = (frame["volume"] - frame["taker_buy_base_volume"]).clip(lower=0.0)
    frame["aggressor_delta"] = frame["taker_buy_base_volume"] - frame["taker_sell_base_volume"]
    frame["aggressor_imbalance"] = frame["aggressor_delta"] / frame["volume"].clip(lower=1e-12)
    frame["taker_buy_ratio"] = frame["taker_buy_base_volume"] / frame["volume"].clip(lower=1e-12)
    return frame


def download_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    cursor = start_ms
    while cursor <= end_ms:
        batch = request_json(FUNDING_ENDPOINT, {
            "symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000,
        })
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
        time.sleep(0.08)
    frame = pd.DataFrame(rows).drop_duplicates("fundingTime")
    if frame.empty:
        return pd.DataFrame(columns=["funding_rate"], index=pd.DatetimeIndex([], name="timestamp"))
    frame["timestamp"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True).dt.tz_localize(None)
    frame["funding_rate"] = frame["fundingRate"].astype(float)
    return frame.set_index("timestamp")[["funding_rate"]].sort_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--start", default="2023-08-13T14:30:00Z")
    parser.add_argument("--end", default="2026-03-21T02:45:00Z")
    parser.add_argument("--output-dir", type=Path, default=Path("data/external"))
    args = parser.parse_args()

    start_ms = int(pd.Timestamp(args.start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(args.end).timestamp() * 1000)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"binance_usdm_{args.symbol.lower()}_{args.interval}"

    flow = download_klines(args.symbol.upper(), args.interval, start_ms, end_ms)
    funding = download_funding(args.symbol.upper(), start_ms, end_ms)
    flow_path = args.output_dir / f"{prefix}_flow.parquet"
    funding_path = args.output_dir / f"binance_usdm_{args.symbol.lower()}_funding.parquet"
    flow.to_parquet(flow_path)
    funding.to_parquet(funding_path)
    metadata = {
        "symbol": args.symbol.upper(), "interval": args.interval,
        "requested_start": args.start, "requested_end": args.end,
        "flow_rows": len(flow), "funding_rows": len(funding),
        "flow_path": str(flow_path), "funding_path": str(funding_path),
    }
    (args.output_dir / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
