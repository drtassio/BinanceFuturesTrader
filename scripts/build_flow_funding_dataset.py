"""Causally enrich the specialist dataset with Binance Futures flow/funding."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FLOW_COLUMNS = ["aggressor_imbalance", "taker_buy_ratio"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("data/featured_data.parquet"))
    parser.add_argument("--flow", type=Path, default=Path("data/external/binance_usdm_btcusdt_15m_flow.parquet"))
    parser.add_argument("--funding", type=Path, default=Path("data/external/binance_usdm_btcusdt_funding.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/featured_data_flow_funding.parquet"))
    args = parser.parse_args()

    base = pd.read_parquet(args.base).sort_index()
    flow = pd.read_parquet(args.flow).sort_index()
    funding = pd.read_parquet(args.funding).sort_index()
    if not isinstance(base.index, pd.DatetimeIndex):
        raise TypeError("base dataset must use a DatetimeIndex")

    # Exact 15m candle timestamps are expected.  merge_asof remains causal if
    # a timestamp is missing: it uses only the last completed observation.
    joined = pd.merge_asof(
        base.reset_index().sort_values("timestamp"),
        flow[FLOW_COLUMNS].reset_index().sort_values("timestamp"),
        on="timestamp", direction="backward",
    ).set_index("timestamp")
    joined = pd.merge_asof(
        joined.reset_index().sort_values("timestamp"),
        funding[["funding_rate"]].reset_index().sort_values("timestamp"),
        on="timestamp", direction="backward",
    ).set_index("timestamp")

    for col in FLOW_COLUMNS + ["funding_rate"]:
        joined[col] = joined[col].ffill().fillna(0.0).astype("float32")

    # Rolling statistics use only current/past completed candles and make the
    # raw flow magnitude comparable across volatility and volume regimes.
    imbalance = joined["aggressor_imbalance"]
    delta_proxy = (imbalance * joined["volume"].astype(float)).astype(float)
    delta_mean = delta_proxy.rolling(32, min_periods=8).mean()
    delta_std = delta_proxy.rolling(32, min_periods=8).std().replace(0.0, np.nan)
    joined["aggressor_delta_z_32"] = ((delta_proxy - delta_mean) / delta_std).fillna(0.0).clip(-8, 8).astype("float32")
    funding_mean = joined["funding_rate"].rolling(32, min_periods=4).mean()
    funding_std = joined["funding_rate"].rolling(32, min_periods=4).std().replace(0.0, np.nan)
    joined["funding_rate_z_32"] = ((joined["funding_rate"] - funding_mean) / funding_std).fillna(0.0).clip(-8, 8).astype("float32")

    # Match the production FeatureEngineeringPipeline naming convention: all
    # non-OHLC features from the primary timeframe receive the _15m suffix.
    joined = joined.rename(columns={
        "aggressor_imbalance": "aggressor_imbalance_15m",
        "taker_buy_ratio": "taker_buy_ratio_15m",
        "funding_rate": "funding_rate_15m",
        "aggressor_delta_z_32": "aggressor_delta_z_32_15m",
        "funding_rate_z_32": "funding_rate_z_32_15m",
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.output)
    print(f"saved={args.output} rows={len(joined)} columns={len(joined.columns)}")
    print("new_columns=aggressor_imbalance_15m,taker_buy_ratio_15m,funding_rate_15m,aggressor_delta_z_32_15m,funding_rate_z_32_15m")


if __name__ == "__main__":
    main()
