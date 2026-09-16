"""Build the leak-free training dataset for the regime specialists.

Merges the raw Binance futures tape (aggressor volume, trade count, quote
volume) and funding onto the featured 15m frame, then applies the shared causal
treatment from feature_engineering.causal_features. The live pipeline calls the
same functions, so a feature used in training is always reproducible live.

    python scripts/build_causal_dataset.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_engineering.causal_features import build_causal_features  # noqa: E402

# Raw tape columns carried through so the tape features can be rebuilt.
TAPE_COLUMNS = [
    "aggressor_imbalance",
    "taker_buy_ratio",
    "trade_count",
    "quote_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def _merge_backward(left: pd.DataFrame, right: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Attach right-hand columns using only observations already published."""
    available = [c for c in columns if c in right.columns]
    if not available:
        return left
    merged = pd.merge_asof(
        left.reset_index().rename(columns={left.index.name or "index": "timestamp"}).sort_values("timestamp"),
        right[available].reset_index().rename(columns={right.index.name or "index": "timestamp"}).sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).set_index("timestamp")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "data" / "featured_data.parquet")
    parser.add_argument("--flow", type=Path, default=ROOT / "data" / "external" / "binance_usdm_btcusdt_15m_flow.parquet")
    parser.add_argument("--funding", type=Path, default=ROOT / "data" / "external" / "binance_usdm_btcusdt_funding.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "featured_data_causal.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.base).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("base dataset must use a DatetimeIndex")
    columns_before = len(df.columns)

    if args.flow.exists():
        flow = pd.read_parquet(args.flow).sort_index()
        df = _merge_backward(df, flow, TAPE_COLUMNS)
    if args.funding.exists():
        funding = pd.read_parquet(args.funding).sort_index()
        df = _merge_backward(df, funding, ["funding_rate"])

    for col in TAPE_COLUMNS + ["funding_rate"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Colunas "*_tf_*" vem de um construtor historico antigo e NAO existem no
    # pipeline ao vivo (scripts/verify_live_parity.py lista 38 delas). Um modelo
    # que as usasse receberia zeros em producao. Elas saem antes de qualquer
    # coisa, para que nenhum seletor de features possa escolhe-las.
    legacy = [c for c in df.columns if isinstance(c, str) and "_tf_" in c]
    df = df.drop(columns=legacy)
    print("colunas legadas sem equivalente ao vivo removidas: %d" % len(legacy))

    df, meta = build_causal_features(df)
    meta["dropped_legacy_columns"] = legacy

    # The higher timeframe shift leaves genuine warm-up NaNs at the head, and
    # the rolling windows need history. Filling either would put back exactly
    # the value the shift removed, so the warm-up is dropped instead.
    warmup = 400
    df = df.iloc[warmup:]
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.ffill().fillna(0.0)
    df = df.replace([np.inf, -np.inf], 0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)

    meta.update({
        "base": str(args.base),
        "rows": len(df),
        "columns_before": columns_before,
        "columns_after": len(df.columns),
        "warmup_rows_dropped": warmup,
        "start": str(df.index.min()),
        "end": str(df.index.max()),
    })
    Path(str(args.output) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("trend_features", "tape_features")}, indent=2))
    print("trend_features=%d tape_features=%d" % (len(meta["trend_features"]), len(meta["tape_features"])))
    print("tape:", ", ".join(meta["tape_features"]))


if __name__ == "__main__":
    main()
