"""Repair the base 15m dataset before the causal build (bugs B1 and B3).

B1. The *_5m indicator columns of data/featured_data.parquet were merged from a
    5m cache that ends in June 2024, so from then on every row repeats the last
    value (rsi_5m is identical on 75% of the rows since 2024-09). Here they are
    recomputed from the real 5m candles in data/external_5m with the code the
    live bot runs (NativeIndicators.calculate_all_features, "_5m" suffix, then
    merge_asof backward onto the 15m index, as FeatureEngineeringPipeline.
    create_features does). Rows before the 5m history starts keep their old
    values; they are outside the training window.

B3. regime / regime_confidence are relabelled with the causal detector
    (forward filtering and hysteresis instead of Viterbi and segment smoothing).

The input is never overwritten.

    python scripts/repair_base_dataset.py
    python scripts/build_causal_dataset.py --base data/featured_data_repaired.parquet \
        --output data/featured_data_causal.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_engineering.crypto_regime_detector import CryptoRegimeDetector  # noqa: E402
from feature_engineering.native_indicators import NativeIndicators  # noqa: E402

OHLCV = ["open", "high", "low", "close", "volume"]


def recompute_5m(base: pd.DataFrame, five: pd.DataFrame) -> tuple:
    featured = NativeIndicators.calculate_all_features(five[OHLCV].copy(), "BTCUSDT", "5m")
    featured = featured.loc[:, ~featured.columns.duplicated()]
    featured = featured.rename(columns={c: "%s_5m" % c for c in featured.columns})
    # "*_tf_*" are legacy columns build_causal_dataset.py drops anyway.
    old_cols = [c for c in base.columns if c.endswith("_5m") and "_tf_" not in c]
    missing = sorted(set(old_cols) - set(featured.columns))
    if missing:
        raise SystemExit("5m recalculado nao gerou %d colunas da base: %s" % (len(missing), missing[:10]))
    merged = pd.merge_asof(
        base[[]].sort_index(), featured[old_cols].sort_index(),
        left_index=True, right_index=True, direction="backward",
    )
    covered = base.index >= featured.index.min()
    out = base.copy()
    for col in old_cols:
        values = merged[col].ffill().fillna(0.0)
        out.loc[covered, col] = values[covered].astype(out[col].dtype, errors="ignore")
    return out, old_cols, featured.index.min(), int(covered.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "data" / "featured_data.parquet")
    parser.add_argument("--five-minute", type=Path,
                        default=ROOT / "data" / "external_5m" / "binance_usdm_btcusdt_5m_flow.parquet")
    parser.add_argument("--detector", type=Path, default=ROOT / "models_ai" / "crypto_regime_detector.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "featured_data_repaired.parquet")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    base = pd.read_parquet(args.base).sort_index()
    five = pd.read_parquet(args.five_minute).sort_index()
    print("base %d x %d (%s .. %s) | 5m %d (%s .. %s)" % (
        len(base), len(base.columns), base.index.min(), base.index.max(),
        len(five), five.index.min(), five.index.max()))

    before = base.loc[base.index >= five.index.min() + pd.Timedelta(days=30), "rsi_5m"]
    out, cols_5m, first_5m, covered = recompute_5m(base, five)
    after = out.loc[before.index, "rsi_5m"]
    print("5m: %d colunas recalculadas em %d linhas | rsi_5m repetido: %.1f%% -> %.1f%%" % (
        len(cols_5m), covered, 100 * (before.diff() == 0).mean(), 100 * (after.diff() == 0).mean()))

    detector = CryptoRegimeDetector.load(str(args.detector))
    if detector is None or not detector.is_trained:
        raise SystemExit("detector de regime ausente em %s" % args.detector)
    labels = detector.predict(out)
    changed = float((labels["regime"].to_numpy() != out["regime"].to_numpy()).mean())
    out["regime"] = labels["regime"].to_numpy().astype(out["regime"].dtype)
    out["regime_confidence"] = labels["confidence"].to_numpy()
    if "regime_name" in out.columns:
        out["regime_name"] = labels["regime_name"].to_numpy()
    print("regime causal: %.1f%% das barras mudaram de rotulo | %s" % (
        100 * changed, out["regime"].value_counts(normalize=True).round(3).to_dict()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output)
    meta = {
        "base": str(args.base), "five_minute": str(args.five_minute), "detector": str(args.detector),
        "columns_5m": cols_5m, "first_5m_bar": str(first_5m), "rows_5m_recomputed": covered,
        "regime_relabelled_fraction": changed,
    }
    Path(str(args.output) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("gravado %s" % args.output)


if __name__ == "__main__":
    main()
