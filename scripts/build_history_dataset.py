"""Training dataset for the trend specialists: 2020 onward, built as the live bot builds it.

The previous dataset began in August 2023 and was assembled offline from a
historical feature frame. It had two problems this one does not:

* Too short to judge trend following. A long-only trend rule loses money in
  sideways or falling years by design and earns it in rising ones; a split of
  a 2.5-year window gives a validation or holdout block that is a single
  regime. From February 2020 the blocks each span several.
* Not what the bot sees. Its regime labels came from Viterbi decoding over the
  whole series and its meta-model columns were fitted on that; live, both are
  computed on a 600-bar window. Neither carried out-of-sample information
  (regime IC ~0.01; meta-model AUC 0.49-0.52), so they are left out here, and
  the live pipeline's values for them are dropped before any environment sees
  a frame (feature_engineering.training_schema).

Input: data/history_live_*.parquet from scripts/build_forward_dataset.py
--skip-regime (live pipeline, read-only). The causal trend structure is then
recomputed over the continuous series, which is the converged value of what
the live window produces (checked: within 0.05 std for the 200-bar EMAs).

    python scripts/build_history_dataset.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_engineering.causal_features import add_trend_structure  # noqa: E402

DROPPED_PREFIXES = ("regime", "tp_", "ml_", "trend_pred_")
BAR = pd.Timedelta(minutes=15)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=str(ROOT / "data" / "history_live_*.parquet"))
    parser.add_argument("--base-schema", type=Path, default=ROOT / "models_ai" / "training_frame_columns.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "history_causal.parquet")
    args = parser.parse_args()

    paths = sorted(p for p in glob.glob(args.inputs) if not p.endswith(".stitched.parquet"))
    metas = [json.loads(Path(p + ".meta.json").read_text(encoding="utf-8")) for p in paths]
    unstable = [p for p, m in zip(paths, metas) if not m.get("stable") or m.get("gaps")]
    if unstable:
        print("BLOQUEADO: segmentos reprovados na validacao contra o bot ao vivo: %s" % unstable)
        return 1
    frame = pd.concat([pd.read_parquet(p) for p in paths]).sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="first")]
    gaps = int((frame.index.to_series().diff().dropna() > BAR).sum())
    if gaps:
        print("BLOQUEADO: %d lacunas entre os segmentos" % gaps)
        return 1

    base = json.loads(args.base_schema.read_text(encoding="utf-8"))["columns"]
    keep = [c for c in base if not str(c).startswith(DROPPED_PREFIXES)]
    missing = [c for c in keep if c not in frame.columns]
    if missing:
        print("BLOQUEADO: colunas do esquema ausentes no historico: %s" % missing[:10])
        return 1
    frame = frame[keep]
    frame, trend_columns = add_trend_structure(frame)
    new_columns = [c for c in trend_columns if c not in keep]
    columns = keep + new_columns
    frame = frame[columns]
    frame.to_parquet(args.output)

    close = frame["close"].astype(float)
    yearly = close.resample("A").last() / close.resample("A").first() - 1
    meta = {"rows": len(frame), "start": str(frame.index[0]), "end": str(frame.index[-1]), "gaps": gaps,
            "segments": paths, "dropped_prefixes": list(DROPPED_PREFIXES), "added_columns": new_columns,
            "columns": columns, "buy_and_hold_by_year": {str(k.year): float(v) for k, v in yearly.items()}}
    Path(str(args.output) + ".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("%d barras de %s a %s | %d colunas (+%s) | B&H por ano: %s" % (
        len(frame), frame.index[0], frame.index[-1], len(columns), new_columns,
        " ".join("%s:%+.0f%%" % (k, 100 * v) for k, v in meta["buy_and_hold_by_year"].items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
