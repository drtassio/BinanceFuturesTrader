"""The exact column set the specialists were trained on.

The trading environment decides part of its own observation and mechanics from
which columns the frame happens to contain. Keys missing from the frame are
appended to the observation as extras (zero in training), and a present
'ema_trend', 'tp_uncertainty' or 'tp_duration_median' changes the trend exit,
the stop distance and the grace period. The live pipeline produces 473 columns;
training had 387. Handed a live frame as is, the environment built a 120-wide
frame for a model trained on 123 (the live observation guard refused the model
outright) and priced stops differently from the backtest.

Every frame that did not come from the training parquet goes through
align_to_training_frame before it reaches an environment: live observation,
forward test, teacher replay. models_ai/training_frame_columns.json is written
from the training dataset and changes only with it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "models_ai" / "training_frame_columns.json"


def training_columns() -> List[str]:
    return list(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["columns"])


def align_to_training_frame(frame: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """Same columns, same order as training; fails if one is missing.

    A model trained on another dataset passes that dataset's columns, recorded
    in its run's feature_contract.json as training_frame_columns.
    """
    columns = list(columns) if columns else training_columns()
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError("frame sem %d colunas do treino: %s" % (len(missing), missing[:10]))
    return frame[columns]


def write_schema(dataset: Path) -> List[str]:
    columns = list(pd.read_parquet(dataset).columns)
    SCHEMA_PATH.write_text(json.dumps({"dataset": dataset.name, "columns": columns}, indent=1), encoding="utf-8")
    return columns


if __name__ == "__main__":
    written = write_schema(ROOT / "data" / "featured_data_causal.parquet")
    print("%d colunas gravadas em %s" % (len(written), SCHEMA_PATH))
