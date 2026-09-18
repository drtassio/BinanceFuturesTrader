import json

import pandas as pd
import pytest

import feature_engineering.training_schema as schema


def test_live_frame_is_cut_to_training_columns_in_training_order(tmp_path, monkeypatch):
    path = tmp_path / "columns.json"
    path.write_text(json.dumps({"columns": ["close", "atr_15m", "tp_prior_dir"]}))
    monkeypatch.setattr(schema, "SCHEMA_PATH", path)
    live = pd.DataFrame({"tp_uncertainty": [0.2], "tp_prior_dir": [0.8], "ema_trend": [1],
                         "close": [100.0], "atr_15m": [1.5]})
    aligned = schema.align_to_training_frame(live)
    # Live-only columns change the environment's extras and stops; they must go.
    assert list(aligned.columns) == ["close", "atr_15m", "tp_prior_dir"]


def test_missing_training_column_is_an_error_not_a_default(tmp_path, monkeypatch):
    path = tmp_path / "columns.json"
    path.write_text(json.dumps({"columns": ["close", "ml_p_long"]}))
    monkeypatch.setattr(schema, "SCHEMA_PATH", path)
    with pytest.raises(ValueError, match="ml_p_long"):
        schema.align_to_training_frame(pd.DataFrame({"close": [1.0]}))


def test_regime_detector_guard_rejects_a_refit(tmp_path):
    from feature_engineering.crypto_regime_detector import canonical_detector_matches

    refit = tmp_path / "crypto_regime_detector.pkl"
    refit.write_bytes(b"not the detector the dataset was labelled with")
    ok, detail = canonical_detector_matches(str(refit))
    assert ok is False and "esperado" in detail
