"""Retrain the Temporal Autoencoder on the repaired dataset, into a staging dir.

1. Cuts data/featured_data_causal.parquet to the specialists' window and writes
   data/featured_data_causal_leg5m.parquet (same rows as before).
2. Trains the autoencoder on the TRAIN block only (the old one saw rows up to
   2026-03-21, inside validation), with the previous input columns minus the
   *_1m ones (bug B1: frozen in history, not used by the bot). The *_5m inputs
   stay; repair_base_dataset.py rebuilt them from real 5m candles.
3. Writes the 32 latents (hidden_feature_*) into the leg5m dataset.

The model goes to <staging>/autoencoder, never to models_ai/: the live
specialists were trained on the old latents and must be replaced together with
it (scripts/promote_model.py, then the autoencoder files by hand).

    python scripts/retrain_autoencoder_staging.py --staging models_ai/staging_b1b3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import AIConfig  # noqa: E402
from feature_engineering.temporal_autoencoder import TemporalAutoencoderPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal", type=Path, default=ROOT / "data" / "featured_data_causal.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "featured_data_causal_leg5m.parquet")
    parser.add_argument("--start", default="2024-09-18 11:30:00")
    parser.add_argument("--end", default="2026-09-18 12:00:00")
    parser.add_argument("--train-end", default="2026-02-11 11:45:00", help="fim do bloco de treino dos especialistas")
    parser.add_argument("--staging", type=Path, default=ROOT / "models_ai" / "staging_b1b3")
    parser.add_argument("--old-columns", type=Path,
                        default=ROOT / "models_ai" / "autoencoder" / "autoencoder_feature_columns.json")
    parser.add_argument("--hyperparams", type=Path,
                        default=ROOT / "models_ai" / "autoencoder" / "autoencoder_hyperparams_scientific.json")
    args = parser.parse_args()

    df = pd.read_parquet(args.causal).sort_index().loc[args.start:args.end]
    df = df.drop(columns=[c for c in df.columns if c.startswith("hidden_") or c == "sdae_recon_error"])
    print("janela dos especialistas: %d barras (%s .. %s)" % (len(df), df.index.min(), df.index.max()))

    old = json.loads(args.old_columns.read_text(encoding="utf-8"))
    columns = [c for c in old if not (c.lower().endswith("_1m") or "_1m_" in c.lower())]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit("colunas do autoencoder ausentes no dataset: %s" % missing[:10])
    print("entradas do autoencoder: %d (antes %d, sem %d de 1m)" % (len(columns), len(old), len(old) - len(columns)))

    class StagingConfig(AIConfig):
        MODEL_DIR = str(args.staging)

    args.staging.mkdir(parents=True, exist_ok=True)
    pipeline = TemporalAutoencoderPipeline(StagingConfig())
    hyper = json.loads(args.hyperparams.read_text(encoding="utf-8"))
    hyper["input_dim"] = len(columns)
    pipeline.hyperparams = hyper
    pipeline.feature_columns = columns

    train = df.loc[:args.train_end]
    print("treino do autoencoder: %d barras ate %s" % (len(train), train.index.max()))
    t0 = time.time()
    if not pipeline._train_final_model(train):
        raise SystemExit("treino do autoencoder falhou")
    pipeline._save_state()
    print("autoencoder treinado em %.0f s -> %s" % (time.time() - t0, pipeline.model_dir))

    t0 = time.time()
    enriched = pipeline.apply_hidden_features_temporal(df)
    hidden = [c for c in enriched.columns if c.startswith("hidden_feature_")]
    if len(hidden) != 32:
        raise SystemExit("esperava 32 latentes, veio %d" % len(hidden))
    enriched.to_parquet(args.output)
    print("latentes aplicadas em %.0f s | %s: %d x %d" % (
        time.time() - t0, args.output.name, len(enriched), len(enriched.columns)))


if __name__ == "__main__":
    main()
