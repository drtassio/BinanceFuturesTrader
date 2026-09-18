"""Enrich featured_data_causal_leg5m.parquet with 32 Temporal Autoencoder latent vectors.

This ensures the RL agents learn the deep correlations between:
- 32 Temporal Autoencoder latent vectors (hidden_feature_0 .. hidden_feature_31)
- Order flow & Aggressor imbalance (aggressor_imbalance, taker_buy_ratio, cz_cvd_z_16)
- 15m, 1h, 4h indicators and trend
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import AIConfig
from feature_engineering.temporal_autoencoder import TemporalAutoencoderPipeline


def main():
    source_path = ROOT / "data" / "featured_data_causal_leg5m.parquet"
    print(f"Carregando {source_path}...")
    df = pd.read_parquet(source_path)
    print(f"Shape original: {df.shape}")
    
    config_ai = AIConfig()
    pipeline = TemporalAutoencoderPipeline(config_ai)
    
    print("Calculando latentes do Temporal Autoencoder na GPU...")
    t0 = time.time()
    enriched_df = pipeline.apply_hidden_features_temporal(df)
    t1 = time.time()
    
    hf_cols = [c for c in enriched_df.columns if "hidden_feature" in c]
    print(f"Latentes geradas ({len(hf_cols)}) em {t1 - t0:.1f}s: {hf_cols[:5]} ... {hf_cols[-1]}")
    print(f"Novo shape: {enriched_df.shape}")
    
    # Save back to data/featured_data_causal_leg5m.parquet
    print(f"Salvando dataset enriquecido em {source_path}...")
    enriched_df.to_parquet(source_path, index=True)
    print("Salvo com sucesso!")


if __name__ == "__main__":
    main()
