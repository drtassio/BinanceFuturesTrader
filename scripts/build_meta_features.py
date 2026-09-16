"""Attach the supervised edge estimate to the training dataset.

Adds four columns consumed by the RL specialists:

    ml_p_long   probability that a long opened here closes profitable after costs
    ml_p_short  the same for a short
    ml_edge     ml_p_long - ml_p_short, the directional opinion in [-1, 1]
    ml_conf     max of the two, how strongly the model feels about it

Every value is produced walk-forward: the model scoring a block is fitted only
on bars whose barrier had already resolved before that block opened. Nothing
here leaks into the RL agent, and the columns are honest out-of-sample
estimates over the whole file, holdout included.

    python scripts/build_meta_features.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learning.meta_labeler import (  # noqa: E402
    BarrierConfig,
    build_labels,
    cross_validate,
    fit_final_bundle,
    select_feature_columns,
    walk_forward_predict,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "featured_data_causal.parquet")
    parser.add_argument("--output", type=Path, default=None, help="default: overwrite --data")
    # Barrier geometry. Wide targets are not a preference, they are arithmetic:
    # a round trip costs ~0.10%, so a 0.4% target hands a quarter of the move to
    # the exchange, while the measured edge per trade is a few basis points.
    parser.add_argument("--profit-atr", type=float, default=8.0)
    parser.add_argument("--stop-atr", type=float, default=3.0)
    parser.add_argument("--max-bars", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=15_000)
    parser.add_argument("--step", type=int, default=5_000)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--bundle", type=Path, default=ROOT / "models_ai" / "meta_labeler.joblib",
                        help="modelo final usado AO VIVO para gerar as colunas ml_*")
    args = parser.parse_args()
    output = args.output or args.data

    df = pd.read_parquet(args.data).sort_index()
    print("dataset %d barras x %d colunas" % (len(df), len(df.columns)))

    cfg = BarrierConfig(profit_atr=args.profit_atr, stop_atr=args.stop_atr, max_bars=args.max_bars)
    labels = build_labels(df, cfg)
    print("barreira %.0f/%.0f ATR em %d barras | acerto long=%.3f short=%.3f | duracao media %.0f barras"
          % (cfg.profit_atr, cfg.stop_atr, cfg.max_bars,
             labels["y_long"].mean(), labels["y_short"].mean(), labels["span"].mean()))

    features = select_feature_columns(df)
    # Drop anything already produced by a previous run, otherwise the model
    # would be handed its own past output as an input.
    features = [c for c in features if not c.startswith("ml_")]
    print("preditores estacionarios: %d" % len(features))

    report = {}
    if not args.skip_cv:
        print("validacao cruzada purgada...")
        report = cross_validate(df, labels, features, n_splits=4, embargo=cfg.max_bars + 96)
        for side in ("long", "short"):
            r = report[side]
            print("  %-5s AUC=%.4f +-%.4f | acerto base=%.3f" % (side, r["auc_mean"], r["auc_std"], r["base_rate"]))

    print("previsao walk-forward...")
    predictions = walk_forward_predict(
        df, labels, features,
        warmup=args.warmup, step=args.step, embargo=cfg.max_bars + 96,
    )
    for column in predictions.columns:
        df[column] = predictions[column].to_numpy()

    df.to_parquet(output)

    # O bot precisa produzir as mesmas quatro colunas em tempo real. As colunas
    # do dataset sao walk-forward (honestas); este pacote e ajustado em todo o
    # historico resolvido e serve SO para inferencia ao vivo — nunca para
    # pontuar linhas historicas, que ele ja viu.
    import joblib
    selected = predictions.attrs.get("selected_features") or []
    bundle = fit_final_bundle(df, labels, selected, cfg, embargo=cfg.max_bars + 96)
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.bundle)
    print("modelo ao vivo salvo em %s (%d features, treinado ate %s)"
          % (args.bundle, len(selected), bundle["trained_until"]))

    meta = {
        "barrier": {"profit_atr": cfg.profit_atr, "stop_atr": cfg.stop_atr, "max_bars": cfg.max_bars},
        "predictors": len(features),
        "warmup": args.warmup,
        "step": args.step,
        "cross_validation": report,
        "columns_added": list(predictions.columns),
        "selected_features": list(selected),
        "live_bundle": str(args.bundle),
    }
    Path(str(output) + ".meta_model.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    scored = predictions["ml_p_long"] != 0.5
    print("salvo em %s" % output)
    print("barras com opiniao do modelo: %d de %d (%.1f%%)"
          % (scored.sum(), len(df), 100.0 * scored.sum() / len(df)))
    print("ml_edge: media=%+.4f desvio=%.4f" % (df["ml_edge"].mean(), df["ml_edge"].std()))


if __name__ == "__main__":
    main()
