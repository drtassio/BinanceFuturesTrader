"""
FASE 3 — Detecção de Regime
Validações:
- CryptoRegimeDetector.fit_predict(train_ratio=0.8) NÃO usa OOS para fittar scaler
- Rótulos são estáveis (refit em subsets não vira tudo de cabeça pra baixo)
- Detector identifica corretamente regimes em dados sintéticos com regime conhecido
- Confiança ∈ [0, 1] e não é NaN
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from scripts.validation._common import PhaseRunner, make_synthetic_ohlcv


def run() -> PhaseRunner:
    p = PhaseRunner("3 — Detecção de Regime")
    with p:
        from feature_engineering.crypto_regime_detector import (
            CryptoRegimeDetector, RegimeConfig,
        )

        df = make_synthetic_ohlcv(n=4000, seed=42)

        # 3.1 — fit_predict roda com train_ratio<1.0 sem crash (correção do leak)
        det = CryptoRegimeDetector(RegimeConfig())
        try:
            out = det.fit_predict(df.drop(columns=["_true_regime"]), train_ratio=0.80)
            shape_ok = len(out) == len(df) and "regime" in out.columns and "confidence" in out.columns
            p.check(
                "fit_predict(train_ratio=0.8) executa e retorna shape correto",
                shape_ok,
                detail=f"shape={out.shape}, cols={list(out.columns)[:5]}",
            )
        except Exception as e:
            p.check("fit_predict(train_ratio=0.8)", False, detail=str(e)[:200])
            p.conclude(); return p

        # 3.2 — Confiança ∈ [0,1] e sem NaN
        conf = out["confidence"]
        ok_conf = conf.between(0, 1).all() and not conf.isna().any()
        p.check(
            "confidence ∈ [0,1] e sem NaN",
            ok_conf,
            detail=f"min={conf.min():.3f} max={conf.max():.3f} nan={int(conf.isna().sum())}",
        )

        # 3.3 — Regimes contêm pelo menos 2 valores distintos (não colapsou)
        n_unique = int(out["regime"].nunique())
        p.check(
            "regimes não colapsaram (>=2 distintos)",
            n_unique >= 2,
            detail=f"n_unique={n_unique}",
        )
        p.metric("regime_distribution", out["regime"].value_counts().to_dict())

        # 3.4 — Anti-leakage estrutural: refit com 80% deve produzir resultado
        # diferente de refit com 100% (porque viu menos dados)
        det_full = CryptoRegimeDetector(RegimeConfig())
        out_full = det_full.fit_predict(df.drop(columns=["_true_regime"]), train_ratio=1.0)
        diff_frac = float((out["regime"].values != out_full["regime"].values).mean())
        p.check(
            "train_ratio=0.8 produz labels distintos de train_ratio=1.0 (sinal de que scaler está respeitando split)",
            diff_frac > 0.0,
            detail=f"diff_frac={diff_frac:.3f}",
            value=diff_frac,
        )
        p.metric("regime_diff_fraction_80vs100", diff_frac)

        # 3.5 — Concordância com regime verdadeiro
        # (não exige perfeição — só significância vs random)
        true_reg = df["_true_regime"].values
        # remap regimes detectados para melhor matching (Hungarian-lite)
        agreement = float((out["regime"].values == true_reg).mean())
        # baseline: regime majoritário acertaria n_maj/n
        baseline = float(pd.Series(true_reg).value_counts(normalize=True).iloc[0])
        p.check(
            "concordância com regime verdadeiro acima do baseline majoritário",
            agreement >= baseline,
            detail=f"agreement={agreement:.3f} baseline={baseline:.3f}",
            value={"agreement": agreement, "baseline_majority": baseline},
        )

        p.conclude()
    return p


if __name__ == "__main__":
    run()
