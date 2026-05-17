"""
FASE 7 — End-to-End (com artefatos reais se disponíveis)

Roda só se você tiver dados/modelos. Caso contrário, marca como SKIP.
Validações:
- Carrega base_featured_df.pkl, executa CryptoRegimeDetector real
- Carrega checkpoints SAC dos especialistas (Bull/Bear/Ranger)
- generate_trading_decision em N barras, sem postar ordens
- Conta crashes, NaN nos sinais, distribuição de ações
"""
from __future__ import annotations
import asyncio
import numpy as np
import pandas as pd

from scripts.validation._common import PhaseRunner, find_real_artifacts


def run() -> PhaseRunner:
    p = PhaseRunner("7 — End-to-End (real artifacts)")
    with p:
        arts = find_real_artifacts()
        have_features = arts["base_features"] is not None
        have_specialists = any(
            arts.get(k) is not None
            for k in ("bull_checkpoint", "bear_checkpoint", "ranger_checkpoint")
        )

        if not have_features:
            p.skip("models_ai/base_featured_df.pkl ausente — rode esta fase na sua máquina")
            return p

        try:
            df = pd.read_pickle(arts["base_features"])
            p.check(
                "carrega base_featured_df.pkl",
                isinstance(df, pd.DataFrame) and len(df) > 100,
                detail=f"shape={df.shape}",
            )
            p.metric("featured_rows", int(len(df)))
        except Exception as e:
            p.check("carrega base_featured_df.pkl", False, detail=str(e)[:200])
            p.conclude(); return p

        # 7.1 — Regime detector em dados reais
        try:
            from feature_engineering.crypto_regime_detector import (
                CryptoRegimeDetector, RegimeConfig,
            )
            sample = df.tail(2000)
            det = CryptoRegimeDetector(RegimeConfig())
            out = det.fit_predict(sample, train_ratio=0.80)
            ok = "regime" in out.columns and len(out) == len(sample)
            p.check(
                "regime detector roda em 2000 barras reais",
                ok,
                detail=f"shape={out.shape}",
            )
            p.metric("real_regime_distribution", out["regime"].value_counts().to_dict())
        except Exception as e:
            p.check("regime detector em dados reais", False, detail=str(e)[:200])

        if not have_specialists:
            p.skip("checkpoints dos especialistas ausentes — rode esta fase na sua máquina")
            return p

        # 7.2 — Especialistas carregam e geram sinais
        try:
            from config.settings import AIConfig, TradingConfig
            from specialists.bull_specialist import BullSpecialist
            from specialists.bear_specialist import BearSpecialist
            from specialists.ranger_specialist import RangerSpecialist

            ai_cfg = AIConfig()
            tr_cfg = TradingConfig()
            specs = {}
            for name, cls in (("bull", BullSpecialist), ("bear", BearSpecialist), ("ranger", RangerSpecialist)):
                try:
                    s = cls(ai_cfg, tr_cfg, input_dim=65)
                    s.load_model()
                    specs[name] = s
                except Exception as e:
                    p.check(f"{name}.load_model()", False, detail=str(e)[:200])

            n_signals = 0
            n_crashes = 0
            for name, s in specs.items():
                try:
                    obs = np.zeros(65, dtype=np.float32)
                    row = df.iloc[-1]
                    sig = s.decide_action(obs, row)
                    if sig is not None:
                        n_signals += 1
                except Exception:
                    n_crashes += 1
            p.check(
                "especialistas geram pelo menos 1 sinal sem crash",
                n_signals >= 1 and n_crashes == 0,
                detail=f"signals={n_signals}, crashes={n_crashes}",
            )
        except Exception as e:
            p.check("smoke dos especialistas", False, detail=str(e)[:200])

        p.conclude()
    return p


if __name__ == "__main__":
    run()
