"""
FASE 2 — Features
Validações:
- Indicadores nativos produzem valores finitos e dentro de ranges esperados
- ffill/bfill em vez de fillna(0) preserva mediana
- Look-ahead probe: features de t não devem correlacionar com retorno futuro
  acima do esperado por puro acaso (sanity check)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from scripts.validation._common import PhaseRunner, make_synthetic_ohlcv


def run() -> PhaseRunner:
    p = PhaseRunner("2 — Features")
    with p:
        from feature_engineering.native_indicators import NativeIndicators

        df = make_synthetic_ohlcv(n=3000, seed=42)
        ind = NativeIndicators()

        # 2.1 — Indicadores nativos rodam e produzem finitos
        try:
            close = df["close"]
            rsi = ind.rsi(close, length=14)
            atr = ind.atr(df["high"], df["low"], close, length=14)
            adx_out = ind.adx(df["high"], df["low"], close, length=14)
            # adx pode retornar tupla (adx, +di, -di) ou Series
            adx_series = adx_out[0] if isinstance(adx_out, tuple) else adx_out
            ok_finite = (
                rsi.dropna().between(0, 100).all()
                and atr.dropna().gt(0).all()
                and adx_series.dropna().between(0, 100).all()
            )
            p.check(
                "indicadores nativos: RSI∈[0,100], ATR>0, ADX∈[0,100]",
                ok_finite,
                detail=f"RSI[{rsi.min():.1f},{rsi.max():.1f}] ADX[{adx_series.min():.1f},{adx_series.max():.1f}]",
            )
            p.metric("rsi_range", [float(rsi.min()), float(rsi.max())])
        except Exception as e:
            p.check("indicadores nativos rodam", False, detail=str(e))

        # 2.2 — ffill/bfill preserva mediana melhor que fillna(0)
        series = df["close"].pct_change()
        with_nans = series.copy()
        idx = np.random.RandomState(0).choice(len(with_nans), 200, replace=False)
        with_nans.iloc[idx] = np.nan
        median_true = float(series.median())
        median_zero = float(with_nans.fillna(0).median())
        median_ffill = float(with_nans.ffill().bfill().median())
        err_zero = abs(median_zero - median_true)
        err_ffill = abs(median_ffill - median_true)
        p.check(
            "ffill/bfill aproxima a mediana melhor que fillna(0)",
            err_ffill <= err_zero,
            detail=f"err_ffill={err_ffill:.2e} vs err_zero={err_zero:.2e}",
            value={"err_ffill": err_ffill, "err_zero": err_zero},
        )
        p.metric("median_true", median_true)

        # 2.3 — Look-ahead probe: rolling sem shift deve correlacionar
        # com retorno futuro MAIS que rolling com shift(1)
        ret_next = df["close"].pct_change().shift(-1)
        roll_max_leak = df["high"].rolling(20).max()
        roll_max_safe = df["high"].rolling(20).max().shift(1)
        c_leak = float(roll_max_leak.corr(ret_next))
        c_safe = float(roll_max_safe.corr(ret_next))
        p.check(
            "look-ahead probe: shift(1) reduz correlação espúria com retorno futuro",
            abs(c_safe) <= abs(c_leak) + 1e-6,
            detail=f"|corr_leak|={abs(c_leak):.4f} vs |corr_safe|={abs(c_safe):.4f}",
            value={"corr_leak": c_leak, "corr_safe": c_safe},
        )

        # 2.4 — Sanity: pct_change finito após o primeiro bar
        pct = df["close"].pct_change().iloc[1:]
        p.check(
            "pct_change finito após o primeiro bar",
            np.isfinite(pct).all(),
            detail=f"min={pct.min():.4f} max={pct.max():.4f}",
        )

        p.conclude()
    return p


if __name__ == "__main__":
    run()
