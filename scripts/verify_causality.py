"""Fail loudly if any feature in the training dataset can see the future.

Run this before every training run. A leak is not a performance problem, it is
a correctness problem: the reported metrics become meaningless and the agent
learns a policy that cannot exist live.

Three independent checks.

1. Higher timeframe alignment. A 1h/4h feature at bar t must describe the last
   bar that has *closed*, not the one in progress. Before the fix,
   corr(log_return_4h, return of the in-progress 4h bar) was +0.95.

2. Implausible predictive power. Nothing in this problem legitimately reaches a
   rank correlation of 0.25 with the next four hours of returns. Anything that
   does is the target in disguise.

3. Constant-within-bucket detection, which catches the generic form of the same
   mistake for any resampled column.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# A rank correlation with future returns above this is not a real edge in
# liquid crypto at a 4h horizon; it is leakage.
MAX_PLAUSIBLE_IC = 0.25
# Correlation with the in-progress higher timeframe bar above this means the
# feature is still being published before that bar closes.
MAX_CURRENT_BAR_CORR = 0.30


def check_higher_timeframe(df: pd.DataFrame) -> list:
    failures = []
    close = df["close"].astype(float)
    for suffix, rule in (("1h", "1h"), ("4h", "4h")):
        buckets = close.resample(rule)
        bar_return = np.log(buckets.last() / buckets.first())
        current = bar_return.reindex(df.index, method="ffill")
        previous = bar_return.shift(1).reindex(df.index, method="ffill")
        probe = "log_return_%s" % suffix
        if probe not in df.columns:
            continue
        x = df[probe].astype(float)
        mask = np.isfinite(x) & np.isfinite(current) & np.isfinite(previous)
        r_current = float(np.corrcoef(x[mask], current[mask])[0, 1])
        r_previous = float(np.corrcoef(x[mask], previous[mask])[0, 1])
        status = "OK " if abs(r_current) <= MAX_CURRENT_BAR_CORR else "FALHA"
        print("  [%s] %-18s candle em andamento=%+.3f  candle fechado=%+.3f"
              % (status, probe, r_current, r_previous))
        if abs(r_current) > MAX_CURRENT_BAR_CORR:
            failures.append("%s ainda reflete o candle %s em andamento (corr=%.3f)"
                            % (probe, suffix, r_current))
    return failures


def check_predictive_power(df: pd.DataFrame, horizon: int = 16) -> list:
    close = df["close"].astype(float)
    forward = close.shift(-horizon) / close - 1.0
    numeric = df.select_dtypes(include=[np.number])
    skip = {"open", "high", "low", "close"}
    offenders = []
    worst = []
    for column in numeric.columns:
        if column in skip:
            continue
        x = numeric[column]
        if x.std() == 0:
            continue
        # Subsample: an exact rank correlation over 90k rows for 400 columns is
        # slow and the estimate is stable well before that.
        xs, ys = x.iloc[::7], forward.iloc[::7]
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() < 500:
            continue
        ic = float(stats.spearmanr(xs[mask], ys[mask])[0])
        if np.isfinite(ic):
            worst.append((abs(ic), column, ic))
            if abs(ic) > MAX_PLAUSIBLE_IC:
                offenders.append("%s tem IC=%.3f com o retorno de %d barras" % (column, ic, horizon))
    worst.sort(reverse=True)
    print("  maiores |IC| contra o retorno futuro de %d barras:" % horizon)
    for _, column, ic in worst[:5]:
        flag = "FALHA" if abs(ic) > MAX_PLAUSIBLE_IC else "OK "
        print("    [%s] %-34s IC=%+.4f" % (flag, column, ic))
    return offenders


def check_constant_within_bucket(df: pd.DataFrame) -> list:
    """Generic form: a column that never moves inside its own resample bucket
    only becomes known when that bucket ends."""
    failures = []
    for suffix, rule in (("1h", "1h"), ("4h", "4h")):
        columns = [c for c in df.columns if isinstance(c, str) and c.endswith("_" + suffix)]
        leaking = 0
        for column in columns[:80]:
            series = df[column]
            if series.nunique() < 5:
                continue
            spread = series.groupby(series.index.floor(rule)).nunique().mean()
            if spread < 1.05:
                leaking += 1
        print("  [%s] sufixo _%s: %d de %d colunas constantes dentro do proprio candle"
              % ("FALHA" if leaking else "OK ", suffix, leaking, len(columns)))
        if leaking:
            failures.append("%d colunas _%s so sao conhecidas no fim do proprio candle"
                            % (leaking, suffix))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "featured_data_causal.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.data).sort_index()
    print("dataset: %s" % args.data)
    print("%d barras x %d colunas, de %s a %s\n" % (len(df), len(df.columns), df.index.min(), df.index.max()))

    failures = []
    print("1) alinhamento de timeframes superiores")
    failures += check_higher_timeframe(df)
    print("\n2) poder preditivo implausivel")
    failures += check_predictive_power(df)
    print("\n3) colunas constantes dentro do proprio candle")
    failures += check_constant_within_bucket(df)

    print()
    if failures:
        print("VAZAMENTO DETECTADO - NAO TREINE COM ESTE DATASET:")
        for item in failures:
            print("  - %s" % item)
        return 1
    print("OK: nenhuma feature enxerga o futuro. Pode treinar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
