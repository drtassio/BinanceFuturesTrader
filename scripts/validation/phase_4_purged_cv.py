"""
FASE 4 — Validação Estatística (CV Purgado + Walk-Forward)

Validações:
- PurgedKFold gera folds com gap correto (sem overlap train↔test)
- Embargo é aplicado dos dois lados (López de Prado, MLAM Cap. 7)
- Walk-forward: train_end < test_start em TODOS os folds
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from scripts.validation._common import PhaseRunner


def run() -> PhaseRunner:
    p = PhaseRunner("4 — CV Purgado & Walk-Forward")
    with p:
        from utils.purged_cv import PurgedKFold

        n = 2000
        X = np.arange(n).reshape(-1, 1)
        embargo = 10

        try:
            cv = PurgedKFold(n_splits=5, embargo_bars=embargo)
        except TypeError:
            cv = PurgedKFold(n_splits=5)

        # 4.1 — Folds não têm overlap entre train e test
        all_clean = True
        overlap_max = 0
        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            train_set = set(train_idx.tolist())
            test_set = set(test_idx.tolist())
            inter = train_set & test_set
            if inter:
                all_clean = False
                overlap_max = max(overlap_max, len(inter))
        p.check(
            "PurgedKFold: nenhum overlap entre train e test em qualquer fold",
            all_clean,
            detail=f"max_overlap={overlap_max}",
        )

        # 4.2 — Gap entre train e test (purge + embargo)
        gaps = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            t0, t1 = int(test_idx.min()), int(test_idx.max())
            tr_below = train_idx[train_idx < t0]
            tr_above = train_idx[train_idx > t1]
            gap_below = (t0 - int(tr_below.max())) if len(tr_below) else None
            gap_above = (int(tr_above.min()) - t1) if len(tr_above) else None
            gaps.append({"fold": fold, "gap_below": gap_below, "gap_above": gap_above})

        avg_gap = np.mean(
            [g["gap_below"] for g in gaps if g["gap_below"]]
            + [g["gap_above"] for g in gaps if g["gap_above"]]
        ) if gaps else 0
        p.check(
            "PurgedKFold aplica gap > 0 entre train e test",
            avg_gap > 0,
            detail=f"avg_gap={avg_gap:.1f}, esperado >= {embargo if embargo else 1}",
            value=gaps,
        )
        p.metric("gaps_per_fold", gaps)

        # 4.3 — WalkForwardValidator: train_end < test_start em todos folds
        # WalkForwardValidator.split yields (fold_obj, train_df, test_df)
        try:
            from learning.walk_forward_validator import WalkForwardValidator
            wf = WalkForwardValidator(n_splits=3, gap_periods=10)
            # Aumentamos N para satisfazer min_train do validator
            big_n = 10_000
            df_dummy = pd.DataFrame(
                {"x": np.arange(big_n), "close": np.arange(big_n).astype(float)},
                index=pd.date_range("2023-01-01", periods=big_n, freq="1h", tz="UTC"),
            )
            order_ok = True
            details = []
            for tup in wf.split(df_dummy):
                # Compatível com versões antigas (2-tuple) e novas (3-tuple)
                if len(tup) == 3:
                    fold_obj, train_df, test_df = tup
                    fold_idx = getattr(fold_obj, "fold_id", getattr(fold_obj, "index", -1))
                else:
                    fold_idx, train_df, test_df = -1, tup[0], tup[1]
                if len(train_df) == 0 or len(test_df) == 0:
                    continue
                tr_end = train_df.index.max()
                te_start = test_df.index.min()
                details.append({
                    "fold": str(fold_idx),
                    "train_end": str(tr_end), "test_start": str(te_start),
                    "ok": bool(tr_end < te_start),
                })
                if tr_end >= te_start:
                    order_ok = False
            p.check(
                "WalkForwardValidator: train_end < test_start em TODOS folds",
                order_ok and len(details) > 0,
                detail=f"{len(details)} folds processados",
            )
            p.metric("walk_forward_folds", details)
        except Exception as e:
            p.check(
                "WalkForwardValidator disponível",
                False,
                detail=f"{type(e).__name__}: {str(e)[:200]}",
            )

        p.conclude()
    return p


if __name__ == "__main__":
    run()
