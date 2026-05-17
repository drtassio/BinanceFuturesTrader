#!/usr/bin/env python3
"""
Orquestrador da suíte de validação científica do BinanceFuturesTrader.

Uso:
    python scripts/validate_pipeline.py            # roda todas as fases
    python scripts/validate_pipeline.py --phase 6  # roda só a fase 6
    python scripts/validate_pipeline.py --skip 7   # pula a fase 7

Saídas:
    reports/validation/report.json  — JSON estruturado de todas as fases
    Exit code 0 se todas passaram, 1 caso contrário.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Garante que o repo está no sys.path quando rodado de qualquer lugar
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validation._common import write_report

PHASE_ORDER = [
    ("1", "scripts.validation.phase_1_imports"),
    ("2", "scripts.validation.phase_2_features"),
    ("3", "scripts.validation.phase_3_regime"),
    ("4", "scripts.validation.phase_4_purged_cv"),
    ("5", "scripts.validation.phase_5_specialists"),
    ("6", "scripts.validation.phase_6_scientific"),
    ("7", "scripts.validation.phase_7_e2e"),
    ("8", "scripts.validation.phase_8_anti_gaming"),
    ("9", "scripts.validation.phase_9_bootstrap"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Suite de validação científica")
    ap.add_argument("--phase", action="append", help="rodar apenas estas fases (1-7)")
    ap.add_argument("--skip", action="append", help="pular estas fases")
    ap.add_argument("--report", default="report.json", help="nome do arquivo de relatório")
    args = ap.parse_args(argv)

    selected = set(args.phase or [str(n) for n, _ in PHASE_ORDER])
    skipped = set(args.skip or [])

    results = []
    for num, mod_name in PHASE_ORDER:
        if num not in selected or num in skipped:
            continue
        try:
            mod = __import__(mod_name, fromlist=["run"])
            phase = mod.run()
            results.append(phase.result)
        except Exception as e:
            print(f"\n❌ FASE {num} falhou ao importar/executar: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            from scripts.validation._common import PhaseResult
            results.append(PhaseResult(
                phase=num, passed=False, duration_s=0.0,
                error=f"{type(e).__name__}: {e}",
            ))

    report_path = write_report(results, filename=args.report)
    n_pass = sum(1 for r in results if r.passed and not r.skipped)
    n_skip = sum(1 for r in results if r.skipped)
    n_fail = sum(1 for r in results if not r.passed)

    print(f"\n{'='*70}\nRESUMO\n{'='*70}")
    for r in results:
        if r.skipped:
            icon, suffix = "⊘", f" — SKIP: {r.skip_reason}"
        elif r.passed:
            icon, suffix = "✅", ""
        else:
            icon, suffix = "❌", ""
        print(f"  {icon} Fase {r.phase:<35} ({r.duration_s}s){suffix}")
    print(f"\n{n_pass} passaram, {n_skip} puladas, {n_fail} falharam")
    print(f"Relatório: {report_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
