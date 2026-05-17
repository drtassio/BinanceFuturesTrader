"""
Utilidades compartilhadas pelas fases de validação:
- gerador de dados sintéticos OHLCV (regimes controlados)
- estruturas de resultado e impressão padronizada
- localização de artefatos reais (dados/modelos) se existirem
"""
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports" / "validation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PhaseResult:
    phase: str
    passed: bool
    duration_s: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PhaseRunner:
    """Context manager para padronizar execução de cada fase."""

    def __init__(self, name: str):
        self.result = PhaseResult(phase=name, passed=False, duration_s=0.0)
        self._t0 = 0.0

    def __enter__(self):
        print(f"\n{'='*70}\nFASE {self.result.phase}\n{'='*70}")
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.result.duration_s = round(time.time() - self._t0, 3)
        if exc is not None:
            self.result.passed = False
            self.result.error = f"{exc_type.__name__}: {exc}"
            print(f"❌ FASE {self.result.phase} EXCEPTION: {self.result.error}")
            return True  # suppress to keep suite running
        if self.result.skipped:
            status = f"⊘ SKIP — {self.result.skip_reason}"
        elif self.result.passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"{status} ({self.result.duration_s}s)")
        return True

    def skip(self, reason: str):
        """Marca a fase como pulada (não conta como falha)."""
        self.result.skipped = True
        self.result.skip_reason = reason
        self.result.passed = True  # SKIP não bloqueia exit code

    def check(self, name: str, condition: bool, detail: str = "", value: Any = None):
        ok = bool(condition)
        self.result.checks.append({"name": name, "passed": ok, "detail": detail, "value": value})
        icon = "✓" if ok else "✗"
        line = f"  {icon} {name}"
        if detail:
            line += f" — {detail}"
        print(line)
        return ok

    def metric(self, key: str, value: Any):
        self.result.metrics[key] = value

    def conclude(self):
        """Marca como passed se todos os checks passaram (e havia pelo menos um)."""
        if not self.result.checks:
            self.result.passed = False
            return
        self.result.passed = all(c["passed"] for c in self.result.checks)


def make_synthetic_ohlcv(
    n: int = 4000,
    seed: int = 42,
    regime_blocks: Optional[List[tuple]] = None,
) -> pd.DataFrame:
    """
    Gera OHLCV sintético com regimes controlados.

    regime_blocks: lista de (n_bars, mu, sigma) por bloco.
    Default: bull(1500)→ranger(1500)→bear(1000).
    """
    rng = np.random.default_rng(seed)
    if regime_blocks is None:
        regime_blocks = [
            (1500, +0.0004, 0.008),   # bull: drift positivo
            (1500, +0.0000, 0.004),   # ranger: drift zero, baixa vol
            (1000, -0.0005, 0.012),   # bear: drift negativo, alta vol
        ]
    rets = []
    regimes = []
    for i, (nb, mu, sig) in enumerate(regime_blocks):
        rets.append(rng.normal(mu, sig, nb))
        regimes.extend([i] * nb)
    rets = np.concatenate(rets)[:n]
    regimes = np.array(regimes[:n])

    price = 50000.0 * np.exp(np.cumsum(rets))
    high = price * (1 + np.abs(rng.normal(0, 0.001, len(price))))
    low = price * (1 - np.abs(rng.normal(0, 0.001, len(price))))
    open_ = np.concatenate([[price[0]], price[:-1]])
    volume = rng.lognormal(6.0, 0.5, len(price))
    index = pd.date_range("2023-01-01", periods=len(price), freq="1h", tz="UTC")

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": price, "volume": volume,
        "_true_regime": regimes,
    }, index=index)
    return df


def find_real_artifacts() -> Dict[str, Optional[Path]]:
    """Localiza dados/modelos reais se existirem (None se não)."""
    candidates = {
        "historical_data_dir": REPO_ROOT / "logs" / "historical_data",
        "base_features": REPO_ROOT / "models_ai" / "base_featured_df.pkl",
        "autoencoder": REPO_ROOT / "models_ai" / "temporal_autoencoder.pth",
        "bull_checkpoint": REPO_ROOT / "checkpoints_ai" / "bull_specialist_sac.zip",
        "bear_checkpoint": REPO_ROOT / "checkpoints_ai" / "bear_specialist_sac.zip",
        "ranger_checkpoint": REPO_ROOT / "checkpoints_ai" / "ranger_specialist_sac.zip",
    }
    return {k: (v if v.exists() else None) for k, v in candidates.items()}


def write_report(results: List[PhaseResult], filename: str = "report.json") -> Path:
    path = REPORTS_DIR / filename
    payload = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "n_phases": len(results),
        "n_passed": sum(1 for r in results if r.passed),
        "n_failed": sum(1 for r in results if not r.passed),
        "phases": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
