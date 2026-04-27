#!/usr/bin/env python3
"""
Automação: roda a Fase 2 (diagnósticos) iterativamente até obter trials Optuna positivos.

Estratégia:
- Executa scripts/run_trend_specialist_diagnostics.py em dados reais (último ano), com gates relaxados e pruning desligado
- Após cada rodada, lê o estudo Optuna (sqlite em logs/) e verifica o melhor trial completo
- Se best.value > 0, para; caso contrário, ajusta pesos pró-tendência e repete

Pesos ajustados por rodada:
- TREND_PF_WEIGHT (↑), TREND_DD_WEIGHT (↓), TREND_SCALP_WEIGHT (↓ leve), TREND_SCORE_SHIFT (↑)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import optuna


REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"
OPTUNA_DB = LOGS_DIR / "trend_specialist_sac_optuna.db"
STUDY_NAME = "trend_specialist_sac_optimization_v3"


def run_round(env_vars: dict, n_trials: int = 1, last_years: int = 1, timesteps: int = 12000, eval_episodes: int = 4, reset_db: bool = False) -> int:
    env = os.environ.copy()
    env.update(env_vars)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_trend_specialist_diagnostics.py"),
        "--relaxed-gates",
        "--no-pruner",
        "--timesteps", str(timesteps),
        "--eval-episodes", str(eval_episodes),
        "--n-trials", str(n_trials),
        "--last-years", str(last_years),
    ]
    if reset_db:
        cmd.append("--reset-db")
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    return proc.returncode


def read_counts() -> Tuple[int, int]:
    """Retorna (num_positivos_com_surf, total_trials_completos)."""
    if not OPTUNA_DB.exists():
        return 0, 0
    storage = f"sqlite:///{OPTUNA_DB}"
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    except Exception:
        return 0, 0
    positives_with_surf = 0
    completed_total = 0
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            continue
        completed_total += 1
        is_surf = bool(t.user_attrs.get("is_trend_surf", False))
        score_final = float(t.user_attrs.get("score_final", t.value))
        if (t.value > 0.0 or score_final > 0.0) and is_surf:
            positives_with_surf += 1
    return positives_with_surf, completed_total


def next_weights(pf_w: float, dd_w: float, scalp_w: float, shift: float) -> Tuple[float, float, float, float]:
    pf_w = min(1.00, pf_w + 0.10)
    dd_w = max(0.20, dd_w - 0.10)
    scalp_w = max(0.005, scalp_w - 0.002)
    shift = min(2.00, shift + 0.20)
    return pf_w, dd_w, scalp_w, shift


def main() -> None:
    required_positives = int(os.getenv("AUTO_P2_REQUIRED_POSITIVES", "3"))
    max_rounds = int(os.getenv("AUTO_P2_MAX_ROUNDS", "10"))
    last_years = int(os.getenv("AUTO_P2_LAST_YEARS", "1"))
    timesteps = int(os.getenv("AUTO_P2_TIMESTEPS", "12000"))
    eval_episodes = int(os.getenv("AUTO_P2_EVALS", "4"))
    trials_per_round = int(os.getenv("AUTO_P2_TRIALS_PER_ROUND", "2"))

    # Inicialização pró-tendência moderada
    pf_w, dd_w, scalp_w, shift = 0.50, 0.60, 0.010, 1.20

    for round_idx in range(1, max_rounds + 1):
        env_vars = {
            "TREND_PF_WEIGHT": f"{pf_w}",
            "TREND_DD_WEIGHT": f"{dd_w}",
            "TREND_SCALP_WEIGHT": f"{scalp_w}",
            "TREND_SCORE_SHIFT": f"{shift}",
            # Guardas tolerantes
            "TREND_MEAN_REWARD_FLOOR": "-0.2",
            "TREND_GUARD_STEP": "12000",
            "TREND_GUARD_EVALS": "4",
        }
        print(f"[AUTO P2] Rodada {round_idx}: PF={pf_w:.2f}, DD={dd_w:.2f}, SCALP={scalp_w:.3f}, SHIFT={shift:.2f}")
        rc = run_round(env_vars, n_trials=trials_per_round, last_years=last_years, timesteps=timesteps, eval_episodes=eval_episodes, reset_db=(round_idx == 1))
        if rc != 0:
            print(f"[AUTO P2] Rodada {round_idx} terminou com código {rc}. Continuando ajustes...")
        positives_with_surf, completed_total = read_counts()
        print(f"[AUTO P2] Trials completos: {completed_total} | positivos com surf: {positives_with_surf}/{required_positives}")
        if positives_with_surf >= required_positives:
            print("[AUTO P2] Sucesso: número requerido de trials positivos com surf atingido. Encerrando automação.")
            return
        # Ajuste de pesos para próxima rodada
        pf_w, dd_w, scalp_w, shift = next_weights(pf_w, dd_w, scalp_w, shift)

    print("[AUTO P2] Limite de rodadas atingido sem atingir 3 trials positivos com surf. Considere rever dados/ambiente.")


if __name__ == "__main__":
    main()


