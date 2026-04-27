#!/usr/bin/env python3
"""
Ferramenta de diagnóstico para os estudos do TrendSpecialist/Optuna.

O script percorre o estudo persistido (sqlite) e gera um sumário dos trials,
incluindo motivo de poda, passo em que ocorreu e estatísticas básicas das
avaliações gravadas no diretório de logs (`logs/optuna_eval/...`).

Exemplo de uso:

    python scripts/analyze_optuna_trials.py \
        --study-name trend_specialist_sac_optimization_v3 \
        --storage sqlite:///logs/trend_specialist_sac_optuna.db

    python scripts/analyze_optuna_trials.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import optuna

DEFAULT_STUDY_NAME = "trend_specialist_sac_optimization_v3"
DEFAULT_STORAGE = "sqlite:///logs/trend_specialist_sac_optuna.db"
DEFAULT_LOG_DIR = Path("logs") / "optuna_eval" / "OptunaFinancialMetrics"


def load_evaluations(trial_number: int, log_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    trial_dir = log_dir / f"trial_{trial_number}"
    eval_file = trial_dir / "evaluations.npz"
    if not eval_file.exists():
        return None
    try:
        data = np.load(eval_file)
    except Exception:
        return None
    return {k: data[k] for k in data.files}


def summarize_evaluations(eval_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    timesteps = eval_data.get("timesteps")
    results = eval_data.get("results")
    if timesteps is None or results is None:
        return {}
    flat_rewards: List[float] = []
    for episode_rewards in results:
        try:
            flat_rewards.extend(float(r) for r in episode_rewards)
        except Exception:
            continue
    if not flat_rewards:
        return {"num_evals": len(timesteps)}
    return {
        "num_evals": len(timesteps),
        "mean_reward": statistics.mean(flat_rewards),
        "median_reward": statistics.median(flat_rewards),
        "min_reward": min(flat_rewards),
        "max_reward": max(flat_rewards),
    }


def build_trial_row(trial: optuna.trial.FrozenTrial, eval_summary: Optional[Dict[str, float]]) -> Dict[str, object]:
    attrs = trial.user_attrs
    row: Dict[str, object] = {
        "number": trial.number,
        "state": str(trial.state),
        "value": trial.value,
        "pruned_timesteps": attrs.get("pruned_timesteps"),
        "forced_prune_reason": attrs.get("forced_prune_reason"),
        "best_mean_reward": attrs.get("best_mean_reward"),
        "gate_stats": attrs.get("gate_stats"),
    }
    if eval_summary:
        row.update({f"eval_{k}": v for k, v in eval_summary.items()})
    return row


def print_table(rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        print("Nenhum trial encontrado.")
        return
    headers = list(rows[0].keys())
    widths = {h: max(len(h), *(len(str(row.get(h, ""))) for row in rows)) for h in headers}
    def _fmt(row: Dict[str, object]) -> str:
        return " | ".join(f"{str(row.get(h, '')):<{widths[h]}}" for h in headers)
    print(_fmt({h: h for h in headers}))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(_fmt(row))


def dump_json(rows: Iterable[Dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(list(rows), fh, ensure_ascii=False, indent=2)
    print(f"Resumo salvo em {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnóstico de trials Optuna do TrendSpecialist.")
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME, help="Nome do estudo Optuna.")
    parser.add_argument("--storage", default=DEFAULT_STORAGE, help="URL de storage do Optuna (sqlite://...).")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Diretório base dos logs de avaliação (npz).")
    parser.add_argument("--export-json", type=Path, help="Opcional: caminho para exportar o resumo em JSON.")
    parser.add_argument("--summary-only", action="store_true", help="Mostra apenas contagem de estados (pruned, complete, fail).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    log_dir = Path(args.log_dir)
    rows: List[Dict[str, object]] = []

    for trial in study.get_trials(deepcopy=False):
        eval_summary = None
        eval_data = load_evaluations(trial.number, log_dir)
        if eval_data:
            eval_summary = summarize_evaluations(eval_data)
        row = build_trial_row(trial, eval_summary)
        rows.append(row)

    if args.summary_only:
        counts: Dict[str, int] = {}
        for row in rows:
            state = row.get("state", "UNKNOWN")
            counts[state] = counts.get(state, 0) + 1
        print("Resumo por estado:")
        for state, count in sorted(counts.items()):
            print(f"  {state}: {count}")
        reason_counts: Dict[str, int] = {}
        for row in rows:
            reason = row.get("forced_prune_reason")
            if reason:
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
        if reason_counts:
            print("Motivos de poda:")
            for reason, count in sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True):
                print(f"  {reason}: {count}")
    else:
        print_table(rows)

    if args.export_json:
        dump_json(rows, args.export_json)


if __name__ == "__main__":
    main()
