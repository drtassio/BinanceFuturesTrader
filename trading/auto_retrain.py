"""Retrain an agent at startup when its model is missing from models_ai.

For each agent in LIVE_AGENTS whose model, scaler or feature contract is
missing, run the guided training (cloud/train_guided.py) with the recipe in
models_ai/training_recipes.json, then promote the checkpoint the training
picked on validation (scripts/promote_model.py). Agents whose files exist are
never retrained. Training takes from about 40 minutes to a few hours.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "models_ai" / "training_recipes.json"
logger = get_logger("AutoRetrain")


def missing_agents(model_dir: Path, agents: List[str]) -> List[str]:
    missing = []
    for agent in agents:
        files = [model_dir / ("%s_specialist_sac.zip" % agent), model_dir / ("%s_specialist_scaler.joblib" % agent),
                 model_dir / ("%s_feature_contract.json" % agent)]
        if not all(f.exists() for f in files):
            missing.append(agent)
    return missing


async def _run(cmd: List[str], env: dict, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(ROOT), env=env, stdout=log,
                                                    stderr=asyncio.subprocess.STDOUT)
        return await proc.wait()


def _newest_run(output: Path, agent: str, before: set) -> Optional[Path]:
    runs = [p for p in output.glob("%s_guided_*" % agent) if p.is_dir() and p not in before
            and (p / "models" / ("%s_specialist_sac.zip" % agent)).exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


async def retrain_missing(model_dir: Path, agents: List[str], output: Path = ROOT / "cloud" / "artifacts" / "auto",
                          extra_args: Optional[List[str]] = None) -> bool:
    """Train and promote every listed agent whose files are missing. False if any fails."""
    todo = missing_agents(model_dir, agents)
    if not todo:
        return True
    recipes = json.loads(RECIPES.read_text(encoding="utf-8"))
    dataset = ROOT / recipes["dataset"]
    if not dataset.exists():
        logger.critical("[RETREINO] Faltam modelos %s, mas o dataset de treino %s nao existe.", todo, dataset)
        return False
    env = dict(os.environ, **recipes.get("env", {}))
    for agent in todo:
        recipe = recipes["agents"].get(agent)
        if recipe is None:
            logger.critical("[RETREINO] Sem receita de treino para %s em %s.", agent, RECIPES)
            return False
        before = set(output.glob("%s_guided_*" % agent))
        log_path = ROOT / "logs" / "auto_retrain" / ("%s.log" % agent)
        cmd = [sys.executable, "cloud/train_guided.py", "--agent", agent, "--data", str(dataset),
               "--rule", str(ROOT / recipe["rule"]), "--output", str(output)] + list(recipe["args"]) + list(extra_args or [])
        logger.warning("[RETREINO] Modelo do agente %s ausente em %s: treinando agora (log: %s). "
                       "Isso leva de 40 minutos a algumas horas.", agent, model_dir, log_path)
        code = await _run(cmd, env, log_path)
        run = _newest_run(output, agent, before)
        if code != 0 or run is None:
            logger.critical("[RETREINO] Treino do agente %s falhou (codigo %s). Veja %s.", agent, code, log_path)
            return False
        promote = [sys.executable, "scripts/promote_model.py", "--agent", agent, "--run", str(run),
                   "--dest", str(model_dir)]
        code = await _run(promote, env, ROOT / "logs" / "auto_retrain" / ("%s_promote.log" % agent))
        if code != 0 or missing_agents(model_dir, [agent]):
            logger.critical("[RETREINO] Promocao do agente %s falhou. Veja logs/auto_retrain/%s_promote.log.", agent, agent)
            return False
        logger.warning("[RETREINO] Agente %s retreinado e promovido de %s.", agent, run)
    return True
