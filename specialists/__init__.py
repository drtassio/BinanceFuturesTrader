# -----------------------------------------------------------------------------
# ARQUIVO: specialists/__init__.py
# -----------------------------------------------------------------------------
"""Agentes especialistas: Bull (so comprado) e Bear (so vendido), treinados em
cloud/train_guided.py e operados ao vivo pelo espelho (trading/agent_mirror.py)."""

from specialists.trend_specialist import TrendSpecialist, TrendFollowingEnv
from specialists.base_regime_specialist import BaseRegimeSpecialist
from specialists.bull_specialist import BullSpecialist
from specialists.bear_specialist import BearSpecialist

__all__ = ['TrendSpecialist', 'TrendFollowingEnv', 'BaseRegimeSpecialist', 'BullSpecialist', 'BearSpecialist']
