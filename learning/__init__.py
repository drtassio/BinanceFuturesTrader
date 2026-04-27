# -----------------------------------------------------------------------------
# ARQUIVO: learning/__init__.py
# -----------------------------------------------------------------------------

"""
Pacote de Aprendizado de Máquina e Reforço.

Este pacote contém os componentes centrais para o treinamento e adaptação
dos modelos de inteligência artificial do sistema de trading.

Componentes Principais:
- CurriculumEngine: Orquestra o treinamento progressivo da IA.
- MetaLearner: Implementa meta-aprendizado (MAML) para adaptação rápida.
- RegimeSwitch: Detecta regimes de mercado e roteia para especialistas.
- WalkForwardValidator: Validação Walk-Forward para HPO científico.
- OnlineTrendLearner: Adaptação online RRL para trend surfing.
"""

from .curriculum_engine import CurriculumEngine
# [REMOVIDO] ProfitabilityPredictor - simplificação do pipeline
# from .profitability_predictor import ProfitabilityPredictor
from .online_learner import (
    OnlineTrendLearner,
    AdaptivePositionManager,
    ExperienceBuffer,
    create_online_learner
)

# Define a API pública deste pacote
__all__ = [
    # Core
    'CurriculumEngine',
    # [REMOVIDO] 'ProfitabilityPredictor' - simplificação do pipeline
    
    # Online Learning
    'OnlineTrendLearner',
    'AdaptivePositionManager',
    'ExperienceBuffer',
    'create_online_learner'
]
