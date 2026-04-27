# -----------------------------------------------------------------------------
# ARQUIVO: specialists/__init__.py
# -----------------------------------------------------------------------------

"""
Pacote dos Agentes Especialistas (Specialist Agents).

Na arquitetura de Aprendizado por Reforço Hierárquico (HRL), os especialistas
são modelos de RL de nível inferior, cada um treinado para se destacar em um
regime de mercado específico (ex: tendência, lateralização, alta volatilidade).

Eles recebem um sinal de alto nível do `HRLMaster` e são responsáveis por
refinar a decisão, determinando os pontos ótimos de entrada/saída, o tamanho
preciso da posição e as táticas de gerenciamento de trade dentro de seu
domínio de especialidade.

Componentes:
- TrendSpecialist: Otimizado para mercados em clara tendência de alta ou baixa.
- BullSpecialist: Especialista em tendências de alta (Long only).
- BearSpecialist: Especialista em tendências de baixa (Short only).
- RangerSpecialist: Especialista em mercados laterais (reversão à média).
- SoftGatingNetwork: Rede de gating suave para seleção de especialistas.
- MixtureOfExpertsRouter: Roteador que combina outputs de múltiplos especialistas.
"""

# Core specialists
from specialists.trend_specialist import TrendSpecialist, TrendFollowingEnv
from specialists.base_regime_specialist import BaseRegimeSpecialist
from specialists.bull_specialist import BullSpecialist
from specialists.bear_specialist import BearSpecialist
from specialists.ranger_specialist import RangerSpecialist

# MoE Gating
from specialists.soft_gating_network import (
    SoftGatingNetwork,
    MixtureOfExpertsRouter,
    FeatureAttention,
    create_gating_network
)

__all__ = [
    # Base classes
    'TrendSpecialist',
    'TrendFollowingEnv',
    'BaseRegimeSpecialist',
    
    # Regime specialists
    'BullSpecialist',
    'BearSpecialist',
    'RangerSpecialist',
    
    # MoE components
    'SoftGatingNetwork',
    'MixtureOfExpertsRouter',
    'FeatureAttention',
    'create_gating_network'
]
