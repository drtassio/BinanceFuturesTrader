# -----------------------------------------------------------------------------
# ARQUIVO: specialists/bear_specialist.py
# -----------------------------------------------------------------------------
"""
Especialista 'O Urso' (BearSpecialist).
Focado exclusivamente em tendências de baixa confirmadas.
Opera apenas em posições Short.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from specialists.base_regime_specialist import BaseRegimeSpecialist
from specialists.trend_specialist import TrendFollowingEnv
from config.settings import AIConfig, TradingConfig
from utils.logger import get_logger

logger = get_logger("BearSpecialist")


class BearSpecialist(BaseRegimeSpecialist):
    """
    Especialista em tendências de baixa.
    Treina exclusivamente com dados de regime Bear.
    Opera apenas em posições Short (venda).
    """
    
    def __init__(
        self, 
        config: AIConfig,
        trading_config: TradingConfig,
        input_dim: int,
        profitability_predictor: Optional[Any] = None
    ):
        super().__init__(
            config=config,
            trading_config=trading_config,
            regime_type='bear',
            input_dim=input_dim,
            profitability_predictor=profitability_predictor
        )
        
        # Ajusta hiperparâmetros específicos para Bear
        self.model_hyperparams = getattr(self, "model_hyperparams", {})
        self.model_hyperparams.update({
            "trend_class_weight_boost": 1.2,  # Viés para baixa
            "min_confidence": 0.65,           # Exige alta confiança
            "position_bias": "short_only",     # Apenas Short
        })
        
        logger.info("🐻 Bear Specialist configurado para operações Short only")

    def get_action(self, state: np.ndarray, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retorna ação, com penalidade suave para sinais contrários (Long).
        
        🔬 Dr. Tensor Fix (Finding #1): Replaced hard mask with soft penalty.
        Hard masking breaks policy gradient - agent can't learn WHY buying in Bear is bad.
        Now we flag regime mismatches for penalty in reward function instead.
        
        Args:
            state: Estado do mercado (features)
            context: Contexto adicional
            
        Returns:
            Dicionário com a ação (com flag de regime mismatch se aplicável)
        """
        action_dict = super().get_action(state, context)
        
        # 🔬 Dr. Tensor: Soft penalty instead of hard block
        # The agent can still TRY to buy, but will be penalized in reward
        side = action_dict.get('side')
        if side == 'buy' or side == 1:
            # Flag for reward function to apply penalty (not blocking!)
            action_dict['_regime_mismatch'] = True
            # Reduce confidence to modulate ensemble weight (soft penalty)
            action_dict['confidence'] = action_dict.get('confidence', 0.5) * 0.3
            logger.debug("🐻 Bear: Flagged buy signal for soft penalty (not blocked)")
        else:
            action_dict['_regime_mismatch'] = False
            
        return action_dict


class BearTradingEnv(TrendFollowingEnv):
    """
    Ambiente RL especializado para o Urso.
    Aplica penalidade de regime mismatch diretamente no reward de trade.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 🔬 Dr. Tensor Fix: Permitir exploração de sinais contrários para punição suave
        self.relaxed_gates = True
        # [FIX SHORT-ONLY] BearSpecialist nunca abre Long, nem durante warmup
        self._specialist_short_only = True
    
    def _compute_trade_reward(
        self,
        pnl_realized: float,
        trade_return_pct: float,
        duration_steps: int,
        exit_reason,
        atr_pct: float,
    ) -> float:
        """
        Reward de trade do BearSpecialist.
        Adiciona penalidade quando o agente fecha um trade Long (contra especialidade Bear).
        """
        # Reward base herdado (Sharpe-adjusted, transaction costs, etc.)
        base_reward = super()._compute_trade_reward(
            pnl_realized=pnl_realized,
            trade_return_pct=trade_return_pct,
            duration_steps=duration_steps,
            exit_reason=exit_reason,
            atr_pct=atr_pct,
        )
        
        # [BUG 3 FIX] Penalidade de mismatch REMOVIDA daqui.
        # O compute_scientific_reward() em _step_training já aplica -8.0
        # via info['_regime_mismatch']. Aplicar as duas causava gradientes de
        # escala inconsistente (-10 a -12 vs outros steps em ±1–3).
        # A única fonte de verdade para a penalidade de mismatch é o scientific_reward.
        
        return float(np.clip(base_reward, -25.0, 25.0))
