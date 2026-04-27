# -----------------------------------------------------------------------------
# ARQUIVO: learning/regime_switch.py
# -----------------------------------------------------------------------------
"""
Módulo responsável pela detecção e alternância de regimes de mercado.
Atua como um 'Master' leve que direciona o fluxo para o especialista correto.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from utils.logger import get_logger

logger = get_logger("RegimeSwitch")

class RegimeSwitch:
    """
    Detecta o regime de mercado atual e fornece probabilidades para roteamento.
    
    Regimes suportados:
    - 'bull': Tendência de alta
    - 'bear': Tendência de baixa
    - 'ranger': Mercado lateral / Range
    
    Atualmente utiliza a saída do TrendPredictor (tp_regime_*) como oráculo.
    Futuramente pode integrar um HMM (Hidden Markov Model) dedicado.
    """
    
    REGIMES = ['bull', 'bear', 'ranger']
    
    def __init__(self, smoothing_window: int = 5):
        self.smoothing_window = smoothing_window
        self.history: Dict[str, list] = {r: [] for r in self.REGIMES}
        
    def detect_current_regime(self, df: pd.DataFrame) -> str:
        """
        Retorna o regime dominante atual baseado na última linha do DataFrame.
        Espera que o DataFrame contenha as colunas 'tp_regime_up', 'tp_regime_down', 'tp_regime_sideways'.
        """
        probs = self.get_regime_probabilities(df)
        
        # Retorna a chave com maior probabilidade
        return max(probs, key=probs.get)

    def get_regime_probabilities(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extrai e suaviza as probabilidades de regime do DataFrame.
        """
        if df is None or df.empty:
            logger.warning("DataFrame vazio ou None em detect_current_regime. Defaulting to 'ranger'.")
            return {'bull': 0.0, 'bear': 0.0, 'ranger': 1.0}

        # Mapeamento das colunas do TrendPredictor para nossos regimes
        col_map = {
            'bull': 'tp_regime_up',
            'bear': 'tp_regime_down',
            'ranger': 'tp_regime_sideways'
        }
        
        # Verifica se as colunas existem
        missing = [col for col in col_map.values() if col not in df.columns]
        if missing:
            # Se faltar colunas, tenta inferir de forma simples ou retorna default
            # logger.debug(f"Colunas de regime ausentes: {missing}. Usando heurística simples se possível.")
            return self._fallback_regime_detection(df)

        # Pega a última linha
        last_row = df.iloc[-1]
        
        raw_probs = {}
        for regime, col in col_map.items():
            val = float(last_row.get(col, 0.0))
            raw_probs[regime] = val
            
        # Normaliza para garantir soma 1.0
        total = sum(raw_probs.values())
        if total > 0:
            probs = {k: v / total for k, v in raw_probs.items()}
        else:
            probs = {'bull': 0.0, 'bear': 0.0, 'ranger': 1.0}
            
        return probs

    def _fallback_regime_detection(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Heurística simples caso as colunas tp_regime_* não existam.
        Baseado em EMA Trend e Volatilidade.
        """
        if 'close' not in df.columns:
            return {'bull': 0.0, 'bear': 0.0, 'ranger': 1.0}
            
        # Tenta usar ema_trend se existir
        if 'ema_trend' in df.columns:
            trend = df['ema_trend'].iloc[-1]
            if trend == 1:
                return {'bull': 0.8, 'bear': 0.1, 'ranger': 0.1}
            elif trend == -1:
                return {'bull': 0.1, 'bear': 0.8, 'ranger': 0.1}
        
        # Default para ranger se não tiver informação suficiente
        return {'bull': 0.1, 'bear': 0.1, 'ranger': 0.8}
