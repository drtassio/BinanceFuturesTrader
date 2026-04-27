"""
config/regime_constants.py
--------------------------
[BUG 10 FIX] Configuração CANÔNICA do CryptoRegimeDetector.

Uma única fonte de verdade para os pesos do detector de regime.
Antes, scientific_data_processor.py e base_regime_specialist.py usavam
pesos diferentes, causando divergência entre os labels do dataset e a
filtragem dos dados de treino dos especialistas.

Regra: SEMPRE importar CANONICAL_REGIME_CONFIG daqui. NUNCA definir
pesos locais de RegimeConfig em outros módulos.
"""

try:
    from learning.regime_switch import RegimeConfig
    
    # weight_funding mantido em 0.10 (sinal real mas ruidoso no crypto).
    # weight_gmm=0.35 é valor de consenso entre 0.30 (sdp) e 0.40 (base_regime).
    CANONICAL_REGIME_CONFIG = RegimeConfig(
        adx_period=9,
        adx_trend_threshold=20.0,
        min_regime_duration=8,
        weight_hmm=0.35,
        weight_gmm=0.35,
        weight_adx=0.20,
        weight_funding=0.10,
    )

except Exception:
    # Fallback seguro se RegimeConfig não estiver disponível no contexto
    CANONICAL_REGIME_CONFIG = None
