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
    from feature_engineering.crypto_regime_detector import RegimeConfig
    
    # [FIX] ADX elevado para corrigir Ranger=2.3% (root cause: HMM+GMM=70% suprimia ADX).
    # Com ADX=0.45, Ranger esperado 12-15% (lateral market detection correta).
    # weight_funding=0.0: funding é fator de risco, não voto de regime (F1 FIX).
    CANONICAL_REGIME_CONFIG = RegimeConfig(
        adx_period=9,
        adx_trend_threshold=20.0,
        min_regime_duration=5,
        weight_hmm=0.30,
        weight_gmm=0.25,
        weight_adx=0.45,
        weight_funding=0.0,
        drawdown_window=20,       # Sincronizado
        autocorr_window=15,       # Sincronizado
        confidence_threshold=0.75,  # [M6 FIX] Threshold crypto-específico
    )

except Exception:
    # Fallback seguro se RegimeConfig não estiver disponível no contexto
    CANONICAL_REGIME_CONFIG = None
