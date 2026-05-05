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
    
    # [FIX] Configuração sincronizada com a volatilidade atual do mercado.
    CANONICAL_REGIME_CONFIG = RegimeConfig(
        adx_period=9,
        adx_trend_threshold=20.0,
        min_regime_duration=8,
        weight_hmm=0.35,
        weight_gmm=0.35,
        weight_adx=0.20,
        weight_funding=0.10,
        drawdown_window=20,    # Sincronizado
        autocorr_window=15     # Sincronizado
    )

except Exception:
    # Fallback seguro se RegimeConfig não estiver disponível no contexto
    CANONICAL_REGIME_CONFIG = None
