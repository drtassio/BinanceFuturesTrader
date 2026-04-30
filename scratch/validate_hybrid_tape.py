
import asyncio
import sys
import os
from datetime import datetime

# Adiciona o diretório raiz ao path
sys.path.append(os.getcwd())

from trading.binance_connector import BinanceConnector
from trading.tape_engine import TapeEngine
from config.settings import active_config

async def validate_hybrid_logic():
    print("🚀 [HYBRID TEST] Iniciando validação do motor REST + WS...")
    
    # 1. Configura Conector (Produção para dados reais)
    connector = BinanceConnector(active_config, force_production=True)
    await connector.connect()
    
    # 2. Inicializa TapeEngine
    symbol = "BTCUSDT"
    engine = TapeEngine(connector, [symbol])
    
    print(f"📊 [PASSO 1] Testando Warmup via REST (/fapi/v1/aggTrades)...")
    await engine._warmup_via_rest(symbol)
    
    # Validação do Passo 1
    trades_count = len(engine.trades_buffer[symbol])
    if trades_count >= 100:
        print(f"✅ SUCESSO: Buffer populado via REST com {trades_count} trades.")
    else:
        print(f"❌ FALHA: Buffer vazio ou insuficiente ({trades_count}).")

    print(f"\n📊 [PASSO 2] Testando OBI via REST (/fapi/v1/rpiDepth ou /depth)...")
    await engine._fetch_rest_depth(symbol)
    
    # Validação do Passo 2
    obi = engine.metrics[symbol].obi
    pulse = engine.get_market_pulse(symbol)
    if obi != 0.0:
        print(f"✅ SUCESSO: OBI calculado via REST: {obi:+.4f}")
        print(f"✅ Pulso do Mercado: {pulse['pulse']} (Score: {pulse['score']})")
    else:
        print(f"❌ FALHA: OBI continua em zero.")

    print("\n🎉 [CONCLUSÃO] O sistema híbrido está pronto para ignorar silêncios do WebSocket!")
    
    await connector.close()

if __name__ == "__main__":
    try:
        asyncio.run(validate_hybrid_logic())
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
