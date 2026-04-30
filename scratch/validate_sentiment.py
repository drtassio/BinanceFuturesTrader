
import asyncio
import sys
import os
from datetime import datetime

# Adiciona o diretório raiz ao path
sys.path.append(os.getcwd())

from trading.onchain_engine import OnChainEngine
from config.settings import active_config

async def validate_sentiment_bridge():
    print("🧠 [SENTIMENT TEST] Iniciando validação da ponte de sentimento...")
    
    # 1. Inicializa o Engine
    engine = OnChainEngine(active_config)
    
    # O valor inicial DEVE ser 50 (cache padrão)
    print(f"📍 Valor Inicial no Cache: {engine.sentiment_cache['value']}")
    
    # 2. Força atualização Sincronizada (como no novo start())
    print("📡 Solicitando dados da Alternative.me...")
    await engine._refresh_sentiment()
    
    # 3. Verifica se o valor mudou
    current_val = engine.sentiment_cache['value']
    current_signal = engine.sentiment_cache['signal']
    
    print(f"📊 Valor Pós-Atualização: {current_val} ({current_signal})")
    
    if current_val != 50:
        print(f"✅ SUCESSO: O motor conseguiu baixar o valor real ({current_val}).")
    else:
        print(f"❌ FALHA: O motor continuou em 50. Verifique conexão com alternative.me.")

    # 4. Testa a interface que o run_bot usa
    pulse = engine.get_onchain_sentiment_signal()
    print(f"🔌 Interface do run_bot retornou: {pulse['score']} - {pulse['signal']}")
    
    if pulse['score'] == current_val:
        print("✅ SUCESSO: A interface está transmitindo o valor correto.")
    else:
        print("❌ FALHA: Discrepância entre cache e interface.")

    print("\n💡 NOTA: Se este teste der SUCESSO e o bot continuar mostrando 50, o problema está no loop de atualização do run_bot.py.")

if __name__ == "__main__":
    try:
        asyncio.run(validate_sentiment_bridge())
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
