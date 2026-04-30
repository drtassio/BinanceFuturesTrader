
import asyncio
import json
import os
import sys
import glob
from datetime import datetime

# Ajuste de Path
sys.path.append(os.getcwd())

from trading.binance_connector import BinanceConnector
from config.settings import active_config

async def compare_bot_vs_market():
    print("🔍 [VALIDATOR] Iniciando auditoria de sincronização...")
    
    # 1. Busca o último estado registrado pelo bot (AIMonitor)
    log_files = glob.glob("logs/ai_events/*.jsonl")
    if not log_files:
        print("❌ Erro: Nenhum log do bot encontrado em logs/ai_events/")
        return
    
    latest_log = max(log_files, key=os.path.getmtime)
    bot_state = None
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                bot_state = json.loads(lines[-1])
    except Exception as e:
        print(f"❌ Erro ao ler log: {e}")
        return

    if not bot_state:
        print("❌ Erro: Não foi possível ler o estado do bot.")
        return

    # 2. Busca dados REAIS da Binance agora
    connector = BinanceConnector(active_config, force_production=True)
    await connector.connect()
    
    symbol = bot_state.get('context', {}).get('symbol', 'BTCUSDT')
    print(f"📊 Auditando Símbolo: {symbol}")
    
    # Busca preço real
    ticker_data = await connector._make_request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol}, signed=False)
    real_price = float(ticker_data['price']) if ticker_data else 0.0
    
    # Busca OBI real
    depth = await connector._make_request("GET", "/fapi/v1/depth", params={"symbol": symbol, "limit": 10}, signed=False)
    real_obi = 0.0
    if depth and 'bids' in depth and 'asks' in depth:
        best_bid_qty = sum(float(b[1]) for b in depth['bids'][:5])
        best_ask_qty = sum(float(a[1]) for a in depth['asks'][:5])
        real_obi = (best_bid_qty - best_ask_qty) / (best_bid_qty + best_ask_qty)

    # 3. COMPARAÇÃO
    bot_context = bot_state.get('context', {})
    bot_price = bot_context.get('price', 0)
    bot_obi = bot_context.get('obi', 0)
    bot_pulse = bot_context.get('tape_pulse', 'UNKNOWN')
    bot_regime = bot_context.get('regime', 'UNKNOWN')
    
    price_diff = abs(bot_price - real_price)
    price_diff_pct = (price_diff / real_price) * 100 if real_price > 0 else 0
    
    print("\n" + "═"*50)
    print(f"🕒 HORA DO LOG (BOT): {bot_state.get('timestamp')}")
    print(f"🕒 HORA ATUAL (MKT): {datetime.utcnow().isoformat()}")
    print("═"*50)
    
    # Validação de Preço
    status_price = "✅ OK" if price_diff_pct < 0.05 else "❌ DISCREPÂNCIA"
    print(f"💰 PREÇO: Bot: ${bot_price:,.2f} | Real: ${real_price:,.2f} | Diff: {price_diff_pct:.4f}% {status_price}")
    
    # Validação de Microestrutura (OBI)
    obi_diff = abs(bot_obi - real_obi)
    status_obi = "✅ OK" if obi_diff < 0.3 else "⚠️ VOLÁTIL"
    print(f"📟 OBI:   Bot: {bot_obi:.4f} | Real: {real_obi:.4f} | Status: {status_obi}")
    
    # Validação de Tape e Regime
    print(f"📟 TAPE:   {bot_pulse}")
    print(f"🐂 REGIME: {bot_regime.upper()}")
    
    print("═"*50)
    if price_diff_pct < 0.1:
        print("🟢 CONCLUSÃO: O bot está vendo a 'fotografia' correta do mercado.")
    else:
        print("🔴 AVISO: O bot está com delay. Verifique sua conexão.")
    
    await connector.close()

if __name__ == "__main__":
    asyncio.run(compare_bot_vs_market())
