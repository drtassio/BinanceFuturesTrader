
import asyncio
import json
import os
import sys
import glob
import time
from datetime import datetime
from trading.binance_connector import BinanceConnector
from config.settings import active_config

# Ajuste de Path
sys.path.append(os.getcwd())

AUDIT_LOG_PATH = "logs/market_audits.log"

async def perform_audit(connector, last_known_line=None):
    """Realiza uma auditoria comparando o log do bot com a Binance Real."""
    
    # 1. Busca o último estado registrado pelo bot (AIMonitor)
    log_files = glob.glob("logs/ai_events/*.jsonl")
    if not log_files:
        return None, "Nenhum log encontrado"
    
    latest_log = max(log_files, key=os.path.getmtime)
    bot_state = None
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                bot_state = json.loads(lines[-1])
    except Exception as e:
        return None, f"Erro ao ler log: {e}"

    if not bot_state:
        return None, "Estado do bot vazio"

    # Se já processamos essa linha e não é um evento especial, pula
    if last_known_line == str(bot_state):
        return last_known_line, "Sem novidades"

    symbol = bot_state.get('context', {}).get('symbol', 'BTCUSDT')
    
    # 2. Busca dados REAIS da Binance
    ticker_data = await connector._make_request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol}, signed=False)
    real_price = float(ticker_data['price']) if ticker_data else 0.0
    
    # 3. COMPARAÇÃO
    bot_context = bot_state.get('context', {})
    bot_price = bot_context.get('price', 0)
    bot_pulse = bot_context.get('tape_pulse', 'UNKNOWN')
    event_type = bot_state.get('event_type', 'HEARTBEAT')
    
    # Corrige busca de timestamp
    bot_ts = bot_state.get('timestamp') or bot_state.get('event_time') or "N/A"
    
    price_diff_pct = (abs(bot_price - real_price) / real_price * 100) if real_price > 0 else 0
    status = "✅ OK" if price_diff_pct < 0.1 else "❌ LAG"
    
    audit_entry = (
        f"[{datetime.now().strftime('%H:%M:%S')}] EVENT: {event_type} | "
        f"Price Bot: ${bot_price:,.2f} | Real: ${real_price:,.2f} | "
        f"Diff: {price_diff_pct:.4f}% | Pulse: {bot_pulse} | Status: {status}\n"
    )
    
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(audit_entry)
        
    print(audit_entry.strip())
    return str(bot_state), "Auditado"

async def market_auditor_loop():
    print(f"🚀 [AUDITOR] Iniciando monitoramento contínuo...")
    print(f"📄 Salvando auditorias em: {AUDIT_LOG_PATH}")
    
    connector = BinanceConnector(active_config, force_production=True)
    await connector.connect()
    
    last_line = None
    last_10min_audit = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Auditoria Periódica (A cada 10 minutos = 600 segundos)
            if current_time - last_10min_audit >= 600:
                last_line, _ = await perform_audit(connector, None) # Força auditoria
                last_10min_audit = current_time
            
            # Monitoramento de Eventos (Ordens ou Mudanças de Decisão)
            # Checa a cada 5 segundos se houve algo novo no log
            new_last_line, msg = await perform_audit(connector, last_line)
            if "Sem novidades" not in msg:
                last_line = new_last_line
                
            await asyncio.sleep(5)
            
    except KeyboardInterrupt:
        print("🛑 Auditor finalizado pelo usuário.")
    finally:
        await connector.close()

if __name__ == "__main__":
    if not os.path.exists("logs"):
        os.makedirs("logs")
    asyncio.run(market_auditor_loop())
