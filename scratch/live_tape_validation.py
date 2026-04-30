
import asyncio
import sys
import os
from datetime import datetime

# Adiciona o diretório raiz ao path para encontrar os módulos do bot
sys.path.append(os.getcwd())

from trading.binance_connector import BinanceConnector
from trading.tape_engine import TapeEngine
from config.settings import active_config

async def run_live_validation():
    print("📡 [LIVE TEST] Conectando aos servidores de Produção da Binance...")
    
    # 1. Configurar Conector (Forçamos Mainnet para pegar dados reais de alto volume)
    connector = BinanceConnector(active_config, force_production=True)
    await connector.connect()
    
    # 2. Inicializar TapeEngine
    symbol = "BTCUSDT"
    engine = TapeEngine(connector, [symbol])
    engine.is_running = True
    
    print(f"✅ Conectado. Aguardando trades reais de {symbol}...")
    print("---------------------------------------------------------")

    # Contador para o teste fechar sozinho após validar
    trades_collected = 0
    start_time = datetime.now()

    async def custom_handler(msg):
        nonlocal trades_collected
        
        # DIAGNÓSTICO BRUTO: Imprime as primeiras 5 mensagens não importa o que sejam
        if engine._msg_received_count < 5:
            print(f"\n🔍 [RAW MSG #{engine._msg_received_count}] {str(msg)[:200]}...")
        
        # Passa para o motor real processar
        await engine._handle_ws_message(msg)
        
        # Verifica o tipo de evento recebido
        data = msg.get('data', msg)
        event = data.get('e', 'desconhecido')
        
        # Verifica se o buffer cresceu
        current_len = len(engine.trades_buffer[symbol])
        
        # Print de status a cada 50 msgs para não inundar o terminal
        if engine._msg_received_count % 50 == 0 or current_len > trades_collected:
            trades_collected = current_len
            obi = engine.metrics[symbol].obi
            print(f"⏱️ {datetime.now().strftime('%H:%M:%S')} | Msgs: {engine._msg_received_count} | Evento: {event} | Trades: {current_len}/5 | OBI: {obi:+.4f}")

    # Inicia o stream real com um MIX de canais (incluindo o que sabemos que funciona)
    streams = [
        f"{symbol.lower()}@bookTicker", # Sabemos que funciona
        f"{symbol.lower()}@aggTrade",   # O que queremos
        f"{symbol.lower()}@ticker"      # Teste extra
    ]
    print(f"📡 Monitorando Mix de Streams: {streams}")
    ws_task = asyncio.create_task(connector.start_websocket_stream(streams, custom_handler))

    # Loop de monitoramento
    while trades_collected < 5:
        await asyncio.sleep(1)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Log de batimento cardíaco
        if engine._msg_received_count > 0 and engine._msg_received_count % 100 == 0:
            pulse = engine.get_market_pulse(symbol)
            print(f"💓 Vivo | Msgs: {engine._msg_received_count} | OBI: {pulse['obi']:+.4f} | Trades: {trades_collected}")
            
        if elapsed > 15:
            if trades_collected == 0:
                print("\n⚠️ ALERTA: Recebemos dados de BOOK, mas ZERO dados de TRADE.")
                print("Isso sugere que a Binance está enviando trades em um canal com nome diferente.")
            break
    
    if trades_collected >= 5:
        print("---------------------------------------------------------")
        print("🎉 [SUCESSO] O TapeEngine processou dados REAIS da Binance!")
        final_pulse = engine.get_market_pulse(symbol)
        print(f"📊 Resultado Final -> Pulso: {final_pulse['pulse']} | Score: {final_pulse['score']}")
    
    # Limpeza
    ws_task.cancel()
    await connector.close()
    print("🔌 Conexão encerrada.")

if __name__ == "__main__":
    try:
        asyncio.run(run_live_validation())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
