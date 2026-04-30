
import asyncio
import aiohttp
import json
import time
import sys

async def raw_binance_test():
    url = "wss://fstream.binance.com/ws/btcusdt@aggTrade"
    print(f"[LIVE TEST] Tentando conexao BRUTA em: {url}")
    sys.stdout.flush()
    
    async with aiohttp.ClientSession() as session:
        try:
            print("[INFO] Sessao criada, conectando...")
            sys.stdout.flush()
            async with session.ws_connect(url, timeout=10) as ws:
                print("[OK] Conectado ao servidor de trades da Binance!")
                sys.stdout.flush()
                
                count = 0
                start_time = time.time()
                
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        count += 1
                        print(f"[{count}/5] Trade Recebido! Preco: {data.get('p')} | Qtd: {data.get('q')}")
                        sys.stdout.flush()
                        
                        if count >= 5:
                            print("\n[SUCCESS] A Binance esta enviando trades via /ws/.")
                            break
                    
                    if time.time() - start_time > 20:
                        print("\n[TIMEOUT] A conexao abriu, mas nenhum trade foi enviado em 20s.")
                        break
                        
        except Exception as e:
            print(f"[ERROR] Erro de Conexao: {e}")
            sys.stdout.flush()

if __name__ == "__main__":
    print("[MAIN] Iniciando script...")
    sys.stdout.flush()
    try:
        asyncio.run(raw_binance_test())
    except Exception as e:
        print(f"[CRITICAL] Falha ao rodar loop: {e}")
