
import asyncio
import os
import sys
from binance.error import ClientError
from binance.um_futures import UMFutures

# Adiciona o diretório raiz ao path para importar configs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import TradingConfig

async def main():
    print("--- DEBUG RAW BALANCE ---")
    
    # Carrega chaves do ambiente ou config
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
    api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET")
    
    print(f"API Key: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")
    
    if not api_key or not api_secret:
        print("Erro: Chaves API não encontradas no ambiente.")
        return

    # Cliente Sincrono para debug rápido (usando lib oficial se disponível, ou requests)
    # O bot usa aiohttp, mas aqui vamos usar um requests simples ou a lib oficial se instalada.
    # Vamos usar requests direto para garantir que não há middleware atrapalhando.
    import requests
    import time
    import hmac
    import hashlib
    
    base_url = "https://testnet.binancefuture.com"
    endpoint = "/fapi/v2/balance"
    
    timestamp = int(time.time() * 1000)
    params = f"timestamp={timestamp}"
    
    signature = hmac.new(api_secret.encode('utf-8'), params.encode('utf-8'), hashlib.sha256).hexdigest()
    
    url = f"{base_url}{endpoint}?{params}&signature={signature}"
    headers = {"X-MBX-APIKEY": api_key}
    
    print(f"Requesting: {endpoint}")
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        print(f"Status: {response.status_code}")
        print("Raw Response Balance:")
        import json
        print(json.dumps(data, indent=2))
        
        # Check Positions
        endpoint_pos = "/fapi/v2/positionRisk"
        timestamp = int(time.time() * 1000)
        params = f"timestamp={timestamp}"
        signature = hmac.new(api_secret.encode('utf-8'), params.encode('utf-8'), hashlib.sha256).hexdigest()
        url_pos = f"{base_url}{endpoint_pos}?{params}&signature={signature}"
        
        print(f"\nRequesting: {endpoint_pos}")
        response_pos = requests.get(url_pos, headers=headers)
        data_pos = response_pos.json()
        
        print("Raw Response Positions:")
        # Filter non-zero for clarity
        non_zero = [p for p in data_pos if float(p.get('positionAmt', 0)) != 0]
        if non_zero:
            print(json.dumps(non_zero, indent=2))
        else:
            print("No non-zero positions found.")
            # Print first 2 for structure check
            if len(data_pos) > 0:
                print("First 2 positions (structure check):")
                print(json.dumps(data_pos[:2], indent=2))
            else:
                 print(json.dumps(data_pos, indent=2))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
