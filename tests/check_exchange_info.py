# Script to check exchange info for TRAILING_STOP_MARKET support
import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://testnet.binancefuture.com"

print("[DEBUG] Checking exchange info for BTCUSDT...")

resp = requests.get(f"{BASE_URL}/fapi/v1/exchangeInfo")
data = resp.json()

# Find BTCUSDT symbol info
for symbol in data.get('symbols', []):
    if symbol.get('symbol') == 'BTCUSDT':
        print(f"\nSymbol: {symbol['symbol']}")
        print(f"Order Types: {symbol.get('orderTypes', [])}")
        print(f"Time in Force: {symbol.get('timeInForce', [])}")
        
        # Check filters
        for f in symbol.get('filters', []):
            if 'callback' in f.get('filterType', '').lower():
                print(f"Callback Filter: {f}")
            if f.get('filterType') == 'TRAILING_DELTA':
                print(f"Trailing Delta Filter: {f}")
        
        break
else:
    print("BTCUSDT not found!")
