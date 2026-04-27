"""
Comprehensive debug: Test ALL balance endpoints on testnet.
Check for non-USDT assets and position details.
"""
import asyncio
import os
from dotenv import load_dotenv
import aiohttp
import time
import hashlib
import hmac
import json

load_dotenv(override=True)

API_KEY = os.getenv("BINANCE_TESTNET_API_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET")
BASE_URL = "https://testnet.binancefuture.com"

print(f"[KEY] Using: {API_KEY[:8]}...{API_KEY[-8:]}")
print(f"[KEY] Length: {len(API_KEY)}")
has_quotes = 'yes' if API_KEY.startswith('"') else 'no'
print(f"[KEY] Has quotes: {has_quotes}")

async def make_signed_request(session, method, endpoint, extra_params=None):
    timestamp = int(time.time() * 1000)
    params = f"timestamp={timestamp}"
    if extra_params:
        params = f"{extra_params}&{params}"
    
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        params.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    url = f"{BASE_URL}{endpoint}?{params}&signature={signature}"
    headers = {"X-MBX-APIKEY": API_KEY}
    
    if method == 'GET':
        async with session.get(url, headers=headers) as resp:
            return resp.status, await resp.json()
    else:
        async with session.post(url, headers=headers) as resp:
            return resp.status, await resp.json()

async def main():
    async with aiohttp.ClientSession() as session:
        # Test 1: /fapi/v2/balance - Individual asset balances
        print("\n=== TEST 1: /fapi/v2/balance ===")
        status, data = await make_signed_request(session, 'GET', '/fapi/v2/balance')
        if status == 200:
            print(f"[OK] Got {len(data)} assets")
            for asset in data:
                bal = float(asset.get('balance', 0))
                cross_pnl = float(asset.get('crossUnPnl', 0))
                avail = float(asset.get('availableBalance', 0))
                if bal != 0 or cross_pnl != 0 or avail != 0:
                    print(f"   [{asset['asset']}] Balance={bal:.4f}, CrossPnL={cross_pnl:.4f}, Available={avail:.4f}")
            non_zero = [a for a in data if float(a.get('balance', 0)) != 0]
            if not non_zero:
                print("   [WARN] ALL balances are zero!")
        else:
            print(f"[ERR] {status}: {data}")

        # Test 2: /fapi/v2/account - Full account with positions
        print("\n=== TEST 2: /fapi/v2/account (positions) ===")
        status, data = await make_signed_request(session, 'GET', '/fapi/v2/account')
        if status == 200:
            print(f"   totalWalletBalance: {data.get('totalWalletBalance')}")
            print(f"   totalUnrealizedProfit: {data.get('totalUnrealizedProfit')}")
            print(f"   totalMarginBalance: {data.get('totalMarginBalance')}")
            print(f"   availableBalance: {data.get('availableBalance')}")
            print(f"   totalCrossWalletBalance: {data.get('totalCrossWalletBalance')}")
            
            # Check ALL positions (even with 0 amt but non-zero leverage changes)
            positions = data.get('positions', [])
            btc_pos = [p for p in positions if 'BTC' in p.get('symbol', '')]
            print(f"\n   BTC-related positions ({len(btc_pos)}):")
            for p in btc_pos:
                print(f"      {p['symbol']}: amt={p['positionAmt']}, leverage={p['leverage']}x, entry={p['entryPrice']}, margin={p.get('initialMargin', 'N/A')}")
        else:
            print(f"[ERR] {status}: {data}")

        # Test 3: /fapi/v1/positionRisk - Current positions
        print("\n=== TEST 3: /fapi/v1/positionRisk (BTCUSDT) ===")
        status, data = await make_signed_request(session, 'GET', '/fapi/v1/positionRisk', 'symbol=BTCUSDT')
        if status == 200:
            for p in data:
                print(f"   {p['symbol']}: amt={p['positionAmt']}, leverage={p['leverage']}x, entry={p['entryPrice']}, uPnL={p.get('unRealizedProfit', 'N/A')}, marginType={p.get('marginType', 'N/A')}")
        else:
            print(f"[ERR] {status}: {data}")

        # Test 4: /fapi/v1/openOrders
        print("\n=== TEST 4: /fapi/v1/openOrders ===")
        status, data = await make_signed_request(session, 'GET', '/fapi/v1/openOrders')
        if status == 200:
            print(f"   Open orders: {len(data)}")
            for o in data:
                print(f"   - {o['symbol']} {o['type']} {o['side']} qty={o['origQty']}")
        else:
            print(f"[ERR] {status}: {data}")

        # Test 5: /fapi/v1/allOrders - Recent order history
        print("\n=== TEST 5: /fapi/v1/allOrders (BTCUSDT, last 10) ===")
        status, data = await make_signed_request(session, 'GET', '/fapi/v1/allOrders', 'symbol=BTCUSDT&limit=10')
        if status == 200:
            print(f"   Total orders returned: {len(data)}")
            for o in data[-5:]:  # Show last 5
                print(f"   - ID:{o['orderId']} {o['symbol']} {o['type']} {o['side']} qty={o['origQty']} status={o['status']}")
        else:
            print(f"[ERR] {status}: {data}")

        print("\n" + "=" * 60)
        print("[DONE] Complete diagnostics finished.")

if __name__ == "__main__":
    asyncio.run(main())
