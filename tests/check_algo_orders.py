# Check for existing algo orders and cancel them
import os
import time
import hashlib
import hmac
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")

BASE_URL = "https://testnet.binancefuture.com"

def _sign(params, secret):
    query_string = urlencode(params)
    signature = hmac.new(
        secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def get_server_time():
    resp = requests.get(f"{BASE_URL}/fapi/v1/time")
    return resp.json()['serverTime']

print("[DEBUG] Checking open algo orders...")

server_time = get_server_time()
headers = {'X-MBX-APIKEY': TESTNET_API_KEY}

# Get open algo orders
params = {
    'symbol': 'BTCUSDT',
    'timestamp': server_time,
}
params['signature'] = _sign(params, TESTNET_API_SECRET)

resp = requests.get(f"{BASE_URL}/fapi/v1/openOrders", params=params, headers=headers)
print(f"Open Algo Orders: {resp.status_code}")
print(f"Response: {resp.text}")

# If there are orders, cancel them
if resp.status_code == 200 and resp.json():
    orders = resp.json()
    print(f"\nFound {len(orders)} open algo orders. Cancelling...")
    for order in orders:
        algo_id = order.get('algoId')
        print(f"  Cancelling algoId: {algo_id}")
        
        cancel_params = {
            'symbol': 'BTCUSDT',
            'algoId': algo_id,
            'timestamp': get_server_time(),
        }
        cancel_params['signature'] = _sign(cancel_params, TESTNET_API_SECRET)
        
        cancel_resp = requests.delete(f"{BASE_URL}/fapi/v1/algoOrder", params=cancel_params, headers=headers)
        print(f"  Cancel response: {cancel_resp.status_code} - {cancel_resp.text}")
