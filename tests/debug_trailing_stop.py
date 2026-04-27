# Sync test with exact same request that the connector is making
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

print("[DEBUG] Testing algoOrder with same params as connector...")
print(f"API Key: {TESTNET_API_KEY[:8]}...")

server_time = get_server_time()

# Exact params from connector (using new format without reduceOnly)
params = {
    'symbol': 'BTCUSDT',
    'side': 'BUY',
    'type': 'TRAILING_STOP_MARKET',
    'algoType': 'CONDITIONAL',
    'quantity': 0.005,
    'callbackRate': 4.0,  # Percentual
    'timestamp': server_time,
}

params['signature'] = _sign(params, TESTNET_API_SECRET)

headers = {
    'X-MBX-APIKEY': TESTNET_API_KEY
}

print(f"\nSending request to: {BASE_URL}/fapi/v1/algoOrder")
print(f"Params: {params}")

resp = requests.post(f"{BASE_URL}/fapi/v1/algoOrder", params=params, headers=headers)

print(f"\nStatus: {resp.status_code}")
print(f"Response: {resp.text}")
