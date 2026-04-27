import asyncio
import os
from dotenv import load_dotenv
from config.settings import Config
from trading.binance_connector import BinanceConnector
from utils.logger import get_logger

# Load env
load_dotenv()

# Setup config
config = Config()
# Ensure testnet
config.BINANCE_TESTNET = True

async def main():
    print("reproducing balance error...")
    connector = BinanceConnector(config)
    await connector.connect()
    
    print("Fetching account summary (balance/positions)...")
    summary = await connector.get_account_summary()
    
    print(f"Summary keys: {summary.keys()}")
    print(f"Total Value: {summary.get('total_value')}")
    
    await connector.close()

if __name__ == "__main__":
    asyncio.run(main())
