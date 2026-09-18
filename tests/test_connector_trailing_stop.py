import asyncio
import os
from dotenv import load_dotenv
from config.settings import Config, TradingConfig
from trading.binance_connector import BinanceConnector
from utils.logger import get_logger

# Load env
load_dotenv()

# Setup config
config = Config()
# Ensure testnet
config.BINANCE_TESTNET = True

async def main():
    print("Testing BinanceConnector trailing stop logic...")
    connector = BinanceConnector(config)
    await connector.connect()
    
    symbol = "BTCUSDT"
    side = "BUY"
    quantity = 0.005
    callback_rate = 4.0 # Percentual (4.0 = 4%), faixa aceita pela Binance: 0.1 a 10

    print(f"Placing trailing stop with callbackRate={callback_rate}%...")
    result = await connector.place_trailing_stop_order(symbol, side, quantity, callback_rate)
    
    if result:
        print("SUCCESS! Order placed.")
        print(result)
    else:
        print("FAILED.")
        
    await connector.close()

if __name__ == "__main__":
    asyncio.run(main())
