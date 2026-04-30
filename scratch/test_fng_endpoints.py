import asyncio
import aiohttp

async def test_binance_fng():
    # Test official Futures API (unlikely)
    async with aiohttp.ClientSession() as session:
        url_futures = "https://fapi.binance.com/fapi/v1/fearGreedIndex"
        try:
            async with session.get(url_futures) as resp:
                print(f"Futures API Status: {resp.status}")
                if resp.status == 200:
                    print(await resp.json())
        except Exception as e:
            print(f"Futures API Error: {e}")

        # Test Binance BAPI (Internal)
        url_bapi = "https://www.binance.com/bapi/growth/v1/friendly/growth-paas/fear-greed-index/get-index"
        try:
            async with session.get(url_bapi) as resp:
                print(f"BAPI Status: {resp.status}")
                if resp.status == 200:
                    print(await resp.json())
        except Exception as e:
            print(f"BAPI Error: {e}")

        # Test Alternative.me
        url_alt = "https://api.alternative.me/fng/"
        try:
            async with session.get(url_alt) as resp:
                print(f"Alternative.me Status: {resp.status}")
                if resp.status == 200:
                    print(await resp.json())
        except Exception as e:
            print(f"Alternative.me Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_binance_fng())
