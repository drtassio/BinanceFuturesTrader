import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading.execution_engine import ExecutionEngine, OrderSide
from trading.binance_connector import BinanceConnector
from config.settings import TradingConfig
from trading.portfolio import PortfolioOptimizer

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrailingStopTest")

async def run_test():
    logger.info("🚀 Starting Trailing Stop Logic Verification")

    # 1. Mock Dependencies
    mock_config = MagicMock(spec=TradingConfig)
    mock_config.DEFAULT_STOP_LOSS_PCT = 0.02
    
    mock_connector = MagicMock(spec=BinanceConnector)
    mock_connector.get_symbol_info = AsyncMock(return_value={'quantityPrecision': 3})
    mock_connector.place_trailing_stop_order = AsyncMock(return_value={'orderId': 12345, 'status': 'NEW'})
    mock_connector.has_trailing_stop_order = AsyncMock(return_value=False) # Default: No order
    
    # Mock Portfolio Positions
    mock_portfolio = MagicMock(spec=PortfolioOptimizer)
    
    # Define a Mock Position Class
    class MockPosition:
        def __init__(self, qty):
            self.quantity = qty
            
    # Start with empty
    mock_portfolio.positions = {}
    
    start_state = {'live_trading_enabled': True} # Start Enabled for Monitor to work

    # 2. Initialize ExecutionEngine
    engine = ExecutionEngine(mock_config, mock_connector, mock_portfolio, start_state)
    
    # 3. Define Test Data
    symbol = "BTCUSDT"
    position_qty = 1.23456789 
    
    logger.info("🧪 Test 1: Verify Quantity Rounding in _place_trailing_stop")
    await engine._place_trailing_stop(symbol, OrderSide.BUY, position_qty, stop_loss_pct=0.02)
    
    # Verify call
    if mock_connector.place_trailing_stop_order.call_count == 1:
        args, kwargs = mock_connector.place_trailing_stop_order.call_args
        actual_qty = kwargs['quantity']
        expected_qty = 1.235
        if actual_qty == expected_qty:
            logger.info(f"✅ Quantity Rounded Correctly: {actual_qty}")
        else:
            logger.error(f"❌ Quantity Rounding Failed! Expected {expected_qty}, got {actual_qty}")
    else:
        logger.error("❌ _place_trailing_stop did NOT call connector!")


    logger.info("🧪 Test 2: Verify Monitoring Logic (Idempotency)")
    
    # Inject Mock Position
    mock_portfolio.positions = {symbol: MockPosition(1.235)}
    
    # Case A: No Trailing Stop -> Should Place Order
    logger.info("   ├─ Case A: No Existing Trailing Stop -> Expect Placement")
    mock_connector.place_trailing_stop_order.reset_mock()
    mock_connector.has_trailing_stop_order.return_value = False
    
    await engine._manage_existing_positions()
    
    if mock_connector.place_trailing_stop_order.call_count == 1:
        logger.info("   ✅ Correctly blocked monitoring loop from placing duplicate? No, it PLACED order correctly.")
    else:
        logger.error(f"   ❌ Failed to place order! Count: {mock_connector.place_trailing_stop_order.call_count}")

    # Case B: Trailing Stop Exists -> Should NOT Place Order
    logger.info("   ├─ Case B: Existing Trailing Stop -> Expect NO Placement")
    mock_connector.place_trailing_stop_order.reset_mock()
    mock_connector.has_trailing_stop_order.return_value = True # Simulate that connector found an order
    
    await engine._manage_existing_positions()
    
    if mock_connector.place_trailing_stop_order.call_count == 0:
        logger.info("   ✅ Correctly skipped placement because order exists.")
    else:
        logger.error(f"   ❌ FAIL: Placed duplicate order! Count: {mock_connector.place_trailing_stop_order.call_count}")

    logger.info("🏁 Logic Verification Complete")

if __name__ == "__main__":
    asyncio.run(run_test())
