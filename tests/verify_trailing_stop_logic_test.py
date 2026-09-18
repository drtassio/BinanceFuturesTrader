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


class MockPosition:
    def __init__(self, qty: float, entry_price: float = 60000.0, mark_price: float = 60300.0):
        self.quantity = qty
        self.entry_price = entry_price
        self.mark_price = mark_price
        self.leverage = 5


async def run_test() -> int:
    logger.info("🚀 Starting Trailing Stop / Emergency Stop Verification")
    failures = 0

    # 1. Mock Dependencies
    mock_config = MagicMock(spec=TradingConfig)
    mock_config.DEFAULT_STOP_LOSS_PCT = 0.02

    mock_connector = MagicMock(spec=BinanceConnector)
    mock_connector.get_symbol_info = AsyncMock(return_value={'quantityPrecision': 3})
    mock_connector.place_trailing_stop_order = AsyncMock(return_value={'orderId': 12345, 'status': 'NEW'})
    mock_connector.place_stop_loss_order = AsyncMock(return_value={'algoId': 777})
    mock_connector.get_open_orders = AsyncMock(return_value=[])
    mock_connector.get_ticker_price = AsyncMock(return_value={'price': '60300'})
    mock_connector.has_open_position = AsyncMock(return_value=True)
    mock_connector.cancel_all_algo_orders = AsyncMock(return_value=0)

    mock_portfolio = MagicMock(spec=PortfolioOptimizer)
    mock_portfolio.positions = {}

    start_state = {'live_trading_enabled': True}  # Monitor só atua em live

    # 2. Initialize ExecutionEngine
    engine = ExecutionEngine(mock_config, mock_connector, mock_portfolio, start_state)
    engine._get_cached_symbol_info = AsyncMock(return_value={'quantityPrecision': 3, 'pricePrecision': 2})

    symbol = "BTCUSDT"
    position_qty = 1.23456789

    logger.info("🧪 Test 1: Quantity rounding in _place_trailing_stop")
    await engine._place_trailing_stop(symbol, OrderSide.BUY, position_qty, stop_loss_pct=0.02, activation_price=61200.0)
    if mock_connector.place_trailing_stop_order.call_count == 1:
        kwargs = mock_connector.place_trailing_stop_order.call_args.kwargs
        if kwargs['quantity'] == 1.235 and kwargs['side'] == 'SELL' and kwargs['activation_price'] == 61200.0:
            logger.info(f"✅ Quantidade arredondada ({kwargs['quantity']}), lado SELL e ativação repassados")
        else:
            failures += 1
            logger.error(f"❌ Parâmetros inesperados: {kwargs}")
    else:
        failures += 1
        logger.error("❌ _place_trailing_stop não chamou o conector!")

    logger.info("🧪 Test 2: Monitor garante Stop Loss fixo (não conta trailing como proteção)")
    mock_portfolio.positions = {symbol: MockPosition(1.235)}

    logger.info("   ├─ Case A: sem ordens de proteção -> coloca Stop Loss de emergência")
    mock_connector.place_stop_loss_order.reset_mock()
    mock_connector.get_open_orders.return_value = []
    await engine._manage_existing_positions()
    if mock_connector.place_stop_loss_order.call_count == 1:
        args = mock_connector.place_stop_loss_order.call_args.args
        logger.info(f"   ✅ Stop de emergência colocado: lado {args[1]} @ {args[3]:.2f}")
        if args[1] != 'SELL' or not args[3] < 60300.0:
            failures += 1
            logger.error("   ❌ Stop de emergência no lado/preço errado para um LONG")
    else:
        failures += 1
        logger.error(f"   ❌ Stop não colocado! Count: {mock_connector.place_stop_loss_order.call_count}")

    logger.info("   ├─ Case B: só trailing aberto -> ainda coloca Stop Loss fixo")
    mock_connector.place_stop_loss_order.reset_mock()
    mock_connector.get_open_orders.return_value = [{'type': 'TRAILING_STOP_MARKET', 'orderId': 1}]
    await engine._manage_existing_positions()
    if mock_connector.place_stop_loss_order.call_count == 1:
        logger.info("   ✅ Trailing não armado não conta como proteção: Stop Loss colocado")
    else:
        failures += 1
        logger.error(f"   ❌ Esperava Stop Loss com só trailing aberto. Count: {mock_connector.place_stop_loss_order.call_count}")

    logger.info("   ├─ Case C: STOP_MARKET aberto -> não duplica")
    mock_connector.place_stop_loss_order.reset_mock()
    mock_connector.get_open_orders.return_value = [{'type': 'STOP_MARKET', 'orderId': 2}]
    await engine._manage_existing_positions()
    if mock_connector.place_stop_loss_order.call_count == 0:
        logger.info("   ✅ Stop existente detectado, nenhuma ordem duplicada")
    else:
        failures += 1
        logger.error(f"   ❌ Stop duplicado! Count: {mock_connector.place_stop_loss_order.call_count}")

    logger.info(f"🏁 Verificação concluída: {'OK' if failures == 0 else f'{failures} falha(s)'}")
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(run_test()))
