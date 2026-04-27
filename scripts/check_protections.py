import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from colorama import Fore, Style, init

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import TradingConfig
from trading.binance_connector import BinanceConnector
from trading.execution_engine import ExecutionEngine, ExecutionOrder
from trading.risk_manager import RiskManager
from models.trade_schema import Signal, Action, OrderType, OrderSide

from trading.portfolio import PortfolioOptimizer

# Initialize Colorama
init(autoreset=True)

# Helper for logging
def log_step(step_name):
    print(f"\n{Fore.CYAN}🔵 [TEST] {step_name}...{Style.RESET_ALL}")

def log_success(msg):
    print(f"{Fore.GREEN}✅ {msg}{Style.RESET_ALL}")

def log_fail(msg):
    print(f"{Fore.RED}❌ {msg}{Style.RESET_ALL}")

def log_info(msg):
    print(f"{Fore.YELLOW}ℹ️ {msg}{Style.RESET_ALL}")

# Setup basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CheckProtections")
logging.getLogger("BinanceConnector").setLevel(logging.WARNING)

async def check_protections():
    print(f"{Fore.WHITE}{Style.BRIGHT}🛡️ VALIDAÇÃO DE PROTEÇÕES DO BOT SHIELD{Style.RESET_ALL}")
    print("="*60)

    config = TradingConfig()
    connector = BinanceConnector(config)
    risk_manager = RiskManager(config)
    portfolio = PortfolioOptimizer(config)
    
    # Mock AI Controller (not needed for this test)
    # Mock system state
    system_state = {"recent_trades": [], "command_feedback": ""}
    
    execution_engine = ExecutionEngine(
        config=config, 
        connector=connector, 
        portfolio=portfolio, 
        system_state=system_state
    )

    symbol = "BTCUSDT"
    qty = 0.005 # Small quantity
    
    # 1. Connect
    log_step("1. Conectando à Binance Testnet")
    await connector.connect()
    
    try:
        # Check Balance
        summary = await connector.get_account_summary()
        if not summary or summary['total_value'] <= 0:
            log_fail("Saldo insuficiente ou erro de conexão.")
            return

        log_success(f"Conectado. Saldo: ${summary['total_value']:.2f}")

        # ---------------------------------------------------------
        # 2. PLACE MARKET ORDER VIA EXECUTION ENGINE
        # ---------------------------------------------------------
        log_step("2. Enviando Ordem de Compra (Market) via ExecutionEngine")
        
        # Create a Signal
        signal = Signal(
            symbol=symbol,
            action=Action.BUY,
            confidence=0.95,
            position_size_pct=0.1,
            leverage=10.0,
            stop_loss=0.02, # 2% SL
            take_profit=0.04, # 4% TP
            profit_probability=0.85
        )
        
        # Create execution order
        import time
        order = ExecutionOrder(
            signal=signal,
            strategy="MARKET",
            order_id=f"test_{int(time.time())}"
        )
        order.total_quantity = qty
        
        # Execute
        await execution_engine._execute_market_real(order)
        
        if order.status.value == "FILLED":
            log_success(f"Ordem de Entrada Preenchida! ID: {order.child_order_ids[-1]}")
        else:
            log_fail(f"Ordem de Entrada Falhou. Status: {order.status.value}")
            return

        # ---------------------------------------------------------
        # 3. VERIFY PROTECTIONS (SL & TP)
        # ---------------------------------------------------------
        log_step("3. Verificando Ordens de Proteção (SL & TP)")
        log_info("Aguardando 5 segundos para propagação...")
        await asyncio.sleep(5)
        
        open_orders = await connector.get_open_orders(symbol)
        
        sl_found = False
        tp_found = False
        
        for o in open_orders:
            o_type = o.get('type')
            o_side = o.get('side')
            o_price = float(o.get('stopPrice', 0))
            o_reduce = o.get('reduceOnly')
            
            if o_type == 'STOP_MARKET' and o_side == 'SELL':
                log_success(f"STOP LOSS Encontrado! ID: {o['orderId']} @ {o_price} (Type: {o_type})")
                sl_found = True
            elif o_type == 'TAKE_PROFIT_MARKET' and o_side == 'SELL':
                log_success(f"TAKE PROFIT Encontrado! ID: {o['orderId']} @ {o_price} (Type: {o_type})")
                tp_found = True
            else:
                log_info(f"Outra ordem encontrada: {o_type} {o_side} @ {o_price}")

        if sl_found and tp_found:
            log_success("✅ AMBAS AS PROTEÇÕES (SL & TP) ESTÃO ATIVAS!")
        elif sl_found:
            log_fail("⚠️ Apenas STOP LOSS foi encontrado. TAKE PROFIT faltando.")
        elif tp_found:
            log_fail("⚠️ Apenas TAKE PROFIT foi encontrado. STOP LOSS faltando.")
        else:
            log_fail("❌ NENHUMA PROTEÇÃO ENCONTRADA!")

        # ---------------------------------------------------------
        # 4. VERIFY LIQUIDATION MONITOR
        # ---------------------------------------------------------
        log_step("4. Testando Monitoramento de Liquidação (Simulação)")
        # Update portfolio with real positions
        summary = await connector.get_account_summary()
        positions_data = summary.get('positions', {})
        
        # Mocking positions in RiskManager
        risk_manager.positions = {}
        from models.trade_schema import Position
        for s, p in positions_data.items():
            pos_obj = Position(
                symbol=s,
                quantity=float(p.get('quantity', 0)),
                entry_price=float(p.get('entry_price', 0)),
                timestamp=datetime.utcnow(),
                leverage=int(p.get('leverage', 1)),
                liquidation_price=float(p.get('liquidationPrice', 0) or 0),
                mark_price=float(p.get('markPrice', 0) or 0)
            )
            risk_manager.positions[s] = pos_obj
            
        # Check normal state
        log_info("Verificando estado normal...")
        alerts = risk_manager.check_liquidation_risk()
        if not alerts:
            log_success("Nenhum alerta em condições normais (Esperado).")
        else:
            log_fail(f"Alerta inesperado: {alerts}")
            
        # Simulate DANGER state
        if symbol in risk_manager.positions:
            pos = risk_manager.positions[symbol]
            original_mark = pos.mark_price
            original_liq = pos.liquidation_price
            
            # Set mark price very close to liquidation price (1% diff)
            # Long: Liq < Mark. Danger: Mark -> Liq.
            # Short: Liq > Mark. Danger: Mark -> Liq.
            
            if original_liq > 0:
                if pos.quantity > 0: # Long
                     pos.mark_price = original_liq * 1.01 # 1% above liq
                else: # Short
                     pos.mark_price = original_liq * 0.99 # 1% below liq
                     
                log_info(f"Simulando Mark Price {pos.mark_price:.2f} vs Liq {original_liq:.2f}...")
                
                alerts = risk_manager.check_liquidation_risk()
                if alerts:
                    log_success(f"✅ Alerta de Liquidação Disparado: {alerts[0]}")
                else:
                    log_fail("❌ Falha ao disparar alerta de liquidação na simulação!")
            else:
                log_info("Skipping Liq simulation (Liq Price is 0/None).")
        else:
            log_info(f"Posição {symbol} não encontrada para simulação de liquidação. (Talvez fechada?)")

        # ---------------------------------------------------------
        # 5. CLEANUP
        # ---------------------------------------------------------
        log_step("5. Limpeza (Fechando Posições e Ordens)")
        await connector.cancel_all_orders(symbol)
        
        # Close position
        await connector.place_order({
            'symbol': symbol,
            'side': 'SELL',
            'type': 'MARKET',
            'quantity': qty
        })
        log_success("Posição fechada e ordens canceladas.")

    except Exception as e:
        log_fail(f"Erro Crítico: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await connector.close()
        print("\n" + "="*60)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(check_protections())
