import asyncio
import logging
import os
import sys
from datetime import datetime
from colorama import Fore, Style, init

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import TradingConfig
from trading.binance_connector import BinanceConnector
from trading.risk_manager import RiskManager
from models.trade_schema import OrderSide, OrderType

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

# Setup basic logger (suppress debug for clarity)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VerificationScript")
logging.getLogger("BinanceConnector").setLevel(logging.WARNING)

async def verify_readiness():
    print(f"{Fore.WHITE}{Style.BRIGHT}🚀 INICIANDO VERIFICAÇÃO AUTOMATIZADA DO BOT SHIELD{Style.RESET_ALL}")
    print("="*60)

    config = TradingConfig()
    connector = BinanceConnector(config)
    risk_manager = RiskManager(config)

    symbol = "BTCUSDT"
    qty = 0.005 # Small quantity for testing
    
    # Inicializa conexão
    log_info("Inicializando conexão com Binance...")
    await connector.connect()

    try:
        # ---------------------------------------------------------
        # 1. TESTE DE CONECTIVIDADE E SALDO
        # ---------------------------------------------------------
        log_step("1. Verificando Conectividade e Saldo")
        account_info = await connector.get_account_summary()
        
        if account_info and account_info['total_value'] > 0:
            log_success(f"Conectado! Saldo Total: ${account_info['total_value']:.2f}")
            log_success(f"Saldo Disponível: ${account_info['cash']:.2f}")
            log_success(f"Margem Usada: ${account_info.get('margin_used', 0):.2f}")
        else:
            log_fail("Falha ao obter saldo ou saldo zero.")
            return

        # ---------------------------------------------------------
        # 2. TESTE DE ALAVANCAGEM
        # ---------------------------------------------------------
        log_step("2. Testando Ajuste Dinâmico de Alavancagem")
        target_lev = 25
        success = await connector.set_leverage_for_symbol(symbol, target_lev)
        if success:
            log_success(f"Alavancagem de {symbol} definida para {target_lev}x")
        else:
            log_info(f"Ajuste de alavancagem falhou (esperado se houver posição aberta/margem insuficiente).")

        # ---------------------------------------------------------
        # 3. TESTE DE ORDEM LIMIT (MAKERS)
        # ---------------------------------------------------------
        log_step("3. Testando Ordem LIMIT (Maker)")
        
        # Obter precisão do símbolo
        symbol_info = await connector.get_symbol_info(symbol)
        price_precision = symbol_info['pricePrecision'] if symbol_info else 2
        
        current_price_data = await connector.get_ticker_price(symbol)
        if not current_price_data:
             log_fail("Não foi possível obter preço atual.")
             return
        
        price = float(current_price_data['price'])
        limit_price = price * 0.95 # 5% abaixo
        
        # Formatar preço corretamente
        formatted_price = f"{limit_price:.{price_precision}f}"
        
        log_info(f"Preço Atual: {price}, Put Limit Buy @ {formatted_price}")
        
        order_params = {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'LIMIT',
            'timeInForce': 'GTC',
            'quantity': qty,
            'price': formatted_price
        }
        
        limit_order = await connector.place_order(order_params)
        if limit_order and 'orderId' in limit_order:
            log_success(f"Ordem LIMIT criada: ID {limit_order['orderId']}")
            
            # Cancelar a ordem
            log_step("4. Cancelando Ordem LIMIT")
            cancel_res = await connector.cancel_order(symbol, order_id=limit_order['orderId'])
            if cancel_res:
                log_success(f"Ordem {limit_order['orderId']} cancelada.")
            else:
                log_fail("Falha ao cancelar ordem LIMIT.")
        else:
            log_fail(f"Falha ao criar ordem LIMIT. Params: {order_params}")

        # ---------------------------------------------------------
        # 4. TESTE DE ORDEM MARKET (TAKER) & POSIÇÃO
        # ---------------------------------------------------------
        log_step("5. Testando Ordem MARKET (Abrir Posição)")
        
        # Snapshot inicial da posição
        initial_summary = await connector.get_account_summary()
        initial_pos = initial_summary.get('positions', {}).get(symbol, {})
        initial_qty = float(initial_pos.get('quantity', 0.0))
        
        market_params = {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'MARKET',
            'quantity': qty
        }
        
        market_order = await connector.place_order(market_params)
        
        if market_order and 'orderId' in market_order:
            log_success(f"Ordem MARKET executada: ID {market_order['orderId']}")
            
            # Aguarda um pouco para atualização
            log_info("Aguardando 5 segundos para atualização de posição...")
            await asyncio.sleep(5)
            
            # Verificar Posição
            log_step("6. Verificando Posição Aberta")
            account_summary = await connector.get_account_summary()
            positions = account_summary.get('positions', {})
            btc_pos = positions.get(symbol)
            new_qty = float(btc_pos['quantity']) if btc_pos else 0.0
            
            if new_qty != initial_qty:
                log_success(f"Posição Alterada: {initial_qty} -> {new_qty} BTC")
                if btc_pos:
                    log_success(f"PnL Não Realizado: ${btc_pos['unrealized_pnl']:.4f}")
                    log_success(f"Margem Usada (Calc): ${btc_pos.get('leverage', 'N/A')}x")
            else:
                log_fail(f"Posição não alterada após ordem MARKET. Qty: {new_qty}")
                
            # ---------------------------------------------------------
            # 5. TESTE DE HARD STOP (SEM AUTOMACAO)
            # ---------------------------------------------------------
            log_step("7. Testando Hard Stop (Stop Loss Fixo)")
            # Stop Loss 2% abaixo do preço atual (aprox)
            market_price = float(initial_summary['total_value'] / initial_summary['cash']) if initial_summary['cash'] > 0 else 0 # Falho
            # Melhor pegar ticker de novo
            ticker = await connector.get_ticker_price(symbol)
            curr_price = float(ticker['price'])
            sl_price = curr_price * 0.98
            
            sl_res = await connector.place_stop_loss_order(
                symbol=symbol,
                side='SELL',
                quantity=qty,
                stop_price=sl_price
            )
            
            if sl_res and 'orderId' in sl_res:
                log_success(f"Hard Stop colocado: ID {sl_res['orderId']} @ {sl_price:.2f}")
                
                # Cancelar SL
                await connector.cancel_order(symbol, order_id=sl_res['orderId'])
                log_success("Hard Stop cancelado para limpeza.")
            else:
                log_fail("Falha ao colocar Hard Stop.")

            # ---------------------------------------------------------
            # 6. TESTE DE TRAILING STOP
            # ---------------------------------------------------------
            log_step("8. Testando Trailing Stop")
            # Trailing Stop para Venda (proteger Long)
            ts_res = await connector.place_trailing_stop_order(
                symbol=symbol,
                side='SELL',
                quantity=qty,
                callback_rate=1.0 # 1%
            )
            
            if ts_res and 'orderId' in ts_res:
                log_success(f"Trailing Stop colocado: ID {ts_res['orderId']}")
                
                # Cancelar TS
                await connector.cancel_order(symbol, order_id=ts_res['orderId'])
                log_success("Trailing Stop cancelado para limpeza.")
            else:
                log_fail("Falha ao colocar Trailing Stop.")

            # ---------------------------------------------------------
            # 6. FECHAR POSIÇÃO
            # ---------------------------------------------------------
            log_step("8. Fechando Posição (Market)")
            close_params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': qty
            }
            close_order = await connector.place_order(close_params)
            if close_order and 'orderId' in close_order:
                log_success(f"Posição Fechada: ID {close_order['orderId']}")
            else:
                log_fail("Falha ao fechar posição.")

        else:
            log_fail("Falha ao executar ordem MARKET.")

    except Exception as e:
        log_fail(f"Erro Crítico durante verificação: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if connector and connector.session:
            log_info("Fechando conexão...")
            await connector.close()

    print("\n" + "="*60)
    print(f"{Fore.WHITE}{Style.BRIGHT}🏁 VERIFICAÇÃO CONCLUÍDA{Style.RESET_ALL}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(verify_readiness())
