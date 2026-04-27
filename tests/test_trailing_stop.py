"""
Script de teste para colocar Trailing Stop manualmente via API Binance Testnet.
"""
import asyncio
import os
import sys

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.binance_connector import BinanceConnector
from config.settings import TradingConfig

async def main():
    print("🔧 [TESTE] Iniciando teste de Trailing Stop...")
    
    # Inicializa o conector
    config = TradingConfig()
    connector = BinanceConnector(config)
    await connector.connect()
    
    symbol = "BTCUSDT"
    
    # 1. Verifica posições abertas
    print("\n📋 [1] Buscando posições abertas...")
    account_summary = await connector.get_account_summary()
    positions = account_summary.get('positions', {})
    
    print(f"   Posições encontradas: {len(positions)}")
    for sym, pos in positions.items():
        qty = pos.get('quantity', 0)
        entry = pos.get('entry_price', 0)
        pnl = pos.get('unrealized_pnl', 0)
        print(f"   - {sym}: {qty:.6f} @ ${entry:.2f} (PnL: ${pnl:.2f})")
    
    # 2. Verifica ordens abertas
    print("\n📋 [2] Buscando ordens abertas...")
    open_orders = await connector.get_open_orders(symbol)
    print(f"   Ordens abertas para {symbol}: {len(open_orders)}")
    for order in open_orders:
        print(f"   - {order.get('type')} | {order.get('side')} | Qty: {order.get('origQty')} | Status: {order.get('status')}")
    
    # 3. Verifica se já tem trailing stop
    print("\n🔍 [3] Verificando trailing stop existente...")
    has_trailing = await connector.has_trailing_stop_order(symbol)
    print(f"   Trailing stop existe: {has_trailing}")
    
    # 4. Se não tem trailing stop e tem posição, coloca um
    if not has_trailing and symbol in positions:
        pos_data = positions[symbol]
        quantity = pos_data.get('quantity', 0)
        
        if quantity != 0:
            print(f"\n🎯 [4] Colocando Trailing Stop para posição {symbol}...")
            
            # quantity < 0 = SHORT, trailing stop = BUY
            # quantity > 0 = LONG, trailing stop = SELL
            trailing_side = "BUY" if quantity < 0 else "SELL"
            abs_quantity = abs(quantity)
            callback_rate = 4.0  # 4% callback rate
            
            print(f"   Side: {trailing_side}")
            print(f"   Quantity: {abs_quantity:.6f}")
            print(f"   Callback Rate: {callback_rate}%")
            
            result = await connector.place_trailing_stop_order(
                symbol=symbol,
                side=trailing_side,
                quantity=abs_quantity,
                callback_rate=callback_rate
            )
            
            if result:
                print(f"\n✅ SUCESSO! Trailing Stop colocado!")
                print(f"   Order ID: {result.get('orderId')}")
                print(f"   Status: {result.get('status')}")
            else:
                print(f"\n❌ FALHA ao colocar Trailing Stop!")
        else:
            print("\n⚠️ Posição com quantidade zero.")
    else:
        if has_trailing:
            print("\n✅ Trailing stop já existe!")
        else:
            print("\n⚠️ Nenhuma posição encontrada para colocar trailing stop.")
    
    await connector.close()
    print("\n🏁 Teste concluído!")

if __name__ == "__main__":
    asyncio.run(main())
