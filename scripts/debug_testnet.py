"""
Script de Debug para verificar conexão com Binance Testnet e a função get_account_summary completa
"""
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Config
from trading.binance_connector import BinanceConnector

async def debug_testnet():
    print("=" * 60)
    print("🔍 DEBUG: Testando get_account_summary() completo")
    print("=" * 60)
    
    config = Config()
    
    # Criar conector em modo Testnet
    connector = BinanceConnector(config, force_production=False)
    
    print(f"\n🌐 Conectando ao: {connector.base_url}")
    
    await connector.connect()
    
    # Testar get_account_summary completo
    print("\n" + "=" * 60)
    print("💰 Testando get_account_summary()...")
    summary = await connector.get_account_summary()
    
    print(f"\n📊 Resumo da Conta:")
    print(f"   - Total Value: ${summary.get('total_value', 0):,.2f}")
    print(f"   - Cash (Available): ${summary.get('cash', 0):,.2f}")
    print(f"   - Unrealized PnL: ${summary.get('unrealized_pnl', 0):,.2f}")
    print(f"   - Positions Count: {summary.get('positions_count', 0)}")
    
    positions = summary.get('positions', {})
    if positions:
        print(f"\n📊 Detalhes das Posições:")
        for symbol, pos in positions.items():
            print(f"\n   🔥 {symbol}:")
            print(f"      - Quantidade: {pos.get('quantity')}")
            print(f"      - Entry Price: ${pos.get('entry_price'):,.2f}")
            print(f"      - Mark Price: ${pos.get('mark_price'):,.2f}")
            print(f"      - PnL: ${pos.get('unrealized_pnl'):,.2f}")
            print(f"      - Leverage: {pos.get('leverage')}x")
            print(f"      - Margin Type: {pos.get('margin_type')}")
    else:
        print("\n   ℹ️ Nenhuma posição aberta")
    
    await connector.close()
    
    print("\n" + "=" * 60)
    print("✅ Debug completo!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(debug_testnet())
