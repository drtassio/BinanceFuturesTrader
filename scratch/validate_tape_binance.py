
import asyncio
import unittest
from datetime import datetime
from trading.tape_engine import TapeEngine

class TestBinanceDataIngestion(unittest.TestCase):
    def setUp(self):
        # Mock minimalista
        class MockConnector:
            def __init__(self): self.testnet = False
            async def start_websocket_stream(self, s, cb): pass
        
        self.engine = TapeEngine(MockConnector(), ["BTCUSDT"])

    def test_combined_stream_ingestion(self):
        """Valida se o bot processa corretamente o formato exato da Binance Futures."""
        
        # 1. Simular aggTrade (Formato real de Combined Stream da Binance)
        trade_msg = {
            "stream": "btcusdt@aggTrade",
            "data": {
                "e": "aggTrade",
                "E": 1672531200000,
                "s": "BTCUSDT",
                "a": 1234567,
                "p": "76100.00",
                "q": "0.500",      # Quantidade
                "f": 100,
                "l": 105,
                "T": 1672531200000, # Timestamp
                "m": False          # Buyer is aggressor (BUY)
            }
        }
        
        # Simula o processamento de 5 trades para sair do warmup
        for i in range(5):
            asyncio.run(self.engine._handle_ws_message(trade_msg))
        
        self.assertEqual(len(self.engine.trades_buffer["BTCUSDT"]), 5)
        print("✅ [OK] Ingestão de aggTrade (trades) validada.")

    def test_book_ticker_ingestion(self):
        """Valida se o OBI é calculado a partir do bookTicker real."""
        book_msg = {
            "stream": "btcusdt@bookTicker",
            "data": {
                "u": 400900217,
                "s": "BTCUSDT",
                "b": "76000.00",
                "B": "100.0", # Bid Qty
                "a": "76001.00",
                "A": "50.0"   # Ask Qty
            }
        }
        
        asyncio.run(self.engine._handle_ws_message(book_msg))
        
        # OBI = (100 - 50) / (100 + 50) = 50 / 150 = 0.333...
        obi = self.engine.metrics["BTCUSDT"].obi
        self.assertAlmostEqual(obi, 0.3333, places=4)
        print(f"✅ [OK] Ingestão de bookTicker (OBI) validada: {obi:.4f}")

    def test_warmup_to_pulse(self):
        """Valida o ciclo completo: Trade -> Cálculo -> Pulse."""
        # Adiciona trades
        for i in range(10):
            msg = {"stream": "btcusdt@aggTrade", "data": {"s": "BTCUSDT", "q": "1", "m": False, "T": 1672531200000}}
            asyncio.run(self.engine._handle_ws_message(msg))
        
        # Força cálculo
        self.engine._calculate_metrics("BTCUSDT")
        
        # Verifica pulso
        pulse = self.engine.get_market_pulse("BTCUSDT")
        self.assertNotEqual(pulse["pulse"], "WARMUP...")
        print(f"✅ [OK] Transição WARMUP -> {pulse['pulse']} validada.")

if __name__ == "__main__":
    unittest.main()
