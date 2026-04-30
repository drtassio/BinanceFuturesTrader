
import asyncio
import unittest
from unittest.mock import MagicMock
from datetime import datetime, timedelta
from trading.tape_engine import TapeEngine, _TapeTick, TapeMetrics

class TestTapePulse(unittest.TestCase):
    def setUp(self):
        # Mock do conector
        self.mock_connector = MagicMock()
        self.mock_connector.testnet = False
        self.mock_connector.ws_base_url = "wss://fstream.binance.com"
        
        # Inicializa o TapeEngine para BTCUSDT
        self.symbol = "BTCUSDT"
        self.engine = TapeEngine(self.mock_connector, [self.symbol])

    def test_warmup_transition(self):
        """Verifica se o bot sai do WARMUP após 5 trades."""
        # Estado inicial: WARMUP
        pulse = self.engine.get_market_pulse(self.symbol)
        self.assertEqual(pulse["pulse"], "WARMUP...")
        
        # Simula 5 trades via WebSocket (usando o formato de Combined Stream)
        for i in range(5):
            msg = {
                "stream": "btcusdt@aggTrade",
                "data": {
                    "e": "aggTrade",
                    "s": "BTCUSDT",
                    "q": "1.5",
                    "m": False,  # Buyer is aggressor (BUY)
                    "T": int(datetime.utcnow().timestamp() * 1000)
                }
            }
            # Chama o handler diretamente (simulando o callback do conector)
            asyncio.run(self.engine._handle_ws_message(msg))
            
        # Verifica se o buffer tem 5 itens
        self.assertEqual(len(self.engine.trades_buffer[self.symbol]), 5)
        
        # Executa o cálculo manual (já que o loop de análise é assíncrono)
        self.engine._calculate_metrics(self.symbol)
        
        # Agora deve ter um pulso real
        pulse = self.engine.get_market_pulse(self.symbol)
        self.assertNotEqual(pulse["pulse"], "WARMUP...")
        self.assertEqual(pulse["pulse"], "STRONG_BULLISH") # 5 BUYs seguidos = Bullish
        self.assertGreater(pulse["score"], 0)

    def test_obi_calculation(self):
        """Verifica se o OBI (Order Book Imbalance) é calculado corretamente."""
        # Simula uma mensagem de bookTicker
        msg = {
            "stream": "btcusdt@bookTicker",
            "data": {
                "u": 12345,
                "s": "BTCUSDT",
                "b": "60000.00",
                "B": "10.0",  # Bid Qty: 10
                "a": "60001.00",
                "A": "2.0"    # Ask Qty: 2
            }
        }
        asyncio.run(self.engine._handle_ws_message(msg))
        
        # OBI = (10 - 2) / (10 + 2) = 8 / 12 = 0.666...
        metrics = self.engine.metrics[self.symbol]
        self.assertAlmostEqual(metrics.obi, 0.6666666666666666, places=7)
        
        # Pulso deve refletir o OBI mesmo em WARMUP (se configurado para tal)
        pulse = self.engine.get_market_pulse(self.symbol)
        self.assertAlmostEqual(pulse["obi"], 0.6666666666666666, places=7)

    def test_stream_identification_robustness(self):
        """Verifica se o motor identifica o trade mesmo sem o campo 'e'."""
        msg = {
            "stream": "btcusdt@aggTrade",
            "data": {
                # Campo 'e' (event type) OMITIDO de propósito
                "s": "BTCUSDT",
                "q": "0.5",
                "m": True, # Sell
                "T": int(datetime.utcnow().timestamp() * 1000)
            }
        }
        asyncio.run(self.engine._handle_ws_message(msg))
        
        # Deve ter sido adicionado ao buffer pelo nome do stream
        self.assertEqual(len(self.engine.trades_buffer[self.symbol]), 1)
        self.assertFalse(self.engine.trades_buffer[self.symbol][0].is_buy)

if __name__ == "__main__":
    unittest.main()
