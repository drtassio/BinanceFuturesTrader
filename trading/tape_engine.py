# -----------------------------------------------------------------------------
# ARQUIVO: trading/tape_engine.py
# -----------------------------------------------------------------------------

"""
Motor de Análise de Tape (Tape Reading Engine).

Este módulo analisa a microestrutura do mercado em tempo real, processando o
fluxo de trades (tape) e o livro de ordens para extrair insights sobre a
pressão de compra/venda, liquidez e a presença de players institucionais.
"""

import asyncio
import threading
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Deque, Optional, Tuple, Any
import numpy as np

from utils.logger import get_logger , LOG_LEVEL_DEBUG
from .binance_connector import BinanceConnector
from models.trade_schema import Trade as TradeSchema, OrderSide

logger = get_logger("TapeEngine" , LOG_LEVEL_DEBUG)

class TapeMetrics:
    """Estrutura para armazenar as métricas calculadas pelo TapeEngine."""
    def __init__(self):
        self.aggressor_delta: float = 0.0
        self.volume_imbalance: float = 0.0
        self.vpin: float = 0.0  # Volume-Synchronized Probability of Informed Trading
        self.avg_trade_size_buy: float = 0.0
        self.avg_trade_size_sell: float = 0.0
        self.trade_rate_per_sec: float = 0.0
        self.large_trades_count: int = 0
        self.last_update_time: Optional[datetime] = None # Adicionado para rastrear a atualização
        
        # [NOVO] Order Book Imbalance (OBI)
        self.obi: float = 0.0  # (BidQty - AskQty) / (BidQty + AskQty)
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0

class TapeEngine:
    """
    Analisa o fluxo de ordens e o livro de ordens em tempo real.
    """

    def __init__(self, connector: BinanceConnector, symbols: List[str]):
        self.connector = connector
        self.symbols = symbols
        # Buffer de trades para cálculo de métricas (mantém os últimos 5000 trades por símbolo)
        self.trades_buffer: Dict[str, Deque[TradeSchema]] = {s: deque(maxlen=5000) for s in symbols}
        # Dicionário para armazenar as métricas calculadas por símbolo
        self.metrics: Dict[str, TapeMetrics] = {s: TapeMetrics() for s in symbols}
        self.is_running = False
        self._analysis_task: Optional[asyncio.Task] = None
        self._ws_task: Optional[asyncio.Task] = None

        # Parâmetros de análise
        self.short_window_size = 100  # trades para cálculo de imbalance de curtíssimo prazo
        self.vpin_bucket_volume_ratio = 0.02 # Cada bucket VPIN terá 2% do volume diário

    async def start(self):
        """Inicia os loops de coleta e análise de dados."""
        if self.is_running:
            logger.warning("TapeEngine já está em execução.")
            return

        logger.info(f"Iniciando TapeEngine para os símbolos: {self.symbols}")
        self.is_running = True
        
        # Inicia o loop de análise que roda em background
        # Este loop calcula as métricas a partir dos dados no trades_buffer
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        
        
        # Inicia a coleta de dados via WebSocket (stream de trades agregados + BOOK TICKER)
        # [MODIFICAÇÃO] Adicionado bookTicker para OBI
        streams = []
        for symbol in self.symbols:
            streams.append(f"{symbol.lower()}@aggTrade")
            streams.append(f"{symbol.lower()}@bookTicker")
            
        self._ws_task = asyncio.create_task(self.connector.start_websocket_stream(streams, self._handle_ws_message))
        
        logger.info("TapeEngine iniciado com sucesso.")

    async def stop(self):
        """Para os loops de coleta e análise."""
        if not self.is_running:
            return
            
        logger.info("Parando TapeEngine...")
        self.is_running = False
        
        if self._ws_task:
            self._ws_task.cancel()
        if self._analysis_task:
            self._analysis_task.cancel()
            
        try:
            # Aguarda a finalização das tasks para garantir que os recursos sejam liberados
            await asyncio.gather(self._ws_task, self._analysis_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass
            
        logger.info("TapeEngine parado.")

    async def _handle_ws_message(self, msg: Dict):
        """Callback para processar mensagens do WebSocket de trades."""
        try:
            if 'e' in msg and msg['e'] == 'aggTrade':
                symbol = msg['s']
                # Cria um objeto TradeSchema a partir dos dados do WebSocket
                trade = TradeSchema(
                    trade_id=str(msg['a']),
                    order_id="", 
                    symbol=symbol,
                    side=OrderSide.SELL if msg['m'] else OrderSide.BUY, # msg['m'] == True significa que o trade é feito por um Maker (ordem Limit)
                    quantity=float(msg['q']),
                    executed_price=float(msg['p']),
                    fee=0, fee_asset="", 
                    timestamp=datetime.fromtimestamp(msg['T'] / 1000.0)
                )
                # Adiciona o trade ao buffer do símbolo correspondente
                if symbol in self.trades_buffer:
                    self.trades_buffer[symbol].append(trade)
            
            # [NOVO] Handler para bookTicker (OBI em tempo real)
            elif 'u' in msg and 's' in msg: # 'u' é uniqueID do bookTicker
                symbol = msg['s']
                if symbol in self.metrics:
                    best_bid_qty = float(msg['B'])
                    best_ask_qty = float(msg['A'])
                    best_bid_price = float(msg['b'])
                    best_ask_price = float(msg['a'])
                    
                    # Cálculo OBI: (Bid - Ask) / (Bid + Ask) -> Range [-1, 1]
                    total_qty = best_bid_qty + best_ask_qty
                    obi = (best_bid_qty - best_ask_qty) / total_qty if total_qty > 0 else 0.0
                    
                    self.metrics[symbol].obi = obi
                    self.metrics[symbol].best_bid = best_bid_price
                    self.metrics[symbol].best_ask = best_ask_price
                    
        except Exception as e:
            logger.error(f"Erro ao processar mensagem do WebSocket no TapeEngine: {msg} | Erro: {e}")

    async def _analysis_loop(self):
        """Loop de fundo que calcula as métricas periodicamente."""
        while self.is_running:
            try:
                # Itera sobre todos os símbolos monitorados
                for symbol in self.symbols:
                    # Calcula as métricas apenas se houver trades suficientes no buffer
                    if len(self.trades_buffer[symbol]) > self.short_window_size:
                        self._calculate_metrics(symbol)
                await asyncio.sleep(1)  # Calcula as métricas a cada 1 segundo
            except asyncio.CancelledError:
                logger.info("Loop de análise do TapeEngine cancelado.")
                break
            except Exception as e:
                logger.exception(f"Erro no loop de análise do TapeEngine: {e}")
                await asyncio.sleep(5) # Espera mais em caso de erro

    def _calculate_metrics(self, symbol: str):
        """
        Calcula todas as métricas para um determinado símbolo usando a janela de trades recente.
        """
        metrics = self.metrics[symbol]
        trades = list(self.trades_buffer[symbol]) # Cria uma cópia para análise segura
        
        if not trades:
            return

        # Usa apenas a janela de trades mais recente para os cálculos
        recent_trades = trades[-self.short_window_size:]
        
        # 1. Delta do Agressor e Imbalance de Volume
        buy_volume = sum(t.quantity for t in recent_trades if t.side == OrderSide.BUY)
        sell_volume = sum(t.quantity for t in recent_trades if t.side == OrderSide.SELL)
        total_volume = buy_volume + sell_volume
        
        metrics.aggressor_delta = buy_volume - sell_volume
        metrics.volume_imbalance = (buy_volume - sell_volume) / (total_volume + 1e-9)

        # 2. Tamanho médio dos trades
        buy_trades = [t.quantity for t in recent_trades if t.side == OrderSide.BUY]
        sell_trades = [t.quantity for t in recent_trades if t.side == OrderSide.SELL]
        metrics.avg_trade_size_buy = np.mean(buy_trades) if buy_trades else 0
        metrics.avg_trade_size_sell = np.mean(sell_trades) if sell_trades else 0
        
        # 3. Taxa de Trades
        time_span_seconds = (recent_trades[-1].timestamp - recent_trades[0].timestamp).total_seconds()
        metrics.trade_rate_per_sec = len(recent_trades) / time_span_seconds if time_span_seconds > 0 else 0

        # 4. Contagem de Grandes Trades (ex: > 10 * média)
        avg_trade_size = total_volume / len(recent_trades) if recent_trades else 0
        large_trade_threshold = avg_trade_size * 10
        metrics.large_trades_count = sum(1 for t in recent_trades if t.quantity > large_trade_threshold)

        # 5. VPIN (cálculo mais intensivo, pode ser feito com menos frequência)
        # O VPIN requer um volume de dados maior para ser robusto
        if len(trades) > 1000: 
            metrics.vpin = self._calculate_vpin(trades)
        else:
            metrics.vpin = 0.5 # Valor padrão se não houver dados suficientes
        
        metrics.last_update_time = datetime.utcnow()
        logger.debug(f"📊 [TAPE METRICS] {symbol}: OBI={metrics.obi:.4f}, Imbalance={metrics.volume_imbalance:.4f}, VPIN={metrics.vpin:.4f}, Rate={metrics.trade_rate_per_sec:.2f} trades/s")

    def _calculate_vpin(self, trades: List[TradeSchema]) -> float:
        """Calcula o VPIN (Volume-Synchronized Probability of Informed Trading)."""
        df = pd.DataFrame.from_records([t.__dict__ for t in trades])
        # Assina o volume (+ para BUY, - para SELL)
        df['signed_vol'] = df.apply(lambda row: row['quantity'] if row['side'] == OrderSide.BUY else -row['quantity'], axis=1)
        
        # Cria buckets de volume
        total_volume = df['quantity'].sum()
        # Define o tamanho do bucket com base no volume total e na razão VPIN
        bucket_size = total_volume * self.vpin_bucket_volume_ratio
        
        if bucket_size == 0: return 0.5

        # Cria um índice para os buckets de volume
        df['cum_vol'] = df['quantity'].cumsum()
        df['bucket'] = (df['cum_vol'] // bucket_size).astype(int)
        
        # Calcula o imbalance por bucket
        bucket_imbalance = df.groupby('bucket')['signed_vol'].sum().abs()
        
        # VPIN é a média do imbalance normalizado
        vpin = (bucket_imbalance / bucket_size).mean()
        
        # Retorna o VPIN, garantindo que seja um float válido
        return float(vpin) if not np.isnan(vpin) else 0.5

    def get_metrics(self, symbol: str) -> Optional[TapeMetrics]:
        """Retorna as métricas mais recentes para um símbolo."""
        return self.metrics.get(symbol)

    def get_market_pulse(self, symbol: str) -> Dict[str, Any]:
        """
        Retorna um pulso de mercado de alto nível (sentimento de curto prazo).
        """
        metrics = self.get_metrics(symbol)
        if not metrics:
            return {"pulse": "NEUTRAL", "confidence": 0.5}

        # Lógica para determinar o pulso
        score = 0
        # Imbalance de volume tem peso alto
        score += metrics.volume_imbalance * 2 
        # VPIN alto indica toxicidade, geralmente precede quedas (VPIN > 0.5 sugere toxicidade)
        # Penaliza se o VPIN for alto e positivamente correlacionado com o volume imbalance
        score -= (metrics.vpin - 0.5) * 2

        if score > 0.7:
            pulse = "BULLISH"
        elif score < -0.7:
            pulse = "BEARISH"
        else:
            pulse = "NEUTRAL"
        
        confidence = min(1.0, abs(score))
        return {"pulse": pulse, "confidence": confidence, "score": score}