# -----------------------------------------------------------------------------
# ARQUIVO: trading/execution_engine.py
# -----------------------------------------------------------------------------

"""
Motor de Execução de Ordens (Execution Engine).

Este módulo recebe um `Signal` do AIController e utiliza estratégias avançadas como
TWAP e SMART para minimizar o impacto no mercado (slippage) e otimizar o preço
de execução para ordens maiores, considerando alavancagem e taxas reais.
"""

import asyncio
import time
from datetime import datetime, timezone 
from typing import Dict, List, Optional, Any, Callable 
import math 
import uuid # Para gerar IDs únicos de trades

from utils.logger import get_logger
from config.settings import TradingConfig 
from models.trade_schema import Signal, Trade, OrderStatus, OrderSide, OrderType, Order, Action 
from .binance_connector import BinanceConnector 
from .portfolio import PortfolioOptimizer 

logger = get_logger("ExecutionEngine")

class ExecutionOrder:
    """
    Representa uma meta-ordem gerenciada pelo ExecutionEngine.
    Esta é a abstração interna da ordem que encapsula o sinal e o estado de execução.
    """
    def __init__(self, signal: Signal, strategy: str, order_id: str):
        if not isinstance(signal, Signal):
            raise TypeError("🚨 [ERRO EXEC] 'signal' deve ser uma instância de Signal.")
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("🚨 [ERRO EXEC] 'strategy' deve ser uma string não vazia.")
        if not isinstance(order_id, str) or not order_id:
            raise ValueError("🚨 [ERRO EXEC] 'order_id' deve ser uma string não vazia.")

        self.id: str = order_id # ID único desta meta-ordem interna
        self.signal: Signal = signal # O sinal de trading que originou esta ordem
        self.strategy: str = strategy # Estratégia de execução utilizada (MARKET, LIMIT, TWAP)
        self.status: OrderStatus = OrderStatus.PENDING # Status atual da meta-ordem
        self.created_at: datetime = datetime.utcnow().replace(tzinfo=timezone.utc) # Timestamp de criação
        
        self.total_quantity: float = 0.0 # Quantidade total que a ordem pretende negociar (em base asset)
        self.filled_quantity: float = 0.0 # Quantidade já preenchida
        self.avg_fill_price: float = 0.0 # Preço médio de preenchimento
        self.notional_value_at_creation: float = 0.0 # Valor nocional da ordem no momento da criação

        # Lista de IDs das ordens "filhas" enviadas para a exchange (e.g., fatias TWAP)
        self.child_order_ids: List[str] = [] 
        # Lista de objetos Trade que representam os preenchimentos desta ordem
        self.trades: List[Trade] = [] 
        
        logger.info(f"🆕 [EXEC] Nova meta-ordem criada: {self.id} (Símbolo: {self.signal.symbol}, Ação: {self.signal.action.value}, Estratégia: {self.strategy}, Alavancagem: {self.signal.leverage:.2f}x).")
        
    @property
    def is_complete(self) -> bool:
        """Verifica se a meta-ordem foi completamente preenchida, cancelada, rejeitada ou expirou."""
        # Considera que a ordem está completa se a quantidade preenchida é próxima da total
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED] or \
               math.isclose(self.filled_quantity, self.total_quantity, rel_tol=1e-5) # Tolerância para floats

    def update_with_fill(self, fill_data: Dict[str, Any]) -> Optional[Trade]:
        """
        Atualiza o estado da meta-ordem com dados de preenchimento e retorna
        o objeto Trade recém-criado para logging externo.
        """
        if not isinstance(fill_data, dict):
            logger.error(f"❌ [ERRO EXEC] 'fill_data' inválido para update_with_fill. Deve ser um dicionário. Recebido: {type(fill_data)}.")
            return None

        try:
            current_order_executed_qty = float(fill_data.get('executedQty', 0.0))
            current_order_cum_quote_qty = float(fill_data.get('cumQuote', 0.0))
            order_status_str = fill_data.get('status', 'NEW')
            
            newly_filled_qty = current_order_executed_qty - self.filled_quantity
            
            if newly_filled_qty <= 1e-9 and OrderStatus(order_status_str) == self.status:
                return None # Nenhuma atualização necessária

            current_trade_price = 0.0
            if current_order_executed_qty > 0:
                current_trade_price = current_order_cum_quote_qty / current_order_executed_qty

            self.filled_quantity = current_order_executed_qty
            self.avg_fill_price = current_trade_price

            logger.info(f"✅ [EXEC] Preenchimento para meta-ordem {self.id}: {newly_filled_qty:.4f} {self.signal.symbol} @ {current_trade_price:.2f}.")

            self.status = OrderStatus(order_status_str)
            if math.isclose(self.filled_quantity, self.total_quantity, rel_tol=1e-5): 
                self.status = OrderStatus.FILLED
            elif self.filled_quantity > 0:
                self.status = OrderStatus.PARTIALLY_FILLED
            
            # Só cria um objeto Trade se houve preenchimento novo positivo
            if newly_filled_qty > 1e-9:
                trade_timestamp_ms = fill_data.get('updateTime', int(time.time()*1000))
                trade_timestamp = datetime.fromtimestamp(trade_timestamp_ms / 1000.0, tz=timezone.utc)
                
                new_trade = Trade(
                    trade_id=str(fill_data.get('tradeId', f"fill_{self.id}_{int(trade_timestamp.timestamp()*1000)}")), 
                    order_id=fill_data.get('clientOrderId', self.id), 
                    symbol=self.signal.symbol,
                    side=self.signal.action, 
                    quantity=newly_filled_qty,
                    executed_price=current_trade_price,
                    fee=float(fill_data.get('commission', 0.0)),
                    fee_asset=fill_data.get('commissionAsset', 'USDT'),
                    timestamp=trade_timestamp
                )
                self.trades.append(new_trade)
                
                # Retorna o objeto de trade para que o chamador (ExecutionEngine) possa logá-lo
                return new_trade
            
            return None
            
        except Exception as e:
            logger.error(f"❌ [ERRO EXEC] Erro ao atualizar meta-ordem {self.id} com fill data: {e}. Data: {fill_data}", exc_info=True)
            self.status = OrderStatus.REJECTED
            return None 


class ExecutionEngine:
    """
    Gerencia a execução de ordens usando estratégias avançadas (e.g., TWAP, SMART).
    Opera em modo Paper Trading ou Live Trading, considerando alavancagem e taxas.
    """

    def __init__(self, config: TradingConfig, connector: BinanceConnector, portfolio: PortfolioOptimizer, system_state: Dict, trade_log_callback: Optional[Callable] = None):
        if not isinstance(config, TradingConfig):
            raise TypeError("🚨 [ERRO EXEC] 'config' deve ser uma instância de TradingConfig.")
        if not isinstance(connector, BinanceConnector):
            raise TypeError("🚨 [ERRO EXEC] 'connector' deve ser uma instância de BinanceConnector.")
        if not isinstance(portfolio, PortfolioOptimizer):
            raise TypeError("🚨 [ERRO EXEC] 'portfolio' deve ser uma instância de PortfolioOptimizer.")
        if not isinstance(system_state, dict):
            raise TypeError("🚨 [ERRO EXEC] 'system_state' deve ser um dicionário.")

        self.config = config
        self.connector = connector
        self.portfolio = portfolio
        self.system_state = system_state # Para diferenciar live/paper e registrar trades recentes na UI
        
        # <<-- ESTA LINHA CORRIGE O ERRO -->>
        self.trade_log_callback = trade_log_callback

        # Dicionário de ordens ativas gerenciadas pelo engine: {'meta_order_id': ExecutionOrder object}
        self.active_orders: Dict[str, ExecutionOrder] = {} 
        # Fila de ordens a serem processadas pelo loop de execução
        self.order_queue = asyncio.Queue() 
        
        self._exec_task: Optional[asyncio.Task] = None # Task para o loop principal de execução
        self._monitor_task: Optional[asyncio.Task] = None # Task para o loop de monitoramento de ordens reais

        logger.info("⚙️ [EXEC] ExecutionEngine inicializado.")

    async def start(self):
        """Inicia os loops de execução e monitoramento de ordens."""
        logger.info("🚀 [EXEC] Motor de Execução iniciado.")
        self._exec_task = asyncio.create_task(self._execution_loop(), name="ExecutionEngine_ExecLoop")
        self._monitor_task = asyncio.create_task(self._monitor_orders_loop(), name="ExecutionEngine_MonitorLoop")

    async def stop(self):
        """Para os loops de execução e monitoramento, e aguarda suas finalizações."""
        logger.info("🛑 [EXEC] Parando o Motor de Execução...")
        
        # Cancela as tasks
        if self._exec_task: 
            self._exec_task.cancel()
        if self._monitor_task: 
            self._monitor_task.cancel()
        
        # Aguarda as tasks serem canceladas e finalizar (com tratamento de exceção)
        try:
            # return_exceptions=True para que gather não falhe se uma task levantar CancelledError
            await asyncio.gather(self._exec_task, self._monitor_task, return_exceptions=True)
            logger.info("✅ [EXEC] Tasks do ExecutionEngine canceladas e finalizadas.")
        except asyncio.CancelledError:
            logger.info("✅ [EXEC] Tasks do ExecutionEngine foram canceladas.")
        except Exception as e:
            logger.error(f"❌ [ERRO EXEC] Exceção durante o desligamento do ExecutionEngine: {e}", exc_info=True)
            
        logger.info("✅ [EXEC] Motor de Execução parado.")

    async def submit_order(self, signal: Signal):
        """
        Submete um novo sinal de trading para execução.
        Calcula a quantidade, alavancagem e escolhe a estratégia de execução.

        Args:
            signal (Signal): O sinal de trading a ser executado.
        """
        if not isinstance(signal, Signal):
            logger.error(f"❌ [ERRO EXEC] Tentativa de submeter objeto não-Signal. Ignorando: {signal}")
            return

        # [POSITION LIMIT] Verifica se já existe uma posição aberta
        has_position = await self._check_position_limit(signal.symbol)
        
        if has_position:
            # Verifica se é uma ordem de fechamento ou redução
            is_reducing = False
            current_pos = self.portfolio.positions.get(signal.symbol)
            
            if signal.action == Action.CLOSE:
                is_reducing = True
                logger.info(f"ℹ️ [EXEC] Sinal CLOSE recebido. Ignorando limite de posição para fechar {signal.symbol}.")
            
            elif current_pos:
                # Se existe posição no portfolio, verifica se a ação é oposta
                if current_pos.quantity > 0 and signal.action == Action.SELL:
                    is_reducing = True
                elif current_pos.quantity < 0 and signal.action == Action.BUY:
                    is_reducing = True
                
                if is_reducing:
                    logger.info(f"ℹ️ [EXEC] Sinal {signal.action} oposto à posição atual ({current_pos.quantity}). Permitindo redução/fechamento.")

            if not is_reducing:
                logger.warning(f"⚠️ [EXEC] LIMITE DE POSIÇÃO: Já existe uma posição aberta para {signal.symbol} e sinal não é de redução. Ordem ignorada.")
                self.system_state["command_feedback"] = f"Limite atingido: Já existe posição aberta para {signal.symbol}."
                return

        order_id = f"exec_{signal.symbol}_{int(time.time() * 1000)}_{signal.action.value.lower()}"
        
        # Lógica SMART para escolher a melhor estratégia de execução
        strategy = self._choose_smart_strategy(signal)
        
        exec_order = ExecutionOrder(signal, strategy, order_id)
        
        # 1. Obter o valor total do portfólio (equity)
        portfolio_total_value_usd = self.portfolio.get_total_value()
        
        # 2. Calcular a margem a ser alocada para este trade (em USD)
        # signal.position_size_pct é a % do capital total que será usada como MARGEM
        margin_to_allocate_usd = portfolio_total_value_usd * signal.position_size_pct
        
        # 3. Obter o preço atual do ativo
        # Usar o preço mais atualizado do portfolio (que é atualizado pelo main loop)
        current_price = self.portfolio.get_current_price(signal.symbol)
        
        if current_price <= 0:
            logger.error(f"❌ [ERRO EXEC] Não foi possível obter o preço atual para {signal.symbol} ou preço é zero. Ordem {order_id} REJEITADA.")
            exec_order.status = OrderStatus.REJECTED
            self.active_orders[order_id] = exec_order 
            self.system_state["command_feedback"] = f"Erro: Ordem {order_id} REJEITADA (Preço {signal.symbol} inválido)."
            return

        # 4. Calcular o valor nocional da ordem (em USD)
        # Notional Value = Margem * Alavancagem
        notional_trade_value_usd = margin_to_allocate_usd * signal.leverage
        exec_order.notional_value_at_creation = notional_trade_value_usd 

        # 5. Calcular a quantidade do ativo (em unidades da base asset)
        # Quantidade = Valor Nocional / Preço Atual
        quantity = notional_trade_value_usd / current_price
        
        # Arredondar a quantidade para a precisão mínima do ativo na Binance
        # Obter precisão dinâmica
        symbol_info = await self.connector.get_symbol_info(signal.symbol)
        qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
        
        exec_order.total_quantity = round(quantity, qty_precision) 
        
        if exec_order.total_quantity <= 0:
            logger.warning(f"⚠️ [EXEC] Quantidade calculada para ordem {order_id} é zero ou negativa. Ordem REJEITADA. Margem alocar: ${margin_to_allocate_usd:,.2f}, Valor Nocional: ${notional_trade_value_usd:,.2f}, Preço: {current_price}.")
            exec_order.status = OrderStatus.REJECTED
            self.active_orders[order_id] = exec_order
            self.system_state["command_feedback"] = f"Aviso: Ordem {order_id} REJEITADA (Quantidade zero)."
            return

        self.active_orders[order_id] = exec_order # Adiciona ao dicionário de ordens ativas
        await self.order_queue.put(exec_order) # Coloca a ordem na fila para processamento
        
        logger.info(f"📦 [EXEC] Meta-ordem {order_id} ({signal.action.value} {exec_order.total_quantity:.4f} {signal.symbol}) enfileirada com estratégia: {strategy}, Alavancagem: {signal.leverage:.2f}x. Valor Nocional: ${notional_trade_value_usd:,.2f}, Margem: ${margin_to_allocate_usd:,.2f}.")
        self.system_state["command_feedback"] = f"Ordem {order_id} ({signal.action.value} {exec_order.total_quantity:.4f} {signal.symbol}) enfileirada. Nocional: ${notional_trade_value_usd:,.2f}."

    async def _execution_loop(self):
        """
        Loop principal que processa as meta-ordens da fila.
        Despacha cada ordem para a estratégia de execução apropriada (Paper ou Real).
        """
        logger.info("⚙️ [EXEC LOOP] Loop de execução de ordens iniciado.")
        while True:
            try:
                # Aguarda a próxima ordem na fila (bloqueia até que uma ordem esteja disponível)
                order = await self.order_queue.get() 
                
                # Verifica o status da ordem antes de processar
                if order.status == OrderStatus.PENDING:
                    asyncio.create_task(self._execute_order(order)) # Cria uma task para executar a ordem, não bloqueia a fila
                else:
                    logger.debug(f"ℹ️ [EXEC LOOP] Ordem {order.id} não está PENDING ({order.status.value}). Ignorando.")
                
                # Marca a tarefa da fila como concluída
                self.order_queue.task_done()

            except asyncio.CancelledError:
                logger.info("🛑 [EXEC LOOP] Loop de execução de ordens cancelado.")
                break 
            except Exception as e:
                logger.exception(f"❌ [ERRO EXEC LOOP] Erro inesperado no loop de execução de ordens: {e}")
                await asyncio.sleep(1) 


    async def _monitor_orders_loop(self):
        """
        Loop que monitora o status das ordens e posições ativas.
        Gerencia trailing stops para posições órfãs.
        """
        monitor_interval = 10 # segundos entre as checagens
        logger.info(f"⚙️ [MONITOR] Loop de monitoramento de ordens e posições iniciado (checa a cada {monitor_interval}s).")
        while True:
            try:
                await asyncio.sleep(monitor_interval)
                
                is_live = self.system_state.get('live_trading_enabled', False)
                if not is_live:
                    continue 

                # Gerenciar posições existentes (Trailing Stops)
                await self._manage_existing_positions()

            except asyncio.CancelledError:
                logger.info("🛑 [MONITOR] Loop de monitoramento de ordens cancelado.")
                break
            except Exception as e:
                logger.error(f"❌ [ERRO MONITOR] Erro inesperado no loop de monitoramento: {e}", exc_info=True)
                await asyncio.sleep(monitor_interval) 

    async def _manage_existing_positions(self):
        """
        Verifica posições carregadas no Portfolio e garante que tenham proteção (Trailing Stop).
        Também monitora alavancagem.
        """
        try:
            # Itera sobre uma cópia das posições para segurança
            positions_copy = list(self.portfolio.positions.items())
            
            for symbol, position in positions_copy:
                if position.quantity == 0:
                    continue

                if not symbol.endswith('USDT'): # Foca em pares USDT
                    continue

                # Verifica se há ordens abertas (especialmente Trailing Stop)
                # Isso depende de get_open_orders ou has_trailing_stop_order ser eficiente
                has_stop = await self.connector.has_trailing_stop_order(symbol)
                
                if not has_stop:
                    logger.warning(f"⚠️ [MONITOR] Posição exposta detectada para {symbol} ({position.quantity}). Criando Trailing Stop de emergência...")
                    
                    # Determina o lado e quantidade (valor absoluto)
                    if position.quantity > 0:
                        position_side = OrderSide.BUY # Long
                    else:
                        position_side = OrderSide.SELL # Short
                    
                    qty = abs(position.quantity)
                    
                    # Usa o stop loss padrão se não soubermos o original
                    await self._place_trailing_stop(
                        symbol=symbol,
                        side=position_side,
                        quantity=qty,
                        stop_loss_pct=self.config.DEFAULT_STOP_LOSS_PCT
                    )
        except Exception as e:
            logger.error(f"❌ [ERRO MONITOR] Erro em _manage_existing_positions: {e}", exc_info=True) 


    async def _execute_order(self, order: ExecutionOrder):
        """
        Despacha a meta-ordem para a estratégia de execução apropriada
        (simulação para Paper Trading ou chamada real para Live Trading).
        """
        is_live = self.system_state.get('live_trading_enabled', False)
        
        # Lógica para ajustar a alavancagem para o símbolo ANTES de enviar a ordem (crucial para Live Trading)
        if is_live and self.config.MAX_LEVERAGE > 1:
            try:
                # Ajusta a alavancagem dinamicamente antes de enviar a ordem
                await self.connector.set_leverage_for_symbol(order.signal.symbol, int(order.signal.leverage))
                logger.debug(f"🌐 [LIVE] Alavancagem ajustada para {order.signal.symbol} para {int(order.signal.leverage)}x.")
            except Exception as e:
                logger.error(f"❌ [ERRO LIVE] Falha ao ajustar alavancagem para {order.signal.symbol} para {order.signal.leverage:.2f}x: {e}. Ordem pode falhar.", exc_info=True)
                order.status = OrderStatus.REJECTED
                self.system_state["command_feedback"] = f"Erro: Ordem {order.id} REJEITADA (Falha ao ajustar alavancagem)."
                return

        if is_live:
            logger.info(f"🟢 [LIVE EXEC] Iniciando execução REAL da meta-ordem {order.id} com a estratégia {order.strategy}.")
            order.status = OrderStatus.OPEN 
            strategy_map = {
                "MARKET": self._execute_market_real, 
                "LIMIT": self._execute_limit_real,
                "TWAP": self._execute_twap_real,
            }
            execution_function = strategy_map.get(order.strategy, self._execute_market_real) 
            await execution_function(order)
        else:
            logger.info(f"📝 [PAPER EXEC] Simulando execução de meta-ordem {order.id} com estratégia {order.strategy}.")
            await self._execute_paper_trade(order)

        # Após a execução (seja real ou simulada), atualiza o feedback na UI
        if order.status == OrderStatus.FILLED:
            self.system_state["command_feedback"] = f"Ordem {order.id} ({order.signal.action.value} {order.total_quantity:.4f} {order.signal.symbol}) PREENCHIDA. Nocional: ${order.notional_value_at_creation:,.2f}."
        elif order.status == OrderStatus.REJECTED:
            self.system_state["command_feedback"] = f"Ordem {order.id} ({order.signal.symbol}) REJEITADA."


    def _choose_smart_strategy(self, signal: Signal) -> str:
        """
        Lógica para a estratégia SMART escolher a melhor execução.
        Aprimorada para considerar o tamanho da ordem em relação ao limite de TWAP.
        """
        # Calcular o valor nocional da ordem em relação ao portfólio
        portfolio_value = self.portfolio.get_total_value()
        notional_value_pct = signal.position_size_pct * signal.leverage
        
        # Se o valor nocional for maior que o threshold de TWAP (geralmente 5% do portfólio)
        if notional_value_pct > self.config.TWAP_THRESHOLD_PCT: 
            # Se a ordem for muito grande, use TWAP para minimizar slippage.
            return "TWAP"
        
        # Se a confiança é alta e o trade é menor, usar MARKET para execução rápida
        if signal.confidence > self.config.MIN_CONFIDENCE_FOR_LARGE_TRADE:
            return "MARKET"

        # Caso contrário, pode ser MARKET ou LIMIT dependendo do contexto.
        # Por padrão, vamos usar MARKET para trades pequenos se não for TWAP.
        return "MARKET"

    async def _execute_paper_trade(self, order: ExecutionOrder):
        """
        Simula a execução de uma ordem no portfólio de papel, considerando taxas e slippage.
        """
        if order.total_quantity <= 0:
            logger.warning(f"⚠️ [PAPER] Ordem de papel {order.id} tem quantidade zero. Marcando como REJEITADA.")
            order.status = OrderStatus.REJECTED
            return
                
        simulated_price = self.portfolio.get_current_price(order.signal.symbol)
        if simulated_price <= 0:
            logger.error(f"❌ [PAPER] Não foi possível obter preço para {order.signal.symbol}. Ordem {order.id} REJEITADA.")
            order.status = OrderStatus.REJECTED
            return

        slippage_factor = 1.0 + self.config.SLIPPAGE_TOLERANCE_PCT if order.signal.action == OrderSide.BUY else 1.0 - self.config.SLIPPAGE_TOLERANCE_PCT
        executed_price = round(simulated_price * slippage_factor, 4)
        executed_notional_value = order.total_quantity * executed_price
        
        is_maker = (order.strategy == "LIMIT")
        fee_rate = self.config.MAKER_FEE if is_maker else self.config.TAKER_FEE
        simulated_fee_amount = executed_notional_value * fee_rate
        
        simulated_trade = Trade(
            trade_id=f"paper_{order.id}_{uuid.uuid4().hex[:8]}",
            order_id=order.id,
            symbol=order.signal.symbol,
            side=order.signal.action, 
            quantity=order.total_quantity,
            executed_price=executed_price,
            fee=simulated_fee_amount,
            fee_asset='USDT', 
            timestamp=datetime.utcnow().replace(tzinfo=timezone.utc)
        )
        
        self.portfolio.update_from_trade(simulated_trade)
        
        order.filled_quantity = order.total_quantity
        order.avg_fill_price = executed_price
        order.status = OrderStatus.FILLED
        order.trades.append(simulated_trade)
        order.notional_value_at_creation = executed_notional_value

        logger.info(f"📝 [PAPER] Ordem {order.id} ({order.signal.action.value} {order.total_quantity:.4f}) SIMULADA @ ${executed_price:.2f}.")
        
        # Cria o dicionário para log e estado da UI
        trade_log_data = {
            "timestamp": simulated_trade.timestamp.isoformat(),
            "symbol": simulated_trade.symbol, 
            "action": simulated_trade.side.value,
            "quantity": simulated_trade.quantity, 
            "price": simulated_trade.executed_price,
            "status": "PAPER_FILLED",
            "notional_value": executed_notional_value,
            "leverage": order.signal.leverage, 
            "profit_probability": order.signal.profit_probability,
            "order_id": order.id,
            "client_order_id": order.id
        }
        
        # Adiciona ao estado do sistema para exibição na UI
        self.system_state["recent_trades"].append(trade_log_data)
        
        # Chama o callback para salvar o trade em arquivo
        if self.trade_log_callback:
            self.trade_log_callback(trade_log_data)

        # [TRAILING STOP] Em Paper Trading, não devemos colocar ordens reais na Binance.
        pass


    async def _execute_market_real(self, order: ExecutionOrder):
        """
        Executa uma ordem a mercado real na Binance Futures.
        """
        if order.total_quantity <= 0:
            logger.warning(f"⚠️ [LIVE] Ordem a mercado {order.id} tem quantidade zero. Marcando como REJEITADA.")
            order.status = OrderStatus.REJECTED
            return

        client_order_id = f"mkt_{uuid.uuid4().hex}"
        order.child_order_ids.append(client_order_id)
        # Obter precisão dinâmica
        symbol_info = await self.connector.get_symbol_info(order.signal.symbol)
        qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
        
        params = {
            "symbol": order.signal.symbol,
            "side": order.signal.action.value, 
            "type": OrderType.MARKET.value,
            "quantity": round(order.total_quantity, qty_precision), 
            "newClientOrderId": client_order_id,
            "newOrderRespType": "FULL", 
        }
        
        logger.info(f"🌐 [LIVE] Enviando ordem a mercado real {client_order_id} para {order.signal.symbol} (Qtd: {params['quantity']}, Alavancagem: {order.signal.leverage:.2f}x)...")
        result = await self.connector.place_order(params)
        
        if result:
            # Captura o trade retornado por update_with_fill para logar
            new_trade = order.update_with_fill(result)
            
            # Se um novo trade (preenchimento total ou parcial) ocorreu, ele é logado
            if new_trade and self.trade_log_callback:
                trade_log_data = {
                    "timestamp": new_trade.timestamp.isoformat(),
                    "symbol": new_trade.symbol,
                    "action": new_trade.side.value,
                    "quantity": new_trade.quantity,
                    "price": new_trade.executed_price,
                    "status": order.status.value,
                    "notional_value": new_trade.quantity * new_trade.executed_price,
                    "leverage": order.signal.leverage,
                    "profit_probability": order.signal.profit_probability,
                    "order_id": result.get('orderId', order.id),
                    "client_order_id": result.get('clientOrderId', order.id)
                }
                self.system_state["recent_trades"].append(trade_log_data)
                self.trade_log_callback(trade_log_data)
            
            # Se não preencheu imediatamente, tenta consultar o status algumas vezes (polling)
            if order.status != OrderStatus.FILLED:
                original_status = order.status.value
                logger.info(f"⏳ [LIVE] Ordem {result.get('orderId')} retornou status {original_status}. Iniciando polling de preenchimento...")
                
                for i in range(5): # Tenta por 5 segundos
                    await asyncio.sleep(1)
                    updated_order_info = await self.connector.get_order(order.signal.symbol, order_id=result.get('orderId'))
                    
                    if updated_order_info:
                        current_status = updated_order_info.get('status')
                        # Se mudou para FILLED, atualiza e processa
                        if current_status == 'FILLED':
                            logger.info(f"✅ [LIVE] Polling: Ordem {result.get('orderId')} agora está FILLED!")
                            
                            # Atualiza trade com dados finais se possível
                            # Nota: update_with_fill lida com diffs, mas aqui queremos garantir o estado final
                            new_trade = order.update_with_fill(updated_order_info)
                            
                            # Se update_with_fill não retornou trade (pq já tinha processado parcial?), força criação se necessário
                            # Mas update_with_fill deve lidar.
                            break
                        elif current_status == 'CANCELED' or current_status == 'REJECTED':
                             logger.warning(f"⚠️ [LIVE] Polling: Ordem {result.get('orderId')} foi {current_status}.")
                             order.status = OrderStatus(current_status)
                             break
                        else:
                             # Ainda NEW ou PARTIALLY_FILLED
                             pass
            
            if order.status == OrderStatus.FILLED:
                logger.info(f"✅ [LIVE] Ordem a mercado {result.get('orderId')} para {order.id} PREENCHIDA.")
                
                # Assegura que temos um objeto trade para referenciar o preço de execução
                # Se new_trade for None (pq foi preenchido no primeiro update ou algo assim),
                # precisamos recuperar o último trade ou criar uma referência.
                if not new_trade and order.trades:
                     new_trade = order.trades[-1]

                if new_trade:
                    # [SAFETY] Implementação de Stop Loss Automático (Hard Stop)
                    # Prioridade: Hard Stop > Trailing Stop
                    sl_pct = order.signal.stop_loss if order.signal.stop_loss > 0 else self.config.DEFAULT_STOP_LOSS_PCT
                    
                    if sl_pct > 0:
                        await self._place_initial_hard_stop(new_trade, sl_pct)

                    # [SAFETY] Implementação de Take Profit Automático
                    tp_pct = order.signal.take_profit if order.signal.take_profit > 0 else self.config.DEFAULT_TAKE_PROFIT_PCT
                    if tp_pct > 0:
                        await self._place_initial_take_profit(new_trade, tp_pct)
                else:
                    logger.warning(f"⚠️ [SAFETY] Não foi possível obter objeto Trade para colocar proteções em {order.id}.")
                
                # [TRAILING STOP] Opcional: Coloca trailing stop também se configurado
                # (Binance permite ambos, mas cuidado com conflitos de margem. Vamos priorizar Hard Stop por enquanto)
                # await self._place_trailing_stop(...) 
            else:
                logger.warning(f"⚠️ [LIVE] Ordem a mercado {result.get('orderId')} para {order.id} não foi FILLED após polling (Status: {order.status.value}).")
        else:
            logger.error(f"❌ [ERRO LIVE] Falha crítica ao enviar ordem a mercado para {order.id}. Resultado nulo do conector.")
            order.status = OrderStatus.REJECTED
            order.signal.explanation["rejection_reason"] = "Falha de comunicação com a exchange ou erro na ordem."


    async def _execute_limit_real(self, order: ExecutionOrder):
        """
        Executa uma ordem limitada real na Binance Futures.
        """
        ticker = await self.connector.get_ticker_price(order.signal.symbol)
        current_market_price = float(ticker['price']) if ticker and 'price' in ticker else 0.0
    
        if current_market_price <= 0:
            logger.error(f"❌ [LIVE] Não foi possível obter preço de mercado para {order.signal.symbol} para ordem limite. Ordem {order.id} REJEITADA.")
            order.status = OrderStatus.REJECTED
            self.system_state["command_feedback"] = f"Erro: Ordem {order.id} REJEITADA (Preço de mercado inválido para ordem limite)."
            return
    
        limit_price = 0.0
        if order.signal.action == OrderSide.BUY:
            # Tentar comprar um pouco abaixo do preço de mercado
            limit_price = current_market_price * (1.0 - self.config.SLIPPAGE_TOLERANCE_PCT * 0.1)
        elif order.signal.action == OrderSide.SELL:
            # Tentar vender um pouco acima do preço de mercado
            limit_price = current_market_price * (1.0 + self.config.SLIPPAGE_TOLERANCE_PCT * 0.1)
    
        if limit_price <= 0:
            logger.error(f"❌ [LIVE] Preço limite calculado para {order.id} é zero ou negativo. Executando como ordem de mercado como fallback.")
            await self._execute_market_real(order)
            return
    
        client_order_id = f"lim_{uuid.uuid4().hex}"
        order.child_order_ids.append(client_order_id)
    
        # Obter precisão dinâmica
        symbol_info = await self.connector.get_symbol_info(order.signal.symbol)
        qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
        price_precision = int(symbol_info['pricePrecision']) if symbol_info else 2

        params = {
            "symbol": order.signal.symbol,
            "side": order.signal.action.value,
            "type": OrderType.LIMIT.value,
            "quantity": round(order.total_quantity, qty_precision),
            "price": round(limit_price, price_precision), # Arredonda o preço para a precisão da exchange
            "timeInForce": "GTC", # Good-Till-Canceled
            "newClientOrderId": client_order_id,
            "newOrderRespType": "FULL",
        }
    
        logger.info(f"🌐 [LIVE] Enviando ordem limite real {client_order_id} para {order.signal.symbol} (Qtd: {params['quantity']}, Preço: {params['price']})...")
        result = await self.connector.place_order(params)
    
        if result:
            # <<-- INÍCIO DA CORREÇÃO -->>
            # Captura o trade retornado por update_with_fill para logar
            new_trade = order.update_with_fill(result)
    
            # Se um novo trade (preenchimento total ou parcial) ocorreu, ele é logado
            if new_trade and self.trade_log_callback:
                trade_log_data = {
                    "timestamp": new_trade.timestamp.isoformat(),
                    "symbol": new_trade.symbol,
                    "action": new_trade.side.value,
                    "quantity": new_trade.quantity,
                    "price": new_trade.executed_price,
                    "status": order.status.value,
                    "notional_value": new_trade.quantity * new_trade.executed_price,
                    "leverage": order.signal.leverage,
                    "profit_probability": order.signal.profit_probability,
                    "order_id": result.get('orderId', order.id),
                    "client_order_id": result.get('clientOrderId', order.id)
                }
                self.system_state["recent_trades"].append(trade_log_data)
                self.trade_log_callback(trade_log_data)
            # <<-- FIM DA CORREÇÃO -->>
    
            if order.status == OrderStatus.FILLED:
                logger.info(f"✅ [LIVE] Ordem limite {result.get('orderId')} para {order.id} PREENCHIDA IMEDIATAMENTE.")
                # [SAFETY] Hard Stop para Limit Orders preenchidas imediatamente
                sl_pct = order.signal.stop_loss if order.signal.stop_loss > 0 else self.config.DEFAULT_STOP_LOSS_PCT
                if sl_pct > 0:
                     await self._place_initial_hard_stop(new_trade, sl_pct)
                
                # [SAFETY] Take Profit para Limit Orders
                tp_pct = order.signal.take_profit if order.signal.take_profit > 0 else self.config.DEFAULT_TAKE_PROFIT_PCT
                if tp_pct > 0:
                     await self._place_initial_take_profit(new_trade, tp_pct)
                     
            elif order.status == OrderStatus.PARTIALLY_FILLED:
                logger.info(f"⏳ [LIVE] Ordem limite {result.get('orderId')} para {order.id} PARCIALMENTE PREENCHIDA.")
                # [SAFETY] Hard Stop para a parte preenchida?
                # Idealmente sim, mas complexo. Vamos deixar para o monitoramento de posição ou preenchimento total.
            else:
                logger.info(f"🅿️ [LIVE] Ordem limite {result.get('orderId')} para {order.id} COLOCADA (Status: {order.status.value}). Aguardando preenchimento.")
        else:
            logger.error(f"❌ [ERRO LIVE] Falha crítica ao enviar ordem limite para {order.id}. Resultado nulo do conector.")
            order.status = OrderStatus.REJECTED
            order.signal.explanation["rejection_reason"] = "Falha de comunicação com a exchange ou erro na ordem limite."
    
    
    async def _execute_twap_real(self, order: ExecutionOrder):
        """
        Executa uma ordem usando a estratégia TWAP (Time-Weighted Average Price) real.
        Divide a ordem em fatias e as envia em intervalos regulares.
        """
        if order.total_quantity <= 0:
            logger.warning(f"⚠️ [LIVE] Ordem TWAP {order.id} tem quantidade zero. Marcando como REJEITADA.")
            order.status = OrderStatus.REJECTED
            return
    
        total_duration_seconds = self.config.TWAP_DURATION_MINUTES * 60
        # Número de fatias (Ex: 60 fatias para 15 minutos = 1 fatia a cada 15s)
        num_slices = max(2, int(total_duration_seconds / 15))
    
        slice_quantity_base = order.total_quantity / num_slices
        slice_interval_seconds = total_duration_seconds / num_slices
    
        logger.info(f"⏳ [LIVE] Executando TWAP para meta-ordem {order.id}: {num_slices} fatias de aproximadamente {slice_quantity_base:.4f} {order.signal.symbol} a cada {slice_interval_seconds:.1f}s.")
    
        for i in range(num_slices):
            if order.is_complete or order.status == OrderStatus.CANCELED or self.system_state.get('shutdown_imminent', False):
                logger.info(f"🎉 [LIVE] Execução TWAP da meta-ordem {order.id} concluída ou cancelada prematuramente.")
                break
    
            try:
                remaining_qty = order.total_quantity - order.filled_quantity
    
                if math.isclose(remaining_qty, 0.0, abs_tol=1e-9):
                    order.status = OrderStatus.FILLED
                    logger.info(f"✅ [LIVE] TWAP para {order.id}: Quantidade restante insignificante. Ordem marcada como FILLED.")
                    break
    
                current_slice_qty = min(slice_quantity_base, remaining_qty)
                
                # Obter precisão dinâmica (idealmentee cacheado ou passado)
                symbol_info = await self.connector.get_symbol_info(order.signal.symbol)
                qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
                
                current_slice_qty = round(current_slice_qty, qty_precision)
    
                if current_slice_qty <= 0:
                    logger.info(f"ℹ️ [LIVE] TWAP para {order.id}: Quantidade da fatia {i+1} é zero ou negativa. Finalizando TWAP.")
                    break
    
                client_order_id = f"twap_{uuid.uuid4().hex[:25]}"
                order.child_order_ids.append(client_order_id)
    
                slice_params = {
                    "symbol": order.signal.symbol,
                    "side": order.signal.action.value,
                    "type": OrderType.MARKET.value, # Geralmente TWAP usa ordens a mercado
                    "quantity": current_slice_qty,
                    "newClientOrderId": client_order_id,
                    "newOrderRespType": "FULL",
                }
    
                logger.info(f"🌐 [LIVE] Enviando fatia TWAP {i+1}/{num_slices} para {order.id} (Qtd: {current_slice_qty:.4f})...")
                result = await self.connector.place_order(slice_params)
    
                if result:
                    # <<-- INÍCIO DA CORREÇÃO -->>
                    # Captura o trade da fatia para logar
                    new_trade = order.update_with_fill(result)
    
                    # Se a fatia resultou em um trade, ele é logado
                    if new_trade and self.trade_log_callback:
                        trade_log_data = {
                            "timestamp": new_trade.timestamp.isoformat(),
                            "symbol": new_trade.symbol,
                            "action": new_trade.side.value,
                            "quantity": new_trade.quantity,
                            "price": new_trade.executed_price,
                            "status": order.status.value, # Status atual da ordem PAI
                            "notional_value": new_trade.quantity * new_trade.executed_price,
                            "leverage": order.signal.leverage,
                            "profit_probability": order.signal.profit_probability,
                            "order_id": result.get('orderId', order.id),
                            "client_order_id": result.get('clientOrderId', order.id)
                        }
                        self.system_state["recent_trades"].append(trade_log_data)
                        self.trade_log_callback(trade_log_data)
                    # <<-- FIM DA CORREÇÃO -->>
    
                    logger.info(f"✅ [LIVE] Fatia TWAP {i+1}/{num_slices} para ordem {order.id} enviada e processada. Status: {order.status.value}.")
                else:
                    logger.warning(f"⚠️ [LIVE] Falha ao enviar fatia TWAP {i+1} para ordem {order.id}. Resultado nulo. Tentando próxima fatia.")
    
                if i < num_slices - 1:
                    await asyncio.sleep(slice_interval_seconds)
            except asyncio.CancelledError:
                logger.info(f"🛑 [LIVE] Execução TWAP da meta-ordem {order.id} cancelada.")
                order.status = OrderStatus.CANCELED
                break
            except Exception as e:
                logger.error(f"❌ [ERRO LIVE] Erro na execução da fatia TWAP {i+1} para ordem {order.id}: {e}", exc_info=True)
                await asyncio.sleep(slice_interval_seconds / 2)
    
        if not order.is_complete:
            logger.warning(f"⚠️ [LIVE] Loop TWAP para meta-ordem {order.id} finalizado, mas a ordem não está completa. Status final: {order.status.value}.")
    
            if order.filled_quantity > 0 and math.isclose(order.filled_quantity, order.total_quantity, rel_tol=1e-5):
                order.status = OrderStatus.FILLED
                logger.info(f"✅ [LIVE] Forçando status FILLED para TWAP {order.id} devido a preenchimento quase completo.")
            else:
                logger.warning(f"⚠️ [LIVE] TWAP {order.id} não preenchida completamente. Quantidade faltante: {order.total_quantity - order.filled_quantity:.4f}.")
    
    async def _check_position_limit(self, symbol: str) -> bool:
        """
        Verifica se já existe uma posição aberta para o símbolo.
        Retorna True se o limite foi atingido (já existe posição), False caso contrário.
        
        Args:
            symbol: Par de trading
        
        Returns:
            True se já existe posição aberta, False caso contrário
        """
        try:
            # Primeiro, verifica o portfolio interno
            if symbol in self.portfolio.positions and self.portfolio.positions[symbol].quantity != 0:
                logger.info(f"🔒 [EXEC] Posição existente encontrada no portfolio para {symbol}.")
                return True
            
            # Depois, verifica na exchange para garantir consistência
            is_live = self.system_state.get('live_trading_enabled', False)
            if is_live:
                account_summary = await self.connector.get_account_summary()
                positions = account_summary.get('positions', {})
                if symbol in positions and positions[symbol].get('quantity', 0) != 0:
                    logger.info(f"🔒 [EXEC] Posição existente encontrada na exchange para {symbol}.")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"❌ [ERRO EXEC] Erro ao verificar limite de posição: {e}", exc_info=True)
            # Em caso de erro, assume que pode haver posição (seguro)
            return True

    async def _place_trailing_stop(self, symbol: str, side: OrderSide, quantity: float, stop_loss_pct: Optional[float] = None):
        """
        Coloca uma ordem de trailing stop na exchange para proteger uma posição.
        
        Args:
            symbol: Par de trading
            side: Lado da POSIÇÃO (OrderSide.BUY para Long, OrderSide.SELL para Short).
                  O trailing stop será colocado no lado OPOSTO.
            quantity: Quantidade da posição (em valor absoluto)
            stop_loss_pct: Percentual de callback (e.g., 0.01 para 1%). Se None, usa config.
        """
        try:
            # Calcula o callback rate
            sl_pct = stop_loss_pct or self.config.DEFAULT_STOP_LOSS_PCT
            callback_rate = abs(sl_pct) * 100
            
            # O lado do trailing stop é o oposto da posição
            if side == OrderSide.BUY:
                trailing_side = "SELL"
            else:
                trailing_side = "BUY"
            
            # Obter precisão do símbolo para quantity
            symbol_info = await self.connector.get_symbol_info(symbol)
            qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
            final_qty = round(quantity, qty_precision)

            logger.info(f"🎯 [EXEC] Colocando Trailing Stop para {symbol}...")
            logger.info(f"   ├─ Posição: {side.value} {quantity:.6f} -> Ajustado: {final_qty}")
            logger.info(f"   └─ Trailing Stop: {trailing_side} @ {callback_rate:.1f}% callback")
            
            result = await self.connector.place_trailing_stop_order(
                symbol=symbol,
                side=trailing_side,
                quantity=final_qty,
                callback_rate=callback_rate
            )
            
            if result:
                order_id = result.get('orderId', 'N/A')
                logger.info(f"✅ [EXEC] Trailing Stop {order_id} colocado com sucesso!")
                self.system_state["command_feedback"] = f"Trailing Stop ativo para {symbol} @ {callback_rate:.1f}%"
            else:
                logger.warning(f"⚠️ [EXEC] Falha ao colocar trailing stop para {symbol}")
        except Exception as e:
            logger.error(f"❌ [ERRO EXEC] Erro ao colocar Trailing Stop para {symbol}: {e}", exc_info=True)

    async def _place_initial_hard_stop(self, trade: Trade, stop_loss_pct: float):
        """
        Coloca uma ordem de Stop Loss Fixo (Hard Stop) imediatamente após a entrada.
        
        Args:
            trade: Objeto Trade com detalhes da execução
            stop_loss_pct: Percentual do stop (ex: 0.02 para 2%)
        """
        try:
            if not trade or stop_loss_pct <= 0:
                return

            # Calcula preço do stop
            entry_price = trade.executed_price
            symbol = trade.symbol
            quantity = abs(trade.quantity)
            
            if trade.side == OrderSide.BUY:
                stop_price = entry_price * (1.0 - stop_loss_pct)
                sl_side = "SELL"
            else: # SELL (Short)
                stop_price = entry_price * (1.0 + stop_loss_pct)
                sl_side = "BUY"
                
            logger.info(f"🛑 [SAFETY] Colocando Hard Stop para {symbol}...")
            logger.info(f"   ├─ Entrada: {entry_price}")
            logger.info(f"   └─ Stop Price: {stop_price:.2f} ({stop_loss_pct:.1%})")
            
            # Chama o conector
            result = await self.connector.place_stop_loss_order(
                symbol=symbol,
                side=sl_side,
                quantity=quantity,
                stop_price=stop_price
            )
            
            if result:
                order_id = result.get('orderId', 'N/A')
                logger.info(f"✅ [SAFETY] Hard Stop {order_id} ativado com sucesso!")
                self.system_state["command_feedback"] = f"Hard Stop ativo para {symbol} @ {stop_price:.2f}"
            else:
                logger.critical(f"🚨 [PERIGO] Falha ao colocar Hard Stop para {symbol}! Posição desprotegida!")
                self.system_state["command_feedback"] = f"PERIGO: Falha ao colocar Stop Loss para {symbol}!"
                
        except Exception as e:
            logger.error(f"❌ [ERRO SAFETY] Erro crítico ao colocar Hard Stop: {e}", exc_info=True)

    async def _place_initial_take_profit(self, trade: Trade, take_profit_pct: float):
        """
        Coloca uma ordem de Take Profit Automático imediatamente após a entrada.
        """
        try:
            if not trade or take_profit_pct <= 0:
                return

            # Calcula preço do TP
            entry_price = trade.executed_price
            symbol = trade.symbol
            quantity = abs(trade.quantity)
            
            if trade.side == OrderSide.BUY:
                stop_price = entry_price * (1.0 + take_profit_pct)
                tp_side = "SELL"
            else: # SELL (Short)
                stop_price = entry_price * (1.0 - take_profit_pct)
                tp_side = "BUY"
                
            logger.info(f"🎯 [SAFETY] Colocando Take Profit para {symbol}...")
            logger.info(f"   ├─ Entrada: {entry_price}")
            logger.info(f"   └─ TP Price: {stop_price:.2f} (+{take_profit_pct:.1%})")
            
            # Chama o conector
            result = await self.connector.place_take_profit_order(
                symbol=symbol,
                side=tp_side,
                quantity=quantity,
                stop_price=stop_price
            )
            
            if result:
                order_id = result.get('orderId', 'N/A')
                logger.info(f"✅ [SAFETY] Take Profit {order_id} ativado com sucesso!")
                self.system_state["command_feedback"] = f"Take Profit ativo para {symbol} @ {stop_price:.2f}"
            else:
                logger.warning(f"⚠️ [SAFETY] Falha ao colocar Take Profit para {symbol}.")
                
        except Exception as e:
            logger.error(f"❌ [ERRO SAFETY] Erro ao colocar Take Profit: {e}", exc_info=True)
