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
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Set
import math
import uuid # Para gerar IDs únicos de trades
import numpy as np

from utils.logger import get_logger
from config.settings import TradingConfig
from models.trade_schema import Signal, Trade, OrderStatus, OrderSide, OrderType, Order, Action, Position
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
        
        # [NOVO] Flag para ordens que visam apenas reduzir a posição (Safety)
        self.reduce_only: bool = False 
        
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

        # [FIX 7] Cache de symbol_info para evitar chamadas repetidas à API por ordem
        # Chave: symbol (str) → valor: {'quantityPrecision': int, 'pricePrecision': int}
        self._symbol_info_cache: Dict[str, Optional[Dict]] = {}

        # [ANTI-SPAM] Rastreia falhas consecutivas de trailing stop por símbolo.
        # Após MAX_STOP_FAILURES tentativas falhas, aumenta o intervalo de retry (backoff).
        self._stop_fail_count: Dict[str, int] = {}
        self._stop_monitor_tick: int = 0   # contador de ciclos do monitor
        self.MAX_STOP_FAILURES = 3         # tentativas antes de ativar backoff
        self.STOP_BACKOFF_CYCLES = 6       # tenta a cada N ciclos quando em backoff (6 × 10s = 60s)
        
        # [PRIORIDADE 1] Contador para Reconciliação Periódica
        self._reconciliation_tick: int = 0
        self.RECONCILIATION_INTERVAL_CYCLES = 6 # 1 min (6 * 10s)
        
        # [RISK VETO] Registro de símbolos vetados por margem para evitar ordens conflitantes
        self._margin_veto_registry: Dict[str, datetime] = {}

        # [SOFTWARE STOP] Stop por software quando a API da exchange rejeita todas as tentativas de stop
        self._software_stops: Dict[str, float] = {}  # symbol → preço de disparo calculado
        
        # [PROFIT SHIELD v2.0] Controle de proteção de lucro e realização parcial
        self._partial_realizations: Dict[str, bool] = {} # symbol → True se já realizou parcial
        self._breakeven_activated: Dict[str, bool] = {}  # symbol → True se BE está ativo

        # [RISK GATE] Referência opcional ao RiskManager para gates de risco intra-TWAP
        # Injetada externamente via set_risk_manager() para evitar dependência circular.
        self._risk_manager: Optional[Any] = None

        # [TWAP PERSISTENCE] Arquivo para salvar/recuperar estado de TWAP em andamento
        state_dir = os.path.join(os.path.dirname(__file__), '..', 'state')
        os.makedirs(state_dir, exist_ok=True)
        self._twap_state_file = os.path.join(state_dir, 'twap_pending.json')

        logger.info("⚙️ [EXEC] ExecutionEngine inicializado.")

    def set_risk_manager(self, risk_manager: Any) -> None:
        """
        Injeta o RiskManager para que o ExecutionEngine possa realizar
        gates de risco intra-TWAP (drawdown, risk-off mode, etc.).
        """
        self._risk_manager = risk_manager
        logger.info("⚙️ [EXEC] RiskManager injetado no ExecutionEngine.")

    def _is_risk_off(self) -> bool:
        """Retorna True se o RiskManager sinalizar modo risk-off (drawdown alto ou margem crítica)."""
        if self._risk_manager is None:
            return False
        try:
            profile = self._risk_manager.get_current_risk_profile()
            return bool(profile.get('is_risk_off_mode', False))
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # [TWAP PERSISTENCE] Salvar / carregar / retomar ordens TWAP interrompidas
    # -------------------------------------------------------------------------
    def _save_twap_state(self, order: 'ExecutionOrder', slice_idx: int, total_slices: int,
                         filled_qty: float, total_qty: float, sl_pct: float, tp_pct: float):
        """Persiste o estado atual de uma execução TWAP em disco."""
        state = {
            'order_id':     order.id,
            'symbol':       order.signal.symbol,
            'action':       order.signal.action.value,
            'slice_idx':    slice_idx,
            'total_slices': total_slices,
            'filled_qty':   filled_qty,
            'total_qty':    total_qty,
            'leverage':     order.signal.leverage,
            'sl_pct':       sl_pct,
            'tp_pct':       tp_pct,
            'saved_at':     datetime.utcnow().isoformat(),
        }
        try:
            with open(self._twap_state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"[TWAP STATE] Falha ao salvar estado: {e}")

    def _clear_twap_state(self):
        """Remove o arquivo de estado TWAP após conclusão ou cancelamento."""
        try:
            if os.path.exists(self._twap_state_file):
                os.remove(self._twap_state_file)
                logger.info("[TWAP STATE] Estado TWAP limpo.")
        except Exception as e:
            logger.warning(f"[TWAP STATE] Falha ao limpar estado: {e}")

    def _load_twap_state(self) -> Optional[Dict]:
        """Carrega estado TWAP pendente do disco (para retomar após restart)."""
        if not os.path.exists(self._twap_state_file):
            return None
        try:
            with open(self._twap_state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            logger.info(
                f"[TWAP STATE] Estado pendente encontrado: {state.get('symbol')} "
                f"{state.get('action')} slice {state.get('slice_idx')}/{state.get('total_slices')} "
                f"({state.get('filled_qty', 0):.4f}/{state.get('total_qty', 0):.4f} preenchido)"
            )
            return state
        except Exception as e:
            logger.warning(f"[TWAP STATE] Falha ao carregar estado: {e}")
            return None

    async def resume_pending_twap(self):
        """
        Chamado no startup para retomar qualquer TWAP que foi interrompido.
        Recria a ExecutionOrder e envia as fatias restantes.
        """
        state = self._load_twap_state()
        if not state:
            return

        logger.warning(
            f"[TWAP RESUME] Retomando TWAP interrompido: {state['symbol']} "
            f"{state['action']} — {state['filled_qty']:.4f}/{state['total_qty']:.4f} BTC preenchido"
        )

        # Reconstrói um sinal mínimo para recriar a ExecutionOrder
        try:
            remaining_qty_base = state['total_qty'] - state['filled_qty']
            if remaining_qty_base <= 1e-6:
                logger.info("[TWAP RESUME] Ordem ja totalmente preenchida. Limpando estado.")
                self._clear_twap_state()
                return

            # Cria sinal sintético com a quantidade restante
            from models.trade_schema import Signal, Action
            resume_signal = Signal(
                symbol=state['symbol'],
                action=Action(state['action']),
                confidence=0.7,  # valor padrão para retomada
                leverage=state.get('leverage', 1.0),
                stop_loss=state.get('sl_pct', 0.03),
                take_profit=state.get('tp_pct', 0.09),
                position_size_pct=remaining_qty_base,
                explanation={'reason': f"TWAP resumido após restart (fatia {state['slice_idx']}/{state['total_slices']})"},
            )

            # Recria e requeue a ordem
            resume_order = ExecutionOrder(resume_signal, 'TWAP', f"resume_{state['order_id']}")
            resume_order.total_quantity = remaining_qty_base
            resume_order.filled_quantity = 0.0
            # [FIX] Agenda enfileiramento somente após live_trading_enabled=True para evitar execução em Paper
            asyncio.create_task(self._enqueue_when_live(resume_order))
            logger.info(f"[TWAP RESUME] Ordem {resume_order.id} agendada (aguardando live_trading_enabled=True).")
        except Exception as e:
            logger.error(f"[TWAP RESUME] Falha ao retomar TWAP: {e}", exc_info=True)
            self._clear_twap_state()

    async def _enqueue_when_live(self, order: 'ExecutionOrder'):
        """Aguarda live_trading_enabled=True antes de enfileirar a ordem. Evita execução em Paper mode no restart."""
        for _ in range(120):
            if self.system_state.get('live_trading_enabled', False):
                break
            await asyncio.sleep(1)
        if not self.system_state.get('live_trading_enabled', False):
            logger.error(f"[TWAP RESUME] Timeout aguardando modo live para {order.id}. Estado TWAP limpo.")
            self._clear_twap_state()
            return
        await self.order_queue.put(order)
        logger.info(f"[TWAP RESUME] ✅ Ordem {order.id} enfileirada após live_trading_enabled=True.")

    async def _cancel_orphan_orders(self):
        """
        Cancela ordens GTC abertas que não correspondem a stops/TP ativos da posição atual.
        Executado no startup para limpar ordens zumbi de sessões anteriores.
        Preserva STOP_MARKET e TAKE_PROFIT_MARKET — cancela todo o resto (LIMIT, LIMIT_MAKER, etc.).
        """
        try:
            # Pequena pausa para garantir que a sessão aiohttp está totalmente inicializada
            await asyncio.sleep(0.5)
            primary_symbol = getattr(self.config, 'PRIMARY_PAIR', 'BTCUSDT')
            open_orders = await self.connector.get_open_orders(primary_symbol)
            if not open_orders:
                logger.info(f"[STARTUP] Nenhuma ordem aberta encontrada para {primary_symbol}. Nada a limpar.")
                return

            # Tipos que DEVEM ser preservados (proteções de posição)
            protected_types = {'STOP_MARKET', 'TAKE_PROFIT_MARKET', 'TRAILING_STOP_MARKET', 'STOP'}

            # Log de diagnóstico: mostra todos os tipos encontrados
            type_summary = {}
            for o in open_orders:
                t = o.get('type', 'UNKNOWN')
                type_summary[t] = type_summary.get(t, 0) + 1
            logger.info(f"[STARTUP] {len(open_orders)} ordens abertas em {primary_symbol}: {type_summary}")

            cancelled = 0
            for o in open_orders:
                order_type = o.get('type', '')
                if order_type not in protected_types:
                    oid = o.get('orderId')
                    try:
                        await self.connector._make_request('DELETE', '/fapi/v1/order', params={
                            'symbol': primary_symbol, 'orderId': oid
                        }, signed=True)
                        cancelled += 1
                        logger.warning(
                            f"🗑️ [STARTUP] Ordem zumbi cancelada: {oid} "
                            f"({order_type} {o.get('side')} @ {o.get('price', o.get('stopPrice', '?'))})"
                        )
                    except Exception as ce:
                        logger.warning(f"⚠️ [STARTUP] Falha ao cancelar ordem {oid}: {ce}")

            if cancelled:
                logger.warning(f"[STARTUP] ✅ {cancelled}/{len(open_orders)} ordem(ns) zumbi cancelada(s) para {primary_symbol}.")
            else:
                logger.info(f"[STARTUP] Todas as {len(open_orders)} ordens abertas são proteções ativas. Nenhuma cancelada.")
        except Exception as e:
            logger.warning(f"⚠️ [STARTUP] Erro ao limpar ordens zumbi: {e}")

    async def start(self):
        """Inicia os loops de execução e monitoramento de ordens."""
        logger.info("[EXEC] Motor de Execucao iniciado.")
        # [STARTUP] Cancela ordens LIMIT GTC orphãs de sessões anteriores
        await self._cancel_orphan_orders()
        # Tenta retomar TWAP interrompido antes de iniciar loops normais
        await self.resume_pending_twap()
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
            
        logger.info("✅ [EXEC] Motor de Execucao parado.")

    # -------------------------------------------------------------------------
    # [FIX 7] Cache de symbol_info — evita N chamadas à API por ordem
    # -------------------------------------------------------------------------
    async def _get_cached_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Retorna symbol_info com cache em memória.
        Na primeira chamada busca da exchange; nas subsequentes retorna do cache.
        Cache é válido durante toda a sessão (precisions não mudam durante o trading).
        """
        if symbol not in self._symbol_info_cache:
            try:
                # [BUG FIX] Chamar o conector, não a si mesmo (recursão infinita)
                info = await self.connector.get_symbol_info(symbol)
                self._symbol_info_cache[symbol] = info
                logger.debug(f"[CACHE] symbol_info armazenado para {symbol}: qty_prec={info.get('quantityPrecision') if info else 'N/A'}")
            except Exception as e:
                logger.warning(f"⚠️ [CACHE] Falha ao buscar symbol_info para {symbol}: {e}. Usando defaults.")
                self._symbol_info_cache[symbol] = None
        return self._symbol_info_cache[symbol]

    # -------------------------------------------------------------------------
    # [FIX 8] Timeout automático para ordens LIMIT que não preenchem
    # -------------------------------------------------------------------------
    async def _cancel_limit_if_not_filled(
        self, order: 'ExecutionOrder', exchange_order_id: str,
        timeout_seconds: int = 300
    ):
        """
        Monitora uma ordem LIMIT pendente. Se não for preenchida dentro do timeout,
        cancela na exchange e re-executa como MARKET (fallback seguro).
        Roda como background task para não bloquear o execution_loop.
        """
        await asyncio.sleep(timeout_seconds)

        if order.status == OrderStatus.FILLED or order.status == OrderStatus.CANCELED:
            return  # Já resolvida — nada a fazer

        try:
            logger.warning(
                f"⏱️ [LIMIT TIMEOUT] Ordem limite {exchange_order_id} ({order.id}) nao preencheu em "
                f"{timeout_seconds}s. Cancelando e re-executando como MARKET..."
            )
            cancel_result = await self.connector.cancel_order(order.signal.symbol, exchange_order_id)
            if cancel_result:
                logger.info(f"✅ [LIMIT TIMEOUT] Ordem {exchange_order_id} cancelada com sucesso.")
            else:
                logger.warning(f"⚠️ [LIMIT TIMEOUT] Cancelamento retornou vazio — pode ja ter preenchido.")

            # Consulta status final antes de reenviar
            final_status = await self.connector.get_order(order.signal.symbol, order_id=exchange_order_id)
            if final_status and final_status.get('status') == 'FILLED':
                logger.info(f"ℹ️ [LIMIT TIMEOUT] Ordem {exchange_order_id} ja estava FILLED. Nao re-executa.")
                order.update_with_fill(final_status)
                return

            # Re-executa como MARKET
            order.status = OrderStatus.PENDING
            order.child_order_ids.clear()
            logger.info(f"🔄 [LIMIT TIMEOUT] Re-executando {order.id} como MARKET...")
            await self._execute_market_real(order)

        except asyncio.CancelledError:
            pass  # Bot desligando — ignora
        except Exception as e:
            logger.error(f"❌ [LIMIT TIMEOUT] Erro ao cancelar/re-executar {order.id}: {e}", exc_info=True)

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
        current_pos = self.portfolio.positions.get(signal.symbol)
        is_reducing = False
        
        if has_position:
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
        
        # [FIX] PRIORIDADE 2 e 3: Se for fechamento/redução, usa quantidade exata e reduceOnly
        if is_reducing and current_pos:
            # Se for CLOSE ou se a % do sinal for >= 90% da posição atual, fecha tudo
            if signal.action == Action.CLOSE or signal.position_size_pct >= 0.9:
                quantity = abs(current_pos.quantity)
                exec_order.reduce_only = True
                logger.info(f"🛡️ [EXEC] Ordem de FECHAMENTO TOTAL para {signal.symbol}. Qtd exata: {quantity:.6f}")
            else:
                # Redução parcial: ainda usa a quantidade calculada, mas marca como reduceOnly
                exec_order.reduce_only = True
                logger.info(f"🛡️ [EXEC] Ordem de REDUÇÃO PARCIAL para {signal.symbol}. reduceOnly ativado.")

        # Arredondar a quantidade para a precisão mínima do ativo na Binance
        symbol_info = await self._get_cached_symbol_info(signal.symbol)
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

                # [PRIORIDADE 1] Reconciliação Periódica (Sincronização com Exchange)
                self._reconciliation_tick += 1
                if self._reconciliation_tick >= self.RECONCILIATION_INTERVAL_CYCLES:
                    self._reconciliation_tick = 0
                    asyncio.create_task(self.reconcile_with_exchange())

            except asyncio.CancelledError:
                logger.info("🛑 [MONITOR] Loop de monitoramento de ordens cancelado.")
                break
            except Exception as e:
                logger.error(f"❌ [ERRO MONITOR] Erro inesperado no loop de monitoramento: {e}", exc_info=True)
                await asyncio.sleep(monitor_interval) 

    async def _manage_existing_positions(self):
        """
        Verifica posições carregadas no Portfolio e garante que tenham proteção (Stop).
        Implementa backoff exponencial para evitar spam quando o endpoint falha repetidamente.
        """
        self._stop_monitor_tick += 1
        try:
            positions_copy = list(self.portfolio.positions.items())

            for symbol, position in positions_copy:
                if position.quantity == 0:
                    # Posição fechada: limpa contadores
                    self._stop_fail_count.pop(symbol, None)
                    self._partial_realizations.pop(symbol, None)
                    self._breakeven_activated.pop(symbol, None)
                    continue

                if not symbol.endswith('USDT'):
                    continue

                # [BACKOFF] Se houve muitas falhas, reduz a frequência de tentativas
                fail_count = self._stop_fail_count.get(symbol, 0)
                if fail_count >= self.MAX_STOP_FAILURES:
                    # Só tenta a cada STOP_BACKOFF_CYCLES ciclos (default 60s)
                    if self._stop_monitor_tick % self.STOP_BACKOFF_CYCLES != 0:
                        continue
                    logger.debug(
                        f"[MONITOR] {symbol}: backoff ativo ({fail_count} falhas). "
                        f"Tentativa {self._stop_monitor_tick // self.STOP_BACKOFF_CYCLES}."
                    )

                # [VETO CHECK] Se foi vetado recentemente por margem, ignora a criação de stop para não gerar conflito
                last_veto = self._margin_veto_registry.get(symbol)
                if last_veto:
                    time_diff = (datetime.utcnow() - last_veto).total_seconds()
                    if time_diff < 300: # 5 minutos de cooldown
                        logger.debug(f"[MONITOR] {symbol}: Ignorando monitoramento de stop devido a veto de margem recente ({time_diff:.0f}s atrás).")
                        continue
                    else:
                        # Limpa veto antigo
                        self._margin_veto_registry.pop(symbol, None)

                # [PROFIT SHIELD v2.0] Aplica lógica de proteção de lucro antes de gerenciar stops normais
                await self._apply_profit_protection_logic(symbol, position)

                # Verifica se já existe alguma ordem de proteção (trailing stop OU stop market)
                open_orders = await self.connector.get_open_orders(symbol)
                has_stop = any(
                    o.get('type') in ('TRAILING_STOP_MARKET', 'STOP_MARKET', 'STOP')
                    for o in (open_orders or [])
                )

                # [SOFTWARE STOP] Verifica se o stop por software está ativo e se foi disparado
                software_stop_price = self._software_stops.get(symbol)
                if software_stop_price is not None:
                    ticker = await self.connector.get_ticker_price(symbol)
                    current_price = float(ticker.get('price', 0)) if ticker else 0.0
                    if current_price > 0:
                        stop_breached = (
                            (position.quantity > 0 and current_price <= software_stop_price) or
                            (position.quantity < 0 and current_price >= software_stop_price)
                        )
                        if stop_breached:
                            logger.warning(
                                f"[MONITOR] 🚨 SOFTWARE STOP DISPARADO: {symbol} @ {current_price:.2f} "
                                f"(stop @ {software_stop_price:.2f}). Fechando posição via MARKET..."
                            )
                            close_side = "SELL" if position.quantity > 0 else "BUY"
                            sym_info = await self._get_cached_symbol_info(symbol)
                            qty_prec = int(sym_info['quantityPrecision']) if sym_info else 3
                            close_qty = round(abs(position.quantity), qty_prec)
                            close_result = await self.connector.place_order({
                                'symbol': symbol, 'side': close_side, 'type': 'MARKET',
                                'quantity': close_qty, 'reduceOnly': 'true',
                            })
                            if close_result and close_result.get('orderId'):
                                logger.warning(f"[MONITOR] ✅ Stop software executado: {symbol} fechado @ market. OrderId={close_result.get('orderId')}")
                                self._software_stops.pop(symbol, None)
                        else:
                            # Stop ainda não disparado → trata como proteção ativa
                            has_stop = True

                if not has_stop:
                    logger.warning(
                        f"[MONITOR] Posicao exposta: {symbol} ({position.quantity}). "
                        f"Criando stop de emergencia (tentativa #{fail_count + 1})..."
                    )
                    position_side = OrderSide.BUY if position.quantity > 0 else OrderSide.SELL
                    qty = abs(position.quantity)

                    result = await self._place_trailing_stop(
                        symbol=symbol,
                        side=position_side,
                        quantity=qty,
                        stop_loss_pct=self.config.DEFAULT_STOP_LOSS_PCT
                    )

                    if result:
                        # Stop colocado com sucesso: reseta contador
                        self._stop_fail_count[symbol] = 0
                    else:
                        self._stop_fail_count[symbol] = fail_count + 1
                        if self._stop_fail_count[symbol] >= self.MAX_STOP_FAILURES:
                            logger.warning(
                                f"[MONITOR] {symbol}: {self._stop_fail_count[symbol]} falhas ao colocar stop. "
                                f"Ativando backoff de {self.STOP_BACKOFF_CYCLES * 10}s entre tentativas."
                            )
                else:
                    # Stop ativo: reseta contador de falhas
                    if self._stop_fail_count.get(symbol, 0) > 0:
                        logger.info(f"[MONITOR] {symbol}: stop ativo detectado. Resetando backoff.")
                    self._stop_fail_count[symbol] = 0

        except Exception as e:
            logger.error(f"[MONITOR] Erro em _manage_existing_positions: {e}", exc_info=True) 


    async def _apply_profit_protection_logic(self, symbol: str, position: Position):
        """
        [PROFIT SHIELD v2.0] Implementa proteção dinâmica de lucros:
        1. Break-Even (ROI > 2%)
        2. Realização Parcial 50% (ROI > 4%)
        3. Trailing Agressivo (ROI > 5%)
        """
        try:
            if position.quantity == 0 or position.entry_price <= 0:
                return

            # 1. Calcular ROI Alavancado (considerando mark_price do monitor)
            # Mark price pode vir do conector ou estar na classe Position
            mark_price = position.mark_price
            if mark_price <= 0:
                # Fallback para ticker se mark_price não estiver disponível
                ticker = await self.connector.get_ticker_price(symbol)
                mark_price = float(ticker.get('price', 0)) if ticker else 0.0
            
            if mark_price <= 0:
                return

            # ROI = (Direção) * (Variação Preço) * Alavancagem
            direction = 1 if position.quantity > 0 else -1
            price_change_pct = (mark_price - position.entry_price) / position.entry_price
            roi_pct = direction * price_change_pct * position.leverage

            # ROI em formato percentual (ex: 0.052 = 5.2%)
            roi_display = roi_pct * 100
            
            # --- FASE 1: BREAK-EVEN (ROI > 2%) ---
            if roi_display >= 2.0 and not self._breakeven_activated.get(symbol):
                # Move o stop para o preço de entrada + 0.5% de "folga" para taxas
                fee_buffer = 0.005 # 0.5%
                be_price = position.entry_price * (1 + (direction * fee_buffer))
                
                self._software_stops[symbol] = be_price
                self._breakeven_activated[symbol] = True
                
                logger.info(f"🛡️ [PROFIT SHIELD] {symbol} atingiu {roi_display:.2f}% ROI. BREAK-EVEN ATIVADO @ {be_price:.2f}")

            # --- FASE 2: REALIZAÇÃO PARCIAL 50% (ROI > 4%) ---
            if roi_display >= 4.0 and not self._partial_realizations.get(symbol):
                logger.warning(f"💰 [PROFIT SHIELD] {symbol} atingiu {roi_display:.2f}% ROI. Executando REALIZAÇÃO PARCIAL (50%).")
                
                close_side = "SELL" if position.quantity > 0 else "BUY"
                sym_info = await self._get_cached_symbol_info(symbol)
                qty_prec = int(sym_info['quantityPrecision']) if sym_info else 3
                
                # Fecha metade da posição atual
                partial_qty = round(abs(position.quantity) * 0.5, qty_prec)
                
                if partial_qty > 0:
                    result = await self.connector.place_order({
                        'symbol': symbol, 'side': close_side, 'type': 'MARKET',
                        'quantity': partial_qty, 'reduceOnly': 'true',
                    })
                    if result and result.get('orderId'):
                        self._partial_realizations[symbol] = True
                        logger.warning(f"✅ [PROFIT SHIELD] Realização parcial executada para {symbol}. Qtd: {partial_qty}")
                    else:
                        logger.error(f"❌ [PROFIT SHIELD] Falha ao realizar parcial para {symbol}: {result}")

            # --- FASE 3: TRAILING AGRESSIVO (ROI > 5%) ---
            if roi_display >= 5.0:
                # Aperta o software stop para garantir que saia com pelo menos 3.5% de ROI
                # Distância de segurança de ~1.5% do preço atual
                tight_offset = 0.015 / position.leverage 
                target_stop = mark_price * (1 - (direction * tight_offset))
                
                # Apenas atualiza se o novo stop for mais lucrativo que o anterior (Trailing)
                current_sw_stop = self._software_stops.get(symbol)
                is_better = False
                if current_sw_stop is None:
                    is_better = True
                else:
                    if direction > 0: # Long
                        is_better = target_stop > current_sw_stop
                    else: # Short
                        is_better = target_stop < current_sw_stop
                
                if is_better:
                    self._software_stops[symbol] = target_stop
                    logger.info(f"🦾 [PROFIT SHIELD] {symbol} em lucro forte ({roi_display:.2f}%). Trailing Software Stop apertado para {target_stop:.2f}")

        except Exception as e:
            logger.error(f"❌ [ERRO PROFIT SHIELD] Falha ao aplicar lógica para {symbol}: {e}")

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
                "LADDER": self._execute_ladder_entry,
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
        """
        # Calcular o valor nocional da ordem em relação ao portfólio
        portfolio_value = self.portfolio.get_total_value()
        notional_value_usd = portfolio_value * signal.position_size_pct * signal.leverage
        
        # [NEW] Estratégia LADDER para ordens grandes (Server-Side Slicing)
        if notional_value_usd > 1000: # Threshold de $1000 para laddering
            return "LADDER"

        # Se o valor nocional for maior que o threshold de TWAP (geralmente 5% do portfólio)
        if (signal.position_size_pct * signal.leverage) > self.config.TWAP_THRESHOLD_PCT: 
            return "TWAP"
        
        return "MARKET"

    async def _execute_ladder_entry(self, order: ExecutionOrder):
        """
        [NEW] Executa uma entrada escalonada (Ladder/Grid) usando Batch Orders na Binance.
        Slica a ordem em 5 fatias: 1 Market + 4 Limit.
        """
        try:
            symbol = order.signal.symbol
            total_qty = order.total_quantity
            side = order.signal.action.value
            
            ticker = await self.connector.get_ticker_price(symbol)
            current_price = float(ticker['price']) if ticker else 0.0
            
            if current_price <= 0:
                await self._execute_market_real(order)
                return

            # Obter precisão dinâmica
            symbol_info = await self._get_cached_symbol_info(symbol)
            qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
            price_precision = int(symbol_info['pricePrecision']) if symbol_info else 2

            # Fatiamento Ponderado: 40% Market, 20% L1, 15% L2, 15% L3, 10% L4
            weights = [0.40, 0.20, 0.15, 0.15, 0.10]
            batch_orders = []
            
            direction = 1 if side == 'SELL' else -1 # No SELL, queremos vender mais ALTO. No BUY, comprar mais BAIXO.
            
            for i, weight in enumerate(weights):
                slice_qty = round(total_qty * weight, qty_precision)
                slice_notional = slice_qty * current_price
                
                if slice_qty <= 0 or slice_notional < 55:
                    # Se uma fatia pequena falhar na regra de $55, redistribuímos para a primeira (Market)
                    if i > 0 and len(batch_orders) > 0:
                        # Incrementa a primeira ordem (Market) com o que sobrou
                        first_qty = float(batch_orders[0]['quantity'])
                        batch_orders[0]['quantity'] = f"{first_qty + slice_qty:.{qty_precision}f}"
                        continue
                    else:
                        # Se for a primeira e for pequena, aborta ladder e faz market único
                        logger.info(f"⚠️ [LADDER] Ordem muito pequena para fatiamento ponderado. Executando como MARKET único.")
                        await self._execute_market_real(order)
                        return

                if i == 0:
                    # 1. Primeira fatia: MARKET (40%)
                    batch_orders.append({
                        'symbol': symbol, 'side': side, 'type': 'MARKET', 
                        'quantity': f"{slice_qty:.{qty_precision}f}"
                    })
                else:
                    # 2. Fatias LIMIT (Escalonadas)
                    offset = 0.001 * i # 0.1% a 0.4%
                    target_price = current_price * (1 + (direction * offset))
                    batch_orders.append({
                        'symbol': symbol, 'side': side, 'type': 'LIMIT',
                        'quantity': f"{slice_qty:.{qty_precision}f}",
                        'price': f"{target_price:.{price_precision}f}",
                        'timeInForce': 'GTC'
                    })

            logger.info(f"🧱 [LADDER PONDERADO] Enviando BATCH (40/20/15/15/10) para {symbol}...")
            results = await self.connector.place_batch_orders(batch_orders)
            
            if results and isinstance(results, list):
                # Processa o primeiro resultado como o principal para o estado da ExecutionOrder
                main_res = results[0]
                if main_res.get('orderId'):
                    order.status = OrderStatus.OPEN
                    order.update_with_fill(main_res)
                    logger.info(f"✅ [LADDER] Batch enviado com sucesso. ID Principal: {main_res.get('orderId')}")
            else:
                logger.error(f"❌ [LADDER] Falha ao enviar Batch: {results}")
                await self._execute_market_real(order) # Fallback para market se batch falhar

        except Exception as e:
            logger.error(f"❌ [ERRO LADDER] Falha na estratégia escalonada: {e}")
            await self._execute_market_real(order)

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
        symbol_info = await self._get_cached_symbol_info(order.signal.symbol)
        qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
        
        params = {
            "symbol": order.signal.symbol,
            "side": order.signal.action.value, 
            "type": OrderType.MARKET.value,
            "quantity": round(order.total_quantity, qty_precision), 
            "newClientOrderId": client_order_id,
            "newOrderRespType": "FULL", 
        }
        
        # [FIX] Adiciona reduceOnly se a meta-ordem exigir
        if order.reduce_only:
            params["reduceOnly"] = "true"

        
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
                    # [BUG FIX] Atualiza o Portfolio com o trade real — sem isso o RiskManager opera com dados velhos
                    self.portfolio.update_from_trade(new_trade)
                    logger.debug(f"✅ [LIVE] Portfolio atualizado com trade real: {new_trade.side.value} {new_trade.quantity:.4f} @ {new_trade.executed_price:.2f}")

                    # [SAFETY] Bracket Orders (Server-Side) ativado imediatamente
                    sl_pct = order.signal.stop_loss if (order.signal.stop_loss is not None and order.signal.stop_loss > 0) else self.config.DEFAULT_STOP_LOSS_PCT
                    tp_pct = order.signal.take_profit if (order.signal.take_profit is not None and order.signal.take_profit > 0) else self.config.DEFAULT_TAKE_PROFIT_PCT
                    
                    # 1. Stop Loss na Exchange (Condicional)
                    await self._place_initial_hard_stop(new_trade, sl_pct)
                    
                    # 2. Take Profit na Exchange (Condicional)
                    if tp_pct > 0:
                        await self._place_initial_take_profit(new_trade, tp_pct)

                    # 3. Trailing Stop (Híbrido)
                    await self._place_trailing_stop(
                        symbol=new_trade.symbol,
                        side=new_trade.side,
                        quantity=abs(new_trade.quantity),
                        stop_loss_pct=sl_pct
                    )
                else:
                    logger.warning(f"⚠️ [SAFETY] Nao foi possivel obter objeto Trade para colocar protecoes em {order.id}.")
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
        symbol_info = await self._get_cached_symbol_info(order.signal.symbol)
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
        
        if order.reduce_only:
            params["reduceOnly"] = "true"

    
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
                if new_trade:
                    # [BUG FIX] Atualiza o Portfolio com o trade real
                    self.portfolio.update_from_trade(new_trade)
                    # [SAFETY] Hard Stop imediato
                    sl_pct = order.signal.stop_loss if (order.signal.stop_loss is not None and order.signal.stop_loss > 0) else self.config.DEFAULT_STOP_LOSS_PCT
                    if sl_pct > 0:
                        await self._place_initial_hard_stop(new_trade, sl_pct)
                    # [SAFETY] Take Profit
                    tp_pct = order.signal.take_profit if (order.signal.take_profit is not None and order.signal.take_profit > 0) else self.config.DEFAULT_TAKE_PROFIT_PCT
                    if tp_pct > 0:
                        await self._place_initial_take_profit(new_trade, tp_pct)
                    # [BUG FIX] Trailing Stop imediato para Limit Orders
                    await self._place_trailing_stop(
                        symbol=new_trade.symbol,
                        side=new_trade.side,
                        quantity=abs(new_trade.quantity),
                        stop_loss_pct=sl_pct
                    )
            elif order.status == OrderStatus.PARTIALLY_FILLED:
                logger.info(f"⏳ [LIVE] Ordem limite {result.get('orderId')} para {order.id} PARCIALMENTE PREENCHIDA.")
                # Monitor loop (10s) cobre o fill restante; proteções serão colocadas quando FILLED.
            else:
                exchange_oid = str(result.get('orderId', ''))
                timeout_s = getattr(self.config, 'LIMIT_ORDER_TIMEOUT_SECONDS', 300)
                logger.info(
                    f"🅿️ [LIVE] Ordem limite {exchange_oid} para {order.id} COLOCADA. "
                    f"Timeout automatico em {timeout_s}s se nao preencher."
                )
                # [FIX 8] Inicia watchdog: cancela e re-executa como MARKET se nao preencher
                asyncio.create_task(
                    self._cancel_limit_if_not_filled(order, exchange_oid, timeout_s),
                    name=f"limit_timeout_{order.id}"
                )
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
        
        # Obter precisão dinâmica para calcular o fatiamento seguro
        symbol_info = await self._get_cached_symbol_info(order.signal.symbol)
        qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
        min_qty = 10**(-qty_precision)
        
        # Número de fatias desejado (Ex: 60 fatias para 15 minutos = 1 fatia a cada 15s)
        desired_slices = max(2, int(total_duration_seconds / 15))
        
        # [FIX NOCIONAL MÍNIMO] Binance Futures exige nocional >= $50 para BTCUSDT
        current_price = self.portfolio.get_current_price(order.signal.symbol)
        if current_price <= 0:
             # Fallback: consulta o preço atual diretamente na exchange
             ticker = await self.connector.get_ticker_price(order.signal.symbol)
             current_price = float(ticker['price']) if ticker and 'price' in ticker else 0.0
             
        min_notional_binance = 50.5 # $50 + margem
        total_notional = order.total_quantity * current_price
        
        # O número máximo de fatias é o nocional total dividido pelo mínimo exigido
        # Se total é $200, max_slices é 200 / 50 = 4.
        max_slices_allowed = max(1, int(total_notional / min_notional_binance))
        
        # O número final de fatias é o menor entre o desejado (pelo tempo) e o permitido (pelo dinheiro)
        num_slices = min(desired_slices, max_slices_allowed)
        
        # Ajuste final pela precisão da quantidade
        max_slices_by_qty = max(1, int(order.total_quantity / (min_qty * 1.1)))
        num_slices = max(1, min(num_slices, max_slices_by_qty))
        
        slice_quantity_base = order.total_quantity / num_slices
        slice_interval_seconds = total_duration_seconds / num_slices
    
        # SL/TP percentuais para persistência de estado
        sl_pct_order = order.signal.stop_loss if (order.signal.stop_loss is not None and order.signal.stop_loss > 0) else self.config.DEFAULT_STOP_LOSS_PCT
        tp_pct_order = order.signal.take_profit if (order.signal.take_profit is not None and order.signal.take_profit > 0) else self.config.DEFAULT_TAKE_PROFIT_PCT

        logger.info(
            f"[LIVE] TWAP {order.id}: {num_slices} fatias (Limite Nocional: {max_slices_allowed}). "
            f"Total: ${total_notional:.2f}, Fatia: ~${(total_notional/num_slices):.2f}. "
            f"Intervalo: {slice_interval_seconds:.1f}s."
        )

        # [TWAP PERSISTENCE] Salva estado inicial antes de começar
        self._save_twap_state(order, 0, num_slices, 0.0, order.total_quantity, sl_pct_order, tp_pct_order)

        for i in range(num_slices):
            # [RISK GATE] Cancela TWAP se o RiskManager entrar em modo risk-off (drawdown alto, etc.)
            if self._is_risk_off():
                logger.warning(
                    f"🛑 [TWAP RISK GATE] RiskManager em modo RISK-OFF. "
                    f"Cancelando TWAP {order.id} na fatia {i+1}/{num_slices}. "
                    f"Qty já executada: {order.filled_quantity:.4f}/{order.total_quantity:.4f}"
                )
                order.status = OrderStatus.CANCELED
                self._save_twap_state(order, i, num_slices, order.filled_quantity, order.total_quantity, sl_pct_order, tp_pct_order)
                break

            # [POSITION CHECK] A partir da 2ª fatia: verifica se a posição ainda existe na exchange.
            # Se o stop loss disparou entre fatias, a posição foi fechada e não devemos abrir nova.
            if i > 0 and order.filled_quantity > 0:
                try:
                    pos_risk = await self.connector._make_request(
                        'GET', '/fapi/v2/positionRisk',
                        params={'symbol': order.signal.symbol, 'recvWindow': 5000}, signed=True
                    )
                    exchange_qty = 0.0
                    if pos_risk:
                        exchange_qty = sum(
                            float(p.get('positionAmt', 0))
                            for p in pos_risk if p.get('symbol') == order.signal.symbol
                        )
                    if abs(exchange_qty) < 1e-6:
                        logger.warning(
                            f"🛑 [TWAP POSITION CHECK] Posição {order.signal.symbol} encerrada na exchange "
                            f"(stop disparado?) antes da fatia {i+1}/{num_slices}. "
                            f"Cancelando fatias restantes."
                        )
                        order.status = OrderStatus.CANCELED
                        self._clear_twap_state()
                        break
                except Exception as pos_check_err:
                    logger.warning(f"⚠️ [TWAP POSITION CHECK] Falha ao verificar posição: {pos_check_err}. Continuando TWAP.")

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
                symbol_info = await self._get_cached_symbol_info(order.signal.symbol)
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
                    # <<-- INÍCIO DA CORREÇÃO (Proteções por Fatia) -->>
                    # Captura o trade da fatia para logar
                    new_trade = order.update_with_fill(result)
    
                    # Se a fatia resultou em um trade, ele é logado e as proteções são atualizadas
                    if new_trade:
                        if self.trade_log_callback:
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
                        
                        # Atualiza o Portfolio com o trade real
                        self.portfolio.update_from_trade(new_trade)
                        
                        # [SAFETY] Coloca/Atualiza proteções na Binance para a posição atualizada
                        sl_pct = order.signal.stop_loss if (order.signal.stop_loss is not None and order.signal.stop_loss > 0) else self.config.DEFAULT_STOP_LOSS_PCT
                        tp_pct = order.signal.take_profit if (order.signal.take_profit is not None and order.signal.take_profit > 0) else self.config.DEFAULT_TAKE_PROFIT_PCT

                        # [CONFIDENCE-BASED TRAILING] Ajusta callback do trailing stop pela confiança do sinal.
                        # Alta confiança → trailing mais largo (trade respira mais, captura mais upside)
                        # Baixa confiança → trailing mais apertado (protege capital mais rápido)
                        # Fórmula: callback = sl_pct × (0.5 + confiança × 1.0), clampado em [0.5×sl, 2×sl]
                        confidence = float(order.signal.confidence or 0.60)
                        confidence_factor = 0.5 + confidence * 1.0  # range [0.5, 1.5] para conf [0.0, 1.0]
                        trailing_sl_pct = float(np.clip(sl_pct * confidence_factor, sl_pct * 0.5, sl_pct * 2.0))

                        logger.info(
                            f"🎯 [SAFETY] Proteções para {new_trade.symbol}: "
                            f"SL={sl_pct:.1%} | TP={tp_pct:.1%} | "
                            f"Trailing={trailing_sl_pct:.1%} (conf={confidence:.0%})"
                        )

                        # Coloca proteções na exchange: Hard Stop + TP + Trailing
                        try:
                            await self._place_initial_hard_stop(new_trade, sl_pct)
                            if tp_pct > 0:
                                await self._place_initial_take_profit(new_trade, tp_pct)
                            await self._place_trailing_stop(
                                symbol=new_trade.symbol,
                                side=new_trade.side,
                                quantity=abs(self.portfolio.get_position_quantity(new_trade.symbol)),
                                stop_loss_pct=trailing_sl_pct  # trailing usa callback dinâmico
                            )
                        except Exception as protect_err:
                            logger.warning(f"⚠️ [TWAP SAFETY] Erro ao colocar proteções para fatia: {protect_err}")
                    # <<-- FIM DA CORREÇÃO -->>

                    # [TWAP PERSISTENCE] Atualiza estado após cada fatia bem-sucedida
                    self._save_twap_state(
                        order, i + 1, num_slices,
                        order.filled_quantity, order.total_quantity,
                        sl_pct_order, tp_pct_order
                    )

                    logger.info(
                        f"[LIVE] Fatia TWAP {i+1}/{num_slices} OK. "
                        f"Preenchido: {order.filled_quantity:.4f}/{order.total_quantity:.4f}"
                    )
                else:
                    logger.warning(
                        f"[LIVE] Falha na fatia TWAP {i+1} para {order.id}. Tentando proxima fatia."
                    )

                if i < num_slices - 1:
                    await asyncio.sleep(slice_interval_seconds)
            except asyncio.CancelledError:
                # [TWAP PERSISTENCE] Salva estado ao ser cancelado para retomada posterior
                self._save_twap_state(
                    order, i, num_slices,
                    order.filled_quantity, order.total_quantity,
                    sl_pct_order, tp_pct_order
                )
                logger.info(f"[LIVE] TWAP {order.id} cancelado na fatia {i+1}. Estado salvo para retomada.")
                order.status = OrderStatus.CANCELED
                break
            except Exception as e:
                logger.error(f"[LIVE] Erro na fatia TWAP {i+1} para {order.id}: {e}", exc_info=True)
                # Salva estado mesmo em caso de erro
                self._save_twap_state(
                    order, i, num_slices,
                    order.filled_quantity, order.total_quantity,
                    sl_pct_order, tp_pct_order
                )
                await asyncio.sleep(slice_interval_seconds / 2)
    
        if not order.is_complete:
            logger.warning(f"⚠️ [LIVE] Loop TWAP para meta-ordem {order.id} finalizado, mas a ordem nao esta completa. Status final: {order.status.value}.")

            if order.filled_quantity > 0 and math.isclose(order.filled_quantity, order.total_quantity, rel_tol=1e-5):
                order.status = OrderStatus.FILLED
                logger.info(f"✅ [LIVE] Forcando status FILLED para TWAP {order.id} devido a preenchimento quase completo.")
            else:
                logger.warning(f"⚠️ [LIVE] TWAP {order.id} nao preenchida completamente. Quantidade faltante: {order.total_quantity - order.filled_quantity:.4f}.")

        # [BUG FIX] TWAP nao colocava protecoes. Agora coloca Hard Stop + Trailing Stop ao concluir.
        if order.status == OrderStatus.FILLED and order.trades:
            last_trade = order.trades[-1]
            # Atualiza portfolio com o trade TWAP consolidado
            self.portfolio.update_from_trade(last_trade)
            sl_pct = order.signal.stop_loss if order.signal.stop_loss > 0 else self.config.DEFAULT_STOP_LOSS_PCT
            if sl_pct > 0:
                await self._place_initial_hard_stop(last_trade, sl_pct)
            tp_pct = order.signal.take_profit if order.signal.take_profit > 0 else self.config.DEFAULT_TAKE_PROFIT_PCT
            if tp_pct > 0:
                await self._place_initial_take_profit(last_trade, tp_pct)
            await self._place_trailing_stop(
                symbol=last_trade.symbol,
                side=last_trade.side,
                quantity=abs(order.filled_quantity),
                stop_loss_pct=sl_pct
            )
            logger.info(f"[TWAP] Protecoes colocadas para TWAP {order.id}: Hard Stop + TP + Trailing Stop.")

        # [TWAP PERSISTENCE] Limpa estado após conclusão (sucesso ou cancelamento completo)
        self._clear_twap_state()

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
            symbol_info = await self._get_cached_symbol_info(symbol)
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

            if result and result.get('software_stop'):
                sw_price = result.get('stop_price', 0.0)
                self._software_stops[symbol] = sw_price
                logger.warning(f"[EXEC] 🛡️ Stop SOFTWARE ativado para {symbol} @ {sw_price:.2f} (API rejeitou todas as tentativas)")
                self.system_state["command_feedback"] = f"Stop SOFTWARE {symbol} @ {sw_price:.2f}"
                return result
            elif result and result.get('orderId'):
                order_id = result.get('orderId', 'N/A')
                order_type = result.get('type', 'STOP')
                logger.info(f"[EXEC] Stop {order_type} {order_id} colocado para {symbol} @ {callback_rate:.1f}%")
                self.system_state["command_feedback"] = f"Stop ativo {symbol} @ {callback_rate:.1f}%"
                return result
            else:
                logger.warning(f"[EXEC] Falha ao colocar stop para {symbol}")
                return None
        except Exception as e:
            logger.error(f"[EXEC] Erro ao colocar Trailing Stop para {symbol}: {e}", exc_info=True)
            return None

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

    async def reconcile_with_exchange(self):
        """
        [PRIORIDADE 1] Sincroniza o estado interno (Portfolio e RiskManager)
        com os dados reais da Exchange. Corrige desvios de PnL e quantidade.
        """
        try:
            logger.info("🔄 [RECONCILIATION] Iniciando sincronização total com a Binance...")
            summary = await self.connector.get_account_summary()
            
            if not summary or 'total_value' not in summary:
                logger.warning("⚠️ [RECONCILIATION] Falha ao obter resumo da conta. Abortando sync.")
                return

            # 1. Reconcilia Posições no Portfolio
            self.portfolio.reconcile_positions(summary.get('positions', {}))
            
            # 2. Atualiza Caixa e Margem
            self.portfolio.cash = summary.get('cash', self.portfolio.cash)
            self.portfolio.margin_used = summary.get('margin_used', self.portfolio.margin_used)
            
            logger.info(
                f"✅ [RECONCILIATION] Sync concluído. Equity: ${summary['total_value']:,.2f}, "
                f"Cash: ${summary['cash']:,.2f}, Margem: ${summary['margin_used']:,.2f}, "
                f"Posições: {summary['positions_count']}"
            )
            
        except Exception as e:
            logger.error(f"❌ [RECONCILIATION] Erro durante sincronização: {e}", exc_info=True)

    def register_margin_veto(self, symbol: str):
        """
        Registra que um trade foi vetado por margem para este símbolo.
        Isso suprime o monitoramento de stops por um curto período para evitar conflitos.
        """
        from datetime import datetime
        self._margin_veto_registry[symbol] = datetime.utcnow()
        logger.warning(f"🛡️ [EXEC] Registro de VETO POR MARGEM para {symbol}. Monitoramento de stop será suprimido por 5 minutos.")
