# -----------------------------------------------------------------------------
# ARQUIVO: trading/portfolio.py
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import replace

from utils.logger import get_logger
from config.settings import TradingConfig
from models.trade_schema import Position, Trade, OrderSide

logger = get_logger("PortfolioOptimizer")


class PortfolioOptimizer:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.initial_capital = self.config.INITIAL_CAPITAL
        self.cash = self.config.INITIAL_CAPITAL  # Capital disponível em USDT
        self.positions: Dict[
            str, Position
        ] = {}  # Posições abertas: {'symbol': Position}
        self.portfolio_value_history = pd.Series(
            dtype=float
        )  # Histórico do valor total do portfólio
        self.last_prices: Dict[
            str, float
        ] = {}  # Armazena os últimos preços conhecidos para cálculo de PnL não realizado
        self._last_funding_timestamp: datetime = datetime.now(timezone.utc)

        # Novos atributos para gerenciar a alavancagem e exposição
        self.margin_used = 0.0  # Total de margem alocada em posições abertas
        self.total_notional_value = (
            0.0  # Valor nocional total de todas as posições abertas
        )

        # Inicializa o histórico com o capital inicial
        self.portfolio_value_history[datetime.now(timezone.utc)] = self.initial_capital

        logger.info(
            f"💰 [PORTFOLIO] PortfolioOptimizer inicializado. Capital Inicial: ${self.initial_capital:,.2f}."
        )

    def update_from_trade(self, trade: Trade):
        """
        Atualiza o estado do portfólio com base em um trade executado.
        Gerencia o capital, posições e PnL realizado.
        Atenção: Este método assume que a "margem" para abrir uma posição
        já foi descontada do `cash` antes, ou será gerenciada aqui.
        Para futuros, `cash` é o saldo disponível, e a margem é alocada.
        """
        symbol = trade.symbol
        executed_quantity_abs = trade.quantity  # Quantidade absoluta do trade
        executed_price = trade.executed_price
        fee = trade.fee

        logger.debug(
            f"💰 [PORTFOLIO] Processando trade para atualização de portfólio: {trade.side.value} {executed_quantity_abs:.4f} {symbol} @ ${executed_price:.2f} (Fee: ${fee:.4f})."
        )

        # Ajusta o cash pelas taxas
        self.cash -= fee
        logger.debug(f"💰 [PORTFOLIO] Cash após taxa: ${self.cash:,.2f}.")

        current_position = self.positions.get(symbol)

        # Determine o lado do trade em relação à posição
        trade_is_long = trade.side == OrderSide.BUY
        trade_is_short = trade.side == OrderSide.SELL

        # Cenário 1: Não há posição existente ou a posição é na direção oposta (reversão)
        if current_position is None or (
            np.sign(current_position.quantity) != (1 if trade_is_long else -1)
            and abs(current_position.quantity) > 1e-9
        ):
            # Se for uma reversão (ex: SHORT para BUY), primeiro fechar a posição existente (implícito)
            # e depois abrir a nova. O PnL realizado é calculado aqui para a parte fechada.

            pnl_realized_trade = 0.0
            margin_released = 0.0
            notional_value_reduced = 0.0

            if current_position is not None:
                # Quantidade que pode ser "fechada" pela contra-ordem
                qty_to_close_by_trade = min(
                    abs(current_position.quantity), executed_quantity_abs
                )

                if (
                    current_position.quantity > 0 and trade_is_short
                ):  # Fechando LONG com SELL
                    pnl_realized_trade = (
                        executed_price - current_position.entry_price
                    ) * qty_to_close_by_trade
                elif (
                    current_position.quantity < 0 and trade_is_long
                ):  # Fechando SHORT com BUY
                    pnl_realized_trade = (
                        current_position.entry_price - executed_price
                    ) * qty_to_close_by_trade

                self.cash += pnl_realized_trade  # Adiciona PnL realizado ao caixa
                logger.debug(
                    f"💰 [PORTFOLIO] PnL Realizado na reversão de {qty_to_close_by_trade:.4f} {symbol}: ${pnl_realized_trade:,.2f}. Cash atualizado: ${self.cash:,.2f}."
                )

                # Ajusta a margem e o valor nocional da posição que está sendo fechada/reduzida
                if qty_to_close_by_trade > 0:
                    # Margem por unidade * quantidade fechada
                    # Precisamos da alavancagem média da posição original para calcular a margem por unidade
                    # Simplificação: assume que a margem é uma % do valor nocional.
                    # Vamos usar a lógica de que a `margin_used` é atualizada diretamente no `RiskManager`
                    # para evitar complexidade duplicada de alavancagem aqui.
                    # Por enquanto, o PortfolioOptimizer apenas rastreia o PnL e a quantidade.
                    pass  # Margem será tratada no RiskManager e no loop principal

            # Atualiza a posição com a nova quantidade e preço de entrada
            new_quantity = (
                current_position.quantity
                + (executed_quantity_abs if trade_is_long else -executed_quantity_abs)
                if current_position
                else (
                    executed_quantity_abs if trade_is_long else -executed_quantity_abs
                )
            )

            if abs(new_quantity) < 1e-9:  # Posição completamente fechada
                if symbol in self.positions:
                    del self.positions[symbol]
                logger.info(
                    f"✅ [PORTFOLIO] Posicao em {symbol} completamente fechada."
                )
            else:
                # [BUG FIX] Distinguir entre reversao total (nova direcao) e fechamento parcial:
                # - Reversao total (sinal oposto, new_quantity muda de sinal): usar executed_price como novo entry
                # - Fechamento parcial (mesma direcao, new_quantity reducao): manter entry_price original
                is_reversal = (
                    current_position is not None
                    and abs(new_quantity) > 1e-9
                    and (new_quantity > 0) != (current_position.quantity > 0)
                )
                if current_position is None or is_reversal:
                    new_entry = executed_price  # Nova posicao na direcao oposta
                else:
                    new_entry = (
                        current_position.entry_price
                    )  # Fechamento parcial: manter entry original
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=new_quantity,
                    entry_price=new_entry,
                    timestamp=trade.timestamp,
                )
                logger.debug(
                    f"📈 [PORTFOLIO] Posicao em {symbol}: {new_quantity:.4f} @ ${new_entry:.2f} ({'reversao' if is_reversal else 'fechamento parcial'})."
                )

        # Cenário 2: Aumentando uma posição existente na mesma direção
        elif (current_position.quantity > 0 and trade_is_long) or (
            current_position.quantity < 0 and trade_is_short
        ):
            # Adiciona ao PnL Realizado do trade se o trade for de fechamento de posição oposta

            # Recalcula o preço médio ponderado da posição
            old_notional_value = (
                current_position.quantity * current_position.entry_price
            )
            new_notional_value = old_notional_value + (
                executed_quantity_abs * executed_price
                if trade_is_long
                else -executed_quantity_abs * executed_price
            )

            new_quantity = current_position.quantity + (
                executed_quantity_abs if trade_is_long else -executed_quantity_abs
            )
            new_entry_price = new_notional_value / (
                new_quantity + np.finfo(float).eps
            )  # Evita divisão por zero

            self.positions[symbol] = replace(
                current_position,
                quantity=new_quantity,
                entry_price=new_entry_price,
                timestamp=trade.timestamp,
            )
            logger.debug(
                f"📊 [PORTFOLIO] Posição em {symbol} AUMENTADA. Nova Qtd: {new_quantity:.4f}, Novo Entry: ${new_entry_price:.2f}."
            )

        # Cenário 3: Fechando parcialmente uma posição existente na mesma direção
        # Este cenário é coberto pelo Cenário 1 se a quantidade for menor e houver reversão de lado,
        # ou se o trade.action for explicitamente CLOSE.
        # No seu sistema, trades são BUY/SELL. Se BUY quando LONG, aumenta. Se SELL quando LONG, fecha.
        # Então, o Cenário 1 já lida com o fechamento.

        # Atualiza o preço de mercado mais recente
        self.last_prices[symbol] = executed_price

        # A margem e o valor nocional serão recalculados na próxima atualização do RiskManager,
        # pois dependem da alavancagem que pode ser dinâmica.

        logger.info(
            f"💰 [PORTFOLIO] Estado do Portfólio atualizado após trade de {trade.symbol}. Cash: ${self.cash:,.2f}, Posições Ativas: {len(self.positions)}."
        )
        # Atualiza o valor total do portfólio no histórico imediatamente
        self.update_portfolio_value({symbol: executed_price})

    def update_portfolio_value(
        self, current_prices: Dict[str, float], timestamp: Optional[datetime] = None
    ):
        """
        Atualiza o valor total do portfólio e o PnL não realizado das posições abertas.
        Esta função é chamada periodicamente, não apenas ao executar trades.
        Args:
            current_prices (Dict[str, float]): Dicionário {símbolo: preço_atual}.
            timestamp (Optional[datetime]): Timestamp do evento. Se None, usa datetime.now(utc).
        """
        self.last_prices.update(current_prices)  # Atualiza os últimos preços conhecidos

        total_unrealized_pnl = 0.0
        current_notional_value = 0.0  # Valor nocional total das posições abertas

        for symbol, position in self.positions.items():
            if symbol in self.last_prices:
                current_market_price = self.last_prices[symbol]

                # Calcula PnL não realizado da posição
                pnl = (current_market_price - position.entry_price) * position.quantity
                total_unrealized_pnl += pnl

                # Atualiza o objeto Position com o PnL não realizado
                self.positions[symbol] = replace(position, unrealized_pnl=pnl)

                # Soma o valor nocional da posição (quantidade * preço de mercado atual)
                current_notional_value += abs(position.quantity) * current_market_price
            else:
                logger.warning(
                    f"⚠️ [PORTFOLIO] Preço atual para {symbol} não disponível para cálculo de PnL não realizado."
                )
                # Usa o PnL não realizado anterior ou 0.0 se não tiver preço atual

        self.total_notional_value = current_notional_value

        # [FIX 9] Recalcula margin_used a partir das posições abertas.
        # Necessário em paper trading onde a exchange não sincroniza esse valor.
        # margin = notional / leverage. Se a posição não tem leverage definida, assume 1x.
        computed_margin = 0.0
        for symbol, position in self.positions.items():
            price = self.last_prices.get(symbol, position.entry_price)
            notional = abs(position.quantity) * price
            lev = float(getattr(position, "leverage", None) or 1.0)
            lev = max(lev, 1.0)  # nunca divide por menos de 1
            computed_margin += notional / lev
        # Só sobrescreve se não houve sync recente da exchange (live mode sincroniza externamente)
        if computed_margin > 0.0 or self.margin_used == 0.0:
            self.margin_used = computed_margin

        # O valor total do portfólio é o cash + PnL não realizado
        # Assumindo que 'cash' já reflete as margens e PnLs realizados
        # [FIX C1] Deduz funding rate para posições (perpétuos)
        # Funding é pago por TODAS as posições abertas, independente de alavancagem.
        # Binance Perp: funding a cada 8h. Cobramos a cada 8h de wall-clock.
        _now_ts = timestamp if timestamp else datetime.now(timezone.utc)
        if not hasattr(self, "_last_funding_timestamp"):
            self._last_funding_timestamp = _now_ts
        _funding_elapsed_h = (
            _now_ts - self._last_funding_timestamp
        ).total_seconds() / 3600.0
        if _funding_elapsed_h >= 8.0:
            for symbol, position in self.positions.items():
                price = self.last_prices.get(symbol, position.entry_price)
                qty = float(position.quantity)
                notional = qty * price
                funding_8h = abs(notional) * 0.0003 * (8.0 / 24.0)
                self.cash += funding_8h if qty < 0 else -funding_8h
            self._last_funding_timestamp = _now_ts
        total_value = self.cash + total_unrealized_pnl

        # Sincronização de Timezone para evitar erro de Comparação
        ts = timestamp if timestamp else datetime.now(timezone.utc)

        if not self.portfolio_value_history.empty:
            last_ts = self.portfolio_value_history.index[-1]
            try:
                # Se histórico é aware e novo é naive -> torna aware
                if last_ts.tzinfo is not None and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                # Se histórico é naive e novo é aware -> torna naive
                elif last_ts.tzinfo is None and ts.tzinfo is not None:
                    ts = ts.replace(tzinfo=None)
            except:
                # Fallback genérico se tzinfo check falhar
                ts = datetime.now(timezone.utc)

        # Adiciona ao histórico do portfólio
        new_entry = pd.Series([total_value], index=[ts])
        self.portfolio_value_history = pd.concat(
            [self.portfolio_value_history, new_entry]
        ).sort_index()

        logger.debug(
            f"💰 [PORTFOLIO] Valor do Portfólio atualizado: ${total_value:,.2f}. PnL Não Realizado: ${total_unrealized_pnl:,.2f}."
        )

    def get_total_value(self) -> float:
        """Retorna o valor total atual do portfólio (capital + PnL flutuante)."""
        if not self.portfolio_value_history.empty:
            return self.portfolio_value_history.iloc[-1]
        return self.initial_capital

    def get_current_price(self, symbol: str) -> float:
        """Retorna o último preço conhecido para um símbolo."""
        return self.last_prices.get(symbol, 0.0)

    def get_detailed_status(self) -> Dict[str, Any]:
        """
        Retorna um dicionário detalhado do status atual do portfólio,
        incluindo informações sobre margem e alavancagem.
        """
        positions_details = []
        for symbol, pos in self.positions.items():
            # Para cada posição, podemos querer mostrar a alavancagem individual
            # Se a alavancagem não for armazenada na posição, teremos que inferir
            # ou obtê-la do RiskManager se ele rastrear por posição.
            # Por simplicidade aqui, usamos o valor nocional.
            current_market_price = self.last_prices.get(symbol, pos.entry_price)
            position_notional_value = abs(pos.quantity) * current_market_price

            # Margem usada para esta posição (assumindo que seja Notional / Leverage da posição)
            # Como a Position dataclass não tem 'leverage', assumimos que o RiskManager calcula.
            # Para um sistema completo, a Position dataclass deveria ter a leverage associada.
            # Aqui, para simulação, vamos estimar a margem por posição se tivermos a alavancagem total.

            positions_details.append(
                {
                    "symbol": symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "timestamp": pos.timestamp.isoformat(),
                    "current_market_price": current_market_price,
                    "notional_value": position_notional_value,
                    "leverage": getattr(pos, "leverage", 1),
                }
            )

        # Calcular a margem utilizada total e alavancagem total
        # Estes são os campos que o RiskManager deve preencher/manter atualizados.
        # Aqui, o Portfolio apenas os calcula se as posições tiverem um "leverage" associado.
        # Por simplicidade, vamos estimar a alavancagem total aqui com base no total_notional_value
        # e o capital total (get_total_value()).

        # Assumindo que 'capital_for_leverage_calc' é o valor do portfólio que pode ser alavancado
        # (geralmente o patrimônio líquido total do portfólio).
        capital_for_leverage_calc = (
            self.get_total_value()
        )  # Ou self.initial_capital se a alavancagem for sobre isso
        total_leverage_ratio = self.total_notional_value / (
            capital_for_leverage_calc + np.finfo(float).eps
        )

        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,  # Saldo disponível
            "total_value": self.get_total_value(),  # Patrimônio líquido total (equity)
            "unrealized_pnl": sum(p.unrealized_pnl for p in self.positions.values()),
            "positions_count": len(self.positions),
            "margin_used": self.margin_used,  # Margem ativamente usada em posições abertas
            "total_notional_value": self.total_notional_value,  # Valor nocional de todas as posições abertas
            "total_leverage_ratio": total_leverage_ratio,  # Alavancagem total do portfólio
            "positions": positions_details,
        }

    def reconcile_positions(self, exchange_positions: Dict[str, Dict]):
        """
        [PRIORIDADE 1] Reconciliação Total de Posições.
        Sobrescreve as posições locais com os dados reais vindos da exchange.
        Útil para corrigir desvios após quedas de conexão ou ordens externas.
        """
        if not isinstance(exchange_positions, dict):
            return

        logger.info(
            f"🔄 [PORTFOLIO] Iniciando reconciliação com {len(exchange_positions)} posições reais..."
        )

        # Sincroniza posições
        new_positions = {}
        for symbol, data in exchange_positions.items():
            new_positions[symbol] = Position(
                symbol=symbol,
                quantity=data["quantity"],
                entry_price=data["entry_price"],
                timestamp=datetime.now(timezone.utc),
                unrealized_pnl=data.get("unrealized_pnl", 0.0),
                leverage=data.get("leverage", 1),
            )
            # Atualiza o preço atual conhecido
            if "mark_price" in data:
                self.last_prices[symbol] = data["mark_price"]

        # Log de mudanças
        added = set(new_positions.keys()) - set(self.positions.keys())
        removed = set(self.positions.keys()) - set(new_positions.keys())
        if added:
            logger.info(f"➕ [RECONCILIATION] Posições detectadas na exchange: {added}")
        if removed:
            logger.info(
                f"➖ [RECONCILIATION] Posições fechadas externamente removidas: {removed}"
            )

        self.positions = new_positions
        logger.info("✅ [PORTFOLIO] Reconciliação concluída com sucesso.")

    def load_state(self, state: Dict[str, Any]):
        """Carrega o estado do portfólio a partir de um dicionário."""
        self.initial_capital = state.get("initial_capital", self.config.INITIAL_CAPITAL)
        self.cash = state.get("cash", self.initial_capital)
        self.margin_used = state.get("margin_used", 0.0)
        self.total_notional_value = state.get("total_notional_value", 0.0)
        self.positions.clear()
        for pos_data in state.get("positions", []):
            self.positions[pos_data["symbol"]] = Position(
                symbol=pos_data["symbol"],
                quantity=pos_data["quantity"],
                entry_price=pos_data["entry_price"],
                timestamp=datetime.fromisoformat(pos_data["timestamp"]),
                unrealized_pnl=pos_data.get(
                    "unrealized_pnl", 0.0
                ),  # Assume 0.0 se não estiver presente
                leverage=pos_data.get("leverage", 1),
            )
        # Recarrega o histórico de valor do portfólio se salvo
        if (
            "portfolio_value_history" in state
            and state["portfolio_value_history"] is not None
        ):
            # Assume que o histórico é salvo como Dict[isoformat_str, float] ou List[Dict]
            if isinstance(state["portfolio_value_history"], dict):
                # Converte chaves de string ISO para datetime e valores para float
                history_dict = {
                    datetime.fromisoformat(k): v
                    for k, v in state["portfolio_value_history"].items()
                }
                self.portfolio_value_history = pd.Series(history_dict).sort_index()
            elif isinstance(
                state["portfolio_value_history"], list
            ):  # Se for lista de {'timestamp', 'equity'}
                timestamps = [
                    datetime.fromisoformat(item["timestamp"])
                    for item in state["portfolio_value_history"]
                ]
                equities = [item["equity"] for item in state["portfolio_value_history"]]
                self.portfolio_value_history = pd.Series(
                    equities, index=timestamps
                ).sort_index()
            logger.info(
                f"✅ [PORTFOLIO] Histórico do portfólio restaurado com {len(self.portfolio_value_history)} entradas."
            )
        else:
            logger.info(
                "ℹ️ [PORTFOLIO] Histórico do portfólio não encontrado no estado. Iniciando novo histórico."
            )
            self.portfolio_value_history = pd.Series(dtype=float)
            self.portfolio_value_history.loc[datetime.utcnow()] = (
                self.get_total_value()
            )  # Inicia com o valor total atual

        logger.info(
            f"✅ [PORTFOLIO] Estado do portfólio carregado: {len(self.positions)} posições, Cash: ${self.cash:,.2f}."
        )
