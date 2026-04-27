# -----------------------------------------------------------------------------
# ARQUIVO: models/trade_schema.py
# -----------------------------------------------------------------------------

"""
Definições de Esquemas de Dados (Data Schemas) para o sistema de trading.

Este módulo define as estruturas de dados fundamentais usadas para representar
sinais de trading, ordens, posições, trades executados e dados de mercado.
A utilização de dataclasses e Enums garante consistência, tipagem forte
e facilidade de serialização/desserialização em todo o sistema.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, List # Adicionado List para `fills` em Order


class OrderSide(str, Enum):
    """Lado da ordem: COMPRA (BUY) ou VENDA (SELL)."""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    """Tipo de ordem: MERCADO (MARKET) ou LIMITE (LIMIT)."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET" # Ordem de stop-market
    STOP_LIMIT = "STOP_LIMIT"   # Ordem de stop-limit
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET" # Ordem de trailing stop
    OCO = "OCO" # One-Cancels-Other (Uma Cancela a Outra)

class OrderStatus(str, Enum):
    """Status de uma ordem na exchange ou no sistema."""
    NEW = "NEW"                 # Ordem recém-criada
    PARTIALLY_FILLED = "PARTIALLY_FILLED" # Parcialmente preenchida
    FILLED = "FILLED"           # Completamente preenchida
    CANCELED = "CANCELED"       # Cancelada pelo usuário/sistema
    PENDING_CANCEL = "PENDING_CANCEL" # Cancelamento pendente
    REJECTED = "REJECTED"       # Rejeitada pela exchange
    EXPIRED = "EXPIRED"         # Expirou (e.g., GTC/FOK/IOC)
    OPEN = "OPEN"               # Ordem aberta (aguardando preenchimento)
    PENDING = "PENDING"         # Ordem interna aguardando envio à exchange

class Action(str, Enum):
    """Ação de trading recomendada pela IA ou a ser executada."""
    BUY = "BUY"      # Comprar (abrir posição LONG ou fechar SHORT)
    SELL = "SELL"    # Vender (abrir posição SHORT ou fechar LONG)
    HOLD = "HOLD"    # Manter a posição atual ou não fazer nada
    CLOSE = "CLOSE"  # Fechar a posição existente (independentemente do lado)
    DELEGATE = "DELEGATE"
@dataclass
class Signal:
    """
    Representa um sinal de trading gerado pela inteligência artificial.
    Este é o output primário do AIController.
    """
    symbol: str             # Símbolo do ativo (e.g., "BTCUSDT")
    action: Action          # Ação recomendada (BUY, SELL, HOLD, CLOSE)
    confidence: float       # Nível de confiança da IA na decisão (0.0 a 1.0)
    
    # Adicionado para dimensionamento de posição e alavancagem
    position_size_pct: float = 0.0 # Tamanho da posição como % do capital a ser usada como MARGEM
    leverage: float = 1.0        # Alavancagem desejada para o trade (e.g., 1.0, 5.0, 10.0)

    stop_loss: Optional[float] = None   # Preço de Stop Loss (se aplicável)
    take_profit: Optional[float] = None # Preço de Take Profit (se aplicável)

    # Adicionado para o Modelo Preditivo de Probabilidade de Lucratividade (P)
    profit_probability: float = 0.5 # Probabilidade de o trade ser lucrativo (0.0 a 1.0)

    # [MELHORIA] Duração estimada do trend em candles/timeframes
    estimated_duration: Optional[float] = None # Duração estimada em candles

    # Campo para explicações e metadados da decisão (para XAI e depuração)
    explanation: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validações básicas após a inicialização
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confiança deve estar entre 0.0 e 1.0.")
        if not (0.0 <= self.position_size_pct <= 1.0):
             # position_size_pct pode ser mais que 1.0 se for o valor nocional em relação ao capital
             # Mas se for porcentagem da MARGEM, então 0.0 a 1.0 é mais adequado.
             # Ajustado para ser uma % de MARGEM, então 0 a 1.
            raise ValueError("position_size_pct deve estar entre 0.0 e 1.0 (representando % da margem).")
        if not (1.0 <= self.leverage): # Alavancagem deve ser no mínimo 1.0
            raise ValueError("Alavancagem deve ser igual ou superior a 1.0.")
        if not (0.0 <= self.profit_probability <= 1.0):
            raise ValueError("profit_probability deve estar entre 0.0 e 1.0.")
                
        # Adicionamos Action.DELEGATE à lista de ações válidas.
        if self.action not in [Action.BUY, Action.SELL, Action.HOLD, Action.CLOSE, Action.DELEGATE]:
            raise ValueError(f"Ação inválida: {self.action}")


@dataclass
class Order:
    """
    Representa uma ordem enviada ou a ser enviada para a exchange.
    Pode ser uma ordem "filha" de uma meta-ordem do ExecutionEngine.
    """
    order_id: str               # ID único da ordem (gerado pelo bot ou pela exchange)
    symbol: str                 # Símbolo do ativo (e.g., "BTCUSDT")
    side: OrderSide             # Lado da ordem (BUY/SELL)
    order_type: OrderType       # Tipo da ordem (MARKET, LIMIT, etc.)
    quantity: float             # Quantidade do ativo a ser negociada
    price: Optional[float] = None # Preço da ordem (para LIMIT, STOP_LIMIT)
    status: OrderStatus = OrderStatus.NEW # Status atual da ordem
    created_at: datetime = field(default_factory=datetime.utcnow) # Timestamp de criação
    client_order_id: Optional[str] = None # Seu ID único para a ordem
    exchange_order_id: Optional[str] = None # ID da ordem na exchange
    
    # Campos para preenchimentos
    filled_quantity: float = 0.0 # Quantidade já preenchida
    avg_fill_price: float = 0.0  # Preço médio de preenchimento
    commission: float = 0.0      # Comissão paga
    commission_asset: Optional[str] = None # Ativo da comissão (e.g., "USDT", "BNB")

    # Lista de preenchimentos parciais (útil para TWAP, VWAP)
    fills: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # Validações adicionais para ordens
        if self.quantity <= 0:
            raise ValueError("Quantidade da ordem deve ser positiva.")
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError(f"Preço é obrigatório para ordens do tipo {self.order_type.value}.")
        if self.price is not None and self.price <= 0:
            raise ValueError("Preço da ordem deve ser positivo.")


@dataclass
class Position:
    """
    Representa uma posição de trading aberta no portfólio.
    """
    symbol: str         # Símbolo do ativo
    quantity: float     # Quantidade do ativo (positiva para LONG, negativa para SHORT)
    entry_price: float  # Preço médio de entrada da posição
    timestamp: datetime # Timestamp da abertura ou última modificação da posição
    
    # O PnL não realizado é calculado dinamicamente pelo PortfolioOptimizer
    unrealized_pnl: float = 0.0 
    
    # Adicionado para exibir a alavancagem configurada na exchange (ex: 75x)
    leverage: int = 1 
    
    # [SAFETY] Campos para monitoramento de risco de liquidação
    liquidation_price: float = 0.0
    mark_price: float = 0.0 

    def __post_init__(self):
        if self.quantity == 0:
            raise ValueError("Quantidade da posição não pode ser zero.")
        if self.entry_price <= 0:
            raise ValueError("Preço de entrada da posição deve ser positivo.")


@dataclass
class Trade:
    """
    Representa um trade individual que foi executado (total ou parcialmente).
    """
    trade_id: str           # ID único do trade (gerado pela exchange ou simulado)
    order_id: str           # ID da ordem que originou este trade
    symbol: str             # Símbolo do ativo
    side: OrderSide         # Lado do trade (BUY/SELL)
    quantity: float         # Quantidade executada neste trade
    executed_price: float   # Preço de execução deste trade
    fee: float              # Taxa paga por este trade
    fee_asset: str          # Ativo no qual a taxa foi paga (e.g., "USDT", "BNB")
    timestamp: datetime     # Timestamp da execução do trade

    # PnL realizado é calculado e adicionado por Backtester ou PortfolioOptimizer
    # quando o trade fecha uma posição. Não é uma propriedade intrínseca do Trade.
    # Adicionar aqui para log de backtest (ver _create_backtest_result_object)
    pnl: Optional[float] = None 

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError("Quantidade do trade deve ser positiva.")
        if self.executed_price <= 0:
            raise ValueError("Preço de execução do trade deve ser positivo.")
        if self.fee < 0:
            raise ValueError("Taxa do trade não pode ser negativa.")