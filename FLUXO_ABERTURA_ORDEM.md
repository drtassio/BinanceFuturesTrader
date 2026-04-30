# 🚀 Fluxo de Abertura de Ordem - Bot Shield

## Visão Geral do Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SISTEMA DE TRADING                              │
│                                                                           │
│  AIController → ExecutionEngine → BinanceConnector → Binance API        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Fluxo Detalhado (Passo a Passo)

### **1️⃣ Geração do Sinal (AIController)**

**Arquivo:** `trading/ai_controller.py`

```
AIController.generate_signal()
    ↓
    Análise de mercado via Specialists (Trend, Bull, Bear, Ranger)
    ↓
    MoE Router seleciona melhor especialista
    ↓
    Retorna Signal(symbol, action, confidence, leverage, position_size_pct, stop_loss, take_profit)
```

**Dados do Signal:**
```python
Signal(
    symbol='BTCUSDT',           # Par de trading
    action=Action.BUY,          # SELL, BUY ou CLOSE
    confidence=0.85,            # 0-1 nível de confiança
    leverage=2.0,               # Alavancagem (1x a MAX_LEVERAGE)
    position_size_pct=0.05,     # % do capital (5%)
    stop_loss=0.03,             # 3% de stop loss
    take_profit=0.09,           # 9% de take profit
    profit_probability=0.72,    # Prob. de lucro estimada
    explanation={...}           # Explicação da IA
)
```

---

### **2️⃣ Submissão da Ordem (ExecutionEngine)**

**Arquivo:** `trading/execution_engine.py`

```
ExecutionEngine.submit_order(signal: Signal)
    ↓
    ✅ Validações:
       1. Type check → Signal é válido?
       2. Position Limit → Já existe posição aberta para esse símbolo?
       3. Price Check → Preço atual disponível?
       4. Quantity Check → Quantidade > 0?
    ↓
    Calcula Quantidade:
       margin = portfolio_total_value × position_size_pct
       notional_value = margin × leverage
       quantity = notional_value / current_price
    ↓
    Escolhe Estratégia (SMART):
       if notional_value_pct > TWAP_THRESHOLD_PCT:
           strategy = "TWAP"  (para ordens grandes)
       elif confidence > MIN_CONFIDENCE:
           strategy = "MARKET"  (para rápidas)
       else:
           strategy = "MARKET"  (default)
    ↓
    Cria ExecutionOrder (wrapper interno)
    ↓
    Enfileira para processamento async
```

**Classe ExecutionOrder:**
```python
ExecutionOrder:
    id: str                          # ID único (ex: exec_BTCUSDT_1234567_buy)
    signal: Signal                   # Signal original
    strategy: str                    # "MARKET", "LIMIT" ou "TWAP"
    status: OrderStatus              # PENDING → OPEN → FILLED
    total_quantity: float            # Qtd total a executar
    filled_quantity: float           # Qtd já preenchida
    avg_fill_price: float            # Preço médio de execução
    notional_value_at_creation: float # Valor nocional (em USD)
    child_order_ids: List[str]       # IDs das ordens filhas (TWAP)
    trades: List[Trade]              # Preenchimentos
```

---

### **3️⃣ Execução da Ordem**

#### **Modo: LIVE TRADING (Real)**

**Arquivo:** `execution_engine.py:_execute_order()`

```
_execute_order(order: ExecutionOrder)
    ↓
    🌐 Ajusta Alavancagem:
       await connector.set_leverage_for_symbol(symbol, leverage)
    ↓
    Seleciona Estratégia de Execução:
```

---

#### **3.1️⃣ Estratégia MARKET (Execução Rápida)**

```
_execute_market_real(order)
    ↓
    Prepara Parâmetros:
    {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "type": "MARKET",
        "quantity": 0.01,
        "newClientOrderId": "mkt_abc123...",
        "newOrderRespType": "FULL"
    }
    ↓
    📤 Envia Ordem à Binance:
       result = await connector.place_order(params)
    ↓
    📨 Resposta da Binance:
    {
        "orderId": 123456789,
        "clientOrderId": "mkt_abc123...",
        "status": "FILLED",
        "executedQty": 0.01,
        "cumQuote": 42500.50,
        "avgPrice": 42500.50,
        "tradeId": 987654321
    }
    ↓
    ✅ Atualiza ExecutionOrder:
       order.update_with_fill(result)
       order.status = OrderStatus.FILLED
       order.avg_fill_price = 42500.50
       order.trades.append(new_trade)
    ↓
    📊 Atualiza Portfolio:
       portfolio.update_from_trade(new_trade)
    ↓
    🛑 Coloca Proteções (Hard Stop):
       await _place_initial_hard_stop(trade, sl_pct=0.03)
       → Coloca STOP_MARKET @ 41285.49 (3% abaixo)
    ↓
    🎯 Coloca Take Profit:
       await _place_initial_take_profit(trade, tp_pct=0.09)
       → Coloca TP @ 46325.55 (9% acima)
    ↓
    🎪 Coloca Trailing Stop:
       await _place_trailing_stop(symbol, side, qty, 3%)
       → Trailing com callback 3%
```

---

#### **3.2️⃣ Estratégia LIMIT (Melhor Preço)**

```
_execute_limit_real(order)
    ↓
    Calcula Preço Limite:
       current_price = 42500
       if BUY:
           limit_price = 42500 × (1 - 0.5%) = 42287.50
       if SELL:
           limit_price = 42500 × (1 + 0.5%) = 42712.50
    ↓
    📤 Envia Ordem Limitada:
       type = "LIMIT"
       timeInForce = "GTC"  (Good Till Canceled)
    ↓
    ⏱️ Se não preencher em 300s:
       Inicia background task _cancel_limit_if_not_filled()
       Cancela ordem LIMIT
       Re-executa como MARKET (fallback seguro)
    ↓
    Mesmo fluxo de proteções do MARKET
```

---

#### **3.3️⃣ Estratégia TWAP (Ordens Grandes)**

```
_execute_twap_real(order)
    ↓
    Calcula Fatias:
       total_duration = TWAP_DURATION_MINUTES × 60
       desired_slices = total_duration / 15
       num_slices = min(desired_slices, max_slices_allowed)
       slice_quantity = total_quantity / num_slices
    ↓
    💾 Salva Estado (para retomada após restart):
       _save_twap_state()
    ↓
    Loop para cada fatia:
       for i in range(num_slices):
           ↓
           Calcula quantidade da fatia
           ↓
           📤 Envia MARKET order para fatia
           ↓
           ✅ Atualiza ExecutionOrder
           ↓
           📊 Atualiza Portfolio
           ↓
           🛑 Atualiza Proteções:
              Ajusta Hard Stop para nova posição agregada
              Ajusta Trailing Stop
           ↓
           💾 Salva estado
           ↓
           ⏳ Aguarda intervalo (slice_duration)
    ↓
    Após última fatia:
       Coloca proteções finais (se não houver)
       Limpa arquivo de estado TWAP
```

---

### **4️⃣ Monitoramento de Ordens**

**Arquivo:** `execution_engine.py:_monitor_orders_loop()`

```
Loop a cada 10 segundos:
    ↓
    _manage_existing_positions()
    ↓
    Para cada posição aberta:
        ↓
        🔍 Verifica open_orders na exchange
        ↓
        Se NÃO tem stop:
            ⚠️ PERIGO! Posição desprotegida
            ↓
            Cria stop de emergência (fallback)
            ↓
            Implementa Backoff exponencial:
               Se 3+ falhas → tenta a cada 60s
    ↓
    Se tem stop ativo:
        ✅ Reseta contador de falhas
```

---

### **5️⃣ Modo PAPER TRADING (Simulado)**

```
_execute_paper_trade(order)
    ↓
    Simula Execução:
       simulated_price = portfolio.get_current_price(symbol)
       slippage_factor = 1.001 (0.1% slippage)
       executed_price = simulated_price × slippage_factor
    ↓
    Calcula Taxa:
       fee = executed_notional × fee_rate
       (0.001% maker ou 0.002% taker)
    ↓
    Cria Trade Simulado:
       Trade(
           trade_id=f"paper_{order.id}",
           symbol=symbol,
           side=action,
           quantity=total_qty,
           executed_price=executed_price,
           fee=fee_amount,
           timestamp=now
       )
    ↓
    Atualiza Portfolio:
       portfolio.update_from_trade(paper_trade)
    ↓
    Adiciona ao Histórico de Trades:
       system_state["recent_trades"].append(trade_data)
    ↓
    NÃO coloca stop (é simulado)
```

---

## 🔄 Estados da Ordem

```
PENDING  → OPEN  → FILLED
  ↓       ↓         ↓
  └───REJECTED    CANCELED
  └───EXPIRED

Estados possíveis:
├─ PENDING        Recém criada, aguardando processamento
├─ OPEN           Enviada para exchange
├─ PARTIALLY_FILLED  Preenchimento parcial
├─ FILLED         Completamente preenchida
├─ CANCELED       Cancelada pelo usuário
├─ REJECTED       Falha na submissão
└─ EXPIRED        Expirou (LIMIT timeout)
```

---

## 📊 Fluxo de Dados - Detalhado

```
┌─────────────────────────────────────────────────────────────────┐
│  1. AIController gera Signal                                     │
│     └─> Signal(BTCUSDT, BUY, 0.85 conf, 2x, 5%, 3%SL, 9%TP)   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. ExecutionEngine.submit_order(signal)                        │
│     ├─ Valida signal                                            │
│     ├─ Calcula: margin = $10,000 × 5% = $500                  │
│     ├─ Calcula: notional = $500 × 2x = $1,000                 │
│     ├─ Calcula: qty = $1,000 / $42,500 = 0.0235 BTC           │
│     ├─ Escolhe MARKET (confidence alta)                        │
│     └─ Enfileira ExecutionOrder                                │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. ExecutionEngine._execution_loop() processa                  │
│     ├─ Ajusta alavancagem (2x)                                 │
│     └─ Chama _execute_market_real(order)                       │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. BinanceConnector.place_order()                              │
│     ├─ POST /fapi/v1/order                                     │
│     │   {                                                       │
│     │    "symbol": "BTCUSDT",                                 │
│     │    "side": "BUY",                                       │
│     │    "type": "MARKET",                                    │
│     │    "quantity": 0.0235,                                  │
│     │    "newClientOrderId": "mkt_xyz..."                     │
│     │   }                                                      │
│     └─ Retorna resposta da exchange                            │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. Resposta da Binance                                         │
│     {                                                           │
│      "orderId": 999888777,                                     │
│      "status": "FILLED",                                       │
│      "executedQty": 0.0235,                                    │
│      "cumQuote": 1000.00,  (BTC × 42500)                      │
│      "avgPrice": 42500.00                                      │
│     }                                                           │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. ExecutionOrder atualizada                                   │
│     ├─ status = FILLED                                         │
│     ├─ filled_quantity = 0.0235                               │
│     ├─ avg_fill_price = 42500.00                              │
│     ├─ trades.append(Trade(...))                              │
│     └─ Criado objeto Trade                                     │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  7. Portfolio atualizado                                        │
│     ├─ Posição: +0.0235 BTC                                    │
│     ├─ Entry Price: 42500                                      │
│     ├─ Mark Price: 42500                                       │
│     ├─ Unrealized PnL: 0                                       │
│     └─ Margin Used: $500                                       │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  8. Proteções colocadas                                         │
│     ├─ Hard Stop @ 41245 (42500 × 0.97)                       │
│     │   └─ STOP_MARKET SELL 0.0235 BTC @ 41245                │
│     ├─ Take Profit @ 46325 (42500 × 1.09)                     │
│     │   └─ TP MARKET SELL 0.0235 BTC @ 46325                  │
│     └─ Trailing Stop                                           │
│         └─ TRAILING_STOP_MARKET 3% callback                    │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│  9. Sistema aguarda eventos                                     │
│     ├─ Monitor loop: verifica posições a cada 10s             │
│     ├─ WebSocket: monitora tick price em tempo real            │
│     ├─ Se preço toca 46325 → TP executado automaticamente     │
│     ├─ Se preço cai para 41245 → SL executado automaticamente │
│     └─ Se preço sobe 3% e cai → Trailing stop executado       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔐 Validações de Segurança

```
✅ ANTES DE SUBMETER:
├─ Type Check: Signal é válido?
├─ Position Limit: Apenas 1 posição aberta por símbolo
├─ Quantity Check: Quantidade > 0
└─ Price Check: Preço disponível?

✅ DURANTE A EXECUÇÃO:
├─ Timeout: LIMIT orders canceladas após 5 min
├─ Alavancagem: Ajustada para o símbolo antes de enviar
├─ Precisão: Quantidade arredondada para precisão do symbol
└─ Notional Mínimo: Binance exige $50 mín (BTCUSDT)

✅ APÓS A EXECUÇÃO:
├─ Hard Stop: Colocado imediatamente
├─ Take Profit: Colocado imediatamente
├─ Trailing Stop: Colocado imediatamente (ou retomado)
└─ Portfolio: Sincronizado com execução
```

---

## 📈 Exemplo Prático Completo

### **Cenário: BUY BTCUSDT com TWAP**

```
ENTRADA:
├─ AIController gera sinal:
│  ├─ Symbol: BTCUSDT
│  ├─ Action: BUY
│  ├─ Confidence: 0.87
│  ├─ Leverage: 3.0x
│  ├─ Position Size: 10% do capital ($10,000)
│  ├─ Stop Loss: 2%
│  └─ Take Profit: 12%
│
└─ Portfolio: $100,000 total
   └─ Capital para trade: $10,000 × 3x = $30,000 nocional
      └─ BTC a $42,500 = 0.7059 BTC

EXECUÇÃO (TWAP - 4 fatias, intervalo 15s):
├─ Fatia 1: 0.1765 BTC @ 42510 ✅ FILLED
│  └─ Hard Stop @ 41659, TP @ 47660, Trailing 2%
│
├─ Fatia 2: 0.1765 BTC @ 42520 ✅ FILLED
│  └─ Proteções atualizadas para 0.353 BTC total
│
├─ Fatia 3: 0.1765 BTC @ 42515 ✅ FILLED
│  └─ Proteções atualizadas para 0.5295 BTC total
│
└─ Fatia 4: 0.1764 BTC @ 42505 ✅ FILLED
   └─ Posição final: 0.7059 BTC @ avg 42512.5

SAÍDA:
├─ Preço sobe para 47660 → TP executado
│  └─ VENDA 0.7059 BTC @ ~47660
│  └─ Lucro: (47660 - 42512.5) × 0.7059 = ~3,655 USD
│  └─ Retorno: 12.2% (conf. esperado)
│
└─ Ou preço cai para 41659 → SL executado
   └─ VENDA 0.7059 BTC @ ~41659
   └─ Perda: (41659 - 42512.5) × 0.7059 = -603 USD
   └─ Retorno: -2.0% (conf. esperado)
```

---

## 🛠️ Componentes Principais

| Componente | Arquivo | Responsabilidade |
|-----------|---------|------------------|
| **AIController** | `ai_controller.py` | Gera sinais via IA |
| **ExecutionEngine** | `execution_engine.py` | Orquestra execução |
| **BinanceConnector** | `binance_connector.py` | API da Binance |
| **PortfolioOptimizer** | `portfolio.py` | Rastreia posições |
| **RiskManager** | `risk_manager.py` | Valida riscos |
| **ExecutionOrder** | `execution_engine.py` | Ordem interna |
| **Trade** | `trade_schema.py` | Preenchimento |

---

## 🚨 Fluxo de Erro

```
❌ Erro em qualquer etapa:
├─ Signal inválido → REJEITADA na submit_order
├─ Preço indisponível → REJEITADA (price check)
├─ Quantidade zero → REJEITADA (quantity check)
├─ API Binance offline → ERRO, retry automático
├─ Timeout LIMIT → Cancelada + re-executada como MARKET
└─ Falha de leverage → REJEITADA com explicação

Todos os erros são logados com:
├─ Timestamp
├─ Order ID
├─ Detalhes técnicos
└─ Feedback visual na UI
```

---

## 📝 Checklist de Execução

```
□ Signal validado
□ Quantidade calculada
□ Estratégia escolhida
□ Alavancagem ajustada
□ Ordem enviada
□ Resposta recebida
□ Order atualizada
□ Portfolio sincronizado
□ Hard Stop colocado
□ Take Profit colocado
□ Trailing Stop colocado
□ Feedback exibido na UI
```

---

**Última atualização:** 29/04/2026  
**Status:** ✅ Sistema operacional e monitorado
