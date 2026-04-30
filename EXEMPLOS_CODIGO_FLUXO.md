# 🔧 Exemplos de Código Real - Fluxo de Abertura de Ordem

## 📌 Índice
1. [AIController → Signal](#1-aicontroller--signal)
2. [ExecutionEngine.submit_order](#2-executionenginesubmit_order)
3. [Cálculo de Quantidade](#3-cálculo-de-quantidade)
4. [Execução MARKET](#4-execução-market)
5. [Execução LIMIT](#5-execução-limit)
6. [Execução TWAP](#6-execução-twap)
7. [Proteções](#7-proteções)
8. [Monitoramento](#8-monitoramento)

---

## 1. AIController → Signal

### Geração do Sinal

**Arquivo:** `trading/ai_controller.py`

```python
class AIController:
    """Centraliza a inteligência do Bot"""
    
    async def generate_signal(self, symbol: str) -> Optional[Signal]:
        """
        Gera um sinal de trading baseado na análise dos especialistas.
        
        Fluxo:
        1. Obtém dados de mercado
        2. Processa features
        3. Consulta especialistas (Mixture of Experts)
        4. MOE Router seleciona o melhor
        5. Retorna Signal
        """
        
        # 1️⃣ Obtém dados de mercado (últimas 4h)
        klines = await self.data_provider.get_klines(
            symbol=symbol,
            interval='4h',
            limit=100
        )
        
        # 2️⃣ Processa features via pipeline
        features = self.feature_pipeline.process(klines)
        # features contém: MA, RSI, MACD, Bollinger Bands, Volume, etc.
        
        # 3️⃣ Consulta cada especialista
        trend_expert_output = self.specialists['trend'].predict(features)
        # Output: {
        #   'action': 'BUY',
        #   'confidence': 0.85,
        #   'expected_return': 0.09,
        #   'reasoning': 'Trend following ascending channel'
        # }
        
        bull_expert_output = self.specialists['bull'].predict(features)
        bear_expert_output = self.specialists['bear'].predict(features)
        ranger_expert_output = self.specialists['ranger'].predict(features)
        
        # 4️⃣ MOE Router seleciona o melhor
        expert_outputs = {
            'trend': trend_expert_output,
            'bull': bull_expert_output,
            'bear': bear_expert_output,
            'ranger': ranger_expert_output
        }
        
        best_expert = self.moe_router.select_expert(
            features=features,
            expert_outputs=expert_outputs
        )
        # best_expert = 'trend' (por exemplo)
        
        selected_output = expert_outputs[best_expert]
        
        # 5️⃣ Cria Signal a partir da seleção
        signal = Signal(
            symbol=symbol,
            action=Action.BUY,  # ou SELL, CLOSE
            confidence=selected_output['confidence'],  # 0.85
            leverage=self.config_ai.DEFAULT_LEVERAGE,  # 2.0
            position_size_pct=self.config_trading.POSITION_SIZE_PCT,  # 0.05
            stop_loss=self.config_ai.DEFAULT_STOP_LOSS_PCT,  # 0.03
            take_profit=self.config_ai.DEFAULT_TAKE_PROFIT_PCT,  # 0.09
            profit_probability=selected_output['expected_return'],  # 0.72
            explanation={
                'expert': best_expert,
                'reasoning': selected_output['reasoning'],
                'features_used': list(features.keys())
            }
        )
        
        logger.info(f"🎯 Signal gerado: {signal.symbol} {signal.action.value} "
                   f"conf={signal.confidence:.2f} lev={signal.leverage:.1f}x")
        
        return signal
    
    async def submit_signal(self, signal: Signal):
        """Submete o signal ao ExecutionEngine"""
        await self.execution_engine.submit_order(signal)
```

**Output Example:**
```
Signal(
    symbol='BTCUSDT',
    action=<Action.BUY: 'BUY'>,
    confidence=0.85,
    leverage=2.0,
    position_size_pct=0.05,
    stop_loss=0.03,
    take_profit=0.09,
    profit_probability=0.72,
    price=42500.00,
    timestamp=2026-04-29T10:30:45.123456Z,
    explanation={
        'expert': 'trend',
        'reasoning': 'Trend following ascending channel',
        'features_used': ['sma_50', 'sma_200', 'rsi', 'macd', 'bb_upper']
    }
)
```

---

## 2. ExecutionEngine.submit_order

### Submissão e Validação

**Arquivo:** `trading/execution_engine.py` (linhas 376-468)

```python
async def submit_order(self, signal: Signal):
    """
    Submete um novo sinal de trading para execução.
    
    Validações:
    1. Type check
    2. Position limit
    3. Price check
    4. Quantity check
    """
    
    # 1️⃣ TYPE CHECK
    if not isinstance(signal, Signal):
        logger.error(f"❌ Tentativa de submeter objeto não-Signal: {signal}")
        return
    
    # 2️⃣ POSITION LIMIT CHECK
    # Verifica se já existe uma posição aberta
    has_position = await self._check_position_limit(signal.symbol)
    
    if has_position:
        # Verifica se é uma ordem de fechamento ou redução
        is_reducing = False
        current_pos = self.portfolio.positions.get(signal.symbol)
        
        if signal.action == Action.CLOSE:
            # É uma ordem de fechamento explícita
            is_reducing = True
            logger.info(f"ℹ️ Sinal CLOSE recebido. Ignorando limite de posição.")
        
        elif current_pos:
            # Verifica se a ação é oposta (reduz posição)
            if current_pos.quantity > 0 and signal.action == Action.SELL:
                # LONG position + SELL signal = redução
                is_reducing = True
            elif current_pos.quantity < 0 and signal.action == Action.BUY:
                # SHORT position + BUY signal = redução
                is_reducing = True
        
        if not is_reducing:
            # Rejeita a ordem
            logger.warning(
                f"⚠️ LIMITE DE POSIÇÃO: Já existe posição aberta para {signal.symbol}"
            )
            self.system_state["command_feedback"] = \
                f"Limite atingido: Já existe posição aberta para {signal.symbol}."
            return
    
    # 3️⃣ PRICE CHECK
    current_price = self.portfolio.get_current_price(signal.symbol)
    
    if current_price <= 0:
        logger.error(
            f"❌ Não foi possível obter preço para {signal.symbol}. "
            f"Ordem REJEITADA."
        )
        # Cria ordem rejeitada
        order_id = f"exec_{signal.symbol}_{int(time.time() * 1000)}_reject"
        exec_order = ExecutionOrder(signal, "REJECTED", order_id)
        exec_order.status = OrderStatus.REJECTED
        self.active_orders[order_id] = exec_order
        return
    
    # 4️⃣ CRIAR ID ÚNICA
    order_id = f"exec_{signal.symbol}_{int(time.time() * 1000)}_{signal.action.value.lower()}"
    
    # 5️⃣ ESCOLHER ESTRATÉGIA (SMART)
    strategy = self._choose_smart_strategy(signal)
    # strategy = "MARKET" ou "LIMIT" ou "TWAP"
    
    # 6️⃣ CRIAR EXECUTION ORDER
    exec_order = ExecutionOrder(signal, strategy, order_id)
    
    # Status agora é PENDING
    logger.info(f"🆕 Nova meta-ordem criada: {order_id} ({strategy})")
    
    # Continua no próximo passo (cálculo de quantidade)
    ...
```

---

## 3. Cálculo de Quantidade

### Margin → Notional → Quantity

**Arquivo:** `trading/execution_engine.py` (linhas 422-468)

```python
async def submit_order(self, signal: Signal):
    """Continuação: Cálculo da quantidade"""
    
    # ... (código anterior: validações)
    
    # 1️⃣ OBTER VALOR TOTAL DO PORTFÓLIO
    portfolio_total_value_usd = self.portfolio.get_total_value()
    # Retorna: 100000.00 (em USD)
    
    # 2️⃣ CALCULAR MARGEM A ALOCAR
    # signal.position_size_pct é a % do capital que será MARGEM
    margin_to_allocate_usd = portfolio_total_value_usd * signal.position_size_pct
    # Cálculo: 100000 × 0.05 = 5000 USD
    
    # 3️⃣ OBTER PREÇO ATUAL
    current_price = self.portfolio.get_current_price(signal.symbol)
    # Retorna: 42500.00 (preço atual do BTC)
    
    if current_price <= 0:
        logger.error(f"❌ Preço inválido para {signal.symbol}")
        exec_order.status = OrderStatus.REJECTED
        return
    
    # 4️⃣ CALCULAR VALOR NOCIONAL
    # Notional Value = Margem × Alavancagem
    notional_trade_value_usd = margin_to_allocate_usd * signal.leverage
    # Cálculo: 5000 × 2.0 = 10000 USD
    
    exec_order.notional_value_at_creation = notional_trade_value_usd
    
    # 5️⃣ CALCULAR QUANTIDADE
    # Quantidade = Valor Nocional / Preço Atual
    quantity = notional_trade_value_usd / current_price
    # Cálculo: 10000 / 42500 = 0.2353 BTC
    
    # 6️⃣ ARREDONDAR PARA PRECISÃO DO SYMBOL
    symbol_info = await self._get_cached_symbol_info(signal.symbol)
    qty_precision = int(symbol_info['quantityPrecision'])  # 4 casas
    # quantityPrecision para BTCUSDT = 4
    
    exec_order.total_quantity = round(quantity, qty_precision)
    # Resultado: 0.2353 BTC (4 casas decimais)
    
    # 7️⃣ VALIDAÇÃO FINAL
    if exec_order.total_quantity <= 0:
        logger.warning(
            f"⚠️ Quantidade calculada é zero ou negativa. "
            f"Margem: ${margin_to_allocate_usd:,.2f}, "
            f"Nocional: ${notional_trade_value_usd:,.2f}, "
            f"Preço: {current_price}"
        )
        exec_order.status = OrderStatus.REJECTED
        return
    
    # 8️⃣ ENFILEIRARAR PARA PROCESSAMENTO
    self.active_orders[order_id] = exec_order
    await self.order_queue.put(exec_order)
    
    # 9️⃣ FEEDBACK
    logger.info(
        f"📦 Meta-ordem enfileirada: {order_id} "
        f"({signal.action.value} {exec_order.total_quantity:.4f} {signal.symbol}) "
        f"Estratégia: {strategy}, "
        f"Alavancagem: {signal.leverage:.2f}x, "
        f"Nocional: ${notional_trade_value_usd:,.2f}, "
        f"Margem: ${margin_to_allocate_usd:,.2f}"
    )
    
    self.system_state["command_feedback"] = \
        f"Ordem enfileirada: {exec_order.total_quantity:.4f} {signal.symbol} @ Nocional ${notional_trade_value_usd:,.2f}"
```

**Exemplo Numérico:**
```
Portfolio Value:     $100,000.00
Position Size %:     5%
├─ Margem Alocada:   $5,000.00

Leverage:            2.0x
├─ Valor Nocional:   $10,000.00

Preço BTC:           $42,500.00
├─ Quantidade:       0.2353 BTC

Precisão:            4 casas
├─ Final Qty:        0.2353 BTC
```

---

## 4. Execução MARKET

### Ordem de Mercado em Tempo Real

**Arquivo:** `trading/execution_engine.py` (linhas 737-855)

```python
async def _execute_market_real(self, order: ExecutionOrder):
    """
    Executa uma ordem a mercado real na Binance Futures.
    
    Fluxo:
    1. Validação
    2. Preparação de parâmetros
    3. Envio à Binance
    4. Recepção e processamento
    5. Atualização de portfolio
    6. Colocação de proteções
    """
    
    # 1️⃣ VALIDAÇÃO
    if order.total_quantity <= 0:
        logger.warning(f"⚠️ Quantidade zero. Rejeitando.")
        order.status = OrderStatus.REJECTED
        return
    
    # 2️⃣ PREPARAÇÃO
    client_order_id = f"mkt_{uuid.uuid4().hex}"
    order.child_order_ids.append(client_order_id)
    
    # Obter precision dinâmica
    symbol_info = await self._get_cached_symbol_info(order.signal.symbol)
    qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
    
    # 3️⃣ PARÂMETROS DA ORDEM
    params = {
        "symbol": order.signal.symbol,           # "BTCUSDT"
        "side": order.signal.action.value,       # "BUY"
        "type": OrderType.MARKET.value,          # "MARKET"
        "quantity": round(order.total_quantity, qty_precision),  # 0.2353
        "newClientOrderId": client_order_id,     # "mkt_a1b2c3d4e5f6..."
        "newOrderRespType": "FULL",              # Response completa
    }
    
    logger.info(
        f"🌐 Enviando ordem a mercado: {client_order_id} para "
        f"{order.signal.symbol} (Qtd: {params['quantity']}, "
        f"Alavancagem: {order.signal.leverage:.2f}x)"
    )
    
    # 4️⃣ ENVIO À BINANCE
    result = await self.connector.place_order(params)
    
    if result:
        logger.info(
            f"✅ Ordem {result.get('orderId')} ({result.get('status')}) "
            f"para {result.get('symbol')} enviada com sucesso."
        )
        
        # 5️⃣ ATUALIZAR EXECUTION ORDER
        new_trade = order.update_with_fill(result)
        # update_with_fill():
        # - Processa result da Binance
        # - Cria objeto Trade
        # - Atualiza status
        # - Retorna Trade criado
        
        # 6️⃣ LOG & CALLBACK
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
            
            # Adiciona ao estado da UI
            self.system_state["recent_trades"].append(trade_log_data)
            
            # Salva em arquivo
            self.trade_log_callback(trade_log_data)
        
        # 7️⃣ POLLING SE NÃO PREENCHEU IMEDIATAMENTE
        if order.status != OrderStatus.FILLED:
            logger.info(
                f"⏳ Ordem {result.get('orderId')} retornou "
                f"{order.status.value}. Iniciando polling..."
            )
            
            for i in range(5):  # Tenta por 5 segundos
                await asyncio.sleep(1)
                updated_order_info = await self.connector.get_order(
                    order.signal.symbol,
                    order_id=result.get('orderId')
                )
                
                if updated_order_info:
                    current_status = updated_order_info.get('status')
                    
                    if current_status == 'FILLED':
                        logger.info(
                            f"✅ Polling: Ordem {result.get('orderId')} "
                            f"agora está FILLED!"
                        )
                        new_trade = order.update_with_fill(updated_order_info)
                        break
                    elif current_status in ['CANCELED', 'REJECTED']:
                        logger.warning(
                            f"⚠️ Polling: Ordem foi {current_status}"
                        )
                        order.status = OrderStatus(current_status)
                        break
        
        # 8️⃣ ATUALIZAR PORTFOLIO
        if order.status == OrderStatus.FILLED and new_trade:
            self.portfolio.update_from_trade(new_trade)
            logger.debug(
                f"✅ Portfolio atualizado: {new_trade.side.value} "
                f"{new_trade.quantity:.4f} @ {new_trade.executed_price:.2f}"
            )
            
            # 9️⃣ COLOCAR PROTEÇÕES
            sl_pct = (order.signal.stop_loss 
                     if order.signal.stop_loss > 0 
                     else self.config.DEFAULT_STOP_LOSS_PCT)
            
            if sl_pct > 0:
                await self._place_initial_hard_stop(new_trade, sl_pct)
            
            tp_pct = (order.signal.take_profit 
                     if order.signal.take_profit > 0 
                     else self.config.DEFAULT_TAKE_PROFIT_PCT)
            
            if tp_pct > 0:
                await self._place_initial_take_profit(new_trade, tp_pct)
            
            # Trailing Stop
            await self._place_trailing_stop(
                symbol=new_trade.symbol,
                side=new_trade.side,
                quantity=abs(new_trade.quantity),
                stop_loss_pct=sl_pct
            )
            
            logger.info(f"✅ Proteções colocadas para {order.id}")
    else:
        logger.error(f"❌ Falha ao enviar ordem para {order.id}")
        order.status = OrderStatus.REJECTED
```

**Resposta da Binance - Exemplo:**
```json
{
  "orderId": 123456789,
  "symbol": "BTCUSDT",
  "status": "FILLED",
  "side": "BUY",
  "type": "MARKET",
  "clientOrderId": "mkt_a1b2c3d4e5f6g7h8i9j0",
  "transactTime": 1704060045567,
  "price": 0,
  "origQty": "0.2353",
  "executedQty": "0.2353",
  "cummulativeQuoteQty": "10000.4525",
  "status": "FILLED",
  "timeInForce": "IOC",
  "type": "MARKET",
  "side": "BUY",
  "stopPrice": "0",
  "icebergQty": "0",
  "time": 1704060045567,
  "updateTime": 1704060045567,
  "isWorking": true,
  "origQuoteOrderQty": "10000",
  "selfTradePreventionMode": "NONE",
  "avgPrice": "42525.00",
  "fills": [
    {
      "price": "42525.00",
      "qty": "0.2353",
      "commission": "0.00471",
      "commissionAsset": "BTC",
      "tradeId": 987654321
    }
  ]
}
```

---

## 5. Execução LIMIT

### Ordem Limitada com Timeout

**Arquivo:** `trading/execution_engine.py` (linhas 857-968)

```python
async def _execute_limit_real(self, order: ExecutionOrder):
    """
    Executa uma ordem limitada na Binance Futures.
    
    Se não preencher em 300s, cancela e re-executa como MARKET.
    """
    
    # 1️⃣ OBTER PREÇO ATUAL
    ticker = await self.connector.get_ticker_price(order.signal.symbol)
    current_market_price = float(ticker['price']) if ticker else 0.0
    
    if current_market_price <= 0:
        logger.error(f"❌ Preço de mercado inválido. Rejeitando ordem limite.")
        order.status = OrderStatus.REJECTED
        return
    
    # 2️⃣ CALCULAR PREÇO LIMITE
    # Estratégia: tentar comprar/vender 0.5% melhor que o preço de mercado
    limit_price = 0.0
    
    if order.signal.action == OrderSide.BUY:
        # Tentar comprar UM POUCO ABAIXO do preço de mercado
        limit_price = current_market_price * (1.0 - self.config.SLIPPAGE_TOLERANCE_PCT * 0.1)
        # Cálculo: 42500 × (1 - 0.005 × 0.1) = 42500 × 0.9995 = 42487.50
    else:
        # Tentar vender UM POUCO ACIMA do preço de mercado
        limit_price = current_market_price * (1.0 + self.config.SLIPPAGE_TOLERANCE_PCT * 0.1)
    
    # 3️⃣ PREPARAR PARÂMETROS
    symbol_info = await self._get_cached_symbol_info(order.signal.symbol)
    qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
    price_precision = int(symbol_info['pricePrecision']) if symbol_info else 2
    
    params = {
        "symbol": order.signal.symbol,
        "side": order.signal.action.value,
        "type": OrderType.LIMIT.value,
        "quantity": round(order.total_quantity, qty_precision),
        "price": round(limit_price, price_precision),
        "timeInForce": "GTC",  # Good Till Canceled
        "newClientOrderId": f"lim_{uuid.uuid4().hex}",
        "newOrderRespType": "FULL",
    }
    
    logger.info(
        f"🌐 Enviando ordem limite: {order.signal.symbol} "
        f"(Preço: {params['price']})"
    )
    
    # 4️⃣ ENVIAR ORDEM
    result = await self.connector.place_order(params)
    
    if result:
        # 5️⃣ ATUALIZAR
        new_trade = order.update_with_fill(result)
        
        if order.status == OrderStatus.FILLED:
            logger.info(f"✅ Ordem limite PREENCHIDA IMEDIATAMENTE")
            
            if new_trade:
                self.portfolio.update_from_trade(new_trade)
                # Colocar proteções (mesmo fluxo do MARKET)
                ...
        
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            logger.info(f"⏳ Ordem limite PARCIALMENTE PREENCHIDA")
            # Monitor loop cobre o fill restante
        
        else:
            # 6️⃣ TIMEOUT AUTOMÁTICO
            exchange_oid = str(result.get('orderId', ''))
            timeout_s = getattr(self.config, 'LIMIT_ORDER_TIMEOUT_SECONDS', 300)
            
            logger.info(
                f"🅿️ Ordem limite COLOCADA. "
                f"Timeout automático em {timeout_s}s se não preencher."
            )
            
            # Inicia background task para cancelar se não preencher
            asyncio.create_task(
                self._cancel_limit_if_not_filled(order, exchange_oid, timeout_s),
                name=f"limit_timeout_{order.id}"
            )
```

### _cancel_limit_if_not_filled - Background Task

```python
async def _cancel_limit_if_not_filled(
    self, 
    order: 'ExecutionOrder', 
    exchange_order_id: str,
    timeout_seconds: int = 300
):
    """
    Background task que monitora uma LIMIT order.
    Se não preencher em timeout, cancela e re-executa como MARKET.
    """
    
    # 1️⃣ AGUARDA TIMEOUT
    await asyncio.sleep(timeout_seconds)
    # Aguarda 5 minutos (300 segundos)
    
    # 2️⃣ VERIFICA SE JÁ FOI RESOLVIDA
    if order.status == OrderStatus.FILLED or order.status == OrderStatus.CANCELED:
        return  # Já resolvida, nada a fazer
    
    try:
        # 3️⃣ CANCELA ORDEM NA EXCHANGE
        logger.warning(
            f"⏱️ Ordem limite {exchange_order_id} não preencheu em "
            f"{timeout_seconds}s. Cancelando e re-executando como MARKET..."
        )
        
        cancel_result = await self.connector.cancel_order(
            order.signal.symbol,
            exchange_order_id
        )
        
        if cancel_result:
            logger.info(f"✅ Ordem {exchange_order_id} cancelada com sucesso.")
        
        # 4️⃣ CONSULTA STATUS FINAL
        final_status = await self.connector.get_order(
            order.signal.symbol,
            order_id=exchange_order_id
        )
        
        if final_status and final_status.get('status') == 'FILLED':
            logger.info(f"ℹ️ Ordem já estava FILLED. Não re-executa.")
            order.update_with_fill(final_status)
            return
        
        # 5️⃣ RE-EXECUTA COMO MARKET (Fallback Seguro)
        order.status = OrderStatus.PENDING
        order.child_order_ids.clear()
        
        logger.info(f"🔄 Re-executando {order.id} como MARKET...")
        await self._execute_market_real(order)
        
    except asyncio.CancelledError:
        pass  # Bot desligando
    except Exception as e:
        logger.error(f"❌ Erro ao cancelar/re-executar: {e}", exc_info=True)
```

---

## 6. Execução TWAP

### Time-Weighted Average Price para Ordens Grandes

**Arquivo:** `trading/execution_engine.py` (linhas 970-1180)

```python
async def _execute_twap_real(self, order: ExecutionOrder):
    """
    Executa ordem TWAP: divide em múltiplas fatias.
    
    Exemplo: 0.2353 BTC em 4 fatias a cada 15 segundos
    ├─ Fatia 1: 0.0588 BTC @ T+0s
    ├─ Fatia 2: 0.0588 BTC @ T+15s
    ├─ Fatia 3: 0.0588 BTC @ T+30s
    └─ Fatia 4: 0.0589 BTC @ T+45s
    
    Benefícios:
    - Minimiza impacto no mercado
    - Melhor preço médio
    - Reduz slippage
    """
    
    # 1️⃣ VALIDAÇÃO
    if order.total_quantity <= 0:
        logger.warning(f"⚠️ Quantidade TWAP zero. Rejeitando.")
        order.status = OrderStatus.REJECTED
        return
    
    # 2️⃣ CALCULAR NÚMERO DE FATIAS
    total_duration_seconds = self.config.TWAP_DURATION_MINUTES * 60
    # Duração: 15 minutos × 60 = 900 segundos
    
    symbol_info = await self._get_cached_symbol_info(order.signal.symbol)
    qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
    min_qty = 10**(-qty_precision)
    
    # Número de fatias desejado (1 fatia a cada 15s)
    desired_slices = max(2, int(total_duration_seconds / 15))
    # Cálculo: 900 / 15 = 60 fatias
    
    # Obtém preço atual para calcular notional
    current_price = self.portfolio.get_current_price(order.signal.symbol)
    min_notional_binance = 50.5  # $50 mínimo por ordem
    total_notional = order.total_quantity * current_price
    
    # Número máximo de fatias baseado no notional mínimo
    max_slices_allowed = max(1, int(total_notional / min_notional_binance))
    # Cálculo: 10000 / 50.5 = 198 fatias possíveis (limitado pelo min $50)
    
    # Número final de fatias
    num_slices = min(desired_slices, max_slices_allowed)
    # Resultado: min(60, 198) = 60 fatias
    
    # Cálculo de quantidade por fatia
    slice_quantity_base = order.total_quantity / num_slices
    # Cálculo: 0.2353 / 60 = 0.00392 BTC por fatia
    
    slice_interval_seconds = total_duration_seconds / num_slices
    # Cálculo: 900 / 60 = 15 segundos entre fatias
    
    logger.info(
        f"[TWAP] {order.id}: {num_slices} fatias, "
        f"Total: ${total_notional:.2f}, "
        f"Fatia: ~${(total_notional/num_slices):.2f}, "
        f"Intervalo: {slice_interval_seconds:.1f}s"
    )
    
    # 3️⃣ SALVAR ESTADO (para retomada após restart)
    sl_pct = (order.signal.stop_loss 
             if order.signal.stop_loss > 0 
             else self.config.DEFAULT_STOP_LOSS_PCT)
    tp_pct = (order.signal.take_profit 
             if order.signal.take_profit > 0 
             else self.config.DEFAULT_TAKE_PROFIT_PCT)
    
    self._save_twap_state(
        order, 0, num_slices, 0.0, order.total_quantity, sl_pct, tp_pct
    )
    
    # 4️⃣ LOOP DE EXECUÇÃO
    for i in range(num_slices):
        # Verificar se deve cancelar
        if (order.is_complete or 
            order.status == OrderStatus.CANCELED or 
            self.system_state.get('shutdown_imminent', False)):
            logger.info(f"🎉 TWAP {order.id} concluído ou cancelado")
            break
        
        try:
            # 4.1️⃣ Calcular quantidade restante
            remaining_qty = order.total_quantity - order.filled_quantity
            # Exemplo fatia 1: 0.2353 - 0.0000 = 0.2353
            
            if math.isclose(remaining_qty, 0.0, abs_tol=1e-9):
                order.status = OrderStatus.FILLED
                logger.info(f"✅ Quantidade restante insignificante. FILLED.")
                break
            
            # 4.2️⃣ Quantidade desta fatia
            current_slice_qty = min(slice_quantity_base, remaining_qty)
            current_slice_qty = round(current_slice_qty, qty_precision)
            
            if current_slice_qty <= 0:
                logger.info(f"ℹ️ Fatia {i+1} quantidade zero. Finalizando.")
                break
            
            # 4.3️⃣ Preparar parâmetros
            client_order_id = f"twap_{uuid.uuid4().hex[:25]}"
            order.child_order_ids.append(client_order_id)
            
            slice_params = {
                "symbol": order.signal.symbol,
                "side": order.signal.action.value,
                "type": OrderType.MARKET.value,
                "quantity": current_slice_qty,
                "newClientOrderId": client_order_id,
                "newOrderRespType": "FULL",
            }
            
            # 4.4️⃣ ENVIAR FATIA
            logger.info(
                f"🌐 Fatia TWAP {i+1}/{num_slices}: "
                f"Enviando {current_slice_qty:.4f}..."
            )
            
            result = await self.connector.place_order(slice_params)
            
            if result:
                # 4.5️⃣ Atualizar ExecutionOrder
                new_trade = order.update_with_fill(result)
                
                if new_trade:
                    # Log da fatia
                    if self.trade_log_callback:
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
                            "client_order_id": result.get('clientOrderId', order.id),
                            "twap_slice": f"{i+1}/{num_slices}"  # Novo campo
                        }
                        self.system_state["recent_trades"].append(trade_log_data)
                        self.trade_log_callback(trade_log_data)
                    
                    # 4.6️⃣ Atualizar Portfolio
                    self.portfolio.update_from_trade(new_trade)
                    
                    # 4.7️⃣ ATUALIZAR PROTEÇÕES
                    # Para cada fatia, ajusta as proteções para a posição agregada
                    try:
                        await self._place_initial_hard_stop(new_trade, sl_pct)
                        if tp_pct > 0:
                            await self._place_initial_take_profit(new_trade, tp_pct)
                        
                        # Trailing stop com quantidade total
                        current_position_qty = self.portfolio.get_position_quantity(
                            new_trade.symbol
                        )
                        await self._place_trailing_stop(
                            symbol=new_trade.symbol,
                            side=new_trade.side,
                            quantity=abs(current_position_qty),
                            stop_loss_pct=sl_pct
                        )
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Erro ao atualizar proteções: {e}"
                        )
                
                # 4.8️⃣ Salvar estado após fatia bem-sucedida
                self._save_twap_state(
                    order, i + 1, num_slices,
                    order.filled_quantity, order.total_quantity,
                    sl_pct, tp_pct
                )
                
                logger.info(
                    f"✅ Fatia {i+1}/{num_slices} OK. "
                    f"Preenchido: {order.filled_quantity:.4f}/"
                    f"{order.total_quantity:.4f}"
                )
            else:
                logger.warning(f"⚠️ Falha na fatia {i+1}. Tentando próxima...")
            
            # 4.9️⃣ AGUARDAR INTERVALO
            if i < num_slices - 1:
                logger.info(
                    f"⏳ Aguardando {slice_interval_seconds:.0f}s "
                    f"até próxima fatia..."
                )
                await asyncio.sleep(slice_interval_seconds)
        
        except asyncio.CancelledError:
            # Salvar estado ao cancelamento
            self._save_twap_state(
                order, i, num_slices,
                order.filled_quantity, order.total_quantity,
                sl_pct, tp_pct
            )
            order.status = OrderStatus.CANCELED
            logger.info(f"🛑 TWAP cancelado na fatia {i+1}. Estado salvo.")
            break
        
        except Exception as e:
            logger.error(f"❌ Erro na fatia {i+1}: {e}", exc_info=True)
            self._save_twap_state(
                order, i, num_slices,
                order.filled_quantity, order.total_quantity,
                sl_pct, tp_pct
            )
            await asyncio.sleep(slice_interval_seconds / 2)
    
    # 5️⃣ FINALIZAÇÃO
    if not order.is_complete:
        if (order.filled_quantity > 0 and 
            math.isclose(order.filled_quantity, order.total_quantity, rel_tol=1e-5)):
            order.status = OrderStatus.FILLED
            logger.info(f"✅ Forçando FILLED para TWAP {order.id}")
        else:
            logger.warning(
                f"⚠️ TWAP não preenchida completamente. "
                f"Faltam: {order.total_quantity - order.filled_quantity:.4f}"
            )
    
    # 6️⃣ COLOCAR PROTEÇÕES FINAIS
    if order.status == OrderStatus.FILLED and order.trades:
        last_trade = order.trades[-1]
        self.portfolio.update_from_trade(last_trade)
        
        if sl_pct > 0:
            await self._place_initial_hard_stop(last_trade, sl_pct)
        if tp_pct > 0:
            await self._place_initial_take_profit(last_trade, tp_pct)
        
        await self._place_trailing_stop(
            symbol=last_trade.symbol,
            side=last_trade.side,
            quantity=abs(order.filled_quantity),
            stop_loss_pct=sl_pct
        )
        logger.info(f"[TWAP] Proteções finais colocadas para {order.id}")
    
    # 7️⃣ LIMPAR ESTADO
    self._clear_twap_state()
```

---

## 7. Proteções

### Hard Stop, Take Profit, Trailing Stop

**Arquivo:** `trading/execution_engine.py` (linhas 1213-1351)

```python
async def _place_initial_hard_stop(self, trade: Trade, stop_loss_pct: float):
    """
    Coloca Stop Loss Fixo imediatamente após entrada.
    """
    if not trade or stop_loss_pct <= 0:
        return
    
    entry_price = trade.executed_price  # 42525.00
    symbol = trade.symbol  # "BTCUSDT"
    quantity = abs(trade.quantity)  # 0.2353
    
    # Calcular preço do stop
    if trade.side == OrderSide.BUY:
        # Para LONG: stop ABAIXO do entry
        stop_price = entry_price * (1.0 - stop_loss_pct)
        # Cálculo: 42525 × (1 - 0.03) = 42525 × 0.97 = 41249.25
        sl_side = "SELL"  # Vender quando atingir stop
    else:
        # Para SHORT: stop ACIMA do entry
        stop_price = entry_price * (1.0 + stop_loss_pct)
        # Cálculo: 42525 × (1 + 0.03) = 42525 × 1.03 = 43800.75
        sl_side = "BUY"  # Comprar quando atingir stop
    
    logger.info(
        f"🛑 Colocando Hard Stop para {symbol}: "
        f"Entry: {entry_price}, Stop: {stop_price} ({stop_loss_pct:.1%})"
    )
    
    # Enviar ordem de stop loss
    result = await self.connector.place_stop_loss_order(
        symbol=symbol,
        side=sl_side,
        quantity=quantity,
        stop_price=stop_price
    )
    
    if result:
        order_id = result.get('orderId', 'N/A')
        logger.info(f"✅ Hard Stop {order_id} ativado com sucesso!")
        self.system_state["command_feedback"] = \
            f"Hard Stop ativo para {symbol} @ {stop_price:.2f}"
    else:
        logger.critical(
            f"🚨 PERIGO! Falha ao colocar Hard Stop para {symbol}! "
            f"Posição desprotegida!"
        )


async def _place_initial_take_profit(self, trade: Trade, take_profit_pct: float):
    """
    Coloca Take Profit Automático após entrada.
    """
    if not trade or take_profit_pct <= 0:
        return
    
    entry_price = trade.executed_price  # 42525.00
    symbol = trade.symbol
    quantity = abs(trade.quantity)
    
    # Calcular preço do TP
    if trade.side == OrderSide.BUY:
        # Para LONG: TP ACIMA do entry
        stop_price = entry_price * (1.0 + take_profit_pct)
        # Cálculo: 42525 × (1 + 0.09) = 42525 × 1.09 = 46312.25
        tp_side = "SELL"
    else:
        # Para SHORT: TP ABAIXO do entry
        stop_price = entry_price * (1.0 - take_profit_pct)
        tp_side = "BUY"
    
    logger.info(
        f"🎯 Colocando Take Profit para {symbol}: "
        f"Entry: {entry_price}, TP: {stop_price} (+{take_profit_pct:.1%})"
    )
    
    result = await self.connector.place_take_profit_order(
        symbol=symbol,
        side=tp_side,
        quantity=quantity,
        stop_price=stop_price
    )
    
    if result:
        order_id = result.get('orderId', 'N/A')
        logger.info(f"✅ Take Profit {order_id} ativado com sucesso!")
        self.system_state["command_feedback"] = \
            f"Take Profit ativo para {symbol} @ {stop_price:.2f}"


async def _place_trailing_stop(
    self, 
    symbol: str, 
    side: OrderSide, 
    quantity: float, 
    stop_loss_pct: Optional[float] = None
):
    """
    Coloca Trailing Stop (segue o preço para cima).
    
    Exemplo:
    - Entrada BTC @ 42525
    - Trailing 3%
    - Preço sobe para 44000 (+3.5%)
      → Trailing stop ajusta para 44000 × 0.97 = 42680
    - Preço cai para 43000
      → Se < 42680 → Stop executado
    - Preço volta a subir para 45000
      → Trailing stop ajusta para 45000 × 0.97 = 43650
    """
    try:
        sl_pct = stop_loss_pct or self.config.DEFAULT_STOP_LOSS_PCT
        callback_rate = abs(sl_pct) * 100
        # Cálculo: 0.03 × 100 = 3% callback rate
        
        # Lado oposto à posição
        if side == OrderSide.BUY:
            trailing_side = "SELL"  # Stop de proteção é vender
        else:
            trailing_side = "BUY"
        
        logger.info(
            f"🎯 Colocando Trailing Stop: {symbol} "
            f"{trailing_side} @ {callback_rate:.1f}% callback"
        )
        
        result = await self.connector.place_trailing_stop_order(
            symbol=symbol,
            side=trailing_side,
            quantity=quantity,
            callback_rate=callback_rate
        )
        
        if result and result.get('orderId'):
            order_id = result.get('orderId', 'N/A')
            logger.info(
                f"✅ Trailing Stop {order_id} colocado @ {callback_rate:.1f}%"
            )
            self.system_state["command_feedback"] = \
                f"Trailing Stop ativo {symbol} @ {callback_rate:.1f}%"
            return result
        
    except Exception as e:
        logger.error(f"❌ Erro ao colocar Trailing Stop: {e}", exc_info=True)
        return None
```

---

## 8. Monitoramento

### Loop de Monitoramento de Posições

**Arquivo:** `trading/execution_engine.py` (linhas 497-602)

```python
async def _monitor_orders_loop(self):
    """
    Loop que monitora posições abertas e proteções.
    Executa a cada 10 segundos.
    """
    monitor_interval = 10  # segundos
    
    logger.info(
        f"⚙️ Loop de monitoramento iniciado (checa a cada {monitor_interval}s)"
    )
    
    while True:
        try:
            await asyncio.sleep(monitor_interval)
            
            # Verificar se live trading está ativo
            is_live = self.system_state.get('live_trading_enabled', False)
            if not is_live:
                continue
            
            # Gerenciar posições existentes
            await self._manage_existing_positions()
        
        except asyncio.CancelledError:
            logger.info("🛑 Loop de monitoramento cancelado")
            break
        except Exception as e:
            logger.error(f"❌ Erro no monitoramento: {e}", exc_info=True)
            await asyncio.sleep(monitor_interval)


async def _manage_existing_positions(self):
    """
    Para cada posição aberta, verifica se tem proteção.
    Se não tiver, cria um stop de emergência.
    """
    try:
        # Copiar posições para evitar modificação durante iteração
        positions_copy = list(self.portfolio.positions.items())
        
        for symbol, position in positions_copy:
            # Ignorar posições fechadas
            if position.quantity == 0:
                self._stop_fail_count.pop(symbol, None)
                continue
            
            # Ignorar símbolos não-USDT
            if not symbol.endswith('USDT'):
                continue
            
            logger.debug(f"🔍 Monitorando posição aberta: {symbol} ({position.quantity})")
            
            # Implementar backoff exponencial se muitas falhas
            fail_count = self._stop_fail_count.get(symbol, 0)
            
            if fail_count >= self.MAX_STOP_FAILURES:
                # Reduzir frequência de tentativas
                if self._stop_monitor_tick % self.STOP_BACKOFF_CYCLES != 0:
                    continue
            
            # Verificar veto de margem recente
            last_veto = self._margin_veto_registry.get(symbol)
            if last_veto:
                time_diff = (datetime.utcnow() - last_veto).total_seconds()
                if time_diff < 300:  # 5 minutos de cooldown
                    continue
                else:
                    self._margin_veto_registry.pop(symbol, None)
            
            # Obter ordens abertas para este símbolo
            open_orders = await self.connector.get_open_orders(symbol)
            
            # Verificar se já tem alguma ordem de stop
            has_stop = any(
                o.get('type') in ('TRAILING_STOP_MARKET', 'STOP_MARKET', 'STOP')
                for o in (open_orders or [])
            )
            
            if not has_stop:
                # ⚠️ POSIÇÃO DESPROTEGIDA!
                logger.warning(
                    f"🚨 Posição exposta: {symbol} ({position.quantity:.4f}). "
                    f"Criando stop de emergência..."
                )
                
                # Preparar ordem de stop
                position_side = OrderSide.BUY if position.quantity > 0 else OrderSide.SELL
                qty = abs(position.quantity)
                
                # Tentar colocar stop
                result = await self._place_trailing_stop(
                    symbol=symbol,
                    side=position_side,
                    quantity=qty,
                    stop_loss_pct=self.config.DEFAULT_STOP_LOSS_PCT
                )
                
                if result:
                    # Stop colocado com sucesso
                    self._stop_fail_count[symbol] = 0
                    logger.info(f"✅ Stop de emergência colocado para {symbol}")
                else:
                    # Stop falhou
                    self._stop_fail_count[symbol] = fail_count + 1
                    
                    if self._stop_fail_count[symbol] >= self.MAX_STOP_FAILURES:
                        logger.warning(
                            f"⚠️ {symbol}: {self._stop_fail_count[symbol]} "
                            f"falhas consecutivas. Ativando backoff."
                        )
            else:
                # Stop ativo: reset contador
                if self._stop_fail_count.get(symbol, 0) > 0:
                    logger.info(f"✅ {symbol}: Stop ativo detectado. Resetando backoff.")
                
                self._stop_fail_count[symbol] = 0
        
        # Incrementar tick do monitor
        self._stop_monitor_tick += 1
    
    except Exception as e:
        logger.error(f"❌ Erro em _manage_existing_positions: {e}", exc_info=True)
```

---

## Resumo do Fluxo Completo

```
AIController.generate_signal()
    ↓
ExecutionEngine.submit_order(signal)
    ├─ Valida
    ├─ Calcula quantidade
    ├─ Escolhe estratégia
    └─ Enfileira
    ↓
_execution_loop() (async)
    ↓
_execute_order(order)
    ├─ Ajusta alavancagem
    └─ Chama estratégia:
        ├─ _execute_market_real() → MARKET order
        ├─ _execute_limit_real() → LIMIT order (com timeout)
        └─ _execute_twap_real() → TWAP (múltiplas fatias)
    ↓
BinanceConnector.place_order(params)
    ↓
Binance API executa
    ↓
Resposta processada
    ├─ order.update_with_fill()
    ├─ portfolio.update_from_trade()
    └─ Proteções colocadas:
        ├─ Hard Stop (3%)
        ├─ Take Profit (9%)
        └─ Trailing Stop (3%)
    ↓
_monitor_orders_loop() aguarda 10s
    ├─ Verifica posições abertas
    ├─ Verifica se tem proteções
    └─ Cria stop de emergência se necessário
    ↓
WebSocket monitora preço em tempo real
    ├─ Preço atinge TP → Venda automática
    ├─ Preço atinge SL → Perda limitada
    └─ Preço sobe 3% → Trailing stop segue
    ↓
Posição fechada → Trade registrado
```

---

**Documentação Completa de Código**  
*Bot Shield - Sistema de Trading com IA*  
**Status:** ✅ Operacional e Documentado
