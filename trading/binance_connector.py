
# -----------------------------------------------------------------------------
# ARQUIVO: trading/binance_connector.py
# -----------------------------------------------------------------------------

import asyncio
import aiohttp
import json
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable
import sys

from utils.logger import get_logger
from config.settings import Config, TradingConfig 

logger = get_logger("BinanceConnector")

shutdown_event = asyncio.Event()

class BinanceConnector:
    """
    Connector para Binance Futures API.
    
    Suporta modo híbrido: dados de produção + ordens na testnet.
    - force_production=True: Usa API de produção para buscar dados (klines, tickers)
    - force_production=False (default): Usa a configuração BINANCE_TESTNET do .env
    
    Para Paper Trading ideal:
    - Crie um connector para dados: BinanceConnector(config, force_production=True)
    - Crie um connector para trading: BinanceConnector(config, force_production=False)
    """
    
    def __init__(self, config: Config, force_production: bool = False):
        if not isinstance(config, Config):
            raise TypeError("🚨 [ERRO CONECTOR] 'config' deve ser uma instância da classe Config.")
        self.config = config
        self.force_production = force_production

        # <<<<<<<<<<<<<<<<< ESTA LINHA É CRÍTICA E DEVE ESTAR PRESENTE! >>>>>>>>>>>>>>>>>>>
        self.trading_config = TradingConfig() # Instancia TradingConfig para acesso aos seus atributos
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # Determina se deve usar testnet ou produção
        # force_production=True ignora BINANCE_TESTNET e usa produção
        self.testnet = False if force_production else self.config.BINANCE_TESTNET
        
        # Seleciona API keys baseado no modo
        if self.testnet:
            # Modo Testnet: usa chaves da testnet se disponíveis
            testnet_key = getattr(self.config, 'BINANCE_TESTNET_API_KEY', None)
            testnet_secret = getattr(self.config, 'BINANCE_TESTNET_API_SECRET', None)
            
            if testnet_key and testnet_secret:
                self.api_key = testnet_key
                self.api_secret = testnet_secret
                masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key else "None"
                logger.info(f"🔑 [CONECTOR] Usando chaves API específicas da TESTNET. (Key: {masked_key})")
            else:
                self.api_key = self.config.BINANCE_API_KEY
                self.api_secret = self.config.BINANCE_API_SECRET
                masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key else "None"
                logger.warning(f"⚠️ [CONECTOR] Chaves TESTNET não encontradas. Usando chaves padrão/production para testnet. (Key: {masked_key})")
        else:
            # Modo Produção (ou force_production=True)
            self.api_key = self.config.BINANCE_API_KEY
            self.api_secret = self.config.BINANCE_API_SECRET

        if not self.api_key or not self.api_secret:
            logger.critical("❌ [ERRO CONECTOR] API_KEY ou API_SECRET da Binance não configurados. As operações falharão.")

        # Configura URLs baseado no modo
        if self.testnet:
            self.base_url = "https://testnet.binancefuture.com"
            self.ws_base_url = "wss://stream.binancefuture.com"
            logger.info("📡 [CONECTOR] Usando ambiente Binance TESTNET.")
        else:
            self.base_url = "https://fapi.binance.com"
            self.ws_base_url = "wss://fstream.binance.com"
            mode_str = "(force_production)" if force_production else ""
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key else "None"
            logger.info(f"📡 [CONECTOR] Usando ambiente Binance PRODUÇÃO {mode_str}. (Key: {masked_key})")

        self.session: Optional[aiohttp.ClientSession] = None
        self._timestamp_offset: int = 0
        self._ws_reconnect_attempts: int = 0
        self.max_ws_reconnect_attempts = 10
        self.ws_reconnect_delay_base = 5
        self.symbol_info_cache: Dict[str, Any] = {} # Cache para exchange info

    async def _calculate_timestamp_offset(self) -> int:
        """
        Calcula o offset de tempo entre o servidor local e o servidor da Binance
        para evitar erros de 'timestamp outside of recvWindow'.
        """
        logger.info("⏱️ [CONECTOR] Sincronizando tempo com o servidor Binance...")
        try:
            # Faz uma requisição pública de tempo (não requer autenticação)
            server_time_response = await self._make_request('GET', '/fapi/v1/time', signed=False) 
            
            if server_time_response and 'serverTime' in server_time_response:
                server_time = int(server_time_response['serverTime'])
                local_time = int(time.time() * 1000)
                offset = server_time - local_time
                self._timestamp_offset = offset
                logger.info(f"✅ [CONECTOR] Sincronização de tempo concluída. Offset: {offset}ms.")
                return offset
            else:
                logger.warning("⚠️ [CONECTOR] Não foi possível obter a hora do servidor da Binance para calcular o offset. Resposta inesperada.")
                return 0 # Retorna 0 se não conseguir obter o tempo
        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Falha ao calcular o offset de horário da Binance: {e}", exc_info=True)
            return 0 # Em caso de erro, assume offset zero (pode causar problemas em requisições assinadas)

    async def connect(self):
        """
        Estabelece a conexão HTTP persistente e sincroniza o tempo.
        Deve ser chamado uma única vez no início da aplicação.
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            logger.info("🔗 [CONECTOR] Nova sessão aiohttp criada.")
        else:
            logger.info("🔗 [CONECTOR] Sessão aiohttp já ativa.")

        # Recalcula o offset de tempo a cada conexão para garantir precisão
        await self._calculate_timestamp_offset()
        logger.info("✅ [CONECTOR] Conexão com a Binance estabelecida e tempo sincronizado.")

    async def close(self):
        """
        Fecha a sessão aiohttp e limpa os recursos.
        Deve ser chamado ao desligar a aplicação.
        """
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            logger.info("👋 [CONECTOR] Sessão aiohttp fechada com sucesso.")
        else:
            logger.info("👋 [CONECTOR] Nenhuma sessão aiohttp ativa para fechar.")

    def _generate_signature(self, data: Dict[str, Any]) -> str:
        """
        Gera a assinatura HMAC SHA256 necessária para requisições autenticadas.
        """
        query_string = '&'.join([f"{key}={value}" for key, value in sorted(data.items())])
        return hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, signed: bool = False, headers: Optional[Dict] = None) -> Any:
        """
        Faz uma requisição HTTP assíncrona para a API da Binance.

        Args:
            method (str): Método HTTP ('GET', 'POST', 'PUT', 'DELETE').
            endpoint (str): O caminho do endpoint da API (e.g., '/fapi/v1/klines').
            params (Optional[Dict]): Dicionário de parâmetros da requisição.
            signed (bool): Se a requisição precisa ser assinada (autenticada).
            headers (Optional[Dict]): Cabeçalhos HTTP adicionais.

        Returns:
            Any: A resposta JSON da API, ou None em caso de erro.
        """
        if self.session is None or self.session.closed:
            logger.error(f"❌ [ERRO CONECTOR] Sessão aiohttp não está ativa para {method} {endpoint}. Chame connect() primeiro.")
            raise ConnectionError("Sessão aiohttp não está ativa. Chame connect() primeiro.")

        params = params.copy() if params else {}
        req_headers = headers.copy() if headers else {}
        
        if signed:
            if not self.api_key or not self.api_secret:
                logger.error(f"❌ [ERRO CONECTOR] Tentativa de requisição assinada sem API_KEY/SECRET configurados para {endpoint}.")
                return None
            params['timestamp'] = int(time.time() * 1000) + self._timestamp_offset
            
            # 1. Gera assinatura com os parâmetros atuais (incluindo timestamp)
            signature = self._generate_signature(params)
            
            # 2. Ordena os parâmetros para envio
            # Aiohttp envia na ordem da lista de tuplas.
            params_list = sorted(params.items())
            
            # 3. Adiciona a assinatura NO FINAL
            params_list.append(('signature', signature))
            
            # 4. Adiciona cabeçalho
            req_headers['X-MBX-APIKEY'] = self.api_key
            
            # Atualiza params para a lista ordenada com assinatura no final
            params = params_list

        url = f"{self.base_url}{endpoint}"
        # CORREÇÃO: Acessa EXECUTION_TIMEOUT_SECONDS da instância self.trading_config
        timeout = aiohttp.ClientTimeout(total=self.trading_config.EXECUTION_TIMEOUT_SECONDS)

        try:
            async with self.session.request(method, url, params=params, headers=req_headers, timeout=timeout) as response:
                # Lê o corpo ANTES de verificar o status para evitar Connection closed
                response_text = await response.text()
                
                if response.status >= 400:
                    # Tenta parsear como JSON para mensagem de erro detalhada
                    try:
                        error_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        error_data = response_text
                    logger.error(f"❌ [ERRO CONECTOR] Erro da API da Binance ({response.status}) para {endpoint}: {error_data}")
                    return None
                
                # Sucesso - parseia JSON
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ [CONECTOR] Resposta não-JSON de {endpoint}: {response_text}")
                    return response_text
                    
        except aiohttp.ClientConnectorError as e:
            logger.error(f"❌ [ERRO CONECTOR] Falha de conexão para {endpoint}: {e}. Verifique sua rede ou URL base.", exc_info=True)
            return None
        except aiohttp.ServerTimeoutError:
            logger.warning(f"⏰ [TIMEOUT CONECTOR] Requisição para {endpoint} excedeu o tempo limite de {timeout.total}s.")
            return None
        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Erro inesperado na requisição para {endpoint}: {e}", exc_info=True)
            return None
    
    async def get_kline_data(self, symbol: str, interval: str, limit: int = 500, start_ts: Optional[int] = None, end_ts: Optional[int] = None) -> Optional[List[List]]:
        """
        Obtém dados de klines (velas) para um símbolo e intervalo.
        """
        if not symbol or not interval:
            logger.error("❌ [ERRO CONECTOR] 'symbol' e 'interval' são obrigatórios para get_kline_data.")
            return None
        
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if start_ts: 
            params['startTime'] = start_ts
        if end_ts: 
            params['endTime'] = end_ts
            
        logger.debug(f"📊 [CONECTOR] Buscando klines para {symbol}-{interval} (limit={limit}, start={start_ts}, end={end_ts})...")
        data = await self._make_request('GET', '/fapi/v1/klines', params)
        if data is None:
            logger.warning(f"⚠️ [CONECTOR] Nenhuma dado de kline retornado para {symbol}-{interval}.")
        return data

    async def get_account_summary(self) -> Dict[str, Any]:
        """
        Obtém um resumo da conta Binance Futures (saldo, PnL não realizado, posições).
        Em modo PAPER_TRADING_PURE, retorna dados simulados.
        """
        # PAPER_TRADING_PURE: Retorna dados simulados para evitar chamadas autenticadas bloqueadas
        if getattr(self.config, 'PAPER_TRADING_PURE', False):
            logger.debug("📝 [CONECTOR] Modo PAPER_TRADING_PURE: Retornando saldo simulado.")
            return {
                'total_value': self.trading_config.INITIAL_CAPITAL,
                'cash': self.trading_config.INITIAL_CAPITAL,
                'unrealized_pnl': 0.0,
                'positions_count': 0,
                'positions': {}
            }
        
        logger.debug("💰 [CONECTOR] Buscando resumo da conta...")
        summary = {'total_value': 0.0, 'cash': 0.0, 'unrealized_pnl': 0.0, 'positions_count': 0, 'positions': {}}
        
        try:
            # Obter saldos
            balance_data = await self._make_request('GET', '/fapi/v2/balance', params={'recvWindow': 60000}, signed=True)
            if balance_data:
                # Para contratos USDT-Margined (BTCUSDT etc.), apenas USDT é a margem real.
                # BTC e USDC são pools de margem separados para contratos COIN-M e USDC-M.
                primary_asset = 'USDT'
                total_balance = 0.0
                total_available = 0.0

                for asset in balance_data:
                    asset_name = asset.get('asset', '')
                    bal = float(asset.get('balance', 0.0))
                    avail = float(asset.get('availableBalance', 0.0))
                    cross_pnl = float(asset.get('crossUnPnl', 0.0))

                    if bal == 0 and cross_pnl == 0:
                        continue

                    logger.debug(f"💰 [CONECTOR] Asset {asset_name}: Balance={bal:.4f}, Available={avail:.4f}, CrossPnL={cross_pnl:.4f}")

                    # Soma apenas USDT (margem dos contratos USDT-M)
                    if asset_name == primary_asset:
                        total_balance = bal
                        total_available = avail

                summary['total_value'] = total_balance
                summary['cash'] = total_available

                if total_balance > 0:
                    logger.info(f"💰 [CONECTOR] Saldo USDT: ${total_balance:,.2f}, Disponível: ${total_available:,.2f}")
            else:
                logger.warning("⚠️ [CONECTOR] Não foi possível obter dados de balanço da Binance.")

            # Obter posições abertas e PnL não realizado
            positions_data = await self._make_request('GET', '/fapi/v2/positionRisk', params={'recvWindow': 60000}, signed=True)
            if positions_data:
                pnl = 0.0
                margin_used = 0.0
                count = 0
                positions_dict = {}
                for pos in positions_data:
                    # Filtra posições com quantidade diferente de zero
                    pos_amt = float(pos.get('positionAmt', 0.0))
                    if pos_amt != 0:
                        pnl += float(pos.get('unrealizedProfit', 0.0) or 0.0)
                        
                        # Calcula a margem utilizada nesta posição
                        mark_price = float(pos.get('markPrice', 0.0))
                        leverage = int(pos.get('leverage', 1))
                        if leverage > 0:
                            margin_used += (abs(pos_amt) * mark_price) / leverage
                        
                        count += 1
                        # Adiciona detalhes da posição para reconciliação
                        symbol = pos.get('symbol', '')
                        positions_dict[symbol] = {
                            'quantity': pos_amt,
                            'entry_price': float(pos.get('entryPrice', 0.0)),
                            'mark_price': mark_price,
                            'unrealized_pnl': float(pos.get('unrealizedProfit', 0.0) or 0.0),
                            'leverage': leverage,
                            'margin_type': pos.get('marginType', 'cross'),
                            'liquidation_price': float(pos.get('liquidationPrice', 0.0))
                        }
                summary['unrealized_pnl'] = pnl
                summary['margin_used'] = margin_used
                summary['positions_count'] = count
                summary['positions'] = positions_dict
                # Adiciona PnL não realizado ao valor total para uma visão mais completa do patrimônio
                summary['total_value'] += pnl 
            else:
                logger.warning("⚠️ [CONECTOR] Não foi possível obter dados de posição da Binance.")

            logger.info(f"✅ [CONECTOR] Resumo da conta obtido: Total: ${summary['total_value']:.2f}, Caixa: ${summary['cash']:.2f}, PnL: ${summary['unrealized_pnl']:.2f}.")
            return summary

        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Erro ao obter resumo da conta: {e}", exc_info=True)
            return summary # Retorna o resumo parcial ou padrão em caso de erro

    async def set_leverage_for_symbol(self, symbol: str, leverage: int) -> bool:
        """
        Define a alavancagem para um símbolo específico na Binance Futures.
        Endpoint: POST /fapi/v1/leverage
        """
        if not symbol or leverage < 1:
            logger.error(f"❌ [ERRO CONECTOR] Parâmetros inválidos para set_leverage: Symbol={symbol}, Leverage={leverage}")
            return False

        try:
            logger.info(f"⚙️ [CONECTOR] Ajustando alavancagem de {symbol} para {leverage}x...")
            params = {
                'symbol': symbol,
                'leverage': leverage
            }
            
            # Chama o endpoint da Binance
            result = await self._make_request('POST', '/fapi/v1/leverage', params=params, signed=True)
            
            if result and 'leverage' in result:
                new_lev = int(result['leverage'])
                logger.info(f"✅ [CONECTOR] Alavancagem para {symbol} definida com sucesso para {new_lev}x (Max Notional: {result.get('maxNotionalValue', 'N/A')}).")
                return new_lev == leverage
            else:
                logger.warning(f"⚠️ [CONECTOR] Falha ao definir alavancagem para {symbol}. Resposta: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Exceção ao definir alavancagem: {e}", exc_info=True)
            return False

    async def place_order(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Coloca uma nova ordem na Binance Futures.
        """
        if not params or 'symbol' not in params or 'side' not in params or 'type' not in params or 'quantity' not in params:
            logger.error("❌ [ERRO CONECTOR] Parâmetros de ordem incompletos para place_order.")
            return None
        
        # Parâmetros adicionais para alavancagem
        if 'leverage' in params:
            # Enviar a alavancagem para a API, se suportado
            # A API da Binance Futures tem um endpoint separado para ajustar a alavancagem de um par
            # /fapi/v1/leverage (POST)
            # Isso precisa ser feito ANTES de enviar a ordem, se a alavancagem da conta/símbolo precisa ser alterada.
            # Idealmente, você chamaria set_leverage_for_symbol() aqui se a alavancagem desejada for diferente da atual.
            pass # A implementação de set_leverage_for_symbol() seria necessária

        logger.info(f"📈 [CONECTOR] Enviando ordem: {params.get('side')} {params.get('quantity')} {params.get('symbol')} ({params.get('type')})...")
        try:
            # Endpoint para ordens de futuros
            result = await self._make_request('POST', '/fapi/v1/order', params=params, signed=True)
            if result:
                logger.info(f"✅ [CONECTOR] Ordem {result.get('orderId')} ({result.get('status')}) para {result.get('symbol')} enviada com sucesso.")
            else:
                logger.warning(f"⚠️ [CONECTOR] Falha ao enviar ordem: {params}. Resultado nulo.")
            return result
        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Exceção ao enviar ordem: {e}", exc_info=True)
            return None

    async def cancel_order(self, symbol: str, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Cancela uma ordem aberta na Binance Futures.
        Pode ser cancelada por orderId (da exchange) ou clientOrderId (seu ID).
        """
        if not symbol:
            logger.error("❌ [ERRO CONECTOR] 'symbol' é obrigatório para cancelar ordem.")
            return None
        if not order_id and not client_order_id:
            logger.error("❌ [ERRO CONECTOR] 'order_id' ou 'client_order_id' é obrigatório para cancelar ordem.")
            return None

        params = {'symbol': symbol}
        if order_id: params['orderId'] = order_id
        if client_order_id: params['origClientOrderId'] = client_order_id

        logger.info(f"🗑️ [CONECTOR] Cancelando ordem para {symbol}: {order_id if order_id else client_order_id}...")
        try:
            result = await self._make_request('DELETE', '/fapi/v1/order', params=params, signed=True)
            if result:
                logger.info(f"✅ [CONECTOR] Ordem {result.get('orderId')} ({result.get('status')}) cancelada com sucesso.")
            else:
                logger.warning(f"⚠️ [CONECTOR] Falha ao cancelar ordem para {symbol}.")
            return result
        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Exceção ao cancelar ordem: {e}", exc_info=True)
            return None

    async def get_ticker_price(self, symbol: str) -> Optional[Dict[str, str]]:
        """
        Obtém o último preço de ticker para um símbolo.
        Útil para obter preços atuais sem precisar de klines.
        """
        if not symbol:
            logger.error("❌ [ERRO CONECTOR] 'symbol' é obrigatório para get_ticker_price.")
            return None

        params = {'symbol': symbol}
        logger.debug(f"💲 [CONECTOR] Buscando preço de ticker para {symbol}...")
        data = await self._make_request('GET', '/fapi/v1/ticker/price', params=params)
        if data is None:
            logger.warning(f"⚠️ [CONECTOR] Não foi possível obter o preço do ticker para {symbol}.")
        return data # Retorna um dicionário como {'symbol': 'BTCUSDT', 'price': '60000.00'}


    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Obtém informações de precisão e filtros para um símbolo.
        Usa cache para evitar excesso de requisições.
        """
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        
        logger.info(f"🔎 [CONECTOR] Buscando exchangeInfo para {symbol}...")
        try:
            exchange_info = await self._make_request('GET', '/fapi/v1/exchangeInfo', signed=False)
            if exchange_info and 'symbols' in exchange_info:
                for s in exchange_info['symbols']:
                    # Armazena no cache
                    info = {
                        'quantityPrecision': s['quantityPrecision'],
                        'pricePrecision': s['pricePrecision'],
                        'filters': s['filters']
                    }
                    self.symbol_info_cache[s['symbol']] = info
                
                if symbol in self.symbol_info_cache:
                    return self.symbol_info_cache[symbol]
            
            logger.warning(f"⚠️ [CONECTOR] Símbolo {symbol} não encontrado no exchangeInfo.")
            return None
        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Erro ao obter exchangeInfo: {e}", exc_info=True)
            return None

    async def place_trailing_stop_order(self, symbol: str, side: str, quantity: float,  
                                         callback_rate: float) -> Optional[Dict[str, Any]]:
        """
        Coloca uma ordem de Trailing Stop Market na Binance Futures via Algo Order endpoint.
        
        Desde 2025-12-09, ordens condicionais como TRAILING_STOP_MARKET devem usar
        o endpoint /fapi/v1/algoOrder.
        
        Args:
            symbol: Par de trading (e.g., "BTCUSDT")
            side: "BUY" (para fechar SHORT) ou "SELL" (para fechar LONG)
            quantity: Tamanho da posição a fechar
            callback_rate: Percentual de callback (1.0 = 1%, 4.0 = 4%). Range: 0.1-5.0
        
        Returns:
            Resultado da ordem ou None em caso de erro
        """
        if not symbol or not side or quantity <= 0:
            logger.error("❌ [ERRO CONECTOR] Parâmetros inválidos para trailing stop.")
            return None
        
        # Suporte para Basis Points (ex: 400 = 4%)
        # Se o valor for maior que 5, assumimos que é basis points e convertemos
        if callback_rate > 5.0:
            logger.info(f"ℹ️ [CONECTOR] callbackRate {callback_rate} detectado como basis points. Convertendo para {callback_rate / 100}%")
            callback_rate = callback_rate / 100.0

        # Callback rate deve estar entre 0.1% e 5%
        callback_rate = max(0.1, min(5.0, callback_rate))
        
        # Parâmetros para Order endpoint (Standard Trailing Stop)
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'TRAILING_STOP_MARKET',
            'quantity': quantity, # Caller must ensure precision or Binance rejects. 
            'callbackRate': callback_rate,  # Percentual (4.0 = 4%)
        }
        
        logger.info(f"🎯 [CONECTOR] Colocando Trailing Stop (Standard): {side} {quantity} {symbol} @ {callback_rate}% callback...")
        
        try:
            # Endpoint padrão de ordens (suporta TRAILING_STOP_MARKET)
            result = await self._make_request('POST', '/fapi/v1/order', params=params, signed=True)
            if result:
                order_id = result.get('orderId', 'N/A')
                status = result.get('status', 'UNKNOWN')
                logger.info(f"✅ [CONECTOR] Trailing Stop {order_id} colocado! Status: {status}, Callback: {callback_rate}%")
            else:
                logger.warning(f"⚠️ [CONECTOR] Falha ao colocar trailing stop para {symbol}.")
            return result
        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Exceção ao colocar trailing stop: {e}", exc_info=True)
            return None

    async def place_stop_loss_order(self, symbol: str, side: str, quantity: float, stop_price: float) -> Optional[Dict[str, Any]]:
        """
        Coloca uma ordem de Stop Loss Market (Reduce Only).
        
        Args:
            symbol: Par de trading
            side: 'SELL' (para Long) ou 'BUY' (para Short)
            quantity: Quantidade a fechar
            stop_price: Preço de disparo do stop
        """
        if not symbol or not side or quantity <= 0 or stop_price <= 0:
            logger.error(f"❌ [ERRO CONECTOR] Parâmetros inválidos para Stop Loss: {symbol} {side} {quantity} @ {stop_price}")
            return None

        # Obter precisão de preço/quantidade para formatar corretamente
        # Se falhar, usa padrão 2/3 casas
        symbol_info = await self.get_symbol_info(symbol)
        price_precision = int(symbol_info['pricePrecision']) if symbol_info else 2
        qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'STOP_MARKET',
            'quantity': f"{quantity:.{qty_precision}f}",
            'stopPrice': f"{stop_price:.{price_precision}f}",
            'reduceOnly': 'true'
        }
        
        logger.info(f"🛡️ [CONECTOR] Colocando Stop Loss: {side} {params['quantity']} {symbol} @ {params['stopPrice']} (ReduceOnly)...")
        
        try:
            result = await self._make_request('POST', '/fapi/v1/order', params=params, signed=True)
            if result:
                logger.info(f"✅ [CONECTOR] Stop Loss {result.get('orderId')} colocado com sucesso.")
            else:
                logger.warning(f"⚠️ [CONECTOR] Falha ao colocar Stop Loss para {symbol}.")
            return result
        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Exceção ao colocar Stop Loss: {e}", exc_info=True)
            return None

    async def place_take_profit_order(self, symbol: str, side: str, quantity: float, stop_price: float) -> Optional[Dict[str, Any]]:
        """
        Coloca uma ordem de Take Profit Market (Reduce Only).
        
        Args:
            symbol: Par de trading
            side: 'SELL' (para Long) ou 'BUY' (para Short)
            quantity: Quantidade a fechar
            stop_price: Preço de disparo do TP
        """
        if not symbol or not side or quantity <= 0 or stop_price <= 0:
            logger.error(f"❌ [ERRO CONECTOR] Parâmetros inválidos para Take Profit: {symbol} {side} {quantity} @ {stop_price}")
            return None

        # Obter precisão de preço/quantidade
        symbol_info = await self.get_symbol_info(symbol)
        price_precision = int(symbol_info['pricePrecision']) if symbol_info else 2
        qty_precision = int(symbol_info['quantityPrecision']) if symbol_info else 3
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'TAKE_PROFIT_MARKET',
            'quantity': f"{quantity:.{qty_precision}f}",
            'stopPrice': f"{stop_price:.{price_precision}f}",
            'reduceOnly': 'true'
        }
        
        logger.info(f"🎯 [CONECTOR] Colocando Take Profit: {side} {params['quantity']} {symbol} @ {params['stopPrice']} (ReduceOnly)...")
        
        try:
            result = await self._make_request('POST', '/fapi/v1/order', params=params, signed=True)
            if result:
                logger.info(f"✅ [CONECTOR] Take Profit {result.get('orderId')} colocado com sucesso.")
            else:
                logger.warning(f"⚠️ [CONECTOR] Falha ao colocar Take Profit para {symbol}.")
            return result
        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Exceção ao colocar Take Profit: {e}", exc_info=True)
            return None

    async def get_order(self, symbol: str, order_id: Optional[str] = None, orig_client_order_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Consulta o status de uma ordem específica.
        
        Args:
            symbol: Par de trading
            order_id: ID da ordem (opcional)
            orig_client_order_id: Client Order ID (opcional)
        """
        if not symbol or (not order_id and not orig_client_order_id):
            return None
            
        params = {'symbol': symbol}
        if order_id:
            params['orderId'] = order_id
        if orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
            
        try:
            return await self._make_request('GET', '/fapi/v1/order', params=params, signed=True)
        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Erro ao buscar ordem {order_id or orig_client_order_id}: {e}")
            return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtém todas as ordens abertas para um símbolo ou para todos os símbolos.
        
        Args:
            symbol: Par de trading (opcional, se None retorna todas as ordens)
        
        Returns:
            Lista de ordens abertas
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        logger.debug(f"📋 [CONECTOR] Buscando ordens abertas{' para ' + symbol if symbol else ''}...")
        
        try:
            result = await self._make_request('GET', '/fapi/v1/openOrders', params=params, signed=True)
            if result is None:
                return []
            logger.info(f"📋 [CONECTOR] {len(result)} ordens abertas encontradas{' para ' + symbol if symbol else ''}.")
            return result
        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Exceção ao buscar ordens abertas: {e}", exc_info=True)
            return []

    async def cancel_all_orders(self, symbol: str) -> bool:
        """
        Cancela todas as ordens abertas para um símbolo.
        
        Args:
            symbol: Par de trading (obrigatório)
        
        Returns:
            True se sucesso, False em caso de erro
        """
        if not symbol:
            logger.error("❌ [ERRO CONECTOR] 'symbol' é obrigatório para cancel_all_orders.")
            return False
        
        params = {'symbol': symbol}
        
        logger.info(f"🗑️ [CONECTOR] Cancelando todas as ordens abertas para {symbol}...")
        
        try:
            result = await self._make_request('DELETE', '/fapi/v1/allOpenOrders', params=params, signed=True)
            if result:
                logger.info(f"✅ [CONECTOR] Todas as ordens para {symbol} foram canceladas.")
                return True
            else:
                logger.warning(f"⚠️ [CONECTOR] Nenhuma ordem para cancelar ou falha ao cancelar para {symbol}.")
                return False
        except Exception as e:
            logger.error(f"❌ [ERRO CONECTOR] Exceção ao cancelar ordens: {e}", exc_info=True)
            return False

    async def has_trailing_stop_order(self, symbol: str) -> bool:
        """
        Verifica se existe uma ordem de trailing stop para o símbolo.
        
        Args:
            symbol: Par de trading
        
        Returns:
            True se existe trailing stop, False caso contrário
        """
        open_orders = await self.get_open_orders(symbol)
        for order in open_orders:
            if order.get('type') == 'TRAILING_STOP_MARKET':
                logger.info(f"✅ [CONECTOR] Trailing stop existente encontrado para {symbol}: Order ID {order.get('orderId')}")
                return True
        return False


    async def start_websocket_stream(self, streams: List[str], callback: Callable):
        """
        Inicia e gerencia uma conexão WebSocket persistente para streams de dados.
        Reconecta automaticamente em caso de falha.
        """
        if not streams or not callback:
            logger.error("❌ [ERRO CONECTOR] 'streams' (lista) e 'callback' são obrigatórios para start_websocket_stream.")
            return

        # Binance suporta múltiplos streams em uma única conexão
        url = f"{self.ws_base_url}/stream?streams={'/'.join(streams)}"
        logger.info(f"🌐 [WEBSOCKET] Tentando conectar ao WebSocket: {url}")

        while not shutdown_event.is_set(): # Loop externo para gerenciar reconexões
            try:
                # Usa uma nova sessão aiohttp para cada tentativa de conexão WS
                # Isso é importante para evitar problemas com sessões corrompidas
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=30) as ws: # heartbeat para manter a conexão viva
                        logger.info("✅ [WEBSOCKET] Conexão WebSocket estabelecida.")
                        self._ws_reconnect_attempts = 0 # Reseta o contador após conexão bem-sucedida

                        # Loop interno para processar mensagens do WebSocket
                        async for msg in ws:
                            if shutdown_event.is_set():
                                logger.info("🛑 [WEBSOCKET] Evento de desligamento recebido. Fechando WebSocket.")
                                break # Sai do loop interno

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    await callback(data) # Chama o callback com os dados processados
                                except json.JSONDecodeError:
                                    logger.error(f"❌ [ERRO WEBSOCKET] Falha ao decodificar JSON do WebSocket: {msg.data[:100]}...")
                                except Exception as e:
                                    logger.error(f"❌ [ERRO WEBSOCKET] Erro no callback de mensagem do WebSocket: {e}", exc_info=True)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"❌ [ERRO WEBSOCKET] Erro de WebSocket recebido: {msg.data}")
                                break # Quebra o loop interno para tentar reconectar
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.warning("⚠️ [WEBSOCKET] Conexão WebSocket fechada pelo servidor.")
                                break # Quebra o loop interno para tentar reconectar
            except aiohttp.ClientConnectorError as e:
                self._ws_reconnect_attempts += 1
                delay = min(self.ws_reconnect_delay_base * (2 ** (self._ws_reconnect_attempts - 1)), 60) # Backoff exponencial, máximo 60s
                logger.error(f"❌ [ERRO WEBSECTOR] Falha de conexão WebSocket ({e}). Tentativa {self._ws_reconnect_attempts}/{self.max_ws_reconnect_attempts}. Reconectando em {delay}s...")
                if self._ws_reconnect_attempts >= self.max_ws_reconnect_attempts:
                    logger.critical("❌ [ERRO WEBSOCKET] Número máximo de tentativas de reconexão WebSocket excedido. Desistindo.")
                    # Em caso de falha crítica, talvez defina o shutdown_event para parar o bot principal
                    shutdown_event.set() 
                await asyncio.sleep(delay)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                if shutdown_event.is_set():
                    break
                logger.warning("⏰ [WEBSOCKET] Timeout ou cancelamento na conexão. Tentando novamente...")
                await asyncio.sleep(5)
            except Exception as e:
                if shutdown_event.is_set():
                    break
                self._ws_reconnect_attempts += 1
                delay = min(self.ws_reconnect_delay_base * (2 ** (self._ws_reconnect_attempts - 1)), 60)
                logger.exception(f"❌ [ERRO WEBSOCKET] Erro inesperado na conexão WebSocket: {e}. Tentativa {self._ws_reconnect_attempts}/{self.max_ws_reconnect_attempts}. Reconectando em {delay}s...")
                if self._ws_reconnect_attempts >= self.max_ws_reconnect_attempts:
                    logger.critical("❌ [ERRO WEBSOCKET] Número máximo de tentativas de reconexão WebSocket excedido. Desistindo.")
                    shutdown_event.set()
                await asyncio.sleep(delay)
            
            if shutdown_event.is_set():
                break # Sai do loop externo se o evento de desligamento for setado

