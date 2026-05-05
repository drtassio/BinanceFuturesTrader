# -----------------------------------------------------------------------------
# ARQUIVO: trading/tape_engine.py
# -----------------------------------------------------------------------------

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Deque, Optional, Any
from utils.logger import get_logger, LOG_LEVEL_DEBUG
from .binance_connector import BinanceConnector

logger = get_logger("TapeEngine", LOG_LEVEL_DEBUG)

@dataclass
class _TapeTick:
    quantity: float
    is_buy: bool
    timestamp: datetime

class TapeMetrics:
    def __init__(self):
        self.aggressor_delta: float = 0.0
        self.volume_imbalance: float = 0.0
        self.vpin: float = 0.0
        self.trade_rate_per_sec: float = 0.0
        self.last_update_time: Optional[datetime] = None
        self.obi: float = 0.0
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        # Volume relativo
        self.relative_volume: float = 1.0  # 1.0 = volume normal
        self.avg_volume_2h: float = 0.0
        # ── LOB Depth Profile (Fase 1) ────────────────────────────────────────
        # depth_slope: nível0 / nível9 — alto = livro raso/frágil; baixo = profundo
        self.depth_slope_bid: float = 1.0
        self.depth_slope_ask: float = 1.0
        # zone_imbalance: pressão imediata (níveis 0-2) vs intenção (níveis 3-9)
        self.zone_imbalance_near: float = 0.0
        self.zone_imbalance_mid: float = 0.0
        # wall: nível com volume > 3× a média — suporte/resistência dinâmica
        self.wall_bid: bool = False
        self.wall_ask: bool = False
        self.wall_bid_price: float = 0.0
        self.wall_ask_price: float = 0.0

class TapeEngine:
    def __init__(self, connector: BinanceConnector, symbols: List[str]):
        self.connector = connector
        self.symbols = [s.upper() for s in symbols]
        self.trades_buffer: Dict[str, Deque[_TapeTick]] = {s: deque(maxlen=5000) for s in self.symbols}
        self.volume_history: Dict[str, Deque[float]] = {s: deque(maxlen=120) for s in self.symbols} # 120 min = 2h
        self.metrics: Dict[str, TapeMetrics] = {s: TapeMetrics() for s in self.symbols}
        self.is_running = False
        self._analysis_task: Optional[asyncio.Task] = None
        self._ws_task: Optional[asyncio.Task] = None
        self.short_window_size = 50 # Janela de trade flow (aumentada para 50 trades)
        self._msg_received_count = 0
        self._last_vol_sync = datetime.utcnow()
        self._current_min_vol = {s: 0.0 for s in self.symbols}

    async def start(self):
        if self.is_running: return
        self.is_running = True
        
        # 1. Warmup Instantâneo via REST (Garante saída do WARMUP em < 1s)
        for symbol in self.symbols:
            await self._warmup_via_rest(symbol)
            
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        
        streams = []
        for symbol in self.symbols:
            s_lower = symbol.lower()
            streams.append(f"{s_lower}@aggTrade")
            streams.append(f"{s_lower}@bookTicker")

        logger.info(f"📡 [TAPE] Iniciando WebSockets (Canais: {streams})")
        self._ws_task = asyncio.create_task(self.connector.start_websocket_stream(streams, self._handle_ws_message))

    async def stop(self):
        self.is_running = False
        if self._ws_task: self._ws_task.cancel()
        if self._analysis_task: self._analysis_task.cancel()

    async def _handle_ws_message(self, msg: Dict):
        """Handler de alta performance para mensagens do WebSocket."""
        self._msg_received_count += 1
        try:
            stream = msg.get('stream', '')
            data = msg.get('data', msg)
            symbol_raw = data.get('s', '').upper()
            
            target_symbol = next((s for s in self.symbols if s.upper() == symbol_raw), None)
            if not target_symbol and stream:
                for s in self.symbols:
                    if s.lower() in stream.lower():
                        target_symbol = s; break
            
            if not target_symbol: return

            # Processar aggTrade
            if 'aggTrade' in stream or data.get('e') == 'aggTrade':
                qty = float(data['q'])
                tick = _TapeTick(
                    quantity=qty,
                    is_buy=not bool(data['m']),
                    timestamp=datetime.fromtimestamp(data.get('T', 0) / 1000.0)
                )
                self.trades_buffer[target_symbol].append(tick)
                self._current_min_vol[target_symbol] += qty
                
            # Processar bookTicker
            elif 'bookTicker' in stream or ('u' in data and 'b' in data):
                bid_q, ask_q = float(data['B']), float(data['A'])
                m = self.metrics[target_symbol]
                m.best_bid, m.best_ask = float(data['b']), float(data['a'])
                if (bid_q + ask_q) > 0:
                    m.obi = (bid_q - ask_q) / (bid_q + ask_q)
        except Exception:
            pass

    async def _warmup_via_rest(self, symbol: str):
        """Busca os últimos 500 trades via REST para inicializar o bot instantaneamente."""
        try:
            logger.info(f"📥 [TAPE] {symbol}: Carregando histórico inicial via REST...")
            # Endpoint: /fapi/v1/aggTrades
            data = await self.connector._make_request("GET", "/fapi/v1/aggTrades", params={"symbol": symbol, "limit": 500})
            if data and isinstance(data, list):
                for t in data:
                    tick = _TapeTick(
                        quantity=float(t['q']),
                        is_buy=not bool(t['m']),
                        timestamp=datetime.fromtimestamp(t['T'] / 1000.0)
                    )
                    self.trades_buffer[symbol].append(tick)
                self._calculate_metrics(symbol)
                logger.info(f"✅ [TAPE] {symbol}: Warmup concluído via REST ({len(data)} trades).")
        except Exception as e:
            logger.warning(f"⚠️ [TAPE] Falha no warmup REST para {symbol}: {e}")

    async def _analysis_loop(self):
        while self.is_running:
            try:
                for symbol in self.symbols:
                    # 1. Se o buffer tem dados, calcula métricas
                    if len(self.trades_buffer[symbol]) >= 5:
                        self._calculate_metrics(symbol)
                    
                    # 2. FALLBACK TRADES: Se não recebemos trades via WS nos últimos 15s
                    m = self.metrics[symbol]
                    now = datetime.utcnow()
                    if not m.last_update_time or (now - m.last_update_time).total_seconds() > 15:
                        await self._poll_rest_trades(symbol)
                    
                    # 3. FALLBACK OBI: Se o OBI via WS está parado, busca via REST (rpiDepth)
                    await self._fetch_rest_depth(symbol)
                        
                await asyncio.sleep(3) # Intervalo seguro para não estourar limite de peso da API
            except Exception as e:
                logger.error(f"Erro no loop de análise: {e}")
                await asyncio.sleep(5)

    async def _poll_rest_trades(self, symbol: str):
        """Busca trades recentes via REST se o WebSocket falhar."""
        try:
            data = await self.connector._make_request("GET", "/fapi/v1/aggTrades", params={"symbol": symbol, "limit": 50})
            if data and isinstance(data, list):
                count = 0
                for t in data:
                    tick = _TapeTick(
                        quantity=float(t['q']),
                        is_buy=not bool(t['m']),
                        timestamp=datetime.fromtimestamp(t['T'] / 1000.0)
                    )
                    # Só adiciona se for mais novo que o último no buffer (simplificado)
                    self.trades_buffer[symbol].append(tick)
                    count += 1
                if count > 0:
                    self._calculate_metrics(symbol)
                    logger.debug(f"🔄 [TAPE] {symbol}: Sincronizado via REST ({count} trades).")
        except Exception as e:
            logger.debug(f"Falha ao pollear trades: {e}")

    async def _fetch_rest_depth(self, symbol: str):
        """Busca o Order Book (RPI) via REST conforme sua documentação."""
        try:
            # Usando /fapi/v1/rpiDepth se disponível, senão /fapi/v1/depth
            endpoint = "/fapi/v1/rpiDepth" 
            data = await self.connector._make_request("GET", endpoint, params={"symbol": symbol, "limit": 1000}, signed=False)
            
            # Se rpiDepth falhar (erro 404), tenta o depth comum
            if not data:
                data = await self.connector._make_request("GET", "/fapi/v1/depth", params={"symbol": symbol, "limit": 100}, signed=False)
            
            if data and 'bids' in data and 'asks' in data:
                bids = data['bids']
                asks = data['asks']
                if not bids or not asks:
                    return

                best_bid_qty = float(bids[0][1])
                best_ask_qty = float(asks[0][1])
                m = self.metrics[symbol]
                m.best_bid, m.best_ask = float(bids[0][0]), float(asks[0][0])
                total_qty = best_bid_qty + best_ask_qty
                if total_qty > 0:
                    m.obi = (best_bid_qty - best_ask_qty) / total_qty
                    m.last_update_time = datetime.utcnow()

                # ── Perfil de profundidade com os níveis brutos disponíveis ──
                self._calculate_depth_profile(m, bids, asks)

        except Exception:
            pass

    def _calculate_depth_profile(self, m: 'TapeMetrics', bids: list, asks: list):
        """
        Calcula métricas de perfil de profundidade do LOB a partir dos níveis brutos.

        Usa os primeiros 20 níveis disponíveis.

        Métricas produzidas:
          depth_slope_bid/ask   — vol_nível0 / vol_nível9  (livro raso vs profundo)
          zone_imbalance_near   — imbalance nos níveis 0-2 (pressão imediata)
          zone_imbalance_mid    — imbalance nos níveis 3-9 (intenção de curto prazo)
          wall_bid / wall_ask   — nível com vol > 3× média (suporte/resistência)
          wall_bid/ask_price    — preço do muro detectado
        """
        try:
            n = min(20, len(bids), len(asks))
            if n < 5:
                return

            bid_vols = [float(bids[i][1]) for i in range(n)]
            ask_vols = [float(asks[i][1]) for i in range(n)]

            # ── 1. Depth Slope ─────────────────────────────────────────────────
            # Compara o nível mais próximo (0) com o nível 9 (distante).
            # Slope alto (>5): liquidez concentrada no topo — livro frágil, fácil de mover.
            # Slope baixo (<2): liquidez distribuída — livro resistente, slippage alto.
            ref = min(9, n - 1)
            m.depth_slope_bid = bid_vols[0] / (bid_vols[ref] + 1e-8)
            m.depth_slope_ask = ask_vols[0] / (ask_vols[ref] + 1e-8)

            # ── 2. Zone Imbalance ──────────────────────────────────────────────
            # Near (0-2): quem está comprando/vendendo agora.
            # Mid  (3-9): quem está posicionado para o próximo movimento.
            near_end = min(3, n)
            mid_end  = min(10, n)

            bid_near = sum(bid_vols[:near_end])
            ask_near = sum(ask_vols[:near_end])
            tot_near = bid_near + ask_near
            m.zone_imbalance_near = (bid_near - ask_near) / tot_near if tot_near > 0 else 0.0

            bid_mid = sum(bid_vols[near_end:mid_end])
            ask_mid = sum(ask_vols[near_end:mid_end])
            tot_mid = bid_mid + ask_mid
            m.zone_imbalance_mid = (bid_mid - ask_mid) / tot_mid if tot_mid > 0 else 0.0

            # ── 3. Wall Detection ──────────────────────────────────────────────
            # Nível com volume > 3× a média dos demais → muro de suporte/resistência.
            # Pega o mais próximo do melhor preço (índice mais baixo).
            avg_bid = sum(bid_vols) / len(bid_vols)
            avg_ask = sum(ask_vols) / len(ask_vols)
            wall_thr = 3.0

            m.wall_bid = False
            m.wall_ask = False
            m.wall_bid_price = 0.0
            m.wall_ask_price = 0.0

            for i in range(n):
                if not m.wall_bid and bid_vols[i] > avg_bid * wall_thr:
                    m.wall_bid       = True
                    m.wall_bid_price = float(bids[i][0])
                if not m.wall_ask and ask_vols[i] > avg_ask * wall_thr:
                    m.wall_ask       = True
                    m.wall_ask_price = float(asks[i][0])
                if m.wall_bid and m.wall_ask:
                    break

        except Exception:
            pass   # falha silenciosa — métricas ficam com defaults seguros

    def _calculate_metrics(self, symbol: str):
        m = self.metrics[symbol]
        trades = list(self.trades_buffer[symbol])
        if not trades: return
        
        # Sincroniza histórico de volume a cada minuto
        now = datetime.utcnow()
        if (now - self._last_vol_sync).total_seconds() >= 60:
            for s in self.symbols:
                self.volume_history[s].append(self._current_min_vol[s])
                self._current_min_vol[s] = 0.0
            self._last_vol_sync = now

        recent = trades[-self.short_window_size:]
        buys = sum(t.quantity for t in recent if t.is_buy)
        sells = sum(t.quantity for t in recent if not t.is_buy)
        vol = buys + sells
        
        if vol > 0:
            m.aggressor_delta = (buys - sells)
            m.volume_imbalance = (buys - sells) / vol
            m.vpin = abs(m.volume_imbalance)
            
        # Taxa real de trades (usa timestamps reais — mais preciso que estimativa fixa)
        if len(recent) >= 2:
            dt_recent = (recent[-1].timestamp - recent[0].timestamp).total_seconds()
            if dt_recent > 0:
                m.trade_rate_per_sec = len(recent) / dt_recent
                avg_vol_per_trade    = vol / len(recent)
                estimated_min_vol    = avg_vol_per_trade * m.trade_rate_per_sec * 60
            else:
                estimated_min_vol = vol * 12   # fallback: 50 trades em ~5s → ×12 = 1 min
        else:
            estimated_min_vol = vol * 12

        # Taxa global (buffer completo) — para trade_rate_per_sec preciso
        if len(trades) >= 2:
            dt_all = (trades[-1].timestamp - trades[0].timestamp).total_seconds()
            if dt_all > 0:
                m.trade_rate_per_sec = len(trades) / dt_all

        # Cálculo de Volume Relativo (RVol) — comparado à média dos últimos 120 min
        if self.volume_history[symbol]:
            m.avg_volume_2h = sum(self.volume_history[symbol]) / len(self.volume_history[symbol])
            if m.avg_volume_2h > 0:
                m.relative_volume = estimated_min_vol / m.avg_volume_2h

        m.last_update_time = datetime.utcnow()

    def get_market_pulse(self, symbol: str) -> Dict[str, Any]:
        m = self.metrics.get(symbol)
        if not m or not m.last_update_time:
            return {
                "pulse": "WARMUP...", "score": 0.0,
                "obi": m.obi if m else 0.0, "relative_volume": 1.0,
                "depth_slope": 1.0, "depth_quality": 1.0,
                "zone_imbalance_near": 0.0, "zone_imbalance_mid": 0.0,
                "wall_bid": False, "wall_ask": False,
                "wall_bid_price": 0.0, "wall_ask_price": 0.0,
            }

        rvol = m.relative_volume   # 1.0 = normal, 2.5+ = institucional

        # ── Score base: imbalance direcional + pressão do livro ───────────────
        base_score = (m.volume_imbalance * 0.7) + (m.obi * 0.3)

        # ── Amplificador por volume relativo (baleias amplificam) ─────────────
        if rvol >= 2.5:
            amp = 1.5
        elif rvol >= 1.5:
            amp = 1.25
        else:
            amp = 1.0

        score = base_score * amp

        # ── Penalidade de qualidade de livro (Fase 1) ─────────────────────────
        # Livro raso = liquidez concentrada só no nível 0 → ruído alto, possível
        # manipulação por 1 ordem grande. Penaliza o score para não gerar sinais
        # EXTREME/STRONG com base em microestrutura frágil.
        #
        #   depth_slope > 8 : quase toda liquidez no nível 0 → penaliza 25%
        #   depth_slope 4-8 : livro moderado                 → penaliza 10%
        #   depth_slope < 4 : livro distribuído              → sem penalidade
        avg_slope = (m.depth_slope_bid + m.depth_slope_ask) / 2.0
        if avg_slope > 8.0:
            depth_quality = 0.75
        elif avg_slope > 4.0:
            depth_quality = 0.90
        else:
            depth_quality = 1.0

        score     = score * depth_quality
        abs_score = abs(score)
        bullish   = score > 0

        # ── Classificação ─────────────────────────────────────────────────────
        if rvol >= 2.5 and abs_score >= 0.35:
            p = "EXTREME_BULLISH" if bullish else "EXTREME_BEARISH"
        elif abs_score >= 0.40:
            p = "STRONG_BULLISH"  if bullish else "STRONG_BEARISH"
        elif abs_score >= 0.15:
            p = "BULLISH"         if bullish else "BEARISH"
        else:
            p = "NEUTRAL"

        return {
            "pulse":                p,
            "score":                round(score, 4),
            "obi":                  round(m.obi, 4),
            "vpin":                 round(m.vpin, 4),
            "delta":                round(m.aggressor_delta, 2),
            "relative_volume":      round(rvol, 2),
            # ── Depth profile (Fase 1) ────────────────────────────────────────
            "depth_slope":          round(avg_slope, 2),
            "depth_quality":        round(depth_quality, 2),
            "zone_imbalance_near":  round(m.zone_imbalance_near, 4),
            "zone_imbalance_mid":   round(m.zone_imbalance_mid, 4),
            "wall_bid":             m.wall_bid,
            "wall_ask":             m.wall_ask,
            "wall_bid_price":       round(m.wall_bid_price, 2),
            "wall_ask_price":       round(m.wall_ask_price, 2),
        }