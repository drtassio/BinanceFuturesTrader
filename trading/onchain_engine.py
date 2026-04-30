import aiohttp
import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from utils.logger import get_logger
from config.settings import Config

logger = get_logger("SentimentEngine")

class OnChainEngine:
    """
    SentimentEngine utilizando a API da Alternative.me (padrão de mercado para bots).
    Fornece o Fear & Greed Index para modulação contextual da IA.
    """
    def __init__(self, config: Config):
        self.config = config
        self.sentiment_cache: Dict[str, Any] = {
            "value": 50,
            "classification": "Neutral",
            "timestamp": datetime.now(timezone.utc),
            "signal": "NEUTRAL",
            "confidence": 0.1
        }
        self.is_running = False
        self._update_thread: Optional[threading.Thread] = None
        self.api_url = "https://api.alternative.me/fng/"
        
        logger.info("🧠 [SENTIMENT] SentimentEngine (Alternative.me) inicializado.")

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        logger.info("🚀 [SENTIMENT] SentimentEngine iniciado (Refresh em background).")

    def stop(self):
        self.is_running = False
        if self._update_thread:
            self._update_thread.join(timeout=5)
        logger.info("🛑 [SENTIMENT] SentimentEngine parado.")

    def _update_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Faz a primeira busca IMEDIATAMENTE na thread de fundo
        try:
            loop.run_until_complete(self._refresh_sentiment())
        except Exception as e:
            logger.warning(f"⚠️ [SENTIMENT] Falha na busca inicial: {e}")

        while self.is_running:
            try:
                loop.run_until_complete(self._refresh_sentiment())
                # Aumentado para cada 15 minutos (900s) para maior precisão
                time.sleep(900)
            except Exception as e:
                logger.error(f"❌ [SENTIMENT] Erro no loop: {e}")
                time.sleep(300)

    async def _refresh_sentiment(self):
        """Busca o índice na Alternative.me e mapeia para sinal estratégico."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and len(data['data']) > 0:
                            item = data['data'][0]
                            value = int(item.get('value', 50))
                            classification = item.get('value_classification', 'Neutral')
                            
                            # Mapeamento ESTRATÉGICO CONTRARIAN
                            # 0-24: Extreme Fear -> BULLISH (Buy the blood)
                            # 25-44: Fear -> NEUTRAL-BULLISH
                            # 45-55: Neutral -> NEUTRAL
                            # 56-74: Greed -> BEARISH (Risk of top)
                            # 75-100: Extreme Greed -> BEARISH (Sell the hype)
                            
                            signal = "NEUTRAL"
                            confidence = 0.5
                            
                            if value <= 24:
                                signal = "BULLISH"
                                confidence = 0.8 + (24 - value) / 24 * 0.2
                            elif value <= 44:
                                signal = "BULLISH"
                                confidence = 0.5 + (44 - value) / 20 * 0.2
                            elif value >= 75:
                                signal = "BEARISH"
                                confidence = 0.8 + (value - 75) / 25 * 0.2
                            elif value >= 56:
                                signal = "BEARISH"
                                confidence = 0.5 + (value - 56) / 20 * 0.2
                            
                            self.sentiment_cache = {
                                "value": value,
                                "classification": classification,
                                "timestamp": datetime.now(timezone.utc),
                                "signal": signal,
                                "confidence": round(confidence, 2)
                            }
                            logger.info(f"🧠 [SENTIMENT] Fear & Greed: {value} ({classification}) -> SINAL: {signal} (Conf: {confidence:.2f})")
                        else:
                            logger.warning("⚠️ [SENTIMENT] Resposta da API sem dados.")
                    else:
                        logger.warning(f"⚠️ [SENTIMENT] Falha na API (Status {response.status}).")
                        
        except Exception as e:
            logger.error(f"❌ [SENTIMENT] Falha ao atualizar sentimento: {e}")

    def get_onchain_sentiment_signal(self, asset: str = 'BTC') -> Dict[str, Any]:
        """Interface para compatibilidade com AIController."""
        sc = self.sentiment_cache
        return {
            "signal": sc["signal"],
            "confidence": sc["confidence"],
            "score": sc["value"],
            "metrics": {"fng_value": sc["value"], "classification": sc["classification"]},
            "reasons": [f"Market Sentiment ({sc['classification']}): {sc['value']}"]
        }