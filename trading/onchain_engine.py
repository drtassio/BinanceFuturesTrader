import requests
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import json
from utils.logger import get_logger
from config.settings import Config

logger = get_logger("OnChainEngine")

class OnChainEngine:
    def __init__(self, config: Config):
        if not isinstance(config, Config):
            raise TypeError("🚨 [ERRO ONCHAIN] 'config' deve ser uma instância da classe Config.")
        self.config = config
        self.api_keys = {
            'santiment': self.config.SANTIMENT_API_KEY,
        }
        self.base_urls = {
            'santiment': "https://api.santiment.net/graphql",
        }
        self.metrics_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = {
            'daily': timedelta(hours=12),
            'hourly': timedelta(minutes=30)
        }
        self.is_running = False
        self._update_thread: Optional[threading.Thread] = None
        logger.info("📡 [ONCHAIN] OnChainEngine inicializado.")

        if not self.api_keys.get('santiment'):
            logger.warning("⚠️ [ALERTE ONCHAIN] SANTIMENT_API_KEY não configurada. Métricas da Santiment não serão coletadas.")

    def start(self):
        if self.is_running:
            logger.warning("⚠️ [ONCHAIN] OnChainEngine já está em execução. Nenhuma ação tomada.")
            return

        logger.info("🚀 [ONCHAIN] Iniciando OnChainEngine...")
        self.is_running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        logger.info("✅ [ONCHAIN] OnChainEngine iniciado com sucesso em thread separada.")

    def stop(self):
        if not self.is_running:
            logger.info("ℹ️ [ONCHAIN] OnChainEngine não está em execução para parar.")
            return

        logger.info("🛑 [ONCHAIN] Parando OnChainEngine...")
        self.is_running = False
        if self._update_thread:
            self._update_thread.join(timeout=10)
            if self._update_thread.is_alive():
                logger.warning("⚠️ [ONCHAIN] Thread de atualização do OnChainEngine não parou dentro do tempo limite.")
            else:
                logger.info("✅ [ONCHAIN] OnChainEngine parado com sucesso.")
        self._update_thread = None

    def _update_loop(self):
        while self.is_running:
            try:
                logger.info("🔄 [ONCHAIN] Atualizando dados on-chain para ativos monitorados...")
                self.get_all_metrics_for_asset('BTC')
                self.get_all_metrics_for_asset('ETH')

                sleep_duration = self.cache_ttl['hourly'].total_seconds()
                logger.info(f"⏳ [ONCHAIN] Próxima atualização on-chain em {sleep_duration / 60:.1f} minutos.")
                time.sleep(sleep_duration)
            except Exception as e:
                logger.exception(f"❌ [ERRO ONCHAIN] Erro no loop de atualização do OnChainEngine: {e}")
                time.sleep(60 * 5)  # Espera 5 minutos em caso de erro

    def get_all_metrics_for_asset(self, asset: str = 'BTC') -> Dict[str, Any]:
        if not isinstance(asset, str) or not asset:
            logger.error("❌ [ERRO ONCHAIN] 'asset' deve ser uma string não vazia para get_all_metrics_for_asset.")
            return {}

        if asset not in self.metrics_cache:
            self.metrics_cache[asset] = {}

        santiment_slug_map = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin'}
        slug = santiment_slug_map.get(asset, None)
        if slug is None:
            logger.warning(f"⚠️ [ONCHAIN] Asset '{asset}' não mapeado para um slug Santiment conhecido. Pulando buscas Santiment.")
            return {k: v['value'] for k, v in self.metrics_cache.get(asset, {}).items()}

        if not self.api_keys.get('santiment'):
            logger.debug(f"ℹ️ [ONCHAIN] Pulando busca Santiment: API Key ausente.")
            return {}

        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)

        metrics_to_fetch_templates = {
            'active_addresses_daily': {
                'metric_name': 'daily_active_addresses', 'interval': '1d', 'cache_type': 'daily',
            },
            #'social_volume_total': {
               # 'metric_name': 'social_volume_total', 'interval': '1h', 'cache_type': 'hourly',#3
           # },
            'price_usd_hourly': {
                'metric_name': 'price_usd', 'interval': '1h', 'cache_type': 'hourly',
            },
        }

        for metric_key, details in metrics_to_fetch_templates.items():
            metric_name = details['metric_name']
            cache_type = details['cache_type']
            query_interval = details['interval']

            if self._is_cache_valid(asset, metric_key, cache_type):
                logger.debug(f"ℹ️ [ONCHAIN] Métrica '{metric_key}' para {asset} está no cache e é válida.")
                continue

            try:
                # A função de fallback agora gerencia suas próprias datas de início
                value = self._fetch_with_date_fallback(
                    asset, metric_name, slug, query_interval, end_date=now_utc
                )

                if value is not None:
                    self.metrics_cache[asset][metric_key] = {
                        'value': value,
                        'timestamp': datetime.utcnow().replace(tzinfo=timezone.utc)
                    }
                    logger.info(f"✅ [ONCHAIN] Métrica '{metric_key}' para {asset} atualizada: {value}.")
                else:
                    logger.warning(f"⚠️ [ONCHAIN] Falha ao obter valor para a métrica '{metric_key}' para {asset}.")

            except Exception as e:
                logger.error(f"❌ [ERRO ONCHAIN] Exceção ao processar métrica '{metric_key}': {e}", exc_info=True)

        return {k: v['value'] for k, v in self.metrics_cache.get(asset, {}).items()}

    def _fetch_with_date_fallback(self, asset: str, metric_name: str, slug: str, interval: str,
                                 end_date: datetime) -> Optional[float]:
        """
        Tenta buscar dados da Santiment, começando com um histórico de 3 dias
        e diminuindo para 2 e 1 dia em caso de erro de restrição de plano.
        """
        for days_ago in [3, 2, 1]:
            start_date = end_date - timedelta(days=days_ago)
            logger.info(f"ℹ️ [ONCHAIN] Tentando buscar '{metric_name}' para {asset} com histórico de {days_ago} dia(s).")
            
            from_date_iso = start_date.isoformat()
            to_date_iso = end_date.isoformat()

            query = f"""
            {{
                getMetric(metric:"{metric_name}"){{
                    timeseriesData(slug:"{slug}", from:"{from_date_iso}", to:"{to_date_iso}", interval:"{interval}"){{
                        datetime
                        value
                    }}
                }}
            }}
            """

            try:
                response_data = self._fetch_santiment_graphql_metric(query)

                if response_data and 'errors' in response_data:
                    # Verifica se o erro é relacionado ao intervalo de datas permitido pelo plano
                    is_date_error = any("allowed interval" in error.get('message', '').lower() for error in response_data['errors'])

                    if is_date_error and days_ago > 1:
                        # Se for um erro de data e ainda houver tentativas com menos dias, continua para a próxima iteração
                        logger.warning(f"⚠️ [ONCHAIN FALLBACK] Restrição de plano para {days_ago} dias. Tentando um período menor.")
                        continue
                    else:
                        # Se for outro tipo de erro GraphQL, ou se for a última tentativa (1 dia), registra o erro e falha
                        logger.error(f"❌ [ONCHAIN] Erro GraphQL final para '{metric_name}': {response_data['errors']}")
                        return None

                elif response_data and 'data' in response_data:
                    timeseries_data = response_data['data'].get('getMetric', {}).get('timeseriesData')
                    if timeseries_data:
                        # Pega o último ponto de dado não nulo da série temporal
                        for point in reversed(timeseries_data):
                            if 'value' in point and point['value'] is not None:
                                logger.info(f"✅ [ONCHAIN] Sucesso ao buscar '{metric_name}' com histórico de {days_ago} dia(s).")
                                return float(point['value'])
                        
                        # Se todos os pontos de dados forem nulos
                        logger.warning(f"⚠️ [ONCHAIN] A busca para '{metric_name}' retornou dados, mas todos os valores são nulos.")
                        return None
                    else:
                        # Se a query teve sucesso mas não retornou dados (ex: métrica não existe para o ativo)
                        logger.warning(f"⚠️ [ONCHAIN] A busca para '{metric_name}' teve sucesso, mas não retornou dados.")
                        return None # Retorna None e encerra o loop para esta métrica

            except Exception as e:
                logger.error(f"❌ [ONCHAIN FALLBACK] Erro inesperado na tentativa de {days_ago} dias para '{metric_name}': {e}", exc_info=True)
                # Permite que o loop continue para a próxima tentativa

        logger.warning(f"⚠️ [ONCHAIN FALLBACK] Falha em todas as tentativas (3, 2 e 1 dia) para a métrica '{metric_name}'.")
        return None

    def _is_cache_valid(self, asset: str, metric: str, cache_type: str) -> bool:
        if asset in self.metrics_cache and metric in self.metrics_cache[asset]:
            last_update = self.metrics_cache[asset][metric]['timestamp']
            if last_update.tzinfo is None:
                last_update = last_update.replace(tzinfo=timezone.utc)
            time_since_last_update = datetime.utcnow().replace(tzinfo=timezone.utc) - last_update
            if cache_type in self.cache_ttl and time_since_last_update < self.cache_ttl[cache_type]:
                return True
        return False

    def _fetch_santiment_graphql_metric(self, query_graphql: str) -> Optional[Dict]:
        api_key = self.api_keys.get('santiment')
        if not api_key:
            return None

        url = self.base_urls.get('santiment')
        if not url:
            return None

        headers = {'Content-Type': 'application/json', 'Authorization': f'Apikey {api_key}'}
        payload = {'query': query_graphql}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()

            if 'errors' in data:
                # O warning agora é tratado na função que chama, para dar um contexto melhor
                return data

            return data

        except requests.exceptions.RequestException as req_err:
            logger.error(f"❌ [ERRO ONCHAIN] Erro na requisição Santiment GraphQL: {req_err}", exc_info=True)
            return None
        except json.JSONDecodeError:
            logger.error(f"❌ [ERRO ONCHAIN] Falha ao decodificar JSON da Santiment GraphQL. Resposta: {response.text[:200]}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"❌ [ERRO ONCHAIN] Erro inesperado em _fetch_santiment_graphql_metric: {e}", exc_info=True)
            return None

    def get_onchain_sentiment_signal(self, asset: str = 'BTC') -> Dict[str, Any]:
        metrics = self.get_all_metrics_for_asset(asset)
        if not metrics:
            logger.warning(f"⚠️ [ONCHAIN] Nenhuma métrica Santiment disponível para {asset}. Sinal NEUTRAL com baixa confiança.")
            return {"signal": "NEUTRAL", "confidence": 0.1, "reason": "No Santiment data available.", "metrics": {}, "reasons": []}

        score = 0
        reasons = []

        sant_active_addresses = metrics.get('active_addresses_daily')
        if sant_active_addresses is not None:
            if sant_active_addresses > 500000:
                score += 1.0
                reasons.append(f"Santiment Endereços Ativos ({sant_active_addresses:.0f}) alto (bullish)")
            elif sant_active_addresses < 100000:
                score -= 1.0
                reasons.append(f"Santiment Endereços Ativos ({sant_active_addresses:.0f}) baixo (bearish)")
            else:
                reasons.append(f"Santiment Endereços Ativos ({sant_active_addresses:.0f}) neutro")

        sant_social_volume = metrics.get('social_volume_total')
        if sant_social_volume is not None:
            if sant_social_volume > 1000:
                score += 0.7
                reasons.append(f"Santiment Volume Social ({sant_social_volume:.0f}) alto (bullish)")
            elif sant_social_volume < 100:
                score -= 0.3
                reasons.append(f"Santiment Volume Social ({sant_social_volume:.0f}) baixo (bearish)")
            else:
                reasons.append(f"Santiment Volume Social ({sant_social_volume:.0f}) neutro")

        sant_price_usd = metrics.get('price_usd_hourly')
        if sant_price_usd is not None:
            pass

        signal = "NEUTRAL"
        if score >= 1.0:
            signal = "BULLISH"
        elif score <= -1.0:
            signal = "BEARISH"

        confidence_base = 0.5
        max_possible_score = 1.7
        confidence = confidence_base + (abs(score) / max_possible_score) * (1.0 - confidence_base)
        confidence = min(1.0, max(0.0, confidence))

        logger.info(f"📈 [ONCHAIN] Sinal de sentimento On-Chain (Santiment) para {asset}: {signal} (Confiança: {confidence:.2f}, Score: {score:.2f}). Razões: {', '.join(reasons)}")

        return {"signal": signal, "confidence": confidence, "score": score, "metrics": metrics, "reasons": reasons}