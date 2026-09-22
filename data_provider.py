# -----------------------------------------------------------------------------
# ARQUIVO: data_provider.py
# -----------------------------------------------------------------------------

"""
Provedor de Dados de Mercado (Data Provider).

Este módulo é a fonte centralizada para todos os dados de mercado necessários
pelo sistema. Ele interage com o `BinanceConnector` para buscar dados brutos,
gerencia um cache local para performance, e utiliza o `FeatureEngineeringPipeline`
para transformar os dados brutos em features para a IA.
"""

import pandas as pd
import numpy as np  # 🔧 Fix: Added missing numpy import
import asyncio
import time  # 🔧 Fix: Added missing time import
from datetime import datetime, timedelta, timezone 
from typing import Dict, Any, Optional, List, Tuple
import sys 
import os
import pickle # Para salvar DataFrames

from utils.logger import get_logger , LOG_LEVEL_DEBUG 
from trading.binance_connector import BinanceConnector
from feature_engineering import FeatureEngineeringPipeline 
from config.settings import DataConfig, AIConfig, TradingConfig 

logger = get_logger("DataProvider", LOG_LEVEL_DEBUG)

class DataProvider:
    """
    Fornece dados de mercado brutos e transformados para outros componentes do bot.
    Gerencia o cache de dados e a integração com o pipeline de engenharia de features.
    """
    # Minutos por barra, para converter a janela pedida em numero de barras.
    _TIMEFRAME_MINUTES = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
        '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080,
    }

    def _validate_historical_coverage(self, frame, timeframe, start, end):
        """A janela historica recebida cobre de fato o periodo pedido?

        Um download truncado nao levanta erro: a exchange simplesmente devolve
        menos barras. O treino entao roda sobre um pedaco do periodo, e o bot
        ao vivo decide com um historico mais curto do que os indicadores
        precisam, sem que nada apareca como falha.

        Devolve (valido, relatorio); o relatorio sempre traz os numeros, mesmo
        quando reprova, para o log dizer o que faltou.
        """
        import pandas as _pd

        minutes = self._TIMEFRAME_MINUTES.get(str(timeframe).lower())
        min_ratio = float(getattr(self.data_config, 'HISTORICAL_MIN_COVERAGE_RATIO', 0.995))
        max_gap_multiplier = float(getattr(self.data_config, 'HISTORICAL_MAX_GAP_MULTIPLIER', 3.0))

        report = {
            'timeframe': str(timeframe),
            'requested_start': str(start),
            'requested_end': str(end),
            'rows': 0 if frame is None else int(len(frame)),
            'coverage_ratio': 0.0,
            'start_ok': False,
            'end_ok': False,
            'largest_gap_bars': 0.0,
            'reason': '',
        }
        if frame is None or len(frame) == 0 or minutes is None:
            report['reason'] = 'frame vazio' if minutes is not None else f'timeframe desconhecido: {timeframe}'
            return False, report

        index = _pd.DatetimeIndex(frame.index)
        start_ts, end_ts = _pd.Timestamp(start), _pd.Timestamp(end)
        # Alinha fuso: comparar um indice consciente com um timestamp ingenuo
        # levantaria excecao, e a origem do desalinhamento costuma ser o chamador.
        if index.tz is not None and start_ts.tz is None:
            start_ts, end_ts = start_ts.tz_localize(index.tz), end_ts.tz_localize(index.tz)
        elif index.tz is None and start_ts.tz is not None:
            index = index.tz_localize(None)
            start_ts, end_ts = start_ts.tz_localize(None), end_ts.tz_localize(None)

        bar = _pd.Timedelta(minutes=minutes)
        expected = max(1, int((end_ts - start_ts) / bar))
        report['expected_rows'] = expected
        report['coverage_ratio'] = round(min(1.0, len(index) / expected), 6)
        # Uma barra de tolerancia em cada ponta: a exchange pode devolver o
        # candle seguinte ou omitir o ultimo ainda em formacao.
        report['start_ok'] = bool(index.min() <= start_ts + bar)
        report['end_ok'] = bool(index.max() >= end_ts - 2 * bar)

        if len(index) > 1:
            gaps = index.to_series().diff().dropna()
            report['largest_gap_bars'] = round(float(gaps.max() / bar), 3) if len(gaps) else 0.0

        problems = []
        if report['coverage_ratio'] < min_ratio:
            problems.append('cobertura %.4f abaixo de %.4f' % (report['coverage_ratio'], min_ratio))
        if not report['start_ok']:
            problems.append('inicio faltando')
        if not report['end_ok']:
            problems.append('fim faltando')
        if report['largest_gap_bars'] > max_gap_multiplier:
            problems.append('buraco de %.1f barras' % report['largest_gap_bars'])
        report['reason'] = '; '.join(problems) if problems else 'ok'
        return (not problems), report

    def __init__(self, connector: BinanceConnector, feature_pipeline: FeatureEngineeringPipeline):
        if not isinstance(connector, BinanceConnector):
            raise TypeError("🚨 [ERRO DATA PROVIDER] 'connector' deve ser uma instância de BinanceConnector.")
        if not isinstance(feature_pipeline, FeatureEngineeringPipeline):
            raise TypeError("🚨 [ERRO DATA PROVIDER] 'feature_pipeline' deve ser uma instância de FeatureEngineeringPipeline.")

        self.connector = connector
        self.feature_pipeline = feature_pipeline
        self.data_config = DataConfig()
        self.ai_config = AIConfig()
        self.trading_config = TradingConfig() # Acessa as configurações de timeframes

        # Cache de dados brutos de klines: {'symbol_interval': pd.DataFrame}
        self.data_cache: Dict[str, pd.DataFrame] = {} 
        self._lock = asyncio.Lock() # Garante acesso thread-safe ao cache

        # Define os timeframes a serem buscados, combinando o primário e os secundários/micro
        self.all_timeframes = self.trading_config.ALL_TRADING_TIMEFRAMES
        
        # Parâmetros de Rate Limiting para coleta histórica (ajustados para Binance Futures)
        self.historical_fetch_delay = 0.5 # Atraso de 0.5s entre as requisições para evitar rate limit 
        
        # Diretório para salvar dados históricos persistentes
        self.HISTORICAL_DATA_DIR = os.path.join(os.getcwd(), "logs", "historical_data")
        os.makedirs(self.HISTORICAL_DATA_DIR, exist_ok=True)

        logger.info(f"📊 [DATA PROVIDER] DataProvider inicializado. Cache de dados e configurações carregadas.")

    async def _fetch_and_format_data(
        self, 
        symbol: str, 
        interval: str, 
        limit: int, 
        start_ts: Optional[int] = None, 
        end_ts: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetches and formats kline data with STATIONARY transformations.
        
        CRITICAL CHANGE (Audit Fix P0):
        - NO LONGER returns raw OHLC prices
        - Returns log-returns and percentage-based features
        - Enforces mathematical stationarity for deep learning
        
        Mathematical Justification:
            Raw prices P_t ~ I(1) (unit root, non-stationary)
            Log-returns r_t = log(P_t / P_{t-1}) ~ I(0) (stationary)
            
        Returns:
            DataFrame with stationary features:
            - log_return: ln(close / prev_close)
            - log_return_volume: ln(volume / prev_volume)  
            - hl_range_pct: (high - low) / close
            - oc_move_pct: (close - open) / open
            - hc_wick_upper: (high - close) / close
            - lc_wick_lower: (close - low) / close
            - realized_vol_20: rolling std of log_return (20 periods)
            - realized_vol_100: rolling std of log_return (100 periods)
            - close_reference: ONLY for position sizing (NOT training)
        """
        if not symbol or not interval or limit <= 0:
            logger.error("❌ Invalid parameters for _fetch_and_format_data")
            return None

        logger.debug(f"🔍 Fetching raw klines for {symbol}-{interval} (limit={limit})")
        try:
            raw_klines = await self.connector.get_kline_data(symbol, interval, limit, start_ts, end_ts)
            
            if not raw_klines:
                logger.warning(f"⚠️ No klines returned for {symbol}-{interval}")
                return None

            columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            df = pd.DataFrame(raw_klines, columns=columns)
            
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            # Convert OHLCV and Binance's taker-buy field to numeric.
            # A API devolve tudo como texto; as colunas de fluxo mantidas mais
            # abaixo precisam ser numericas para as features de tape.
            for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume',
                        'quote_asset_volume', 'number_of_trades', 'taker_buy_quote_asset_volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

            if df.empty:
                logger.warning(f"⚠️ DataFrame empty after cleaning for {symbol}-{interval}")
                return None

            # ========== CRITICAL: STATIONARITY INJECTION (Audit Fix P0) ==========
            logger.debug(f"🧮 [STATIONARITY] Applying log-returns transformation for {symbol}-{interval}")
            
            # 1. Log-returns (primary stationary feature)
            # Mathematical: r_t = ln(P_t / P_{t-1})
            df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
            df['log_return_volume'] = np.log(df['volume'] / (df['volume'].shift(1) + 1e-10)).fillna(0)
            
            # 2. Percentage-based features (scale-invariant)
            df['hl_range_pct'] = ((df['high'] - df['low']) / (df['close'] + 1e-10)).fillna(0)
            df['oc_move_pct'] = ((df['close'] - df['open']) / (df['open'] + 1e-10)).fillna(0)
            df['hc_wick_upper'] = ((df['high'] - df['close']) / (df['close'] + 1e-10)).fillna(0)
            df['lc_wick_lower'] = ((df['close'] - df['low']) / (df['close'] + 1e-10)).fillna(0)
            
            # 3. Realized volatility (rolling std of log-returns)
            df['realized_vol_20'] = df['log_return'].rolling(20, min_periods=5).std().fillna(0)
            df['realized_vol_100'] = df['log_return'].rolling(100, min_periods=20).std().fillna(0)

            # Historical taker flow is available in Futures klines.  These
            # features describe completed-candle aggression; they are not a
            # synthetic order book and therefore remain causal.
            sell_volume = (df['volume'] - df['taker_buy_base_asset_volume']).clip(lower=0.0)
            delta = df['taker_buy_base_asset_volume'] - sell_volume
            df['aggressor_imbalance'] = delta / df['volume'].clip(lower=1e-12)
            df['taker_buy_ratio'] = df['taker_buy_base_asset_volume'] / df['volume'].clip(lower=1e-12)
            delta_mean = delta.rolling(32, min_periods=8).mean()
            delta_std = delta.rolling(32, min_periods=8).std().replace(0, np.nan)
            df['aggressor_delta_z_32'] = ((delta - delta_mean) / delta_std).fillna(0.0).clip(-8, 8).astype('float32')
            
            # 4. Keep close ONLY as reference for position sizing (NOT for training)
            df['close_reference'] = df['close']
            
            # 5. Keep stationary features AND raw prices (for feature engineering)
            # Raw prices will be dropped in TrendPredictor before training
            features_to_keep = [
                'open', 'high', 'low', 'close', 'volume',  # ✅ RESTORED for Feature Engineering
                'log_return', 'log_return_volume',
                'hl_range_pct', 'oc_move_pct', 
                'hc_wick_upper', 'lc_wick_lower',
                'realized_vol_20', 'realized_vol_100',
                'close_reference', 'aggressor_imbalance', 'taker_buy_ratio', 'aggressor_delta_z_32',
                # Colunas brutas do tape. Sao elas que alimentam as features de
                # fluxo (tamanho medio de trade, intensidade, price improvement
                # de compradores e vendedores) no modulo causal compartilhado com
                # o treino. Descarta-las aqui zerava cinco features ao vivo que o
                # especialista viu com valores reais durante todo o treino.
                'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            ]
            
            # 6. Clean infinities and remaining NaNs
            # Ensure we only keep columns that exist (in case some were dropped earlier)
            existing_cols = [c for c in features_to_keep if c in df.columns]
            result_df = df[existing_cols].replace([np.inf, -np.inf], 0).fillna(0)
            
            logger.debug(
                f"✅ [STATIONARITY] Stationary features created for {symbol}-{interval}. "
                f"Raw OHLCV kept for Feature Engineering. Shape: {result_df.shape}"
            )
            
            return result_df

        except Exception as e:
            logger.error(f"❌ [ERRO DATA PROVIDER] Falha ao buscar e formatar dados para {symbol}-{interval}: {e}", exc_info=True)
            return None

    # No arquivo data_provider.py

    async def get_latest_features(self, symbol: str, tape_metrics: Optional[Dict[str, Any]] = None, sentiment_metrics: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """
        Obtém o DataFrame mais recente com features para o loop de trading.
        Busca dados de múltiplos timeframes e os combina.
        """
        if not symbol:
            logger.error("❌ [ERRO DATA PROVIDER] 'symbol' é obrigatório para get_latest_features.")
            return None

        async with self._lock:
            raw_dfs_multi_tf: Dict[str, pd.DataFrame] = {}

            # Loop para buscar e atualizar o cache para todos os timeframes
            for interval in self.all_timeframes:
                cache_key = f"{symbol}_{interval}"
                required_data_points = self.ai_config.LOOKBACK_WINDOW_MAIN + 52

                if cache_key not in self.data_cache or len(self.data_cache[cache_key]) < required_data_points:
                    initial_df = await self._fetch_and_format_data(symbol, interval, limit=required_data_points)
                    if initial_df is None:
                        logger.warning(f"⚠️ [DATA PROVIDER] Não foi possível buscar dados iniciais para {cache_key}.")
                        return None
                    self.data_cache[cache_key] = initial_df

                last_timestamp_in_cache = int(self.data_cache[cache_key].index[-1].timestamp() * 1000)
                
                # Calcula quantas velas estão faltando desde o último timestamp no cache
                current_time = int(time.time() * 1000)
                interval_ms = self._get_interval_milliseconds(interval)
                expected_candles_since_last = max(1, (current_time - last_timestamp_in_cache) // interval_ms)
                
                # Busca exatamente o número de velas ausentes para garantir continuidade
                # Adiciona um buffer de segurança (máximo 100 velas para evitar requests muito grandes)
                candles_to_fetch = min(expected_candles_since_last + 5, 100)
                
                latest_klines_df = await self._fetch_and_format_data(symbol, interval, limit=candles_to_fetch, start_ts=last_timestamp_in_cache)

                if latest_klines_df is not None and not latest_klines_df.empty:
                    # [FIX A9] Filtra a vela em formação ANTES de colocar no cache.
                    # Se fizermos isso apenas na extração de features, a vela parcial
                    # pode sobrescrever a versão antiga fechada via keep='last', e 
                    # na próxima chamada seu timestamp pode parecer fechado.
                    closed_mask_new = (latest_klines_df.index.asi8 // 10**6 + interval_ms) <= current_time
                    latest_klines_df = latest_klines_df.loc[closed_mask_new]
                    
                    if not latest_klines_df.empty:
                        combined_df = pd.concat([self.data_cache[cache_key], latest_klines_df])
                        self.data_cache[cache_key] = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()

                cached = self.data_cache[cache_key]
                # Apenas garantia redundante (o cache já deve estar limpo)
                closed_mask = (cached.index.asi8 // 10**6 + interval_ms) <= current_time
                data_for_features = cached.loc[closed_mask].tail(required_data_points)
                if len(data_for_features) > 0:
                    raw_dfs_multi_tf[interval] = data_for_features
                    
                    # [STALENESS CHECK] Se o candle mais recente tiver mais de 2 períodos de idade, loga aviso
                    last_time = data_for_features.index[-1].timestamp() * 1000
                    if current_time - last_time > (interval_ms * 2):
                        logger.warning(f"⚠️ [DATA PROVIDER] Dados stale detectados em {symbol}-{interval}. Última vela: {data_for_features.index[-1]}")

            if not raw_dfs_multi_tf:
                logger.error("❌ [DATA PROVIDER] Nenhuma kline válida encontrada para criar features.")
                return None
            
            # Adicionamos a palavra-chave 'await' aqui, pois create_features é uma função assíncrona.
            # Funding is published every eight hours. Align the last rate
            # already published with each primary-timeframe candle.
            primary_tf = self.trading_config.PRIMARY_TIMEFRAME_TRADING
            primary_df = raw_dfs_multi_tf.get(primary_tf)
            if primary_df is not None and not primary_df.empty:
                try:
                    start_ms = int((primary_df.index.min() - pd.Timedelta(days=1)).timestamp() * 1000)
                    end_ms = int(primary_df.index.max().timestamp() * 1000)
                    funding_rows = await self.connector.get_funding_rate_history(symbol, start_ms, end_ms)
                    if funding_rows:
                        funding = pd.DataFrame(funding_rows)
                        funding.index = pd.to_datetime(funding['fundingTime'], unit='ms', utc=True)
                        series = funding.sort_index()['fundingRate'].astype(float)
                        primary_df = primary_df.copy()
                        primary_df['funding_rate'] = series.reindex(primary_df.index, method='ffill').fillna(0.0)
                        rates = primary_df['funding_rate']
                        rate_std = rates.rolling(32, min_periods=4).std().replace(0.0, np.nan)
                        primary_df['funding_rate_z_32'] = ((rates - rates.rolling(32, min_periods=4).mean()) / rate_std).fillna(0.0).clip(-8, 8).astype('float32')
                        raw_dfs_multi_tf[primary_tf] = primary_df
                except Exception as exc:
                    logger.warning(f"[DATA PROVIDER] Funding unavailable; using neutral 0: {exc}")

            featured_df = await self.feature_pipeline.create_features(
                raw_dfs_multi_tf,
                symbol,
                self.trading_config.PRIMARY_TIMEFRAME_TRADING,
                fit_scaler=False,
                tape_metrics=tape_metrics,
                onchain_metrics=sentiment_metrics
            )
                
            if featured_df is None or featured_df.empty:
                logger.error(f"❌ [ERRO DATA PROVIDER] FeatureEngineeringPipeline retornou um DataFrame vazio ou None.")
                return None

            logger.debug(f"✅ [DATA PROVIDER] Features multi-timeframe geradas. Shape: {featured_df.shape}.")
            return featured_df
    
    def _get_historical_file_path(self, symbol: str, interval: str) -> str:
        """Gera o caminho do arquivo para o cache histórico persistente."""
        return os.path.join(self.HISTORICAL_DATA_DIR, f"{symbol}_{interval}_historical.pkl")

    def _load_historical_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Tenta carregar dados históricos persistentes do disco."""
        filepath = self._get_historical_file_path(symbol, interval)
        if os.path.exists(filepath):
            try:
                # Usamos pickle para DataFrames com índices complexos (datetime)
                data = pd.read_pickle(filepath)
                # [FIX B1] Verifica se o cache é obsoleto (última vela muito antiga).
                # Um buraco de meses no cache não será preenchido adequadamente pelo _fetch_historical_batch
                # de forma transparente (pode gerar ffill() na frente).
                if not data.empty:
                    last_time = data.index[-1].timestamp()
                    current_time = time.time()
                    if current_time - last_time > (30 * 24 * 60 * 60):  # 30 dias
                        logger.warning(f"⚠️ [DATA PROVIDER] Cache {filepath} obsoleto (último dado em {data.index[-1]}). Descartando.")
                        data = pd.DataFrame()
                        
                if not data.empty:
                    logger.info(f"✅ [DATA PROVIDER] Dados históricos carregados de '{filepath}'. Total de linhas: {len(data)}.")
                    return data
            except Exception as e:
                logger.warning(f"⚠️ [DATA PROVIDER] Cache corrompido ou incompatível em '{filepath}': {e}. Removendo cache corrompido...")
                try:
                    os.remove(filepath)
                except OSError:
                    pass

        # Fallback para data/historical em formato parquet (versionado no Git)
        parquet_path = os.path.join("data", "historical", f"{symbol}_{interval}_historical.parquet")
        if os.path.exists(parquet_path):
            try:
                data = pd.read_parquet(parquet_path)
                logger.info(f"✅ [DATA PROVIDER] Dados históricos (parquet) carregados de '{parquet_path}'. Total de linhas: {len(data)}.")
                return data
            except Exception as e:
                logger.warning(f"⚠️ [DATA PROVIDER] Falha ao carregar parquet '{parquet_path}': {e}")

        # Fallback para dados particionados (ex: 1m part1 + part2)
        p1 = os.path.join("data", "historical", f"{symbol}_{interval}_part1.parquet")
        p2 = os.path.join("data", "historical", f"{symbol}_{interval}_part2.parquet")
        if os.path.exists(p1) and os.path.exists(p2):
            try:
                d1 = pd.read_parquet(p1)
                d2 = pd.read_parquet(p2)
                data = pd.concat([d1, d2]).sort_index()
                data = data[~data.index.duplicated(keep='first')]
                logger.info(f"✅ [DATA PROVIDER] Dados históricos particionados ({interval}) reconstruídos com sucesso. Total de linhas: {len(data)}.")
                return data
            except Exception as e:
                logger.warning(f"⚠️ [DATA PROVIDER] Falha ao reconstruir partes parquet: {e}")

        return None

    def _get_interval_milliseconds(self, interval: str) -> int:
        """Converte um intervalo de tempo (ex: '15m', '1h', '4h') para milissegundos."""
        if 'm' in interval:
            minutes = int(interval.replace('m', ''))
            return minutes * 60 * 1000
        elif 'h' in interval:
            hours = int(interval.replace('h', ''))
            return hours * 60 * 60 * 1000
        elif 'd' in interval:
            days = int(interval.replace('d', ''))
            return days * 24 * 60 * 60 * 1000
        else:
            return 60 * 1000  # Fallback para 1 minuto

    def _save_historical_data(self, df: pd.DataFrame, symbol: str, interval: str):
        """Salva dados históricos para persistência no disco."""
        filepath = self._get_historical_file_path(symbol, interval)
        try:
            # Garante que o diretório exista antes de salvar
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Salva o DataFrame em pickle
            df.to_pickle(filepath)
            logger.info(f"💾 [DATA PROVIDER] Dados históricos para {symbol}-{interval} salvos em '{filepath}'.")
        except Exception as e:
            logger.error(f"❌ [ERRO DATA PROVIDER] Falha ao salvar dados históricos em '{filepath}': {e}", exc_info=True)

    async def get_data_for_training(self, symbol: str, start_date: str, end_date: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Obtém dados históricos para treinamento, buscando em todos os timeframes configurados.
        Implementa persistência e busca incremental.
        """
        if not symbol or not start_date or not end_date:
            logger.error("❌ [ERRO DATA PROVIDER] Todos os parâmetros são obrigatórios para get_data_for_training.")
            return None

        # Converte datas para timestamps UTC
        try:
            start_dt_utc = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            end_dt_utc = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError as e:
            logger.error(f"❌ [ERRO DATA PROVIDER] Formato de data inválido ('%Y-%m-%d'): {e}. Usar 'YYYY-MM-DD'.")
            return None

        logger.info(f"⏳ [DATA PROVIDER] Iniciando busca de dados históricos para treinamento para todos os timeframes de {start_dt_utc.strftime('%Y-%m-%d')} a {end_dt_utc.strftime('%Y-%m-%d')}...")

        # Dicionário para armazenar os resultados por timeframe
        raw_dfs_multi_tf: Dict[str, pd.DataFrame] = {}

        # Busca dados para cada timeframe em paralelo
        tasks = []
        for interval in self.all_timeframes:
            
            # 1. Tenta carregar dados persistentes
            historical_df = self._load_historical_data(symbol, interval)
            
            fetch_start_ts = int(start_dt_utc.timestamp() * 1000)
            
            if historical_df is not None and not historical_df.empty:
                # Se dados existentes, encontra o último timestamp e começa a busca a partir dele
                last_saved_timestamp = historical_df.index[-1]
                
                # Para evitar duplicatas e garantir a integridade do candle de transição,
                # começamos a buscar a partir do próximo intervalo após o último salvo
                
                # Converte o intervalo de string para timedelta para adicionar ao timestamp
                if 'm' in interval: time_delta = timedelta(minutes=int(interval.replace('m', '')))
                elif 'h' in interval: time_delta = timedelta(hours=int(interval.replace('h', '')))
                elif 'd' in interval: time_delta = timedelta(days=int(interval.replace('d', '')))
                else: time_delta = timedelta(minutes=1) # Fallback
                
                # Ajusta a data de início da busca para o próximo candle
                fetch_start_dt = last_saved_timestamp + time_delta
                fetch_start_ts = int(fetch_start_dt.timestamp() * 1000)
                
                logger.info(f"🔄 [DATA PROVIDER] Dados persistentes encontrados para {symbol}-{interval}. Buscando dados incrementais a partir de {fetch_start_dt.strftime('%Y-%m-%d %H:%M UTC')}.")
            else:
                logger.info(f"🆕 [DATA PROVIDER] Dados persistentes não encontrados para {symbol}-{interval}. Buscando dados completos.")

            # Verifica se já temos dados até o final do período solicitado
            if fetch_start_ts > int(end_dt_utc.timestamp() * 1000):
                 logger.info(f"✅ [DATA PROVIDER] Dados para {symbol}-{interval} já estão atualizados até {end_dt_utc.strftime('%Y-%m-%d %H:%M UTC')}.")
                 if historical_df is not None:
                     raw_dfs_multi_tf[interval] = historical_df
                 continue


            # Adiciona a tarefa de busca incremental/completa
            tasks.append(self._fetch_historical_batch(
                symbol, 
                interval, 
                fetch_start_ts, 
                int(end_dt_utc.timestamp() * 1000), 
                historical_df=historical_df # Passa o DataFrame existente para concatenação
            ))
        
        # O resultado será uma lista de (interval, DataFrame) ou None, se a busca foi pulada
        results = await asyncio.gather(*tasks)

        # Agrupar os resultados dos fetches em raw_dfs_multi_tf e salvar a persistência
        for interval, fetched_df in results:
            if fetched_df is not None and not fetched_df.empty:
                raw_dfs_multi_tf[interval] = fetched_df
                # Salva o DataFrame completo (dados antigos + novos)
                self._save_historical_data(fetched_df, symbol, interval)
            else:
                logger.warning(f"⚠️ [DATA PROVIDER] Nenhum dado coletado para o timeframe {interval} na busca atual.")
                # Se o fetched_df for None ou vazio, ainda precisamos dos dados antigos se existirem
                if interval not in raw_dfs_multi_tf:
                     historical_df = self._load_historical_data(symbol, interval)
                     if historical_df is not None:
                         raw_dfs_multi_tf[interval] = historical_df


        if not raw_dfs_multi_tf:
            logger.error("❌ [ERRO DATA PROVIDER] Nenhuma dado histórico coletado ou carregado para treinamento em nenhum timeframe.")
            return None

        logger.info(f"✅ [DATA PROVIDER] Coleta e/ou carregamento de dados históricos concluído. Timeframes coletados: {list(raw_dfs_multi_tf.keys())}.")
        return raw_dfs_multi_tf

    async def _fetch_historical_batch(self, symbol: str, interval: str, start_ts: int, end_ts: int, historical_df: Optional[pd.DataFrame] = None) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        Busca dados históricos para um único timeframe, lidando com paginação e rate limiting.
        """
        all_data_frames = []
        current_ts = start_ts
        request_limit = 1500 # Máximo de klines por requisição

        # Converte o intervalo para milissegundos para cálculo de avanço de tempo
        interval_ms = 0
        if 'm' in interval: interval_ms = int(interval.replace('m', '')) * 60 * 1000
        elif 'h' in interval: interval_ms = int(interval.replace('h', '')) * 3600 * 1000
        elif 'd' in interval: interval_ms = int(interval.replace('d', '')) * 86400 * 1000
        
        if interval_ms == 0:
            logger.error(f"❌ [ERRO DATA PROVIDER] Intervalo '{interval}' inválido. Não é possível calcular intervalo em ms.")
            return interval, historical_df

        logger.debug(f"⏳ [DATA PROVIDER] Buscando dados para {interval} de {datetime.fromtimestamp(start_ts/1000, tz=timezone.utc)} a {datetime.fromtimestamp(end_ts/1000, tz=timezone.utc)}...")

        while current_ts < end_ts:
            
            data_batch = await self._fetch_and_format_data(symbol, interval, limit=request_limit, start_ts=current_ts, end_ts=end_ts)
            
            if data_batch is None or data_batch.empty:
                logger.info(f"ℹ️ [DATA PROVIDER] Nenhuma dado adicional ou dados vazios retornados na busca para treinamento. Finalizando coleta.")
                break # Sai do loop se não houver mais dados ou erro

            all_data_frames.append(data_batch)
            
            # Avança o timestamp para o início do próximo candle após o último candle recebido
            if data_batch.index.empty:
                logger.warning("⚠️ [DATA PROVIDER] Lote de dados vazio, não é possível avançar o timestamp. Encerrando loop.")
                break

            last_open_time_in_batch_ms = int(data_batch.index[-1].timestamp() * 1000)
            current_ts = last_open_time_in_batch_ms + interval_ms # Começa a próxima busca do início do próximo candle

            # Evita loops infinitos ou exceder end_ts
            if current_ts >= end_ts and interval_ms > 0: 
                break
            
            # --- IMPLEMENTAÇÃO DO RATE LIMITING ---
            # Pausa para evitar Rate Limits da Binance ao buscar dados históricos
            await asyncio.sleep(self.historical_fetch_delay)
        
        # Combina os dados existentes (se carregados) com os dados recém-buscados
        if historical_df is not None:
            all_data_frames.insert(0, historical_df)

        if not all_data_frames:
            return interval, None

        full_df = pd.concat(all_data_frames)
        # Remove quaisquer linhas duplicadas que possam ter sido coletadas devido a sobreposições de tempo
        full_df = full_df[~full_df.index.duplicated(keep='first')].sort_index() 

        if full_df.empty:
            logger.warning(f"⚠️ [DATA PROVIDER] DataFrame consolidado de treinamento está vazio após remover duplicatas.")
            return interval, None

        logger.info(f"✅ [DATA PROVIDER] Coleta histórica para {interval} concluída. Total: {len(full_df)} linhas. Período: {full_df.index.min()} a {full_df.index.max()}.")
        return interval, full_df
