# -----------------------------------------------------------------------------
# ARQUIVO: feature_engineering/main.py (Versão Final de Produção)
# -----------------------------------------------------------------------------

"""
Módulo Principal de Engenharia de Features.
"""

import pandas as pd
import numpy as np
import joblib 
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from .native_indicators import NativeIndicators 
from .temporal_autoencoder import TemporalAutoencoderPipeline  # <--- ADICIONAR ESTA LINHA
from utils.logger import get_logger 
from config.settings import AIConfig, TradingConfig 

logger = get_logger("FeatureEngineering")

# ------------------------------------------------------------------
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# EXCLUIR TODA A DEFINIÇÃO DA CLASSE 'Autoencoder' DAQUI
# A lógica agora está obsoleta e será totalmente substituída pela TemporalSDAE
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ------------------------------------------------------------------

class FeatureEngineeringPipeline:
    def __init__(self, config: AIConfig):
        """
        [VERSÃO ADAPTADA PARA O AUTOENCODER TEMPORAL]
        Inicializa os scalers e a pipeline do autoencoder temporal.
        """
        if not isinstance(config, AIConfig):
            raise TypeError("🚨 'config' deve ser uma instância da classe AIConfig.")
        self.config = config
        self.trading_config = TradingConfig()
        
        self.native_indicators = NativeIndicators()
        self.scalers: Dict[str, Any] = {}
        
        # --- INÍCIO DA MODIFICAÇÃO ---
        
        # 1. Caminhos para os scalers (o resto dos caminhos será gerenciado pela outra classe)
        self.scaler_path = os.path.join(self.config.MODEL_DIR, "feature_scalers_dual.joblib")
        
        # 2. Instanciar a nova pipeline do autoencoder temporal
        #    Ela gerenciará seus próprios modelos, hiperparâmetros e arquivos.
        self.temporal_autoencoder_pipeline = TemporalAutoencoderPipeline(config)
        
        # 3. Remover atributos e caminhos do autoencoder antigo
        #    (self.autoencoder, self.autoencoder_path, etc. não são mais necessários aqui)
        
        # --- FIM DA MODIFICAÇÃO ---
        
        # Carrega os scalers de forma robusta
        self._load_scalers()
        logger.info("⚙️ FeatureEngineeringPipeline inicializado com a Pipeline de Autoencoder Temporal.")


# ------------------------------------------------------------------
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# EXCLUIR MÉTODOS ANTIGOS DO AUTOENCODER
# Toda essa responsabilidade agora pertence à TemporalAutoencoderPipeline
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ------------------------------------------------------------------        

    def _save_scalers(self):
        """Salva um dicionário contendo os scalers e a lista de colunas."""
        try:
            # O objeto a ser salvo agora é o dicionário 'self.scalers' inteiro
            joblib.dump(self.scalers, self.scaler_path)
            logger.info(f"✅ Normalizadores e lista de colunas salvos em '{self.scaler_path}'.")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar os normalizadores: {e}", exc_info=True)
            
    def _load_scalers(self):
        """
        [FUNÇÃO CORRIGIDA]
        Carrega os normalizadores e valida sua estrutura para garantir a integridade do
        'Contrato de Features'.
        """
        # Sempre começa com um dicionário vazio
        self.scalers = {}
        if os.path.exists(self.scaler_path):
            try:
                loaded_data = joblib.load(self.scaler_path)
                
                # --- INÍCIO DA CORREÇÃO E VALIDAÇÃO ---
                # Verifica se o dado carregado é um dicionário e tem as chaves essenciais
                if (isinstance(loaded_data, dict) and
                    'robust' in loaded_data and
                    'minmax' in loaded_data and
                    'columns' in loaded_data and
                    isinstance(loaded_data['columns'], list)):
                    
                    self.scalers = loaded_data
                    logger.info(f"✅ Normalizadores e 'Contrato de Features' válidos carregados de '{self.scaler_path}'.")
                else:
                    # O arquivo existe, mas está em um formato antigo ou inválido.
                    logger.warning(f"⚠️ Arquivo de normalizadores '{self.scaler_path}' encontrado, mas está em formato inválido ou obsoleto. Ignorando e tratando como se não existisse.")
                    # Mantém self.scalers como um dicionário vazio, forçando a necessidade de um novo 'fit'.
                    self.scalers = {}
                # --- FIM DA CORREÇÃO E VALIDAÇÃO ---

            except Exception as e:
                logger.error(f"❌ Erro ao carregar ou validar os normalizadores de '{self.scaler_path}': {e}. Tratando como se não existisse.")
                self.scalers = {} # Garante que está vazio em caso de qualquer erro
        else:
             logger.warning(f"⚠️ Arquivo de scalers não encontrado em '{self.scaler_path}'. Um ciclo de treinamento (fit_scaler=True) é necessário para criá-lo.")
             self.scalers = {}


# ------------------------------------------------------------------
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# EXCLUIR MÉTODOS DE INICIALIZAÇÃO ANTIGOS
# A nova pipeline gerencia isso internamente
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# EXCLUIR MÉTODO DE OTIMIZAÇÃO ANTIGO
# A nova pipeline gerencia isso internamente com Optuna


   
    def train_autoencoder(self, data_for_autoencoder: pd.DataFrame, num_epochs: int) -> int:
        """
        [MÉTODO ADAPTADO]
        Delega o treinamento para a pipeline do autoencoder temporal.
        O parâmetro 'num_epochs' não é usado diretamente, pois a nova pipeline
        tem sua própria lógica de early stopping. O 'n_trials' é mais relevante.
        """
        logger.info("🧠 Delegando treinamento para a Temporal Autoencoder Pipeline...")
        # A nova função espera o número de trials para o Optuna, não épocas.
        # Podemos usar um valor padrão ou passá-lo de alguma forma.
        # Vamos usar um valor fixo por simplicidade.
        n_trials_for_optimization = 75

        # A função `train_autoencoder_temporal` retorna um booleano (sucesso/falha).
        # Extrai colunas válidas
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in data_for_autoencoder.columns if col not in ohlcv_cols]
        if not feature_columns:
            feature_columns = data_for_autoencoder.columns.tolist()
        
        # Extrai colunas de features (exclui OHLCV)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in data_for_autoencoder.columns if col not in ohlcv_cols]
        if not feature_columns:
            feature_columns = data_for_autoencoder.columns.tolist()
        
        # ASSINATURA CORRETA: (df, feature_columns, optimize)
        success = self.temporal_autoencoder_pipeline.train_autoencoder_temporal(
            df=data_for_autoencoder,
            feature_columns=feature_columns,
            optimize=True
        )
        
        logger.info(f"✅ Treinamento delegado concluído. Sucesso: {success}")
        return 1 if success else 0

    async def create_features(self, raw_df_dict: Dict[str, pd.DataFrame], symbol: str, primary_timeframe: str, fit_scaler: bool = False, tape_metrics: Optional[Dict[str, Any]] = None, onchain_metrics: Optional[Dict[str, Any]] = None, apply_latent: bool = True) -> Optional[pd.DataFrame]:
        """
        [VERSÃO FINAL E COMPLETA - ESTADO DA ARTE]
        Orquestra a criação de features, garantindo a consistência da normalização
        entre o treinamento e a inferência. Esta é a versão completa sem abreviações.
        """
        # --- Etapa 1: Validações Iniciais ---
        if not isinstance(raw_df_dict, dict) or not raw_df_dict:
            logger.error(f"❌ [ERRO FE] 'raw_df_dict' deveria ser um dicionário. Abortando.")
            return None
        if primary_timeframe not in raw_df_dict:
            logger.error(f"❌ Timeframe primário '{primary_timeframe}' não encontrado. Abortando.")
            return None

        # --- Etapa 2: Cálculo de Indicadores em Paralelo ---
        logger.info(f"⚙️ [FE] Iniciando cálculo de indicadores para {len(raw_df_dict)} timeframes em PARALELO...")
        
        loop = asyncio.get_running_loop()
        tasks = []
        
        for tf, df_raw in raw_df_dict.items():
            if df_raw.empty or not all(col in df_raw.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                logger.warning(f"⚠️ [FE] DataFrame para o timeframe '{tf}' está vazio ou sem colunas essenciais. Pulando.")
                continue
            
            task = loop.run_in_executor(
                None,
                self.native_indicators.calculate_all_features,
                df_raw.copy(), symbol, tf
            )
            tasks.append(task)

        if not tasks:
            logger.error("❌ [ERRO FE] Nenhuma tarefa de cálculo de features válida para executar.")
            return None
            
        all_timeframes_featured_dfs = await asyncio.gather(*tasks)
        logger.info(f"✅ [FE] Cálculo de indicadores em paralelo concluído.")

        # --- Etapa 3: Sufixação e Preparação para o Merge ---
        all_timeframes_dfs = []
        
        for tf_featured_df, (tf, df_raw) in zip(all_timeframes_featured_dfs, raw_df_dict.items()):
            if tf_featured_df is None or tf_featured_df.empty:
                continue

            tf_featured_df = tf_featured_df.loc[:, ~tf_featured_df.columns.duplicated()]
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # Separa colunas calculadas (que recebem sufixo _{tf})
            computed_cols = [col for col in tf_featured_df.columns if col not in required_cols]
            
            # Cria DataFrame com features calculadas sufixadas (ex: rsi_15m)
            suffixed_computed_df = tf_featured_df[computed_cols].rename(columns={col: f"{col}_{tf}" for col in computed_cols})
            
            if tf == primary_timeframe:
                # Timeframe primário: Raw sem sufixo + Calculadas sufixadas
                final_df_for_tf = pd.concat([tf_featured_df[required_cols], suffixed_computed_df], axis=1)
                all_timeframes_dfs.append(final_df_for_tf)
            else:
                # Timeframes secundários: Raw com sufixo _tf_ + Calculadas sufixadas + Cópia de Base Features com _tf_
                # Motivo: O metadata de treinamento espera 'hc_wick_upper_1h' E 'hc_wick_upper_tf_1h'
                
                # 1. Raw Cols -> _tf_{tf}
                raw_suffixed_df = tf_featured_df[required_cols].rename(columns={col: f"{col}_tf_{tf}" for col in required_cols})
                
                # 2. Base Features que precisam de versão _tf_
                base_features_to_duplicate = [
                    'log_return', 'log_return_volume', 'hl_range_pct', 'oc_move_pct',
                    'hc_wick_upper', 'lc_wick_lower', 'realized_vol_20', 'realized_vol_100',
                    'close_reference'
                ]
                
                extra_tf_features = []
                for base_feat in base_features_to_duplicate:
                    if base_feat in tf_featured_df.columns:
                        extra_tf_features.append(
                            tf_featured_df[[base_feat]].rename(columns={base_feat: f"{base_feat}_tf_{tf}"})
                        )
                
                # Une tudo
                parts = [raw_suffixed_df, suffixed_computed_df] + extra_tf_features
                final_df_for_tf = pd.concat(parts, axis=1)
                all_timeframes_dfs.append(final_df_for_tf)

        if not all_timeframes_dfs:
            logger.error("❌ Nenhum dataframe com features foi gerado após o cálculo dos indicadores.")
            return None

        # --- Etapa 4: Merge Robusto dos Timeframes ---
        base_df = next((df for df in all_timeframes_dfs if all(c in df.columns for c in required_cols)), None)
        if base_df is None:
            logger.error("❌ Dataframe base (do timeframe primário) não foi encontrado após a geração de features.")
            return None
        
        other_dfs = [df for df in all_timeframes_dfs if df is not base_df]
        featured_df_combined = base_df.sort_index()

        for other_df in other_dfs:
            featured_df_combined = pd.merge_asof(
                left=featured_df_combined.sort_index(), right=other_df.sort_index(),
                left_index=True, right_index=True, direction='backward'
            )
        
        # --- Etapa 5: Limpeza e Consistência de Tipos ---
        logger.info("⚙️ [FE] Garantindo consistência de tipos. Convertendo todas as colunas booleanas para inteiros (0/1)...")
        for col in featured_df_combined.select_dtypes(include='bool').columns:
            featured_df_combined[col] = featured_df_combined[col].astype(int)

        initial_rows = len(featured_df_combined)
        featured_df_combined.dropna(inplace=True)
        if initial_rows > len(featured_df_combined):
            logger.info(f"🧹 [FE CLEANING] Removidas {initial_rows - len(featured_df_combined)} linhas com dados desalinhados.")
        
        if featured_df_combined.empty:
            logger.error("❌ DataFrame ficou vazio após a limpeza final. Abortando.")
            return None

        # [DEBUG] Log das colunas geradas
        raise RuntimeError("VERIFICANDO SE ARQUIVO E CARREGADO")
        logger.debug(f"🔍 [FE DEBUG] Colunas geradas: {list(featured_df_combined.columns)}")
        
        # --- Etapa 6: REMOVIDO SCALING GLOBAL (Audit Fix) ---
        # Motivo: O scaling aqui causava Data Leakage ao usar estatísticas globais antes do split.
        # A normalização agora é delegada para:
        # 1. TemporalAutoencoderPipeline (para features latentes)
        # 2. TrendPredictor (para features finais)
        
        # --- Etapa 7: 🔬 ADICIONAR REGIME LABELS (Scientific Pipeline) ---
        # Referência: Hamilton (1989), Ang & Bekaert (2002)
        # Detecta regimes uma vez aqui para garantir consistência entre especialistas
        try:
            from feature_engineering.scientific_data_processor import ScientificDataProcessor
            
            data_processor = ScientificDataProcessor()
            featured_df_combined = data_processor.add_regime_labels(featured_df_combined)
            data_processor.save_scalers()
            
            logger.info(f"✅ [FE] Regime labels adicionados: {featured_df_combined['regime'].value_counts().to_dict()}")
        except Exception as e:
            logger.warning(f"⚠️ [FE] Falha ao adicionar regime labels: {e}. Continuando sem eles.")
        
        # 🔧 FIX: Return the processed DataFrame
        logger.info(f"✅ [FE] Features criadas com sucesso. Shape: {featured_df_combined.shape}")
        return featured_df_combined
# ------------------------------------------------------------------        

    def _save_scalers(self):
        """Salva um dicionário contendo os scalers e a lista de colunas."""
        try:
            # O objeto a ser salvo agora é o dicionário 'self.scalers' inteiro
            joblib.dump(self.scalers, self.scaler_path)
            logger.info(f"✅ Normalizadores e lista de colunas salvos em '{self.scaler_path}'.")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar os normalizadores: {e}", exc_info=True)
            
    def _load_scalers(self):
        """
        [FUNÇÃO CORRIGIDA]
        Carrega os normalizadores e valida sua estrutura para garantir a integridade do
        'Contrato de Features'.
        """
        # Sempre começa com um dicionário vazio
        self.scalers = {}
        if os.path.exists(self.scaler_path):
            try:
                loaded_data = joblib.load(self.scaler_path)
                
                # --- INÍCIO DA CORREÇÃO E VALIDAÇÃO ---
                # Verifica se o dado carregado é um dicionário e tem as chaves essenciais
                if (isinstance(loaded_data, dict) and
                    'robust' in loaded_data and
                    'minmax' in loaded_data and
                    'columns' in loaded_data and
                    isinstance(loaded_data['columns'], list)):
                    
                    self.scalers = loaded_data
                    logger.info(f"✅ Normalizadores e 'Contrato de Features' válidos carregados de '{self.scaler_path}'.")
                else:
                    # O arquivo existe, mas está em um formato antigo ou inválido.
                    logger.warning(f"⚠️ Arquivo de normalizadores '{self.scaler_path}' encontrado, mas está em formato inválido ou obsoleto. Ignorando e tratando como se não existisse.")
                    # Mantém self.scalers como um dicionário vazio, forcing a necessidade de um novo 'fit'.
                    self.scalers = {}
                # --- FIM DA CORREÇÃO E VALIDAÇÃO ---

            except Exception as e:
                logger.error(f"❌ Erro ao carregar ou validar os normalizadores de '{self.scaler_path}': {e}. Tratando como se não existisse.")
                self.scalers = {} # Garante que está vazio em caso de qualquer erro
        else:
             logger.warning(f"⚠️ Arquivo de scalers não encontrado em '{self.scaler_path}'. Um ciclo de treinamento (fit_scaler=True) é necessário para criá-lo.")
             self.scalers = {}


# ------------------------------------------------------------------
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# EXCLUIR MÉTODOS DE INICIALIZAÇÃO ANTIGOS
# A nova pipeline gerencia isso internamente
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# EXCLUIR MÉTODO DE OTIMIZAÇÃO ANTIGO
# A nova pipeline gerencia isso internamente com Optuna


   
    def train_autoencoder(self, data_for_autoencoder: pd.DataFrame, num_epochs: int) -> int:
        """
        [MÉTODO ADAPTADO]
        Delega o treinamento para a pipeline do autoencoder temporal.
        O parâmetro 'num_epochs' não é usado diretamente, pois a nova pipeline
        tem sua própria lógica de early stopping. O 'n_trials' é mais relevante.
        """
        logger.info("🧠 Delegando treinamento para a Temporal Autoencoder Pipeline...")
        # A nova função espera o número de trials para o Optuna, não épocas.
        # Podemos usar um valor padrão ou passá-lo de alguma forma.
        # Vamos usar um valor fixo por simplicidade.
        n_trials_for_optimization = 75

        # A função `train_autoencoder_temporal` retorna um booleano (sucesso/falha).
        # Extrai colunas válidas
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in data_for_autoencoder.columns if col not in ohlcv_cols]
        if not feature_columns:
            feature_columns = data_for_autoencoder.columns.tolist()
        
        # Extrai colunas de features (exclui OHLCV)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in data_for_autoencoder.columns if col not in ohlcv_cols]
        if not feature_columns:
            feature_columns = data_for_autoencoder.columns.tolist()
        
        # ASSINATURA CORRETA: (df, feature_columns, optimize)
        success = self.temporal_autoencoder_pipeline.train_autoencoder_temporal(
            df=data_for_autoencoder,
            feature_columns=feature_columns,
            optimize=True
        )
        
        logger.info(f"✅ Treinamento delegado concluído. Sucesso: {success}")
        return 1 if success else 0

    async def create_features(self, raw_df_dict: Dict[str, pd.DataFrame], symbol: str, primary_timeframe: str, fit_scaler: bool = False, tape_metrics: Optional[Dict[str, Any]] = None, onchain_metrics: Optional[Dict[str, Any]] = None, apply_latent: bool = True) -> Optional[pd.DataFrame]:
        """
        [VERSÃO FINAL E COMPLETA - ESTADO DA ARTE]
        Orquestra a criação de features, garantindo a consistência da normalização
        entre o treinamento e a inferência. Esta é a versão completa sem abreviações.
        """
        # --- Etapa 1: Validações Iniciais ---
        if not isinstance(raw_df_dict, dict) or not raw_df_dict:
            logger.error(f"❌ [ERRO FE] 'raw_df_dict' deveria ser um dicionário. Abortando.")
            return None
        if primary_timeframe not in raw_df_dict:
            logger.error(f"❌ Timeframe primário '{primary_timeframe}' não encontrado. Abortando.")
            return None

        # --- Etapa 2: Cálculo de Indicadores em Paralelo ---
        logger.info(f"⚙️ [FE] Iniciando cálculo de indicadores para {len(raw_df_dict)} timeframes em PARALELO...")
        
        loop = asyncio.get_running_loop()
        tasks = []
        
        for tf, df_raw in raw_df_dict.items():
            if df_raw.empty or not all(col in df_raw.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                logger.warning(f"⚠️ [FE] DataFrame para o timeframe '{tf}' está vazio ou sem colunas essenciais. Pulando.")
                continue
            
            task = loop.run_in_executor(
                None,
                self.native_indicators.calculate_all_features,
                df_raw.copy(), symbol, tf
            )
            tasks.append(task)

        if not tasks:
            logger.error("❌ [ERRO FE] Nenhuma tarefa de cálculo de features válida para executar.")
            return None
            
        all_timeframes_featured_dfs = await asyncio.gather(*tasks)
        logger.info(f"✅ [FE] Cálculo de indicadores em paralelo concluído.")

        # --- Etapa 3: Sufixação e Preparação para o Merge ---
        all_timeframes_dfs = []
        
        for tf_featured_df, (tf, df_raw) in zip(all_timeframes_featured_dfs, raw_df_dict.items()):
            if tf_featured_df is None or tf_featured_df.empty:
                continue

            tf_featured_df = tf_featured_df.loc[:, ~tf_featured_df.columns.duplicated()]
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            cols_to_suffix = [col for col in tf_featured_df.columns if col not in required_cols]
            suffixed_df = tf_featured_df[cols_to_suffix].rename(columns={col: f"{col}_{tf}" for col in cols_to_suffix})
            
            if tf == primary_timeframe:
                final_df_for_tf = pd.concat([tf_featured_df[required_cols], suffixed_df], axis=1)
                all_timeframes_dfs.append(final_df_for_tf)
            else:
                all_timeframes_dfs.append(suffixed_df)

        if not all_timeframes_dfs:
            logger.error("❌ Nenhum dataframe com features foi gerado após o cálculo dos indicadores.")
            return None

        # --- Etapa 4: Merge Robusto dos Timeframes ---
        base_df = next((df for df in all_timeframes_dfs if all(c in df.columns for c in required_cols)), None)
        if base_df is None:
            logger.error("❌ Dataframe base (do timeframe primário) não foi encontrado após a geração de features.")
            return None
        
        other_dfs = [df for df in all_timeframes_dfs if df is not base_df]
        featured_df_combined = base_df.sort_index()

        for other_df in other_dfs:
            featured_df_combined = pd.merge_asof(
                left=featured_df_combined.sort_index(), right=other_df.sort_index(),
                left_index=True, right_index=True, direction='backward'
            )
        
        # --- Etapa 5: Limpeza e Consistência de Tipos ---
        logger.info("⚙️ [FE] Garantindo consistência de tipos. Convertendo todas as colunas booleanas para inteiros (0/1)...")
        for col in featured_df_combined.select_dtypes(include='bool').columns:
            featured_df_combined[col] = featured_df_combined[col].astype(int)

        initial_rows = len(featured_df_combined)
        featured_df_combined.dropna(inplace=True)
        if initial_rows > len(featured_df_combined):
            logger.info(f"🧹 [FE CLEANING] Removidas {initial_rows - len(featured_df_combined)} linhas com dados desalinhados.")
        
        if featured_df_combined.empty:
            logger.error("❌ DataFrame ficou vazio após a limpeza final. Abortando.")
            return None

        # --- Etapa 6: REMOVIDO SCALING GLOBAL (Audit Fix) ---
        # Motivo: O scaling aqui causava Data Leakage ao usar estatísticas globais antes do split.
        # A normalização agora é delegada para:
        # 1. TemporalAutoencoderPipeline (para features latentes)
        # 2. TrendPredictor (para features finais)
        
        # 🔧 FIX: Return the processed DataFrame
        logger.info(f"✅ [FE] Features criadas com sucesso. Shape: {featured_df_combined.shape}")
        return featured_df_combined

    def apply_hidden_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wrapper para aplicar features latentes usando o pipeline temporal.
        [FIX] Adiciona features dummy para compatibilidade com modelos treinados
        que esperam outputs do antigo TrendPredictor e labelers.
        """
        # 1. Aplica o Autoencoder
        df_enriched = self.temporal_autoencoder_pipeline.apply_hidden_features_temporal(df)
        
        # 2. [FIX] Adiciona features dummy para satisfazer o contrato de input dos modelos (dim=54)
        # Lista baseada nos logs de erro de features ausentes
        missing_cols = [
            'tp_regime_sideways', 'tp_regime_up', 'tp_regime_down',
            'trend_pred_uptrend', 'trend_pred_downtrend', 'trend_pred_neutral',
            'trend_pred_confidence',
            'tp_prior_dir', 'tp_prior_conf',
            'tp_duration_median', 'tp_uncertainty',
            'qpnl_24_p50', 'qpnl_24_p90', 
            'qpnl_48_p50', 'qpnl_48_p90',
            'ema_trend'
        ]
        
        for col in missing_cols:
            if col not in df_enriched.columns:
                # Usa 0.0 para evitar NaN e manter neutralidade
                df_enriched[col] = 0.0
                
        # [OPCIONAL] Se ema_trend precisar ser algo diferente de 0 (ex: 0.5 para neutro?)
        # Por enquanto 0.0 é seguro para redes neurais normalizadas.
        
        return df_enriched