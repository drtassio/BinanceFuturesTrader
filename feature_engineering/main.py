# -----------------------------------------------------------------------------
# ARQUIVO: feature_engineering/main.py (Versao Final de Producao)
# -----------------------------------------------------------------------------

"""
Modulo Principal de Engenharia de Features.
"""

import pandas as pd
import numpy as np
import joblib 
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from .native_indicators import NativeIndicators 
from .temporal_autoencoder import TemporalAutoencoderPipeline
from utils.logger import get_logger 
from config.settings import AIConfig, TradingConfig 

logger = get_logger("FeatureEngineering")

class FeatureEngineeringPipeline:
    def __init__(self, config: AIConfig):
        """
        [VERSAO ADAPTADA PARA O AUTOENCODER TEMPORAL]
        Inicializa os scalers e a pipeline do autoencoder temporal.
        """
        if not isinstance(config, AIConfig):
            raise TypeError("'config' deve ser uma instancia da classe AIConfig.")
        self.config = config
        self.trading_config = TradingConfig()
        
        self.native_indicators = NativeIndicators()
        self.scalers: Dict[str, Any] = {}
        
        # 1. Caminhos para os scalers
        self.scaler_path = os.path.join(self.config.MODEL_DIR, "feature_scalers_dual.joblib")
        
        # 2. Instanciar a nova pipeline do autoencoder temporal
        self.temporal_autoencoder_pipeline = TemporalAutoencoderPipeline(config)
        
        # Carrega os scalers de forma robusta
        self._load_scalers()
        logger.info("FeatureEngineeringPipeline inicializado com a Pipeline de Autoencoder Temporal.")

    def _save_scalers(self):
        """Salva um dicionario contendo os scalers e a lista de colunas."""
        try:
            joblib.dump(self.scalers, self.scaler_path)
            logger.info(f"Normalizadores e lista de colunas salvos em '{self.scaler_path}'.")
        except Exception as e:
            logger.error(f"Erro ao salvar os normalizadores: {e}", exc_info=True)
            
    def _load_scalers(self):
        """
        Carrega os normalizadores e valida sua estrutura para garantir a integridade do
        'Contrato de Features'.
        """
        self.scalers = {}
        if os.path.exists(self.scaler_path):
            try:
                loaded_data = joblib.load(self.scaler_path)
                if (isinstance(loaded_data, dict) and
                    'robust' in loaded_data and
                    'minmax' in loaded_data and
                    'columns' in loaded_data and
                    isinstance(loaded_data['columns'], list)):
                    
                    self.scalers = loaded_data
                    logger.info(f"Normalizadores e 'Contrato de Features' validos carregados de '{self.scaler_path}'.")
                else:
                    logger.warning(f"Arquivo de normalizadores '{self.scaler_path}' obsoleto. Ignorando.")
                    self.scalers = {}
            except Exception as e:
                logger.error(f"Erro ao carregar scalers de '{self.scaler_path}': {e}.")
                self.scalers = {}
        else:
             logger.warning(f"Arquivo de scalers nao encontrado em '{self.scaler_path}'.")
             self.scalers = {}

    def train_autoencoder(self, data_for_autoencoder: pd.DataFrame, num_epochs: int) -> int:
        """
        Delega o treinamento para a pipeline do autoencoder temporal.
        """
        logger.info("Delegando treinamento para a Temporal Autoencoder Pipeline...")
        
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in data_for_autoencoder.columns if col not in ohlcv_cols]
        if not feature_columns:
            feature_columns = data_for_autoencoder.columns.tolist()
        
        success = self.temporal_autoencoder_pipeline.train_autoencoder_temporal(
            df=data_for_autoencoder,
            feature_columns=feature_columns,
            optimize=True
        )
        
        logger.info(f"Treinamento delegado concluido. Sucesso: {success}")
        return 1 if success else 0

    async def create_features(self, raw_df_dict: Dict[str, pd.DataFrame], symbol: str, primary_timeframe: str, fit_scaler: bool = False, tape_metrics: Optional[Dict[str, Any]] = None, onchain_metrics: Optional[Dict[str, Any]] = None, apply_latent: bool = True) -> Optional[pd.DataFrame]:
        """
        [VERSAO FINAL E COMPLETA - ESTADO DA ARTE]
        Orquestra a criacao de features, garantindo a consistencia da normalizacao.
        """
        if not isinstance(raw_df_dict, dict) or not raw_df_dict:
            logger.error("'raw_df_dict' invalido.")
            return None
        if primary_timeframe not in raw_df_dict:
            logger.error(f"Timeframe primario '{primary_timeframe}' nao encontrado.")
            return None

        logger.info(f"Iniciando calculo de indicadores para {len(raw_df_dict)} timeframes em PARALELO...")
        
        loop = asyncio.get_running_loop()
        tasks = []
        for tf, df_raw in raw_df_dict.items():
            if df_raw.empty or not all(col in df_raw.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                logger.warning(f"DataFrame para '{tf}' invalido. Pulando.")
                continue
            
            task = loop.run_in_executor(
                None,
                self.native_indicators.calculate_all_features,
                df_raw.copy(), symbol, tf
            )
            tasks.append(task)

        if not tasks:
            logger.error("Nenhuma tarefa valida.")
            return None
            
        all_timeframes_featured_dfs = await asyncio.gather(*tasks)
        logger.info("Calculo de indicadores concluido.")

        all_timeframes_dfs = []
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        for tf_featured_df, (tf, df_raw) in zip(all_timeframes_featured_dfs, raw_df_dict.items()):
            if tf_featured_df is None or tf_featured_df.empty:
                continue

            tf_featured_df = tf_featured_df.loc[:, ~tf_featured_df.columns.duplicated()]
            
            if tf == primary_timeframe:
                cols_to_suffix = [col for col in tf_featured_df.columns if col not in required_cols]
                suffixed_df = tf_featured_df[cols_to_suffix].rename(columns={col: f"{col}_{tf}" for col in cols_to_suffix})
                final_df_for_tf = pd.concat([tf_featured_df[required_cols], suffixed_df], axis=1)
                all_timeframes_dfs.append(final_df_for_tf)
            else:
                 # SINGLE SUFFIXING STRATEGY for secondary timeframes (Fix BUG M2)
                 cols_all = tf_featured_df.columns.tolist()
                 final_secondary_df = tf_featured_df[cols_all].rename(columns={col: f"{col}_{tf}" for col in cols_all})
                 final_secondary_df = final_secondary_df.loc[:, ~final_secondary_df.columns.duplicated()]
                 all_timeframes_dfs.append(final_secondary_df)

        if not all_timeframes_dfs:
            return None

        base_df = next((df for df in all_timeframes_dfs if all(c in df.columns for c in required_cols)), None)
        if base_df is None:
            return None
        
        other_dfs = [df for df in all_timeframes_dfs if df is not base_df]
        featured_df_combined = base_df.sort_index()

        for other_df in other_dfs:
            featured_df_combined = pd.merge_asof(
                left=featured_df_combined.sort_index(), right=other_df.sort_index(),
                left_index=True, right_index=True, direction='backward'
            )
        
        logger.info("Garantindo consistencia de tipos...")
        for col in featured_df_combined.select_dtypes(include='bool').columns:
            featured_df_combined[col] = featured_df_combined[col].astype(int)

        featured_df_combined.dropna(inplace=True)
        if featured_df_combined.empty:
            return None

        # ETAPA 7: ADICIONAR REGIME LABELS (Hamilton 1989)
        try:
            from feature_engineering.scientific_data_processor import ScientificDataProcessor
            data_processor = ScientificDataProcessor()
            featured_df_combined = data_processor.add_regime_labels(featured_df_combined)
            data_processor.save_scalers()
            logger.info(f"Regime labels adicionados: {featured_df_combined['regime'].value_counts().to_dict()}")
        except Exception as e:
            logger.warning(f"Falha ao adicionar regime labels: {e}")
        
        logger.info(f"Features criadas com sucesso. Shape: {featured_df_combined.shape}")
        return featured_df_combined

    def apply_hidden_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wrapper para aplicar features latentes usando o pipeline temporal.
        """
        # 1. Aplica o Autoencoder (Adiciona Latents + Regime + Confianca)
        df_enriched = self.temporal_autoencoder_pipeline.apply_hidden_features_temporal(df)
        
        # 2. Compatibilidade Legada
        missing_cols = [
            'tp_regime_sideways', 'tp_regime_up', 'tp_regime_down',
            'trend_pred_uptrend', 'trend_pred_downtrend', 'trend_pred_neutral',
            'trend_pred_confidence', 'tp_prior_dir', 'tp_prior_conf',
            'tp_duration_median', 'tp_uncertainty',
            'qpnl_24_p50', 'qpnl_24_p90', 'qpnl_48_p50', 'qpnl_48_p90',
            'ema_trend'
        ]
        
        for col in missing_cols:
            if col not in df_enriched.columns:
                df_enriched[col] = 0.0
                
        # [FIX] Alias para 'regime'
        if 'regime' not in df_enriched.columns:
            df_enriched['regime'] = df_enriched.get('regime_val', 2)

        # [BUG 9 FIX] Mapeamento CONTINUO (identico ao ai_controller.py L1124).
        # Antes: mapeamento discreto {0:1.0, 1:-1.0, 2:0.0} fazia prior_dir = +-1.0 fixo,
        # eliminando o efeito de confianca e impedindo o low_confidence_gate de funcionar.
        # Agora: prior_dir e modulado pela regime_confidence -> gates filtram corretamente.
        if 'regime' in df_enriched.columns and 'regime_confidence' in df_enriched.columns:
            conditions = [
                df_enriched['regime'] == 0,  # Bull -> positivo
                df_enriched['regime'] == 1,  # Bear -> negativo
                df_enriched['regime'] >= 2,  # Ranger -> neutro
            ]
            choices = [
                df_enriched['regime_confidence'],   # Bull: +conf
                -df_enriched['regime_confidence'],  # Bear: -conf
                0.0                                  # Ranger: 0
            ]
            df_enriched['tp_prior_dir'] = np.select(conditions, choices, default=0.0)
        else:
            df_enriched['tp_prior_dir'] = 0.0

        df_enriched['tp_prior_conf'] = df_enriched.get('regime_confidence', 0.5)
        df_enriched['tp_uncertainty'] = 1.0 - df_enriched['tp_prior_conf']

        return df_enriched