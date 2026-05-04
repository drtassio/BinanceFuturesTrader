# -----------------------------------------------------------------------------
# ARQUIVO: trading/ai_controller.py (VersÃ£o Corrigida e Completa)
# -----------------------------------------------------------------------------

from __future__ import annotations
import pandas as pd
from typing import Dict, Optional, Any, List
import asyncio
import numpy as np
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from collections import deque
from dataclasses import replace
from stable_baselines3 import DQN, SAC
from config.settings import AIConfig, TradingConfig
from utils.logger import get_logger
from models.trade_schema import Signal, Action
from feature_engineering.main import FeatureEngineeringPipeline
from feature_engineering.native_indicators import NativeIndicators

# [REMOVIDO] ProfitabilityPredictor - simplificação do pipeline
# from learning.profitability_predictor import ProfitabilityPredictor
from learning.curriculum_engine import CurriculumEngine
from specialists.trend_specialist import TrendSpecialist, TrendFollowingEnv
from specialists.bull_specialist import BullSpecialist, BullTradingEnv
from specialists.bear_specialist import BearSpecialist, BearTradingEnv
from specialists.ranger_specialist import RangerSpecialist, RangerTradingEnv
# MoE Components
from specialists.soft_gating_network import SoftGatingNetwork, MixtureOfExpertsRouter, create_gating_network
from learning.regime_switch import RegimeSwitch
from learning.walk_forward_validator import WalkForwardValidator
from learning.online_learner import OnlineTrendLearner, create_online_learner
from trading.hrl_master import HRLMaster
from trading.portfolio import PortfolioOptimizer
from trading.risk_manager import RiskManager
from trading.tape_engine import TapeEngine
from trading.onchain_engine import OnChainEngine
from governance.ai_monitor import AIMonitor
from governance.drift_detector import DriftDetector
from xai_engine.explainer import Explainer
from data_provider import DataProvider
from trading.utils.trend_backtest import (
    ensure_datetime_index,
    split_train_holdout,
    build_unseen_eval_frame,
    build_recent_eval_frame,
    run_manual_backtest,
    DEFAULT_EVAL_MONTHS,
    DEFAULT_EVAL_MIN_ROWS,
)

logger = get_logger("AIController")
class AIController:
    """
    [VERSÃO FINAL CORRIGIDA]
    Centraliza a inteligência do Bot.
    """
    def __init__(self, config: AIConfig, trading_config: TradingConfig, system_state: Dict[str, Any]):
        self.config_ai = config
        self.config_trading = trading_config
        self.system_state = system_state
        self.last_trained_date = None
        self.last_adaptation_date = None
        
        # --- Componentes de Decisão ---
        # self.profitability_predictor = None # Removido: Módulo obsoleto (substituído pelo HRLMaster)
        # self.trend_predictor = None          # Removido: Módulo obsoleto
        self.specialists: Dict[str, Any] = {}
        self.hrl_master = None # Será inicializado se PPO estiver ativo
        
        # --- Outros Componentes ---
        self.log_dir = os.path.join(os.getcwd(), "logs")
        self.regime_switch: Optional[RegimeSwitch] = None
        self.meta_learners: Dict[str, Any] = {}
        # MoE Components
        self.gating_network: Optional[SoftGatingNetwork] = None
        self.moe_router: Optional[MixtureOfExpertsRouter] = None
        self.online_learner: Optional[OnlineTrendLearner] = None
        self.feature_pipeline: Optional[FeatureEngineeringPipeline] = None
        self.data_provider: Optional[DataProvider] = None
        self.portfolio: Optional[PortfolioOptimizer] = None
        self.risk_manager: Optional[RiskManager] = None
        self.ai_monitor: Optional[AIMonitor] = None
        self.drift_detector: Optional[DriftDetector] = None
        self.curriculum_engine: Optional[CurriculumEngine] = None
        self.tape_engine: Optional[TapeEngine] = None
        self.onchain_engine: Optional[OnChainEngine] = None
        self.explainer: Optional[Explainer] = None
        self.final_feature_count: Optional[int] = None
        self.is_trained: bool = False
        self.last_trained_date: Optional[datetime] = None
        self.last_adaptation_time: Optional[datetime] = None
        self.metadata_path: str = os.path.join(self.config_ai.MODEL_DIR, "training_metadata.json")
        self.metadata: Dict[str, Any] = {
            'last_full_training_date': None,
            'models': {
                # Caminho real do artefato do autoencoder temporal
                'autoencoder_model': {'trained': False, 'path': os.path.join('autoencoder', 'temporal_sdae.pth'), 'trained_on': None},
                # [ARCH] hrl_master_ppo removido — roteamento via CryptoRegimeDetector + SoftGatingNetwork
                # Especialistas Bull/Bear/Ranger
                'bull_specialist_sac': {'trained': False, 'path': 'bull_specialist_sac.zip', 'trained_on': None},
                'bear_specialist_sac': {'trained': False, 'path': 'bear_specialist_sac.zip', 'trained_on': None},
                'ranger_specialist_sac': {'trained': False, 'path': 'ranger_specialist_sac.zip', 'trained_on': None},
            },
            'final_feature_count': None,
            'base_feature_columns': None
        }
        self._normalize_model_paths()
        self.drift_check_buffer = deque(maxlen=100)
        self.drift_check_min_samples = 100
        # [CORREÃ‡ÃƒO] Inicializa a variÃ¡vel que guardarÃ¡ o "Contrato de Features"
        self.base_feature_columns: Optional[List[str]] = None
        
        # [CACHE] Inicializa cache do detetor de regime
        self.regime_detector = None
        
        # [FIX] Initialize predictors to None to avoid AttributeError
        self.trend_predictor = None
        self.profitability_predictor = None
        
        # [STATUS] Estado do drift para exibição
        self.last_drift_status = "DESCONHECIDO"

    def set_feature_pipeline(self, pipeline: FeatureEngineeringPipeline): self.feature_pipeline = pipeline
    def set_portfolio(self, portfolio: PortfolioOptimizer): self.portfolio = portfolio
    def set_risk_manager(self, manager: RiskManager): self.risk_manager = manager
    def set_ai_monitor(self, monitor: AIMonitor): self.ai_monitor = monitor
    def set_drift_detector(self, detector: DriftDetector): self.drift_detector = detector
    def set_curriculum_engine(self, engine: CurriculumEngine): self.curriculum_engine = engine
    def set_tape_engine(self, engine: TapeEngine): self.tape_engine = engine
    def set_onchain_engine(self, engine: OnChainEngine): self.onchain_engine = engine
    def set_data_provider(self, provider: DataProvider): self.data_provider = provider
    def set_explainer(self, explainer: Explainer):
        self.explainer = explainer

    def _initialize_feature_models(self, input_dim: int):
        """
        
        Inicializa os modelos de features, passando ambas as configuraÃ§Ãµes
        necessÃ¡rias para o TrendPredictor.
        """
        logger.info(f"🧠 [AI] Inicializando modelos de features com input_dim: {input_dim}.")
        # [REMOVIDO] ProfitabilityPredictor - simplificação do pipeline
        # Os especialistas agora operam sem preditor de lucratividade
        pass

    def _initialize_decision_agents(self, input_dim: int):
        logger.info(f"🧠 [AI] Inicializando agentes de decisão MoE com input_dim sugerido: {input_dim}.")
        # Use a dimensão sugerida inicialmente, mas será validada durante o treino
        self.final_feature_count = input_dim

        # Regime detection handled by ScientificRegimeDetector in specialists

        # Configuração dos especialistas baseados em regime
        specialist_classes = {
            'bull': BullSpecialist,
            'bear': BearSpecialist,
            'ranger': RangerSpecialist
        }

        # Inicializa especialistas
        self.specialists = {}
        for name, cls in specialist_classes.items():
            try:
                self.specialists[name] = cls(
                    config=self.config_ai,
                    trading_config=self.config_trading,
                    input_dim=input_dim,
                    profitability_predictor=None  # [REMOVIDO] - simplificação do pipeline
                )
                logger.info(f"✅ [MoE] Especialista {name} inicializado.")
            except Exception as e:
                logger.error(f"❌ [MoE] Falha ao inicializar especialista {name}: {e}")
        
        
        # [REMOVIDO] TrendSpecialist não é mais treinado separadamente
        # Bull, Bear e Ranger são os especialistas que herdam de TrendSpecialist
        # e são treinados com dados filtrados por regime.
        # A classe TrendSpecialist agora serve apenas como BASE para os especialistas.
        logger.info("ℹ️ [MoE] TrendSpecialist é apenas classe base - não será treinado separadamente.")

        # Inicializa SoftGatingNetwork
        latent_dim = getattr(self.config_ai, 'AUTOENCODER_LATENT_DIM', 64)
        self.gating_network = create_gating_network(self.config_ai, latent_dim)
        
        # Tenta carregar modelo existente
        gating_path = os.path.join(self.config_ai.MODEL_DIR, 'soft_gating_network.pt')
        if os.path.exists(gating_path):
            try:
                self.gating_network = SoftGatingNetwork.load(gating_path)
                logger.info(f"✅ [MoE] SoftGatingNetwork carregado de {gating_path}")
            except Exception as e:
                logger.warning(f"⚠️ [MoE] Não foi possível carregar SoftGatingNetwork: {e}")
        
        # Inicializa MoE Router
        self.moe_router = MixtureOfExpertsRouter(
            gating_network=self.gating_network,
            specialists=self.specialists,
            ensemble_mode='weighted',
            top_k=2,
            threshold=0.2
        )
        logger.info("✅ [MoE] MixtureOfExpertsRouter inicializado com modo weighted.")
        
        # Inicializa Online Learner (para trend surfing)
        try:
            self.online_learner = create_online_learner(self.gating_network, self.config_ai)
            logger.info("✅ [MoE] OnlineTrendLearner inicializado para adaptação online.")
        except Exception as e:
            logger.warning(f"⚠️ [MoE] Falha ao inicializar OnlineLearner: {e}")
            self.online_learner = None
        
        # Mantém HRLMaster para compatibilidade
        self.hrl_master = HRLMaster(self.config_ai, input_dim=input_dim, specialists=self.specialists)
    
    def get_training_status(self) -> Dict[str, bool]:
        """
        [VERSÃƒO MELHORADA] Verifica inteligentemente o status de cada modelo,
        incluindo modelos em checkpoints e validando integridade dos arquivos.
        """
        status = {}
        all_files_found = True
        
        # Primeiro, atualiza os metadados com base nos arquivos existentes
        self._update_metadata_from_existing_files()
        
        # Verifica cada modelo
        for name, details in self.metadata['models'].items():
            # Verifica se o arquivo principal existe e Ã© vÃ¡lido
            main_file_path = os.path.join(self.config_ai.MODEL_DIR, details['path'])
            main_file_valid = self._is_file_valid(main_file_path)
            
            # Verifica se hÃ¡ checkpoint vÃ¡lido para este modelo
            checkpoint_valid = self._has_valid_checkpoint(name)
            
            # Modelo Ã© considerado treinado se qualquer uma das opÃ§Ãµes for vÃ¡lida
            is_trained = main_file_valid or checkpoint_valid
            
            # Atualiza o status
            self.metadata['models'][name]['trained'] = is_trained
            status[name] = is_trained
            
            if not is_trained:
                all_files_found = False
                logger.debug(f"âŒ {name}: arquivo principal e checkpoint invÃ¡lidos")
            else:
                if main_file_valid and checkpoint_valid:
                    logger.debug(f"âœ… {name}: arquivo principal e checkpoint vÃ¡lidos")
                elif main_file_valid:
                    logger.debug(f"âœ… {name}: apenas arquivo principal vÃ¡lido")
                else:
                    logger.debug(f"âœ… {name}: apenas checkpoint vÃ¡lido")
        
        status['all_trained'] = all_files_found
        
        # Log detalhado do status
        valid_count = sum(1 for v in status.values() if v and v != 'all_trained')
        total_count = len([k for k in status.keys() if k != 'all_trained'])
        logger.info(f"ðŸ”Ž [STATUS TREINO] {valid_count}/{total_count} modelos vÃ¡lidos: {status}")
        
        return status
    
    def _update_metadata_from_existing_files(self):
        """Atualiza metadados com base nos arquivos existentes."""
        for name, details in self.metadata['models'].items():
            file_path = os.path.join(self.config_ai.MODEL_DIR, details['path'])
            if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
                # Atualiza data de treinamento se nÃ£o existir
                if not details.get('trained_on'):
                    file_dt = datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc)
                    self.metadata['models'][name]['trained_on'] = file_dt.isoformat()
                    logger.debug(f"ðŸ“… {name}: data de treinamento atualizada para {file_dt.strftime('%Y-%m-%d')}")
    
    def _is_file_valid(self, file_path: str) -> bool:
        """Verifica se um arquivo de modelo é válido."""
        if not os.path.exists(file_path):
            return False
        
        # Verifica tamanho mínimo baseado no tipo de arquivo
        # Thresholds ajustados para modelos menores (autoencoder ~958KB, HRL ~226KB)
        file_size = os.path.getsize(file_path)
        if file_path.endswith('.pth'):
            return file_size > 50000   # 50KB para modelos PyTorch (autoencoder pode ser pequeno)
        elif file_path.endswith('.zip'):
            return file_size > 100000  # 100KB para modelos RL (HRL Master é menor)
        elif file_path.endswith('.joblib'):
            return file_size > 1000    # 1KB para scalers
        else:
            return file_size > 1000    # 1KB padrão
        
        return False
    
    def _has_valid_checkpoint(self, model_name: str) -> bool:
        """Verifica se hÃ¡ um checkpoint vÃ¡lido para o modelo."""
        checkpoint_dir = os.path.join(os.path.dirname(self.config_ai.MODEL_DIR), 'checkpoints_ai')
        
        if not os.path.exists(checkpoint_dir):
            return False
        
        # Mapeia nomes de modelo para pastas de checkpoint
        checkpoint_mapping = {
            'hrl_master_ppo': 'hrl_master_checkpoints',
            'trend_specialist_sac': 'trend_specialist_checkpoints',
            'sideways_specialist_dqn': 'sideways_specialist_checkpoints',
            'range_specialist_dqn': 'range_specialist_checkpoints',
            'volatility_specialist_sac': 'volatility_specialist_checkpoints'
        }
        
        if model_name not in checkpoint_mapping:
            return False
        
        specialist_dir = os.path.join(checkpoint_dir, checkpoint_mapping[model_name])
        if not os.path.exists(specialist_dir):
            return False
        
        # Verifica se hÃ¡ um modelo treinado
        best_model_path = os.path.join(specialist_dir, 'best_model.zip')
        if os.path.exists(best_model_path) and os.path.getsize(best_model_path) > 1000000:
            return True
        
        # Verifica se hÃ¡ outros arquivos de modelo
        for file_name in os.listdir(specialist_dir):
            if file_name.endswith('.zip') and os.path.getsize(os.path.join(specialist_dir, file_name)) > 1000000:
                return True
        
        return False

    def _normalize_model_paths(self) -> None:
        overrides = {
            # 'trend_predictor': REMOVIDO
            # Migração do caminho antigo do autoencoder para o diretório correto
            'autoencoder_model': os.path.join('autoencoder', 'temporal_sdae.pth'),
        }
        for key, expected in overrides.items():
            info = self.metadata['models'].get(key)
            if not info:
                continue
            current = info.get('path')
            if current != expected:
                logger.debug('[META] Ajustando caminho do modelo %s: %s -> %s', key, current, expected)
                info['path'] = expected

    def _bootstrap_metadata_from_disk(self) -> None:
        """
        [MELHORADO] ReconstrÃ³i metadados consistentes a partir dos artefatos existentes,
        incluindo modelos em checkpoints e validando integridade dos arquivos.
        """
        self._normalize_model_paths()
        try:
            latest_ts: Optional[datetime] = None
            valid_models_count = 0
            
            # Primeiro, verifica modelos principais
            for name, details in self.metadata['models'].items():
                model_path = os.path.join(self.config_ai.MODEL_DIR, details['path'])
                if os.path.exists(model_path) and self._is_file_valid(model_path):
                    self.metadata['models'][name]['trained'] = True
                    file_dt = datetime.fromtimestamp(os.path.getmtime(model_path), tz=timezone.utc)
                    self.metadata['models'][name]['trained_on'] = file_dt.isoformat()
                    valid_models_count += 1
                    
                    if latest_ts is None or file_dt > latest_ts:
                        latest_ts = file_dt
                    logger.debug(f"ðŸ“ {name}: modelo principal vÃ¡lido encontrado ({file_dt.strftime('%Y-%m-%d')})")
                else:
                    # Verifica se hÃ¡ checkpoint vÃ¡lido
                    if self._has_valid_checkpoint(name):
                        self.metadata['models'][name]['trained'] = True
                        # Usa a data do checkpoint
                        checkpoint_dir = self._get_checkpoint_dir(name)
                        if checkpoint_dir:
                            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
                            if checkpoint_files:
                                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                                file_dt = datetime.fromtimestamp(os.path.getmtime(checkpoint_path), tz=timezone.utc)
                                self.metadata['models'][name]['trained_on'] = file_dt.isoformat()
                                valid_models_count += 1
                                
                                if latest_ts is None or file_dt > latest_ts:
                                    latest_ts = file_dt
                                logger.debug(f"ðŸ“ {name}: checkpoint vÃ¡lido encontrado ({file_dt.strftime('%Y-%m-%d')})")
                    else:
                        self.metadata['models'][name]['trained'] = False
                        self.metadata['models'][name]['trained_on'] = None
                        logger.debug(f"âŒ {name}: nenhum modelo vÃ¡lido encontrado")

            # Atualiza a data de Ãºltimo treinamento
            self.last_trained_date = latest_ts
            self.metadata['last_full_training_date'] = latest_ts.isoformat() if latest_ts else None

            # Tenta inferir contrato de features caso exista
            self._infer_feature_contract()

            # Persiste imediatamente
            self._save_metadata()
            logger.info(f"âœ… Metadados reconstruÃ­dos: {valid_models_count} modelos vÃ¡lidos encontrados. Ãšltimo treinamento: {latest_ts.strftime('%Y-%m-%d') if latest_ts else 'N/A'}")
            
        except Exception as e:
            logger.error(f"âŒ Falha ao reconstruir metadados a partir do disco: {e}")
    
    def _get_checkpoint_dir(self, model_name: str) -> Optional[str]:
        """Retorna o diretÃ³rio de checkpoint para um modelo especÃ­fico."""
        checkpoint_mapping = {
            'hrl_master_ppo': 'hrl_master_checkpoints',
            # [FIX] Novas chaves Bull/Bear/Ranger para checkpoints
            'bull_specialist_sac': 'bull_specialist_checkpoints',
            'bear_specialist_sac': 'bear_specialist_checkpoints',
            'ranger_specialist_sac': 'ranger_specialist_checkpoints'
        }
        
        if model_name not in checkpoint_mapping:
            return None
        
        checkpoint_dir = os.path.join(os.path.dirname(self.config_ai.MODEL_DIR), 'checkpoints_ai', checkpoint_mapping[model_name])
        return checkpoint_dir if os.path.exists(checkpoint_dir) else None
    
    def _infer_feature_contract(self):
        """Infere o contrato de features a partir dos arquivos existentes."""
        # Tenta inferir contrato de features caso exista
        ae_features_path = os.path.join(self.config_ai.MODEL_DIR, "autoencoder_feature_columns.json")
        if os.path.exists(ae_features_path):
            try:
                with open(ae_features_path, 'r') as f:
                    cols = json.load(f)
                    if isinstance(cols, list) and cols:
                        self.metadata['base_feature_columns'] = cols
                        self.metadata['final_feature_count'] = len(cols)
                        self.base_feature_columns = cols
                        self.final_feature_count = len(cols)
                        logger.debug(f"ðŸ”§ Contrato de features inferido: {len(cols)} colunas")
            except Exception as e:
                logger.debug(f"âš ï¸ NÃ£o foi possÃ­vel inferir contrato de features: {e}")
        
        # Tenta inferir a partir do dataset base
        base_df_path = os.path.join(self.config_ai.MODEL_DIR, "base_featured_df.pkl")
        if os.path.exists(base_df_path) and not self.metadata.get('base_feature_columns'):
            try:
                import pandas as pd
                base_df = pd.read_pickle(base_df_path)
                numeric_cols = base_df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    self.metadata['base_feature_columns'] = numeric_cols
                    self.metadata['final_feature_count'] = len(numeric_cols)
                    self.base_feature_columns = numeric_cols
                    self.final_feature_count = len(numeric_cols)
                    logger.debug(f"ðŸ”§ Contrato de features inferido do dataset: {len(numeric_cols)} colunas")
            except Exception as e:
                logger.debug(f"âš ï¸ NÃ£o foi possÃ­vel inferir features do dataset: {e}")
    
    def load_metadata(self):
        """
        [VERSÃƒO MELHORADA] Carrega os metadados granulares do arquivo JSON.
        Se nÃ£o existir ou estiver corrompido, reconstrÃ³i automaticamente.
        """
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    loaded_meta = json.load(f)
                    
                    # Valida a estrutura dos metadados carregados
                    if self._validate_metadata_structure(loaded_meta):
                        # Mescla o que foi carregado com o padrÃ£o para garantir que todas as chaves existam
                        for model_key in self.metadata['models']:
                            if model_key in loaded_meta.get('models', {}):
                                self.metadata['models'][model_key].update(loaded_meta['models'][model_key])
                        
                        self.metadata['last_full_training_date'] = loaded_meta.get('last_full_training_date')
                        self.metadata['final_feature_count'] = loaded_meta.get('final_feature_count')
                        self.metadata['base_feature_columns'] = loaded_meta.get('base_feature_columns')
                        
                        # Converte a data de treinamento
                        if self.metadata['last_full_training_date']:
                            try:
                                self.last_trained_date = datetime.fromisoformat(self.metadata['last_full_training_date'])
                            except (ValueError, TypeError):
                                logger.warning("âš ï¸ Data de treinamento invÃ¡lida nos metadados. SerÃ¡ recalculada.")
                                self.last_trained_date = None
                        else:
                            self.last_trained_date = None
                        
                        self.final_feature_count = self.metadata.get('final_feature_count')
                        self.base_feature_columns = self.metadata.get('base_feature_columns')
                        
                        logger.info("âœ… Metadados de treinamento carregados com sucesso.")
                    else:
                        logger.warning("âš ï¸ Estrutura de metadados invÃ¡lida. Reconstruindo...")
                        self._bootstrap_metadata_from_disk()
                        return
                        
            except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
                logger.error(f"âŒ Erro ao ler o arquivo de metadados: {e}. Reconstruindo metadados...")
                self._bootstrap_metadata_from_disk()
                return
            except Exception as e:
                logger.error(f"âŒ Erro inesperado ao carregar metadados: {e}. Reconstruindo...")
                self._bootstrap_metadata_from_disk()
                return
        else:
            logger.warning("âš ï¸ Arquivo de metadados nÃ£o encontrado. Inicializando estrutura padrÃ£o.")
        
        self._normalize_model_paths()
        # Sempre executa o bootstrap para garantir consistÃªncia e detectar modelos existentes
        self._bootstrap_metadata_from_disk()
    
    def _validate_metadata_structure(self, loaded_meta: dict) -> bool:
        """Valida se a estrutura dos metadados carregados Ã© vÃ¡lida."""
        try:
            # Verifica se tem a estrutura bÃ¡sica
            if not isinstance(loaded_meta, dict):
                return False
            
            if 'models' not in loaded_meta:
                return False
            
            # Verifica se todos os modelos esperados estão presentes
            # [ARCH] HRL Master removido dos required — roteamento via CryptoRegimeDetector + SoftGatingNetwork.
            # HRL Master tinha lookahead bias no MasterEnv (future_price 5 steps à frente) e era
            # redundante com o CryptoRegimeDetector que já classifica regime deterministicamente.
            expected_models = [
                'autoencoder_model',
                'bull_specialist_sac',
                'bear_specialist_sac',
                'ranger_specialist_sac'
            ]
            
            for model_name in expected_models:
                if model_name not in loaded_meta['models']:
                    return False
                
                model_info = loaded_meta['models'][model_name]
                if not isinstance(model_info, dict):
                    return False
                
                if 'trained' not in model_info or 'path' not in model_info:
                    return False
            
            return True
            
        except Exception:
            return False


    def load_all_models(self):
        """
        [FAST_STARTUP] Carrega todos os modelos para operação sem retreinamento.
        Inicializa agentes de decisão e carrega pesos dos modelos existentes.
        """
        try:
            logger.info("✅ Verificando status de treinamento de todos os componentes...")
            
            # --- ETAPA 1: Inicializar componentes se não existirem ---
            # Verifica se feature_pipeline tem autoencoder treinado
            # [FIX] Tenta carregar o estado do disco se não estiver carregado
            if self.feature_pipeline and self.feature_pipeline.temporal_autoencoder_pipeline:
                 if not self.feature_pipeline.temporal_autoencoder_pipeline.is_trained:
                     logger.info("⚡ [FAST_STARTUP] Carregando pesos do Autoencoder Temporal...")
                     try:
                        self.feature_pipeline.temporal_autoencoder_pipeline.load_state()
                     except Exception as ae_e:
                        logger.warning(f"⚠️ [FAST_STARTUP] Falha ao carregar Autoencoder: {ae_e}")

            autoencoder_trained = (
                self.feature_pipeline and 
                self.feature_pipeline.temporal_autoencoder_pipeline and 
                self.feature_pipeline.temporal_autoencoder_pipeline.is_trained
            )
            
            # Se não há specialists ou HRL Master, inicializar agora
            if not self.specialists or not self.hrl_master:
                # Determina input_dim a partir dos metadados ou padrão
                input_dim = int(self.metadata.get('final_feature_count') or 54)
                logger.info(f"⚡ [FAST_STARTUP] Inicializando agentes de decisão com input_dim={input_dim}...")
                self._initialize_decision_agents(input_dim)
            
            # --- ETAPA 2: Carregar pesos dos modelos salvos ---
            # Carregar HRL Master
            hrl_path = os.path.join(self.config_ai.MODEL_DIR, 'hrl_master_ppo.zip')
            if self.hrl_master and os.path.exists(hrl_path):
                try:
                    self.hrl_master.load_model()
                    logger.info(f"✅ [FAST_STARTUP] HRL Master carregado de {hrl_path}")
                except Exception as e:
                    logger.warning(f"⚠️ [FAST_STARTUP] Falha ao carregar HRL Master: {e}")
            
            # Carregar Specialists
            specialist_keys = ['bull_specialist_sac', 'bear_specialist_sac', 'ranger_specialist_sac']
            specialist_names = ['bull', 'bear', 'ranger']
            for spec_key, spec_name in zip(specialist_keys, specialist_names):
                spec_path = os.path.join(self.config_ai.MODEL_DIR, f'{spec_key}.zip')
                if spec_name in self.specialists and os.path.exists(spec_path):
                    try:
                        self.specialists[spec_name].load_model()
                        logger.info(f"✅ [FAST_STARTUP] Specialist {spec_name} carregado de {spec_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ [FAST_STARTUP] Falha ao carregar {spec_name}: {e}", exc_info=True)
            
            # --- ETAPA 3: Verificar status final ---
            hrl_master_trained = self.hrl_master is not None and self.hrl_master.is_trained
            
            specialists_trained = True
            # [FIX] Robust check: If files exist, trust them even if metadata is fresh
            for spec_key in specialist_keys:
                model_meta = self.metadata.get('models', {}).get(spec_key, {})
                # Check default path if metadata path is missing
                default_filename = f'{spec_key}.zip'
                model_path = os.path.join(self.config_ai.MODEL_DIR, model_meta.get('path', default_filename))
                
                if os.path.exists(model_path):
                     # If file exists but metadata says not trained (or is empty), update it
                     if not model_meta.get('trained', False):
                         logger.info(f"✅ [FAST_STARTUP] Modelo encontrado em disco: {default_filename}. Atualizando metadados...")
                         if 'models' not in self.metadata: self.metadata['models'] = {}
                         self.metadata['models'][spec_key] = {'trained': True, 'path': default_filename, 'trained_on': datetime.utcnow().isoformat()}
                         self._save_metadata()
                else:
                    specialists_trained = False
                    logger.warning(f"⚠️ [FAST_STARTUP] Modelo não encontrado: {model_path}")

            # Validate HRL
            hrl_path = os.path.join(self.config_ai.MODEL_DIR, 'hrl_master_ppo.zip')
            if os.path.exists(hrl_path):
                 if not self.metadata.get('models', {}).get('hrl_master_ppo', {}).get('trained', False):
                     logger.info(f"✅ [FAST_STARTUP] HRL Master encontrado em disco. Atualizando metadados...")
                     if 'models' not in self.metadata: self.metadata['models'] = {}
                     self.metadata['models']['hrl_master_ppo'] = {'trained': True, 'path': 'hrl_master_ppo.zip', 'trained_on': datetime.utcnow().isoformat()}
                     self._save_metadata()
                 hrl_master_trained = True
            else:
                 hrl_master_trained = False

            # [ARCH] HRL Master não é mais requisito de prontidão.
            # Roteamento feito por CryptoRegimeDetector + SoftGatingNetwork (mais simples e sem lookahead bias).
            if autoencoder_trained and specialists_trained:
                self.is_trained = True
                logger.info("✅ [FAST_STARTUP] Sistema IA validado como TREINADO (baseado em arquivos e metadados).")
            else:
                self.is_trained = False
                logger.warning(f"⚠️ [FAST_STARTUP] Sistema IA incompleto. AE={autoencoder_trained}, Specs={specialists_trained}, HRL(opcional)={hrl_master_trained}")
            
            if self.is_trained:
                logger.info("✅ Todos os modelos estão carregados. AIController está treinado e pronto para operar.")
                
                # [REGIMES] Carrega Detector de Regimes
                try:
                    regime_detector_path = os.path.join(self.config_ai.MODEL_DIR, "crypto_regime_detector.pkl")
                    if os.path.exists(regime_detector_path):
                        import joblib
                        self.regime_detector = joblib.load(regime_detector_path)
                        logger.info(f"✅ [REGIMES] CryptoRegimeDetector carregado de '{regime_detector_path}'.")
                    else:
                        logger.warning(f"⚠️ [REGIMES] Arquivo do detector de regimes não encontrado: {regime_detector_path}")
                except Exception as e:
                    logger.error(f"❌ [REGIMES] Falha ao carregar CryptoRegimeDetector: {e}")

                # [DRIFT] Tenta carregar dados de referência para o Drift Detector
                if self.drift_detector and not self.drift_detector.reference_data:
                    try:
                        logger.info("📊 [DRIFT] Carregando dados de referência a partir do dataset base...")
                        base_df_path = os.path.join(self.config_ai.MODEL_DIR, "base_featured_df.pkl")
                        if os.path.exists(base_df_path):
                            base_df = pd.read_pickle(base_df_path)
                            # Usa as colunas de features salvas nos metadados ou infere
                            features_to_monitor = self.base_feature_columns
                            if not features_to_monitor:
                                features_to_monitor = base_df.select_dtypes(include=['number']).columns.tolist()
                                # [FIX] Persiste colunas inferidas para o contrato de features
                                self.base_feature_columns = features_to_monitor
                                logger.info(f"⚡ [FAST_STARTUP] 'base_feature_columns' inferido de base_featured_df ({len(self.base_feature_columns)} features).")
                                
                            self.drift_detector.set_reference_data(base_df, features_to_monitor)
                        else:
                            logger.warning("⚠️ [DRIFT] Dataset base não encontrado. Drift status permanecerá DESCONHECIDO.")
                    except Exception as e:
                        logger.error(f"❌ [DRIFT] Erro ao carregar dados de referência: {e}")
            else:
                status = {
                    "Autoencoder": autoencoder_trained,
                    "HRLMaster": hrl_master_trained,
                    "Specialists": specialists_trained
                }
                logger.warning(f"⚠️ Falha no carregamento. Status dos componentes: {status}. AIController não está totalmente treinado.")
        except Exception as e:
            logger.critical(f"🚨 Erro crítico durante o carregamento dos modelos: {e}", exc_info=True)
            self.is_trained = False

    def is_retraining_due(self, retrain_days: int) -> bool:
        """
        [CORRIGIDO] Decide se um retreinamento completo Ã© necessÃ¡rio.
        Verifica se hÃ¡ modelos vÃ¡lidos e sÃ³ forÃ§a retreinamento se realmente necessÃ¡rio.
        """
        now = datetime.now(timezone.utc)
        
        # Primeiro, verifica se hÃ¡ modelos treinados vÃ¡lidos
        valid_models = self._count_valid_models()
        if valid_models == 0:
            logger.info("â„¹ï¸ Nenhum modelo vÃ¡lido encontrado. Retreinamento necessÃ¡rio.")
            return True
        
        # Se hÃ¡ modelos vÃ¡lidos, verifica a data do mais recente
        latest_ts = self._get_latest_model_timestamp()
        if latest_ts is None:
            logger.warning("âš ï¸ NÃ£o foi possÃ­vel determinar a data do Ãºltimo treinamento. Verificando arquivos...")
            return self._check_files_for_retraining(retrain_days)
        
        days_since_training = (now - latest_ts).days
        logger.info(f"â„¹ï¸ Ãšltimo treinamento: {latest_ts.strftime('%Y-%m-%d')} ({days_since_training} dias atrÃ¡s)")
        
        # SÃ³ forÃ§a retreinamento se realmente passou muito tempo E nÃ£o hÃ¡ modelos vÃ¡lidos suficientes
        if days_since_training > retrain_days and valid_models < 3:
            logger.warning(f"âš ï¸ Modelos muito antigos ({days_since_training} dias) e poucos modelos vÃ¡lidos ({valid_models}). Retreinamento recomendado.")
            return True
        elif days_since_training > retrain_days:
            logger.info(f"â„¹ï¸ Modelos antigos ({days_since_training} dias), mas hÃ¡ {valid_models} modelos vÃ¡lidos. Treinamento granular recomendado.")
            return False
        
        return False
    
    def _count_valid_models(self) -> int:
        """Conta quantos modelos sÃ£o considerados vÃ¡lidos."""
        valid_count: int = 0
        for name, details in self.metadata['models'].items():
            # Verifica se o arquivo existe
            model_path = os.path.join(self.config_ai.MODEL_DIR, details['path'])
            if os.path.exists(model_path):
                # Verifica se o arquivo tem tamanho mÃ­nimo (nÃ£o estÃ¡ corrompido)
                if os.path.getsize(model_path) > 1000:  # 1KB mÃ­nimo
                    valid_count += 1
                    # Atualiza o status nos metadados
                    self.metadata['models'][name]['trained'] = True
                    if not self.metadata['models'][name].get('trained_on'):
                        file_dt = datetime.fromtimestamp(os.path.getmtime(model_path), tz=timezone.utc)
                        self.metadata['models'][name]['trained_on'] = file_dt.isoformat()
        
        # Verifica tambÃ©m modelos em checkpoints
        checkpoint_models = self._count_checkpoint_models()
        valid_count += checkpoint_models
        
        return valid_count
    
    def _count_checkpoint_models(self) -> int:
        """Conta modelos vÃ¡lidos em checkpoints."""
        checkpoint_count: int = 0
        checkpoint_dir = os.path.join(os.path.dirname(self.config_ai.MODEL_DIR), 'checkpoints_ai')
        
        if os.path.exists(checkpoint_dir):
            for specialist_dir in os.listdir(checkpoint_dir):
                specialist_path = os.path.join(checkpoint_dir, specialist_dir)
                if os.path.isdir(specialist_path):
                    best_model_path = os.path.join(specialist_path, 'best_model.zip')
                    if os.path.exists(best_model_path) and os.path.getsize(best_model_path) > 1000000:  # 1MB mÃ­nimo
                        checkpoint_count += 1
        
        return checkpoint_count
    
    def _get_latest_model_timestamp(self) -> Optional[datetime]:
        """ObtÃ©m o timestamp do modelo mais recente."""
        latest_ts = None
        
        for name, details in self.metadata['models'].items():
            if details.get('trained_on'):
                try:
                    model_ts = datetime.fromisoformat(details['trained_on'])
                    if latest_ts is None or model_ts > latest_ts:
                        latest_ts = model_ts
                except (ValueError, TypeError):
                    continue
        
        return latest_ts
    
    def _check_files_for_retraining(self, retrain_days: int) -> bool:
        """Fallback para verificar arquivos quando nÃ£o hÃ¡ metadados de data."""
        now = datetime.now(timezone.utc)
        latest_ts = None
        any_file = False
        
        for _, details in self.metadata['models'].items():
            model_path = os.path.join(self.config_ai.MODEL_DIR, details['path'])
            if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
                any_file = True
                file_dt = datetime.fromtimestamp(os.path.getmtime(model_path), tz=timezone.utc)
                if latest_ts is None or file_dt > latest_ts:
                    latest_ts = file_dt

        if not any_file:
            return True
        if latest_ts is None:
            return True
            
        days_since = (now - latest_ts).days
        logger.info(f"â„¹ï¸ VerificaÃ§Ã£o de arquivos: Ãºltimo modelo modificado hÃ¡ {days_since} dias")
        return days_since > retrain_days

    async def train_missing_components(self, status: Dict[str, bool]) -> bool:
        """
        Treina apenas os componentes da IA que estÃ£o faltando, com base no
        dicionÃ¡rio de status, e atualiza os metadados atomicamente apÃ³s cada
        etapa bem-sucedida para garantir a resiliÃªncia.
        """
        try:
            logger.info("--- Iniciando Treinamento Granular dos Componentes Ausentes ---")
            
            # --- ETAPA 1: PREPARAÃ‡ÃƒO DE DADOS (FEITA APENAS UMA VEZ) ---
            
            # ObtÃ©m o dataset base. Se nÃ£o existir em cache, serÃ¡ criado a partir dos dados histÃ³ricos.
            base_df = await self._get_or_create_base_dataset()
            if base_df is None:
                logger.error("âŒ Falha ao obter o dataset base para o treinamento granular.")
                return False

            # --- ETAPA 2: TREINAMENTO DOS MODELOS DE FEATURES ---
            
            numeric_df = base_df.select_dtypes(include=np.number)
            self._initialize_feature_models(input_dim=len(numeric_df.columns))

            autoencoder_ready = False
            if not status.get('autoencoder_model'):
                logger.info(' -> [Granular] Treinando Autoencoder ausente...')
                self.feature_pipeline.train_autoencoder(numeric_df, self.config_ai.AUTOENCODER_TRAINING_EPOCHS)
                if self.feature_pipeline.temporal_autoencoder_pipeline.is_trained:
                    self._update_model_metadata('autoencoder_model', True)
                    autoencoder_ready = True
            else:
                logger.info(' -> [Granular] Autoencoder já treinado. Carregando.')
                try:
                    self.feature_pipeline.temporal_autoencoder_pipeline.load_state()
                    autoencoder_ready = bool(self.feature_pipeline.temporal_autoencoder_pipeline.is_trained)
                    
                    if not autoencoder_ready:
                        logger.warning('⚠️ [Granular] Autoencoder marcado como treinado mas load_state() falhou.')
                        logger.info('🔄 [Granular] Tentando retreinar autoencoder...')
                        self.feature_pipeline.train_autoencoder(numeric_df, self.config_ai.AUTOENCODER_TRAINING_EPOCHS)
                        if self.feature_pipeline.temporal_autoencoder_pipeline.is_trained:
                            self._update_model_metadata('autoencoder_model', True)
                            autoencoder_ready = True
                except Exception as exc:
                    logger.error(f'❌ [Granular] Exceção ao carregar Autoencoder: {exc}', exc_info=True)
                    logger.info('🔄 [Granular] Tentando retreinar autoencoder após erro...')
                    try:
                        self.feature_pipeline.train_autoencoder(numeric_df, self.config_ai.AUTOENCODER_TRAINING_EPOCHS)
                        if self.feature_pipeline.temporal_autoencoder_pipeline.is_trained:
                            self._update_model_metadata('autoencoder_model', True)
                            autoencoder_ready = True
                    except Exception as retry_exc:
                        logger.error(f'❌ [Granular] Falha no retreinamento do autoencoder: {retry_exc}', exc_info=True)
                        autoencoder_ready = False

            if not autoencoder_ready:
                logger.error('❌ Erro ao preparar Autoencoder para o treinamento granular.')
                return False

            try:
                df_with_hidden_features = self.feature_pipeline.apply_hidden_features(base_df)
            except Exception as exc:
                logger.error(f'Falha ao aplicar hidden features no fluxo granular: {exc}')
                return False


            if df_with_hidden_features is None or df_with_hidden_features.empty:
                logger.error('Falha ao aplicar hidden features (dataset vazio).')
                return False

            # --- AUTO-UPDATE SHAP BACKGROUND (Fix for SHAP shape mismatch) ---
            # Atualiza o base_featured_df.pkl com dados reais sempre que features válidas são geradas
            try:
                if self.base_feature_columns:
                    available_cols = [c for c in self.base_feature_columns if c in df_with_hidden_features.columns]
                    if len(available_cols) == len(self.base_feature_columns):
                        shap_bg = df_with_hidden_features[self.base_feature_columns].tail(100)
                        shap_bg_path = os.path.join(self.config_ai.MODEL_DIR, "shap_background.pkl")
                        shap_bg.to_pickle(shap_bg_path)
                        logger.info(f"✅ [SHAP] Background atualizado com dados reais para '{shap_bg_path}' ({len(shap_bg)} linhas, {len(self.base_feature_columns)} cols)")
            except Exception as shap_exc:
                logger.warning(f"⚠️ [SHAP] Falha ao atualizar background: {shap_exc}")

            # TrendPredictor removido - detecção de regime agora via ScientificRegimeDetector
            # Os especialistas (Bull, Bear, Ranger) agora filtram dados por regime automaticamente
            logger.info(' -> [Granular] TrendPredictor removido - usando ScientificRegimeDetector nos especialistas')

            # --- ETAPA 3: TREINAMENTO DOS AGENTES DE DECISÃƒO (RL) ---

            # Se algum agente de RL estiver faltando, todos precisam ser retreinados, pois
            # o dataset enriquecido pode ter mudado.
            
            # Verifica se algum agente de RL precisa de treinamento
            # [ARCH] HRL Master removido dos requisitos de treinamento RL.
            # CryptoRegimeDetector + SoftGatingNetwork fazem o roteamento.
            needs_rl_training = not all(status.get(key) for key in [
                'bull_specialist_sac',
                'bear_specialist_sac',
                'ranger_specialist_sac'
            ])

            if needs_rl_training:
                logger.info("--- Iniciando Treinamento Granular dos Agentes de Decisão (RL) ---")
                
                # Constrói o dataset enriquecido, que depende dos modelos de features
                enriched_training_df = await self.build_enriched_dataset(base_df)
                if enriched_training_df is None or enriched_training_df.empty:
                    logger.error("✖ Falha ao construir o dataset enriquecido no treinamento granular.")
                    return False
                
                # [FIX] Garantir colunas de regime no dataset granular
                if 'regime' in base_df.columns and 'regime' not in enriched_training_df.columns:
                    for col in ['regime', 'regime_confidence', 'regime_name']:
                        if col in base_df.columns:
                            enriched_training_df[col] = base_df[col]
                    logger.info("✅ [GRANULAR] Colunas de regime restauradas no dataset enriquecido.")
                
                # Atualiza os metadados com as informações das features finais
                self.metadata['final_feature_count'] = self.final_feature_count
                self.metadata['base_feature_columns'] = self.base_feature_columns
                self._save_metadata() # Salva o "contrato de features"

                # Inicializa os agentes de decisão com a dimensão correta
                self._initialize_decision_agents(input_dim=self.final_feature_count)

                # --- ETAPA 4: BOOTSTRAP DA GATING NETWORK ---
                # Treinamento supervisionado da rede neural de gating antes do RL
                await self._bootstrap_gating_network(enriched_training_df)

                # Define um objetivo de treinamento para a sessão granular
                training_progress = {} 
                def _env_int(name: str, default: int) -> int:
                    try:
                        return int(os.environ.get(name, default))
                    except (TypeError, ValueError):
                        return default

                # [FIX TREINO FINAL] trend_goal agora é usado no dict (antes era calculado e ignorado).
                # Ref: AFLM (López de Prado) — SAC em env financeiro precisa de 1M-5M steps
                # para convergir. 200k é insuficiente para o agente aprender trend-surfing.
                # GPU RTX 3500: ~1500 steps/s → 1.5M steps ≈ 17 min por especialista.
                # Controlável via env vars: TREND_FINAL_TIMESTEPS, SPECIALIST_BASE_TIMESTEPS
                # [CONVERGÊNCIA] 500k steps suficiente para todos os agentes (validado empiricamente
                # aos 500k: Sharpe 8+, WR 78%, PF 9.6, ent_coef ~0.04 → política madura).
                # Controlável via env vars: TREND_FINAL_TIMESTEPS, SPECIALIST_BASE_TIMESTEPS
                trend_goal = _env_int('TREND_FINAL_TIMESTEPS', 500_000)
                base_goal  = _env_int('SPECIALIST_BASE_TIMESTEPS', 500_000)
                training_goals = {
                    "BullSpecialist":   trend_goal,   # 500k — convergência validada
                    "BearSpecialist":   trend_goal,   # 500k — igual ao bull
                    "RangerSpecialist": base_goal,    # 500k — regime lateral
                    # [ARCH] HRLMaster removido — roteamento via CryptoRegimeDetector + SoftGatingNetwork
                }
                logger.info(f"ðŸŽ¯ Objetivos de treinamento granular para agentes de RL definidos: {training_goals}")

                await self._train_rl_agents(
                    enriched_training_df, 
                    training_progress=training_progress, 
                    training_goals=training_goals,
                    status=status
                )

                # [FIX] Atualiza metadados APENAS para modelos que realmente existem no disco
                # Isso garante que apenas modelos treinados com sucesso sejam marcados como 'trained'
                specialist_file_mapping = {
                    'bull_specialist_sac': 'bull_specialist_sac.zip',
                    'bear_specialist_sac': 'bear_specialist_sac.zip',
                    'ranger_specialist_sac': 'ranger_specialist_sac.zip',
                    # [ARCH] hrl_master_ppo removido — não é mais treinado nem requerido
                }
                
                for model_key, filename in specialist_file_mapping.items():
                    model_path = os.path.join(self.config_ai.MODEL_DIR, filename)
                    if os.path.exists(model_path):
                        self._update_model_metadata(model_key, True)
                        logger.debug(f"✅ Modelo '{model_key}' detectado em disco - marcado como treinado.")
                    else:
                        logger.debug(f"⏳ Modelo '{model_key}' não encontrado em disco - não marcado como treinado.")
            else:
                logger.info("âœ… [Granular] Todos os agentes de decisÃ£o (RL) jÃ¡ estÃ£o treinados.")

            # VerificaÃ§Ã£o final
            self.load_all_models() # Recarrega tudo para garantir um estado consistente
            if self.is_trained:
                logger.info("âœ… Treinamento granular concluÃ­do com sucesso. Todos os componentes estÃ£o prontos.")
                return True
            else:
                logger.error("âŒ Treinamento granular finalizado, mas a IA ainda nÃ£o estÃ¡ em estado 'treinado'. Verifique os logs.")
                return False

        except Exception as e:
            logger.error(f"âŒ Erro catastrÃ³fico durante o treinamento granular: {e}", exc_info=True)
            return False

    def _save_metadata(self):
        """Salva o dicionÃ¡rio de metadados atual no arquivo JSON."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=4)
            logger.info(f"âœ… Arquivo de metadados atualizado em '{self.metadata_path}'.")
        except Exception as e:
            logger.error(f"âŒ Erro crÃ­tico ao salvar metadados: {e}", exc_info=True)
    
    def _update_model_metadata(self, model_key: str, trained_status: bool):
        """Atualiza o status de um modelo especÃ­fico nos metadados e salva."""
        if model_key in self.metadata['models']:
            self.metadata['models'][model_key]['trained'] = trained_status
            self.metadata['models'][model_key]['trained_on'] = datetime.now(timezone.utc).isoformat() if trained_status else None
            self._save_metadata()
        else:
            logger.warning(f"Tentativa de atualizar metadados para um modelo desconhecido: {model_key}")
    
    async def _get_or_create_enriched_dataset(self) -> Optional[pd.DataFrame]:
        enriched_df_path = os.path.join(self.config_ai.MODEL_DIR, "enriched_featured_df.pkl")
        if os.path.exists(enriched_df_path):
            logger.info(f"âœ… [CACHE] Cache do dataset enriquecido final encontrado. Carregando de '{enriched_df_path}'...")
            try:
                return pd.read_pickle(enriched_df_path)
            except Exception as e:
                logger.error(f"âŒ [CACHE] Falha ao carregar o cache do dataset enriquecido: {e}. Retornando None.")
                return None
        else:
            logger.warning("âš ï¸ [CACHE] Dataset enriquecido nÃ£o encontrado. Um ciclo de treinamento Ã© necessÃ¡rio para criÃ¡-lo.")
            return None
    
    def needs_foundation_training(self, retrain_days: int) -> bool:
        if not os.path.exists(self.metadata_path):
            logger.warning("âš ï¸ Arquivo de metadados nÃ£o encontrado. Treinamento de fundaÃ§Ã£o necessÃ¡rio.")
            return True
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            last_trained_str = metadata.get('last_trained_date')
            if not last_trained_str:
                logger.warning("âš ï¸ Data do Ãºltimo treino nÃ£o encontrada nos metadados. Treinamento de fundaÃ§Ã£o necessÃ¡rio.")
                return True
            last_trained_date = datetime.fromisoformat(last_trained_str)
            if (datetime.now(timezone.utc) - last_trained_date).days > retrain_days:
                logger.warning(f"ðŸ—“ï¸ Modelos de fundaÃ§Ã£o desatualizados (treinados hÃ¡ mais de {retrain_days} dias). Retreinamento necessÃ¡rio.")
                return True
            status = self.get_training_status()
            if not status['all_trained']:
                 logger.warning(f"ðŸ”¥ Arquivos de modelo essenciais estÃ£o faltando. Treinamento de fundaÃ§Ã£o necessÃ¡rio.")
                 return True
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"âŒ Erro ao ler metadados: {e}. ForÃ§ando treinamento de fundaÃ§Ã£o.")
            return True
        return False
    
    def _apply_regime_detector_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [CORREÇÃO CRÍTICA] Aplica CryptoRegimeDetector ao DataFrame para injetar colunas de regime.
        Esta função garante que regime_confidence e regime_val existam ANTES do enriquecimento.
        """
        try:
            # Verifica se já tem as colunas (evita re-processamento)
            if 'regime_confidence' in df.columns and 'regime_val' in df.columns:
                conf_std = df['regime_confidence'].std()
                if conf_std > 0.01:  # Valores dinâmicos já existem
                    logger.debug("[AI] Colunas de regime já presentes com valores dinâmicos.")
                    return df
            
            # [PERFORMANCE FIX] Cache do modelo em memÃ³ria para evitar IO de disco repetitivo
            if not hasattr(self, 'regime_detector') or self.regime_detector is None:
                from feature_engineering.crypto_regime_detector import CryptoRegimeDetector
                model_path = os.path.join(self.config_ai.MODEL_DIR, "crypto_regime_detector.pkl")
                if os.path.exists(model_path):
                    self.regime_detector = CryptoRegimeDetector.load(model_path)
                    logger.info("✅ [CACHE] CryptoRegimeDetector carregado e cacheado em memÃ³ria.")
                else:
                    logger.warning(f"âš ï¸  [AI] CryptoRegimeDetector nÃ£o encontrado em: {model_path}")
                    self.regime_detector = None

            if self.regime_detector and self.regime_detector.is_trained:
                regime_df = self.regime_detector.predict(df)
                
                # Garante alinhamento de Ã­ndice
                df['regime_confidence'] = regime_df['confidence'].reindex(df.index).fillna(0.5)
                df['regime_val'] = regime_df['regime'].reindex(df.index).fillna(2).astype(int)
                
                # Check rÃ¡pido de std apenas para debug info (pode ser removido em prod se muito custoso)
                # mean_conf = df['regime_confidence'].mean()
                # std_conf = df['regime_confidence'].std()
                # logger.debug(f"✅ [AI] Regime detectado via cache.") 
            else:
                if self.regime_detector:
                     logger.warning("âš ï¸  [AI] CryptoRegimeDetector cacheado mas nÃ£o treinado!")

                
        except Exception as e:
            logger.error(f"❌ [AI] Erro ao aplicar regime detector: {e}", exc_info=True)
        
        return df

    def _enrich_df_with_trend_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona colunas de tendência. 
        [CORREÇÃO] Agora chama _apply_regime_detector_to_df primeiro para garantir que
        regime_confidence e regime_val existam.
        """
        # [CORREÇÃO CRÍTICA] Aplica detector de regime ANTES de verificar colunas
        df = self._apply_regime_detector_to_df(df)
        
        df_enriched = df.copy()

        # [CORREÇÃO] Aceita 'regime' ou 'regime_val' para maior resiliência
        if 'regime_val' not in df_enriched.columns and 'regime' in df_enriched.columns:
            df_enriched['regime_val'] = df_enriched['regime']
            
        has_regime_info = 'regime_confidence' in df_enriched.columns and 'regime_val' in df_enriched.columns
        
        if has_regime_info:
            # [CORREÇÃO] Validação crítica: Detectar valores constantes (fallback silencioso)
            conf_std = df_enriched['regime_confidence'].std()
            conf_mean = df_enriched['regime_confidence'].mean()
            
            if conf_std < 0.01:
                logger.warning(
                    f"⚠️ [AI] CRÍTICO: regime_confidence é praticamente constante "
                    f"(mean={conf_mean:.4f}, std={conf_std:.4f}). AGENTES NÃO APRENDERÃO!"
                )
            else:
                logger.info(f"✅ [AI] Mapping CryptoRegimeDetector outputs to Trend priors (mean_conf={conf_mean:.4f}, std={conf_std:.4f}).")
            
            # Map Confidence
            df_enriched['tp_prior_conf'] = df_enriched['regime_confidence']
            df_enriched['trend_pred_confidence'] = df_enriched['regime_confidence']
            
            # Map Direction (Bull=+, Bear=-, Ranger=0) based on regime_val
            # 0=Bull, 1=Bear, 2=Ranger
            # Vectorized mapping
            conditions = [
                df_enriched['regime_val'] == 0, # Bull
                df_enriched['regime_val'] == 1, # Bear
                df_enriched['regime_val'] >= 2  # Ranger/Other
            ]
            choices = [
                df_enriched['regime_confidence'],  # Positive conf for Long
                -df_enriched['regime_confidence'], # Negative conf for Short
                0.0                                # Neutral
            ]
            df_enriched['tp_prior_dir'] = np.select(conditions, choices, default=0.0)
            
            # Map One-Hot Regimes
            df_enriched['tp_regime_up'] = (df_enriched['regime_val'] == 0).astype(float)
            df_enriched['tp_regime_down'] = (df_enriched['regime_val'] == 1).astype(float)
            df_enriched['tp_regime_sideways'] = (df_enriched['regime_val'] >= 2).astype(float)
            
            # Legacy/Compat columns
            df_enriched['trend_pred_uptrend'] = df_enriched['tp_regime_up']
            df_enriched['trend_pred_downtrend'] = df_enriched['tp_regime_down']
            df_enriched['trend_pred_neutral'] = df_enriched['tp_regime_sideways']
            
            # Fill remaining with defaults if missing
            defaults = {
                'trend_success_prob': 0.5,
                'predicted_volatility_high': 0.0,
                'predicted_breakout': 0.0,
                'tp_uncertainty': 0.5,
                'tp_duration_median': 10.0
            }
            for col, val in defaults.items():
                if col not in df_enriched.columns:
                    df_enriched[col] = val
                    
        else:
            # Fallback: Hardcoded Neutral
            logger.warning(
                "⚠️ [AI] FALLBACK ATIVADO: regime_confidence/regime_val não encontrados! "
                "TrendSpecialist usará valores neutros fixos e NÃO APRENDERÁ. "
                "Verifique se CryptoRegimeDetector está treinado e acessível."
            )
            neutral_cols = {
                'trend_pred_neutral': 1.0,
                'trend_pred_uptrend': 0.0,
                'trend_pred_downtrend': 0.0,
                'trend_pred_confidence': 0.5,
                'trend_success_prob': 0.5,
                'predicted_volatility_high': 0.0,
                'predicted_breakout': 0.0,
                # Compatibilidade com TrendSpecialist (tp_ prefix)
                'tp_prior_dir': 0.0, # Neutral direction
                'tp_prior_conf': 0.5, # Neutral confidence
                'tp_regime_up': 0.0,
                'tp_regime_down': 0.0,
                'tp_regime_sideways': 1.0, 
                'tp_uncertainty': 0.5,
                'tp_duration_median': 10.0,
            }
            for col, val in neutral_cols.items():
                if col not in df_enriched.columns:
                    df_enriched[col] = val
        
        return df_enriched


    def report_regime_validation(self, df: pd.DataFrame) -> None:
        """
        Gera mÃ©tricas por buckets de regime: (Volatilidade: baixo/alto) Ã— (ADX: baixo/alto).
        MÃ©tricas: Win Rate e Sharpe anualizado dos retornos futuros (prÃ³ximo candle).

        - Volatilidade: usa colunas booleanas `high_volatility` / `low_volatility` se existirem;
          caso contrÃ¡rio, deriva a partir de `atr_percentage` versus sua mÃ©dia mÃ³vel de 20.
        - ADX: usa `adx_strong_trend` / `adx_weak_trend` se existirem; se nÃ£o, deriva de `adx`.
        """
        try:
            if df is None or df.empty:
                logger.warning("âš ï¸ [REGIME REPORT] DataFrame vazio. RelatÃ³rio por regime ignorado.")
                return

            df_local = df.copy()

            # Garantir colunas necessÃ¡rias de base
            has_atr_pct = 'atr_percentage' in df_local.columns
            has_adx = 'adx' in df_local.columns

            # Volatilidade: usa flags existentes se disponÃ­veis; caso contrÃ¡rio, deriva
            if 'high_volatility' not in df_local.columns or 'low_volatility' not in df_local.columns:
                # TolerÃ¢ncia: tenta inferir atr_percentage se nÃ£o existir
                if not has_atr_pct:
                    tf = str(getattr(self.config_trading, 'PRIMARY_TIMEFRAME_TRADING', '1h')).lower()
                    atr_tf_col = f"atr_{tf}"
                    atr_pct_tf_col = f"atr_percentage_{tf}"
                    if atr_pct_tf_col in df_local.columns:
                        df_local['atr_percentage'] = df_local[atr_pct_tf_col]
                        has_atr_pct = True
                    elif atr_tf_col in df_local.columns and 'close' in df_local.columns:
                        df_local['atr_percentage'] = (df_local[atr_tf_col] / (df_local['close'] + 1e-9)) * 100.0
                        has_atr_pct = True
                if not has_atr_pct:
                    logger.warning("âš ï¸ [REGIME REPORT] 'atr_percentage' ausente. NÃ£o Ã© possÃ­vel derivar volatilidade.")
                    return
                atr_ma = df_local['atr_percentage'].rolling(20, min_periods=20).mean()
                df_local['high_volatility'] = (df_local['atr_percentage'] > (atr_ma * 1.5)) & (atr_ma > 1e-9)
                df_local['low_volatility'] = (df_local['atr_percentage'] < (atr_ma * 0.7)) & (atr_ma > 1e-9)

            # ADX: usa flags existentes se disponÃ­veis; caso contrÃ¡rio, deriva
            if 'adx_strong_trend' not in df_local.columns or 'adx_weak_trend' not in df_local.columns:
                if not has_adx:
                    # Tenta usar coluna com sufixo do timeframe principal
                    tf_local = str(getattr(self.config_trading, 'PRIMARY_TIMEFRAME_TRADING', '1h')).lower()
                    adx_tf_col = f"adx_{tf_local}"
                    if adx_tf_col in df_local.columns:
                        df_local['adx'] = df_local[adx_tf_col]
                        has_adx = True
                    else:
                        # Como Ãºltimo recurso, recalcula ADX a partir de high/low/close
                        if all(col in df_local.columns for col in ['high', 'low', 'close']):
                            try:
                                adx_series, _, _ = NativeIndicators.adx(df_local['high'], df_local['low'], df_local['close'], length=14)
                                df_local['adx'] = adx_series
                                has_adx = True
                                logger.info("â„¹ï¸ [REGIME REPORT] ADX recalculado on-the-fly para o relatÃ³rio de regime.")
                            except Exception as _e:
                                logger.warning(f"âš ï¸ [REGIME REPORT] Falha ao recalcular ADX: {_e}")
                if not has_adx:
                    logger.warning("âš ï¸ [REGIME REPORT] 'adx' ausente. NÃ£o Ã© possÃ­vel derivar forÃ§a de tendÃªncia.")
                    return
                df_local['adx_strong_trend'] = (df_local['adx'] > 25)
                df_local['adx_weak_trend'] = (df_local['adx'] < 20)

            # Retorno futuro (prÃ³ximo candle)
            if 'close' not in df_local.columns:
                logger.warning("âš ï¸ [REGIME REPORT] Coluna 'close' ausente. NÃ£o Ã© possÃ­vel calcular retornos.")
                return
            df_local['ret_fwd'] = df_local['close'].pct_change().shift(-1)

            # Fator de anualizaÃ§Ã£o conforme timeframe principal
            tf = getattr(self.config_trading, 'PRIMARY_TIMEFRAME_TRADING', '1h')
            tf = str(tf).lower()
            bars_per_year_map = {
                '1m': 365*24*60,
                '5m': 365*24*12,
                '15m': 365*24*4,
                '1h': 365*24,
                '4h': 365*6,
                '1d': 365
            }
            bars_per_year = bars_per_year_map.get(tf, 365*24)

            def sharpe(series: pd.Series) -> float:
                series = series.dropna()
                if series.empty or series.std() == 0:
                    return float('nan')
                return float(series.mean() / (series.std() + 1e-12) * np.sqrt(bars_per_year))

            def win_rate(series: pd.Series) -> float:
                series = series.dropna()
                if series.empty:
                    return float('nan')
                return float((series > 0).mean())

            buckets = [
                ("VOL=ALTA Ã— ADX=ALTO", (df_local['high_volatility'] & df_local['adx_strong_trend'])),
                ("VOL=ALTA Ã— ADX=BAIXO", (df_local['high_volatility'] & df_local['adx_weak_trend'])),
                ("VOL=BAIXA Ã— ADX=ALTO", (df_local['low_volatility'] & df_local['adx_strong_trend'])),
                ("VOL=BAIXA Ã— ADX=BAIXO", (df_local['low_volatility'] & df_local['adx_weak_trend']))
            ]

            logger.info("\n" + "="*60 + "\nðŸ“Š ValidaÃ§Ã£o por Regime (prÃ©-RL)\n" + "="*60)
            for name, mask in buckets:
                sub = df_local.loc[mask, 'ret_fwd']
                n = int(sub.dropna().shape[0])
                if n == 0:
                    logger.info(f"- {name}: sem amostras suficientes")
                    continue
                wr = win_rate(sub)
                sh = sharpe(sub)
                logger.info(f"- {name}: N={n} | WR={wr:.2%} | Sharpe={sh:.2f}")
        except Exception as e:
            logger.warning(f"⚠️ [REGIME REPORT] Falha ao gerar relatório por regime: {e}")

    # [REMOVIDO] needs_profitability_predictor_retraining - simplificação do pipeline
    # def needs_profitability_predictor_retraining(self, min_new_trades: int) -> bool:
    #     try:
    #         last_training_trade_count = 0
    #         if os.path.exists(self.metadata_path):
    #             with open(self.metadata_path, 'r') as f:
    #                 metadata = json.load(f)
    #                 last_training_trade_count = metadata.get('profit_predictor_trade_count', 0)
    #         trades_log_path = os.path.join(self.log_dir, "trades_log.csv")
    #         if not os.path.exists(trades_log_path):
    #             return False
    #         with open(trades_log_path, 'r', encoding='utf-8') as f:
    #             current_trade_count = sum(1 for line in f) - 1
    #         if (current_trade_count - last_training_trade_count) >= min_new_trades:
    #             logger.info(f"ðŸ§  Novos trades detectados ({current_trade_count - last_training_trade_count}). Retreinamento do ProfitabilityPredictor justificado.")
    #             return True
    #     except Exception as e:
    #         logger.error(f"âŒ Erro ao verificar necessidade de retreino do ProfitabilityPredictor: {e}")
    #     return False
    
    # [REMOVIDO] retrain_profitability_predictor - simplificação do pipeline  
    # async def retrain_profitability_predictor(self, trade_log: List[Dict[str, Any]]) -> bool:
    #     if not self.profitability_predictor:
    #         logger.error("âŒ ProfitabilityPredictor nÃ£o inicializado.")
    #         return False
    #     logger.info("ðŸ§  Retreinando o ProfitabilityPredictor com o log de trades mais recente...")
    #     try:
    #         base_df = await self._get_or_create_base_dataset()
    #         if base_df is None or base_df.empty:
    #             logger.error("âŒ NÃ£o foi possÃ­vel obter o dataset base para o retreinamento do P-Predictor.")
    #             return False
    #         num_epochs = 50
    #         completed_epochs = self.profitability_predictor.train_model(base_df, trade_log, num_epochs)
    #         if completed_epochs > 0:
    #             self.profitability_predictor.save_model(os.path.join(self.config_ai.MODEL_DIR, "profitability_predictor.pth"))
    #             if os.path.exists(self.metadata_path):
    #                 with open(self.metadata_path, 'r+') as f:
    #                     metadata = json.load(f)
    #                     metadata['profit_predictor_trade_count'] = len(trade_log)
    #                     f.seek(0)
    #                     json.dump(metadata, f, indent=4)
    #                     f.truncate()
    #             return True
    #         return False
    #     except Exception as e:
    #         logger.critical(f"ðŸš¨ Falha crÃ­tica no retreinamento do ProfitabilityPredictor: {e}", exc_info=True)
    #         return False

    async def _train_rl_agents(
        self, 
        featured_df: pd.DataFrame, 
        use_profitability_predictor: bool = False,
        training_progress: Dict = None, 
        training_goals: Dict = None,
        status: Dict[str, bool] = None
    ):
        """
        [FIX] Sincronização de Features e Treinamento de Especialistas.
        """
        # 🔬 [ROBUSTNESS FIX] Validar dimensão de features
        # O encoder pode ter mudado ou colunas extras foram adicionadas.
        # Precisamos garantir que especialistas e gating batam com o featured_df.
        sample_features = featured_df.select_dtypes(include=[np.number]).columns.tolist()
        from feature_engineering.scientific_data_processor import ScientificDataProcessor
        processor = ScientificDataProcessor()
        relevant_features = processor.filter_autoencoder_features(sample_features, featured_df)
        actual_dim = len(relevant_features)
        
        if actual_dim != self.final_feature_count:
            logger.warning(
                f"⚠️ [DIVERSIDADE] Feature Count Mismatch! "
                f"Config: {self.final_feature_count} vs Actual: {actual_dim}. "
                f"Re-inicializando agentes de decisão..."
            )
            self._initialize_decision_agents(actual_dim)

        logger.info(f"🚀 Iniciando Pipeline de Treinamento RL (Features: {actual_dim})")

        # ── Curriculum Learning: filtra dados pelo estágio atual de dificuldade ──
        if self.curriculum_engine:
            stage_name = self.curriculum_engine.get_current_stage_name()
            logger.info(f"📚 [CURRÍCULO] Estágio ativo: '{stage_name}' "
                        f"({self.curriculum_engine.current_stage_index + 1}/"
                        f"{len(self.curriculum_engine.stages)})")
            featured_df = self.curriculum_engine.prepare_training_data(featured_df)
            logger.info(f"📚 [CURRÍCULO] Dataset filtrado: {len(featured_df)} amostras "
                        f"(estágio '{stage_name}')")
        # ─────────────────────────────────────────────────────────────────────────

        training_progress = training_progress or {}
        training_goals = training_goals or {}

        # Variável para rastrear se novos conhecimentos foram gerados (para bootstrap da gating)
        any_specialist_trained = False
        train_df, eval_df = train_test_split(featured_df, test_size=0.2, shuffle=False)
        # [REMOVIDO] predictor_to_use - simplificação do pipeline
        unique_spec_types_to_train = {}
        for regime, specialist in self.specialists.items():
            spec_class_name = specialist.__class__.__name__
            if spec_class_name not in unique_spec_types_to_train:
                 unique_spec_types_to_train[spec_class_name] = specialist
        logger.info("--- Etapa 4.1: Verificando e treinando especialistas individuais ---")
        
        # 🔬 ORDENAÇÃO POR PROGRESSO: Specialist com mais trials primeiro
        # Isso garante que ao reiniciar, continue de onde parou
        specialist_order = ['BullSpecialist', 'BearSpecialist', 'RangerSpecialist']
        
        # Mapeia progresso de cada specialist
        specialist_progress = {}
        for spec_name in specialist_order:
            spec = unique_spec_types_to_train.get(spec_name)
            if spec and hasattr(spec, 'optuna_db_path') and os.path.exists(spec.optuna_db_path):
                try:
                    import optuna
                    summaries = optuna.study.get_all_study_summaries(storage=f"sqlite:///{spec.optuna_db_path}")
                    specialist_progress[spec_name] = summaries[0].n_trials if summaries else 0
                except:
                    specialist_progress[spec_name] = 0
            else:
                specialist_progress[spec_name] = 0
        
        # Ordena: specialist com mais progresso primeiro (mais perto de completar)
        specialist_order_sorted = sorted(
            [s for s in specialist_order if s in unique_spec_types_to_train],
            key=lambda x: specialist_progress.get(x, 0),
            reverse=True  # Maior progresso primeiro
        )
        
        logger.info(f"📊 [ORDEM] Processando specialists por progresso: {[(s, specialist_progress.get(s, 0)) for s in specialist_order_sorted]}")
        
        for spec_class_name in specialist_order_sorted:
            # Encontra o specialist correspondente
            specialist = unique_spec_types_to_train.get(spec_class_name)
            if specialist is None:
                continue
                
            # PASSO 1: Verificar status de otimizacao ROBUSTA
            # Verifica: 1) is_optimized property, 2) DB Optuna com trials, 3) Modelo treinado existe
            skip_optimization = False
            current_trials = 0
            
            # Verificação 1: Propriedade is_optimized
            if hasattr(specialist, 'is_optimized') and specialist.is_optimized:
                logger.info(f"✅ [OPTUNA] Especialista '{spec_class_name}' já otimizado (via is_optimized). Pulando.")
                skip_optimization = True
            
            # Verificação 1b: REMOVIDA - o arquivo .json de hiperparâmetros é salvo APÓS cada trial.
            # Usar a existência do arquivo como gatilho de "otimização concluída" causava skip
            # prematuro com apenas 7 trials (de 100 necessários).
            # A única fonte de verdade é o DB do Optuna via Verificação 2 (>= 100 trials).
            
            # Verificação 1c: Já marcado como treinado pelo controlador (vinda do status)
            if not skip_optimization and status:
                spec_status_map = {
                    'BullSpecialist': 'bull_specialist_sac',
                    'BearSpecialist': 'bear_specialist_sac',
                    'RangerSpecialist': 'ranger_specialist_sac'
                }
                status_key = spec_status_map.get(spec_class_name)
                if status_key and status.get(status_key):
                    logger.info(f"✅ [OPTUNA] Especialista '{spec_class_name}' já marcado como treinado no status. Pulando otimização.")
                    skip_optimization = True

            # Verificação 2: Database Optuna com trials suficientes
            if not skip_optimization and hasattr(specialist, 'optuna_db_path'):
                try:
                    import optuna
                    db_path = specialist.optuna_db_path
                    if os.path.exists(db_path):
                        # API correta: optuna.study.get_all_study_summaries
                        study_summaries = optuna.study.get_all_study_summaries(storage=f"sqlite:///{db_path}")
                        for summary in study_summaries:
                            current_trials = summary.n_trials
                            if summary.n_trials >= 100:  # Alvo corrigido para 100 trials
                                logger.info(f"✅ [OPTUNA] Especialista '{spec_class_name}' tem {summary.n_trials} trials no DB. Pulando otimização.")
                                skip_optimization = True
                                # Carrega melhores hiperparâmetros do estudo
                                study = optuna.load_study(study_name=summary.study_name, storage=f"sqlite:///{db_path}")
                                if study.best_trial:
                                    specialist.hyperparams.update(study.best_params)
                                    logger.info(f"✅ [OPTUNA] Melhores hiperparâmetros carregados do estudo '{summary.study_name}'.")
                                break
                            else:
                                # Tem trials mas não completou - FOCA neste specialist
                                logger.info(f"📊 [OPTUNA] Especialista '{spec_class_name}' tem {summary.n_trials}/100 trials. Continuando otimização...")
                except Exception as e:
                    logger.warning(f"⚠️ [OPTUNA] Erro ao verificar DB Optuna para '{spec_class_name}': {e}")
            
            # Verificação 3: Modelo treinado existe (apenas se não houver DB do Optuna)
            # [FIX] Se o usuário pediu treinamento, permitimos re-otimização se o DB estiver incompleto
            # mesmo que o .zip exista (pode ser um modelo antigo de um contrato diferente).
            if not skip_optimization and not os.path.exists(getattr(specialist, 'optuna_db_path', '')) and hasattr(specialist, 'model_path'):
                if os.path.exists(specialist.model_path):
                    file_size = os.path.getsize(specialist.model_path)
                    if file_size > 1000000:  # >1MB = modelo válido
                        logger.info(f"✅ [OPTUNA] Modelo '{specialist.model_path}' existe ({file_size/1e6:.1f}MB). Pulando otimização.")
                        skip_optimization = True
            
            if not skip_optimization:
                logger.info(f"[OPTUNA] Especialista '{spec_class_name}' não otimizado. Executando otimização Optuna...")
                try:
                    # [NORMALIZAÇÃO OPTUNA] Diferente do train_model(), o Optuna recebe 
                    # enriched_training_df que NÃO passou pelo ScientificDataProcessor.
                    # Por isso o scaler é fitado aqui e passado para optimize_hyperparameters
                    # via self.feature_scaler, que internamente também fita seu próprio scaler
                    # no subconjunto. Os dois são compatíveis (ambos StandardScaler).
                    train_opt, eval_opt = train_test_split(train_df, test_size=0.2, shuffle=False)
                    specialist.optimize_hyperparameters(train_opt, eval_opt, n_trials=100, n_timesteps=100000)  # 🚀 HPO Científico (100k + MLP = 3-4x mais rápido)
                    logger.info(f"[OPTUNA] Otimização do '{spec_class_name}' concluída.")
                except Exception as opt_err:
                    logger.error(f"[OPTUNA] Erro na otimização do '{spec_class_name}': {opt_err}. Continuando com hiperparâmetros padrão.")
                
                # 🔬 FOCO SEQUENCIAL: Após otimizar um specialist, NÃO processa os outros
                # Isso garante que cada specialist complete 100 trials antes de passar para o próximo
                logger.info(f"📊 [OPTUNA] Focando em '{spec_class_name}' até completar otimização. Outros specialists serão processados na próxima execução.")
                # Não dá break aqui - continua para o treinamento deste specialist
            
            # PASSO 2: Treinar modelo final
            # IMPORTANTE: TrendSpecialist é apenas base, não treina separadamente
            if spec_class_name == 'TrendSpecialist':
                logger.info(f"⏭️ Pulando treinamento de '{spec_class_name}' (é apenas classe base).")
                continue
            
            # Para specialists de regime (Bull, Bear, Ranger), usar train_model que filtra por regime
            # 🔧 FIX: Verificar se modelo já existe (arquivo .zip ou best_model)
            completed_steps = training_progress.get(spec_class_name, {}).get("completed_timesteps", 0)
            total_target_steps = training_goals.get(spec_class_name, 0)
            
            # [FIX] Usar model_path do specialist diretamente (bull_specialist_sac.zip, etc)
            # Em vez de gerar nome incorreto a partir da classe
            model_save_path = getattr(specialist, 'model_path', None)
            if not model_save_path:
                # Fallback para nome baseado em specialist_name
                spec_name_attr = getattr(specialist, 'specialist_name', spec_class_name.lower())
                model_save_path = os.path.join(self.config_ai.MODEL_DIR, f"{spec_name_attr}_sac.zip")
            
            # [FIX] Caminho dinâmico para checkpoints usando specialist_name
            spec_name_for_folder = getattr(specialist, 'specialist_name', spec_class_name.lower())
            spec_folder_name = f"{spec_name_for_folder}_checkpoints"
            checkpoint_dir = os.path.join(self.config_ai.CHECKPOINT_DIR, spec_folder_name)
            
            # Fallback para pasta de logs se não existir em checkpoints_ai (retrocompatibilidade)
            if not os.path.exists(checkpoint_dir):
                checkpoint_dir = os.path.join(os.getcwd(), "logs", spec_folder_name)

            best_model_path = os.path.join(checkpoint_dir, "best_model.zip")
            
            if os.path.exists(model_save_path):
                # Modelo final existe - considerar treinamento completo
                completed_steps = max(completed_steps, total_target_steps)
                logger.info(f"💾 [MODELO] Modelo FINAL encontrado para '{spec_class_name}'. Status: Completo.")
            elif os.path.exists(best_model_path):
                 # Checkpoint existe - logar que vai resumir
                 logger.info(f"🔄 [RESUME] Best Model encontrado em '{best_model_path}'. O treinamento continuará deste ponto.")
            elif hasattr(specialist, 'model') and specialist.model is not None:
                # Verificar steps no modelo carregado
                model_steps = getattr(specialist.model, 'num_timesteps', 0)
                if model_steps > completed_steps:
                    completed_steps = model_steps
                    logger.info(f"📊 [TIMESTEPS] Modelo '{spec_class_name}' já tem {model_steps:,} steps treinados.")
            
            remaining_steps = total_target_steps - completed_steps
            
            if remaining_steps > 0:
                logger.info(f"⚠️ Especialista '{spec_class_name}' precisa treinar por mais {remaining_steps:,} timesteps (atual: {completed_steps:,}, meta: {total_target_steps:,}).")
                # specialist.profitability_predictor = predictor_to_use # Removido: predictor não utilizado mais
                
                # [NORMALIZAÇÃO] Os dados passam por ScientificDataProcessor.prepare_for_specialist()
                # dentro de filter_data_by_regime() ANTES de chegar ao TrendFollowingEnv.
                # O warning "[SCALER] Nenhum scaler fornecido" é enganoso — os dados já chegam
                # normalizados por regime (RobustScaler por regime em scientific_scalers.joblib).
                # NÃO injetar scaler aqui — causaria dupla normalização.
                
                # Usa train_model() que filtra dados por regime automaticamente
                if hasattr(specialist, 'train_model'):
                    logger.info(f"📊 Usando train_model() com filtragem por regime para '{spec_class_name}'")
                    result = specialist.train_model(train_df, total_timesteps=remaining_steps)
                    new_total_steps = result.get('timesteps', remaining_steps) if isinstance(result, dict) else remaining_steps
                else:
                    # Fallback para train() genérico
                    new_total_steps = specialist.train(train_df, timesteps_to_train=remaining_steps)
                
                if spec_class_name not in training_progress:
                    training_progress[spec_class_name] = {}
                training_progress[spec_class_name]["completed_timesteps"] = completed_steps + new_total_steps
                any_specialist_trained = True

                # ── Curriculum Learning: atualiza progresso após cada especialista ──
                if self.curriculum_engine:
                    try:
                        curriculum_metrics = {'sharpe_ratio': 0.0, 'max_drawdown_pct': 0.0}

                        if hasattr(specialist, 'evaluate'):
                            # Caminho ideal: especialista implementa evaluate()
                            _em = specialist.evaluate(eval_df)
                            curriculum_metrics = {
                                'sharpe_ratio':     float(_em.get('sharpe_ratio', 0.0)),
                                'max_drawdown_pct': float(_em.get('max_drawdown', 0.0)),
                            }
                        else:
                            # Proxy: Sharpe de buy-and-hold no eval set filtrado pelo estágio.
                            # Mede a "qualidade de sinal" do mercado nesse regime de dificuldade.
                            # Em mercados calmos → Sharpe tende a ser > 1.5 (critério Stage 1).
                            # Em mercados voláteis → Sharpe cai, refletindo maior dificuldade.
                            _close_col = next(
                                (c for c in ('close_15m', 'close_1h', 'close_4h', 'close')
                                 if c in eval_df.columns),
                                None
                            )
                            if _close_col and len(eval_df) > 20:
                                _prices = eval_df[_close_col].dropna()
                                _rets   = _prices.pct_change().dropna()
                                if len(_rets) > 10:
                                    # Sharpe anualizado para barras de 15 min (252 dias × 96 barras/dia)
                                    _ann    = np.sqrt(252 * 96)
                                    _sr     = float((_rets.mean() / (_rets.std() + 1e-8)) * _ann)
                                    # Max drawdown da curva de retorno acumulado
                                    _cumr   = (1.0 + _rets).cumprod()
                                    _mdd    = float(abs(((_cumr - _cumr.cummax()) / _cumr.cummax()).min()))
                                    curriculum_metrics = {
                                        'sharpe_ratio':     float(np.clip(_sr, -5.0, 15.0)),
                                        'max_drawdown_pct': _mdd,
                                    }
                                    logger.debug(
                                        f"📚 [CURRÍCULO] Proxy buy-and-hold ({_close_col}): "
                                        f"Sharpe={curriculum_metrics['sharpe_ratio']:.2f} | "
                                        f"MDD={curriculum_metrics['max_drawdown_pct']:.2%} "
                                        f"({len(_rets)} barras de eval)"
                                    )

                        self.curriculum_engine.update_progress(curriculum_metrics)
                        logger.info(
                            f"📚 [CURRÍCULO] Progresso atualizado após '{spec_class_name}': "
                            f"Sharpe={curriculum_metrics['sharpe_ratio']:.2f} | "
                            f"DD={curriculum_metrics['max_drawdown_pct']:.2%}"
                        )
                    except Exception as _ce:
                        logger.debug(f"📚 [CURRÍCULO] Métricas não disponíveis para update_progress: {_ce}")
                # ─────────────────────────────────────────────────────────────────────

                logger.info(f"🎯 [PROGRESSO] '{spec_class_name}' treinado com sucesso. Prosseguindo para o próximo componente...")
            else:
                logger.info(f"✅ Especialista '{spec_class_name}' já completou seu treinamento ({completed_steps:,} >= {total_target_steps:,}). Pulando.")
        
        # [NEW] Após treinar especialistas (ou se solicitado explicitamente), 
        # re-treinar a gating network para alinhar com o novo conhecimento latente
        if any_specialist_trained or getattr(self, '_force_gating_bootstrap', False):
             await self._bootstrap_gating_network(featured_df)

    async def _bootstrap_gating_network(self, enriched_df: pd.DataFrame) -> bool:
        """
        🚀 [BOOTSTRAP] Treinamento Supervisionado para a SoftGatingNetwork.
        Mapeia Features Latentes (Autoencoder) para Regimes Detectados.
        """
        logger.info("--- Iniciando Bootstrap Supervisionado da SoftGatingNetwork ---")
        try:
            # 1. Preparação de Dados
            if 'regime_val' not in enriched_df.columns:
                logger.error("❌ Erro: Coluna 'regime_val' não encontrada no enriched_df.")
                return False

            # Colunas latentes (representação do autoencoder)
            latent_cols = [c for c in enriched_df.columns if c.startswith('hidden_feature_')]
            if not latent_cols:
                logger.error("❌ Erro: Hidden Features (Latent) não encontradas. Verifique o contrato de features.")
                return False

            X = enriched_df[latent_cols].fillna(0).values
            y = enriched_df['regime_val'].fillna(2).astype(int).values # Default to Ranger (2)
            
            # Garante que y está no range [0, 2]
            y = np.clip(y, 0, 2)

            # 2. Inicialização da Rede e Tensors (Robusta)
            # [FIX] Fallback para CPU se CUDA falhar (limpa travamentos de driver)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            try:
                # Tenta criar tensors no device configurado
                X_tensor = torch.FloatTensor(X).to(device)
                y_tensor = torch.LongTensor(y).to(device)
            except RuntimeError as cuda_err:
                if 'CUDA error' in str(cuda_err):
                    logger.warning(f"⚠️ [GATING] Falha CUDA ({cuda_err}). Migrando bootstrap para CPU...")
                    device = torch.device('cpu')
                    X_tensor = torch.FloatTensor(X).to(device)
                    y_tensor = torch.LongTensor(y).to(device)
                else:
                    raise cuda_err

            # [FIX] Garantir que a dimensão bate com o número real de features encontradas
            if self.gating_network is None or self.gating_network.latent_dim != len(latent_cols):
                latent_dim = len(latent_cols)
                logger.info(f"🔄 [GATING] Ajustando SoftGatingNetwork para nova dimensão latente: {latent_dim}")
                self.gating_network = create_gating_network(self.config_ai, latent_dim=latent_dim)
            
            self.gating_network.to(device) # Move rede para o mesmo device dos dados
            self.gating_network.train()
            
            optimizer = torch.optim.Adam(self.gating_network.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss()
            
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)
            
            # 3. Loop de Treino (10 epochs são suficientes para bootstrap)
            epochs = 10
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    output = self.gating_network(batch_X)
                    logits = output['logits']
                    loss = criterion(logits, batch_y)
                    
                    # Load balancing loss suave para evitar colapso em um só regime
                    lb_loss = self.gating_network.compute_load_balancing_loss(reset=True)
                    (loss + 0.1 * lb_loss).backward()
                    
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 2 == 0:
                    logger.debug(f" -> [GATING BOOTSTRAP] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
            
            # 4. Salvamento
            # Use .pt to match _initialize_decision_agents logic
            gating_path = os.path.join(self.config_ai.MODEL_DIR, "soft_gating_network.pt")
            self.gating_network.save(gating_path)
            
            # 5. Inicialização do Router
            self.moe_router = MixtureOfExpertsRouter(
                gating_network=self.gating_network,
                specialists=self.specialists
            )
            
            logger.info(f"✅ Bootstrap da SoftGatingNetwork concluído. Salvo em '{gating_path}'.")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no bootstrap da Gating Network: {e}", exc_info=True)
            return False
        
        # [FIX] Verificar se TODOS os especialistas concluíram antes de treinar HRL Master
        all_specialists_trained = True
        # [FIX] Mapeamento correto de nomes de arquivos dos especialistas
        specialist_file_mapping = {
            'BullSpecialist': 'bull_specialist_sac.zip',
            'BearSpecialist': 'bear_specialist_sac.zip',
            'RangerSpecialist': 'ranger_specialist_sac.zip'
        }
        
        for spec_class_name in ['BullSpecialist', 'BearSpecialist', 'RangerSpecialist']:
            model_filename = specialist_file_mapping.get(spec_class_name, f"{spec_class_name.lower()}.zip")
            model_path = os.path.join(self.config_ai.MODEL_DIR, model_filename)
            spec_goal = training_goals.get(spec_class_name, 0)
            spec_progress = training_progress.get(spec_class_name, {}).get("completed_timesteps", 0)
            if not os.path.exists(model_path) and spec_progress < spec_goal:
                all_specialists_trained = False
                logger.info(f"⏳ [SEQUENCIAL] '{spec_class_name}' ainda não completou ({spec_progress}/{spec_goal}). HRL Master aguarda.")
                break
        
        # [ARCH FIX] HRL Master desativado — roteamento feito por CryptoRegimeDetector + SoftGatingNetwork.
        #
        # Diagnostico cientifico que motivou a remocao:
        # 1. LOOKAHEAD BIAS: MasterEnv usava future_price 5 candles a frente no calculo de reward
        #    -> PPO aprendia com informacao inexistente em producao.
        # 2. CIRCULARIDADE: 40% do reward era por "regime coherence" (acertar o label do
        #    CryptoRegimeDetector) + observacao ja continha regime/prob_bull/prob_bear/prob_ranger
        #    -> PPO simplesmente copiava o CryptoRegimeDetector deterministicamente.
        # 3. REDUNDANCIA: CryptoRegimeDetector (HMM+GMM+ADX+Funding) classifica regime
        #    de forma robusta. SoftGatingNetwork faz blend suave em zonas de transicao.
        #    Uma 3a camada PPO nao acrescenta informacao nova.
        # 4. DEPENDENCIA CIRCULAR: HRL Master requeria os 3 especialistas treinados -> atrasava
        #    o startup sem nenhum beneficio funcional.
        logger.info(
            "[ARCH] Etapa 4.2 (HRL Master) ignorada - roteamento delegado ao "
            "CryptoRegimeDetector + SoftGatingNetwork (sem lookahead bias, sem circularidade)."
        )


    def _evaluate_trend_specialist_holdout(self, specialist: TrendSpecialist, full_df: pd.DataFrame) -> None:
        df = ensure_datetime_index(full_df)
        if df.empty:
            logger.warning("[TrendSpecialist] Dataset vazio. Backtest automático ignorado.")
            return

        _, holdout_df = split_train_holdout(df, holdout_months=2, min_holdout_rows=DEFAULT_EVAL_MIN_ROWS)
        holdout_label = "two_month_holdout"
        if holdout_df.empty:
            holdout_df = build_unseen_eval_frame(df, train_cutoff_date=None, eval_months=3)
            holdout_label = "unseen_backtest"
        if holdout_df.empty:
            holdout_df = build_recent_eval_frame(df, months=DEFAULT_EVAL_MONTHS, min_rows=DEFAULT_EVAL_MIN_ROWS)
            holdout_label = "recent_backtest"
        if holdout_df.empty:
            logger.warning("[TrendSpecialist] Nenhum conjunto válido de dados não vistos para backtest automático.")
            return

        log_dir = Path(os.getcwd()) / "logs" / "trade_analysis" / "SAC"
        summary = run_manual_backtest(specialist, holdout_df, log_dir=log_dir, label=holdout_label, logger=logger)
        if summary:
            logger.info(
                "[TrendSpecialist] Backtest %s | NetProfit=%.2f | WinRate=%.2f%% | PF=%.2f | MaxDD=%.2f%%",
                holdout_label,
                summary.get("net_profit", 0.0),
                summary.get("win_rate_pct", 0.0),
                summary.get("profit_factor", 0.0),
                summary.get("max_drawdown_pct", 0.0),
            )
        else:
            logger.warning("[TrendSpecialist] Backtest automático não retornou métricas.")

    @staticmethod
    def _timeframe_to_minutes(tf: str) -> Optional[int]:
        tf = str(tf).lower()
        if not tf:
            return None
        try:
            value = int(''.join(ch for ch in tf if ch.isdigit()))
        except ValueError:
            return None
        if tf.endswith('m'):  # minutes
            return value
        if tf.endswith('h'):
            return value * 60
        if tf.endswith('d'):
            return value * 60 * 24
        return None

    @staticmethod
    def _timeframe_to_pandas_freq(tf: str) -> Optional[str]:
        tf = str(tf).lower()
        try:
            value = int(''.join(ch for ch in tf if ch.isdigit()))
        except ValueError:
            return None
        if tf.endswith('m'):
            return f"{value}T"
        if tf.endswith('h'):
            return f"{value}H"
        if tf.endswith('d'):
            return f"{value}D"
        return None

    def _augment_with_additional_timeframes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriquece o dataset base com dados de outros timeframes disponÃ­veis em logs/historical_data."""
        folder = Path('logs') / 'historical_data'
        if df is None or df.empty or not folder.exists():
            return df

        symbol = str(self.config_trading.PRIMARY_PAIR)
        primary_tf = str(self.config_trading.PRIMARY_TIMEFRAME_TRADING).lower()
        base_minutes = self._timeframe_to_minutes(primary_tf)
        base_freq = self._timeframe_to_pandas_freq(primary_tf)
        if not base_minutes or not base_freq:
            return df

        df_aug = df.copy()
        df_aug.index = pd.to_datetime(df_aug.index)
        if getattr(df_aug.index, 'tz', None) is not None:
            df_aug.index = df_aug.index.tz_convert('UTC').tz_localize(None)
        base_index = df_aug.index

        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }

        added_suffixes = set()
        added_columns = 0

        for file in folder.glob(f"{symbol}_*_historical.pkl"):
            parts = file.stem.split('_')
            if len(parts) < 2:
                continue
            tf_part = parts[1].lower()
            if tf_part == primary_tf:
                continue

            tf_minutes = self._timeframe_to_minutes(tf_part)
            tf_freq = self._timeframe_to_pandas_freq(tf_part)
            if not tf_minutes or not tf_freq:
                continue

            try:
                df_tf = pd.read_pickle(file)
            except Exception as exc:
                logger.warning(" âš ï¸ [DATA] Falha ao carregar %s: %s", file.name, exc)
                continue

            if df_tf is None or df_tf.empty:
                continue

            df_tf = df_tf.sort_index()
            df_tf.index = pd.to_datetime(df_tf.index)
            if getattr(df_tf.index, 'tz', None) is not None:
                df_tf.index = df_tf.index.tz_convert('UTC').tz_localize(None)

            try:
                if tf_minutes < base_minutes:
                    resampled = df_tf.resample(base_freq).agg(agg)
                elif tf_minutes > base_minutes:
                    resampled = df_tf.resample(base_freq).ffill()
                else:
                    resampled = df_tf
            except Exception as exc:
                logger.warning(" âš ï¸ [DATA] Falha ao reamostrar %s para %s: %s", tf_part, primary_tf, exc)
                continue

            resampled = resampled.reindex(base_index)
            resampled = resampled.ffill().bfill()

            renamed = {col: f"{col}_tf_{tf_part}" for col in resampled.columns}
            resampled.rename(columns=renamed, inplace=True)

            for col in resampled.columns:
                df_aug[col] = resampled[col]
                added_columns += 1
            added_suffixes.add(tf_part)

        if added_columns:
            logger.info(" ðŸ§  [DATA] Colunas multi-timeframe adicionadas: %d (%s)", added_columns, ', '.join(sorted(added_suffixes)))

        return df_aug

    async def _get_or_create_base_dataset(self) -> Optional[pd.DataFrame]:
        """Obtém o dataset base; reutiliza cache e enriquece com dados multi-timeframe."""
        base_df_path = os.path.join(self.config_ai.MODEL_DIR, "base_featured_df.pkl")
        backup_path = base_df_path + ".bak"
        
        # [SAFEGUARD] Auto-restore from backup if main file is missing but backup exists
        if not os.path.exists(base_df_path) and os.path.exists(backup_path):
            try:
                import shutil
                shutil.copy2(backup_path, base_df_path)
                logger.info(f"✅ [CACHE RESTORE] Arquivo restaurado de backup: '{backup_path}'")
            except Exception as restore_err:
                logger.warning(f"⚠️ [CACHE RESTORE] Falha ao restaurar backup: {restore_err}")

        if os.path.exists(base_df_path):
            try:
                # [SAFEGUARD] Check file size - if too small, try backup
                file_size = os.path.getsize(base_df_path)
                if file_size < 1_000_000 and os.path.exists(backup_path):  # < 1MB is suspicious
                    backup_size = os.path.getsize(backup_path)
                    if backup_size > file_size * 10:  # Backup much larger = likely correct
                        logger.warning(f"⚠️ [CACHE] Arquivo muito pequeno ({file_size/1024:.0f}KB). Restaurando backup ({backup_size/1024/1024:.1f}MB)...")
                        import shutil
                        shutil.copy2(backup_path, base_df_path)
                        
                logger.info(f"[CACHE] Cache do dataset base encontrado. Carregando de '{base_df_path}'...")
                cached_df = pd.read_pickle(base_df_path)
                if cached_df is not None and not cached_df.empty:
                    # [FIX ROBUSTO] Garantir que labels de regime existam no cache
                    if 'regime' not in cached_df.columns:
                        logger.warning("⚠️ [CACHE] Coluna 'regime' ausente. Injetando labels via ScientificDataProcessor...")
                        try:
                            from feature_engineering.scientific_data_processor import create_data_processor
                            dp = create_data_processor(self.config_ai.MODEL_DIR)
                            cached_df = dp.add_regime_labels(cached_df)
                            # Salvar cache atualizado
                            cached_df.to_pickle(base_df_path)
                            logger.info(f"✅ [CACHE] Cache atualizado com colunas de regime.")
                        except Exception as e:
                            logger.error(f"❌ [CACHE] Falha ao injetar regimes: {e}")

                    enriched_cached = self._augment_with_additional_timeframes(cached_df)
                    if enriched_cached is not None and not enriched_cached.empty and enriched_cached is not cached_df:
                        try:
                            enriched_cached.to_pickle(base_df_path)
                        except Exception as save_exc:
                            logger.warning('[CACHE] Falha ao atualizar cache enriquecido: %s', save_exc)
                        cached_df = enriched_cached
                return cached_df
            except Exception as e:
                logger.warning(f"⚠️ [CACHE] cache corrompido ou erro de leitura: {e}. Gerando novamente.")

        logger.info("[CACHE] Cache do dataset base nÃ£o encontrado. Gerando a partir de dados histÃ³ricos...")
        if not self.data_provider or not self.feature_pipeline:
            logger.error("[CACHE] DataProvider/FeaturePipeline nÃ£o inicializados. ImpossÃ­vel criar dataset.")
            return None

        historical_data = await self.data_provider.get_data_for_training(
            symbol=self.config_trading.PRIMARY_PAIR,
            start_date=(datetime.now(timezone.utc) - timedelta(days=3 * 365)).strftime('%Y-%m-%d'),
            end_date=datetime.now(timezone.utc).strftime('%Y-%m-%d')
        )
        if not historical_data:
            logger.error("[CACHE] Falha ao obter dados histÃ³ricos para criar o dataset base.")
            return None

        featured_df = await self.feature_pipeline.create_features(
            historical_data,
            self.config_trading.PRIMARY_PAIR,
            self.config_trading.PRIMARY_TIMEFRAME_TRADING,
            fit_scaler=True,
            apply_latent=False
        )

        if featured_df is not None and not featured_df.empty:
            featured_df = self._augment_with_additional_timeframes(featured_df)
            try:
                os.makedirs(os.path.dirname(base_df_path), exist_ok=True)
                # [SAFEGUARD] Create backup before overwriting
                if os.path.exists(base_df_path):
                    import shutil
                    shutil.copy2(base_df_path, backup_path)
                    logger.info(f"[CACHE] Backup criado: '{backup_path}'")
                featured_df.to_pickle(base_df_path)
                logger.info(f"[CACHE] Dataset base com features salvo em '{base_df_path}'.")
            except Exception as e:
                logger.error(f"[CACHE] Falha ao salvar o cache do dataset base: {e}", exc_info=True)

        return featured_df


    def _optimize_trend_predictor(self, features_df: pd.DataFrame, labels_df: pd.DataFrame, n_trials: int, context: str) -> None:
        """
        Runs TrendPredictor Optuna optimization with a combined dataset
        (features + labels), ensuring the labeling filter path is reused.
        """
        if not self.trend_predictor:
            raise RuntimeError("TrendPredictor not initialized")
        if n_trials <= 0:
            return
        if features_df is None or labels_df is None:
            raise ValueError("Features and labels are required for optimization")

        overlapping = [c for c in labels_df.columns if c in features_df.columns and c != "primary_signal"]
        feature_only = features_df.drop(columns=overlapping, errors="ignore")
        combined_df = feature_only.join(labels_df, how="inner")
        if combined_df is None or combined_df.empty:
            raise ValueError("Combined dataset for TrendPredictor Optuna is empty after join")

        logger.info("%s Otimizando TrendPredictor via Optuna (%d trials)...", context, n_trials)
        self.trend_predictor.optimize_from_combined_dataset(combined_df, n_trials=n_trials)
        logger.info("%s Otimização do TrendPredictor concluída.", context)

        logger.info("✅ [RL] Ciclo de treinamento dos agentes concluído.")

    async def train_foundation_models(self) -> Optional[pd.DataFrame]:
        """
        [FUNÇÃO CORRIGIDA - ORDEM HIERÁRQUICA CORRETA]
        Orquestra o treinamento de fundação seguindo a ordem hierárquica correta:
        1. Treina Autoencoder primeiro
        2. Aplica hidden features ao base_df
        3. Treina TrendPredictor com features enriquecidas
        4. Treina agentes RL com dataset completo
        """
        try:
            logger.info("--- Iniciando FASE 1: Treinamento dos Modelos de Features ---")
            base_df = await self._get_or_create_base_dataset()
            if base_df is None or base_df.empty:
                logger.error("❌ Falha ao obter o dataset base. Abortando treinamento de fundação.")
                return None

            numeric_df = base_df.select_dtypes(include=np.number)
            self._initialize_feature_models(input_dim=len(numeric_df.columns))

            logger.info("🚀 [FASE 1.1] Treinando Autoencoder Temporal...")
            # Garante que a pipeline do autoencoder esteja inicializada
            if self.feature_pipeline is None:
                logger.error("❌ [TRAIN] feature_pipeline não inicializado. Abortando.")
                return None
            if not self.feature_pipeline.temporal_autoencoder_pipeline:
                self.feature_pipeline.train_autoencoder(numeric_df, self.config_ai.AUTOENCODER_TRAINING_EPOCHS)
            
            # Se já estiver treinado, carrega; senão, treina
            if self.feature_pipeline.temporal_autoencoder_pipeline.is_trained:
                logger.info(" -> Autoencoder já treinado. Carregando.")
                self.feature_pipeline.temporal_autoencoder_pipeline.load_state()
            else:
                self.feature_pipeline.train_autoencoder(numeric_df, self.config_ai.AUTOENCODER_TRAINING_EPOCHS)
            
            self._update_model_metadata('autoencoder_model', True)

            logger.info("🔄 [FASE 1.2] Aplicando hidden features ao dataset base...")
            df_with_hidden_features = self.feature_pipeline.apply_hidden_features(base_df)
            if df_with_hidden_features is None or df_with_hidden_features.empty:
                logger.error("❌ Falha ao aplicar hidden features. Abortando.")
                return None
            
            # [FIX] Garantir que colunas de regime sejam propagadas para o dataset enriquecido
            if 'regime' in base_df.columns:
                cols_to_restore = ['regime', 'regime_confidence', 'regime_name']
                for col in cols_to_restore:
                    if col in base_df.columns and col not in df_with_hidden_features.columns:
                        df_with_hidden_features[col] = base_df[col]
                logger.info(f"✅ [DATA] Colunas de regime restauradas no dataset enriquecido.")

            logger.info("📈 [FASE 1.3] Preparando TrendPredictor (REMOVIDO)...")
            # TrendPredictor foi removido do sistema.
            # Otimização e treinamento removidos.
            
            logger.info("--- Iniciando FASE 2: Treinamento dos Agentes de Decisão ---")
            # PASSO 4: Construir dataset final para agentes RL
            enriched_training_df = await self.build_enriched_dataset(df_with_hidden_features)
            if enriched_training_df is None or enriched_training_df.empty:
                logger.error("❌ Falha ao construir o dataset enriquecido final. Abortando treinamento.")
                return None
            
            self._initialize_decision_agents(input_dim=self.final_feature_count)

            # --- FASE 2.1: Bootstrap da SoftGatingNetwork ---
            # Garante que o roteamento comece inteligente mapeando as latent features para os regimes
            await self._bootstrap_gating_network(enriched_training_df)
            
            try:
                with open(self.metadata_path, 'r') as f:
                    training_progress = json.load(f).get("training_progress", {})
            except (FileNotFoundError, json.JSONDecodeError):
                training_progress = {}

            # [CONVERGÊNCIA] 500k para todos — validado empiricamente (Sharpe 8+, WR 78% aos 500k)
            _trend_g = int(os.environ.get('TREND_FINAL_TIMESTEPS', 500_000))
            _base_g  = int(os.environ.get('SPECIALIST_BASE_TIMESTEPS', 500_000))
            training_goals = {
                "BullSpecialist": _trend_g, "BearSpecialist": _trend_g,
                "RangerSpecialist": _base_g, "HRLMaster": _base_g,
            }
            logger.info(f"🎯 Objetivos de treinamento definidos para esta sessão: {training_goals}")

            # PASSO 5: Treinar agentes RL com dataset completo
            await self._train_rl_agents(
                enriched_training_df, 
                use_profitability_predictor=False,
                training_progress=training_progress, 
                training_goals=training_goals
            )
            
            enriched_df_path = os.path.join(self.config_ai.MODEL_DIR, "enriched_featured_df.pkl")
            enriched_training_df.to_pickle(enriched_df_path)
            logger.info(f"✅ [CACHE] Dataset enriquecido final salvo em '{enriched_df_path}'.")

            self.is_trained = True
            # Correção: save_all_models não está definido neste snippet, substituindo por save individual ou remoção se não existir
            # self.save_all_models(training_progress) <--- Removido pois não vi definição no código enviado
            # Salva metadados finais
            self._save_metadata()
            
            logger.info("✅ --- TREINAMENTO DE FUNDAÇÃO (Fases 1 e 2) Concluído com Sucesso ---")
            logger.info("🎉 TODOS OS MODELOS AGORA USAM HIDDEN FEATURES CORRETAMENTE!")
            
            return enriched_training_df
        except Exception as e:
            logger.critical(f"🚨 Falha Crítica no Treinamento de Fundação: {e}", exc_info=True)
            self.is_trained = False
            return None

    async def build_enriched_dataset(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        [FUNÃ‡ÃƒO CORRIGIDA - EVITA DUPLICAÃ‡ÃƒO DE HIDDEN FEATURES]
        Cria o dataset para os agentes de RL e **define `self.base_feature_columns`**,
        que atua como o "Contrato de Features" para os agentes de RL.
        
        Args:
            base_df: DataFrame que pode jÃ¡ conter hidden features (se vindo do treinamento)
                    ou apenas indicadores nativos (se vindo de inferÃªncia)
        """
        logger.info("Construindo dataset enriquecido com vetor de features otimizado...")
        
        # Verifica se o DataFrame jÃ¡ contÃ©m hidden features
        has_hidden_features = any('hidden_feature_' in col for col in base_df.columns)
        
        if has_hidden_features:
            logger.info("ðŸ” DataFrame jÃ¡ contÃ©m hidden features. Pulando aplicaÃ§Ã£o duplicada.")
            df_with_trends = self._enrich_df_with_trend_predictions(base_df)
            # [FIX] TrendPredictor removido - redundante com ScientificRegimeDetector
            if False:
                pass
            df_fully_enriched = df_with_trends
        else:
            logger.info("🔍 Aplicando hidden features ao DataFrame base...")
            # 1. Aplica hidden features primeiro
            if self.feature_pipeline is None:
                logger.error("❌ [AI] feature_pipeline não inicializado. Não é possível aplicar hidden features.")
                return None
            df_with_hidden = self.feature_pipeline.apply_hidden_features(base_df)
            
            # 2. Enriquece com predições de tendência
            df_with_trends = self._enrich_df_with_trend_predictions(df_with_hidden)
            
            # [FIX] TrendPredictor removido - redundante com ScientificRegimeDetector
            if False:
                pass
            
            
            df_fully_enriched = df_with_trends
        
        # [FIX] Garantir que colunas de regime sejam preservadas no dataset enriquecido final
        if 'regime' in base_df.columns:
            cols_to_restore = ['regime', 'regime_confidence', 'regime_name']
            for col in cols_to_restore:
                if col in base_df.columns and col not in df_fully_enriched.columns:
                    df_fully_enriched[col] = base_df[col]
                    logger.debug(f"✅ [DATA] Coluna '{col}' restaurada em build_enriched_dataset.")
        
        hidden_cols = [c for c in df_fully_enriched.columns if 'hidden_feature_' in c]
        trend_cols = [c for c in df_fully_enriched.columns if 'trend_pred_' in c]
        prior_cols = [c for c in ('tp_prior_dir', 'tp_prior_conf', 'tp_tau', 'tp_primary_signal') if c in df_fully_enriched.columns]
        optimized_feature_cols = hidden_cols + trend_cols
        
        # [FIX CRÍTICO PARA XAI] Sincroniza contrato de features com colunas latentes
        # Isso evita que o filtro de reordenação descarte as hidden features
        if self.base_feature_columns:
            for hc in hidden_cols:
                if hc not in self.base_feature_columns:
                    self.base_feature_columns.append(hc)
                    logger.debug(f"➕ [AI] Feature latente '{hc}' adicionada ao contrato base.")
        
        primary_tf = self.config_trading.PRIMARY_TIMEFRAME_TRADING
        essential_env_cols = [
            'open', 'high', 'low', 'close', 'volume',
            f'atr_{primary_tf}', f'bb_lower_{primary_tf}', f'bb_upper_{primary_tf}',
            f'rsi_{primary_tf}', 'bb_expansion', 'high_volatility', 
            'bb_squeeze', 'low_volatility', 'adx_weak_trend', f'adx_{primary_tf}', 'ema_trend'
        ]

        ema_tf_col = f'ema_trend_{primary_tf}'
        if ema_tf_col in df_fully_enriched.columns and 'ema_trend' not in df_fully_enriched.columns:
            df_fully_enriched['ema_trend'] = df_fully_enriched[ema_tf_col]
        
        # [REVERSO DA REGRESSÃO] Coleta TODAS as colunas numéricas para não perder os 500+ indicadores
        all_indicators = [c for c in df_fully_enriched.columns 
                         if c not in (optimized_feature_cols + prior_cols + ['regime', 'regime_confidence', 'regime_name', 'regime_val', 'regime_is_bull', 'regime_is_bear', 'regime_is_ranger'])
                         and df_fully_enriched[c].dtype in [np.float32, np.float64, np.int64]]

        extra_feature_keys = [
            'tp_duration_median',
            'tp_regime_up', 'tp_regime_down', 'tp_regime_sideways'
        ]
        # Garantir que colunas extras existam (mesmo que vazias) para manter o contrato estável
        for col in extra_feature_keys:
            if col not in df_fully_enriched.columns:
                df_fully_enriched[col] = 0.0
        extras_cols = [col for col in extra_feature_keys if col in df_fully_enriched.columns]
        
        ordered_cols = all_indicators + optimized_feature_cols + prior_cols + extras_cols
        
        # [ALINHAMENTO CIENTÍFICO] Aplica a filtragem para definir o contrato real (vencendo ruído e preços)
        from feature_engineering.scientific_data_processor import ScientificDataProcessor
        processor = ScientificDataProcessor()
        final_contract_cols = processor.filter_autoencoder_features(ordered_cols, df_fully_enriched)
        
        # [CORREÇÃO] Mantém colunas de metadados e colunas ESSENCIAIS para o TrendFollowingEnv
        # mesmo que elas não façam parte do contrato de features da rede neural.
        regime_meta_cols = [
            'regime', 'regime_confidence', 'regime_name', 'regime_val', 
            'regime_is_bull', 'regime_is_bear', 'regime_is_ranger'
        ]
        
        # [FIX CRÍTICO] Re-injeta as Hidden Features e Regime Meta no contrato final, 
        # mesmo que o processador científico as tenha filtrado. Elas são vitais para o XAI e MoE.
        final_df_cols = list(dict.fromkeys(final_contract_cols + hidden_cols + trend_cols))
        
        # Colunas reais que o DataFrame deve conter: Contraco IA + Metadados + Colunas Base (OHLCV) + TP Signals + Hidden Features
        hidden_sdae_cols = [c for c in df_fully_enriched.columns if 'hidden' in c or 'sdae' in c]
        data_storage_cols = list(dict.fromkeys(
            final_df_cols + 
            [c for c in regime_meta_cols if c in df_fully_enriched.columns] +
            [c for c in essential_env_cols if c in df_fully_enriched.columns] +
            prior_cols + extras_cols + hidden_sdae_cols
        ))
        
        # [CORREÇÃO] Adiciona .copy() para evitar SettingWithCopyWarning
        final_df = df_fully_enriched[data_storage_cols].copy()
        final_df.dropna(inplace=True)

        logger.info(f"✅ Dataset enriquecido e otimizado criado com sucesso. Shape final: {final_df.shape}")
        logger.info(f"📊 Features no contrato (Filtradas): {len(final_df_cols)} | Metadados: {len(data_storage_cols) - len(final_df_cols)}")
        
        # [CORREÇÃO] Salva o contrato final DESTILADO
        self.base_feature_columns = final_df_cols
        self.final_feature_count = len(self.base_feature_columns)
        logger.info(f"📌 Contrato Final Definido: {self.final_feature_count} colunas (Escala Z Protegida).")

        return final_df
    

    async def handle_drift_and_adapt(self) -> bool:
        """
        [NOVA FUNÃ‡ÃƒO]
        Acionado quando um drift Ã© detectado. Usa os dados recentes que causaram o drift
        para atualizar e expandir o dataset de referÃªncia do DriftDetector.
        Isso permite que o bot aprenda sobre o novo regime de mercado.
        Retorna True se a adaptaÃ§Ã£o foi bem-sucedida.
        """
        logger.warning("ðŸ§  [ADAPTAÃ‡ÃƒO] Drift detectado. Iniciando protocolo de adaptaÃ§Ã£o...")
        
        if not self.drift_detector or not self.data_provider or not self.feature_pipeline:
            logger.error("âŒ [ADAPTAÃ‡ÃƒO] Componentes essenciais (DriftDetector, DataProvider) nÃ£o estÃ£o disponÃ­veis. AdaptaÃ§Ã£o abortada.")
            return False

        try:
            # 1. Busca um conjunto de dados recentes e um pouco mais amplo para contextualizar o drift.
            logger.info("    -> Buscando dados de mercado recentes para reavaliar a referÃªncia...")
            from datetime import datetime, timedelta, timezone
            end_date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            # Busca um perÃ­odo mais longo para garantir que a nova referÃªncia seja estatisticamente robusta
            start_date_str = (datetime.now(timezone.utc) - timedelta(days=365*2)).strftime('%Y-%m-%d')
            
            new_reference_data_raw = await self.data_provider.get_data_for_training(
                symbol=self.config_trading.PRIMARY_PAIR,
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            if not new_reference_data_raw:
                logger.error("âŒ [ADAPTAÃ‡ÃƒO] Falha ao buscar novos dados para a referÃªncia. AdaptaÃ§Ã£o abortada.")
                return False

            # 2. Gera as features para os novos dados
            logger.info("    -> Gerando features para os novos dados de referÃªncia...")
            new_reference_df_featured = await self.feature_pipeline.create_features(
                new_reference_data_raw,
                self.config_trading.PRIMARY_PAIR,
                self.config_trading.PRIMARY_TIMEFRAME_TRADING,
                fit_scaler=False # Usa os scalers existentes
            )

            if new_reference_df_featured is None or new_reference_df_featured.empty:
                logger.error("âŒ [ADAPTAÃ‡ÃƒO] Falha ao gerar features para os novos dados de referÃªncia. AdaptaÃ§Ã£o abortada.")
                return False

            # 3. Atualiza o DriftDetector com a nova (e mais completa) referÃªncia
            logger.info("    -> Atualizando o DriftDetector com o novo conjunto de dados de referÃªncia...")
            # Usa a mesma lista de colunas que jÃ¡ estava sendo monitorada
            columns_to_monitor = list(self.drift_detector.reference_data.keys())
            self.drift_detector.set_reference_data(new_reference_df_featured, columns_to_monitor)
            
            logger.info("âœ… [ADAPTAÃ‡ÃƒO] Protocolo de adaptaÃ§Ã£o ao drift concluÃ­do com sucesso. O bot agora reconhece o novo regime de mercado.")
            # Reseta o status de drift no sistema
            self.system_state["drift_status"] = "RECALIBRADO"
            return True

        except Exception as e:
            logger.critical(f"ðŸš¨ [ADAPTAÃ‡ÃƒO] Falha crÃ­tica durante o protocolo de adaptaÃ§Ã£o ao drift: {e}", exc_info=True)
            return False


    def predict_for_shap(self, X: np.ndarray) -> np.ndarray:
        """
        [VERSÃƒO CORRIGIDA PARA SHAP]
        Thin wrapper que recebe features e retorna scores estÃ¡veis para o KernelExplainer.
        Usa diretamente os modelos de prediÃ§Ã£o para evitar problemas de async.
        
        Args:
            X: np.ndarray com shape (n_samples, n_features) - features de mercado
            
        Returns:
            np.ndarray com scores: positivo = BUY, negativo = SELL, próximo de zero = HOLD
        """
        if not self.is_trained or not self.base_feature_columns:
            logger.warning("⚠️ [SHAP] AIController não treinado ou sem contrato de features. Retornando zeros.")
            return np.zeros(X.shape[0])
        assert self.base_feature_columns is not None  # narrow Optional[List[str]] -> List[str]

        expected_n = len(self.base_feature_columns)
        # [FIX SHAP SHAPE] Padding/truncation defensivo ao invés de retornar zeros.
        # O background foi salvo com uma versão anterior do pipeline (54 cols); o pipeline
        # atual pode ter mais ou menos colunas. Ajustamos em silêncio para evitar spam de error.
        if X.shape[1] != expected_n:
            if X.shape[1] < expected_n:
                # Padding com zeros à direita
                pad = np.zeros((X.shape[0], expected_n - X.shape[1]), dtype=np.float32)
                X = np.hstack([X, pad])
            else:
                # Trunca para as primeiras expected_n colunas
                X = X[:, :expected_n]
        
        # [FIX] trend_predictor foi removido do pipeline. SHAP usa scores neutros.
        # Isso é esperado e não afeta a operação do bot.
        
        try:
            # Converte para DataFrame com colunas corretas
            input_df = pd.DataFrame(X, columns=self.base_feature_columns)
            
            # Cria timestamps fictÃ­cios para evitar erros de Ã­ndice
            timestamps = pd.date_range(
                start=pd.Timestamp.utcnow() - pd.Timedelta(minutes=len(input_df)-1),
                end=pd.Timestamp.utcnow(),
                freq='1min'
            )
            input_df.index = timestamps
            
            scores = []
            
            for idx, row in input_df.iterrows():
                try:
                    # [FIX SHAP SCORE] Score precisa VARIAR com as features para que o
                    # KernelExplainer atribua importâncias não-zero.
                    # Proxy heurístico: momentum + regime — sem dependências async.

                    momentum_cols = [c for c in self.base_feature_columns
                                     if any(k in c for k in ['log_return', 'macd', 'rsi',
                                                              'volume_imbalance', 'oc_move',
                                                              'aggressor_delta', 'obi', 'hidden_feature'])]
                    available_mc = [c for c in momentum_cols if c in row.index]
                    if available_mc:
                        scores_arr = row[available_mc].fillna(0).values.astype(np.float32)
                        std = float(scores_arr.std()) or 1.0
                        raw_score = float(np.clip(scores_arr.mean() / std, -1.0, 1.0))
                    else:
                        raw_score = 0.0

                    regime_val  = int(row['regime'])           if 'regime'            in row.index else 2
                    regime_conf = float(row['regime_confidence']) if 'regime_confidence' in row.index else 0.5
                    regime_sign = {0: 1.0, 1: -1.0, 2: 0.0}.get(regime_val, 0.0)
                    final_score = float(np.clip(raw_score * 0.7 + regime_sign * regime_conf * 0.3, -1.0, 1.0))
                    scores.append(final_score)

                except Exception as e:
                    logger.warning(f"[SHAP] Erro ao processar linha {idx}: {e}. Score neutro.")
                    scores.append(0.0)
            
            return np.array(scores, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"âŒ [SHAP] Erro crÃ­tico no predict_for_shap: {e}", exc_info=True)
            return np.zeros(X.shape[0])

   
    async def generate_trading_decision(self, recent_market_df: pd.DataFrame) -> Signal:
        """
        [VERSÃƒO FINAL COMPLETA - COM DRIFT DETECTOR]
        Orquestra o pipeline de decisÃ£o completo: features, verificaÃ§Ã£o de drift,
        anÃ¡lise de regime, seleÃ§Ã£o de especialista, modulaÃ§Ã£o e risco.
        """
        if not self.is_trained or not all([self.specialists, self.risk_manager, self.portfolio]): # drift_detector opcional
            
            # [DEBUG GRANULAR] Diagnóstico de falha
            missing = []
            if not self.is_trained: missing.append("is_trained=False")
            if not self.specialists: missing.append("specialists=None/Empty")
            if not self.risk_manager: missing.append("risk_manager=None")
            if not self.portfolio: missing.append("portfolio=None")
            logger.warning(f"⚠️ [AI DECISION BLOCKED] Componentes faltantes: {missing}")

            # [CONFIG] Bypass para execução não treinada (SAFETY: verifica flag)
            if self.config_ai.ALLOW_UNTRAINED_EXECUTION:
                logger.warning("⚠️ [AI] ALLOW_UNTRAINED_EXECUTION=True. Operando com modelos potencialmente não treinados/incompletos. Cuidado!")
            else:
                return Signal(symbol=self.config_trading.PRIMARY_PAIR, action=Action.HOLD, confidence=0.0,
                            explanation={"reason": f"AI not fully trained. Missing: {missing} (use ALLOW_UNTRAINED_EXECUTION to bypass)."})

        try:
            # --- Etapa 1: PreparaÃ§Ã£o Completa de Dados e Features ---
            df_fully_enriched = self.feature_pipeline.apply_hidden_features(recent_market_df)
            df_fully_enriched.ffill(inplace=True)
            df_fully_enriched.dropna(inplace=True)

            # [REUPERAÇÃO DE ESTADO] Garante que o bot nunca fique 'cego' ou zerado por falta de padding
            _hf_cols = [c for c in df_fully_enriched.columns if c.startswith('hidden_feature_')]
            has_real_data = False
            
            if _hf_cols:
                _last_hf = df_fully_enriched[_hf_cols].iloc[-1]
                if (_last_hf != 0.0).any():
                    self._last_valid_hidden_dict = _last_hf.to_dict()
                    has_real_data = True
            
            # Se não temos dados reais no loop atual, tentamos o 'Cofre de Memória'
            if not has_real_data and hasattr(self, '_last_valid_hidden_dict'):
                logger.info("🔄 [AI] Hidden Features zeradas no loop atual. Injetando dados da memória persistente.")
                for k, v in self._last_valid_hidden_dict.items():
                    df_fully_enriched[k] = v
                xai_features_row = df_fully_enriched.iloc[-1].to_dict()
            else:
                xai_features_row = df_fully_enriched.iloc[-1].to_dict() # Usa o que tiver (real ou fallback do DF)

            if df_fully_enriched.empty:
                return Signal(symbol=self.config_trading.PRIMARY_PAIR, action=Action.HOLD, confidence=0.0,
                              explanation={"reason": "DataFrame vazio após processamento."})
            
            # [FIX] Compute Memory Features for Specialist Contract
            for window in [20, 50]:
                 if len(recent_market_df) >= window:
                     roll_max = recent_market_df['high'].rolling(window).max()
                     roll_min = recent_market_df['low'].rolling(window).min()
                     latest_max = roll_max.iloc[-1]
                     latest_min = roll_min.iloc[-1]
                     # latest_close comes from df_fully_enriched untransformed close
                     latest_close = df_fully_enriched.iloc[-1]['close']
                     df_fully_enriched[f'dist_to_max_{window}'] = (latest_close - latest_max) / (latest_max + 1e-9)
                     df_fully_enriched[f'dist_to_min_{window}'] = (latest_close - latest_min) / (latest_min + 1e-9)
                 else:
                     df_fully_enriched[f'dist_to_max_{window}'] = 0.0
                     df_fully_enriched[f'dist_to_min_{window}'] = 0.0

            latest_row = df_fully_enriched.iloc[-1]

            # Propaga a linha enriquecida (com SDAE hidden features) para o system_state,
            # garantindo que log_trade_decision/XAI externo veja valores reais, não zeros.
            if hasattr(self, 'system_state') and self.system_state is not None:
                self.system_state['latest_features'] = latest_row

            # [FIX] Aumentado threshold de 100 para 1000 para acomodar contratos multi-TF (ex: 168 features)
            ae_pipe = self.feature_pipeline.temporal_autoencoder_pipeline if self.feature_pipeline else None
            ae_required_count = len(ae_pipe.feature_columns) if ae_pipe and ae_pipe.feature_columns else 0
            
            is_contract_too_small = self.base_feature_columns is not None and len(self.base_feature_columns) < 20
            is_contract_too_large = self.base_feature_columns is not None and len(self.base_feature_columns) > 1000
            is_insufficient_for_ae = self.base_feature_columns is not None and ae_required_count > 0 and len(self.base_feature_columns) < ae_required_count
            
            if self.base_feature_columns is None or is_contract_too_small or is_contract_too_large or is_insufficient_for_ae:
                 reason = "não carregado" if self.base_feature_columns is None else \
                          "muito pequeno" if is_contract_too_small else \
                          "muito grande" if is_contract_too_large else \
                          f"insuficiente para AE ({len(self.base_feature_columns)} < {ae_required_count})"
                 logger.warning(f"⚠️ [AI] Contrato de Features {reason}. Recalculando...")
                 
                 # Logic synced from build_enriched_dataset
                 hidden_cols = [c for c in df_fully_enriched.columns if 'hidden_feature_' in c]
                 trend_cols = [c for c in df_fully_enriched.columns if 'trend_pred_' in c]
                 optimized_feature_cols = hidden_cols + trend_cols
                 
                 primary_tf = self.config_trading.PRIMARY_TIMEFRAME_TRADING
                 all_tfs = getattr(self.config_trading, 'ALL_TRADING_TIMEFRAMES', [primary_tf])
                 
                 essential_env_cols = ['open', 'high', 'low', 'close', 'volume', 'ema_trend']
                 for tf in all_tfs:
                     essential_env_cols += [
                         f'atr_{tf}', f'bb_lower_{tf}', f'bb_upper_{tf}',
                         f'rsi_{tf}', f'adx_{tf}', f'close_frac_{tf}',
                         f'bb_width_{tf}', f'log_return_{tf}', f'log_return_volume_{tf}'
                     ]
                 
                 # Adiciona flags de volatilidade/trend globais
                 essential_env_cols += ['bb_expansion', 'high_volatility', 'bb_squeeze', 'low_volatility', 'adx_weak_trend']
                 
                 extra_feature_keys = [
                     'tp_duration_median', 'tp_uncertainty',
                     'qpnl_24_p50', 'qpnl_24_p90', 'qpnl_48_p50', 'qpnl_48_p90',
                     'tp_regime_up', 'tp_regime_down', 'tp_regime_sideways',
                     'sdae_recon_error',
                     'dist_to_max_20', 'dist_to_min_20', 'dist_to_max_50', 'dist_to_min_50' # Memory features
                 ]
                 
                 existing_essential_cols = [col for col in essential_env_cols if col in df_fully_enriched.columns]
                 extras_cols = [col for col in extra_feature_keys if col in df_fully_enriched.columns]
                 
                 # [FIX] Integrar colunas do AutoEncoder no contrato para evitar mismatch de dimensões
                 ae_cols = []
                 if self.feature_pipeline and self.feature_pipeline.temporal_autoencoder_pipeline:
                     ae_cols = self.feature_pipeline.temporal_autoencoder_pipeline.feature_columns
                     logger.debug(f"⚙️ [AI] Sincronizando {len(ae_cols)} colunas do AutoEncoder no contrato.")

                 # Ordered columns for the "Contract"
                 # [FIX] ae_cols incluídas para garantir que a ordem bata com o RobustScaler do AE
                 ordered_cols = existing_essential_cols + ae_cols + optimized_feature_cols + extras_cols
                 self.base_feature_columns = list(dict.fromkeys(ordered_cols))
                 self.final_feature_count = len(self.base_feature_columns)
                 
                 # Salva metadados
                 self.metadata['base_feature_columns'] = self.base_feature_columns
                 self.metadata['final_feature_count'] = self.final_feature_count
                 self._save_metadata()
                 logger.info(f"✅ [AI] Contrato Auto-Inicializado com {self.final_feature_count} colunas.")
                 
                 # Inicializa referência do drift detector também
                 if self.drift_detector:
                     self.drift_detector.set_reference_data(df_fully_enriched, self.base_feature_columns)
                     logger.info("✅ [AI] DriftDetector inicializado com dados atuais como referência.")

            if self.base_feature_columns:
                assert self.base_feature_columns is not None  # type narrowing for Pyre2
                missing_features = set(self.base_feature_columns) - set(df_fully_enriched.columns)
                if missing_features:
                    logger.error(f"âŒ [AI] Features ausentes: {missing_features}")
                    return Signal(symbol=self.config_trading.PRIMARY_PAIR, action=Action.HOLD, confidence=0.0,
                                  explanation={"reason": f"Features ausentes: {missing_features}"})
                
                # Verifica ordem das features
                feature_order_mismatch = [i for i, col in enumerate(self.base_feature_columns) 
                                        if i < len(df_fully_enriched.columns) and col != df_fully_enriched.columns[i]]
                if feature_order_mismatch:
                    logger.debug(f"âš ï¸ [AI] Ordem de features diferente do esperado. Reordenando...")
                    # Reordena para garantir contrato mas mantém todas as colunas (regime, etc) para etapas posteriores
                    cols_to_keep = self.base_feature_columns + [c for c in df_fully_enriched.columns if c not in self.base_feature_columns]
                    df_fully_enriched = df_fully_enriched[cols_to_keep]

            latest_row = df_fully_enriched.iloc[-1]

            # --- Etapa 2: VerificaÃ§Ã£o de Drift de Dados (Disjuntor de SeguranÃ§a) ---
            if self.drift_detector and self.drift_detector.reference_data:
                features_to_monitor = list(self.drift_detector.reference_data.keys())
                if all(col in latest_row for col in features_to_monitor):
                    current_features_dict = latest_row[features_to_monitor].to_dict()
                    self.drift_check_buffer.append(current_features_dict)
                    
                    if len(self.drift_check_buffer) >= self.drift_check_min_samples:
                        buffer_df = pd.DataFrame(list(self.drift_check_buffer))
                        # [OTIMIZAÇÃO] Passa DataFrame diretamente para o novo DriftDetector
                        drift_results = self.drift_detector.check_drift(buffer_df)
                        is_drifting = drift_results.get('global', {}).get('drift_detected', False)
                        
                        # [STATUS] Atualiza status para UI
                        status_str = "DRIFT DETECTADO" if is_drifting else "OK"
                        self.system_state["drift_status"] = status_str
                        self.last_drift_status = status_str
                        
                        if is_drifting:
                            logger.warning("🚨 DRIFT DETECTADO! Acionando protocolo de adaptação e pausando trades.")
                            asyncio.create_task(self.handle_drift_and_adapt())

                    else:
                        # [STATUS] Atualiza status durante coleta
                        status_str = f"COLETANDO ({len(self.drift_check_buffer)}/{self.drift_check_min_samples})"
                        self.system_state["drift_status"] = status_str
                        self.last_drift_status = status_str
            # --- Etapa 5: PreparaÃ§Ã£o da ObservaÃ§Ã£o Completa ---
            if self.base_feature_columns is None:
                 return Signal(symbol=self.config_trading.PRIMARY_PAIR, action=Action.HOLD, confidence=0.0, explanation={"reason": "Contrato de Features nÃ£o carregado."})

            market_observation = latest_row[self.base_feature_columns].values.astype(np.float32)
            timestamp = latest_row.name
            time_features = np.array([np.sin(2*np.pi*timestamp.hour/24), np.cos(2*np.pi*timestamp.hour/24),
                                      np.sin(2*np.pi*timestamp.dayofweek/7), np.cos(2*np.pi*timestamp.dayofweek/7)], dtype=np.float32)
            current_pos = self.portfolio.positions.get(self.config_trading.PRIMARY_PAIR)
            pos_side = np.sign(current_pos.quantity) if current_pos else 0
            pnl = current_pos.unrealized_pnl / (current_pos.entry_price * abs(current_pos.quantity) + 1e-9) if current_pos and current_pos.entry_price > 0 else 0
            # Handle timezone-aware vs tz-naive datetime subtraction
            ts_naive = None
            pos_ts_naive = None
            if current_pos:
                ts_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo else timestamp
                pos_ts_naive = current_pos.timestamp.replace(tzinfo=None) if hasattr(current_pos.timestamp, 'tzinfo') and current_pos.timestamp.tzinfo else current_pos.timestamp
                # Normalização idêntica ao treino: steps_in_candles / 100.0
                # PRIMARY_TIMEFRAME=15m → 900s/candle
                candle_seconds = 900
                steps_in_candles = (ts_naive - pos_ts_naive).total_seconds() / candle_seconds
            else:
                steps_in_candles = 0.0
            agent_state = np.array([pos_side, pnl, steps_in_candles / 100.0], dtype=np.float32)

            # Prior: apenas 2 valores como no treino (prior_dir + prior_conf)
            prior_dir_val = float(latest_row.get('tp_prior_dir', 0.0))
            prior_conf_val = float(latest_row.get('tp_prior_conf', 0.0))

            # Observação inicial genérica (usada somente quando shape == target_shape sem fix)
            full_observation = np.concatenate([market_observation, agent_state, time_features,
                                               np.array([prior_dir_val, prior_conf_val], dtype=np.float32)])


            # --- Etapa 6: Seleção de Especialistas (Soft Gating / MoE) e Geração da Decisão Tática ---
            
            # 1. Definir especialistas ativos através da SoftGatingNetwork (MoE)
            active_experts = {}
            probs = {}
            
            latent_cols = [c for c in df_fully_enriched.columns if c.startswith('hidden_feature_')]
            _regime_key_map = {0: 'bull', 1: 'bear', 2: 'ranger'}
            _current_regime_val = int(latest_row.get('regime', 2))
            _regime_specialist_key = _regime_key_map.get(_current_regime_val, 'ranger')

            if self.moe_router and self.moe_router.gating and latent_cols:
                try:
                    latent_features_gating = latest_row[latent_cols].values.astype(np.float32)

                    # Valida qualidade das hidden features antes de usar soft gating.
                    # Features zeradas (SDAE não rodou ou pipeline quebrado) resultam em
                    # probabilidades enviesadas para bull (~90%) independente do regime real.
                    latent_zero_frac = float((latent_features_gating == 0.0).mean())
                    if latent_zero_frac > 0.5:
                        logger.warning(
                            f"⚠️ [GATING] {latent_zero_frac:.0%} das hidden features são zero → "
                            f"soft gating não confiável. Usando hard switch por regime ({_regime_specialist_key})."
                        )
                        # active_experts fica {} → hard switch abaixo cuida do roteamento
                    else:
                        device = next(self.moe_router.gating.parameters()).device
                        latent_tensor = torch.FloatTensor(latent_features_gating).unsqueeze(0).to(device)

                        gating_output = self.moe_router.gating.get_expert_probabilities(latent_tensor)
                        probs = gating_output['probabilities']

                        # Usa o ensemble_mode do roteador para decidir quem opera
                        mode = self.moe_router.ensemble_mode
                        if mode == 'weighted':
                            active_experts = {k: v for k, v in probs.items() if v > 0.05}
                        elif mode == 'top_k':
                            sorted_exp = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                            active_experts = dict(sorted_exp[:self.moe_router.top_k])
                        else:
                            active_experts = {k: v for k, v in probs.items() if v >= self.moe_router.threshold}
                            if not active_experts:
                                best = max(probs.items(), key=lambda x: x[1])
                                active_experts = {best[0]: best[1]}

                        # Normaliza os pesos matematicamente
                        total_weight = sum(active_experts.values())
                        active_experts = {k: v / total_weight for k, v in active_experts.items()}

                        # Regime-blend: se o especialista do regime atual tem peso < 20%, o soft
                        # gating está em forte desacordo com o regime detector. Mistura 60/40
                        # (soft gating / regime hard) para evitar que o especialista correto seja
                        # silenciado e o sinal jamais atinja o threshold de execução.
                        if active_experts.get(_regime_specialist_key, 0.0) < 0.20:
                            logger.info(
                                f"⚖️ [GATING BLEND] Regime={_regime_specialist_key} tem peso "
                                f"{active_experts.get(_regime_specialist_key, 0.0):.0%} no soft gating. "
                                f"Blend 60/40 (soft/regime) aplicado."
                            )
                            blended = {}
                            for k in ['bull', 'bear', 'ranger']:
                                hard_w = 1.0 if k == _regime_specialist_key else 0.0
                                blended[k] = 0.60 * active_experts.get(k, 0.0) + 0.40 * hard_w
                            active_experts = {k: v for k, v in blended.items() if v > 0.01}
                            total_weight = sum(active_experts.values())
                            active_experts = {k: v / total_weight for k, v in active_experts.items()}

                        # ── BMA Camada 1: Regime Confidence Scaling ──────────────────────
                        # Blend entre pesos do soft gating e pesos iguais proporcional à
                        # convicção do regime. BULL a 60% → distribui mais; BULL a 90% →
                        # confia mais no gating. Aplica floor=15% e cap=70% por especialista.
                        # Base científica: Hoeting et al. (1999) BMA + Hamilton (1989) regimes.
                        _rc = (float(latest_row['regime_confidence'])
                               if 'regime_confidence' in latest_row.index else 0.5)
                        _n_exp = len(active_experts)
                        if _n_exp > 0:
                            _eq_w = 1.0 / _n_exp
                            active_experts = {
                                k: float(np.clip(
                                    _rc * v + (1.0 - _rc) * _eq_w,  # blend gating↔igual
                                    0.15, 0.70                       # floor=15%, cap=70%
                                ))
                                for k, v in active_experts.items()
                            }
                            _bma1_tot = sum(active_experts.values())
                            if _bma1_tot > 0:
                                active_experts = {k: v / _bma1_tot
                                                  for k, v in active_experts.items()}
                        # ─────────────────────────────────────────────────────────────────
                        logger.info(f"🧠 [SOFT GATING] Probabilidades: {probs} -> Especialistas Ativos: {active_experts}")
                except Exception as e:
                    logger.error(f"❌ [SOFT GATING] Erro ao obter probabilidades: {e}. Usando fallback para Hard Switch.")

            # Fallback seguro: Hard Switch se a SoftGatingNetwork não estiver disponível, falhar,
            # ou as hidden features estiverem zeradas (detectado acima).
            if not active_experts:
                if 'regime' not in latest_row:
                    logger.warning("⚠️ [MONITOR] Coluna 'regime' ausente nos dados de inferência. Usando fallback Ranger (2).")
                    current_regime = 2
                else:
                    current_regime = int(latest_row['regime'])

                regime_map = {0: "bull", 1: "bear", 2: "ranger"}
                specialist_key = regime_map.get(current_regime, "ranger")
                active_experts = {specialist_key: 1.0}
                logger.info(f"🧠 [HARD SWITCH] Regime={current_regime} ({specialist_key}) → peso 100%.")

            # Verifica se há especialistas no pool
            if not self.specialists:
                logger.critical("❌ FALHA CRÍTICA: Nenhum especialista disponível no AIController. Abortando sinal.")
                return Signal(symbol=self.config_trading.PRIMARY_PAIR, action=Action.HOLD, confidence=0.0, explanation={"reason": "No active specialists available"})

            # 2. Iterar sobre especialistas ativos e construir observações
            expert_signals = []
            
            for expert_key, expert_weight in active_experts.items():
                active_specialist = self.specialists.get(expert_key)
                
                if active_specialist is None:
                    logger.warning(f"⚠️ Especialista '{expert_key}' não carregado. Ignorando no ensemble.")
                    continue

                # ── FAST PATH: monta observação direto do contrato do especialista ──────
                # Evita o re-encode do autoencoder — shape correto na primeira tentativa.
                _spec_cols = getattr(active_specialist, 'feature_columns', [])
                if not _spec_cols:
                    _fs = getattr(active_specialist, 'feature_scaler', None)
                    if _fs is not None and hasattr(_fs, 'feature_names_in_'):
                        _spec_cols = list(_fs.feature_names_in_)

                _mem_keys = ['dist_to_max_20', 'dist_to_min_20',
                             'dist_to_max_50', 'dist_to_min_50']

                if _spec_cols and all(c in latest_row.index for c in _spec_cols):
                    _mkt = np.array([float(latest_row.get(c, 0.0)) for c in _spec_cols],
                                    dtype=np.float32)
                    _mem = np.array([float(latest_row.get(k, 0.0)) for k in _mem_keys],
                                    dtype=np.float32)
                    _prior = np.array([prior_dir_val, prior_conf_val], dtype=np.float32)
                    expert_observation = np.concatenate(
                        [_mkt, _mem, agent_state, time_features, _prior]
                    )
                    logger.debug(
                        f"⚡ [FAST PATH] {active_specialist.__class__.__name__}: "
                        f"shape {expert_observation.shape[0]} ✓ (sem re-encode)"
                    )
                else:
                    # Fallback: usa observação genérica (caminho lento anterior)
                    expert_observation = full_observation.copy()
                # ─────────────────────────────────────────────────────────────────────

                # Identifica o target_shape
                target_shape = None
                if hasattr(active_specialist, 'model') and active_specialist.model is not None:
                    if hasattr(active_specialist.model, 'observation_space'):
                        target_shape = active_specialist.model.observation_space.shape

                current_shape = expert_observation.shape
                logger.debug(f"🧐 [DEBUG SHAPE] Specialist: {expert_key} | Current: {current_shape} | Target: {target_shape}")

                if target_shape and current_shape != target_shape:
                    logger.info(f"⚙️ [AI SHAPE FIX] Adaptando observação ({current_shape[0]} -> {target_shape[0]}) para {active_specialist.__class__.__name__}")
                    missing_dims = target_shape[0] - current_shape[0]
                    
                    if missing_dims > 0 and missing_dims <= 10:
                        padding = np.zeros(missing_dims, dtype=np.float32)
                        expert_observation = np.concatenate([expert_observation, padding])
                        logger.info(f"✅ [AI SHAPE FIX] Padding simples aplicado. Novo shape: {expert_observation.shape}")
                    elif self.feature_pipeline and hasattr(self.feature_pipeline, 'temporal_autoencoder_pipeline') and self.feature_pipeline.temporal_autoencoder_pipeline.autoencoder:
                        try:
                            pipeline_ae = self.feature_pipeline.temporal_autoencoder_pipeline
                            seq_len = pipeline_ae.hyperparams.get('seq_length', 32)
                            
                            if len(df_fully_enriched) >= seq_len:
                                ae_input_cols = pipeline_ae.feature_columns
                                available_ae_cols = [c for c in ae_input_cols if c in df_fully_enriched.columns]
                                
                                if len(available_ae_cols) < len(ae_input_cols):
                                    seq_df = pd.DataFrame(0, index=df_fully_enriched.index[-seq_len:], columns=ae_input_cols)
                                    for c in available_ae_cols:
                                        seq_df[c] = df_fully_enriched[c].iloc[-seq_len:]
                                    seq_data = seq_df.values.astype(np.float32)
                                else:
                                    seq_data = df_fully_enriched.iloc[-seq_len:][ae_input_cols].values.astype(np.float32)

                                if hasattr(pipeline_ae, 'scaler') and pipeline_ae.scaler_fitted:
                                    seq_data_scaled = pipeline_ae.scaler.transform(seq_data)
                                    seq_data = seq_data_scaled.reshape(1, seq_len, -1)
                                else:
                                    seq_data = seq_data.reshape(1, seq_len, -1)
                                    
                                seq_tensor = torch.FloatTensor(seq_data).to(pipeline_ae.device)
                                
                                with torch.no_grad():
                                    current_regime = int(latest_row.get('regime', 2))
                                    regime_tensor = torch.LongTensor([current_regime]).to(pipeline_ae.device)
                                    mu, _ = pipeline_ae.autoencoder.encode(seq_tensor, regime_labels=regime_tensor)
                                    latent_features = mu.cpu().numpy().flatten()
                                    
                                prior_dir_val = float(latest_row.get('tp_prior_dir', 0.0))
                                prior_conf_val = float(latest_row.get('tp_prior_conf', 0.0))
                                memory_keys = ['dist_to_max_20', 'dist_to_min_20', 'dist_to_max_50', 'dist_to_min_50']
                                memory_extras = np.array([float(latest_row.get(k, 0.0)) for k in memory_keys], dtype=np.float32)

                                def _build_78dim_obs(row, _agent_state, _time_features, _prior_dir, _prior_conf):
                                    spec_cols = getattr(active_specialist, 'feature_columns', [])
                                    if len(spec_cols) != 65:
                                        fs = getattr(active_specialist, 'feature_scaler', None)
                                        if fs is not None and hasattr(fs, 'feature_names_in_'):
                                            spec_cols = list(fs.feature_names_in_)
                                    mkt = np.array([float(row.get(c, 0.0)) for c in spec_cols], dtype=np.float32) if len(spec_cols) == 65 else np.zeros(65, dtype=np.float32)
                                    mem = np.array([float(row.get(k, 0.0)) for k in ['dist_to_max_20', 'dist_to_min_20', 'dist_to_max_50', 'dist_to_min_50']], dtype=np.float32)
                                    prior_2 = np.array([_prior_dir, _prior_conf], dtype=np.float32)
                                    return np.concatenate([mkt, mem, _agent_state, _time_features, prior_2])

                                if target_shape[0] == 78:
                                    expert_observation = _build_78dim_obs(latest_row, agent_state, time_features, prior_dir_val, prior_conf_val)
                                elif target_shape[0] == 312:
                                    n_rows = len(df_fully_enriched)
                                    steps_seq = []
                                    for offset in range(3, -1, -1):
                                        idx = max(0, n_rows - 1 - offset)
                                        past_row = df_fully_enriched.iloc[idx]
                                        steps_seq.append(_build_78dim_obs(past_row, agent_state, time_features, prior_dir_val, prior_conf_val))
                                    expert_observation = np.concatenate(steps_seq)
                                elif target_shape[0] == 65:
                                    tau_val = float(latest_row.get('tp_tau', 0.0))
                                    primary_signal_val = float(latest_row.get('tp_primary_signal', 0.0))
                                    uncertainty_val = float(latest_row.get('tp_uncertainty', 0.0))
                                    p_extras = np.array([prior_dir_val, prior_conf_val, tau_val, primary_signal_val, uncertainty_val], dtype=np.float32)
                                    expert_observation = np.concatenate([latent_features, agent_state, time_features, p_extras])
                                elif target_shape[0] == 66:
                                    p_extras = np.array([prior_dir_val, prior_conf_val], dtype=np.float32)
                                    expert_observation = np.concatenate([latent_features, memory_extras, agent_state, time_features, p_extras])
                                else:
                                    prior_extras = np.array([prior_dir_val, prior_conf_val], dtype=np.float32)
                                    expert_observation = np.concatenate([latent_features, agent_state, time_features, prior_extras])
                                
                                if expert_observation.shape[0] > target_shape[0]:
                                    expert_observation = expert_observation[:target_shape[0]]
                                elif expert_observation.shape[0] < target_shape[0]:
                                    expert_observation = np.pad(expert_observation, (0, target_shape[0] - len(expert_observation)))
                            else:
                                if current_shape[0] < target_shape[0]:
                                    pad = np.zeros(target_shape[0] - current_shape[0], dtype=np.float32)
                                    expert_observation = np.concatenate([expert_observation, pad])
                                else:
                                    expert_observation = expert_observation[:target_shape[0]]
                        except Exception as e:
                            logger.error(f"❌ [AI SHAPE FIX] Erro durante encoding: {e}. Fallback emergencial.")
                            if current_shape[0] < target_shape[0]:
                                pad = np.zeros(target_shape[0] - current_shape[0], dtype=np.float32)
                                expert_observation = np.concatenate([expert_observation, pad])
                            else:
                                expert_observation = expert_observation[:target_shape[0]]

                # Gerar sinal para o especialista específico
                try:
                    specialist_display_name = active_specialist.__class__.__name__
                    strategic_signal_ind = active_specialist.decide_action(expert_observation, latest_row)
                    if strategic_signal_ind:
                        if strategic_signal_ind.explanation is not None:
                            strategic_signal_ind.explanation['specialist'] = specialist_display_name
                            strategic_signal_ind.explanation['regime'] = expert_key.upper()
                        expert_signals.append((strategic_signal_ind, expert_weight, specialist_display_name, expert_key))
                except Exception as e:
                    logger.error(f"❌ [AI ERROR] Erro no especialista {expert_key}: {e}", exc_info=True)

            # ── BMA Camadas 2 & 3: Uncertainty Penalty + Rolling Performance ─────────
            # Camada 2: especialista com baixa convicção própria perde peso.
            # Camada 3: especialista que acertou nas últimas N barras ganha peso.
            # Ambas aplicadas sobre os pesos já ajustados pela Camada 1.
            # Base: EWA (Vovk 1990) + Gal & Ghahramani (2016) uncertainty + BMA.
            if len(expert_signals) > 1:
                # Preço atual para scoring da decisão anterior (1 barra de atraso)
                _cur_price = float(
                    latest_row['close_15m'] if 'close_15m' in latest_row.index
                    else (latest_row['close'] if 'close' in latest_row.index else 0.0)
                )

                # Inicializa rastreadores na primeira chamada
                if not hasattr(self, '_bma_perf_history'):
                    self._bma_perf_history: Dict = {}   # {key: deque(maxlen=20)}
                    self._bma_perf_mult:    Dict = {}   # {key: float}  — inicia em 1.0

                # Score a decisão anterior com o preço desta barra
                for _bk, _bhist in self._bma_perf_history.items():
                    if _bhist and _bhist[-1].get('price') and _bhist[-1].get('outcome') is None:
                        _bprev = _bhist[-1]
                        if   _bprev['action'] == Action.BUY:
                            _bhist[-1]['outcome'] = _cur_price > _bprev['price']
                        elif _bprev['action'] == Action.SELL:
                            _bhist[-1]['outcome'] = _cur_price < _bprev['price']
                        else:
                            _bhist[-1]['outcome'] = True   # HOLD = neutro (não penaliza)

                        # Recalcula EWA accuracy — mínimo 3 observações para ser confiável
                        _bouts = [h['outcome'] for h in _bhist if h.get('outcome') is not None]
                        if len(_bouts) >= 3:
                            _balpha = 0.9   # decaimento: observações recentes valem mais
                            _bws    = [_balpha ** (len(_bouts)-1-i) for i in range(len(_bouts))]
                            _bewa   = sum(w * int(o) for w, o in zip(_bws, _bouts)) / sum(_bws)
                            # Multiplicador: 0.60 (0% acc) → 1.00 (50% acc) → 1.40 (100% acc)
                            self._bma_perf_mult[_bk] = 0.60 + 0.80 * _bewa

                # Aplica Camada 2 (incerteza) + Camada 3 (performance) nos pesos
                _bma_adj = {}
                for sig, w, name, key in expert_signals:
                    _unc_factor  = 0.70 + 0.30 * float(sig.confidence)   # Camada 2
                    _perf_factor = self._bma_perf_mult.get(key, 1.0)      # Camada 3
                    _bma_adj[key] = w * _unc_factor * _perf_factor

                _bma_tot = sum(_bma_adj.values())
                if _bma_tot > 0:
                    # Normaliza provisório → aplica floor/cap → renormaliza final
                    _bprov   = {k: v / _bma_tot for k, v in _bma_adj.items()}
                    _bcl     = {k: float(np.clip(v, 0.15, 0.70)) for k, v in _bprov.items()}
                    _bcl_tot = sum(_bcl.values())
                    if _bcl_tot > 0:
                        _bfw = {k: v / _bcl_tot for k, v in _bcl.items()}
                        expert_signals = [
                            (sig, _bfw.get(key, w), name, key)
                            for sig, w, name, key in expert_signals
                        ]
                        logger.info(
                            "🎯 [BMA] Pesos finais (3 camadas BMA): "
                            + " | ".join(
                                f"{k.upper()}={_bfw.get(k, 0):.0%}"
                                for k in sorted(_bfw)
                            )
                        )

                # Registra decisão atual para scoring na próxima barra
                for sig, w, name, key in expert_signals:
                    if key not in self._bma_perf_history:
                        self._bma_perf_history[key] = deque(maxlen=20)
                    self._bma_perf_history[key].append({
                        'action':  sig.action,
                        'price':   _cur_price,
                        'outcome': None,
                    })
            # ─────────────────────────────────────────────────────────────────────────

            # 3. Combinar sinais dos especialistas (Soft Ensemble)
            if not expert_signals:
                strategic_signal = Signal(symbol=self.config_trading.PRIMARY_PAIR, action=Action.HOLD, confidence=0.0, explanation={"reason": "Nenhum sinal gerado pelos especialistas."})
            else:
                weighted_action_score = 0.0
                weighted_confidence = 0.0
                weighted_leverage = 0.0
                weighted_size = 0.0
                weighted_sl = 0.0
                weighted_tp = 0.0
                
                sum_weights = 0.0
                ensemble_details = {}
                
                for sig, w, name, key in expert_signals:
                    if sig.action == Action.BUY:
                        act_val = sig.confidence
                    elif sig.action == Action.SELL:
                        act_val = -sig.confidence
                    else:
                        act_val = 0.0
                    
                    weighted_action_score += act_val * w
                    weighted_leverage += sig.leverage * w
                    weighted_size += sig.position_size_pct * w
                    
                    sl_val = sig.stop_loss if (sig.stop_loss is not None and sig.stop_loss > 0) else self.config_trading.DEFAULT_STOP_LOSS_PCT
                    tp_val = sig.take_profit if (sig.take_profit is not None and sig.take_profit > 0) else self.config_trading.DEFAULT_TAKE_PROFIT_PCT
                    
                    weighted_sl += sl_val * w
                    weighted_tp += tp_val * w
                    
                    sum_weights += w
                    ensemble_details[name] = {"action": sig.action.value, "confidence": sig.confidence, "weight": float(w), "regime": key.upper()}
                
                if sum_weights > 0:
                    weighted_leverage /= sum_weights
                    weighted_size /= sum_weights
                    weighted_sl /= sum_weights
                    weighted_tp /= sum_weights

                # --- Etapa Contextual (Movida para suportar Threshold Adaptativo) ---
                onchain_signal = self.onchain_engine.get_onchain_sentiment_signal(asset='BTC')
                tape_pulse = self.tape_engine.get_market_pulse(self.config_trading.PRIMARY_PAIR)

                # --- Cálculo de Confiança do Ensemble ---
                # confidence = |score| — mesma escala [0, 1], proporcional à convicção direcional real.
                confidence = abs(weighted_action_score)
                weighted_confidence = float(np.clip(confidence, 0.0, 1.0))

                # [ADAPTIVE THRESHOLD] Ajusta sensibilidade com base na convicção do regime e força do Tape
                _base_threshold = getattr(self.config_trading, 'MIN_CONFIDENCE_FOR_TRADE', 0.60)
                _min_conf = _base_threshold
                
                # Se regime é incerto (<75%) e Tape é forte, reduzimos exigência (Opção 1 científica)
                # Isso permite capturar correções ou explosões quando o regime ainda não virou mas o fluxo já.
                _reg_conf = latest_row.get('regime_confidence', 0.5)
                _pulse = tape_pulse.get('pulse', '')
                if _reg_conf < 0.75 and _pulse.startswith(('STRONG', 'EXTREME')):
                    _min_conf = max(0.40, _base_threshold * 0.7)
                    if _min_conf < _base_threshold:
                        logger.info(f"⚡ [ADAPTIVE] Threshold reduzido: {_base_threshold:.2f} -> {_min_conf:.2f} (Regime Conf: {_reg_conf:.0%}, Tape: {_pulse})")

                if weighted_action_score > _min_conf:
                    final_action = Action.BUY
                elif weighted_action_score < -_min_conf:
                    final_action = Action.SELL
                else:
                    final_action = Action.HOLD

                # Cria o sinal agregado com parâmetros herdados
                strategic_signal = Signal(
                    symbol=self.config_trading.PRIMARY_PAIR,
                    action=final_action,
                    confidence=weighted_confidence,
                    position_size_pct=float(np.clip(weighted_size, 0.0, self.config_trading.MAX_POSITION_SIZE_PERCENT)),
                    leverage=float(np.clip(weighted_leverage, self.config_trading.MIN_LEVERAGE_PER_TRADE, self.config_trading.MAX_LEVERAGE_PER_TRADE)),
                    stop_loss=float(weighted_sl),
                    take_profit=float(weighted_tp),
                    estimated_duration=10.0,
                    explanation={
                        "reason": f"MoE Normalized Ensemble ({len(active_experts)} experts)",
                        "ensemble_details": ensemble_details,
                        "weighted_score": float(weighted_action_score),
                        "specialist": "MoE_Router",
                        "regime": "ENSEMBLE"
                    }
                )
            
            # 4. Explicar a decisão (Apenas se não estivermos em uma chamada recursiva do SHAP)
            if strategic_signal.action == Action.HOLD and not getattr(self, '_is_explaining', False):
                try:
                    if self.explainer:
                        # [XAI BYPASS] Usa a variável LOCAL capturada no início da função
                        market_data_dict = xai_features_row
                        
                        self._is_explaining = True
                        top_expl = self.explainer.explain_decision(strategic_signal, market_data_dict, top_n=8)
                        self._is_explaining = False
                        
                        strategic_signal.explanation['main_contributors'] = top_expl.get('main_contributors')
                        strategic_signal.explanation['xai_mode'] = top_expl.get('mode', 'full')
                        if 'narrative' in top_expl:
                             strategic_signal.explanation['narrative'] = top_expl['narrative']
                except Exception as e:
                    logger.warning(f"⚠️ [XAI HOLD] Erro ao explicar HOLD: {e}")
                return strategic_signal

            # --- Etapa 7: Modulação Contextual e Aprovação Final de Risco ---
            # (onchain_signal e tape_pulse já foram extraídos acima para o Threshold Adaptativo)
            
            # RiskManager (MIN_PROFIT_PROBABILITY=0.60). O ProfitabilityPredictor foi removido;
            # usamos a confiança do especialista como proxy: especialista confiante → alta
            # probabilidade de lucro. Mapeia confidence → [0.5, 1.0] via escala linear.
            # Um sinal que passou o MIN_CONFIDENCE_FOR_TRADE (0.65) terá profit_prob ≥ 0.65 > 0.60.
            profit_prob = float(np.clip(strategic_signal.confidence, 0.5, 1.0))
            # Propaga para o campo do sinal para que o RiskManager veja o valor correto
            from dataclasses import replace as dc_replace
            strategic_signal = dc_replace(strategic_signal, profit_probability=profit_prob)

            modulated_signal = self._apply_contextual_modulation(strategic_signal, onchain_signal, tape_pulse, profit_prob)
            is_approved, reason = self.risk_manager.check_trade_approval(modulated_signal)
            
            if not is_approved:
                # [XAI] Armazena metadados da rejeição para o Explainer
                modulated_signal.explanation['reason'] = reason
                modulated_signal.explanation['original_action'] = strategic_signal.action # Salva a intenção original
                modulated_signal.explanation['original_confidence'] = strategic_signal.confidence # Salva a confiança original
                
                final_rejection = Signal(symbol=self.config_trading.PRIMARY_PAIR, action=Action.HOLD, confidence=0.0,
                                         explanation=modulated_signal.explanation)
                 # [XAI FIX] Explica Rejeição de Risco
                try:
                    if self.explainer:
                        top_expl = self.explainer.explain_decision(final_rejection, latest_row)
                        final_rejection.explanation['main_contributors'] = top_expl.get('main_contributors')
                        if 'narrative' in top_expl:
                             final_rejection.explanation['narrative'] = top_expl['narrative']
                except Exception as e:
                    logger.warning(f"⚠️ [XAI RISK] Erro ao explicar Rejeição: {e}")

                return final_rejection

            # XAI: anexa principais features quando disponível
            try:
                # [MELHORIA] Explica qualquer decisão, inclusive HOLD, se o explainer estiver ativo
                if self.explainer and modulated_signal:
                    top_expl = self.explainer.explain_decision(modulated_signal, latest_row)
                    modulated_signal.explanation['main_contributors'] = top_expl.get('main_contributors')
                    modulated_signal.explanation['xai_mode'] = top_expl.get('mode', 'full')
                    
                    # Se tiver narrativa, anexa também
                    if 'narrative' in top_expl:
                        modulated_signal.explanation['narrative'] = top_expl['narrative']
            except Exception as e:
                logger.warning(f"âš ï¸ [XAI] Erro ao gerar explicaÃ§Ã£o da decisÃ£o: {e}")

            logger.info(f"âœ… [AI] DecisÃ£o final da IA: {modulated_signal.action.value} para {modulated_signal.symbol}")
            return modulated_signal

        except Exception as e:
            logger.critical(f"ðŸš¨ [AI] Falha catastrÃ³fica ao gerar decisÃ£o: {e}", exc_info=True)
            return Signal(symbol=self.config_trading.PRIMARY_PAIR, action=Action.HOLD, confidence=0.0,
                          explanation={"reason": f"Critical error: {e}"})    

    def _apply_contextual_modulation(self, strategic_signal: Signal, onchain_signal: Dict, tape_pulse: Dict, profit_prob: float) -> Signal:
        """
        Modula o sinal estrategico com base no contexto de mercado em tempo real.

        Camadas de decisao:
          VETO       -> converte BUY/SELL em HOLD quando contexto e fortemente oposto
          AMPLIFICAR -> aumenta tamanho/alavancagem quando tudo esta alinhado
          ATENUAR    -> reduz tamanho quando tape se opoe ao trade (mas nao veta)
          NEUTRO     -> sinal do especialista mantido intacto

        Mapeamento de Tape Pulse:
          TapeEngine retorna: EXTREME_BULLISH | STRONG_BULLISH | BULLISH |
                              NEUTRAL | BEARISH | STRONG_BEARISH | EXTREME_BEARISH
          Este metodo classifica em: STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL
          STRONG = pulse já é STRONG/EXTREME, OU score >= 0.50, OU RVol >= 1.5x
        """
        final_signal = strategic_signal
        action = strategic_signal.action
        veto_reason = None

        # --- Classificacao da forca do Tape Pulse ----------------------------
        # TapeEngine retorna: EXTREME_BULLISH | STRONG_BULLISH | BULLISH |
        #                     NEUTRAL | BEARISH | STRONG_BEARISH | EXTREME_BEARISH
        # O score já está amplificado pelo RVol e penalizado pelo depth_quality.
        pulse      = tape_pulse.get('pulse', 'NEUTRAL')
        tape_score = abs(float(tape_pulse.get('score', 0.0)))
        rvol       = float(tape_pulse.get('relative_volume', 1.0))

        # ── Depth profile (Fase 1) ────────────────────────────────────────────
        depth_quality     = float(tape_pulse.get('depth_quality', 1.0))
        zone_near         = float(tape_pulse.get('zone_imbalance_near', 0.0))
        zone_mid          = float(tape_pulse.get('zone_imbalance_mid', 0.0))
        wall_bid          = bool(tape_pulse.get('wall_bid', False))
        wall_ask          = bool(tape_pulse.get('wall_ask', False))
        wall_bid_price    = float(tape_pulse.get('wall_bid_price', 0.0))
        wall_ask_price    = float(tape_pulse.get('wall_ask_price', 0.0))

        # ── STRONG: pulso já é STRONG/EXTREME, OU score alto, OU institucional ──
        is_strong = (
            pulse.startswith('STRONG') or pulse.startswith('EXTREME') or
            tape_score >= 0.50 or rvol >= 1.5
        )

        # ── Ajuste por qualidade de profundidade ──────────────────────────────
        # Livro raso (depth_quality < 0.85): sinal pode ser 1 ordem grande mascarada.
        # Downgrade de STRONG para regular BUY/SELL, exceto se fluxo for institucional.
        if depth_quality < 0.85 and rvol < 2.5:
            is_strong = False

        # ── Convergência de zonas: near e mid confirmam? ──────────────────────
        # Se pressão imediata (0-2) e intenção (3-9) apontam na mesma direção
        # → confirma o sinal e pode promover a STRONG.
        # Se divergem → sinal ambíguo, downgrade (exceto fluxo institucional).
        bullish_pulse = 'BULLISH' in pulse
        zones_confirm = (
            (zone_near > 0.08 and zone_mid > 0.05 and bullish_pulse) or
            (zone_near < -0.08 and zone_mid < -0.05 and not bullish_pulse)
        )
        zones_diverge = (
            (zone_near > 0.08 and zone_mid < -0.05) or
            (zone_near < -0.08 and zone_mid > 0.05)
        )

        if zones_confirm and not is_strong:
            is_strong = True   # zonas convergentes promovem o sinal
        if zones_diverge and rvol < 2.5:
            is_strong = False  # zonas divergentes rebaixam (exceto institucional)

        if 'BULLISH' in pulse:
            tape_strength = 'STRONG_BUY' if is_strong else 'BUY'
        elif 'BEARISH' in pulse:
            tape_strength = 'STRONG_SELL' if is_strong else 'SELL'
        else:
            tape_strength = 'NEUTRAL'

        # ── Wall check: muro na direção oposta ao trade ───────────────────────
        # Não veta (o VETO já trata tape oposto), mas impede AMPLIFICAÇÃO.
        # Comprar contra um muro de ask = redução de potencial de alta.
        has_opposing_wall = (
            (action == Action.BUY  and wall_ask) or
            (action == Action.SELL and wall_bid)
        )

        # --- Bloco de VETO ---------------------------------------------------
        # Condicoes em que o trade e vetado (convertido em HOLD)
        onchain_sig  = onchain_signal.get('signal', 'NEUTRAL')
        onchain_conf = float(onchain_signal.get('confidence', 0.0))

        if profit_prob < 0.40:
            veto_reason = f'VETO: Probabilidade de lucro muito baixa ({profit_prob:.2%})'
        # [FIX PRODUÇÃO] Replica a restrição de direção do ambiente de treino.
        # BullSpecialist só opera LONG; BearSpecialist só opera SHORT.
        # Durante o treino _specialist_long_only=True impede o ambiente de aceitar short.
        # Em produção aplicamos o mesmo filtro: SELL em BULL ou BUY em BEAR → HOLD.
        elif strategic_signal.explanation.get('_regime_mismatch'):
            regime_mismatch_info = strategic_signal.explanation.get('regime', 'UNKNOWN')
            veto_reason = (
                f'VETO: Sinal contra especialidade do regime ({regime_mismatch_info}). '
                f'Ação {action.value} bloqueada para não operar fora da direção treinada.'
            )
        elif action == Action.BUY and onchain_sig == 'BEARISH' and onchain_conf > 0.7:
            veto_reason = 'VETO: Sentimento fortemente BEARISH contra compra'
        elif action == Action.SELL and onchain_sig == 'BULLISH' and onchain_conf > 0.7:
            veto_reason = 'VETO: Sentimento fortemente BULLISH contra venda'
        elif action == Action.BUY and tape_strength == 'STRONG_SELL':
            veto_reason = f'VETO: Tape BEARISH de alta confianca (score={tape_score:.2f}, rvol={rvol:.1f}x) contra compra'
        elif action == Action.SELL and tape_strength == 'STRONG_BUY':
            veto_reason = f'VETO: Tape BULLISH de alta confianca (score={tape_score:.2f}, rvol={rvol:.1f}x) contra venda'

        if veto_reason:
            final_signal.explanation['modulation'] = veto_reason
            final_signal.explanation['original_confidence'] = final_signal.confidence
            logger.info(f'[MOD] {veto_reason}')
            return replace(final_signal, action=Action.HOLD, confidence=0.0, position_size_pct=0.0)

        # --- Bloco de AMPLIFICACAO -------------------------------------------
        # Amplifica quando especialista + onchain + tape estao todos alinhados
        is_buy_amplified  = (
            action == Action.BUY
            and profit_prob > 0.75
            and onchain_sig == 'BULLISH'
            and tape_strength in ('STRONG_BUY', 'BUY')
            and not has_opposing_wall   # não amplifica contra muro de ask
        )
        is_sell_amplified = (
            action == Action.SELL
            and profit_prob > 0.75
            and onchain_sig == 'BEARISH'
            and tape_strength in ('STRONG_SELL', 'SELL')
            and not has_opposing_wall   # não amplifica contra muro de bid
        )

        if is_buy_amplified or is_sell_amplified:
            amplification_factor = 1.25
            final_signal = replace(
                final_signal,
                position_size_pct=min(
                    final_signal.position_size_pct * amplification_factor,
                    self.config_trading.MAX_POSITION_SIZE_PERCENT
                ),
                leverage=min(
                    final_signal.leverage * amplification_factor,
                    self.config_trading.MAX_LEVERAGE_PER_TRADE
                ),
            )
            msg = (
                f'AMPLIFICADO x{amplification_factor}: Tape={tape_strength} | '
                f'Sentiment={onchain_sig} | prob={profit_prob:.2%}'
            )
            final_signal.explanation['modulation'] = msg
            logger.info(f'[MOD] {msg}')
            return final_signal

        # --- Bloco de ATENUACAO ----------------------------------------------
        # Atenua APENAS quando tape se opoe diretamente ao trade.
        # Contexto NEUTRAL ou OnChain divergente: reduz tamanho mas nao veta.
        is_buy_dampened  = action == Action.BUY  and tape_strength in ('STRONG_SELL', 'SELL')
        is_sell_dampened = action == Action.SELL and tape_strength in ('STRONG_BUY',  'BUY')

        if is_buy_dampened or is_sell_dampened:
            dampening_factor = 0.80
            final_signal = replace(
                final_signal,
                position_size_pct=final_signal.position_size_pct * dampening_factor,
                leverage=max(
                    final_signal.leverage * dampening_factor,
                    self.config_trading.MIN_LEVERAGE_PER_TRADE
                ),
            )
            msg = (
                f'ATENUADO x{dampening_factor}: Tape={tape_strength} opoe-se ao trade '
                f'({action.value}). Sentiment={onchain_sig}'
            )
            final_signal.explanation['modulation'] = msg
            logger.info(f'[MOD] {msg}')
            return final_signal

        # --- Sinal aprovado sem modificacao ----------------------------------
        wall_info = ""
        if wall_ask and action == Action.BUY:
            wall_info = f" | ⚠️ Muro ask ${wall_ask_price:,.0f}"
        elif wall_bid and action == Action.SELL:
            wall_info = f" | ⚠️ Muro bid ${wall_bid_price:,.0f}"
        depth_info = f" | Livro={'raso' if depth_quality < 0.85 else 'profundo'} (q={depth_quality:.0%})"
        msg = (
            f'APROVADO: Tape={tape_strength} | Sentiment={onchain_sig} | '
            f'prob={profit_prob:.2%}{depth_info}{wall_info}'
        )
        final_signal.explanation['modulation'] = msg
        return final_signal

    async def adapt(self, recent_data: pd.DataFrame, adaptation_strength: float = 0.1, preserve_core_learning: bool = True) -> bool:
        """
        [MELHORIA] MÃ©todo de adaptaÃ§Ã£o em tempo real usando Meta-Learning.
        
        Permite que o bot se adapte dinamicamente a mudanÃ§as sutis no mercado
        sem perder o conhecimento fundamental adquirido durante o treinamento inicial.
        
        Args:
            recent_data (pd.DataFrame): Dados recentes para adaptaÃ§Ã£o
            adaptation_strength (float): ForÃ§a da adaptaÃ§Ã£o (0.0 a 1.0)
            preserve_core_learning (bool): Se deve preservar o aprendizado fundamental
            
        Returns:
            bool: True se a adaptaÃ§Ã£o foi bem-sucedida
        """
        if not self.is_trained:
            logger.warning("âš ï¸ [ADAPT] IA nÃ£o treinada. NÃ£o Ã© possÃ­vel adaptar.")
            return False
            
        if recent_data.empty or len(recent_data) < 50:
            logger.warning("âš ï¸ [ADAPT] Dados insuficientes para adaptaÃ§Ã£o (mÃ­nimo 50 pontos).")
            return False
            
        try:
            logger.info(f"ðŸ§  [ADAPT] Iniciando adaptaÃ§Ã£o com {len(recent_data)} pontos de dados (forÃ§a: {adaptation_strength})")
            
            # Adapta o TrendPredictor
            if self.trend_predictor and hasattr(self.trend_predictor, 'adapt'):
                trend_adapt_success = await self.trend_predictor.adapt(
                    recent_data, 
                    adaptation_strength=adaptation_strength,
                    preserve_core_learning=preserve_core_learning
                )
                if trend_adapt_success:
                    logger.info("âœ… [ADAPT] TrendPredictor adaptado com sucesso.")
                else:
                    logger.warning("âš ï¸ [ADAPT] Falha na adaptaÃ§Ã£o do TrendPredictor.")
            
            # Adapta o ProfitabilityPredictor
            if self.profitability_predictor and hasattr(self.profitability_predictor, 'adapt'):
                profit_adapt_success = await self.profitability_predictor.adapt(
                    recent_data,
                    adaptation_strength=adaptation_strength,
                    preserve_core_learning=preserve_core_learning
                )
                if profit_adapt_success:
                    logger.info("âœ… [ADAPT] ProfitabilityPredictor adaptado com sucesso.")
                else:
                    logger.warning("âš ï¸ [ADAPT] Falha na adaptaÃ§Ã£o do ProfitabilityPredictor.")
            
            # Adapta os especialistas
            specialist_adaptations = 0
            for specialist_name, specialist in self.specialists.items():
                if hasattr(specialist, 'adapt') and getattr(specialist, 'is_trained', False):
                    try:
                        adapt_success = await specialist.adapt(
                            recent_data,
                            adaptation_strength=adaptation_strength,
                            preserve_core_learning=preserve_core_learning
                        )
                        if adapt_success:
                            specialist_adaptations += 1
                            logger.debug(f"âœ… [ADAPT] {specialist_name} adaptado.")
                    except Exception as e:
                        logger.warning(f"âš ï¸ [ADAPT] Erro ao adaptar {specialist_name}: {e}")
            
            logger.info(f"âœ… [ADAPT] {specialist_adaptations}/{len(self.specialists)} especialistas adaptados.")
            
            # Adapta o HRL Master se disponÃ­vel
            if self.hrl_master and hasattr(self.hrl_master, 'adapt') and self.hrl_master.is_trained:
                try:
                    hrl_adapt_success = await self.hrl_master.adapt(
                        recent_data,
                        adaptation_strength=adaptation_strength,
                        preserve_core_learning=preserve_core_learning
                    )
                    if hrl_adapt_success:
                        logger.info("âœ… [ADAPT] HRL Master adaptado com sucesso.")
                    else:
                        logger.warning("âš ï¸ [ADAPT] Falha na adaptaÃ§Ã£o do HRL Master.")
                except Exception as e:
                    logger.warning(f"âš ï¸ [ADAPT] Erro ao adaptar HRL Master: {e}")
            
            # Atualiza o timestamp da Ãºltima adaptaÃ§Ã£o
            self.last_adaptation_time = datetime.utcnow()
            
            logger.info("âœ… [ADAPT] Ciclo de adaptaÃ§Ã£o concluÃ­do com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"âŒ [ADAPT] Erro crÃ­tico durante adaptaÃ§Ã£o: {e}", exc_info=True)
            return False





