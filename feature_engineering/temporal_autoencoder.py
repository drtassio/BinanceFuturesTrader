# -----------------------------------------------------------------------------
# ARQUIVO: feature_engineering/temporal_autoencoder.py
# -----------------------------------------------------------------------------

"""
Sequence Denoising Autoencoder (SDAE) Temporal para Features Financeiras.

Este módulo implementa um autoencoder temporal robusto que:
1. Captura padrões em diferentes escalas temporais usando CNN dilatada
2. Gera representações latentes estáveis para séries financeiras
3. Evita vazamento temporal através de janelas deslizantes
4. Integra-se perfeitamente com o pipeline de features existente
"""

import os
import json
import warnings
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.exceptions import TrialPruned

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from utils.logger import get_logger
from config.settings import AIConfig
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from collections import Counter
import joblib

# SCIENTIFIC CORRECTIONS: Novos módulos para validação e CV
from feature_engineering.financial_stationarity import StationarityValidator
from utils.purged_cv import PurgedKFold, TimeSeriesSplitWithEmbargo
# from learning.regime_switch import RegimeSwitch  # CVAE: Importa RegimeSwitch (REMOVIDO)

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

logger = get_logger("TemporalAutoencoder")


class LazyTimeSeiresDataset(torch.utils.data.Dataset):
    """
    [OTIMIZACAO] Dataset que gera janelas temporais sob demanda (Lazy Loading).
    
    Evita a criação de arrays 3D gigantes em memória (N, T, F), 
    economizando ~10x RAM em datasets grandes.
    """
    
    def __init__(self, data: np.ndarray, seq_length: int, regime_labels: Optional[np.ndarray] = None):
        """
        Args:
            data: Array 2D (n_samples, n_features)
            seq_length: Tamanho da janela temporal
            regime_labels: Array 1D (n_samples,) com IDs de regime
        """
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        self.n_samples = len(data) - seq_length + 1
        
        # CVAE: Armazena labels de regime se fornecidos
        if regime_labels is not None:
            self.regime_labels = torch.LongTensor(regime_labels)
        else:
            self.regime_labels = None
        
    def __len__(self):
        return max(0, self.n_samples)
    
    def __getitem__(self, idx):
        # Fatia a janela sob demanda (Zero-Copy view se possível)
        window = self.data[idx : idx + self.seq_length]
        
        if self.regime_labels is not None:
            # Retorna o regime do final da janela (contexto atual)
            # idx + seq_length - 1 é o índice do último elemento da janela
            regime = self.regime_labels[idx + self.seq_length - 1]
            return window, regime
            
        return (window,)


class TemporalSDAE(nn.Module):
    """
    [MELHORIA] Sequence Denoising Autoencoder com arquitetura temporal robusta.
    
    Encoder: CNN dilatada (1, 2, 4) + GRU bidirecional + pooling
    Decoder: GRU + ConvTranspose1d simétrico + reconstrução
    """
    
    def __init__(self, input_dim: int, cnn_out_channels: int, kernel_size: int,
                 gru_hidden: int, gru_layers: int, latent_dim: int, 
                 seq_length: int, dropout: float = 0.1,
                 p_mask: float = 0.25, sigma: float = 0.12,
                 num_regimes: int = 3, regime_emb_dim: int = 4):  # CVAE: Novos parâmetros
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.p_mask = p_mask
        self.sigma = sigma
        self.regime_emb_dim = regime_emb_dim
        
        # CVAE: Embedding de Regime
        # 0=Bull, 1=Bear, 2=Ranger (conforme RegimeSwitch)
        self.regime_embedding = nn.Embedding(num_regimes, regime_emb_dim)
        
        # Encoder: CNN dilatada com diferentes escalas
        # Distribui canais de saída garantindo soma exata = cnn_out_channels
        _branches = [1, 2, 4]
        _base = cnn_out_channels // len(_branches)
        _rem = cnn_out_channels - _base * len(_branches)
        self._encoder_branch_outs = [
            _base + (1 if i < _rem else 0) for i in range(len(_branches))
        ]
        self.encoder_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=self._encoder_branch_outs[i],
                kernel_size=kernel_size,
                padding='same',
                dilation=dil,
                bias=False
            ) for i, dil in enumerate(_branches)
        ])

        # Normalização robusta a batch-size e sem exigência de canais fixos exatos
        self.encoder_out_channels = sum(self._encoder_branch_outs)
        # Fix: Garante que num_groups é divisível por encoder_out_channels
        num_groups = max(1, min(8, self.encoder_out_channels // 8))
        # Ajusta num_groups para ser divisível pelo número de canais
        while self.encoder_out_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.encoder_norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=self.encoder_out_channels
        )
        self.encoder_relu = nn.ReLU()
        self.encoder_dropout = nn.Dropout(dropout)
        
        # Encoder: GRU bidirecional
        # CVAE: Input size aumenta com o embedding do regime
        self.encoder_gru = nn.GRU(
            input_size=cnn_out_channels + regime_emb_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=True
        )
        
        self.gru_output_dim = gru_hidden * 2
        self.latent_pre = nn.Sequential(
            nn.Linear(self.gru_output_dim, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.latent_mu = nn.Linear(gru_hidden, latent_dim)
        self.latent_logvar = nn.Linear(gru_hidden, latent_dim)
        
        # Decoder: GRU para desdobrar o latente
        # CVAE: Input size aumenta com o embedding do regime
        self.decoder_gru = nn.GRU(
            input_size=latent_dim + regime_emb_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=False
        )
        
        # Decoder: Arquitetura Refinada (Conv1d direto)
        # self.decoder_upsample removido (redundante)
        
        mid_channels = max(64, input_dim // 2)
        self.decoder_conv_1 = nn.Conv1d(
            in_channels=gru_hidden,
            out_channels=mid_channels,
            kernel_size=3,
            padding='same', # Mantém comprimento temporal
            bias=False
        )
        
        self.decoder_final_proj = nn.Conv1d(
            in_channels=mid_channels,
            out_channels=input_dim,
            kernel_size=1,
            padding=0,
            bias=False
        )
        
        # Normalização no decoder
        decoder_num_groups = max(1, min(8, input_dim // 8))
        while input_dim % decoder_num_groups != 0 and decoder_num_groups > 1:
            decoder_num_groups -= 1
            
        self.decoder_norm = nn.GroupNorm(
            num_groups=decoder_num_groups,
            num_channels=input_dim
        )
        self.decoder_relu = nn.ReLU()
        self.decoder_dropout = nn.Dropout(dropout)
        
        # Inicialização de pesos
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Inicializa pesos usando Kaiming (He) para ReLU e Xavier para outros."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            # Kaiming Initialization é superior para ReLU para evitar vanishing gradients
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def forward(self, x: torch.Tensor, regime_labels: Optional[torch.Tensor] = None, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass com denoising e masking durante treino.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_dim)
            regime_labels: Tensor de inteiros (batch_size,) com IDs do regime (0,1,2)
            training: Se está em modo treino (aplica denoising)
            
        Returns:
            Tuple (reconstructed, mu, logvar, latent)
        """
        batch_size, seq_length, input_dim = x.shape
        
        # PATCH C: Denoising configurável via hiperparâmetros otimizados
        if training:
            # Usa parâmetros otimizados pelo Optuna (self.p_mask, self.sigma)
            # Mascara posições aleatórias
            mask = (torch.rand_like(x) > self.p_mask).float()
            
            # Ruído gaussiano APENAS nas posições mascaradas
            noise = torch.randn_like(x) * self.sigma
            x = x * mask + noise * (1 - mask)  # Ruído só nas posições mascaradas
        
        # Encoder: CNN dilatada
        x_conv = x.permute(0, 2, 1)  # (batch, input_dim, seq_length)
        conv_outputs = []
        
        for conv in self.encoder_convs:
            conv_out = conv(x_conv)
            conv_outputs.append(conv_out)
        
        # Concatena saídas das CNNs
        x_conv = torch.cat(conv_outputs, dim=1)  # (batch, encoder_out_channels, seq_length)
        x_conv = self.encoder_norm(x_conv)
        x_conv = self.encoder_relu(x_conv)
        x_conv = self.encoder_dropout(x_conv)
        
        # Encoder: GRU bidirecional
        x_gru = x_conv.permute(0, 2, 1)  # (batch, seq_length, cnn_out_channels)
        
        # CVAE: Injeta regime embedding no Encoder
        if regime_labels is not None:
            regime_emb = self.regime_embedding(regime_labels)  # (batch, emb_dim)
            regime_emb_seq = regime_emb.unsqueeze(1).expand(-1, seq_length, -1)  # (batch, seq_len, emb_dim)
            x_gru = torch.cat([x_gru, regime_emb_seq], dim=-1)  # Concatena na dimensão de features
        else:
            # Fallback se não houver labels (usa embedding zero ou default)
            # Para robustez, usamos regime 2 (Ranger/Neutro) como default
            dummy_labels = torch.full((batch_size,), 2, dtype=torch.long, device=x.device)
            regime_emb = self.regime_embedding(dummy_labels)
            regime_emb_seq = regime_emb.unsqueeze(1).expand(-1, seq_length, -1)
            x_gru = torch.cat([x_gru, regime_emb_seq], dim=-1)

        gru_out, _ = self.encoder_gru(x_gru)
        
        # Pooling temporal: usa o último passo da sequência
        last_hidden = gru_out[:, -1, :]  # (batch, gru_hidden * 2)
        
        latent_base = self.latent_pre(last_hidden)
        mu = self.latent_mu(latent_base)
        logvar = self.latent_logvar(latent_base)
        latent = self._reparameterize(mu, logvar, training=training)
        latent_expanded = latent.unsqueeze(1).expand(-1, seq_length, -1)  # (batch, seq_length, latent_dim)
        
        # Decoder: GRU
        # CVAE: Injeta regime embedding no Decoder (concatena com latente)
        latent_conditioned = torch.cat([latent_expanded, regime_emb_seq], dim=-1)
        
        decoder_gru_out, _ = self.decoder_gru(latent_conditioned)
        
        # Decoder: Nova arquitetura (Upsample + Conv1d)
        decoder_conv = decoder_gru_out.permute(0, 2, 1)  # (batch, gru_hidden, seq_length)
        
        # Passo 1: Refinamento com Conv1d (Upsample removido)
        x_refined = self.decoder_conv_1(decoder_conv)
        
        # Passo 3: Projeção final
        reconstructed = self.decoder_final_proj(x_refined)  # (batch, input_dim, seq_length)
        reconstructed = self.decoder_norm(reconstructed)
        
        if reconstructed.shape != (batch_size, input_dim, seq_length):
            raise RuntimeError(
                f"Decoder alterou shape: esperado ({batch_size}, {input_dim}, {seq_length}), "
                f"got {reconstructed.shape}"
            )
        
        reconstructed = self.decoder_relu(reconstructed)
        reconstructed = self.decoder_dropout(reconstructed)
        
        # Retorna para formato original
        reconstructed = reconstructed.permute(0, 2, 1)  # (batch, seq_length, input_dim)
        
        return reconstructed, mu, logvar, latent

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def encode(self, x: torch.Tensor, regime_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Método para inferência: retorna média e log variância."""
        self.eval()
        with torch.no_grad():
            batch_size, seq_length, _ = x.shape
            
            # Aplica apenas a parte do encoder
            x_conv = x.permute(0, 2, 1)
            conv_outputs = []
            
            for conv in self.encoder_convs:
                conv_out = conv(x_conv)
                conv_outputs.append(conv_out)
            
            x_conv = torch.cat(conv_outputs, dim=1)
            x_conv = self.encoder_norm(x_conv)
            x_conv = self.encoder_relu(x_conv)
            
            x_gru = x_conv.permute(0, 2, 1)
            
            # CVAE: Injeta regime embedding (ou fallback)
            if regime_labels is not None:
                regime_emb = self.regime_embedding(regime_labels)
            else:
                # Fallback: usa regime neutro (Ranger = 2)
                dummy_labels = torch.full((batch_size,), 2, dtype=torch.long, device=x.device)
                regime_emb = self.regime_embedding(dummy_labels)
            
            regime_emb_seq = regime_emb.unsqueeze(1).expand(-1, seq_length, -1)
            x_gru = torch.cat([x_gru, regime_emb_seq], dim=-1)
            
            gru_out, _ = self.encoder_gru(x_gru)
            
            last_hidden = gru_out[:, -1, :]
            latent_base = self.latent_pre(last_hidden)
            mu = self.latent_mu(latent_base)
            logvar = self.latent_logvar(latent_base)
            return mu, logvar


class TemporalAutoencoderPipeline:
    """
    Pipeline completo para treinamento e aplicação do SDAE temporal.
    """
    
    def __init__(self, config_ai: AIConfig):
        self.config_ai = config_ai
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Modelo e hiperparâmetros
        self.autoencoder: Optional[TemporalSDAE] = None
        self.hyperparams: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self.scaler = RobustScaler()
        self.scaler_fitted = False
        
        # SCIENTIFIC CORRECTIONS: Validadores e configurações
        self.stationarity_validator = StationarityValidator()
        self.enforce_stationarity = getattr(config_ai, 'ENFORCE_STATIONARITY', True)
        self.use_walk_forward_cv = getattr(config_ai, 'USE_WALK_FORWARD_CV', False)
        self.min_compression_ratio = getattr(config_ai, 'MIN_COMPRESSION_RATIO', 8.0)  # FIXED: Increased from 4.0 to 8.0
        self.embargo_bars = getattr(config_ai, 'EMBARGO_BARS', None)  # Auto-calculate if None
        
        # Hardening: seed determinística para reprodutibilidade
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        # Diretórios e caminhos
        self._setup_directories()
        
        # Carrega estado se existir
        self.load_state()
        
        
        # [CACHE] Regime Detector
        self.regime_detector = None
        
        logger.info(f"🚀 [SDAE] Pipeline inicializado no dispositivo: {self.device}")
    
    def set_debug_mode(self, enabled: bool = True):
        """Configura modo debug para detecção de anomalias."""
        if enabled:
            torch.autograd.set_detect_anomaly(True)
            logger.info("🔍 [SDAE] Modo debug ativado - detecção de anomalias habilitada")
        else:
            torch.autograd.set_detect_anomaly(False)
            logger.info("🔍 [SDAE] Modo debug desativado")
    
    def _setup_directories(self):
        """Configura diretórios para persistência."""
        self.model_dir = os.path.join(self.config_ai.MODEL_DIR, "autoencoder")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.hyperparams_path = os.path.join(self.model_dir, "autoencoder_hyperparams.json")
        self.feature_columns_path = os.path.join(self.model_dir, "autoencoder_feature_columns.json")
        self.scaler_path = os.path.join(self.model_dir, "autoencoder_scaler.joblib")
        self.model_path = os.path.join(self.model_dir, "temporal_sdae.pth")
    
    def load_state(self):
        """Carrega estado salvo se existir."""
        try:
            if os.path.exists(self.hyperparams_path):
                with open(self.hyperparams_path, 'r') as f:
                    self.hyperparams = json.load(f)
                logger.info("✅ [SDAE] Hiperparâmetros carregados.")
            else:
                logger.warning(f"⚠️ [SDAE] Arquivo de hiperparâmetros não encontrado: {self.hyperparams_path}")
                return
            
            if os.path.exists(self.feature_columns_path):
                with open(self.feature_columns_path, 'r') as f:
                    self.feature_columns = json.load(f)
                logger.info("✅ [SDAE] Colunas de features carregadas.")

                # Ajuste defensivo: sincroniza tamanho das colunas com input_dim
                try:
                    input_dim = int(self.hyperparams.get('input_dim', len(self.feature_columns)))
                    if len(self.feature_columns) != input_dim:
                        logger.warning(
                            f"⚠️ [SDAE] Mismatch entre feature_columns({len(self.feature_columns)}) e input_dim({input_dim}). Ajustando."
                        )
                        self.feature_columns = self.feature_columns[:input_dim]
                except Exception as e:
                    logger.warning(f"⚠️ [SDAE] Erro ao ajustar feature_columns: {e}")
            else:
                logger.warning(f"⚠️ [SDAE] Arquivo de feature_columns não encontrado: {self.feature_columns_path}")
            
            if os.path.exists(self.scaler_path):
                try:
                    self.scaler = joblib.load(self.scaler_path)
                    self.scaler_fitted = True
                    logger.info("✅ [SDAE] Scaler carregado.")
                except Exception as e:
                    logger.warning(f"⚠️ [SDAE] Erro ao carregar scaler: {e}")
                    self.scaler = RobustScaler()
                    self.scaler_fitted = False
            else:
                logger.warning(f"⚠️ [SDAE] Scaler não encontrado: {self.scaler_path}")
                self.scaler_fitted = False
            
            if os.path.exists(self.model_path) and self.hyperparams:
                try:
                    self._load_model()
                    logger.info("✅ [SDAE] Modelo carregado com sucesso.")
                except Exception as model_exc:
                    logger.error(f"❌ [SDAE] Erro ao carregar modelo: {model_exc}", exc_info=True)
                    self.autoencoder = None  # Garante que is_trained retorna False
                    logger.warning(f"⚠️ [SDAE] Modelo incompatível ou corrompido: {model_exc}. Necessário retreinar.")
            else:
                missing = []
                if not os.path.exists(self.model_path):
                    missing.append(f"model_path ({self.model_path})")
                if not self.hyperparams:
                    missing.append("hyperparams")
                logger.warning(f"⚠️ [SDAE] Não foi possível carregar modelo. Faltando: {', '.join(missing)}")
                
        except Exception as e:
            logger.error(f"❌ [SDAE] Erro ao carregar estado: {e}", exc_info=True)
            self.autoencoder = None  # Garante que is_trained retorna False
            raise  # Propaga exceção
    
    def _load_model(self):
        """Carrega o modelo treinado."""
        if not self.hyperparams:
            raise ValueError("Hiperparâmetros não carregados. Não é possível criar modelo.")
        
        try:
            # Valida hiperparâmetros obrigatórios
            required_params = ['input_dim', 'cnn_out_channels', 'kernel_size', 'gru_hidden', 
                             'gru_layers', 'latent_dim', 'seq_length', 'dropout']
            missing_params = [p for p in required_params if p not in self.hyperparams]
            if missing_params:
                raise ValueError(f"Hiperparâmetros faltando: {missing_params}")
            
            self.autoencoder = TemporalSDAE(
                input_dim=self.hyperparams['input_dim'],
                cnn_out_channels=self.hyperparams['cnn_out_channels'],
                kernel_size=self.hyperparams['kernel_size'],
                gru_hidden=self.hyperparams['gru_hidden'],
                gru_layers=self.hyperparams['gru_layers'],
                latent_dim=self.hyperparams['latent_dim'],
                seq_length=self.hyperparams['seq_length'],
                dropout=self.hyperparams['dropout'],
                p_mask=self.hyperparams.get('p_mask', 0.25),  # Fallback para modelos antigos
                sigma=self.hyperparams.get('sigma', 0.12),
                num_regimes=self.hyperparams.get('num_regimes', 3),
                regime_emb_dim=self.hyperparams.get('regime_emb_dim', 4)
            )
            
            logger.info(f"🔍 [SDAE] Carregando pesos do modelo de: {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.autoencoder.load_state_dict(state_dict)
            self.autoencoder.to(self.device)
            self.autoencoder.eval()
            logger.info(f"✅ [SDAE] Pesos carregados. Modelo em modo eval no device: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ [SDAE] Falha ao criar/carregar modelo: {e}", exc_info=True)
            self.autoencoder = None
            raise
    
    def _save_state(self):
        """Salva estado atual."""
        try:
            # Salva hiperparâmetros
            with open(self.hyperparams_path, 'w') as f:
                json.dump(self.hyperparams, f, indent=4)
            
            # Salva colunas de features
            with open(self.feature_columns_path, 'w') as f:
                json.dump(self.feature_columns, f, indent=4)
            
            # Salva scaler
            if self.scaler_fitted:
                joblib.dump(self.scaler, self.scaler_path)
            
            logger.info("✅ [SDAE] Estado salvo com sucesso.")
            
        except Exception as e:
            logger.error(f"❌ [SDAE] Erro ao salvar estado: {e}")
    
    def _sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [FIX] Substitui inf/-inf por NaN e preenche NaNs/Infs.
        Garante que RobustScaler e Training não falhem com 'value too large'.
        """
        # Substitui inf/-inf por NaN
        df_clean = df.replace([np.inf, -np.inf], np.nan)
        
        # Check for columns that are ALL NaN (and drop them)
        null_counts = df_clean.isnull().sum()
        all_null_cols = null_counts[null_counts == len(df_clean)].index.tolist()
        if all_null_cols:
            logger.warning(f"⚠️ [SDAE] Dropping {len(all_null_cols)} columns that are ALL NaN/Inf: {all_null_cols[:5]}...")
            df_clean = df_clean.drop(columns=all_null_cols)
        
        # Forward fill para preencher buracos temporais
        df_clean = df_clean.ffill()
        
        # Backward fill para o início
        df_clean = df_clean.bfill()
        
        # Se ainda sobrar NaN (dataset inteiro vazio?), preencher com 0
        if df_clean.isnull().any().any():
            logger.warning("⚠️ [SDAE] Ainda existem NaNs após ffill/bfill. Preenchendo com 0.")
            df_clean = df_clean.fillna(0)
            
        return df_clean

    def _filter_features_for_autoencoder(self, df: pd.DataFrame, feature_columns: List[str]) -> List[str]:
        """
        🔧 FIX: Filtra features que causam problemas no autoencoder.
        
        Remove colunas com valores absolutos muito altos (preços, EMAs, volumes cumulativos)
        que causam loss MSE muito alta mesmo após normalização.
        
        Mantém apenas:
        - Retornos (log returns, pct changes)
        - Indicadores normalizados (RSI, Stoch, ADX, etc.)
        - Features binárias (crosses, patterns)
        - Features z-scored ou percentage-based
        """
        # Prefixos de colunas a EXCLUIR (valores absolutos)
        exclude_prefixes = [
            'open', 'high', 'low', 'close',  # Preços absolutos
            'volume',  # Volume absoluto (não volume_z ou roc_volume)
            'ema_',  # EMAs são preços absolutos
            'bb_upper', 'bb_middle', 'bb_lower',  # Bollinger absolutos
            'vwap', 'psar',  # Preços absolutos
            'obv',  # On-Balance Volume (cumulativo)
            'pvt', 'adl',  # Volume cumulativo
            'macd_line', 'macd_signal',  # Valores em escala de preço
            'atr_1',  # ATR absoluto
            'tp_prior_',  # Regime signals from TrendPredictor/Detector
            'regime_',  # Confidence and other regime metadata
        ]
        
        # Sufixos/palavras a EXCLUIR
        exclude_contains = [
            '_tf_',  # Timeframes absolutos (open_tf_1h, etc.)
            'close_reference',  # Referência de preço
        ]
        
        # Filtrar colunas
        filtered = []
        excluded = []
        
        for col in feature_columns:
            col_lower = col.lower()
            
            # Check prefixes
            should_exclude = False
            for prefix in exclude_prefixes:
                if col_lower.startswith(prefix):
                    # Exceções: manter variantes normalizadas e FRACDIFF
                    if any(x in col_lower for x in ['_pct', '_percentage', '_z', '_ratio', '_roc', 'log_return', '_frac']):
                        should_exclude = False
                        break
                    should_exclude = True
                    break
            
            # Check contains
            if not should_exclude:
                for pattern in exclude_contains:
                    if pattern in col_lower:
                        should_exclude = True
                        break
            
            if should_exclude:
                excluded.append(col)
            else:
                filtered.append(col)
        
        logger.info(f"🔍 [SDAE] Feature filter: kept {len(filtered)}, excluded {len(excluded)} (absolute-value columns)")
        if excluded and len(excluded) <= 20:
            logger.debug(f"   Excluded: {excluded}")
        
        # Garantir mínimo de features
        if len(filtered) < 30:
            logger.warning(f"⚠️ [SDAE] Muito poucas features ({len(filtered)}). Mantendo todas com RobustScaler.")
            return feature_columns
        
        return filtered
    
    # =========================================================================
    # SCIENTIFIC CORRECTIONS: Novos Métodos de Validação
    # =========================================================================
    
    def _calculate_optimal_embargo(self, df: pd.DataFrame) -> int:
        """
        Calcula embargo ótimo baseado em autocorrelação e seq_length.
        
        Referência: De Prado (2018), Section 7.4.3 - "The Embargo Method"
        """
        if self.embargo_bars is not None:
            return int(self.embargo_bars)
        
        seq_based = self.hyperparams.get('seq_length', 64) if self.hyperparams else 64
        
        try:
            autocorr_based = seq_based
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                sample_col = numeric_cols[0]
                clean_series = df[sample_col].dropna()
                
                if len(clean_series) > 50:
                    autocorr_1 = pd.Series(clean_series).autocorr(lag=1)
                    if not np.isnan(autocorr_1) and autocorr_1 > 0:
                        autocorr_1 = min(autocorr_1, 0.99)
                        autocorr_based = max(1, int(np.ceil(1 / (1 - autocorr_1))))
        
        except Exception as e:
            logger.warning(f"⚠️ [SDAE] Erro ao calcular autocorrelação: {e}")
            autocorr_based = seq_based
        
        embargo = max(seq_based, autocorr_based, 10)
        logger.info(f"📊 [SDAE] Embargo calculado: {embargo} barras")
        return embargo
    
    def _validate_bottleneck(self, input_dim: int, latent_dim: int) -> None:
        """Valida que o bottleneck é suficientemente restritivo."""
        compression_ratio = input_dim / latent_dim
        if compression_ratio < self.min_compression_ratio:
            raise ValueError(
                f"❌ [SDAE] Bottleneck muito fraco: {compression_ratio:.2f} < {self.min_compression_ratio}."
            )
        logger.info(f"✅ [SDAE] Bottleneck validado: {compression_ratio:.1f}:1 compression")
    
    def _detect_identity_mapping(self, latent_effective_rank: float, latent_dim: int, input_dim: int) -> bool:
        """Detecta se autoencoder está aprendendo função identidade."""
        utilization_ratio = latent_effective_rank / latent_dim
        compression_vs_input = latent_effective_rank / input_dim
        
        is_identity = False
        
        if utilization_ratio > 0.95:
            logger.warning(f"⚠️ [SDAE] Utilização latente muito alta ({utilization_ratio:.2%}).")
            if compression_vs_input > 0.70:
                logger.error(f"❌ [SDAE] IDENTITY MAPPING DETECTADO!")
                is_identity = True
        elif utilization_ratio < 0.30:
            logger.warning(f"⚠️ [SDAE] Utilização latente baixa ({utilization_ratio:.2%}).")
        else:
            logger.info(f"✅ [SDAE] Utilização latente saudável: {utilization_ratio:.2%}")
        
        return is_identity

    def _compute_latent_pca_diagnostics(self, latent_activations: np.ndarray) -> Dict[str, float]:
        """Analisa espaço latente usando PCA."""
        try:
            pca = PCA()
            pca.fit(latent_activations)
            
            eigenvalues = pca.explained_variance_
            probs = eigenvalues / eigenvalues.sum()
            effective_rank = np.exp(-np.sum(probs * np.log(probs + 1e-12)))
            
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            dims_for_80pct = int(np.argmax(cumsum_var >= 0.80) + 1)
            dims_for_95pct = int(np.argmax(cumsum_var >= 0.95) + 1)
            
            return {
                'effective_rank': float(effective_rank),
                'dims_for_80pct': dims_for_80pct,
                'dims_for_95pct': dims_for_95pct,
                'top_eigenvalue': float(eigenvalues[0]),
                'eigenvalue_ratio': float(eigenvalues[0] / eigenvalues.sum())
            }
        except Exception as e:
            logger.warning(f"⚠️ [SDAE] Erro em PCA diagnostics: {e}")
            return {'effective_rank': 0.0, 'dims_for_80pct': 0, 'dims_for_95pct': 0, 'top_eigenvalue': 0.0, 'eigenvalue_ratio': 0.0}

    def _get_regime_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        🔬 Dr. Tensor: UNIFIED regime detection using CryptoRegimeDetector.
        
        Uses HMM + GMM + ADX + Funding Rate ensemble for accurate regime detection.
        This replaces the old ScientificRegimeDetector (ADX only).
        
        🔧 FIX: Now loads from disk if trained, saves after training.
        """
        try:
            from feature_engineering.crypto_regime_detector import CryptoRegimeDetector, RegimeConfig, create_regime_detector
            import os
            
            # [CACHE] Use cached detector if available
            if self.regime_detector is None:
                model_path = os.path.join(os.getcwd(), "models_ai", "crypto_regime_detector.pkl")
                if os.path.exists(model_path):
                     logger.info(f"🔄 [SDAE] Carregando CryptoRegimeDetector do disco para cache...")
                     self.regime_detector = CryptoRegimeDetector.load(model_path)
            
            if self.regime_detector is not None and self.regime_detector.is_trained:
                # logger.info("✅ [SDAE] CryptoRegimeDetector loaded from cache. Using predict_online...")
                # Use existing trained model for online prediction
                result_df = self.regime_detector.predict(df)  # predict without retraining
            else:
                # 2. Train new detector
                logger.info("📊 [SDAE] Training new CryptoRegimeDetector...")
                # [FIX] confidence_threshold corrigido de 0.75 para 0.55.
                # O threshold 0.75 era inalcançável com os pesos do ensemble,
                # resultando em confidence=0.5 constante e regime labels neutros
                # para todos os candles. Os regime embeddings do CVAE ficavam inúteis.
                config = RegimeConfig(
                    adx_period=9,
                    adx_trend_threshold=20.0,
                    confidence_threshold=0.55,
                    min_regime_duration=8,
                    weight_hmm=0.30,
                    weight_gmm=0.35,
                    weight_adx=0.20,
                    weight_funding=0.15
                )
                
                detector = CryptoRegimeDetector(config)
                result_df = detector.fit_predict(df)
                
                # 3. Save after training
                detector.save(model_path)
            
            logger.info(f"✅ [SDAE] CryptoRegimeDetector: Bull={sum(result_df['regime']==0)}, "
                       f"Bear={sum(result_df['regime']==1)}, Ranger={sum(result_df['regime']==2)}")
            
            return result_df['regime'].values
            
        except ImportError:
            logger.error("❌ [SDAE] crypto_regime_detector not found. Using fallback.")
            return np.full(len(df), 2)  # Fallback: Ranger
        except Exception as e:
            logger.warning(f"⚠️ [SDAE] CryptoRegimeDetector failed: {e}. Using fallback.")
            return np.full(len(df), 2)  # Fallback: Ranger
    
    # =========================================================================
    # CRITICAL VALIDATIONS: OOD and Permutation Tests
    # =========================================================================
    
    def _legacy_validate_ood_reconstruction(self, df: pd.DataFrame, ood_periods: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """
        [LEGACY — substituído pela versão array-based em validate_ood_reconstruction]
        Valida robustez em períodos out-of-distribution.

        Args:
            df: DataFrame completo com dados históricos
            ood_periods: Lista de (start_date, end_date, label) para testar
                         Ex: [('2020-03-01', '2020-04-30', 'COVID_CRASH'),
                              ('2021-11-01', '2021-12-31', 'BULL_PEAK')]

        Returns:
            Dict com erros por período

        Referência: Goodfellow et al. (2016) - Deep Learning, Cap. 5.2
        """
        if not self.is_trained:
            raise ValueError("❌ Modelo não treinado. Execute train_autoencoder_temporal() primeiro.")
        
        logger.info("🔬 [SDAE] Iniciando validação OOD (Out-of-Distribution)...")
        results = {}
        
        # Baseline: erro médio em dados normais (excluindo períodos OOD)
        normal_mask = pd.Series(True, index=df.index)
        for start, end, _ in ood_periods:
            normal_mask &= ~df.index.to_series().between(start, end)
        
        if normal_mask.sum() < self.hyperparams['seq_length']:
            logger.error("❌ Dados normais insuficientes para baseline")
            return {'error': 'insufficient_data'}
        
        baseline_error = self._legacy_compute_reconstruction_error_df(df[normal_mask])
        results['baseline'] = float(baseline_error)
        logger.info(f"📊 Baseline error (dados normais): {baseline_error:.4f}")

        # Testa cada período OOD
        for start, end, label in ood_periods:
            try:
                ood_data = df[df.index.to_series().between(start, end)]

                if len(ood_data) < self.hyperparams['seq_length']:
                    logger.warning(f"⚠️ Período {label} muito curto ({len(ood_data)} < {self.hyperparams['seq_length']}), pulando")
                    continue

                ood_error = self._legacy_compute_reconstruction_error_df(ood_data)
                degradation = (ood_error - baseline_error) / (baseline_error + 1e-8)
                
                acceptable = degradation < 0.5  # < 50% aumento
                
                results[label] = {
                    'error': float(ood_error),
                    'degradation': float(degradation),
                    'acceptable': acceptable
                }
                
                status_icon = "✅" if acceptable else "❌"
                logger.info(f"{status_icon} OOD {label}: error={ood_error:.4f}, degradation={degradation:+.1%}")
                
            except Exception as e:
                logger.error(f"❌ Erro ao testar período {label}: {e}")
                results[label] = {'error': 'failed', 'exception': str(e)}
        
        # Critério de aprovação
        test_results = [r for r in results.values() if isinstance(r, dict) and 'acceptable' in r]
        if not test_results:
            logger.warning("⚠️ Nenhum período OOD válido testado")
            return results
        
        all_acceptable = all(r['acceptable'] for r in test_results)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"VALIDAÇÃO OOD COMPLETA")
        logger.info(f"{'='*70}")
        logger.info(f"Baseline Error:  {baseline_error:.4f}")
        logger.info(f"Períodos testados: {len(test_results)}")
        logger.info(f"Períodos aprovados: {sum(r['acceptable'] for r in test_results)}/{len(test_results)}")
        
        if all_acceptable:
            logger.info("✅ APROVADO: Modelo robusto a períodos OOD")
        else:
            logger.error("❌ REPROVADO: Modelo falha em generalizar para períodos OOD")
        
        logger.info(f"{'='*70}\n")
        
        return results
    
    def _legacy_permutation_test_reconstruction(self, df: pd.DataFrame, n_permutations: int = 50) -> Dict[str, Any]:
        """
        [LEGACY — substituído pela versão array-based em permutation_test_reconstruction]
        Testa se padrões aprendidos são genuínos via bootstrap temporal.

        Método: Permuta blocos temporais e verifica se reconstrução degrada.
        Se modelo aprendeu padrões temporais reais, permutações devem piorar erro.

        Args:
            df: DataFrame com features
            n_permutations: Número de permutações (50-100 recomendado)

        Returns:
            Dict com p-value e interpretação

        Referência: De Prado (2018) - Advances in Financial ML, Cap. 12
        """
        if not self.is_trained:
            raise ValueError("❌ Modelo não treinado. Execute train_autoencoder_temporal() primeiro.")
        
        logger.info(f"🔬 [SDAE] Iniciando teste de permutação ({n_permutations} permutações)...")
        
        # 1. Erro de reconstrução em dados originais
        real_error = self._legacy_compute_reconstruction_error_df(df)
        logger.info(f"📊 Erro real: {real_error:.4f}")

        # 2. Erros em dados permutados
        permuted_errors = []
        block_size = self.hyperparams['seq_length'] * 2  # Permuta blocos de 2x seq_length

        for i in range(n_permutations):
            try:
                # Permuta blocos preservando estrutura local
                df_perm = self._block_permutation(df, block_size=block_size)
                perm_error = self._legacy_compute_reconstruction_error_df(df_perm)
                permuted_errors.append(perm_error)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"   Permutação {i+1}/{n_permutations}: error={perm_error:.4f}")
            
            except Exception as e:
                logger.warning(f"⚠️ Erro na permutação {i+1}: {e}")
                continue
        
        if len(permuted_errors) < 10:
            logger.error("❌ Permutações insuficientes para teste estatístico")
            return {'error': 'insufficient_permutations'}
        
        # 3. Calcula p-value
        # H0: Modelo não capturou padrões temporais (erro em perm == erro real)
        # H1: Modelo capturou padrões (erro em perm > erro real)
        permuted_errors = np.array(permuted_errors)
        p_value = (permuted_errors <= real_error).mean()
        
        # 4. Interpretação
        result = {
            'real_error': float(real_error),
            'perm_error_mean': float(permuted_errors.mean()),
            'perm_error_std': float(permuted_errors.std()),
            'p_value': float(p_value),
            'n_permutations': len(permuted_errors),
            'significant': p_value < 0.05,
            'interpretation': None
        }
        
        if p_value < 0.01:
            result['interpretation'] = "✅ FORTE evidência de padrões genuínos (p<0.01)"
        elif p_value < 0.05:
            result['interpretation'] = "✅ Evidência de padrões genuínos (p<0.05)"
        elif p_value < 0.10:
            result['interpretation'] = "⚠️ Evidência fraca de padrões (p<0.10)"
        else:
            result['interpretation'] = "❌ SEM evidência de padrões genuínos (p>=0.10) - ALERTA!"
        
        logger.info(f"\n{'='*70}")
        logger.info(f"TESTE DE PERMUTAÇÃO")
        logger.info(f"{'='*70}")
        logger.info(f"Erro Real:       {real_error:.4f}")
        logger.info(f"Erro Permutado:  {permuted_errors.mean():.4f} ± {permuted_errors.std():.4f}")
        logger.info(f"P-value:         {p_value:.4f}")
        logger.info(f"N permutações:   {len(permuted_errors)}")
        logger.info(f"Conclusão:       {result['interpretation']}")
        logger.info(f"{'='*70}\n")
        
        return result
    
    def _legacy_compute_reconstruction_error_df(self, df: pd.DataFrame) -> float:
        """
        [LEGACY — substituído pela versão array-based em _compute_reconstruction_error]
        Helper para calcular erro de reconstrução médio.

        Args:
            df: DataFrame com features originais

        Returns:
            Erro médio de reconstrução (MAE)
        """
        try:
            # Aplica autoencoder
            df_with_latents = self.apply_hidden_features_temporal(df)
            
            # Usa erro já calculado
            if 'sdae_recon_error' in df_with_latents.columns:
                errors = df_with_latents['sdae_recon_error'].dropna()
                if len(errors) == 0:
                    logger.warning("⚠️ Nenhum erro de reconstrução disponível")
                    return np.nan
                return float(errors.mean())
            
            logger.warning("⚠️ Coluna 'sdae_recon_error' não encontrada")
            return np.nan
            
        except Exception as e:
            logger.error(f"❌ Erro ao calcular reconstruction error: {e}")
            return np.nan
    
    def _block_permutation(self, df: pd.DataFrame, block_size: int) -> pd.DataFrame:
        """
        Permuta blocos temporais preservando estrutura local.
        
        Args:
            df: DataFrame original
            block_size: Tamanho do bloco a permutar
            
        Returns:
            DataFrame com blocos permutados
        """
        n = len(df)
        n_blocks = n // block_size
        
        # Cria blocos
        blocks = []
        for i in range(n_blocks):
            start = i * block_size
            end = min(start + block_size, n)
            blocks.append(df.iloc[start:end].copy())
        
        # Sobra (se houver)
        if n % block_size != 0:
            blocks.append(df.iloc[n_blocks * block_size:].copy())
        
        # Permuta blocos
        np.random.shuffle(blocks)
        
        # Reconstrói DataFrame
        df_permuted = pd.concat(blocks, ignore_index=True)
        df_permuted.index = df.index  # Mantém índice original (apenas ordem é permutada)
        
        return df_permuted


    def _run_training_loop(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        max_epochs: int,
        patience: int,
        beta_kl: float,
        kl_annealing_epochs: int = 10,
        lambda_sparse: float = 0.0,
        is_trial: bool = False,
        trial: "optuna.trial.Trial | None" = None,
        writer: "SummaryWriter | None" = None,
        clip_grad: float | None = 1.0,
        debug_assert_shapes: bool = False,
    ) -> float:
        """
        Loop DRY de treinamento com suporte a CVAE.
        """
        use_amp = (getattr(self, "device", torch.device("cpu")).type == "cuda")
        scaler = GradScaler(enabled=use_amp)
        recon_criterion = nn.L1Loss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(max_epochs):
            # -------------------- TREINAMENTO --------------------
            model.train()
            total_train_loss = total_recon_loss = total_kl_loss = total_sparse_loss = 0.0

            for batch in train_loader:
                # CVAE: Desempacota batch
                if len(batch) == 2:
                    batch_x, batch_regime = batch
                    batch_regime = batch_regime.to(self.device, non_blocking=True)
                else:
                    batch_x = batch[0]
                    batch_regime = None
                
                batch_x = batch_x.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=use_amp):
                    # CVAE: Passa regime_labels
                    reconstructed_x, mu, logvar, _ = model(batch_x, regime_labels=batch_regime, training=True)

                    if debug_assert_shapes:
                        assert reconstructed_x.shape == batch_x.shape, \
                            f"Decoder alterou shape: got {reconstructed_x.shape}, want {batch_x.shape}"

                    # KL Annealing: Aumenta beta gradualmente para evitar posterior collapse
                    # beta_current = beta_kl * min(1.0, epoch / kl_annealing_epochs)
                    # Implementação Linear
                    if kl_annealing_epochs > 0:
                        anneal_factor = min(1.0, (epoch + 1) / kl_annealing_epochs)
                        current_beta = beta_kl * anneal_factor
                    else:
                        current_beta = beta_kl

                    recon_loss = recon_criterion(reconstructed_x, batch_x)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    l1_latent = torch.mean(torch.abs(mu))
                    loss = recon_loss + current_beta * kl_loss + lambda_sparse * l1_latent

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                grad_total_norm = None
                if clip_grad is not None:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                else:
                    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                    if grads:
                        grad_total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), p=2)

                scaler.step(optimizer)
                scaler.update()

                total_train_loss += float(loss.item())
                total_recon_loss += float(recon_loss.item())
                total_kl_loss += float(kl_loss.item())
                total_sparse_loss += float(l1_latent.item())

            avg_train_loss = total_train_loss / max(1, len(train_loader))
            avg_train_recon = total_recon_loss / max(1, len(train_loader))
            avg_train_kl = total_kl_loss / max(1, len(train_loader))
            avg_train_sparse = total_sparse_loss / max(1, len(train_loader))

            # -------------------- VALIDAÇÃO ---------------------
            avg_val_loss, avg_val_recon, avg_val_kl, diagnostics = self._validate_epoch(
                model=model,
                val_loader=val_loader,
                recon_criterion=recon_criterion,
                beta_kl=beta_kl,
                lambda_sparse=lambda_sparse,
                debug_assert_shapes=debug_assert_shapes,
            )

            # ---------------- LOGGING & SCHEDULER --------------
            if writer:
                writer.add_scalar("Loss/Train_Total", avg_train_loss, epoch)
                writer.add_scalar("Loss/Train_Reconstruction", avg_train_recon, epoch)
                writer.add_scalar("Loss/Train_KL", avg_train_kl, epoch)
                writer.add_scalar("Loss/Validation_Total", avg_val_loss, epoch)
                writer.add_scalar("Loss/Validation_Reconstruction", avg_val_recon, epoch)
                writer.add_scalar("Loss/Validation_KL", avg_val_kl, epoch)
                
                writer.add_scalar("Latent/Effective_Rank", diagnostics['effective_rank'], epoch)
                writer.add_scalar("Latent/Dims_80pct", diagnostics['dims_for_80pct'], epoch)
                
                writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
                if grad_total_norm is not None:
                    writer.add_scalar("Gradients/Norm", float(grad_total_norm), epoch)
                
                # FIXED: Check for identity mapping during training
                if diagnostics.get('effective_rank', 0) > 0:
                    latent_dim = model.latent_dim if hasattr(model, 'latent_dim') else 32
                    input_dim = model.input_dim if hasattr(model, 'input_dim') else 241
                    is_identity = self._detect_identity_mapping(diagnostics['effective_rank'], latent_dim, input_dim)
                    writer.add_scalar("Diagnostics/Identity_Mapping", float(is_identity), epoch)
                    if is_identity and epoch > 10:
                        logger.error(f"❌ [SDAE] Identity mapping detected at epoch {epoch}. Stopping training.")
                        break
                
                writer.flush()
            
            # 📊 [CLEAN LOGGING] Log limpo a cada 5 épocas
            if (epoch + 1) % 5 == 0 or epoch == 0:
                progress = (epoch + 1) / max_epochs * 100
                bar_len = 20
                filled = int(bar_len * (epoch + 1) / max_epochs)
                bar = "█" * filled + "░" * (bar_len - filled)
                
                eff_rank = diagnostics.get('effective_rank', 0)
                sparsity = diagnostics.get('sparsity_ratio', 0)
                
                print(f"[{bar}] {progress:5.1f}% | Epoch {epoch+1:3d}/{max_epochs} | "
                      f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                      f"EffRank: {eff_rank:.1f} | Sparsity: {sparsity:.1%}")

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            if is_trial and trial is not None:
                trial.report(avg_val_loss, step=epoch)
                if trial.should_prune():
                    raise TrialPruned()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                try:
                    torch.save(model.state_dict(), self.model_path)
                except Exception as e:
                    logger.warning(f"[SDAE] Falha ao salvar checkpoint: {e}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"🛑 Early stopping na época {epoch+1} (best_val={best_val_loss:.6f})")
                    break

        try:
            if os.path.exists(self.model_path):
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except Exception as e:
            logger.warning(f"[SDAE] Falha ao carregar melhor checkpoint: {e}")

        return best_val_loss

    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader,
        recon_criterion: nn.Module,
        beta_kl: float,
        lambda_sparse: float = 0.0,
        debug_assert_shapes: bool = False,
    ) -> tuple[float, float, float, dict]:
        """
        Validação com perda variacional (recon + β * KL + λ * L1) + diagnósticos.
        Retorna: (avg_total_loss, avg_recon_loss, avg_kl_loss, error_diagnostics)
        """
        model.eval()
        total_val_loss = total_recon_loss = total_kl_loss = total_sparse_loss = 0.0
        all_sample_errors = []
        all_latent_activations = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    batch_x, batch_regime = batch
                    batch_regime = batch_regime.to(self.device, non_blocking=True)
                else:
                    batch_x = batch[0]
                    batch_regime = None
                
                batch_x = batch_x.to(self.device, non_blocking=True)
                
                # CVAE: Passa regime_labels
                reconstructed_x, mu, logvar, _ = model(batch_x, regime_labels=batch_regime, training=False)

                if debug_assert_shapes:
                    assert reconstructed_x.shape == batch_x.shape, \
                        f"[VAL] Decoder alterou shape: got {reconstructed_x.shape}, want {batch_x.shape}"

                recon_loss = recon_criterion(reconstructed_x, batch_x)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                l1_latent = torch.mean(torch.abs(mu))
                loss = recon_loss + beta_kl * kl_loss + lambda_sparse * l1_latent

                total_val_loss   += float(loss.item())
                total_recon_loss += float(recon_loss.item())
                total_kl_loss    += float(kl_loss.item())
                total_sparse_loss += float(l1_latent.item())
                
                sample_errors = torch.mean(torch.abs(reconstructed_x - batch_x), dim=(1,2))
                all_sample_errors.extend(sample_errors.cpu().numpy().tolist())
                
                all_latent_activations.append(mu.cpu().numpy())

        avg_total = total_val_loss / max(1, len(val_loader))
        avg_recon = total_recon_loss / max(1, len(val_loader))
        avg_kl    = total_kl_loss / max(1, len(val_loader))
        avg_sparse = total_sparse_loss / max(1, len(val_loader))
        
        if all_sample_errors:
            error_stats = {
                'p50': float(np.percentile(all_sample_errors, 50)),
                'p95': float(np.percentile(all_sample_errors, 95)),
                'p99': float(np.percentile(all_sample_errors, 99)),
                'std': float(np.std(all_sample_errors)),
                'all_errors': all_sample_errors
            }
        else:
            error_stats = {'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'std': 0.0, 'all_errors': []}
        
        if all_latent_activations:
            latents = np.concatenate(all_latent_activations, axis=0)
            sparsity_ratio = float((np.abs(latents) < 0.1).mean())
            latent_variance = float(np.var(latents, axis=0).mean())
            pca_diag = self._compute_latent_pca_diagnostics(latents)
        else:
            sparsity_ratio = 0.0
            latent_variance = 0.0
            pca_diag = {
                'effective_rank': 0.0,
                'dims_for_80pct': 0,
                'dims_for_95pct': 0,
            }
        
        diagnostics = {
            'error_stats': error_stats,
            'sparsity_ratio': sparsity_ratio,
            'latent_variance': latent_variance,
            'avg_sparse_loss': avg_sparse,
            'effective_rank': pca_diag['effective_rank'],
            'dims_for_80pct': pca_diag['dims_for_80pct'],
            'dims_for_95pct': pca_diag['dims_for_95pct'],
        }
        
        return avg_total, avg_recon, avg_kl, diagnostics

    def _create_sequences(self, data: np.ndarray, seq_length: int) -> np.ndarray:
        """Cria sequências usando stride tricks para eficiência.
        
        🚀 [MEMORY OPTIMIZATION] Retorna uma VIEW (strided array) sem copiar dados.
        A cópia para o device deve ser feita por batch para evitar OOM.
        """
        if len(data) < seq_length:
            return np.array([], dtype=np.float32)
            
        # Stride tricks para criar view (N-T+1, T, F) sem copiar dados
        shape = (len(data) - seq_length + 1, seq_length, data.shape[1])
        strides = (data.strides[0], data.strides[0], data.strides[1])
        
        # [FIX] Removido .copy() que causava alocação de 8GB+ em datasets longos
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    def optimize_hyperparameters(self, df: pd.DataFrame, n_trials: int = 100) -> Dict[str, Any]:  # OTIMIZADO: Era 20
        """Otimiza hiperparâmetros usando Optuna."""
        logger.info(f"🧪 [SDAE] Iniciando otimização de hiperparâmetros ({n_trials} trials)...")
        
        # CVAE: Gera labels de regime uma vez para todo o processo
        # [FIX] Sanitize BEFORE regime detection and usage
        df_clean = self._sanitize_data(df)
        
        regime_labels = self._get_regime_labels(df_clean)
        
        def objective(trial):
            # Espaço de busca
            input_dim = int(self.hyperparams.get('input_dim', len(self.feature_columns)))
            
            # SCIENTIFIC CORRECTIONS: Bottleneck constraint
            max_latent = input_dim // int(self.min_compression_ratio)
            # OTIMIZADO: Range restrito 20-28 para melhor compression
            max_latent = min(28, input_dim // int(self.min_compression_ratio))  # Cap em 28
            latent_dim = trial.suggest_int('latent_dim', 20, max(20, max_latent))
            
            params = {
                'input_dim': input_dim,
                'cnn_out_channels': trial.suggest_int('cnn_out_channels', 16, 64, step=8),
                'kernel_size': trial.suggest_int('kernel_size', 3, 7, step=2),
                'gru_hidden': trial.suggest_int('gru_hidden', 16, 64, step=8),
                'gru_layers': trial.suggest_int('gru_layers', 1, 2),
                'latent_dim': latent_dim,
                'seq_length': trial.suggest_int('seq_length', 32, 96, step=16),
                'dropout': trial.suggest_float('dropout', 0.1, 0.4),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'p_mask': trial.suggest_float('p_mask', 0.25, 0.40),  # OTIMIZADO: Menos agressivo
                'sigma': trial.suggest_float('sigma', 0.15, 0.25),  # OTIMIZADO: Menos ruído
                'vae_beta': trial.suggest_float('vae_beta', 5e-3, 5e-2, log=True),  # OTIMIZADO: Range mais alto
                'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-3, log=True),
            }
            
            # Treina trial
            base_loss = self._train_trial(df_clean, params, trial, regime_labels)
            
            # MULTI-OBJECTIVE: Adiciona penalties baseadas em métricas secundárias
            penalty = 0.0
            
            # Penalty 1: Compression ratio baixo
            compression_ratio = input_dim / latent_dim
            if compression_ratio < self.min_compression_ratio:
                penalty += (self.min_compression_ratio - compression_ratio) * 0.5
            
            # Penalty 2: Effective rank prediction (estimativa conservadora)
            # Se latent_dim muito alto, penaliza (risco de subutilização)
            if latent_dim > 32:
                penalty += (latent_dim - 32) * 0.02
            
            # Penalty 3: VAE beta muito baixo (risco de collapse)
            if params['vae_beta'] < 0.008:  # 0.008 = threshold
                penalty += (0.008 - params['vae_beta']) * 50.0
            
            # Loss final = base + penalties
            final_loss = base_loss + penalty
            
            # Log para debug
            if trial.number % 10 == 0:
                logger.info(f"   Trial {trial.number}: base_loss={base_loss:.4f}, penalty={penalty:.4f}, final={final_loss:.4f}")
            
            return final_loss

        # 🔧 FIX: Usar SQLite para persistir estudos e permitir retomada
        import os
        optuna_db_path = os.path.join(os.getcwd(), "logs", "autoencoder_optuna_study.db")
        os.makedirs(os.path.dirname(optuna_db_path), exist_ok=True)
        storage_url = f"sqlite:///{optuna_db_path}"
        
        study = optuna.create_study(
            direction='minimize',
            study_name='autoencoder_temporal_v1',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            storage=storage_url,
            load_if_exists=True  # 🔧 Continua de onde parou!
        )
        
        # Calcula quantos trials ainda precisam ser executados
        completed_trials = len([t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]])
        remaining_trials = max(0, n_trials - completed_trials)
        
        if remaining_trials > 0:
            logger.info(f"📊 [SDAE] Optuna: {completed_trials} trials já completos. Executando mais {remaining_trials}...")
            study.optimize(objective, n_trials=remaining_trials)
        else:
            logger.info(f"✅ [SDAE] Optuna: {completed_trials} trials já completos. Nenhum trial adicional necessário.")
        
        
        logger.info("✅ [SDAE] Otimização concluída.")
        logger.info(f"   Melhores params: {study.best_params}")
        logger.info(f"   Melhor loss: {study.best_value:.6f}")
        
        # Atualiza hiperparâmetros com os melhores encontrados
        self.hyperparams.update(study.best_params)
        return study.best_params

    def _train_trial(self, df: pd.DataFrame, params: Dict[str, Any], trial: optuna.trial.Trial, regime_labels: np.ndarray) -> float:
        """Treina um modelo trial e retorna a loss de validação."""
        try:
            # Seleção de colunas
            input_dim = params['input_dim']
            use_cols = self.feature_columns[:input_dim]
            df_selected = df[use_cols]
            
            # SCIENTIFIC CORRECTIONS: Purged CV Split
            if self.use_walk_forward_cv:
                # Usa apenas o último fold para validação rápida no trial
                embargo = self._calculate_optimal_embargo(df)
                cv = TimeSeriesSplitWithEmbargo(n_splits=3, embargo_bars=embargo)
                splits = list(cv.split(df_selected.values))
                train_idx, val_idx = splits[-1]
            else:
                # Split simples temporal (80/20)
                split_idx = int(len(df_selected) * 0.8)
                train_idx = np.arange(split_idx)
                val_idx = np.arange(split_idx, len(df_selected))

            # [FIX LEAKAGE] Scaler fitado apenas no split de treino do trial.
            # Anteriormente era fitado no dataset inteiro na primeira chamada,
            # vazando estatísticas de validação para todos os trials subsequentes.
            trial_scaler = RobustScaler()
            raw_values = df_selected.values
            trial_scaler.fit(raw_values[train_idx])
            numeric_data = trial_scaler.transform(raw_values)
            
            # CVAE: Passa regime_labels. .copy() garante contiguidade se train_idx for não-contíguo.
            train_dataset = LazyTimeSeiresDataset(
                numeric_data[train_idx].copy(), 
                params['seq_length'],
                regime_labels[train_idx].copy()
            )
            val_dataset = LazyTimeSeiresDataset(
                numeric_data[val_idx].copy(), 
                params['seq_length'],
                regime_labels[val_idx].copy()
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=params['batch_size'], 
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=params['batch_size'], 
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            # Cria modelo
            model = TemporalSDAE(
                input_dim=params['input_dim'],
                cnn_out_channels=params['cnn_out_channels'],
                kernel_size=params['kernel_size'],
                gru_hidden=params['gru_hidden'],
                gru_layers=params['gru_layers'],
                latent_dim=params['latent_dim'],
                seq_length=params['seq_length'],
                dropout=params['dropout'],
                p_mask=params['p_mask'],
                sigma=params['sigma'],
                num_regimes=3,           # Fixo conforme arquitetura
                regime_emb_dim=4         # Fixo conforme arquitetura
            ).to(self.device)
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
            
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            
            # Treina
            val_loss = self._run_training_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                max_epochs=20,  # Menos épocas para trials
                patience=5,
                beta_kl=params['vae_beta'],
                lambda_sparse=params['lambda_sparse'],
                is_trial=True,
                trial=trial
            )
            
            return val_loss
            
        except TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"⚠️ [SDAE] Trial falhou: {e}")
            return float('inf')

    def train_autoencoder_temporal(self, df: pd.DataFrame, feature_columns: List[str], optimize: bool = True) -> bool:
        """
        Treina o autoencoder temporal final.
        
        🔧 CORREÇÃO GRANULAR: Verifica se já está treinado antes de re-otimizar.
        """
        try:
            # 🔧 FIX GRANULAR: Se já está treinado e temos modelo válido, pular treino
            if self.autoencoder is not None and os.path.exists(self.model_path):
                logger.info("✅ [SDAE] Autoencoder já treinado. Pulando otimização e treinamento.")
                logger.info(f"   - Modelo: {self.model_path}")
                logger.info(f"   - Hyperparams: latent_dim={self.hyperparams.get('latent_dim', '?')}, input_dim={self.hyperparams.get('input_dim', '?')}")
                return True
            
            # Verifica se temos hiperparâmetros salvos (de otimização anterior)
            if os.path.exists(self.hyperparams_path) and os.path.exists(self.model_path):
                logger.info("🔄 [SDAE] Hiperparâmetros e modelo existentes. Carregando ao invés de re-treinar...")
                try:
                    self.load_state()
                    if self.autoencoder is not None:
                        logger.info("✅ [SDAE] Modelo carregado com sucesso. Treinamento não necessário.")
                        return True
                except Exception as load_err:
                    logger.warning(f"⚠️ [SDAE] Falha ao carregar modelo existente: {load_err}. Prosseguindo com treino.")
            
            logger.info("🚀 [SDAE] Iniciando treinamento do autoencoder temporal...")
            
            # [FIX] Sanitize data first
            df = self._sanitize_data(df)

            # 🔧 FIX: Filtrar colunas que têm valores absolutos muito altos
            # (preços, EMAs, volumes cumulativos) - estas causam loss muito alta
            filtered_columns = self._filter_features_for_autoencoder(df, feature_columns)
            logger.info(f"📊 [SDAE] Features após filtragem: {len(filtered_columns)}/{len(feature_columns)}")
            
            # Define colunas de features (filtradas)
            self.feature_columns = filtered_columns
            
            # Otimiza ou carrega hiperparâmetros padrão
            if optimize or not self.hyperparams:
                # 🔧 FIX: Usar número de colunas FILTRADAS, não originais
                self.hyperparams['input_dim'] = len(filtered_columns)
                self.optimize_hyperparameters(df)
            
            # Treina modelo final
            success = self._train_final_model(df)
            
            if success:
                self._save_state()
                
            return success
            
        except Exception as e:
            logger.error(f"❌ [SDAE] Erro fatal no treinamento: {e}", exc_info=True)
            return False

    def _train_final_model(self, df: pd.DataFrame) -> bool:
        """Treina o modelo final com os melhores hiperparâmetros."""
        try:
            # Prepara dados
            input_dim = int(self.hyperparams['input_dim'])
            use_cols = self.feature_columns[:input_dim]
            df_selected = df[use_cols]
            
            # CVAE: Gera labels de regime
            regime_labels = self._get_regime_labels(df)
            
            # [FIX LEAKAGE] Split ANTES do fit do scaler.
            # Anteriormente o scaler era fitado no dataset inteiro (incluindo validação),
            # vazando estatísticas do futuro para o treino.
            split_idx = int(len(df_selected) * 0.9)
            
            train_raw = df_selected.values[:split_idx]
            val_raw   = df_selected.values[split_idx:]
            
            # Fit APENAS no treino (sem ver validação)
            self.scaler.fit(train_raw)
            self.scaler_fitted = True
            
            train_data = self.scaler.transform(train_raw)
            val_data   = self.scaler.transform(val_raw)
            
            train_regime = regime_labels[:split_idx]
            val_regime = regime_labels[split_idx:]
            
            # Datasets
            train_dataset = LazyTimeSeiresDataset(
                train_data, 
                self.hyperparams['seq_length'],
                train_regime
            )
            val_dataset = LazyTimeSeiresDataset(
                val_data, 
                self.hyperparams['seq_length'],
                val_regime
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.hyperparams['batch_size'], 
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.hyperparams['batch_size'], 
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            # Modelo final
            self.autoencoder = TemporalSDAE(
                input_dim=self.hyperparams['input_dim'],
                cnn_out_channels=self.hyperparams['cnn_out_channels'],
                kernel_size=self.hyperparams['kernel_size'],
                gru_hidden=self.hyperparams['gru_hidden'],
                gru_layers=self.hyperparams['gru_layers'],
                latent_dim=self.hyperparams['latent_dim'],
                seq_length=self.hyperparams['seq_length'],
                dropout=self.hyperparams['dropout'],
                p_mask=self.hyperparams.get('p_mask', 0.25),
                sigma=self.hyperparams.get('sigma', 0.12),
                num_regimes=3,
                regime_emb_dim=4
            ).to(self.device)
            
            # Garante que parâmetros do CVAE sejam persistidos
            self.hyperparams['num_regimes'] = 3
            self.hyperparams['regime_emb_dim'] = 4
            
            # Otimizador e scheduler
            optimizer = optim.AdamW(
                self.autoencoder.parameters(),
                lr=self.hyperparams['learning_rate'],
                weight_decay=self.hyperparams['weight_decay']
            )
            
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
            
            # TensorBoard
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            log_dir = f"logs/tensorboard/sdae/{timestamp}"
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
            
            beta_kl = self.hyperparams.get('vae_beta', 1e-3)
            lambda_sparse = self.hyperparams.get('lambda_sparse', 0.0)

            # Treina
            best_val_loss = self._run_training_loop(
                model=self.autoencoder,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                max_epochs=50,
                patience=20,
                beta_kl=beta_kl,
                lambda_sparse=lambda_sparse,
                is_trial=False,
                trial=None,
                writer=writer,
                clip_grad=1.0,
                debug_assert_shapes=False,
            )
            
            writer.close()
            
            # Salva colunas usadas
            self.feature_columns = df_selected.columns.tolist()
            
            logger.info(f"✅ [SDAE] Modelo final treinado. Loss final: {best_val_loss:.6f}")
            logger.info(f"📊 [SDAE] TensorBoard logs salvos em: {log_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [SDAE] Erro no treinamento final: {e}", exc_info=True)
            return False

    def apply_hidden_features_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [MELHORIA] Aplica features latentes do SDAE temporal ao DataFrame.
        Também anexa informações de regime (confiança, classe) para uso pelos agentes.
        """
        if self.autoencoder is None:
            raise ValueError("Autoencoder não treinado. Execute train_autoencoder_temporal() primeiro.")
        
        if not self.feature_columns:
            raise ValueError("Colunas de features não definidas.")
        
        try:
            logger.debug("🔍 [SDAE] Aplicando features latentes temporais...")
            
            # [FIX] Sanitize data first (handle infs/nans)
            df = self._sanitize_data(df)
            
            # Verifica colunas
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Colunas ausentes: {missing_cols}")
            
            input_dim = int(self.hyperparams.get('input_dim', len(self.feature_columns)))
            use_cols = self.feature_columns[:input_dim]
            df_selected = df[use_cols]
            
            # --- REGIME DETECTION INTEGRATION ---
            # Explicitly load detector to get full details (confidence, probs), not just labels
            from feature_engineering.crypto_regime_detector import CryptoRegimeDetector
            import os
            
            regime_df = None
            
            # [CACHE] Use cached detector
            if self.regime_detector is None:
                 model_path = os.path.join(os.getcwd(), "models_ai", "crypto_regime_detector.pkl")
                 if os.path.exists(model_path):
                     try:
                         logger.info(f"🔄 [SDAE] Carregando CryptoRegimeDetector do disco para cache (apply_hidden)...")
                         self.regime_detector = CryptoRegimeDetector.load(model_path)
                     except Exception as e:
                         logger.warning(f"⚠️ [SDAE] Failed to load detector to cache: {e}")

            if self.regime_detector and self.regime_detector.is_trained:
                try:
                    regime_df = self.regime_detector.predict(df)
                    # logger.info("✅ [SDAE] Regime details loaded from CryptoRegimeDetector (Cached).")
                except Exception as e:
                    logger.warning(f"⚠️ [SDAE] Failed to predict regimes: {e}")

            if regime_df is not None:
                regime_labels_full = regime_df['regime'].values
            else:
                # Fallback: Call _get_regime_labels (which might train/load if it can) or default to Ranger
                # Since we tried loading above, likely _get_regime_labels will also fail or re-train.
                # Let's rely on _get_regime_labels as a backup for the *labels* needed for CVAE
                regime_labels_full = self._get_regime_labels(df)
            
            # Scaling
            # [FIX DIMENSIONAL MISMATCH] Force df_selected to match scaler's input dimension
            # Scikit-learn scalers store 'n_features_in_' which must match the input to transform()
            if self.scaler_fitted and hasattr(self.scaler, 'n_features_in_'):
                expected_dim = self.scaler.n_features_in_
                if df_selected.shape[1] > expected_dim:
                    logger.warning(
                        f"🛡️ [SDAE] Feature Pipeline ballooned to {df_selected.shape[1]}. "
                        f"Slicing the data to {expected_dim} to match the trained Scaler."
                    )
                    df_selected = df_selected.iloc[:, :expected_dim]
                elif df_selected.shape[1] < expected_dim:
                    logger.error(
                        f"❌ [SDAE] Feature Pipeline provided fewer columns than the Scaler expects "
                        f"({df_selected.shape[1]} < {expected_dim}). Autoencoder will likely crash."
                    )

            if not self.scaler_fitted:
                logger.warning("⚠️ [SDAE] Scaler not fitted! Using raw data.")
                numeric_data = df_selected.values
            else:
                # [FIX 266 vs 252] This is the exact line where the mismatch usually crashes
                numeric_data = self.scaler.transform(df_selected.values)
            
            # Cria sequências
            seq_length = self.hyperparams['seq_length']
            original_len = len(numeric_data)
            pad_applied  = 0

            if original_len < seq_length:
                # [FIX] Janela de inferência curta (ex: live trading com 40 linhas).
                # Solução: replica a primeira linha para preencher o início até seq_length.
                # O latent gerado representa o último ponto disponível — válido para decisão.
                pad_rows    = seq_length - original_len
                padding     = np.tile(numeric_data[0:1, :], (pad_rows, 1))
                numeric_data = np.vstack([padding, numeric_data])
                pad_applied  = pad_rows
                logger.warning(
                    f"⚠️ [SDAE] Janela curta ({original_len} rows < seq_length={seq_length}). "
                    f"Padding de {pad_rows} linhas aplicado para inferência."
                )

            sequences = self._create_sequences(numeric_data, seq_length)
            if len(sequences) == 0:
                raise ValueError("Sequências vazias após padding")
            
            # Gera features latentes
            latents: List[np.ndarray] = []
            logvars: List[np.ndarray] = []
            recon_errors: List[np.ndarray] = []
            self.autoencoder.to(self.device)
            self.autoencoder.eval()
            
            with torch.no_grad():
                batch_size = 128
                for i in range(0, len(sequences), batch_size):
                    batch_sequences = sequences[i:i + batch_size]
                    batch_tensor = torch.FloatTensor(batch_sequences).to(self.device)
                    
                    # CVAE: Pega regimes correspondentes ao final de cada janela
                    batch_regime_indices = np.arange(i, i + len(batch_sequences)) + seq_length - 1
                    batch_regime_indices = np.clip(batch_regime_indices, 0, len(regime_labels_full) - 1)
                    
                    batch_regimes = torch.LongTensor(regime_labels_full[batch_regime_indices]).to(self.device)
                    
                    batch_reconstructed, batch_mu, batch_logvar, _ = self.autoencoder(
                        batch_tensor, regime_labels=batch_regimes, training=False
                    )
                    
                    latents.append(batch_mu.detach().cpu().numpy())
                    logvars.append(batch_logvar.detach().cpu().numpy())
                    
                    batch_reconstructed_np = batch_reconstructed.detach().cpu().numpy()
                    batch_error = np.mean(
                        (batch_reconstructed_np[:, -1, :] - batch_sequences[:, -1, :]) ** 2,
                        axis=1,
                    )
                    recon_errors.append(batch_error)
            
            # Concatena
            all_latents = np.vstack(latents)
            all_logvars = np.vstack(logvars) if logvars else np.zeros_like(all_latents)
            all_recon_errors = np.concatenate(recon_errors) if recon_errors else np.zeros(len(all_latents), dtype=np.float32)
            hidden_uncertainty = np.exp(0.5 * all_logvars).mean(axis=1).astype(np.float32)
            
            # Cria DataFrame
            latent_columns = [f'hidden_feature_{i}' for i in range(all_latents.shape[1])]
            # [FIX PADDING] Com padding aplicado, o índice inicial desloca para trás:
            # Sem padding: latent_start_idx = seq_length - 1
            # Com padding: latent_start_idx = seq_length - 1 - pad_applied = original_len - 1
            latent_start_idx = seq_length - 1 - pad_applied
            latent_end_idx   = len(df)

            if latent_start_idx >= len(df):
                raise ValueError(f"Sequência muito longa ({seq_length}) para DataFrame pequeno ({len(df)})")
            
            latent_df = pd.DataFrame(
                all_latents,
                columns=latent_columns,
                index=df.index[latent_start_idx:latent_end_idx]
            )
            recon_error_series = pd.Series(
                all_recon_errors,
                index=df.index[latent_start_idx:latent_end_idx],
                name='sdae_recon_error',
            )
            uncertainty_series = pd.Series(
                hidden_uncertainty,
                index=df.index[latent_start_idx:latent_end_idx],
                name='hidden_uncertainty',
            )
            
            # Merge
            result_df = df.copy()
            result_df = result_df.join(latent_df, how='left')
            result_df = result_df.join(recon_error_series, how='left')
            result_df = result_df.join(uncertainty_series, how='left')

            # Append Regime Details if available
            if regime_df is not None:
                # Ensure index alignment
                regime_df_aligned = regime_df.reindex(result_df.index)
                result_df['regime_confidence'] = regime_df_aligned['confidence'].fillna(0.0)
                result_df['regime_val'] = regime_df_aligned['regime'].fillna(2).astype(int) # 2=Ranger default
                
                # Add friendly columns for downstream mapping
                # Assuming 0=Bull, 1=Bear, 2=Ranger from CryptoRegimeDetector
                result_df['regime_is_bull'] = (result_df['regime_val'] == 0).astype(int)
                result_df['regime_is_bear'] = (result_df['regime_val'] == 1).astype(int)
                result_df['regime_is_ranger'] = (result_df['regime_val'] == 2).astype(int)
                
                # FIX: Alias to 'regime' as expected by AIController
                result_df['regime'] = result_df['regime_val']
            else:
                 # Fallback: use labels found earlier
                labels_aligned = regime_labels_full[latent_start_idx:latent_end_idx] # Indices shifted by seq_length
                # Padding at start
                labels_padded = np.pad(labels_aligned, (latent_start_idx, 0), mode='edge')
                result_df['regime_val'] = labels_padded[:len(result_df)]
                result_df['regime_confidence'] = 0.5 # Default
                
                # FIX: Alias to 'regime' as expected by AIController
                result_df['regime'] = result_df['regime_val']

            
            # [FIX AGRESSIVO] Forward fill para garantir que a última linha tenha dados
            # Se o AE não conseguiu processar o candle atual (ex: delay de dados), 
            # usamos o estado latente imediatamente anterior.
            result_df[latent_columns] = result_df[latent_columns].ffill().bfill()
            result_df['sdae_recon_error'] = result_df['sdae_recon_error'].ffill().bfill().fillna(0.0)
            result_df['hidden_uncertainty'] = result_df['hidden_uncertainty'].ffill().bfill().fillna(0.0)
            
            # Fill regime cols if they were added
            if 'regime_confidence' in result_df.columns:
                result_df['regime_confidence'] = result_df['regime_confidence'].ffill().fillna(0.5) 
                result_df['regime_val'] = result_df['regime_val'].ffill().fillna(2)
                if 'regime' in result_df.columns:
                     result_df['regime'] = result_df['regime_val']

            # Log de Verificação Final para a última linha (Crítico para XAI)
            last_latents = result_df[latent_columns].iloc[-1]
            nan_count = last_latents.isna().sum()
            zero_count = (last_latents == 0.0).sum()
            if nan_count > 0 or zero_count == len(latent_columns):
                logger.warning(f"⚠️ [SDAE] Atenção: Última linha contém {nan_count} NaNs e {zero_count} Zeros nas Hidden Features.")
            else:
                sample_val = last_latents.values[:3].tolist()
                logger.info(f"✅ [SDAE] Latentes prontas para a decisão: {len(latent_columns)} cols | Amostra: {sample_val}")

            # DEBUG
            if 'regime' not in result_df.columns:
                logger.error(f"🚨 [SDAE DEBUG] 'regime' column MISSING in result_df! Columns: {result_df.columns.tolist()}")
            else:
                 # logger.debug("[SDAE DEBUG] 'regime' column present.")
                 pass

            logger.debug(f"✅ [SDAE] Features latentes aplicadas: {len(latent_columns)} colunas")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ [SDAE] Erro ao aplicar features latentes: {e}", exc_info=True)
            raise

    @property
    def is_trained(self) -> bool:
        """Retorna True se o modelo está treinado."""
        return self.autoencoder is not None and hasattr(self.autoencoder, 'state_dict')

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo treinado."""
        if not self.autoencoder:
            return {"status": "não treinado"}
        
        return {
            "status": "treinado",
            "hyperparams": self.hyperparams,
            "feature_columns": self.feature_columns,
            "model_path": self.model_path,
            "device": str(self.device)
        }
    
    def sanity_check(self, test_input_dim: int = 241, test_seq_length: int = 64) -> Dict[str, Any]:
        """Checks rápidos (antes de colocar para treinar longo)"""
        try:
            logger.info("🧪 [SDAE] Executando sanity check...")
            
            if not self.autoencoder:
                return {"status": "erro", "message": "Modelo não treinado"}
            
            results = {}
            test_dims = [test_input_dim, test_input_dim + 1, test_input_dim - 1]
            test_lengths = [test_seq_length, test_seq_length // 2, test_seq_length * 2]
            
            for input_dim in test_dims:
                for seq_length in test_lengths:
                    if seq_length < 10: continue
                        
                    batch_size = 8
                    x = torch.randn(batch_size, seq_length, input_dim).to(self.device)
                    
                    try:
                        reconstructed, mu, logvar, _ = self.autoencoder(x, training=False)
                        shape_preserved = (reconstructed.shape == x.shape)
                        results[f"shape_{input_dim}x{seq_length}"] = {
                            "input_shape": list(x.shape),
                            "output_shape": list(reconstructed.shape),
                            "preserved": shape_preserved
                        }
                    except Exception as e:
                        results[f"shape_{input_dim}x{seq_length}"] = {"error": str(e)}
            
            logger.info("✅ [SDAE] Sanity check concluído")
            return {"status": "sucesso", "results": results}
            
        except Exception as e:
            logger.error(f"❌ [SDAE] Erro no sanity check: {e}", exc_info=True)
            return {"status": "erro", "message": str(e)}
    
    def analyze_latent_semantics(self, df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Analisa a semântica dos latentes correlacionando com indicadores conhecidos."""
        if self.autoencoder is None: return {}
        
        try:
            logger.info("🔍 [SDAE] Analisando semântica dos latentes...")
            
            reference_indicators = {
                'momentum': ['rsi', 'macd_line', 'macd_hist', 'roc', 'stoch_k'],
                'volatility': ['atr', 'atr_percentage', 'bb_width'],
                'trend': ['ema_trend', 'adx', 'ema_fast_slope', 'ema_slow_slope'],
                'volume': ['volume', 'obv', 'vwap'],
            }
            
            all_indicators = []
            for category, indicators in reference_indicators.items():
                all_indicators.extend(indicators)
            
            available_indicators = [ind for ind in all_indicators if ind in df.columns]
            if not available_indicators: return {}
            
            latent_dim = int(self.hyperparams.get('latent_dim', 32))
            semantics = {}
            
            for i in range(latent_dim):
                latent_col = f'hidden_feature_{i}'
                if latent_col not in df.columns: continue
                
                correlations = {}
                for indicator in available_indicators:
                    try:
                        valid_mask = df[[latent_col, indicator]].notna().all(axis=1)
                        if valid_mask.sum() < 10: continue
                        
                        corr = df.loc[valid_mask, latent_col].corr(df.loc[valid_mask, indicator])
                        if not np.isnan(corr):
                            correlations[indicator] = float(corr)
                    except: continue
                
                if not correlations: continue
                
                strongest_indicator = max(correlations.items(), key=lambda x: abs(x[1]))
                indicator_name, correlation_value = strongest_indicator
                
                category = 'unknown'
                for cat, indicators in reference_indicators.items():
                    if indicator_name in indicators:
                        category = cat
                        break
                
                semantics[i] = {
                    'name': f"{category}_proxy_{i}",
                    'indicator': indicator_name,
                    'correlation': correlation_value,
                    'category': category
                }
            
            semantics_path = os.path.join(self.model_dir, "latent_semantics.json")
            with open(semantics_path, 'w') as f:
                json.dump(semantics, f, indent=4)
            
            return semantics
            
        except Exception as e:
            logger.error(f"❌ [SDAE] Erro ao analisar semântica: {e}")
            return {}

    def _compute_adaptive_sequence_length(self, df: pd.DataFrame, base_seq_length: Optional[int] = None) -> np.ndarray:
        """Calcula seq_length adaptativo baseado em regime de volatilidade."""
        if base_seq_length is None:
            base_seq_length = int(self.hyperparams.get('seq_length', 64))
        
        try:
            if 'atr_percentage' not in df.columns:
                return np.full(len(df), base_seq_length, dtype=np.int32)
            
            atr_pct = pd.to_numeric(df['atr_percentage'], errors='coerce')
            atr_ma = atr_pct.rolling(window=20, min_periods=1).mean()
            
            q25 = float(atr_ma.quantile(0.25))
            q75 = float(atr_ma.quantile(0.75))
            
            short_seq = max(16, base_seq_length // 2)
            long_seq = min(128, int(base_seq_length * 1.5))
            
            seq_lengths = np.where(
                atr_ma > q75, short_seq,
                np.where(atr_ma < q25, long_seq, base_seq_length)
            ).astype(np.int32)
            
            return seq_lengths
            
        except Exception:
            return np.full(len(df), base_seq_length, dtype=np.int32)
    
    def apply_hidden_features_temporal_adaptive(self, df: pd.DataFrame, use_adaptive_seq_length: bool = True) -> pd.DataFrame:
        """Versão adaptativa de apply_hidden_features_temporal (Experimental)."""
        logger.info("⚠️ [SDAE] Modo adaptativo ainda experimental. Usando modo padrão.")
        return self.apply_hidden_features_temporal(df)
    
    # =========================================================================
    # SCIENTIFIC VALIDATION METHODS
    # =========================================================================
    
    def validate_ood_reconstruction(
        self, 
        df: pd.DataFrame, 
        ood_periods: List[tuple]
    ) -> Dict[str, Any]:
        """
        🔬 Valida reconstrução em períodos Out-of-Distribution (crashes, rallies).
        
        Referência: Hendrycks & Gimpel (2017) - "A Baseline for Detecting Misclassified 
        and Out-of-Distribution Examples"
        
        Args:
            df: DataFrame com features
            ood_periods: Lista de (start_date, end_date, label)
            
        Returns:
            Dicionário com erros por período e se são aceitáveis
        """
        if self.autoencoder is None:
            logger.warning("⚠️ [SDAE] Modelo não treinado para validação OOD")
            return {'error': 'Model not trained'}
        
        try:
            import torch
            from sklearn.preprocessing import RobustScaler
            
            # Usa as colunas do modelo
            use_cols = self.feature_columns[:self.hyperparams.get('input_dim', len(self.feature_columns))]
            available_cols = [c for c in use_cols if c in df.columns]
            
            # [FIX] Usa o scaler treinado em vez de criar um novo.
            # [SCIENTIFIC FIX] ffill/bfill antes de zero — preserva propriedades
            # estatísticas da feature em vez de injetar "zero" como observação real.
            df_numeric = df[available_cols].copy().ffill().bfill().fillna(0)
            if self.scaler_fitted:
                scaled_data = self.scaler.transform(df_numeric.values)
            else:
                logger.warning("⚠️ [SDAE] Scaler não fitado. OOD usará fit_transform (apenas fallback).")
                scaled_data = RobustScaler().fit_transform(df_numeric.values)
            
            # Baseline: últimos 30% dos dados (excluindo OOD periods)
            baseline_start = int(len(scaled_data) * 0.7)
            baseline_error = self._compute_reconstruction_error(scaled_data[baseline_start:])
            
            results = {'baseline': float(baseline_error) if not np.isnan(baseline_error) else 0.0}
            
            # Testa cada período OOD
            for start_date, end_date, label in ood_periods:
                try:
                    period_mask = (df.index >= start_date) & (df.index <= end_date)
                    period_data = scaled_data[period_mask.values]
                    
                    ood_error = self._compute_reconstruction_error(period_data)
                    
                    # Degradação aceitável: até 50% do baseline
                    degradation = (ood_error / baseline_error - 1) if baseline_error > 0 else 0
                    acceptable = degradation < 0.5  # Max 50% degradação
                    
                    results[label] = {
                        'error': float(ood_error) if not np.isnan(ood_error) else None,
                        'degradation': float(degradation),
                        'acceptable': acceptable
                    }
                except Exception as e:
                    results[label] = {'error': None, 'acceptable': False, 'exception': str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"❌ [SDAE] Erro na validação OOD: {e}")
            return {'error': str(e)}

    def _compute_reconstruction_error(self, data: np.ndarray) -> float:
        """Helper para calcular erro de reconstrução com proteção de memória."""
        import torch
        seq_length = self.hyperparams.get('sequence_length', 32)
        if len(data) < seq_length:
            return np.nan
            
        self.autoencoder.eval()
        errors = []
        with torch.no_grad():
            batch_size = 128
            # Usa stride para não processar cada milisegundo (performance)
            stride = 5
            for i in range(0, len(data) - seq_length + 1, batch_size * stride):
                batch_data = []
                for j in range(i, min(i + batch_size * stride, len(data) - seq_length + 1), stride):
                    batch_data.append(data[j : j + seq_length])
                
                if not batch_data: continue
                batch_tensor = torch.FloatTensor(np.array(batch_data)).to(self.device)
                recon, _, _, _ = self.autoencoder(batch_tensor, regime_labels=None, training=False)
                err = torch.mean((recon - batch_tensor) ** 2, dim=(1, 2))
                errors.extend(err.cpu().numpy())
        
        return float(np.mean(errors)) if errors else np.nan

    def permutation_test_reconstruction(
        self, 
        df: pd.DataFrame, 
        n_permutations: int = 50
    ) -> Dict[str, Any]:
        """
        🔬 Teste de permutação para verificar se modelo aprendeu padrões genuínos.
        """
        if self.autoencoder is None:
            return {'error': 'Model not trained', 'p_value': 1.0}
        
        try:
            import torch
            from sklearn.preprocessing import RobustScaler
            
            use_cols = self.feature_columns[:self.hyperparams.get('input_dim', len(self.feature_columns))]
            available_cols = [c for c in use_cols if c in df.columns]
            
            # [SCIENTIFIC FIX] ffill/bfill antes de zero (mesma justificativa acima).
            df_numeric = df[available_cols].copy().ffill().bfill().fillna(0)

            if self.scaler_fitted:
                scaled_data = self.scaler.transform(df_numeric.values)
            else:
                scaled_data = RobustScaler().fit_transform(df_numeric.values)
            
            # Testa nos últimos 20%
            test_start = int(len(scaled_data) * 0.8)
            test_data = scaled_data[test_start:]
            
            # Erro real
            real_error = self._compute_reconstruction_error(test_data)
            
            # Erros permutados
            perm_errors = []
            for i in range(n_permutations):
                perm_data = test_data.copy()
                np.random.shuffle(perm_data)
                err = self._compute_reconstruction_error(perm_data)
                if not np.isnan(err):
                    perm_errors.append(err)
            
            if not perm_errors:
                return {'error': 'No valid permutation errors', 'p_value': 1.0}
            
            p_value = float(np.mean([1 if pe <= real_error else 0 for pe in perm_errors]))
            
            return {
                'real_error': float(real_error),
                'perm_error_mean': float(np.mean(perm_errors)),
                'p_value': p_value,
                'interpretation': "✅ PADRÕES DETECTADOS" if p_value < 0.05 else "❌ SEM PADRÕES TEMPORAIS"
            }
            
        except Exception as e:
            logger.error(f"❌ [SDAE] Erro no teste de permutação: {e}")
            return {'error': str(e), 'p_value': 1.0}

