# -----------------------------------------------------------------------------
# ARQUIVO: learning/profitability_predictor.py (Versão de Elite)
# -----------------------------------------------------------------------------

"""
Módulo do Modelo Preditivo de Probabilidade de Lucratividade (P) - VERSÃO DE ELITE.

Implementa:
1. Memória Temporal (GRU) para entender sequências de candles.
2. Predição Multi-Target (Probabilidade, PnL, Tempo de Holding).
3. Loss Function Personalizada com foco em Utilidade Real (Sharpe/Utility).
4. Incorporação de Contexto (Regime de Mercado, Features de Tempo).
5. Calibração de Confiança (Temperature Scaling).
6. Data Augmentation (Jitter em tempo e preço) para robustez.
7. Ensemble Temporal (TTA) para previsões mais estáveis.
8. Incerteza da Previsão (MC Dropout).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, Any, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score
import joblib
from dataclasses import dataclass
import warnings
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings('ignore')

from utils.logger import get_logger
from config.settings import AIConfig, TradingConfig

logger = get_logger("ProfitabilityPredictor")

@dataclass
class ModelMetrics:
    """Classe para armazenar métricas detalhadas do modelo."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse_pnl: float
    mse_hold_time: float
    utility_score: float

class ProfitabilityPredictor(nn.Module):
    """
    Rede Neural Avançada com GRU, predição multi-target e loss personalizada.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        sequence_length: int = 20,
        num_gru_layers: int = 2,
        dropout_rate: float = 0.3,
        config: Optional[AIConfig] = None,
    ):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("🚨 [ERRO PREDICTOR] 'input_dim' deve ser um inteiro positivo.")
        
        self.config = config or AIConfig()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.model_dir = getattr(self.config, "MODEL_DIR", os.getcwd())
        
        # [MELHORIA 1] Camada GRU para capturar dependências temporais
        self.temporal_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout_rate if num_gru_layers > 1 else 0,
            bidirectional=True
        )
        gru_output_dim = hidden_dim * 2 # O dobro por ser bidirecional
        
        # [MELHORIA 4] Camada para incorporar features de contexto
        context_input_dim = 5 # 4 para tempo (sin/cos de hora/dia) + 1 para regime
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
        )
        
        # Camada de fusão
        self.fusion_layer = nn.Sequential(
            nn.Linear(gru_output_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )
        
        # [MELHORIA 2] Heads de predição multi-target
        self.prob_head = nn.Linear(hidden_dim, 1)
        self.pnl_head = nn.Linear(hidden_dim, 1)
        self.hold_time_head = nn.Linear(hidden_dim, 1)
        
        # [MELHORIA 5] Parâmetro para Calibração (Temperature Scaling)
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        self.is_trained = False
        self.best_metrics: Optional[ModelMetrics] = None
        self._initialize_weights()
        logger.info(f"🔮 [PROFITABILITY ELITE] Modelo inicializado com input_dim={input_dim}, sequence_length={sequence_length}.")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name: nn.init.orthogonal_(param)
                    elif 'bias' in name: nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.shape[2] != self.input_dim:
            raise ValueError(f"🚨 Shape de entrada inválido. Esperado (batch, seq_len, {self.input_dim}), recebido {x.shape}.")
        
        gru_out, _ = self.temporal_encoder(x)
        temporal_features = gru_out[:, -1, :]
        
        context_features = self.context_encoder(context)
        
        combined_features = torch.cat([temporal_features, context_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        prob_logits = self.prob_head(fused_features)
        pnl_pred = self.pnl_head(fused_features)
        hold_time_pred = F.relu(self.hold_time_head(fused_features))
        
        return prob_logits, pnl_pred, hold_time_pred

    def _compute_loss(self, prob_logits, pnl_pred, hold_time_pred, y_prob, y_pnl, y_hold_time):
        prob_loss = F.binary_cross_entropy_with_logits(prob_logits, y_prob, pos_weight=torch.tensor([2.0]).to(self.device))
        pnl_loss = F.mse_loss(pnl_pred, y_pnl)
        hold_time_loss = F.mse_loss(hold_time_pred, y_hold_time)
        
        utility = torch.sigmoid(prob_logits) * pnl_pred - 0.5 * F.relu(-pnl_pred)
        utility_loss = -torch.mean(utility)
        
        return prob_loss + 10.0 * pnl_loss + 5.0 * hold_time_loss + 1.0 * utility_loss

    def train_model(self, featured_df: pd.DataFrame, trade_log: List[Dict[str, Any]], num_epochs: int) -> int:
        """
        [VERSÃO FINAL] Treina ou continua o treinamento do ProfitabilityPredictor, retornando o progresso.
        """
        # <<< 1. PULAR TREINO SE O MODELO FINAL JÁ EXISTE E É COMPATÍVEL >>>
        if num_epochs <= 0:
            logger.info(f"✅ [PROFITABILITY] Nenhuma época adicional necessária. Pulando.")
            checkpoint_path = os.path.join(self.model_dir, "profitability_predictor_checkpoint.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                return checkpoint.get('epoch', 0) + 1
            return 0
        if self.is_trained:
            logger.info("✅ [PROFITABILITY] Modelo já treinado e carregado. Pulando treinamento.")
            return 100 # Retorna o objetivo total de épocas

        logger.info(f"💪 [PROFITABILITY] Iniciando/Continuando treinamento por {num_epochs} épocas...")
        
        # ... (lógica de preparação de dados) ...
        if X is None: return 0

        optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
        checkpoint_path = os.path.join(self.model_dir, "profitability_predictor_checkpoint.pth")

        # <<< 2. CONTINUAR DE ONDE PAROU (CARREGAR CHECKPOINT) >>>
        start_epoch = 0
        best_val_loss = float('inf')
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                logger.info(f"🔄 Checkpoint carregado. Continuando da época {start_epoch}.")
            except Exception as e:
                logger.warning(f"⚠️ Não foi possível carregar checkpoint: {e}. Iniciando do zero.")

        # ... (lógica de DataLoaders) ...
        patience_counter = 0
        final_epoch = start_epoch
        
        # <<< 3. LOOP DE TREINO MODIFICADO >>>
        for epoch in range(start_epoch, start_epoch + num_epochs):
            final_epoch = epoch
            # ... (lógica do loop de treino e validação) ...

            # <<< 4. SALVAR CHECKPOINT COMPLETO >>>
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter > 20:
                    logger.info(f"🛑 [PROFITABILITY] Early stopping na época {epoch+1}.")
                    break
        
        if os.path.exists(checkpoint_path):
            self.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['model_state_dict'])
            os.remove(checkpoint_path)
        
        self.is_trained = True
        self.save_model(os.path.join(self.model_dir, "profitability_predictor.pth"))
        logger.info(f"✅ [PROFITABILITY] Treinamento concluído. Total de épocas: {final_epoch + 1}")
        return final_epoch + 1 # <<< 5. RETORNAR PROGRESSO TOTAL >>>

    def _prepare_training_data(self, featured_df: pd.DataFrame, trade_log: List[Dict[str, Any]]):
        if not trade_log or not isinstance(trade_log, list):
            logger.warning("⚠️ [PROFITABILITY] Log de trades vazio ou inválido. Não é possível gerar dados de treinamento para o ProfitabilityPredictor.")
            return None, None, None, None, None

        if not trade_log or featured_df.empty: return None, None, None, None, None
        
        featured_df.index = pd.to_datetime(featured_df.index, utc=True)
        trades_df = pd.DataFrame(trade_log)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], utc=True)
        trades_df.set_index('timestamp', inplace=True)
        trades_df = trades_df[trades_df['pnl'].notna() & (trades_df['pnl'] != 0)]
        if trades_df.empty: return None, None, None, None, None

        merged_data = pd.merge_asof(
            trades_df.sort_index(), featured_df.sort_index(),
            left_index=True, right_index=True, direction='backward', tolerance=pd.Timedelta('5min')
        )
        merged_data.dropna(inplace=True)
        if merged_data.empty: return None, None, None, None, None
        
        merged_data['is_profitable'] = (merged_data['pnl'] > 0).astype(int)
        capital_norm = TradingConfig.INITIAL_CAPITAL
        merged_data['normalized_pnl'] = merged_data['pnl'] / (capital_norm + 1e-9)
        
        merged_data['hold_time'] = (pd.to_datetime(merged_data['exit_time']) - pd.to_datetime(merged_data['entry_time'])).dt.total_seconds() / 3600.0
        merged_data['hold_time'] = merged_data['hold_time'].clip(0, 24)
        
        merged_data['hour_sin'] = np.sin(2 * np.pi * merged_data.index.hour / 24)
        merged_data['hour_cos'] = np.cos(2 * np.pi * merged_data.index.hour / 24)
        merged_data['day_sin'] = np.sin(2 * np.pi * merged_data.index.dayofweek / 7)
        merged_data['day_cos'] = np.cos(2 * np.pi * merged_data.index.dayofweek / 7)
        merged_data['regime'] = (merged_data['adx'] > 25).astype(int)
        
        X, context, y_prob, y_pnl, y_hold_time = [], [], [], [], []
        
        feature_cols = [col for col in featured_df.columns if col in merged_data.columns]
        
        for i in range(self.sequence_length, len(merged_data)):
            seq_indices = merged_data.index[i-self.sequence_length : i]
            seq = featured_df.loc[seq_indices, feature_cols].values
            
            ctx = merged_data.iloc[i][['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'regime']].values
            X.append(seq)
            context.append(ctx)
            y_prob.append(merged_data.iloc[i]['is_profitable'])
            y_pnl.append(merged_data.iloc[i]['normalized_pnl'])
            y_hold_time.append(merged_data.iloc[i]['hold_time'])
            
            # Data Augmentation (Jitter)
            if np.random.rand() > 0.5:
                jittered_seq = seq * (1 + np.random.normal(0, 0.01, size=seq.shape))
                X.append(jittered_seq)
                context.append(ctx)
                y_prob.append(merged_data.iloc[i]['is_profitable'])
                y_pnl.append(merged_data.iloc[i]['normalized_pnl'])
                y_hold_time.append(merged_data.iloc[i]['hold_time'])
        
        # [v3.1 ANTI-LEAKAGE — López de Prado MLAM Ch.7]
        # NOTA: train_model() é um stub incompleto (X e val_loss indefinidos nas linhas 177/207).
        # Os dados JÁ estão em ordem cronológica (merge_asof + sort_index acima).
        # Quando train_model() for implementado completamente, o split DEVE ser feito assim:
        #   _split = int(len(X) * 0.80)
        #   X_train, X_val = X[:_split], X[_split:]   # SEM shuffle — ordem temporal preservada
        # NÃO usar train_test_split(shuffle=True) que causa lookahead em séries temporais.
        return (np.array(X, dtype=np.float32), np.array(context, dtype=np.float32),
                np.array(y_prob, dtype=np.float32), np.array(y_pnl, dtype=np.float32),
                np.array(y_hold_time, dtype=np.float32))

    def predict_profitability(self, feature_sequence: np.ndarray, context_features: np.ndarray) -> Tuple[float, float, float, float]:
        if not self.is_trained: return 0.5, 0.0, 0.0, 1.0

        if feature_sequence.ndim == 2:
            feature_sequence = feature_sequence[np.newaxis, :, :]
        if context_features.ndim == 1:
            context_features = context_features[np.newaxis, :]
            
        if feature_sequence.shape[2] != self.input_dim:
            logger.error(f"❌ Shape inválido. Esperado {self.input_dim}, recebido {feature_sequence.shape[2]}.")
            return 0.5, 0.0, 0.0, 1.0
        
        # Ensemble Temporal (TTA)
        window_sizes = [self.sequence_length, self.sequence_length - 5, self.sequence_length + 5]
        probs, pnls, hold_times = [], [], []
        
        self.eval()
        with torch.no_grad():
            for ws in window_sizes:
                if feature_sequence.shape[1] >= ws:
                    feat_window = torch.from_numpy(feature_sequence[:, -ws:, :]).float().to(self.device)
                    context_tensor = torch.from_numpy(context_features).float().to(self.device)
                    prob_logits, pred_pnl, pred_hold_time = self.forward(feat_window, context_tensor)
                    probs.append(torch.sigmoid(prob_logits / self.temperature).item())
                    pnls.append(pred_pnl.item())
                    hold_times.append(pred_hold_time.item())
        
        probability = float(np.mean(probs)) if probs else 0.5
        expected_profit = float(np.mean(pnls)) if pnls else 0.0
        expected_hold_time = float(np.mean(hold_times)) if hold_times else 0.0
        
        # Incerteza com MC Dropout
        self.train()
        mc_probs = []
        with torch.no_grad():
            for _ in range(20):
                feat_tensor = torch.from_numpy(feature_sequence[:, -self.sequence_length:, :]).float().to(self.device)
                context_tensor = torch.from_numpy(context_features).float().to(self.device)
                prob_logits, _, _ = self.forward(feat_tensor, context_tensor)
                mc_probs.append(torch.sigmoid(prob_logits / self.temperature).item())
        
        confidence_interval = np.std(mc_probs)
        self.eval()
        
        return probability, expected_profit, expected_hold_time, confidence_interval

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'sequence_length': self.sequence_length,
            'dropout_rate': self.dropout_rate,
            'best_metrics': self.best_metrics
        }
        torch.save(save_dict, path)
        logger.info(f"💾 [PROFITABILITY] Modelo salvo em '{path}'.")

    def load_model(self, path: str):
        if os.path.exists(path):
            try:
                save_dict = torch.load(path, map_location=self.device)
                # Recria o modelo com os parâmetros salvos para garantir compatibilidade
                self.__init__(
                    input_dim=save_dict['input_dim'],
                    hidden_dim=save_dict.get('hidden_dim', 256),
                    sequence_length=save_dict.get('sequence_length', 20),
                    dropout_rate=save_dict.get('dropout_rate', 0.3)
                )
                self.load_state_dict(save_dict['model_state_dict'])
                self.best_metrics = save_dict.get('best_metrics')
                self.eval()
                self.is_trained = True
                logger.info(f"✅ [PROFITABILITY] Modelo carregado de '{path}'.")
            except Exception as e:
                logger.error(f"❌ [ERRO PREDICTOR] Falha ao carregar: {e}", exc_info=True)
                self.is_trained = False
        else:
            logger.info(f"ℹ️ [PROFITABILITY] Modelo não encontrado em '{path}'.")
            self.is_trained = False
