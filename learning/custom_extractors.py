# -----------------------------------------------------------------------------
# ARQUIVO: learning/custom_extractors.py
# -----------------------------------------------------------------------------
"""
🧠 Custom Feature Extractors para RL com Stable Baselines 3.

Este módulo implementa extractors personalizados para processar features
temporais com LSTM e Attention antes de passar para as redes policy/value.

Reference: 
- Stable Baselines3 Custom Policy documentation
- Vaswani et al. (2017) - Attention Is All You Need
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np


class HybridLSTMAttentionExtractor(BaseFeaturesExtractor):
    """
    Feature extractor híbrido que combina:
    1. LSTM para capturar dependências temporais REAIS (requer VecFrameStack)
    2. Multi-Head Attention para foco em features relevantes por timestep
    3. Layer Norm para estabilidade de treinamento

    Com VecFrameStack(n_stack=N):
        obs shape:  (batch, obs_dim * N)          ← entrada do SB3
        reshape:    (batch, N, obs_dim)            ← sequência temporal real
        LSTM vê N timesteps reais → captura tendência, momentum, reversão

    Sem VecFrameStack (n_stack=1, legado):
        obs shape:  (batch, obs_dim)
        LSTM recebe seq_len=1 → equivalente a MLP (comportamento anterior)

    Projetado para funcionar com SAC/PPO em ambientes de trading.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_stack: int = 1,           # Número de frames empilhados via VecFrameStack
        latent_dim: int = 32,
        features_dim: int = 128,
        lstm_hidden_size: int = 64,
        num_heads: int = 4,
        dropout: float = 0.15,
    ):
        # features_dim é o tamanho do output final
        super().__init__(observation_space, features_dim)

        self.n_stack = n_stack
        self.latent_dim = latent_dim
        self.lstm_hidden_size = lstm_hidden_size

        # Dimensão total da observação (obs_dim * n_stack quando frame stacking ativo)
        total_obs_dim = int(np.prod(observation_space.shape))

        # Dimensão de um único frame (o que o LSTM processa por timestep)
        if n_stack > 1:
            if total_obs_dim % n_stack != 0:
                raise ValueError(
                    f"HybridLSTMAttentionExtractor: obs_dim={total_obs_dim} não é "
                    f"divisível por n_stack={n_stack}. Verifique VecFrameStack."
                )
            self.single_obs_dim = total_obs_dim // n_stack
        else:
            self.single_obs_dim = total_obs_dim

        # 1. Input projection: single_obs_dim → lstm_hidden_size*2
        #    Projeta cada frame individualmente antes de entrar no LSTM
        self.input_proj = nn.Sequential(
            nn.Linear(self.single_obs_dim, lstm_hidden_size * 2),
            nn.LayerNorm(lstm_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. LSTM for temporal dependencies (unidirectional for online inference)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size * 2,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0,  # Dropout only if > 1 layer
            bidirectional=False,  # Causal LSTM for online trading
        )

        # 3. Multi-Head Self-Attention sobre os hidden states do LSTM
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(lstm_hidden_size)

        # 4. Output projection to features_dim
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid extractor.

        Args:
            observations: (batch_size, obs_dim * n_stack)  ← saída do VecFrameStack (flat)
                       ou (batch_size, obs_dim)             ← sem frame stacking (legado)

        Returns:
            features: (batch_size, features_dim)
        """
        batch_size = observations.shape[0]

        if self.n_stack > 1:
            # VecFrameStack ativo: reshape frames empilhados em sequência temporal real
            # (batch, obs_dim * n_stack) → (batch, n_stack, obs_dim)
            x = observations.view(batch_size, self.n_stack, self.single_obs_dim)
        elif observations.dim() == 2:
            # Legado sem frame stacking: adiciona dimensão temporal unitária
            # (batch, obs_dim) → (batch, 1, obs_dim)
            x = observations.unsqueeze(1)
        else:
            x = observations  # Já é 3D

        # x shape: (batch, seq_len, single_obs_dim)
        # seq_len = n_stack (sequência real) ou 1 (legado)

        # 1. Project each timestep independently
        x = self.input_proj(x)   # (batch, seq_len, lstm_hidden*2)

        # 2. LSTM processing — processa sequência temporal
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden)

        # 3. Self-attention sobre todos os hidden states da sequência
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.attention_norm(lstm_out + attn_out)  # Residual connection

        # 4. Pega o último timestep (mais recente) e projeta para output
        last_hidden = x[:, -1, :]  # (batch, lstm_hidden)
        features = self.output_proj(last_hidden)  # (batch, features_dim)

        return features


class AttentionMLPExtractor(BaseFeaturesExtractor):
    """
    🔬 [SCIENTIFIC REFACTOR] 
    Extractor que usa Multi-Head Attention para ponderar a importância de cada feature.
    Ideal para datasets com muitas colunas (~264) onde o SAC precisa focar no que importa.
    
    Ao contrário da LSTM, este é 100% compatível com SAC padrão (sem frame stacking).
    """
    
    def __init__(
        self, 
        observation_space: spaces.Box,
        features_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)
        
        input_dim = int(np.prod(observation_space.shape))
        self.input_dim = input_dim
        
        # 1. Project to a larger space for attention (must be divisible by num_heads)
        self.embed_dim = ((input_dim // num_heads) + 1) * num_heads
        self.proj = nn.Linear(input_dim, self.embed_dim)
        
        # 2. Multi-Head Attention block
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(self.embed_dim)
        
        # 3. Final MLP transition
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Project and add pseudo-sequence dimension for multihead attention
        # (batch, input_dim) -> (batch, 1, embed_dim)
        x = self.proj(observations).unsqueeze(1)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Remove pseudo-sequence and project to features_dim
        x = x.squeeze(1)
        return self.mlp(x)


class SimpleMLP(BaseFeaturesExtractor):
    """
    Simple MLP extractor as fallback - no LSTM/Attention.
    Faster training, good for simpler environments.
    """
    
    def __init__(
        self, 
        observation_space: spaces.Box,
        features_dim: int = 128,
        hidden_sizes: tuple = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)
        
        input_dim = int(np.prod(observation_space.shape))
        
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, features_dim))
        layers.append(nn.LayerNorm(features_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
