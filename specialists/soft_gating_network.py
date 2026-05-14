# -----------------------------------------------------------------------------
# ARQUIVO: specialists/soft_gating_network.py
# -----------------------------------------------------------------------------
"""
Attention-based Soft Gating Network for Mixture of Experts specialist selection.

Scientific basis:
- Jacobs et al. (1991) "Adaptive Mixtures of Local Experts"
- Shazeer et al. (2017) "Sparsely-Gated Mixture-of-Experts Layer"
- Vaswani et al. (2017) "Attention Is All You Need"

Key advantages over hard (discrete) gating:
1. Smooth regime transitions (no whipsaw during Bull→Bear)
2. Uncertainty quantification via entropy of gating probabilities
3. End-to-end differentiable training
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from collections import deque

from utils.logger import get_logger

logger = get_logger("SoftGatingNetwork")


class FeatureAttention(nn.Module):
    """
    Learns which latent features are most important for gating decisions.
    Acts as a soft feature selector before the gating head.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Attention weights in [0, 1]
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [batch, input_dim]
            
        Returns:
            attended_features: Features weighted by attention [batch, input_dim]
            attention_weights: The attention weights for interpretability [batch, input_dim]
        """
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        return attended_features, attention_weights


class SoftGatingNetwork(nn.Module):
    """
    Attention-based soft gating for expert selection in MoE architecture.
    
    Instead of hard (discrete) selection of one expert, produces soft
    probability distribution over all experts, enabling:
    - Smooth transitions between market regimes
    - Ensemble predictions during uncertain periods
    - Uncertainty quantification via gating entropy
    """
    
    EXPERT_NAMES = ['bull', 'bear', 'ranger']
    
    def __init__(
        self, 
        latent_dim: int, 
        num_experts: int = 3, 
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
        min_expert_weight: float = 0.05  # Minimum weight for load balancing
    ):
        """
        Args:
            latent_dim: Dimension of autoencoder latent features
            num_experts: Number of specialists (default 3: Bull, Bear, Ranger)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate for regularization
            temperature: Softmax temperature (>1 = exploration, <1 = exploitation)
            min_expert_weight: Minimum probability floor for load balancing
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.min_expert_weight = min_expert_weight
        
        # Feature attention module
        self.feature_attention = FeatureAttention(latent_dim, hidden_dim, dropout)
        
        # Gating head: produces logits for each expert
        self.gating_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Auxiliary: regime momentum predictor for transition detection
        self.momentum_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output in [-1, 1]: negative = bearish momentum, positive = bullish
        )
        
        # Tracking for load balancing loss (Switch Transformer Eq.4)
        # expert_prob_sum: Σ p_i across tokens → P_i = expert_prob_sum / usage_count
        # expert_dispatch_count: count tokens where expert i is argmax → f_i = count / usage_count
        self.register_buffer('expert_prob_sum', torch.zeros(num_experts))
        self.register_buffer('expert_dispatch_count', torch.zeros(num_experts))
        self.register_buffer('usage_count', torch.tensor(0.0))
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"✅ SoftGatingNetwork initialized: latent_dim={latent_dim}, "
            f"num_experts={num_experts}, hidden_dim={hidden_dim}, temp={temperature}"
        )
        
    def _init_weights(self):
        """Xavier/Glorot initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(
        self, 
        latent_features: torch.Tensor, 
        temperature: Optional[float] = None,
        return_entropy: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: compute gating probabilities and optional metrics.
        
        Args:
            latent_features: Autoencoder latent features [batch, latent_dim]
            temperature: Override default temperature (for annealing)
            return_entropy: Whether to compute and return entropy
            
        Returns:
            Dict with:
                - probabilities: Expert selection probabilities [batch, num_experts]
                - attention_weights: Feature attention weights [batch, latent_dim]
                - momentum: Regime momentum signal [batch, 1]
                - entropy: (optional) Gating entropy for uncertainty [batch]
                - selected_expert: Argmax expert index [batch]
        """
        temp = temperature if temperature is not None else self.temperature
        
        # Apply feature attention
        attended_features, attention_weights = self.feature_attention(latent_features)
        
        # Compute gating logits
        logits = self.gating_head(attended_features)
        
        # Soft probabilities with temperature scaling
        probabilities = F.softmax(logits / temp, dim=-1)
        
        # Apply minimum expert weight (load balancing)
        if self.min_expert_weight > 0:
            probabilities = (
                probabilities * (1 - self.num_experts * self.min_expert_weight) + 
                self.min_expert_weight
            )
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        
        # Compute regime momentum
        momentum = self.momentum_head(attended_features)
        
        # Track expert usage for load balancing (Switch Transformer Eq.4)
        if self.training:
            with torch.no_grad():
                n = probabilities.shape[0]
                self.expert_prob_sum += probabilities.sum(dim=0)
                top_expert = probabilities.argmax(dim=-1)
                for i in range(self.num_experts):
                    self.expert_dispatch_count[i] += (top_expert == i).sum().float()
                self.usage_count += n
        
        result = {
            'probabilities': probabilities,
            'attention_weights': attention_weights,
            'momentum': momentum,
            'selected_expert': probabilities.argmax(dim=-1),
            'logits': logits
        }
        
        if return_entropy:
            # Shannon entropy: H = -sum(p * log(p))
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)
            result['entropy'] = entropy
            
        return result
    
    def get_expert_probabilities(self, latent_features: torch.Tensor) -> Dict[str, float]:
        """
        Get named expert probabilities for a single observation.
        Convenience method for inference.
        
        Returns:
            Dict mapping expert name to probability, e.g. {'bull': 0.6, 'bear': 0.1, 'ranger': 0.3}
        """
        self.eval()
        with torch.no_grad():
            if latent_features.dim() == 1:
                latent_features = latent_features.unsqueeze(0)
            
            output = self.forward(latent_features, return_entropy=True)
            probs = output['probabilities'][0].cpu().numpy()
            entropy = output['entropy'][0].cpu().item()
            momentum = output['momentum'][0].cpu().item()
            
        return {
            'probabilities': {name: float(probs[i]) for i, name in enumerate(self.EXPERT_NAMES)},
            'entropy': entropy,
            'momentum': momentum,
            'dominant_expert': self.EXPERT_NAMES[np.argmax(probs)],
            'confidence': float(np.max(probs))
        }
    
    def compute_load_balancing_loss(self, reset: bool = True) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss (Switch Transformer Eq.4).

        L_aux = N * Σ_i (f_i * P_i)
        where:
          f_i = fraction of tokens dispatched to expert i (via argmax)
          P_i = mean routing probability for expert i
          N   = number of experts (scaling constant)

        f_i e P_i são acumulados em buffers SEPARADOS no forward().
        Chamar com reset=True uma vez por EPOCH (não por batch).

        Reference: Fedus et al. (2022) "Switch Transformers", Eq.4
        """
        if self.usage_count < 1:
            return torch.tensor(0.0, device=self.expert_prob_sum.device)

        P_i = self.expert_prob_sum / self.usage_count
        f_i = self.expert_dispatch_count / self.usage_count
        loss = self.num_experts * torch.dot(f_i, P_i)

        if reset:
            self.expert_prob_sum.zero_()
            self.expert_dispatch_count.zero_()
            self.usage_count.zero_()

        return loss
    
    def get_feature_importance(self, latent_features: torch.Tensor) -> np.ndarray:
        """
        Get average attention weights across a batch for feature importance analysis.
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.feature_attention(latent_features)
            return attention_weights.mean(dim=0).cpu().numpy()
            
    def save(self, path: str):
        """Save model state and config."""
        config = {
            'latent_dim': self.latent_dim,
            'num_experts': self.num_experts,
            'hidden_dim': self.hidden_dim,
            'temperature': self.temperature,
            'min_expert_weight': self.min_expert_weight
        }
        torch.save({
            'state_dict': self.state_dict(),
            'config': config
        }, path)
        logger.info(f"✅ SoftGatingNetwork saved to {path}")
        
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'SoftGatingNetwork':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        logger.info(f"✅ SoftGatingNetwork loaded from {path}")
        return model


class MixtureOfExpertsRouter:
    """
    High-level router that combines SoftGatingNetwork with specialists.
    Handles the weighted ensemble of expert outputs.
    """
    
    def __init__(
        self,
        gating_network: SoftGatingNetwork,
        specialists: Dict[str, Any],  # {'bull': BullSpecialist, 'bear': BearSpecialist, 'ranger': RangerSpecialist}
        ensemble_mode: str = 'weighted',  # 'weighted', 'top_k', 'threshold'
        top_k: int = 2,
        threshold: float = 0.2
    ):
        """
        Args:
            gating_network: Trained SoftGatingNetwork
            specialists: Dict mapping expert name to specialist instance
            ensemble_mode: How to combine expert outputs
                - 'weighted': Weighted average by gating probabilities
                - 'top_k': Only use top k experts
                - 'threshold': Only use experts above probability threshold
            top_k: Number of experts to use in top_k mode
            threshold: Minimum probability in threshold mode
        """
        self.gating = gating_network
        self.specialists = specialists
        self.ensemble_mode = ensemble_mode
        self.top_k = top_k
        self.threshold = threshold
        
        logger.info(f"✅ MixtureOfExpertsRouter initialized with mode={ensemble_mode}")
        
    def route(
        self, 
        latent_features: torch.Tensor,
        observation: np.ndarray,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Route observation to specialists and combine outputs.
        
        Args:
            latent_features: Autoencoder latent features for gating
            observation: Full observation for specialists
            context: Additional context (price, regime info, etc.)
            
        Returns:
            Combined action dict with confidence and routing info
        """
        # Get gating probabilities
        gating_output = self.gating.get_expert_probabilities(latent_features)
        probs = gating_output['probabilities']
        
        # Determine which experts to use based on mode
        if self.ensemble_mode == 'weighted':
            active_experts = {k: v for k, v in probs.items() if v > 0.01}
        elif self.ensemble_mode == 'top_k':
            sorted_experts = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            active_experts = dict(sorted_experts[:self.top_k])
        elif self.ensemble_mode == 'threshold':
            active_experts = {k: v for k, v in probs.items() if v >= self.threshold}
            if not active_experts:
                # Fallback to best expert if none above threshold
                best = max(probs.items(), key=lambda x: x[1])
                active_experts = {best[0]: best[1]}
        else:
            active_experts = probs
            
        # Normalize active expert weights
        total_weight = sum(active_experts.values())
        active_experts = {k: v / total_weight for k, v in active_experts.items()}
        
        # Get actions from active experts
        expert_actions = {}
        for expert_name, weight in active_experts.items():
            specialist = self.specialists.get(expert_name)
            if specialist and hasattr(specialist, 'get_action'):
                try:
                    action = specialist.get_action(observation, context)
                    expert_actions[expert_name] = {
                        'action': action,
                        'weight': weight
                    }
                except Exception as e:
                    logger.warning(f"⚠️ Expert {expert_name} failed: {e}")
                    
        # Combine actions (weighted average of continuous signals)
        combined_action = self._combine_actions(expert_actions)
        
        return {
            'action': combined_action,
            'gating_probabilities': probs,
            'active_experts': active_experts,
            'entropy': gating_output['entropy'],
            'momentum': gating_output['momentum'],
            'dominant_expert': gating_output['dominant_expert'],
            'confidence': gating_output['confidence']
        }
        
    def _combine_actions(self, expert_actions: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine expert actions using weighted average."""
        if not expert_actions:
            return {'action': 0, 'side': 0, 'confidence': 0.0}
            
        # Weighted average of action values
        weighted_action = 0.0
        weighted_confidence = 0.0
        combined_side = 0
        
        for expert_name, data in expert_actions.items():
            action = data['action']
            weight = data['weight']
            
            if isinstance(action, dict):
                action_val = action.get('action', 0)
                conf = action.get('confidence', 0.5)
                side = action.get('side', 0)
            else:
                action_val = action if isinstance(action, (int, float)) else 0
                conf = 0.5
                side = 0
                
            weighted_action += float(action_val) * weight
            weighted_confidence += conf * weight
            combined_side += int(side) * weight
            
        return {
            'action': weighted_action,
            'side': int(np.sign(combined_side)) if abs(combined_side) > 0.3 else 0,
            'confidence': weighted_confidence,
            'ensemble_size': len(expert_actions)
        }


# Standalone utility for integration
def create_gating_network(
    config: Any,
    latent_dim: Optional[int] = None,
    device: str = 'auto'
) -> SoftGatingNetwork:
    """
    Factory function to create SoftGatingNetwork from config.
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # Try to get latent dim from config
    if latent_dim is None:
        latent_dim = getattr(config, 'AUTOENCODER_LATENT_DIM', 64)
        
    hidden_dim = getattr(config, 'GATING_HIDDEN_DIM', 64)
    dropout = getattr(config, 'GATING_DROPOUT', 0.1)
    temperature = getattr(config, 'GATING_TEMPERATURE', 1.0)
    
    model = SoftGatingNetwork(
        latent_dim=latent_dim,
        num_experts=3,
        hidden_dim=hidden_dim,
        dropout=dropout,
        temperature=temperature
    ).to(device)
    
    return model
