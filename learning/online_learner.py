# -----------------------------------------------------------------------------
# ARQUIVO: learning/online_learner.py
# -----------------------------------------------------------------------------
"""
Online Learning Module for Adaptive Trend Surfing.

Implements Recurrent Reinforcement Learning (RRL) inspired online adaptation
for trading models. Allows the model to adapt to evolving market conditions
during live trading without full retraining.

Scientific References:
- Moody & Saffell (2001) "Learning to Trade via Direct Reinforcement"
- Sutton & Barto (2018) "Reinforcement Learning: An Introduction" (TD-learning)
- Gold & Shadlen (2007) "The Neural Basis of Decision Making" (evidence accumulation)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Deque
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import copy

from utils.logger import get_logger

logger = get_logger("OnlineLearner")


@dataclass
class TradeOutcome:
    """Records a completed trade for online learning."""
    observation: np.ndarray
    action: np.ndarray
    reward: float  # PnL or shaped reward
    duration: int  # Trade duration in candles
    regime: str  # bull, bear, ranger
    timestamp: datetime
    entry_price: float
    exit_price: float
    

class ExperienceBuffer:
    """
    Prioritized experience buffer for online learning.
    Uses TD-error based prioritization to focus on surprising outcomes.
    """
    
    def __init__(
        self,
        capacity: int = 500,
        priority_alpha: float = 0.6,  # Priority exponent
        priority_beta: float = 0.4,   # IS weight exponent (annealed to 1)
        beta_increment: float = 0.001
    ):
        self.capacity = capacity
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.beta_increment = beta_increment
        
        self.buffer: Deque[TradeOutcome] = deque(maxlen=capacity)
        self.priorities: Deque[float] = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def add(self, outcome: TradeOutcome, priority: Optional[float] = None):
        """Add experience with priority (defaults to max)."""
        self.buffer.append(outcome)
        self.priorities.append(priority if priority is not None else self.max_priority)
        
    def sample(self, batch_size: int) -> Tuple[List[TradeOutcome], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling.
        
        Returns:
            experiences: List of TradeOutcome
            indices: Array of sampled indices
            weights: Importance sampling weights
        """
        n = len(self.buffer)
        if n == 0:
            return [], np.array([]), np.array([])
            
        batch_size = min(batch_size, n)
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.priority_alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)
        
        # Compute importance sampling weights
        self.priority_beta = min(1.0, self.priority_beta + self.beta_increment)
        weights = (n * probs[indices]) ** (-self.priority_beta)
        weights = weights / weights.max()
        
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, indices, weights
        
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(priority, 1e-6)
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self) -> int:
        return len(self.buffer)


class OnlineTrendLearner:
    """
    RRL-inspired online adaptation for trend following.
    
    Key features:
    - Incremental weight updates based on recent trades
    - Exponential recency weighting (recent trades matter more)
    - Gradient accumulation with clipping for stability
    - Separate adaptation rates for different regimes
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-5,  # Much smaller than offline training
        adaptation_window: int = 20,  # Number of recent trades
        reward_decay: float = 0.95,   # Exponential decay for recency
        min_trades_for_update: int = 5,
        gradient_clip: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 1e-6,
        regime_specific_lr: bool = True  # Different LR per regime
    ):
        """
        Args:
            model: PyTorch model to adapt (policy network)
            learning_rate: Base learning rate for online updates
            adaptation_window: Window size for trade history
            reward_decay: Exponential decay factor for recency weighting
            min_trades_for_update: Minimum trades before first update
            gradient_clip: Max gradient norm
            momentum: SGD momentum
            weight_decay: L2 regularization
            regime_specific_lr: If True, scale LR by regime volatility
        """
        self.model = model
        self.base_lr = learning_rate
        self.adaptation_window = adaptation_window
        self.reward_decay = reward_decay
        self.min_trades_for_update = min_trades_for_update
        self.gradient_clip = gradient_clip
        self.regime_specific_lr = regime_specific_lr
        
        # Optimizer with momentum for smoother updates
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer(capacity=adaptation_window * 10)
        
        # Tracking
        self.recent_rewards: Deque[float] = deque(maxlen=adaptation_window)
        self.regime_stats: Dict[str, Dict[str, float]] = {
            'bull': {'trades': 0, 'avg_reward': 0, 'lr_scale': 1.0},
            'bear': {'trades': 0, 'avg_reward': 0, 'lr_scale': 1.0},
            'ranger': {'trades': 0, 'avg_reward': 0, 'lr_scale': 1.0}
        }
        
        # Store initial weights for reset capability
        self.initial_state = copy.deepcopy(model.state_dict())
        
        # Metrics
        self.total_updates = 0
        self.cumulative_loss = 0.0
        
        logger.info(
            f"✅ OnlineTrendLearner initialized: "
            f"lr={learning_rate}, window={adaptation_window}, decay={reward_decay}"
        )
        
    def record_trade(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        duration: int,
        regime: str,
        entry_price: float,
        exit_price: float
    ):
        """
        Record a completed trade outcome.
        
        Args:
            observation: Market state at trade entry
            action: Action taken (direction, size, etc.)
            reward: Realized PnL or shaped reward
            duration: Trade duration in candles
            regime: Market regime (bull, bear, ranger)
            entry_price: Trade entry price
            exit_price: Trade exit price
        """
        outcome = TradeOutcome(
            observation=observation,
            action=action,
            reward=reward,
            duration=duration,
            regime=regime,
            timestamp=datetime.now(),
            entry_price=entry_price,
            exit_price=exit_price
        )
        
        # Calculate initial priority (proportional to |reward|)
        priority = abs(reward) + 0.1  # Ensure non-zero
        self.experience_buffer.add(outcome, priority)
        
        # Update tracking
        self.recent_rewards.append(reward)
        
        # Update regime statistics
        stats = self.regime_stats[regime]
        n = stats['trades']
        stats['avg_reward'] = (stats['avg_reward'] * n + reward) / (n + 1)
        stats['trades'] = n + 1
        
        logger.debug(
            f"📝 Trade recorded: regime={regime}, reward={reward:.4f}, "
            f"duration={duration}, buffer_size={len(self.experience_buffer)}"
        )
        
    def should_update(self) -> bool:
        """Check if we have enough data for an update."""
        return len(self.experience_buffer) >= self.min_trades_for_update
        
    def compute_baseline(self) -> float:
        """
        Compute reward baseline for variance reduction.
        Uses exponentially weighted moving average.
        """
        if not self.recent_rewards:
            return 0.0
            
        weights = [self.reward_decay ** (len(self.recent_rewards) - i - 1) 
                   for i in range(len(self.recent_rewards))]
        weight_sum = sum(weights)
        
        return sum(w * r for w, r in zip(weights, self.recent_rewards)) / weight_sum
        
    def adapt(self, batch_size: int = 8) -> Dict[str, float]:
        """
        Perform online adaptation step.
        
        Uses policy gradient with baseline subtraction:
        L = -sum( log π(a|s) * (R - baseline) )
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dict with training metrics
        """
        if not self.should_update():
            return {'skipped': True, 'reason': 'insufficient_data'}
            
        self.model.train()
        
        # Sample experiences
        experiences, indices, is_weights = self.experience_buffer.sample(batch_size)
        
        if len(experiences) == 0:
            return {'skipped': True, 'reason': 'empty_sample'}
            
        # Compute baseline
        baseline = self.compute_baseline()
        
        # Prepare batch
        observations = torch.FloatTensor(np.array([e.observation for e in experiences]))
        actions = torch.FloatTensor(np.array([e.action for e in experiences]))
        rewards = np.array([e.reward for e in experiences])
        regimes = [e.regime for e in experiences]
        
        # Compute advantages (reward - baseline)
        advantages = rewards - baseline
        
        # Apply regime-specific learning rate scaling
        if self.regime_specific_lr:
            lr_scales = np.array([self.regime_stats[r]['lr_scale'] for r in regimes])
            advantages = advantages * lr_scales
            
        advantages = torch.FloatTensor(advantages)
        is_weights = torch.FloatTensor(is_weights)
        
        # Forward pass to get action probabilities
        self.optimizer.zero_grad()
        
        try:
            # Assume model has method to get log probability
            if hasattr(self.model, 'get_action_log_prob'):
                log_probs = self.model.get_action_log_prob(observations, actions)
            else:
                # Fallback: compute via forward pass
                output = self.model(observations)
                if isinstance(output, dict):
                    output = output.get('probabilities', output.get('logits', None))
                if output is not None:
                    log_probs = F.log_softmax(output, dim=-1)
                    # Get log prob of taken actions
                    action_indices = actions.argmax(dim=-1) if actions.dim() > 1 else actions.long()
                    log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze()
                else:
                    logger.warning("⚠️ Cannot compute log probs from model output")
                    return {'skipped': True, 'reason': 'no_log_probs'}
                    
        except Exception as e:
            logger.warning(f"⚠️ Forward pass failed: {e}")
            return {'skipped': True, 'reason': str(e)}
            
        # Policy gradient loss with importance sampling
        loss = -(log_probs * advantages * is_weights).mean()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.gradient_clip
        )
        
        # Update weights
        self.optimizer.step()
        
        # Update priorities based on TD error (approximated by advantage magnitude)
        new_priorities = np.abs(advantages.detach().numpy()) + 0.1
        self.experience_buffer.update_priorities(indices, new_priorities)
        
        # Update tracking
        self.total_updates += 1
        self.cumulative_loss += loss.item()
        
        metrics = {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'baseline': baseline,
            'mean_advantage': advantages.mean().item(),
            'batch_size': len(experiences),
            'total_updates': self.total_updates
        }
        
        logger.debug(f"🔄 Online update #{self.total_updates}: loss={loss.item():.6f}")
        
        return metrics
        
    def update_regime_lr_scales(self):
        """
        Dynamically adjust learning rate scales per regime based on performance.
        
        Low-performing regimes get higher LR (more adaptation needed)
        High-performing regimes get lower LR (preserve good behavior)
        """
        all_rewards = [s['avg_reward'] for s in self.regime_stats.values()]
        if not any(all_rewards):
            return
            
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards) + 1e-6
        
        for regime, stats in self.regime_stats.items():
            if stats['trades'] < 5:
                continue
                
            # Z-score of regime performance
            z = (stats['avg_reward'] - mean_reward) / std_reward
            
            # Scale LR inversely with performance
            # Bad performance → higher LR (more adaptation)
            # Good performance → lower LR (preserve)
            stats['lr_scale'] = np.clip(1.0 - 0.3 * z, 0.5, 2.0)
            
        logger.debug(f"📊 Regime LR scales updated: {self.regime_stats}")
        
    def reset_to_initial(self):
        """Reset model weights to initial state."""
        self.model.load_state_dict(copy.deepcopy(self.initial_state))
        logger.info("🔄 Model reset to initial weights")
        
    def save_checkpoint(self, path: str):
        """Save online learner state."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'regime_stats': self.regime_stats,
            'total_updates': self.total_updates,
            'cumulative_loss': self.cumulative_loss
        }
        torch.save(checkpoint, path)
        logger.info(f"✅ OnlineLearner checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str):
        """Load online learner state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.regime_stats = checkpoint['regime_stats']
        self.total_updates = checkpoint['total_updates']
        self.cumulative_loss = checkpoint['cumulative_loss']
        logger.info(f"✅ OnlineLearner checkpoint loaded from {path}")


class AdaptivePositionManager:
    """
    Uses online learning signals to adjust position sizing and holding periods.
    
    Implements the "trend surfing" logic:
    - Increase position size as trend strengthens
    - Extend holding period during strong trends
    - Quick exit when trend momentum weakens
    """
    
    def __init__(
        self,
        base_position_size: float = 1.0,
        max_position_scale: float = 2.0,
        momentum_threshold: float = 0.3,
        min_hold_periods: int = 5,
        max_hold_periods: int = 100
    ):
        self.base_position_size = base_position_size
        self.max_position_scale = max_position_scale
        self.momentum_threshold = momentum_threshold
        self.min_hold_periods = min_hold_periods
        self.max_hold_periods = max_hold_periods
        
        # Trend tracking
        self.momentum_history: Deque[float] = deque(maxlen=20)
        self.current_position_scale = 1.0
        self.hold_extension = 0
        
    def update_momentum(self, momentum: float, gating_confidence: float):
        """
        Update momentum signal and adjust position parameters.
        
        Args:
            momentum: Trend momentum from gating network [-1, 1]
            gating_confidence: Max gating probability
        """
        self.momentum_history.append(momentum)
        
        if len(self.momentum_history) < 5:
            return
            
        # Compute momentum statistics
        recent_momentum = np.array(list(self.momentum_history))
        avg_momentum = np.mean(recent_momentum[-5:])
        momentum_trend = np.mean(np.diff(recent_momentum[-5:])) if len(recent_momentum) > 5 else 0
        
        # Strong, consistent momentum → increase position
        if abs(avg_momentum) > self.momentum_threshold and abs(momentum_trend) < 0.1:
            # Trend is strong and stable
            self.current_position_scale = min(
                self.max_position_scale,
                1.0 + abs(avg_momentum) * gating_confidence
            )
            self.hold_extension = int(10 * gating_confidence)
        else:
            # Reduce position scale
            self.current_position_scale = max(1.0, self.current_position_scale * 0.9)
            self.hold_extension = max(0, self.hold_extension - 1)
            
    def get_position_size(self) -> float:
        """Get current recommended position size."""
        return self.base_position_size * self.current_position_scale
        
    def get_min_hold(self) -> int:
        """Get current minimum hold period."""
        return self.min_hold_periods + self.hold_extension
        
    def should_exit_early(self, current_pnl_pct: float, hold_periods: int) -> bool:
        """
        Determine if position should exit early based on trend momentum.
        
        Returns True if momentum has reversed significantly.
        """
        if len(self.momentum_history) < 5:
            return False
            
        recent = list(self.momentum_history)[-5:]
        
        # Check for momentum reversal
        if current_pnl_pct > 0:
            # In profit: exit if momentum reversing
            return np.mean(recent) * np.sign(current_pnl_pct) < -self.momentum_threshold
        else:
            # In loss: exit if momentum continuing against us
            return hold_periods > self.min_hold_periods and \
                   abs(np.mean(recent)) > self.momentum_threshold


# Convenience factory function
def create_online_learner(
    model: nn.Module,
    config: Any,
    device: str = 'auto'
) -> OnlineTrendLearner:
    """Factory function to create OnlineTrendLearner from config."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    lr = getattr(config, 'ONLINE_LEARNING_RATE', 1e-5)
    window = getattr(config, 'ONLINE_ADAPTATION_WINDOW', 20)
    decay = getattr(config, 'ONLINE_REWARD_DECAY', 0.95)
    
    return OnlineTrendLearner(
        model=model.to(device),
        learning_rate=lr,
        adaptation_window=window,
        reward_decay=decay
    )
