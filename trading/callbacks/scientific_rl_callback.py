# -----------------------------------------------------------------------------
# ARQUIVO: trading/callbacks/scientific_rl_callback.py
# -----------------------------------------------------------------------------
"""
🔬 Scientific RL Callback for TensorBoard Metrics

Implements advanced quantitative finance metrics for RL training monitoring:
- Sharpe Ratio (Sharpe, 1966)
- Sortino Ratio (Sortino & van der Meer, 1991)
- Calmar Ratio (Young, 1991)
- Kelly Criterion (Kelly, 1956)
- Maximum Drawdown (Burke, 1994)
- Win Rate & Profit Factor

References:
- Marcos López de Prado, "Advances in Financial Machine Learning" (2018)
- Chan, E., "Algorithmic Trading" (2013)
"""

import numpy as np
from typing import Dict, Any, Optional, List
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

class ScientificRLCallback(BaseCallback):
    """
    🔬 Advanced RL training callback with scientifically-proven metrics.
    
    Logs to TensorBoard:
    - Sharpe/Sortino/Calmar ratios
    - Maximum Drawdown
    - Win Rate & Profit Factor
    - Entropy coefficient (SAC)
    - Action distribution statistics
    """
    
    def __init__(
        self,
        eval_freq: int = 1000,
        log_freq: int = 100,
        risk_free_rate: float = 0.0,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.risk_free_rate = risk_free_rate
        
        # Metrics storage
        self.episode_returns: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)
        self.trade_pnls: deque = deque(maxlen=500)
        self.actions_history: deque = deque(maxlen=1000)
        
        # Running statistics
        self.equity_curve: List[float] = [1.0]
        self.peak_equity = 1.0
        self.n_trades = 0
        self.n_wins = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
    def _on_step(self) -> bool:
        # Log every log_freq steps
        if self.num_timesteps % self.log_freq == 0:
            self._log_action_statistics()
        
        # Full evaluation every eval_freq steps
        if self.num_timesteps % self.eval_freq == 0:
            self._log_scientific_metrics()
            self._log_training_diagnostics()
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at end of each rollout."""
        # Get episode info from VecEnv (DummyVecEnv or SubprocVecEnv)
        try:
            # [SCIENTIFIC FIX] Tenta consumir e limpar histórico para evitar contagem duplicada
            if hasattr(self.training_env, 'env_method'):
                try:
                    # Tenta chamar consume_trade_history nos envs internos
                    histories_list = self.training_env.env_method('consume_trade_history')
                    # Se não lançou erro, assume que funcionou
                    for history in histories_list:
                        if history:
                            for pnl in history:
                                self._record_trade(pnl)
                    return # Sucesso - não usar fallback
                except Exception:
                    # Método não existe ou falhou - tentar fallback
                    pass

            # Fallback legacy (risco de double counting se o env não limpar o histórico)
            if hasattr(self.training_env, 'get_attr'):
                trade_histories = self.training_env.get_attr('trade_return_history')
                for trade_history in trade_histories:
                    if trade_history:
                        for pnl in trade_history:
                            self._record_trade(pnl)
            # Fallback: try direct access for non-vectorized envs
            elif hasattr(self.training_env, 'envs'):
                for env in self.training_env.envs:
                    if hasattr(env, 'trade_return_history'):
                        for pnl in getattr(env, 'trade_return_history', []):
                            self._record_trade(pnl)
        except Exception:
            pass  # Silently ignore if can't access
    
    def _record_trade(self, pnl: float) -> None:
        """Record a trade for metrics calculation."""
        self.trade_pnls.append(pnl)
        self.n_trades += 1
        
        if pnl > 0:
            self.n_wins += 1
            self.total_profit += pnl
        else:
            self.total_loss += abs(pnl)
        
        # Update equity curve
        new_equity = self.equity_curve[-1] * (1 + pnl)
        self.equity_curve.append(new_equity)
        self.peak_equity = max(self.peak_equity, new_equity)
    
    def _log_action_statistics(self) -> None:
        """Log action distribution to detect policy collapse."""
        if len(self.actions_history) < 10:
            return
        
        actions = np.array(list(self.actions_history))
        
        # Action diversity metrics
        action_mean = float(np.mean(actions))
        action_std = float(np.std(actions))
        action_min = float(np.min(actions))
        action_max = float(np.max(actions))
        
        self.logger.record("actions/mean", action_mean)
        self.logger.record("actions/std", action_std)
        self.logger.record("actions/min", action_min)
        self.logger.record("actions/max", action_max)
        
        # Policy collapse detection
        is_collapsed = action_std < 0.05
        self.logger.record("diagnostics/policy_collapsed", int(is_collapsed))
        
        if is_collapsed and self.verbose:
            print(f"⚠️ [CALLBACK] Policy collapse detected! action_std={action_std:.4f}")
    
    def _log_scientific_metrics(self) -> None:
        """Log scientifically-proven performance metrics."""
        if len(self.trade_pnls) < 10:
            return
        
        returns = np.array(list(self.trade_pnls))
        
        # 1. Sharpe Ratio (annualized, assuming 252 trading days)
        sharpe = self._calculate_sharpe(returns)
        self.logger.record("metrics/sharpe_ratio", sharpe)
        
        # 2. Sortino Ratio (downside deviation only)
        sortino = self._calculate_sortino(returns)
        self.logger.record("metrics/sortino_ratio", sortino)
        
        # 3. Calmar Ratio (return / max drawdown)
        calmar = self._calculate_calmar()
        self.logger.record("metrics/calmar_ratio", calmar)
        
        # 4. Maximum Drawdown
        max_dd = self._calculate_max_drawdown()
        self.logger.record("metrics/max_drawdown", max_dd)
        
        # 5. Win Rate
        win_rate = self.n_wins / max(1, self.n_trades)
        self.logger.record("metrics/win_rate", win_rate)
        
        # 6. Profit Factor
        profit_factor = self.total_profit / max(0.01, self.total_loss)
        self.logger.record("metrics/profit_factor", min(10.0, profit_factor))
        
        # 7. Kelly Criterion (optimal position sizing)
        kelly = self._calculate_kelly(returns)
        self.logger.record("metrics/kelly_criterion", kelly)
        
        # 8. Average Trade
        avg_trade = float(np.mean(returns))
        self.logger.record("metrics/avg_trade_return", avg_trade)
        
        if self.verbose and self.num_timesteps % (self.eval_freq * 10) == 0:
            print(f"\n📊 [METRICS] Step {self.num_timesteps}:")
            print(f"   Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | Calmar: {calmar:.3f}")
            print(f"   WinRate: {win_rate:.1%} | PF: {profit_factor:.2f} | Kelly: {kelly:.2%}")
            print(f"   MaxDD: {max_dd:.2%} | AvgTrade: {avg_trade:.4f}")
    
    def _log_training_diagnostics(self) -> None:
        """Log SAC-specific training diagnostics."""
        if hasattr(self.model, 'ent_coef'):
            ent_coef = self.model.ent_coef
            # Handle 'auto_X.X' string format used when auto-tuning
            if isinstance(ent_coef, str):
                # Skip logging if it's auto mode (e.g., 'auto_0.1')
                pass
            elif hasattr(ent_coef, 'item'):
                self.logger.record("diagnostics/entropy_coef", float(ent_coef.item()))
            else:
                try:
                    self.logger.record("diagnostics/entropy_coef", float(ent_coef))
                except (ValueError, TypeError):
                    pass  # Skip if can't convert
        
        if hasattr(self.model, 'log_ent_coef'):
            if self.model.log_ent_coef is not None:
                import torch
                log_ent = self.model.log_ent_coef.detach().cpu().item()
                self.logger.record("diagnostics/log_entropy_coef", log_ent)
        
        # Episode statistics
        if len(self.episode_returns) > 0:
            self.logger.record("episode/mean_return", np.mean(self.episode_returns))
            self.logger.record("episode/std_return", np.std(self.episode_returns))
        
        if len(self.episode_lengths) > 0:
            self.logger.record("episode/mean_length", np.mean(self.episode_lengths))
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe Ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - self.risk_free_rate / 252
        std = np.std(excess_returns)
        if std < 1e-8:
            return 0.0
        return float(np.sqrt(252) * np.mean(excess_returns) / std)
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino Ratio (downside deviation)."""
        if len(returns) < 2:
            return 0.0
        target = self.risk_free_rate / 252
        excess_returns = returns - target
        # [D-A1 FIX] Semi-desvio canônico: RMSE de min(r - target, 0) sobre TODOS os retornos.
        # np.std(returns[returns<0]) usava desvio em torno da média dos negativos (não do target),
        # e descartava as observações acima do target, inflando o Sortino em até 59%.
        downside = np.minimum(returns - target, 0.0)
        downside_rmse = float(np.sqrt(np.mean(downside ** 2)))
        if downside_rmse < 1e-8:
            return 0.0
        return float(np.sqrt(252) * np.mean(excess_returns) / downside_rmse)
    
    def _calculate_calmar(self) -> float:
        """Calculate Calmar Ratio (annual return / max drawdown)."""
        max_dd = self._calculate_max_drawdown()
        if max_dd < 0.01:
            return 0.0
        total_return = self.equity_curve[-1] / self.equity_curve[0] - 1
        return float(total_return / max_dd)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate Maximum Drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return float(np.max(drawdown))
    
    def _calculate_kelly(self, returns: np.ndarray) -> float:
        """Calculate Kelly Criterion for optimal position sizing."""
        if len(returns) < 10:
            return 0.0
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss < 1e-8:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Fractional Kelly (more conservative)
        return float(np.clip(kelly * 0.5, 0, 0.25))
    
    def record_action(self, action: float) -> None:
        """Record an action for analysis."""
        self.actions_history.append(action)


class EntropyRegularizationCallback(BaseCallback):
    """
    🔬 Callback to prevent entropy collapse in SAC.
    
    Based on: Haarnoja et al. (2018) "Soft Actor-Critic: Off-Policy 
    Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    
    If entropy drops too low, adjusts target entropy to encourage exploration.
    """
    
    def __init__(
        self,
        min_entropy_threshold: float = -10.0,
        check_freq: int = 1000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.min_entropy_threshold = min_entropy_threshold
        self.check_freq = check_freq
        self.entropy_warnings = 0
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0:
            return True
        
        # Check entropy coefficient
        if hasattr(self.model, 'log_ent_coef') and self.model.log_ent_coef is not None:
            import torch
            with torch.no_grad():
                log_ent = self.model.log_ent_coef.item()
                ent_coef = np.exp(log_ent)
                
                if log_ent < self.min_entropy_threshold:
                    self.entropy_warnings += 1
                    if self.verbose:
                        print(f"⚠️ [ENTROPY] Low entropy detected: {ent_coef:.6f} (log={log_ent:.2f})")
                    
                    # Reset entropy coefficient to higher value
                    if self.entropy_warnings >= 3:
                        self.model.log_ent_coef.data.fill_(np.log(0.2))
                        if self.verbose:
                            print("🔄 [ENTROPY] Reset entropy coefficient to 0.2")
                        self.entropy_warnings = 0
        
        return True


class ActionDiversityCallback(BaseCallback):
    """
    🔬 Callback to monitor and encourage action diversity.
    
    Detects policy collapse early and can add exploration noise.
    """
    
    def __init__(
        self,
        check_freq: int = 500,
        min_action_std: float = 0.1,
        exploration_noise: float = 0.1,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.min_action_std = min_action_std
        self.exploration_noise = exploration_noise
        self.recent_actions: deque = deque(maxlen=200)
        self.collapse_counter = 0
        
    def _on_step(self) -> bool:
        # Record action
        if hasattr(self, 'locals') and 'actions' in self.locals:
            actions = self.locals['actions']
            if actions is not None:
                self.recent_actions.append(float(np.mean(actions)))
        
        if self.num_timesteps % self.check_freq != 0:
            return True
        
        if len(self.recent_actions) < 50:
            return True
        
        actions_array = np.array(list(self.recent_actions))
        action_std = float(np.std(actions_array))
        
        if action_std < self.min_action_std:
            self.collapse_counter += 1
            if self.verbose:
                print(f"⚠️ [ACTION] Low diversity: std={action_std:.4f} (min={self.min_action_std})")
            
            # Add exploration noise to actor if available
            if self.collapse_counter >= 5 and hasattr(self.model, 'actor'):
                import torch
                with torch.no_grad():
                    for param in self.model.actor.parameters():
                        noise = torch.randn_like(param) * self.exploration_noise
                        param.add_(noise)
                if self.verbose:
                    print(f"🔄 [ACTION] Added exploration noise to actor parameters")
                self.collapse_counter = 0
        else:
            self.collapse_counter = max(0, self.collapse_counter - 1)
        
        return True
