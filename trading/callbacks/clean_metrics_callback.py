# -----------------------------------------------------------------------------
# ARQUIVO: trading/callbacks/clean_metrics_callback.py
# -----------------------------------------------------------------------------
"""
📊 Clean Metrics Callback - Logging Conciso para Treinamento

Imprime apenas métricas essenciais a cada N steps/episódios:
- Métricas de Trading (Win Rate, Profit Factor, Sharpe)
- Matriz de Confusão (Precisão por tipo de ação)
- Progresso de Otimização Optuna

Referências:
- de Prado (2018) - "Advances in Financial Machine Learning"
- Chan (2013) - "Algorithmic Trading"
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback


class CleanMetricsCallback(BaseCallback):
    """
    📊 Callback de métricas limpas - Log conciso a cada N steps.
    
    Imprime:
    - Métricas de performance (Sharpe, Sortino, Win Rate, PF)
    - Matriz de confusão de ações (Long/Short/Hold accuracy)
    - Progresso do episódio
    """
    
    def __init__(
        self,
        print_freq: int = 5000,
        verbose: int = 1,
        show_confusion_matrix: bool = True
    ):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.show_confusion_matrix = show_confusion_matrix
        
        # Contadores para matriz de confusão
        self.action_counts = defaultdict(int)  # Total de cada ação
        self.action_outcomes = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0})
        
        # Métricas agregadas
        self.episode_returns: List[float] = []
        self.trade_pnls: List[float] = []
        self.trade_durations: List[int] = []
        
        # Contadores
        self.n_episodes = 0
        self.n_trades = 0
        self.n_wins = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
    def _on_step(self) -> bool:
        # 1. NOVO: Coleta trades pendentes a cada passo para tempo-real
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                self._collect_env_metrics(env)

        # 2. Rastreia a ação atual para a contagem total (candles)
        if 'actions' in self.locals:
            actions = self.locals['actions']
            for action_val in actions:
                if isinstance(action_val, np.ndarray):
                    # Lógica TrendSpecialist
                    val = action_val[0]
                    if val > 0.3: act_idx = 1
                    elif val < -0.3: act_idx = 2
                    else: act_idx = 0
                else:
                    act_idx = int(action_val)
                
                # Registra apenas a ocorrência (hold/base)
                action_map = {0: 'hold', 1: 'long', 2: 'short'}
                action_name = action_map.get(act_idx, 'unknown')
                self.action_counts[action_name] += 1
        
        # 3. Log a cada print_freq steps
        if self.num_timesteps % self.print_freq == 0 and self.num_timesteps > 0:
            self._print_clean_metrics()
        return True
    
    def _on_rollout_end(self) -> None:
        """Coleta métricas do ambiente ao final do rollout."""
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                self._collect_env_metrics(env)
    
    def _collect_env_metrics(self, env) -> None:
        """Coleta métricas de um ambiente de forma resiliente."""
        # Unpack environments if they are VecEnvs (SB3 standard)
        try:
            u_env = env.unwrapped if hasattr(env, 'unwrapped') else env
            # 1. Verifica se o episódio acabou para o retorno
            # Algumas envs SB3 usam 'dones' ou reset automático
            if hasattr(u_env, 'current_step') and u_env.current_step == 0:
                if hasattr(u_env, 'net_worth') and hasattr(u_env, 'initial_balance'):
                    # Retorno percentual do episódio
                    ep_return = (u_env.net_worth / u_env.initial_balance) - 1
                    # Sanity check: não registrar se o env acabou de começar sem dados
                    if hasattr(u_env, 'steps_in_episode') and u_env.steps_in_episode > 1:
                        self.episode_returns.append(ep_return)
                        self.n_episodes += 1
        
            # 2. Coleta histórico de trades
            # Checamos tanto trade_return_history quanto o PnL do trade atual se ele fechou
            trades_to_process = []
            if hasattr(u_env, 'trade_return_history') and len(u_env.trade_return_history) > 0:
                trades_to_process.extend(u_env.trade_return_history)
                u_env.trade_return_history = [] # Limpa para o próximo passo
            
            for pnl in trades_to_process:
                # Determina o tipo de ação baseado na posição do trade recém-fechado.
                # [FIX RANGER] last_action_type é zerado no reset de posição (trend_specialist
                # L2661/2746), então quando lemos aqui já é 0. Usamos last_closed_trade_side
                # que o env preserva ANTES de zerar. Funciona para Bull, Bear E Ranger.
                act_type = 'unknown'

                # 1ª tentativa: side preservado do último trade fechado (mais confiável)
                closed_side = int(getattr(u_env, 'last_closed_trade_side', 0) or 0)
                if closed_side > 0: act_type = 'long'
                elif closed_side < 0: act_type = 'short'

                # 2ª tentativa (trade ainda aberto ou campo ausente): posição atual
                if act_type == 'unknown' and hasattr(u_env, 'last_action_type'):
                    if u_env.last_action_type > 0: act_type = 'long'
                    elif u_env.last_action_type < 0: act_type = 'short'

                # 3ª tentativa (fallback legado p/ especialistas single-side): nome
                if act_type == 'unknown':
                    spec_name = str(getattr(u_env, 'specialist_name', '')).lower()
                    if "bull" in spec_name: act_type = 'long'
                    elif "bear" in spec_name: act_type = 'short'

                self._record_trade(pnl, act_type)
        except Exception:
            pass # Silencioso durante o loop de treino

    def _record_trade(self, pnl: float, action_type: str) -> None:
        """Registra o desfecho de um trade na matriz."""
        self.trade_pnls.append(pnl)
        self.n_trades += 1
        
        # Registra no PnL acumulado da ação
        self.action_outcomes[action_type]['pnl'] += pnl
        
        if pnl > 0:
            self.n_wins += 1
            self.total_profit += pnl
            self.action_outcomes[action_type]['wins'] += 1
        else:
            self.total_loss += abs(pnl)
            self.action_outcomes[action_type]['losses'] += 1
        
        # NOTA: Não incrementamos action_counts aqui pois ele já é incrementado 
        # a cada candle no _on_step para mostrar "Duração/Ocorrência".
    
    def record_action_decision(self, action: int, outcome: str, pnl: float = 0.0) -> None:
        """
        Registra decisão de ação para matriz de confusão.
        
        Args:
            action: 0=Hold, 1=Long, 2=Short
            outcome: 'win', 'loss', 'hold'
            pnl: PnL do trade se aplicável
        """
        action_map = {0: 'hold', 1: 'long', 2: 'short'}
        action_name = action_map.get(action, 'unknown')
        
        self.action_counts[action_name] += 1
        if outcome == 'win':
            self.action_outcomes[action_name]['wins'] += 1
            self.action_outcomes[action_name]['pnl'] += pnl
        elif outcome == 'loss':
            self.action_outcomes[action_name]['losses'] += 1
            self.action_outcomes[action_name]['pnl'] += pnl
    
    def _print_clean_metrics(self) -> None:
        """Imprime métricas limpas e concisas."""
        print("\n" + "="*70)
        print(f"📊 MÉTRICAS | Step: {self.num_timesteps:,}")
        print("="*70)
        
        # 1. Performance Metrics
        if len(self.trade_pnls) >= 5:
            returns = np.array(self.trade_pnls[-100:])  # Últimos 100 trades
            
            win_rate = self.n_wins / max(1, self.n_trades)
            profit_factor = self.total_profit / max(0.01, self.total_loss)
            sharpe = self._calc_sharpe(returns)
            sortino = self._calc_sortino(returns)
            avg_trade = np.mean(returns) * 100  # Em %
            max_dd = self._calc_max_dd()
            
            print(f"\n📈 PERFORMANCE:")
            print(f"   Win Rate: {win_rate:.1%} | PF: {profit_factor:.2f} | Trades: {self.n_trades}")
            print(f"   Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | MaxDD: {max_dd:.2%}")
            print(f"   Avg Trade: {avg_trade:+.3f}%")
        else:
            print(f"\n📈 PERFORMANCE: Aguardando trades (atual: {self.n_trades})")
        
        # 2. Confusion Matrix
        if self.show_confusion_matrix:
            self._print_confusion_matrix()
        
        # 3. Episode Progress
        if len(self.episode_returns) > 0:
            recent_returns = self.episode_returns[-10:]
            avg_ep_return = np.mean(recent_returns) * 100
            print(f"\n🎯 EPISÓDIOS: {self.n_episodes} | Retorno Médio: {avg_ep_return:+.2f}%")
        
        print("="*70 + "\n")
    
    def _print_confusion_matrix(self) -> None:
        """Imprime matriz de confusão de ações."""
        import logging
        logger = logging.getLogger("CleanMetrics")
        logger.info(f"📋 MATRIZ DE CONFUSÃO (Ações):")
        logger.info(f"   {'Ação':<10} {'Total':>8} {'Wins':>8} {'Loss':>8} {'WR':>8} {'PnL':>10}")
        header_sep = "-" * 52
        logger.info(f"   {header_sep}")
        
        # [FIX] Inclui 'unknown' para não perder silenciosamente trades sem side atribuído
        for action in ['long', 'short', 'hold', 'unknown']:
            total = self.action_counts.get(action, 0)
            outcome = self.action_outcomes.get(action, {'wins': 0, 'losses': 0, 'pnl': 0.0})
            wins = outcome.get('wins', 0)
            losses = outcome.get('losses', 0)
            pnl = outcome.get('pnl', 0.0)

            # Só pula se não houve NENHUM evento (nem decisão nem outcome)
            if total == 0 and wins == 0 and losses == 0:
                continue

            wr = wins / max(1, wins + losses)
            logger.info(f"   {action.upper():<10} {total:>8} {wins:>8} {losses:>8} {wr:>7.1%} {pnl:>+10.4f}")
    
    def _calc_sharpe(self, returns: np.ndarray) -> float:
        """Calcula Sharpe Ratio."""
        if len(returns) < 2:
            return 0.0
        std = np.std(returns)
        if std < 1e-8:
            return 0.0
        return float(np.sqrt(252) * np.mean(returns) / std)
    
    def _calc_sortino(self, returns: np.ndarray) -> float:
        """Calcula Sortino Ratio."""
        if len(returns) < 2:
            return 0.0
        downside = returns[returns < 0]
        if len(downside) < 2:
            return 0.0
        downside_std = np.std(downside)
        if downside_std < 1e-8:
            return 0.0
        return float(np.sqrt(252) * np.mean(returns) / downside_std)
    
    def _calc_max_dd(self) -> float:
        """Calcula Maximum Drawdown a partir dos retornos por trade.

        Usa janela deslizante dos últimos 200 trades para refletir
        a performance RECENTE, evitando que a fase de exploração inicial
        do SAC (primeiros ~60k steps) contamine permanentemente o MaxDD.

        Usa trade_pnls (não episode_returns) porque episode_returns é capturado
        após o reset do env — nesse ponto net_worth já voltou ao initial_balance,
        resultando em retorno 0 e MaxDD sempre 0.
        """
        if len(self.trade_pnls) < 2:
            return 0.0
        # Janela deslizante: últimos 200 trades (consistente com Sharpe/Sortino [-100])
        recent_pnls = self.trade_pnls[-200:]
        equity = np.cumprod(1 + np.array(recent_pnls))
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        return float(np.max(dd))


class OptunaProgressCallback:
    """
    📊 Callback para logging limpo durante otimização Optuna.
    
    Uso:
        study.optimize(objective, n_trials=100, callbacks=[OptunaProgressCallback()])
    """
    
    def __init__(self, print_freq: int = 10):
        self.print_freq = print_freq
        self.best_value = float('inf')
        self.best_params = {}
    
    def __call__(self, study, trial) -> None:
        """Chamado após cada trial."""
        trial_num = trial.number + 1
        
        # Atualiza melhor valor
        if trial.value is not None and trial.value < self.best_value:
            self.best_value = trial.value
            self.best_params = trial.params.copy()
        
        # Log a cada print_freq trials
        if trial_num % self.print_freq == 0:
            print(f"\n{'='*60}")
            print(f"🔬 OPTUNA | Trial {trial_num}/{study.n_trials if hasattr(study, 'n_trials') else '?'}")
            print(f"{'='*60}")
            print(f"   Último Trial: {trial.value:.4f}")
            print(f"   Melhor Trial: {self.best_value:.4f} (Trial #{study.best_trial.number if study.best_trial else '?'})")
            
            # Parâmetros do melhor trial
            if self.best_params:
                print(f"\n   📋 Melhores Params:")
                for key, value in list(self.best_params.items())[:5]:  # Top 5 params
                    if isinstance(value, float):
                        print(f"      {key}: {value:.6f}")
                    else:
                        print(f"      {key}: {value}")
            print(f"{'='*60}\n")


class EpochMetricsLogger:
    """
    📊 Logger de métricas por época para autoencoder e modelos.
    
    Uso:
        logger = EpochMetricsLogger()
        for epoch in range(n_epochs):
            ...
            logger.log_epoch(epoch, train_loss, val_loss, extra_metrics)
    """
    
    def __init__(self, print_freq: int = 5, total_epochs: int = 100):
        self.print_freq = print_freq
        self.total_epochs = total_epochs
        self.history = defaultdict(list)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        extra_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Loga métricas de uma época.
        
        Args:
            epoch: Número da época (0-indexed)
            train_loss: Loss de treino
            val_loss: Loss de validação
            extra_metrics: Métricas adicionais (opcional)
        """
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        
        if extra_metrics:
            for key, value in extra_metrics.items():
                self.history[key].append(value)
        
        # Imprime a cada print_freq épocas
        if (epoch + 1) % self.print_freq == 0 or epoch == 0:
            self._print_epoch_summary(epoch + 1)
    
    def _print_epoch_summary(self, epoch: int) -> None:
        """Imprime resumo da época."""
        train_loss = self.history['train_loss'][-1]
        val_loss = self.history['val_loss'][-1]
        
        # Calcula trend
        if len(self.history['val_loss']) > 5:
            recent = self.history['val_loss'][-5:]
            trend = "↓" if recent[-1] < recent[0] else "↑" if recent[-1] > recent[0] else "→"
        else:
            trend = "→"
        
        progress = epoch / self.total_epochs * 100
        bar_len = 20
        filled = int(bar_len * epoch / self.total_epochs)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        print(f"[{bar}] {progress:5.1f}% | Epoch {epoch:3d}/{self.total_epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} {trend}")
        
        # Extra metrics
        for key in self.history:
            if key not in ['train_loss', 'val_loss'] and self.history[key]:
                value = self.history[key][-1]
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")


def format_optuna_trial_log(trial_number: int, value: float, params: Dict[str, Any]) -> str:
    """
    Formata log de trial Optuna de forma concisa.
    
    Returns:
        String formatada para log
    """
    # Seleciona parâmetros mais importantes
    key_params = ['latent_dim', 'learning_rate', 'batch_size', 'gru_hidden']
    param_str = " | ".join([
        f"{k}={params[k]:.4g}" if isinstance(params.get(k), float) else f"{k}={params.get(k)}"
        for k in key_params if k in params
    ])
    
    return f"Trial {trial_number:3d}: Loss={value:.4f} | {param_str}"
