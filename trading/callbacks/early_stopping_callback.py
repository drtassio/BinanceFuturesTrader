# -----------------------------------------------------------------------------
# ARQUIVO: trading/callbacks/early_stopping_callback.py
# -----------------------------------------------------------------------------

"""
Callback personalizado para Early Stopping (Parada Antecipada) em Stable Baselines3.

Este callback monitora o desempenho do agente durante o treinamento e interrompe
o processo se a recompensa média não melhorar por um determinado número de avaliações.
"""

import numpy as np
import os
from typing import Optional, Dict, Any, List
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


class EarlyStoppingCallback(BaseCallback):
    """
    Callback personalizado para Early Stopping baseado em avaliação periódica (EvalCallback).
    Para ser usado como callback filho em EvalCallback via 'callback_after_eval'.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: int = 0):
        """
        :param patience: Número de avaliações consecutivas sem melhora antes de parar.
        :param min_delta: Mínima diferença de recompensa para considerar uma melhora.
        :param verbose: Nível de verbosidade.
        """
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0

        self.eval_freq: Optional[int] = None
        self.eval_log_path: Optional[str] = None

        if self.verbose > 0:
            print(f"[EarlyStoppingCallback] Inicializado com patience={patience}, min_delta={min_delta}")

    def _init_callback(self) -> None:
        """
        Chamado uma vez quando o callback é registrado.
        Verifica se está sendo usado como filho de um EvalCallback.
        """
        if self.parent is not None and isinstance(self.parent, EvalCallback):
            self.eval_freq = self.parent.eval_freq
            self.eval_log_path = self.parent.log_path
            if self.verbose > 0:
                print(f"[EarlyStoppingCallback] Detectado uso por EvalCallback. Frequência de avaliação: {self.eval_freq}.")
        else:
            if self.verbose > 0:
                print("⚠️ [EarlyStoppingCallback] Não detectado uso por EvalCallback. O early stopping pode não funcionar corretamente.")

    def _on_step(self) -> bool:
        """Chamado a cada passo. Não é usado aqui."""
        return True

    def _on_rollout_end(self) -> None:
        """Chamado ao fim de cada rollout. Não utilizado neste callback."""
        pass

    def _on_training_end(self) -> None:
        """Chamado ao fim do treinamento."""
        if self.verbose > 0:
            print("[EarlyStoppingCallback] Treinamento finalizado.")

    def _safe_mean(self, values: List[float]) -> float:
        """Calcula a média de uma lista, retornando 0.0 se a lista estiver vazia."""
        return np.mean(values) if len(values) > 0 else 0.0

    def _on_evaluation_end(self, mean_reward: float, logs: Dict[str, Any]) -> bool:
        """
        Lógica de Early Stopping com base na recompensa média da avaliação.
        Retorna False se o treinamento deve ser interrompido.
        """
        current_mean_reward = mean_reward

        if self.verbose > 0:
            print(f"[EarlyStopping] Avaliação finalizada. Recompensa média: {current_mean_reward:.4f}, melhor: {self.best_mean_reward:.4f}")

        if current_mean_reward > self.best_mean_reward + self.min_delta:
            self.best_mean_reward = current_mean_reward
            self.no_improvement_count = 0
            if self.verbose > 0:
                print(f"[EarlyStopping] Nova melhor recompensa média: {self.best_mean_reward:.4f}. Resetando contador.")
        else:
            self.no_improvement_count += 1
            if self.verbose > 0:
                print(f"[EarlyStopping] Sem melhora. Contador: {self.no_improvement_count}/{self.patience}")

        if self.no_improvement_count >= self.patience:
            if self.verbose > 0:
                print(f"[EarlyStopping] Interrompendo treinamento após {self.patience} avaliações sem melhora.")
            return False  # Para o treinamento

        return True  # Continua o treinamento
