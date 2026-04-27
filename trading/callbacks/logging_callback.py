# trading/callbacks/logging_callback.py

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from utils.logger import get_logger

logger = get_logger("RLTraining")

class LoggingCallback(BaseCallback):
    """
    Um callback da Stable Baselines 3 que loga o progresso do treinamento
    para ser exibido na UI.
    """
    def __init__(self, check_freq: int, verbose: int = 0):
        super(LoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Tenta obter a recompensa média do buffer de informações do episódio
            if self.model.ep_info_buffer:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                logger.info(f"🤖 [TREINO RL] Timesteps: {self.num_timesteps} | Recompensa Média (últimos 100 ep): {mean_reward:.2f}")
            else:
                 logger.info(f"🤖 [TREINO RL] Timesteps: {self.num_timesteps}")
        return True