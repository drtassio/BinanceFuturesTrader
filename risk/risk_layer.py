"""
Camada de gestão de risco para BTCUSDT Perpetual Futures.

Funções:
1. Kill switch: pausa o bot quando drawdown > limiar
2. Position sizing: Kelly fracionário (0.25x)
3. Funding cost monitoring: reduz posição quando funding > 0.05%/8h
4. Circuit breaker: bloqueia em flash crashes (vol > 5σ)
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("RiskLayer")


@dataclass
class RiskConfig:
    max_drawdown_intraday: float = 0.05
    max_drawdown_total: float = 0.25
    kelly_fraction: float = 0.25
    max_position_size: float = 0.20
    funding_threshold: float = 0.0005
    flash_crash_sigma: float = 5.0
    cooldown_seconds: int = 86_400


@dataclass
class RiskState:
    is_paused: bool = False
    pause_until: float = 0.0
    high_water_mark: float = 0.0
    current_equity: float = 0.0
    last_action: str = "none"


class RiskLayer:
    def __init__(self, initial_capital: float, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.state = RiskState(
            high_water_mark=initial_capital, current_equity=initial_capital
        )

    def check_kill_switch(self) -> bool:
        now = time.time()
        if self.state.is_paused:
            if now < self.state.pause_until:
                return True
            self.state.is_paused = False
            logger.info("✅ [RISK] Cooldown finalizado. Retomando trading.")
        return False

    def update_equity(self, new_equity: float) -> Dict[str, Any]:
        self.state.current_equity = new_equity
        if new_equity > self.state.high_water_mark:
            self.state.high_water_mark = new_equity

        dd = 1 - (new_equity / max(self.state.high_water_mark, 1e-9))
        result = {"drawdown": dd, "kill_switch_active": False}

        if dd >= self.config.max_drawdown_intraday:
            self.state.is_paused = True
            self.state.pause_until = time.time() + self.config.cooldown_seconds
            self.state.last_action = "kill_switch_intraday"
            result["kill_switch_active"] = True
            logger.error(
                f"🛑 [RISK] KILL SWITCH ATIVADO. DD={dd:.2%}. "
                f"Pausa de {self.config.cooldown_seconds}s."
            )
        return result

    def reset_kill_switch(self, confirmation: str = "") -> bool:
        if confirmation.strip().lower() != "confirmo":
            logger.warning("[RISK] Reset do kill-switch requer confirmacao 'confirmo'.")
            return False
        if not self.state.is_paused:
            return False
        self.state.is_paused = False
        self.state.pause_until = 0.0
        self.state.last_action = "kill_switch_manual_reset"
        logger.info("[RISK] Kill-switch resetado manualmente. Trading liberado.")
        return True

    def calculate_position_size(
        self, win_prob: float, win_loss_ratio: float, current_capital: float
    ) -> float:
        if win_loss_ratio <= 0:
            return 0.0
        kelly_full = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly_fractional = max(0.0, kelly_full * self.config.kelly_fraction)
        size = min(kelly_fractional, self.config.max_position_size)
        return size * current_capital

    def check_funding_risk(self, funding_rate_8h: float) -> float:
        abs_funding = abs(funding_rate_8h)
        if abs_funding > self.config.funding_threshold * 2:
            return 0.0
        if abs_funding > self.config.funding_threshold:
            return 0.5
        return 1.0

    def check_flash_crash(self, returns_window: pd.Series) -> bool:
        if len(returns_window) < 100:
            return False
        std = returns_window.iloc[:-1].std()
        last = abs(returns_window.iloc[-1])
        if last > self.config.flash_crash_sigma * std:
            logger.warning(
                f"⚠️ [RISK] Flash crash detectado: |ret|={last:.4f} "
                f"({last / std:.1f}σ). Bloqueando entradas."
            )
            return True
        return False
