"""
Camada de execução com slippage realista e cálculo de taxa Binance.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from utils.logger import get_logger

logger = get_logger("ExecutionLayer")


@dataclass
class ExecutionConfig:
    taker_fee: float = 0.0004
    maker_fee: float = 0.0002
    base_slippage_bps: float = 2.0
    impact_coefficient: float = 0.5
    max_acceptable_slippage_bps: float = 10.0


class ExecutionLayer:
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

    def estimate_slippage_bps(
        self, order_size_usd: float, avg_volume_usd_per_min: float
    ) -> float:
        if avg_volume_usd_per_min <= 0:
            return self.config.max_acceptable_slippage_bps
        impact_ratio = order_size_usd / avg_volume_usd_per_min
        slippage = (
            self.config.base_slippage_bps
            + self.config.impact_coefficient * impact_ratio * 100
        )
        return min(slippage, self.config.max_acceptable_slippage_bps * 2)

    def simulate_execution(
        self,
        side: str,
        size_usd: float,
        mid_price: float,
        avg_volume_usd_per_min: float,
        is_maker: bool = False,
    ) -> Dict[str, Any]:
        slippage_bps = self.estimate_slippage_bps(size_usd, avg_volume_usd_per_min)

        if slippage_bps > self.config.max_acceptable_slippage_bps:
            return {
                "executed": False,
                "reason": f"Slippage {slippage_bps:.1f} bps excede limite",
            }

        slippage_factor = slippage_bps / 10_000.0
        if side == "buy":
            executed_price = mid_price * (1 + slippage_factor)
        else:
            executed_price = mid_price * (1 - slippage_factor)

        fee_rate = self.config.maker_fee if is_maker else self.config.taker_fee
        fee_usd = size_usd * fee_rate

        return {
            "executed": True,
            "executed_price": executed_price,
            "slippage_bps": slippage_bps,
            "fee_usd": fee_usd,
            "total_cost_bps": slippage_bps + fee_rate * 10_000,
        }
