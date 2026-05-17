"""
FASE 1 — Imports & Smoke
Verifica que todos os módulos críticos importam, que enums/dataclasses
estão acessíveis, e que os 4 bugs P0 que corrigimos não regrediram.
"""
from __future__ import annotations
from scripts.validation._common import PhaseRunner


CRITICAL_MODULES = [
    "config.settings",
    "models.trade_schema",
    "utils.purged_cv",
    "utils.physics_sensors",
    "feature_engineering.main",
    "feature_engineering.crypto_regime_detector",
    "feature_engineering.scientific_data_processor",
    "feature_engineering.temporal_autoencoder",
    "feature_engineering.native_indicators",
    "learning.profitability_predictor",
    "learning.walk_forward_validator",
    "learning.regime_switch",
    "specialists.trend_specialist",
    "specialists.base_regime_specialist",
    "specialists.bull_specialist",
    "specialists.bear_specialist",
    "specialists.ranger_specialist",
    "trading.ai_controller",
    "trading.portfolio",
    "trading.risk_manager",
    "trading.stress_tester",
    "trading.backtester",
    "governance.drift_detector",
    "governance.learning_monitor",
]


def run() -> PhaseRunner:
    p = PhaseRunner("1 — Imports & Smoke")
    with p:
        # 1.1 — Todos os módulos importam
        failed = []
        for mod in CRITICAL_MODULES:
            try:
                __import__(mod)
            except Exception as e:
                failed.append((mod, type(e).__name__, str(e)[:200]))
        p.check(
            f"importa {len(CRITICAL_MODULES)} módulos críticos",
            len(failed) == 0,
            detail=f"{len(failed)} falharam: {failed[:3]}" if failed else "todos OK",
            value=failed,
        )

        # 1.2 — Bug P0 #1: base_regime_specialist.direction
        from specialists.base_regime_specialist import BaseRegimeSpecialist
        import inspect
        src = inspect.getsource(BaseRegimeSpecialist.decide_action)
        p.check(
            "BaseRegimeSpecialist.decide_action usa self._specialist_direction",
            "_specialist_direction" in src,
            detail="bug P0 #1 corrigido",
        )

        # 1.3 — Bug P0 #2: quantum_metrics computado antes do loop
        from trading.ai_controller import AIController
        src = inspect.getsource(AIController.generate_trading_decision)
        first_use = src.find("quantum_metrics.get")
        first_assign = src.find("quantum_metrics = get_market_chaos_metrics")
        p.check(
            "ai_controller: quantum_metrics atribuído ANTES do primeiro uso",
            first_assign != -1 and first_assign < first_use,
            detail=f"assign@{first_assign} < use@{first_use}",
        )

        # 1.4 — Bug P0 #3: run_bot timedelta sem import local
        from pathlib import Path
        run_bot_src = (Path(__file__).resolve().parents[2] / "run_bot.py").read_text()
        bad = "from datetime import datetime, timedelta, timezone" in run_bot_src and run_bot_src.count(
            "from datetime import datetime, timedelta, timezone"
        ) > 1
        p.check(
            "run_bot.py: sem import local de timedelta que sombreava o topo",
            not bad,
            detail="bug P0 #3 corrigido (UnboundLocalError)",
        )

        # 1.5 — Bug P0 #4: stress_tester importa Action/OrderSide/Trade
        from trading import stress_tester as st
        has_action = hasattr(st, "Action")
        has_orderside = hasattr(st, "OrderSide")
        has_trade = hasattr(st, "Trade")
        p.check(
            "stress_tester: Action/OrderSide/Trade importados",
            has_action and has_orderside and has_trade,
            detail=f"Action={has_action} OrderSide={has_orderside} Trade={has_trade}",
        )

        # 1.6 — Bug P0 #5: profitability_predictor.train_model implementado
        from learning.profitability_predictor import ProfitabilityPredictor
        src = inspect.getsource(ProfitabilityPredictor.train_model)
        is_stub = "if X is None: return 0" in src and "_prepare_training_data" not in src
        p.check(
            "profitability_predictor.train_model: implementação real (não stub)",
            not is_stub,
            detail="bug P0 #5 corrigido",
        )

        p.conclude()
    return p


if __name__ == "__main__":
    run()
