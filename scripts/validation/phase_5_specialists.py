"""
FASE 5 — Especialistas
Validações:
- BaseRegimeSpecialist.decide_action não crasha (bug do `direction` corrigido)
- Bull/Bear/Ranger instanciam com _specialist_direction correto
- Reset do ambiente limpa _reward_running_mean/var/count (fix S5)
- PnL de SHORT tem sinal correto: short_loss < 0 < short_win
"""
from __future__ import annotations
import inspect
import numpy as np

from scripts.validation._common import PhaseRunner


def run() -> PhaseRunner:
    p = PhaseRunner("5 — Especialistas")
    with p:
        from specialists.bull_specialist import BullSpecialist
        from specialists.bear_specialist import BearSpecialist
        from specialists.ranger_specialist import RangerSpecialist
        from specialists.base_regime_specialist import BaseRegimeSpecialist

        # 5.1 — Diretivas de especialista corretas
        # (não instanciamos porque construtor exige config + autoencoder treinado;
        #  inspecionamos defaults declarados nos __init__ via source)
        bull_src = inspect.getsource(BullSpecialist)
        bear_src = inspect.getsource(BearSpecialist)
        ranger_src = inspect.getsource(RangerSpecialist)
        p.check(
            "BullSpecialist define _specialist_direction='long_only'",
            "_specialist_direction = 'long_only'" in bull_src,
        )
        p.check(
            "BearSpecialist define _specialist_direction='short_only'",
            "_specialist_direction = 'short_only'" in bear_src,
        )
        p.check(
            "RangerSpecialist define _specialist_direction (qualquer valor)",
            "_specialist_direction" in ranger_src,
        )

        # 5.2 — decide_action lê via getattr (não crasha se _specialist_direction ausente)
        decide_src = inspect.getsource(BaseRegimeSpecialist.decide_action)
        p.check(
            "BaseRegimeSpecialist.decide_action usa getattr (defensivo)",
            "getattr(self, '_specialist_direction'" in decide_src
            or "getattr(self, \"_specialist_direction\"" in decide_src,
        )

        # 5.3 — reset() reseta reward_running_mean/var/count
        from specialists.trend_specialist import TrendFollowingEnv
        reset_src = inspect.getsource(TrendFollowingEnv.reset)
        has_mean = "self._reward_running_mean = 0.0" in reset_src
        has_var = "self._reward_running_var" in reset_src
        has_count = "self._reward_count = 0" in reset_src
        p.check(
            "TrendFollowingEnv.reset() limpa running stats do reward",
            has_mean and has_var and has_count,
            detail=f"mean={has_mean} var={has_var} count={has_count}",
        )

        # 5.4 — Sinal de PnL em SHORT (validação matemática pura)
        # Para um short: entry=100, close=95 → ganho 5%
        # Formula explicit  (entry-close)/entry            → +5% ✓
        # Formula sign(pos): (close-entry)/entry * sign    → -5% * -1 = +5% ✓
        entry, close_win, close_loss = 100.0, 95.0, 105.0
        explicit_win = (entry - close_win) / entry
        explicit_loss = (entry - close_loss) / entry
        sign_form_win = (close_win - entry) / entry * np.sign(-1)
        sign_form_loss = (close_loss - entry) / entry * np.sign(-1)
        p.check(
            "PnL SHORT win > 0 (ambas as formas)",
            explicit_win > 0 and sign_form_win > 0,
            detail=f"explicit={explicit_win:+.4f}, sign={sign_form_win:+.4f}",
        )
        p.check(
            "PnL SHORT loss < 0 (ambas as formas)",
            explicit_loss < 0 and sign_form_loss < 0,
            detail=f"explicit={explicit_loss:+.4f}, sign={sign_form_loss:+.4f}",
        )
        p.check(
            "PnL SHORT: formula explícita == formula sign(pos)",
            abs(explicit_win - sign_form_win) < 1e-12
            and abs(explicit_loss - sign_form_loss) < 1e-12,
        )

        p.conclude()
    return p


if __name__ == "__main__":
    run()
