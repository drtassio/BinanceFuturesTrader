# -----------------------------------------------------------------------------
# ARQUIVO: trading/ai_controller.py
# -----------------------------------------------------------------------------
"""Decisao ao vivo: o espelho dos agentes treinados (LIVE_POLICY=agent_mirror).

A cada candle de 15m fechado, cada agente aprovado (e os de diagnostico, so na
testnet) e repassado pelo mesmo ambiente do treino sobre os ultimos candles
fechados (trading/agent_mirror.py). A conta copia a posicao que o agente tem no
fim desse repasse, mas so no candle em que ele entra; a saida, o stop e o
tamanho sao os do ambiente. Nao ha votacao, regime, SHAP nem adaptacao online:
o bot faz o que foi medido no backtest e aprovado.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from config.settings import AIConfig, TradingConfig
from models.trade_schema import Action, Signal
from utils.logger import get_logger

logger = get_logger("AIController")


class AIController:
    def __init__(self, config: AIConfig, trading_config: TradingConfig, system_state: Dict[str, Any]):
        self.config_ai = config
        self.config_trading = trading_config
        self.system_state = system_state
        self.live_policy = str(getattr(trading_config, 'LIVE_POLICY', 'agent_mirror')).strip().lower()
        if self.live_policy != 'agent_mirror':
            raise ValueError("LIVE_POLICY deve ser agent_mirror (o modo por votacao 'sac' foi removido)")

        self.feature_pipeline = None
        self.portfolio = None
        self.risk_manager = None
        self.ai_monitor = None
        self.tape_engine = None
        self.onchain_engine = None
        self.data_provider = None

        self.is_trained = False
        self.teacher_ready = False
        self.mirror_agents: Dict[str, Any] = {}
        self.mirror_diagnostic = []
        self.mirror_history = None
        self.mirror_paths: Dict[str, pd.DataFrame] = {}
        self.mirror_view: Optional[Dict[str, Any]] = None
        self.mirror_stop_target = None
        self._teacher_cache = None
        self._teacher_last_order = None

    def set_feature_pipeline(self, pipeline): self.feature_pipeline = pipeline
    def set_portfolio(self, portfolio): self.portfolio = portfolio
    def set_risk_manager(self, manager): self.risk_manager = manager
    def set_ai_monitor(self, monitor): self.ai_monitor = monitor
    def set_tape_engine(self, engine): self.tape_engine = engine
    def set_onchain_engine(self, engine): self.onchain_engine = engine
    def set_data_provider(self, provider): self.data_provider = provider

    def _mirror_to_signal(self, shadows, bar, policy_name: str, label: str) -> Signal:
        """Order that makes the account hold what the replayed environments hold."""
        import time as _time
        from trading import teacher_policy as teacher

        symbol = self.config_trading.PRIMARY_PAIR

        def hold(reason, **extra):
            if self.mirror_view is not None:
                self.mirror_view["reason"] = reason
            return Signal(symbol=symbol, action=Action.HOLD, confidence=0.0,
                          explanation={"reason": reason, "specialist": label,
                                       "regime": label.upper(), "policy": policy_name, **extra})

        position = self.portfolio.positions.get(symbol)
        live_side = int(np.sign(position.quantity)) if position is not None and position.quantity else 0
        decision = teacher.mirror(shadows, live_side)
        details = {"bar": str(bar), "shadows": {s.agent: s.side for s in shadows}}
        # What the terminal panel shows (run_bot.log_mirror_panel): each agent's
        # simulated trade, the account and what the bot does about them.
        history = getattr(self, "mirror_history", None)
        close = None
        if history is not None and not history.frame.empty and "close" in history.frame:
            close = float(history.frame["close"].iloc[-1])
        self.mirror_view = {"bar": bar, "close": close, "shadows": list(shadows), "account_side": live_side,
                            "account_qty": abs(float(position.quantity)) if live_side else 0.0,
                            "action": decision.action, "reason": decision.reason,
                            "diagnostic": list(getattr(self, "mirror_diagnostic", []))}
        # The account holds the environment's position: hand its current stop
        # to the execution engine, which moves the exchange stop when it tightens.
        self.mirror_stop_target = None
        if (decision.action == "hold" and live_side != 0 and decision.shadow is not None
                and decision.shadow.side == live_side and decision.shadow.stop_price):
            self.mirror_stop_target = (symbol, live_side, abs(float(position.quantity)),
                                       float(decision.shadow.stop_price))
        if decision.action == "hold":
            return hold(decision.reason, **details)

        key = (bar, decision.action, decision.side)
        if self._teacher_last_order and self._teacher_last_order[0] == key \
                and _time.monotonic() - self._teacher_last_order[1] < 180:
            return hold("ordem desta barra ja enviada; aguardando execucao", **details)

        cfg = self.config_trading
        action = Action.BUY if decision.side > 0 else Action.SELL
        explanation = {"reason": decision.reason, "specialist": label, "regime": label.upper(),
                       "policy": policy_name, **details}
        if decision.action == "close":
            # position_size_pct >= 0.9 faz o ExecutionEngine fechar a quantidade
            # exata com reduceOnly.
            signal = Signal(symbol=symbol, action=action, confidence=1.0, position_size_pct=1.0,
                            leverage=float(getattr(position, 'leverage', 1.0) or 1.0),
                            stop_loss=0.0, take_profit=0.0, explanation=explanation)
        else:
            shadow = decision.shadow
            fraction = float(shadow.notional_fraction)
            # O ambiente limita o nocional, nao a margem. Na corretora a margem
            # por posicao e limitada (MAX_POSITION_SIZE_PERCENT), entao a
            # alavancagem da ordem e a menor que acomoda o MESMO nocional do
            # backtest; o risco continua sendo nocional x distancia do stop.
            max_margin = float(cfg.MAX_POSITION_SIZE_PERCENT) * 0.95
            leverage = float(np.clip(np.ceil(fraction / max_margin), max(1.0, float(cfg.MIN_LEVERAGE_PER_TRADE)),
                                     float(cfg.MAX_LEVERAGE_PER_TRADE)))
            size_pct = float(np.clip(fraction / leverage, 0.0, max_margin))
            # A saida real e a do ambiente, avaliada no fechamento da barra. O
            # stop na corretora so cobre o bot parado: fica ao dobro da
            # distancia do stop do ambiente.
            stop_distance = abs((shadow.stop_price or 0.0) - shadow.entry_price) / max(shadow.entry_price, 1e-9)
            catastrophe = float(np.clip(2.0 * stop_distance, 0.01, 0.20))
            explanation.update(notional_fraction=fraction, env_stop_price=shadow.stop_price,
                               env_entry_price=shadow.entry_price)
            signal = Signal(symbol=symbol, action=action, confidence=1.0, position_size_pct=size_pct,
                            leverage=leverage, stop_loss=catastrophe, take_profit=0.0, explanation=explanation)

        approved, reason = self.risk_manager.check_trade_approval(signal)
        if not approved:
            return hold("vetado pelo risco: %s" % reason, **details)
        self._teacher_last_order = (key, _time.monotonic())
        logger.info("[%s] %s %s | margem %.1f%% x %.0fx | stop de catastrofe %.2f%% | %s", label,
                    decision.action, action.value, signal.position_size_pct * 100, signal.leverage,
                    (signal.stop_loss or 0.0) * 100, decision.reason)
        return signal

    async def prepare_agent_mirror(self) -> bool:
        """Load the approved specialists and make sure the replay history is complete."""
        from trading import agent_mirror as mirror

        model_dir = Path(str(self.config_ai.MODEL_DIR))
        names = [a.strip() for a in str(getattr(self.config_trading, 'LIVE_AGENTS', 'bull')).split(',') if a.strip()]
        diagnostic = [a.strip() for a in str(getattr(self.config_trading, 'TESTNET_DIAGNOSTIC_AGENTS', '')).split(',')
                      if a.strip()]
        if diagnostic and not bool(getattr(self.config_trading, 'BINANCE_TESTNET', False)):
            logger.critical("[MIRROR] TESTNET_DIAGNOSTIC_AGENTS=%s so vale na testnet. Nenhuma ordem sera enviada.",
                            diagnostic)
            self.teacher_ready = False
            return False
        names = [n for n in names if n not in diagnostic]
        self.mirror_diagnostic = diagnostic
        approved, detail = mirror.approval_is_valid(model_dir)
        if approved:
            # A valid approval of one agent must not let another, unapproved
            # agent listed in LIVE_AGENTS trade next to it.
            try:
                approved_agents = set(json.loads(mirror.approval_path(model_dir).read_text(encoding="utf-8")).get("agents", []))
            except (OSError, ValueError):
                approved_agents = set()
            missing = [n for n in names if n not in approved_agents]
            if missing:
                approved, detail = False, "agentes em LIVE_AGENTS sem aprovacao: %s" % missing
        if not approved and bool(getattr(self.config_ai, 'REQUIRE_OOS_POLICY_APPROVAL', True)):
            logger.critical("[MIRROR] Especialistas sem aprovacao valida (%s). Nenhuma ordem sera enviada.", detail)
            self.teacher_ready = False
            return False
        if not approved:
            logger.warning("[MIRROR] Operando os modelos de models_ai sem aprovacao registrada "
                           "(REQUIRE_OOS_POLICY_APPROVAL=False): %s", detail)
        self.mirror_agents = {}
        for name in names:
            try:
                self.mirror_agents[name] = mirror.load_specialist(name, model_dir)
            except Exception as exc:
                logger.critical("[MIRROR] Nao foi possivel carregar %s: %s", name, exc)
                self.teacher_ready = False
                return False
        for name in diagnostic:
            # Failed approval: trades only on the testnet, to diagnose it on
            # bars no model has seen.
            try:
                self.mirror_agents[name] = mirror.load_specialist(name, model_dir / "shadow")
            except Exception as exc:
                logger.critical("[MIRROR] Nao foi possivel carregar %s (diagnostico): %s", name, exc)
                self.teacher_ready = False
                return False
            logger.warning("[MIRROR] %s NAO APROVADO operando na TESTNET para diagnostico (models_ai/shadow).", name)
        self.mirror_history = mirror.LiveHistory()
        newest = pd.Timestamp.now(tz="UTC").floor("15min") - mirror.BAR
        missing = self.mirror_history.missing_since(newest)
        if missing is not None:
            logger.info("[MIRROR] Reconstruindo historico de candles fechados desde %s...", missing)
            await mirror.rebuild(self.mirror_history, missing, newest + mirror.BAR)
        self.teacher_ready = True
        self.is_trained = True
        logger.info("[MIRROR] Especialistas prontos: %s | historico %d barras ate %s", list(self.mirror_agents),
                    len(self.mirror_history.frame), self.mirror_history.frame.index[-1])
        return True

    async def _agent_mirror_decision(self, recent_market_df: pd.DataFrame) -> Signal:
        from trading import agent_mirror as mirror

        symbol = self.config_trading.PRIMARY_PAIR
        if not self.teacher_ready or not self.risk_manager or not self.portfolio:
            return Signal(symbol=symbol, action=Action.HOLD, confidence=0.0,
                          explanation={"reason": "espelho indisponivel", "policy": mirror.POLICY_NAME})
        now = pd.Timestamp.now(tz="UTC")
        # Same enrichment as the training frame and the history rebuild
        # (build_forward_dataset): the specialists observe the autoencoder
        # latents, which create_features does not add. Appending rows without
        # them left the newest bar with the previous bar's latents after ffill.
        frame = self.feature_pipeline.apply_hidden_features(recent_market_df.copy())
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        newest = self.mirror_history.append_newest(frame, now)
        if not self.mirror_history.is_complete(newest):
            start = self.mirror_history.missing_since(newest)
            logger.warning("[MIRROR] Historico incompleto desde %s: reconstruindo antes de decidir.", start)
            await mirror.rebuild(self.mirror_history, start, newest + mirror.BAR)
            if not self.mirror_history.is_complete(newest):
                return Signal(symbol=symbol, action=Action.HOLD, confidence=0.0,
                              explanation={"reason": "historico incompleto", "policy": mirror.POLICY_NAME})
        if self._teacher_cache is None or self._teacher_cache[0] != newest:
            history = self.mirror_history.frame.loc[:newest]
            shadows, self.mirror_paths = [], {}
            for name, (agent, contract) in self.mirror_agents.items():
                state, path = await asyncio.to_thread(mirror.replay_with_path, agent, contract, history, name)
                shadows.append(state)
                self.mirror_paths[name] = path
            self._teacher_cache = (newest, shadows)
            logger.info("[MIRROR] barra %s | %s", newest,
                        " ".join("%s=%+d%s" % (s.agent, s.side, "*" if s.entered_on_last_bar else "") for s in shadows))
        return self._mirror_to_signal(self._teacher_cache[1], newest, mirror.POLICY_NAME, "AgentMirror")

    async def generate_trading_decision(self, recent_market_df: pd.DataFrame) -> Signal:
        """Decisao deste ciclo: a do espelho dos agentes."""
        try:
            return await self._agent_mirror_decision(recent_market_df)
        except Exception as exc:
            logger.critical("[MIRROR] Falha ao decidir: %s", exc, exc_info=True)
            return Signal(symbol=self.config_trading.PRIMARY_PAIR, action=Action.HOLD, confidence=0.0,
                          explanation={"reason": "falha do espelho: %s" % exc, "policy": "agent_mirror"})
