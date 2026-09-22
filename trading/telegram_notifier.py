# -----------------------------------------------------------------------------
# ARQUIVO: trading/telegram_notifier.py
# DESCRIÇÃO: Alertas instantâneos + relatório horário via Telegram Bot API.
#
# Eventos que disparam alerta imediato:
#   • Posição aberta / fechada (com PnL)
#   • Stop loss disparado
#   • Risk-off ativado / desativado
#   • WebSocket desconectado
#   • Erro crítico no bot
#   • Tape WARMUP por mais de 5 min
#   • Bot iniciado / encerrado
#
# Relatório horário (gerado pelo LLM local via Ollama):
#   • Status geral do sistema
#   • Posição atual com PnL em tempo real
#   • O que a IA está vendo no mercado
#   • Erros das últimas 1h
#   • Decisões tomadas
# -----------------------------------------------------------------------------

import asyncio
import aiohttp
import json
import os
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from utils.logger import get_logger

logger = get_logger("TelegramNotifier")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
REPORT_INTERVAL   = timedelta(minutes=30)
SEND_TIMEOUT      = 15   # segundos para cada envio
MAX_ERRORS_BUFFER = 50   # máximo de erros guardados para o relatório
# Nome mostrado ao usuario: o Bull opera so comprado e o Bear so vendido.
AGENT_NAMES = {"bull": "agente LONG", "bear": "agente SHORT"}

# ---------------------------------------------------------------------------
# Prompt do relatório horário para o LLM
# ---------------------------------------------------------------------------
HOURLY_PROMPT = """Você é um analista de trading. Gere um relatório horário CONCISO em português para o dono do bot.

ESTADO DO BOT ({hora}):
- Par: {symbol} | Preço: ${price:,.2f}
- Regime: {regime} | Tape: {tape_pulse} (Score: {tape_score:+.2f})
- Última decisão: {action} | Confiança: {confidence:.1%}
- Sistema: {health} | Conexão Binance: {connection}
- Modo: {mode}

POSIÇÃO ABERTA:
{position_text}

ERROS NAS ÚLTIMAS {error_window}h:
{errors_text}

DECISÕES NAS ÚLTIMAS {decision_window}h:
{decisions_text}

INSTRUÇÕES:
- Máximo 8 linhas
- Tom de analista experiente, direto
- Destaque se há algo preocupante ou oportunidade interessante
- Se houver erros, explique em linguagem simples o que pode estar causando
- Termine com uma frase sobre o que aguardar na próxima hora
"""


class TelegramNotifier:
    """
    Envia alertas instantâneos e relatórios horários via Telegram.

    Uso no bot:
        notifier = TelegramNotifier(token, chat_id, narrator=narrator)
        await notifier.check_availability()

        # Alertas pontuais:
        await notifier.alert_position_opened(position_dict, signal)
        await notifier.alert_position_closed(symbol, pnl, reason)
        await notifier.alert_error("Descrição do erro", level="CRITICAL")
        await notifier.alert_risk_off(drawdown_pct)

        # No monitor loop (a cada minuto):
        await notifier.maybe_hourly_report(system_state, portfolio, tape_pulse)
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        narrator=None,          # OllamaNarrator — para gerar texto do relatório
        ollama_url: str = "http://localhost:11434/api/generate",
        ollama_model: str = "qwen3.5:4b",
    ):
        self.token        = token
        self.chat_id      = str(chat_id)
        self.narrator     = narrator
        self.ollama_url   = ollama_url
        self.ollama_model = ollama_model
        self._available   = None          # None = não testado ainda

        # Buffers para o relatório horário
        self._error_buffer:    deque = deque(maxlen=MAX_ERRORS_BUFFER)
        self._decision_buffer: deque = deque(maxlen=100)
        # Primeiro relatório LLM sai 1 minuto após o boot (bot precisa de 1 ciclo completo)
        # Depois disso, cadência normal de 1 hora
        self._last_report: Optional[datetime] = datetime.utcnow() - REPORT_INTERVAL + timedelta(minutes=1)
        self._generating_report: bool = False

        # Controle de anti-spam (evita duplicar o mesmo alerta em < 60s)
        self._last_alerts: Dict[str, datetime] = {}
        # Anti-spam específico para análises do Narrator (evita flood)
        self._last_narrator_forward: Optional[datetime] = None

    # ── Verificação de disponibilidade ──────────────────────────────────────

    async def check_availability(self) -> bool:
        """Verifica se o token/chat_id são válidos via getMe."""
        if not self.token or not self.chat_id:
            logger.warning(
                "⚠️ [TELEGRAM] TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID não configurados. "
                "Adicione ao .env para ativar notificações."
            )
            self._available = False
            return False
        try:
            url = TELEGRAM_API.format(token=self.token, method="getMe")
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    data = await r.json()
                    if data.get("ok"):
                        bot_name = data["result"].get("username", "bot")
                        logger.info(
                            f"✅ [TELEGRAM] Bot @{bot_name} autenticado. "
                            f"Notificações ATIVAS → chat {self.chat_id}"
                        )
                        self._available = True
                        return True
                    else:
                        logger.warning(f"⚠️ [TELEGRAM] Token inválido: {data}")
        except Exception as e:
            logger.warning(f"⚠️ [TELEGRAM] Falha ao conectar: {e}")
        self._available = False
        return False

    # ── Envio de mensagem base ───────────────────────────────────────────────

    async def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Envia mensagem ao chat. Retorna True se sucesso."""
        if not self._available:
            return False
        try:
            url  = TELEGRAM_API.format(token=self.token, method="sendMessage")
            payload = {
                "chat_id":    self.chat_id,
                "text":       text[:4096],    # limite do Telegram
                "parse_mode": parse_mode,
            }
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=SEND_TIMEOUT)
                ) as r:
                    result = await r.json()
                    if result.get("ok"):
                        return True
                    logger.warning(f"⚠️ [TELEGRAM] Erro ao enviar: {result}")
        except Exception as e:
            logger.warning(f"⚠️ [TELEGRAM] Falha no envio: {e}")
        return False

    async def send_photo(self, path, caption: str = "") -> bool:
        """Envia uma imagem (ex.: o gráfico do espelho) ao chat."""
        if not self._available:
            return False
        try:
            url = TELEGRAM_API.format(token=self.token, method="sendPhoto")
            form = aiohttp.FormData()
            form.add_field("chat_id", self.chat_id)
            form.add_field("caption", caption[:1024])
            form.add_field("parse_mode", "HTML")
            with open(path, "rb") as fh:
                form.add_field("photo", fh.read(), filename="espelho.png", content_type="image/png")
            async with aiohttp.ClientSession() as s:
                async with s.post(url, data=form, timeout=aiohttp.ClientTimeout(total=30)) as r:
                    result = await r.json()
                    if result.get("ok"):
                        return True
                    logger.warning(f"⚠️ [TELEGRAM] Erro ao enviar imagem: {result}")
        except Exception as e:
            logger.warning(f"⚠️ [TELEGRAM] Falha no envio da imagem: {e}")
        return False

    def _anti_spam(self, key: str, min_interval_s: int = 60) -> bool:
        """Retorna True se OK para enviar (não é spam)."""
        last = self._last_alerts.get(key)
        if last and (datetime.utcnow() - last).total_seconds() < min_interval_s:
            return False
        self._last_alerts[key] = datetime.utcnow()
        return True

    # ── Alertas de eventos ───────────────────────────────────────────────────

    async def alert_bot_started(self, mode: str, balance: float, policy: str = ""):
        """Bot iniciado."""
        text = (
            f"🚀 <b>Bot Shield IA — INICIADO</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚙️ Modo: <b>{mode}</b>\n"
            + (f"🪞 {policy}\n" if policy else "") +
            f"💰 Saldo: <b>${balance:,.2f}</b>\n"
            f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        )
        await self._send(text)

    async def alert_bot_stopped(self, reason: str = "Desligamento normal"):
        """Bot encerrado."""
        text = (
            f"🛑 <b>Bot Shield IA — ENCERRADO</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 Motivo: {reason}\n"
            f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        )
        await self._send(text)

    async def alert_position_opened(self, position: Dict, signal: Any = None):
        """Posição aberta."""
        qty       = float(position.get("quantity", 0))
        entry     = float(position.get("entry_price", 0))
        lev       = position.get("leverage", 1)
        symbol    = position.get("symbol", "BTCUSDT")
        direction = "🟢 LONG" if qty > 0 else "🔴 SHORT"
        sl        = getattr(signal, "stop_loss", None) if signal else None
        tp        = getattr(signal, "take_profit", None) if signal else None
        conf      = getattr(signal, "confidence", 0) if signal else 0

        sl_price = round(entry * (1 - sl), 2) if sl and qty > 0 else round(entry * (1 + sl), 2) if sl else "—"
        tp_price = round(entry * (1 + tp), 2) if tp and qty > 0 else round(entry * (1 - tp), 2) if tp else "—"

        text = (
            f"📈 <b>POSIÇÃO ABERTA — {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Direção: {direction}\n"
            f"Quantidade: <b>{abs(qty):.4f} BTC</b>\n"
            f"Entrada: <b>${entry:,.2f}</b>\n"
            f"Alavancagem: <b>{lev}x</b>\n"
            f"Stop Loss: ${sl_price}\n"
            f"Take Profit: ${tp_price}\n"
            f"Confiança da IA: {conf:.1%}\n"
            f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        )
        asyncio.create_task(self._send(text))

    async def alert_position_closed(self, symbol: str, pnl: float, reason: str = ""):
        """Posição fechada com resultado."""
        icon  = "✅" if pnl >= 0 else "❌"
        emoji = "🟢" if pnl >= 0 else "🔴"
        text = (
            f"{icon} <b>POSIÇÃO FECHADA — {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Resultado: {emoji} <b>${pnl:+,.2f}</b>\n"
            f"Motivo: {reason or 'Decisão da IA'}\n"
            f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        )
        asyncio.create_task(self._send(text))

    async def alert_stop_hit(self, symbol: str, stop_price: float, pnl: float):
        """Stop loss disparado."""
        emoji = "🟢" if pnl >= 0 else "🔴"
        text = (
            f"🛡️ <b>STOP DISPARADO — {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Preço de stop: <b>${stop_price:,.2f}</b>\n"
            f"PnL realizado: {emoji} <b>${pnl:+,.2f}</b>\n"
            f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        )
        asyncio.create_task(self._send(text))

    async def alert_risk_off(self, drawdown_pct: float, leverage: float = 0):
        """Modo risk-off ativado."""
        if not self._anti_spam("risk_off", min_interval_s=300):
            return
        text = (
            f"⚠️ <b>RISK-OFF ATIVADO</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📉 Drawdown atual: <b>{drawdown_pct:.1%}</b>\n"
            f"⚡ Alavancagem: <b>{leverage:.1f}x</b>\n"
            f"🚫 Novas entradas bloqueadas até normalizar\n"
            f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        )
        asyncio.create_task(self._send(text))

    async def alert_error(self, message: str, level: str = "ERROR"):
        """Erro no bot. Acumula no buffer e envia se for CRITICAL."""
        timestamp = datetime.utcnow()
        self._error_buffer.append({
            "time":    timestamp.strftime("%H:%M:%S"),
            "level":   level,
            "message": message[:200],
        })

        # Envia imediatamente só para erros críticos
        if level == "CRITICAL" and self._anti_spam(f"critical_{message[:30]}", 120):
            text = (
                f"🚨 <b>ERRO CRÍTICO NO BOT</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚠️ {message[:300]}\n"
                f"🕐 {datetime.now().strftime('%H:%M:%S')}"
            )
            asyncio.create_task(self._send(text))

    async def alert_tape_warmup(self, elapsed_seconds: float):
        """Tape em WARMUP por tempo excessivo."""
        if not self._anti_spam("tape_warmup", min_interval_s=600):
            return
        text = (
            f"📡 <b>TAPE ENGINE — WARMUP PROLONGADO</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏳ Sem dados de microestrutura há <b>{elapsed_seconds/60:.0f} min</b>\n"
            f"OBI e Volume Imbalance usando valor zero (neutro)\n"
            f"Verifique a conexão WebSocket com a Binance\n"
            f"🕐 {datetime.now().strftime('%H:%M:%S')}"
        )
        asyncio.create_task(self._send(text))

    async def forward_narrator_analysis(self, text: str, ctx: Dict):
        """
        Encaminha a análise do OllamaNarrator para o Telegram.
        Anti-spam: máximo 1 mensagem a cada 10 minutos.
        """
        if not self._available:
            return
        now = datetime.utcnow()
        if self._last_narrator_forward and \
                (now - self._last_narrator_forward).total_seconds() < 600:
            return

        action  = ctx.get('action', 'HOLD')
        price   = float(ctx.get('price', 0))
        regime  = ctx.get('regime', '').upper()
        symbol  = ctx.get('symbol', 'BTCUSDT')
        conf    = float(ctx.get('confidence', 0))
        tape    = ctx.get('tape_pulse', 'N/A')
        icons   = {'BUY': '🟢 COMPRA', 'SELL': '🔴 VENDA', 'HOLD': '⚪ AGUARDANDO'}

        msg = (
            f"🤖 <b>ANÁLISE IA LOCAL — {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 ${price:,.0f} | {regime} | {icons.get(action, action)}\n"
            f"🎯 Confiança: {conf:.1%} | 📡 Tape: {tape}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{text}"
        )
        sent = await self._send(msg)
        if sent:
            self._last_narrator_forward = now

    def log_decision(self, action: str, confidence: float, regime: str):
        """Registra decisão no buffer para o relatório horário."""
        self._decision_buffer.append({
            "time":       datetime.utcnow().strftime("%H:%M"),
            "action":     action,
            "confidence": confidence,
            "regime":     regime,
        })

    # ── Espelho dos agentes (LIVE_POLICY=agent_mirror) ───────────────────────
    # Quem decide sao os agentes repassados no ambiente de treino; regime,
    # tape, confianca e SHAP nao entram na ordem, entao as mensagens mostram
    # a posicao de cada agente na simulacao, a conta e o que o bot fez.

    @staticmethod
    def _mirror_agent_lines(view: Dict) -> str:
        price = view.get("close") or 0.0
        diagnostic = set(view.get("diagnostic") or [])
        icons = {"bull": "🐂", "bear": "🐻"}
        lines = []
        for sh in view.get("shadows", []):
            tag = "diagnóstico" if sh.agent in diagnostic else "aprovado"
            head = f"{icons.get(sh.agent, '•')} <b>{AGENT_NAMES.get(sh.agent, sh.agent)}</b> ({tag}): "
            if sh.side == 0:
                lines.append(head + "fora")
                continue
            side = "COMPRADO" if sh.side > 0 else "VENDIDO"
            pnl = sh.side * (price / sh.entry_price - 1) * 100 if price and sh.entry_price else 0.0
            since = ""
            if getattr(sh, "entry_bar", None) is not None:
                since = " desde " + (sh.entry_bar + timedelta(minutes=15)).strftime("%d/%m %H:%M") + " UTC"
            stop = f" | stop ${sh.stop_price:,.0f}" if sh.stop_price else ""
            new = " ★ NOVA ENTRADA" if sh.entered_on_last_bar else ""
            lines.append(head + f"{side}{since} @ ${sh.entry_price:,.0f}{stop} | {pnl:+.2f}%{new}")
        return "\n".join(lines)

    @staticmethod
    def _account_text(view: Dict) -> str:
        side = view.get("account_side", 0)
        if not side:
            return "sem posição"
        return f"{'COMPRADA' if side > 0 else 'VENDIDA'} {view.get('account_qty', 0.0):.4f} BTC"

    def log_mirror_event(self, text: str):
        """Guarda um evento do espelho para o relatório periódico."""
        self._decision_buffer.append({"time": datetime.now().strftime("%H:%M"), "text": text,
                                      "at": datetime.utcnow()})

    async def alert_mirror_order(self, view: Dict, signal: Any, submitted: bool, reason: str = ""):
        """Ordem enviada pelo espelho (abrir ou fechar)."""
        action = view.get("action")
        agent = ""
        if view.get("reason", "").startswith("agente ") and view["reason"].endswith(" entrou"):
            agent = AGENT_NAMES.get(view["reason"].split()[1], view["reason"].split()[1]).upper()
        if action == "open":
            side = "🟢 LONG" if getattr(signal, "action", None) is not None and signal.action.value == "BUY" else "🔴 SHORT"
            title = f"📈 <b>ABRINDO {side} — {agent}</b>"
            expl = signal.explanation or {}
            detail = (f"Entrada ≈ ${float(expl.get('env_entry_price') or view.get('close') or 0):,.0f} | "
                      f"stop ${float(expl.get('env_stop_price') or 0):,.0f}\n"
                      f"Margem {signal.position_size_pct:.1%} × {signal.leverage:.0f}x")
        else:
            title = "📉 <b>FECHANDO POSIÇÃO</b>"
            detail = "O agente saiu na simulação: fechando a conta (reduceOnly)."
        status = "✅ ordem enviada à corretora" if submitted else "⚠️ ordem NÃO foi enviada (ver log)"
        text = (
            f"{title}\n━━━━━━━━━━━━━━━━━━━━━━\n{detail}\n{status}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n{self._mirror_agent_lines(view)}\n"
            f"🕐 {datetime.now().strftime('%d/%m %H:%M')}"
        )
        self.log_mirror_event(f"{'ABRIU' if action == 'open' else 'FECHOU'} {agent} ({'ok' if submitted else 'falhou'})")
        await self._send(text)

    async def alert_mirror_account_change(self, previous_side: int, view: Dict, equity: float):
        """A conta mudou de lado sem uma ordem de fechamento do espelho (ex.: stop na corretora)."""
        now_side = view.get("account_side", 0)
        before = {1: "COMPRADA", -1: "VENDIDA", 0: "sem posição"}.get(previous_side, str(previous_side))
        if now_side == 0:
            title = "🛡️ <b>POSIÇÃO ENCERRADA NA CORRETORA</b>"
            detail = f"A conta estava {before} e agora está sem posição (stop na corretora ou fechamento)."
        else:
            title = "🔄 <b>POSIÇÃO DA CONTA MUDOU</b>"
            detail = f"Antes: {before} | agora: {self._account_text(view)}"
        text = (
            f"{title}\n━━━━━━━━━━━━━━━━━━━━━━\n{detail}\n💰 Saldo: <b>${equity:,.2f}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n{self._mirror_agent_lines(view)}\n"
            f"🕐 {datetime.now().strftime('%d/%m %H:%M')}"
        )
        self.log_mirror_event(f"conta: {before} → {self._account_text(view)}")
        await self._send(text)

    def _mirror_report(self, system_state: Dict, view: Dict, equity: float, errors: list) -> str:
        bar = view.get("bar")
        bar_txt = (bar + timedelta(minutes=15)).strftime("%H:%M") + " UTC" if bar is not None else "—"
        since = datetime.utcnow() - REPORT_INTERVAL
        events = [e for e in self._decision_buffer if "text" in e and e.get("at") and e["at"] >= since]
        events_text = "\n".join(f"[{e['time']}] {e['text']}" for e in events[-8:]) or "nenhuma ordem no período"
        errors_text = f"⚠️ {len(errors)} erros: " + errors[-1] if errors else "✅ sem erros"
        why = system_state.get("mirror_reason_text") or view.get("reason", "")
        return (
            f"📊 <b>ESPELHO DOS AGENTES — {datetime.now().strftime('%d/%m %H:%M')}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💲 BTCUSDT ${float(view.get('close') or 0):,.0f} (candle {bar_txt})\n"
            f"{self._mirror_agent_lines(view)}\n"
            f"🏦 Conta: <b>{self._account_text(view)}</b> | saldo ${equity:,.2f}\n"
            f"▶ Bot: {why}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🧾 Ordens no período:\n{events_text}\n"
            f"{errors_text} | sistema {system_state.get('system_health', '—')}\n"
            f"ℹ️ A ordem sai quando um agente mostra ★ NOVA ENTRADA."
        )

    # ── Relatório horário ────────────────────────────────────────────────────

    async def maybe_hourly_report(
        self,
        system_state: Dict,
        portfolio,
        tape_pulse: Dict,
    ):
        """
        Gera e envia relatório horário se 1h passou desde o último.
        Chamar no monitor_and_log_loop a cada minuto.
        """
        if not self._available or self._generating_report:
            return
        if self._last_report and (datetime.utcnow() - self._last_report) < REPORT_INTERVAL:
            return

        asyncio.create_task(self._generate_hourly_report(system_state, portfolio, tape_pulse))

    async def _generate_hourly_report(self, system_state: Dict, portfolio, tape_pulse: Dict):
        """Gera texto do relatório via LLM e envia ao Telegram."""
        self._generating_report = True
        try:
            mirror_view = system_state.get("mirror_view")
            if mirror_view is not None:
                one_hour_ago = (datetime.utcnow() - REPORT_INTERVAL).strftime("%H:%M:%S")
                errors = [f"{e['message']}" for e in self._error_buffer if e["time"] >= one_hour_ago]
                equity = portfolio.get_total_value() if portfolio is not None else 0.0
                await self._send(self._mirror_report(system_state, mirror_view, equity, errors))
                self._last_report = datetime.utcnow()
                logger.info("✅ [TELEGRAM] Relatório do espelho enviado.")
                return
            # Coleta contexto
            latest_feat = system_state.get("latest_features")
            regime_map  = {0: "BULL 🐂", 1: "BEAR 🐻", 2: "RANGER 🤠"}
            regime      = regime_map.get(
                int(latest_feat.get("regime", 2)) if latest_feat is not None else 2,
                "Desconhecido"
            )
            price = float(latest_feat.get("close", 0)) if latest_feat is not None else 0
            last_expl  = system_state.get("last_explanation", {})
            action     = last_expl.get("decision", "HOLD")
            confidence = float(system_state.get("last_confidence", 0))
            health     = system_state.get("system_health", "desconhecido")
            connection = system_state.get("binance_connection", "desconhecido")
            mode       = "LIVE TRADING" if system_state.get("live_trading_enabled") else "PAPER TRADING"

            # Posição aberta
            position_text = "Nenhuma posição aberta no momento."
            if portfolio and portfolio.positions:
                pos_sym = list(portfolio.positions.keys())[0]
                pos     = portfolio.positions[pos_sym]
                direction = "LONG 🟢" if pos.quantity > 0 else "SHORT 🔴"
                pnl_icon  = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
                position_text = (
                    f"{direction} {abs(pos.quantity):.4f} BTC\n"
                    f"Entrada: ${pos.entry_price:,.2f} | Lev: {pos.leverage}x\n"
                    f"PnL: {pnl_icon} ${pos.unrealized_pnl:+,.2f}"
                )

            # Erros recentes (última 1h)
            one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).strftime("%H:%M:%S")
            recent_errors = [
                f"[{e['time']}] {e['level']}: {e['message']}"
                for e in self._error_buffer
                if e["time"] >= one_hour_ago
            ]
            errors_text = "\n".join(recent_errors[-5:]) if recent_errors else "Nenhum erro detectado."

            # Decisões recentes
            recent_decisions = list(self._decision_buffer)[-10:]
            decisions_text = "\n".join([
                f"[{d['time']}] {d['action']} ({d['confidence']:.0%}) — {d['regime']}"
                for d in recent_decisions
            ]) if recent_decisions else "Sem decisões registradas."

            prompt = HOURLY_PROMPT.format(
                hora             = datetime.now().strftime("%H:%M"),
                symbol           = "BTCUSDT",
                price            = price,
                regime           = regime,
                tape_pulse       = tape_pulse.get("pulse", "NEUTRAL"),
                tape_score       = float(tape_pulse.get("score", 0)),
                action           = action,
                confidence       = confidence,
                health           = health,
                connection       = connection,
                mode             = mode,
                position_text    = position_text,
                errors_text      = errors_text,
                error_window     = 1,
                decisions_text   = decisions_text,
                decision_window  = 1,
            )

            # Chama o LLM
            llm_text = await self._call_ollama(prompt)

            if not llm_text:
                # Fallback sem LLM
                llm_text = (
                    f"Regime: {regime} | Última decisão: {action} ({confidence:.0%})\n"
                    f"Posição: {position_text}\n"
                    f"Sistema: {health} | {mode}"
                )

            # Monta mensagem Telegram
            errors_summary = f"⚠️ {len(recent_errors)} erros" if recent_errors else "✅ Sem erros"
            report = (
                f"📊 <b>RELATÓRIO HORÁRIO — {datetime.now().strftime('%d/%m %H:%M')}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{llm_text}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 ${price:,.2f} | {regime} | {errors_summary}"
            )
            await self._send(report)
            self._last_report = datetime.utcnow()
            logger.info("✅ [TELEGRAM] Relatório horário enviado.")

        except Exception as e:
            logger.warning(f"⚠️ [TELEGRAM] Falha no relatório horário: {e}")
        finally:
            self._generating_report = False

    async def _call_ollama(self, prompt: str, max_tokens: int = 400) -> str:
        """Chama o LLM local e retorna o texto gerado."""
        try:
            full_text = ""
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.ollama_url,
                    json={
                        "model":   self.ollama_model,
                        "prompt":  prompt,
                        "stream":  True,
                        "think":   False,
                        "options": {"temperature": 0.2, "num_predict": max_tokens},
                    },
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as response:
                    if response.status != 200:
                        return ""
                    async for raw_line in response.content:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line.decode("utf-8"))
                            full_text += data.get("response", "")
                            if data.get("done"):
                                break
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue

            import re
            text = full_text.strip()
            if "<think>" in text:
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text

        except Exception as e:
            logger.warning(f"⚠️ [TELEGRAM] Ollama falhou no relatório: {e}")
            return ""
