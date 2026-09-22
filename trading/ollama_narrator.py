# -----------------------------------------------------------------------------
# ARQUIVO: trading/ollama_narrator.py
# DESCRIÇÃO: Narrador IA local via Ollama — gera análise em linguagem natural
# diretamente no terminal, apenas quando o mercado muda significativamente.
# -----------------------------------------------------------------------------

"""
OllamaNarrator

Converte o contexto estruturado do bot (features, regime, tape, posição)
em um relatório em linguagem natural exibido no terminal.

Estratégia de atualização:
  • Só gera nova análise quando o "fingerprint" do mercado muda
  • Fingerprint: action + regime + tape_pulse + banda de confiança + banda de preço + posição
  • Fallback: atualiza a cada 15 min mesmo sem mudança
  • Nunca bloqueia o loop de trading (roda em background via asyncio.create_task)
  • Se Ollama não estiver disponível, desativa silenciosamente
"""

import asyncio
import aiohttp
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("OllamaNarrator")

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL   = "qwen3.5:4b"
BOX_WIDTH       = 72          # largura da caixa no terminal
MAX_INTERVAL    = timedelta(minutes=15)   # atualização forçada se nada mudar
MIN_INTERVAL    = timedelta(seconds=45)   # evita geração em rafaga
OLLAMA_TIMEOUT  = 90          # segundos máximos para o LLM responder
MAX_TOKENS      = 512         # tokens de saída (aumentado para cobrir think + resposta)


# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """Você é um analista sênior de crypto trading. Analise o estado atual do bot e escreva um relatório curto e direto em português.

ESTADO DO BOT AGORA:
- Par: {symbol} | Preço: ${price:,.2f}
- Regime de mercado detectado: {regime}
- Decisão da IA: {action} (Confiança: {confidence:.1%})
- Microestrutura (Tape Pulse): {tape_pulse} | Score: {tape_score:+.2f} | OBI: {obi:+.3f}

POSIÇÃO ABERTA:
{position_text}

FEATURES QUE MAIS INFLUENCIARAM A DECISÃO:
{features_text}

REGRAS DE ESCRITA:
- Máximo 5 linhas curtas
- Linguagem natural, como um analista falaria num rádio de trading
- Explique POR QUE o bot tomou essa decisão com base nas features acima
- Se há posição aberta, comente o status dela
- Termine com o que o bot está aguardando para agir (ou continuar aguardando)
- NÃO repita os números brutos — interprete-os
"""


MIRROR_PROMPT = """Você narra, em português, o que um bot de trading de BTC está enxergando agora. Escreva para o dono do bot, em linguagem natural, como um trader experiente explicando a tela.

COMO O BOT OPERA (não contradiga):
- Não tenta adivinhar reversão. Só entra DEPOIS que a escada se confirma: degraus fortes no gráfico de 15m na mesma direção, velas fortes no 5m, rompimento da máxima (ou mínima) de 12h e o fluxo agressor a favor.
- O agente LONG só compra; o agente SHORT só vende. Cada um surfa a escada e sai quando a estrutura de 4h quebra ou no stop.
- O bot copia na conta a posição do agente, mas só no candle em que o agente entra (não entra atrasado).

O QUE O BOT ESTÁ VENDO (candle de 15m fechado às {bar} UTC, preço ${price:,.1f}):
{facts}

AGENTES E CONTA:
{agents}
Ação do bot agora: {action} — {reason}

CONTEXTO DE MERCADO (só informativo, não decide a ordem):
Regime detectado: {regime} | Tape: {tape_pulse} (score {tape_score:+.2f}) | OBI: {obi:+.2f} | Fear & Greed: {sentiment}

COMO ESCREVER:
- Fale como um trader experiente conversando com o dono do bot enquanto olha o gráfico: tom natural, frases variadas, em 3 a 5 frases, sem listas nem números de ATR.
- Conte o que o mercado está fazendo agora (se tem escada ou só vai-e-vem, para onde o 4h aponta, se o fluxo está comprador ou vendedor, o clima do tape e do sentimento), fiel ao RESUMO DO GRÁFICO: nunca diga que há escada se o resumo diz que não há.
- Diga de forma simples o que o bot está esperando: cite só o que mais falta (1 ou 2 coisas) para o LONG ou o SHORT entrar, sem recitar a lista toda. Se um agente está posicionado, diga como vai o trade e o que faria o bot sair.
- Use só os fatos acima; não invente números. Não fale em "reversão", "confiança" nem "sinal de reversão".
"""


def mirror_facts(row, view: Dict) -> Dict[str, str]:
    """Fatos do candle fechado, nas variaveis que os agentes realmente leem."""
    def f(name, default=0.0):
        try:
            return float(row.get(name, default))
        except (TypeError, ValueError):
            return default

    up, down = int(f("cz_leg_steps_up")), int(f("cz_leg_steps_down"))
    up5, down5 = int(f("cz_steps5_up")), int(f("cz_steps5_down"))
    brk_up, brk_down = f("cz_breakout_up_48"), f("cz_breakout_down_48")
    cvd, body = f("cz_cvd_z_16"), f("cz_step_body_atr")
    s_long, s_short = f("cz_struct_4h_long"), f("cz_struct_4h_short")
    trend4 = f("cz_trend_4h")
    rsi = f("rsi_15m", float("nan"))

    if up >= 3:
        summary = "escada de ALTA formada (%d degraus fortes de alta)" % up
    elif down >= 2:
        summary = "escada de BAIXA formada (%d degraus fortes de baixa)" % down
    elif up or down:
        summary = "sem escada formada: só %d degrau(s) de alta e %d de baixa, ainda não confirma" % (up, down)
    else:
        summary = "sem degraus fortes: mercado de vai-e-vem, nenhuma escada"

    def missing(items):
        left = [text for ok, text in items if not ok]
        return "nada, todas as condições cumpridas" if not left else "; ".join(left)

    long_missing = missing([
        (up >= 3, "mais %d degrau(s) forte(s) de alta no 15m" % max(0, 3 - up)),
        (up5 >= 2, "mais %d vela(s) forte(s) de alta no 5m" % max(0, 2 - up5)),
        (brk_up > 0, "romper a máxima de 12h (está %.1f ATR abaixo)" % abs(brk_up)),
        (body > 0, "fechar um candle de 15m de alta"),
        (cvd > 0, "fluxo agressor virar comprador")])
    short_missing = missing([
        (down >= 2, "mais %d degrau(s) forte(s) de baixa no 15m" % max(0, 2 - down)),
        (down5 >= 2, "mais %d vela(s) forte(s) de baixa no 5m" % max(0, 2 - down5)),
        (brk_down < 0, "romper a mínima de 12h (está %.1f ATR acima)" % abs(brk_down)),
        (body < 0, "fechar um candle de 15m de baixa"),
        (cvd < 0, "fluxo agressor virar vendedor")])

    facts = [
        "- RESUMO DO GRÁFICO: %s." % summary,
        "- FALTA PARA O AGENTE LONG ENTRAR: %s." % long_missing,
        "- FALTA PARA O AGENTE SHORT ENTRAR: %s." % short_missing,
        "- Degraus fortes (corpo >= 1 ATR) nas últimas 12 velas de 15m: %d de alta, %d de baixa." % (up, down),
        "- Velas fortes de 5m dentro dos dois últimos candles de 15m: %d de alta, %d de baixa." % (up5, down5),
        "- Máxima de 12h: preço %s (%.1f ATR). Mínima de 12h: preço %s (%.1f ATR)." % (
            "ACIMA, rompeu" if brk_up > 0 else "abaixo", abs(brk_up),
            "ABAIXO, rompeu" if brk_down < 0 else "acima", abs(brk_down)),
        "- Último candle de 15m: %s (corpo %.1f ATR). Fluxo agressor (CVD 16 velas): %s." % (
            "de alta" if body > 0 else "de baixa" if body < 0 else "neutro", abs(body),
            "comprador" if cvd > 0 else "vendedor" if cvd < 0 else "neutro"),
        "- Estrutura de 4h: %s; tendência de 4h %s%s." % (
            "de alta intacta" if s_long >= 0 else "de alta quebrada",
            "de alta" if trend4 > 0 else "de baixa" if trend4 < 0 else "lateral",
            "" if rsi != rsi else "; RSI 15m %.0f" % rsi),
    ]
    names = {"bull": "agente LONG", "bear": "agente SHORT"}
    price = float(view.get("close") or 0.0)
    agents = []
    for sh in view.get("shadows", []):
        name = names.get(sh.agent, sh.agent)
        if sh.side == 0:
            agents.append("- %s: fora." % name)
            continue
        pnl = sh.side * (price / sh.entry_price - 1) * 100 if price and sh.entry_price else 0.0
        agents.append("- %s: %s na simulação desde a entrada a $%s, resultado %+.2f%%, stop $%s%s." % (
            name, "comprado" if sh.side > 0 else "vendido", "{:,.0f}".format(sh.entry_price), pnl,
            "{:,.0f}".format(sh.stop_price) if sh.stop_price else "—",
            ", ENTROU NESTE CANDLE" if sh.entered_on_last_bar else ""))
    acc = view.get("account_side", 0)
    agents.append("- Conta na Binance: %s." % ("sem posição" if not acc else "comprada" if acc > 0 else "vendida"))
    return {"facts": "\n".join(facts), "agents": "\n".join(agents)}


class OllamaNarrator:
    """
    Gera narrativas em linguagem natural no terminal via Ollama.
    Integração: chamar `await narrator.maybe_narrate(ctx, position)` no loop principal.
    """

    def __init__(self, model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL, telegram=None):
        self.model       = model
        self.ollama_url  = ollama_url
        self.telegram    = telegram   # TelegramNotifier opcional — encaminha análises
        self._last_fp:        Optional[Dict] = None
        self._last_update:    Optional[datetime] = None
        self._is_generating:  bool = False
        self._available:      Optional[bool] = None
        self._check_task:     Optional[asyncio.Task] = None

    # ── Verificação de disponibilidade ──────────────────────────────────────

    async def check_availability(self) -> bool:
        """Testa se Ollama está rodando e o modelo está disponível."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    OLLAMA_TAGS_URL,
                    timeout=aiohttp.ClientTimeout(total=4)
                ) as r:
                    if r.status == 200:
                        data = await r.json()
                        models = [m.get('name', '') for m in data.get('models', [])]
                        model_base = self.model.split(':')[0]
                        model_found = any(model_base in m for m in models)
                        if model_found:
                            logger.info(
                                f"✅ [NARRATOR] Ollama disponível. "
                                f"Modelo '{self.model}' encontrado. Narrativas ATIVAS."
                            )
                            self._available = True
                        else:
                            logger.warning(
                                f"⚠️ [NARRATOR] Ollama disponível mas modelo '{self.model}' "
                                f"não encontrado. Modelos instalados: {models}. "
                                f"Execute: ollama pull {self.model}"
                            )
                            self._available = False
                        return self._available
        except Exception as e:
            logger.warning(
                f"⚠️ [NARRATOR] Ollama não acessível em {OLLAMA_TAGS_URL}: {e}. "
                f"Narrativas desativadas. Para ativar: instale Ollama e execute "
                f"'ollama pull {self.model}'"
            )
        self._available = False
        return False

    # ── Fingerprint e detecção de mudança ───────────────────────────────────

    def _fingerprint(self, ctx: Dict, has_position: bool) -> Dict:
        """
        Resume o estado atual em campos discretos comparáveis.
        Mudança em qualquer campo = nova narrativa.
        """
        confidence = float(ctx.get('confidence', 0))
        price      = float(ctx.get('price', 0))
        tape       = ctx.get('tape_pulse', 'NEUTRAL')

        return {
            'action':       ctx.get('action', 'HOLD'),
            'regime':       ctx.get('regime', 'unknown'),
            # Tape WARMUP não conta como mudança — aguarda dado real
            'tape':         tape if tape != 'WARMUP...' else 'WARMUP',
            'conf_band':    int(confidence * 5),   # bandas de 20%
            'price_band':   round(price / 1000),   # bandas de $1000
            'has_position': has_position,
        }

    def _has_changed(self, new_fp: Dict) -> Tuple[bool, str]:
        """Retorna (mudou, motivo_legível)."""
        if self._last_fp is None:
            return True, "primeira análise do bot"

        old = self._last_fp

        if new_fp['action'] != old['action']:
            return True, f"decisão mudou: {old['action']} → {new_fp['action']}"
        if new_fp['regime'] != old['regime']:
            return True, f"regime mudou: {old['regime'].upper()} → {new_fp['regime'].upper()}"
        if new_fp['tape'] != old['tape'] and new_fp['tape'] != 'WARMUP':
            return True, f"tape mudou: {old['tape']} → {new_fp['tape']}"
        if new_fp['has_position'] != old['has_position']:
            verb = "aberta" if new_fp['has_position'] else "fechada"
            return True, f"posição {verb}"
        if abs(new_fp['conf_band'] - old['conf_band']) >= 1:
            return True, "confiança da IA mudou significativamente"
        if new_fp['price_band'] != old['price_band']:
            return True, f"preço cruzou banda de ${new_fp['price_band'] * 1000:,.0f}"

        # Fallback periódico
        if self._last_update and (datetime.utcnow() - self._last_update) >= MAX_INTERVAL:
            return True, "atualização periódica (15 min)"

        return False, ""

    # ── Construção do prompt ─────────────────────────────────────────────────

    @staticmethod
    def _build_position_text(position: Optional[Dict]) -> str:
        if not position:
            return "  Nenhuma posição aberta no momento."
        qty   = float(position.get('quantity', 0))
        entry = float(position.get('entry_price', 0))
        pnl   = float(position.get('unrealized_pnl', 0))
        lev   = position.get('leverage', 1)
        direction = "LONG 🟢" if qty > 0 else "SHORT 🔴"
        pnl_icon  = "🟢" if pnl >= 0 else "🔴"
        return (
            f"  Direção: {direction} | Tamanho: {abs(qty):.4f} BTC | "
            f"Alavancagem: {lev}x\n"
            f"  Entrada: ${entry:,.2f} | PnL não realizado: {pnl_icon} ${pnl:+,.2f}"
        )

    @staticmethod
    def _build_features_text(top_features: list) -> str:
        if not top_features:
            return "  Nenhuma feature disponível."
        lines = []
        feature_names = {
            'bb_width_1h':         'Largura das Bandas de Bollinger (1h)',
            'realized_vol_20_15m': 'Volatilidade Realizada 20p (15m)',
            'log_return_volume_1h':'Retorno de Volume Logarítmico (1h)',
            'rsi_1h':              'RSI (1h)',
            'rsi_4h':              'RSI (4h)',
            'macd_1h':             'MACD (1h)',
            'atr_1h':              'ATR (1h)',
        }
        for f in top_features[:5]:
            name   = f.get('feature', '')
            value  = f.get('value', '')
            impact = float(f.get('impact_score', 0))
            label  = feature_names.get(name, name)
            direction = "↑ bullish" if impact > 0 else "↓ bearish"
            lines.append(f"  • {label}: {value} ({direction}, peso {abs(impact):.3f})")
        return "\n".join(lines)

    def _build_prompt(self, ctx: Dict, position: Optional[Dict]) -> str:
        return PROMPT_TEMPLATE.format(
            symbol       = ctx.get('symbol', 'BTCUSDT'),
            price        = float(ctx.get('price', 0)),
            regime       = ctx.get('regime', 'desconhecido').upper(),
            action       = ctx.get('action', 'HOLD'),
            confidence   = float(ctx.get('confidence', 0)),
            tape_pulse   = ctx.get('tape_pulse', 'N/A'),
            tape_score   = float(ctx.get('tape_score', 0)),
            obi          = float(ctx.get('obi', 0)),
            position_text= self._build_position_text(position),
            features_text= self._build_features_text(ctx.get('top_features', [])),
        )

    # ── Exibição no terminal ─────────────────────────────────────────────────

    @staticmethod
    def _print_narrative(text: str, ctx: Dict):
        """
        Imprime caixa visual no terminal via print() direto (não logger).
        O logger filtra por nível — o print garante visibilidade imediata.
        """
        price    = float(ctx.get('price', 0))
        action   = ctx.get('action', 'HOLD')
        regime   = ctx.get('regime', '').upper()
        symbol   = ctx.get('symbol', 'BTCUSDT')
        conf     = float(ctx.get('confidence', 0))
        now      = datetime.now().strftime('%H:%M:%S')
        tape     = ctx.get('tape_pulse', 'N/A')

        icons = {'BUY': '🟢 COMPRA', 'SELL': '🔴 VENDA', 'HOLD': '⚪ AGUARDANDO'}
        action_label = icons.get(action, action)

        w = BOX_WIDTH
        header = f" 🤖 ANÁLISE IA LOCAL  •  {now}  •  {symbol} ${price:,.0f} "
        sub    = f" Regime: {regime}  |  {action_label}  |  Confiança: {conf:.1%}  |  Tape: {tape} "
        if ctx.get('_mirror_sub'):
            header = f" 🤖 O QUE O BOT ESTÁ VENDO (IA LOCAL)  •  {now}  •  {symbol} ${price:,.0f} "
            sub    = " " + ctx['_mirror_sub'] + " "

        # Quebra texto em linhas respeitando a largura da caixa
        words  = text.replace('\n', ' ').split()
        lines  = []
        cur    = ""
        for word in words:
            if len(cur) + len(word) + 1 <= w - 4:
                cur += (" " if cur else "") + word
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)

        # Imprime a caixa
        print(f"\n╔{'═' * w}╗", flush=True)
        print(f"║{header:^{w}}║", flush=True)
        print(f"║{sub:^{w}}║", flush=True)
        print(f"╠{'═' * w}╣", flush=True)
        for line in lines:
            print(f"║  {line:<{w - 2}}║", flush=True)
        print(f"╚{'═' * w}╝\n", flush=True)

    # ── Interface pública ────────────────────────────────────────────────────

    async def maybe_narrate(self, ctx: Dict, position: Optional[Dict] = None):
        """
        Ponto de entrada principal. Chamar após cada decisão no main_trading_loop.

        Args:
            ctx: dicionário com symbol, action, confidence, price, regime,
                 tape_pulse, tape_score, obi, top_features
            position: dicionário com quantity, entry_price, unrealized_pnl, leverage
                      (ou None se não há posição)
        """
        # Verifica Ollama na primeira chamada
        if self._available is None:
            await self.check_availability()

        if not self._available or self._is_generating:
            return

        # Rate limit mínimo
        if self._last_update and (datetime.utcnow() - self._last_update) < MIN_INTERVAL:
            return

        has_position = position is not None
        fp = self._fingerprint(ctx, has_position)
        changed, reason = self._has_changed(fp)

        if not changed:
            return

        logger.info(f"🤖 [NARRATOR] Gerando análise... Motivo: {reason}")
        # Lança geração em background — não bloqueia o loop de trading
        asyncio.create_task(self._generate(ctx, position, fp, reason))

    async def maybe_narrate_mirror(self, ctx: Dict):
        """Espelho dos agentes: narra o que o bot ve, uma vez por candle ou quando algo muda.

        ctx: price, symbol, _mirror_key (candle + lados + acao), _mirror_sub (linha do
        cabecalho) e os campos de MIRROR_PROMPT.
        """
        if self._available is None:
            await self.check_availability()
        if not self._available or self._is_generating:
            return
        key = ctx.get('_mirror_key')
        if self._last_fp is not None and self._last_fp.get('mirror') == key:
            return
        ctx = dict(ctx, _prompt=MIRROR_PROMPT.format(**{k: ctx[k] for k in (
            'bar', 'price', 'facts', 'agents', 'action', 'reason', 'regime', 'tape_pulse', 'tape_score',
            'obi', 'sentiment')}))
        logger.info("🤖 [NARRATOR] Lendo o candle %s para o painel...", ctx.get('bar'))
        asyncio.create_task(self._generate(ctx, None, {'mirror': key}, "novo candle"))

    async def _generate(self, ctx: Dict, position: Optional[Dict], fp: Dict, reason: str):
        """Chama o Ollama e exibe o resultado. Roda em background."""
        self._is_generating = True
        try:
            prompt    = ctx.get('_prompt') or self._build_prompt(ctx, position)
            full_text = ""

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.ollama_url,
                    json={
                        "model":   self.model,
                        "prompt":  prompt,
                        "stream":  True,
                        "think":   False,          # desativa thinking mode (Qwen3/3.5)
                        "options": {
                            "temperature": 0.25,
                            "num_predict": MAX_TOKENS,
                            "top_p":       0.9,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT),
                ) as response:
                    if response.status != 200:
                        body = await response.text()
                        logger.warning(f"⚠️ [NARRATOR] Ollama status {response.status}: {body[:200]}")
                        return

                    async for raw_line in response.content:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            data  = json.loads(line.decode('utf-8'))
                            token = data.get('response', '')
                            full_text += token
                            if data.get('done', False):
                                break
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue

            import re
            text = full_text.strip()

            # Remove bloco <think>...</think> caso o modelo ignore o think:false
            if '<think>' in text:
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

            # Remove linhas em branco excessivas
            text = re.sub(r'\n{3,}', '\n\n', text).strip()

            if text:
                self._print_narrative(text, ctx)
                self._last_fp     = fp
                self._last_update = datetime.utcnow()
                # Encaminha análise ao Telegram (com anti-spam interno de 10 min)
                if self.telegram:
                    asyncio.create_task(
                        self.telegram.forward_narrator_analysis(text, ctx)
                    )
            else:
                # Log diagnóstico: mostra os primeiros chars do raw para entender o problema
                preview = full_text[:300] if full_text else "(nenhum token recebido)"
                logger.warning(
                    f"⚠️ [NARRATOR] Resposta vazia após filtro. "
                    f"Raw ({len(full_text)} chars): {preview!r}"
                )

        except asyncio.CancelledError:
            pass
        except aiohttp.ClientConnectorError:
            logger.warning("⚠️ [NARRATOR] Ollama desconectou. Tentará na próxima mudança.")
            self._available = False   # desativa até próxima verificação manual
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ [NARRATOR] Timeout após {OLLAMA_TIMEOUT}s. Modelo lento ou sobrecarregado.")
        except Exception as e:
            logger.warning(f"⚠️ [NARRATOR] Erro inesperado: {e}")
        finally:
            self._is_generating = False
