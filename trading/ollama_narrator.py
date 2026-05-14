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

    async def _generate(self, ctx: Dict, position: Optional[Dict], fp: Dict, reason: str):
        """Chama o Ollama e exibe o resultado. Roda em background."""
        self._is_generating = True
        try:
            prompt    = self._build_prompt(ctx, position)
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
