"""15m candle chart of the mirror, drawn in the terminal.

After every closed bar the panel prints the recent 15m candles from the same
replay the bot trades on: green/red candles, the stretches each agent held in
the simulation (green background long, red background short), entries (▲ long,
▼ short), exits (✖) and a line at the current price. The same chart without
colours goes to logs/charts/espelho.txt.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEXT_PATH = ROOT / "logs" / "charts" / "espelho.txt"
# Lado da conta na Binance a cada candle fechado: os trades REAIS do bot.
ACCOUNT_LOG = ROOT / "logs" / "charts" / "conta_espelho.jsonl"
AGENT_NAMES = {"bull": "agente LONG", "bear": "agente SHORT"}

RESET = "\033[0m"
FG = {"up": "\033[38;5;42m", "down": "\033[38;5;203m", "long": "\033[1;38;5;46m", "short": "\033[1;38;5;196m",
      "exit": "\033[1;38;5;231m", "price": "\033[38;5;75m", "axis": "\033[38;5;245m"}
BG = {1: "\033[48;5;22m", -1: "\033[48;5;52m"}

_console_ready = False


def _enable_console_colors() -> None:
    global _console_ready
    if _console_ready:
        return
    try:
        import colorama
        colorama.just_fix_windows_console()
    except Exception:
        if os.name == "nt":
            os.system("")
    _console_ready = True


def trades_of(path: pd.DataFrame, close: pd.Series) -> List[dict]:
    """Entry/exit pairs of one agent's replayed trajectory."""
    trades, current = [], None
    for bar, side in path["side"].astype(int).items():
        entered = bool(path.at[bar, "entered"])
        if current is not None and (side != current["side"] or entered):
            exit_price = float(close.get(bar, current["entry_price"]))
            current.update(exit_bar=bar, exit_price=exit_price,
                           result=current["side"] * (exit_price / current["entry_price"] - 1) * 100)
            trades.append(current)
            current = None
        if side != 0 and current is None:
            entry = float(path.at[bar, "entry_price"]) or float(close.get(bar, 0.0))
            current = {"side": side, "entry_bar": bar, "entry_price": entry}
    if current is not None:
        trades.append(current)
    return trades


def record_account(bar, side: int, entry_price: float) -> None:
    """Guarda o lado da conta neste candle (um registro por candle)."""
    import json
    ACCOUNT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ACCOUNT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"bar": str(pd.Timestamp(bar)), "side": int(side), "entry_price": float(entry_price or 0.0)}) + "\n")


def real_trades(close: pd.Series) -> List[dict]:
    """Trades que o bot fez de fato na conta, a partir de conta_espelho.jsonl."""
    import json
    if not ACCOUNT_LOG.exists():
        return []
    rows = []
    for line in ACCOUNT_LOG.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    if not rows:
        return []
    log = pd.DataFrame(rows)
    log["bar"] = pd.to_datetime(log["bar"], utc=True)
    if close.index.tz is None:
        log["bar"] = log["bar"].dt.tz_convert(None)
    log = log.drop_duplicates("bar", keep="last").set_index("bar").sort_index()
    trades, current = [], None
    for bar, r in log.iterrows():
        side = int(r["side"])
        if current is not None and side != current["side"]:
            exit_price = float(close.get(bar, current["entry_price"]))
            current.update(exit_bar=bar, exit_price=exit_price,
                           result=current["side"] * (exit_price / current["entry_price"] - 1) * 100)
            trades.append(current)
            current = None
        if side != 0 and current is None:
            entry = float(r.get("entry_price") or 0.0) or float(close.get(bar, 0.0))
            current = {"side": side, "entry_bar": bar, "entry_price": entry, "agent": "bot"}
    if current is not None:
        trades.append(current)
    return trades


def render(history: pd.DataFrame, paths: Dict[str, pd.DataFrame], view: Optional[dict],
           bars_shown: int = 96, height: int = 18) -> str:
    """Coloured text chart of the last bars_shown 15m candles; also saved without colour."""
    _enable_console_colors()
    frame = history.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    bars = frame[["open", "high", "low", "close"]].astype(float).tail(bars_shown)
    index = list(bars.index)
    col = {t: i for i, t in enumerate(index)}
    price = float((view or {}).get("close") or bars["close"].iloc[-1])
    top = max(bars["high"].max(), price)
    bottom = min(bars["low"].min(), price)
    step = (top - bottom) / (height - 1) or 1.0

    def row(p: float) -> int:
        return int(round((top - p) / step))

    width = len(index)
    cells = [[(" ", None, None) for _ in range(width)] for _ in range(height)]

    trades = []
    for agent, path in (paths or {}).items():
        if path is None or path.empty:
            continue
        for tr in trades_of(path, frame["close"]):
            tr["agent"] = agent
            trades.append(tr)
    shade = [None] * width
    for tr in trades:
        end = tr.get("exit_bar", index[-1])
        for t in index:
            if tr["entry_bar"] <= t <= end:
                shade[col[t]] = tr["side"]

    for t, (o, h, l, c) in bars.iterrows():
        i = col[t]
        color = "up" if c >= o else "down"
        for r in range(row(h), row(l) + 1):
            cells[r][i] = ("│", color, shade[i])
        for r in range(min(row(o), row(c)), max(row(o), row(c)) + 1):
            cells[r][i] = ("█", color, shade[i])
        for r in range(height):
            if cells[r][i][0] == " ":
                cells[r][i] = (" ", None, shade[i])

    price_row = row(price)
    for i in range(width):
        ch, fg, bg = cells[price_row][i]
        if ch == " ":
            cells[price_row][i] = ("┈", "price", bg)

    real = real_trades(frame["close"])
    for tr in real:
        if tr["entry_bar"] in col:
            i = col[tr["entry_bar"]]
            if tr["side"] > 0:
                r = min(height - 1, row(bars.at[tr["entry_bar"], "low"]) + 1)
                cells[r][i] = ("▲", "long", cells[r][i][2])
            else:
                r = max(0, row(bars.at[tr["entry_bar"], "high"]) - 1)
                cells[r][i] = ("▼", "short", cells[r][i][2])
        if "exit_bar" in tr and tr["exit_bar"] in col:
            i = col[tr["exit_bar"]]
            if tr["side"] > 0:
                r = max(0, row(bars.at[tr["exit_bar"], "high"]) - 1)
            else:
                r = min(height - 1, row(bars.at[tr["exit_bar"], "low"]) + 1)
            cells[r][i] = ("✖", "exit", cells[r][i][2])

    def fmt(p: float) -> str:
        return "{:,.0f}".format(p)

    lines_c, lines_p = [], []
    for r in range(height):
        colored, plain = [], []
        for ch, fg, bg in cells[r]:
            plain.append(ch)
            colored.append((BG.get(bg, "") if bg else "") + (FG.get(fg, "") if fg else "") + ch + RESET)
        label = ""
        if r == price_row:
            label = " ◄ %s preço atual" % "{:,.1f}".format(price)
        elif r % 3 == 0:
            label = " ─ " + fmt(top - r * step)
        lines_c.append("   " + "".join(colored) + FG["price" if r == price_row else "axis"] + label + RESET)
        lines_p.append("   " + "".join(plain) + label)

    axis = [" "] * width
    for t in index:
        i = col[t]
        if t.minute == 0 and t.hour % 4 == 0 and i + 5 <= width:
            stamp = t.strftime("%d/%m") if t.hour == 0 else t.strftime("%H:%M")
            for k, ch in enumerate(stamp):
                axis[i + k] = ch
    axis_line = "   " + "".join(axis) + "  (UTC)"

    events = []
    for tr in sorted(real, key=lambda d: d["entry_bar"]):
        if tr.get("exit_bar", index[-1]) < index[0]:
            continue
        side = "LONG" if tr["side"] > 0 else "SHORT"
        text = "%s %s BOT ENTROU %s @ %s" % ("▲" if tr["side"] > 0 else "▼",
                                             (tr["entry_bar"] + pd.Timedelta(minutes=15)).strftime("%d/%m %H:%M"),
                                             side, fmt(tr["entry_price"]))
        if "exit_bar" in tr:
            text += "  →  ✖ %s SAI %+.2f%%" % ((tr["exit_bar"] + pd.Timedelta(minutes=15)).strftime("%d/%m %H:%M"), tr["result"])
        else:
            text += "  →  posição aberta: %+.2f%%" % (tr["side"] * (price / tr["entry_price"] - 1) * 100)
        events.append(text)
    legend = ("   fundo verde/vermelho = oportunidades dos agentes LONG/SHORT (simulação) | "
              "▲ ▼ ✖ = entradas e saídas REAIS do bot | ┈ preço atual")
    last_close = (index[-1] + pd.Timedelta(minutes=15)).strftime("%d/%m %H:%M")
    title = "   📈 BTCUSDT perp 15m — últimas %dh até %s UTC" % (len(index) // 4, last_close)
    ev = ["   " + e for e in events[-6:]] or ["   nenhum trade real do bot neste período"]

    colored = "\n".join([title] + lines_c + [FG["axis"] + axis_line + RESET, legend] + ev)
    plain = "\n".join([title] + lines_p + [axis_line, legend] + ev)
    try:
        TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        TEXT_PATH.write_text(plain + "\n", encoding="utf-8")
    except OSError:
        pass
    return colored


# --- Imagem PNG (mesmo estilo dos graficos de marcacoes) ----------------------
CHART_PATH = ROOT / "logs" / "charts" / "espelho.png"
PAGE_PATH = ROOT / "logs" / "charts" / "espelho.html"
_PAGE = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="60"><title>Espelho dos agentes</title>
<style>body{margin:0;background:#fff;font:14px sans-serif;color:#222}img{width:100%%;height:auto;display:block}
p{margin:8px 16px}</style></head><body><img src="espelho.png?t=%(stamp)s" alt="grafico do espelho">
<p>Atualizado a cada candle de 15m (%(closed)s UTC). A pagina recarrega sozinha a cada minuto.</p></body></html>"""


def render_png(history: pd.DataFrame, paths: Dict[str, pd.DataFrame], view: Optional[dict], hours: int = 72) -> Path:
    """Candles, stretches each agent held (green long, red short), entries, exits and current price."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = history.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    bars = frame.loc[frame.index > frame.index[-1] - pd.Timedelta(hours=hours), ["open", "high", "low", "close"]].astype(float)
    x = {t: i for i, t in enumerate(bars.index)}
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=100)
    up = bars["close"] >= bars["open"]
    for mask, color in ((up, "#26a69a"), (~up, "#ef5350")):
        sub = bars[mask]
        idx = [x[t] for t in sub.index]
        ax.vlines(idx, sub["low"], sub["high"], color=color, linewidth=0.7)
        ax.bar(idx, (sub["close"] - sub["open"]).abs().clip(lower=1e-9), bottom=sub[["open", "close"]].min(axis=1),
               width=0.7, color=color, linewidth=0)
    span = bars["high"].max() - bars["low"].min()
    n = 0
    for agent, path in (paths or {}).items():
        if path is None or path.empty:
            continue
        for tr in trades_of(path, frame["close"]):
            end = tr.get("exit_bar", bars.index[-1])
            if end < bars.index[0]:
                continue
            a = x.get(max(tr["entry_bar"], bars.index[0]), 0)
            b = x.get(min(end, bars.index[-1]), len(bars) - 1)
            ax.axvspan(a - 0.5, b + 0.5, color="#2e7d32" if tr["side"] > 0 else "#c62828", alpha=0.12, linewidth=0)
    for tr in real_trades(frame["close"]):
            if tr.get("exit_bar", bars.index[-1]) < bars.index[0]:
                continue
            n += 1
            if tr["entry_bar"] in x:
                i, y = x[tr["entry_bar"]], tr["entry_price"]
                marker, color = ("^", "#1b5e20") if tr["side"] > 0 else ("v", "#b71c1c")
                ax.scatter(i, y, marker=marker, s=120, color=color, zorder=5)
                ax.annotate("BOT ENTROU %s" % ("LONG" if tr["side"] > 0 else "SHORT"), (i, y),
                            xytext=(0, -18 if tr["side"] > 0 else 12), textcoords="offset points",
                            ha="center", fontsize=8, color=color, fontweight="bold")
            if "exit_bar" in tr and tr["exit_bar"] in x:
                i = x[tr["exit_bar"]]
                ax.scatter(i, tr["exit_price"], marker="X", s=100, color="black", zorder=5)
                ax.annotate("SAI %+.1f%%" % tr["result"], (i, tr["exit_price"]), xytext=(6, -14),
                            textcoords="offset points", fontsize=8, fontweight="bold")
    price = float((view or {}).get("close") or bars["close"].iloc[-1])
    ax.axhline(price, color="#1565c0", linestyle="--", linewidth=1.1)
    ax.annotate("preço atual $%s" % "{:,.1f}".format(price), (len(bars) - 1, price), xytext=(6, 5),
                textcoords="offset points", color="#1565c0", fontsize=9, fontweight="bold")
    ticks = [i for i, t in enumerate(bars.index) if t.minute == 0 and t.hour % 12 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([bars.index[i].strftime("%d/%m %Hh") for i in ticks], fontsize=8)
    ax.set_xlim(-1, len(bars) + 18)
    ax.set_ylim(bars["low"].min() - 0.04 * span, bars["high"].max() + 0.07 * span)
    ax.grid(alpha=0.25)
    state = []
    for sh in (view or {}).get("shadows", []):
        txt = "fora" if sh.side == 0 else ("COMPRADO" if sh.side > 0 else "VENDIDO")
        state.append("%s: %s%s" % (AGENT_NAMES.get(sh.agent, sh.agent), txt, " (NOVA ENTRADA)" if sh.entered_on_last_bar else ""))
    acc = (view or {}).get("account_side", 0)
    closed = (bars.index[-1] + pd.Timedelta(minutes=15)).strftime("%d/%m/%Y %H:%M")
    ax.set_title("BTCUSDT perp 15m, últimas %dh até %s UTC: %d trades reais do bot (marcas); fundo = oportunidades dos agentes (verde long, vermelho short)\n%s | conta: %s"
                 % (hours, closed, n, " | ".join(state) or "-", "sem posição" if not acc else ("comprada" if acc > 0 else "vendida")),
                 fontsize=10)
    fig.tight_layout()
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHART_PATH.with_name("espelho.tmp.png")
    fig.savefig(tmp)
    plt.close(fig)
    tmp.replace(CHART_PATH)
    PAGE_PATH.write_text(_PAGE % {"stamp": closed.replace("/", "").replace(" ", "").replace(":", ""), "closed": closed},
                         encoding="utf-8")
    return CHART_PATH
