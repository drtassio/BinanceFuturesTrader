"""
mark_legs.py
Marca no gráfico de 15m TODAS as pernas de tendência dos últimos 2 anos, com
entrada e saída, do jeito que um trader marcaria olhando o gráfico pronto.

O zigzag divide o preço em ondas (reversões de SWING_MOVE). Uma onda é PERNA
de tendência quando anda pelo menos LEG_MIN_MOVE e a pelo menos LEG_MIN_SPEED
por hora; as ondas pequenas ou lentas são o vai-e-vem do lateral. As marcas:

  ENTRADA  primeiro candle que fecha CONFIRM_MOVE além do ponto onde a perna
           começou: o movimento já começou, nada foi adivinhado.
  SAÍDA    primeiro candle que fecha CONFIRM_MOVE de volta a partir do extremo
           da perna: a perna acabou e isso já está confirmado.

As marcas usam o gráfico inteiro (sabem onde cada perna termina). Elas são os
EXEMPLOS para ensinar os agentes, não uma estratégia: ao vivo ninguém sabe onde
a perna termina, e é isso que o modelo precisa aprender a reconhecer.

Saídas:
  data/teacher_leg_labels.parquet     rótulo por candle (+1 long, -1 short, 0 fora)
  data/teacher_legs.parquet           uma linha por perna
  reports/leg_marks/semana_*.png      gráfico de cada semana com as marcas
  reports/leg_marks/index.html
"""

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SWING_MOVE = 0.012        # ondas do zigzag: reversão de pelo menos 1,2%
LEG_MIN_MOVE = 0.02       # perna: onda de pelo menos 2% ...
LEG_MIN_SPEED = 0.001     # ... andando pelo menos 0,1% por hora; ondas lentas são lateral
CONFIRM_MOVE = 0.005      # entrada/saída confirmadas após 0,5% a favor/contra
COST_ROUND_TRIP = 0.0014  # 0,05% taker + 0,02% slippage por lado
YEARS = 2


def load_close():
    df = pd.read_parquet(ROOT_DIR / "data/market_raw/btc_perp_15m.parquet")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()[["open", "high", "low", "close", "volume"]]
    return df[~df.index.duplicated(keep="last")]


def zigzag_pivots(close: np.ndarray, threshold: float):
    """Índices dos fundos e topos alternados com reversão >= threshold."""
    pivots = [0]
    direction = 0
    extreme = 0
    for i in range(1, len(close)):
        if direction == 1 and close[i] > close[extreme]:
            extreme = i
        elif direction == -1 and close[i] < close[extreme]:
            extreme = i
        if direction == 0:
            if close[i] >= close[0] * (1 + threshold):
                direction, extreme = 1, i
            elif close[i] <= close[0] * (1 - threshold):
                direction, extreme = -1, i
        elif direction == 1 and close[i] <= close[extreme] * (1 - threshold):
            pivots.append(extreme)
            direction, extreme = -1, i
        elif direction == -1 and close[i] >= close[extreme] * (1 + threshold):
            pivots.append(extreme)
            direction, extreme = 1, i
    return pivots


def mark_legs(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].to_numpy(float)
    pivots = zigzag_pivots(close, SWING_MOVE)
    legs = []
    for a, b in zip(pivots[:-1], pivots[1:]):
        side = 1 if close[b] > close[a] else -1
        start, end = close[a], close[b]
        move = side * (end / start - 1)
        hours = max((b - a) * 0.25, 0.25)
        if move < LEG_MIN_MOVE or move / hours < LEG_MIN_SPEED:
            continue   # onda pequena ou lenta: vai-e-vem, fica para o lateral
        # entrada: primeiro fechamento CONFIRM_MOVE além do início da perna
        seg = close[a + 1:b + 1]
        hit = np.nonzero(side * (seg / start - 1) >= CONFIRM_MOVE)[0]
        if not len(hit):
            continue
        entry = a + 1 + hit[0]
        # saída: primeiro fechamento CONFIRM_MOVE de volta a partir do extremo
        after = close[b + 1:]
        back = np.nonzero(side * (after / end - 1) <= -CONFIRM_MOVE)[0]
        exit_ = b + 1 + back[0] if len(back) else len(close) - 1
        gross = side * (close[exit_] / close[entry] - 1)
        legs.append({
            "side": "LONG" if side > 0 else "SHORT", "sign": side,
            "start_time": df.index[a], "entry_time": df.index[entry], "extreme_time": df.index[b],
            "exit_time": df.index[exit_],
            "start_price": start, "entry_price": close[entry], "extreme_price": end, "exit_price": close[exit_],
            "leg_move": move, "captured_gross": gross, "captured_net": gross - COST_ROUND_TRIP,
            "hours": (exit_ - entry) * 0.25,
            "start_i": a, "entry_i": entry, "extreme_i": b, "exit_i": exit_,
        })
    return pd.DataFrame(legs)


def per_bar_labels(df: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    target = np.zeros(n, dtype=np.int8)       # posição que o professor quer: +1, -1, 0
    phase = np.full(n, "fora", dtype=object)
    leg_id = np.full(n, -1, dtype=np.int32)
    for k, e in legs.iterrows():
        target[e.entry_i:e.exit_i] = e.sign
        phase[e.start_i:e.entry_i] = "inicio"
        phase[e.entry_i] = "entrada"
        phase[e.entry_i + 1:e.exit_i] = "surf"
        phase[e.exit_i] = "saida"
        leg_id[e.start_i:e.exit_i + 1] = k
    out = df[["open", "high", "low", "close"]].copy()
    out["hs_target_position"] = target
    out["hs_phase"] = phase
    out["hs_leg_id"] = leg_id
    return out


def plot_weeks(df, legs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    for week_start in pd.date_range(df.index.min().normalize(), df.index.max(), freq="7D"):
        week_end = week_start + pd.Timedelta("7D")
        w = df.loc[week_start:week_end]
        if len(w) < 50:
            continue
        fig, ax = plt.subplots(figsize=(18, 6.5))
        up = w.close >= w.open
        for m, c in ((up, "#26a69a"), (~up, "#ef5350")):
            ax.vlines(w.index[m], w.low[m], w.high[m], color=c, lw=0.6)
            ax.bar(w.index[m], (w.close - w.open)[m].abs().clip(lower=1e-6),
                   bottom=np.minimum(w.open, w.close)[m], width=0.0072, color=c, lw=0)
        inside = legs[(legs.exit_time >= week_start) & (legs.entry_time <= week_end)]
        for _, e in inside.iterrows():
            col = "#1b5e20" if e.sign > 0 else "#b71c1c"
            a, b = max(e.entry_time, week_start), min(e.exit_time, week_end)
            ax.axvspan(a, b, color=col, alpha=0.12)
            if week_start <= e.entry_time <= week_end:
                ax.scatter([e.entry_time], [e.entry_price], marker="^" if e.sign > 0 else "v", s=170, color=col, zorder=6)
                ax.annotate("ENTRA %s" % e.side, (e.entry_time, e.entry_price), xytext=(-10, -28 if e.sign > 0 else 18),
                            textcoords="offset points", fontsize=9, color=col, weight="bold")
            if week_start <= e.exit_time <= week_end:
                ax.scatter([e.exit_time], [e.exit_price], marker="X", s=130, color="black", zorder=6)
                ax.annotate("SAI %+.1f%%" % (e.captured_net * 100), (e.exit_time, e.exit_price), xytext=(4, 10),
                            textcoords="offset points", fontsize=9, weight="bold")
        ax.set_title("BTCUSDT perp 15m, semana de %s: %d pernas (verde = long, vermelho = short, sem cor = lateral/fora)"
                     % (week_start.strftime("%d/%m/%Y"), len(inside)))
        ax.grid(alpha=0.2)
        fig.tight_layout()
        name = "semana_%s.png" % week_start.strftime("%Y%m%d")
        fig.savefig(out_dir / name, dpi=75)
        plt.close(fig)
        names.append(name)
    html = ("<!doctype html><meta charset='utf-8'><title>Pernas marcadas</title>"
            "<style>body{font-family:sans-serif;margin:16px}img{max-width:100%}</style>")
    html += "<h1>Pernas marcadas — %d exemplos</h1>" % len(legs) + "".join("<img loading='lazy' src='%s'>" % n for n in names)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    df = load_close()
    df = df.loc[df.index.max() - pd.DateOffset(years=YEARS) - pd.Timedelta("7D"):]
    legs = mark_legs(df)
    start = df.index.max() - pd.DateOffset(years=YEARS)
    legs = legs[legs.entry_time >= start].reset_index(drop=True)
    labels = per_bar_labels(df, legs).loc[start:]

    labels.to_parquet(ROOT_DIR / "data/teacher_leg_labels.parquet")
    legs.to_parquet(ROOT_DIR / "data/teacher_legs.parquet")
    out_dir = ROOT_DIR / "reports/leg_marks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("semana_*.png"):
        old.unlink()

    print("=" * 90)
    print("PERNAS MARCADAS (%s a %s)" % (start.date(), df.index.max()))
    print("=" * 90)
    for side in ("LONG", "SHORT"):
        s = legs[legs.side == side]
        print("  %-5s %4d pernas | movimento mediano %.2f%% | capturado líquido mediano %+.2f%% | duração mediana %.1fh"
              % (side, len(s), s.leg_move.median() * 100, s.captured_net.median() * 100, s.hours.median()))
    print("  posição desejada em %.1f%% dos candles (long %.1f%%, short %.1f%%); lateral/fora %.1f%%" % (
        (labels.hs_target_position != 0).mean() * 100, (labels.hs_target_position > 0).mean() * 100,
        (labels.hs_target_position < 0).mean() * 100, (labels.hs_target_position == 0).mean() * 100))
    if not args.no_charts:
        names = plot_weeks(df.loc[start:], legs, out_dir)
        print("  gráficos: %d semanas em %s" % (len(names), out_dir))


if __name__ == "__main__":
    main()
