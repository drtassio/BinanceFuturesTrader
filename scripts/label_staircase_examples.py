"""
label_staircase_examples.py
Rotula TODAS as escadas dos últimos 2 anos no gráfico de 15m, com o contexto
dos timeframes maiores (1h, 4h, 1D) antes e depois de cada exemplo.

A regra do professor (a mesma do gráfico):
  1. CAIXA   - o preço fica preso numa faixa estreita nas 12h anteriores.
  2. NASCE   - 2+ degraus grandes seguidos (corpo >= 1.5 x ATR 15m) na mesma direção.
  3. CONFIRMA- o fechamento sai da caixa. Só aqui o professor entra (nunca antes).
  4. SURFA   - segura enquanto a estrutura de 4h não quebra (stop sobe/desce com ela).
  5. SAI     - fechamento de 15m do outro lado do stop estrutural.

Saídas:
  data/teacher_staircase_labels.parquet   - rótulo por candle de 15m (+ features causais)
  data/teacher_staircase_events.parquet   - uma linha por exemplo (escada)
  reports/staircase_examples/             - gráfico 15m/1h/4h de cada exemplo + index.html

Colunas com prefixo hs_ usam o futuro (hindsight): servem para AVALIAR e SEGMENTAR,
nunca como observação do agente.
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

# Parâmetros fixos. Escolhidos pelo PF do período de treino numa grade de 72
# combinações; todas as 40 combinações lucrativas no treino também foram no holdout.
STEP_BODY_ATR = 1.5      # corpo mínimo de um degrau, em ATR(14) de 15m
MIN_STEPS = 2            # degraus consecutivos para confirmar
BOX_BARS = 48            # caixa = 12h de 15m antes dos degraus
BOX_GAP = 3              # a caixa termina 3 candles antes (exclui os próprios degraus)
BOX_MAX_ATR = 8.0        # caixa comprimida: largura <= 8 x ATR
STOP_ATR_4H = 0.5        # folga do stop estrutural de 4h
FEE_PER_SIDE = 0.0005 + 0.0002   # taker 0.05% + slippage 0.02%
HOLDOUT_START = pd.Timestamp("2026-01-24", tz="UTC")
YEARS = 2


def _utc(idx):
    idx = pd.to_datetime(idx)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def load():
    df = pd.read_parquet(ROOT_DIR / "data/market_raw/btc_perp_15m.parquet")
    df.index = _utc(df.index)
    df = df.sort_index()[["open", "high", "low", "close", "volume", "taker_buy_base"]]
    df = df[~df.index.duplicated(keep="last")]
    fund = pd.read_parquet(ROOT_DIR / "data/market_raw/btc_funding.parquet")
    fund.index = _utc(fund.index)
    return df, fund["fundingRate"].astype(float).sort_index()


def _atr(d, n):
    tr = np.maximum(d.high - d.low, np.maximum((d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()))
    return tr.rolling(n).mean()


def _htf(d15, rule):
    """Candle de timeframe maior, carimbado no início do último 15m que o compõe:
    o 15m só enxerga o candle maior depois que ele fechou (sem lookahead)."""
    h = d15.resample(rule, closed="left", label="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    h.index = h.index + pd.Timedelta(rule) - pd.Timedelta("15min")
    return h


def build_features(d, funding):
    f = d.copy()
    f["atr_15m"] = _atr(d, 14)
    f["step_body_atr"] = (d.close - d.open) / f.atr_15m
    up = (f.step_body_atr >= STEP_BODY_ATR) & (d.close > d.close.shift())
    dn = (f.step_body_atr <= -STEP_BODY_ATR) & (d.close < d.close.shift())
    f["step_count_up"] = up.groupby((~up).cumsum()).cumsum().astype(int)
    f["step_count_down"] = dn.groupby((~dn).cumsum()).cumsum().astype(int)

    f["box_high"] = d.high.rolling(BOX_BARS).max().shift(BOX_GAP)
    f["box_low"] = d.low.rolling(BOX_BARS).min().shift(BOX_GAP)
    f["box_width_atr"] = (f.box_high - f.box_low) / f.atr_15m
    f["box_breakout_atr"] = np.where(d.close > f.box_high, (d.close - f.box_high) / f.atr_15m,
                            np.where(d.close < f.box_low, (d.close - f.box_low) / f.atr_15m, 0.0))

    delta = 2 * d.taker_buy_base - d.volume
    f["flow_imbalance"] = delta / (d.volume + 1e-8)
    f["cvd_z"] = delta.rolling(16).sum() / (delta.rolling(96).std() * 4 + 1e-8)
    f["vol_expansion"] = d.volume / (d.volume.rolling(20).mean() + 1e-8)

    for rule, fast, slow in (("1h", 9, 21), ("4h", 12, 26), ("1D", 20, 50)):
        h = _htf(d, rule)
        tag = rule.lower()
        h[f"trend_{tag}"] = np.where(h.close.ewm(span=fast).mean() > h.close.ewm(span=slow).mean(), 1, -1)
        cols = [f"trend_{tag}"]
        if rule == "4h":
            h["atr_4h"] = _atr(h, 14)
            h["struct_low_4h"] = h.low.rolling(2).min() - STOP_ATR_4H * h.atr_4h
            h["struct_high_4h"] = h.high.rolling(2).max() + STOP_ATR_4H * h.atr_4h
            cols += ["atr_4h", "struct_low_4h", "struct_high_4h"]
        f = f.join(h[cols].reindex(f.index, method="ffill"))

    f["dist_4h_stop_atr"] = np.where(f.trend_4h > 0, (d.close - f.struct_low_4h), (f.struct_high_4h - d.close)) / f.atr_15m
    # funding é cobrado a cada 8h: taxa só no candle de 15m que contém o horário de cobrança
    f["funding_charge"] = funding.reindex(f.index).fillna(0.0)
    return f.dropna()


def detect_events(f):
    C, H, L = f.close.values, f.high.values, f.low.values
    us, ds = f.step_count_up.values, f.step_count_down.values
    bw, bh, bl, cz = f.box_width_atr.values, f.box_high.values, f.box_low.values, f.cvd_z.values
    s_lo, s_hi, fund = f.struct_low_4h.values, f.struct_high_4h.values, f.funding_charge.values
    idx, n = f.index, len(f)

    events, pos = [], 0
    for i in range(1, n):
        if pos == 0:
            long_ = us[i] >= MIN_STEPS and bw[i] <= BOX_MAX_ATR and C[i] > bh[i] and cz[i] >= 0
            short = ds[i] >= MIN_STEPS and bw[i] <= BOX_MAX_ATR and C[i] < bl[i] and cz[i] <= 0
            if long_ or short:
                pos = 1 if long_ else -1
                ev = {"side": "BULL" if pos == 1 else "BEAR", "entry_i": i,
                      "birth_i": i - (us[i] if pos == 1 else ds[i]) + 1,
                      "entry_price": C[i], "box_high": bh[i], "box_low": bl[i],
                      "box_width_atr": bw[i], "cvd_z_entry": cz[i], "funding": 0.0}
                ev["box_start_i"] = max(0, ev["birth_i"] - BOX_GAP + 1 - BOX_BARS)
                stop = bl[i] if pos == 1 else bh[i]
                ev["initial_stop"] = stop
                best, worst, best_i = C[i], C[i], i
        else:
            ev["funding"] += fund[i] * pos
            stop = max(stop, s_lo[i]) if pos == 1 else min(stop, s_hi[i])
            fav, adv = (H[i], L[i]) if pos == 1 else (L[i], H[i])
            if (fav - best) * pos > 0:
                best, best_i = fav, i
            if (worst - adv) * pos > 0:
                worst = adv
            hit = C[i] < stop if pos == 1 else C[i] > stop
            if hit or i == n - 1:
                ep = ev["entry_price"]
                ev.update({"exit_i": i, "exit_price": C[i], "exit_reason": "stop_4h" if hit else "end_of_data",
                           "ret_gross": pos * (C[i] - ep) / ep,
                           "hs_mfe": pos * (best - ep) / ep, "hs_mae": pos * (worst - ep) / ep, "hs_mfe_i": best_i,
                           "risk": abs(ep - ev["initial_stop"]) / ep})
                ev["ret_net"] = ev["ret_gross"] - 2 * FEE_PER_SIDE - ev["funding"]
                ev["r_multiple"] = ev["ret_net"] / ev["risk"] if ev["risk"] > 0 else np.nan
                events.append(ev)
                pos = 0

    ev = pd.DataFrame(events)
    for col in ("box_start_i", "birth_i", "entry_i", "exit_i", "hs_mfe_i"):
        ev[col.replace("_i", "_time")] = idx[ev[col].values]
    for tf in ("1h", "4h", "1d"):
        ev[f"trend_{tf}_entry"] = f[f"trend_{tf}"].values[ev.entry_i.values]
    ev["hours"] = (ev.exit_i - ev.entry_i) * 0.25
    ev["outcome"] = np.where(ev.ret_net > 0, "win", "false_breakout")
    ev["split"] = np.where(ev.entry_time >= HOLDOUT_START, "holdout", "train")
    return ev


def label_bars(f, ev):
    n = len(f)
    action = np.zeros(n, dtype=np.int8)
    phase = np.full(n, "none", dtype=object)
    event_id = np.full(n, -1, dtype=np.int32)
    outcome = np.full(n, "", dtype=object)

    # Faixas comprimidas sem escada = território do Range
    phase[(f.box_width_atr.values <= BOX_MAX_ATR)] = "range"
    # caixas que antecederam uma escada (hindsight), depois as escadas por cima
    for k, e in ev.iterrows():
        phase[e.box_start_i:e.birth_i] = "box"
    for k, e in ev.iterrows():
        sign = 1 if e.side == "BULL" else -1
        phase[e.birth_i:e.entry_i + 1] = "birth"
        phase[e.entry_i + 1:e.exit_i] = "surf"
        phase[e.exit_i] = "exit"
        action[e.entry_i:e.exit_i] = sign   # posição desejada após o fechamento do candle
        event_id[e.box_start_i:e.exit_i + 1] = k
        outcome[e.box_start_i:e.exit_i + 1] = e.outcome

    side_of = np.full(n, "", dtype=object)
    for k, e in ev.iterrows():
        side_of[e.birth_i:e.exit_i + 1] = e.side.lower()
    specialist = np.where(np.isin(phase, ["birth", "surf", "exit"]), side_of,
                 np.where(np.isin(phase, ["box", "range"]), "range", "flat"))

    out = f.copy()
    out["teacher_action"] = action
    out["hs_phase"] = phase
    out["hs_specialist"] = specialist
    out["hs_event_id"] = event_id
    out["hs_event_outcome"] = outcome
    return out


def summarize(ev):
    def line(sub, name):
        if sub.empty:
            return f"   {name:22s} 0 exemplos"
        w, l = sub.ret_net[sub.ret_net > 0].sum(), -sub.ret_net[sub.ret_net <= 0].sum()
        comp = ((1 + sub.ret_net).prod() - 1) * 100
        return (f"   {name:22s} {len(sub):4d} exemplos | composto {comp:+7.1f}% | PF {w / l if l else float('inf'):4.2f} | "
                f"acerto {(sub.ret_net > 0).mean() * 100:4.1f}% | R médio {sub.r_multiple.mean():+.2f} | "
                f"duração mediana {sub.hours.median():.1f}h")
    print("\n" + "=" * 100)
    print("ESCADAS ROTULADAS (custos: 0.14% ida+volta + funding real)")
    print("=" * 100)
    print(line(ev, "TODOS"))
    for side in ("BULL", "BEAR"):
        for split in ("train", "holdout"):
            print(line(ev[(ev.side == side) & (ev.split == split)], f"{side} {split}"))
    print(f"\n   Ganho que existia no melhor ponto (hindsight MFE) mediano: {ev.hs_mfe.median() * 100:.2f}% | "
          f"capturado mediano: {ev.ret_net.median() * 100:.2f}%")


def _candles(ax, d, width):
    up = d.close >= d.open
    x = d.index.tz_convert(None)
    for mask, color in ((up, "#26a69a"), (~up, "#ef5350")):
        ax.vlines(x[mask], d.low[mask], d.high[mask], color=color, linewidth=0.6)
        ax.bar(x[mask], (d.close - d.open)[mask].abs().clip(lower=1e-6), bottom=np.minimum(d.open, d.close)[mask],
               width=width, color=color, linewidth=0)


def plot_event(k, e, d15, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [3, 1.6, 1.6]})
    t0, t1 = e.box_start_time - pd.Timedelta("4h"), e.exit_time + pd.Timedelta("4h")
    views = [("15m", d15.loc[t0:t1], pd.Timedelta("15min")),
             ("1h", _htf_plain(d15, "1h").loc[t0 - pd.Timedelta("3D"):t1 + pd.Timedelta("1D")], pd.Timedelta("1h")),
             ("4h", _htf_plain(d15, "4h").loc[t0 - pd.Timedelta("12D"):t1 + pd.Timedelta("4D")], pd.Timedelta("4h"))]
    color = "#26a69a" if e.side == "BULL" else "#ef5350"
    for ax, (tf, d, step) in zip(axes, views):
        _candles(ax, d, step.total_seconds() / 86400 * 0.7)
        naive = lambda t: t.tz_convert(None)
        ax.axvspan(naive(e.box_start_time), naive(e.birth_time), color="#9e9e9e", alpha=0.15, label="caixa")
        ax.axvspan(naive(e.birth_time), naive(e.entry_time), color=color, alpha=0.30, label="degraus (nasce)")
        ax.axvspan(naive(e.entry_time), naive(e.exit_time), color=color, alpha=0.10, label="surf")
        ax.hlines([e.box_high, e.box_low], naive(e.box_start_time), naive(e.birth_time), color="#757575", linestyle="--", linewidth=0.8)
        ax.set_ylabel(tf)
        ax.grid(alpha=0.2)
    ax = axes[0]
    ax.scatter([e.entry_time.tz_convert(None)], [e.entry_price], marker="^" if e.side == "BULL" else "v", color=color, s=90, zorder=5, label="entrada")
    ax.scatter([e.exit_time.tz_convert(None)], [e.exit_price], marker="x", color="black", s=70, zorder=5, label="saída")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"#{k:03d} {e.side} {e.entry_time:%Y-%m-%d %H:%M} UTC | {e.split} | {e.outcome} | "
                 f"líquido {e.ret_net * 100:+.2f}% ({e.r_multiple:+.1f}R) | MFE {e.hs_mfe * 100:+.2f}% | {e.hours:.1f}h", fontsize=10)
    fig.tight_layout()
    path = out_dir / f"event_{k:03d}_{e.side.lower()}_{e.entry_time:%Y%m%d_%H%M}.png"
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path.name


def _htf_plain(d15, rule, _cache={}):
    if rule not in _cache:
        _cache[rule] = d15.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return _cache[rule]


def plot_overview(f, ev, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 6))
    c = f.close.resample("4h").last().dropna()
    ax.plot(c.index.tz_convert(None), c.values, color="#607d8b", linewidth=0.7)
    for _, e in ev.iterrows():
        col = "#26a69a" if e.side == "BULL" else "#ef5350"
        ax.axvspan(e.entry_time.tz_convert(None), e.exit_time.tz_convert(None), color=col,
                   alpha=0.45 if e.outcome == "win" else 0.15)
    ax.axvline(HOLDOUT_START.tz_convert(None), color="black", linestyle="--", linewidth=1)
    ax.text(HOLDOUT_START.tz_convert(None), ax.get_ylim()[1], " holdout →", va="top")
    ax.set_title("BTCUSDT perp — escadas rotuladas (verde = Bull, vermelho = Bear; forte = ganhou, claro = falso rompimento)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=90)
    plt.close(fig)


def write_index(ev, names, out_dir):
    rows = "\n".join(
        f'<figure class="{e.side.lower()} {e.outcome}"><img loading="lazy" src="{nm}"><figcaption>#{k:03d} {e.side} '
        f'{e.entry_time:%Y-%m-%d %H:%M} · {e.split} · {e.ret_net * 100:+.2f}%</figcaption></figure>'
        for (k, e), nm in zip(ev.iterrows(), names))
    html = f"""<!doctype html><meta charset="utf-8"><title>Escadas rotuladas</title>
<style>body{{font-family:sans-serif;margin:16px}}img{{max-width:100%}}figure{{margin:0 0 24px}}
.false_breakout figcaption{{color:#c62828}}.win figcaption{{color:#2e7d32}}</style>
<h1>Escadas rotuladas — {len(ev)} exemplos</h1><img src="overview.png">
{rows}"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    d, funding = load()
    f = build_features(d, funding)
    end = f.index.max()
    start = end - pd.DateOffset(years=YEARS)
    print(f"Janela rotulada: {start:%Y-%m-%d} → {end:%Y-%m-%d %H:%M} UTC (aquecimento de indicadores antes disso)")

    ev = detect_events(f)
    ev = ev[ev.entry_time >= start].reset_index(drop=True)
    labels = label_bars(f, ev).loc[start:]

    featured = pd.read_parquet(ROOT_DIR / "data/featured_data.parquet", columns=["close"])
    labels["in_featured_data"] = labels.index.isin(_utc(featured.index))
    cov = labels.in_featured_data.mean() * 100

    labels.to_parquet(ROOT_DIR / "data/teacher_staircase_labels.parquet")
    ev.to_parquet(ROOT_DIR / "data/teacher_staircase_events.parquet")
    out_dir = ROOT_DIR / "reports/staircase_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    ev.drop(columns=[c for c in ev.columns if c.endswith("_i")]).to_csv(out_dir / "events.csv", index=False)

    summarize(ev)
    print("\nOcupação dos rótulos por candle de 15m:")
    print(labels.hs_phase.value_counts(normalize=True).mul(100).round(1).to_string())
    print(f"\nCobertura do featured_data.parquet (autoencoder/orderbook): {cov:.1f}% dos candles rotulados")

    if not args.no_charts:
        d_plot = d.loc[start - pd.Timedelta("20D"):]
        plot_overview(labels, ev, out_dir)
        names = [plot_event(k, e, d_plot, out_dir) for k, e in ev.iterrows()]
        write_index(ev, names, out_dir)
        print(f"\nGráficos: {len(names)} exemplos em {out_dir}")


if __name__ == "__main__":
    main()
