"""
research_mark_triggers.py
Quais indicadores, vetores do autoencoder e medidas de fluxo acionam as entradas
e saídas marcadas no gráfico (scripts/mark_legs.py), por timeframe.

Duas perguntas diferentes:
  Q1  O que DESCREVE o momento marcado? Entradas marcadas contra candles comuns,
      saídas marcadas contra candles dentro de uma perna. (Descritivo: usa o
      mesmo instante, mas quem escolheu o instante sabia o futuro.)
  Q2  O que SEPARA a entrada que virou perna do falso começo, no mesmo tipo de
      instante (onda confirmada 0,25% a partir do último fundo/topo)? É o que um
      agente precisa para operar. AUC por feature e AUC walk-forward por grupo.

Grupos: 5m (dentro do candle de 15m), 15m, 1h, 4h, fluxo/tape (agressão, CVD,
VPIN, absorção, tamanho dos negócios, funding) e autoencoder (32 latentes do
SDAE temporal, calculados com o modelo e o scaler salvos em models_ai/autoencoder).

    python scripts/research_mark_triggers.py
"""
import json
import sys
import warnings
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data" / "featured_data_causal.parquet"
MARKS = ROOT / "data" / "teacher_leg_labels_fino.parquet"
FLOW = ("cz_aggression", "cz_cvd", "cz_absorption", "cz_flow_price_div", "cz_vpin", "cz_avg_trade_size",
        "cz_trade_intensity", "cz_buy_price", "cz_sell_price", "cz_vwap_dist", "cz_funding", "funding_rate")


def group_of(col: str) -> str:
    if col.startswith("ae_"):
        return "autoencoder"
    if col.startswith(("cz_steps5", "cz_body5")):
        return "5m"
    if col.startswith(FLOW):
        return "fluxo/tape"
    if col.endswith("_4h") or col.startswith(("cz_struct_4h", "cz_trend_4h")):
        return "4h"
    if col.endswith("_1h"):
        return "1h"
    if col.endswith("_15m") or col.startswith("cz_"):
        return "15m"
    return "outros"


def autoencoder_latents(frame: pd.DataFrame) -> pd.DataFrame:
    import torch
    from config.settings import AIConfig
    from feature_engineering.temporal_autoencoder import TemporalAutoencoderPipeline

    pipe = TemporalAutoencoderPipeline(AIConfig())
    if getattr(pipe, "autoencoder", None) is None:
        pipe.load_state()
    cols = list(pipe.feature_columns)
    missing = [c for c in cols if c not in frame.columns]
    x = frame.reindex(columns=cols).ffill().fillna(0.0).to_numpy(np.float32)
    if pipe.scaler_fitted:
        x = pipe.scaler.transform(x).astype(np.float32)
    seq = int(pipe.hyperparams.get("seq_length", 32))
    regime = frame["regime"].fillna(2).astype(int).clip(0, 2).to_numpy() if "regime" in frame else np.full(len(frame), 2)
    model = pipe.autoencoder.eval()
    device = next(model.parameters()).device
    out = np.full((len(frame), int(pipe.hyperparams["latent_dim"])), np.nan, np.float32)
    ends = np.arange(seq - 1, len(frame))
    with torch.no_grad():
        for s in range(0, len(ends), 2048):
            e = ends[s:s + 2048]
            windows = np.stack([x[i - seq + 1:i + 1] for i in e])
            mu, _ = model.encode(torch.from_numpy(windows).to(device),
                                 regime_labels=torch.from_numpy(regime[e]).long().to(device))
            out[e] = mu.cpu().numpy()
    print("  autoencoder: %d latentes, %d colunas de entrada (%d ausentes no dataset preenchidas com 0)"
          % (out.shape[1], len(cols), len(missing)))
    return pd.DataFrame(out, index=frame.index, columns=["ae_%02d" % k for k in range(out.shape[1])])


def auc_table(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    rows = []
    for c in X.columns:
        v = X[c].to_numpy(float)
        ok = np.isfinite(v)
        if ok.sum() < 50 or np.nanstd(v[ok]) == 0 or len(np.unique(y[ok])) < 2:
            continue
        a = roc_auc_score(y[ok], v[ok])
        rows.append((c, group_of(c), a, abs(a - 0.5)))
    return pd.DataFrame(rows, columns=["feature", "grupo", "auc", "forca"]).sort_values("forca", ascending=False)


def walk_forward_auc(X: np.ndarray, y: np.ndarray, t: pd.Series) -> float:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    q = t.dt.to_period("Q")
    quarters = sorted(q.unique())
    p = np.full(len(y), np.nan)
    for qq in quarters[2:]:
        tr, te = (q < qq).values, (q == qq).values
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=60)
        p[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    ok = ~np.isnan(p)
    return float(roc_auc_score(y[ok], p[ok]))


def main():
    import research_5m_patterns as R

    df = pd.read_parquet(DATA).sort_index()
    marks = pd.read_parquet(MARKS)
    marks.index = pd.to_datetime(marks.index)
    df = df.loc[marks.index.min() - pd.Timedelta("2D"):]
    print("dados: %s a %s | %d candles de 15m" % (df.index.min(), df.index.max(), len(df)))
    ae = autoencoder_latents(df)
    df = df.join(ae)
    df = df.loc[marks.index.min():]
    feats = [c for c in df.select_dtypes("number").columns
             if group_of(c) != "outros" and not c.startswith(("ml_", "tp_")) and c not in ("regime",)]
    X = df[feats]
    phase = marks["hs_phase"].reindex(df.index).fillna("fora")
    target = marks["hs_target_position"].reindex(df.index).fillna(0)
    side_at = target.shift(-1).fillna(0)          # lado da perna que começa na entrada

    report = {}
    print("\nQ1  O que descreve os momentos marcados (AUC contra candles comuns; 0,5 = nada)")
    for label, mask, base in (
            ("ENTRADA LONG", (phase == "entrada") & (side_at > 0), phase == "fora"),
            ("ENTRADA SHORT", (phase == "entrada") & (side_at < 0), phase == "fora"),
            ("SAIDA de LONG", (phase == "saida") & (target.shift(1) > 0), (phase == "surf") & (target > 0)),
            ("SAIDA de SHORT", (phase == "saida") & (target.shift(1) < 0), (phase == "surf") & (target < 0))):
        sel = mask | base
        t = auc_table(X[sel], mask[sel].to_numpy().astype(int))
        report["Q1_" + label] = t.head(25).to_dict("records")
        by_group = t.groupby("grupo").forca.max().sort_values(ascending=False)
        print("\n  %s (%d marcas)  | força máxima por grupo: %s" % (
            label, int(mask.sum()), ", ".join("%s %.2f" % (g, 0.5 + v) for g, v in by_group.items())))
        for _, r in t.head(8).iterrows():
            print("     %-34s %-12s AUC %.3f  (%s)" % (r.feature, r.grupo, r.auc, "maior na marca" if r.auc > 0.5 else "menor na marca"))

    print("\nQ2  O que separa a onda que virou perna do falso começo (mesmo instante de confirmação)")
    d5 = R.load()
    d15 = R.resample_15m(d5)
    ev = R.wave_starts(d15.close.to_numpy(float), 0.006, 0.0025, 0.009)
    ev["t"] = d15.index[ev.i]
    ev = ev[ev.t.isin(df.index)].reset_index(drop=True)
    Xe = X.loc[ev.t].reset_index(drop=True)
    signed = [c for c in Xe.columns if not c.startswith(("ae_", "cz_box_width", "cz_vpin", "cz_atr", "cz_rvol",
                                                          "cz_hurst", "cz_entropy", "atr", "cz_trade_intensity"))]
    Xs = Xe.copy()
    Xs[signed] = Xs[signed].to_numpy() * ev.side.to_numpy()[:, None]
    y = ev.y.to_numpy()
    print("  %d começos de onda | viraram perna %.0f%%" % (len(ev), 100 * y.mean()))
    t = auc_table(Xs, y)
    report["Q2_features"] = t.head(30).to_dict("records")
    for _, r in t.head(12).iterrows():
        print("     %-34s %-12s AUC %.3f" % (r.feature, r.grupo, r.auc))
    print("\n  AUC walk-forward (fora da amostra) usando só cada grupo:")
    groups = {}
    for g in ("5m", "15m", "1h", "4h", "fluxo/tape", "autoencoder"):
        cols = [c for c in Xs.columns if group_of(c) == g]
        if cols:
            groups[g] = walk_forward_auc(Xs[cols].to_numpy(float), y, ev.t)
            print("     %-12s %3d features  AUC %.3f" % (g, len(cols), groups[g]))
    groups["todos"] = walk_forward_auc(Xs.to_numpy(float), y, ev.t)
    print("     %-12s %3d features  AUC %.3f" % ("todos", Xs.shape[1], groups["todos"]))
    report["Q2_groups_walk_forward_auc"] = groups
    out = ROOT / "reports" / "mark_triggers.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print("\ndetalhes: %s" % out)


if __name__ == "__main__":
    main()
