"""
research_ic.py
Quais dados que o agente recebe apontam a direção do próximo movimento, medido
como em pesquisa quantitativa (não por AUC):

  IC        correlação de Spearman entre o valor da feature no candle t e o
            retorno futuro de h candles (h = 4: 1 hora, h = 16: 4 horas).
  ESTÁVEL   IC calculado trimestre a trimestre: mesmo sinal em >= 75% dos
            trimestres e estatística t (média/erro padrão entre trimestres) >= 2.
  SPREAD    retorno futuro médio do decil mais alto menos o do mais baixo, em %,
            comparado ao custo de ida e volta de 0,14%.
  ACASO     o mesmo com o retorno futuro embaralhado em blocos de um dia: quantas
            features passariam só por sorte.

Recortes: todos os candles, dentro das pernas de alta marcadas e dentro das
pernas de baixa marcadas (scripts/mark_legs.py, versão fina). Features: tudo o
que o agente observa (15m, 1h, 4h, 5m, fluxo/tape) e os 32 latentes do
autoencoder.

    python scripts/research_ic.py
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
from research_mark_triggers import DATA, MARKS, autoencoder_latents, group_of  # noqa: E402

COST = 0.0014


def per_quarter_ic(x: pd.Series, fwd: pd.Series, quarters: pd.Series) -> np.ndarray:
    out = []
    for _, idx in quarters.groupby(quarters).groups.items():
        a, b = x.loc[idx], fwd.loc[idx]
        ok = a.notna() & b.notna()
        if ok.sum() > 300 and a[ok].nunique() > 5:
            out.append(a[ok].rank().corr(b[ok].rank()))
    return np.array(out, dtype=float)


def evaluate(X: pd.DataFrame, fwd: pd.Series, quarters: pd.Series) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        ics = per_quarter_ic(X[c], fwd, quarters)
        if len(ics) < 4:
            continue
        mean = float(np.nanmean(ics))
        t = mean / (np.nanstd(ics, ddof=1) / np.sqrt(len(ics)) + 1e-12)
        agree = float(np.mean(np.sign(ics) == np.sign(mean)))
        x, y = X[c], fwd
        ok = x.notna() & y.notna()
        try:
            dec = pd.qcut(x[ok].rank(method="first"), 10, labels=False)
            spread = float(y[ok][dec == 9].mean() - y[ok][dec == 0].mean())
        except ValueError:
            spread = np.nan
        rows.append((c, group_of(c), mean, t, agree, spread))
    r = pd.DataFrame(rows, columns=["feature", "grupo", "ic", "t", "trimestres_mesmo_sinal", "spread"])
    r["estavel"] = (r.trimestres_mesmo_sinal >= 0.75) & (r.t.abs() >= 2)
    return r.sort_values("t", key=np.abs, ascending=False)


def block_shuffle(fwd: pd.Series, block: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = fwd.to_numpy()
    blocks = [values[i:i + block] for i in range(0, len(values), block)]
    rng.shuffle(blocks)
    return pd.Series(np.concatenate(blocks)[:len(values)], index=fwd.index)


def main():
    df = pd.read_parquet(DATA).sort_index()
    marks = pd.read_parquet(MARKS)
    marks.index = pd.to_datetime(marks.index)
    df = df.loc[marks.index.min() - pd.Timedelta("2D"):]
    df = df.join(autoencoder_latents(df)).loc[marks.index.min():]
    feats = [c for c in df.select_dtypes("number").columns
             if group_of(c) != "outros" and not c.startswith(("ml_", "tp_")) and c != "regime"]
    X = df[feats]
    close = df["close"].astype(float)
    target = marks["hs_target_position"].reindex(df.index).fillna(0)
    quarters = pd.Series(df.index.to_period("Q").astype(str), index=df.index)

    report = {}
    for h, name in ((4, "1 hora"), (16, "4 horas")):
        fwd = close.shift(-h) / close - 1
        print("\n" + "=" * 110)
        print("RETORNO FUTURO DE %s" % name)
        for cut, mask in (("todos os candles", slice(None)),
                          ("dentro das pernas de ALTA marcadas", target > 0),
                          ("dentro das pernas de BAIXA marcadas", target < 0)):
            Xc, fc, qc = X.loc[mask], fwd.loc[mask], quarters.loc[mask]
            r = evaluate(Xc, fc, qc)
            chance = []
            for seed in range(3):
                rs = evaluate(Xc, block_shuffle(fc, 96, seed), qc)
                chance.append(int(rs.estavel.sum()))
            stable = r[r.estavel]
            paying = stable[stable.spread.abs() > COST]
            print("\n  %s (%d candles): %d de %d features estáveis | por acaso: %s | estáveis com spread > custo: %d"
                  % (cut, len(Xc), len(stable), len(r), chance, len(paying)))
            print("  por grupo (estáveis): %s" % ", ".join(
                "%s %d" % (g, n) for g, n in stable.grupo.value_counts().items()))
            for _, row in r.head(10).iterrows():
                print("     %-32s %-12s IC %+.3f  t %+5.1f  mesmo sinal %3.0f%%  spread %+.2f%% %s"
                      % (row.feature, row.grupo, row.ic, row.t, 100 * row.trimestres_mesmo_sinal,
                         100 * row.spread, "ESTÁVEL" if row.estavel else ""))
            report["%s | %s" % (name, cut)] = {"stable": int(len(stable)), "chance": chance,
                                               "stable_paying_cost": int(len(paying)),
                                               "top": r.head(25).to_dict("records")}
    out = ROOT / "reports" / "research_ic.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print("\ndetalhes: %s" % out)


if __name__ == "__main__":
    main()
