"""Separa as marcações finas (scripts/mark_legs.py --tag fino) em datasets/marcacoes_finas.

    python scripts/mark_legs.py --tag fino --swing 0.006 --leg-min 0.009 --speed 0.0005 --confirm 0.0025
    python scripts/export_marked_legs.py

Estrutura gerada:
  datasets/marcacoes_finas/
    todas_pernas.parquet          uma linha por perna (long e short)
    rotulos_por_candle.parquet    posição desejada por candle de 15m (+1, -1, 0)
    bull/pernas_long.parquet      só as pernas de alta
    bull/rotulos_long.parquet     +1 dentro das pernas de alta, 0 fora
    bull/regra.json               professor usado no treino do Bull
    bull/resumo.json
    bear/...                      o mesmo para as pernas de baixa
    graficos/                     um gráfico por semana com entradas e saídas + index.html
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "marcacoes_finas"


def summary(legs: pd.DataFrame, labels: pd.Series, side: str) -> dict:
    return {
        "lado": side,
        "pernas": int(len(legs)),
        "periodo": [str(legs.entry_time.min()), str(legs.exit_time.max())],
        "movimento_mediano_pct": round(float(legs.leg_move.median() * 100), 2),
        "capturado_liquido_mediano_pct": round(float(legs.captured_net.median() * 100), 2),
        "soma_capturado_liquido_pct": round(float(legs.captured_net.sum() * 100), 1),
        "acerto_pct": round(float((legs.captured_net > 0).mean() * 100), 1),
        "duracao_mediana_h": round(float(legs.hours.median()), 1),
        "candles_posicionado_pct": round(float((labels != 0).mean() * 100), 1),
        "aviso": "marcacoes feitas com o grafico pronto (sabem onde cada perna termina): "
                 "sao exemplos para ensinar, nao uma estrategia reproduzivel ao vivo",
    }


def main() -> int:
    legs = pd.read_parquet(ROOT / "data" / "teacher_legs_fino.parquet")
    labels = pd.read_parquet(ROOT / "data" / "teacher_leg_labels_fino.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    legs.to_parquet(OUT / "todas_pernas.parquet")
    labels.to_parquet(OUT / "rotulos_por_candle.parquet")

    for side, sign, folder in (("LONG", 1, "bull"), ("SHORT", -1, "bear")):
        d = OUT / folder
        d.mkdir(exist_ok=True)
        own = legs[legs.side == side].reset_index(drop=True)
        own.to_parquet(d / ("pernas_%s.parquet" % side.lower()))
        lab = labels.copy()
        lab["hs_target_position"] = (lab.hs_target_position == sign).astype("int8") * sign
        lab.to_parquet(d / ("rotulos_%s.parquet" % side.lower()))
        rule = ROOT / "models_ai" / ("%s_marked_legs_fino_rule.json" % folder)
        if rule.exists():
            shutil.copy2(rule, d / "regra.json")
        (d / "resumo.json").write_text(json.dumps(summary(own, lab.hs_target_position, side), indent=2,
                                                 ensure_ascii=False), encoding="utf-8")
        print("%s: %d pernas" % (folder, len(own)))

    charts = ROOT / "reports" / "leg_marks_fino"
    if charts.exists():
        shutil.copytree(charts, OUT / "graficos", dirs_exist_ok=True)
        print("graficos: %d arquivos" % len(list((OUT / "graficos").iterdir())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
