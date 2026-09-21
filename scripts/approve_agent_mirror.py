"""Approval for specialists that operate through trading/agent_mirror.py.

A specialist may trade only when all of this holds, for the exact files in
models_ai/ (hash-bound; replacing any of them voids the approval):

1. The promoted model is the model of the training run whose report is judged
   (same SHA-256).
2. The mirror replay reproduces that model's backtest (agent_mirror_parity.json).
3. Performance, deterministic policy, blocks the model selection never used for
   training (validation) or never saw at all (holdout):

   * train:      net return > 0
   * validation: net return > 0, profit factor >= 1.2, drawdown <= 15%, >= 20 trades
   * holdout:    profit factor >= 1.1, drawdown <= 15%, >= 20 trades, beats buy and hold,
                 and net return > 0 — except that a directional specialist whose
                 holdout moved against its side by more than 20% must instead keep
                 its loss within 5%. A long-only specialist is not expected to earn
                 in a crash; it is expected to stay out of it.

These criteria were written down on 2026-09-17, before the trend Bull's
results existed. They are printed with every verdict.

    python scripts/approve_agent_mirror.py --agent bull --run cloud/artifacts/trend/bull_guided_<stamp>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading import agent_mirror as mirror  # noqa: E402

SIDE = {"bull": 1, "bear": -1}


def metrics(m: dict) -> dict:
    return {"net": float(m.get("total_return_pct", 0.0) or 0.0), "pf": float(m.get("profit_factor", 0.0) or 0.0),
            "dd": float(m.get("max_drawdown_pct", 1.0) or 0.0), "trades": int(m.get("num_trades", 0) or 0),
            "deterministic": bool(m.get("deterministic", True))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=sorted(SIDE))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models_ai")
    parser.add_argument("--min-trades", type=int, default=20,
                        help="trades minimos por bloco; so outro valor se registrado antes do treino "
                             "(reports/leg_confirm_preregistration.json: 10)")
    parser.add_argument("--specialist-benchmark", action="store_true",
                        help="nao exige bater o buy and hold: o especialista e julgado pelo proprio "
                             "lucro, PF e queda, independente da direcao do mercado. Decisao do usuario "
                             "em 2026-09-21, tomada DEPOIS de ver o holdout do Bull leg_confirm "
                             "(+16.2%% contra +18.8%% do buy and hold); registrada no veredito.")
    args = parser.parse_args()

    agent = args.agent
    report = json.loads((args.run / ("%s_guided_report.json" % agent)).read_text(encoding="utf-8"))
    run_model = args.run / "models" / ("%s_specialist_sac.zip" % agent)
    promoted = args.model_dir / ("%s_specialist_sac.zip" % agent)
    same_model = hashlib.sha256(run_model.read_bytes()).digest() == hashlib.sha256(promoted.read_bytes()).digest()

    parity_ok = False
    try:
        parity = json.loads((args.model_dir / "agent_mirror_parity.json").read_text(encoding="utf-8"))
        current = mirror.artifact_hashes(args.model_dir, [agent])
        parity_ok = bool(parity.get("all_passed")) and all(
            parity["artifact_hashes"].get(name) == digest for name, digest in current.items())
    except (OSError, ValueError, KeyError):
        pass

    train, val, hold = (metrics(report.get(k) or {}) for k in ("train_metrics", "selected_validation", "holdout_metrics"))
    benchmark = float(report["verdict"]["buy_and_hold_return"])
    adverse = SIDE[agent] * benchmark < -0.20
    checks = {
        "modelo promovido = modelo do relatorio": same_model,
        "paridade do espelho com o backtest": parity_ok,
        "treino: retorno > 0": train["net"] > 0,
        "validacao: retorno > 0": val["net"] > 0,
        "validacao: PF >= 1.2": val["pf"] >= 1.2,
        "validacao: DD <= 15%": val["dd"] <= 0.15,
        "validacao: >= %d trades" % args.min_trades: val["trades"] >= args.min_trades,
        "holdout: politica deterministica": hold["deterministic"],
        "holdout: PF >= 1.1": hold["pf"] >= 1.1,
        "holdout: DD <= 15%": hold["dd"] <= 0.15,
        "holdout: >= %d trades" % args.min_trades: hold["trades"] >= args.min_trades,
        ("holdout: perda <= 5%% (mercado %+.0f%% contra o lado)" % (100 * benchmark) if adverse
         else "holdout: retorno > 0"): hold["net"] >= -0.05 if adverse else hold["net"] > 0,
    }
    if not args.specialist_benchmark:
        checks["holdout: bate buy and hold"] = hold["net"] > benchmark
    approved = all(checks.values())
    out = {"approved": approved, "agents": [agent], "run": str(args.run), "generated_at": datetime.now(timezone.utc).isoformat(),
           "benchmark_rule": ("specialist: own profit, PF and drawdown; buy and hold not required "
                              "(user decision 2026-09-21, after the Bull holdout was seen)")
                             if args.specialist_benchmark else "must beat buy and hold",
           "min_trades": args.min_trades,
           "criteria_doc": __doc__, "checks": checks, "train": train, "validation": val, "holdout": hold,
           "holdout_buy_and_hold": benchmark, "periods": report.get("periods"),
           "artifact_hashes": mirror.artifact_hashes(args.model_dir, [agent])}
    mirror.approval_path(args.model_dir).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, passed in checks.items():
        print("  [%s] %s" % ("OK " if passed else "NAO", name))
    print("treino %+.1f%% | validacao %+.1f%% PF %.2f DD %.1f%% %d trades | holdout %+.1f%% PF %.2f DD %.1f%% %d trades (B&H %+.1f%%)" % (
        100 * train["net"], 100 * val["net"], val["pf"], 100 * val["dd"], val["trades"],
        100 * hold["net"], hold["pf"], 100 * hold["dd"], hold["trades"], 100 * benchmark))
    print("APROVADO PARA OPERAR (espelho): %s" % approved)
    return 0 if approved else 1


if __name__ == "__main__":
    sys.exit(main())
