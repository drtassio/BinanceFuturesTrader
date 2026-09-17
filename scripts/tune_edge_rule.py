"""Choose the teacher rule for each specialist on the TRAINING split only.

Runs the real trading environment — same costs, stops, sizing and funding as
the agents — with scripted actions from learning.edge_policy, over a small grid.
Selection never looks at validation or holdout. The chosen rule is then scored
once on validation, and the holdout stays untouched for the final verdict.

    python scripts/tune_edge_rule.py --agent bull
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")

MIN_TRAIN_TRADES = 40


def _env_class(agent):
    from specialists.bull_specialist import BullTradingEnv
    from specialists.bear_specialist import BearTradingEnv
    from specialists.ranger_specialist import RangerTradingEnv
    return {"bull": BullTradingEnv, "bear": BearTradingEnv, "ranger": RangerTradingEnv}[agent]


def run_rule(frame, agent, rule):
    """Drive the environment with the rule over a whole block."""
    from config.settings import AIConfig
    from learning.edge_policy import edge_action

    config = AIConfig()
    config.ECONOMIC_REWARD_ONLY = True
    env = _env_class(agent)(df=frame, config=config, mode="training",
                            specialist_name="%s_specialist" % agent)
    # Mesmas regras de execucao de cloud/train_agent.evaluate.
    env.set_phase3_runtime_tweaks()
    env.reset()
    env.max_steps = len(frame) - 2
    start = int(getattr(env, "start_idx", 0) or 0)
    summary = None
    for _ in range(env.max_steps + 1):
        index = min(start + env.current_step, len(frame) - 1)
        action = edge_action(frame.iloc[index], env.position, agent, rule)
        _, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break
    summaries = env.consume_episode_summaries()
    summary = summaries[-1] if summaries else {}
    return {
        "trades": int(summary.get("num_trades", 0) or 0),
        "net_return": float(summary.get("total_return_pct", 0.0) or 0.0),
        "profit_factor": float(summary.get("profit_factor", 0.0) or 0.0),
        "max_drawdown": float(summary.get("max_drawdown_pct", 0.0) or 0.0),
        "win_rate": float(summary.get("win_rate_pct", 0.0) or 0.0),
        "avg_duration": float(summary.get("avg_trade_duration", 0.0) or 0.0),
    }


def _load_split(which):
    from cloud.train_agent import load_dataset, split_chronological
    train, validation, holdout = split_chronological(load_dataset(ROOT / "data" / "featured_data_causal.parquet"))
    return {"train": train, "validation": validation, "holdout": holdout}[which]


def _worker(job):
    agent, rule_dict, split = job
    from learning.edge_policy import EdgeRule
    rule = EdgeRule(**rule_dict)
    return rule_dict, run_rule(_load_split(split), agent, rule)


def score(result, min_trades=MIN_TRAIN_TRADES):
    """Net return per unit of drawdown, only for rules that actually trade."""
    if result["trades"] < min_trades or result["net_return"] <= 0:
        return -np.inf
    return result["net_return"] / max(result["max_drawdown"], 0.02)


def main() -> int:
    from learning.edge_policy import EdgeRule

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("bull", "bear", "ranger"), required=True)
    parser.add_argument("--workers", type=int, default=6)
    # A selecao no treino e o padrao. O Bear e a excecao documentada: o bloco
    # de treino (ago/2023 a jun/2025) e quase todo de alta e nenhuma regra
    # short acumula trades suficientes ali. Para ele a regra e escolhida na
    # validacao, o unico trecho de baixa antes do holdout, que continua
    # intocado para o veredito final.
    parser.add_argument("--select-on", choices=("train", "validation"), default="train")
    parser.add_argument("--min-trades", type=int, default=MIN_TRAIN_TRADES)
    # Os limiares de probabilidade precisam estar na escala de cada lado. O
    # meta-modelo short tem taxa-base de 26.5%: p_short >= 0.50 fica acima do
    # percentil 99 e a grade original so encontrava 3 a 5 trades por bloco.
    parser.add_argument("--probabilities", default=None,
                        help="lista separada por virgula; padrao por agente")
    parser.add_argument("--edges", default="0.05,0.10,0.20")
    parser.add_argument("--short-probabilities", default="0.30,0.35",
                        help="so Ranger: limiares do lado short, na escala do meta-modelo short")
    # O agente emite sl_mult no maximo 3.0; um stop maior na grade seria
    # cortado no treino e a professora deixaria de ser imitavel.
    parser.add_argument("--stops", default="2.0,3.0")
    args = parser.parse_args()
    default_probabilities = {"bull": "0.50,0.55,0.60", "bear": "0.30,0.35,0.40", "ranger": "0.50,0.55"}
    probabilities = tuple(float(x) for x in (args.probabilities or default_probabilities[args.agent]).split(","))
    edges = tuple(float(x) for x in args.edges.split(","))
    stops = tuple(float(x) for x in args.stops.split(","))
    shorts = (tuple(float(x) for x in args.short_probabilities.split(","))
              if args.agent == "ranger" else (None,))

    grid = []
    for p, p_short, e_in, e_out, regime, sl in itertools.product(
            probabilities, shorts, edges, (-0.05, 0.0), (True, False), stops):
        if e_out >= e_in:
            continue
        grid.append(EdgeRule(enter_probability=p, enter_edge=e_in, exit_edge=e_out,
                             require_regime=regime, sl_mult=sl,
                             enter_probability_short=p_short).as_dict())
    print("%s: %d regras em %s, %d processos" % (args.agent, len(grid), args.select_on, args.workers))

    with Pool(args.workers) as pool:
        results = pool.map(_worker, [(args.agent, g, args.select_on) for g in grid])

    results.sort(key=lambda item: score(item[1], args.min_trades), reverse=True)
    print("\n%-58s %7s %9s %6s %7s %6s %6s" % ("regra (treino)", "trades", "retorno", "PF", "maxDD", "acerto", "dur"))
    for rule, r in results[:10]:
        label = "p>=%.2f in>=%.2f out>%.2f reg=%s sl=%.0f" % (
            rule["enter_probability"], rule["enter_edge"], rule["exit_edge"], rule["require_regime"], rule["sl_mult"])
        if rule.get("enter_probability_short") is not None:
            label += " ps>=%.2f" % rule["enter_probability_short"]
        print("%-58s %7d %8.2f%% %6.2f %6.1f%% %5.1f%% %6.1f" % (
            label, r["trades"], r["net_return"] * 100, r["profit_factor"], r["max_drawdown"] * 100, r["win_rate"], r["avg_duration"]))

    best_rule, best_train = results[0]
    if not np.isfinite(score(best_train, args.min_trades)):
        print("\nNenhuma regra lucrativa com trades suficientes em %s." % args.select_on)
        return 1
    if args.select_on == "validation":
        validation = dict(best_train)
    else:
        validation = run_rule(_load_split("validation"), args.agent, EdgeRule(**best_rule))
    print("\nescolhida: %s" % best_rule)
    print("validacao (vista uma vez): trades=%d retorno=%+.2f%% PF=%.2f maxDD=%.1f%% acerto=%.1f%%" % (
        validation["trades"], validation["net_return"] * 100, validation["profit_factor"],
        validation["max_drawdown"] * 100, validation["win_rate"]))

    out = ROOT / "models_ai" / ("%s_edge_rule.json" % args.agent)
    out.write_text(json.dumps({"agent": args.agent, "rule": best_rule, "train": best_train,
                               "validation": validation, "selected_on": args.select_on}, indent=2), encoding="utf-8")
    print("salvo em %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
