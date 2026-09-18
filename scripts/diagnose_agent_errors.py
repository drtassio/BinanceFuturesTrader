"""Where does a trained specialist lose money, compared with its teacher?

Runs the deterministic policy and the teacher rule through the same training
environment on the TRAIN and VALIDATION blocks only (the holdout is never used
to correct a model), rebuilds every trade from the position path, and reports:

  * agent trades that match a teacher trade, extra entries the teacher never
    made, and teacher entries the agent missed
  * result by exit reason and duration (did it leave too early or too late?)
  * result by the market context at entry: 4h and 1h trend with or against the
    trade, volatility expansion, hour of day

    python scripts/diagnose_agent_errors.py --agent bear --run cloud/artifacts/leg_confirm_run/bear_guided_<stamp>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")

from cloud.train_agent import AGENTS, _episode_environment, load_dataset, split_chronological  # noqa: E402
from config.settings import AIConfig, TradingConfig  # noqa: E402


def _raw(vec_env):
    inner = vec_env
    while hasattr(inner, "venv"):
        inner = inner.venv
    return inner.envs[0]


def path_of(frame: pd.DataFrame, agent_name: str, act) -> pd.DataFrame:
    """Position after each bar, entry price and exit reason, for any action source."""
    from specialists.bull_specialist import BullTradingEnv
    from specialists.bear_specialist import BearTradingEnv

    config = AIConfig()
    config.ECONOMIC_REWARD_ONLY = True
    rows = []
    if act == "teacher":
        env = {"bull": BullTradingEnv, "bear": BearTradingEnv}[agent_name](
            df=frame, config=config, mode="training", specialist_name="%s_specialist" % agent_name)
        env.set_phase3_runtime_tweaks()
        env.reset()
        env.max_steps = len(frame) - 2
        start = int(getattr(env, "start_idx", 0) or 0)
        for _ in range(env.max_steps + 1):
            i = min(start + env.current_step, len(frame) - 1)
            _, _, done, trunc, info = env.step(RULE_ACTION(frame.iloc[i], env.position))
            rows.append((frame.index[i], int(np.sign(env.position)), float(env.entry_price or 0.0),
                         info.get("exit_reason"), float(env.net_worth)))
            if done or trunc:
                break
    else:
        vec = _episode_environment(act, frame, agent_name)
        raw = _raw(vec)
        obs = vec.reset()
        start = int(getattr(raw, "start_idx", 0) or 0)
        for _ in range(len(frame)):
            i = min(start + raw.current_step, len(frame) - 1)
            action, _ = act.model.predict(obs, deterministic=True)
            obs, _, done, infos = vec.step(action)
            if bool(done[0]):
                break
            rows.append((frame.index[i], int(np.sign(raw.position)), float(raw.entry_price or 0.0),
                         infos[0].get("exit_reason"), float(raw.net_worth)))
        vec.close()
    return pd.DataFrame(rows, columns=["bar", "side", "entry_price", "exit_reason", "net_worth"]).set_index("bar")


def trades_of(path: pd.DataFrame) -> pd.DataFrame:
    out, prev, start = [], 0, None
    worth = path.net_worth.to_numpy()
    for k, (bar, row) in enumerate(path.iterrows()):
        if prev == 0 and row.side != 0:
            start, w0 = bar, worth[k - 1] if k else worth[k]
        if prev != 0 and row.side != prev:
            out.append({"entry": start, "exit": bar, "side": prev, "bars": int((bar - start) / pd.Timedelta("15min")),
                        "ret": worth[k] / w0 - 1, "exit_reason": row.exit_reason or "Agent Decision"})
            if row.side != 0:
                start, w0 = bar, worth[k]
        prev = row.side
    return pd.DataFrame(out)


def context(frame: pd.DataFrame, t: pd.DataFrame, side: int) -> pd.DataFrame:
    at = frame.reindex(t.entry)
    t = t.copy()
    t["trend_4h"] = np.where(np.sign(at.get("cz_trend_4h", 0)).to_numpy() == side, "a favor", "contra")
    t["trend_1h"] = np.where(np.sign(at.get("ema_trend_1h", 0)).to_numpy() == side, "a favor", "contra")
    exp = at.get("cz_atr_expansion", pd.Series(1.0, index=at.index)).to_numpy()
    t["volatilidade"] = np.where(exp >= 1.2, "expandindo", np.where(exp <= 0.9, "contraida", "normal"))
    t["horario_utc"] = pd.cut(pd.DatetimeIndex(t.entry).hour, [-1, 7, 15, 23], labels=["00-08h", "08-16h", "16-24h"]).astype(str)
    t["dur"] = pd.cut(t.bars, [-1, 8, 32, 96, 10**6], labels=["<2h", "2-8h", "8-24h", ">24h"]).astype(str)
    return t


def table(t: pd.DataFrame, by: str) -> str:
    g = t.groupby(by).ret.agg(["count", "mean", "sum"])
    g["acerto"] = t.groupby(by).ret.apply(lambda r: (r > 0).mean())
    return "\n".join("      %-22s %4d trades | medio %+.2f%% | soma %+6.1f%% | acerto %3.0f%%"
                     % (k, r["count"], 100 * r["mean"], 100 * r["sum"], 100 * r["acerto"]) for k, r in g.iterrows())


def main() -> int:
    global RULE_ACTION
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=("bull", "bear"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", default=None, help="zip dentro de <run>/models; padrao: <agent>_specialist_sac.zip")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "featured_data_causal_leg5m.parquet")
    ap.add_argument("--rule", type=Path, default=None)
    args = ap.parse_args()

    from learning.edge_policy import load_rule, teacher_action
    from specialists.trend_specialist import ClippedSAC

    rule_path = args.rule or ROOT / "models_ai" / ("%s_leg_confirm_rule.json" % args.agent)
    rule = load_rule(json.loads(rule_path.read_text(encoding="utf-8"))["rule"])
    RULE_ACTION = lambda row, pos: teacher_action(row, pos, args.agent, rule)  # noqa: E731
    side = 1 if args.agent == "bull" else -1

    train_df, val_df, _ = split_chronological(load_dataset(args.data))
    config = AIConfig()
    config.ECONOMIC_REWARD_ONLY = True
    config.MODEL_DIR = str(args.run.resolve() / "models")
    agent = AGENTS[args.agent](config=config, trading_config=TradingConfig(),
                               input_dim=len(train_df.select_dtypes(include="number").columns))
    agent.load_model()
    model_path = args.run.resolve() / "models" / (args.model or "%s_specialist_sac.zip" % args.agent)
    agent.model = ClippedSAC.load(str(model_path), device=agent.model.device)
    print("modelo: %s | professor: %s | stops em ATR de %s" % (model_path.name, rule.as_dict(), AIConfig.ENV_STOP_ATR_TIMEFRAME))

    report = {}
    for name, frame in (("TREINO", train_df), ("VALIDACAO", val_df)):
        a = context(frame, trades_of(path_of(frame, args.agent, agent)), side)
        p = trades_of(path_of(frame, args.agent, "teacher"))
        near = lambda t0, ref: (abs(ref.entry - t0) <= pd.Timedelta("30min")).any() if len(ref) else False  # noqa: E731
        a["origem"] = ["junto com o professor" if near(t0, p) else "entrada EXTRA (professor nao entrou)" for t0 in a.entry]
        missed = [t0 for t0 in p.entry if not ((a.entry <= t0 + pd.Timedelta("30min")) & (a.exit >= t0)).any()] if len(a) else list(p.entry)
        missed_ret = p[p.entry.isin(missed)].ret
        print("\n" + "=" * 100)
        print("%s  %s a %s" % (name, frame.index.min().date(), frame.index.max().date()))
        print("  agente: %d trades, soma %+.1f%%, acerto %.0f%% | professor: %d trades, soma %+.1f%%"
              % (len(a), 100 * a.ret.sum(), 100 * (a.ret > 0).mean(), len(p), 100 * p.ret.sum()))
        print("  entradas do professor que o agente PERDEU: %d (valiam %+.1f%% somadas)" % (len(missed), 100 * missed_ret.sum()))
        for by in ("origem", "exit_reason", "dur", "trend_4h", "trend_1h", "volatilidade", "horario_utc"):
            print("   por %s:\n%s" % (by, table(a, by)))
        report[name] = {"agent_trades": a.assign(entry=a.entry.astype(str), exit=a.exit.astype(str)).to_dict("records"),
                        "teacher_trades": int(len(p)), "missed": int(len(missed)), "missed_sum": float(missed_ret.sum())}
    out = args.run / ("%s_error_diagnosis.json" % args.agent)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("\ndetalhes: %s" % out)
    return 0


RULE_ACTION = None

if __name__ == "__main__":
    sys.exit(main())
