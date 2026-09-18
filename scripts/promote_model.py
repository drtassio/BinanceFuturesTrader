"""Promote a trained specialist from a run directory into models_ai/.

The bot loads each specialist from models_ai/<agent>_specialist_sac.zip and the
scaler from models_ai/<agent>_specialist_scaler.joblib. The scaler's
feature_names_in_ is the authoritative feature contract: production builds the
observation from exactly those columns. A policy promoted without its own
scaler therefore receives an observation assembled for some other model, and
ai_controller pads or truncates the vector to fit, so nothing raises.

This script refuses that. It checks, before copying anything, that the scaler's
feature count reproduces the policy's observation size, then copies both files
together and records their hashes. Promoting a policy changes its hash, which
invalidates any existing OOS approval until scripts/approve_policies_oos.py is
run again.

    python scripts/promote_model.py --agent bull
    python scripts/promote_model.py --agent bull --run cloud/artifacts/bull_20260916T170529
    python scripts/promote_model.py --agent bull --archive ~/Downloads/bull_artifacts_*.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_STACK = 4
# agent_state(3) + time(4) + prior(2) + physics(2), see TrendFollowingEnv.
FIXED_TAIL = 11


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_run(agent: str) -> Path:
    # Inclui execucoes guiadas (<agente>_guided_*); a mais recente vence.
    runs = sorted((ROOT / "cloud" / "artifacts").glob("%s_*" % agent), key=lambda p: p.stat().st_mtime)
    runs = [r for r in runs if (r / "models" / ("%s_specialist_sac.zip" % agent)).exists()]
    if not runs:
        raise SystemExit("nenhuma execucao de %s com modelo salvo em cloud/artifacts" % agent)
    return runs[-1]


def locate(run: Path, name: str, fallback: Path | None = None) -> Path:
    for candidate in (run / "models" / name, run / name):
        if candidate.exists():
            return candidate
    if fallback is not None and fallback.exists():
        return fallback
    raise SystemExit("arquivo ausente na execucao: %s" % name)


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    """Archives are untrusted: reject traversal, links and special devices."""
    destination = destination.resolve()
    members = tar.getmembers()
    for member in members:
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
            raise SystemExit('unsafe archive member: %s' % member.name)
    tar.extractall(destination, members=members)


def archive_run(directory: Path, agent: str) -> Path:
    """Support root packages and nested Kaggle exports without guessing a run."""
    name = '%s_specialist_sac.zip' % agent
    candidates = []
    for model in directory.rglob(name):
        run = model.parent.parent if model.parent.name == 'models' else model.parent
        if ((run / 'feature_contract.json').exists()
                and any((run / suffix / ('%s_specialist_scaler.joblib' % agent)).exists()
                        for suffix in ('models', ''))):
            candidates.append(run)
    if len(candidates) != 1:
        raise SystemExit('archive needs exactly one complete %s run; found %d' % (agent, len(candidates)))
    return candidates[0]


def observation_size(model_path: Path) -> int:
    """Read the policy's observation size without building its extractor."""
    import zipfile

    with zipfile.ZipFile(model_path) as archive:
        data = json.loads(archive.read("data").decode("utf-8"))
    space = data.get("observation_space", {})
    shape = space.get("_shape") or space.get("shape")
    if shape is None:
        raise SystemExit("nao foi possivel ler observation_space de %s" % model_path)
    return int(shape[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("bull", "bear", "ranger"), required=True)
    parser.add_argument("--run", type=Path, help="pasta da execucao (padrao: a mais recente)")
    parser.add_argument("--archive", type=Path, help="pacote .tar.gz baixado do Colab/Kaggle")
    parser.add_argument("--dest", type=Path, default=ROOT / "models_ai")
    args = parser.parse_args()

    agent = args.agent
    model_name = "%s_specialist_sac.zip" % agent
    scaler_name = "%s_specialist_scaler.joblib" % agent

    workdir = None
    if args.archive:
        workdir = Path(tempfile.mkdtemp(prefix="promote_"))
        with tarfile.open(args.archive) as tar:
            safe_extract(tar, workdir)
        run = archive_run(workdir, agent)
    else:
        run = args.run.resolve() if args.run else latest_run(agent)
    print("origem: %s" % run)

    model_path = locate(run, model_name)
    # Execucoes anteriores a correcao gravavam o scaler em models_ai/.
    scaler_path = locate(run, scaler_name, fallback=None if args.archive else ROOT / "models_ai" / scaler_name)

    import joblib

    scaler = joblib.load(scaler_path)
    raw_names = getattr(scaler, "feature_names_in_", None)
    names = [] if raw_names is None else [str(n) for n in raw_names]
    if not names:
        raise SystemExit("o scaler nao tem feature_names_in_: sem contrato de features")

    obs = observation_size(model_path)
    print("politica: observacao %d (%d frames de %d)" % (obs, N_STACK, obs // N_STACK))
    print("scaler  : %d features de mercado" % len(names))

    # Comparar so o tamanho NAO basta. Medido neste repositorio: um modelo
    # treinado com 116 colunas e um scaler de outro treino com 119 geram os
    # mesmos 137 valores por frame, porque as colunas tp_regime_* sairam dos
    # "extras" e entraram no dataset. O tamanho bate e o conteudo nao. O que
    # prova que scaler e politica sao do mesmo treino e o contrato de features
    # gravado pelo treinador, comparado nome a nome e na mesma ordem.
    contract_path = None
    for candidate in (run / "feature_contract.json", run / "models" / "feature_contract.json"):
        if candidate.exists():
            contract_path = candidate
            break
    if contract_path is None:
        raise SystemExit("sem feature_contract.json na execucao: impossivel provar que scaler e politica combinam")
    run_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    trained = [str(c) for c in run_contract.get("feature_columns", [])]
    if trained != names:
        missing = [c for c in trained if c not in names]
        unexpected = [c for c in names if c not in trained]
        detail = "faltando no scaler: %s | sobrando no scaler: %s" % (missing[:6], unexpected[:6])
        if not missing and not unexpected:
            detail = "mesmas colunas em ordem diferente"
        raise SystemExit(
            "INCOMPATIVEL: o scaler (%d colunas) nao e o do treino desta politica (%d colunas). %s"
            % (len(names), len(trained), detail)
        )

    # Confirma tambem pelo proprio ambiente, quando o dataset esta disponivel:
    # e ele que decide o tamanho real da observacao em producao.
    # The environment's observation depends on the stop scale and on which
    # columns the training frame had, both recorded by the trainer.
    dataset = Path(run_contract.get("dataset") or ROOT / "data" / "featured_data_causal.parquet")
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    if dataset.exists():
        import os
        import pandas as pd

        os.environ.setdefault("TREND_SKIP_OPTUNA", "1")
        from config.settings import AIConfig
        from specialists.bull_specialist import BullTradingEnv
        from specialists.bear_specialist import BearTradingEnv
        from specialists.ranger_specialist import RangerTradingEnv

        env_class = {"bull": BullTradingEnv, "bear": BearTradingEnv, "ranger": RangerTradingEnv}[agent]
        AIConfig.ENV_STOP_ATR_TIMEFRAME = run_contract.get("stop_atr_timeframe", "15m")
        from trading.agent_mirror import apply_leverage_bounds
        apply_leverage_bounds(run_contract)
        sample = pd.read_parquet(dataset).sort_index().iloc[-600:].ffill().fillna(0.0)
        if run_contract.get("training_frame_columns"):
            sample = sample[run_contract["training_frame_columns"]]
        env = env_class(df=sample, config=AIConfig(), mode="training",
                        feature_columns=names, specialist_name="%s_specialist" % agent)
        expected = int(env.observation_space.shape[0]) * N_STACK
        print("ambiente: observacao esperada %d" % expected)
        if expected != obs:
            raise SystemExit(
                "INCOMPATIVEL: o ambiente atual monta %d valores com este contrato, a politica espera %d. "
                "O codigo de features mudou desde o treino." % (expected, obs)
            )

    args.dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup = args.dest / "previous" / stamp
    for name in (model_name, scaler_name):
        current = args.dest / name
        if current.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current, backup / name)
    shutil.copy2(model_path, args.dest / model_name)
    shutil.copy2(scaler_path, args.dest / scaler_name)

    contract = {
        "agent": agent,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.archive or run),
        "observation_size": obs,
        "feature_columns": names,
        "model_sha256": sha256(args.dest / model_name),
        "scaler_sha256": sha256(args.dest / scaler_name),
    }
    # The live mirror needs the stop scale and frame the model was trained on.
    for key in ("stop_atr_timeframe", "leverage_bounds", "training_frame_columns", "dataset"):
        if key in run_contract:
            contract[key] = run_contract[key]
    (args.dest / ("%s_feature_contract.json" % agent)).write_text(json.dumps(contract, indent=2), encoding="utf-8")

    print("\npromovido para %s" % args.dest)
    if backup.exists():
        print("versao anterior guardada em %s" % backup)
    print("a aprovacao OOS anterior deixou de valer; rode scripts/approve_policies_oos.py")
    if workdir is not None:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
