"""Train a specialist from a profitable teacher, then let RL improve on it.

Why not SAC from scratch
------------------------
Measured on this dataset, a 300k-step SAC Bull finished validation at -33.6%
and holdout at -46.2% with 441 trades: its deterministic vote hovered around
zero (mean -0.018, std 0.16) and crossed the entry threshold on noise. The
causal edge rule it could have followed made +5.45% on the same validation
block with profit factor 4.7. The signal exists; model-free RL at this sample
size fails to find it.

The standard remedy is learning from demonstrations:

1. Teacher. learning.edge_policy with thresholds chosen on the TRAIN split only
   (scripts/tune_edge_rule.py).
2. Behaviour cloning. The SAC actor is trained to output the teacher's actions
   from its own observation, which already contains the ml_*, tp_* and
   position features the rule reads. The replay buffer is pre-filled with the
   teacher's transitions.
3. Critic warm-up with the actor frozen, so the first policy gradients come from
   a critic that has seen profitable behaviour rather than from random Q-values.
4. Fine-tuning with a TD3+BC-style penalty that decays over time: the policy may
   depart from the teacher only where the critic says it pays.
5. Selection on validation only. The cloned policy is the baseline; a fine-tuned
   checkpoint replaces it only if it scores better there.
6. Holdout, read once, for the verdict.

    python cloud/train_guided.py --agent bull
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")

from cloud.train_agent import (  # noqa: E402
    AGENTS, _episode_environment, buy_and_hold_return, evaluate, judge,
    load_dataset, split_chronological,
)
from config.settings import AIConfig, TradingConfig  # noqa: E402
from learning.edge_policy import load_rule, teacher_action, teacher_inputs  # noqa: E402


def _raw_env(vec_env):
    """TrendFollowingEnv underneath VecNormalize(VecFrameStack(DummyVecEnv))."""
    inner = vec_env
    while hasattr(inner, "venv"):
        inner = inner.venv
    return inner.envs[0]


def score(metrics: dict, min_trades: int = 10) -> float:
    """Return per unit of drawdown, for policies that trade and make money."""
    trades = int(metrics.get("num_trades", 0) or 0)
    net = float(metrics.get("total_return_pct", 0.0) or 0.0)
    pf = float(metrics.get("profit_factor", 0.0) or 0.0)
    dd = float(metrics.get("max_drawdown_pct", 1.0) or 0.0)
    # Do not prefer a high ratio from eight trades over a validated policy
    # that actually meets the bot's minimum activity and drawdown criteria.
    if not all(np.isfinite(value) for value in (net, pf, dd)):
        return -np.inf
    if trades < min_trades or net <= 0.0 or pf < 1.2 or not 0.0 <= dd <= float(AIConfig.OOS_MAX_DRAWDOWN):
        return -np.inf
    return net / max(dd, 0.02)


def summarize(metrics: dict) -> str:
    return "trades=%d retorno=%+.2f%% PF=%.2f maxDD=%.1f%% acerto=%.1f%%" % (
        int(metrics.get("num_trades", 0) or 0),
        float(metrics.get("total_return_pct", 0.0) or 0.0) * 100,
        float(metrics.get("profit_factor", 0.0) or 0.0),
        float(metrics.get("max_drawdown_pct", 0.0) or 0.0) * 100,
        float(metrics.get("win_rate_pct", 0.0) or 0.0))


def collect_teacher(model, agent_name, frame, rule, vec_env, fill_buffer=True):
    """Roll the teacher through a block; optionally fill the replay buffer."""
    raw = _raw_env(vec_env)
    observations, actions = [], []
    obs = vec_env.reset()
    start = int(getattr(raw, "start_idx", 0) or 0)
    for _ in range(len(frame)):
        row = frame.iloc[min(start + raw.current_step, len(frame) - 1)]
        action = teacher_action(row, raw.position, agent_name, rule)[None, :]
        scaled = model.policy.scale_action(action)
        next_obs, reward, done, infos = vec_env.step(action)
        buffer_next = next_obs.copy()
        if bool(done[0]) and "terminal_observation" in infos[0]:
            buffer_next[0] = infos[0]["terminal_observation"]
        if fill_buffer:
            model.replay_buffer.add(obs, buffer_next, scaled, reward, done, infos)
        observations.append(obs[0].copy())
        actions.append(scaled[0].copy())
        obs = next_obs
        if bool(done[0]):
            break
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def _on_position(votes, sides):
    """Bars where a vote holds or opens a position on one of the agent's sides."""
    import torch

    if len(sides) > 1:
        return votes.abs() > 0.05
    return torch.sign(votes) == float(sides[0])


def rule_input_mask(agent, obs_dim, rule, agent_name, n_stack=4, all_features=True):
    """Observation positions the teacher rule actually reads, in the newest frame.

    When all_features=True (default), preserves the complete observation vector so the
    actor learns cross-correlations with the 32 autoencoder latents, aggressor delta,
    order flow, 1h/4h higher-timeframe features, and position states.
    """
    import numpy as _np

    if all_features:
        return _np.ones(obs_dim, dtype=bool)
    frame_dim = obs_dim // n_stack
    columns = list(agent.feature_columns)
    inputs = teacher_inputs(rule, agent_name)
    if inputs.get("all_columns"):
        return _np.ones(obs_dim, dtype=bool)
    keep = _np.zeros(obs_dim, dtype=bool)
    offset = (n_stack - 1) * frame_dim
    for name in inputs["columns"]:
        if name not in columns:
            raise KeyError("observacao sem a entrada da professora: %s" % name)
        keep[offset + columns.index(name)] = True
    extras = frame_dim - len(columns) - 11
    tail = offset + len(columns) + extras
    # Cauda fixa: agent_state(3) = [sign(position), pnl, duracao], time(4),
    # prior(2) = [tp_prior_dir, tp_prior_conf], physics(2). tp_prior_dir e
    # removido das features de mercado pelo ambiente, mas chega aqui.
    keep[tail + 0] = True  # sign(position)
    if inputs["prior_dir"]:
        keep[tail + 7] = True  # tp_prior_dir
    return keep


def _balanced_agreement(model, obs_t, act_t, sides):
    import torch

    with torch.no_grad():
        teacher_on = _on_position(act_t[:, 0], sides)
        clone_on = _on_position(model.actor(obs_t, deterministic=True)[:, 0], sides)
    recall = float(((clone_on & teacher_on).float().sum() / teacher_on.float().sum().clamp(min=1)).item())
    specificity = float(((~clone_on & ~teacher_on).float().sum() / (~teacher_on).float().sum().clamp(min=1)).item())
    return 0.5 * (recall + specificity), recall, 1.0 - specificity


def collect_dagger(model, agent_name, frame, rule, vec_env):
    """The clone acts; the teacher labels every state the clone reaches.

    Plain cloning only trains on states the teacher visits. A small mistake
    takes the clone somewhere the teacher never went, where it errs again:
    measured on validation, the clone matched the teacher's trade count (50 vs
    51) and profit factor (1.94 vs 1.93) yet made +4.52% against +23.96%,
    because it left trends early and the rule's hysteresis then kept it out.
    DAgger (Ross, Gordon and Bagnell, 2011) labels the clone's own trajectory
    with the teacher's decision for the position the clone actually holds, so
    the next round of cloning covers exactly those states.
    """
    raw = _raw_env(vec_env)
    observations, labels = [], []
    obs = vec_env.reset()
    start = int(getattr(raw, "start_idx", 0) or 0)
    for _ in range(len(frame)):
        row = frame.iloc[min(start + raw.current_step, len(frame) - 1)]
        teacher = teacher_action(row, raw.position, agent_name, rule)[None, :]
        clone, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, infos = vec_env.step(clone)
        buffer_next = next_obs.copy()
        if bool(done[0]) and "terminal_observation" in infos[0]:
            buffer_next[0] = infos[0]["terminal_observation"]
        # On-policy transitions also teach the critic what the clone's own
        # mistakes cost, which the teacher's trajectory cannot show.
        model.replay_buffer.add(obs, buffer_next, model.policy.scale_action(clone), reward, done, infos)
        observations.append(obs[0].copy())
        labels.append(model.policy.scale_action(teacher)[0].copy())
        obs = next_obs
        if bool(done[0]):
            break
    return np.asarray(observations, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def behaviour_clone(model, observations, actions, epochs, sides, keep_mask,
                    val_observations=None, val_actions=None, batch_size=512, patience=6):
    """Imitate the teacher without learning its incidental correlates.

    Cloning a threshold rule over ~480 observation dimensions memorised the
    training block (recall 100%, false positives 0.01%) and failed out of
    sample: on the same validation slice the teacher made PF 1.73 and its clone
    PF 0.36. The network had latched onto dimensions the rule never reads, the
    causal confusion of de Haan et al. (2019). Every dimension outside the
    rule's inputs is shuffled across the minibatch at each step, which keeps its
    distribution but destroys any link to the label, so only the true inputs
    carry information. Epochs stop on imitation fidelity measured on the
    validation block, which uses the teacher's actions and no returns.
    """
    import copy
    import torch

    device = model.device
    obs_t = torch.as_tensor(observations, device=device)
    act_t = torch.as_tensor(actions, device=device)
    nuisance = torch.as_tensor(~keep_mask, device=device)
    from learning.policy_imitation import imitation_weights, imitation_fidelity
    weights = imitation_weights(obs_t, act_t, sides)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4, weight_decay=1e-5)
    have_val = val_observations is not None and len(val_observations) > 0
    if have_val:
        val_obs_t = torch.as_tensor(val_observations, device=device)
        val_act_t = torch.as_tensor(val_actions, device=device)
    best = (-1.0, None, -1)
    n = obs_t.shape[0]
    for epoch in range(epochs):
        # Balance semantic decisions, including rare position exits. Applying
        # entry-only weights conflated flat waiting with closing a held trade.
        order = torch.multinomial(weights, n, replacement=True)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch = obs_t[idx].clone()
            shuffled = batch[torch.randperm(batch.shape[0], device=device)]
            batch[:, nuisance] = shuffled[:, nuisance]
            predicted = model.actor(batch, deterministic=True)
            per_sample = ((predicted - act_t[idx]) ** 2).mean(dim=1)
            loss = per_sample.mean()  # sampling already balances the classes
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(idx)
        train_bal, train_recall, train_false = _balanced_agreement(model, obs_t, act_t, sides)
        line = "  clonagem epoca %d: perda=%.4f treino[recall=%.1f%% falso+=%.2f%%]" % (
            epoch, total / n, 100 * train_recall, 100 * train_false)
        if have_val:
            val_bal, val_recall, val_false = _balanced_agreement(model, val_obs_t, val_act_t, sides)
            line += " validacao[recall=%.1f%% falso+=%.2f%%]" % (100 * val_recall, 100 * val_false)
            with torch.no_grad():
                fidelity = imitation_fidelity(val_obs_t, val_act_t[:, 0],
                    model.actor(val_obs_t, deterministic=True)[:, 0], sides)
            val_bal = fidelity['macro_recall']
            exit_text = ('%.1f%%' % (100 * fidelity['exit_recall'])
                         if fidelity['exit_recall'] is not None else 'sem exemplos')
            line += ' decisoes[macro=%.1f%% saidas=%s n=%d]' % (
                100 * val_bal, exit_text, fidelity['exit_count'])
            if val_bal > best[0]:
                best = (val_bal, copy.deepcopy(model.actor.state_dict()), epoch)
        print(line)
        if have_val and epoch - best[2] >= patience:
            break
    if best[1] is not None:
        model.actor.load_state_dict(best[1])
        print("  melhor fidelidade na validacao na epoca %d (acuracia balanceada %.1f%%)" % (best[2], 100 * best[0]))
    return obs_t, act_t, weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "featured_data_causal.parquet")
    parser.add_argument("--rule", type=Path, help="padrao: models_ai/<agente>_edge_rule.json")
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--dagger-iters", type=int, default=3)
    parser.add_argument("--dagger-epochs", type=int, default=12)
    parser.add_argument("--critic-warmup", type=int, default=5000)
    parser.add_argument("--finetune-steps", type=int, default=60000)
    parser.add_argument("--eval-every", type=int, default=20000)
    parser.add_argument("--bc-weight-start", type=float, default=2.0)
    parser.add_argument("--bc-weight-end", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=ROOT / "cloud" / "artifacts")
    parser.add_argument("--max-bars", type=int, default=0, help="so para teste de fumaca: encurta cada bloco")
    parser.add_argument("--bootstrap-from", type=Path, help="execucao anterior cujo modelo e scaler sao reaproveitados")
    parser.add_argument("--min-trades", type=int, default=10, help="minimo de trades na validacao para considerar o checkpoint")
    parser.add_argument("--isolate-teacher-inputs", action="store_true", default=False, help="embaralha dimensoes fora das entradas da professora")
    parser.add_argument('--cpu-threads', type=int, default=1, help='limite de threads PyTorch no processador')
    parser.add_argument('--seed', type=int, default=42, help='semente fixa, sem busca de hiperparametros')
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')

    import torch
    import random
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    rule_path = args.rule or ROOT / "models_ai" / ("%s_edge_rule.json" % args.agent)
    rule = load_rule(json.loads(rule_path.read_text(encoding="utf-8"))["rule"])
    print("stops do ambiente em ATR de %s" % AIConfig.ENV_STOP_ATR_TIMEFRAME)
    print("professora: %s" % rule.as_dict())

    df = load_dataset(args.data)
    train_df, val_df, holdout_df = split_chronological(df)
    if args.max_bars:
        train_df, val_df, holdout_df = (train_df.tail(args.max_bars), val_df.tail(args.max_bars // 3),
                                        holdout_df.tail(args.max_bars // 3))

    config = AIConfig()
    config.ECONOMIC_REWARD_ONLY = True
    run_dir = args.output.resolve() / ("%s_guided_%s" % (args.agent, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")))
    config.MODEL_DIR = str(run_dir / "models")
    config.CHECKPOINT_DIR = str(run_dir / "checkpoints")
    Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    input_dim = len(train_df.select_dtypes(include="number").columns)
    if args.bootstrap_from:
        # Reaproveita modelo e scaler de uma execucao anterior (mesmo contrato),
        # pulando o bootstrap. Os artefatos sao copiados para esta execucao para
        # que politica, scaler e contrato continuem juntos.
        import shutil
        source = args.bootstrap_from.resolve() / "models"
        for name in ("%s_specialist_sac.zip" % args.agent, "%s_specialist_scaler.joblib" % args.agent,
                     "training_metadata.json"):
            if (source / name).exists():
                shutil.copy2(source / name, Path(config.MODEL_DIR) / name)
        agent = AGENTS[args.agent](config=config, trading_config=TradingConfig(), input_dim=input_dim)
        agent.load_model()
        if agent.model is None:
            raise RuntimeError("nao foi possivel carregar o modelo de %s" % source)
    else:
        agent = AGENTS[args.agent](config=config, trading_config=TradingConfig(), input_dim=input_dim)
        # A tiny run builds the ClippedSAC, the scaler and the observation contract
        # through exactly the production code path; its weights are then replaced.
        result = agent.train_model(train_df, total_timesteps=1000)
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError("bootstrap falhou: %s" % result)
    model = agent.model
    model.set_logger(configure(folder=None, format_strings=[]))

    # A professora so pode usar acoes que o agente consegue executar. A grade
    # chegou a escolher sl_mult=5 com o espaco de acao limitado a 3: o agente
    # cortava o stop e a comparacao entre os dois deixava de ser justa.
    from dataclasses import replace as _replace
    low, high = model.action_space.low, model.action_space.high
    feasible = _replace(rule, sl_mult=float(np.clip(rule.sl_mult, low[1], high[1])),
                        leverage=float(np.clip(rule.leverage, low[2], high[2])))
    if feasible != rule:
        print("professora ajustada ao espaco de acao: sl_mult %.1f->%.1f alavancagem %.1f->%.1f" % (
            rule.sl_mult, feasible.sl_mult, rule.leverage, feasible.leverage))
        rule = feasible

    # 1) Teacher on validation, for reference.
    teacher_val = {}
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from tune_edge_rule import run_rule
        teacher_val = run_rule(val_df, args.agent, rule)
        print("professora na validacao: trades=%d retorno=%+.2f%% PF=%.2f" % (
            teacher_val["trades"], teacher_val["net_return"] * 100, teacher_val["profit_factor"]))
    except Exception as exc:
        print("aviso: nao foi possivel medir a professora: %s" % exc)

    # 2) Demonstrations.
    train_env = _episode_environment(agent, train_df, args.agent)
    # O bootstrap cria o modelo com os ambientes paralelos do treino padrao, e
    # o buffer de replay guarda lotes desse tamanho. Demonstracoes e ajuste
    # fino percorrem o bloco de treino num unico episodio cronologico, entao o
    # buffer e recriado para um ambiente.
    from stable_baselines3.common.buffers import ReplayBuffer
    model.replay_buffer = ReplayBuffer(
        model.buffer_size, model.observation_space, model.action_space,
        device=model.device, n_envs=train_env.num_envs,
    )
    model.n_envs = train_env.num_envs
    print("coletando demonstracoes no treino...")
    observations, actions = collect_teacher(model, args.agent, train_df, rule, train_env)
    print("  %d transicoes" % len(actions))

    # 3) Behaviour cloning.
    print("clonagem de comportamento...")
    from learning.edge_policy import SIDES
    # Trajetoria da professora na validacao: so as acoes dela, para medir a
    # fidelidade da imitacao. Nenhum retorno e usado aqui.
    val_env = _episode_environment(agent, val_df, args.agent)
    val_observations, val_actions = collect_teacher(model, args.agent, val_df, rule, val_env, fill_buffer=False)
    val_env.close()
    keep_mask = rule_input_mask(agent, observations.shape[1], rule, args.agent, all_features=not args.isolate_teacher_inputs)
    obs_t, act_t, weights = behaviour_clone(
        model, observations, actions, args.bc_epochs, SIDES[args.agent], keep_mask,
        val_observations=val_observations, val_actions=val_actions)
    cloned = evaluate(agent, val_df, args.agent, deterministic=True)
    print("politica clonada na validacao: %s" % summarize(cloned))
    best = {"label": "clonada", "score": score(cloned, min_trades=args.min_trades), "metrics": cloned}
    best_path = run_dir / "models" / "best_validation.zip"
    model.save(best_path)

    for iteration in range(1, args.dagger_iters + 1):
        print("DAgger %d: clone em malha fechada no treino, rotulado pela professora..." % iteration)
        extra_obs, extra_act = collect_dagger(model, args.agent, train_df, rule, train_env)
        observations = np.concatenate([observations, extra_obs])
        actions = np.concatenate([actions, extra_act])
        obs_t, act_t, weights = behaviour_clone(
            model, observations, actions, args.dagger_epochs, SIDES[args.agent], keep_mask,
            val_observations=val_observations, val_actions=val_actions)
        metrics = evaluate(agent, val_df, args.agent, deterministic=True)
        model.save(run_dir / 'models' / ('dagger_%d_validation.zip' % iteration))
        print("DAgger %d na validacao: %s" % (iteration, summarize(metrics)))
        current = score(metrics, min_trades=args.min_trades)
        if current > best["score"]:
            best.update(label="DAgger %d" % iteration, score=current, metrics=metrics)
            model.save(best_path)
            print("    -> novo melhor na validacao")
    if best["label"] != "clonada":
        # O ajuste fino por RL parte da melhor politica imitada.
        from specialists.trend_specialist import ClippedSAC as _Loader
        model.actor.load_state_dict(_Loader.load(str(best_path), device=model.device).actor.state_dict())
    cloned = best["metrics"]

    # 4) Critic warm-up with the actor frozen.
    print("aquecendo o critico (%d passos, ator congelado)..." % args.critic_warmup)
    model.actor_frozen = True
    model.train(gradient_steps=args.critic_warmup, batch_size=model.batch_size)
    model.actor_frozen = False

    # 5) Fine-tuning with a decaying cloning penalty and validation selection.
    model.bc_observations, model.bc_actions, model.bc_sample_weights = obs_t, act_t, weights
    model.learning_starts = 0
    model.set_env(train_env)

    class Guide(BaseCallback):
        def _on_step(self) -> bool:
            progress = min(1.0, self.num_timesteps / max(1, args.finetune_steps))
            model.bc_weight = args.bc_weight_start + (args.bc_weight_end - args.bc_weight_start) * progress
            if self.num_timesteps % args.eval_every == 0:
                metrics = evaluate(agent, val_df, args.agent, deterministic=True)
                model.save(run_dir / 'models' / ('finetune_%d_validation.zip' % self.num_timesteps))
                current = score(metrics, min_trades=args.min_trades)
                print("  passo %d (bc=%.2f) validacao: %s" % (self.num_timesteps, model.bc_weight, summarize(metrics)))
                if current > best["score"]:
                    best.update(label="ajuste fino %d" % self.num_timesteps, score=current, metrics=metrics)
                    model.save(best_path)
                    print("    -> novo melhor na validacao")
            return True

    print("ajuste fino por RL (%d passos)..." % args.finetune_steps)
    model.learn(total_timesteps=args.finetune_steps, callback=Guide(), reset_num_timesteps=True)

    # 6) Holdout, once, with the policy chosen on validation.
    from specialists.trend_specialist import ClippedSAC
    agent.model = ClippedSAC.load(str(best_path), device=model.device)
    print("\nescolhida pela validacao: %s (%s)" % (best["label"], summarize(best["metrics"])))
    # Training reward and imitation accuracy are not financial performance.
    # Measure the selected deterministic agent on train too, without learning
    # or changing checkpoint selection; only then touch the final holdout.
    trained = evaluate(agent, train_df, args.agent, deterministic=True)
    print("TREINO (politica deterministica selecionada): %s" % summarize(trained))
    holdout = evaluate(agent, holdout_df, args.agent, deterministic=True)
    verdict = judge(holdout, buy_and_hold_return(holdout_df))
    print("HOLDOUT: %s | buy&hold %+.2f%%" % (summarize(holdout), verdict["buy_and_hold_return"] * 100))

    final_path = run_dir / "models" / ("%s_specialist_sac.zip" % args.agent)
    agent.model.save(str(final_path))
    (run_dir / "feature_contract.json").write_text(
        json.dumps({"feature_columns": list(agent.feature_columns),
                    "stop_atr_timeframe": AIConfig.ENV_STOP_ATR_TIMEFRAME,
                    # The leverage range the agent learned in; replay and live
                    # sizing must use the same one.
                    "leverage_bounds": [float(TradingConfig.MIN_LEVERAGE_PER_TRADE),
                                        float(min(TradingConfig.MAX_LEVERAGE_PER_TRADE, AIConfig.TRAINING_LEVERAGE_CAP))],
                    "training_frame_columns": list(df.columns),
                    "dataset": str(args.data)}, indent=2), encoding="utf-8")
    report = {
        "agent": args.agent, "method": "behaviour_cloning+td3bc_finetune",
        "teacher_rule": rule.as_dict(), "teacher_validation": teacher_val,
        "cloned_validation": cloned, "selected": best["label"],
        "selected_validation": best["metrics"], "holdout_metrics": holdout, "verdict": verdict,
        "train_metrics": trained,
        "settings": {"bc_epochs": args.bc_epochs, "dagger_iters": args.dagger_iters,
                     "dagger_epochs": args.dagger_epochs, "critic_warmup": args.critic_warmup,
                     "finetune_steps": args.finetune_steps, "eval_every": args.eval_every,
                     "max_bars": args.max_bars},
        'imitation_objective': 'position_aware_class_balancing_and_macro_recall',
        'seed': args.seed, 'cpu_threads': args.cpu_threads,
        "periods": {"train": [str(train_df.index.min()), str(train_df.index.max())],
                    "validation": [str(val_df.index.min()), str(val_df.index.max())],
                    "holdout": [str(holdout_df.index.min()), str(holdout_df.index.max())]},
        "run_directory": str(run_dir),
    }
    (args.output / ("%s_guided_report.json" % args.agent)).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (run_dir / ("%s_guided_report.json" % args.agent)).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("relatorio: %s" % (run_dir / ("%s_guided_report.json" % args.agent)))
    print("APROVADO PARA OPERAR: %s" % verdict["approved_for_live_trading"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
