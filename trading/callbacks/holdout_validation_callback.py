# -----------------------------------------------------------------------------
# ARQUIVO: trading/callbacks/holdout_validation_callback.py
# -----------------------------------------------------------------------------
"""
🧪 HoldoutValidationCallback — Validação Out-of-Sample Verdadeira

Avalia periodicamente o agente em um dataset TEMPORALMENTE SEPARADO (holdout_df)
que ele NUNCA viu durante o treino nem durante a HPO.

Por que isso importa:
    Trades lucrativos no dataset de treino NÃO provam aprendizado real.
    Q-values baixos e Critic Loss estável também não. A prova científica é
    performance consistente em dados fora-da-amostra (van Hasselt, 2016;
    López de Prado, 2018).

Métricas logadas no TensorBoard (prefixo holdout/):
    - holdout/mean_reward          → reward médio nos episódios do holdout
    - holdout/std_reward           → variabilidade do reward
    - holdout/sharpe               → Sharpe ratio no holdout
    - holdout/avg_trade_pct        → retorno médio por trade (%)
    - holdout/profit_factor        → PF no holdout
    - holdout/win_rate             → WR no holdout
    - holdout/n_trades             → total de trades executados
    - holdout/generalization_ratio → holdout_reward / train_reward
                                     (>0.7 saudável, <0.4 overfit, <0 divergência)

Regra de decisão científica:
    - ratio >= 0.7  → aprendizado real, continuar
    - 0.4 < ratio < 0.7 → zona cinza, monitorar
    - ratio <= 0.4  → OVERFIT, reduzir lr do critic ou early-stop
    - ratio <= 0.0  → DIVERGÊNCIA, parar e recarregar melhor checkpoint

Referências:
    - van Hasselt et al. (2016) — "Deep Reinforcement Learning with Double Q-learning"
    - López de Prado (2018) — "Advances in Financial Machine Learning", Ch. 7 (CV)
    - Henderson et al. (2018) — "Deep RL that Matters" (reprodutibilidade)
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Any, Dict, List
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class HoldoutValidationCallback(BaseCallback):
    """
    Avalia o modelo em um holdout verdadeiramente out-of-sample.

    Args:
        env_builder: Callable[[], VecEnv] — fábrica que constrói o env de holdout
                     sob demanda (evita cópia precoce de dados).
        eval_freq: intervalo em steps entre avaliações (default 25000).
        n_eval_episodes: episódios por avaliação (default 5).
        train_reward_source: callable opcional que retorna o reward médio atual
                             do treino (ex.: lambda: model.logger.name_to_value.get(
                             "rollout/ep_rew_mean", 0.0)). Se fornecido, calcula
                             generalization_ratio.
        verbose: nível de log.
    """

    def __init__(
        self,
        env_builder: Callable[[], Any],
        eval_freq: int = 25_000,
        n_eval_episodes: int = 1,
        stochastic_episodes: int = 3,
        train_reward_source: Optional[Callable[[], float]] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env_builder = env_builder
        self.eval_freq = int(eval_freq)
        # Com holdout curto + deterministic=True, N episódios rodam a MESMA sequência
        # e geram o mesmo reward (std=0). 1 episódio basta para determinística.
        self.n_eval_episodes = int(n_eval_episodes)
        # Avaliação estocástica paralela — revela se o agente CONSEGUE operar
        # com exploração. Se determinística dá 0 trades mas estocástica dá N>0,
        # confirma que a política colapsou para HOLD deterministicamente.
        self.stochastic_episodes = int(stochastic_episodes)
        self.train_reward_source = train_reward_source

        self._eval_env = None
        self._last_eval_step = 0
        self._history: deque = deque(maxlen=20)  # últimas 20 avaliações
        # ── Rastreio autônomo de reward de treino ──
        # Não depende de Monitor wrapper, ep_info_buffer nem do logger (que é
        # limpo após cada dump()). O callback acumula rewards por step e fecha
        # episódios quando done=True — funciona em QUALQUER env SB3.
        # Armazena reward POR STEP (total_ep_reward / n_steps) — elimina o viés
        # de escala quando episódios de treino (~8000 steps) e holdout (~1000
        # steps) têm comprimentos muito diferentes.
        self._cached_train_reward: float = 0.0   # reward/step médio das últimas N ep
        self._current_ep_reward: float = 0.0     # acumulador do episódio atual
        self._current_ep_steps: int = 0          # contador de steps do episódio atual
        self._train_ep_buf: deque = deque(maxlen=100)  # últimos 100 reward/step

    # ------------------------------------------------------------------
    def _build_env_lazy(self):
        """Constrói o env de holdout apenas na primeira avaliação."""
        if self._eval_env is None:
            try:
                self._eval_env = self.env_builder()
                if self.verbose:
                    print(
                        f"🧪 [HOLDOUT] Env de validação out-of-sample construído "
                        f"(lazy init @ step {self.num_timesteps})."
                    )
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ [HOLDOUT] Falha ao construir eval env: {e}")
                self._eval_env = False  # marca como falho, não tenta de novo
        return self._eval_env if self._eval_env else None

    # ------------------------------------------------------------------
    def _extract_trade_metrics(self, env) -> Dict[str, float]:
        """Tenta extrair PF / WR / avg_trade do env (se expuser esses atributos)."""
        metrics = {
            "profit_factor": float("nan"),
            "win_rate": float("nan"),
            "avg_trade_pct": float("nan"),
            "n_trades": 0,
        }
        try:
            # VecEnv → pegamos o env interno [0]
            inner = env
            if hasattr(env, "envs"):
                inner = env.envs[0]
            elif hasattr(env, "venv"):
                try:
                    inner = env.venv.envs[0]
                except Exception:
                    pass
            # Desembrulha Monitor/wrappers
            for _ in range(5):
                if hasattr(inner, "env"):
                    inner = inner.env
                else:
                    break

            # Trades: tenta múltiplas nomenclaturas (TrendFollowingEnv usa
            # `trade_return_history` — list[float] com PnL% por trade fechado).
            trades = (
                getattr(inner, "trade_return_history", None)
                or getattr(inner, "closed_trades", None)
                or getattr(inner, "trade_history", None)
                or getattr(inner, "trades", None)
                or []
            )
            if trades:
                # Extrai PnL — item pode ser float (trade_return_history),
                # dict com 'pnl'/'pnl_pct', ou objeto com atributo.
                pnls: List[float] = []
                for t in trades:
                    if isinstance(t, (int, float)):
                        p = float(t)
                    elif isinstance(t, dict):
                        p = t.get("pnl") or t.get("pnl_pct") or t.get("return_pct")
                    else:
                        p = getattr(t, "pnl", None) or getattr(t, "pnl_pct", None)
                    if p is not None:
                        try:
                            pnls.append(float(p))
                        except (TypeError, ValueError):
                            pass
                if pnls:
                    pnls_arr = np.asarray(pnls, dtype=float)
                    wins = pnls_arr[pnls_arr > 0]
                    losses = pnls_arr[pnls_arr < 0]
                    metrics["n_trades"] = int(len(pnls_arr))
                    metrics["avg_trade_pct"] = float(np.mean(pnls_arr))
                    metrics["win_rate"] = float(len(wins) / len(pnls_arr)) if len(pnls_arr) else 0.0
                    gross_win = float(wins.sum()) if len(wins) else 0.0
                    gross_loss = float(-losses.sum()) if len(losses) else 0.0
                    metrics["profit_factor"] = (
                        (gross_win / gross_loss) if gross_loss > 1e-9 else float("inf")
                    )
        except Exception:
            pass
        return metrics

    # ------------------------------------------------------------------
    def _reset_trade_log(self, env) -> None:
        """Zera o histórico de trades antes de uma avaliação, se possível."""
        try:
            inner = env.envs[0] if hasattr(env, "envs") else env
            for _ in range(5):
                if hasattr(inner, "env"):
                    inner = inner.env
                else:
                    break
            for attr in ("trade_return_history", "closed_trades", "trade_history", "trades"):
                if hasattr(inner, attr):
                    try:
                        setattr(inner, attr, [])
                    except Exception:
                        pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _compute_regime_stats(self, env) -> Dict[str, float]:
        """Calcula ADX/BBW médios do holdout para diagnóstico de regime."""
        out = {
            "regime_adx_mean": float("nan"),
            "regime_bbw_mean": float("nan"),
            "regime_range_ratio": float("nan"),  # % do holdout com ADX<22 & BBW<0.018
        }
        try:
            inner = env.envs[0] if hasattr(env, "envs") else env
            for _ in range(5):
                if hasattr(inner, "env"):
                    inner = inner.env
                else:
                    break
            df = getattr(inner, "df", None)
            if df is None or len(df) == 0:
                return out
            import pandas as pd  # local import
            # procura colunas ADX/BBW (múltiplas nomenclaturas)
            adx_col = next(
                (c for c in df.columns if c.lower().startswith("adx")),
                None,
            )
            bbw_col = next(
                (c for c in df.columns if "bb_width" in c.lower() or "bbw" in c.lower()),
                None,
            )
            if adx_col is not None:
                out["regime_adx_mean"] = float(pd.to_numeric(df[adx_col], errors="coerce").mean())
            if bbw_col is not None:
                out["regime_bbw_mean"] = float(pd.to_numeric(df[bbw_col], errors="coerce").mean())
            if adx_col is not None and bbw_col is not None:
                adx_s = pd.to_numeric(df[adx_col], errors="coerce")
                bbw_s = pd.to_numeric(df[bbw_col], errors="coerce")
                mask = (adx_s < 22.0) & (bbw_s < 0.018)
                out["regime_range_ratio"] = float(mask.mean())
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------
    def _resolve_env_action_threshold(self, env, default: float = 0.03) -> float:
        """
        Lê o threshold mínimo de ação ('intenção de trade') do env por introspecção.
        TrendTradingEnv/RangerTradingEnv expõem `train_min_action_threshold` (≈0.03)
        e `opt_min_action_threshold` (≈0.02). Usamos o menor dos disponíveis para
        não subestimar a atividade do agente (ação pequena já é sinal).
        """
        try:
            inner = env
            if hasattr(env, "envs"):
                inner = env.envs[0]
            elif hasattr(env, "venv"):
                try:
                    inner = env.venv.envs[0]
                except Exception:
                    pass
            for _ in range(5):
                if hasattr(inner, "env"):
                    inner = inner.env
                else:
                    break
            candidates = [
                getattr(inner, "train_min_action_threshold", None),
                getattr(inner, "opt_min_action_threshold", None),
                getattr(inner, "min_action_threshold", None),
            ]
            vals = [float(c) for c in candidates if c is not None]
            if vals:
                # Usa o MENOR threshold — qualquer |a| acima disso é intenção
                return max(1e-4, min(vals))
        except Exception:
            pass
        return float(default)

    # ------------------------------------------------------------------
    def _roll_with_action_log(self, env, deterministic: bool, n_episodes: int):
        """
        Replica evaluate_policy mas:
        1. Conta HOLD/LONG/SHORT por decisão
        2. COLETA TRADES de info['trade'] em cada step — crítico porque VecEnv
           faz auto-reset quando done=True, e reset() zera `trade_return_history`
           DENTRO do env. Se a gente tentar ler esse atributo DEPOIS do loop,
           encontra sempre lista vazia (o bug que estava causando n_trades=0).
        """
        from stable_baselines3.common.vec_env import VecEnv

        action_counts = {"hold": 0, "long": 0, "short": 0}
        ep_rewards: List[float] = []
        ep_steps: List[int] = []       # comprimento real de cada episódio
        collected_trades: List[Dict[str, Any]] = []
        # ── Lê threshold REAL do env (não assume 0.1) ──
        # Em modo treino/eval o env usa `train_min_action_threshold` (≈0.03) como
        # gate de ação significativa. Qualquer |a|>thr é intenção de trade.
        # Fallback para 0.03 se o atributo não for encontrado.
        action_threshold = self._resolve_env_action_threshold(env, default=0.03)

        is_vec = isinstance(env, VecEnv)
        for _ in range(max(1, n_episodes)):
            obs = env.reset()
            done = np.array([False]) if is_vec else False
            ep_r = 0.0
            steps = 0
            max_steps = 50_000  # safety
            while True:
                action, _ = self.model.predict(obs, deterministic=deterministic)

                # classifica a ação (assumindo action space contínuo 1D)
                try:
                    a_val = float(np.asarray(action).flatten()[0])
                    if a_val > action_threshold:
                        action_counts["long"] += 1
                    elif a_val < -action_threshold:
                        action_counts["short"] += 1
                    else:
                        action_counts["hold"] += 1
                except Exception:
                    pass

                if is_vec:
                    obs, reward, done_arr, infos = env.step(action)
                    ep_r += float(np.asarray(reward).flatten()[0])
                    # ── COLETA TRADES ANTES DO AUTO-RESET ──
                    # info['trade'] é emitido pelo env quando um trade fecha
                    # (ver trend_specialist.py L2600). VecEnv propaga via infos[0].
                    try:
                        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
                        if isinstance(info0, dict) and 'trade' in info0:
                            t = info0['trade']
                            if isinstance(t, dict):
                                collected_trades.append(t)
                        # terminal_observation também pode conter info do episode end
                        if isinstance(info0, dict):
                            _te_info = info0.get('terminal_info') or info0
                            if isinstance(_te_info, dict) and _te_info is not info0 and 'trade' in _te_info:
                                t = _te_info['trade']
                                if isinstance(t, dict):
                                    collected_trades.append(t)
                    except Exception:
                        pass
                    if bool(done_arr[0]):
                        break
                else:
                    obs, reward, terminated, truncated, info0 = env.step(action)
                    ep_r += float(reward)
                    try:
                        if isinstance(info0, dict) and 'trade' in info0:
                            t = info0['trade']
                            if isinstance(t, dict):
                                collected_trades.append(t)
                    except Exception:
                        pass
                    if terminated or truncated:
                        break
                steps += 1
                if steps >= max_steps:
                    break
            ep_rewards.append(ep_r)
            ep_steps.append(max(1, steps))

        return ep_rewards, action_counts, collected_trades, ep_steps

    # ------------------------------------------------------------------
    def _trades_to_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Converte lista de trades coletados (info['trade']) em métricas agregadas."""
        metrics = {
            "profit_factor": float("nan"),
            "win_rate": float("nan"),
            "avg_trade_pct": float("nan"),
            "n_trades": 0,
        }
        if not trades:
            return metrics
        pnls: List[float] = []
        pnls_pct: List[float] = []
        for t in trades:
            # pnl é absoluto em USD; derivamos pct quando possível
            p_abs = t.get("pnl")
            entry = t.get("entry_price") or 0.0
            exit_ = t.get("exit_price") or 0.0
            side = str(t.get("side", "")).lower()
            if p_abs is not None:
                try:
                    pnls.append(float(p_abs))
                except (TypeError, ValueError):
                    pass
            # pnl_pct derivado do preço quando disponível
            try:
                if entry and exit_:
                    sign = 1.0 if side.startswith("l") else (-1.0 if side.startswith("s") else 0.0)
                    if sign != 0.0:
                        pct = (float(exit_) - float(entry)) / (float(entry) + 1e-9) * sign
                        pnls_pct.append(pct)
            except Exception:
                pass

        if pnls:
            arr = np.asarray(pnls, dtype=float)
            wins = arr[arr > 0]
            losses = arr[arr < 0]
            metrics["n_trades"] = int(len(arr))
            # avg_trade_pct: usa pct quando disponível, senão usa pnl absoluto
            if pnls_pct:
                metrics["avg_trade_pct"] = float(np.mean(pnls_pct))
            else:
                metrics["avg_trade_pct"] = float(np.mean(arr))
            metrics["win_rate"] = float(len(wins) / len(arr))
            gross_win = float(wins.sum()) if len(wins) else 0.0
            gross_loss = float(-losses.sum()) if len(losses) else 0.0
            metrics["profit_factor"] = (
                (gross_win / gross_loss) if gross_loss > 1e-9 else float("inf")
            )
        return metrics

    # ------------------------------------------------------------------
    def _evaluate(self) -> Optional[Dict[str, float]]:
        env = self._build_env_lazy()
        if env is None:
            return None

        try:
            # ── PASSE 1: DETERMINÍSTICO (política "real" — o que o bot vai fazer em prod) ──
            self._reset_trade_log(env)
            det_rewards, det_actions, det_collected_trades, det_ep_steps = self._roll_with_action_log(
                env, deterministic=True, n_episodes=self.n_eval_episodes
            )
            det_rewards = np.asarray(det_rewards, dtype=float)
            # Normaliza por steps: reward/step é comparável entre episódios de
            # comprimentos distintos (treino ~8000 steps vs holdout ~1000 steps).
            det_steps_arr = np.asarray(det_ep_steps, dtype=float)
            mean_r = float((det_rewards / det_steps_arr).mean()) if len(det_rewards) else 0.0
            # std entre episódios só é significativo com N>1.
            # Com n_eval_episodes=1: std=0 sempre → enganoso → NaN (não logado no TB).
            std_r = (
                float((det_rewards / det_steps_arr).std())
                if len(det_rewards) > 1
                else float("nan")
            )
            # Sharpe entre episódios só existe com N>1. Com n_eval_episodes=1
            # calculamos o Sharpe dos trades individuais (ver abaixo).
            # Usa trades coletados DURANTE o rollout (via info['trade']) porque
            # VecEnv auto-reseta e zera trade_return_history antes de retornarmos.
            det_trades = self._trades_to_metrics(det_collected_trades)
            # Fallback: se info['trade'] não foi emitido, tenta atributos do env
            if det_trades.get("n_trades", 0) == 0:
                fallback = self._extract_trade_metrics(env)
                if fallback.get("n_trades", 0) > 0:
                    det_trades = fallback

            # ── Sharpe dos trades individuais (válido mesmo com 1 episódio) ──
            # Sharpe entre episódios requer N>1 e é sempre 0 com n_eval_episodes=1.
            # Sharpe de trades usa os retornos individuais de cada trade fechado —
            # é a métrica correta para avaliar qualidade de edge em séries curtas.
            sharpe = 0.0
            try:
                _trade_rets: List[float] = []
                for t in det_collected_trades:
                    # Preferência: pct derivado de entry/exit (consistente com avg_trade_pct)
                    entry = t.get("entry_price") or 0.0
                    exit_ = t.get("exit_price") or 0.0
                    side  = str(t.get("side", "")).lower()
                    pct   = None
                    try:
                        if entry and exit_:
                            sign = 1.0 if side.startswith("l") else (-1.0 if side.startswith("s") else 0.0)
                            if sign != 0.0:
                                pct = (float(exit_) - float(entry)) / (float(entry) + 1e-9) * sign
                    except Exception:
                        pass
                    # Fallback: pnl absoluto (mesma escala → consistente entre trades)
                    if pct is None:
                        raw = t.get("pnl")
                        if raw is not None:
                            try:
                                pct = float(raw)
                            except Exception:
                                pass
                    if pct is not None:
                        _trade_rets.append(pct)
                if len(_trade_rets) >= 2:
                    _arr = np.asarray(_trade_rets, dtype=float)
                    _std = float(_arr.std())
                    if _std > 1e-9:
                        sharpe = float(_arr.mean() / _std * np.sqrt(len(_arr)))
            except Exception:
                sharpe = 0.0

            # ── PASSE 2: ESTOCÁSTICO (revela se a política CONSEGUE operar com ruído) ──
            stoch_mean_r = float("nan")
            stoch_n_trades = 0
            stoch_actions = {"hold": 0, "long": 0, "short": 0}
            if self.stochastic_episodes > 0:
                self._reset_trade_log(env)
                stoch_rewards, stoch_actions, stoch_collected_trades, stoch_ep_steps = self._roll_with_action_log(
                    env, deterministic=False, n_episodes=self.stochastic_episodes
                )
                if stoch_rewards:
                    stoch_steps_arr = np.asarray(stoch_ep_steps, dtype=float)
                    stoch_mean_r = float(np.mean(
                        np.asarray(stoch_rewards, dtype=float) / stoch_steps_arr
                    ))
                stoch_trade_metrics = self._trades_to_metrics(stoch_collected_trades)
                if stoch_trade_metrics.get("n_trades", 0) == 0:
                    fb = self._extract_trade_metrics(env)
                    if fb.get("n_trades", 0) > 0:
                        stoch_trade_metrics = fb
                stoch_n_trades = int(stoch_trade_metrics.get("n_trades", 0))

            # ── REGIME STATS do holdout ──
            regime_stats = self._compute_regime_stats(env)

            # ── Distribuição de ações (determinística) ──
            total_det_acts = sum(det_actions.values()) or 1
            det_hold_pct = det_actions["hold"] / total_det_acts
            det_long_pct = det_actions["long"] / total_det_acts
            det_short_pct = det_actions["short"] / total_det_acts

            # det_stoch_gap: quanto a política degrada saindo do modo determinístico.
            # Substitui std_reward (sempre NaN com n_eval_episodes=1 — métrica morta).
            # Interpretação: gap próximo de 0 = política robusta; gap grande = edge frágil.
            det_stoch_gap = (
                float(abs(stoch_mean_r - mean_r))
                if np.isfinite(stoch_mean_r) and np.isfinite(mean_r)
                else float("nan")
            )

            result = {
                "mean_reward": mean_r,
                "det_stoch_gap": det_stoch_gap,   # substitui std_reward
                "sharpe": sharpe,
                "n_episodes": len(det_rewards),
                "action_hold_pct": det_hold_pct,
                "action_long_pct": det_long_pct,
                "action_short_pct": det_short_pct,
                "stoch_mean_reward": stoch_mean_r,
                "stoch_n_trades": stoch_n_trades,
                **det_trades,
                **regime_stats,
            }

            # ── Generalization ratio ──
            # Hierarquia (mais → menos confiável):
            # 1. _cached_train_reward — lido em cada _on_step() ANTES do dump().
            #    Funciona mesmo sem Monitor wrapper no env de treino (o caso comum).
            # 2. train_reward_source (closure do specialist) — fallback externo.
            # 3. ep_info_buffer — só funciona se o env usar Monitor wrapper.
            train_r = 0.0
            # Fonte 1: cache (mais confiável — lido antes do dump)
            if abs(self._cached_train_reward) > 1e-6:
                train_r = self._cached_train_reward
            # Fonte 2: callable externo
            if abs(train_r) < 1e-6 and self.train_reward_source is not None:
                try:
                    v = self.train_reward_source()
                    if v is not None and abs(float(v)) > 1e-6:
                        train_r = float(v)
                except Exception:
                    pass
            # Fonte 3: ep_info_buffer (requer Monitor wrapper)
            if abs(train_r) < 1e-6:
                try:
                    buf = getattr(self.model, 'ep_info_buffer', None)
                    if buf is not None and len(buf) > 0:
                        rewards = [float(ep.get('r', 0.0)) for ep in buf if 'r' in ep]
                        if rewards and abs(float(np.mean(rewards))) > 1e-6:
                            train_r = float(np.mean(rewards))
                except Exception:
                    pass
            if abs(train_r) > 1e-6:
                result["generalization_ratio"] = mean_r / train_r
            else:
                result["generalization_ratio"] = float("nan")

            # Guardamos as raw action counts para uso no print
            result["_det_actions"] = det_actions
            result["_stoch_actions"] = stoch_actions
            return result
        except Exception as e:
            if self.verbose:
                print(f"⚠️ [HOLDOUT] Erro na avaliação: {e}")
            return None

    # ------------------------------------------------------------------
    def _log_and_print(self, metrics: Dict[str, float]) -> None:
        # Logga no TensorBoard — pula chaves internas que começam com "_"
        for k, v in metrics.items():
            if k.startswith("_"):
                continue
            try:
                if isinstance(v, (int, float)) and np.isfinite(v):
                    self.logger.record(f"holdout/{k}", float(v))
            except Exception:
                pass

        if self.verbose:
            ratio = metrics.get("generalization_ratio", float("nan"))
            ratio_str = f"{ratio:+.2f}" if np.isfinite(ratio) else "N/A"
            n_trades_det = int(metrics.get("n_trades", 0))
            n_trades_stoch = int(metrics.get("stoch_n_trades", 0))
            hold_pct = metrics.get("action_hold_pct", 0.0) * 100
            long_pct = metrics.get("action_long_pct", 0.0) * 100
            short_pct = metrics.get("action_short_pct", 0.0) * 100
            regime_ratio = metrics.get("regime_range_ratio", float("nan"))
            adx_mean = metrics.get("regime_adx_mean", float("nan"))
            bbw_mean = metrics.get("regime_bbw_mean", float("nan"))

            # ── Veredito científico multi-dimensional ──
            # Extrai métricas de qualidade de trade (thresholds absolutos no holdout)
            pf  = metrics.get("profit_factor", float("nan"))
            wr  = metrics.get("win_rate", float("nan"))
            pf_ok   = np.isfinite(pf) and pf > 1.5
            pf_good = np.isfinite(pf) and pf > 2.5
            wr_ok   = np.isfinite(wr) and wr > 0.50
            quality_ok   = pf_ok and wr_ok     # PF>1.5 e WR>50%
            quality_good = pf_good and wr_ok   # PF>2.5 e WR>50%

            # Contexto de seletividade: HOLD alto em regime range é CORRETO
            # Um agente de range com 94% dos candles laterais DEVE ser seletivo.
            # Comparar reward/step com treino nesse caso penaliza a disciplina.
            hold_pct_frac = hold_pct / 100.0
            range_regime = np.isfinite(regime_ratio) and regime_ratio > 0.70
            selective_hold = range_regime and hold_pct_frac > 0.40

            # Detecta bias direcional: uma direção domina > 80% das decisões
            dominant_side = None
            bias_severe = False
            if long_pct >= 80:
                dominant_side = f"LONG ({long_pct:.0f}%)"
                bias_severe = long_pct >= 90
            elif short_pct >= 80:
                dominant_side = f"SHORT ({short_pct:.0f}%)"
                bias_severe = short_pct >= 90

            # ── Árvore de decisão ──
            if n_trades_det == 0 and n_trades_stoch == 0:
                if hold_pct >= 80:
                    if range_regime:
                        verdict = "🔴 POLÍTICA PARALISADA — HOLD total mesmo com regime range favorável"
                    else:
                        verdict = "✅ HOLDOUT SEM REGIME — agente corretamente NÃO operou (disciplina OK)"
                elif dominant_side is not None:
                    sev = "SEVERO" if bias_severe else "forte"
                    verdict = (
                        f"🔴 BIAS DIRECIONAL {sev} — {dominant_side} dominante, "
                        f"posição aberta e NUNCA fechada (overfit direcional)"
                    )
                else:
                    verdict = "⚠️  AÇÕES MISTAS mas 0 trades fechados — possível filtro do env bloqueando"

            elif n_trades_det == 0 and n_trades_stoch > 0:
                verdict = "⚠️  DETERMINÍSTICA COLAPSADA — só fecha trades com exploração (conservadora demais)"

            elif dominant_side is not None and bias_severe:
                verdict = f"⚠️  BIAS {dominant_side} — opera mas unidimensional (verificar em 50k+)"

            elif not np.isfinite(ratio):
                verdict = "ℹ️  (aguardando episódios de treino para gen_ratio)"

            else:
                # ── Diagnóstico principal: qualidade de trade + gen_ratio + contexto ──
                #
                # REGRA CIENTÍFICA: "overfit real" = qualidade PIOR no holdout.
                # Se PF e WR são superiores no holdout, o agente generalizou —
                # gen_ratio baixo reflete seletividade (mais HOLD), não memorização.
                #
                if quality_good and selective_hold:
                    # Alta qualidade + HOLD alto em range: disciplina correta
                    if ratio >= 0.35:
                        verdict = (
                            f"✅ APRENDIZADO REAL — edge OOS confirmado "
                            f"(WR {wr*100:.0f}% / PF {pf:.2f}), "
                            f"gen_ratio baixo por seletividade correta em range"
                        )
                    else:
                        verdict = (
                            f"⚠️  ZONA CINZA — qualidade OOS alta (WR {wr*100:.0f}% / PF {pf:.2f}) "
                            f"mas gen_ratio {ratio:+.2f} muito baixo — monitorar tendência"
                        )
                elif quality_good and not selective_hold:
                    # Alta qualidade sem desculpa de seletividade
                    if ratio >= 0.5:
                        verdict = f"✅ APRENDIZADO REAL — generaliza bem (WR {wr*100:.0f}% / PF {pf:.2f} / ratio {ratio:+.2f})"
                    else:
                        verdict = f"⚠️  ZONA CINZA — qualidade OK (WR {wr*100:.0f}% / PF {pf:.2f}) mas gen_ratio {ratio:+.2f} abaixo do esperado"
                elif quality_ok and not quality_good:
                    # Qualidade aceitável
                    if ratio >= 0.4:
                        verdict = f"⚠️  ZONA CINZA — gen_ratio {ratio:+.2f} e trade quality moderada (WR {wr*100:.0f}% / PF {pf:.2f})"
                    elif selective_hold:
                        verdict = f"⚠️  ZONA CINZA — qualidade aceitável com seletividade, aguardar 75k+ para diagnóstico firme"
                    else:
                        verdict = f"🔴 ATENÇÃO — gen_ratio {ratio:+.2f} baixo E qualidade marginal (WR {wr*100:.0f}% / PF {pf:.2f})"
                else:
                    # Qualidade ruim no holdout — ESTE é o overfit real
                    if ratio < 0.0:
                        verdict = "❌ DIVERGÊNCIA — parar e recarregar melhor checkpoint"
                    else:
                        verdict = (
                            f"🔴 OVERFIT REAL — trade quality degradada OOS "
                            f"(WR {wr*100:.0f}% < 50% ou PF {pf:.2f} < 1.5) "
                            f"+ gen_ratio {ratio:+.2f}"
                        )

            print(
                "\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🧪 [HOLDOUT @ step {self.num_timesteps:,}]  VALIDAÇÃO OUT-OF-SAMPLE\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"   ── DETERMINÍSTICA (política de produção) ──\n"
                f"   mean_reward       : {metrics.get('mean_reward', 0):+.6f}\n"
                f"   det_stoch_gap     : {metrics.get('det_stoch_gap', float('nan')):.4f}  (0=robusto, alto=frágil)\n"
                f"   sharpe (trades)   : {metrics.get('sharpe', 0):+.3f}\n"
                f"   n_trades          : {n_trades_det}\n"
                f"   avg_trade_pct     : {metrics.get('avg_trade_pct', float('nan')):+.4f}\n"
                f"   win_rate          : {metrics.get('win_rate', float('nan')):.3f}\n"
                f"   profit_factor     : {metrics.get('profit_factor', float('nan')):.3f}\n"
                f"   ── AÇÕES (distribuição %) ──\n"
                f"   HOLD / LONG / SHORT : {hold_pct:.1f}% / {long_pct:.1f}% / {short_pct:.1f}%\n"
                f"   ── ESTOCÁSTICA (exploração, diagnóstico) ──\n"
                f"   stoch_mean_reward : {metrics.get('stoch_mean_reward', float('nan')):+.4f}\n"
                f"   stoch_n_trades    : {n_trades_stoch}\n"
                f"   ── REGIME DO HOLDOUT ──\n"
                f"   ADX médio         : {adx_mean:.2f}   (< 22 = lateral)\n"
                f"   BBW médio         : {bbw_mean:.4f} (< 0.018 = squeeze)\n"
                f"   %candles em range : {regime_ratio*100 if np.isfinite(regime_ratio) else float('nan'):.1f}%\n"
                f"   ── DIAGNÓSTICO ──\n"
                f"   gen_ratio         : {ratio_str}\n"
                f"   veredito          : {verdict}\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        # ── RASTREIO AUTÔNOMO DE REWARD DE TREINO ──
        # Lê rewards/dones do env de treino via self.locals (disponível em
        # _on_step de qualquer callback SB3, on-policy ou off-policy).
        # Não depende de Monitor wrapper, ep_info_buffer nem do logger.
        try:
            step_reward = float(
                np.asarray(self.locals.get('rewards', [0.0])).flatten()[0]
            )
            step_done = bool(
                np.asarray(self.locals.get('dones', [False])).flatten()[0]
            )
            self._current_ep_reward += step_reward
            self._current_ep_steps += 1
            if step_done:
                # Guarda reward POR STEP — normaliza episódios de comprimentos diferentes
                if self._current_ep_steps > 0:
                    self._train_ep_buf.append(
                        self._current_ep_reward / self._current_ep_steps
                    )
                self._current_ep_reward = 0.0
                self._current_ep_steps = 0
                if len(self._train_ep_buf) > 0:
                    self._cached_train_reward = float(
                        np.mean(list(self._train_ep_buf))
                    )
        except Exception:
            pass

        # Só avalia a cada eval_freq steps (num_timesteps é global ao modelo)
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        metrics = self._evaluate()
        if metrics is None:
            return True

        self._history.append(metrics)
        self._log_and_print(metrics)
        return True

    # ------------------------------------------------------------------
    def _on_training_end(self) -> None:
        # Avaliação final + dump de histórico
        if self.verbose and self._history:
            ratios = [
                m.get("generalization_ratio", float("nan"))
                for m in self._history
                if np.isfinite(m.get("generalization_ratio", float("nan")))
            ]
            rewards = [m.get("mean_reward", 0.0) for m in self._history]
            print(
                "\n🧪 [HOLDOUT] Resumo do treino:\n"
                f"   Avaliações realizadas : {len(self._history)}\n"
                f"   mean_reward (último)  : {rewards[-1]:+.4f}\n"
                f"   mean_reward (média)   : {float(np.mean(rewards)):+.4f}\n"
                + (
                    f"   gen_ratio médio       : {float(np.mean(ratios)):+.3f}\n"
                    if ratios
                    else ""
                )
            )
