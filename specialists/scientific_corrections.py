"""
REWARD CIENTÍFICO — LÓPEZ DE PRADO v5.0
========================================
Baseado em:
  - "Advances in Financial Machine Learning" (AFLM, 2018) — Triple Barrier, Meta-Labeling,
    Anti-Scalping via Fractional Differentiation e CUSUM filters
  - "Machine Learning for Asset Managers" (2020) — HRP, Sharpe/Sortino-based exit quality
  - "Causal Factor Investing" (2023) — Reward por fatores causais, nao correlacoes espurias
  - Conservative Q-Learning (CQL) — Kumar et al. (2020) — via reward clipping para controlar
    Q-value overestimation observada no log (actor loss → -99.7)
  - Calmar Ratio — Young (1991) — return/max_drawdown como medida de qualidade do sistema
  - Sortino Ratio — Sortino & van der Meer (1991) — penalizacao assimetrica do downside
  - Ornstein-Uhlenbeck — Uhlenbeck & Ornstein (1930); Chan (2013) — half-life de mean reversion
  - Bollinger (1992) — BB Width como proxy de volatilidade comprimida para regime lateral

Diagnostico aplicado (Best Trial number=5):
  [P1] 90% duration=1: Anti-Scalping Exponencial (Comp. 1B) + Minimum Hold Reward (Comp. 9)
  [P2] Actor loss -99.7 (Q-overestimation): CQL Reward Clipping (Comp. 3, clip final [-20,20])
  [P3] VOTE_MISALIGNED com confiança 1.0: Penalidade Dinamica proporcional (Comp. 5)
  [P4] Regime mismatch fraca (-0.40 plana): substituida por penalidade dinamica (Comp. 5)
  [P5] Grace period sem sinal de saida: Emergency Exit Signal (Comp. 6)
  [P6] Scalp duration=1 compensa taxas: Anti-Scalping bloqueia via penalidade exponencial
  [v5] Ranger-aware: adaptacoes para mercado lateral (mean reversion) em 3 componentes:
       1A → EMA Convergence Well (substitui Gravity Well para Ranger)
       1B → Anti-Scalping OU-aware (cap no half-life; sem bonus apos 2×HL)
       2  → Entry Precision com thresholds MR (RSI 32/68, Z-score 1.8, ADX/BBW)
       9  → OU Half-Life Completion Bonus (substitui Minimum Hold para Ranger)
       10 → Opportunity Cost com range quality para Ranger (bidirecional)

Componentes (ordem de calculo):
  1A. Gravity Well      (AFLM Ch.3): profundidade ATR; Ranger: EMA Convergence Well
  1B. Anti-Scalping     (AFLM CUSUM): < 5 steps penalidade; Ranger: cap no OU half-life
  2.  Entry Precision   (AFLM Ch.4): RSI/BB extremes; Ranger: + Z-score + ADX + BBW
  3.  Exit Quality      (AFLM Ch.3 Triple Barrier + CQL Clipping): Sharpe × sqrt(dur)
  4.  Delta Drawdown    Penalty incremental de drawdown (nao cumulativa)
  5.  Vote Misaligned   Penalidade DINAMICA proporcional a |conf * dir|
  6.  Emergency Exit    Grace period: sinal positivo quando pnl < -2% apos graca
  7.  Sortino Ratio     Penalidade assimetrica downside vs upside returns
  8.  Calmar Component  Bonus proporcional a return/max_drawdown
  9.  Minimum Hold      Trend: bonus >= 3/5/15 steps; Ranger: OU half-life completion
  10. Opportunity Cost  Trend: penaliza flat/sinal forte; Ranger: penaliza flat/range quality
  11. Entry Friction    Custo transacional no step de abertura
"""

import numpy as np
from typing import Optional, Dict, Any


def compute_scientific_reward(
    env,
    pnl_realized: float,
    trade_return_pct: float,
    duration_steps: int,
    exit_reason: Optional[str],
    current_price: float,
    atr: float,
    prev_price: Optional[float] = None,
    info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    [LOPÉZ DE PRADO SCIENTIFIC REWARD — v4.0]

    Correcoes aplicadas com base no diagnostico do log (Best Trial #5):
    - Anti-Scalping Exponencial elimina incentivo a duration=1
    - CQL via clip final [-20, 20] controla Q-value overestimation
    - Penalidade VOTE_MISALIGNED dinamica (proporcional a conf*dir, max -2.0)
    - Emergency Exit Signal no grace period para perdas > -2%
    - Sortino Ratio e Calmar Ratio como componentes de qualidade do sistema
    - Minimum Hold Bonus reforça transicao de scalping para holding real
    """
    reward = 0.0
    info = info or {}

    # ── Detecção de especialista ──────────────────────────────────────────────
    # Ranger (mercado lateral) usa variantes adaptadas de vários componentes.
    # Bull/Bear (tendência) usam os componentes originais sem alteração.
    _specialist_name = str(getattr(env, 'specialist_name', '')).lower()
    _is_ranger = 'ranger' in _specialist_name
    # Half-life de Ornstein-Uhlenbeck: Ranger usa 15 candles; fallback seguro para outros.
    _mr_halflife = int(getattr(env, 'mean_reversion_halflife', 15)) if _is_ranger else 15

    # Variaveis de contexto reutilizadas em multiplos componentes
    steps_in_pos = int(getattr(env, 'steps_in_position', 0))
    atr_pct = atr / (current_price + 1e-9) if current_price > 0 else 0.01
    depth_in_atr = 0.0  # sera preenchido no Componente 1A se posicao aberta
    # exit_reason setado = trade fechou neste step, mesmo que pnl_realized == 0.0
    # (break-even exato). Isso garante que todos os componentes de fechamento disparam.
    _pnl_finite = pnl_realized is not None and np.isfinite(pnl_realized)
    trade_closed_this_step = _pnl_finite and (pnl_realized != 0 or exit_reason is not None)
    # [ZERO-REWARD GUARD] Break-even puro (pnl=0 sem exit_reason) = flat desnecessário.
    # Não é "neutro" — tem custo de oportunidade. Será penalizado no clip final.

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENTE 1A: GRAVITY WELL + MOMENTUM + RETREAT SIGNAL (AFLM Ch.3)
    # ─────────────────────────────────────────────────────────────────────────
    # Tres sinais per-step que ensinam o agente a "deixar o winner correr":
    #
    # A) GRAVITY WELL (profundidade): quanto mais fundo em lucro, mais reward.
    #    Saturacao em ±3 ATR via tanh para evitar reward explosivo.
    #
    # B) MOMENTUM: bônus quando o trade avanca na direcao correta neste step.
    #    Usa tracker interno _prev_sci_unrealized para delta correto.
    #
    # C) RETREAT SIGNAL (Option B ratchet): penalidade por dar-back de lucro
    #    do pico quando profundidade > 0.5 ATR. Ensina saida organica.
    #
    if env.position != 0 and hasattr(env, 'entry_price') and env.entry_price > 0:
        unrealized_return = (
            (current_price - env.entry_price) / env.entry_price
            * np.sign(env.position)
        )
        depth_in_atr = unrealized_return / (atr_pct + 1e-9)
        leverage = float(min(getattr(env, 'current_leverage', 1.0), 3.0))

        if not _is_ranger:
            # ── TREND (Bull/Bear): Gravity Well clássico ─────────────────────
            # Recompensa profundidade do trade em relação ao preço de entrada.
            # "Deixar o winner correr" — quanto mais fundo em lucro, mais reward.
            depth_reward = 0.20 * np.tanh(depth_in_atr / 3.0) * leverage
            reward += float(np.clip(depth_reward, -0.30, 0.30))
        else:
            # ── RANGER: EMA Convergence Well ─────────────────────────────────
            # Substitui Gravity Well por "Poço de Convergência à EMA".
            #
            # Lógica de Mean Reversion (Chan, 2013):
            #   - O objetivo NÃO é ir mais fundo do preço de entrada.
            #   - O objetivo É convergir em direção à EMA (média).
            #   - Cada step que o preço se aproxima da EMA = reward positivo.
            #   - Cada step que o preço se afasta da EMA = reward negativo.
            #
            # Fórmula:
            #   dist_entry   = |entry_price - EMA| / ATR  (distância na abertura)
            #   dist_current = |current_price - EMA| / ATR (distância atual)
            #   convergence  = (dist_entry - dist_current) / max(dist_entry, 0.5)
            #                  → +1.0 = chegou à EMA | 0.0 = sem movimento | -1.0 = afastou
            #   reward = 0.18 * tanh(convergence * 2.5)   → [-0.18, +0.18] por step
            #
            try:
                ema_val = 0.0
                if hasattr(env, '_get_ema_trend_value') and hasattr(env, 'current_row'):
                    ema_val = float(env._get_ema_trend_value(env.current_row))
                if ema_val <= 0 and hasattr(env, 'df') and hasattr(env, 'start_idx'):
                    # Fallback: tentar leitura direta do df
                    _idx = min(env.start_idx + env.current_step, len(env.df) - 1)
                    _row = env.df.iloc[_idx]
                    for _col in ('ema_trend', 'ema_50', 'ema_20', 'ema_200'):
                        _v = _row.get(_col, 0.0) if hasattr(_row, 'get') else 0.0
                        if _v > 0:
                            ema_val = float(_v)
                            break

                if ema_val > 0:
                    _entry = float(env.entry_price)
                    dist_entry   = abs(_entry        - ema_val) / (atr + 1e-9)
                    dist_current = abs(current_price - ema_val) / (atr + 1e-9)

                    # [EMA WELL GATE] Só ativa o well se a entrada foi suficientemente
                    # distante da EMA (entrada de qualidade em mean reversion).
                    # Se dist_entry < 0.8 ATR, o agente entrou perto da EMA (entrada ruim)
                    # → aplicar penalidade aqui seria injusto e gera -0.20/step desnecessário.
                    # Nesse caso, reward neutro (0): a punição virá via Entry Precision (Comp 2).
                    _EMA_WELL_MIN_DIST = 0.8  # mínimo 0.8 ATR de distância para ativar o well
                    if dist_entry >= _EMA_WELL_MIN_DIST:
                        convergence  = (dist_entry - dist_current) / max(dist_entry, 0.5)
                        ema_well_reward = 0.18 * np.tanh(convergence * 2.5)
                        reward += float(np.clip(ema_well_reward, -0.20, 0.20))
                    # else: entrada perto da EMA → zona neutra (Entry Precision já penalizou na abertura)

                    # Tracker de convergência para Momentum adaptado (abaixo)
                    env._ranger_dist_ema_prev = dist_current
                else:
                    # Fallback: Gravity Well normal se EMA não disponível
                    depth_reward = 0.20 * np.tanh(depth_in_atr / 3.0)
                    reward += float(np.clip(depth_reward, -0.20, 0.20))
            except Exception:
                pass  # Falha silenciosa — não quebra o reward

        # B) Momentum
        curr_unrealized = float(getattr(env, 'pnl_since_entry', unrealized_return))
        if steps_in_pos <= 1:
            env._prev_sci_unrealized = curr_unrealized
            momentum_reward = 0.0
        else:
            prev_unrealized_sci = float(getattr(env, '_prev_sci_unrealized', curr_unrealized))
            delta_return = curr_unrealized - prev_unrealized_sci
            momentum_norm = delta_return / (atr_pct + 1e-9)
            if not _is_ranger:
                # Trend: momentum padrão (avançar na direção da posição = bom)
                momentum_reward = 0.08 * np.tanh(momentum_norm * 5.0)
            else:
                # Ranger: momentum só é bom se estiver convergindo à EMA.
                # O EMA Convergence Well acima já captura isso; aqui apenas
                # penalizamos levemente quando PnL deteriora (saída iminente).
                if curr_unrealized < -0.005 and delta_return < 0:
                    momentum_reward = 0.04 * np.tanh(momentum_norm * 5.0)
                else:
                    momentum_reward = 0.0  # neutro — EMA Well já sinalizou
            env._prev_sci_unrealized = curr_unrealized
        reward += float(np.clip(momentum_reward, -0.15, 0.15))

        # C) Retreat Signal — penaliza erosao do pico (Option B)
        # Aplicado ao Trend; no Ranger, o _should_exit_fast já trata retreats.
        if not _is_ranger:
            peak_price = getattr(env, '_peak_price_since_entry', None)
            if peak_price is not None and float(env.entry_price) > 0:
                peak_return = (
                    (float(peak_price) - float(env.entry_price))
                    / float(env.entry_price)
                    * np.sign(env.position)
                )
                peak_depth_in_atr = peak_return / (atr_pct + 1e-9)
                retreat_atr = peak_depth_in_atr - depth_in_atr
                if retreat_atr > 1.0 and depth_in_atr > 0.5:
                    retreat_pen = -0.12 * np.tanh(retreat_atr / 2.0)
                    reward += float(np.clip(retreat_pen, -0.25, 0.0))

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENTE 1B: ANTI-SCALPING EXPONENCIAL (AFLM CUSUM / position sizing)
    # ─────────────────────────────────────────────────────────────────────────
    # Ref: Lopez de Prado (2018) AFLM — CUSUM filters e fractional diff.
    # mostram que returns muito curtos nao capturam sinal e sao dominados por
    # ruido transacional. Esta penalidade exponencial direciona o agente para
    # trades de duracao suficiente para ter significancia estatistica.
    #
    # Esquema:
    #   duration=1  → penalidade -3.0  (scalp de 1 candle: custo alto)
    #   duration=2  → penalidade -1.84 (ainda caro)
    #   duration=3  → penalidade -0.99 (limiar do Minimum Hold bonus)
    #   duration=5  → penalidade -0.20 (zona neutra: apenas friccao residual)
    #   duration=10 → bonus      +0.50 (holding de qualidade)
    #   duration=20 → bonus      +1.00 (posicao com direcao)
    #   duration=40 → bonus      +1.30 (satura)
    #
    # Formula:
    #   se d < 5  : pen  = -3.0 * exp(-0.5 * (d - 1))  [exponencial decrescente]
    #   se d >= 10: bonus = +1.5 * (1 - exp(-0.07 * (d - 10)))  [crescimento sublinear]
    #   5 <= d < 10: zona neutra (sem ajuste)
    #
    # O componente so e aplicado quando a posicao esta ABERTA (per-step),
    # evitando dupla contagem com o Exit Quality no step de fechamento.
    if env.position != 0 and not trade_closed_this_step:
        if not _is_ranger:
            # ── TREND (Bull/Bear): Anti-Scalping + Hold Bonus clássico ───────
            # Incentiva "deixar o winner correr" indefinidamente.
            if steps_in_pos < 5:
                # Penalidade exponencial decrescente para scalps
                # exp(-0.5 * 0) = 1.0  → d=1: -3.0
                # exp(-0.5 * 1) = 0.61 → d=2: -1.84
                # exp(-0.5 * 2) = 0.37 → d=3: -1.10
                # exp(-0.5 * 3) = 0.22 → d=4: -0.67
                scalp_penalty = -3.0 * np.exp(-0.5 * max(0, steps_in_pos - 1))
                reward += float(np.clip(scalp_penalty, -3.5, 0.0))
            elif steps_in_pos >= 10:
                # Bonus crescente para trades com duracao real, escalado pela lucratividade.
                # depth_in_atr > 0 = lucrativo, < 0 = perdendo.
                # profit_scale: 0.0 se perdendo, 0.3 se neutro, 1.0 se +1 ATR de lucro.
                # Evita recompensar segurar trades perdedores só pelo tempo.
                hold_bonus = 1.5 * (1.0 - np.exp(-0.07 * (steps_in_pos - 10)))
                profit_scale = float(np.clip(0.3 + depth_in_atr * 0.7, 0.0, 1.0))
                reward += float(np.clip(hold_bonus * profit_scale, 0.0, 1.5))
            # 5 <= steps_in_pos < 10: zona neutra — sem ajuste
        else:
            # ── RANGER: SEM Anti-Scalping per-step ───────────────────────────
            # Por que NÃO aplicar anti-scalp per-step no Ranger:
            #
            # 1. O Ranger JÁ tem deterrente de scalping via fee deduction no pnl_base:
            #    net_return = gross_return - 2×TAKER_FEE×leverage
            #    → trades que não cobrem a taxa já recebem pnl_base negativo no fechamento.
            #
            # 2. O OU efficiency JÁ penaliza holds muito curtos OU muito longos:
            #    d < 3    → multiplier 0.10 (quase zero)
            #    d ≤ HL   → multiplier 1.00 (ideal)
            #    d > 3×HL → multiplier -0.20 (penalidade direta)
            #
            # 3. Anti-scalp per-step gerava -5.21 por trade × 38 trades = -198/episódio.
            #    Isso excede qualquer reward positivo (~+20/episódio) → TODOS os trials
            #    têm reward < 0 → Optuna não aprende nada → 100 trials × 34min = 57h perdidos.
            #
            # 4. O _should_exit_fast JÁ controla saídas rápidas (EMA atingida, ADX subiu).
            #    Penalizar PER-STEP o que a arquitetura de saída JÁ trata = dupla contagem.
            #
            # Comportamento: zona neutra total para todos os steps no Ranger.
            # Os sinais de qualidade de duração vêm APENAS do trade reward (OU efficiency)
            # e da EMA Convergence Well per-step (Comp 1A). Não desta função.
            pass  # Ranger: sem anti-scalp per-step. Fee + OU efficiency + EMA Well = suficiente.

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENTE 2: ENTRY PRECISION BONUS (AFLM Ch.4 — Meta-Labeling)
    # ─────────────────────────────────────────────────────────────────────────
    # O regime predictor (sinal primario) diz a DIRECAO.
    # O agente RL (modelo secundario) decide QUANDO entrar.
    # Bonificar entradas em zonas de valor extremo aumenta precisao da estrategia.
    #
    #   Long:  RSI < 35 (oversold)   + preco perto da BB inferior → bonus cumulativo
    #   Short: RSI > 65 (overbought) + preco perto da BB superior → bonus cumulativo
    #   Maximo: +0.8 por entrada (RSI extremo + confluencia de BB)
    #
    if hasattr(env, 'steps_in_position') and env.steps_in_position == 1:
        entry_bonus = 0.0
        try:
            row = getattr(env, 'current_row', None)
            if row is not None:
                def _get(names, default):
                    """Lookup robusto: tenta multiplos nomes de coluna."""
                    for n in names:
                        try:
                            v = row.get(n) if hasattr(row, 'get') else (
                                row[n] if n in row.index else None
                            )
                            if v is not None and np.isfinite(float(v)):
                                return float(v)
                        except Exception:
                            pass
                    return default

                rsi = _get(['rsi_1h', 'rsi_4h', 'rsi_15m', 'rsi'], default=50.0)
                bb_lower = _get(['bb_lower_1h', 'bb_lower_4h', 'bb_lower'], default=0.0)
                bb_upper = _get(['bb_upper_1h', 'bb_upper_4h', 'bb_upper'], default=0.0)

                bb_range = max(bb_upper - bb_lower, 1e-9)
                if bb_lower > 0 and bb_upper > bb_lower:
                    bb_pct_b = (current_price - bb_lower) / bb_range
                else:
                    bb_pct_b = 0.5

                if not _is_ranger:
                    # ── Trend (Bull/Bear): thresholds padrão ─────────────────
                    if env.position > 0:  # Long: comprar perto do fundo
                        if rsi < 35.0:
                            entry_bonus += 0.5 * max(0.0, (35.0 - rsi) / 35.0)
                        if bb_pct_b < 0.25:
                            entry_bonus += 0.3 * (1.0 - bb_pct_b / 0.25)
                    elif env.position < 0:  # Short: vender perto do topo
                        if rsi > 65.0:
                            entry_bonus += 0.5 * max(0.0, (rsi - 65.0) / 35.0)
                        if bb_pct_b > 0.75:
                            entry_bonus += 0.3 * ((bb_pct_b - 0.75) / 0.25)
                else:
                    # ── Ranger: thresholds calibrados para Mean Reversion ─────
                    # Usa RSI 32/68 (alinhado com _RSI_OVERSOLD/_RSI_OVERBOUGHT do Ranger)
                    # BB: posição extrema (< 0.15 para Long, > 0.85 para Short)
                    #
                    # Bônus adicional para Z-score extremo (dupla confirmação MR):
                    z_score = _get(['z_score', 'zscore'], default=0.0)
                    adx_val = _get(['adx'], default=25.0)
                    bb_width = _get(['bb_width', 'bbw'], default=0.025)

                    if env.position > 0:  # Long: entrada oversold
                        if rsi < 32.0:  # Ranger: RSI_OVERSOLD=32
                            entry_bonus += 0.5 * max(0.0, (32.0 - rsi) / 32.0)
                        if bb_pct_b < 0.15:  # Fundo extremo da BB
                            entry_bonus += 0.35 * (1.0 - bb_pct_b / 0.15)
                        if z_score < -1.8:  # Z-score extremo (ZSCORE_OVERSOLD=-1.8)
                            entry_bonus += 0.30 * min(1.0, abs(z_score) / 3.0)

                    elif env.position < 0:  # Short: entrada overbought
                        if rsi > 68.0:  # Ranger: RSI_OVERBOUGHT=68
                            entry_bonus += 0.5 * max(0.0, (rsi - 68.0) / 32.0)
                        if bb_pct_b > 0.85:  # Topo extremo da BB
                            entry_bonus += 0.35 * ((bb_pct_b - 0.85) / 0.15)
                        if z_score > 1.8:   # Z-score extremo (ZSCORE_OVERBOUGHT=+1.8)
                            entry_bonus += 0.30 * min(1.0, abs(z_score) / 3.0)

                    # Bônus de qualidade de range: ADX baixo + BB estreita = entrada ideal
                    if adx_val < 22.0 and bb_width < 0.018:
                        entry_bonus += 0.25  # Range de alta qualidade confirmado
                    elif adx_val < 22.0 or bb_width < 0.018:
                        entry_bonus += 0.10  # Um sinal de qualidade

        except Exception:
            pass  # Falha silenciosa — features indisponiveis nao quebram o reward
        reward += float(np.clip(entry_bonus, 0.0, 1.2 if _is_ranger else 0.8))

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENTE 3: EXIT QUALITY — Triple Barrier + CQL Clipping (AFLM Ch.3)
    # ─────────────────────────────────────────────────────────────────────────
    # Ref CQL: Kumar et al. (2020) "Conservative Q-Learning for Offline RL"
    # O actor loss explodindo para -99.7 indica Q-value overestimation severa.
    # CQL controla isso via pessimism na estimativa: o reward clipping final
    # em [-20, 20] atua como regularizador implicito no target Q.
    #
    # Triple Barrier: Sharpe × sqrt(duration/10) escala pela qualidade temporal.
    # sqrt() e mais conservadora que log1p, penalizando mais fortemente scalps.
    #
    #   dur_mult: sqrt(1/10)=0.316 | sqrt(5/10)=0.707 | sqrt(10/10)=1.0 | sqrt(40/10)=2.0
    #
    if (pnl_realized != 0 and np.isfinite(pnl_realized)
            and not info.get('trade_reward_already_computed', False)):
        trade_sharpe = trade_return_pct / max(atr_pct, 1e-6)
        sharpe_reward = 20.0 * np.tanh(trade_sharpe)
        duration_mult = min(2.5, 1.0 + np.log1p(duration_steps / 10.0))
        if sharpe_reward > 0:
            reward += sharpe_reward * duration_mult
        else:
            reward += sharpe_reward  # perdedores: sem amplificacao por duracao

    # CSL PENALTY — executa SEMPRE para Trend (sinal de processo).
    # [RANGER GUARD] Ranger tem sua própria contabilidade de CSL via pnl_base
    # fee-aware (net_return_pct já deduz 2×taxa) + ADX penalty no trade reward.
    # Aplicar -8.0 aqui em cima disso era a principal fonte de double-counting
    # que matava o signal: 60 trades × ~40% em loss × -8 = -192/episódio extra.
    if pnl_realized != 0 and exit_reason == 'Catastrophe Stop Loss' and not _is_ranger:
        if trade_return_pct > 0:
            reward -= 0.5  # CSL capturou lucro: penalidade leve
        else:
            reward -= 8.0  # CSL em perda: entrada errada ou sem saida a tempo

    # TRIPLE BARRIER — executa SEMPRE para Trend.
    # [RANGER GUARD] Ranger já computa o equivalente via ADX filter + MR convergence
    # + OU efficiency no _compute_trade_reward. Disparar isto aqui gerava até -4.0
    # por Stop Loss, -4.0 por Agent Decision loser e -1.5 por Time Stop em cima do
    # trade reward — double counting de ~-200/episódio sobre ~90 trades.
    # A flag trade_reward_already_computed é setada em trend_specialist.py:2517
    # logo após o _compute_trade_reward ser chamado, pulando este bloco para Ranger.
    if exit_reason and _pnl_finite and not info.get('trade_reward_already_computed', False):
        atr_pct_exit = atr / (current_price + 1e-9) if current_price > 0 else 0.01
        trade_sharpe_exit = trade_return_pct / max(atr_pct_exit, 1e-6)
        # sqrt(d/10): penaliza mais os scalps do que log1p
        # dur=1: 0.316 | dur=5: 0.707 | dur=10: 1.0 | dur=40: 2.0
        dur_mult_exit = float(np.clip(np.sqrt(duration_steps / 10.0), 0.1, 2.0))

        if exit_reason == 'Stop Loss':
            reward -= 4.0

        elif exit_reason == 'Catastrophe Stop Loss' and pnl_realized > 0:
            barrier_bonus = 1.5 * np.tanh(trade_sharpe_exit / 2.0) * dur_mult_exit
            reward += float(np.clip(barrier_bonus, 0.0, 3.0))

        elif 'Agent Decision' in exit_reason:
            if pnl_realized > 0:
                barrier_bonus = 2.0 * np.tanh(trade_sharpe_exit / 2.0) * dur_mult_exit
                reward += float(np.clip(barrier_bonus, 0.0, 4.0))
            elif pnl_realized == 0.0:
                # Break-even: custo de oportunidade fixo. O agente aprende que abrir
                # trade tem custo — fica mais criterioso nas entradas.
                reward -= 0.1
            else:
                # Aplica dur_mult_exit simetricamente para eliminar assimetria ganho/perda.
                # Antes: max win=+4.0, max loss=-2.0 (2× assimetria → Q-overestimation).
                # Agora: max loss=-4.0 (dur=10+), mesma escala dos ganhos.
                barrier_loss = 2.0 * np.tanh(abs(trade_sharpe_exit)) * dur_mult_exit
                reward -= float(np.clip(barrier_loss, 0.0, 4.0))

        elif 'Time Stop' in exit_reason:
            reward -= 1.5  # Dead money opportunity cost

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENTE 4: DELTA DRAWDOWN PENALTY (incremental, nao cumulativa)
    # ─────────────────────────────────────────────────────────────────────────
    # So penaliza quando o drawdown PIORA neste step — evita "black hole"
    # onde uma perda grande no inicio do episodio mata todo o reward futuro.
    current_dd = float(getattr(env, 'episode_max_drawdown', 0.0))
    prev_dd = float(getattr(env, '_prev_scientific_dd', 0.0))
    # [RANGER GUARD] Ranger opera em mercado lateral: drawdowns temporários de 2-5%
    # são parte natural da estratégia (o preço precisa se afastar antes de reverter).
    # Penalizar delta DD aqui ensina o agente a NÃO manter trades contra o fundo,
    # exatamente o oposto do que mean reversion precisa.
    if current_dd > prev_dd and current_dd > 0.02 and not _is_ranger:
        reward -= 5.0 * (current_dd - prev_dd)
    env._prev_scientific_dd = current_dd

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENTE 5: VOTE_MISALIGNED PENALTY DINAMICA
    # ─────────────────────────────────────────────────────────────────────────
    # Substituicao da penalidade plana de -0.40 por penalidade proporcional
    # a |predictor_confidence * predictor_direction|.
    #
    # Diagnostico [P3/P4]: o agente votava SHORT com predictor BULL (+1.0 confianca),
    # mas a penalidade de -0.40 era fraca demais para desencorajar este comportamento.
    #
    # Formula:
    #   signal_strength = |conf| * |dir|       ∈ [0, 1]
    #   penalty = -2.0 * tanh(2.0 * signal_strength)
    #
    # Tabela de exemplos:
    #   conf=1.0, dir=+1.0 vs SHORT: signal=1.0 → penalty = -2.0 * tanh(2.0) ≈ -1.93
    #   conf=0.5, dir=+0.8 vs SHORT: signal=0.4 → penalty = -2.0 * tanh(0.8) ≈ -1.19
    #   conf=0.2, dir=+0.3 vs SHORT: signal=0.06→ penalty = -2.0 * tanh(0.12)≈ -0.24
    #
    # O gate regime_mismatch (DISABLE_GRACE_BLOCK=False) permanece como seletor
    # de quando aplicar; so a MAGNITUDE mudou.
    # [RANGER GUARD] Vote misalignment não faz sentido para Ranger (bidirecional).
    # O Ranger abre LONG em oversold e SHORT em overbought — "direção certa" depende
    # do Z-score/RSI no momento, não do tp_prior_dir. Penalizar aqui ensina o agente
    # a seguir a tendência, contradizendo o propósito do mean reversion.
    if info.get('_regime_mismatch', False) and not _is_ranger:
        prior_conf_val = float(info.get('prior_conf_val', 0.5))
        prior_dir_val = float(info.get('prior_dir_val', 0.0))
        signal_strength = abs(prior_conf_val) * abs(prior_dir_val)
        mismatch_penalty = -2.0 * np.tanh(2.0 * signal_strength)
        reward += float(np.clip(mismatch_penalty, -2.0, 0.0))

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENTE 6: GRACE PERIOD EMERGENCY EXIT SIGNAL
    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostico [P5]: o grace period bloqueava saidas defensivas mas nao
    # sinalizava ao agente que ele DEVIA sair em perdas grandes.
    #
    # Logica:
    #   Se steps_in_position > grace_period AND pnl_since_entry < -2.0%:
    #     → reward POSITIVO para FLAT (incentiva a sair), negativo para manter
    #
    # Este componente inverte o sinal padrao de oportunidade: ao inves de
    # penalizar flat, RECOMPENSA flat quando a posicao esta com perda acima
    # do limiar critico. Funciona como "saida de emergencia positiva".
    grace_period_steps = int(getattr(env, 'grace_period', 5))
    pnl_since_entry = float(getattr(env, 'pnl_since_entry', 0.0))
    emergency_threshold = -0.02  # -2.0% em retorno fracionado

    # [RANGER GATE] Componente 6 DESATIVADO para Ranger.
    # Razão: em mean reversion, o preço PRECISA ir -3% a -5% antes de reverter.
    # Com threshold=-2%, este componente dispara em ~6 steps/trade × 100 trades:
    #   -3.0 × tanh(0.01/0.02) ≈ -1.44/step × 600 steps = -864/episódio
    # Isso dominava todo o sinal de reward e tornava o HPO inviável.
    # O Ranger gerencia drawdown via: Stop Loss fixo + _should_exit_fast + OU efficiency.
    if (not _is_ranger
            and env.position != 0
            and steps_in_pos > grace_period_steps
            and pnl_since_entry < emergency_threshold):
        # Quanto mais negativo o pnl alem do threshold, mais forte o sinal
        excess_loss = abs(pnl_since_entry - emergency_threshold)  # positivo
        # Sinal positivo para FLAT (o agente precisa aprender que sair e bom aqui)
        # Implementado como penalidade por MANTER posicao perdedora pos-grace
        emergency_penalty = -3.0 * np.tanh(excess_loss / 0.02)
        reward += float(np.clip(emergency_penalty, -3.0, 0.0))

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENTE 7: SORTINO RATIO REWARD (Sortino & van der Meer, 1991)
    # ─────────────────────────────────────────────────────────────────────────
    # Penaliza assimetricamente returns negativos vs positivos.
    # O Sortino Ratio trata somente o downside deviation como "risco real",
    # ao contrario do Sharpe que penaliza tanto upside quanto downside volatility.
    #
    # Aplicado no fechamento do trade:
    #   - return positivo: bonus suave (nao penaliza volatilidade positiva)
    #   - return negativo: penalidade proporcional ao downside quadratico
    #
    # Formula:
    #   sortino_component = tanh(trade_return_pct / max_downside)
    #   max_downside = max(|negative_return|, epsilon)  [so o downside conta]
    #
    # COMPONENTE 7: SORTINO RATIO REWARD (Sortino & van der Meer, 1991)
    # ─────────────────────────────────────────────────────────────────────────
    # [RANGER GUARD] Se _compute_trade_reward() já foi chamado (Ranger), não fazer
    # double-counting de Sortino — o OU efficiency do Ranger já cobre assimetria downside.
    _trade_already_computed = info.get('trade_reward_already_computed', False)
    if trade_closed_this_step and np.isfinite(trade_return_pct) and not _trade_already_computed:
        if trade_return_pct >= 0:
            sortino_up = 0.5 * np.tanh(trade_return_pct / max(atr_pct, 1e-6))
            reward += float(np.clip(sortino_up, 0.0, 1.0))
        else:
            downside_sq = (abs(trade_return_pct) ** 2) / max(atr_pct ** 2, 1e-10)
            sortino_down = -1.5 * np.tanh(downside_sq)
            reward += float(np.clip(sortino_down, -2.0, 0.0))

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENTE 8: CALMAR RATIO COMPONENT (Young, 1991)
    # ─────────────────────────────────────────────────────────────────────────
    # Calmar Ratio = CAGR / Max Drawdown — mede qualidade do sistema inteiro.
    # Aqui usado como bonus/penalidade local: bonifica quando o retorno do trade
    # e bom em relacao ao drawdown maximo do episodio.
    #
    # Aplicado no fechamento de trades positivos com drawdown nao-nulo:
    #   calmar_local = trade_return_pct / episode_max_drawdown
    #   bonus = 0.8 * tanh(calmar_local / 5.0)   [satura em ≈ +0.8]
    #
    # Nao aplicado em trades negativos (o Sortino e o Delta DD ja cobrem isso).
    # [RANGER GUARD] Idem ao Sortino — Ranger usa OU efficiency como proxy de Calmar.
    if trade_closed_this_step and trade_return_pct > 0 and current_dd > 0.005 and not _trade_already_computed:
        calmar_local = trade_return_pct / current_dd
        calmar_bonus = 0.8 * np.tanh(calmar_local / 5.0)
        reward += float(np.clip(calmar_bonus, 0.0, 0.8))

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENTE 9: MINIMUM HOLD REWARD
    # ─────────────────────────────────────────────────────────────────────────
    # Bonus pequeno para trades que superam a duracao minima de 3 steps.
    # Escalado proporcionalmente ao retorno em ATR para evitar que trades com
    # PnL~0 recebam bonus positivo (ensinaria que não ganhar nada é bom).
    #
    # quality = clip(trade_return_pct / atr_pct, 0, 1):
    #   PnL = 0       → quality=0.0 → bonus=0.00  ✓
    #   PnL = 0.5 ATR → quality=0.5 → bonus=0.25 (meio bonus)
    #   PnL >= 1 ATR  → quality=1.0 → bonus cheio ✓
    #   PnL < 0       → quality=0.0 → bonus=0.00  ✓
    #
    #   duration >= 3  → +0.30 × quality
    #   duration >= 5  → +0.50 × quality
    #   duration >= 15 → +0.70 × quality
    #
    if trade_closed_this_step and not _trade_already_computed:
        if not _is_ranger:
            # ── Trend (Bull/Bear): Minimum Hold clássico ─────────────────────
            if duration_steps >= 15:
                hold_completion_bonus = 0.70
            elif duration_steps >= 5:
                hold_completion_bonus = 0.50
            elif duration_steps >= 3:
                hold_completion_bonus = 0.30
            else:
                hold_completion_bonus = 0.0
            if hold_completion_bonus > 0:
                quality = float(np.clip(trade_return_pct / max(atr_pct, 1e-6), 0.0, 1.0))
                reward += hold_completion_bonus * quality
        else:
            # ── Ranger: OU Half-Life Completion Bonus ────────────────────────
            # [RANGER GUARD] Ranger usa _compute_trade_reward() separado que já inclui
            # OU efficiency. Este bloco nunca deve executar para Ranger.
            # Mantido apenas como fallback (caso is_ranger seja False erroneamente).
            half_hl = max(1, _mr_halflife // 2)
            if duration_steps < 3:
                hold_completion_bonus = 0.0
            elif duration_steps < half_hl:
                hold_completion_bonus = 0.15
            elif duration_steps <= _mr_halflife:
                hold_completion_bonus = 0.50
            elif duration_steps <= int(1.5 * _mr_halflife):
                hold_completion_bonus = 0.30
            elif duration_steps <= 2 * _mr_halflife:
                hold_completion_bonus = 0.10
            else:
                hold_completion_bonus = 0.0
            if hold_completion_bonus > 0:
                quality = float(np.clip(trade_return_pct / max(atr_pct, 1e-6), 0.0, 1.0))
                reward += hold_completion_bonus * quality

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENTE 10: OPPORTUNITY COST (evita policy collapse para FLAT)
    # ─────────────────────────────────────────────────────────────────────────
    # Nota [BUG FIX C]: Opportunity Cost nao dispara no step de fechamento.
    # O step de fechamento encontra position=0 mas trade_closed=True, portanto
    # nao penaliza o reward do bom trade que acabou de ser fechado.
    if env.position == 0 and not trade_closed_this_step:
        prior_dir = info.get('prior_dir_val', 0.0)
        predictor_conf = info.get('prior_conf_val', 0.0)

        if 'bull' in _specialist_name:
            predictor_strength = max(0.0, prior_dir)
        elif 'bear' in _specialist_name:
            predictor_strength = max(0.0, -prior_dir)
        else:
            # Ranger e genérico: ambas as direções têm oportunidade
            predictor_strength = abs(prior_dir)

        # [RANGER GUARD] Flat base penalty reduzida para Ranger.
        # Mean reversion exige PACIÊNCIA: esperar o squeeze + RSI extremo pode levar
        # 50-200 candles flat. -0.01 × 200 = -2.0/episódio não é dramático, mas
        # -0.003 é suficiente como pressão leve sem dominar o sinal.
        reward -= 0.003 if _is_ranger else 0.01  # flat base penalty por estar flat

        if not _is_ranger:
            # Trend: penaliza flat quando sinal forte existe
            if predictor_strength > 0.1 and predictor_conf > 0.3:
                signal_magnitude = predictor_strength * predictor_conf
                reward -= 0.10 * np.tanh(signal_magnitude * 3.0)
        else:
            # Ranger: penaliza flat quando range quality é alta E há extremo de RSI/Z-score
            # (o agente deve entrar quando há sinal real de reversão, não ficar parado)
            try:
                row = getattr(env, 'current_row', None)
                if row is not None:
                    _get_r = lambda names, dft: next(
                        (float(v) for n in names
                         for v in [row.get(n) if hasattr(row, 'get') else None]
                         if v is not None and np.isfinite(float(v))),
                        dft
                    )
                    adx_now = _get_r(['adx'], 25.0)
                    bb_width_now = _get_r(['bb_width', 'bbw'], 0.025)
                    rsi_now = _get_r(['rsi_1h', 'rsi_4h', 'rsi_15m', 'rsi'], 50.0)
                    # Range quality: lateral confirmado + squeeze + RSI extremo = sinal forte
                    range_quality = 0.0
                    if adx_now < 22.0:
                        range_quality += 0.4
                    if bb_width_now < 0.020:
                        range_quality += 0.3
                    if rsi_now < 35.0 or rsi_now > 65.0:
                        range_quality += 0.3
                    if range_quality > 0.5 and predictor_conf > 0.2:
                        # Alta qualidade de sinal de range e flat → penalizar
                        reward -= 0.08 * np.tanh(range_quality * 2.0)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENTE 11: CUSTO DE ENTRADA (friccao transacional)
    # ─────────────────────────────────────────────────────────────────────────
    # Nota [FIX H v2]: Condicao corrigida para steps_in_position == 0 AND position != 0.
    # Garante disparo apenas no step de ABERTURA (position recen-setada, ainda
    # nao incrementou). 1-step trades: dispara no step de abertura corretamente.
    #
    # Ranger: DESABILITADO — a dedução de taxa no pnl_base já cobre o custo transacional.
    # Formula: net_return = gross_return - 2×TAKER_FEE×leverage
    # Aplicar -0.8 adicional causaria dupla contagem e tornaria todos os Q-values negativos,
    # impedindo convergência (actor_loss diverge para 100+ porque Q(s,a) < 0 sempre).
    if getattr(env, 'steps_in_position', -1) == 0 and env.position != 0 and not _is_ranger:
        reward -= 0.8

    # ─────────────────────────────────────────────────────────────────────────
    # FINALIZACAO: CQL Clipping implicito via clip final
    # ─────────────────────────────────────────────────────────────────────────
    # Ref: Kumar et al. (2020) CQL — Conservative Q-Learning.
    # O reward clipping em [-20, 20] atua como regularizador do target Q-value,
    # contendo a explosao de Q-values observada no log (actor loss → -99.7).
    # Range reduzido de [-25, 25] (v3.0) para [-20, 20] (v4.0) para maior
    # conservadorismo consistente com o diagnostico de Q-overestimation.
    reward_clipped = float(np.clip(reward, -20.0, 20.0))

    # ── ZERO-REWARD GUARD ────────────────────────────────────────────────────
    # Princípio: reward=0 NUNCA deve ser ponto fixo neutro.
    # "Não fazer nada" tem custo de oportunidade — o agente precisa aprender isso.
    #
    # Casos em que reward_clipped == 0.0:
    #   a) Flat num step normal sem sinal → deveria ser penalizado (Comp 10 já trata,
    #      mas pode zerar por outros componentes se todos se cancelam).
    #   b) Trade break-even exato (pnl=0) → custo de oportunidade + fricção transacional.
    #   c) Step neutro (sem posição, sem sinal) → inação tem custo.
    #
    # Solução: aplicar penalidade fixa de -0.50 quando reward final == 0.0 exato,
    # EXCETO se um trade acabou de ser fechado (exit_reason não é nulo). Quando um 
    # trade fecha, é normal o SciRwd ser 0.0 (pois o TradeRwd já lidou com os bônus).
    if reward_clipped == 0.0 and not exit_reason:
        reward_clipped = -0.50

    return reward_clipped


def get_attention_mlp_policy_kwargs(
    features_dim: int = 128,
    num_heads: int = 8,
) -> Dict[str, Any]:
    """
    Attention-MLP Extractor — substitui a "Placebo LSTM" (seq_len=1 sem memoria real).

    Args:
        features_dim: Dimensao do espaco de features extraido (default 128 para
                      treino final, recomenda-se 64 para HPO).
        num_heads:    Numero de cabecas de atencao multi-head (default 8 para
                      treino final, recomenda-se 4 para HPO mais rapido).
    """
    from learning.custom_extractors import AttentionMLPExtractor
    import torch.nn as _nn

    # Garante que net_arch seja coerente com features_dim
    hidden = max(64, features_dim * 2)
    policy_kwargs = {
        'features_extractor_class': AttentionMLPExtractor,
        'features_extractor_kwargs': {
            'features_dim': features_dim,
            'num_heads': num_heads,
            'dropout': 0.1,
        },
        'net_arch': {
            'pi': [hidden, hidden // 2],
            'qf': [hidden, hidden // 2],
        },
        'activation_fn': _nn.GELU,
    }
    return policy_kwargs


def get_hybrid_lstm_policy_kwargs(
    n_stack: int = 4,
    features_dim: int = 128,
    lstm_hidden_size: int = 64,
    num_heads: int = 4,
    dropout: float = 0.15,
) -> Dict[str, Any]:
    """
    HybridLSTMAttentionExtractor com VecFrameStack real.

    Requer que o ambiente tenha sido criado com VecFrameStack(n_stack=n_stack).
    O extractor faz reshape das observações empilhadas:
        (batch, obs_dim * n_stack) → (batch, n_stack, obs_dim)
    alimentando o LSTM com uma sequência temporal real de n_stack candles.

    Pipeline completo:
        Candle t-3 → Candle t-2 → Candle t-1 → Candle t
              ↓           ↓           ↓           ↓
              [      LSTM processa sequência temporal      ]
                          ↓
              [  Attention pondera timesteps relevantes  ]
                          ↓
              [       SAC decide ação                   ]

    Args:
        n_stack:          Número de candles empilhados (deve bater com VecFrameStack)
        features_dim:     Dimensão do vetor de saída para a policy/value network
        lstm_hidden_size: Tamanho do hidden state do LSTM
        num_heads:        Cabeças de atenção multi-head
        dropout:          Dropout para regularização
    """
    from learning.custom_extractors import HybridLSTMAttentionExtractor
    import torch.nn as _nn

    hidden = max(64, features_dim * 2)
    policy_kwargs = {
        'features_extractor_class': HybridLSTMAttentionExtractor,
        'features_extractor_kwargs': {
            'n_stack': n_stack,
            'features_dim': features_dim,
            'lstm_hidden_size': lstm_hidden_size,
            'num_heads': num_heads,
            'dropout': dropout,
        },
        'net_arch': {
            'pi': [hidden, hidden // 2],
            'qf': [hidden, hidden // 2],
        },
        'activation_fn': _nn.GELU,
    }
    return policy_kwargs


def get_mlp_policy_kwargs() -> Dict[str, Any]:
    """MLP simples para Optuna HPO (4x mais rapido que LSTM)."""
    import torch.nn as _nn
    policy_kwargs = {
        'net_arch': {
            'pi': [128, 64],
            'qf': [128, 64],
        },
        'activation_fn': _nn.ReLU,
    }
    return policy_kwargs


# FLAGS DE CONFIGURACAO
DISABLE_PRIOR_VETO = True
DISABLE_REVERSAL_BLOCK = True
DISABLE_GRACE_BLOCK = False


def should_apply_gate_bypass(gate_type: str) -> bool:
    gates_config = {
        "prior_veto": DISABLE_PRIOR_VETO,
        "reversal_block": DISABLE_REVERSAL_BLOCK,
        "grace_period": DISABLE_GRACE_BLOCK,
    }
    return gates_config.get(gate_type, False)


def apply_temperature_scaling(logits: np.ndarray, temperature: float = 1.5) -> np.ndarray:
    """Temperature Scaling para calibrar probabilidades (Guo et al. 2017)."""
    if np.all(logits >= 0) and np.isclose(np.sum(logits), 1.0, atol=1e-3):
        logits = np.log(logits + 1e-10)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)
