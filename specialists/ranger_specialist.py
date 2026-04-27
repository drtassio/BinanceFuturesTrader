"""
Especialista 'O Ranger' (RangerSpecialist) — v2.0
Focado em mercados laterais (sideways) e reversão à média.
Opera em ambas as direções, comprando suporte e vendendo resistência.

ARQUITETURA DE REWARD (5 componentes + fee-awareness):
  1. PnL base (NET)  — retorno líquido = gross − 2×TAKER_FEE×leverage
                        micro-trades abaixo do custo de taxa → reward negativo
  2. Filtro ADX      — penaliza operar em mercado tendencioso (ADX > 25)
  3. BB Squeeze      — bonifica entrada em range de baixa volatilidade
  4. MR Convergência — bonifica fechamento próximo à EMA (reversão concluída)
  5. Eficiência OU   — Ornstein-Uhlenbeck: premia saída dentro do half-life

REFERÊNCIAS:
  - Bollinger (1992): BB Width como proxy de volatilidade comprimida
  - Wilder (1978): ADX/RSI para identificar força e momentum
  - Uhlenbeck & Ornstein (1930): Half-life de processos de reversão
  - Prado (2018): Meta-labeling e filtros de confiança para qualidade de sinal
  - Chan (2013): Pairs trading e estratégias de mean reversion
"""

from typing import Dict, Any, Optional
import optuna
import pandas as pd
import numpy as np

from specialists.base_regime_specialist import BaseRegimeSpecialist
from specialists.trend_specialist import TrendFollowingEnv
from config.settings import AIConfig, TradingConfig
from utils.logger import get_logger

logger = get_logger("RangerSpecialist")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE REGIME LATERAL
# ─────────────────────────────────────────────────────────────────────────────
_ADX_RANGE_MAX      = 22.0   # ADX abaixo disso = mercado lateral confirmado
_ADX_TREND_WARN     = 28.0   # ADX acima disso = tendência emergindo
_ADX_TREND_STRONG   = 35.0   # ADX acima disso = tendência forte (penalidade máxima)
_BBW_SQUEEZE        = 0.018  # BB Width abaixo disso = squeeze de volatilidade
_BBW_NORMAL         = 0.035  # BB Width acima disso = range normal
_ZSCORE_OVERSOLD    = -1.8   # Z-score de entrada compra (calibrado para crypto)
_ZSCORE_OVERBOUGHT  =  1.8   # Z-score de entrada venda
_RSI_OVERSOLD       =  32    # RSI sobrevendido (mais conservador que 30 padrão)
_RSI_OVERBOUGHT     =  68    # RSI sobrecomprado
_MR_HALFLIFE        = 15     # Half-life típico de reversão em crypto (candles 15m)
_DIST_EMA_IDEAL     = 0.40   # Dist ATR ideal ao fechar (reversão quase completa)
_DIST_EMA_GOOD      = 0.80   # Dist ATR boa ao fechar


class RangerSpecialist(BaseRegimeSpecialist):
    """
    Especialista em mercados laterais com mean reversion matemática.

    Estratégia central:
      - Comprar quando preço toca suporte (Z < -1.8, RSI < 32) → esperar EMA
      - Vender quando preço toca resistência (Z > +1.8, RSI > 68) → esperar EMA
      - Sair rapidamente: trades curtos (< 1x half-life) são mais rentáveis em range

    Features de detecção de range:
      - ADX < 22: confirma mercado não-tendencioso
      - BB Width < 0.018: squeeze de volatilidade (melhor momento para range trade)
      - RSI extremo + Z-score extremo: dupla confirmação de reversão

    Treina exclusivamente com dados de regime Ranger (regime_code=2).
    """

    def __init__(
        self,
        config: AIConfig,
        trading_config: TradingConfig,
        input_dim: int,
        profitability_predictor: Optional[Any] = None
    ):
        super().__init__(
            config=config,
            trading_config=trading_config,
            regime_type='ranger',
            input_dim=input_dim,
            profitability_predictor=profitability_predictor
        )

        # Parâmetros de mean reversion (sinais matemáticos)
        self.bb_window           = 20          # Janela Bollinger Band
        self.bb_std              = 2.0         # Desvios padrão para as bandas
        self.rsi_period          = 14          # Período RSI (Wilder, 1978)
        self.z_score_threshold   = abs(_ZSCORE_OVERSOLD)
        self.rsi_overbought      = _RSI_OVERBOUGHT
        self.rsi_oversold        = _RSI_OVERSOLD

        # Hiperparâmetros de modelo específicos do Ranger
        self.model_hyperparams = getattr(self, "model_hyperparams", {})
        self.model_hyperparams.update({
            "mean_reversion_strength": 1.5,
            "min_confidence": 0.55,
            "position_bias": "bidirectional",
            "take_profit_sensitivity": 1.2,
        })

        logger.info(
            "🤠 RangerSpecialist v2.0 — Mean Reversion com ADX/BBW/Z-score/OU half-life"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SINAIS DE MEAN REVERSION (para inferência / get_action)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_mean_reversion_signal(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Sinais matemáticos para reversão à média.

        Retorna:
          z_score             : Distância da média em desvios-padrão (Bollinger)
          rsi                 : Oscilador de momentum [0-100]
          price_position      : Posição [0=fundo, 1=topo] dentro do range recente
          adx                 : Força da tendência [0-100] (ADX < 22 = lateral)
          bb_width            : Largura da Bollinger Band (squeeze < 0.018)
          range_quality_score : Score [0-1] de qualidade do range atual
        """
        defaults = {
            'z_score': 0.0, 'rsi': 50.0, 'price_position_in_range': 0.5,
            'adx': 25.0, 'bb_width': 0.02, 'range_quality_score': 0.5,
        }
        if len(df) < self.bb_window:
            return defaults

        try:
            close = df['close']
            high  = df.get('high', close)
            low   = df.get('low', close)

            # 1. Bollinger Z-Score
            sma    = close.rolling(self.bb_window).mean()
            std    = close.rolling(self.bb_window).std()
            z_score = float((close - sma).iloc[-1] / (std.iloc[-1] + 1e-9))

            # BB Width (proxy de squeeze)
            bb_upper = sma + self.bb_std * std
            bb_lower = sma - self.bb_std * std
            bb_width = float(
                (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / (sma.iloc[-1] + 1e-9)
            )

            # 2. RSI (Wilder, 1978)
            delta = close.diff()
            gain  = delta.where(delta > 0, 0.0).rolling(self.rsi_period).mean()
            loss  = (-delta.where(delta < 0, 0.0)).rolling(self.rsi_period).mean()
            rs    = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
            rsi   = float(100.0 - 100.0 / (1.0 + rs))

            # 3. Posição no range
            r_high = close.rolling(self.bb_window).max().iloc[-1]
            r_low  = close.rolling(self.bb_window).min().iloc[-1]
            price_pos = float((close.iloc[-1] - r_low) / (r_high - r_low + 1e-9))

            # 4. ADX (se já calculado no DF, usar; senão estimar)
            if 'adx' in df.columns:
                adx_val = float(df['adx'].iloc[-1])
            else:
                # Aproximação rápida: desvio padrão de retornos anualizados
                returns_std = float(close.pct_change().rolling(14).std().iloc[-1])
                adx_val = min(50.0, returns_std * 200)
            if np.isnan(adx_val):
                adx_val = 25.0

            # 5. Range Quality Score — combina ADX + BBW + RSI extremidade
            adx_score = max(0.0, 1.0 - adx_val / 40.0)             # 1.0 se ADX=0, 0.0 se ADX≥40
            bbw_score = max(0.0, 1.0 - bb_width / _BBW_NORMAL)     # 1.0 se squeeze, 0 se amplo
            rsi_score = abs(rsi - 50.0) / 50.0                      # 1.0 se RSI extremo
            range_quality = float(
                0.45 * adx_score + 0.35 * bbw_score + 0.20 * rsi_score
            )

            return {
                'z_score': z_score if not np.isnan(z_score) else 0.0,
                'rsi': rsi if not np.isnan(rsi) else 50.0,
                'price_position_in_range': price_pos if not np.isnan(price_pos) else 0.5,
                'adx': adx_val,
                'bb_width': bb_width if not np.isnan(bb_width) else 0.02,
                'range_quality_score': range_quality,
            }

        except Exception as exc:
            logger.warning("⚠️ Mean reversion signal falhou: %s", exc)
            return defaults

    # ─────────────────────────────────────────────────────────────────────────
    # AÇÃO (inferência — não afeta SAC durante treino)
    # ─────────────────────────────────────────────────────────────────────────

    def get_action(self, state: np.ndarray, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ação baseada em sinais de mean reversion matemáticos.

        Filtro ADX: não operar se ADX > _ADX_TREND_WARN (mercado em tendência).
        Sinal duplo: Z-score extremo + RSI extremo = confirmação de reversão.
        """
        action_dict = super().get_action(state, context)

        context = context or {}
        if 'df' in context:
            mr_signal = self.compute_mean_reversion_signal(context['df'])
            context.update(mr_signal)

        z_score       = context.get('z_score', 0.0)
        rsi           = context.get('rsi', 50.0)
        price_pos     = context.get('price_position_in_range', 0.5)
        adx_val       = context.get('adx', 25.0)
        range_quality = context.get('range_quality_score', 0.5)

        # FILTRO ADX: não operar em mercado tendencioso
        if adx_val > _ADX_TREND_WARN:
            action_dict['confidence'] = action_dict.get('confidence', 0.5) * 0.5
            logger.debug("🤠 Ranger: ADX=%.1f > %.0f — reduzindo confiança (mercado em tendência)", adx_val, _ADX_TREND_WARN)
            return action_dict

        # Sinais de reversão com dupla confirmação
        oversold   = z_score < _ZSCORE_OVERSOLD  and rsi < _RSI_OVERSOLD
        overbought = z_score > _ZSCORE_OVERBOUGHT and rsi > _RSI_OVERBOUGHT

        if oversold:
            action_dict['side'] = 1   # Long
            signal_strength = min(1.0, (
                abs(z_score) / 3.5 +
                (_RSI_OVERSOLD - rsi) / 35.0 +
                range_quality
            ) / 3.0)
            action_dict['confidence'] = max(0.6, min(1.0, signal_strength))
            logger.debug("🤠 Ranger LONG: Z=%.2f RSI=%.1f ADX=%.1f RQ=%.2f", z_score, rsi, adx_val, range_quality)

        elif overbought:
            action_dict['side'] = -1  # Short
            signal_strength = min(1.0, (
                abs(z_score) / 3.5 +
                (rsi - _RSI_OVERBOUGHT) / 35.0 +
                range_quality
            ) / 3.0)
            action_dict['confidence'] = max(0.6, min(1.0, signal_strength))
            logger.debug("🤠 Ranger SHORT: Z=%.2f RSI=%.1f ADX=%.1f RQ=%.2f", z_score, rsi, adx_val, range_quality)

        else:
            # Ajuste por posição extrema no range (sem reversão confirmada)
            if price_pos < 0.15 and action_dict.get('side') == 'buy':
                action_dict['confidence'] = min(1.0, action_dict.get('confidence', 0.5) * 1.15)
            elif price_pos > 0.85 and action_dict.get('side') == 'sell':
                action_dict['confidence'] = min(1.0, action_dict.get('confidence', 0.5) * 1.15)

        return action_dict

    # ─────────────────────────────────────────────────────────────────────────
    # HPO — ESPAÇO DE BUSCA ESPECÍFICO DO RANGER
    # ─────────────────────────────────────────────────────────────────────────

    def _suggest_reward_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Override do espaço HPO para parâmetros de reward do Ranger.

        Diferenças vs TrendSpecialist:
          - DURATION_PRESSURE_START menor: trades curtos (15-50 vs 60-180)
          - DURATION_PRESSURE_WEIGHT menor: menos pressão por tempo (0.15-0.45)
          - DRAWDOWN_PENALTY maior: range trades não devem ter grandes drawdowns
        """
        reward_params: Dict[str, Any] = {}

        # Penalidade de drawdown — mais forte no Ranger (sem grandes oscilações)
        reward_params["TREND_DRAWDOWN_PENALTY_WEIGHT"] = trial.suggest_float(
            "TREND_DRAWDOWN_PENALTY_WEIGHT", 0.30, 1.5
        )
        # Penalidade de flip de prior — moderada (Ranger é bidirecional)
        reward_params["TREND_PRIOR_FLIP_PENALTY_WEIGHT"] = trial.suggest_float(
            "TREND_PRIOR_FLIP_PENALTY_WEIGHT", 0.10, 0.80
        )
        reward_params["TREND_PRIOR_FLIP_PENALTY_EXP"] = trial.suggest_float(
            "TREND_PRIOR_FLIP_PENALTY_EXP", 0.5, 1.3
        )
        reward_params["TREND_PRIOR_FLIP_THRESHOLD"] = trial.suggest_float(
            "TREND_PRIOR_FLIP_THRESHOLD", 0.25, 0.55
        )
        # Duration pressure MUITO menor: range trades são curtos por natureza
        # Mínimo elevado (0.20) garante que o agente sempre fecha posições
        reward_params["TREND_DURATION_PRESSURE_WEIGHT"] = trial.suggest_float(
            "TREND_DURATION_PRESSURE_WEIGHT", 0.20, 0.50
        )
        # Pressure começa cedo: mean reversion resolve em 15-40 candles
        reward_params["TREND_DURATION_PRESSURE_START"] = trial.suggest_int(
            "TREND_DURATION_PRESSURE_START", 15, 55
        )
        reward_params["TREND_DURATION_PRESSURE_EXP"] = trial.suggest_float(
            "TREND_DURATION_PRESSURE_EXP", 0.8, 1.4
        )
        return reward_params


# ─────────────────────────────────────────────────────────────────────────────
# AMBIENTE RL — RangerTradingEnv
# ─────────────────────────────────────────────────────────────────────────────

class RangerTradingEnv(TrendFollowingEnv):
    """
    Ambiente RL especializado para o Ranger (mercado lateral / mean reversion).

    Reward de 5 componentes (todos calculados no fechamento do trade):
      1. PnL Base          — retorno escalado sem amplificação de duração longa
      2. Filtro ADX        — penaliza trades abertos em mercado tendencioso
      3. BB Squeeze Bonus  — bonifica entrada em período de baixa volatilidade
      4. MR Convergência   — bonifica fechamento próximo à EMA (reversão real)
      5. Eficiência OU     — recompensa saída dentro do half-life de Ornstein-Uhlenbeck

    Saída dinâmica (_should_exit_fast):
      - Sai mais cedo quando ADX sobe (breakout iminente)
      - Sai quando EMA foi alcançada (reversão completa)
      - Sai após 2x half-life com qualquer lucro
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Half-life de Ornstein-Uhlenbeck para crypto range (candles 15m)
        self.mean_reversion_halflife  = _MR_HALFLIFE
        self.max_duration_multiplier  = 2.5   # Forçar saída em 2.5x half-life

        # Colunas de indicadores pré-calculados no DataFrame
        self.adx_col_ranger   = 'adx'        # Calculado em native_indicators.py
        self.bbw_col_ranger   = 'bb_width'   # Calculado em native_indicators.py
        self.bb_upper_col     = 'bb_upper'
        self.bb_lower_col     = 'bb_lower'
        self.bb_middle_col    = 'bb_middle'

        # [FIX RANGER GATE] Permite Long E Short — direção neutra é condição ideal
        # Quando tp_prior_dir ≈ 0 (lateral), o gate do parent fecha tudo.
        # Esta flag abre os gates especificamente para o Ranger.
        self._allow_both_directions = True

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITÁRIOS DE LEITURA DO DF
    # ─────────────────────────────────────────────────────────────────────────

    def _get_row_safe(self, step_offset: int = 0) -> pd.Series:
        """Retorna linha do DataFrame com bounds checking."""
        idx = min(
            max(0, self.start_idx + self.current_step + step_offset),
            len(self.df) - 1
        )
        return self.df.iloc[idx]

    def _get_entry_row(self, duration_steps: int) -> pd.Series:
        """
        Retorna a linha do DataFrame no momento de ABERTURA do trade.
        Usa back-calculation: entry_step = current_step - duration_steps.
        """
        entry_offset = -int(duration_steps)
        return self._get_row_safe(entry_offset)

    def _get_adx(self, row: pd.Series, default: float = 25.0) -> float:
        """Lê ADX com fallback seguro."""
        try:
            val = float(row.get(self.adx_col_ranger, default))
            return val if not np.isnan(val) else default
        except Exception:
            return default

    def _get_bbw(self, row: pd.Series, default: float = 0.025) -> float:
        """Lê BB Width com fallback seguro."""
        try:
            val = float(row.get(self.bbw_col_ranger, default))
            return val if not np.isnan(val) else default
        except Exception:
            return default

    def _get_atr_safe(self, row: pd.Series, price: float) -> float:
        """Lê ATR com fallback para 1% do preço."""
        try:
            val = float(row.get(self.atr_col, price * 0.01))
            if np.isnan(val) or val <= 1e-9:
                return price * 0.01
            return val
        except Exception:
            return price * 0.01

    # ─────────────────────────────────────────────────────────────────────────
    # FUNÇÃO DE REWARD — NÚCLEO DO RANGER
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_trade_reward(
        self,
        pnl_realized: float,
        trade_return_pct: float,
        duration_steps: int,
        exit_reason,
        atr_pct: float,
    ) -> float:
        """
        Reward de 5 componentes para mean reversion (taxa-consciente).

        Componentes:
        ┌─────────────────────────────┬───────────────────────────────────────┐
        │ Componente                  │ Efeito                                │
        ├─────────────────────────────┼───────────────────────────────────────┤
        │ 1. PnL Base (NET de taxas)  │ retorno_bruto − 2×TAKER_FEE×leverage  │
        │    → taxa de round-trip     │ Micro-trades q/ não cobrem fee → neg  │
        │ 2. Filtro ADX (entrada)     │ Penaliza trade em mercado tendencioso │
        │ 3. BB Squeeze (entrada)     │ Bonifica entrada em squeeze           │
        │ 4. MR Convergência (saída)  │ Bonifica fechamento próximo à EMA     │
        │ 5. Eficiência OU (duração)  │ Premia saída dentro do half-life      │
        └─────────────────────────────┴───────────────────────────────────────┘

        Fee deduction formula:
            roundtrip_fee_pct = 2 × TAKER_FEE × current_leverage
            net_return_pct    = trade_return_pct − roundtrip_fee_pct
            pnl_base          = net_return_pct × 20.0

        Rationale: trade_return_pct já inclui alavancagem (pnl/margem),
        as taxas incidem sobre o notional = margem × leverage. Logo a
        deducão de fee_pct precisa ser escalada por leverage também.
        """
        try:
            exit_row  = self._get_row_safe()
            entry_row = self._get_entry_row(duration_steps)

            exit_price  = float(exit_row.get('close', 0.0))
            entry_price = float(entry_row.get('close', exit_price))
            if exit_price <= 0:
                exit_price = entry_price if entry_price > 0 else 1.0

            # ── 1. PnL BASE (taxa-consciente) ────────────────────────────────
            # trade_return_pct = pnl_bruto / margem = retorno_preço × leverage (BRUTO)
            # Taxas são cobradas sobre o notional: TAKER_FEE × notional por lado.
            # Como % da margem:  fee_pct = TAKER_FEE × leverage  (por lado)
            # Round-trip (entrada + saída): 2 × TAKER_FEE × leverage
            #
            # Exemplo com TAKER_FEE=0.001, leverage=5:
            #   roundtrip_fee_pct = 2 × 0.001 × 5 = 0.010  (1% da margem)
            #   Trade bruto de +0.8% (margin) → net = −0.2% → pnl_base negativo ✓
            #   Trade bruto de +1.5% (margin) → net = +0.5% → pnl_base positivo ✓
            #
            # Isso evita que o agente aprenda a fazer micro-trades rentáveis no gross
            # mas perdedores no líquido, que é o maior risco em mercados laterais.
            fee_rate          = getattr(self.trading_config, 'TAKER_FEE', 0.001)
            leverage          = float(getattr(self, 'current_leverage', 5.0))
            roundtrip_fee_pct = 2.0 * fee_rate * leverage
            net_return_pct    = float(trade_return_pct) - roundtrip_fee_pct
            pnl_base          = net_return_pct * 20.0
            # (trades que não cobrem as taxas → reward negativo mesmo com gross > 0)

            # ── 2. FILTRO ADX (baseado no estado no MOMENTO DA ENTRADA) ──────
            # Se o agente entrou quando o ADX estava alto → penalizar
            # (ensina a SÓ operar em mercado realmente lateral)
            adx_entry = self._get_adx(entry_row)
            adx_exit  = self._get_adx(exit_row)

            adx_penalty = 0.0
            if adx_entry > _ADX_TREND_STRONG:
                # Mercado estava em tendência forte na entrada: penalidade máxima
                adx_penalty = -2.5
            elif adx_entry > _ADX_TREND_WARN:
                # Tendência emergindo: penalidade proporcional
                adx_penalty = -1.5 * ((adx_entry - _ADX_TREND_WARN) / (_ADX_TREND_STRONG - _ADX_TREND_WARN))
            elif adx_entry < _ADX_RANGE_MAX:
                # Mercado lateral confirmado na entrada: pequeno bônus
                adx_penalty = +0.4 * (1.0 - adx_entry / _ADX_RANGE_MAX)

            # Se ADX acelerou DURANTE o trade (breakout após entrada) → penalidade extra
            if adx_exit > adx_entry + 8.0 and pnl_realized < 0:
                adx_penalty -= 0.5  # O breakout causou a perda

            # ── 3. BB SQUEEZE BONUS (estado na ENTRADA) ──────────────────────
            # Entrada em squeeze = melhor condição para range trade
            bbw_entry = self._get_bbw(entry_row)

            bb_bonus = 0.0
            if bbw_entry < _BBW_SQUEEZE:
                # Squeeze forte: máximo bônus de range quality
                bb_bonus = 1.5 * (1.0 - bbw_entry / _BBW_SQUEEZE)
            elif bbw_entry < _BBW_NORMAL:
                # Range normal: bônus pequeno
                bb_bonus = 0.5 * (1.0 - bbw_entry / _BBW_NORMAL)
            # BBW alto = mercado em expansão → sem bônus (não é o momento de range trade)

            # ── 4. MR CONVERGÊNCIA (estado na SAÍDA) ─────────────────────────
            # A essência do mean reversion: o preço RETORNOU à média?
            ema_val    = self._get_ema_trend_value(exit_row)
            atr_exit   = self._get_atr_safe(exit_row, exit_price)
            dist_exit  = abs(exit_price - ema_val) / (atr_exit + 1e-9) if ema_val > 0 else 1.5

            mr_convergence = 0.0
            if pnl_realized > 0:
                if dist_exit < _DIST_EMA_IDEAL:
                    # Reversão quase perfeita: fechou muito próximo da EMA
                    mr_convergence = 3.0 * (1.0 - dist_exit / _DIST_EMA_IDEAL)
                elif dist_exit < _DIST_EMA_GOOD:
                    # Boa reversão: fechou razoavelmente próximo
                    mr_convergence = 1.2 * (1.0 - (dist_exit - _DIST_EMA_IDEAL) / (_DIST_EMA_GOOD - _DIST_EMA_IDEAL))
                # else: fechou lucrativo mas longe da EMA → sem bônus extra (trade de momentum, não MR)
            else:
                # Perdeu E a EMA não foi alcançada → nunca houve reversão real
                if dist_exit > 1.5:
                    mr_convergence = -2.5   # Máxima penalidade: morreu longe da média
                elif dist_exit > 1.0:
                    mr_convergence = -1.0   # Penalidade moderada

            # ── 5. EFICIÊNCIA ORNSTEIN-UHLENBECK (duração) ───────────────────
            # Mean reversion tem half-life: premia saídas rápidas, penaliza overholding
            # Referência: Chan (2013), Tabela 3.1 — Optimal exit for OU process
            hl = self.mean_reversion_halflife  # 15 candles
            if duration_steps <= hl:
                dur_efficiency = 1.00     # Ótimo: saiu dentro do half-life
            elif duration_steps <= int(1.5 * hl):
                dur_efficiency = 0.75     # Bom: dentro de 1.5x half-life
            elif duration_steps <= 2 * hl:
                dur_efficiency = 0.50     # Aceitável: dentro de 2x half-life
            elif duration_steps <= 3 * hl:
                dur_efficiency = 0.25     # Lento: 3x half-life
            else:
                dur_efficiency = -0.20    # Overheld: prejudica generalização

            # PnL escalado pela eficiência de duração
            duration_scaled_pnl = pnl_base * max(0.1, dur_efficiency)
            # (max 0.1 para não zerar reward de trades longos lucrativos completamente)

            # ── TOTAL ─────────────────────────────────────────────────────────
            total = duration_scaled_pnl + adx_penalty + bb_bonus + mr_convergence

            logger.debug(
                "🤠 Ranger reward: gross=%.3f%% fee=%.3f%% net=%.3f%% | "
                "pnl_base=%.3f adx_pen=%.2f bb_bon=%.2f mr_conv=%.2f dur_eff=%.2f → total=%.3f",
                float(trade_return_pct) * 100, roundtrip_fee_pct * 100, net_return_pct * 100,
                duration_scaled_pnl, adx_penalty, bb_bonus, mr_convergence,
                dur_efficiency, total,
            )

            return float(np.clip(total, -25.0, 25.0))

        except Exception as exc:
            logger.warning("⚠️ Erro no reward Ranger: %s", exc)
            # Fallback ao parent em caso de erro inesperado
            return super()._compute_trade_reward(
                pnl_realized, trade_return_pct, duration_steps, exit_reason, atr_pct
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SAÍDA DINÂMICA
    # ─────────────────────────────────────────────────────────────────────────

    def _should_exit_fast(self, current_pnl_pct: float) -> bool:
        """
        Saída dinâmica para mean reversion — 4 critérios:

        1. Limiar base: 0.8% de lucro → sair sempre
        2. ADX alto com lucro: mercado saindo do range → tomar lucro cedo
        3. EMA alcançada com lucro: reversão completa → sair imediatamente
        4. 2x half-life com qualquer lucro positivo → não esperar mais
        """
        # ── Critério base ────────────────────────────────────────────────────
        if current_pnl_pct >= 0.008:
            return True

        try:
            exit_row = self._get_row_safe()

            # ── Critério 2: ADX subindo → breakout iminente ──────────────────
            adx_now = self._get_adx(exit_row)
            if adx_now > _ADX_TREND_WARN and current_pnl_pct > 0.003:
                # Mercado saindo do range: tomar qualquer lucro > 0.3%
                return True

            # ── Critério 3: EMA alcançada ────────────────────────────────────
            ema_val = self._get_ema_trend_value(exit_row)
            if ema_val > 0:
                price    = float(exit_row.get('close', 0.0))
                atr_val  = self._get_atr_safe(exit_row, price)
                dist_ema = abs(price - ema_val) / (atr_val + 1e-9)
                if dist_ema < 0.30 and current_pnl_pct > 0.0:
                    # Preço voltou à EMA e está lucrativo → reversão completa
                    return True

            # ── Critério 4: Ultrapassou 2x half-life ────────────────────────
            if (
                hasattr(self, 'steps_in_position')
                and self.steps_in_position > 2 * self.mean_reversion_halflife
                and current_pnl_pct > 0.002
            ):
                # Ficou 30 candles+ → aceitar lucro mínimo de 0.2%
                return True

        except Exception:
            pass

        return False
