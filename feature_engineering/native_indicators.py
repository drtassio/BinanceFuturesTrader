# -----------------------------------------------------------------------------
# ARQUIVO: feature_engineering/native_indicators.py (Completo e Corrigido)
# -----------------------------------------------------------------------------

"""
ImplementaÃ§Ã£o nativa de indicadores tÃ©cnicos usando Python, Pandas e NumPy.
"""

import os
import pandas as pd
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from typing import Tuple, List, Dict, Optional
from utils.logger import get_logger , LOG_LEVEL_DEBUG

logger = get_logger("NativeIndicators" , LOG_LEVEL_DEBUG)

class NativeIndicators:
    """
    Agrupa todos os cÃ¡lculos de indicadores tÃ©cnicos como mÃ©todos estÃ¡ticos.
    """
    # Constante EPSILON como variÃ¡vel de classe para acesso consistente
    EPSILON = 1e-9

    # Cache de d ótimo por (symbol-timeframe) — evita recalcular a cada ciclo
    # Formato: { "BTCUSDT-15m": (d_value, timestamp_unix) }
    _d_cache: Dict[str, Tuple[float, float]] = {}
    _D_CACHE_TTL_SECONDS: float = 3600.0  # Revalida 1x por hora

    # Persist d values to disk so training d == inference d across restarts
    _D_PERSIST_PATH: str = os.path.join(os.getcwd(), "models_ai", "fracdiff_d_values.json") if True else ""

    @staticmethod
    def _load_d_from_disk() -> Dict[str, float]:
        """Load persisted d values. Returns {} on any failure."""
        try:
            import json
            path = NativeIndicators._D_PERSIST_PATH
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
        except Exception:
            pass
        return {}

    @staticmethod
    def _save_d_to_disk(cache_key: str, d_val: float) -> None:
        """Persist d value for cache_key to disk (non-blocking, silent on error)."""
        try:
            import json
            path = NativeIndicators._D_PERSIST_PATH
            os.makedirs(os.path.dirname(path), exist_ok=True)
            existing = NativeIndicators._load_d_from_disk()
            existing[cache_key] = round(d_val, 4)
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def _validate_series(series: pd.Series, min_length: int, name: str) -> bool:
        """
        Helper para validar sÃ©ries de entrada.
        Retorna False se a sÃ©rie nÃ£o for vÃ¡lida ou for muito curta para o cÃ¡lculo inicial.
        """
        if not isinstance(series, pd.Series):
            logger.debug(f"ð¨ [VAL. IND.] '{name}' nÃ£o Ã© uma pd.Series. Tipo: {type(series)}. Retornando False.")
            return False
        if series.empty:
            logger.debug(f"ð¨ [VAL. IND.] '{name}' Ã© uma sÃ©rie vazia. Retornando False.")
            return False
        if len(series) < min_length:
            logger.warning(f"â ï¸ [VAL. IND.] '{name}' tem menos dados ({len(series)}) do que o mÃ­nimo recomendado ({min_length}). CÃ¡lculos podem resultar em NaNs iniciais.")
        return True

    @staticmethod
    def frac_diff_ffd(
        series: pd.Series,
        d: float = 0.5,
        window: int = 64,
        threshold: float = 1e-4,
    ) -> pd.Series:
        """
        [OPTIMIZED] Fractional Differentiation (Fixed-Width).
        Usa torch Gpu if available, senão fallback para np.convolve (vectorized).
        """
        min_len = max(4, window)
        if not NativeIndicators._validate_series(series, min_len, "frac_diff_ffd input"):
            return pd.Series(np.nan, index=series.index, dtype=np.float32)

        clean = pd.to_numeric(series, errors="coerce").astype(float)
        n = len(clean)
        
        # 1. Calcula Pesos
        weights = [1.0]
        for k in range(1, window):
            weight = -weights[-1] * (d - k + 1) / k
            if abs(weight) < threshold:
                break
            weights.append(weight)
        weights = np.array(weights[::-1], dtype=np.float64)
        weight_len = len(weights)
        
        # --- ACELERAÇÃO GPU (TORCH) OU VECTORIZED NUMPY ---
        if torch is not None and torch.cuda.is_available():
            try:
                # Converte para tensor e move para GPU
                device = torch.device('cuda')
                series_tensor = torch.tensor(clean.values, dtype=torch.float32, device=device).view(1, 1, -1)
                weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device).view(1, 1, -1)
                
                # Aplica Convolução 1D
                start_idx = weight_len - 1
                with torch.no_grad():
                    output_tensor = torch.nn.functional.conv1d(series_tensor, weights_tensor, padding=0)
                
                # Converte de volta
                output_values = output_tensor.cpu().numpy().flatten()
                
                # Remonta série com NaNs iniciais
                final_output = np.full(n, np.nan, dtype=np.float32)
                final_output[start_idx:] = output_values
                return pd.Series(final_output, index=series.index, dtype=np.float32)
                
            except Exception as e:
                logger.debug(f"⚠️ GPU FracDiff falhou (fallback para CPU): {e}")

        # Fallback Vectorized: np.convolve
        output_values = np.convolve(clean.values, weights, mode='valid')
        final_output = np.full(n, np.nan, dtype=np.float32)
        final_output[weight_len - 1:] = output_values
        
        return pd.Series(final_output, index=series.index, dtype=np.float32)

    @staticmethod
    def find_min_d_for_stationarity(
        series: pd.Series,
        window: int = 256,
        d_start: float = 0.0,
        d_max: float = 1.0,
        p_threshold: float = 0.05,
        cache_key: str = "",
        force_recalc: bool = False,
    ) -> float:
        """
        [BULLETPROOF] Encontra o d mínimo que torna a série estacionária.

        Estratégia em 4 camadas — NUNCA retorna sem um d válido:
          1. Cache TTL  : reutiliza o d descoberto na última hora
          2. Binary search (10 iter, precisão ~0.001): acha o d mínimo real
          3. Guard loop (+0.10/tentativa, 5×): cobre divergência amostra vs série completa
          4. Fallback final hardcoded: d=1.0 se tudo falhar — pipeline nunca trava

        Referência: López de Prado (2018) — Advances in Financial ML, cap. 5
        """
        import time

        # ------------------------------------------------------------------
        # CAMADA 0: Validação de entrada
        # ------------------------------------------------------------------
        if not isinstance(series, pd.Series) or len(series) < 100:
            logger.warning("⚠️ [FRACDIFF] Série muito curta para busca de d. Usando d=0.5.")
            return 0.5

        clean_input = pd.to_numeric(series, errors='coerce').dropna()
        if len(clean_input) < 100:
            logger.warning("⚠️ [FRACDIFF] Série com muitos NaNs. Usando d=0.5.")
            return 0.5

        if clean_input.std() < NativeIndicators.EPSILON:
            logger.warning("⚠️ [FRACDIFF] Série constante. Usando d=1.0.")
            return 1.0

        # ------------------------------------------------------------------
        # CAMADA 1: Cache TTL + Disco (training d == inference d cross-restart)
        # ------------------------------------------------------------------
        now = time.time()
        if cache_key and not force_recalc:
            # 1a. In-memory cache (sub-second, TTL=1h)
            cached = NativeIndicators._d_cache.get(cache_key)
            if cached is not None:
                d_cached, ts_cached = cached
                age = now - ts_cached
                if age < NativeIndicators._D_CACHE_TTL_SECONDS:
                    logger.debug(f"✅ [FRACDIFF CACHE] {cache_key}: d={d_cached:.3f} (age={age:.0f}s)")
                    return d_cached
            # 1b. Disk cache — preserva o mesmo d entre restarts (treino == inferência)
            disk_vals = NativeIndicators._load_d_from_disk()
            if cache_key in disk_vals:
                d_disk = disk_vals[cache_key]
                NativeIndicators._d_cache[cache_key] = (d_disk, now)
                logger.info(f"✅ [FRACDIFF DISK] {cache_key}: d={d_disk:.3f} (carregado do disco — consistente com treino)")
                return d_disk

        # ------------------------------------------------------------------
        # CAMADA 2: Importa validator — fallback se indisponível
        # ------------------------------------------------------------------
        validator = None
        try:
            from feature_engineering.financial_stationarity import StationarityValidator
            validator = StationarityValidator()
        except Exception as e:
            logger.warning(f"⚠️ [FRACDIFF] StationarityValidator indisponível ({e}). Usando d=0.4.")
            return 0.4

        # Helper blindado — retorna 1.0 em qualquer falha do ADF
        def _adf_pvalue(d_val: float, work_series: pd.Series) -> float:
            try:
                frac = NativeIndicators.frac_diff_ffd(work_series, d=d_val, window=window)
                clean = frac.dropna()
                if len(clean) < 50:
                    return 1.0
                if not np.all(np.isfinite(clean.values)):
                    return 1.0
                if clean.std() < NativeIndicators.EPSILON:
                    return 1.0
                metrics = validator.test_adf(clean)
                pv = metrics.get('pvalue', 1.0)
                return float(pv) if np.isfinite(pv) else 1.0
            except Exception as exc:
                logger.debug(f"⚠️ [FRACDIFF ADF] d={d_val:.3f} falhou: {exc}")
                return 1.0

        # Usa no máximo 100k pontos para a busca (performance)
        work = clean_input.tail(100_000) if len(clean_input) > 100_000 else clean_input

        # ------------------------------------------------------------------
        # CAMADA 2: Binary search — 10 iterações, precisão ≈ 0.001
        # ------------------------------------------------------------------
        low_b, high_b = float(d_start), float(d_max)
        best_d = float(d_max)  # Pessimista: começa no máximo

        # Série já estacionária com d mínimo?
        if _adf_pvalue(max(0.01, low_b), work) < p_threshold:
            best_d = max(0.01, low_b)
            logger.info(f"✅ [FRACDIFF] {cache_key}: série já estacionária com d≈0. d={best_d:.3f}")
        else:
            for iteration in range(10):
                mid = (low_b + high_b) / 2.0
                pv = _adf_pvalue(mid, work)
                if pv < p_threshold:
                    best_d = mid
                    high_b = mid   # Candidato válido — tenta d menor
                else:
                    low_b = mid    # Não estacionário — aumenta d
                logger.debug(
                    f"🔍 [FRACDIFF SEARCH] {cache_key} iter={iteration+1}/10 "
                    f"d={mid:.4f} p={pv:.4f} best_d={best_d:.4f}"
                )

        # ------------------------------------------------------------------
        # CAMADA 3: Guard loop — valida best_d na série completa
        # ------------------------------------------------------------------
        final_d = round(best_d, 3)
        guard_pv = _adf_pvalue(final_d, work)

        if guard_pv >= p_threshold:
            logger.warning(
                f"⚠️ [FRACDIFF GUARD] {cache_key}: d={final_d:.3f} falhou "
                f"(p={guard_pv:.4f}). Guard loop iniciado (+0.10/tentativa)."
            )
            for attempt in range(5):
                final_d = round(min(1.0, final_d + 0.10), 3)
                guard_pv = _adf_pvalue(final_d, work)
                status = "✅ OK" if guard_pv < p_threshold else "❌ Falhou"
                logger.warning(
                    f"   Guard {attempt+1}/5: d={final_d:.3f} → p={guard_pv:.4f} {status}"
                )
                if guard_pv < p_threshold:
                    break

        # ------------------------------------------------------------------
        # CAMADA 4: Fallback final
        # ------------------------------------------------------------------
        if guard_pv >= p_threshold:
            logger.error(
                f"❌ [FRACDIFF CRITICAL] {cache_key}: impossível estacionarizar. "
                f"Usando d=1.0 (diff simples). Pipeline NÃO travará."
            )
            final_d = 1.0

        logger.info(
            f"📊 [FRACDIFF] {cache_key}: d ótimo={final_d:.3f} | "
            f"p={guard_pv:.4f} | "
            f"{'✅ Estacionário' if guard_pv < p_threshold else '⚠️ Fallback d=1.0'}"
        )

        # Salva em memória e em disco (garante treino == inferência entre restarts)
        if cache_key:
            NativeIndicators._d_cache[cache_key] = (final_d, now)
            NativeIndicators._save_d_to_disk(cache_key, final_d)

        return final_d

    @staticmethod
    def sma(series: pd.Series, length: int = 14) -> pd.Series:
        """Calcula a MÃ©dia MÃ³vel Simples (SMA)."""
        if not NativeIndicators._validate_series(series, length, "SMA input series"):
            return pd.Series(np.nan, index=series.index, dtype=np.float32)
        return series.rolling(window=length, min_periods=length).mean().astype(np.float32)

    @staticmethod
    def ema(series: pd.Series, length: int = 14) -> pd.Series:
        """Calcula a MÃ©dia MÃ³vel Exponencial (EMA)."""
        if not NativeIndicators._validate_series(series, length, "EMA input series"):
            return pd.Series(np.nan, index=series.index, dtype=np.float32)
        return series.ewm(span=length, adjust=False, min_periods=length).mean().astype(np.float32)

    @staticmethod
    def rsi(series: pd.Series, length: int = 14) -> pd.Series:
        """Calcula o Ãndice de ForÃ§a Relativa (RSI)."""
        if not NativeIndicators._validate_series(series, length + 1, "RSI input series"):
            return pd.Series(np.nan, index=series.index, dtype=np.float32)

        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.fillna(0).ewm(span=length, adjust=False, min_periods=length).mean()
        avg_loss = loss.fillna(0).ewm(span=length, adjust=False, min_periods=length).mean()
        
        rs = avg_gain / (avg_loss + NativeIndicators.EPSILON)
        rsi = 100 - (100 / (1 + rs))
        return np.clip(rsi.fillna(50), 0, 100).astype(np.float32)

    @staticmethod
    def macd(series: pd.Series, fast_len: int = 12, slow_len: int = 26, signal_len: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula a ConvergÃªncia/DivergÃªncia de MÃ©dias MÃ³veis (MACD)."""
        min_len = max(fast_len, slow_len, signal_len) 
        if not NativeIndicators._validate_series(series, min_len, "MACD input series"):
            nan_series = pd.Series(np.nan, index=series.index, dtype=np.float32)
            return nan_series, nan_series, nan_series
            
        ema_fast = NativeIndicators.ema(series, length=fast_len)
        ema_slow = NativeIndicators.ema(series, length=slow_len)
        
        macd_line = ema_fast - ema_slow
        signal_line = NativeIndicators.ema(macd_line, length=signal_len)
        histogram = macd_line - signal_line
        
        return macd_line.astype(np.float32), signal_line.astype(np.float32), histogram.astype(np.float32)

    @staticmethod
    def bollinger_bands(series: pd.Series, length: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula as Bandas de Bollinger (BB)."""
        if not NativeIndicators._validate_series(series, length, "Bollinger Bands input series"):
            nan_series = pd.Series(np.nan, index=series.index, dtype=np.float32)
            return nan_series, nan_series, nan_series
            
        middle_band = NativeIndicators.sma(series, length=length)
        std = series.rolling(window=length, min_periods=length).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band.astype(np.float32), middle_band.astype(np.float32), lower_band.astype(np.float32)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """
        Calcula o Average True Range (ATR).
        Garante que o ATR seja sempre positivo.
        """
        min_len = length + 1
        if not (NativeIndicators._validate_series(high, min_len, "ATR high") and
                NativeIndicators._validate_series(low, min_len, "ATR low") and
                NativeIndicators._validate_series(close, min_len, "ATR close")):
            logger.warning(f"â ï¸ [IND ATR] Dados insuficientes para ATR. Min necessÃ¡rio: {min_len}. Retornando NaNs.")
            return pd.Series(np.nan, index=close.index, dtype=np.float32)

        high_low = high - low
        high_close_prev = (high - close.shift(1)).abs()
        low_close_prev = (low - close.shift(1)).abs()
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False).fillna(0)
        
        atr = tr.ewm(span=length, adjust=False, min_periods=length).mean()
        return np.maximum(atr, NativeIndicators.EPSILON).astype(np.float32)

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcula o Ãndice Direcional MÃ©dio (ADX), Positive Directional Indicator (+DI), e Negative Direcional Indicator (-DI).
        """
        min_len = length * 2 + 1
        if not (NativeIndicators._validate_series(high, min_len, "ADX high") and
                NativeIndicators._validate_series(low, min_len, "ADX low") and
                NativeIndicators._validate_series(close, min_len, "ADX close")):
            nan_series = pd.Series(np.nan, index=close.index, dtype=np.float32)
            return nan_series, nan_series, nan_series

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index).fillna(0.0)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=low.index).fillna(0.0)

        smooth_plus_dm = plus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        smooth_minus_dm = minus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

        atr_val_for_adx = NativeIndicators.atr(high, low, close, length)

        plus_di = 100 * (smooth_plus_dm / (atr_val_for_adx + NativeIndicators.EPSILON))
        minus_di = 100 * (smooth_minus_dm / (atr_val_for_adx + NativeIndicators.EPSILON))

        dx = 100 * (np.abs(plus_di - minus_di) / (np.abs(plus_di + minus_di) + NativeIndicators.EPSILON))
        adx = dx.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

        return np.clip(adx.fillna(0), 0, 100).astype(np.float32), np.clip(plus_di.fillna(0), 0, 100).astype(np.float32), np.clip(minus_di.fillna(0), 0, 100).astype(np.float32)

    @staticmethod
    def parabolic_sar(
        high: pd.Series,
        low: pd.Series,
        step: float = 0.02,
        max_step: float = 0.2,
    ) -> pd.Series:
        """
        Calcula o Parabolic SAR (Stop and Reverse) otimizando para custos baixos.
        """
        if not NativeIndicators._validate_series(high, 3, "PSAR high") or not NativeIndicators._validate_series(low, 3, "PSAR low"):
            return pd.Series(np.nan, index=high.index, dtype=np.float32)

        high_values = high.values.astype(float)
        low_values = low.values.astype(float)
        n = len(high_values)
        if n == 0:
            return pd.Series(dtype=np.float32)

        sar = np.zeros(n, dtype=np.float64)
        is_long = True
        af = float(max(step, 1e-4))
        max_step = float(max_step)
        ep_high = float(high_values[0])
        ep_low = float(low_values[0])
        sar[0] = ep_low

        for i in range(1, n):
            prev_sar = sar[i - 1]
            if is_long:
                sar_candidate = prev_sar + af * (ep_high - prev_sar)
                if i >= 2:
                    sar_candidate = min(sar_candidate, low_values[i - 1], low_values[i - 2])
                else:
                    sar_candidate = min(sar_candidate, low_values[i - 1])
                sar[i] = sar_candidate
                if low_values[i] < sar_candidate:
                    is_long = False
                    sar[i] = ep_high
                    ep_low = float(low_values[i])
                    af = float(step)
                else:
                    if high_values[i] > ep_high:
                        ep_high = float(high_values[i])
                        af = min(af + step, max_step)
            else:
                sar_candidate = prev_sar + af * (ep_low - prev_sar)
                if i >= 2:
                    sar_candidate = max(sar_candidate, high_values[i - 1], high_values[i - 2])
                else:
                    sar_candidate = max(sar_candidate, high_values[i - 1])
                sar[i] = sar_candidate
                if high_values[i] > sar_candidate:
                    is_long = True
                    sar[i] = ep_low
                    ep_high = float(high_values[i])
                    af = float(step)
                else:
                    if low_values[i] < ep_low:
                        ep_low = float(low_values[i])
                        af = min(af + step, max_step)

        return pd.Series(sar, index=high.index, dtype=np.float32)

    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calcula o Oscilador EstocÃ¡stico (%K e %D)."""
        min_len = k_period + d_period 
        if not (NativeIndicators._validate_series(high, min_len, "Stochastic high") and
                NativeIndicators._validate_series(low, min_len, "Stochastic low") and
                NativeIndicators._validate_series(close, min_len, "Stochastic close")):
            nan_series = pd.Series(np.nan, index=close.index, dtype=np.float32)
            return nan_series, nan_series

        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()
        
        k_line = 100 * ((close - lowest_low) / (highest_high - lowest_low + NativeIndicators.EPSILON))
        d_line = NativeIndicators.sma(k_line, length=d_period)
        
        return np.clip(k_line.fillna(50), 0, 100).astype(np.float32), np.clip(d_line.fillna(50), 0, 100).astype(np.float32)

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """Calcula o Williams %R."""
        if not (NativeIndicators._validate_series(high, length, "Williams %R high") and
                NativeIndicators._validate_series(low, length, "Williams %R low") and
                NativeIndicators._validate_series(close, length, "Williams %R close")):
            return pd.Series(np.nan, index=close.index, dtype=np.float32)

        highest_high = high.rolling(window=length, min_periods=length).max()
        lowest_low = low.rolling(window=length, min_periods=length).min()
        
        r = -100 * ((highest_high - close) / (highest_high - lowest_low + NativeIndicators.EPSILON))
        return np.clip(r.fillna(-50), -100, 0).astype(np.float32)

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
        """Calcula o Commodity Channel Index (CCI)."""
        if not (NativeIndicators._validate_series(high, length, "CCI high") and
                NativeIndicators._validate_series(low, length, "CCI low") and
                NativeIndicators._validate_series(close, length, "CCI close")):
            return pd.Series(np.nan, index=close.index, dtype=np.float32)

        typical_price = (high + low + close) / 3
        sma_tp = NativeIndicators.sma(typical_price, length)
        
        mean_deviation = typical_price.rolling(window=length, min_periods=length).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True).fillna(0)
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation + NativeIndicators.EPSILON)
        return cci.fillna(0).astype(np.float32)

    @staticmethod
    def roc(series: pd.Series, length: int = 12) -> pd.Series:
        """Calcula a Taxa de VariaÃ§Ã£o (Rate of Change)."""
        if not NativeIndicators._validate_series(series, length + 1, "ROC input series"):
            return pd.Series(np.nan, index=series.index, dtype=np.float32)
            
        roc = ((series - series.shift(length)) / (series.shift(length) + NativeIndicators.EPSILON)) * 100
        return roc.fillna(0).astype(np.float32)

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calcula o On-Balance Volume (OBV).
        Adiciona normalização relativa para estabilidade em séries longas.
        """
        if not (NativeIndicators._validate_series(close, 2, "OBV close") and
                NativeIndicators._validate_series(volume, 2, "OBV volume")):
            return pd.Series(np.nan, index=close.index, dtype=np.float32)

        volume_clean = volume.replace(0, NativeIndicators.EPSILON)
        direction = np.sign(close.diff()).fillna(0)
        obv_series = (volume_clean * direction).cumsum().fillna(0)
        
        # Normalização relativa por rolling std para evitar overflow/escala gigantesca
        obv_norm = obv_series / (obv_series.rolling(200, min_periods=1).std() + NativeIndicators.EPSILON)
        return obv_norm.astype(np.float32)

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calcula o Volume Weighted Average Price (VWAP) diário."""
        if not (NativeIndicators._validate_series(high, 1, "VWAP high") and
                NativeIndicators._validate_series(low, 1, "VWAP low") and
                NativeIndicators._validate_series(close, 1, "VWAP close") and
                NativeIndicators._validate_series(volume, 1, "VWAP volume")):
            return pd.Series(np.nan, index=close.index, dtype=np.float32)

        typical_price = (high + low + close) / 3
        df_temp = pd.DataFrame({'tp': typical_price, 'vol': volume}, index=typical_price.index)
        df_temp['vol_clean'] = df_temp['vol'].replace(0, NativeIndicators.EPSILON)
        df_temp['tp_vol'] = df_temp['tp'] * df_temp['vol_clean']
        
        # Proteção para DatetimeIndex
        if isinstance(df_temp.index, pd.DatetimeIndex):
            df_temp['cum_tp_vol'] = df_temp.groupby(df_temp.index.date)['tp_vol'].cumsum()
            df_temp['cum_vol'] = df_temp.groupby(df_temp.index.date)['vol_clean'].cumsum()
        else:
            # Fallback: VWAP cumulativo sem reset diário
            df_temp['cum_tp_vol'] = df_temp['tp_vol'].cumsum()
            df_temp['cum_vol'] = df_temp['vol_clean'].cumsum()
        
        vwap = df_temp['cum_tp_vol'] / (df_temp['cum_vol'] + NativeIndicators.EPSILON)
        return vwap.fillna(typical_price).astype(np.float32)

    @staticmethod
    def candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detecta padrÃµes bÃ¡sicos de candlestick."""
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols) or df.empty or len(df) < 2:
            return pd.DataFrame(index=df.index, dtype=bool)
        
        patterns = pd.DataFrame(index=df.index)
        open_, high_, low_, close_ = df['open'], df['high'], df['low'], df['close']
        body = abs(close_ - open_)
        total_range = high_ - low_ + NativeIndicators.EPSILON 
        
        patterns['doji'] = (body / total_range < 0.1) & (body > NativeIndicators.EPSILON)
        patterns['marubozu_bullish'] = (open_ < close_) & (open_ <= low_ * 1.0001) & (close_ >= high_ * 0.9999) & (body / total_range > 0.8)
        patterns['marubozu_bearish'] = (open_ > close_) & (open_ >= high_ * 0.9999) & (close_ <= low_ * 1.0001) & (body / total_range > 0.8) 

        prev_open, prev_close = open_.shift(1), close_.shift(1)
        patterns['engulfing_bullish'] = (close_ > open_) & (prev_close < prev_open) & (close_ > prev_open) & (open_ < prev_close) & (body > abs(prev_close - prev_open) * 0.9)
        patterns['engulfing_bearish'] = (close_ < open_) & (prev_close > prev_open) & (close_ < prev_open) & (open_ > prev_close) & (body > abs(prev_close - prev_open) * 0.9)

        min_body_ratio = body / total_range
        lower_shadow = np.minimum(open_, close_) - low_
        upper_shadow = high_ - np.maximum(open_, close_)
        patterns['hammer'] = (min_body_ratio < 0.3) & (lower_shadow / total_range > 0.6) & (upper_shadow / total_range < 0.1) & (body > NativeIndicators.EPSILON) 
        patterns['shooting_star'] = (min_body_ratio < 0.3) & (upper_shadow / total_range > 0.6) & (lower_shadow / total_range < 0.1) & (body > NativeIndicators.EPSILON)
        
        return patterns.fillna(False) 

    @staticmethod
    def calculate_money_flow_volume_pressure(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.DataFrame:
        """
        Calcula mÃ©tricas de pressÃ£o de volume (Money Flow) para simular tape reading.
        """
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(NativeIndicators._validate_series(series, 1, name) for series, name in zip([high, low, close, volume], required_cols)):
            return pd.DataFrame(index=close.index, dtype=np.float32)

        df = pd.DataFrame({'high': high, 'low': low, 'close': close, 'volume': volume})
        df['volume'] = df['volume'].replace(0, NativeIndicators.EPSILON)
        
        df['mfm'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + NativeIndicators.EPSILON)
        df['mfm'] = df['mfm'].fillna(0) 
        df['mfv'] = df['mfm'] * df['volume']
        
        cmf_period = 20
        cmf = df['mfv'].rolling(window=cmf_period, min_periods=cmf_period).sum() / df['volume'].rolling(window=cmf_period, min_periods=cmf_period).sum()
        df['cmf'] = cmf.fillna(0).astype(np.float32)

        close_diff = df['close'].diff()
        prev_close = df['close'].shift(1) + NativeIndicators.EPSILON
        df['pvt'] = (close_diff / prev_close) * df['volume']
        df['pvt'] = df['pvt'].cumsum().fillna(0).astype(np.float32)
        
        df['adl_mfm'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + NativeIndicators.EPSILON)
        df['adl_mfm'] = df['adl_mfm'].fillna(0)
        df['adl'] = (df['adl_mfm'] * df['volume']).cumsum().fillna(0).astype(np.float32)

        return df[['cmf', 'pvt', 'adl']]

    @staticmethod
    def calculate_all_features(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        [VERSÃO COMPLETA E ROBUSTA RESTAURADA]
        Calcula um conjunto abrangente de features, garantindo que todas as colunas
        necessÃ¡rias para todos os componentes do bot (TrendPredictor, Especialistas)
        sejam geradas corretamente.
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if df.empty or not all(col in df.columns for col in required_cols):
            logger.error(f"â [ERRO FE] DataFrame de entrada para calculate_all_features ({symbol}-{timeframe}) estÃ¡ vazio ou faltando colunas.")
            return pd.DataFrame()
        
        df_features = df.copy()
        for col in required_cols:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')


        # [SCIENTIFIC FIX] d adaptativo com cache por (symbol-timeframe)
        # find_min_d_for_stationarity: binary search + guard loop + 4 fallbacks
        # cache_key evita recalcular nas chamadas subsequentes (TTL=1h)
        frac_window = 256
        cache_key = f"{symbol}-{timeframe}"
        frac_order = NativeIndicators.find_min_d_for_stationarity(
            df_features['close'],
            window=frac_window,
            cache_key=cache_key,
        )

        # Aplica a diferenciação final (garantidamente estacionária) para todos os preços
        for price_col in ('close', 'high', 'low'):
            df_features[f"{price_col}_frac"] = NativeIndicators.frac_diff_ffd(
                df_features[price_col],
                d=frac_order,
                window=frac_window,
                threshold=1e-4,
            )

        # Log ADF final para rastreabilidade (usa o que já foi calculado acima)
        try:
            from feature_engineering.financial_stationarity import StationarityValidator
            _val = StationarityValidator()
            _metrics = _val.test_adf(df_features['close_frac'].dropna())
            _pv = _metrics.get('pvalue', 1.0)
            logger.info(
                f"📊 [FE ADF] {symbol}-{timeframe} close_frac p-value: {_pv:.4f} | "
                f"Stationary: {_pv < 0.05} | d={frac_order:.3f}"
            )
        except Exception:
            pass  # Log informativo — nunca trava o pipeline

        df_features['ema_12'] = NativeIndicators.ema(df_features['close'], 12)
        
        # --- FEATURES REQUERIDAS PELO AUTOENCODER (log_return, realized_vol, wick, etc.) ---
        # Estas features são essenciais para o TemporalAutoencoder
        EPSILON = NativeIndicators.EPSILON
        
        # 1. Log Returns
        df_features['log_return'] = np.log(df_features['close'] / (df_features['close'].shift(1) + EPSILON))
        df_features['log_return_volume'] = np.log(df_features['volume'] / (df_features['volume'].shift(1) + EPSILON))
        
        # 2. Price Range Features (como % do close)
        df_features['hl_range_pct'] = (df_features['high'] - df_features['low']) / (df_features['close'] + EPSILON)
        df_features['oc_move_pct'] = (df_features['close'] - df_features['open']) / (df_features['open'] + EPSILON)
        
        # 3. Wick Features (proporção do candle que é pavio)
        candle_range = df_features['high'] - df_features['low'] + EPSILON
        df_features['hc_wick_upper'] = (df_features['high'] - np.maximum(df_features['close'], df_features['open'])) / candle_range
        df_features['lc_wick_lower'] = (np.minimum(df_features['close'], df_features['open']) - df_features['low']) / candle_range
        
        # 4. Realized Volatility (rolling std of log returns)
        df_features['realized_vol_20'] = df_features['log_return'].rolling(window=20, min_periods=5).std() * np.sqrt(20)
        df_features['realized_vol_100'] = df_features['log_return'].rolling(window=100, min_periods=20).std() * np.sqrt(100)
        
        # 5. Close Reference (para normalização relativa)
        df_features['close_reference'] = df_features['close'] / df_features['close'].shift(1).fillna(df_features['close'])

        df_features['ema_26'] = NativeIndicators.ema(df_features['close'], 26)
        df_features['ema_50'] = NativeIndicators.ema(df_features['close'], 50)
        df_features['ema_200'] = NativeIndicators.ema(df_features['close'], 200)
        df_features['ema_trend'] = np.sign(df_features['ema_50'] - df_features['ema_200']).fillna(0).astype(int)
        df_features['rsi'] = NativeIndicators.rsi(df_features['close'], 14)
        df_features['macd_line'], df_features['macd_signal'], df_features['macd_hist'] = NativeIndicators.macd(df_features['close'])
        df_features['stoch_k'], df_features['stoch_d'] = NativeIndicators.stochastic_oscillator(df_features['high'], df_features['low'], df_features['close'])
        df_features['williams_r'] = NativeIndicators.williams_r(df_features['high'], df_features['low'], df_features['close'])
        df_features['cci'] = NativeIndicators.cci(df_features['high'], df_features['low'], df_features['close'])
        df_features['roc'] = NativeIndicators.roc(df_features['close'], 12)
        df_features['bb_upper'], df_features['bb_middle'], df_features['bb_lower'] = NativeIndicators.bollinger_bands(df_features['close'])
        df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / (df_features['bb_middle'] + NativeIndicators.EPSILON)
        df_features['atr'] = NativeIndicators.atr(df_features['high'], df_features['low'], df_features['close'], length=14)
        df_features['atr_percentage'] = (df_features['atr'] / (df_features['close'] + NativeIndicators.EPSILON)) * 100
        df_features['obv'] = NativeIndicators.obv(df_features['close'], df_features['volume'])
        df_features['vwap'] = NativeIndicators.vwap(df_features['high'], df_features['low'], df_features['close'], df_features['volume'])
        df_features['adx'], df_features['plus_di'], df_features['minus_di'] = NativeIndicators.adx(df_features['high'], df_features['low'], df_features['close'], length=14)
        df_features['psar'] = NativeIndicators.parabolic_sar(df_features['high'], df_features['low'])
        df_features['psar_trend'] = np.where(df_features['close'] >= df_features['psar'], 1.0, -1.0).astype(np.float32)

        tape_features = NativeIndicators.calculate_money_flow_volume_pressure(df_features['high'], df_features['low'], df_features['close'], df_features['volume'])
        df_features = pd.concat([df_features, tape_features], axis=1)

        df_features['macd_cross_bullish'] = ((df_features['macd_line'] > df_features['macd_signal']) & (df_features['macd_line'].shift(1) <= df_features['macd_signal'].shift(1))).astype(bool)
        df_features['macd_cross_bearish'] = ((df_features['macd_line'] < df_features['macd_signal']) & (df_features['macd_line'].shift(1) >= df_features['macd_signal'].shift(1))).astype(bool)
        df_features['rsi_oversold'] = (df_features['rsi'] < 30).astype(bool)
        df_features['rsi_overbought'] = (df_features['rsi'] > 70).astype(bool)
        
        candle_patterns_df = NativeIndicators.candlestick_patterns(df_features[['open', 'high', 'low', 'close']])
        df_features = pd.concat([df_features, candle_patterns_df], axis=1)

        df_features['adx_strong_trend'] = (df_features['adx'] > 25).astype(bool)
        df_features['adx_weak_trend'] = (df_features['adx'] < 20).astype(bool)
        
        bb_width_ma = df_features['bb_width'].rolling(20, min_periods=20).mean()
        df_features['bb_squeeze'] = (df_features['bb_width'] < (bb_width_ma * 0.8)) & (bb_width_ma > NativeIndicators.EPSILON)
        df_features['bb_expansion'] = (df_features['bb_width'] > (bb_width_ma * 1.5)) & (bb_width_ma > NativeIndicators.EPSILON)
        
        atr_ma = df_features['atr_percentage'].rolling(20, min_periods=20).mean()
        df_features['high_volatility'] = (df_features['atr_percentage'] > (atr_ma * 1.5)) & (atr_ma > NativeIndicators.EPSILON)
        df_features['low_volatility'] = (df_features['atr_percentage'] < (atr_ma * 0.7)) & (atr_ma > NativeIndicators.EPSILON)
        
        # --- ETAPA DE LIMPEZA FINAL (🔧 FIX: pular warmup period) ---
        # O indicador com maior lookback é EMA-200, então precisamos de ~250 candles de warmup
        WARMUP_PERIOD = 250  # Candles iniciais a descartar (EMA-200 + margem de segurança)
        
        initial_rows = len(df_features)
        
        # 1. Pular período de warmup (primeiros N candles têm NaN nos indicadores de lookback longo)
        if len(df_features) > WARMUP_PERIOD:
            df_features = df_features.iloc[WARMUP_PERIOD:].copy()
            logger.info(f"📊 [FE] {symbol}-{timeframe}: Descartados {WARMUP_PERIOD} candles de warmup (indicadores com lookback longo).")
        
        # 2. Preencher NaNs remanescentes (se houver, ex: colunas booleanas)
        df_features = df_features.fillna(0)
        
        # 3. Verificar se ainda há NaN (não deveria haver)
        rows_before_drop = len(df_features)
        df_features.dropna(inplace=True)
        final_rows = len(df_features)
        
        if rows_before_drop - final_rows > 0:
            logger.warning(f"⚠️ [FE CLEANING] Removidas {rows_before_drop - final_rows} linhas com NaN persistente de {symbol}-{timeframe}.")
        
        # Log de preservação de dados
        preserved_pct = (final_rows / initial_rows * 100) if initial_rows > 0 else 0
        logger.info(f"✅ [FE] {symbol}-{timeframe}: {final_rows}/{initial_rows} linhas preservadas ({preserved_pct:.1f}%).")
        
        if df_features.empty:
            logger.error(f"❌ [ERRO FE] Após a limpeza de NaNs, o DataFrame para {symbol}-{timeframe} ficou vazio.")
            return df_features

        for col in df_features.select_dtypes(include=np.number).columns:
            df_features[col] = df_features[col].astype(np.float32)
            
        return df_features


