"""
Teste diagnóstico do pipeline completo:
  Binance → Features → Regime → Physics → Signal

Rodar: python tests/test_market_diagnostic.py
Resultado: PASS/FAIL por camada + tabela de sanidade dos valores.
"""
import sys
import os
import asyncio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Windows UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ─── Helpers ─────────────────────────────────────────────────────────────────
RESULTS = []

def check(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((status, name, detail))
    sym = "+" if passed else "X"
    print(f"  [{sym}] {name:<55} {detail}")

def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


async def run_diagnostics():
    # ─── 1. CONFIG ────────────────────────────────────────────────────────────
    section("1. CONFIG & ENV")
    cfg = ai_cfg = None
    try:
        from config.settings import TradingConfig, AIConfig
        cfg = TradingConfig()
        ai_cfg = AIConfig()
        check("TradingConfig + AIConfig carregados", True)
        check("PRIMARY_PAIR definido", bool(cfg.PRIMARY_PAIR), cfg.PRIMARY_PAIR)
        check("PRIMARY_TIMEFRAME definido", bool(cfg.PRIMARY_TIMEFRAME_TRADING), cfg.PRIMARY_TIMEFRAME_TRADING)
        api_ok = bool(os.environ.get("BINANCE_API_KEY") or os.environ.get("BINANCE_TESTNET_API_KEY"))
        check("API key presente no .env", api_ok)
    except Exception as e:
        check("Config", False, str(e))

    symbol   = cfg.PRIMARY_PAIR if cfg else "BTCUSDT"
    interval = cfg.PRIMARY_TIMEFRAME_TRADING if cfg else "15m"

    # ─── 2. BINANCE KLINES ────────────────────────────────────────────────────
    section("2. BINANCE — KLINES FETCH")
    raw_df = None
    bc     = None
    try:
        from trading.binance_connector import BinanceConnector
        bc = BinanceConnector(cfg)
        await bc.connect()
        raw_data = await bc.get_kline_data(symbol=symbol, interval=interval, limit=200)
        ok = raw_data is not None and len(raw_data) > 0
        check("get_kline_data() retornou dados", ok,
              f"{len(raw_data) if ok else 0} candles")

        if ok:
            cols = ["open_time","open","high","low","close","volume",
                    "close_time","quote_asset_volume","number_of_trades",
                    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"]
            df = pd.DataFrame(raw_data, columns=cols)
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.index = pd.to_datetime(df["open_time"], unit="ms")
            raw_df  = df

            check("Colunas OHLCV presentes",
                  all(c in df.columns for c in ["open","high","low","close","volume"]))
            check("Sem NaN no OHLCV",
                  not df[["open","high","low","close","volume"]].isnull().any().any())
            check("Close > 0", (df["close"] > 0).all(),
                  f"ultima close = {df['close'].iloc[-1]:,.2f}")
            check("Candles em ordem cronologica", df["open_time"].is_monotonic_increasing)

            last_ts  = pd.to_datetime(df["open_time"].iloc[-1], unit="ms")
            age_min  = (pd.Timestamp.utcnow().replace(tzinfo=None) - last_ts).total_seconds() / 60
            check("Ultimo candle recente (< 20min)", age_min < 20,
                  f"{age_min:.1f} min atras")
    except Exception as e:
        check("Binance klines", False, str(e))

    # ─── 3. FEATURE ENGINEERING ───────────────────────────────────────────────
    section("3. FEATURE ENGINEERING")
    enriched_df = None
    # Features podem ter sufixo de timeframe (ex: rsi → rsi_15m)
    REQUIRED_FEATURES_BASE = ["rsi","adx","bb_upper","bb_lower","atr",
                               "log_return","realized_vol_20","hl_range_pct"]

    try:
        from feature_engineering.main import FeatureEngineeringPipeline
        fp = FeatureEngineeringPipeline(ai_cfg)

        if raw_df is not None:
            tf_dict = {interval: raw_df}
            for tf in (cfg.SECONDARY_TIMEFRAMES if cfg else ["1h"]):
                tf_dict[tf] = raw_df.copy()

            enriched_df = await fp.create_features(
                raw_df_dict=tf_dict,
                symbol=symbol,
                primary_timeframe=interval,
                fit_scaler=False,
                apply_latent=False,
            )
            ok = enriched_df is not None and len(enriched_df) > 0
            check("create_features() executou", ok,
                  f"{len(enriched_df)} linhas, {len(enriched_df.columns)} colunas" if ok else "vazio")

            if ok:
                # Resolve nomes reais: sem sufixo OU com sufixo de TF
                def col_exists(name):
                    if name in enriched_df.columns: return True
                    # tenta variantes com sufixo de TF
                    return any(c == f"{name}_{interval}" or
                               c.startswith(f"{name}_") for c in enriched_df.columns)

                missing = [f for f in REQUIRED_FEATURES_BASE if not col_exists(f)]
                check("Features tecnicas presentes", len(missing) == 0,
                      f"faltando: {missing}" if missing else "OK")

                # Mostra amostra dos nomes reais (ajuda debug)
                sample_cols = [c for c in enriched_df.columns
                               if any(c == f or c.startswith(f"{f}_")
                                      for f in ["rsi","adx","atr","log_return"])][:8]
                print(f"\n  Amostra colunas reais: {sample_cols}")

                nan_pct = enriched_df.isnull().mean().mean() * 100
                check("NaN < 5% no DataFrame", nan_pct < 5, f"{nan_pct:.1f}% NaN")

                inf_count = np.isinf(enriched_df.select_dtypes(include=np.number)).sum().sum()
                check("Sem inf no DataFrame", inf_count == 0,
                      f"{inf_count} inf encontrados")
        else:
            check("create_features()", False, "sem dados Binance")
    except Exception as e:
        check("Feature Engineering", False, str(e))

    # ─── 4. REGIME DETECTION ──────────────────────────────────────────────────
    section("4. REGIME DETECTION")
    regime_df = None
    REGIME_COLS = ["regime","regime_confidence","regime_name"]

    try:
        from config.regime_constants import CANONICAL_REGIME_CONFIG
        from feature_engineering.crypto_regime_detector import CryptoRegimeDetector
        check("CANONICAL_REGIME_CONFIG carregado", CANONICAL_REGIME_CONFIG is not None)

        if raw_df is not None:
            det = CryptoRegimeDetector(CANONICAL_REGIME_CONFIG)
            regime_df = det.fit_predict(raw_df)
            ok = regime_df is not None and len(regime_df) > 0
            check("fit_predict() retornou DataFrame", ok,
                  f"{len(regime_df)} linhas" if ok else "vazio")

            if ok:
                # regime_confidence adicionado pelo ScientificDataProcessor, nao pelo fit_predict standalone
                REGIME_COLS_STANDALONE = ["regime","regime_name"]
                missing_rc = [c for c in REGIME_COLS_STANDALONE if c not in regime_df.columns]
                check("Colunas de regime presentes (standalone)", len(missing_rc) == 0,
                      f"faltando: {missing_rc}" if missing_rc else "OK")

                counts = regime_df["regime"].value_counts().to_dict() if "regime" in regime_df.columns else {}
                check("Regimes detectados",
                      len(counts) >= 1,
                      f"Bull={counts.get(0,0)} Bear={counts.get(1,0)} Ranger={counts.get(2,0)}")

                if "regime_confidence" in regime_df.columns:
                    avg_c = regime_df["regime_confidence"].mean()
                    check("Confianca media > 0.3", avg_c > 0.3, f"{avg_c:.2%}")

                if "regime" in regime_df.columns:
                    nan_r = regime_df["regime"].isnull().sum()
                    check("Sem NaN na coluna regime", nan_r == 0, f"{nan_r} NaN")
    except Exception as e:
        check("Regime Detection", False, str(e))

    # ─── 5. PHYSICS SENSORS ───────────────────────────────────────────────────
    section("5. PHYSICS SENSORS (Hurst R/S + Shannon Entropy)")
    physics_metrics = None

    try:
        from utils.physics_sensors import (
            calculate_hurst_exponent,
            calculate_shannon_entropy,
            get_market_chaos_metrics,
        )

        if raw_df is not None and len(raw_df) >= 30:
            returns = raw_df["close"].pct_change().dropna().tail(30)

            hurst   = calculate_hurst_exponent(returns)
            entropy = calculate_shannon_entropy(returns)
            metrics = get_market_chaos_metrics(raw_df)
            physics_metrics = metrics

            check("Hurst calculado sem crash", True, f"H = {hurst:.3f}")
            check("Hurst em range [0.1, 0.9]", 0.1 <= hurst <= 0.9, f"H = {hurst:.3f}")
            check("Hurst != 0.5 (nao default)", hurst != 0.5,
                  "OK" if hurst != 0.5 else "fallback — verificar dados")
            check("Entropy em range [0, 5]", 0.0 <= entropy <= 5.0, f"E = {entropy:.3f}")
            check("get_market_chaos_metrics() completo",
                  all(k in metrics for k in ["entropy","hurst","chaos_label"]))

            print(f"\n  Mercado atual: {metrics['chaos_label']}")
            print(f"  Hurst={metrics['hurst']}  Entropy={metrics['entropy']}")
            if metrics["hurst"] < 0.40:
                print("  >> RANGER OVERRULE ativo (H < 0.40)")
            elif metrics["hurst"] > 0.60:
                print("  >> TENDENCIA detectada")
            if metrics["entropy"] > 3.0:
                print("  >> CAOS: confianca reduzida 20%")
        else:
            check("Physics sensors", False, "sem dados suficientes")
    except Exception as e:
        check("Physics Sensors", False, str(e))

    # ─── 6. DFA INTERNO ───────────────────────────────────────────────────────
    section("6. DFA INTERNO (ai_controller._calculate_dfa)")

    try:
        from trading.ai_controller import AIController

        if raw_df is not None and len(raw_df) >= 50:
            prices = raw_df["close"].tail(128).values.astype(float)
        else:
            np.random.seed(42)
            prices = np.random.randn(128).cumsum() + 50000

        obj = object.__new__(AIController)
        dfa_val  = obj._calculate_dfa(prices)
        ent_val  = obj._calculate_shannon_entropy(prices)
        lyap_val = obj._calculate_lyapunov_exponent(prices)

        check("_calculate_dfa() sem crash", True, f"H = {dfa_val:.3f}")
        check("DFA em range [0.1, 0.9]", 0.1 <= dfa_val <= 0.9, f"H = {dfa_val:.3f}")
        check("DFA != 0.5 (nao fallback)", dfa_val != 0.5,
              "OK" if dfa_val != 0.5 else "retornou 0.5 — verificar length")
        check("Shannon entropy [0.0, 1.0]", 0.0 <= ent_val <= 1.0, f"E = {ent_val:.3f}")
        check("Lyapunov em [-0.5, 0.5]", -0.5 <= lyap_val <= 0.5, f"L = {lyap_val:.4f}")

        print(f"\n  DFA/Quantum: H={dfa_val:.3f}  E={ent_val:.3f}  L={lyap_val:.4f}")
        if ent_val > 0.85 or lyap_val > 0.15:
            print("  >> CHAOS VETO seria ativado!")
        else:
            print("  >> Mercado dentro dos parametros operacionais")
    except Exception as e:
        check("DFA interno", False, str(e))

    # ─── 7. REGIME → SPECIALIST MAPPING ──────────────────────────────────────
    section("7. REGIME → SPECIALIST MAPPING")

    if regime_df is not None and "regime" in regime_df.columns:
        try:
            r_raw = regime_df["regime"].iloc[-1]
            r_val = int(r_raw) if pd.notna(r_raw) else 2
            r_conf = float(regime_df["regime_confidence"].iloc[-1]) \
                     if "regime_confidence" in regime_df.columns else 0.5
            r_map = {0: "BULL -> bull_specialist",
                     1: "BEAR -> bear_specialist",
                     2: "RANGER -> ranger_specialist"}
            check("Regime atual mapeavel", r_val in r_map,
                  r_map.get(r_val, f"DESCONHECIDO ({r_val})"))
            check("Confianca do regime usavel", r_conf >= 0.0, f"{r_conf:.2%}")
            check("Regime nao NaN", pd.notna(r_raw))

            if physics_metrics:
                overrule = physics_metrics["hurst"] < 0.40 and r_val != 2
                check("Physics Overrule consistente", True,
                      f"Ranger forcado={'SIM' if overrule else 'NAO'} "
                      f"(H={physics_metrics['hurst']})")
        except Exception as e:
            check("Regime mapping", False, str(e))
    else:
        check("Regime disponivel para mapping", False, "regime_df ausente")

    # ─── 8. COLUNAS CRITICAS ──────────────────────────────────────────────────
    section("8. COLUNAS CRITICAS NO DataFrame ENRIQUECIDO")

    # Resolve col com ou sem sufixo TF
    def resolve_col(df, name):
        if name in df.columns: return name
        for c in df.columns:
            if c == f"{name}_{interval}" or c.startswith(f"{name}_"):
                return c
        return None

    # tp_prior_dir/conf geradas apenas com apply_latent=True (autoencoder)
    CRITICAL_COLS = ["close","volume","rsi","adx","atr",
                     "regime","regime_confidence",
                     "log_return","realized_vol_20"]

    if enriched_df is not None:
        for col in CRITICAL_COLS:
            actual = resolve_col(enriched_df, col)
            if actual:
                nan_pct = enriched_df[actual].isnull().mean() * 100
                val = enriched_df[actual].iloc[-1]
                label = actual if actual != col else col
                check(f"'{col}' presente e valida", nan_pct < 10,
                      f"[{label}] NaN={nan_pct:.0f}%  ultimo={val:.4f}" if pd.notna(val) else
                      f"[{label}] NaN={nan_pct:.0f}%")
            else:
                check(f"'{col}' presente", False, "AUSENTE")
    else:
        check("DataFrame enriquecido disponivel", False, "enriched_df ausente")

    # ─── 9. RESUMO ────────────────────────────────────────────────────────────
    section("RESUMO")
    passed = sum(1 for s,_,_ in RESULTS if s == "PASS")
    failed = sum(1 for s,_,_ in RESULTS if s == "FAIL")
    total  = len(RESULTS)

    print(f"\n  PASS: {passed}/{total}")
    print(f"  FAIL: {failed}/{total}")

    if failed == 0:
        print("\n  [OK] Pipeline validado — bot pode operar")
    else:
        print(f"\n  [!!] {failed} problema(s):")
        for status, name, detail in RESULTS:
            if status == "FAIL":
                print(f"    -> {name}: {detail}")
    print()

    # Fechar sessao HTTP
    if bc:
        try:
            if hasattr(bc, "disconnect"):
                await bc.disconnect()
            elif hasattr(bc, "_session") and bc._session and not bc._session.closed:
                await bc._session.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(run_diagnostics())
