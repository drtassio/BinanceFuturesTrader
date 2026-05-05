"""
tests/test_feature_contract.py
-------------------------------
Valida o contrato de features end-to-end:
  1. Pipeline gera todas as colunas esperadas
  2. Nenhuma coluna é constante (phantom / sempre-0)
  3. Scaler normaliza corretamente (sem NaN/Inf, range razoável)
  4. feature_columns do especialista existem no DataFrame enriquecido
  5. dist_to_max/min computados ao vivo e com valores reais
  6. Obs final tem shape correto e sem NaN

Rodar: python -X utf8 tests/test_feature_contract.py
"""
import sys
import os
import asyncio
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ── helpers ──────────────────────────────────────────────────────────────────
RESULTS = []

def chk(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((status, name, detail))
    sym = "+" if passed else "X"
    print(f"  [{sym}] {name:<65} {detail}")

def section(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ── Features esperadas após _filter_to_core_features ─────────────────────────
# Qualquer feature que bata com _TECH_PATTERNS nos TFs aceitos (15m, 1h)
EXPECTED_TECH_FEATURES = [
    # log_return / ranges
    "log_return_15m", "log_return_volume_15m",
    "hl_range_pct_15m", "oc_move_pct_15m",
    "log_return_1h", "hl_range_pct_1h", "oc_move_pct_1h",
    # fracdiff
    "close_frac_15m", "high_frac_15m", "low_frac_15m",
    "close_frac_1h",
    # vol
    "realized_vol_20_15m", "realized_vol_100_15m",
    "realized_vol_20_1h",
    # osciladores
    "rsi_15m", "rsi_1h",
    "stoch_k_15m", "stoch_d_15m",
    "stoch_k_1h", "stoch_d_1h",
    "macd_hist_15m", "macd_hist_1h",
    # tendência
    "adx_15m", "adx_1h",
    "plus_di_15m", "minus_di_15m",
    "plus_di_1h", "minus_di_1h",
    "ema_trend_15m", "ema_trend_1h",
    # volatilidade normalizada
    "atr_percentage_15m", "atr_percentage_1h",
    # volume
    "cmf_15m", "cmf_1h",
    # wicks
    "hc_wick_upper_15m", "lc_wick_lower_15m",
]

# Colunas phantom que NÃO devem aparecer no feature_columns
PHANTOM_FEATURES = [
    "trend_success_prob",
    "predicted_volatility_high",
    "predicted_breakout",
    "qpnl_24_p50", "qpnl_24_p90",
    "qpnl_48_p50", "qpnl_48_p90",
]

# Duplicatas que não devem estar em feature_columns
DUPLICATE_FEATURES = [
    "trend_pred_uptrend", "trend_pred_downtrend",
    "trend_pred_neutral", "trend_pred_confidence",
]


async def run_tests():
    # ── 1. CONFIG ─────────────────────────────────────────────────────────────
    section("1. CONFIG")
    cfg = ai_cfg = None
    try:
        from config.settings import TradingConfig, AIConfig
        cfg = TradingConfig()
        ai_cfg = AIConfig()
        chk("TradingConfig + AIConfig carregados", True)
    except Exception as e:
        chk("Config", False, str(e))
        return

    symbol = cfg.PRIMARY_PAIR
    interval = cfg.PRIMARY_TIMEFRAME_TRADING

    # ── 2. DADOS ──────────────────────────────────────────────────────────────
    section("2. FETCH KLINES")
    raw_df = None
    bc = None
    try:
        from trading.binance_connector import BinanceConnector
        bc = BinanceConnector(cfg)
        await bc.connect()
        raw_data = await bc.get_kline_data(symbol=symbol, interval=interval, limit=500)
        ok = raw_data is not None and len(raw_data) >= 200
        chk("500 candles obtidos", ok, f"{len(raw_data) if raw_data else 0} candles")
        if ok:
            cols = ["open_time","open","high","low","close","volume",
                    "close_time","quote_asset_volume","number_of_trades",
                    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"]
            raw_df = pd.DataFrame(raw_data, columns=cols)
            for c in ["open","high","low","close","volume"]:
                raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")
            raw_df.index = pd.to_datetime(raw_df["open_time"], unit="ms")
    except Exception as e:
        chk("Fetch klines", False, str(e))
        return

    if raw_df is None:
        print("  Sem dados — abortando.")
        return

    # ── 3. FEATURE PIPELINE ───────────────────────────────────────────────────
    section("3. FEATURE PIPELINE")
    enriched_df = None
    try:
        from feature_engineering.main import FeatureEngineeringPipeline
        from config.settings import AIConfig
        fp = FeatureEngineeringPipeline(ai_cfg)

        tf_dict = {interval: raw_df.copy()}
        secondary = getattr(cfg, "SECONDARY_TIMEFRAMES", ["1h"])
        for tf in secondary:
            tf_dict[tf] = raw_df.copy()

        enriched_df = await fp.create_features(
            raw_df_dict=tf_dict,
            symbol=symbol,
            primary_timeframe=interval,
            fit_scaler=False,
            apply_latent=False,
        )
        ok = enriched_df is not None and len(enriched_df) > 50
        chk("create_features() executou", ok,
            f"{len(enriched_df)} linhas, {len(enriched_df.columns)} colunas" if ok else "vazio")
    except Exception as e:
        chk("Feature pipeline", False, str(e))
        return

    if enriched_df is None:
        return

    # Features booleanas/flag que podem ser constantes em janelas curtas de mercado
    # (ex: ema_trend=1.0 durante BULL forte de 500 candles é correto, não phantom)
    BOOLEAN_FLAG_PATTERNS = (
        'ema_trend', 'psar_trend', 'adx_strong_trend', 'adx_weak_trend',
        'bb_squeeze', 'bb_expansion', 'rsi_overbought', 'rsi_oversold',
        'macd_cross', 'volume_pressure',
    )

    # ── 4. COLUNAS ESPERADAS PRESENTES ────────────────────────────────────────
    section("4. COLUNAS ESPERADAS PRESENTES")
    missing_expected = []
    for col in EXPECTED_TECH_FEATURES:
        present = col in enriched_df.columns
        if not present:
            missing_expected.append(col)
        else:
            nan_pct = enriched_df[col].isnull().mean() * 100
            chk(f"  '{col}' presente", nan_pct < 20, f"NaN={nan_pct:.0f}%  ultimo={enriched_df[col].dropna().iloc[-1]:.4f}" if enriched_df[col].dropna().size > 0 else f"NaN={nan_pct:.0f}%")

    if missing_expected:
        chk(f"Colunas faltando ({len(missing_expected)})", False, str(missing_expected[:10]))
    else:
        chk("Todas colunas esperadas presentes", True, f"{len(EXPECTED_TECH_FEATURES)} colunas OK")

    # ── 5. PHANTOM FEATURES AUSENTES ─────────────────────────────────────────
    section("5. PHANTOM FEATURES NAO DEVEM ESTAR EM feature_columns")
    try:
        from feature_engineering.scientific_data_processor import ScientificDataProcessor
        from specialists.trend_specialist import TrendFollowingEnv

        processor = ScientificDataProcessor()
        all_numeric_cols = enriched_df.select_dtypes(include=[np.number]).columns.tolist()
        filtered_cols = processor.filter_autoencoder_features(all_numeric_cols, enriched_df)

        # Simula _filter_to_core_features (instancia dummy)
        dummy_env = object.__new__(TrendFollowingEnv)
        core_cols = dummy_env._filter_to_core_features(filtered_cols)

        for phantom in PHANTOM_FEATURES:
            in_core = phantom in core_cols
            chk(f"  '{phantom}' AUSENTE do feature_columns", not in_core,
                "OK — removido" if not in_core else "ERRO — phantom presente!")

        for dup in DUPLICATE_FEATURES:
            in_core = dup in core_cols
            chk(f"  '{dup}' (duplicata) AUSENTE", not in_core,
                "OK" if not in_core else "ERRO — duplicata presente!")

        chk("Total features no core", 20 <= len(core_cols) <= 100,
            f"{len(core_cols)} features")
        print(f"\n  Core features ({len(core_cols)}): {sorted(core_cols)[:15]}...")

    except Exception as e:
        chk("Filtro de features", False, str(e))

    # ── 6. SEM COLUNAS CONSTANTES (std=0) ────────────────────────────────────
    section("6. SEM COLUNAS CONSTANTES (std=0 = sinal morto)")
    try:
        numeric_df = enriched_df.select_dtypes(include=[np.number])
        stds = numeric_df.std()
        constant_cols = stds[stds < 1e-9].index.tolist()
        # Exclui: (1) metadados de regime, (2) booleanas legítimas (podem ser constantes
        # em janelas curtas de mercado unidirecional — têm variância em datasets completos)
        real_phantom = [c for c in constant_cols
                        if not any(x in c for x in ['regime_name', 'ignore'])
                        and not any(p in c for p in BOOLEAN_FLAG_PATTERNS)]
        # Flags booleanas constantes — warning, não falha
        bool_constant = [c for c in constant_cols
                         if any(p in c for p in BOOLEAN_FLAG_PATTERNS)]
        if bool_constant:
            print(f"\n  [!] Flags booleanas constantes nesta janela (normal em mercado direcional): {bool_constant}")
        chk("Sem colunas phantom constantes", len(real_phantom) == 0,
            f"Phantoms: {real_phantom[:10]}" if real_phantom else "OK")

        # Verifica se phantom conhecidas são constantes
        for ph in PHANTOM_FEATURES:
            if ph in enriched_df.columns:
                std = enriched_df[ph].std()
                chk(f"  '{ph}' é constante (confirma remoção)", std < 1e-6,
                    f"std={std:.6f}")
    except Exception as e:
        chk("Constante check", False, str(e))

    # ── 7. NORMALIZAÇÃO DO SCALER ─────────────────────────────────────────────
    section("7. NORMALIZACAO (scaler aplicado corretamente)")
    try:
        from sklearn.preprocessing import RobustScaler
        import joblib

        models_dir = os.path.join(os.getcwd(), "models_ai")
        scaler = None

        # Tenta carregar scaler real
        for name in ["bull_specialist_scaler.joblib", "bull_scaler.joblib",
                     "feature_scaler_robust.joblib"]:
            path = os.path.join(models_dir, name)
            if os.path.exists(path):
                try:
                    scaler = joblib.load(path)
                    print(f"\n  Scaler carregado: {name}")
                    break
                except Exception:
                    pass

        if scaler is None:
            # Fit novo scaler nos dados atuais para testar pipeline
            print("\n  Scaler nao encontrado — fitando RobustScaler nos dados atuais")
            from feature_engineering.scientific_data_processor import ScientificDataProcessor
            from specialists.trend_specialist import TrendFollowingEnv
            proc = ScientificDataProcessor()
            all_cols = enriched_df.select_dtypes(include=[np.number]).columns.tolist()
            filtered = proc.filter_autoencoder_features(all_cols, enriched_df)
            dummy = object.__new__(TrendFollowingEnv)
            core = dummy._filter_to_core_features(filtered)
            available = [c for c in core if c in enriched_df.columns]
            if available:
                scaler = RobustScaler()
                scaler.fit(enriched_df[available].fillna(0).replace([np.inf, -np.inf], 0))
                scaler.feature_names_in_ = np.array(available)

        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            feat_names = list(scaler.feature_names_in_)
            available = [c for c in feat_names if c in enriched_df.columns]
            missing_from_df = [c for c in feat_names if c not in enriched_df.columns]

            chk("Colunas do scaler presentes no df",
                len(missing_from_df) == 0,
                f"faltando: {missing_from_df[:5]}" if missing_from_df else f"{len(feat_names)} OK")

            if available:
                sample = enriched_df[available].tail(50).fillna(0).replace([np.inf, -np.inf], 0)
                try:
                    scaled = scaler.transform(sample)
                    has_nan = np.isnan(scaled).any()
                    has_inf = np.isinf(scaled).any()
                    max_abs = np.abs(scaled).max()
                    chk("Scaled sem NaN", not has_nan,
                        f"{np.isnan(scaled).sum()} NaN" if has_nan else "OK")
                    chk("Scaled sem Inf", not has_inf,
                        f"{np.isinf(scaled).sum()} Inf" if has_inf else "OK")
                    chk("Scaled max_abs < 50 (sem outliers extremos)", max_abs < 50,
                        f"max_abs={max_abs:.2f}")

                    # Verifica features individualmente problemáticas
                    for i, col in enumerate(available):
                        col_std = scaled[:, i].std()
                        col_max = np.abs(scaled[:, i]).max()
                        is_boolean_flag = any(p in col for p in BOOLEAN_FLAG_PATTERNS)
                        if col_std < 1e-6 and not is_boolean_flag:
                            chk(f"  '{col}' variancia pos-scaler", False,
                                f"std={col_std:.2e} — coluna constante!")
                        if col_max > 30:
                            chk(f"  '{col}' outlier pos-scaler", False,
                                f"max_abs={col_max:.1f}")
                except Exception as e:
                    chk("Scaler.transform()", False, str(e))
        else:
            chk("Scaler disponivel", False, "nao encontrado")

    except Exception as e:
        chk("Normalizacao", False, str(e))

    # ── 8. DIST_TO_MAX/MIN COMPUTADOS AO VIVO ────────────────────────────────
    section("8. DIST_TO_MAX/MIN (memoria temporal ao vivo)")
    try:
        latest_row = enriched_df.iloc[-1]
        mem_live = {}
        for w in [20, 50]:
            h = enriched_df['high'].rolling(w).max().iloc[-1]
            l = enriched_df['low'].rolling(w).min().iloc[-1]
            c = float(latest_row.get('close', latest_row.get('close_15m', 1.0)))
            if 'close' not in enriched_df.columns and 'close_15m' in enriched_df.columns:
                c = float(enriched_df['close_15m'].iloc[-1])
            mem_live[f'dist_to_max_{w}'] = (c - h) / (h + 1e-9)
            mem_live[f'dist_to_min_{w}'] = (c - l) / (l + 1e-9)

        for k, v in mem_live.items():
            is_real = abs(v) > 1e-9
            in_range = -1.0 < v < 0.1
            chk(f"  '{k}' real (nao zero)", is_real, f"{v:.6f}")
            chk(f"  '{k}' em range [-1, 0.1]", in_range, f"{v:.6f}")

    except Exception as e:
        chk("dist_to_max/min", False, str(e))

    # ── 9. OBS FINAL — SHAPE E SANIDADE ───────────────────────────────────────
    section("9. OBSERVACAO FINAL (shape e sanidade)")
    try:
        from feature_engineering.scientific_data_processor import ScientificDataProcessor
        from specialists.trend_specialist import TrendFollowingEnv

        proc = ScientificDataProcessor()
        all_cols = enriched_df.select_dtypes(include=[np.number]).columns.tolist()
        filtered = proc.filter_autoencoder_features(all_cols, enriched_df)
        dummy = object.__new__(TrendFollowingEnv)
        core_cols = dummy._filter_to_core_features(filtered)
        available_core = [c for c in core_cols if c in enriched_df.columns]

        latest = enriched_df.iloc[-1]
        mkt = np.array([float(latest.get(c, 0.0)) for c in available_core], dtype=np.float32)
        mem = np.array([mem_live.get(k, 0.0) for k in
                        ['dist_to_max_20','dist_to_min_20','dist_to_max_50','dist_to_min_50']],
                       dtype=np.float32)

        import datetime
        ts = enriched_df.index[-1]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        agent_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        time_feats  = np.array([
            np.sin(2*np.pi*ts.hour/24), np.cos(2*np.pi*ts.hour/24),
            np.sin(2*np.pi*ts.dayofweek/7), np.cos(2*np.pi*ts.dayofweek/7)
        ], dtype=np.float32)
        prior = np.array([0.0, 0.5], dtype=np.float32)

        obs = np.concatenate([mkt, mem, agent_state, time_feats, prior])
        expected_dim = len(available_core) + 4 + 3 + 4 + 2

        chk("Obs shape bate formula", obs.shape[0] == expected_dim,
            f"{obs.shape[0]} == {len(available_core)}mkt+4mem+3agent+4time+2prior={expected_dim}")
        chk("Obs sem NaN", not np.isnan(obs).any(),
            f"{np.isnan(obs).sum()} NaN" if np.isnan(obs).any() else "OK")
        chk("Obs sem Inf", not np.isinf(obs).any(),
            f"{np.isinf(obs).sum()} Inf" if np.isinf(obs).any() else "OK")
        chk("Obs dim em range [60, 120]", 60 <= obs.shape[0] <= 120,
            f"dim={obs.shape[0]}")

        # Verifica slots de memória não são todos zero
        mem_nonzero = (np.abs(mem) > 1e-9).sum()
        chk("Memoria (dist_to_*) tem valores reais", mem_nonzero == 4,
            f"{mem_nonzero}/4 nao-zero")

        print(f"\n  Obs dim={obs.shape[0]}  mkt={len(available_core)}  "
              f"mem={mem}  prior={prior}")

    except Exception as e:
        chk("Obs final", False, str(e))

    # ── 10. FRACDIFF d PERSISTIDO ─────────────────────────────────────────────
    section("10. FRACDIFF d PERSISTIDO EM DISCO")
    try:
        from feature_engineering.native_indicators import NativeIndicators
        disk_vals = NativeIndicators._load_d_from_disk()
        chk("fracdiff_d_values.json existe", len(disk_vals) > 0,
            str(disk_vals) if disk_vals else "vazio — rode pipeline primeiro")

        for key, d_val in disk_vals.items():
            chk(f"  '{key}' d em [0.1, 1.0]", 0.1 <= d_val <= 1.0, f"d={d_val:.4f}")

    except Exception as e:
        chk("Fracdiff disk", False, str(e))

    # ── RESUMO ────────────────────────────────────────────────────────────────
    section("RESUMO")
    passed = sum(1 for s, _, _ in RESULTS if s == "PASS")
    failed = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    total  = len(RESULTS)

    print(f"\n  PASS: {passed}/{total}")
    print(f"  FAIL: {failed}/{total}")

    if failed == 0:
        print("\n  [OK] Contrato de features validado — agentes recebem obs corretas")
    else:
        print(f"\n  [!!] {failed} problema(s):")
        for status, name, detail in RESULTS:
            if status == "FAIL":
                print(f"    -> {name}: {detail}")
    print()

    if bc:
        try:
            if hasattr(bc, "disconnect"):
                await bc.disconnect()
            elif hasattr(bc, "_session") and bc._session and not bc._session.closed:
                await bc._session.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(run_tests())
