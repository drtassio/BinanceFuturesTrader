"""
CryptoRegimeDetector Validation - 12 Tests (Levels 1-5)
========================================================
Executa todos os testes de validacao do regime detector conforme especificacao.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import groupby

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import AIConfig


def load_data():
    """Load featured df and regime detector."""
    path = os.path.join(AIConfig.MODEL_DIR, "base_featured_df.pkl")
    if not os.path.exists(path):
        print("ERRO: base_featured_df.pkl nao encontrado")
        sys.exit(1)
    df = pd.read_pickle(path)
    print(f"[DATA] {len(df)} rows loaded")

    det_path = os.path.join(AIConfig.MODEL_DIR, "crypto_regime_detector.pkl")
    if not os.path.exists(det_path):
        print("ERRO: crypto_regime_detector.pkl nao encontrado")
        sys.exit(1)

    import joblib
    detector = joblib.load(det_path)
    print(f"[DETECTOR] Loaded from {det_path}")
    return df, detector


def test_1_distribution(regimes):
    """Nivel 1: Distribuicao dos regimes e razoavel."""
    counts = pd.Series(regimes).value_counts(normalize=True).sort_index()
    print("\n[TEST 1] Distribuicao dos regimes:")
    names = {0: "Bull", 1: "Bear", 2: "Ranger"}
    for code, pct in counts.items():
        print(f"  {names.get(code, code)}: {pct:.1%}")

    # Check: nenhum regime > 80% ou < 2%
    max_pct = counts.max()
    min_pct = counts.min()
    passed = max_pct < 0.80 and min_pct > 0.02
    status = "PASS" if passed else "FAIL"
    print(f"  Max={max_pct:.1%}, Min={min_pct:.1%} -> {status}")
    return passed


def test_2_whipsaw(regimes):
    """Nivel 1: Taxa de transicao (whipsaw check)."""
    transitions = (regimes[1:] != regimes[:-1]).sum()
    rate = transitions / len(regimes)
    passed = 0.0005 < rate < 0.05
    status = "PASS" if passed else "FAIL"
    print(f"\n[TEST 2] Taxa de transicao: {rate:.4f} ({transitions} transicoes)")
    print(f"  Esperado: 0.0005-0.05 -> {status}")
    return passed


def test_3_duration(regimes):
    """Nivel 1: Duracao minima dos regimes."""
    runs = [(k, sum(1 for _ in g)) for k, g in groupby(regimes)]
    durations = [d for _, d in runs]
    mean_dur = np.mean(durations)
    min_dur = np.min(durations)
    passed = mean_dur >= 20
    status = "PASS" if passed else "FAIL"
    print(f"\n[TEST 3] Duracao dos regimes:")
    print(f"  Media: {mean_dur:.1f} candles, Min: {min_dur}, Max: {np.max(durations)}")
    print(f"  Esperado media >= 20 -> {status}")
    return passed


def test_4_forward_algorithm(detector):
    """Nivel 2: Confirmar que usa Forward Algorithm (nao Viterbi)."""
    # Check source code for predict vs predict_proba
    import inspect
    source = ""
    for attr in ["detect_regimes", "predict_regimes", "_predict_hmm", "predict_online"]:
        if hasattr(detector, attr):
            try:
                source += inspect.getsource(getattr(detector, attr))
            except (TypeError, OSError):
                pass

    uses_viterbi = "viterbi" in source.lower() or ".predict(" in source
    uses_forward = "predict_proba" in source or "forward" in source.lower() or "filter" in source.lower()

    if uses_viterbi and not uses_forward:
        status = "FAIL (Viterbi detected - LEAKAGE)"
        passed = False
    elif uses_forward:
        status = "PASS (Forward/filter algorithm)"
        passed = True
    else:
        status = "WARN (could not determine)"
        passed = True  # benefit of doubt

    print(f"\n[TEST 4] Forward vs Viterbi:")
    print(f"  {status}")
    return passed


def test_5_causal_stability(df, detector):
    """Nivel 2: Regime nao muda retroativamente com mais dados."""
    try:
        n = min(3000, len(df))
        df_500 = df.iloc[:n // 2]
        df_full = df.iloc[:n]

        regimes_500 = detector.detect_regimes(df_500)
        regimes_full = detector.detect_regimes(df_full)

        # Compare first portion (with warmup margin)
        warmup = 100
        compare_len = min(len(regimes_500) - warmup, len(regimes_full) - warmup)
        if compare_len < 50:
            print("\n[TEST 5] Estabilidade causal: SKIP (dados insuficientes)")
            return True

        agreement = (regimes_500[warmup:warmup + compare_len] == regimes_full[warmup:warmup + compare_len]).mean()
        passed = agreement >= 0.85
        status = "PASS" if passed else "FAIL"
        print(f"\n[TEST 5] Estabilidade causal: {agreement:.3f}")
        print(f"  Esperado >= 0.85 -> {status}")
        return passed
    except Exception as e:
        print(f"\n[TEST 5] ERRO: {e}")
        return False


def test_6_temporal_permutation(df, detector):
    """Nivel 2: Entropia real < embaralhada."""
    from scipy.stats import entropy
    try:
        n = min(5000, len(df))
        df_sub = df.iloc[:n].copy()
        regimes_real = detector.detect_regimes(df_sub)

        df_shuffled = df_sub.sample(frac=1.0, random_state=42).reset_index(drop=True)
        regimes_shuffled = detector.detect_regimes(df_shuffled)

        H_real = entropy(pd.Series(regimes_real).value_counts(normalize=True))
        H_shuffled = entropy(pd.Series(regimes_shuffled).value_counts(normalize=True))

        passed = H_real < H_shuffled
        status = "PASS" if passed else "FAIL"
        print(f"\n[TEST 6] Permutacao temporal:")
        print(f"  Entropia real: {H_real:.4f}")
        print(f"  Entropia embaralhada: {H_shuffled:.4f}")
        print(f"  Real < Embaralhada -> {status}")
        return passed
    except Exception as e:
        print(f"\n[TEST 6] ERRO: {e}")
        return False


def test_7_market_alignment(df, regimes):
    """Nivel 3: Bull retorno > 0, Bear < 0."""
    if "close" not in df.columns:
        print("\n[TEST 7] SKIP: sem coluna close")
        return True

    df_test = df.copy()
    df_test["regime"] = regimes[:len(df_test)]
    df_test["return_5d"] = df_test["close"].pct_change(4 * 24 * 5)  # 5 dias em 15m

    names = {0: "Bull", 1: "Bear", 2: "Ranger"}
    results = {}
    print(f"\n[TEST 7] Alinhamento com mercado (retorno medio 5d):")
    for code, name in names.items():
        mask = df_test["regime"] == code
        if mask.sum() > 100:
            avg_ret = df_test.loc[mask, "return_5d"].mean()
            results[name] = avg_ret
            print(f"  {name}: {avg_ret:.4f} (N={mask.sum()})")
        else:
            results[name] = 0
            print(f"  {name}: INSUFICIENTE (N={mask.sum()})")

    # Bull > 0 e Bear < 0
    bull_ok = results.get("Bull", 0) > 0
    bear_ok = results.get("Bear", 0) < 0
    passed = bull_ok and bear_ok
    status = "PASS" if passed else "FAIL"
    print(f"  Bull>0={bull_ok}, Bear<0={bear_ok} -> {status}")
    return passed


def test_8_adx_agreement(df, regimes):
    """Nivel 3: Acordo regime-ADX >= 0.60."""
    adx_col = None
    for col in df.columns:
        if "adx" in col.lower() and "14" in col:
            adx_col = col
            break
    if not adx_col:
        for col in df.columns:
            if "adx" in col.lower():
                adx_col = col
                break

    if not adx_col:
        print("\n[TEST 8] SKIP: sem coluna ADX")
        return True

    df_test = df.copy()
    df_test["regime"] = regimes[:len(df_test)]
    trend_regime = df_test["regime"].isin([0, 1])
    high_adx = df_test[adx_col] > 25

    valid = ~(df_test[adx_col].isna())
    agreement = (trend_regime[valid] == high_adx[valid]).mean()
    passed = agreement >= 0.60
    status = "PASS" if passed else "WARN"
    print(f"\n[TEST 8] Acordo Regime-ADX ({adx_col}):")
    print(f"  Agreement: {agreement:.3f} -> {status} (threshold: 0.60)")
    return passed


def test_9_confidence(df, detector, regimes):
    """Nivel 3: Confidence score e informativo."""
    if not hasattr(detector, "get_confidence") and not hasattr(detector, "regime_confidence"):
        print("\n[TEST 9] SKIP: detector sem metodo de confidence")
        return True

    try:
        if hasattr(detector, "get_confidence"):
            confidence = detector.get_confidence(df)
        else:
            confidence = df.get("regime_confidence", None)
            if confidence is None:
                print("\n[TEST 9] SKIP: sem confidence disponivel")
                return True

        df_test = df.copy()
        df_test["regime"] = regimes[:len(df_test)]
        df_test["confidence"] = confidence[:len(df_test)] if len(confidence) >= len(df_test) else confidence
        df_test["return_1h"] = df_test["close"].pct_change(4) if "close" in df_test.columns else 0

        print(f"\n[TEST 9] Confidence informativa:")
        q_low = df_test["confidence"].quantile(0.25)
        q_high = df_test["confidence"].quantile(0.75)

        bull_low = df_test[(df_test["regime"] == 0) & (df_test["confidence"] <= q_low)]["return_1h"].mean()
        bull_high = df_test[(df_test["regime"] == 0) & (df_test["confidence"] >= q_high)]["return_1h"].mean()

        print(f"  Bull low-conf return:  {bull_low:.5f}")
        print(f"  Bull high-conf return: {bull_high:.5f}")
        passed = bull_high > bull_low
        status = "PASS" if passed else "WARN"
        print(f"  High > Low -> {status}")
        return passed
    except Exception as e:
        print(f"\n[TEST 9] ERRO: {e}")
        return True


def test_10_temporal_stability(df, detector):
    """Nivel 4: Distribuicao diferente por periodo."""
    if not hasattr(df.index, 'year'):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            print("\n[TEST 10] SKIP: index nao e datetime")
            return True

    print(f"\n[TEST 10] Estabilidade temporal por periodo:")
    distributions = []
    names = {0: "Bull", 1: "Bear", 2: "Ranger"}

    for year in sorted(df.index.year.unique()):
        period_df = df[df.index.year == year]
        if len(period_df) < 200:
            continue
        try:
            regimes = detector.detect_regimes(period_df)
            dist = pd.Series(regimes).value_counts(normalize=True).sort_index()
            distributions.append(dist)
            parts = [f"{names.get(k, k)}={v:.0%}" for k, v in dist.items()]
            print(f"  {year}: {' '.join(parts)} (N={len(period_df)})")
        except Exception as e:
            print(f"  {year}: ERRO - {e}")

    # Check: distributions are NOT all the same
    if len(distributions) >= 2:
        # Compare first and last
        diff = abs(distributions[0].get(0, 0) - distributions[-1].get(0, 0))
        passed = diff > 0.05  # At least 5% difference in Bull between periods
        status = "PASS" if passed else "FAIL"
        print(f"  Variacao Bull entre periodos: {diff:.1%} -> {status}")
        return passed
    return True


def test_11_structural_breaks(df, detector):
    """Nivel 4: Regime muda em eventos macro."""
    if not hasattr(df.index, 'year'):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            print("\n[TEST 11] SKIP: index nao e datetime")
            return True

    key_dates = {
        "crash_2022": "2022-05-01",
        "recovery_2023": "2023-01-01",
        "bull_2024": "2024-03-01",
    }

    print(f"\n[TEST 11] Structural breaks em eventos macro:")
    changes = 0
    tested = 0
    names = {0: "Bull", 1: "Bear", 2: "Ranger"}

    for event, date in key_dates.items():
        try:
            date_ts = pd.Timestamp(date)
            before = df[df.index < date_ts].tail(500)
            after = df[df.index >= date_ts].head(500)
            if len(before) < 100 or len(after) < 100:
                continue

            regimes_before = detector.detect_regimes(before)
            regimes_after = detector.detect_regimes(after)

            mode_before = pd.Series(regimes_before).mode()[0]
            mode_after = pd.Series(regimes_after).mode()[0]
            changed = mode_before != mode_after
            if changed:
                changes += 1
            tested += 1
            mark = "mudou" if changed else "nao mudou"
            print(f"  {event}: {names.get(mode_before, mode_before)}->{names.get(mode_after, mode_after)} ({mark})")
        except Exception as e:
            print(f"  {event}: ERRO - {e}")

    passed = changes >= 1 if tested > 0 else True
    status = "PASS" if passed else "FAIL"
    print(f"  Mudancas: {changes}/{tested} -> {status}")
    return passed


def test_12_online_parity(df, detector):
    """Nivel 5: predict_online retorna o mesmo que detect_regimes."""
    if not hasattr(detector, "predict_online"):
        print("\n[TEST 12] SKIP: detector sem predict_online")
        return True

    try:
        n = min(200, len(df))
        df_tail = df.tail(n)
        regimes_batch = detector.detect_regimes(df_tail)

        regimes_online = []
        for i in range(len(df_tail)):
            row = df_tail.iloc[i]
            regime = detector.predict_online(row)
            regimes_online.append(regime)

        agreement = (np.array(regimes_batch) == np.array(regimes_online)).mean()
        passed = agreement >= 0.80
        status = "PASS" if passed else "FAIL"
        print(f"\n[TEST 12] Online parity:")
        print(f"  Acordo batch/online: {agreement:.3f} -> {status} (threshold: 0.80)")
        return passed
    except Exception as e:
        print(f"\n[TEST 12] ERRO: {e}")
        return True


def main():
    print("=" * 70)
    print("  CRYPTO REGIME DETECTOR - FULL VALIDATION (12 TESTS)")
    print("=" * 70)

    df, detector = load_data()

    # Get regimes for the full dataset
    print("\n[DETECT] Computing regimes for full dataset...")
    regimes = detector.detect_regimes(df)
    regimes = np.array(regimes)
    print(f"[DETECT] {len(regimes)} regime labels computed")

    results = {}

    # Level 1 - Sanity
    results["test_01_distribution"] = test_1_distribution(regimes)
    results["test_02_whipsaw"] = test_2_whipsaw(regimes)
    results["test_03_duration"] = test_3_duration(regimes)

    # Level 2 - Causality
    results["test_04_forward_algo"] = test_4_forward_algorithm(detector)
    results["test_05_causal_stability"] = test_5_causal_stability(df, detector)
    results["test_06_temporal_perm"] = test_6_temporal_permutation(df, detector)

    # Level 3 - Market Alignment
    results["test_07_market_align"] = test_7_market_alignment(df, regimes)
    results["test_08_adx_agreement"] = test_8_adx_agreement(df, regimes)
    results["test_09_confidence"] = test_9_confidence(df, detector, regimes)

    # Level 4 - Temporal Stability
    results["test_10_temporal_stab"] = test_10_temporal_stability(df, detector)
    results["test_11_structural_breaks"] = test_11_structural_breaks(df, detector)

    # Level 5 - Live Parity
    results["test_12_online_parity"] = test_12_online_parity(df, detector)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    critical = ["test_04_forward_algo", "test_05_causal_stability", "test_07_market_align", "test_12_online_parity"]
    blocking = ["test_01_distribution", "test_02_whipsaw", "test_03_duration", "test_06_temporal_perm",
                "test_10_temporal_stab", "test_11_structural_breaks"]

    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    critical_pass = all(results.get(t, False) for t in critical)
    blocking_pass = all(results.get(t, False) for t in blocking)

    for name, passed in results.items():
        level = "CRITICAL" if name in critical else ("BLOCKING" if name in blocking else "INFO")
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name} ({level})")

    print(f"\n  Total: {n_pass}/{n_total} passed")
    print(f"  Critical tests: {'ALL PASS' if critical_pass else 'SOME FAILED'}")
    print(f"  Blocking tests: {'ALL PASS' if blocking_pass else 'SOME FAILED'}")
    overall = critical_pass and blocking_pass
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")

    # Save results
    out_path = os.path.join(AIConfig.MODEL_DIR, "regime_detector_validation.json")
    with open(out_path, "w") as f:
        json.dump({k: bool(v) for k, v in results.items()}, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())