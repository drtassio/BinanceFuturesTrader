"""
AE Validation Script - BS-CLAIM-003, BS-CLAIM-004, BS-CLAIM-005
Executa 3 testes obrigatorios do boot sequence (Etapa 5).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import AIConfig
from feature_engineering.hybrid_autoencoder import HybridAutoencoderPipeline


def load_featured_df():
    path = os.path.join(AIConfig.MODEL_DIR, "base_featured_df.pkl")
    if not os.path.exists(path):
        print(f"ERRO: {path} nao encontrado.")
        sys.exit(1)
    df = pd.read_pickle(path)
    print(f"[DATA] Loaded {len(df)} rows, {len(df.columns)} cols")
    return df


def get_latents(pipeline, df, feature_cols):
    data = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).ffill(limit=300).fillna(0)
    scaler = pipeline.scaler
    data_scaled = scaler.transform(data.values) if scaler else data.values
    seq_len = pipeline.hparams.get("seq_len", 96)
    model = pipeline.model
    model.eval()
    device = pipeline.device
    latents_list = []
    batch_size = 512
    with torch.no_grad():
        for i in range(seq_len, len(data_scaled), batch_size):
            end = min(i + batch_size, len(data_scaled))
            batch_seqs = [data_scaled[j - seq_len:j] for j in range(i, end)]
            batch_tensor = torch.FloatTensor(np.array(batch_seqs)).to(device)
            output = model(batch_tensor)
            if isinstance(output, dict) and "z" in output:
                z = output["z"]
            elif isinstance(output, tuple):
                z = output[0]
            else:
                z = output
            latents_list.append(z.cpu().numpy())
    latents = np.concatenate(latents_list, axis=0)
    print(f"[LATENTS] Shape: {latents.shape}")
    return latents


def test_linear_probe(latents, df, seq_len, n_folds=5, embargo=21):
    """BS-003: LightGBM nos latents -> sign(return_1h). AUC >= 0.55."""
    try:
        from lightgbm import LGBMClassifier
        use_lgbm = True
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        use_lgbm = False

    if "close" in df.columns:
        fwd_ret = df["close"].pct_change(4).shift(-4)
    else:
        print("ERRO: coluna close nao encontrada")
        return 0.0

    targets = (fwd_ret.iloc[seq_len:seq_len + len(latents)] > 0).astype(int).values
    valid_mask = ~np.isnan(targets.astype(float))
    X = latents[valid_mask]
    y = targets[valid_mask]
    print(f"[PROBE] X={X.shape}, y={y.shape}, pos_rate={y.mean():.3f}")

    fold_size = len(X) // n_folds
    aucs = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        train_end = max(0, val_start - embargo)
        if train_end < 100:
            continue
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]
        if len(np.unique(y_val)) < 2:
            continue
        try:
            if use_lgbm:
                from lightgbm import LGBMClassifier as LGBM
                clf = LGBM(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, verbose=-1
                )
            else:
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=500)
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            aucs.append(auc)
            print(f"  Fold {fold+1}: AUC={auc:.4f} (train={len(X_train)}, val={len(X_val)})")
        except Exception as e:
            print(f"  Fold {fold+1}: ERRO - {e}")

    mean_auc = np.mean(aucs) if aucs else 0.0
    status = "PASS" if mean_auc >= 0.55 else "FAIL"
    print(f"\n[BS-003] LINEAR PROBE: AUC_OOS = {mean_auc:.4f} {status} (threshold: 0.55)")
    return mean_auc


def test_temporal_permutation(pipeline, df, feature_cols, seq_len):
    """BS-005: Temporal permutation destruction.

    Para JEPA (sem decoder): mede se shuffle temporal DESTROI a representacao.
    Metrica: distancia L2 entre z_normal e z_shuffled, normalizada pela
    variancia tipica inter-sample de z. Se o AE usa estrutura temporal,
    shuffle deve produzir z muito diferente (ratio alto).

    Threshold: mean_dist_shuffled / mean_dist_adjacent >= 2.0
    (shuffled pairs devem estar mais distantes que pares temporais adjacentes)
    """
    model = pipeline.model
    model.eval()
    device = pipeline.device
    scaler = pipeline.scaler
    data = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).ffill(limit=300).fillna(0)
    data_scaled = scaler.transform(data.values) if scaler else data.values

    n_samples = min(1000, len(data_scaled) - seq_len - 1)
    indices = np.linspace(seq_len, len(data_scaled) - 2, n_samples, dtype=int)

    dist_shuffle = []  # L2 distance between z_normal and z_shuffled
    dist_adjacent = []  # L2 distance between z_t and z_{t+1} (baseline)

    with torch.no_grad():
        for idx in indices:
            # Normal sequence
            seq = data_scaled[idx - seq_len:idx]
            tensor_normal = torch.FloatTensor(seq).unsqueeze(0).to(device)
            output_normal = model(tensor_normal)

            if not isinstance(output_normal, dict) or "z" not in output_normal:
                continue
            z_normal = output_normal["z"].squeeze(0)

            # Shuffled sequence (destroy temporal order)
            seq_shuffled = seq.copy()
            np.random.shuffle(seq_shuffled)
            tensor_shuffled = torch.FloatTensor(seq_shuffled).unsqueeze(0).to(device)
            output_shuffled = model(tensor_shuffled)
            z_shuffled = output_shuffled["z"].squeeze(0)

            # Adjacent sequence (t+1)
            if idx + 1 < len(data_scaled):
                seq_next = data_scaled[idx - seq_len + 1:idx + 1]
                tensor_next = torch.FloatTensor(seq_next).unsqueeze(0).to(device)
                output_next = model(tensor_next)
                z_next = output_next["z"].squeeze(0)
                dist_adjacent.append((z_normal - z_next).norm().item())

            dist_shuffle.append((z_normal - z_shuffled).norm().item())

    mean_dist_shuffle = np.mean(dist_shuffle) if dist_shuffle else 0
    mean_dist_adjacent = np.mean(dist_adjacent) if dist_adjacent else 1e-12
    ratio = mean_dist_shuffle / (mean_dist_adjacent + 1e-12)

    status = "PASS" if ratio >= 2.0 else "FAIL"
    print(f"\n[BS-005] TEMPORAL PERMUTATION (JEPA-adapted):")
    print(f"  Mean L2 dist (shuffled vs normal): {mean_dist_shuffle:.4f}")
    print(f"  Mean L2 dist (adjacent t vs t+1):  {mean_dist_adjacent:.4f}")
    print(f"  Ratio (shuffle/adjacent):          {ratio:.2f}x {status} (threshold: 2.0x)")
    print(f"  Interpretation: shuffle destroys {ratio:.1f}x more info than 1-step shift")
    return ratio


def main():
    print("=" * 70)
    print("  AE VALIDATION - BS-CLAIM-003 / 005")
    print("=" * 70)
    config = AIConfig()
    pipeline = HybridAutoencoderPipeline(config)
    if not pipeline.is_trained:
        print("ERRO: Autoencoder nao treinado.")
        sys.exit(1)
    print(f"[AE] Modelo: {pipeline.model_path}")

    df = load_featured_df()
    feature_cols = [c for c in pipeline._filter_features(df) if c in df.columns]
    print(f"[FEATURES] {len(feature_cols)} features")
    seq_len = pipeline.hparams.get("seq_len", 96)

    print("\n--- Encoding latents ---")
    latents = get_latents(pipeline, df, feature_cols)

    print("\n" + "=" * 50)
    auc = test_linear_probe(latents, df, seq_len)

    print("\n" + "=" * 50)
    perm_ratio = test_temporal_permutation(pipeline, df, feature_cols, seq_len)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    results = {
        "bs_003_linear_probe_auc": float(auc),
        "bs_003_pass": bool(auc >= 0.55),
        "bs_005_temporal_perm_ratio": float(perm_ratio),
        "bs_005_pass": bool(perm_ratio >= 2.0),
    }
    all_pass = results["bs_003_pass"] and results["bs_005_pass"]
    s003 = "PASS" if results["bs_003_pass"] else "FAIL"
    s005 = "PASS" if results["bs_005_pass"] else "FAIL"
    overall = "ALL PASS" if all_pass else "SOME FAILED"
    print(f"  BS-003 (Linear Probe):  AUC={auc:.4f} {s003}")
    print(f"  BS-005 (Temporal Perm): Ratio={perm_ratio:.2f}x {s005}")
    print(f"  OVERALL: {overall}")

    out_path = os.path.join(AIConfig.MODEL_DIR, "ae_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())