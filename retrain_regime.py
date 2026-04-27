
import pandas as pd
import numpy as np
import os
import joblib
from feature_engineering.crypto_regime_detector import CryptoRegimeDetector, RegimeConfig

def main():
    print("🧠 [RE-TREINAMENTO] Atualizando CryptoRegimeDetector com Alpha-Mapping...")
    
    # 1. Carregar dados históricos para treino
    df_path = os.path.join(os.getcwd(), "logs", "historical_data", "BTCUSDT_15m_historical.pkl")
    if not os.path.exists(df_path):
        print(f"❌ Dados não encontrados em {df_path}")
        return
        
    df = pd.read_pickle(df_path)
    print(f"📊 Dados para treino: {len(df)} linhas")
    
    # 2. Configurar novo detector
    config = RegimeConfig(
        confidence_threshold=0.55,
        min_regime_duration=12,  # Aumentado para evitar ruído
        weight_hmm=0.40,
        weight_gmm=0.40,
        weight_adx=0.20
    )
    detector = CryptoRegimeDetector(config)
    
    # 3. Fit e Predict
    print("⚙️ Treinando modelos (HMM, GMM, ADX)...")
    results = detector.fit_predict(df)
    
    # 4. Validar Alpha Manualmente (Horizontal 1h = 4 candles)
    print("🧪 Validando novo Alpha (Manual)...")
    results['close'] = df['close']
    results['returns_1h'] = results['close'].shift(-4) / results['close'] - 1
    
    alpha_bull = results[results['regime'] == 0]['returns_1h'].mean()
    alpha_bear = results[results['regime'] == 1]['returns_1h'].mean()
    
    # 5. Salvar modelo atualizado
    model_path = os.path.join(os.getcwd(), "models_ai", "crypto_regime_detector.pkl")
    detector.save(model_path)
    print(f"💾 Novo modelo salvo em {model_path}")
    
    # 6. Report final
    report = f"\n✅ Resultados do Re-treinamento:\n"
    report += f"   - Forward Return Bull (1h): {alpha_bull:.4%}\n"
    report += f"   - Forward Return Bear (1h): {alpha_bear:.4%}\n"
    
    valid = alpha_bull > alpha_bear
    report += f"   - Financial Validity: {'✅ PASSED' if valid else '❌ FAILED'}\n"
    
    print(report)
    with open("alpha_retrain_results.txt", "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    main()
