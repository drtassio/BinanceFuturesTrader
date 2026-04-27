
import pandas as pd
import numpy as np
import os
import joblib
from feature_engineering.crypto_regime_detector import CryptoRegimeDetector

def main():
    print("🔬 [REIDOR DE REGIME] Validando CryptoRegimeDetector...")
    
    # 1. Carregar detector treinado
    model_path = os.path.join(os.getcwd(), "models_ai", "crypto_regime_detector.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado em {model_path}")
        return
        
    detector = CryptoRegimeDetector.load(model_path)
    print(f"✅ Detector carregado. Treinado: {detector.is_trained}")
    detector.config.confidence_threshold = 0.75
    print(f"🔧 Alterado confidence_threshold para: {detector.config.confidence_threshold}")
    
    # 2. Carregar dados de teste (15m)
    df_path = os.path.join(os.getcwd(), "logs", "historical_data", "BTCUSDT_15m_historical.pkl")
    df = pd.read_pickle(df_path)
    print(f"📊 Dados carregados: {len(df)} linhas (15m)")
    
    # 3. Predição e Validação
    print("⚙️ Executando predição de regime...")
    results = detector.predict(df)
    
    # 4. Estatísticas básica
    counts = results['regime'].value_counts()
    output = "\n📈 Distribuição de Regimes:\n"
    for code, count in counts.items():
        name = detector.REGIME_NAMES.get(code, "Unknown")
        output += f"   - {name} ({code}): {count} ({count/len(results):.1%})\n"
        
    # 5. Validação científica (Forward returns manuais)
    print("\n🧪 Validando Alpha dos Regimes...")
    
    horizons = [1, 4, 8, 16] # 15m, 1h, 2h, 4h
    df['returns'] = df['close'].pct_change().shift(-1) # Próximo candle
    
    alpha_results = []
    for h in horizons:
        fwd_ret = df['close'].shift(-h) / df['close'] - 1
        
        for code, name in detector.REGIME_NAMES.items():
            mask = results['regime'] == code
            avg_ret = fwd_ret[mask].mean()
            alpha_results.append({
                "Horizon": f"{h*15}m",
                "Regime": name,
                "Avg Fut Return": avg_ret
            })
            
    alpha_df = pd.DataFrame(alpha_results)
    print(alpha_df.to_string(index=False))
    
    # 6. Validação científica (Detector metrics)
    metrics = detector.validate_regimes()
    
    output += "\n📊 [ALPHA ANALYSIS]\n"
    output += alpha_df.to_string(index=False)
    
    output += "\n\n✅ Conclusão da Validação:\n"
    output += f"   - Persistence: {metrics['avg_persistence_bars']:.1f} bars\n"
    output += f"   - Model Agreement: {metrics['avg_model_agreement']:.1%}\n"
    output += f"   - Median Confidence: {metrics['mean_confidence']:.1%}\n"
    
    # Validação manual de Alpha
    bull_1h = alpha_df[(alpha_df['Horizon'] == '60m') & (alpha_df['Regime'] == 'Bull')]['Avg Fut Return'].values[0]
    bear_1h = alpha_df[(alpha_df['Horizon'] == '60m') & (alpha_df['Regime'] == 'Bear')]['Avg Fut Return'].values[0]
    
    output += f"   - Forward Return Bull (1h): {bull_1h:.4%}\n"
    output += f"   - Forward Return Bear (1h): {bear_1h:.4%}\n"
    
    valid = bull_1h > bear_1h
    output += f"   - Financial Validity (Bull > Bear): {'✅ PASSED' if valid else '❌ FAILED'}\n"
    
    if not valid:
        output += "\n🚨 [ERRO CRÍTICO] O detector de regime não tem valor preditivo!\n"
        output += "   Bull Return <= Bear Return. Isso confunde o agente e impede o aprendizado.\n"
        
    print(output)
    with open("regime_validation.txt", "w", encoding="utf-8") as f:
        f.write(output)

if __name__ == "__main__":
    main()
