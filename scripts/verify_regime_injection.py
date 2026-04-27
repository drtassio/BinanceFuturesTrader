"""
Script de Diagnóstico: Verificar Injeção de Regime no DataFrame

Este script verifica se o CryptoRegimeDetector está funcionando corretamente
e se as colunas de regime estão sendo injetadas com valores dinâmicos.

Uso:
    python scripts/verify_regime_injection.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 70)
    print("[DIAGNOSTICO] Verificacao de Injecao de Regime")
    print("=" * 70)
    
    # 1. Verificar se CryptoRegimeDetector está treinado
    print("\n[1] Verificando CryptoRegimeDetector...")
    model_path = os.path.join(os.getcwd(), "models_ai", "crypto_regime_detector.pkl")
    
    if not os.path.exists(model_path):
        print(f"[ERRO] Arquivo nao encontrado: {model_path}")
        print("   -> Execute o treinamento do CryptoRegimeDetector primeiro!")
        return False
    
    file_size = os.path.getsize(model_path)
    print(f"[OK] Arquivo encontrado: {model_path} ({file_size / 1024 / 1024:.2f} MB)")
    
    # 2. Carregar detector e verificar status
    try:
        from feature_engineering.crypto_regime_detector import CryptoRegimeDetector
        detector = CryptoRegimeDetector.load(model_path)
        
        if detector and detector.is_trained:
            print("[OK] CryptoRegimeDetector esta TREINADO")
        else:
            print("[ERRO] CryptoRegimeDetector esta carregado mas NAO TREINADO!")
            return False
    except Exception as e:
        print(f"[ERRO] Erro ao carregar detector: {e}")
        return False
    
    # 3. Carregar DataFrame de cache
    print("\n[2] Carregando DataFrame de cache...")
    cache_path = os.path.join(os.getcwd(), "models_ai", "base_featured_df.pkl")
    
    if not os.path.exists(cache_path):
        print(f"[WARN] Cache nao encontrado: {cache_path}")
        print("   -> Tentando criar DataFrame sintetico para teste...")
        
        # Cria DataFrame sintético minimal
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
        df = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 101,
            'low': np.random.randn(1000).cumsum() + 99,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.abs(np.random.randn(1000)) * 1000,
        }, index=dates)
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
    else:
        df = pd.read_pickle(cache_path)
        print(f"[OK] DataFrame carregado: {len(df)} linhas, {len(df.columns)} colunas")
    
    # 4. Verificar colunas de regime existentes
    print("\n[3] Verificando colunas de regime no DataFrame...")
    regime_cols = ['regime_confidence', 'regime_val', 'tp_prior_conf', 'tp_prior_dir']
    
    for col in regime_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"   {col}: mean={mean_val:.4f}, std={std_val:.4f}")
            
            if col in ['regime_confidence', 'tp_prior_conf'] and std_val < 0.01:
                print(f"   [WARN] {col} esta praticamente constante!")
        else:
            print(f"   {col}: NAO ENCONTRADA")
    
    # 5. Aplicar detector ao DataFrame
    print("\n[4] Aplicando CryptoRegimeDetector ao DataFrame...")
    try:
        regime_df = detector.predict(df)
        
        mean_conf = regime_df['confidence'].mean()
        std_conf = regime_df['confidence'].std()
        regime_counts = regime_df['regime'].value_counts()
        
        print(f"[OK] Predicao concluida:")
        print(f"   - Confianca media: {mean_conf:.4f}")
        print(f"   - Confianca std: {std_conf:.4f}")
        print(f"   - Distribuicao de regimes:")
        for regime, count in regime_counts.items():
            regime_name = {0: 'Bull', 1: 'Bear', 2: 'Ranger'}.get(regime, f'Unknown({regime})')
            print(f"     - {regime_name}: {count} ({count/len(regime_df)*100:.1f}%)")
        
        if std_conf < 0.01:
            print("\n[PROBLEMA] Confianca quase constante!")
            print("   -> O detector pode nao estar funcionando corretamente.")
            return False
        else:
            print("\n[SUCESSO] Detector esta gerando valores dinamicos!")
            
    except Exception as e:
        print(f"[ERRO] Erro ao aplicar detector: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Simular enriquecimento
    print("\n[5] Simulando enriquecimento do DataFrame...")
    
    # Simula o que _enrich_df_with_trend_predictions faz
    df['regime_confidence'] = regime_df['confidence'].reindex(df.index).fillna(0.5)
    df['regime_val'] = regime_df['regime'].reindex(df.index).fillna(2).astype(int)
    
    # Map direction
    conditions = [
        df['regime_val'] == 0,  # Bull
        df['regime_val'] == 1,  # Bear
        df['regime_val'] >= 2   # Ranger
    ]
    choices = [
        df['regime_confidence'],   # Positive for Long
        -df['regime_confidence'],  # Negative for Short
        0.0                        # Neutral
    ]
    df['tp_prior_dir'] = np.select(conditions, choices, default=0.0)
    df['tp_prior_conf'] = df['regime_confidence']
    
    print("\n[6] Verificacao final das colunas enriquecidas:")
    for col in ['tp_prior_dir', 'tp_prior_conf']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"   {col}: mean={mean_val:.4f}, std={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
    
    print("\n" + "=" * 70)
    print("[SUCESSO] DIAGNOSTICO COMPLETO - Injecao de regime esta funcionando!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

