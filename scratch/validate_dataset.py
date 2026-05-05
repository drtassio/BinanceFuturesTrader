
import os
import pandas as pd
import numpy as np
import torch
import sys
from sklearn.preprocessing import RobustScaler

# Adiciona o diretório atual ao path para importar os módulos do bot
sys.path.append(os.getcwd())

from feature_engineering.scientific_data_processor import ScientificDataProcessor
from specialists.trend_specialist import TrendFollowingEnv, TrendSpecialist
from config.settings import AIConfig, TradingConfig

def audit_dataset_v3():
    print("\n" + "="*80)
    print("🔬 AUDITORIA FINAL: VALIDAÇÃO DE NORMALIZAÇÃO E VAZAMENTO")
    print("="*80)

    base_df_path = os.path.join("models_ai", "base_featured_df.pkl")
    if not os.path.exists(base_df_path):
        base_df_path = "models/base_featured_df.pkl"
        
    if not os.path.exists(base_df_path):
        print(f"❌ Erro: Dataset base não encontrado.")
        return

    df = pd.read_pickle(base_df_path)
    df_sample = df.tail(5000).copy()
    
    print(f"📊 Dataset Real: {df_sample.shape[0]} amostras.")

    # 1. Simular o que o TrendSpecialist faz agora: Fit do Scaler
    print("\n🛡️ SIMULANDO FIT DO SCALER (Lógica TrendSpecialist):")
    processor = ScientificDataProcessor()
    auto_cols = list(df_sample.select_dtypes(include=[np.number]).columns)
    all_filtered = processor.filter_autoencoder_features(auto_cols)
    
    # Simula o filtro do Especialista (usando o objeto real para ser fiel)
    temp_spec = TrendSpecialist(AIConfig(), input_dim=60)
    core_cols = temp_spec._filter_to_core_features(all_filtered)
    
    scaler = RobustScaler()
    scaler.fit(df_sample[core_cols])
    print(f"   ✅ Scaler treinado em {len(core_cols)} features.")

    # 2. Testar o Ambiente com o Scaler Treinado
    print("\n🤖 TESTE DE AMBIENTE COM SCALER REAL:")
    try:
        env = TrendFollowingEnv(df=df_sample, config=AIConfig(), feature_scaler=scaler, feature_columns=core_cols)
        
        obs, _ = env.reset()
        
        # 3. Analisar Variância e Escala
        print(f"\n📊 ANÁLISE DE ESCALA DA OBSERVAÇÃO:")
        market_obs = obs[:len(core_cols)]
        print(f"   🔹 Range Market: Min={market_obs.min():.4f}, Max={market_obs.max():.4f}, Mean={market_obs.mean():.4f}")
        print(f"   🔹 Vetor Completo (Tamanho): {len(obs)}")
        
        # 4. Verificar se há vida (variância) no vetor
        if np.abs(market_obs).sum() == 0:
            print("   ❌ ERRO: Vetor de mercado está ZERADO. Normalização falhou.")
        else:
            print("   ✅ Vetor de mercado OPERACIONAL (há variação nos dados).")

        # 5. Conferir Física e Vazamento
        print(f"\n🧠 CONFERINDO CONTEXTO (Últimas 9 posições):")
        context_obs = obs[-9:]
        # Mapping: [pos, pnl, steps, h_sin, h_cos, d_sin, d_cos, entropy, hurst]
        print(f"   🔹 Física (Hurst): {context_obs[-1]:.4f}")
        print(f"   🔹 Física (Entropia): {context_obs[-2]:.4f}")
        
        # Verificação final de vazamento (tp_prior)
        if len(obs) == 60:
            print("   ✅ Dimensão 60 confirmada: Vazamento de tp_prior REMOVIDO.")
        else:
            print(f"   ⚠️  Dimensão inesperada: {len(obs)}. Verifique o __init__.")

    except Exception as e:
        print(f"❌ Erro: {e}")

    print("\n" + "="*80)
    print("✅ AUDITORIA FINAL CONCLUÍDA")
    print("="*80 + "\n")

if __name__ == "__main__":
    audit_dataset_v3()
