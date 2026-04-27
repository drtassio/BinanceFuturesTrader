import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import logging

# Adicionar o diretório raiz ao path para importar os módulos
sys.path.append(os.getcwd())

from trading.ai_controller import AIController
from feature_engineering.scientific_data_processor import ScientificDataProcessor
from config.settings import DevelopmentConfig

# Configurar logging mínimo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CONTRACT_TEST")

def test_feature_contract():
    try:
        # 1. Carregar o dataset base (simulando o carregamento do AIController)
        base_df_path = os.path.join(os.getcwd(), 'models_ai', 'base_featured_df.pkl')
        if not os.path.exists(base_df_path):
            print(f"❌ Erro: {base_df_path} não encontrado.")
            return
            
        with open(base_df_path, 'rb') as f:
            df = pickle.load(f)
            
        print(f"✅ Dataset carregado: {df.shape}")
        
        # 2. Inicializar AIController e ScientificDataProcessor
        config = DevelopmentConfig()
        ai_controller = AIController(config)
        processor = ScientificDataProcessor()
        
        # 3. Simular a criação do contrato via AIController
        # Nota: Vamos focar na lógica de seleção de colunas do build_enriched_dataset
        
        # Simular optimized_feature_cols etc.
        hidden_cols = [c for c in df.columns if 'hidden_feature_' in c]
        trend_cols = [c for c in df.columns if 'trend_pred_' in c]
        optimized_feature_cols = hidden_cols + trend_cols
        prior_cols = [c for c in ('tp_prior_dir', 'tp_prior_conf', 'tp_tau', 'tp_primary_signal') if c in df.columns]
        extras_cols = ['tp_duration_median', 'tp_regime_up', 'tp_regime_down', 'tp_regime_sideways']
        
        # Lógica restaurada no AIController
        all_indicators = [c for c in df.columns 
                         if c not in (optimized_feature_cols + prior_cols + ['regime', 'regime_confidence', 'regime_name', 'regime_val', 'regime_is_bull', 'regime_is_bear', 'regime_is_ranger'])
                         and df[c].dtype in [np.float32, np.float64, np.int64]]
        
        ordered_cols = all_indicators + optimized_feature_cols + prior_cols + [e for e in extras_cols if e in df.columns]
        final_df_cols = list(dict.fromkeys(ordered_cols))
        
        print(f"📊 Colunas enviadas para o Processador: {len(final_df_cols)}")
        
        # 4. Filtragem Científica
        kept_cols = processor.filter_autoencoder_features(final_df_cols, df)
        
        print(f"🛡️ Colunas após Filtragem Científica: {len(kept_cols)}")
        
        # 5. Análise de Qualidade (O que o usuário pediu)
        # Verificar se tem colunas com valor constante (ZERO)
        subset = df[kept_cols].fillna(0)
        constant_cols = []
        for col in kept_cols:
            if subset[col].nunique() <= 1:
                constant_cols.append(col)
                
        # Verificar se tem colunas com valores gigantes (que escaparam do filtro)
        leaked_prices = []
        for col in kept_cols:
            if subset[col].abs().max() > 1500 and not any(p in col for p in ['rsi', 'adx', 'cci', 'williams', 'stoch']):
                leaked_prices.append(f"{col} (max: {subset[col].abs().max():.0f})")

        print("\n--- ANALISE DE QUALIDADE ---")
        if constant_cols:
            print(f"⚠️ {len(constant_cols)} Colunas Constant (Ruido): {constant_cols[:5]}...")
        else:
            print("✅ Nenhuma coluna constante (ruído zero) detectada.")
            
        if leaked_prices:
            print(f"❌ {len(leaked_prices)} Colunas com escala absoluta detectadas: {leaked_prices[:5]}...")
        else:
            print("✅ Nenhuma coluna de preço absoluto detectada (Escala OK).")
            
        # Verificar se as hidden features estão lá
        hidden_found = [c for c in kept_cols if 'hidden' in c]
        print(f"🧠 Hidden Features preservadas: {len(hidden_found)}")
        
        # Verificar colunas mortas específicas solicitadas
        dead_check = ['qpnl_24_p50', 'tp_uncertainty', 'sdae_recon_error']
        found_dead = [c for c in dead_check if c in kept_cols]
        if found_dead:
            print(f"❌ Erro: Colunas 'mortas' ainda no contrato: {found_dead}")
        else:
            print("✅ Colunas mortas foram removidas com sucesso.")

    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_contract()
