
import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import logging

# Adicionar o diretório raiz ao path para importar os módulos
sys.path.append(os.getcwd())

from feature_engineering.scientific_data_processor import ScientificDataProcessor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("REAL_CONTRACT_GEN")

def generate_real_contract():
    try:
        # 1. Carregar os dados reais que o bot usa
        data_path = os.path.join(os.getcwd(), 'models_ai', 'base_featured_df.pkl')
        if not os.path.exists(data_path):
            print(f"❌ Erro: Arquivo {data_path} não encontrado.")
            return

        print(f"📂 Carregando dados reais de {data_path}...")
        with open(data_path, 'rb') as f:
            df = pickle.load(f)
        
        print(f"✅ Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas.")

        # 2. Instanciar o processador com as novas regras científicas
        processor = ScientificDataProcessor()
        
        # 3. Identificar colunas potenciais para o contrato (numéricas)
        initial_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remover metadados óbvios que não devem estar no filtro inicial
        meta_to_exclude = ['regime', 'regime_confidence', 'regime_val', 'regime_is_bull', 'regime_is_bear', 'regime_is_ranger']
        candidate_cols = [c for c in initial_cols if c not in meta_to_exclude]
        
        print(f"🔍 Candidatos iniciais (numéricos): {len(candidate_cols)}")

        # 4. EXECUTAR FILTRAGEM REAL
        # Esta função utiliza as regras de whitelist (hidden_feature), fillna(0) e o filtro de escala > 1000
        final_contract = processor.filter_autoencoder_features(candidate_cols, df)
        
        print(f"🛡️ Contrato Final Gerado: {len(final_contract)} colunas.")

        # 5. SALVAR AMOSTRA DO CONTRATO PARA O USUÁRIO
        sample_contract = {
            "total_features": len(final_contract),
            "sample_features": final_contract[:50],  # Mostrar as primeiras 50
            "excluded_count": len(candidate_cols) - len(final_contract),
            "has_hidden_features": any('hidden' in c for c in final_contract),
            "hidden_features_found": [c for c in final_contract if 'hidden' in c]
        }
        
        output_path = os.path.join(os.getcwd(), 'tmp', 'real_contract_sample.json')
        with open(output_path, 'w') as f:
            json.dump(sample_contract, f, indent=4)
        
        print(f"✅ Amostra do contrato salva em: {output_path}")
        
        # 6. Verificação de Integridade Live
        print("\n--- TESTE DE INTEGRIDADE REAL ---")
        df_contract = df[final_contract].fillna(0)
        
        # Verificar escalas
        scales = df_contract.abs().max()
        leaked = scales[scales > 1500]
        # Filtrar indicadores que naturalmente podem ter escala alta mas são normalizados (Ex: CCI)
        real_leaked = [idx for idx, val in leaked.items() if not any(p in idx.lower() for p in ['cci', 'rsi', 'adx', 'stoch', 'williams'])]
        
        if real_leaked:
            print(f"⚠️ Alerta: {len(real_leaked)} colunas com escala suspeita (>1500): {real_leaked[:5]}")
        else:
            print("✅ Verificação de Escala: Nenhuma feature de preço bruto detectada.")

        # Verificar colunas mortas (Zeros)
        zeros = (df_contract == 0).all()
        dead_cols = zeros[zeros == True].index.tolist()
        if dead_cols:
            print(f"⚠️ Alerta: {len(dead_cols)} colunas mortas (totalmente zero): {dead_cols[:5]}")
        else:
            print("✅ Verificação de Sinal: Nenhuma coluna 100% vazia detectada.")

    except Exception as e:
        print(f"❌ Erro fatal ao gerar contrato: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_real_contract()
