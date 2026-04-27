"""
Script de correção simplificado para SHAP Background.
Gera dados sintéticos com a estrutura correta (53 colunas) para o SHAP.
O SHAP precisa apenas da estrutura de colunas, não dos valores reais.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())

from config.settings import AIConfig
from utils.logger import get_logger

logger = get_logger("FixSHAP")

def main():
    logger.info("🚀 Gerando SHAP background com dados sintéticos (estrutura correta)...")
    
    # 1. Carregar lista de colunas esperadas do metadata
    metadata_path = os.path.join(AIConfig.MODEL_DIR, "training_metadata.json")
    
    if not os.path.exists(metadata_path):
        logger.error(f"❌ Metadados não encontrados em: {metadata_path}")
        return
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    expected_cols = metadata.get("base_feature_columns", [])
    
    if not expected_cols:
        logger.error("❌ 'base_feature_columns' não encontrado nos metadados.")
        return
    
    logger.info(f"📋 Metadados carregados. Esperado: {len(expected_cols)} colunas.")
    logger.info(f"   Colunas: {expected_cols[:5]} ... {expected_cols[-3:]}")
    
    # 2. Gerar 100 linhas de dados sintéticos (normalizado entre 0 e 1)
    n_rows = 100
    
    data = {}
    for col in expected_cols:
        if col in ['open', 'high', 'low', 'close', 'volume']:
            # OHLCV podem ter escalas maiores
            data[col] = np.random.uniform(0.5, 1.5, n_rows)
        elif 'hidden_feature' in col:
            # Features latentes do autoencoder, tipicamente [-1, 1]
            data[col] = np.random.uniform(-1, 1, n_rows)
        elif 'regime' in col.lower():
            # Features de regime, one-hot encoded
            data[col] = np.random.choice([0.0, 1.0], n_rows)
        else:
            # Outros indicadores normalizados [0, 1]
            data[col] = np.random.uniform(0, 1, n_rows)
    
    df = pd.DataFrame(data)
    
    # 3. Garantir ordem das colunas
    df = df[expected_cols]
    
    # 4. Salvar
    output_path = os.path.join(AIConfig.MODEL_DIR, "base_featured_df.pkl")
    df.to_pickle(output_path)
    
    logger.info(f"💾 Arquivo salvo em: {output_path}")
    logger.info(f"📊 Shape final: {df.shape}")
    logger.info("✅ SHAP background regenerado com sucesso!")
    logger.info("   ⚠️ NOTA: Dados sintéticos. Valores SHAP iniciais podem ser menos significativos.")
    logger.info("   Reinicie o bot para aplicar.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Erro: {e}", exc_info=True)
