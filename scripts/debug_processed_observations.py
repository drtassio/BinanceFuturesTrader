
import asyncio
import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Adiciona o diretório raiz ao path para permitir importações
sys.path.append(os.getcwd())

from data_provider import DataProvider
from trading.binance_connector import BinanceConnector
from feature_engineering import FeatureEngineeringPipeline
from feature_engineering.scientific_data_processor import ScientificDataProcessor
from config.settings import AIConfig, TradingConfig, Config
from trading.ai_controller import AIController

# Configura log básico para o script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataAuditor")

async def export_full_processed_dataset():
    logger.info("🚀 Iniciando Auditoria Completa de Dados Processados...")
    
    # 1. Configurações
    ai_config = AIConfig()
    trading_config = TradingConfig()
    base_config = Config()
    
    # 2. Inicialização de Componentes
    connector = BinanceConnector(base_config, force_production=True)
    await connector.connect()
    
    try:
        pipeline = FeatureEngineeringPipeline(ai_config)
        dp = DataProvider(connector, pipeline)
        
        symbol = trading_config.PRIMARY_PAIR
        logger.info(f"📊 Buscando dados históricos para {symbol}...")
        
        # Obtém as features enriquecidas (brutas)
        df_raw = await dp.get_latest_features(symbol)
        
        if df_raw is None or df_raw.empty:
            logger.error("❌ Falha ao obter dados.")
            return

        # 3. Simula AIController para obter o contrato
        controller = AIController(ai_config, trading_config, {})
        controller.set_feature_pipeline(pipeline)
        
        logger.info("🛠️ Enriquecendo dataset e extraindo contrato...")
        # build_enriched_dataset faz o merge de latentes e priors
        enriched_df = await controller.build_enriched_dataset(df_raw)
        feature_contract = controller.base_feature_columns
        
        if not feature_contract:
            logger.error("❌ Contrato de features não encontrado.")
            return

        # 4. Aplica Normalização (Técnica usada no Treinamento)
        logger.info("🧪 Aplicando Normalização Científica (RobustScaler)...")
        processor = ScientificDataProcessor()
        
        # Carrega scalers se existirem, senão fit neles (simula treino)
        if not processor.load_scalers():
             logger.warning("⚠️ Scalers não encontrados. Fitando novos scalers para demonstração.")
             # Assume regime 2 (Ranger) como base para a demonstração global
             normalized_df = processor.normalize_for_specialist(enriched_df, regime_code=2, feature_columns=feature_contract, fit=True)
        else:
             logger.info("✅ Scalers carregados com sucesso.")
             normalized_df = processor.normalize_for_specialist(enriched_df, regime_code=2, feature_columns=feature_contract, fit=False)

        # 5. Constrói Vetor de Observação Completo para Auditoria
        # Vamos criar um DataFrame que contém exatamente o que entra no agente SAC
        obs_rows = []
        for idx, row in normalized_df.iterrows():
            # Boco 1: Market
            market_obs = row[feature_contract].values.astype(np.float32)
            
            # Bloco 2: Agent State (MOCK)
            agent_state = np.array([0.0, 0.0, 0.0], dtype=np.float32) # pos, pnl, steps
            
            # Bloco 3: Time Features
            timestamp = idx
            time_feats = np.array([
                np.sin(2*np.pi*timestamp.hour/24), 
                np.cos(2*np.pi*timestamp.hour/24),
                np.sin(2*np.pi*timestamp.dayofweek/7), 
                np.cos(2*np.pi*timestamp.dayofweek/7)
            ], dtype=np.float32)
            
            # Bloco 4: Priors (MOCK)
            priors = np.array([0.0, 0.0], dtype=np.float32)
            
            # Vetor Final
            full_vector = np.concatenate([market_obs, agent_state, time_feats, priors])
            obs_rows.append(full_vector)
            
        # Cria nomes de colunas para o auditor
        col_names = list(feature_contract) + ["pos_side", "pnl", "steps", "sin_h", "cos_h", "sin_d", "cos_d", "prior_dir", "prior_conf"]
        audit_df = pd.DataFrame(obs_rows, index=normalized_df.index, columns=col_names)
        
        # 6. Salva e Exibe
        output_file = "logs/full_feature_audit.csv"
        os.makedirs("logs", exist_ok=True)
        audit_df.to_csv(output_file)
        
        print("\n" + "="*80)
        print("✅ AUDITORIA CONCLUÍDA")
        print("="*80)
        print(f"📄 Arquivo gerado: {output_file}")
        print(f"📐 Dimensões do dataset: {audit_df.shape}")
        print("\n--- ÚLTIMA LINHA (EXEMPLO DE INPUT DO AGENTE) ---")
        # Mostra todos os valores da última linha
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(audit_df.iloc[-1])
        
        # Verifica se ainda há valores fora da escala [ -10, 10 ]
        outliers = audit_df.iloc[-1][(audit_df.iloc[-1].abs() > 10)]
        if not outliers.empty:
            print("\n⚠️ ALERTA: Ainda existem valores fora da escala ideal (>10):")
            print(outliers)
        else:
            print("\n✨ Estabilidade: Todos os valores estão dentro da escala controlada.")

    finally:
        await connector.close()

if __name__ == "__main__":
    asyncio.run(export_full_processed_dataset())
