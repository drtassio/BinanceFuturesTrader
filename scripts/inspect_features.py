
import asyncio
import os
import sys
import pandas as pd
import numpy as np
import torch
import logging

# Adiciona o diretório raiz ao path para permitir importações
sys.path.append(os.getcwd())

from data_provider import DataProvider
from trading.binance_connector import BinanceConnector
from feature_engineering import FeatureEngineeringPipeline
from config.settings import AIConfig, TradingConfig, Config
from trading.ai_controller import AIController
from models.trade_schema import Signal, Action

# Configura log básico para o script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureInspector")

async def inspect_feature_contract():
    logger.info("🚀 Iniciando Inspeção do Contrato de Features...")
    
    # 1. Configurações
    ai_config = AIConfig()
    trading_config = TradingConfig()
    base_config = Config()
    
    # 2. Inicialização de Componentes (Modo Simulação)
    # Criamos o conector com force_production=True para pegar dados reais de klines
    connector = BinanceConnector(base_config, force_production=True)
    await connector.connect()
    
    try:
        pipeline = FeatureEngineeringPipeline(ai_config)
        dp = DataProvider(connector, pipeline)
        
        symbol = trading_config.PRIMARY_PAIR
        logger.info(f"📊 Buscando dados e gerando features para {symbol}...")
        
        # Obtém as features mais recentes
        # Nota: get_latest_features já chama pipeline.create_features
        df_features = await dp.get_latest_features(symbol)
        
        if df_features is None or df_features.empty:
            logger.error("❌ Falha ao obter features do DataProvider.")
            return

        # 3. Inicializa AIController para reconstruir o contrato
        controller = AIController(ai_config, trading_config, {})
        controller.set_feature_pipeline(pipeline)
        controller.data_provider = dp
        
        # build_enriched_dataset define self.base_feature_columns
        # Ele precisa de um DF com colunas base
        logger.info("🛠️ Reconstruindo contrato de features via AIController...")
        enriched_df = await controller.build_enriched_dataset(df_features)
        
        feature_contract = controller.base_feature_columns
        if not feature_contract:
            logger.error("❌ Contrato de features (base_feature_columns) não foi gerado.")
            return

        # 4. Análise de Valores
        latest_row = enriched_df.iloc[-1]
        
        results = []
        for feat in feature_contract:
            val = latest_row.get(feat, np.nan)
            results.append({
                "Feature": feat,
                "Value": val,
                "Type": "Giant" if abs(val) > 10 and feat not in ['volume'] else "Normal",
                "Category": "Indicator" if any(x in feat for x in ['rsi', 'adx', 'ema', 'bb', 'macd']) else "Other"
            })
        
        inspect_df = pd.DataFrame(results)
        
        # 5. Exibição de Resultados
        print("\n" + "="*80)
        print(f"📊 RELATÓRIO TÉCNICO: CONTRATO DE FEATURES ({len(feature_contract)} colunas)")
        print("="*80)
        
        # Mostra as primeiras 20 features
        print("\nTOP 20 FEATURES NO CONTRATO:")
        print(inspect_df.head(20).to_string(index=False))
        
        # Filtra valores gigantes
        giant_vals = inspect_df[inspect_df['Type'] == "Giant"]
        if not giant_vals.empty:
            print("\n⚠️ ALERTA: VALORES GIGANTES DETECTADOS (Abs > 10):")
            # Agrupa por categoria para ver quais indicadores estão descontrolados
            print(giant_vals.to_string(index=False))
        else:
            print("\n✅ Estabilidade: Nenhum valor gigante (Abs > 10) detectado nas features principais.")

        # 6. Verificação do Shape Final para o Agente SAC
        # Simula a Etapa 5 do generate_trading_decision
        market_obs = latest_row[feature_contract].values.astype(np.float32)
        agent_state = np.array([0, 0, 0], dtype=np.float32) # Mock
        time_features = np.array([0, 0, 0, 0], dtype=np.float32) # Mock
        priors = np.array([0, 0], dtype=np.float32) # Mock
        
        full_obs = np.concatenate([market_obs, agent_state, time_features, priors])
        
        print("\n" + "-"*40)
        print(f"📐 SHAPE DO DATASET PARA O AGENTE: {full_obs.shape[0]} dimensões")
        print(f"🔢 EXEMPLO DE INPUT (Vetor): {full_obs[:5]} ... {full_obs[-5:]}")
        print("-"*40)

    finally:
        await connector.close()

if __name__ == "__main__":
    asyncio.run(inspect_feature_contract())
