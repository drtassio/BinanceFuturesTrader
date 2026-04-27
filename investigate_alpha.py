
import pandas as pd
import numpy as np
import os
import joblib
import torch
import asyncio
from data_provider import DataProvider
from trading.binance_connector import BinanceConnector
from feature_engineering.main import FeatureEngineeringPipeline
from config.settings import AIConfig, DataConfig, TradingConfig

async def main():
    print("🚀 [INVESTIGAÇÃO ALPHA] Iniciando análise de poder preditivo...")
    
    # 1. Setup
    config_ai = AIConfig()
    config_trading = TradingConfig()
    
    # Mock connector for DataProvider
    class MockConnector:
        async def get_kline_data(self, *args, **kwargs): return []
    
    feature_pipeline = FeatureEngineeringPipeline(config_ai)
    # data_provider = DataProvider(MockConnector(), feature_pipeline)
    
    # 2. Carregar dados reais do cache histórico
    timeframes = ["15m", "1h", "4h", "1m", "5m"]
    raw_dfs = {}
    
    for tf in timeframes:
        df_path = os.path.join(os.getcwd(), "logs", "historical_data", f"BTCUSDT_{tf}_historical.pkl")
        if not os.path.exists(df_path):
            print(f"⚠️ Aviso: Arquivo não encontrado para {tf}: {df_path}")
            continue
        df_tf = pd.read_pickle(df_path)
        raw_dfs[tf] = df_tf
        print(f"✅ Dados {tf} carregados: {len(df_tf)} linhas")
    
    if "15m" not in raw_dfs:
        print("❌ timeframe primário (15m) não carregado.")
        return
    
    # 3. Gerar features (incluindo o Predictor do Autoencoder)
    print("⚙️ Gerando features e sinais do Predictor...")
    featured_df = await feature_pipeline.create_features(
        raw_dfs, "BTCUSDT", "15m", fit_scaler=False
    )
    
    if featured_df is None:
        print("❌ Falha ao criar features.")
        return
        
    # Aplicar latents (onde tp_prior_dir é gerado)
    featured_df = feature_pipeline.apply_hidden_features(featured_df)
    
    if 'tp_prior_dir' not in featured_df.columns:
        print("❌ tp_prior_dir não encontrado no DataFrame.")
        return
        
    # 4. Análise de Correlação
    print("\n📊 [RESULTADOS DA ANÁLISE]")
    
    # Preço de fechamento para retornos futuros
    # Usamos close_15m se disponível ou apenas close
    close_col = 'close' if 'close' in featured_df.columns else 'close_15m'
    
    # Retornos futuros (Forward Returns)
    horizons = [1, 4, 8, 16, 32] # 15m * X = 15m, 1h, 2h, 4h, 8h
    
    predictor_signal = featured_df['tp_prior_dir']
    
    results = []
    for h in horizons:
        # Retorno percentual logarítmico futuro
        future_return = np.log(featured_df[close_col].shift(-h) / featured_df[close_col])
        
        # Correlação de Pearson
        valid_mask = predictor_signal.notna() & future_return.notna()
        corr = np.corrcoef(predictor_signal[valid_mask], future_return[valid_mask])[0, 1]
        
        # Hit Rate (Acerto de direção)
        # Se prior_dir > 0 e retorno > 0, ou prior_dir < 0 e retorno < 0
        hits = ((predictor_signal > 0) & (future_return > 0)) | ((predictor_signal < 0) & (future_return < 0))
        # Consideramos apenas sinais não neutros
        active_signals = predictor_signal != 0
        hit_rate = hits[active_signals & valid_mask].mean()
        
        results.append({
            "Horizon": f"{h} steps ({h*15}m)",
            "Correlation": corr,
            "Hit Rate": hit_rate
        })
        
    res_df = pd.DataFrame(results)
    output = res_df.to_string(index=False)
    print(output)
    
    with open("alpha_results.txt", "w", encoding="utf-8") as f:
        f.write(output)
        
    max_corr = res_df['Correlation'].max()
    print(f"\n🔥 Correlação Máxima Detectada: {max_corr:.4f}")
    
    with open("alpha_results.txt", "a", encoding="utf-8") as f:
        f.write(f"\n\nMax Correlation: {max_corr:.4f}")
        if max_corr < 0.05:
            f.write("\nWARNING: Low predictive power (< 0.05).")
        else:
            f.write("\nOK: Acceptable predictive power.")

if __name__ == "__main__":
    asyncio.run(main())
