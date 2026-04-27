import asyncio
import pandas as pd
import logging
from config.settings import AIConfig, TradingConfig
from data_provider import DataProvider
from feature_engineering.main import FeatureEngineeringPipeline
from utils.logger import get_logger

# Setup logging
logger = get_logger("RebuildAutoencoder")
logging.basicConfig(level=logging.INFO)

async def main():
    logger.info("🚀 Starting Autoencoder Rebuild Process...")
    
    # Initialize Configs
    ai_config = AIConfig()
    trading_config = TradingConfig()
    
    # Initialize Components
    data_provider = DataProvider(ai_config, trading_config)
    pipeline = FeatureEngineeringPipeline(ai_config)
    
    # Fetch Data
    symbol = "BTCUSDT"
    timeframes = ["15m", "1h", "4h", "1m", "5m"]
    limit = 5000 # Enough for training AE
    
    logger.info(f"📥 Fetching data for {symbol}...")
    raw_data = {}
    
    # Fetch data for all timeframes
    for tf in timeframes:
        logger.info(f"   Fetching {tf}...")
        df = data_provider._fetch_and_format_data(symbol, tf, limit)
        if df is not None and not df.empty:
            raw_data[tf] = df
        else:
            logger.error(f"❌ Failed to fetch data for {tf}")
            return

    # Create Features (RAW ONLY - apply_latent=False)
    logger.info("⚙️ Generating Technical Indicators (Raw)...")
    # Note: We set fit_scaler=False because global scaling is removed/deprecated in favor of internal scaling
    # We set apply_latent=False to avoid circular dependency (AE not trained yet)
    training_data = await pipeline.create_features(
        raw_df_dict=raw_data,
        symbol=symbol,
        primary_timeframe="5m", # Using 5m as primary for high freq
        fit_scaler=False,
        apply_latent=False 
    )
    
    if training_data is None or training_data.empty:
        logger.error("❌ Failed to create training data.")
        return
        
    logger.info(f"✅ Training Data Ready. Shape: {training_data.shape}")
    
    # Train Autoencoder
    logger.info("🧠 Starting Autoencoder Training...")
    # This will:
    # 1. Fit the internal RobustScaler on this data
    # 2. Optimize hyperparameters (if n_trials > 0)
    # 3. Train the final model
    # 4. Save model, scaler, and column metadata
    success = pipeline.train_autoencoder(training_data, num_epochs=50)
    
    if success:
        logger.info("✅ Autoencoder Rebuild Complete! You can now run the bot.")
    else:
        logger.error("❌ Autoencoder Training Failed.")

if __name__ == "__main__":
    asyncio.run(main())
