import sys
import os
import asyncio
import pandas as pd
from datetime import datetime
import traceback

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import AIConfig, TradingConfig
from specialists.trend_specialist import TrendSpecialist
from feature_engineering.main import FeatureEngineeringPipeline
from trading.ai_controller import AIController
from trading.utils.trend_backtest import ensure_datetime_index, split_train_holdout
from utils.logger import get_logger

logger = get_logger("TrendOptimizationScript")

async def main():
    print("🚀 Starting TrendSpecialist Optimization (Corrected Flow with AIController)...")
    
    ai_cfg = AIConfig()
    tr_cfg = TradingConfig()
    
    # Initialize Controller and Pipeline properly
    # We pass an empty system_state for this script
    ctrl = AIController(ai_cfg, tr_cfg, {})
    
    # Manually initialize feature pipeline to ensure we can access scalers
    fe = FeatureEngineeringPipeline(ai_cfg)
    ctrl.set_feature_pipeline(fe)
    
    # Load features
    base_features_path = os.path.join(ai_cfg.MODEL_DIR, "base_featured_df.pkl")
    
    if os.path.exists(base_features_path):
        print(f"Loading features from {base_features_path}...")
        df = pd.read_pickle(base_features_path)
    else:
        print("❌ Base features not found. Please run trend specialist training first to generate features.")
        return

    df = ensure_datetime_index(df)
    
    # Start dynamic regime detection if needed
    if 'regime_confidence' not in df.columns:
        print("⚠️ 'regime_confidence' not found in cache. Running features regeneration (Regime Detection)...")
        from feature_engineering.crypto_regime_detector import CryptoRegimeDetector, RegimeConfig
        
        # Configure detector (using defaults matching train_moe_specialists)
        config_regime = RegimeConfig(
            adx_period=9,
            adx_trend_threshold=20.0,
            confidence_threshold=0.75,
            min_regime_duration=8,
            weight_hmm=0.35,
            weight_gmm=0.30,
            weight_adx=0.20,
            weight_funding=0.15
        )
        detector = CryptoRegimeDetector(config_regime)
        try:
             # Fit predict on the dataframe
             print("   🧠 Fitting CryptoRegimeDetector...")
             regime_df = detector.fit_predict(df)
             # Join back to df
             df['regime'] = regime_df['regime']
             df['regime_confidence'] = regime_df['confidence']
             df['regime_val'] = df['regime'] # Ensure numerical mapping if needed
             print(f"✅ Regime detection complete. Added regime columns. (Mean confidence: {df['regime_confidence'].mean():.4f})")
        except Exception as e:
             print(f"❌ Regime detection failed: {e}")
             traceback.print_exc()

    # Enrich with Trend Predictions (Dynamic Logic)
    print("🔄 Enriching DataFrame with dynamic trend predictions...")
    if hasattr(ctrl, '_enrich_df_with_trend_predictions'):
        # Just in case _enrich needs other components initialized:
        # It relies on self.regime_detector which is usually in specialists, but the method
        # implementation logic (Step 162 summary) suggests it uses columns if present or 
        # defaults.
        # Wait, Step 162 says: "Modified to load and use CryptoRegimeDetector.predict(df) to include regime_confidence..."
        # And "Integrated CryptoRegimeDetector Output into AIController._enrich_df_with_trend_predictions... to dynamically populate..."
        
        # We need to see if _enrich_df_with_trend_predictions requires anything special.
        # It takes 'df' as input.
        try:
             df = ctrl._enrich_df_with_trend_predictions(df)
             print("✅ DataFrame enriched successfully.")
        except Exception as e:
             print(f"❌ Failed to enrich DataFrame: {e}")
             traceback.print_exc()
             return
    else:
        print("⚠️ Controller does not have _enrich_df_with_trend_predictions method!")

    # Check if tp_prior_conf is present (sanity check)
    if 'tp_prior_conf' in df.columns:
        print(f"✅ 'tp_prior_conf' present. Mean: {df['tp_prior_conf'].mean():.4f}")
    else:
        print("❌ 'tp_prior_conf' NOT present after enrichment!")

    # Split data
    train_df, eval_df = split_train_holdout(df, holdout_months=2, min_holdout_rows=1000)
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    input_dim = len(numeric_cols)
    print(f"Numeric columns count: {input_dim}")
    
    print(f"Initializing TrendSpecialist with input_dim={input_dim}")
    specialist = TrendSpecialist(config=ai_cfg, input_dim=input_dim)
    
    # Inject Scaler
    if not fe.scalers:
         print("Attempts to load scalers manually...")
         fe._load_scalers()
    
    if fe.scalers and 'robust' in fe.scalers:
        print("✅ Scaler found in pipeline. Injecting into specialist...")
        specialist.set_feature_scaler(fe.scalers['robust'])
    else:
        print("⚠️ No scaler found in pipeline! creating a quick fit scaler for verification...")
        # Since we are solving the "no scaler" warning, we must provide one.
        # If the pipeline doesn't have it, we fit on train_df (safe flow)
        from sklearn.preprocessing import RobustScaler
        import numpy as np
        scaler = RobustScaler()
        # Fit on numeric columns matches what TrendFollowingEnv uses (minus tp_prior_dir)
        cols_to_fit = [c for c in numeric_cols if c != 'tp_prior_dir']
        scaler.fit(train_df[cols_to_fit].fillna(0).values)
        specialist.set_feature_scaler(scaler)
        print(f"✅ Created and injected a temporary RobustScaler (fitted on {len(cols_to_fit)} features).")

        
    print("Starting Hyperparameter Optimization (5 trials)...")
    try:
        specialist.optimize_hyperparameters(
            train_df=train_df,
            eval_df=eval_df,
            n_trials=20, 
            n_timesteps=3000 # Short timesteps for quick check
        )
        print("✅ Optimization complete.")
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
