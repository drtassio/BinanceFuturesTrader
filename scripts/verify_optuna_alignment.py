
import sys
import os
import pandas as pd
import numpy as np
import torch
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from learning.trend_predictor import TrendPredictor
from config.settings import AIConfig, TradingConfig

def verify_optuna_alignment():
    print("Starting Optuna Alignment Verification...")
    
    # Mock configs
    ai_config = AIConfig()
    trading_config = TradingConfig()
    
    # Instantiate TrendPredictor
    predictor = TrendPredictor(ai_config, trading_config)
    
    # Mock data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 1000
    }, index=dates)
    
    labels_df = pd.DataFrame({
        'primary_signal': np.random.choice([-1, 0, 1], 100),
        'pnl_24': np.random.randn(100) * 0.05,
        'duration': np.random.randint(5, 50, 100)
    }, index=dates)
    
    # Mock trained clone
    trained_clone = MagicMock()
    trained_clone.model_hyperparams = {
        "optuna_financial_eval": {
            "enabled": True,
            "recent_rows": 50,
            "tau": 0.0, # low tau to ensure trades
            "horizon": 24
        },
        "financial_metrics": {"evaluation_horizon": 24},
        "seq_length": 10
    }
    
    # Mock predict output
    class MockOutput:
        def __init__(self):
            self.success_prob = np.random.rand(50)
            self.primary_signal = np.random.choice([-1, 1], 50) # Force signals
            self.lookback_offset = 0
            
    trained_clone.predict.return_value = MockOutput()
    trained_clone.last_selected_tau = 0.0
    
    # Test _evaluate_trial_backtest
    print("Testing _evaluate_trial_backtest...")
    try:
        metrics = predictor._evaluate_trial_backtest(trained_clone, df, labels_df)
        print("Metrics calculated:", metrics)
        
        expected_keys = ["financial_avg_return", "financial_sharpe", "catastrophe_stop_rate", "avg_winner_duration_steps"]
        for k in expected_keys:
            if k not in metrics:
                print(f"FAILED: Missing metric {k}")
            else:
                print(f"OK: {k} = {metrics[k]}")
                
    except Exception as e:
        print(f"FAILED: _evaluate_trial_backtest raised exception: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test _compute_profit_objective
    print("\nTesting _compute_profit_objective (RL-Aligned)...")
    try:
        score = predictor._compute_profit_objective(metrics)
        print(f"Objective Score: {score}")
        
        if not (0.0 <= score <= 1.0) and not (-1.0 <= score <= 1.0): # Depending on normalization
             print(f"WARNING: Score {score} out of expected range?")
             
        # Test sensitivity
        # Case 1: High catastrophe rate
        bad_metrics = metrics.copy()
        bad_metrics["catastrophe_stop_rate"] = 1.0 # 100% catastrophe
        bad_score = predictor._compute_profit_objective(bad_metrics)
        print(f"Bad Score (100% catastrophe): {bad_score}")
        
        if bad_score >= score:
             print("FAILED: Objective did not penalize catastrophe rate correctly.")
        else:
             print("OK: Objective penalized catastrophe rate.")

    except Exception as e:
        print(f"FAILED: _compute_profit_objective raised exception: {e}")
        traceback.print_exc()
        return

    print("\nVerification Completed Successfully.")

if __name__ == "__main__":
    verify_optuna_alignment()
