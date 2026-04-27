import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.temporal_autoencoder import TemporalAutoencoderPipeline
from config.settings import AIConfig
import logging

# Configure logging to stdout
logging.basicConfig(level=logging.INFO)

def verify_refactor():
    print("🧪 Verifying Temporal Autoencoder Refactor...")
    
    config = AIConfig()
    pipeline = TemporalAutoencoderPipeline(config)
    
    # Mock autoencoder for sanity check if not loaded (it won't be trained yet in this script context usually, but we need it to exist)
    # Actually sanity_check checks if self.autoencoder is not None.
    # We can manually initialize it with random weights to test the architecture.
    
    print("⚙️ Initializing dummy model for architecture check...")
    pipeline.hyperparams = {
        'input_dim': 32,
        'cnn_out_channels': 32,
        'kernel_size': 3,
        'gru_hidden': 32,
        'gru_layers': 1,
        'latent_dim': 16,
        'seq_length': 64,
        'dropout': 0.1,
        'p_mask': 0.25,
        'sigma': 0.12
    }
    # pipeline._load_model() # Removed to avoid crash on architecture mismatch
    
    # Let's manually init
    from feature_engineering.temporal_autoencoder import TemporalSDAE
    pipeline.autoencoder = TemporalSDAE(
        input_dim=32,
        cnn_out_channels=32,
        kernel_size=3,
        gru_hidden=32,
        gru_layers=1,
        latent_dim=16,
        seq_length=64
    ).to(pipeline.device)
    
    print("✅ Dummy model initialized.")
    
    # Run sanity check
    results = pipeline.sanity_check(test_input_dim=32, test_seq_length=64)
    print(f"📊 Sanity Check Results: {results}")
    
    if results['status'] == 'sucesso':
        print("✅ Architecture verification passed!")
        return True
    else:
        print("❌ Architecture verification failed!")
        return False

if __name__ == "__main__":
    success = verify_refactor()
    sys.exit(0 if success else 1)
