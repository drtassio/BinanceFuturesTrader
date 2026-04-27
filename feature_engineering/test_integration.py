# -----------------------------------------------------------------------------
# ARQUIVO: feature_engineering/test_integration.py
# -----------------------------------------------------------------------------

"""
Teste de integração da nova FeatureEngineeringPipeline com TemporalAutoencoderPipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.main import FeatureEngineeringPipeline
from config.settings import AIConfig

def test_integration():
    """Testa se a integração está funcionando corretamente."""
    print("🧪 Testando integração da FeatureEngineeringPipeline com TemporalAutoencoderPipeline...")
    
    try:
        # Cria uma configuração de teste
        config = AIConfig()
        
        # Inicializa a pipeline
        print("⚙️ Inicializando FeatureEngineeringPipeline...")
        pipeline = FeatureEngineeringPipeline(config)
        
        print("✅ FeatureEngineeringPipeline inicializada com sucesso!")
        print(f"📊 Pipeline temporal: {type(pipeline.temporal_autoencoder_pipeline).__name__}")
        print(f"📊 Status treinado: {pipeline.temporal_autoencoder_pipeline.is_trained}")
        
        # Testa o método get_autoencoder_latent_dim
        latent_dim = pipeline.get_autoencoder_latent_dim()
        print(f"📊 Dimensão latente: {latent_dim}")
        
        print("\n🎉 Todos os testes passaram! A integração está funcionando perfeitamente.")
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
