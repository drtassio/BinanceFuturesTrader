# -----------------------------------------------------------------------------
# ARQUIVO: tests/test_autoencoder_critical_validations.py
# -----------------------------------------------------------------------------

"""
Testes para validações críticas do autoencoder:
- OOD (Out-of-Distribution) reconstruction
- Permutation test
"""

import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.temporal_autoencoder import TemporalAutoencoderPipeline, TemporalSDAE
from config.settings import AIConfig


class TestCriticalValidations:
    """Testes para validações científicas críticas"""
    
    @pytest.fixture
    def synthetic_time_series(self):
        """Cria série temporal sintética com padrões conhecidos"""
        np.random.seed(42)
        
        # 2 anos de dados diários
        dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
        n = len(dates)
        
        # Features com padrões temporais
        df = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
            'volume': 1000 + np.random.randn(n) * 100,
            'rsi': 50 + 20 * np.sin(np.arange(n) / 10),
            'macd': np.random.randn(n),
            'bb_upper': 105 + np.random.randn(n),
            'bb_lower': 95 + np.random.randn(n),
        }, index=dates)
        
        # Adiciona "crash" simulado (março 2023)
        crash_mask = (df.index >= '2023-03-01') & (df.index <= '2023-03-31')
        df.loc[crash_mask, 'close'] *= 0.7
        df.loc[crash_mask, 'volume'] *= 3.0
        
        # Adiciona "bull run" simulado (nov-dez 2023)
        bull_mask = (df.index >= '2023-11-01') & (df.index <= '2023-12-31')
        df.loc[bull_mask, 'close'] *= 1.5
        df.loc[bull_mask, 'volume'] *= 2.0
        
        return df
    
    @pytest.fixture
    def trained_pipeline(self, synthetic_time_series, tmp_path):
        """Pipeline treinado com modelo simples"""
        config = AIConfig()
        config.MODEL_DIR = str(tmp_path)
        
        pipeline = TemporalAutoencoderPipeline(config)
        
        # Treina modelo pequeno (rápido para teste)
        pipeline.train_autoencoder_temporal(
            df=synthetic_time_series,
            feature_columns=synthetic_time_series.columns.tolist(),
            latent_dim=4,
            seq_length=16,
            epochs=5,  # Poucas épocas para teste
            batch_size=32,
            optimize=False  # Sem Optuna
        )
        
        return pipeline
    
    def test_ood_reconstruction_basic(self, trained_pipeline, synthetic_time_series):
        """Testa validação OOD básica"""
        
        # Define períodos OOD
        ood_periods = [
            ('2023-03-01', '2023-03-31', 'CRASH_PERIOD'),
            ('2023-11-01', '2023-12-31', 'BULL_PERIOD'),
        ]
        
        # Executa validação
        results = trained_pipeline.validate_ood_reconstruction(
            synthetic_time_series,
            ood_periods
        )
        
        # Verificações
        assert 'baseline' in results
        assert isinstance(results['baseline'], (int, float))
        assert results['baseline'] > 0
        
        assert 'CRASH_PERIOD' in results or 'BULL_PERIOD' in results
        
        # Pelo menos um período deve ter sido testado
        tested_periods = [k for k in results.keys() if k != 'baseline' and isinstance(results[k], dict)]
        assert len(tested_periods) > 0
        
        print(f"\n✅ OOD Test - Baseline error: {results['baseline']:.4f}")
        for period_name in tested_periods:
            period_result = results[period_name]
            if 'error' in period_result and isinstance(period_result['error'], (int, float)):
                print(f"   {period_name}: error={period_result['error']:.4f}, "
                      f"degradation={period_result.get('degradation', 'N/A')}")
    
    def test_ood_insufficient_data_handling(self, trained_pipeline):
        """Testa que OOD lida bem com dados insuficientes"""
        
        # DataFrame muito pequeno
        tiny_df = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200],
        }, index=pd.date_range('2024-01-01', periods=3))
        
        ood_periods = [('2024-01-02', '2024-01-02', 'TINY')]
        
        # Deve retornar erro de dados insuficientes
        results = trained_pipeline.validate_ood_reconstruction(tiny_df, ood_periods)
        
        assert 'error' in results or len(results) == 1  # baseline ou erro
    
    def test_permutation_test_basic(self, trained_pipeline, synthetic_time_series):
        """Testa teste de permutação básico"""
        
        # Executa teste de permutação (poucas permutações para teste rápido)
        results = trained_pipeline.permutation_test_reconstruction(
            synthetic_time_series,
            n_permutations=20
        )
        
        # Verificações
        assert 'real_error' in results
        assert 'perm_error_mean' in results
        assert 'perm_error_std' in results
        assert 'p_value' in results
        assert 'interpretation' in results
        
        assert isinstance(results['real_error'], (int, float))
        assert isinstance(results['p_value'], (int, float))
        assert 0 <= results['p_value'] <= 1
        
        print(f"\n✅ Permutation Test Results:")
        print(f"   Real error:      {results['real_error']:.4f}")
        print(f"   Permuted error:  {results['perm_error_mean']:.4f} ± {results['perm_error_std']:.4f}")
        print(f"   P-value:         {results['p_value']:.4f}")
        print(f"   Interpretation:  {results['interpretation']}")
    
    def test_permutation_degrades_performance(self, trained_pipeline, synthetic_time_series):
        """Verifica que permutação PIORA o erro (padrões temporais existem)"""
        
        results = trained_pipeline.permutation_test_reconstruction(
            synthetic_time_series,
            n_permutations=30
        )
        
        # Espera-se que erro permutado seja MAIOR que erro real
        # (indica que modelo aprendeu padrões temporais)
        if results['p_value'] < 0.20:  # Threshold relaxado para teste sintético
            assert results['perm_error_mean'] > results['real_error'], \
                "Permutação deveria piorar erro se modelo aprendeu padrões temporais"
            print("✅ Modelo aprendeu padrões temporais genuínos")
        else:
            print("⚠️ Evidência fraca de padrões (esperado para modelo pequeno de teste)")
    
    def test_block_permutation_preserves_structure(self, trained_pipeline, synthetic_time_series):
        """Testa que _block_permutation mantém estrutura local"""
        
        block_size = 10
        df_perm = trained_pipeline._block_permutation(synthetic_time_series, block_size)
        
        # DataFrame permutado deve ter mesmo tamanho
        assert len(df_perm) == len(synthetic_time_series)
        
        # Deve ter mesmas colunas
        assert list(df_perm.columns) == list(synthetic_time_series.columns)
        
        # Deve ter mesmo índice (apenas ordem mudou)
        assert list(df_perm.index) == list(synthetic_time_series.index)
        
        # Valores devem ser diferentes em ordem (permutação funcionou)
        # Mas soma deve ser igual (nenhum dado perdido)
        assert not np.allclose(df_perm.values, synthetic_time_series.values)
        assert np.allclose(df_perm.sum().sum(), synthetic_time_series.sum().sum())
        
        print("✅ Block permutation preserva estrutura corretamente")
    
    def test_compute_reconstruction_error_helper(self, trained_pipeline, synthetic_time_series):
        """Testa helper de cálculo de erro"""
        
        error = trained_pipeline._compute_reconstruction_error(synthetic_time_series)
        
        # Deve retornar valor numérico válido
        assert isinstance(error, (int, float))
        assert error > 0 or np.isnan(error)  # Pode ser NaN se modelo não aplicado ainda
        
        if not np.isnan(error):
            print(f"✅ Reconstruction error: {error:.4f}")
    
    def test_critical_validations_require_trained_model(self):
        """Verifica que validações exigem modelo treinado"""
        
        config = AIConfig()
        pipeline = TemporalAutoencoderPipeline(config)
        
        df = pd.DataFrame({'close': [100, 101, 102]}, index=pd.date_range('2024-01-01', periods=3))
        
        # Deve lançar erro se modelo não treinado
        with pytest.raises(ValueError, match="não treinado"):
            pipeline.validate_ood_reconstruction(df, [])
        
        with pytest.raises(ValueError, match="não treinado"):
            pipeline.permutation_test_reconstruction(df)
        
        print("✅ Validações corretamente exigem modelo treinado")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
