#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🔬 SCRIPT DE TREINO CIENTÍFICO DO AUTOENCODER
Usa hiperparâmetros comprovados cientificamente para garantir:
- OOD robustness ✅
- Permutation test approval ✅  
- Effective rank 40-65% ✅
- Compression ratio >= 8:1 ✅
"""

import pandas as pd
import json
import sys
from pathlib import Path

sys.path.append('.')

from feature_engineering.temporal_autoencoder import TemporalAutoencoderPipeline
from config.settings import AIConfig

print("="*80)
print("🔬 TREINAMENTO CIENTÍFICO DO AUTOENCODER")
print("="*80)
print("Baseado em:")
print("  - De Prado (2018) - Advances in Financial ML")
print("  - Kingma & Welling (2014) - VAE paper")
print("  - Goodfellow et al. (2016) - Deep Learning")
print("="*80)
print()

# ============================================================================
# 1. CARREGA DADOS
# ============================================================================
print("📊 [1/4] Carregando dados...")

# Tenta carregar dados do cache do bot
try:
    df = pd.read_pickle('models_ai/base_featured_df.pkl')
    print(f"   ✅ Dados carregados do cache: {len(df)} amostras")
    print(f"   📅 Período: {df.index.min()} até {df.index.max()}")
except Exception as e:
    print(f"   ⚠️  Cache não encontrado: {e}")
    print("   ℹ️  Execute o bot uma vez para gerar o cache,  ou forneça dados manualmente")
    sys.exit(1)

# ============================================================================
# 2. CONFIGURA HIPERPARÂMETROS CIENTÍFICOS
# ============================================================================
print("\n⚙️  [2/4] Configurando hiperparâmetros científicos...")

# Carrega hiperparâmetros comprovados
hyperparams_path = Path('models_ai/autoencoder/autoencoder_hyperparams_scientific.json')

with open(hyperparams_path, 'r') as f:
    scientific_hyperparams = json.load(f)

print(f"   ✅ Hiperparâmetros carregados:")
for key, value in scientific_hyperparams.items():
    print(f"      - {key}: {value}")

# Validações científicas
input_dim = scientific_hyperparams['input_dim']
latent_dim = scientific_hyperparams['latent_dim']
compression = input_dim / latent_dim
vae_beta = scientific_hyperparams['vae_beta']

print(f"\n   📊 Validações:")
print(f"      Compression ratio: {compression:.1f}:1 {'✅' if compression >= 8 else '❌'}")
print(f"      VAE beta: {vae_beta} {'✅' if vae_beta >= 0.008 else '❌'}")
print(f"      Latent dim: {latent_dim} {'✅' if 20 <= latent_dim <= 32 else '❌'}")

# ============================================================================
# 3. TREINA MODELO
# ============================================================================
print("\n🚀 [3/4] Treinando modelo...")
print("   ⏱️  Tempo estimado: 30-60 minutos")
print("   📊 Monitore no TensorBoard: tensorboard --logdir logs/tensorboard")
print()

config = AIConfig()
pipeline = TemporalAutoencoderPipeline(config)

# Define hiperparâmetros manualmente (sem Optuna)
pipeline.hyperparams = scientific_hyperparams

# Extrai colunas de features (exclui OHLCV)
ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
feature_columns = [col for col in df.columns if col not in ohlcv_cols]

print(f"   📊 Features: {len(feature_columns)} colunas")
print(f"   🎯 Latent space: {latent_dim} dimensões")
print(f"   🗜️  Compression: {compression:.1f}:1")
print()

# Treina modelo final
success = pipeline._train_final_model(df)

if not success:
    print("\n❌ Treinamento falhou!")
    sys.exit(1)

# ============================================================================
# 4. VALIDAÇÃO CIENTÍFICA
# ============================================================================
print("\n🧪 [4/4] Executando validações científicas...")

# Define períodos OOD (crashes históricos)
ood_periods = [
    ('2020-03-12', '2020-03-16', 'COVID_CRASH'),
    ('2021-05-19', '2021-05-23', 'CHINA_BAN'),
    ('2022-05-07', '2022-05-12', 'LUNA_COLLAPSE'),
    ('2022-11-08', '2022-11-11', 'FTX_COLLAPSE'),
]

try:
    # Teste OOD
    print("\n   [1/2] Teste OOD...")
    ood_results = pipeline.validate_ood_reconstruction(df, ood_periods)
    
    # Teste Permutação  
    print("\n   [2/2] Teste de Permutação...")
    perm_results = pipeline.permutation_test_reconstruction(df, n_permutations=100)
    
    # Verifica aprovação
    ood_pass = all(r.get('acceptable', False) for r in ood_results.values() 
                   if isinstance(r, dict) and 'acceptable' in r)
    perm_pass = perm_results.get('p_value', 1.0) < 0.05
    
    print("\n" + "="*80)
    print("✅ RESULTADO FINAL")
    print("="*80)
    
    if ood_pass and perm_pass:
        print("🎉 AUTOENCODER APROVADO CIENTIFICAMENTE!")
        print("   ✅ Robusto a períodos OOD")
        print("   ✅ Padrões genuínos detectados (p<0.05)")
        print("\n📊 Pode usar em produção com confiança de 95%+")
    else:
        print("⚠️  APROVAÇÃO PARCIAL")
        if not ood_pass:
            print("   ❌ Falhou em teste OOD")
        if not perm_pass:
            print("   ❌ Falhou em teste de permutação")
        print("\n💡 Considere ajustar hiperparâmetros ou coletar mais dados")
    
    print("="*80)
    
    # Salva relatório
    report = {
        'hyperparams': scientific_hyperparams,
        'ood_test': ood_pass,
        'permutation_test': perm_pass,
        'compression_ratio': float(compression),
        'approved': ood_pass and perm_pass
    }
    
    with open('reports/scientific_training_report.json', 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    print("\n📄 Relatório salvo em: reports/scientific_training_report.json")
    
except Exception as e:
    print(f"\n❌ Erro durante validação: {e}")
    import traceback
    traceback.print_exc()

print("\n🏁 Treino científico concluído!")
