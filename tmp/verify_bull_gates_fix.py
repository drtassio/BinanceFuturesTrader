import pandas as pd
import numpy as np
import os
import sys

# Adiciona o diretório raiz ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from specialists.bull_specialist import BullTradingEnv
from config.settings import AIConfig, TradingConfig

def verify_fix():
    print(" iniciando verificação do fix do BullSpecialist e Prior Pressure...")
    
    # Criar dados mockados
    num_steps = 100
    dates = pd.date_range('2026-01-01', periods=num_steps, freq='5T')
    df = pd.DataFrame({
        'open': np.random.rand(num_steps) * 100,
        'high': np.random.rand(num_steps) * 100,
        'low': np.random.rand(num_steps) * 100,
        'close': np.random.rand(num_steps) * 100,
        'volume': np.random.rand(num_steps) * 1000,
        'tp_prior_dir': np.random.uniform(-1, 1, num_steps),  # Mix de direções
        'tp_prior_conf': np.random.uniform(0, 1, num_steps),
        'tp_regime_up': np.random.rand(num_steps),
        'tp_regime_down': np.random.rand(num_steps),
        'tp_regime_sideways': np.random.rand(num_steps),
        'tp_uncertainty': np.zeros(num_steps),
        'tp_duration_median': np.ones(num_steps) * 50
    }, index=dates)
    
    # Garantir algumas colunas extras necessárias pelo TrendFollowingEnv
    df['atr_5m'] = df['close'] * 0.01
    df['ema_trend_5m'] = 0.0
    
    config = AIConfig()
    trading_config = TradingConfig()
    
    # 1. Testar BullTradingEnv em modo Otimização
    print("\n--- TESTE 1: BullTradingEnv (Optimization Mode) ---")
    env = BullTradingEnv(df, config, mode='optimization')
    env.reset()
    
    shorts_allowed = 0
    prior_pressure_exits = 0
    
    # Simular 50 steps de exploração (exploring=True por default no início se steps < warmup)
    env.exploration_warmup_steps = 1000 # Forçar exploring=True
    
    for i in range(50):
        # Ação forçando voto curto (negativo)
        # action: [voto, sl_mult, leverage]
        action = np.array([-1.0, 1.0, 1.0])
        obs, reward, done, truncated, info = env.step(action)
        
        # Verificar trend_allows_short nos logs seria difícil, vamos olhar o comportamento
        if env.position < 0:
            shorts_allowed += 1
            
        if info.get('exit_reason') == 'Agent Decision - prior pressure':
            prior_pressure_exits += 1
            
    print(f"Número de SHORTS abertos em BullSpecialist: {shorts_allowed} (Esperado: 0)")
    print(f"Saídas por Prior Pressure durante Exploração: {prior_pressure_exits} (Esperado: 0)")
    
    # 2. Testar Prior Pressure em modo normal após warmup
    print("\n--- TESTE 2: Prior Pressure desativado em Warmup ---")
    # Resetar env
    env.reset()
    env.exploration_warmup_steps = 100 # Exploring=True
    
    # Forçar posição LONG
    env.position = 1
    env.entry_price = 50
    env.initial_notional_value = 1000
    env.steps_in_position = 50 # Passou o grace period
    
    # Forçar prior oposto
    env.df.iloc[env.start_idx + env.current_step, env.df.columns.get_loc('tp_prior_dir')] = -0.9
    env.df.iloc[env.start_idx + env.current_step, env.df.columns.get_loc('tp_prior_conf')] = 0.9
    
    # Step com voto que confirmaria a saída se não fosse o fix
    action = np.array([-0.8, 1.0, 1.0]) 
    _, _, _, _, info = env.step(action)
    
    if info.get('exit_reason') == 'Agent Decision - prior pressure':
        print("FALHA: Prior Pressure ainda ativo durante exploring=True")
    else:
        print("SUCESSO: Prior Pressure desativado durante exploring=True")

    # 3. Testar Prior Pressure ATIVADO após warmup
    print("\n--- TESTE 3: Prior Pressure ativado após Warmup ---")
    env.current_step = 200 # Passou o warmup de 100
    # Forçar posição
    env.position = 1
    env.steps_in_position = 50
    
    # Forçar prior oposto
    env.df.iloc[env.start_idx + env.current_step, env.df.columns.get_loc('tp_prior_dir')] = -0.9
    env.df.iloc[env.start_idx + env.current_step, env.df.columns.get_loc('tp_prior_conf')] = 0.9
    
    action = np.array([-0.8, 1.0, 1.0])
    _, _, _, _, info = env.step(action)
    
    if info.get('exit_reason') == 'Agent Decision - prior pressure':
        print("SUCESSO: Prior Pressure ativado após exploring=False")
    else:
        print(f"FALHA: Prior Pressure não ativou. Exit Reason: {info.get('exit_reason')}")

if __name__ == "__main__":
    verify_fix()
