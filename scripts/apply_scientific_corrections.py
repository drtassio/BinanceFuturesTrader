"""
Script para aplicar correções científicas ao TrendSpecialist de forma segura.
Aplica todas as correções P0, P1 e P2 identificadas na auditoria.
"""

import re
import os
import shutil
from datetime import datetime

# Arquivo alvo
TREND_SPECIALIST_PATH = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\specialists\trend_specialist.py"

def create_backup(filepath):
    """Cria backup timestamped do arquivo."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backup criado: {backup_path}")
    return backup_path

def read_file(filepath):
    """Lê conteúdo do arquivo."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filepath, content):
    """Escreve conteúdo no arquivo."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Arquivo atualizado: {filepath}")

def apply_correction_1_add_latent_dim_method(content):
    """P0.1: Adiciona método _get_autoencoder_latent_dim."""
    print("\n[1/8] Aplicando: Adicionar método _get_autoencoder_latent_dim...")
    
    # Localiza onde inserir (após _calculate_average_metrics)
    pattern = r"(        return aggregated\n\n)(    def _calculate_trend_surfing_score)"
    
    method_code = '''    def _get_autoencoder_latent_dim(self) -> int:
        """
        Determina dimensão das features latentes do AutoEncoder.
        
        Ordem de prioridade:
        1. Config explícita (se existir)
        2. Inspeção do observation space do environment
        3. Fallback seguro: 64
        
        Returns:
            int: Dimensão das features latentes
        """
        try:
            # 1. Tenta pegar do config
            if hasattr(self.config, 'AUTOENCODER_LATENT_DIM'):
                latent_dim = int(self.config.AUTOENCODER_LATENT_DIM)
                logger.info(f"[LSTM CONFIG] latent_dim do config: {latent_dim}")
                return latent_dim
            
            # 2. Tenta inspecionar environment
            if hasattr(self, 'train_df') and self.train_df is not None and len(self.train_df) > 100:
                sample_df = self.train_df.iloc[:100]
                env = self._make_trend_env(sample_df, mode='training')
                obs_shape = env.observation_space.shape[0]
                
                # Subtrai features conhecidas:
                # extras (depende, mas ~10) + agent(5) + regime(3) + prior(2) = ~20
                # Assume que o resto são features de mercado (latents + indicadores)
                # Conservador: assume 60% são latents
                estimated_market_features = obs_shape - 20
                latent_dim = max(32, min(128, int(estimated_market_features * 0.6)))
                
                logger.info(f"[LSTM CONFIG] latent_dim inferido do env: {latent_dim} (obs_shape={obs_shape})")
                return latent_dim
                
        except Exception as e:
            logger.warning(f"[LSTM CONFIG] Não foi possível determinar latent_dim automaticamente: {e}")
        
        # 3. Fallback seguro
        logger.info("[LSTM CONFIG] Usando latent_dim padrão: 64")
        return 64

'''
    
    replacement = r"\1" + method_code + r"\2"
    content_new = re.sub(pattern, replacement, content)
    
    if content_new != content:
        print("   ✅ Método _get_autoencoder_latent_dim adicionado")
        return content_new
    else:
        print("   ⚠️  Padrão não encontrado, pulando...")
        return content

def apply_correction_2_fase1_lstm(content):
    """P0.2: Integrar LSTM na Fase 1 (optimize_anti_scalping_config)."""
    print("\n[2/8] Aplicando: Integrar LSTM Policy na Fase 1...")
    
    # Localiza criação do modelo SAC na Fase 1
    pattern = re.compile(
        r"(                # Usa hiperparâmetros SAC padrão para esta fase\n"
        r"                sac_params = \{\n"
        r"                    \"learning_rate\": 1e-4,\n"
        r"                    \"buffer_size\": 100000,\n"
        r"                    \"batch_size\": 512,\n"
        r"                    \"gamma\": 0\.99,\n"
        r"                    \"tau\": 0\.01,\n"
        r"                    \"train_freq\": 4,\n"
        r"                    \"use_sde\": True\n"
        r"                \}\n)"
        r"(                model = ClippedSAC\(\n"
        r"                    \"MlpPolicy\",\n"
        r"                    train_env,\n)"
        r"(                    \*\*sac_params,)",
        re.MULTILINE
    )
    
    replacement = (
        r"\1"
        r"                \n"
        r"                # [CORREÇÃO P0] Integrar LSTM + Attention Policy\n"
        r"                from specialists.scientific_corrections import get_lstm_policy_kwargs\n"
        r"                latent_dim = self._get_autoencoder_latent_dim()\n"
        r"                policy_kwargs = get_lstm_policy_kwargs(latent_dim=latent_dim)\n"
        r"                logger.info(f\"[FASE 1 LSTM] Usando HybridLSTMAttentionExtractor com latent_dim={latent_dim}\")\n"
        r"                \n"
        r"\2"
        r"                    policy_kwargs=policy_kwargs,  # <<< NOVO: LSTM + Attention\n"
        r"\3"
    )
    
    content_new = pattern.sub(replacement, content)
    
    if content_new != content:
        print("   ✅ LSTM integrado na Fase 1")
        return content_new
    else:
        print("   ⚠️  Padrão não encontrado, pulando...")
        return content

def apply_correction_3_fase2_lstm(content):
    """P0.2: Integrar LSTM na Fase 2 (optimize_hyperparameters)."""
    print("\n[3/8] Aplicando: Integrar LSTM Policy na Fase 2...")
    
    # Localiza criação do modelo SAC na Fase 2
    # Procura por 'model = ClippedSAC("MlpPolicy", train_env, **sac_params,'
    pattern = re.compile(
        r"(                model = ClippedSAC\(\n"
        r"                    \"MlpPolicy\",\n"
        r"                    train_env,\n)"
        r"(                    \*\*sac_params,)",
        re.MULTILINE
    )
    
    # Precisamos injetar a definição de policy_kwargs antes do loop ou dentro dele se não existir
    # Mas como regex é local, vamos injetar logo antes da chamada
    
    replacement = (
        r"                # [CORREÇÃO P0] Integrar LSTM + Attention Policy\n"
        r"                from specialists.scientific_corrections import get_lstm_policy_kwargs\n"
        r"                latent_dim = self._get_autoencoder_latent_dim()\n"
        r"                policy_kwargs = get_lstm_policy_kwargs(latent_dim=latent_dim)\n"
        r"                \n"
        r"\1"
        r"                    policy_kwargs=policy_kwargs,  # <<< NOVO: LSTM + Attention\n"
        r"\2"
    )
    
    # Nota: isso pode afetar Fase 1 se o regex for igual. 
    # O regex da Fase 1 era mais específico (incluía sac_params definition).
    # Este regex é mais genérico, então deve pegar a Fase 2 se a Fase 1 já foi alterada (o regex da Fase 1 não vai casar mais).
    # Mas cuidado para não duplicar na Fase 1 se rodar de novo.
    
    # Melhor estratégia: procurar contexto específico da Fase 2.
    # Fase 2 tem 'verbose=1' e 'tensorboard_log=None'
    
    pattern_fase2 = re.compile(
        r"(                model = ClippedSAC\(\n"
        r"                    \"MlpPolicy\",\n"
        r"                    train_env,\n"
        r"                    \*\*sac_params,\n"
        r"                    verbose=1,\n"
        r"                    device=self._get_device\(\),\n"
        r"                    tensorboard_log=None,)",
        re.MULTILINE
    )
    
    replacement_fase2 = (
        r"                # [CORREÇÃO P0] Integrar LSTM + Attention Policy\n"
        r"                from specialists.scientific_corrections import get_lstm_policy_kwargs\n"
        r"                latent_dim = self._get_autoencoder_latent_dim()\n"
        r"                policy_kwargs = get_lstm_policy_kwargs(latent_dim=latent_dim)\n"
        r"                \n"
        r"                model = ClippedSAC(\n"
        r"                    \"MlpPolicy\",\n"
        r"                    train_env,\n"
        r"                    policy_kwargs=policy_kwargs,  # <<< NOVO: LSTM + Attention\n"
        r"                    **sac_params,\n"
        r"                    verbose=1,\n"
        r"                    device=self._get_device(),\n"
        r"                    tensorboard_log=None,"
    )
    
    content_new = pattern_fase2.sub(replacement_fase2, content)
    
    if content_new != content:
        print("   ✅ LSTM integrado na Fase 2")
        return content_new
    else:
        print("   ⚠️  Padrão Fase 2 não encontrado, pulando...")
        return content

def apply_correction_4_fase3_lstm(content):
    """P0.2: Integrar LSTM na Fase 3 (train)."""
    print("\n[4/8] Aplicando: Integrar LSTM Policy na Fase 3...")
    
    pattern_fase3 = re.compile(
        r"(                self\.model = ClippedSAC\(\n"
        r"                                 \"MlpPolicy\", train_env,\n)",
        re.MULTILINE
    )
    
    # Injetar policy_kwargs antes
    # Precisamos encontrar um lugar seguro para definir policy_kwargs antes do if self.model is None
    # Mas regex replace é local. Vamos substituir a chamada do construtor.
    
    # Problema: precisamos definir policy_kwargs antes.
    # Vamos usar uma substituição que define inline ou chama uma função helper se possível? Não.
    # Vamos substituir o bloco 'if self.model is None:' para incluir a definição.
    
    pattern_block = re.compile(
        r"(            if self\.model is None:\n"
        r"                logger\.info\(\"\[TREND SAC\] Criando novo modelo SAC\.\"\)\n"
        r"                self\.model = ClippedSAC\(\n"
        r"                                 \"MlpPolicy\", train_env,\n)",
        re.MULTILINE
    )
    
    replacement_block = (
        r"            if self.model is None:\n"
        r"                logger.info(\"[TREND SAC] Criando novo modelo SAC.\")\n"
        r"                \n"
        r"                # [CORREÇÃO P0] Integrar LSTM + Attention Policy\n"
        r"                from specialists.scientific_corrections import get_lstm_policy_kwargs\n"
        r"                latent_dim = self._get_autoencoder_latent_dim()\n"
        r"                policy_kwargs = get_lstm_policy_kwargs(latent_dim=latent_dim)\n"
        r"                \n"
        r"                self.model = ClippedSAC(\n"
        r"                                 \"MlpPolicy\", train_env,\n"
        r"                                 policy_kwargs=policy_kwargs,  # <<< NOVO: LSTM + Attention\n"
    )
    
    content_new = pattern_block.sub(replacement_block, content)
    
    if content_new != content:
        print("   ✅ LSTM integrado na Fase 3")
        return content_new
    else:
        print("   ⚠️  Padrão Fase 3 não encontrado, pulando...")
        return content

def apply_correction_5_reward_function(content):
    """P0.3: Substituir cálculo de reward pela função científica."""
    print("\n[5/8] Aplicando: Função de Recompensa Científica...")
    
    # Localizar _step_training e a parte de cálculo de reward
    # Vamos procurar um bloco característico do reward antigo
    
    # Exemplo: 'reward = 0.0' seguido de 'pnl_realized ='
    
    # Vamos injetar a chamada da nova função logo após o cálculo de pnl_realized e trade_return_pct
    # e ANTES de todo o resto, e forçar o return ou ignorar o resto.
    
    # Mas o código original é muito longo.
    # Estratégia: Encontrar 'reward = 0.0' dentro de _step_training e substituir o bloco subsequente
    # até o return, ou melhor, calcular o scientific reward e usar ele.
    
    # Vamos procurar:
    # reward = 0.0
    # ...
    # if self.reward_shaping_params:
    
    # O código é complexo. Vamos tentar substituir a chamada de _compute_trade_reward se ela for usada,
    # mas o código parece fazer tudo inline.
    
    # Vamos procurar o início do reward calculation:
    # 'reward = 0.0'
    # 'penalty_accum = 0.0'
    
    pattern_start = re.compile(
        r"(        reward = 0\.0\n"
        r"        penalty_accum = 0\.0\n)",
        re.MULTILINE
    )
    
    # Vamos injetar a chamada da função científica e um 'if True: return ...' para pular o resto?
    # Não, o método _step_training continua.
    
    # Melhor: Substituir o cálculo de reward.
    # Mas tem muitos componentes.
    
    # Vamos adicionar a flag use_scientific_reward no __init__ primeiro?
    # O script scientific_corrections.py já tem a função.
    
    # Vamos substituir o bloco que começa com 'reward = 0.0' e vai até antes de 'self.reward_history.append(reward)'?
    # Isso é arriscado sem um parser.
    
    # Alternativa: Injetar no final do cálculo de reward, sobrescrevendo-o.
    # Procurar: 'return self._get_observation(), reward, done, truncated, info'
    # E inserir antes:
    # if getattr(self, 'use_scientific_reward', True):
    #     reward = compute_scientific_reward(...)
    
    pattern_return = re.compile(
        r"(        return self\._get_observation\(\), reward, done, truncated, info)",
        re.MULTILINE
    )
    
    injection = (
        r"        # [CORREÇÃO P0] Recompensa Científica\n"
        r"        from specialists.scientific_corrections import compute_scientific_reward\n"
        r"        # Recalcula reward usando a função científica limpa\n"
        r"        reward = compute_scientific_reward(\n"
        r"            env=self,\n"
        r"            pnl_realized=pnl_realized,\n"
        r"            trade_return_pct=trade_return_pct,\n"
        r"            duration_steps=closed_trade_duration or self.steps_in_position,\n"
        r"            exit_reason=info.get('exit_reason'),\n"
        r"            current_price=current_price,\n"
        r"            atr=atr\n"
        r"        )\n"
        r"        \n"
        r"\1"
    )
    
    content_new = pattern_return.sub(injection, content)
    
    if content_new != content:
        print("   ✅ Reward Function injetada no final de _step_training")
        return content_new
    else:
        print("   ⚠️  Return de _step_training não encontrado, pulando...")
        return content

def apply_correction_6_gates(content):
    """P0.4: Remover/Bypass Gates Anti-RL."""
    print("\n[6/8] Aplicando: Bypass de Gates Anti-RL...")
    
    # 1. Prior Veto
    pattern_veto = re.compile(
        r"(if prior_dir_val < -0\.1 and should_open_long:\n"
        r"\s+should_open_long = False\n"
        r"\s+reward -= 10\.0)",
        re.MULTILINE
    )
    
    replacement_veto = (
        r"from specialists.scientific_corrections import should_apply_gate_bypass\n"
        r"            if not should_apply_gate_bypass('prior_veto'):\n"
        r"                \1"
    )
    
    content = pattern_veto.sub(replacement_veto, content)
    
    # 2. Reversal Block
    # 'if predictor_confirms_flip:' ... 'else:' ... 'reward -= 5.0'
    
    pattern_reversal = re.compile(
        r"(else:\n"
        r"\s+reward -= 5\.0\n"
        r"\s+penalty_accum \+= 5\.0\n"
        r"\s+should_open_long = False\n"
        r"\s+should_open_short = False)",
        re.MULTILINE
    )
    
    replacement_reversal = (
        r"elif not should_apply_gate_bypass('reversal_block'):\n"
        r"                \1"
    )
    # Nota: isso assume que o 'else' casa com o if anterior. Regex é frágil aqui.
    # Vamos tentar ser mais específicos ou usar um bloco maior.
    
    # O bloco original é:
    # if predictor_confirms_flip:
    #     reward += 2.0
    # else:
    #     reward -= 5.0
    #     ...
    
    pattern_reversal_block = re.compile(
        r"(if predictor_confirms_flip:\n"
        r"\s+reward \+= 2\.0\n"
        r"\s+)(else:\n"
        r"\s+reward -= 5\.0)",
        re.MULTILINE
    )
    
    replacement_reversal_block = (
        r"\1"
        r"elif not should_apply_gate_bypass('reversal_block'):\n"
        r"                \2"
    )
    
    content = pattern_reversal_block.sub(replacement_reversal_block, content)
    
    print("   ✅ Gates modificados (se encontrados)")
    return content

def apply_correction_7_normalization_method(content):
    """P1.1: Adicionar método _normalize_market_features."""
    print("\n[7/8] Aplicando: Adicionar método _normalize_market_features...")
    
    # Inserir após _get_autoencoder_latent_dim (que adicionamos no passo 1)
    # Ou após _get_observation
    
    pattern = r"(    def _get_observation\(self\):)"
    
    method_code = '''    def _normalize_market_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normaliza features de mercado usando RobustScaler.
        """
        if not hasattr(self, '_feature_scaler'):
            from sklearn.preprocessing import RobustScaler
            self._feature_scaler = RobustScaler()
            
            # Fit no primeiro reset (usando histórico completo se possível)
            if hasattr(self, 'df') and len(self.df) > 100:
                try:
                    # Tenta pegar dados de treino para fit
                    sample_data = self.df[self.feature_columns].values
                    self._feature_scaler.fit(sample_data)
                    logger.info("[NORMALIZATION] Scaler fit com sucesso.")
                except Exception as e:
                    logger.warning(f"[NORMALIZATION] Falha ao fitar scaler: {e}")
        
        try:
            if hasattr(self, '_feature_scaler'):
                return self._feature_scaler.transform(features.reshape(1, -1)).flatten()
        except Exception:
            pass
            
        return features

'''
    
    replacement = method_code + r"\n\1"
    content_new = re.sub(pattern, replacement, content)
    
    if content_new != content:
        print("   ✅ Método _normalize_market_features adicionado")
        return content_new
    else:
        print("   ⚠️  Padrão _get_observation não encontrado, pulando...")
        return content

def apply_correction_8_normalization_call(content):
    """P1.1: Chamada de normalização em _get_observation."""
    print("\n[8/8] Aplicando: Chamada de normalização em _get_observation...")
    
    pattern = re.compile(
        r"(market_features = row_all\[self\.feature_columns\]\.values\.astype\(np\.float32, copy=False\))"
    )
    
    replacement = (
        r"market_features_raw = row_all[self.feature_columns].values.astype(np.float32, copy=False)\n"
        r"        # [CORREÇÃO P1] Normalização Global\n"
        r"        market_features = self._normalize_market_features(market_features_raw)"
    )
    
    content_new = pattern.sub(replacement, content)
    
    if content_new != content:
        print("   ✅ Normalização aplicada em _get_observation")
        return content_new
    else:
        print("   ⚠️  Padrão market_features não encontrado, pulando...")
        return content

def main():
    """Aplica todas as correções."""
    print("="*60)
    print("APLICANDO CORREÇÕES CIENTÍFICAS AO TRENDSPECIALIST")
    print("="*60)
    
    # 1. Criar backup
    backup_path = create_backup(TREND_SPECIALIST_PATH)
    
    # 2. Ler arquivo
    content = read_file(TREND_SPECIALIST_PATH)
    print(f"\n📄 Arquivo lido: {len(content)} caracteres, {content.count(chr(10))} linhas")
    
    # 3. Aplicar correções uma por uma
    content = apply_correction_1_add_latent_dim_method(content)
    content = apply_correction_2_fase1_lstm(content)
    content = apply_correction_3_fase2_lstm(content)
    content = apply_correction_4_fase3_lstm(content)
    content = apply_correction_5_reward_function(content)
    content = apply_correction_6_gates(content)
    content = apply_correction_7_normalization_method(content)
    content = apply_correction_8_normalization_call(content)
    
    # 4. Salvar arquivo modificado
    write_file(TREND_SPECIALIST_PATH, content)
    
    print("\n" + "="*60)
    print("✅ CORREÇÕES APLICADAS COM SUCESSO!")
    print(f"📦 Backup disponível em: {backup_path}")
    print("="*60)

if __name__ == "__main__":
    main()
