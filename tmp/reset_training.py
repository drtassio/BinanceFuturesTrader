import os
import glob
import shutil
import json

# Configurações de caminhos
BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "models_ai")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints_ai")
METADATA_PATH = os.path.join(MODELS_DIR, "training_metadata.json")

def cleanup():
    print("🧹 [CLEANUP] Iniciando limpeza seletiva para fresh start de treinamento...")

    # 1. Deletar bancos de dados do Optuna (.db)
    # Isso força o reinício do ciclo de 100 trials por especialista
    db_files = glob.glob(os.path.join(MODELS_DIR, "*.db"))
    for f in db_files:
        try:
            os.remove(f)
            print(f"✅ Removido: {os.path.basename(f)}")
        except Exception as e:
            print(f"❌ Erro ao remover {f}: {e}")

    # 2. Deletar modelos de especialistas SAC (.zip)
    # Mantemos o autoencoder pois ele é a base e demora para treinar
    spec_models = ["bull_specialist_sac.zip", "bear_specialist_sac.zip", "ranger_specialist_sac.zip", "hrl_master_ppo.zip"]
    for model in spec_models:
        path = os.path.join(MODELS_DIR, model)
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"✅ Removido modelo: {model}")
            except Exception as e:
                print(f"❌ Erro ao remover {model}: {e}")

    # 3. Deletar Gating Network (para novo bootstrap com dimensões corretas)
    gating_network = os.path.join(MODELS_DIR, "soft_gating_network.pt")
    if os.path.exists(gating_network):
        os.remove(gating_network)
        print("✅ Removido: soft_gating_network.pt")

    # 4. Limpar Checkpoints
    if os.path.exists(CHECKPOINTS_DIR):
        print(f"📂 Limpando diretório de checkpoints: {CHECKPOINTS_DIR}")
        for item in os.listdir(CHECKPOINTS_DIR):
            item_path = os.path.join(CHECKPOINTS_DIR, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                 print(f"⚠️ Falha ao remover checkpoint {item}: {e}")

    # 5. Resetar Training Metadata
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r') as f:
                data = json.load(f)
            
            # Resetar status de treinamento dos agentes (mantendo autoencoder)
            for model_key in ["bull_specialist_sac", "bear_specialist_sac", "ranger_specialist_sac", "hrl_master_ppo"]:
                if model_key in data.get("models", {}):
                    data["models"][model_key]["trained"] = False
                    data["models"][model_key]["trained_on"] = None
            
            with open(METADATA_PATH, 'w') as f:
                json.dump(data, f, indent=4)
            print("✅ Metadata de treinamento resetado com sucesso.")
        except Exception as e:
            print(f"❌ Erro ao resetar metadata: {e}")

    print("\n🚀 [PRONTO] O ambiente está limpo. Ao rodar 'python run_bot.py', o sistema iniciará a otimização de 100 trials do zero.")

if __name__ == "__main__":
    cleanup()
