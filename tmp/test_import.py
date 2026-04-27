import os
import sys

# Adiciona o diretório raiz ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

print("Tentando importar BullTradingEnv...")
try:
    from specialists.bull_specialist import BullTradingEnv
    print("Sucesso!")
except Exception as e:
    print(f"Erro: {e}")
    import traceback
    traceback.print_exc()
