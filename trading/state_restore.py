# -----------------------------------------------------------------------------
# ARQUIVO: trading/state_restore.py
# -----------------------------------------------------------------------------

import os
import json # Pode ser usado para metadados ou se o estado for JSON-serializável
import pickle # Para serializar e deserializar objetos Python
from datetime import datetime, timezone # Importar timezone para UTC
from typing import Dict, Any, Optional

from utils.logger import get_logger
from config.settings import Config, AIConfig # Importar Config e AIConfig

logger = get_logger("StateRestore")

class StateRestore:
    """
    Gerencia o salvamento e restauração de checkpoints do estado do sistema.
    Isso é crucial para a persistência de dados (e.g., portfólio, AI state)
    entre sessões do bot, permitindo reinícios sem perda de progresso.
    """
    def __init__(self, config: Config, ai_config: AIConfig):
        if not isinstance(config, Config):
            raise TypeError("🚨 [ERRO RESTORE] 'config' deve ser uma instância da classe Config.")
        if not isinstance(ai_config, AIConfig):
            raise TypeError("🚨 [ERRO RESTORE] 'ai_config' deve ser uma instância da classe AIConfig.")

        self.config = config
        self.ai_config = ai_config
        # O diretório de checkpoints é configurado em AIConfig
        self.checkpoint_dir = self.ai_config.CHECKPOINT_DIR 

        try:
            # Cria o diretório de checkpoints se ele não existir
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            logger.info(f"💾 [RESTORE] StateRestore inicializado. Diretório de checkpoints: '{self.checkpoint_dir}'.")
        except OSError as e:
            logger.critical(f"🚨 [CRÍTICO RESTORE] Não foi possível criar o diretório de checkpoints '{self.checkpoint_dir}': {e}. Salvamento/Restauração podem falhar.", exc_info=True)
        except Exception as e:
            logger.critical(f"🚨 [CRÍTICO RESTORE] Erro inesperado ao inicializar StateRestore: {e}.", exc_info=True)


    def save_checkpoint(self, state: Dict[str, Any], name_prefix: str = "system_state"):
        """
        Salva um checkpoint completo do estado do sistema em um arquivo .pkl.
        Recomenda-se salvar estados que sejam facilmente serializáveis com pickle.

        Args:
            state (Dict[str, Any]): Dicionário contendo o estado do sistema a ser salvo.
                                     Ex: {'portfolio': portfolio.get_state(), 'ai_state': ai_controller.get_state()}.
            name_prefix (str): Prefixo para o nome do arquivo do checkpoint (e.g., "portfolio", "full_state").
        """
        if not isinstance(state, dict):
            logger.error(f"❌ [ERRO RESTORE] 'state' para salvar checkpoint deve ser um dicionário. Recebido: {type(state)}. Abortando salvamento.")
            return
        if not isinstance(name_prefix, str) or not name_prefix:
            logger.error(f"❌ [ERRO RESTORE] 'name_prefix' inválido. Deve ser uma string não vazia. Abortando salvamento.")
            return

        # Gera um timestamp no formato UTC para o nome do arquivo
        timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{name_prefix}_{timestamp}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        try:
            # Usa 'wb' para escrever em modo binário
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"✅ [RESTORE] Checkpoint '{filename}' salvo com sucesso em '{filepath}'.")
        except (IOError, OSError) as e:
            logger.error(f"❌ [ERRO RESTORE] Falha de I/O ao salvar o checkpoint '{filename}': {e}. Verifique permissões/espaço em disco.", exc_info=True)
        except pickle.PicklingError as e:
            logger.error(f"❌ [ERRO RESTORE] Erro de serialização (pickle) ao salvar o checkpoint '{filename}': {e}. Certifique-se de que o estado é serializável.", exc_info=True)
        except Exception as e:
            logger.critical(f"🚨 [CRÍTICO RESTORE] Erro inesperado ao salvar o checkpoint '{filename}': {e}.", exc_info=True)


    def restore_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Restaura o estado do checkpoint mais recente encontrado no diretório.
        Prioriza o arquivo com o timestamp mais recente no nome.

        Returns:
            Optional[Dict[str, Any]]: O dicionário de estado restaurado, ou None se nenhum
                                      checkpoint for encontrado ou se houver um erro.
        """
        latest_checkpoint_file = "unknown"
        try:
            # Lista todos os arquivos .pkl no diretório de checkpoints
            if not os.path.exists(self.checkpoint_dir):
                 logger.info(f"ℹ️ [RESTORE] Diretório inexistente: {self.checkpoint_dir}. Nada a restaurar.")
                 return None

            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pkl")]
            
            if not checkpoints:
                logger.info("ℹ️ [RESTORE] Nenhum checkpoint (.pkl) encontrado para restauração no diretório. Iniciando do zero.")
                return None

            # Ordena os checkpoints pelo timestamp embutido no nome do arquivo
            # Assume o formato "prefix_YYYYMMDD_HHMMSS.pkl"
            try:
                checkpoints.sort(key=lambda f: datetime.strptime(f.split('_')[-1].split('.')[0], "%Y%m%d_%H%M%S"), reverse=True)
            except ValueError:
                # Fallback: Ordena por data de modificação se o nome não bater
                checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(self.checkpoint_dir, f)), reverse=True)

            latest_checkpoint_file = checkpoints[0] # Pega o mais recente

            filepath = os.path.join(self.checkpoint_dir, latest_checkpoint_file)
            
            logger.info(f"🔄 [RESTORE] Tentando restaurar estado do checkpoint: '{latest_checkpoint_file}' de '{filepath}'.")

            # Usa 'rb' para ler em modo binário
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            logger.info(f"✅ [RESTORE] Estado restaurado com sucesso do checkpoint: '{latest_checkpoint_file}'.")
            return state
        except (FileNotFoundError, IOError, OSError) as e:
            logger.error(f"❌ [ERRO RESTORE] Falha de I/O ao restaurar do checkpoint '{latest_checkpoint_file}': {e}.", exc_info=True)
            return None
        except pickle.UnpicklingError as e:
            logger.error(f"❌ [ERRO RESTORE] Erro de deserialização (pickle) ao restaurar o checkpoint '{latest_checkpoint_file}': {e}.", exc_info=True)
            logger.warning("⚠️ [ALERTE RESTORE] O checkpoint pode estar corrompido ou é incompatível. Tentando iniciar o bot do zero.")
            return None 
        except Exception as e:
            logger.critical(f"🚨 [CRÍTICO RESTORE] Erro inesperado ao restaurar do checkpoint '{latest_checkpoint_file}': {e}. Abortando.", exc_info=True)
            return None