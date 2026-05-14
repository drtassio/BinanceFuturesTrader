# -----------------------------------------------------------------------------
# ARQUIVO: utils/logger.py
# -----------------------------------------------------------------------------

import logging
import os
import sys
import threading
import io
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from collections import deque
from typing import Deque, List, Optional, Tuple, Dict, Any

_loggers: Dict[str, logging.Logger] = {}
_handlers: List[logging.Handler] = []
_phase_file_handlers: Dict[str, logging.Handler] = {}
_lock = threading.Lock()

# --- Constantes para Níveis de Log ---
LOG_LEVEL_DEBUG = logging.DEBUG
LOG_LEVEL_INFO = logging.INFO
LOG_LEVEL_WARNING = logging.WARNING
LOG_LEVEL_ERROR = logging.ERROR
LOG_LEVEL_CRITICAL = logging.CRITICAL

def _prepare_utf8_stream(stream: Any) -> Any:
    """
    Garante que o stream suporte unicode (UTF-8), mesmo em consoles cp1252.
    """
    reconfig = getattr(stream, "reconfigure", None)
    if callable(reconfig):
        try:
            reconfig(encoding="utf-8", errors="replace")
            return stream
        except Exception:
            pass
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        try:
            return io.TextIOWrapper(buffer, encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass
    return stream

class UIRedirectHandler(logging.Handler):
    """
    Um handler de log que redireciona os registros para um deque,
    que pode ser usado para exibição em uma interface de usuário (CLI ou Web UI).
    """
    def __init__(self, log_deque: Deque[str], level: int = LOG_LEVEL_INFO):
        super().__init__(level=level)
        if not isinstance(log_deque, deque):
            raise TypeError("🚨 [ERRO LOGGER] 'log_deque' deve ser uma instância de collections.deque.")
        self.log_deque = log_deque
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
        
    def emit(self, record: logging.LogRecord):
        """
        Adiciona o registro formatado ao deque.
        """
        try:
            log_entry = self.format(record)
            self.log_deque.append(log_entry)
        except Exception:
            self.handleError(record)
            sys.stderr.write(f"🚨 [ERRO LOGGER] Falha ao emitir log para UI: {record.getMessage()}\n")

class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    Handler de rotação de arquivos que é mais seguro em ambientes multithread,
    especialmente no Windows, ao garantir que o arquivo seja fechado antes da rotação.
    """
    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        
        super().doRollover()
        
def _setup_file_handler(log_dir: str, log_filename: str, level: int) -> 'SafeTimedRotatingFileHandler':
    """
    Função interna para configurar o handler de arquivo rotativo.
    """
    file_handler = None
    for handler in _handlers:
        if isinstance(handler, SafeTimedRotatingFileHandler):
            file_handler = handler
            break
    
    if file_handler is None:
        project_log_base_dir = os.path.join(os.getcwd(), "logs")
        app_logs_dir = os.path.join(project_log_base_dir, "app_logs")
        
        os.makedirs(app_logs_dir, exist_ok=True)
        log_file_path = os.path.join(app_logs_dir, log_filename)

        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'
        formatter = logging.Formatter(log_format)

        file_handler = SafeTimedRotatingFileHandler(
            log_file_path,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level) # Nível passado como argumento (será DEBUG)
        _handlers.append(file_handler)
        logging.getLogger(__name__).info("[LOGGER SETUP] Handler de arquivo configurado: '%s' com level %s.", log_file_path, logging.getLevelName(level))
    
    return file_handler

def setup_ui_logging(log_deque: Deque[str], level: int = LOG_LEVEL_INFO):
    """
    Configura o handler da UI para o logger raiz.
    """
    with _lock:
        root_logger = logging.getLogger()
        
        if any(isinstance(h, UIRedirectHandler) for h in root_logger.handlers):
            return

        # Modifica o nível padrão do handler da UI para DEBUG para que os logs do Autoencoder (DEBUG) sejam exibidos
        # A implementação atual em run_bot.py já passa LOG_LEVEL_INFO, mas o handler deve ser criado no nível desejado.
        ui_handler = UIRedirectHandler(log_deque, level=LOG_LEVEL_DEBUG) 
        
        # Define o nível do logger raiz para DEBUG para que todas as mensagens passem
        root_logger.setLevel(LOG_LEVEL_DEBUG) 
        root_logger.addHandler(ui_handler)
        _handlers.append(ui_handler)
        logging.getLogger(__name__).info("[LOGGER SETUP] Handler da UI configurado para o logger raiz com level %s.", logging.getLevelName(LOG_LEVEL_DEBUG))

def setup_phase_logger(phase_name: str, logger_name: Optional[str] = None) -> str:
    """
    [DESATIVADO] Configura um arquivo de log dedicado para a fase indicada.
    Funcionalidade desativada conforme pedido do usuário para reduzir I/O de logs.
    """
    return ""



def get_logger(name: str, level: int = LOG_LEVEL_INFO) -> logging.Logger:
    """
    Obtém uma instância de logger configurada para um módulo específico.
    """
    with _lock:
        if name in _loggers:
            return _loggers[name]

        logger_instance = logging.getLogger(name)
        # Logger instance em DEBUG: a filtragem por nível acontece nos handlers.
        # Console handler=INFO (tela limpa), file handler=DEBUG (arquivo completo).
        # Se o logger ficasse em INFO, as mensagens debug seriam descartadas ANTES
        # de chegar ao file handler, independente do nível deste.
        logger_instance.setLevel(LOG_LEVEL_DEBUG)

        # Garante que as mensagens passem pelo root logger, cujo nível será DEBUG
        logger_instance.propagate = True

        # --- Configuração do handler de arquivo para o logger raiz (se ainda não configurado) ---
        root_logger = logging.getLogger()
        file_handler_present = any(isinstance(h, SafeTimedRotatingFileHandler) for h in root_logger.handlers) or \
                               any(isinstance(h, SafeTimedRotatingFileHandler) for h in _handlers)

        if not file_handler_present:
            # Passa LOG_LEVEL_DEBUG para o handler de arquivo
            file_h = _setup_file_handler("logs", "trading_bot.log", LOG_LEVEL_DEBUG)
            root_logger.addHandler(file_h)

        _loggers[name] = logger_instance
        return logger_instance

def _configure_root_logger():
    """
    Configura o logger raiz no inicio da aplicacao.
    """
    with _lock:
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            safe_stdout = _prepare_utf8_stream(sys.stdout)
            if safe_stdout is not sys.stdout:
                sys.stdout = safe_stdout
            safe_stderr = _prepare_utf8_stream(sys.stderr)
            if safe_stderr is not sys.stderr:
                sys.stderr = safe_stderr

            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            # Nivel do console pode ser INFO para evitar muita verbosidade na tela,
            # mas o arquivo tera DEBUG.
            console_handler.setLevel(LOG_LEVEL_INFO)
            root_logger.addHandler(console_handler)
            _handlers.append(console_handler)
            # O nivel do root logger deve permanecer em DEBUG para liberar todos os logs.
            root_logger.setLevel(LOG_LEVEL_DEBUG)

_configure_root_logger()
