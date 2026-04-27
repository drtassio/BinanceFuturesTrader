# -----------------------------------------------------------------------------
# ARQUIVO: governance/ai_monitor.py
# -----------------------------------------------------------------------------

"""
Módulo de Monitoramento da IA (AI Monitor).

Responsável por registrar, auditar e fornecer insights sobre o comportamento
do sistema de IA em tempo real. Atua como a "caixa-preta" do avião,
registrando decisões, contextos e eventos críticos para análise post-mortem,
depuração e garantia de conformidade.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import threading # Para garantir thread-safety no acesso a session_events, se necessário

from utils.logger import get_logger # Importação do logger centralizado

# Logger centralizado para registrar as operações do próprio monitor
monitor_logger = get_logger("AIMonitor")

class AIMonitor:
    """
    Monitora e registra eventos do sistema de IA do bot de trading.
    Gera logs explicativos e indicadores de estabilidade para governança e depuração.
    Os eventos são salvos em arquivos JSONL e mantidos em memória para acesso rápido.
    """

    def __init__(self, log_dir: str = "logs/ai_events", store_json: bool = True):
        """
        Inicializa o AIMonitor.

        Args:
            log_dir (str): Diretório base para salvar os logs de eventos da IA.
                           Será criado dentro da raiz do projeto.
            store_json (bool): Se True, salva cada evento em um arquivo JSON Lines (.jsonl).
                               Se False, os eventos são apenas logados via logger principal.
        """
        self.log_dir = log_dir # Caminho base
        # Garante que log_dir seja um caminho absoluto ou relativo ao diretório de trabalho atual
        if not os.path.isabs(self.log_dir):
            self.log_dir = os.path.join(os.getcwd(), self.log_dir)

        self.store_json = store_json
        self.session_events: List[Dict[str, Any]] = [] # Lista de eventos na sessão atual (em memória)
        self._lock = threading.Lock() # Lock para proteger self.session_events e operações de arquivo

        try:
            os.makedirs(self.log_dir, exist_ok=True)
            monitor_logger.info(f"👁️‍🗨️ [MONITOR] AIMonitor inicializado. Armazenamento de eventos em JSONL: {self.store_json}. Diretório: '{self.log_dir}'.")
        except OSError as e:
            self.store_json = False # Desativa o salvamento em arquivo se o diretório não puder ser criado
            monitor_logger.error(f"❌ [ERRO MONITOR] Não foi possível criar o diretório de logs '{self.log_dir}': {e}. O armazenamento em JSONL foi desativado.", exc_info=True)
        except Exception as e:
            self.store_json = False
            monitor_logger.critical(f"🚨 [CRÍTICO MONITOR] Erro inesperado ao inicializar AIMonitor: {e}. Desativando armazenamento em JSONL.", exc_info=True)


    def log_event(self, event_type: str, message: str, level: str = "INFO", context: Optional[Dict[str, Any]] = None):
        """
        Registra um evento relevante no comportamento da IA.
        O evento é adicionado à lista de eventos da sessão e, opcionalmente, salvo em arquivo.

        Args:
            event_type (str): Tipo do evento (e.g., 'DECISION', 'REGIME_SHIFT', 'RISK_ALERT', 'TRAINING_START').
                              Deve ser uma string curta e descritiva.
            message (str): Uma mensagem descritiva sobre o evento.
            level (str): Nível de severidade do evento ('INFO', 'WARNING', 'ERROR', 'CRITICAL').
                         Será usado para o log centralizado e para o campo 'level' do JSON.
            context (Optional[Dict[str, Any]]): Dicionário com dados contextuais relevantes
                                                 (e.g., features, confiança do sinal, id da ordem, métricas).
        """
        # Validação de entrada
        if not isinstance(event_type, str) or not event_type:
            monitor_logger.error(f"❌ [ERRO MONITOR] 'event_type' inválido: '{event_type}'. Deve ser uma string não vazia. Ignorando evento.")
            return
        if not isinstance(message, str) or not message:
            monitor_logger.error(f"❌ [ERRO MONITOR] 'message' inválida: '{message}'. Deve ser uma string não vazia. Ignorando evento.")
            return
        if not isinstance(level, str) or level.upper() not in ["INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"]:
            monitor_logger.warning(f"⚠️ [MONITOR] Nível de log '{level}' inválido. Usando 'INFO' para o evento '{event_type}'.")
            level = "INFO"
        if context is not None and not isinstance(context, dict):
            monitor_logger.warning(f"⚠️ [MONITOR] 'context' deve ser um dicionário ou None. Recebido: {type(context)}. Usando dicionário vazio.")
            context = {}

        event = {
            "timestamp_utc": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(), # Timestamp em UTC ISO format
            "event_type": event_type.upper(), # Garante maiúsculas para consistência
            "level": level.upper(),
            "message": message,
            "context": context if context is not None else {} # Garante que context é um dict
        }

        with self._lock: # Protege o acesso à lista de eventos em memória
            self.session_events.append(event)
            # Limita o tamanho da lista em memória para evitar consumo excessivo de RAM
            # (Adeque o maxlen se precisar de mais histórico em memória para get_recent_events)
            if len(self.session_events) > 1000: # Exemplo: manter os últimos 1000 eventos em memória
                self.session_events.pop(0)

        # Loga o evento no logger principal da aplicação para visibilidade centralizada (console/arquivo de log geral)
        # Usa `default=str` para serializar objetos não JSON-serializáveis (e.g., Enums, datetime) no contexto
        log_message_for_main_logger = f"[{event['event_type']}] {event['message']} | Contexto: {json.dumps(event['context'], default=str, ensure_ascii=False)}"
        
        # Obtém o método de log apropriado com base no nível
        log_method = getattr(monitor_logger, level.lower(), monitor_logger.info)
        log_method(log_message_for_main_logger)

        # Salva o evento em um arquivo JSON Lines dedicado, se habilitado
        if self.store_json:
            self._save_to_file(event)


    def _save_to_file(self, event: Dict[str, Any]):
        """
        Salva um único evento como uma nova linha em um arquivo de log JSON Lines (.jsonl).
        Os arquivos são rotacionados diariamente com base na data UTC.
        """
        if not self.store_json:
            return # Não faz nada se o armazenamento em JSONL estiver desativado

        try:
            # Cria um arquivo de log por dia para facilitar o gerenciamento e a análise
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            # Usa '.jsonl' (JSON Lines) para indicar que cada linha é um objeto JSON válido
            path = os.path.join(self.log_dir, f"ai_events_{date_str}.jsonl")
            
            with self._lock: # Protege a operação de escrita no arquivo
                # Usa 'a' para modo append. Cada evento é uma linha JSON, sem vírgulas entre eles.
                # `ensure_ascii=False` para preservar caracteres UTF-8.
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
        except (IOError, OSError) as e:
            monitor_logger.error(f"❌ [ERRO MONITOR] Falha de I/O ao salvar evento da IA no arquivo '{path}': {e}. Verifique permissões/espaço em disco.", exc_info=True)
            # Em caso de falha persistente, pode-se desativar self.store_json aqui para evitar futuros erros.
        except Exception as e:
            monitor_logger.critical(f"🚨 [CRÍTICO MONITOR] Erro inesperado ao salvar log de evento da IA: {e}. Evento: {event}", exc_info=True)


    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Retorna os N eventos mais recentes registrados na sessão atual (em memória).

        Args:
            count (int): O número de eventos recentes a serem retornados.

        Returns:
            List[Dict[str, Any]]: Uma lista dos eventos mais recentes, do mais antigo para o mais novo.
        """
        if not isinstance(count, int) or count < 0:
            monitor_logger.error(f"❌ [ERRO MONITOR] 'count' deve ser um inteiro não negativo. Recebido: {count}. Retornando lista vazia.")
            return []

        with self._lock: # Protege o acesso à lista em memória
            return list(self.session_events[-count:]) # Retorna uma cópia para evitar modificações externas

    def get_summary(self) -> Dict[str, Any]:
        """
        Fornece um resumo da atividade da IA na sessão atual.
        Conta o total de eventos, por nível de severidade e por tipo de evento.

        Returns:
            Dict[str, Any]: Um dicionário com estatísticas agregadas da sessão.
        """
        summary = {
            "total_events_in_session": 0,
            "info_count": 0,
            "warning_count": 0,
            "error_count": 0,
            "critical_count": 0,
            "debug_count": 0,
            "event_types_count": {} # Contagem por tipo de evento (e.g., 'DECISION': 10)
        }
        
        with self._lock: # Protege o acesso à lista em memória para o resumo
            summary["total_events_in_session"] = len(self.session_events)
            for event in self.session_events:
                level_key = event.get("level", "INFO").lower() + "_count"
                if level_key in summary:
                    summary[level_key] += 1
                
                event_type = event.get("event_type", "UNKNOWN").upper()
                summary["event_types_count"][event_type] = summary["event_types_count"].get(event_type, 0) + 1
        
        summary["summary_generated_utc"] = datetime.utcnow().isoformat()
        return summary

    def reset_session(self):
        """
        Limpa todos os eventos da sessão atual da memória (`self.session_events`).
        Isso não afeta os arquivos de log JSONL já salvos em disco.
        Útil para iniciar uma nova "sessão" de monitoramento sem os eventos antigos em RAM.
        """
        with self._lock: # Protege a operação de limpeza
            self.session_events.clear()
        monitor_logger.info("🗑️ [MONITOR] Sessão do AIMonitor foi resetada. Eventos em memória limpos.")