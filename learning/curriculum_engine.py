# -----------------------------------------------------------------------------
# ARQUIVO: learning/curriculum_engine.py
# -----------------------------------------------------------------------------

"""
Motor de Aprendizado por Currículo (Curriculum Learning Engine).
"""

import pandas as pd
import numpy as np
import os
import joblib # Usado para salvar e carregar objetos Python
from typing import Dict, List, Any, Optional
from datetime import datetime # Para timestamps nos logs

from utils.logger import get_logger , LOG_LEVEL_DEBUG
from config.settings import AIConfig, TradingConfig # Importação da classe de configuração de IA

logger = get_logger("CurriculumEngine", LOG_LEVEL_DEBUG)

class CurriculumEngine:
    """
    Gerencia o progresso do treinamento da IA através de estágios de dificuldade definidos.
    """

    def __init__(self, config: AIConfig):
        """
        Inicializa o CurriculumEngine.

        Args:
            config (AIConfig): Objeto de configuração da IA contendo as definições dos estágios.
        """
        if not isinstance(config, AIConfig):
            raise TypeError("🚨 [ERRO CURRÍCULO] 'config' deve ser uma instância da classe AIConfig.")
        self.config = config
        
        # Carrega a definição dos estágios do currículo a partir da configuração
        if not isinstance(self.config.CURRICULUM_STAGES, list) or not self.config.CURRICULUM_STAGES:
            logger.critical("❌ [ERRO CURRÍCULO] A configuração CURRICULUM_STAGES está vazia ou não é uma lista. O currículo não funcionará.")
            self.stages = []
        else:
            self.stages = self.config.CURRICULUM_STAGES
        
        # Caminho onde o estado atual do currículo será salvo/carregado
        self.state_path = os.path.join(self.config.CHECKPOINT_DIR, "curriculum_state.joblib")
        
        # --- Estado interno do currículo ---
        self.current_stage_index: int = 0 
        self.stage_performance_history: Dict[int, List[Dict[str, float]]] = {i: [] for i in range(len(self.stages))}
        
        # Obter o timeframe primário do TradingConfig para filtrar corretamente
        self.primary_timeframe = TradingConfig.PRIMARY_TIMEFRAME_TRADING
        
        # Tenta carregar um estado anterior do currículo se existir
        self.load_state()
        logger.info(f"📚 [CURRÍCULO] CurriculumEngine inicializado. Total de estágios: {len(self.stages)}. Estágio atual: {self.get_current_stage_name()}.")


    def get_current_stage(self) -> Dict[str, Any]:
        """
        Retorna a configuração do estágio de treinamento atual.
        """
        if self.current_stage_index >= len(self.stages):
            logger.info("🎉 [CURRÍCULO] Currículo concluído! Operando no estágio final (repetindo o último).")
            return self.stages[-1] 
        return self.stages[self.current_stage_index]

    def get_current_stage_name(self) -> str:
        """Helper para retornar o nome do estágio atual ou indicar conclusão."""
        if self.current_stage_index >= len(self.stages):
            return "Currículo Concluído"
        return self.stages[self.current_stage_index].get('name', f"Estágio {self.current_stage_index + 1}")


    def prepare_training_data(self, featured_df: pd.DataFrame) -> pd.DataFrame:
        """
        [LÓGICA CORRIGIDA] Filtra o DataFrame de features. Se o resultado for muito pequeno,
        retorna o DataFrame original não filtrado para garantir que o treinamento sempre ocorra.
        """
        if not isinstance(featured_df, pd.DataFrame) or featured_df.empty:
            logger.error("❌ [ERRO CURRÍCULO] DataFrame de features de entrada está vazio.")
            return pd.DataFrame()

        stage = self.get_current_stage()
        logger.info(f"🧪 [CURRÍCULO] Preparando dados para o estágio: '{stage['name']}' ({self.current_stage_index + 1}/{len(self.stages)}).")

        # Inicia com uma cópia do DataFrame completo para usar como fallback
        original_df = featured_df.copy()
        filtered_df = featured_df.copy()

        # Define as colunas a serem usadas para filtragem
        volatility_col = f'atr_percentage_{self.primary_timeframe}'
        adx_col = f'adx_{self.primary_timeframe}'
        
        # Filtrar por Volatilidade
        if 'volatility_max' in stage and volatility_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[volatility_col] <= stage['volatility_max']]
        
        # Filtrar por Clareza da Tendência
        if 'trend_clarity_min' in stage and adx_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[adx_col] >= stage['trend_clarity_min']]

        if 'trend_clarity_max' in stage and adx_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[adx_col] <= stage['trend_clarity_max']]

        # --- INÍCIO DA CORREÇÃO LÓGICA ---
        # Define um limite mínimo razoável de amostras para um estágio
        min_samples_for_stage = 500
        
        if len(filtered_df) < min_samples_for_stage:
            logger.warning(f"⚠️ [ALERTE CURRÍCULO] A filtragem para o estágio '{stage['name']}' resultou em apenas {len(filtered_df)} amostras (mínimo: {min_samples_for_stage}).")
            logger.warning("Usando o conjunto de dados de treinamento COMPLETO (não filtrado) como fallback para este estágio, para garantir o aprendizado.")
            # Retorna o DataFrame original sem filtro como fallback
            return original_df
        # --- FIM DA CORREÇÃO LÓGICA ---
        
        logger.info(f"✅ [CURRÍCULO] Dados filtrados para o estágio '{stage['name']}': {len(filtered_df)} amostras prontas para treinamento.")
        return filtered_df
    
    def update_progress(self, episode_metrics: Dict[str, float]):
        """
        Atualiza o progresso do currículo com as métricas do último episódio de treinamento.
        """
        if not isinstance(episode_metrics, dict):
            logger.error(f"❌ [ERRO CURRÍCULO] 'episode_metrics' deve ser um dicionário. Recebido: {type(episode_metrics)}. Ignorando atualização de progresso.")
            return

        if self.is_completed():
            logger.info("ℹ️ [CURRÍCULO] Currículo já concluído. Nenhuma atualização de progresso necessária.")
            return

        current_stage_idx = self.current_stage_index
        if current_stage_idx not in self.stage_performance_history:
            self.stage_performance_history[current_stage_idx] = [] 

        self.stage_performance_history[current_stage_idx].append(episode_metrics)
        logger.debug(f"📈 [CURRÍCULO] Métricas do episódio registradas para o estágio {self.get_current_stage_name()}. Total de episódios neste estágio: {len(self.stage_performance_history[current_stage_idx])}.")

        stage = self.get_current_stage()
        history = self.stage_performance_history[current_stage_idx]

        # ── evaluation_frequency calibrado para ciclos de treino ────────────────
        # Antes: max(20, episodes/10) → p/ 50 000 episodes dava 5 000 chamadas
        # Agora: usa 'min_training_runs' do estágio (padrão 5), mínimo 3
        # Reflete que update_progress() é chamado 1x por especialista por treino
        # (não 1x por passo RL como era a intenção original do código).
        evaluation_frequency = max(3, int(stage.get('min_training_runs', 5)))

        if len(history) < evaluation_frequency:
            logger.debug(
                f"📚 [CURRÍCULO] Aguardando mínimo de {evaluation_frequency} ciclos "
                f"para avaliar '{stage['name']}'. Atual: {len(history)}."
            )
            return

        recent_eval_window = min(len(history), 50)
        recent_performance = pd.DataFrame(history[-recent_eval_window:])

        if recent_performance.empty:
            logger.warning(f"⚠️ [ALERTE CURRÍCULO] Nenhum dado de performance recente para avaliar o estágio {stage['name']}.")
            return

        avg_performance = recent_performance.mean(numeric_only=True).to_dict()
        logger.info(
            f"📊 [CURRÍCULO] Avaliação — estágio '{stage['name']}' "
            f"(últimos {recent_eval_window} ciclos): {avg_performance}"
        )

        # Verifica se os critérios de sucesso foram atingidos
        criteria = stage.get('success_criteria', {})
        all_criteria_met = True
        reasons_not_met = []

        for metric, threshold in criteria.items():
            current_metric_val = avg_performance.get(metric)

            if current_metric_val is None:
                all_criteria_met = False
                reasons_not_met.append(f"Métrica '{metric}' não encontrada nas métricas de performance.")
                continue

            if 'drawdown' in metric:
                if current_metric_val > threshold:
                    all_criteria_met = False
                    reasons_not_met.append(f"❌ {metric} ({current_metric_val:.2%}) > limite ({threshold:.2%})")
            else:
                if current_metric_val < threshold:
                    all_criteria_met = False
                    reasons_not_met.append(f"❌ {metric} ({current_metric_val:.2f}) < limite ({threshold:.2f})")

        if all_criteria_met:
            logger.info(f"🎉 [CURRÍCULO] Critérios de sucesso para o estágio '{stage['name']}' ATINGIDOS!")
            self._advance_stage()

        # [FIX PLATEAU] Plateau só faz sentido com histórico suficiente
        elif self.detect_plateau(patience=min(30, max(10, len(history) // 2))):
            logger.warning(
                f"🔄 [CURRÍCULO] Plateau detectado no estágio '{stage['name']}'. "
                f"Retrocedendo 1 estágio para reforçar aprendizado base."
            )
            if self.current_stage_index > 0:
                self.trigger_retraining(performance_degradation_level=0.3)

        else:
            # ── Válvula de segurança: avança forçado após max_training_runs ────
            # Garante que o currículo sempre progride mesmo se o proxy de Sharpe
            # não atingir o threshold (ex.: mercado bear, métricas ruidosas).
            max_runs = stage.get('max_training_runs', evaluation_frequency * 4)
            if len(history) >= max_runs:
                logger.warning(
                    f"⏫ [CURRÍCULO] Limite de {max_runs} ciclos de treino atingido "
                    f"no estágio '{stage['name']}' sem critérios atingidos. "
                    f"Avançando forçadamente para o próximo estágio."
                )
                self._advance_stage()
            else:
                logger.info(
                    f"⏳ [CURRÍCULO] Critérios de '{stage['name']}' ainda não atingidos "
                    f"({'; '.join(reasons_not_met)}). "
                    f"Ciclo {len(history)}/{max_runs} — continuando treinamento."
                )

    def _advance_stage(self):
        """
        Avança para o próximo estágio do currículo.
        """
        if self.current_stage_index < len(self.stages) - 1:
            self.current_stage_index += 1
            stage = self.get_current_stage()
            logger.info(f"🚀 [CURRÍCULO] Currículo AVANÇOU para o estágio {self.current_stage_index + 1}: '{stage['name']}'.")
            # Limpa o histórico de performance do novo estágio para começar do zero
            if self.current_stage_index not in self.stage_performance_history:
                self.stage_performance_history[self.current_stage_index] = []
            else:
                self.stage_performance_history[self.current_stage_index].clear() 
            self.save_state()
        else:
            logger.info("✅ [CURRÍCULO] Parabéns! Todos os estágios do currículo foram CONCLUÍDOS.")
            self.current_stage_index = len(self.stages) 
            self.save_state()

    def detect_plateau(self, patience: int = 30, min_delta: float = 0.01) -> bool:
        """
        Detecta se o agente está em plateau (sem melhoria significativa).

        Args:
            patience: Número de episódios sem melhoria para declarar plateau
            min_delta: Melhoria mínima para não ser considerado plateau

        Returns:
            True se plateau detectado, False caso contrário

        Referência: Bengio et al. 2012 - "Practical Recommendations for Gradient-Based Training"
        """
        if self.is_completed():
            return False

        history = self.stage_performance_history.get(self.current_stage_index, [])
        if len(history) < patience:
            return False

        # Pega os últimos 'patience' episódios
        recent = history[-patience:]
        recent_df = pd.DataFrame(recent)

        if recent_df.empty or 'sharpe_ratio' not in recent_df.columns:
            return False

        recent_sharpe = recent_df['sharpe_ratio'].dropna().values
        if len(recent_sharpe) < patience // 2:
            return False

        # Calcula melhoria linear via regressão simples
        x = np.arange(len(recent_sharpe), dtype=float)
        if len(x) < 2:
            return False

        slope = np.polyfit(x, recent_sharpe, 1)[0]
        is_plateau = abs(slope) < min_delta

        if is_plateau:
            logger.warning(
                f"⚠️ [CURRÍCULO] PLATEAU detectado no estágio '{self.get_current_stage_name()}'. "
                f"Slope de Sharpe nos últimos {patience} eps: {slope:.4f} (< {min_delta}). "
                f"Considere aumentar learning_rate ou retreinar do estágio anterior."
            )

        return is_plateau

    def trigger_retraining(self, performance_degradation_level: float):
        """
        Retrocede o currículo para um estágio anterior em caso de degradação da performance.
        """
        if not (0.0 <= performance_degradation_level <= 1.0):
            logger.error(f"❌ [ERRO CURRÍCULO] 'performance_degradation_level' deve estar entre 0.0 e 1.0. Recebido: {performance_degradation_level}. Ignorando.")
            return

        if self.current_stage_index == 0:
            logger.warning("⚠️ [CURRÍCULO] Já está no PRIMEIRO estágio. Não é possível retroceder mais para retreinamento adaptativo.")
            return

        stages_to_rewind = int(np.ceil(self.current_stage_index * performance_degradation_level))
        new_stage_index = max(0, self.current_stage_index - stages_to_rewind)
        
        if new_stage_index == self.current_stage_index:
            logger.info(f"ℹ️ [CURRÍCULO] Degradação {performance_degradation_level:.2%} não suficiente para retroceder estágio. Permanecendo no estágio atual.")
            return

        logger.warning(f"🚨 [CURRÍCULO] Degradação de performance detectada ({performance_degradation_level:.2%})! Retrocedendo de '{self.get_current_stage_name()}' "
                       f"para '{self.stages[new_stage_index]['name']}' (Estágio {new_stage_index + 1}) para retreinamento adaptativo.")
        
        self.current_stage_index = new_stage_index
        # Limpa o histórico de performance dos estágios a partir do novo ponto para forçar o reaprendizado.
        for i in range(new_stage_index, len(self.stages)):
            if i in self.stage_performance_history:
                self.stage_performance_history[i].clear() 
            else:
                self.stage_performance_history[i] = [] 
            
        self.save_state() 

    def is_completed(self) -> bool:
        """Verifica se todos os estágios do currículo foram concluídos."""
        return self.current_stage_index >= len(self.stages)

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna o estado atual do progresso do currículo para monitoramento.
        """
        if self.is_completed():
            return {
                "status": "Concluído",
                "current_stage_name": "N/A - Currículo Concluído",
                "current_stage_index": self.current_stage_index,
                "total_stages": len(self.stages),
                "completion_timestamp": datetime.utcnow().isoformat()
            }

        stage = self.get_current_stage()
        history = self.stage_performance_history[self.current_stage_index]
        
        avg_recent_performance = {}
        if history:
            recent_eval_window = min(len(history), 50)
            avg_recent_performance = pd.DataFrame(history[-recent_eval_window:]).mean(numeric_only=True).to_dict()

        return {
            "status": "Em Progresso",
            "current_stage_name": stage.get('name', f"Estágio {self.current_stage_index + 1}"),
            "current_stage_index": self.current_stage_index,
            "total_stages": len(self.stages),
            "episodes_in_current_stage": len(history),
            "required_episodes_for_stage": stage.get('episodes', 'N/A'),
            "success_criteria_for_stage": stage.get('success_criteria', {}),
            "recent_performance_avg": avg_recent_performance,
            "last_updated_utc": datetime.utcnow().isoformat()
        }

    def save_state(self):
        """Salva o estado atual do currículo em disco para persistência."""
        state = {
            'current_stage_index': self.current_stage_index,
            'stage_performance_history': self.stage_performance_history,
            'last_save_time_utc': datetime.utcnow().isoformat()
        }
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            joblib.dump(state, self.state_path)
            logger.info(f"✅ [CURRÍCULO] Estado do currículo salvo com sucesso em '{self.state_path}'.")
        except Exception as e:
            logger.error(f"❌ [ERRO CURRÍCULO] Falha ao salvar o estado do currículo em '{self.state_path}': {e}", exc_info=True)

    def load_state(self):
        """Carrega o estado do currículo do disco, se existir."""
        if os.path.exists(self.state_path):
            try:
                state = joblib.load(self.state_path)
                self.current_stage_index = state.get('current_stage_index', 0)
                loaded_history = state.get('stage_performance_history', {})
                self.stage_performance_history = {i: loaded_history.get(i, []) for i in range(len(self.stages))}

                logger.info(f"✅ [CURRÍCULO] Estado do currículo carregado de '{self.state_path}'. "
                            f"Iniciando no estágio {self.current_stage_index + 1} ('{self.get_current_stage_name()}').")
            except Exception as e:
                logger.error(f"❌ [ERRO CURRÍCULO] Falha ao carregar o estado do currículo de '{self.state_path}': {e}. Iniciando do zero.", exc_info=True)
                self.current_stage_index = 0
                self.stage_performance_history = {i: [] for i in range(len(self.stages))}
        else:
            logger.info("ℹ️ [CURRÍCULO] Nenhum estado de currículo salvo encontrado. Iniciando do primeiro estágio.")
            self.current_stage_index = 0
            self.stage_performance_history = {i: [] for i in range(len(self.stages))}