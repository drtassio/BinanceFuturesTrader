# No arquivo xai_engine/explainer.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

# --- NOVAS IMPORTAÇÕES CRÍTICAS ---
# import shap  # [LAZY LOAD] Movido para dentro da classe para economizar memória no import inicial


# --- Importações existentes ---
from models.trade_schema import Signal, Action
from utils.logger import get_logger

import logging
logger = get_logger("Explainer")

# Silencia logs internos matemáticos da biblioteca SHAP para limpar o terminal
logging.getLogger('shap').setLevel(logging.WARNING)

class Explainer:
    """
    [VERSÃO ESTADO DA ARTE]
    Fornece explicações para as decisões da IA usando SHAP (SHapley Additive exPlanations).
    """
    def __init__(self, ai_controller: 'AIController', background_data: pd.DataFrame, production_mode: bool = False):
        """
        Inicializa o Explainer com o modelo de IA e dados de referência para o SHAP.
        
        Args:
            ai_controller: A instância do AIController que contém a lógica de decisão.
            background_data: Um DataFrame de amostra (ex: 100-200 linhas dos dados de treino)
                             que o SHAP usa como referência para as explicações.
        """
        logger.info("🗣️ [EXPLAINER] Inicializando Explainer com motor SHAP...")
        self.ai_controller = ai_controller
        self.background_data = background_data
        self.production_mode = production_mode
        
        # [MELHORIA] Contador para explicações em produção
        self.explanation_counter = 0
        self.explanation_interval = 10  # Explica 1 a cada 10 decisões em produção
        
        # O KernelExplainer é agnóstico de modelo e pode usar qualquer função preditora.
        # Ele funciona criando perturbações nos dados de entrada para ver como a saída muda.
        
        # [FIX] Guarda as colunas esperadas para garantir alinhamento dimensional
        self.expected_features = list(self.background_data.columns)
        
        # Amostra os dados de background para o SHAP (máximo 100 pontos para performance)
        background_sample = self.background_data.sample(
            n=min(100, len(self.background_data)), 
            random_state=42
        ).values
        
        if not production_mode:
            # Modo completo para backtests
            try:
                import shap
                self.shap_explainer = shap.KernelExplainer(
                    self.ai_controller.predict_for_shap,
                    background_sample
                )
                logger.info("✅ [EXPLAINER] Motor SHAP completo inicializado para backtests.")
            except ImportError:
                logger.warning("⚠️ [EXPLAINER] SHAP não instalado ou falha ao importar. Explicações detalhadas desativadas.")
                self.shap_explainer = None
            except Exception as e:
                logger.warning(f"⚠️ [EXPLAINER] Erro ao inicializar SHAP: {e}")
                self.shap_explainer = None
        else:
            # Modo rápido para produção
            self.shap_explainer = None
            logger.info("✅ [EXPLAINER] Modo rápido inicializado para produção (explicações amostrais).")

    def _get_feature_importance_from_shap(self, market_features_row: pd.Series, top_n: int = 5, source: str = "Unknown") -> List[Dict[str, Any]]:
        """Usa o SHAP para calcular a contribuição de cada feature."""
        try:
            # [FIX] Garante que market_features_row seja uma Series para processamento uniforme
            if isinstance(market_features_row, dict):
                market_features_row = pd.Series(market_features_row)

            # [FIX] Filtra a linha para conter APENAS as features esperadas pelo SHAP
            if hasattr(self, 'expected_features') and self.expected_features:
                # --- Alinhamento Robusto de Features ---
                # Garante que os valores sejam numéricos e limpa o índice
                market_features_row.index = [str(i).strip().lower().replace(' ', '_') for i in market_features_row.index]
                market_features_row = pd.to_numeric(market_features_row, errors='coerce').fillna(0.0)
                
                expected_map = {str(f).strip().lower().replace(' ', '_'): f for f in self.expected_features}
                current_series = market_features_row
                
                aligned_values = []
                for expected_norm in expected_map.keys():
                    found_val = None
                    
                    # 1. Busca Direta
                    if expected_norm in current_series.index:
                        found_val = current_series[expected_norm]
                    
                    # 2. Busca por Prefixo/Sufixo se falhou ou veio ZERO (pode ser zero real, mas checamos)
                    if found_val is None or (found_val == 0.0 and 'hidden' in expected_norm):
                        for idx_name in current_series.index:
                            if expected_norm in idx_name or idx_name in expected_norm:
                                found_val = current_series[idx_name]
                                if found_val != 0.0: break # Achamos um valor real
                    
                    aligned_values.append(found_val if found_val is not None else 0.0)
                
                market_features_row = pd.Series(aligned_values, index=self.expected_features)
                
                # --- VALIDAÇÃO PÓS-ALINHAMENTO ---
                final_non_zero = (market_features_row != 0.0).sum()
                logger.debug(f"🔍 [XAI ALIGN] Alinhamento concluído. {final_non_zero} features com valor real (incl. hidden).")
            
            # Calcula os SHAP values para a observação atual
            shap_values = self.shap_explainer.shap_values(market_features_row.values, nsamples=100)
            
            # SHAP pode retornar uma lista ou array, normalizamos para array
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Pega o primeiro elemento se for lista
            
            contrib_df = pd.DataFrame({
                'feature': market_features_row.index,
                'value': market_features_row.values,
                'shap_value': shap_values
            })
            
            # Ordena pela contribuição absoluta para encontrar as mais importantes
            contrib_df['abs_shap'] = contrib_df['shap_value'].abs()
            top_contributors = contrib_df.sort_values(by='abs_shap', ascending=False).head(top_n)

            contributors_list = []
            for _, row in top_contributors.iterrows():
                effect = "positiva (apoiando COMPRA ou contra VENDA)" if row['shap_value'] > 0 else "negativa (apoiando VENDA ou contra COMPRA)"
                reason = f"Contribuiu com {row['shap_value']:.4f} para o score final, indicando uma influência {effect}."
                
                # [PRECISION FIX] Aumenta precisão para Hidden Features (valores latentes do Autoencoder)
                precision = 10 if "hidden" in str(row['feature']).lower() else 6
                val_str = f"{row['value']:.{precision}f}"
                
                contributors_list.append({
                    "feature": row['feature'],
                    "value": val_str,
                    "impact_score": f"{row['shap_value']:.4f}", # O SHAP value é o nosso novo 'impact_score'
                    "reason": reason
                })
            return contributors_list
        except Exception as e:
            logger.error(f"❌ [EXPLAINER] Erro ao calcular SHAP values: {e}", exc_info=True)
            return [{"feature": "Erro na análise SHAP", "value": "N/A", "reason": str(e)}]

    def _get_feature_importance_fast(self, market_features_row: pd.Series, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        [MELHORIA] Método rápido para análise de features em produção.
        
        Usa heurísticas simples em vez de SHAP para economizar tempo de processamento.
        """
        try:
            # Identifica features com valores extremos ou mudanças significativas
            features_analysis = []
            
            for feature, value in market_features_row.items():
                # Calcula score baseado em magnitude e tipo de feature
                abs_value = abs(value)
                
                # Features de tendência (maior peso)
                if any(trend_word in feature.lower() for trend_word in ['trend', 'ema', 'sma', 'macd']):
                    score = abs_value * 2.0
                # Features de volatilidade
                elif any(vol_word in feature.lower() for vol_word in ['atr', 'volatility', 'bb']):
                    score = abs_value * 1.5
                # Features de volume
                elif any(vol_word in feature.lower() for vol_word in ['volume', 'obv']):
                    score = abs_value * 1.2
                # Outras features
                else:
                    score = abs_value
                
                features_analysis.append({
                    'feature': feature,
                    'value': value,
                    'score': score
                })
            
            # Ordena por score e pega as top_n
            features_analysis.sort(key=lambda x: x['score'], reverse=True)
            top_features = features_analysis[:top_n]
            
            contributors_list = []
            for item in top_features:
                # [PRECISION FIX]
                precision = 10 if "hidden" in str(item['feature']).lower() else 6
                val_raw = item['value']
                val_str = f"{val_raw:.{precision}f}"
                
                contributors_list.append({
                    "feature": item['feature'],
                    "value": val_str,
                    "impact_score": f"{item['score']:.4f}",
                    "reason": f"Feature com valor {val_str} e score de impacto {item['score']:.4f}."
                })
            
            return contributors_list
            
        except Exception as e:
            logger.error(f"❌ [EXPLAINER] Erro no método rápido: {e}")
            return [{"feature": "Erro na análise rápida", "value": "N/A", "reason": str(e)}]

    def explain_decision(self, signal: Signal, market_features_row: pd.Series, top_n: int = 5) -> Dict[str, Any]:
        """
        Gera uma explicação detalhada para uma decisão de trading usando SHAP.
        """
        if not isinstance(signal, Signal):
            return {"narrative": "Erro: Sinal inválido."}
        
        # [MELHORIA] Modo rápido para produção
        if self.production_mode:
            self.explanation_counter += 1
            if self.explanation_counter % self.explanation_interval != 0:
                # Retorna explicação simplificada para a maioria das decisões
                return {
                    "narrative": f"Decisão {signal.action.value} com confiança {signal.confidence:.2%}. "
                                 f"Explicação detalhada disponível a cada {self.explanation_interval} decisões.",
                    "mode": "production_fast",
                    "explanation_counter": self.explanation_counter
                }
        
        logger.info(f"🗣️ [EXPLAINER] Gerando explicação SHAP para a decisão: {signal.action.value}...")
        
        # [FIX] Extrai a origem para o diagnóstico de Raio-X
        source = signal.explanation.get('specialist', 'Ensemble')
        
        # A obtenção da importância das features agora usa SHAP
        if self.shap_explainer:
            contributors = self._get_feature_importance_from_shap(market_features_row, top_n=top_n, source=source)
        else:
            # Fallback para modo rápido
            contributors = self._get_feature_importance_fast(market_features_row, top_n=top_n)
        
        narrative = self._generate_narrative(signal, contributors)
        
        explanation = {
            "decision": signal.action.value,
            "confidence": f"{signal.confidence:.2%}",
            "specialist_in_charge": signal.explanation.get("specialist", "N/A"),
            "main_contributors": contributors,
            "narrative": narrative,
            "timestamp_utc": datetime.utcnow().isoformat()
        }
        return explanation

    def _generate_narrative(self, signal: Signal, contributors: List[Dict]) -> str:
        """Gera uma narrativa textual explicativa e organizada da decisão."""
        action_desc = signal.action.value.upper()
        specialist = signal.explanation.get("specialist", "Motor de IA")
        # Se houve veto de risco, o motivo está no 'reason'
        reason = signal.explanation.get("reason", "")
        original_action = signal.explanation.get("original_action", None)
        
        # --- 1. Cabeçalho da Decisão ---
        if signal.action == Action.HOLD:
            if original_action:
                # Caso de Veto de Risco
                narrative = f"🛑 TRADE VETADO PELO RISCO ({original_action.value.upper()} -> HOLD)\n"
                narrative += f"📌 Motivo: {reason if reason else 'Filtro de segurança ativado.'}"
            else:
                # Caso de HOLD natural do modelo
                narrative = f"⚖️ DECISÃO: MANTER AGUARDANDO (HOLD)\n"
                narrative += f"📌 Motivo: {reason if reason else 'Incerteza do mercado ou falta de sinal claro.'}"
            
            narrative += f"\n🎯 Confiança do Sinal: {signal.confidence:.2%}"
        else:
            # Caso de Execução
            narrative = f"🚀 DECISÃO: EXECUTAR {action_desc}\n"
            narrative += f"👤 Especialista: {specialist}\n"
            narrative += f"🎯 Confiança: {signal.confidence:.2%}"
            if reason:
                narrative += f"\n📌 Contexto: {reason}"

        # --- 2. Painel de Votação do Ensemble (MoE) ---
        ensemble_details = signal.explanation.get("ensemble_details")
        if ensemble_details:
            narrative += "\n\n🗳️ VOTAÇÃO DO ENSEMBLE (MoE):"
            narrative += "\n" + "─" * 60
            narrative += f"\n{'AGENTE':<12} | {'PESO':<6} | {'VOTO':<6} | {'CONF.':<6}"
            narrative += "\n" + "─" * 60
            
            for name, details in ensemble_details.items():
                icon = "🐂" if "bull" in name.lower() else "🐻" if "bear" in name.lower() else "🤠"
                weight_pct = f"{details.get('weight', 0)*100:>5.1f}%"
                action = details.get('action', 'HOLD').upper()
                conf = f"{details.get('confidence', 0)*100:>5.1f}%"
                
                # Formatação visual do voto
                action_color = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
                narrative += f"\n{icon} {name.capitalize():<9} | {weight_pct} | {action_color} {action:<4} | {conf}"
            
            # Adiciona o Score Ponderado Final
            weighted_score = signal.explanation.get("weighted_score", 0.0)
            score_bar = "┠" + ("█" * int(abs(weighted_score)*10)) + ("░" * (10 - int(abs(weighted_score)*10))) + "┨"
            narrative += "\n" + "─" * 60
            narrative += f"\n📊 Score Final Ponderado: {weighted_score:+.4f} {score_bar}"
            narrative += "\n" + "─" * 60

        # --- 2. Análise de Fatores (SHAP) ---
        if contributors:
            narrative += "\n\n📊 FATORES QUE MAIS INFLUENCIARAM A IA:"
            narrative += "\n" + "─" * 60
            for c in contributors:
                impact_val = float(c['impact_score'])
                icon = "📈" if impact_val > 0 else "📉"
                feat_name = c['feature'].replace('_', ' ').title()
                # Formatação tipo tabela com largura ajustada para precisão (14 chars para Val)
                narrative += f"\n{icon} {feat_name:<25} | Val: {str(c['value']):>14} | Imp: {impact_val:+.4f}"
            narrative += "\n" + "─" * 60
        
        return narrative