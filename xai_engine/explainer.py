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

logger = get_logger("Explainer")

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

    def _get_feature_importance_from_shap(self, market_features_row: pd.Series, top_n: int = 5) -> List[Dict[str, Any]]:
        """Usa o SHAP para calcular a contribuição de cada feature."""
        try:
            # [FIX] Filtra a linha para conter APENAS as features esperadas pelo SHAP
            # Evita IndexError se market_features_row tiver mais colunas que background_data
            if hasattr(self, 'expected_features') and self.expected_features:
                 # Robust intersection using pandas index method to match valid columns
                 valid_features = market_features_row.index.intersection(self.expected_features)
                 
                 # Select valid columns and then reindex to fill missing expected columns with 0
                 # This guarantees the output series has exactly self.expected_features in correct order
                 market_features_row = market_features_row[valid_features].reindex(self.expected_features, fill_value=0)
            
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
                contributors_list.append({
                    "feature": row['feature'],
                    "value": f"{row['value']:.4f}",
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
                effect = "positiva" if item['value'] > 0 else "negativa"
                contributors_list.append({
                    "feature": item['feature'],
                    "value": f"{item['value']:.4f}",
                    "impact_score": f"{item['score']:.4f}",
                    "reason": f"Feature com valor {effect} ({item['value']:.4f}) e score de impacto {item['score']:.4f}."
                })
            
            return contributors_list
            
        except Exception as e:
            logger.error(f"❌ [EXPLAINER] Erro no método rápido: {e}")
            return [{"feature": "Erro na análise rápida", "value": "N/A", "reason": str(e)}]

    def explain_decision(self, signal: Signal, market_features_row: pd.Series) -> Dict[str, Any]:
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
        
        # A obtenção da importância das features agora usa SHAP
        if self.shap_explainer:
            contributors = self._get_feature_importance_from_shap(market_features_row)
        else:
            # Fallback para modo rápido
            contributors = self._get_feature_importance_fast(market_features_row)
        
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
        """Gera uma narrativa textual explicativa da decisão."""
        action_desc = signal.action.value.upper()
        specialist = signal.explanation.get("specialist", "Motor de IA")
        reason = signal.explanation.get("reason", "N/A")
        
        if signal.action == Action.HOLD:
            narrative = f"O sistema decidiu aguardar ({action_desc}). Motivo principal: {reason if reason else 'Incerteza do mercado ou falta de sinal claro'}."
            narrative += f"\nConfiança atual: {signal.confidence:.2%}."
        else:
            narrative = f"O especialista '{specialist}' recomendou uma ação de {action_desc} com confiança de {signal.confidence:.2%}."

        if contributors:
            narrative += "\n\nA análise SHAP revela que a decisão foi influenciada principalmente por:"
            for c in contributors:
                narrative += f"\n• {c['feature']} (valor: {c['value']}): {c['reason']}"
        
        return narrative