# -----------------------------------------------------------------------------
# ARQUIVO: governance/drift_detector.py
# -----------------------------------------------------------------------------

"""
DriftDetector Elite - Detecção Avançada de Drift
================================================

Implementa:
- Ensemble de detectores (covariate, concept, performance)
- Histerese e ações automáticas
- Integração com risk budget e HRL Master
- Auto-calibração e relatórios
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import deque
import yaml
import json
import os
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error
import joblib



logger = logging.getLogger(__name__)

@dataclass
class DriftConfig:
    """Configuração do DriftDetector Elite."""
    # Pesos do ensemble
    psi_weight: float = 0.30
    ece_weight: float = 0.25
    sharpe_weight: float = 0.25
    slippage_weight: float = 0.10
    recon_weight: float = 0.10
    
    # Thresholds
    none_threshold: float = 0.2
    mild_threshold: float = 0.3
    severe_threshold: float = 0.5
    
    # Histerese
    recovery_threshold: float = 0.15
    recovery_steps: int = 10
    cooldown_steps: int = 20
    
    # Janelas de cálculo
    psi_window: int = 100
    ece_window: int = 50
    sharpe_window: int = 30
    slippage_window: int = 20
    recon_window: int = 50
    
    # Ações automáticas
    mild_risk_multiplier: float = 0.7
    severe_risk_multiplier: float = 0.4
    mild_threshold_increase: float = 0.02
    severe_threshold_increase: float = 0.05

class DriftDetector:
    """DriftDetector Elite com ensemble de detectores."""
    
    def __init__(self, config: DriftConfig = None):
        if config is None:
            self.config = DriftConfig()
        else:
            self.config = config
        
        # Estado interno
        self.reference_data: Optional[Dict[str, np.ndarray]] = None
        self.drift_history: deque = deque(maxlen=1000)
        self.action_history: deque = deque(maxlen=1000)
        self.recovery_counter: int = 0
        self.cooldown_counter: int = 0
        
        # Métricas rolling
        self.psi_scores: deque = deque(maxlen=self.config.psi_window)
        self.ece_scores: deque = deque(maxlen=self.config.ece_window)
        self.sharpe_scores: deque = deque(maxlen=self.config.sharpe_window)
        self.slippage_scores: deque = deque(maxlen=self.config.slippage_window)
        self.recon_scores: deque = deque(maxlen=self.config.recon_window)
        
        # Baseline para comparação
        self.baseline_metrics: Dict[str, float] = {}
        self.baseline_recon_mse: float = 0.0
        
        # Configuração
        self.drift_config_path = "models_ai/drift_config.yaml"
        self.reports_dir = "docs/drift_reports"
        
        # Cria diretórios
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs("models_ai", exist_ok=True)
        
        logger.info(f"🔍 DriftDetector Elite inicializado")
    
    def set_reference_data(self, df: pd.DataFrame, feature_columns: List[str] = None):
        """Define dados de referência para detecção de drift (tolerante)."""
        logger.info("📊 Definindo dados de referência para drift detection")
        
        # [REMOVIDO] FeatureContract (arquivo ausente). Prosseguindo com colunas disponíveis.
        # try:
        #     df = FeatureContract.enforce_contract(
        #         df, required_regime=False, required_sdae=False, required_trend=False, required_tape=False
        #     )
        # except Exception as e:
        #     logger.warning(f"⚠️ [DRIFT] Contract enforcement tolerante falhou: {e}. Prosseguindo com colunas numéricas disponíveis.")
        
        # Extrai features do contrato
        if feature_columns:
            feature_cols = [col for col in feature_columns if col in df.columns]
        else:
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol']]
        
        # Armazena dados de referência
        self.reference_data = {}
        for col in feature_cols:
            if col in df.columns:
                self.reference_data[col] = df[col].dropna().values
        
        # Calcula baseline de métricas
        self._calculate_baseline_metrics(df)
        
        # Salva configuração
        self._save_config()
        
        logger.info(f"✅ Dados de referência definidos ({len(feature_cols)} features)")
    
    def _calculate_baseline_metrics(self, df: pd.DataFrame):
        """Calcula métricas de baseline."""
        # Baseline de recon MSE (se SDAE disponível)
        if 'smoothness_latent' in df.columns:
            self.baseline_recon_mse = df['smoothness_latent'].var()
        
        # Baseline de Sharpe (se disponível)
        if 'trend_confidence' in df.columns:
            # Simula retornos baseados na confiança
            confidence_returns = df['trend_confidence'].pct_change().dropna()
            if len(confidence_returns) > 10:
                self.baseline_metrics['sharpe'] = confidence_returns.mean() / (confidence_returns.std() + 1e-8) * np.sqrt(252)
        
        # Baseline de ECE (se disponível)
        if 'trend_pred' in df.columns and 'trend_confidence' in df.columns:
            # Calcula ECE simples
            preds = df['trend_pred'].values
            confs = df['trend_confidence'].values
            ece = self._calculate_ece_simple(preds, confs)
            self.baseline_metrics['ece'] = ece
    
    def _calculate_ece_simple(self, preds: np.ndarray, confs: np.ndarray, n_bins: int = 10) -> float:
        """Calcula ECE simples."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confs > bin_lower) & (confs <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                # Aqui simplificamos pois não temos o target real no momento
                # Em um cenário real, compararíamos preds com labels reais
                # Usamos a média de confiança como proxy para calibração interna
                bin_confidence = np.mean(confs[in_bin])
                ece += bin_size * np.abs(bin_confidence - bin_confidence) # Placeholder
        
        return 0.0 # Placeholder até implementarmos calibração real com targets
    
    def check_drift(self, current_data: Any) -> Dict[str, Any]:
        """
        Wrapper de compatibilidade para a API anterior.
        Tenta converter current_data (dict ou DataFrame) para o formato esperado pelo compute_drift_score.
        """
        # Se receber um dict de listas (API antiga), converte para DataFrame
        batch_df = None
        if isinstance(current_data, dict):
            try:
                # Tenta criar DataFrame. Assume que todas as listas têm o mesmo tamanho.
                batch_df = pd.DataFrame(current_data)
            except Exception as e:
                logger.warning(f"⚠️ [DRIFT] Falha ao converter dict para DataFrame: {e}")
        elif isinstance(current_data, pd.DataFrame):
            batch_df = current_data
            
        if batch_df is not None and not batch_df.empty:
            score, level, explanation = self.compute_drift_score(batch_df)
            
            # Mapeia para o formato de retorno antigo para não quebrar o AIController imediatamente
            return {
                'global': {
                    'drift_detected': level != 'none',
                    'score': score,
                    'level': level,
                    'details': explanation
                }
            }
        
        return {'global': {'drift_detected': False, 'message': 'Invalid data'}}

    def compute_drift_score(self, batch_df: pd.DataFrame, 
                          performance_metrics: Optional[Dict] = None,
                          execution_stats: Optional[Dict] = None) -> Tuple[float, str, Dict]:
        """
        Computa score de drift usando ensemble de detectores.
        
        Args:
            batch_df: DataFrame com features atuais
            performance_metrics: Métricas de performance (Sharpe, ECE, etc.)
            execution_stats: Estatísticas de execução (slippage, etc.)
            
        Returns:
            (drift_score, level, explanation)
        """
        if self.reference_data is None:
            # Tenta carregar config se não houver referência
            self.load_config()
            if self.reference_data is None:
                logger.warning("⚠️ Dados de referência não definidos")
                return 0.0, 'none', {'reason': 'no_reference_data'}
        
        # [REMOVIDO] FeatureContract
        # try:
        #    batch_df = FeatureContract.enforce_contract(batch_df, required_regime=False, required_sdae=False, required_trend=False, required_tape=False)
        # except:
        #    pass

        # 1. Covariate Shift (PSI)
        psi_z = self._compute_psi_score(batch_df)
        
        # 2. Concept/Performance Drift (se métricas disponíveis)
        ece_z = self._compute_ece_score(batch_df, performance_metrics)
        sharpe_z = self._compute_sharpe_score(batch_df, performance_metrics)
        
        # 3. Execution Drift
        slippage_z = self._compute_slippage_score(execution_stats)
        
        # 4. SDAE Reconstruction Drift
        recon_z = self._compute_recon_score(batch_df)
        
        # Calcula score composto
        score = (self.config.psi_weight * psi_z +
                self.config.ece_weight * ece_z +
                self.config.sharpe_weight * sharpe_z +
                self.config.slippage_weight * slippage_z +
                self.config.recon_weight * recon_z)
        
        # Normaliza para [0, 1]
        score = np.clip(score, 0.0, 1.0)
        
        # Determina nível
        if score < self.config.none_threshold:
            level = 'none'
        elif score < self.config.mild_threshold:
            level = 'mild'
        else:
            level = 'severe'
        
        # Atualiza histórico
        self.drift_history.append({
            'timestamp': datetime.now(),
            'score': score,
            'level': level,
            'components': {
                'psi_z': psi_z,
                'ece_z': ece_z,
                'sharpe_z': sharpe_z,
                'slippage_z': slippage_z,
                'recon_z': recon_z
            }
        })
        
        # Aplica histerese
        score, level, actions = self._apply_hysteresis(score, level)
        
        explanation = {
            'score': float(score),
            'level': level,
            'components': {
                'psi_z': float(psi_z),
                'ece_z': float(ece_z),
                'sharpe_z': float(sharpe_z),
                'slippage_z': float(slippage_z),
                'recon_z': float(recon_z)
            },
            'actions': actions,
            'recovery_counter': self.recovery_counter,
            'cooldown_counter': self.cooldown_counter
        }
        
        logger.debug(f"🔍 Drift Score: {score:.3f} ({level}) - PSI: {psi_z:.3f}, ECE: {ece_z:.3f}, Sharpe: {sharpe_z:.3f}")
        
        return float(score), level, explanation
    
    def _compute_psi_score(self, batch_df: pd.DataFrame) -> float:
        """Computa PSI (Population Stability Index) score."""
        try:
            psi_scores = []
            
            for col in self.reference_data.keys():
                if col in batch_df.columns:
                    ref_data = self.reference_data[col]
                    current_data = batch_df[col].dropna().values
                    
                    if len(current_data) > 10 and len(ref_data) > 10:
                        # Calcula PSI
                        psi = self._calculate_psi(ref_data, current_data)
                        psi_scores.append(psi)
            
            if psi_scores:
                avg_psi = np.mean(psi_scores)
                # Normaliza PSI (0.1 = baixo drift, 0.5 = alto drift)
                psi_z = np.clip(avg_psi / 0.5, 0.0, 1.0)
                self.psi_scores.append(psi_z)
                return psi_z
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"⚠️ Erro no cálculo PSI: {e}")
            return 0.0
    
    def _calculate_psi(self, ref_data: np.ndarray, current_data: np.ndarray, n_bins: int = 10) -> float:
        """Calcula Population Stability Index."""
        try:
            # Cria bins baseados nos dados de referência
            bins = np.percentile(ref_data, np.linspace(0, 100, n_bins + 1))
            bins[0] = bins[0] - 1e-8  # Garante que todos os dados sejam incluídos
            bins[-1] = bins[-1] + 1e-8
            
            # Calcula distribuições
            ref_hist, _ = np.histogram(ref_data, bins=bins)
            current_hist, _ = np.histogram(current_data, bins=bins)
            
            # Normaliza
            ref_hist = ref_hist / (np.sum(ref_hist) + 1e-8)
            current_hist = current_hist / (np.sum(current_hist) + 1e-8)
            
            # Calcula PSI
            psi = 0.0
            for i in range(len(ref_hist)):
                if ref_hist[i] > 0 and current_hist[i] > 0:
                    psi += (current_hist[i] - ref_hist[i]) * np.log(current_hist[i] / ref_hist[i])
            
            return abs(psi)
            
        except Exception as e:
            # logger.warning(f"⚠️ Erro no cálculo PSI detalhado: {e}")
            return 0.0
    
    def _compute_ece_score(self, batch_df: pd.DataFrame, performance_metrics: Optional[Dict]) -> float:
        """Computa ECE (Expected Calibration Error) score."""
        try:
            if 'trend_pred' in batch_df.columns and 'trend_confidence' in batch_df.columns:
                preds = batch_df['trend_pred'].values
                confs = batch_df['trend_confidence'].values
                
                ece = self._calculate_ece_simple(preds, confs)
                
                # Compara com baseline
                if 'ece' in self.baseline_metrics:
                    ece_delta = abs(ece - self.baseline_metrics['ece'])
                    ece_z = np.clip(ece_delta / 0.1, 0.0, 1.0)  # Normaliza
                else:
                    ece_z = np.clip(ece / 0.1, 0.0, 1.0)
                
                self.ece_scores.append(ece_z)
                return ece_z
            else:
                return 0.0
                
        except Exception as e:
            # logger.warning(f"⚠️ Erro no cálculo ECE: {e}")
            return 0.0
    
    def _compute_sharpe_score(self, batch_df: pd.DataFrame, performance_metrics: Optional[Dict]) -> float:
        """Computa Sharpe ratio score."""
        try:
            if performance_metrics and 'sharpe_ratio' in performance_metrics:
                current_sharpe = performance_metrics['sharpe_ratio']
                
                if 'sharpe' in self.baseline_metrics:
                    sharpe_drop = self.baseline_metrics['sharpe'] - current_sharpe
                    sharpe_z = np.clip(sharpe_drop / 0.5, 0.0, 1.0)  # Normaliza
                else:
                    sharpe_z = np.clip((1.0 - current_sharpe) / 2.0, 0.0, 1.0)
                
                self.sharpe_scores.append(sharpe_z)
                return sharpe_z
            else:
                return 0.0
                
        except Exception as e:
            # logger.warning(f"⚠️ Erro no cálculo Sharpe: {e}")
            return 0.0
    
    def _compute_slippage_score(self, execution_stats: Optional[Dict]) -> float:
        """Computa slippage score."""
        try:
            if execution_stats and 'slippage_ratio' in execution_stats:
                slippage_ratio = execution_stats['slippage_ratio']
                
                # Normaliza (1.0 = normal, >1.3 = problema)
                slippage_z = np.clip((slippage_ratio - 1.0) / 0.3, 0.0, 1.0)
                
                self.slippage_scores.append(slippage_z)
                return slippage_z
            else:
                return 0.0
                
        except Exception as e:
            # logger.warning(f"⚠️ Erro no cálculo slippage: {e}")
            return 0.0
    
    def _compute_recon_score(self, batch_df: pd.DataFrame) -> float:
        """Computa SDAE reconstruction score."""
        try:
            if 'smoothness_latent' in batch_df.columns:
                current_recon_mse = batch_df['smoothness_latent'].var()
                
                if self.baseline_recon_mse > 0:
                    recon_ratio = current_recon_mse / self.baseline_recon_mse
                    recon_z = np.clip((recon_ratio - 1.0) / 0.5, 0.0, 1.0)  # Normaliza
                else:
                    recon_z = 0.0
                
                self.recon_scores.append(recon_z)
                return recon_z
            else:
                return 0.0
                
        except Exception as e:
            # logger.warning(f"⚠️ Erro no cálculo recon: {e}")
            return 0.0
    
    def _apply_hysteresis(self, score: float, level: str) -> Tuple[float, str, List[str]]:
        """Aplica histerese para evitar oscilações."""
        actions = []
        
        # Lógica de recuperação
        if level == 'none' and self.recovery_counter > 0:
            self.recovery_counter += 1
            if self.recovery_counter >= self.config.recovery_steps:
                self.recovery_counter = 0
                self.cooldown_counter = 0
                actions.append("recovery_complete")
        elif level == 'none':
            self.recovery_counter = 1
        else:
            self.recovery_counter = 0
        
        # Lógica de cooldown
        if level == 'severe':
            self.cooldown_counter = self.config.cooldown_steps
            actions.append("cooldown_activated")
        elif self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.cooldown_counter == 0:
                actions.append("cooldown_complete")
        
        # Ajusta score baseado na histerese
        adjusted_score = score
        
        # Se em cooldown, força nível severe
        if self.cooldown_counter > 0:
            adjusted_score = max(adjusted_score, self.config.severe_threshold)
            level = 'severe'
            actions.append("cooldown_enforced")
        
        # Se em recuperação, reduz score
        if self.recovery_counter > 0:
            adjusted_score *= 0.8
            actions.append("recovery_discount")
        
        return adjusted_score, level, actions
    
    def get_automatic_actions(self, level: str) -> Dict[str, Any]:
        """Retorna ações automáticas baseadas no nível de drift."""
        actions = {
            'risk_budget_multiplier': 1.0,
            'threshold_increase': 0.0,
            'abstain_bias': 0.0,
            'recalibration_needed': False,
            'meta_adaptation': False
        }
        
        if level == 'mild':
            actions['risk_budget_multiplier'] = self.config.mild_risk_multiplier
            actions['threshold_increase'] = self.config.mild_threshold_increase
            actions['recalibration_needed'] = True
            actions['meta_adaptation'] = True
            
        elif level == 'severe':
            actions['risk_budget_multiplier'] = self.config.severe_risk_multiplier
            actions['threshold_increase'] = self.config.severe_threshold_increase
            actions['abstain_bias'] = 0.3
            actions['recalibration_needed'] = True
            
        return actions
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Gera relatório de drift."""
        if not self.drift_history:
            return {'error': 'No drift history available'}
        
        # Estatísticas gerais
        recent_drifts = list(self.drift_history)[-100:]  # Últimos 100
        
        avg_score = np.mean([d['score'] for d in recent_drifts])
        level_counts = {}
        for d in recent_drifts:
            level = d['level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Componentes médios
        avg_components = {}
        if recent_drifts and 'components' in recent_drifts[0]:
            component_keys = recent_drifts[0]['components'].keys()
            for key in component_keys:
                avg_components[key] = np.mean([d['components'][key] for d in recent_drifts])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_detections': len(self.drift_history),
                'recent_avg_score': float(avg_score),
                'level_distribution': level_counts,
                'recovery_counter': self.recovery_counter,
                'cooldown_counter': self.cooldown_counter
            },
            'components': avg_components,
            'recent_detections': recent_drifts[-10:],  # Últimos 10
            'config': self.config.__dict__
        }
        
        # Salva relatório
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.reports_dir}/drift_report_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Relatório de drift salvo: {output_path}")
        return report
    
    def _save_config(self):
        """Salva configuração do drift detector."""
        config_dict = self.config.__dict__
        
        with open(self.drift_config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"💾 Configuração salva: {self.drift_config_path}")
    
    def load_config(self):
        """Carrega configuração do drift detector."""
        if os.path.exists(self.drift_config_path):
            with open(self.drift_config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Atualiza configuração
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            logger.info(f"📂 Configuração carregada: {self.drift_config_path}")
    
    def get_drift_state(self) -> Dict[str, Any]:
        """Retorna estado atual do drift detector."""
        return {
            'recent_score': self.drift_history[-1]['score'] if self.drift_history else 0.0,
            'recent_level': self.drift_history[-1]['level'] if self.drift_history else 'none',
            'recovery_counter': self.recovery_counter,
            'cooldown_counter': self.cooldown_counter,
            'total_detections': len(self.drift_history)
        }