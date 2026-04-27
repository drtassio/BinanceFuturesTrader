# -----------------------------------------------------------------------------
# ARQUIVO: feature_engineering/financial_stationarity.py
# -----------------------------------------------------------------------------

"""
Validação de Estacionariedade para Séries Financeiras.

Implementa testes estatísticos e transformações para garantir estacionariedade:
1. ADF Test (Augmented Dickey-Fuller) - Testa presença de raiz unitária
2. KPSS Test - Testa estacionariedade contra tendência estocástica
3. Fractional Differencing (De Prado 2018) - Balanceia estacionariedade e memória

Referências Acadêmicas:
    - Dickey & Fuller (1979): "Distribution of the Estimators for Autoregressive 
      Time Series with a Unit Root"
    - Kwiatkowski et al. (1992): "Testing the Null Hypothesis of Stationarity 
      Against the Alternative of a Unit Root"
    - De Prado (2018): "Advances in Financial Machine Learning", Chapter 5
"""

import warnings
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
try:
    from statsmodels.tsa.stattools import adfuller, kpss
except ImportError as e:
    adfuller = None
    kpss = None
    # Use logger if available, otherwise print
    try:
        from utils.logger import get_logger
        import_logger = get_logger("FinancialStationarity")
        import_logger.warning(f"⚠️ [STATIONARITY] statsmodels not found or failed to import: {e}")
    except:
        print(f"Warning: statsmodels not found. Stationarity tests will fail. Error: {e}")
try:
    import torch
except ImportError:
    torch = None
from scipy import stats
from utils.logger import get_logger

logger = get_logger("FinancialStationarity")

warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')


class StationarityValidator:
    """
    Validador de estacionariedade para séries temporais financeiras.
    
    Aplica múltiplos testes para robustez:
    - ADF (null: unit root exists → non-stationary)
    - KPSS (null: stationary → if rejected, non-stationary)
    """
    
    def __init__(self, adf_significance: float = 0.05, kpss_significance: float = 0.05):
        """
        Args:
            adf_significance: Nível de significância para ADF test
            kpss_significance: Nível de significância para KPSS test
        """
        self.adf_alpha = adf_significance
        self.kpss_alpha = kpss_significance
    
    def test_adf(self, series: pd.Series, maxlag: Optional[int] = None) -> Dict[str, Any]:
        """
        Augmented Dickey-Fuller Test para raiz unitária.
        
        H0: Série tem raiz unitária (não estacionária)
        H1: Série é estacionária
        
        Args:
            series: Série temporal a testar
            maxlag: Máximo de lags para o teste (None = auto)
            
        Returns:
            Dict com estatística, p-value, e resultado
        """
        try:
            # Remove NaNs
            clean_series = series.dropna()
            
            # [SCIENTIFIC FIX] O teste ADF deve ser feito em blocos CONTÍGUOS para preservar
            # as propriedades de autocorrelação e detecção de raiz unitária.
            # Amostragem sistemática (iloc[::step]) distorce o teste estatístico.
            if len(clean_series) > 50000:
                clean_series = clean_series.tail(50000)
            
            # Executa ADF
            result = adfuller(clean_series, maxlag=maxlag, autolag='AIC')
            adf_stat, pvalue, usedlag, nobs, critical_values, icbest = result
            
            # Rejeita H0 se pvalue < alpha → série é estacionária
            is_stationary = pvalue < self.adf_alpha
            
            return {
                'test': 'ADF',
                'statistic': float(adf_stat),
                'pvalue': float(pvalue),
                'used_lag': int(usedlag),
                'n_obs': int(nobs),
                'critical_values': {k: float(v) for k, v in critical_values.items()},
                'is_stationary': is_stationary,
                'interpretation': 'Stationary' if is_stationary else 'Non-stationary (unit root)'
            }
            
        except Exception as e:
            logger.warning(f"ADF test falhou: {e}")
            return {
                'test': 'ADF',
                'statistic': np.nan,
                'pvalue': np.nan,
                'is_stationary': False,
                'error': str(e)
            }
    
    def test_kpss(self, series: pd.Series, regression: str = 'c') -> Dict[str, Any]:
        """
        KPSS Test para estacionariedade.
        
        H0: Série é estacionária
        H1: Série tem raiz unitária
        
        Args:
            series: Série temporal a testar
            regression: 'c' (constant) ou 'ct' (constant + trend)
            
        Returns:
            Dict com estatística, p-value, e resultado
        """
        try:
            clean_series = series.dropna()
            
            # [SCIENTIFIC FIX] KPSS contíguo (tail 50k)
            if len(clean_series) > 50000:
                clean_series = clean_series.tail(50000)
            
            # Executa KPSS
            kpss_stat, pvalue, lags, critical_values = kpss(
                clean_series, 
                regression=regression,
                nlags='auto'
            )
            
            # NÃO rejeita H0 se pvalue > alpha → série é estacionária
            is_stationary = pvalue > self.kpss_alpha
            
            return {
                'test': 'KPSS',
                'statistic': float(kpss_stat),
                'pvalue': float(pvalue),
                'lags': int(lags),
                'critical_values': {k: float(v) for k, v in critical_values.items()},
                'is_stationary': is_stationary,
                'interpretation': 'Stationary' if is_stationary else 'Non-stationary (trend/drift)'
            }
            
        except Exception as e:
            logger.warning(f"KPSS test falhou: {e}")
            return {
                'test': 'KPSS',
                'statistic': np.nan,
                'pvalue': np.nan,
                'is_stationary': False,
                'error': str(e)
            }
    
    def validate_series(self, series: pd.Series, name: str = "") -> Dict[str, Any]:
        """
        Valida estacionariedade usando ADF + KPSS (abordagem robusta).
        
        Casos:
        1. ADF: stationary, KPSS: stationary → STATIONARY
        2. ADF: non-stationary, KPSS: non-stationary → NON-STATIONARY
        3. ADF: stationary, KPSS: non-stationary → TREND-STATIONARY (diferencing needed)
        4. ADF: non-stationary, KPSS: stationary → DIFFERENCE-STATIONARY (rare)
        
        Args:
            series: Série a validar
            name: Nome da série (para logging)
            
        Returns:
            Dict com resultados combinados
        """
        adf_result = self.test_adf(series)
        kpss_result = self.test_kpss(series)
        
        # Decisão combinada (conservadora)
        both_stationary = adf_result['is_stationary'] and kpss_result['is_stationary']
        
        # Classifica tipo de não-estacionariedade
        if both_stationary:
            status = 'STATIONARY'
        elif not adf_result['is_stationary'] and not kpss_result['is_stationary']:
            status = 'NON-STATIONARY (unit root + trend)'
        elif adf_result['is_stationary'] and not kpss_result['is_stationary']:
            status = 'TREND-STATIONARY (needs detrending)'
        else:
            status = 'DIFFERENCE-STATIONARY'
        
        return {
            'name': name,
            'status': status,
            'is_stationary': both_stationary,
            'adf': adf_result,
            'kpss': kpss_result
        }
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        significance_level: float = 0.05,
        max_cols_to_test: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Valida estacionariedade de todas as colunas de um DataFrame.
        
        Args:
            df: DataFrame com séries temporais
            significance_level: Nível de significância (override do __init__)
            max_cols_to_test: Máximo de colunas a testar (None = todas)
            
        Returns:
            Dict com estatísticas agregadas
        """
        # Override significance levels temporariamente
        original_adf_alpha = self.adf_alpha
        original_kpss_alpha = self.kpss_alpha
        self.adf_alpha = significance_level
        self.kpss_alpha = significance_level
        
        try:
            cols_to_test = df.columns[:max_cols_to_test] if max_cols_to_test else df.columns
            
            results = []
            stationary_cols = []
            non_stationary_cols = []
            
            logger.info(f"🔍 [Stationarity] Testando {len(cols_to_test)} colunas...")
            
            for col in cols_to_test:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                    
                result = self.validate_series(df[col], name=col)
                results.append(result)
                
                if result['is_stationary']:
                    stationary_cols.append(col)
                else:
                    non_stationary_cols.append(col)
            
            pct_stationary = len(stationary_cols) / len(results) * 100 if results else 0
            
            logger.info(
                f"✅ [Stationarity] Resultado: {len(stationary_cols)}/{len(results)} "
                f"estacionárias ({pct_stationary:.1f}%)"
            )
            
            return {
                'total_tested': len(results),
                'n_stationary': len(stationary_cols),
                'n_non_stationary': len(non_stationary_cols),
                'pct_stationary': pct_stationary,
                'stationary': stationary_cols,
                'non_stationary': non_stationary_cols,
                'details': results
            }
            
        finally:
            # Restore original alphas
            self.adf_alpha = original_adf_alpha
            self.kpss_alpha = original_kpss_alpha
    
    def fractional_diff(
        self, 
        series: pd.Series, 
        d: float = 0.5, 
        threshold: float = 1e-5
    ) -> pd.Series:
        """
        Fractional Differencing (De Prado 2018, Chapter 5).
        
        Formula:
            X_t^d = Σ(k=0 to ∞) w_k * X_{t-k}
            onde w_k = (-1)^k * Γ(d+1) / (Γ(k+1) * Γ(d-k+1))
        
        Balanceia:
        - d=0: série original (máxima memória, não estacionária)
        - d=1: primeira diferença (estacionária, perde memória)
        - 0<d<1: compromisso ótimo
        
        Args:
            series: Série temporal
            d: Ordem fracionária (0.4-0.6 típico para finanças)
            threshold: Corta pesos quando |w_k| < threshold
            
        Returns:
            Série fracionalmente diferenciada
        """
        # Calcula pesos
        weights = self._get_frac_diff_weights(d, threshold)
        
        # --- ACELERAÇÃO GPU (TORCH) OU VECTORIZED NUMPY ---
        if torch is not None and torch.cuda.is_available():
            try:
                # Converte para tensor e move para GPU
                device = torch.device('cuda')
                series_tensor = torch.tensor(series.values, dtype=torch.float32, device=device).view(1, 1, -1)
                weights_tensor = torch.tensor(weights[::-1].copy(), dtype=torch.float32, device=device).view(1, 1, -1)
                
                # Padding inicial para manter alinhamento
                padding = len(weights) - 1
                
                # Aplica Convolução 1D (equivalente ao dot product móvel)
                with torch.no_grad():
                    output_tensor = torch.nn.functional.conv1d(series_tensor, weights_tensor, padding=0)
                
                # Converte de volta para CPU/Pandas
                output_values = output_tensor.cpu().numpy().flatten()
                
                # Remonta a série com NaNs iniciais
                final_values = np.empty(len(series))
                final_values[:] = np.nan
                final_values[padding:] = output_values
                
                return pd.Series(final_values, index=series.index)
            except Exception as torch_err:
                logger.warning(f"⚠️ GPU FracDiff falhou, usando fallback CPU: {torch_err}")
        
        # Fallback Vectorized Numpy (Convolve) - Muito mais rápido que loop Python
        padding = len(weights) - 1
        output_values = np.convolve(series.values, weights[::-1], mode='valid')
        
        final_values = np.empty(len(series))
        final_values[:] = np.nan
        final_values[padding:] = output_values
        
        return pd.Series(final_values, index=series.index)
    
    def _get_frac_diff_weights(self, d: float, threshold: float = 1e-5) -> np.ndarray:
        """
        Calcula pesos para fractional differencing.
        
        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k
        
        Args:
            d: Ordem fracionária
            threshold: Para quando |w_k| < threshold
            
        Returns:
            Array de pesos
        """
        weights = [1.0]
        k = 1
        
        while True:
            w_k = -weights[-1] * (d - k + 1) / k
            
            if abs(w_k) < threshold:
                break
                
            weights.append(w_k)
            k += 1
            
            # Safety: limita a 1000 pesos
            if k > 1000:
                break
        
        return np.array(weights)
    
    def make_stationary(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        method: str = 'fractional_diff',
        d_optimal: float = 0.5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transforma DataFrames para garantir estacionariedade.
        
        Args:
            train_df: DataFrame de treino
            val_df: DataFrame de validação
            method: 'fractional_diff', 'diff', ou 'returns'
            d_optimal: Ordem fracionária (se method='fractional_diff')
            
        Returns:
            (train_transformed, val_transformed)
        """
        logger.info(f"🔧 [Stationarity] Aplicando transformação: {method}")
        
        if method == 'fractional_diff':
            # Aplica fractional differencing em cada coluna
            train_transformed = train_df.copy()
            val_transformed = val_df.copy()
            
            for col in train_df.columns:
                if not pd.api.types.is_numeric_dtype(train_df[col]):
                    continue
                
                # Concatena train+val para manter continuidade temporal
                full_series = pd.concat([train_df[col], val_df[col]])
                diff_series = self.fractional_diff(full_series, d=d_optimal)
                
                # Separa novamente
                train_transformed[col] = diff_series.iloc[:len(train_df)]
                val_transformed[col] = diff_series.iloc[len(train_df):]
            
            # Remove NaNs iniciais
            train_transformed = train_transformed.dropna()
            val_transformed = val_transformed.dropna()
            
        elif method == 'diff':
            # Primeira diferença simples
            train_transformed = train_df.diff().dropna()
            val_transformed = val_df.diff().dropna()
            
        elif method == 'returns':
            # Log returns
            train_transformed = np.log(train_df / train_df.shift(1)).dropna()
            val_transformed = np.log(val_df / val_df.shift(1)).dropna()
            
        else:
            raise ValueError(f"Método inválido: {method}")
        
        logger.info(
            f"✅ [Stationarity] Transformação completa. "
            f"Train: {len(train_df)} → {len(train_transformed)}, "
            f"Val: {len(val_df)} → {len(val_transformed)}"
        )
        
        return train_transformed, val_transformed
    
    def find_optimal_d(
        self, 
        series: pd.Series, 
        d_range: Tuple[float, float] = (0.1, 1.0),
        step: float = 0.1,
        p_threshold: float = 0.05
    ) -> float:
        """
        Encontra a menor ordem fracionária d que garante estacionariedade.
        
        Objetivo (De Prado): Maximizar preservação de memória mantendo p-value < threshold.
        """
        d_values = np.arange(d_range[0], d_range[1] + step, step)
        
        for d in d_values:
            diff_series = self.fractional_diff(series, d=d)
            diff_series_clean = diff_series.dropna()
            
            if len(diff_series_clean) < 20:
                continue
            
            adf_result = self.test_adf(diff_series_clean)
            pvalue = adf_result.get('pvalue', 1.0)
            
            if pvalue < p_threshold:
                logger.info(
                    f"📊 [Stationarity] d={d:.2f} atingiu p-value={pvalue:.4f} (< {p_threshold}). "
                    f"Parando busca para preservar memória."
                )
                return float(d)
        
        logger.warning(f"⚠️ [Stationarity] Nenhum d no range atingiu p < {p_threshold}. Retornando d=1.0 (Standard Diff)")
        return 1.0


# Exemplo de uso
if __name__ == "__main__":
    # Cria série não-estacionária (random walk)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    series = pd.Series(prices)
    
    validator = StationarityValidator()
    
    # Testa série original
    print("=== Teste: Série Original (Random Walk) ===")
    result_original = validator.validate_series(series, name="prices")
    print(f"Status: {result_original['status']}")
    print(f"ADF p-value: {result_original['adf']['pvalue']:.4f}")
    print(f"KPSS p-value: {result_original['kpss']['pvalue']:.4f}")
    
    # Aplica fractional differencing
    print("\n=== Fractional Differencing (d=0.5) ===")
    diff_series = validator.fractional_diff(series, d=0.5)
    result_diff = validator.validate_series(diff_series.dropna(), name="frac_diff")
    print(f"Status: {result_diff['status']}")
    print(f"ADF p-value: {result_diff['adf']['pvalue']:.4f}")
    print(f"KPSS p-value: {result_diff['kpss']['pvalue']:.4f}")
    
    # Encontra d ótimo
    print("\n=== Otimizando d ===")
    optimal_d = validator.find_optimal_d(series)
    print(f"d ótimo: {optimal_d:.2f}")
