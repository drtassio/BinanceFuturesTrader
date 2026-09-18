import numpy as np
import pandas as pd
from typing import Dict, Any

def calculate_shannon_entropy(series: pd.Series, bins: int = 10) -> float:
    """Calcula a Entropia de Shannon (Desordem)"""
    try:
        # Usamos bins para discretizar os retornos e calcular a distribuição
        counts, _ = np.histogram(series, bins=bins)
        prob = counts / (len(series) + 1e-12)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log2(prob))
        # Normaliza para escala 0-5 para facilitar leitura (Shannnon Entropy para 10 bins max é log2(10) ≈ 3.32)
        return float(np.clip(entropy, 0.0, 5.0))
    except Exception:
        return 0.0

def calculate_hurst_exponent(series: pd.Series) -> float:
    """
    Calcula o Expoente de Hurst via R/S (Rescaled Range Analysis).
    H > 0.5: Tendência (Persistente)
    H < 0.5: Reversão à Média (Anti-persistente)
    H = 0.5: Movimento Aleatório (Ruído)
    """
    try:
        arr = np.array(series, dtype=float)
        n = len(arr)
        if n < 20:
            return 0.5

        lags = sorted(set([max(4, n // 8), max(8, n // 4), max(12, n // 2)]))
        rs_values, valid_lags = [], []

        for lag in lags:
            chunks = [arr[i:i + lag] for i in range(0, n - lag + 1, lag)]
            if len(chunks) < 2:
                continue
            rs_list = []
            for chunk in chunks:
                dev = np.cumsum(chunk - np.mean(chunk))
                r = np.max(dev) - np.min(dev)
                s = np.std(chunk, ddof=1)
                if s > 1e-9:
                    rs_list.append(r / s)
            if rs_list:
                rs_values.append(np.mean(rs_list))
                valid_lags.append(lag)

        if len(valid_lags) < 2:
            return 0.5

        poly = np.polyfit(np.log(valid_lags), np.log(rs_values), 1)
        return float(np.clip(poly[0], 0.1, 0.9))
    except Exception:
        return 0.5

def get_market_chaos_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Retorna um dicionário com métricas de física do mercado."""
    try:
        if df is None or len(df) < 30:
            return {"entropy": 0.0, "hurst": 0.5, "chaos_label": "Iniciando...", "shannon_entropy": 0.0, "hurst_exponent": 0.5, "market_state": "Iniciando..."}
        
        returns = df['close'].pct_change().dropna().tail(30)
        
        entropy = calculate_shannon_entropy(returns)
        hurst = calculate_hurst_exponent(returns)
        
        # Labels de interpretação
        if entropy > 3.0:
            label = "CAÓTICO (Alto Risco)"
        elif hurst < 0.40:
            label = "REVERSÃO (Ranger)"
        elif hurst > 0.60:
            label = "TENDÊNCIA (Forte)"
        else:
            label = "ESTÁVEL"
            
        return {
            "entropy": round(entropy, 2),
            "hurst": round(hurst, 2),
            "chaos_label": label,
            # Chaves lidas pelo AIController e pelo ambiente de treino
            "shannon_entropy": round(entropy, 2),
            "hurst_exponent": round(hurst, 2),
            "market_state": label,
        }
    except Exception:
        return {"entropy": 0.0, "hurst": 0.5, "chaos_label": "Erro no Sensor", "shannon_entropy": 0.0, "hurst_exponent": 0.5, "market_state": "Erro no Sensor"}


def rolling_chaos_metrics(close, lookback: int = 30):
    """
    Entropia e Hurst por candle, idênticos a get_market_chaos_metrics() aplicado à janela
    que termina em cada candle. Usado no pré-cálculo do ambiente de treino e na inferência ao vivo.
    Retorna (entropy, hurst) como arrays numpy do mesmo tamanho de `close`.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    entropy = np.zeros(n, dtype=np.float32)
    hurst = np.full(n, 0.5, dtype=np.float32)
    for i in range(lookback - 1, n):
        window = close[max(0, i - lookback): i + 1]
        metrics = get_market_chaos_metrics(pd.DataFrame({"close": window}))
        entropy[i] = metrics["entropy"]
        hurst[i] = metrics["hurst"]
    return entropy, hurst
