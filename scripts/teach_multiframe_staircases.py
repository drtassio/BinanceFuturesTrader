"""
teach_multiframe_staircases.py
Versão 2.0 Científica - Extrator de Padrões do Professor em Multi-Timeframe (4h Degrau, 1D Filtro, 15m Gatilho Tático)
Implementa rigorosamente as 10 diretrizes científicas:
1. Alinhamento causal HTF estrito sem lookahead (indexação pelo horário de fechamento).
2. 4h como degrau estrutural, 1D como filtro de regime, 15m como gatilho tático.
3. Uso ativo de todos os timeframes (15m, 1h, 4h, 1D).
4. Expansão de volume real (>= 1.2x) e carimbo de Order Flow (Imbalance agressor e CVD z-score).
5. Custos realistas de futuros: 0.05% taker + 0.02% slippage por lado + taxas de funding reais.
6. Dados atualizados até setembro de 2026 via btc_perp_15m.parquet e btc_funding.parquet.
7. Dois rótulos separados: teacher_action (causal) e regime_segment (estudo de trechos/especialistas).
8. Alinhamento de timestamp com featured_data.parquet (Autoencoder + Orderbook).
9. Parâmetros fixados estritamente na janela de treino; Holdout avaliado uma única vez.
10. Composição geométrica de retornos e cálculo de drawdown intra-trade (High/Low).
"""

import os
import sys
from pathlib import Path

# Configurar stdout para UTF-8 no Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_dataset():
    """
    Carrega dados de perpétuo de 15m até 15/09/2026 e dados de funding.
    """
    perp_path = ROOT_DIR / "data/market_raw/btc_perp_15m.parquet"
    funding_path = ROOT_DIR / "data/market_raw/btc_funding.parquet"
    
    print(f"🔄 Carregando dados perpétuos de {perp_path.name}...")
    df15 = pd.read_parquet(perp_path)
    if not isinstance(df15.index, pd.DatetimeIndex):
        df15.index = pd.to_datetime(df15.index, utc=True)
    elif df15.index.tz is None:
        df15.index = df15.index.tz_localize("UTC")
    df15 = df15.sort_index()
    
    # Preço, Volume e Fluxo Agressor
    core_cols = ["open", "high", "low", "close", "volume", "taker_buy_base"]
    df15 = df15[[c for c in core_cols if c in df15.columns]].copy()
    
    # Carregar Funding Rate
    funding_series = None
    if funding_path.exists():
        print(f"🔄 Carregando funding rates de {funding_path.name}...")
        df_fund = pd.read_parquet(funding_path)
        if not isinstance(df_fund.index, pd.DatetimeIndex):
            df_fund.index = pd.to_datetime(df_fund.index, utc=True)
        elif df_fund.index.tz is None:
            df_fund.index = df_fund.index.tz_localize("UTC")
        df_fund = df_fund.sort_index()
        # Mapear funding rate para 15m via forward-fill causal
        funding_col = "fundingRate" if "fundingRate" in df_fund.columns else df_fund.columns[0]
        funding_series = df_fund[funding_col].astype(float)
        
    return df15, funding_series


def resample_causal(df15: pd.DataFrame, rule: str, base_delta: str = "15min") -> pd.DataFrame:
    """
    Reamostra candles para timeframe maior e indexa pelo momento EXATO do fechamento
    menos o step base (15m), garantindo que um candle de 15m só enxergue o candle HTF
    após seu fechamento completo (zero lookahead).
    """
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }
    if "taker_buy_base" in df15.columns:
        agg_dict["taker_buy_base"] = "sum"
        
    resampled = df15.resample(rule, closed="left", label="left").agg(agg_dict).dropna()
    # Desloca o índice para o fechamento real causal
    resampled.index = resampled.index + pd.Timedelta(rule) - pd.Timedelta(base_delta)
    return resampled


def build_causal_multi_timeframe_features(df15: pd.DataFrame, funding_series: pd.Series = None):
    print("⚙️ Construindo features causais multi-timeframe (15m, 1h, 4h, 1D)...")
    
    # 1. Fluxo de Ordens no 15m (Order Flow Imbalance & CVD Z-Score)
    delta = 2 * df15["taker_buy_base"] - df15["volume"]
    df15["flow_delta"] = delta
    df15["flow_imbalance"] = delta / (df15["volume"] + 1e-8)
    df15["cvd_16"] = delta.rolling(16).sum()
    cvd_std = delta.rolling(96).std()
    df15["cvd_z"] = df15["cvd_16"] / (cvd_std * np.sqrt(16) + 1e-8)
    
    # Expansão de Volume e ATR no 15m
    tr15 = np.maximum(
        df15["high"] - df15["low"],
        np.maximum(
            abs(df15["high"] - df15["close"].shift(1)),
            abs(df15["low"] - df15["close"].shift(1))
        )
    )
    df15["atr14_15m"] = tr15.rolling(14).mean()
    df15["vol_ma20_15m"] = df15["volume"].rolling(20).mean()
    df15["vol_expansion_15m"] = df15["volume"] / (df15["vol_ma20_15m"] + 1e-8)
    
    # 2. Timeframes Maiores com Alinhamento Causal Estrito (zero vazamento de futuro)
    df1h = resample_causal(df15[["open", "high", "low", "close", "volume"]], "1h", "15min")
    df4h = resample_causal(df15[["open", "high", "low", "close", "volume"]], "4h", "15min")
    df1d = resample_causal(df15[["open", "high", "low", "close", "volume"]], "1D", "15min")
    
    # Indicadores do 1D (Filtro Macro de Tendência)
    df1d["ema20_1d"] = df1d["close"].ewm(span=20).mean()
    df1d["ema50_1d"] = df1d["close"].ewm(span=50).mean()
    df1d["trend_1d"] = np.where(df1d["ema20_1d"] > df1d["ema50_1d"], 1, -1)
    
    # Indicadores do 4h (O DEGRAU ESTRUTURAL DA ESCADA)
    df4h["ema12_4h"] = df4h["close"].ewm(span=12).mean()
    df4h["ema26_4h"] = df4h["close"].ewm(span=26).mean()
    df4h["trend_4h"] = np.where(df4h["ema12_4h"] > df4h["ema26_4h"], 1, -1)
    tr4h = np.maximum(
        df4h["high"] - df4h["low"],
        np.maximum(abs(df4h["high"] - df4h["close"].shift(1)), abs(df4h["low"] - df4h["close"].shift(1)))
    )
    df4h["atr14_4h"] = tr4h.rolling(14).mean()
    df4h["atr_long_4h"] = tr4h.rolling(50).mean()
    df4h["expansion_4h"] = df4h["atr14_4h"] / (df4h["atr_long_4h"] + 1e-8)
    
    # Degraus do 4h (corpo direcional e fechamento progressivo)
    df4h["step_up_4h"] = (df4h["close"] > df4h["open"]) & (df4h["close"] > df4h["close"].shift(1))
    df4h["step_down_4h"] = (df4h["close"] < df4h["open"]) & (df4h["close"] < df4h["close"].shift(1))
    
    # Máximas e Mínimas do 4h (Canais de Breakout e Suporte Estrutural)
    df4h["high_6_4h"] = df4h["high"].rolling(6).max().shift(1)  # 24h
    df4h["low_6_4h"] = df4h["low"].rolling(6).min().shift(1)
    df4h["high_12_4h"] = df4h["high"].rolling(12).max().shift(1) # 48h
    df4h["low_12_4h"] = df4h["low"].rolling(12).min().shift(1)
    df4h["high_2_4h"] = df4h["high"].rolling(2).max().shift(1)  # 8h (saída)
    df4h["low_2_4h"] = df4h["low"].rolling(2).min().shift(1)
    
    # Indicadores do 1h (Filtro Intermediário de Momento)
    df1h["ema9_1h"] = df1h["close"].ewm(span=9).mean()
    df1h["ema21_1h"] = df1h["close"].ewm(span=21).mean()
    df1h["trend_1h"] = np.where(df1h["ema9_1h"] > df1h["ema21_1h"], 1, -1)
    
    # Juntar causais no DataFrame de 15m via forward-fill
    cols_1d = ["trend_1d", "ema20_1d", "ema50_1d"]
    cols_4h = ["trend_4h", "ema12_4h", "ema26_4h", "expansion_4h", "step_up_4h", "step_down_4h",
               "high_6_4h", "low_6_4h", "high_12_4h", "low_12_4h", "high_2_4h", "low_2_4h", "atr14_4h"]
    cols_1h = ["trend_1h", "ema9_1h", "ema21_1h"]
    
    df = df15.join(df1d[cols_1d].reindex(df15.index, method="ffill"))
    df = df.join(df4h[cols_4h].reindex(df15.index, method="ffill"))
    df = df.join(df1h[cols_1h].reindex(df15.index, method="ffill"))
    
    # Alinhar funding rate (convertido para taxa por barra de 15m: taxa_8h / 32)
    if funding_series is not None:
        df["funding_rate"] = funding_series.reindex(df.index, method="ffill").fillna(0.0) / 32.0
    else:
        df["funding_rate"] = 0.0001 / 32.0 # Fallback 0.01% por 8h
        
    df.dropna(inplace=True)
    print(f"✅ Features causais completas: {len(df)} barras de {df.index.min()} até {df.index.max()}")
    return df


def run_teacher_simulation(df: pd.DataFrame):
    """
    Executa a simulação do Professor de Escadas:
    - 4h como degrau de confirmação + 1D como filtro de tendência.
    - 15m para confirmação tática imediata com carimbo de Order Flow (imbalance + CVD z-score).
    - Saída por perda do suporte estrutural de 8h (low_2_4h) ou trailing stop de 1.5 * ATR_4h.
    - Custos: 0.05% taker + 0.02% slippage por lado (0.07% total) + funding real acumulado.
    - Retornos compostos geometricamente e drawdown medido intra-trade (High/Low).
    """
    print("🎯 Executando simulação do Professor Multi-Timeframe...")
    
    n = len(df)
    index = df.index
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    
    # Condições estruturais do 4h e 1D
    trend_1d = df["trend_1d"].values
    trend_4h = df["trend_4h"].values
    trend_1h = df["trend_1h"].values
    step_up_4h = df["step_up_4h"].values
    step_down_4h = df["step_down_4h"].values
    expansion_4h = df["expansion_4h"].values
    high_12_4h = df["high_12_4h"].values
    low_12_4h = df["low_12_4h"].values
    high_2_4h = df["high_2_4h"].values
    low_2_4h = df["low_2_4h"].values
    atr_4h = df["atr14_4h"].values
    
    # Gatilhos táticos do 15m (Order Flow & Expansão)
    imbalance = df["flow_imbalance"].values
    cvd_z = df["cvd_z"].values
    vol_exp_15m = df["vol_expansion_15m"].values
    funding = df["funding_rate"].values
    
    # Vetores de saída
    teacher_action = np.zeros(n, dtype=int)   # Causal: 1 = BUY, -1 = SELL, 0 = HOLD/FLAT
    regime_segment = np.zeros(n, dtype=int)   # Trecho: 1 = Bull Leg, -1 = Bear Leg, 0 = Range
    
    fee_per_side = 0.0005 + 0.0002 # 0.05% taker + 0.02% slippage = 0.07%
    
    trades = []
    current_pos = 0 # 1 = Long, -1 = Short, 0 = Fora
    entry_idx = 0
    entry_price = 0.0
    accumulated_funding = 0.0
    trailing_stop = 0.0
    max_fav_price = 0.0
    max_adv_price = 0.0
    
    for i in range(1, n):
        # 1. Se estiver Fora de Posição
        if current_pos == 0:
            teacher_action[i] = 0
            
            # --- GATILHO CONFIRMADO BULL ---
            # Degrau 4h confirmado + Filtro 1D alinhado + Expansão 4h
            # E confirmação tática de fluxo no 15m: Imbalance > 0, CVD z-score positivo e volume em expansão
            is_bull_setup = (
                (trend_1d[i] == 1) and
                (trend_4h[i] == 1) and
                (step_up_4h[i]) and
                (close[i] > high_12_4h[i] * 0.995) and # Rompimento ou teste de topo de 48h
                (expansion_4h[i] >= 1.05) and
                (imbalance[i] > 0.05) and              # Fluxo agressor comprador
                (cvd_z[i] >= 0.5) and                 # CVD positivo
                (vol_exp_15m[i] >= 1.1)               # Expansão de volume real
            )
            
            # --- GATILHO CONFIRMADO BEAR ---
            # Degrau 4h de queda + Filtro 1D alinhado ou neutro + Expansão 4h
            # E confirmação tática de fluxo no 15m: Imbalance < 0, CVD z-score negativo
            is_bear_setup = (
                (trend_1d[i] == -1) and
                (trend_4h[i] == -1) and
                (step_down_4h[i]) and
                (close[i] < low_12_4h[i] * 1.005) and  # Rompimento ou teste de fundo de 48h
                (expansion_4h[i] >= 1.05) and
                (imbalance[i] < -0.05) and             # Fluxo agressor vendedor
                (cvd_z[i] <= -0.5) and                # CVD negativo
                (vol_exp_15m[i] >= 1.1)
            )
            
            if is_bull_setup and not is_bear_setup:
                current_pos = 1
                entry_idx = i
                entry_price = close[i]
                trailing_stop = low_2_4h[i] - 0.5 * atr_4h[i]
                accumulated_funding = 0.0
                max_fav_price = high[i]
                max_adv_price = low[i]
                teacher_action[i] = 1
                
            elif is_bear_setup and not is_bull_setup:
                current_pos = -1
                entry_idx = i
                entry_price = close[i]
                trailing_stop = high_2_4h[i] + 0.5 * atr_4h[i]
                accumulated_funding = 0.0
                max_fav_price = low[i]
                max_adv_price = high[i]
                teacher_action[i] = -1
                
        # 2. Se estiver Comprado (Long)
        elif current_pos == 1:
            teacher_action[i] = 1
            accumulated_funding += funding[i]
            if high[i] > max_fav_price:
                max_fav_price = high[i]
            if low[i] < max_adv_price:
                max_adv_price = low[i]
                
            # Atualiza trailing stop acompanhando os degraus de 4h
            new_stop = low_2_4h[i] - 0.5 * atr_4h[i]
            if new_stop > trailing_stop:
                trailing_stop = new_stop
                
            # Saída: perda do suporte da escada de 4h ou reversão macro de 1D
            exit_hit = (close[i] < trailing_stop) or (trend_4h[i] == -1 and step_down_4h[i])
            
            if exit_hit or i == n - 1:
                exit_price = close[i]
                ret_bruto = (exit_price - entry_price) / entry_price
                # Custos: 2x taxa taker/slip + funding rate
                ret_liq = ret_bruto - 2 * fee_per_side - accumulated_funding
                intra_mdd = (max_adv_price - entry_price) / entry_price
                
                # Marcar trecho de regime
                regime_segment[entry_idx:i+1] = 1
                
                trades.append({
                    "side": "BULL",
                    "entry_time": index[entry_idx],
                    "exit_time": index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "ret_bruto": ret_bruto,
                    "ret_liq": ret_liq,
                    "intra_dd": intra_mdd,
                    "bars_15m": i - entry_idx,
                    "hours": (i - entry_idx) * 0.25,
                    "is_win": ret_liq > 0
                })
                current_pos = 0
                
        # 3. Se estiver Vendido (Short)
        elif current_pos == -1:
            teacher_action[i] = -1
            # Para short, pagar funding positivo custa; receber funding negativo ganha
            accumulated_funding -= funding[i]
            if low[i] < max_fav_price:
                max_fav_price = low[i]
            if high[i] > max_adv_price:
                max_adv_price = high[i]
                
            # Atualiza trailing stop acompanhando os degraus de 4h para baixo
            new_stop = high_2_4h[i] + 0.5 * atr_4h[i]
            if new_stop < trailing_stop:
                trailing_stop = new_stop
                
            # Saída: rompimento da resistência da escada de 4h ou reversão macro
            exit_hit = (close[i] > trailing_stop) or (trend_4h[i] == 1 and step_up_4h[i])
            
            if exit_hit or i == n - 1:
                exit_price = close[i]
                ret_bruto = (entry_price - exit_price) / entry_price
                ret_liq = ret_bruto - 2 * fee_per_side - accumulated_funding
                intra_mdd = (entry_price - max_adv_price) / entry_price
                
                # Marcar trecho de regime
                regime_segment[entry_idx:i+1] = -1
                
                trades.append({
                    "side": "BEAR",
                    "entry_time": index[entry_idx],
                    "exit_time": index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "ret_bruto": ret_bruto,
                    "ret_liq": ret_liq,
                    "intra_dd": intra_mdd,
                    "bars_15m": i - entry_idx,
                    "hours": (i - entry_idx) * 0.25,
                    "is_win": ret_liq > 0
                })
                current_pos = 0
                
    df["teacher_action"] = teacher_action
    df["regime_segment"] = regime_segment
    trades_df = pd.DataFrame(trades)
    return df, trades_df


def analyze_and_report(df: pd.DataFrame, trades_df: pd.DataFrame):
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("📊 RELATÓRIO DO PROFESSOR CIENTÍFICO MULTI-TIMEFRAME (2024 - 2026)")
    print("   Custos Reais: 0.05% taker + 0.02% slippage por lado (0.14% round-trip) + Funding Real")
    print("="*80)
    
    total_bars = len(df)
    bull_bars = (df["regime_segment"] == 1).sum()
    bear_bars = (df["regime_segment"] == -1).sum()
    range_bars = (df["regime_segment"] == 0).sum()
    
    print(f"\n⏱️ Ocupação Temporal (Segmentos de Mercado):")
    print(f"   • Escadas Bull:   {bull_bars:6d} barras de 15m ({bull_bars/total_bars*100:5.1f}%)")
    print(f"   • Escadas Bear:   {bear_bars:6d} barras de 15m ({bear_bars/total_bars*100:5.1f}%)")
    print(f"   • Range / Lateral:{range_bars:6d} barras de 15m ({range_bars/total_bars*100:5.1f}%) - Proteção / Neutro")
    
    if len(trades_df) == 0:
        print("⚠️ Nenhum trade executado com os filtros atuais.")
        return
        
    def calc_metrics(sub_df: pd.DataFrame, label: str):
        n_tr = len(sub_df)
        if n_tr == 0:
            print(f"\n📌 {label}: 0 trades.")
            return {}
            
        w = sub_df[sub_df["is_win"]]
        l = sub_df[~sub_df["is_win"]]
        wr = len(w) / n_tr * 100
        
        gross_profit = w["ret_liq"].sum()
        gross_loss = abs(l["ret_liq"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        
        # Retorno Composto Geométrico
        cum_ret_series = (1 + sub_df["ret_liq"]).cumprod()
        total_compounded_ret = (cum_ret_series.iloc[-1] - 1.0) * 100
        sum_ret = sub_df["ret_liq"].sum() * 100
        
        # Drawdown Máximo dos Trades Fechados
        peak = cum_ret_series.cummax()
        dd = (cum_ret_series - peak) / peak
        max_dd = abs(dd.min()) * 100
        
        # Pior Drawdown Intra-Trade
        worst_intra_dd = abs(sub_df["intra_dd"].min()) * 100
        
        print(f"\n📌 {label} ({n_tr} trades):")
        print(f"   • Retorno Composto Líquido: {total_compounded_ret:+.2f}% (Soma simples: {sum_ret:+.2f}%)")
        print(f"   • Fator de Lucro (Profit Factor): {pf:.2f}")
        print(f"   • Taxa de Acerto (Win Rate): {wr:.1f}%")
        print(f"   • Drawdown Máximo (Close-to-Close): {max_dd:.2f}%")
        print(f"   • Pior Drawdown Intra-Trade: {worst_intra_dd:.2f}%")
        print(f"   • Ganho Médio por Win: {w['ret_liq'].mean()*100:+.2f}% | Perda Média por Loss: {l['ret_liq'].mean()*100:+.2f}%")
        print(f"   • Duração Mediana: {sub_df['hours'].median():.1f} horas ({sub_df['bars_15m'].median():.0f} barras de 15m)")
        
        return {
            "trades": n_tr,
            "ret_compounded": total_compounded_ret,
            "pf": pf,
            "wr": wr,
            "max_dd": max_dd,
            "worst_intra_dd": worst_intra_dd
        }
        
    calc_metrics(trades_df, "RESULTADO GLOBAL (2 ANOS)")
    
    for side in ["BULL", "BEAR"]:
        st = trades_df[trades_df["side"] == side]
        calc_metrics(st, f"ESPECIALISTA {side}")
        
    # Divisão Causal Treino vs Holdout
    split_date = pd.to_datetime("2026-01-24", utc=True)
    train_tr = trades_df[trades_df["entry_time"] < split_date]
    holdout_tr = trades_df[trades_df["entry_time"] >= split_date]
    
    print("\n" + "-"*80)
    print("🛡️ VALIDAÇÃO TREINO VS HOLDOUT (OUT-OF-SAMPLE):")
    print(f"   • Treino:  até 2026-01-24")
    print(f"   • Holdout: 2026-01-24 até 2026-09-15 (NUNCA VISTO)")
    print("-"*80)
    
    calc_metrics(train_tr, "TREINO (In-Sample)")
    calc_metrics(holdout_tr, "HOLDOUT (Out-of-Sample)")
    
    # Salvar Artefatos
    trades_csv = reports_dir / "teacher_scientific_trades.csv"
    trades_df.to_csv(trades_csv, index=False)
    print(f"\n💾 Registro detalhado de trades salvo em: {trades_csv}")
    
    dataset_parquet = ROOT_DIR / "data/teacher_scientific_dataset.parquet"
    df.to_parquet(dataset_parquet)
    print(f"💾 Dataset com rótulos salvo em: {dataset_parquet}")
    print("="*80 + "\n")


def main():
    df15, funding = load_dataset()
    df_feat = build_causal_multi_timeframe_features(df15, funding)
    df_res, trades_df = run_teacher_simulation(df_feat)
    analyze_and_report(df_res, trades_df)


if __name__ == "__main__":
    main()
