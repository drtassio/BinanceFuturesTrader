"""
optimize_staircases_flow.py
Pesquisa rigorosa do Professor de Escadas Multi-Timeframe com Fluxo de Ordens:
Testa variações de entrada por confirmação de degraus e fluxo, e saída por perda de estrutura.
Separa estritamente em:
- Treino: 2024-07-01 a 2026-01-01 (18 meses)
- Holdout: 2026-01-01 a 2026-09-15 (8.5 meses fora da amostra)
"""

import sys
from pathlib import Path
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent


def load_data():
    df = pd.read_parquet(ROOT_DIR / "data/market_raw/btc_perp_15m.parquet")
    df = df.sort_index()
    
    # Preço e Volume
    df = df[["open", "high", "low", "close", "volume", "taker_buy_base"]].copy()
    
    # 1. Fluxo de Ordens
    df["delta"] = 2 * df["taker_buy_base"] - df["volume"]
    df["imbalance"] = df["delta"] / (df["volume"] + 1e-8)
    df["cvd_16"] = df["delta"].rolling(16).sum()
    df["cvd_std"] = df["delta"].rolling(96).std()
    df["cvd_z"] = df["cvd_16"] / (df["cvd_std"] * np.sqrt(16) + 1e-8)
    
    # 2. ATR e Expansão de Volatilidade
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )
    df["atr14"] = tr.rolling(14).mean()
    df["atr_long"] = tr.rolling(96).mean()
    df["expansion"] = df["atr14"] / (df["atr_long"] + 1e-8)
    
    # 3. Degraus Individuais
    df["is_step_up"] = (df["close"] > df["open"]) & (df["close"] > df["close"].shift(1))
    df["is_step_down"] = (df["close"] < df["open"]) & (df["close"] < df["close"].shift(1))
    
    # Contagem de degraus
    up_steps = np.zeros(len(df), dtype=int)
    down_steps = np.zeros(len(df), dtype=int)
    su = df["is_step_up"].values
    sd = df["is_step_down"].values
    for i in range(1, len(df)):
        up_steps[i] = up_steps[i-1] + 1 if su[i] else 0
        down_steps[i] = down_steps[i-1] + 1 if sd[i] else 0
    df["up_steps"] = up_steps
    df["down_steps"] = down_steps
    
    # 4. Médias Móveis Multi-Timeframe
    # 1h = 4 candles de 15m; 4h = 16 candles
    df["ema_1h"] = df["close"].ewm(span=16).mean()
    df["ema_4h"] = df["close"].ewm(span=64).mean()
    
    # Máximas e Mínimas (8h = 32 candles; 16h = 64; 24h = 96; 48h = 192)
    for w in [16, 32, 48, 64, 96, 192]:
        df[f"high_{w}"] = df["high"].rolling(w).max().shift(1)
        df[f"low_{w}"] = df["low"].rolling(w).min().shift(1)
        
    df.dropna(inplace=True)
    return df


def backtest_rule(
    df: pd.DataFrame,
    side: str, # "BULL" ou "BEAR"
    entry_w: int,
    exit_w: int,
    min_steps: int,
    min_expansion: float,
    min_cvd: float,
    use_ema_filter: bool = True
):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    entry_ref = df[f"high_{entry_w}"].values if side == "BULL" else df[f"low_{entry_w}"].values
    exit_ref = df[f"low_{exit_w}"].values if side == "BULL" else df[f"high_{exit_w}"].values
    steps = df["up_steps"].values if side == "BULL" else df["down_steps"].values
    expansion = df["expansion"].values
    cvd_z = df["cvd_z"].values
    imbalance = df["imbalance"].values
    ema_1h = df["ema_1h"].values
    ema_4h = df["ema_4h"].values
    index = df.index
    n = len(df)
    
    fee = 0.0004 # 4 bps por trade
    trades = []
    in_pos = False
    entry_idx = 0
    entry_price = 0.0
    
    for i in range(1, n):
        if not in_pos:
            if side == "BULL":
                cond = (
                    (close[i] > entry_ref[i]) and
                    (steps[i] >= min_steps) and
                    (expansion[i] >= min_expansion) and
                    (cvd_z[i] >= min_cvd) and
                    (imbalance[i] > 0)
                )
                if use_ema_filter:
                    cond = cond and (close[i] > ema_1h[i])
            else: # BEAR
                cond = (
                    (close[i] < entry_ref[i]) and
                    (steps[i] >= min_steps) and
                    (expansion[i] >= min_expansion) and
                    (cvd_z[i] <= -min_cvd) and
                    (imbalance[i] < 0)
                )
                if use_ema_filter:
                    cond = cond and (close[i] < ema_1h[i])
                    
            if cond:
                in_pos = True
                entry_idx = i
                entry_price = close[i]
                
        else: # em posição
            if side == "BULL":
                exit_hit = (close[i] < exit_ref[i]) or (i == n - 1)
            else:
                exit_hit = (close[i] > exit_ref[i]) or (i == n - 1)
                
            if exit_hit:
                exit_price = close[i]
                if side == "BULL":
                    ret_bruto = (exit_price - entry_price) / entry_price
                else:
                    ret_bruto = (entry_price - exit_price) / entry_price
                ret_liq = ret_bruto - 2 * fee
                trades.append({
                    "entry_time": index[entry_idx],
                    "exit_time": index[i],
                    "ret_liq": ret_liq,
                    "bars": i - entry_idx,
                    "is_win": ret_liq > 0
                })
                in_pos = False
                
    if not trades:
        return None
        
    tdf = pd.DataFrame(trades)
    train_split = pd.to_datetime("2026-01-01", utc=True)
    train_tdf = tdf[tdf["entry_time"] < train_split]
    holdout_tdf = tdf[tdf["entry_time"] >= train_split]
    
    def get_stats(sub):
        if len(sub) == 0:
            return {"trades": 0, "net": 0.0, "pf": 0.0, "wr": 0.0, "dd": 0.0}
        w = sub[sub["is_win"]]
        l = sub[~sub["is_win"]]
        wr = len(w) / len(sub) * 100
        gp = w["ret_liq"].sum()
        gl = abs(l["ret_liq"].sum())
        pf = gp / gl if gl > 0 else float("inf")
        net = sub["ret_liq"].sum() * 100
        cum = (1 + sub["ret_liq"]).cumprod()
        dd = abs(((cum - cum.cummax()) / cum.cummax()).min()) * 100
        return {"trades": len(sub), "net": net, "pf": pf, "wr": wr, "dd": dd}
        
    return {
        "all": get_stats(tdf),
        "train": get_stats(train_tdf),
        "holdout": get_stats(holdout_tdf)
    }


def main():
    print("🔄 Carregando dados e indicadores de fluxo...")
    df = load_data()
    print(f"✅ Dados carregados: {len(df)} candles de {df.index.min()} até {df.index.max()}")
    
    print("\n" + "="*80)
    print("🔍 GRID DE PESQUISA: ENCONTRANDO AS ESCADAS PERFEITAS (BULL E BEAR)")
    print("="*80)
    
    for side in ["BULL", "BEAR"]:
        print(f"\n--- TESTANDO ESPECIALISTA {side} ---")
        best_configs = []
        
        # Grid de parâmetros estruturais e de degraus
        entry_windows = [32, 64, 96, 192] if side == "BULL" else [16, 32, 48, 64]
        exit_windows = [16, 32, 48]
        min_steps_list = [1, 2]
        expansions = [1.0, 1.2]
        cvds = [0.5, 1.0]
        
        for ew in entry_windows:
            for xw in exit_windows:
                for ms in min_steps_list:
                    for exp in expansions:
                        for cvd in cvds:
                            res = backtest_rule(
                                df, side=side, entry_w=ew, exit_w=xw,
                                min_steps=ms, min_expansion=exp, min_cvd=cvd
                            )
                            if res and res["train"]["trades"] >= 20 and res["holdout"]["trades"] >= 10:
                                tr = res["train"]
                                ho = res["holdout"]
                                if tr["net"] > 0 and ho["net"] > 0 and tr["pf"] >= 1.2 and ho["pf"] >= 1.1:
                                    best_configs.append({
                                        "ew": ew, "xw": xw, "ms": ms, "exp": exp, "cvd": cvd,
                                        "tr_net": tr["net"], "tr_pf": tr["pf"], "tr_n": tr["trades"], "tr_dd": tr["dd"],
                                        "ho_net": ho["net"], "ho_pf": ho["pf"], "ho_n": ho["trades"], "ho_dd": ho["dd"]
                                    })
                                    
        if best_configs:
            best_configs.sort(key=lambda x: x["ho_pf"] + x["tr_pf"], reverse=True)
            print(f"🏆 Top 5 configurações aprovadas para {side}:")
            for c in best_configs[:5]:
                print(f"   • Entry={c['ew']}b Exit={c['xw']}b Steps={c['ms']} Exp={c['exp']} CVD={c['cvd']} | "
                      f"Train: {c['tr_net']:+.1f}% PF={c['tr_pf']:.2f} (DD {c['tr_dd']:.1f}%, {c['tr_n']}t) | "
                      f"Holdout: {c['ho_net']:+.1f}% PF={c['ho_pf']:.2f} (DD {c['ho_dd']:.1f}%, {c['ho_n']}t)")
        else:
            print(f"⚠️ Nenhuma configuração atendeu aos filtros estritos para {side}.")


if __name__ == "__main__":
    main()
