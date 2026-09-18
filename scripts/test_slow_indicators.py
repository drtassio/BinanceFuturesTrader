"""Evaluate slower 1h and 4h indicators alongside 15m and 5m for Bull and Bear.

Tests:
1. Baseline leg_confirm (15m staircase + 5m acceleration + CVD + 4h structure exit)
2. + 1h EMA trend (ema_trend_1h > 0 for Bull, < 0 for Bear)
3. + 4h Trend (cz_trend_4h > 0 for Bull, < 0 for Bear)
4. + 4h EMA trend (ema_trend_4h > 0 for Bull, < 0 for Bear)
5. + 1h MACD momentum (macd_hist_1h > 0 for Bull, < 0 for Bear)
6. + 4h MACD momentum (macd_hist_4h > 0 for Bull, < 0 for Bear)
7. + Confluence (1h trend AND 4h trend agreeing)
8. + ADX filter (adx_1h > 20 or adx_4h > 20: requiring active trending market)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloud.train_agent import load_dataset, split_chronological
from learning.edge_policy import LegConfirmRule, SIDES


def evaluate_variant(df: pd.DataFrame, agent: str, rule: LegConfirmRule, extra_filter=None) -> dict:
    """Run simulation with optional extra 1h/4h filter function."""
    side = SIDES[agent][0]
    
    # Pre-extract series for fast vectorized row lookup
    steps_col = "cz_leg_steps_up" if side > 0 else "cz_leg_steps_down"
    breakout_col = "cz_breakout_up_48" if side > 0 else "cz_breakout_down_48"
    steps5_col = "cz_steps5_up" if side > 0 else "cz_steps5_down"
    struct_col = "cz_struct_4h_long" if side > 0 else "cz_struct_4h_short"
    
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    steps = df[steps_col].to_numpy()
    breakout = df[breakout_col].to_numpy()
    body = df["cz_step_body_atr"].to_numpy()
    cvd = df["cz_cvd_z_16"].to_numpy()
    steps5 = df[steps5_col].to_numpy()
    struct = df[struct_col].to_numpy()
    trend_4h = df["cz_trend_4h"].to_numpy() if "cz_trend_4h" in df else np.zeros(len(df))
    
    # Custom filter mask
    if extra_filter is not None:
        custom_mask = extra_filter(df, side).to_numpy()
    else:
        custom_mask = np.ones(len(df), dtype=bool)
        
    cost_per_side = 0.0005
    n = len(df)
    
    position = 0
    entry_price = 0.0
    trades = []
    equity = 1.0
    equity_curve = [1.0]
    entry_bar = 0
    
    for i in range(1, n):
        curr_price = close[i]
        
        if position == side:
            # Check exit: 4h structure break
            if struct[i] < 0.0:
                # Close trade
                pnl_pct = side * (curr_price - entry_price) / entry_price - 2 * cost_per_side
                equity *= (1.0 + pnl_pct * rule.leverage)
                trades.append({
                    "pnl_pct": pnl_pct,
                    "leveraged_pnl": pnl_pct * rule.leverage,
                    "bars": i - entry_bar,
                })
                position = 0
                entry_price = 0.0
        elif position == 0:
            # Check entry
            confirmed = (
                steps[i] >= rule.min_steps
                and side * breakout[i] > 0.0
                and side * body[i] > 0.0
                and side * cvd[i] > 0.0
                and steps5[i] >= rule.min_steps_5m
                and custom_mask[i]
            )
            if rule.require_trend_4h:
                confirmed = confirmed and side * trend_4h[i] > 0.0
                
            if confirmed:
                position = side
                entry_price = curr_price
                entry_bar = i
                
        equity_curve.append(equity)
        
    # Close open trade at end
    if position != 0:
        pnl_pct = side * (close[-1] - entry_price) / entry_price - 2 * cost_per_side
        equity *= (1.0 + pnl_pct * rule.leverage)
        trades.append({
            "pnl_pct": pnl_pct,
            "leveraged_pnl": pnl_pct * rule.leverage,
            "bars": n - 1 - entry_bar,
        })
        
    num_trades = len(trades)
    if num_trades == 0:
        return {
            "trades": 0, "net_return": 0.0, "profit_factor": 0.0,
            "win_rate": 0.0, "max_drawdown": 0.0, "avg_bars": 0.0
        }
        
    wins = [t["leveraged_pnl"] for t in trades if t["leveraged_pnl"] > 0]
    losses = [abs(t["leveraged_pnl"]) for t in trades if t["leveraged_pnl"] <= 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)
    win_rate = len(wins) / num_trades * 100.0
    net_return = equity - 1.0
    
    # Max drawdown
    eq_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(eq_arr)
    dd = (peak - eq_arr) / np.maximum(peak, 1e-9)
    max_dd = float(np.max(dd))
    avg_bars = float(np.mean([t["bars"] for t in trades]))
    
    return {
        "trades": num_trades,
        "net_return": net_return,
        "profit_factor": pf,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "avg_bars": avg_bars
    }


def main():
    data_path = ROOT / "data" / "featured_data_causal_leg5m.parquet"
    print("Carregando dados de %s..." % data_path)
    df = load_dataset(data_path)
    train_df, val_df, holdout_df = split_chronological(df)
    
    print("Dataset carregado: Treino=%d, Val=%d, Holdout=%d barras" % (len(train_df), len(val_df), len(holdout_df)))
    
    # Filter definitions
    variants = [
        ("1. Base (15m+5m+CVD, sem 4h trend)",
         LegConfirmRule(min_steps=3, min_steps_5m=2, require_trend_4h=False, sl_mult=1.5, leverage=3.0),
         None),
        ("2. Base + cz_trend_4h",
         LegConfirmRule(min_steps=3, min_steps_5m=2, require_trend_4h=True, sl_mult=1.5, leverage=3.0),
         None),
        ("3. Base + ema_trend_1h",
         LegConfirmRule(min_steps=3, min_steps_5m=2, require_trend_4h=False, sl_mult=1.5, leverage=3.0),
         lambda d, s: (d["ema_trend_1h"] > 0) if s > 0 else (d["ema_trend_1h"] < 0)),
        ("4. Base + ema_trend_4h",
         LegConfirmRule(min_steps=3, min_steps_5m=2, require_trend_4h=False, sl_mult=1.5, leverage=3.0),
         lambda d, s: (d["ema_trend_4h"] > 0) if s > 0 else (d["ema_trend_4h"] < 0)),
        ("5. Base + cz_trend_4h + ema_trend_1h",
         LegConfirmRule(min_steps=3, min_steps_5m=2, require_trend_4h=True, sl_mult=1.5, leverage=3.0),
         lambda d, s: (d["ema_trend_1h"] > 0) if s > 0 else (d["ema_trend_1h"] < 0)),
        ("6. Base + 1h Momentum (macd_hist_1h)",
         LegConfirmRule(min_steps=3, min_steps_5m=2, require_trend_4h=False, sl_mult=1.5, leverage=3.0),
         lambda d, s: (d["macd_hist_1h"] > 0) if s > 0 else (d["macd_hist_1h"] < 0)),
        ("7. Base + 4h Momentum (macd_hist_4h)",
         LegConfirmRule(min_steps=3, min_steps_5m=2, require_trend_4h=False, sl_mult=1.5, leverage=3.0),
         lambda d, s: (d["macd_hist_4h"] > 0) if s > 0 else (d["macd_hist_4h"] < 0)),
        ("8. Confluência Total (cz_trend_4h + ema_trend_1h + macd_hist_1h)",
         LegConfirmRule(min_steps=3, min_steps_5m=2, require_trend_4h=True, sl_mult=1.5, leverage=3.0),
         lambda d, s: ((d["ema_trend_1h"] > 0) & (d["macd_hist_1h"] > 0)) if s > 0 else ((d["ema_trend_1h"] < 0) & (d["macd_hist_1h"] < 0))),
        ("9. Confluência + ADX_1h > 20 (tendência ativa)",
         LegConfirmRule(min_steps=3, min_steps_5m=2, require_trend_4h=True, sl_mult=1.5, leverage=3.0),
         lambda d, s: (((d["ema_trend_1h"] > 0) & (d["macd_hist_1h"] > 0) & (d["adx_1h"] > 20)) if s > 0
                       else ((d["ema_trend_1h"] < 0) & (d["macd_hist_1h"] < 0) & (d["adx_1h"] > 20)))),
    ]

    for agent in ["bull", "bear"]:
        print("\n" + "=" * 95)
        print("RESULTADOS EXPERIMENTAIS: AGENTE %s" % agent.upper())
        print("=" * 95)
        header = f"{'Variante':<45} | {'TRAIN (Ret/PF/Wr/Tr)':<22} | {'VAL (Ret/PF/Wr/Tr)':<22} | {'HOLDOUT (Ret/PF/Wr/Tr)':<22}"
        print(header)
        print("-" * 95)
        
        for name, rule, extra_filter in variants:
            res_tr = evaluate_variant(train_df, agent, rule, extra_filter)
            res_val = evaluate_variant(val_df, agent, rule, extra_filter)
            res_ho = evaluate_variant(holdout_df, agent, rule, extra_filter)
            
            str_tr = f"{res_tr['net_return']*100:+5.1f}%/{res_tr['profit_factor']:4.2f}/{res_tr['win_rate']:4.0f}%/{res_tr['trades']:2d}"
            str_val = f"{res_val['net_return']*100:+5.1f}%/{res_val['profit_factor']:4.2f}/{res_val['win_rate']:4.0f}%/{res_val['trades']:2d}"
            str_ho = f"{res_ho['net_return']*100:+5.1f}%/{res_ho['profit_factor']:4.2f}/{res_ho['win_rate']:4.0f}%/{res_ho['trades']:2d}"
            
            print(f"{name:<45} | {str_tr:<22} | {str_val:<22} | {str_ho:<22}")

if __name__ == "__main__":
    main()
