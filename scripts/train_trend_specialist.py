import asyncio
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Adiciona o diretório raiz ao path para importações
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config.settings import AIConfig, TradingConfig
from feature_engineering.main import FeatureEngineeringPipeline
from specialists.trend_specialist import TrendSpecialist, TrendFollowingEnv
from trading.utils.trend_backtest import (
	check_data_freshness,
	ensure_datetime_index,
	split_train_holdout,
	build_unseen_eval_frame,
	build_recent_eval_frame,
	run_manual_backtest,
	DEFAULT_EVAL_MONTHS,
	DEFAULT_EVAL_MIN_ROWS,
)

HIST_DIR = Path('logs/historical_data')
BASE_FEATURES = Path('models_ai/base_featured_df.pkl')
CHECKPOINT_DIR = Path('checkpoints_ai/trend_specialist_checkpoints')
EVAL_LOG_DIR = Path('logs/trade_analysis/SAC')


def load_or_build_features(force_rebuild: bool = False) -> pd.DataFrame:
	ai_cfg = AIConfig()
	fe = FeatureEngineeringPipeline(ai_cfg)
	tr_cfg = TradingConfig()
	symbol = tr_cfg.PRIMARY_PAIR
	primary_tf = tr_cfg.PRIMARY_TIMEFRAME_TRADING

	if BASE_FEATURES.exists() and not force_rebuild:
		return pd.read_pickle(BASE_FEATURES)

	timeframes = ['1m', '5m', '15m', '1h', '4h']
	raw = {}
	for tf in timeframes:
		p = HIST_DIR / f"{symbol}_{tf}_historical.pkl"
		if not p.exists():
			continue
		df = pd.read_pickle(p)
		if not isinstance(df.index, pd.DatetimeIndex):
			df.index = pd.to_datetime(df.index, utc=True)
		raw[tf] = df.sort_index()
	if primary_tf not in raw:
		raise RuntimeError(f"Missing primary timeframe {primary_tf} in historical data")

	featured = asyncio.run(fe.create_features(raw, symbol, primary_tf, fit_scaler=False))
	if featured is None or featured.empty:
		raise RuntimeError('Failed to generate features')
	BASE_FEATURES.parent.mkdir(parents=True, exist_ok=True)
	featured.to_pickle(BASE_FEATURES)
	return featured


def read_financial_metrics() -> dict:
    if not CHECKPOINT_DIR.exists():
        return {}
    candidates = sorted(CHECKPOINT_DIR.glob('obs_*/financial_eval_metrics.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            if isinstance(data, list) and data:
                # pega o último registro válido
                for item in reversed(data):
                    if isinstance(item, dict):
                        return item
        except Exception:
            continue
    return {}


def build_recent_eval_frame(featured: pd.DataFrame, months: int, min_rows: int) -> pd.DataFrame:
	return tb_build_recent_eval_frame(featured, months, min_rows)


def build_unseen_eval_frame(featured: pd.DataFrame, train_cutoff_date: str = None, eval_months: int = 3) -> pd.DataFrame:
	return tb_build_unseen_eval_frame(featured, train_cutoff_date=train_cutoff_date, eval_months=eval_months)


def run_manual_backtest(agent: TrendSpecialist, eval_df: pd.DataFrame, log_dir: Path, label: str = "manual_backtest") -> Dict[str, Any]:
	if eval_df is None or eval_df.empty:
		print("[BACKTEST] DataFrame de avaliação está vazio. Backtest manual não executado.")
		return {}

	log_dir = Path(log_dir)
	log_dir.mkdir(parents=True, exist_ok=True)

	base_env = getattr(agent, '_env', None)
	model = getattr(base_env, 'model', None)
	if model is None:
		try:
			agent.load_model()
			base_env = getattr(agent, '_env', None)
			model = getattr(base_env, 'model', None)
		except Exception as exc:
			print(f"[BACKTEST] Não foi possível carregar o modelo SAC salvo: {exc}")
			return {}

	if model is None:
		print("[BACKTEST] Modelo SAC indisponível após o treinamento. Backtest manual não executado.")
		return {}

	agent_cfg = getattr(agent, 'config', None) or getattr(agent, 'config_ai', None)
	if agent_cfg is None:
		print("[BACKTEST] Configuração do agente não encontrada. Backtest manual não executado.")
		return {}

	try:
		eval_env = TrendFollowingEnv(
			df=eval_df,
			config=agent_cfg,
			profitability_predictor=getattr(agent, 'profitability_predictor', None),
			mode='evaluation'
		)
	except TypeError:
		# Assinatura alternativa (fallback para versões antigas)
		eval_env = TrendFollowingEnv(eval_df, agent_cfg)

	obs, _ = eval_env.reset()
	cumulative_reward = 0.0
	trades: List[Dict[str, Any]] = []
	nav_records: List[Dict[str, Any]] = []

	def _fmt_ts(idx: int) -> str:
		if idx < 0 or idx >= len(eval_env.df.index):
			idx = len(eval_env.df.index) - 1
		ts = eval_env.df.index[idx]
		if isinstance(ts, pd.Timestamp):
			if ts.tzinfo is None:
				try:
					ts = ts.tz_localize('UTC')
				except (TypeError, ValueError):
					pass
			return ts.isoformat()
		if isinstance(ts, datetime):
			return ts.isoformat()
		# fallback for non-temporal indexes
		return str(ts)

	nav_records.append({
		"step": 0,
		"timestamp": _fmt_ts(0),
		"net_worth": float(eval_env.net_worth),
		"reward": 0.0,
	})

	done = False
	step_idx = 0
	safety_cap = len(eval_env.df) * 2

	while not done and step_idx < safety_cap:
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = eval_env.step(action)
		cumulative_reward += float(reward)
		done = bool(terminated or truncated)
		step_idx += 1

		current_step = min(eval_env.current_step, len(eval_env.df) - 1)
		nav_records.append({
			"step": step_idx,
			"timestamp": _fmt_ts(current_step),
			"net_worth": float(eval_env.net_worth),
			"reward": float(reward),
		})

		trade_info = info.get("trade")
		if trade_info:
			trade_entry = dict(trade_info)
			trade_entry["step"] = step_idx
			trade_entry["timestamp"] = _fmt_ts(current_step)
			trades.append(trade_entry)

		if eval_env.current_step >= eval_env.max_steps:
			done = True

	net_array = np.asarray(eval_env.net_worth_history, dtype=np.float64)
	if net_array.size:
		running_max = np.maximum.accumulate(net_array)
		drawdowns = (running_max - net_array) / np.maximum(running_max, 1e-9)
		max_drawdown_pct = float(drawdowns.max() * 100.0)
	else:
		max_drawdown_pct = 0.0

	trades_df = pd.DataFrame(trades)
	total_trades = int(len(trades_df))
	profitable = trades_df[trades_df["pnl"] > 0] if total_trades else pd.DataFrame()
	losing = trades_df[trades_df["pnl"] <= 0] if total_trades else pd.DataFrame()

	win_rate_pct = float((len(profitable) / total_trades * 100.0)) if total_trades else 0.0
	avg_profit = float(profitable["pnl"].mean()) if not profitable.empty else 0.0
	avg_loss = float(losing["pnl"].mean()) if not losing.empty else 0.0
	total_profit = float(profitable["pnl"].sum()) if not profitable.empty else 0.0
	total_loss_abs = float(abs(losing["pnl"].sum())) if not losing.empty else 0.0
	profit_factor = float(total_profit / total_loss_abs) if total_loss_abs > 0 else float('inf') if total_profit > 0 else 0.0
	avg_duration = float(trades_df["duration_steps"].mean()) if total_trades and "duration_steps" in trades_df.columns else 0.0

	# Matriz de confusão: direção prevista (lado do trade) vs direção real (variação de preço)
	confusion = {
		"TP": 0,
		"TN": 0,
		"FP": 0,
		"FN": 0,
		"accuracy": 0.0,
		"precision": 0.0,
		"recall": 0.0,
		"f1": 0.0,
	}
	if total_trades and {"side", "entry_price", "exit_price"}.issubset(trades_df.columns):
		pred_up = (trades_df["side"].astype(str).str.lower() == "long").astype(int)
		actual_up = (trades_df["exit_price"] > trades_df["entry_price"]).astype(int)
		# TP: previu long e preço subiu | TN: previu short e preço caiu
		TP = int(((pred_up == 1) & (actual_up == 1)).sum())
		TN = int(((pred_up == 0) & (actual_up == 0)).sum())
		FP = int(((pred_up == 1) & (actual_up == 0)).sum())
		FN = int(((pred_up == 0) & (actual_up == 1)).sum())
		total_cm = max(TP + TN + FP + FN, 1)
		accuracy = (TP + TN) / total_cm
		precision = TP / max(TP + FP, 1)
		recall = TP / max(TP + FN, 1)
		f1 = (2 * precision * recall / max(precision + recall, 1e-9)) if (precision + recall) > 0 else 0.0
		confusion.update({
			"TP": TP,
			"TN": TN,
			"FP": FP,
			"FN": FN,
			"accuracy": float(accuracy),
			"precision": float(precision),
			"recall": float(recall),
			"f1": float(f1),
		})

	final_net_worth = float(eval_env.net_worth)
	initial_capital = float(eval_env.initial_balance)
	net_profit = final_net_worth - initial_capital
	net_return_pct = (net_profit / initial_capital * 100.0) if initial_capital else 0.0

	ts_label = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
	nav_path = log_dir / f"{label}_nav_{ts_label}.csv"
	trades_path = log_dir / f"{label}_trades_{ts_label}.csv"
	summary_path = log_dir / f"{label}_summary_{ts_label}.json"

	nav_df = pd.DataFrame(nav_records)
	nav_df.to_csv(nav_path, index=False)
	if total_trades:
		trades_df.to_csv(trades_path, index=False)
	else:
		trades_path = None

	summary: Dict[str, Any] = {
		"window_start": str(eval_df.index.min()),
		"window_end": str(eval_df.index.max()),
		"rows_evaluated": int(len(eval_df)),
		"final_net_worth": final_net_worth,
		"net_profit": float(net_profit),
		"net_return_pct": float(net_return_pct),
		"total_reward": float(cumulative_reward),
		"total_trades": total_trades,
		"win_rate_pct": float(win_rate_pct),
		"profit_factor": float(profit_factor),
		"avg_profit": float(avg_profit),
		"avg_loss": float(avg_loss),
		"avg_trade_duration_steps": float(avg_duration),
		"max_drawdown_pct": float(max_drawdown_pct),
		"confusion_matrix": confusion,
		"nav_path": str(nav_path),
		"trades_path": str(trades_path) if trades_path else "",
		"log_dir": str(log_dir),
	}

	with open(summary_path, 'w', encoding='utf-8') as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)
	summary["summary_path"] = str(summary_path)

	print(f"[BACKTEST] Execução concluída. NetWorth final: {final_net_worth:.2f} | Lucro: {net_profit:.2f} ({net_return_pct:.2f}%) | Trades: {total_trades}")
	if total_trades and confusion["TP"] + confusion["TN"] + confusion["FP"] + confusion["FN"] > 0:
		print("[BACKTEST] Matriz de Confusão (direção prevista vs real):")
		print(f"         Actual Up   Actual Down")
		print(f"Pred Up     {confusion['TP']:>6}        {confusion['FP']:>6}")
		print(f"Pred Down   {confusion['FN']:>6}        {confusion['TN']:>6}")
		print(f"         Acc={confusion['accuracy']:.3f} | Prec={confusion['precision']:.3f} | Recall={confusion['recall']:.3f} | F1={confusion['f1']:.3f}")

	return summary


def main():
    ai_cfg = AIConfig()
    os.environ.setdefault('TREND_FORCE_OPTIMIZE', '1')

    featured = load_or_build_features()
    featured = ensure_datetime_index(featured)

    is_fresh, latest_ts, staleness_hours = check_data_freshness(featured)
    if not is_fresh:
        print(f"[DATA CHECK] ⚠️ Dados desatualizados ({staleness_hours:.1f}h desde o último candle em {latest_ts}). Reconstruindo features...")
        featured = load_or_build_features(force_rebuild=True)
        featured = ensure_datetime_index(featured)
        is_fresh, latest_ts, staleness_hours = check_data_freshness(featured)
        if not is_fresh:
            print(f"[DATA CHECK] ⚠️ Após reconstrução, os dados ainda estão defasados ({staleness_hours:.1f}h). Verifique a atualização dos históricos antes de prosseguir.")
        else:
            print(f"[DATA CHECK] ✅ Features atualizadas até {latest_ts}.")
    else:
        print(f"[DATA CHECK] ✅ Dados atualizados até {latest_ts}.")

    train_frame, holdout_frame = split_train_holdout(featured, holdout_months=2, min_holdout_rows=DEFAULT_EVAL_MIN_ROWS)
    if not holdout_frame.empty:
        print(f"[HOLDOUT] Janela de 2 meses reservada para validação final: {holdout_frame.index.min()} -> {holdout_frame.index.max()} ({len(holdout_frame)} linhas)")
        print(f"[HOLDOUT] Dados disponíveis para treino: {train_frame.index.min()} -> {train_frame.index.max()} ({len(train_frame)} linhas)")
        training_source = train_frame
    else:
        print("[HOLDOUT] ⚠️ Não foi possível reservar dois meses para holdout. Usando dataset completo e fallback 80/20 para avaliação.")
        training_source = featured
    numeric_cols = featured.select_dtypes(include='number').columns
    input_dim = len(numeric_cols)
    agent = TrendSpecialist(config=ai_cfg, input_dim=input_dim)

    try:
        target_total_timesteps = int(os.getenv('TREND_TARGET_TIMESTEPS', str(getattr(ai_cfg, 'TOTAL_TRAINING_TIMESTEPS', 300_000))))
    except ValueError:
        target_total_timesteps = getattr(ai_cfg, 'TOTAL_TRAINING_TIMESTEPS', 300_000)
    target_total_timesteps = max(target_total_timesteps, 1)

    try:
        base_steps = int(os.getenv('TREND_STEPS_PER_CYCLE', '15000'))
    except ValueError:
        base_steps = 15000
    try:
        step_increment = int(os.getenv('TREND_STEP_INCREMENT', '5000'))
    except ValueError:
        step_increment = 5000
    base_steps = max(base_steps, 1)
    step_increment = max(step_increment, 1)

    agent.hyperparams.update({
        "learning_rate": 1e-4,
        "buffer_size": 500_000,
        "batch_size": 256,
        "gamma": 0.999,
        "tau": 0.005,
        "train_freq": 1,
        "use_sde": True,
    })

    agent.config.CHECKPOINT_FREQ = max(1000, int(getattr(agent.config, 'CHECKPOINT_FREQ', 10000)))

    current_total = 0
    try:
        agent.load_model()
        env = getattr(agent, '_env', None)
        model_obj = getattr(env, 'model', None) if env else None
        if model_obj is not None:
            current_total = getattr(model_obj, 'num_timesteps', 0) or 0
    except Exception:
        current_total = 0

    print(f"[INFO] Timesteps atuais do modelo: {current_total} | Alvo final: {target_total_timesteps}")

    def _estimate_cycles(remaining: int) -> int:
        total = 0
        cycles = 0
        while total < remaining and cycles < 200:
            cycles += 1
            total += base_steps + (cycles - 1) * step_increment
        return cycles

    env_cycles_raw = os.getenv('TREND_MAX_CYCLES', '').strip()
    if env_cycles_raw:
        try:
            max_cycles = max(int(env_cycles_raw), 1)
        except ValueError:
            max_cycles = max(_estimate_cycles(max(target_total_timesteps - current_total, 0)) + 2, 10)
    else:
        max_cycles = max(_estimate_cycles(max(target_total_timesteps - current_total, 0)) + 2, 10)

    trading_cfg = TradingConfig()
    fallback_capital = getattr(ai_cfg, 'INITIAL_CAPITAL', 0.0)
    initial_capital = float(getattr(trading_cfg, 'INITIAL_CAPITAL', fallback_capital))
    best_net_profit = -float('inf')
    patience_counter = 0
    max_patience = 3
    cycle = 0

    while current_total < target_total_timesteps and cycle < max_cycles:
        cycle += 1
        planned_steps = base_steps + (cycle - 1) * step_increment
        remaining_for_target = max(target_total_timesteps - current_total, 0)
        steps_per_cycle = min(planned_steps, remaining_for_target if remaining_for_target > 0 else planned_steps)
        if steps_per_cycle <= 0:
            break

        print(f"[CYCLE {cycle}] Treinando {steps_per_cycle} timesteps (restante para alvo: {remaining_for_target})...")
        total_after_cycle = agent.train(training_source, steps_per_cycle)
        if total_after_cycle:
            current_total = total_after_cycle

        metrics = read_financial_metrics() or {}
        mean_final_net_worth = float(metrics.get('mean_final_net_worth', 0.0))
        net_profit = mean_final_net_worth - initial_capital if mean_final_net_worth > 0 else float(metrics.get('net_profit', -1.0))
        win_rate = float(metrics.get('win_rate_pct', 0.0))
        num_trades = int(metrics.get('num_trades', 0))
        profit_factor = float(metrics.get('profit_factor', 0.0))

        print(f"[CYCLE {cycle}] Total timesteps: {current_total} / {target_total_timesteps} | NetWorth: {mean_final_net_worth:.2f} | NetProfit: {net_profit:.2f} | WinRate: {win_rate:.1f}% | Trades: {num_trades} | PF: {profit_factor:.2f}")

        success_condition = net_profit > 0 and win_rate > 45.0 and num_trades > 10
        if success_condition and current_total >= target_total_timesteps:
            print('[SUCCESS] Criterio de lucratividade e alvo de timesteps alcancados.')
            break

        if net_profit > best_net_profit:
            best_net_profit = net_profit
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"[ADAPT] Sem melhoria por {patience_counter} ciclos. Ajustando hiperparametros...")
            agent.hyperparams["learning_rate"] = max(5e-6, agent.hyperparams["learning_rate"] * 0.7)
            agent.hyperparams["gamma"] = min(0.9995, agent.hyperparams["gamma"] + 0.001)
            patience_counter = 0

        if win_rate < 40.0:
            agent.hyperparams["learning_rate"] = max(1e-5, agent.hyperparams["learning_rate"] * 0.9)
        elif win_rate > 60.0:
            agent.hyperparams["learning_rate"] = min(5e-4, agent.hyperparams["learning_rate"] * 1.1)

    if current_total < target_total_timesteps:
        remaining = target_total_timesteps - current_total
        if remaining > 0:
            print(f"[FINAL PUSH] Executando treino adicional de {remaining} timesteps para atingir o alvo.")
            total_after_final = agent.train(training_source, remaining)
            if total_after_final:
                current_total = total_after_final

    print(f"[DONE] Treinamento concluido. Timesteps finais: {current_total}")

    # Backtest em dados não vistos (split temporal)
    print("\n" + "="*60)
    print("🔍 EXECUTANDO BACKTEST EM DADOS NÃO VISTOS")
    print("="*60)
    
    try:
        train_cutoff_date = os.getenv('TREND_TRAIN_CUTOFF_DATE', None)  # Ex: '2024-12-31'
        eval_months = int(os.getenv('TREND_EVAL_MONTHS', '3'))
    except ValueError:
        train_cutoff_date = None
        eval_months = 3

    eval_df = holdout_frame
    eval_label = "two_month_holdout"
    if eval_df.empty:
        print('[HOLDOUT] ⚠️ Holdout dedicado indisponível. Usando divisão temporal 80/20 como fallback.')
        eval_df = build_unseen_eval_frame(featured, train_cutoff_date=train_cutoff_date, eval_months=eval_months)
        eval_label = "unseen_backtest"
    if eval_df.empty:
        print('[BACKTEST] ⚠️ Janela de dados não vistos vazia. Tentando backtest com dados recentes...')
        try:
            eval_months_fallback = int(os.getenv('TREND_EVAL_MONTHS', str(DEFAULT_EVAL_MONTHS)))
        except ValueError:
            eval_months_fallback = DEFAULT_EVAL_MONTHS
        try:
            eval_min_rows = int(os.getenv('TREND_EVAL_MIN_ROWS', str(DEFAULT_EVAL_MIN_ROWS)))
        except ValueError:
            eval_min_rows = DEFAULT_EVAL_MIN_ROWS
        eval_df = build_recent_eval_frame(featured, months=eval_months_fallback, min_rows=eval_min_rows)
        eval_label = "recent_backtest"
    
    if eval_df.empty:
        print('[BACKTEST] ❌ Nenhuma janela de avaliação disponível. Backtest não executado.')
    else:
        print(f"[BACKTEST] Executando avaliação final '{eval_label}' ({eval_df.index.min()} -> {eval_df.index.max()}) com {len(eval_df)} linhas.")
        backtest_summary = run_manual_backtest(agent, eval_df, EVAL_LOG_DIR, label=eval_label)
        if backtest_summary:
            print(f"\n📊 RESULTADOS DO BACKTEST ({eval_label}):")
            print(f"   📅 Período: {backtest_summary['window_start']} -> {backtest_summary['window_end']}")
            print(f"   📈 Linhas analisadas: {backtest_summary['rows_evaluated']}")
            print(f"   💰 NetWorth Final: {backtest_summary['final_net_worth']:.2f}")
            print(f"   📊 Net Profit: {backtest_summary['net_profit']:.2f} ({backtest_summary['net_return_pct']:.2f}%)")
            print(f"   🎯 Win Rate: {backtest_summary['win_rate_pct']:.2f}%")
            print(f"   ⚡ Profit Factor: {backtest_summary['profit_factor']:.2f}")
            print(f"   📉 Max Drawdown: {backtest_summary['max_drawdown_pct']:.2f}%")
            print(f"   🔄 Total Trades: {backtest_summary['total_trades']}")
            print(f"\n📁 Arquivos gerados:")
            print(f"   📊 NAV: {backtest_summary['nav_path']}")
            if backtest_summary.get('trades_path'):
                print(f"   📋 Trades: {backtest_summary['trades_path']}")
            print(f"   📄 Summary: {backtest_summary.get('summary_path', '')}")
            
            # Avaliação de performance
            is_profitable = backtest_summary['net_profit'] > 0
            good_winrate = backtest_summary['win_rate_pct'] > 50.0
            good_pf = backtest_summary['profit_factor'] > 1.2
            acceptable_dd = backtest_summary['max_drawdown_pct'] < 20.0
            
            print(f"\n🎯 AVALIAÇÃO DE PERFORMANCE:")
            print(f"   {'✅' if is_profitable else '❌'} Lucrativo: {backtest_summary['net_profit']:.2f}")
            print(f"   {'✅' if good_winrate else '❌'} Win Rate: {backtest_summary['win_rate_pct']:.1f}%")
            print(f"   {'✅' if good_pf else '❌'} Profit Factor: {backtest_summary['profit_factor']:.2f}")
            print(f"   {'✅' if acceptable_dd else '❌'} Drawdown: {backtest_summary['max_drawdown_pct']:.1f}%")
            
            overall_score = sum([is_profitable, good_winrate, good_pf, acceptable_dd])
            print(f"\n🏆 SCORE GERAL: {overall_score}/4")
            if overall_score >= 3:
                print("🎉 EXCELENTE! Modelo pronto para trading real.")
            elif overall_score >= 2:
                print("👍 BOM! Modelo promissor, considere ajustes menores.")
            else:
                print("⚠️ ATENÇÃO! Modelo precisa de mais treinamento ou ajustes.")



if __name__ == '__main__':
	main()
