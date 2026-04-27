from __future__ import annotations

import os
import csv
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from utils.logger import get_logger


logger = get_logger("TradeAnalysisCallback")


class TradeAnalysisCallback(BaseCallback):
    """
    Coleta e analisa trades executados pelo agente durante a avaliação.

    - Compatível com DummyVecEnv (usa o primeiro ambiente da pilha)
    - Salva CSV/JSON de análise em logs/trade_analysis/<agent_name>/
    - Imprime métricas essenciais no final do treinamento
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 1000,
        n_eval_episodes: int = 5,
        verbose: int = 1,
        log_dir: str | None = None,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.trade_logs: List[Dict[str, Any]] = []
        self.log_dir = log_dir or os.path.join(os.getcwd(), "logs", "trade_analysis")
        self.pnl_series: List[float] = []
        self.action_stats: Dict[str, Dict[str, int]] = {}
        self.regime_stats: Dict[str, Dict[str, int]] = {}
        self.total_steps: int = 0

    # Mantemos _on_step para futura extensão; hoje a análise roda no final
    def _on_step(self) -> bool:  # noqa: D401
        # Poderíamos rodar avaliações parciais aqui (e.g. cada eval_freq)
        return True

    def _on_training_end(self) -> None:
        try:
            agent_name = self.model.__class__.__name__ if hasattr(self, "model") else "UnknownAgent"
            logger.info("=" * 60)
            logger.info(f"📊 ANÁLISE FINAL DE TRADES PARA O AGENTE: {agent_name}")
            logger.info("=" * 60)

            # 1) Avaliação padrão (recompensa média)
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
            logger.info(f"  - Recompensa Média Final: {mean_reward:.2f} +/- {std_reward:.2f}")

            # 2) Coleta de trades (simulação determinística)
            self._collect_trade_data_vec()

            # 3) Análise e logging
            summary = self._analyze_trades_and_log()

            # 4) Persistência
            self._persist_results(agent_name, summary)
            logger.info("=" * 60 + "\n")
        except Exception as e:
            logger.error(f"❌ [TRADE ANALYSIS] Falha ao gerar análise de trades: {e}", exc_info=True)

    # --- Coleta de trades (compatível com VecEnv) ---
    def _collect_trade_data_vec(self) -> None:
        self.trade_logs = []
        env = self.eval_env
        try:
            for _ in range(self.n_eval_episodes):
                obs = env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = env.step(action)

                    # VecEnv: usa o primeiro subambiente
                    info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
                    if isinstance(dones, (list, tuple, np.ndarray)):
                        done = bool(dones[0])
                    else:
                        done = bool(dones)

                    trade = info0.get("trade") if isinstance(info0, dict) else None
                    if trade:
                        self.trade_logs.append(trade)
        except Exception as e:
            logger.warning(f"⚠️ [TRADE ANALYSIS] Falha parcial na coleta de trades: {e}")

    # --- Análise e logging ---
    def _analyze_trades_and_log(self) -> Dict[str, Any]:
        if not self.trade_logs:
            logger.warning("Nenhum trade executado durante a avaliação. Sem análise.")
            return {"total_trades": 0}

        df = pd.DataFrame(self.trade_logs)
        total_trades = len(df)
        profitable = df[df["pnl"] > 0]
        losing = df[df["pnl"] <= 0]

        win_rate = float(len(profitable) / total_trades * 100.0) if total_trades else 0.0
        avg_profit = float(profitable["pnl"].mean()) if not profitable.empty else 0.0
        avg_loss = float(losing["pnl"].mean()) if not losing.empty else 0.0
        total_profit = float(profitable["pnl"].sum()) if not profitable.empty else 0.0
        total_loss_abs = float(abs(losing["pnl"].sum())) if not losing.empty else 0.0
        profit_factor = float(total_profit / total_loss_abs) if total_loss_abs > 0 else float("inf")

        # Duração
        avg_duration_all = float(df["duration_steps"].mean()) if "duration_steps" in df.columns else 0.0
        avg_duration_profit = float(profitable["duration_steps"].mean()) if "duration_steps" in profitable.columns and not profitable.empty else 0.0
        avg_duration_loss = float(losing["duration_steps"].mean()) if "duration_steps" in losing.columns and not losing.empty else 0.0

        most_profitable_trade = df.loc[df["pnl"].idxmax()].to_dict() if not df.empty else {}

        logger.info("\n  --- Resumo da Performance de Trading ---")
        logger.info(f"     - Total de Trades Executados: {total_trades}")
        logger.info(f"     - Taxa de Acerto (Win Rate): {win_rate:.2f}%")
        logger.info(f"     - Fator de Lucro (Profit Factor): {profit_factor:.2f}")
        logger.info(f"     - Média de Ganho (trades vencedores): {avg_profit:,.2f}")
        logger.info(f"     - Média de Perda (trades perdedores): {avg_loss:,.2f}")

        logger.info("\n  --- Duração dos Trades (em candles) ---")
        logger.info(f"     - Média (todos): {avg_duration_all:.1f}")
        logger.info(f"     - Média (vencedores): {avg_duration_profit:.1f}")
        logger.info(f"     - Média (perdedores): {avg_duration_loss:.1f}")

        summary = {
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "avg_duration_all": avg_duration_all,
            "avg_duration_profit": avg_duration_profit,
            "avg_duration_loss": avg_duration_loss,
            "most_profitable_trade": most_profitable_trade,
        }
        return summary

    # --- Persistência ---
    def _persist_results(self, agent_name: str, summary: Dict[str, Any]) -> None:
        try:
            ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            out_dir = os.path.join(self.log_dir, agent_name)
            os.makedirs(out_dir, exist_ok=True)

            # CSV de trades
            trades_csv = os.path.join(out_dir, f"trades_{ts}.csv")
            if self.trade_logs:
                fieldnames = sorted({k for t in self.trade_logs for k in t.keys()})
                with open(trades_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.trade_logs)

            # JSON de resumo
            summary_json = os.path.join(out_dir, f"summary_{ts}.json")
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ [TRADE ANALYSIS] Falha ao persistir resultados: {e}")



