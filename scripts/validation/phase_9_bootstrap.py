"""
FASE 9 — Bootstrap Backtest & N-Trade Adapt Trigger

Valida o pipeline pós-treino e o gatilho de adaptação por contagem de trades:

1. BacktestConfig expõe as 3 chaves novas (BOOTSTRAP_ENABLED, BOOTSTRAP_BACKTEST_DAYS,
   MIN_TRADES_FOR_ADAPT) com valores sensatos.
2. AIController tem o método _run_bootstrap_backtest com a assinatura correta.
3. AIController.train_missing_components chama _run_bootstrap_backtest após
   o treino dos especialistas (verificação por análise do código).
4. run_bot.system_state inicializa trades_since_last_adapt e total_closed_trades.
5. save_trade_to_log incrementa o contador SE status indicar trade fechado.
6. O loop de adaptação tem dois gatilhos independentes (tempo OR trades).
7. Após adapt bem-sucedido, trades_since_last_adapt é zerado.

Sem necessidade de torch/data/modelos reais — todos os checks são estruturais
(via inspect / análise de código) ou simulação isolada do contador.
"""
from __future__ import annotations
import inspect

from scripts.validation._common import PhaseRunner


def run() -> PhaseRunner:
    p = PhaseRunner("9 — Bootstrap Backtest & N-Trade Adapt")
    with p:
        from config.settings import BacktestConfig

        # 9.1 — Config keys presentes e razoáveis
        p.check(
            "BacktestConfig.BOOTSTRAP_ENABLED existe",
            hasattr(BacktestConfig, 'BOOTSTRAP_ENABLED'),
            detail=f"value={getattr(BacktestConfig, 'BOOTSTRAP_ENABLED', None)}",
        )
        p.check(
            "BacktestConfig.BOOTSTRAP_BACKTEST_DAYS no range [7, 365]",
            7 <= int(getattr(BacktestConfig, 'BOOTSTRAP_BACKTEST_DAYS', 0)) <= 365,
            detail=f"value={getattr(BacktestConfig, 'BOOTSTRAP_BACKTEST_DAYS', None)}",
        )
        p.check(
            "BacktestConfig.MIN_TRADES_FOR_ADAPT no range [5, 500]",
            5 <= int(getattr(BacktestConfig, 'MIN_TRADES_FOR_ADAPT', 0)) <= 500,
            detail=f"value={getattr(BacktestConfig, 'MIN_TRADES_FOR_ADAPT', None)}",
        )
        p.check(
            "BacktestConfig.BOOTSTRAP_OUTPUT_PATH definido",
            bool(getattr(BacktestConfig, 'BOOTSTRAP_OUTPUT_PATH', None)),
            detail=f"path={getattr(BacktestConfig, 'BOOTSTRAP_OUTPUT_PATH', None)}",
        )

        # 9.2 — AIController._run_bootstrap_backtest existe e é async
        from trading.ai_controller import AIController
        method = getattr(AIController, '_run_bootstrap_backtest', None)
        p.check(
            "AIController._run_bootstrap_backtest existe",
            method is not None,
        )
        if method is not None:
            p.check(
                "_run_bootstrap_backtest é coroutine",
                inspect.iscoroutinefunction(method),
            )

        # 9.3 — train_missing_components chama _run_bootstrap_backtest
        src = inspect.getsource(AIController.train_missing_components)
        p.check(
            "train_missing_components chama _run_bootstrap_backtest após treino",
            "_run_bootstrap_backtest" in src,
        )
        p.check(
            "chamada está APÓS verificação is_trained (ordem correta)",
            src.find("is_trained") < src.find("_run_bootstrap_backtest"),
        )

        # 9.4 — run_bot.system_state inicializa contadores
        import run_bot
        ss = run_bot.system_state
        p.check(
            "system_state['trades_since_last_adapt'] inicializa em 0",
            ss.get('trades_since_last_adapt') == 0,
            detail=f"value={ss.get('trades_since_last_adapt')}",
        )
        p.check(
            "system_state['total_closed_trades'] inicializa em 0",
            ss.get('total_closed_trades') == 0,
            detail=f"value={ss.get('total_closed_trades')}",
        )

        # 9.5 — save_trade_to_log incrementa contador para status fechado
        # (e NÃO incrementa para status aberto/pendente)
        # Reset antes do teste
        ss['trades_since_last_adapt'] = 0
        ss['total_closed_trades'] = 0

        # FILLED → deve incrementar
        try:
            run_bot.save_trade_to_log({
                'timestamp': '2025-01-01T00:00:00Z', 'symbol': 'BTCUSDT',
                'action': 'BUY', 'quantity': 0.01, 'price': 50000,
                'status': 'FILLED', 'leverage': 1, 'profit_probability': 0.6,
                'notional_value': 500, 'order_id': 'test1', 'client_order_id': 'c1',
            })
        except Exception as e:
            print(f"  [WARN] save_trade_to_log lançou: {e} — possivelmente normal se CSV não pôde ser escrito")
        p.check(
            "save_trade_to_log(status=FILLED) incrementa trades_since_last_adapt",
            ss.get('trades_since_last_adapt') == 1,
            detail=f"value após FILLED={ss.get('trades_since_last_adapt')}",
        )

        # NEW (pendente) → NÃO deve incrementar
        prev_count = ss['trades_since_last_adapt']
        try:
            run_bot.save_trade_to_log({
                'timestamp': '2025-01-01T00:01:00Z', 'symbol': 'BTCUSDT',
                'action': 'BUY', 'quantity': 0.01, 'price': 50001,
                'status': 'NEW', 'leverage': 1, 'profit_probability': 0.6,
                'notional_value': 500, 'order_id': 'test2', 'client_order_id': 'c2',
            })
        except Exception:
            pass
        p.check(
            "save_trade_to_log(status=NEW) NÃO incrementa contador",
            ss.get('trades_since_last_adapt') == prev_count,
            detail=f"value após NEW={ss.get('trades_since_last_adapt')} (esperado {prev_count})",
        )

        # 9.6 — Loop tem ambos os gatilhos (tempo OR trades)
        run_bot_src = inspect.getsource(run_bot)
        has_time_trigger = "trigger_time" in run_bot_src
        has_trade_trigger = "trigger_trades" in run_bot_src
        has_or_logic = "trigger_time or trigger_trades" in run_bot_src
        p.check(
            "Loop tem trigger_time, trigger_trades e usa OR entre eles",
            has_time_trigger and has_trade_trigger and has_or_logic,
            detail=f"time={has_time_trigger} trades={has_trade_trigger} or={has_or_logic}",
        )

        # 9.7 — Após adapt bem-sucedido, contador é zerado
        has_reset = "system_state['trades_since_last_adapt'] = 0" in run_bot_src
        p.check(
            "Após adaptação bem-sucedida, trades_since_last_adapt é zerado",
            has_reset,
        )

        # 9.8 — Simulação end-to-end do contador: 30 FILLEDs devem disparar o threshold
        ss['trades_since_last_adapt'] = 0
        for i in range(30):
            try:
                run_bot.save_trade_to_log({
                    'timestamp': f'2025-01-02T00:{i:02d}:00Z', 'symbol': 'BTCUSDT',
                    'action': 'BUY' if i % 2 == 0 else 'SELL',
                    'quantity': 0.01, 'price': 50000 + i,
                    'status': 'FILLED', 'leverage': 1, 'profit_probability': 0.6,
                    'notional_value': 500, 'order_id': f'sim{i}', 'client_order_id': f'cs{i}',
                })
            except Exception:
                pass
        threshold = int(BacktestConfig.MIN_TRADES_FOR_ADAPT)
        p.check(
            f"Após 30 FILLEDs, contador >= MIN_TRADES_FOR_ADAPT ({threshold})",
            ss.get('trades_since_last_adapt') >= threshold,
            detail=f"count={ss.get('trades_since_last_adapt')} vs threshold={threshold}",
        )

        p.conclude()
    return p


if __name__ == "__main__":
    run()
