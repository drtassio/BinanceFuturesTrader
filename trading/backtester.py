# -----------------------------------------------------------------------------
# ARQUIVO: trading/backtester.py
# -----------------------------------------------------------------------------

"""
Motor de Backtesting (Backtester).

Este módulo simula a execução de uma estratégia de trading sobre dados históricos,
proporcionando uma avaliação detalhada de sua performance e viabilidade antes
de arriscar capital real.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone 
import uuid 
import json 
import math # Para math.isclose

from utils.logger import get_logger
from config.settings import BacktestConfig, TradingConfig 
from models.backtest_result import BacktestResult 
from models.trade_schema import Signal, Action, OrderSide, OrderType, Trade 

# As classes abaixo são importadas apenas para tipagem, pois há dependência circular.
# A injeção de dependência via parâmetros é o ideal.
from trading.ai_controller import AIController
from trading.portfolio import PortfolioOptimizer
from trading.risk_manager import RiskManager
from xai_engine.explainer import Explainer
logger = get_logger("Backtester")

class Backtester:
    """
    Executa simulações de estratégias de trading em dados históricos,
    simulando taxas, slippage e gerenciamento de portfólio,
    considerando alavancagem e custos reais.
    """

    def __init__(self, config: BacktestConfig, db_session: Any): # db_session pode ser de SQLAlchemy
        if not isinstance(config, BacktestConfig):
            raise TypeError("🚨 [ERRO BACKTESTER] 'config' deve ser uma instância da classe BacktestConfig.")
        self.config = config
        self.trading_config = TradingConfig() # Instancia para acessar as taxas e alavancagem
        self.db_session = db_session # Sessão de banco de dados para salvar resultados
        logger.info("🧪 [BACKTESTER] Backtester inicializado.")
      
    # Em trading/backtester.py

    async def run(self, historical_data_multi_tf: Dict[str, pd.DataFrame], ai_controller: AIController,
                  portfolio: PortfolioOptimizer, risk_manager: RiskManager,
                  symbol: str, timeframe: str, model_version: str, explainer: Optional[Explainer] = None,
                  explain_decisions: bool = False) -> Optional[BacktestResult]:
        """
        [VERSÃO FINAL COMPLETA - COM EXPLICAÇÕES CONFIGURÁVEIS]
        Executa um backtest completo e gera explicações para as decisões de trade
        apenas se a configuração BACKTEST_WITH_EXPLANATIONS estiver ativada.
        """
        if timeframe not in historical_data_multi_tf:
            logger.error(f"❌ [ERRO BACKTESTER] Timeframe principal '{timeframe}' não encontrado. Abortando.")
            return None
        
        base_historical_df = historical_data_multi_tf[timeframe]
        if base_historical_df.empty:
            logger.error("❌ [ERRO BACKTESTER] DataFrame para backtest está vazio. Abortando.")
            return None
        
        should_explain = explain_decisions or self.config.BACKTEST_WITH_EXPLANATIONS
        explainer_status = "ATIVADAS" if should_explain and explainer else "DESATIVADAS para performance"
        logger.info(f"🧪 [BACKTESTER] Iniciando backtest OTIMIZADO para {symbol} ({timeframe}) com EXPLICAÇÕES {explainer_status}.")
        logger.info(f"Período: {base_historical_df.index.min()} a {base_historical_df.index.max()}. Capital: ${self.config.INITIAL_CAPITAL:,.2f}.")
        
        portfolio.initial_capital = self.config.INITIAL_CAPITAL
        portfolio.cash = self.config.INITIAL_CAPITAL
        portfolio.positions.clear()
        portfolio.portfolio_value_history = pd.Series(dtype=float)
        if not base_historical_df.index.empty:
            portfolio.portfolio_value_history[base_historical_df.index[0]] = self.config.INITIAL_CAPITAL
        
        from tqdm import tqdm
        
        trade_log: List[Dict[str, Any]] = []
        start_index = self.config.WARMUP_PERIOD 

        # [USABILITY] Progress bar for backtest execution
        iterator = tqdm(range(start_index, len(base_historical_df)), desc=f"🧪 Backtesting {symbol}", unit="bar")
        
        for i in iterator:
            current_timestamp = base_historical_df.index[i]
            current_row = base_historical_df.iloc[i]
            current_price = current_row.get('close')

            if pd.isna(current_price) or current_price <= 0:
                continue

            portfolio.update_portfolio_value({symbol: current_price}, timestamp=current_timestamp)
            risk_manager.update_portfolio_state(
                portfolio.get_total_value(), portfolio.cash, portfolio.positions,
                portfolio.total_notional_value, portfolio.margin_used
            )
            
            # [PERFORMANCE] Slice data to avoid O(N^2) complexity in Autoencoder
            # Keep enough history for rolling windows (regime detector needs ~100) and sequence length (60)
            window_size = 300 
            start_window = max(0, i + 1 - window_size)
            data_window_for_ai = base_historical_df.iloc[start_window : i+1]
            
            signal = await ai_controller.generate_trading_decision(data_window_for_ai)

            if should_explain and explainer and signal and signal.action != Action.HOLD:
                try:
                    latest_features_for_explanation = data_window_for_ai.iloc[-1]
                    explanation = explainer.explain_decision(signal, latest_features_for_explanation)
                    logger.info(f"🧠 [IA Backtest - Passo {i}] Decisão: {explanation.get('narrative', 'Explicação indisponível.')}")
                    signal.explanation['full_xai_explanation'] = explanation
                except Exception as e:
                    logger.error(f"❌ [BACKTESTER] Falha ao gerar explicação para o trade no passo {i}: {e}", exc_info=True)

            if signal is None or signal.action == Action.HOLD:
                continue

            if signal.position_size_pct > 0:
                trade_notional_value = portfolio.get_total_value() * signal.position_size_pct * signal.leverage
                
                if trade_notional_value < self.trading_config.MIN_NOTIONAL_VALUE:
                    continue

                trade_quantity = trade_notional_value / current_price
                
                simulated_trade = self._simulate_execution(signal, current_row, trade_quantity)
                
                if simulated_trade:
                    simulated_trade['explanation'] = signal.explanation.get('full_xai_explanation', {})
                    
                    executed_trade_obj = Trade(
                        trade_id=simulated_trade['trade_id'],
                        order_id=f"bt_order_{uuid.uuid4().hex[:8]}",
                        symbol=symbol,
                        side=signal.action,
                        quantity=simulated_trade['quantity'],
                        executed_price=simulated_trade['price'],
                        fee=simulated_trade['fee'],
                        fee_asset='USDT',
                        timestamp=simulated_trade['timestamp']
                    )
                    portfolio.update_from_trade(executed_trade_obj)
                    trade_log.append(simulated_trade)

        if not trade_log:
            logger.warning("⚠️ [BACKTESTER] Nenhum trade foi executado. Não é possível calcular métricas de performance.")
            return BacktestResult(
                test_id=f"bt_no_trades_{uuid.uuid4().hex[:8]}", model_version=model_version,
                start_date=base_historical_df.index.min(), end_date=base_historical_df.index.max(),
                total_return_pct=0.0, max_drawdown_pct=0.0, total_trades=0
            )

        metrics = self._calculate_performance_metrics(portfolio.portfolio_value_history.copy(), trade_log)
        
        equity_curve_data = portfolio.portfolio_value_history.reset_index().rename(columns={'index':'timestamp', 0:'equity'})
        equity_curve_data['timestamp'] = equity_curve_data['timestamp'].astype(str)
        
        backtest_result = self._create_backtest_result_object(
            metrics, equity_curve_data.to_dict('records'),
            trade_log, symbol, timeframe, model_version
        )
        
        logger.info(f"✅ [BACKTESTER] Backtest OTIMIZADO concluído. Retorno Final: {metrics.get('total_return_pct', 0):.2f}%, Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}.")
        return backtest_result
    
    async def run_walk_forward(
        self,
        historical_data_multi_tf: Dict[str, pd.DataFrame],
        ai_controller,
        portfolio,
        risk_manager,
        symbol: str,
        timeframe: str,
        model_version: str,
        n_splits: int = 5,
        gap_periods: int = 100,
        min_test_bars: int = 500,
    ) -> Optional[Dict]:
        """
        [WALK-FORWARD BACKTESTING] Avaliação out-of-sample com N folds cronológicos.

        Evita overfitting ao backtest único: cada fold testa num período FUTURO
        ao treino, com gap de embargo entre eles.

        Schema de splits (ex: n_splits=5):
        |=====Train_1====|gap|=Test_1=|
                |=====Train_2====|gap|=Test_2=|
                        |=====Train_3====|gap|=Test_3=|
                                ...

        Args:
            n_splits:      Número de folds (padrão 5 — 5 períodos out-of-sample)
            gap_periods:   Barras de embargo entre treino e teste (padrão 100)
            min_test_bars: Barras mínimas por fold de teste (padrão 500)

        Returns:
            Dict com métricas agregadas de todos os folds ou None se falhar.
        """
        if timeframe not in historical_data_multi_tf:
            logger.error(f"❌ [WF] Timeframe '{timeframe}' não encontrado.")
            return None

        base_df = historical_data_multi_tf[timeframe]
        n = len(base_df)
        
        # Calcula tamanho de cada fold de teste
        test_size = max(min_test_bars, n // (n_splits + 1))
        
        fold_results = []
        logger.info(
            f"🔬 [WALK-FORWARD] Iniciando {n_splits} folds | "
            f"Dataset: {n} barras | Teste/fold: {test_size} barras | Gap: {gap_periods}"
        )
        
        for fold_idx in range(n_splits):
            # Posição do fold (da frente para trás nos últimos folds)
            test_end   = n - (n_splits - fold_idx - 1) * test_size
            test_start = test_end - test_size
            train_end  = test_start - gap_periods

            if train_end < 200 or test_start >= test_end:
                logger.warning(f"⚠️ [WF] Fold {fold_idx}: dados insuficientes. Pulando.")
                continue

            train_df = base_df.iloc[:train_end].copy()
            test_df  = base_df.iloc[test_start:test_end].copy()

            logger.info(
                f"📊 [WF] Fold {fold_idx+1}/{n_splits}: "
                f"Treino [{base_df.index[0].date()} → {base_df.index[train_end-1].date()}] "
                f"({len(train_df)} barras) | "
                f"Teste [{test_df.index[0].date()} → {test_df.index[-1].date()}] "
                f"({len(test_df)} barras)"
            )

            try:
                # Reseta estado do portfólio para cada fold
                from trading.portfolio import PortfolioOptimizer
                from trading.risk_manager import RiskManager
                fold_portfolio = PortfolioOptimizer(self.trading_config)
                fold_risk      = RiskManager(self.trading_config)
                fold_portfolio.cash = self.config.INITIAL_CAPITAL

                # Roda o backtest simples no fold de TESTE apenas
                fold_result = await self.run(
                    historical_data_multi_tf={timeframe: test_df},
                    ai_controller=ai_controller,
                    portfolio=fold_portfolio,
                    risk_manager=fold_risk,
                    symbol=symbol,
                    timeframe=timeframe,
                    model_version=f"{model_version}_fold{fold_idx}"
                )

                if fold_result:
                    fold_results.append({
                        'fold':           fold_idx,
                        'test_start':     str(test_df.index[0].date()),
                        'test_end':       str(test_df.index[-1].date()),
                        'total_return':   fold_result.total_return_pct,
                        'max_drawdown':   fold_result.max_drawdown_pct,
                        'sharpe':         fold_result.sharpe_ratio or 0.0,
                        'sortino':        fold_result.sortino_ratio or 0.0,
                        'profit_factor':  fold_result.profit_factor or 0.0,
                        'total_trades':   fold_result.total_trades,
                        'win_rate':       fold_result.win_rate_pct,
                    })
                    logger.info(
                        f"✅ [WF] Fold {fold_idx+1}: "
                        f"Ret={fold_result.total_return_pct:.1f}% | "
                        f"DD={fold_result.max_drawdown_pct:.1f}% | "
                        f"Sharpe={fold_result.sharpe_ratio:.2f} | "
                        f"Trades={fold_result.total_trades}"
                    )

            except Exception as e:
                logger.error(f"❌ [WF] Fold {fold_idx} falhou: {e}", exc_info=True)
                fold_results.append({
                    'fold': fold_idx, 'error': str(e),
                    'sharpe': -2.0, 'total_return': -100.0,
                    'max_drawdown': 100.0, 'total_trades': 0
                })

        if not fold_results:
            logger.error("❌ [WF] Nenhum fold concluído com sucesso.")
            return None

        # Agrega métricas — Sharpe médio, pior drawdown (conservador)
        sharpes       = [f.get('sharpe', 0) for f in fold_results]
        returns       = [f.get('total_return', 0) for f in fold_results]
        drawdowns     = [abs(f.get('max_drawdown', 0)) for f in fold_results]
        profit_factors = [f.get('profit_factor', 0) for f in fold_results]

        agg = {
            'n_folds':              len(fold_results),
            'avg_sharpe':           float(np.mean(sharpes)),
            'min_sharpe':           float(np.min(sharpes)),
            'avg_return_pct':       float(np.mean(returns)),
            'worst_drawdown_pct':   float(np.max(drawdowns)),   # conservador
            'avg_profit_factor':    float(np.mean(profit_factors)),
            'positive_folds':       sum(1 for r in returns if r > 0),
            'fold_details':         fold_results,
        }

        logger.info(
            f"📈 [WALK-FORWARD] Resultado Final: "
            f"Folds={len(fold_results)} | "
            f"Sharpe médio={agg['avg_sharpe']:.2f} (min={agg['min_sharpe']:.2f}) | "
            f"Retorno médio={agg['avg_return_pct']:.1f}% | "
            f"Pior DD={agg['worst_drawdown_pct']:.1f}% | "
            f"Folds positivos={agg['positive_folds']}/{len(fold_results)}"
        )
        return agg
    

    def _simulate_execution(self, signal: Signal, data_row: pd.Series, quantity: float, is_maker_order: bool = False) -> Optional[Dict[str, Any]]:
        """
        [MELHORADO] Simula a execução de um trade de forma mais realista.
        Considera high/low para simular execução dentro da vela e aplica taxas e slippage.
        Considera a alavancagem para o cálculo do valor nocional para taxas.

        Args:
            signal (Signal): O sinal que originou o trade.
            data_row (pd.Series): A linha de dados do candle atual (inclui 'close', 'high', 'low', 'volume').
            quantity (float): A quantidade do ativo a ser negociada (em base asset).
            is_maker_order (bool): True se a ordem for Maker (LIMIT), False se for Taker (MARKET).

        Returns:
            Optional[Dict]: Dicionário com detalhes da execução simulada ou None em caso de erro.
        """
        if not isinstance(signal, Signal):
            logger.error(f"❌ [ERRO BACKTESTER] 'signal' inválido para _simulate_execution.")
            return None
        if not isinstance(data_row, pd.Series) or 'close' not in data_row or 'volume' not in data_row:
            logger.error(f"❌ [ERRO BACKTESTER] 'data_row' inválida ou sem 'close'/'volume' para _simulate_execution.")
            return None
        if not isinstance(quantity, (float, int)) or quantity <= 0:
            logger.error(f"❌ [ERRO BACKTESTER] 'quantity' inválida ({quantity}) para _simulate_execution. Deve ser positiva.")
            return None

        # [MELHORIA] Simulação mais realista considerando high/low da vela
        close_price = data_row['close']
        high_price = data_row.get('high', close_price)
        low_price = data_row.get('low', close_price)
        
        if close_price <= 0 or high_price <= 0 or low_price <= 0:
            logger.warning(f"⚠️ [BACKTESTER] Preços zero ou negativos para simulação de execução. Retornando None.")
            return None
        
        # [MELHORIA] Determina o preço de execução baseado no tipo de ordem e volatilidade da vela
        if is_maker_order:
            # Para ordens LIMIT, simula execução no preço limite (mais favorável)
            if signal.action == Action.BUY:
                # Ordem de compra LIMIT: executa no preço mais baixo possível dentro da vela
                execution_price = low_price
            else:
                # Ordem de venda LIMIT: executa no preço mais alto possível dentro da vela
                execution_price = high_price
        else:
            # Para ordens MARKET, simula execução considerando a volatilidade da vela
            if signal.action == Action.BUY:
                # Ordem de compra MARKET: executa entre low e close, tendendo para o high
                execution_price = low_price + (close_price - low_price) * 0.7
            else:
                # Ordem de venda MARKET: executa entre close e high, tendendo para o low
                execution_price = close_price + (high_price - close_price) * 0.3
        
        # Garante que o preço de execução está dentro dos limites da vela
        execution_price = max(low_price, min(high_price, execution_price))
        
        # Usa o preço de execução calculado para os cálculos subsequentes
        price = execution_price

        # Valor nocional da operação simulada
        trade_value_notional = quantity * price
        
        # Simulação de slippage
        slippage_pct = 0.0
        if self.config.SLIPPAGE_MODEL == 'volume_based' and data_row['volume'] > 0:
            participation_ratio = trade_value_notional / (data_row['volume'] * price + np.finfo(float).eps) # Quantidade em USD vs Volume do candle em USD
            slippage_pct = self.config.SLIPPAGE_IMPACT_FACTOR * (participation_ratio ** 0.5)
            slippage_pct = min(slippage_pct, self.config.SLIPPAGE_TOLERANCE_PCT) 
        elif self.config.SLIPPAGE_MODEL == 'constant':
            slippage_pct = self.config.SLIPPAGE_CONSTANT

        execution_price = price
        if signal.action == Action.BUY:
            execution_price = price * (1 + slippage_pct)
        elif signal.action == Action.SELL or signal.action == Action.CLOSE: 
            execution_price = price * (1 - slippage_pct)
            
        if execution_price <= 0:
            logger.error(f"❌ [ERRO BACKTESTER] Preço de execução simulado é não-positivo ({execution_price:.2f}). Ajustando para preço original.")
            execution_price = price 

        # Calcular a taxa de transação (agora com MAKER/TAKER e possível BNB Discount)
        fee_rate = self.trading_config.MAKER_FEE if is_maker_order else self.trading_config.TAKER_FEE
        
        # Lógica para desconto BNB e taxas USDC (simplificada para backtest)
        # Se você tiver lógica de BNB holdings para o bot, isso seria simulado aqui.
        # Por enquanto, aplicamos o desconto se o par for BNBUSDT ou se assumir que BNB é usado.
        # Ex: if symbol == 'BNBUSDT' or self.config.SIMULATE_BNB_DISCOUNT_ALWAYS:
        #    fee_rate *= self.trading_config.BNB_DISCOUNT_FEE_MULTIPLIER

        # Se for um contrato USDC
        if signal.symbol.endswith('USDC'):
            fee_rate = self.trading_config.USDC_MAKER_FEE if is_maker_order else self.trading_config.USDC_TAKER_FEE

        fee = trade_value_notional * fee_rate # Taxa baseada no valor nocional
        fee = max(0.0, fee) # Garante que a taxa não seja negativa

        return {
            "trade_id": f"sim_trade_{uuid.uuid4().hex[:12]}",
            "timestamp": data_row.name,
            # [FIX] entry_time é o campo que train_from_trade_log() do ProfitabilityPredictor busca
            "entry_time": data_row.name,
            "exit_time": None,       # preenchido ao fechar posição
            "exit_price": None,      # preenchido ao fechar posição
            "symbol": signal.symbol,
            "action": signal.action.value,
            "quantity": quantity,
            "price": execution_price,
            "fee": fee,
            "slippage_pct": slippage_pct,
            "leverage": signal.leverage,
            "notional_value": trade_value_notional,
            "profit_probability": signal.explanation.get("profit_probability", 0.5),
            "pnl": 0.0,  # PnL realizado (preenchido ao fechar posição)
        }

    def _calculate_performance_metrics(self, equity_curve_series: pd.Series, trade_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula um conjunto completo de métricas de performance a partir da curva de equity
        e do log de trades.
        """
        if equity_curve_series.empty or len(equity_curve_series) < 2:
            logger.warning("⚠️ [BACKTESTER] Curva de equity insuficiente para calcular métricas de performance. Retornando vazio.")
            return {}
        if not trade_log:
            logger.warning("⚠️ [BACKTESTER] Log de trades vazio. Não é possível calcular métricas de trades.")
            return {}

        # Garante que o índice é datetime e está em UTC (ou um timezone consistente)
        if not isinstance(equity_curve_series.index, pd.DatetimeIndex):
            equity_curve_series.index = pd.to_datetime(equity_curve_series.index, utc=True)
        equity_curve_series = equity_curve_series.sort_index()

        initial_capital = equity_curve_series.iloc[0]
        final_capital = equity_curve_series.iloc[-1]
        
        # Retornos diários/período
        # pct_change pode introduzir NaNs no primeiro elemento, .dropna() remove-os
        returns = equity_curve_series.pct_change().dropna()
        
        if returns.empty:
            logger.warning("⚠️ [BACKTESTER] Retornos vazios após pct_change. Retornando métricas parciais.")
            return {
                "initial_capital": initial_capital, "final_capital": final_capital,
                "total_return_pct": ((final_capital / initial_capital) - 1) * 100 if initial_capital > 0 else 0,
                "total_trades": len(trade_log), "winning_trades": 0, "losing_trades": 0, "win_rate_pct": 0,
                "max_drawdown_pct": 0, "annualized_return_pct": 0, "volatility_pct": 0
            }

        total_return = ((final_capital / initial_capital) - 1)
        
        # --- Drawdown ---
        # Retornos acumulados
        cumulative_returns = (1 + returns).cumprod()
        # Pico anterior (valor mais alto atingido até o momento)
        peak = cumulative_returns.expanding(min_periods=1).max()
        # Drawdown = (Valor Atual - Pico) / Pico
        drawdown = (cumulative_returns - peak) / (peak + np.finfo(float).eps) 
        max_drawdown = drawdown.min()
        
        # --- Ratios Ajustados ao Risco ---
        # Anualização depende do timeframe dos dados
        # Frequência de dados (diária, por hora, por minuto)
        # Assumir que o timeframe da equity_curve é o `timeframe` passado para o run().
        # Precisa mapear o `timeframe` (e.g., '1m', '1h', '1d') para o número de períodos por ano.
        periods_per_day_map = {
            '1m': 24 * 60, '5m': 24 * 12, '15m': 24 * 4, '30m': 24 * 2, '1h': 24, '2h': 12, '4h': 6, '6h': 4, '8h': 3, '12h': 2, '1d': 1
        }
        periods_per_year = periods_per_day_map.get(self.trading_config.PRIMARY_TIMEFRAME_TRADING, 24 * 60) * 365.25 # padrão 1m
        
        sharpe_ratio = (returns.mean() / (returns.std() + np.finfo(float).eps)) * np.sqrt(periods_per_year)
        
        # [D-A1 FIX] Semi-desvio canônico: RMSE de min(r, 0) sobre todos os retornos.
        # downside_returns.std() usava desvio em torno da média dos negativos (não de 0).
        _downside_sq = np.minimum(returns.values, 0.0) ** 2
        _downside_rmse = float(np.sqrt(_downside_sq.mean())) if len(_downside_sq) > 0 else 0.0
        sortino_ratio = (returns.mean() / (_downside_rmse + np.finfo(float).eps)) * np.sqrt(periods_per_year) if _downside_rmse > 1e-9 else np.inf
        
        annualized_return = returns.mean() * periods_per_year
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        annualized_volatility = returns.std() * np.sqrt(periods_per_year)

        # --- Estatísticas de Trades ---
        pnls = [t['pnl'] for t in trade_log if 'pnl' in t and t['pnl'] is not None] # Apenas PnLs de trades fechados
        
        if not pnls:
            logger.warning("⚠️ [BACKTESTER] Nenhum PnL válido nos logs de trade. Estatísticas de trade incompletas.")
            pnls = [0.0] # Para evitar erros de divisão por zero/empty list
        
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        win_rate = len(winning_trades) / len(pnls) if pnls else 0 # Taxa de acerto
        
        total_gross_profit = sum(winning_trades)
        total_gross_loss = abs(sum(losing_trades))
        profit_factor = total_gross_profit / (total_gross_loss + np.finfo(float).eps) if total_gross_loss != 0 else np.inf

        avg_trade_pnl_pct = (sum(pnls) / len(pnls) / initial_capital * 100) if pnls and initial_capital > 0 else 0

        # Média do período de manutenção dos trades (se o trade_log incluir timestamps de abertura/fechamento)
        avg_holding_period_hours = 0.0
        # Isto exigiria que o trade_log registrasse 'entry_timestamp' e 'exit_timestamp'
        # para posições, e não apenas trades individuais. Por enquanto, mantemos 0.0

        var_95 = abs(np.percentile(returns, 5)) if not returns.empty else 0.0
        cvar_95 = abs(returns[returns <= np.percentile(returns, 5)].mean()) if not returns.empty and len(returns[returns <= np.percentile(returns, 5)]) > 0 else 0.0

        # --- Novas Métricas de Alavancagem e P ---
        # Extrair alavancagem e P de cada trade logado
        logged_leverages = [t['leverage'] for t in trade_log if 'leverage' in t]
        logged_profit_probs = [t['profit_probability'] for t in trade_log if 'profit_probability' in t]

        avg_leverage_per_trade = np.mean(logged_leverages) if logged_leverages else 1.0
        avg_profit_probability_per_trade = np.mean(logged_profit_probs) if logged_profit_probs else 0.5


        metrics = {
            "initial_capital": float(initial_capital),
            "final_capital": float(final_capital),
            "total_return_pct": total_return * 100,
            "annualized_return_pct": annualized_return * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "volatility_pct": annualized_volatility * 100,
            "sharpe_ratio": float(sharpe_ratio) if np.isfinite(sharpe_ratio) else 0.0,
            "sortino_ratio": float(sortino_ratio) if np.isfinite(sortino_ratio) else 0.0,
            "calmar_ratio": float(calmar_ratio) if np.isfinite(calmar_ratio) else 0.0,
            "total_trades": len(trade_log),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate_pct": win_rate * 100,
            "avg_trade_pnl_pct": avg_trade_pnl_pct,
            "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else 0.0,
            "var_95_pct": var_95 * 100,
            "cvar_95_pct": cvar_95 * 100,
            "avg_holding_period_hours": avg_holding_period_hours,
            "avg_leverage_per_trade": float(avg_leverage_per_trade), # Nova métrica
            "avg_profit_probability_per_trade": float(avg_profit_probability_per_trade), # Nova métrica
        }
        logger.debug(f"📊 [BACKTESTER] Métricas calculadas: {metrics}")
        return metrics

    def _create_backtest_result_object(self, metrics: Dict[str, Any], equity_curve: List[Dict[str, Any]], trade_log: List[Dict[str, Any]], symbol: str, timeframe: str, model_version: str, params: Optional[Dict[str, Any]] = None) -> BacktestResult:
        """
        Cria um objeto SQLAlchemy BacktestResult a partir das métricas e logs.
        """
        # Converte Timestamps para strings ISO para serialização JSON
        for item in equity_curve: 
            if isinstance(item['timestamp'], datetime):
                item['timestamp'] = item['timestamp'].isoformat()
        # Incluir leverage e profit_probability no trade_log JSON
        for item in trade_log: 
            if isinstance(item['timestamp'], datetime):
                item['timestamp'] = item['timestamp'].isoformat()
            # Garante que alavancagem e P sejam incluídos no JSON, mesmo que None
            item['leverage'] = item.get('leverage')
            item['profit_probability'] = item.get('profit_probability')

        # Calcula retornos mensais para armazenamento
        monthly_returns = {}
        equity_series = pd.Series([item['equity'] for item in equity_curve], index=pd.to_datetime([item['timestamp'] for item in equity_curve]))
        if not equity_series.empty:
            monthly_returns_series = equity_series.resample('M').last().pct_change().dropna()
            monthly_returns = monthly_returns_series.apply(lambda x: x * 100).to_dict() 
            monthly_returns = {k.isoformat(): v for k, v in monthly_returns.items()}


        result_id = f"bt_{uuid.uuid4().hex[:12]}"
        logger.info(f"🧪 [BACKTESTER] Criando objeto BacktestResult com ID: {result_id}.")
        return BacktestResult(
            test_id=result_id,
            model_version=model_version,
            start_date=pd.to_datetime(equity_curve[0]['timestamp']),
            end_date=pd.to_datetime(equity_curve[-1]['timestamp']),
            symbols=json.dumps([symbol]), 
            timeframe=timeframe,
            parameters=params if params else {}, 
            
            initial_capital=metrics.get('initial_capital', 0.0),
            final_capital=metrics.get('final_capital', 0.0),
            total_return_pct=metrics.get('total_return_pct', 0.0),
            annualized_return_pct=metrics.get('annualized_return_pct', 0.0),
            max_drawdown_pct=metrics.get('max_drawdown_pct', 0.0),
            volatility_pct=metrics.get('volatility_pct', 0.0),
            sharpe_ratio=metrics.get('sharpe_ratio'),
            sortino_ratio=metrics.get('sortino_ratio'),
            calmar_ratio=metrics.get('calmar_ratio'),
            total_trades=metrics.get('total_trades', 0),
            winning_trades=metrics.get('winning_trades', 0),
            losing_trades=metrics.get('losing_trades', 0),
            win_rate_pct=metrics.get('win_rate_pct', 0.0),
            avg_trade_pnl_pct=metrics.get('avg_trade_pnl_pct', 0.0),
            profit_factor=metrics.get('profit_factor'),
            var_95_pct=metrics.get('var_95_pct'),
            cvar_95_pct=metrics.get('cvar_95_pct'),
            avg_holding_period_hours=metrics.get('avg_holding_period_hours'),
            avg_leverage_per_trade=metrics.get('avg_leverage_per_trade'), # Nova métrica
            avg_profit_probability_per_trade=metrics.get('avg_profit_probability_per_trade'), # Nova métrica

            equity_curve=json.dumps(equity_curve), 
            trade_log=json.dumps(trade_log), 
            monthly_returns=json.dumps(monthly_returns) 
        )

    def _save_results(self, result: BacktestResult):
        """Salva o objeto BacktestResult na sessão do banco de dados."""
        if not isinstance(result, BacktestResult):
            logger.error("❌ [ERRO BACKTESTER] Objeto 'result' inválido. Deve ser uma instância de BacktestResult. Não será salvo.")
            return

        try:
            self.db_session.add(result)
            self.db_session.commit()
            logger.info(f"✅ [BACKTESTER] Resultado do backtest {result.test_id} salvo no banco de dados.")
        except Exception as e:
            logger.error(f"❌ [ERRO BACKTESTER] Falha ao salvar resultado do backtest no banco de dados para {result.test_id}: {e}", exc_info=True)
            self.db_session.rollback() # Reverte a transação em caso de erro