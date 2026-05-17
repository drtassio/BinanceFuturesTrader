# -----------------------------------------------------------------------------
# ARQUIVO: trading/stress_tester.py
# -----------------------------------------------------------------------------

"""
Módulo de Teste de Estresse (Stress Tester).

Este componente submete o sistema de trading completo a cenários de mercado
extremos e simulados para avaliar sua robustez, resiliência e comportamento
em condições adversas.
"""

import numpy as np
import pandas as pd
import copy
import asyncio
from datetime import datetime, timedelta, timezone 
from typing import Dict, List, Any, Tuple
import math
import uuid # Para gerar IDs únicos de trade

from utils.logger import get_logger
from config.settings import StressTestConfig, TradingConfig
# [BUG FIX] Estes enums/dataclasses eram referenciados sem importar.
from models.trade_schema import Action, OrderSide, Trade
# Classes importadas apenas para tipagem, pois há dependência circular
from trading.ai_controller import AIController
from trading.portfolio import PortfolioOptimizer
from trading.risk_manager import RiskManager
from feature_engineering import FeatureEngineeringPipeline

logger = get_logger("StressTester")

class StressTester:
    """
    Simula cenários de mercado extremos para testar a resiliência do sistema de IA.
    Avalia como o bot se comporta sob condições de mercado adversas.
    """

    def __init__(self, config: StressTestConfig, feature_pipeline: FeatureEngineeringPipeline):
        if not isinstance(config, StressTestConfig):
            raise TypeError("🚨 [ERRO STRESS] 'config' deve ser uma instância da classe StressTestConfig.")
        if not isinstance(feature_pipeline, FeatureEngineeringPipeline):
            raise TypeError("🚨 [ERRO STRESS] 'feature_pipeline' deve ser uma instância de FeatureEngineeringPipeline.")

        self.config = config
        self.trading_config = TradingConfig() # Acessar os timeframes configurados
        self.feature_pipeline = feature_pipeline # Injeta o pipeline de features
        self.scenarios = self.config.SCENARIOS # Dicionário de cenários configurados

        logger.info(f"🧪 [STRESS] StressTester inicializado com {len(self.scenarios)} cenários.")

    async def run_all_scenarios(self, ai_controller: AIController, portfolio: PortfolioOptimizer, risk_manager: RiskManager) -> List[Dict[str, Any]]:
        """
        Executa todos os cenários de teste de estresse definidos na configuração.
        """
        if not isinstance(ai_controller, AIController) or not isinstance(portfolio, PortfolioOptimizer) or not isinstance(risk_manager, RiskManager):
            logger.error("❌ [ERRO STRESS] Componentes AIController, PortfolioOptimizer ou RiskManager inválidos para run_all_scenarios. Abortando.")
            return []

        all_results = []
        logger.info(f"🧪 [STRESS] Iniciando bateria de testes de estresse com {len(self.scenarios)} cenários.")
        
        for scenario_name in self.scenarios.keys():
            logger.info(f"\n--- Iniciando Teste de Estresse: {scenario_name} ---")
            try:
                result = await self.run_scenario(scenario_name, ai_controller, portfolio, risk_manager)
                all_results.append(result)
            except Exception as e:
                logger.critical(f"🚨 [CRÍTICO STRESS] Erro fatal ao executar cenário '{scenario_name}': {e}. Este cenário não será incluído nos resultados.", exc_info=True)
                all_results.append({"scenario_name": scenario_name, "status": "FAILED_CRITICAL", "error": str(e)})

        logger.info("✅ [STRESS] Bateria de testes de estresse concluída.")
        return all_results

    async def run_scenario(self, scenario_name: str, ai_controller: AIController, portfolio: PortfolioOptimizer, risk_manager: RiskManager) -> Dict[str, Any]:
        """
        Executa um único cenário de teste de estresse.
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"🚨 [ERRO STRESS] Cenário de estresse '{scenario_name}' não encontrado na configuração.")

        scenario_params = self.scenarios[scenario_name]
        logger.info(f"⚙️ [STRESS] Executando cenário '{scenario_name}' com parâmetros: {scenario_params}")

        # 1. Isolar o ambiente de teste
        # Cria cópias profundas dos componentes para um teste isolado
        test_ai_controller, test_portfolio, test_risk_manager = self._create_isolated_env(ai_controller, portfolio, risk_manager)
        
        initial_portfolio_value = test_portfolio.get_total_value()
        logger.info(f"💰 [STRESS] Capital inicial para o cenário '{scenario_name}': ${initial_portfolio_value:,.2f}.")

        # 2. Gerar dados de mercado sintéticos multi-timeframe e processá-los com o pipeline de features
        logger.info(f"📊 [STRESS] Gerando dados sintéticos multi-timeframe para o cenário '{scenario_name}'...")
        
        # O `_generate_stress_data` agora retorna um dicionário de DataFrames {timeframe: OHLCV DataFrame}
        raw_stress_data_multi_tf = self._generate_stress_data(scenario_params) 

        if not raw_stress_data_multi_tf or not raw_stress_data_multi_tf.get(self.trading_config.PRIMARY_TIMEFRAME_TRADING).empty:
            logger.error(f"❌ [ERRO STRESS] Dados sintéticos para '{scenario_name}' estão vazios ou o timeframe primário está faltando. Abortando cenário.")
            return {"scenario_name": scenario_name, "status": "FAILED_EMPTY_DATA", "parameters": scenario_params}
        
        # Processar os dados sintéticos através do pipeline de features
        # O `create_features` espera o dicionário multi-timeframe
        logger.info(f"⚙️ [STRESS] Processando dados sintéticos com o FeatureEngineeringPipeline (multi-timeframe) para '{scenario_name}'...")
        
        # Usar o timeframe primário para o `create_features` e passar o dicionário
        stress_df_featured = self.feature_pipeline.create_features(
            raw_stress_data_multi_tf, 
            self.trading_config.PRIMARY_PAIR,  # [CORREÇÃO] Usa PRIMARY_PAIR em vez de 'STRESS_ASSET' hardcoded
            self.trading_config.PRIMARY_TIMEFRAME_TRADING, 
            fit_scaler=False, # Usar scalers pré-treinados
            fit_autoencoder=False # Usar autoencoder pré-treinado
        ) 
        
        if stress_df_featured is None or stress_df_featured.empty:
            logger.error(f"❌ [ERRO STRESS] Falha na engenharia de features para dados de estresse de '{scenario_name}'. Abortando cenário.")
            return {"scenario_name": scenario_name, "status": "FAILED_FEATURE_ENGINEERING", "parameters": scenario_params}

        # O RiskManager precisa de dados históricos de retorno (apenas 'close' do TF primário é suficiente para VaR)
        test_risk_manager.add_historical_data({self.trading_config.PRIMARY_PAIR: stress_df_featured[['close']]})
        logger.info(f"📊 [STRESS] Dados sintéticos (features) prontos. Total de {len(stress_df_featured)} pontos. Primeira entrada: {stress_df_featured.index[0]}.")


        # 3. Executar a simulação passo a passo
        simulation_log = []
        min_portfolio_value = initial_portfolio_value
        max_portfolio_value = initial_portfolio_value 
        
        # O loop de simulação itera sobre o timeframe primário (que é o `stress_df_featured`)
        for i in range(len(stress_df_featured)):
            current_timestamp = stress_df_featured.index[i]
            
            # O AIController precisa do DataFrame de Features completo (incluindo multi-timeframe e ocultas)
            # A linha `i` de `stress_df_featured` contém todas as features para este timestamp.
            
            # Obter o preço atual para a simulação
            current_price = stress_df_featured['close'].iloc[i]
            if current_price <= 0:
                logger.warning(f"⚠️ [STRESS] Preço de fechamento inválido ({current_price}) no passo {i}. Pulando cálculo.")
                current_price = test_portfolio.get_current_price(self.trading_config.PRIMARY_PAIR) 
                if current_price <= 0: continue 

            # Atualiza o portfólio e o risco antes de gerar a decisão para este passo
            test_portfolio.update_portfolio_value({self.trading_config.PRIMARY_PAIR: current_price})
            # O `update_portfolio_state` do RiskManager agora espera total_notional_value e margin_used
            test_risk_manager.update_portfolio_state(
                test_portfolio.get_total_value(), 
                test_portfolio.cash, 
                test_portfolio.positions,
                test_portfolio.total_notional_value,
                test_portfolio.margin_used # Assume que portfolio.py já calcula estes valores
            )
            
            current_portfolio_value = test_portfolio.get_total_value()
            max_portfolio_value = max(max_portfolio_value, current_portfolio_value)
            min_portfolio_value = min(min_portfolio_value, current_portfolio_value)

            # Gera a decisão da IA para o passo atual
            # AIController espera o DataFrame completo (do início até o passo i)
            data_window_for_ai = stress_df_featured.iloc[:i+1]

            # Await na decisão da IA, pois o AIController usa funções assíncronas (como get_ticker_price)
            # e a lógica de decisão pode ser complexa.
            signal = await test_ai_controller.generate_trading_decision(
                data_window_for_ai, 
                test_portfolio, 
                test_risk_manager
            )
            
            # 4. Simula a execução do trade
            if signal and signal.action != Action.HOLD and signal.position_size_pct > 0:
                
                # O `RiskManager` já ajustou `signal.position_size_pct` (margem) e `signal.leverage`.
                # Calculamos o valor nocional para simulação
                trade_notional_value = test_portfolio.get_total_value() * signal.position_size_pct * signal.leverage
                
                if current_price > 0:
                    trade_quantity = trade_notional_value / current_price
                    
                    # Simula a execução do trade (aplicando slippage e fees)
                    simulated_trade_details = self._simulate_execution(signal, stress_df_featured.iloc[i], trade_quantity, is_maker_order=False)

                    if simulated_trade_details:
                        # Para o backtester, update_from_trade já gerencia caixa e posições
                        # Precisamos saber se este trade é uma abertura ou fechamento para calcular o PnL realizado.
                        
                        current_pos = test_portfolio.positions.get(signal.symbol)
                        
                        # Calcula PnL realizado se for um trade de fechamento (oposto à posição atual)
                        pnl_realized = 0.0
                        
                        if current_pos:
                            is_closing_trade = (signal.action == OrderSide.BUY and current_pos.quantity < 0) or \
                                               (signal.action == OrderSide.SELL and current_pos.quantity > 0)
                            
                            if is_closing_trade:
                                pnl_realized = (simulated_trade_details['price'] - current_pos.entry_price) * current_pos.quantity # Simplificação: PnL = (preço_saida - preço_entrada) * qtd
                                # Ajuste de sinal para PnL de short
                                if current_pos.quantity < 0:
                                     pnl_realized *= -1 # Se short, PnL = (preço_entrada - preço_saida) * abs(qtd)

                        
                        # Cria o objeto Trade para atualizar o portfólio de teste
                        mock_trade_obj = Trade(
                            trade_id=f"st_trade_{uuid.uuid4().hex[:8]}",
                            order_id=f"st_order_{uuid.uuid4().hex[:8]}",
                            symbol=signal.symbol,
                            side=signal.action,
                            quantity=simulated_trade_details['quantity'],
                            executed_price=simulated_trade_details['price'],
                            fee=simulated_trade_details['fee'],
                            fee_asset='USDT',
                            timestamp=current_timestamp,
                            pnl=pnl_realized # Adiciona PnL realizado
                        )
                        
                        test_portfolio.update_from_trade(mock_trade_obj) # Atualiza o estado do portfólio

                        # Log da simulação
                        log_entry = {
                            'step': i,
                            'timestamp': current_timestamp.isoformat(),
                            'price': float(current_price),
                            'portfolio_value': float(test_portfolio.get_total_value()),
                            'signal_action': signal.action.value,
                            'trade_executed': True,
                            'trade_details': {
                                'action': signal.action.value,
                                'qty': simulated_trade_details['quantity'],
                                'price': simulated_trade_details['price'],
                                'fee': simulated_trade_details['fee'],
                                'leverage': signal.leverage,
                                'pnl_realized': pnl_realized
                            }
                        }
                        simulation_log.append(log_entry)
                        logger.debug(f"📈 [STRESS] Simulação: {signal.action.value} {trade_quantity:.4f} {signal.symbol} @ ${current_price:.2f} (Alavancagem: {signal.leverage:.2f}x). PnL Realizado: ${pnl_realized:,.2f}.")

            # Se não houve trade, apenas loga o estado
            if not signal or signal.action == Action.HOLD or signal.position_size_pct <= 0:
                log_entry = {
                    'step': i,
                    'timestamp': current_timestamp.isoformat(),
                    'price': float(current_price),
                    'portfolio_value': float(test_portfolio.get_total_value()),
                    'signal_action': signal.action.value if signal else "HOLD_NO_SIGNAL",
                    'trade_executed': False
                }
                simulation_log.append(log_entry)
                logger.debug(f"ℹ️ [STRESS] Sinal HOLD no passo {i}.")
            
        # 5. Avaliar os resultados do cenário
        final_portfolio_value = test_portfolio.get_total_value()
        
        # Recalcula o drawdown máximo com o histórico de valores do portfólio
        portfolio_values_series = pd.Series([entry['portfolio_value'] for entry in simulation_log], 
                                            index=[pd.to_datetime(entry['timestamp']) for entry in simulation_log])
        if not portfolio_values_series.empty:
            cumulative_returns_sim = (portfolio_values_series.pct_change().dropna() + 1).cumprod()
            peak_sim = cumulative_returns_sim.expanding(min_periods=1).max()
            drawdown_sim = (cumulative_returns_sim - peak_sim) / (peak_sim + np.finfo(float).eps)
            max_drawdown_calculated = abs(drawdown_sim.min()) if not drawdown_sim.empty else 0.0
        else:
            max_drawdown_calculated = 0.0
            
        # O cenário passa se o drawdown máximo NÃO exceder o limite aceitável
        max_allowable_drawdown = scenario_params.get('max_allowable_drawdown', 0.25) 
        passed = max_drawdown_calculated <= max_allowable_drawdown

        result = {
            "scenario_name": scenario_name,
            "parameters": scenario_params,
            "passed": passed,
            "initial_portfolio_value": float(initial_portfolio_value),
            "final_portfolio_value": float(final_portfolio_value),
            "min_portfolio_value_achieved": float(min_portfolio_value),
            "max_drawdown_pct": max_drawdown_calculated * 100,
            "max_allowable_drawdown_pct": max_allowable_drawdown * 100,
            "simulation_log_summary": {
                "total_steps": len(simulation_log),
                "buy_signals": sum(1 for log in simulation_log if log['signal_action'] == Action.BUY.value),
                "sell_signals": sum(1 for log in simulation_log if log['signal_action'] == Action.SELL.value),
                "hold_signals": sum(1 for log in simulation_log if log['signal_action'] == Action.HOLD.value),
                "trade_executed_count": sum(1 for log in simulation_log if log.get('trade_executed', False))
            },
            # Inclui um preview do log de simulação para análise
            "full_simulation_log_preview": simulation_log[:10] + (simulation_log[-10:] if len(simulation_log) > 20 else [])
        }
        
        status_text = "APROVADO" if passed else "REPROVADO"
        logger.info(f"--- Resultado do Teste '{scenario_name}': {status_text} ---")
        logger.info(f"Drawdown Máximo: {result['max_drawdown_pct']:.2f}%, Permitido: {result['max_allowable_drawdown_pct']:.2f}%, Valor Final: ${result['final_portfolio_value']:.2f}")

        return result

    def _create_isolated_env(self, ai_controller: AIController, portfolio: PortfolioOptimizer, risk_manager: RiskManager) -> Tuple[AIController, PortfolioOptimizer, RiskManager]:
        """
        Cria cópias profundas dos componentes essenciais para um teste isolado.
        """
        logger.debug("⚙️ [STRESS] Criando ambiente de teste isolado (deep copy)...")
        # É crucial que deepcopy funcione para todos os objetos internos.
        # Objetos como loggers ou sessões de DB não devem ser copiados, mas são injetados.
        
        # Copia os objetos
        test_ai_controller = copy.deepcopy(ai_controller)
        test_portfolio = copy.deepcopy(portfolio)
        test_risk_manager = copy.deepcopy(risk_manager)

        # Re-injetar dependências para garantir que apontem para as *cópias*
        # (se o AIController ou outros componentes armazenam referências diretas a Portfolio/RiskManager)
        test_ai_controller.set_portfolio(test_portfolio)
        test_ai_controller.set_risk_manager(test_risk_manager)
        
        # Limpar o estado do AI Monitor para o teste, para que não polua o log principal
        if test_ai_controller.ai_monitor:
            test_ai_controller.ai_monitor.reset_session()
        
        logger.debug("✅ [STRESS] Ambiente de teste isolado criado com sucesso.")
        return test_ai_controller, test_portfolio, test_risk_manager

    def _generate_stress_data(self, params: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Gera dados OHLCV sintéticos multi-timeframe para o cenário de estresse.

        Args:
            params (Dict[str, Any]): Parâmetros específicos para a geração de dados do cenário.

        Returns:
            Dict[str, pd.DataFrame]: Dicionário de DataFrames OHLCV sintéticos por timeframe.
        """
        # Definir a duração total do cenário com base nos parâmetros
        duration_minutes = params.get('duration_minutes', 0)
        duration_hours = params.get('duration_hours', 0)
        duration_days = params.get('duration_days', 0)
        
        total_duration_minutes = duration_minutes + duration_hours * 60 + duration_days * 24 * 60
        if total_duration_minutes <= 0:
            logger.error("❌ [ERRO STRESS] Duração total do cenário é zero ou negativa.")
            return {}

        # Gerar dados sintéticos apenas para o timeframe de 1 minuto (o mais granular)
        base_tf = '1m'
        timesteps = pd.to_datetime(pd.date_range(start=datetime.now(timezone.utc), periods=total_duration_minutes, freq='T'))
        
        price = np.zeros(total_duration_minutes)
        price[0] = 100.0  # Preço inicial base
        logger.debug(f"📊 [STRESS] Gerando {total_duration_minutes} minutos de dados sintéticos para {base_tf}.")

        # --- Lógica de Geração de Preço para o Cenário ---
        # Simula o choque de preço (e.g., flash crash)
        price_drop_pct = params.get('price_drop_pct', 0.0)
        shock_duration_minutes = int(total_duration_minutes * params.get('shock_duration_ratio', 0.1)) 
        
        # Volatilidade base para ruído
        base_volatility_noise = 0.001 
        vol_multiplier = params.get('volatility_multiplier', 1.0)
        
        # Simula spread (para liquidez, afeta High/Low)
        spread_multiplier = params.get('spread_multiplier', 1.0)

        for t in range(1, total_duration_minutes):
            current_price_t_minus_1 = price[t-1]
            if current_price_t_minus_1 <= 0: 
                current_price_t_minus_1 = price[0] 

            # Fase de choque (queda abrupta)
            if t < shock_duration_minutes and price_drop_pct > 0:
                drop_per_step = -price_drop_pct / shock_duration_minutes
                price[t] = current_price_t_minus_1 * (1 + drop_per_step)
            else:
                # Fase de recuperação/estabilização
                recovery_pct = params.get('recovery_pct', 0.0)
                # O preço alvo para recuperação pode ser o inicial ou um percentual dele
                target_price_recovery = price[0] * (1 - price_drop_pct * (1 - recovery_pct)) if price_drop_pct > 0 else price[0]
                
                # Recuperação gradual em direção ao alvo
                price[t] = current_price_t_minus_1 + (target_price_recovery - current_price_t_minus_1) * 0.02 
            
            # Adiciona ruído de volatilidade (sempre presente)
            noise = np.random.normal(0, base_volatility_noise * vol_multiplier)
            price[t] *= (1 + noise)
            
            # Garante que o preço não caia abaixo de um mínimo razoável (evitar zeros/negativos)
            price[t] = max(price[t], price[0] * 0.01) 

        # Cria o DataFrame OHLCV para o timeframe base (1m)
        df_raw = pd.DataFrame(index=timesteps, data={'close': price})
        
        # `open` é o `close` anterior
        df_raw['open'] = df_raw['close'].shift(1).fillna(df_raw['close'].iloc[0])
        
        # `high` e `low` baseados no `open` e `close` com um spread aleatório
        # O spread é influenciado pelo `spread_multiplier` para simular liquidez
        random_spread = np.random.uniform(0.0005, 0.005, size=len(df_raw)) * spread_multiplier
        df_raw['high'] = df_raw[['open', 'close']].max(axis=1) * (1 + random_spread)
        df_raw['low'] = df_raw[['open', 'close']].min(axis=1) * (1 - random_spread)
        
        # Volume: Poisson para simular eventos de volume (picos/baixas)
        base_volume = params.get('base_volume', 1000)
        volume_divider = params.get('volume_divider', 1.0)
        df_raw['volume'] = np.random.poisson(base_volume / volume_divider, size=len(df_raw)) + 1 
        df_raw['volume'] = df_raw['volume'].astype(float)

        # Garante que não há NaNs resultantes das operações de shift/cálculo
        df_raw.ffill(inplace=True)
        df_raw.bfill(inplace=True)
        
        if df_raw.isnull().any().any():
            logger.error("❌ [ERRO STRESS] NaNs persistentes no DataFrame de dados sintéticos.")

        # 3. Gerar DataFrames para outros Timeframes (Resampling)
        # Usa o dataframe de 1m para gerar os timeframes maiores
        
        # Obter a lista de timeframes a serem usados
        all_timeframes_to_generate = self.trading_config.ALL_TRADING_TIMEFRAMES
        
        # Dicionário para armazenar os DataFrames gerados
        generated_dfs: Dict[str, pd.DataFrame] = {base_tf: df_raw}

        # Resampling para timeframes maiores (e.g., '1h', '4h')
        for tf in all_timeframes_to_generate:
            if tf != base_tf:
                # Usa a função resample do Pandas para agregar dados
                try:
                    resampled_df = df_raw.resample(tf).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    
                    if not resampled_df.empty:
                         generated_dfs[tf] = resampled_df
                    else:
                         logger.warning(f"⚠️ [STRESS] Falha ao gerar dados sintéticos para o timeframe '{tf}'. DataFrame vazio após resampling.")
                except Exception as e:
                    logger.error(f"❌ [ERRO STRESS] Erro ao resamplear dados para o timeframe '{tf}': {e}")

        logger.info(f"✅ [STRESS] Geração de dados sintéticos concluída para timeframes: {list(generated_dfs.keys())}.")
        return generated_dfs