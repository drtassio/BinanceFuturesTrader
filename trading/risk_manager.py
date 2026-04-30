# -----------------------------------------------------------------------------
# ARQUIVO: trading/risk_manager.py
# -----------------------------------------------------------------------------

"""
Módulo de Gerenciamento de Risco (Risk Manager).

Este componente é o guardião do capital do sistema. Ele monitora continuamente
o risco do portfólio, valida todas as decisões de trading antes da execução e
garante que o sistema opere dentro dos limites de risco predefinidos.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime 

from utils.logger import get_logger
from config.settings import TradingConfig
from models.trade_schema import Signal, Position, OrderSide, Action

# Adiciona importação de NativeIndicators para o EPSILON
from feature_engineering.native_indicators import NativeIndicators

logger = get_logger("RiskManager")

class RiskManager:
    """
    Gerencia o risco em nível de portfólio e de posição individual,
    implementando lógicas de VaR, stop-loss dinâmico e validação pré-trade.
    """

    def __init__(self, config: TradingConfig):
        if not isinstance(config, TradingConfig):
            raise TypeError("🚨 [ERRO RISCO] 'config' deve ser uma instância da classe TradingConfig.")
        self.config = config
        
        # Estado atual do portfólio, atualizado externamente
        self.portfolio_value: float = 0.0 # Equity Total
        self.cash: float = 0.0 # Cash disponível
        self.positions: Dict[str, Position] = {} # Posições abertas
        
        # Dicionário para armazenar séries de retornos históricos de cada símbolo
        self.historical_returns: Dict[str, pd.Series] = {} 
        
        # Valor de pico do portfólio, para cálculo de drawdown
        self.peak_portfolio_value: float = self.config.INITIAL_CAPITAL 
        # Drawdown atual (calculado na atualização do estado)
        self.current_drawdown: float = 0.0 

        # Atributos para rastreamento de alavancagem (serão passados pelo PortfolioOptimizer/AIController)
        self.margin_used: float = 0.0 
        self.total_notional_value: float = 0.0 
        self.current_leverage_ratio: float = 0.0 

        logger.info(f"🛡️ [RISCO] RiskManager inicializado. Limites de Risco: Max Drawdown {self.config.MAX_DRAWDOWN_PERCENT:.2%}, Max VaR {self.config.MAX_VAR_1D_PERCENT:.2%}.")

    def update_portfolio_state(self, portfolio_value: float, cash: float, positions: Dict[str, Position], 
                               total_notional_value: float, margin_used: float):
        """
        Atualiza o estado do portfólio para os cálculos de risco.
        Recebe informações de alavancagem e margem diretamente do PortfolioOptimizer.

        Args:
            portfolio_value (float): O valor total atual do portfólio (equity).
            cash (float): O saldo de caixa disponível.
            positions (Dict[str, Position]): Dicionário de posições abertas.
            total_notional_value (float): Valor nocional total de todas as posições abertas.
            margin_used (float): Margem total alocada em posições abertas.
        """
        if not isinstance(portfolio_value, (int, float)) or portfolio_value < 0:
            logger.error(f"❌ [ERRO RISCO] 'portfolio_value' inválido: {portfolio_value}. Deve ser um número não negativo.")
            return
        if not isinstance(cash, (int, float)) or cash < 0:
            logger.error(f"❌ [ERRO RISCO] 'cash' inválido: {cash}. Deve ser um número não negativo.")
            return
        if not isinstance(positions, dict):
            logger.error(f"❌ [ERRO RISCO] 'positions' inválido. Deve ser um dicionário. Recebido: {type(positions)}.")
            return

        self.portfolio_value = portfolio_value 
        self.cash = cash 
        self.positions = positions 
        
        # Recebe os valores precisos do PortfolioOptimizer/AIController
        self.total_notional_value = total_notional_value
        self.margin_used = margin_used

        # Atualiza o pico do portfólio para cálculo de drawdown
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        
        # Calcula o drawdown atual
        if self.peak_portfolio_value > 0:
            self.current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        else:
            self.current_drawdown = 0.0 

        # Calcula a alavancagem atual do portfólio (exposição nocional / equity)
        if self.portfolio_value > 0:
            self.current_leverage_ratio = self.total_notional_value / self.portfolio_value
        else:
            self.current_leverage_ratio = 0.0

        # [MELHORIA] Envelopes de risco dinâmicos
        self._update_risk_envelopes()
        
        logger.debug(f"🛡️ [RISCO] Estado do portfólio atualizado. Equity: ${self.portfolio_value:,.2f}, Cash: ${self.cash:,.2f}, "
                     f"Drawdown: {self.current_drawdown:.2%}, Notional: ${self.total_notional_value:,.2f}, "
                     f"Margem Usada: ${self.margin_used:,.2f}, Alavancagem: {self.current_leverage_ratio:.2f}x.")

    def _update_risk_envelopes(self):
        """
        [MELHORIA] Atualiza envelopes de risco dinâmicos baseados no estado atual do portfólio.
        
        Implementa curvas de corte suaves que reduzem automaticamente o tamanho das posições
        conforme o risco aumenta, antes de atingir os limites hard.
        """
        # Envelope de alavancagem dinâmico
        if self.current_leverage_ratio > self.config.MAX_LEVERAGE_PER_TRADE * 0.7:  # 70% do limite máximo
            self.leverage_reduction_factor = max(0.3, 1.0 - (self.current_leverage_ratio / self.config.MAX_LEVERAGE_PER_TRADE))
            logger.warning(f"⚠️ [RISCO] Alavancagem alta ({self.current_leverage_ratio:.2f}x). Reduzindo posições por fator {self.leverage_reduction_factor:.2f}")
        else:
            self.leverage_reduction_factor = 1.0
        
        # Envelope de drawdown dinâmico
        if self.current_drawdown > self.config.MAX_DRAWDOWN_PERCENT * 0.6:  # 60% do limite máximo
            self.drawdown_reduction_factor = max(0.2, 1.0 - (self.current_drawdown / self.config.MAX_DRAWDOWN_PERCENT))
            logger.warning(f"⚠️ [RISCO] Drawdown alto ({self.current_drawdown:.2%}). Reduzindo posições por fator {self.drawdown_reduction_factor:.2f}")
        else:
            self.drawdown_reduction_factor = 1.0
        
        # Fator de redução combinado (usa o mais restritivo)
        self.combined_reduction_factor = min(self.leverage_reduction_factor, self.drawdown_reduction_factor)

    def _get_current_price_from_positions(self, symbol: str) -> float:
        """
        Recupera o preço atual do ativo. Em um sistema real, isso viria do DataProvider.
        Como o `PortfolioOptimizer` já atualiza os `positions` com o PnL não realizado,
        podemos inferir que ele tem acesso aos preços atuais. No entanto, para evitar
        dependência circular, o `RiskManager` deve ser atualizado com os preços
        ou receber o preço atual no `update_portfolio_state`.
        
        Para esta implementação, manteremos o placeholder para ilustrar que o preço
        deve ser o preço de mercado atual, não o preço de entrada.
        """
        # A implementação real precisa buscar o preço de mercado atual.
        # Se `update_portfolio_state` não recebe os preços, este método é um placeholder.
        if symbol in self.positions:
            return self.positions[symbol].entry_price # Usar entry_price como fallback se não tiver acesso ao preço atual
        return 0.0 

    def add_historical_data(self, data: Dict[str, pd.DataFrame]):
        """
        Adiciona dados históricos de retorno (log-returns) para cálculos de VaR.

        Args:
            data (Dict[str, pd.DataFrame]): Dicionário onde a chave é o símbolo
                                             e o valor é um DataFrame com a coluna 'close'.
        """
        if not isinstance(data, dict):
            logger.error(f"❌ [ERRO RISCO] 'data' deve ser um dicionário. Recebido: {type(data)}.")
            return

        for symbol, df in data.items():
            if not isinstance(df, pd.DataFrame) or df.empty or 'close' not in df.columns:
                logger.warning(f"⚠️ [ALERTE RISCO] Dados históricos inválidos ou vazios para '{symbol}'. Ignorando.")
                continue
            
            # Calcula retornos logarítmicos
            # np.finfo(float).eps é usado para evitar log(0) e divisão por zero
            returns = np.log(df['close'] / (df['close'].shift(1) + np.finfo(float).eps)).dropna()
            
            if returns.empty:
                logger.warning(f"⚠️ [ALERTE RISCO] Retornos vazios calculados para '{symbol}'. Dados insuficientes ou estáticos.")
                continue

            self.historical_returns[symbol] = returns
            logger.debug(f"📊 [RISCO] Retornos históricos para '{symbol}' atualizados. {len(returns)} pontos.")

    def check_trade_approval(self, signal: Signal) -> Tuple[bool, str]:
        """
        Valida um sinal de trading contra as regras de risco predefinidas antes da execução.
        """
        if not isinstance(signal, Signal):
            reason = f"❌ [ERRO RISCO] Sinal inválido recebido para validação de risco. Tipo: {type(signal)}. Rejeitado."
            logger.error(reason)
            return False, reason

        if signal.action == Action.HOLD or signal.action == "HOLD":  # [BUG FIX] comparar enum corretamente
            return True, "✅ Sinal HOLD: Nenhuma alteração de risco, aprovação automática."

        # [SAFETY] Validador de Qualidade do Sinal (Filtro Anti-Alucinação)
        is_quality_good, quality_reason = self.validate_signal_quality(signal)
        if not is_quality_good:
            logger.warning(quality_reason)
            return False, quality_reason

        # 1. Determinar se o trade é de fechamento (redução de nocional) ou abertura/aumento
        is_closing_trade = False
        current_pos = self.positions.get(signal.symbol)
        
        # Compara usando .value ou cast para string para evitar problemas de tipos de Enum diferentes
        action_val = signal.action.value if hasattr(signal.action, 'value') else str(signal.action)
        
        if current_pos:
            # Se a direção do trade for oposta à posição atual, é um fechamento parcial ou total
            if (action_val == "BUY" and current_pos.quantity < 0) or \
               (action_val == "SELL" and current_pos.quantity > 0) or \
               (action_val == "CLOSE"): 
                is_closing_trade = True
                logger.info(f"🛡️ [RISCO] Ordem de FECHAMENTO detectada para {signal.symbol}. Ignorando limites de margem para redução de risco.")

        # 2. Checagem de Margem Disponível (o sinal usa signal.position_size_pct como % de margem)
        initial_margin_needed_for_trade = self.portfolio_value * signal.position_size_pct
        
        # [CORREÇÃO] Calcular cash disponível considerando a margem já utilizada
        available_cash = self.cash - self.margin_used
        
        if not is_closing_trade and available_cash < initial_margin_needed_for_trade:
            reason = (f"❌ [REJEITADO] Caixa disponível (${available_cash:,.2f}) insuficiente para "
                      f"cobrir a margem inicial necessária (${initial_margin_needed_for_trade:,.2f}) para o trade ({signal.symbol}).")
            logger.warning(reason)
            return False, reason

        # 1. Checagem de Drawdown Máximo (Regra de Saída em Crise)
        if self.current_drawdown > self.config.MAX_DRAWDOWN_PERCENT:
            reason = (f"❌ [REJEITADO] Drawdown atual ({self.current_drawdown:.2%}) "
                      f"excede o limite de {self.config.MAX_DRAWDOWN_PERCENT:.2%}. "
                      "Modo de segurança ativado: Nenhuma nova posição de risco será aberta.")
            logger.warning(f"🚨 {reason} Equity atual: ${self.portfolio_value:,.2f}, Pico: ${self.peak_portfolio_value:,.2f}.")
            return False, reason

        # 2. Checagem de Alavancagem Máxima da Conta (Gerenciamento da exposição nocional)
        # Calcula a alavancagem projetada se o trade for executado.
        
        # Valor nocional do trade a ser aberto (em USD)
        trade_notional_value = self.portfolio_value * signal.position_size_pct * signal.leverage
        
        projected_total_notional_value = self.total_notional_value
        if not is_closing_trade: 
            projected_total_notional_value += trade_notional_value
        else: 
            # A lógica de fechamento é mais complexa, mas para aprovação, a exposição total não deve exceder o limite
            # se a nova posição resultante (após a reversão) for maior que a exposição original.
            # Por simplicidade, assumimos que trades de fechamento são aprovados quanto à exposição,
            # a menos que sejam uma reversão que exceda o limite de alavancagem total.
            
            # Se for uma reversão (ex: de Long 1x para Short 2x), a exposição aumenta.
            # Precisamos calcular o nocional líquido da posição resultante e adicionar à exposição total.
            
            # Para este cenário, se is_closing_trade é True, confiamos na lógica do PortfolioOptimizer para gerenciar
            # o PnL realizado. Para o RiskManager, se a ação for BUY ou SELL, a margem e nocional serão ajustados.
            pass

        projected_leverage_ratio = projected_total_notional_value / (self.portfolio_value + np.finfo(float).eps)

        # [FIX] Ordens de FECHAMENTO sempre passam nesta checagem, mesmo com alavancagem projetada alta.
        # O objetivo é REDUZIR a exposição, não aumentá-la.
        if not is_closing_trade and projected_leverage_ratio > self.config.MAX_TOTAL_EXPOSURE_PERCENT: 
            reason = (f"❌ [REJEITADO] Alavancagem projetada do portfólio ({projected_leverage_ratio:.2f}x) "
                      f"excederia o limite de {self.config.MAX_TOTAL_EXPOSURE_PERCENT:.2f}x (Exposição total).")
            logger.warning(reason)
            return False, reason

        # 3. Checagem de Tamanho Máximo da Posição Individual (margem usada)
        # O signal.position_size_pct já representa a porcentagem da MARGEM.
        
        projected_margin_used = self.margin_used
        if not is_closing_trade:
            projected_margin_used += initial_margin_needed_for_trade
        
        # Checa se a margem projetada excede o limite (MAX_POSITION_SIZE_PERCENT)
        # [FIX] Se for uma ordem de fechamento, permitimos mesmo que o limite esteja excedido,
        # pois o objetivo é reduzir a exposição e o risco.
        margin_ratio = projected_margin_used / (self.portfolio_value + np.finfo(float).eps)
        logger.debug(f"📊 [RISCO] Check Margem: {projected_margin_used:.2f} / {self.portfolio_value:.2f} = {margin_ratio:.2%} (Limite: {self.config.MAX_POSITION_SIZE_PERCENT:.2%}). Closing: {is_closing_trade}")
        
        if not is_closing_trade and margin_ratio > self.config.MAX_POSITION_SIZE_PERCENT:
             reason = (f"❌ [REJEITADO] Margem total utilizada projetada "
                       f"({projected_margin_used / (self.portfolio_value + np.finfo(float).eps):.2%}) "
                       f"excederia o limite de margem individual de {self.config.MAX_POSITION_SIZE_PERCENT:.2%}.")
             logger.warning(reason)
             return False, reason
             
        # Checar se a alavancagem proposta pelo agente está dentro dos limites permitidos
        if not (self.config.MIN_LEVERAGE_PER_TRADE <= signal.leverage <= self.config.MAX_LEVERAGE_PER_TRADE):
            reason = (f"❌ [REJEITADO] Alavancagem solicitada ({signal.leverage:.2f}x) fora dos limites permitidos "
                      f"({self.config.MIN_LEVERAGE_PER_TRADE:.2f}x - {self.config.MAX_LEVERAGE_PER_TRADE:.2f}x).")
            logger.warning(reason)
            return False, reason

        # 4. Checagem de VaR (Value at Risk) do Portfólio
        # Calcula o VaR projetado do portfólio com o novo trade hipotético
        projected_var_value = self.calculate_portfolio_var(add_trade_signal=signal)
        
        if projected_var_value > self.config.MAX_VAR_1D_PERCENT:
            reason = (f"❌ [REJEITADO] VaR projetado do portfólio ({projected_var_value:.2%}) "
                      f"excederia o limite de {self.config.MAX_VAR_1D_PERCENT:.2%}. "
                      f"VaR do portfólio atual: {self.calculate_portfolio_var():.2%}.")
            logger.warning(reason)
            return False, reason

        # [MELHORIA] Aplicação de envelopes de risco dinâmicos
        if hasattr(self, 'combined_reduction_factor') and self.combined_reduction_factor < 1.0:
            # Aplica redução dinâmica no tamanho da posição
            original_position_size = signal.position_size_pct
            signal.position_size_pct *= self.combined_reduction_factor
            
            logger.info(f"🛡️ [RISCO] Aplicando envelope dinâmico: posição reduzida de {original_position_size:.2%} para {signal.position_size_pct:.2%} "
                       f"(fator de redução: {self.combined_reduction_factor:.2f})")
            
            # Recalcula margem necessária com o novo tamanho
            adjusted_margin_needed = self.portfolio_value * signal.position_size_pct
            if self.cash < adjusted_margin_needed:
                reason = f"❌ [REJEITADO] Mesmo com redução dinâmica, caixa insuficiente para margem ajustada (${adjusted_margin_needed:,.2f})"
                logger.warning(reason)
                return False, reason
            # [BUG FIX] Não retornar False aqui — posição foi reduzida e caixa é suficiente.
            # O sinal continua aprovado com o tamanho reduzido pelo envelope dinâmico.
            
        logger.info(f"✅ [APROVADO] Sinal para {signal.symbol} ({signal.action.value}, Confiança: {signal.confidence:.2%}, P: {signal.profit_probability:.2%}, Alavancagem: {signal.leverage:.2f}x) aprovado pela gestão de risco.")
        return True, "✅ Aprovado"

    def validate_signal_quality(self, signal: Signal) -> Tuple[bool, str]:
        """
        [SAFETY] Valida a qualidade intrínseca do sinal gerado pela IA.
        Atua como um filtro 'Anti-Alucinação' para evitar trades com baixa convicção matemática.
        """
        # 1. Checagem de Confiança Mínima do Modelo (Decision Confidence)
        # O modelo diz o quão confiante está na sua própria previsão.
        if signal.confidence < self.config.MIN_CONFIDENCE_FOR_TRADE:
             return False, (f"❌ [QUALIDADE] Confiança do modelo ({signal.confidence:.2%}) "
                            f"abaixo do mínimo aceitável ({self.config.MIN_CONFIDENCE_FOR_TRADE:.2%}).")

        # 2. Checagem de Probabilidade de Lucratividade (Predictive Probability)
        # O modelo 'Profit Predictor' estima a chance de sucesso do trade.
        if signal.profit_probability < self.config.MIN_PROFIT_PROBABILITY:
             return False, (f"❌ [QUALIDADE] Probabilidade de lucro estimada ({signal.profit_probability:.2%}) "
                            f"abaixo do limiar de segurança ({self.config.MIN_PROFIT_PROBABILITY:.2%}).")

        return True, "✅ Sinal Aprovado"

    def check_liquidation_risk(self) -> List[str]:
        """
        Verifica se alguma posição está perigosamente próxima do preço de liquidação.
        
        Returns:
            List[str]: Lista de mensagens de alerta, se houver.
        """
        alerts = []
        try:
            for symbol, pos in self.positions.items():
                if pos.quantity == 0:
                    continue
                
                # Se não temos dados de liquidação ou mark price, não podemos verificar
                if pos.liquidation_price <= 0 or pos.mark_price <= 0:
                    continue
                
                # Cálculo da distância percentual
                distance_pct = abs(pos.mark_price - pos.liquidation_price) / pos.mark_price
                
                # Se estiver a menos de 5% do monitor de liquidação -> PERIGO
                # (Binance costuma liquidar antes de chegar a 0% de margem, cuidado)
                if distance_pct < 0.05:
                    alert_msg = (f"🚨 PERIGO DE LIQUIDAÇÃO: {symbol} está a APENAS {distance_pct:.2%} "
                                 f"do preço de liquidação! (Mark: {pos.mark_price}, Liq: {pos.liquidation_price})")
                    alerts.append(alert_msg)
                    logger.critical(alert_msg)
                    
        except Exception as e:
            logger.error(f"❌ [ERRO RISCO] Erro ao verificar risco de liquidação: {e}")
            
        return alerts

    def calculate_position_size(self, symbol: str, confidence: float, volatility: float,
                                 profit_probability: float, desired_leverage: float) -> Tuple[float, float]:
        """
        Calcula o tamanho da posição recomendado (em termos de margem alocada)
        e a alavancagem final a ser usada, com base na volatilidade (paridade de risco),
        confiança da IA, probabilidade de lucratividade e alavancagem desejada.
        """
        if not isinstance(symbol, str) or not symbol:
            logger.error("❌ [ERRO RISCO] 'symbol' inválido para calculate_position_size.")
            return 0.0, 1.0
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            logger.error(f"❌ [ERRO RISCO] 'confidence' inválida: {confidence}. Deve ser entre 0 e 1.")
            return 0.0, 1.0
        if not isinstance(volatility, (int, float)) or volatility <= 0:
            logger.warning(f"⚠️ [ALERTE RISCO] Volatilidade para {symbol} é zero/negativa ({volatility}). "
                           "Usando volatilidade padrão mínima para evitar divisão por zero.")
            volatility = 0.005 # Valor mínimo razoável, 0.5%
        if not isinstance(profit_probability, (int, float)) or not (0.0 <= profit_probability <= 1.0):
            logger.error(f"❌ [ERRO RISCO] 'profit_probability' inválida: {profit_probability}. Deve ser entre 0 e 1.")
            return 0.0, 1.0
        if not isinstance(desired_leverage, (int, float)) or not (self.config.MIN_LEVERAGE_PER_TRADE <= desired_leverage <= self.config.MAX_LEVERAGE):
            logger.warning(f"⚠️ [ALERTE RISCO] Alavancagem desejada inválida ({desired_leverage}). Limitando ao range [{self.config.MIN_LEVERAGE_PER_TRADE}, {self.config.MAX_LEVERAGE}].")
            desired_leverage = np.clip(desired_leverage, self.config.MIN_LEVERAGE_PER_TRADE, self.config.MAX_LEVERAGE)
            
        # 1. Ajuste o risco alvo por trade com base na confiança e probabilidade de lucratividade
        # Um trade com alta confiança E alta probabilidade pode arriscar mais.
        # Ajuste de (profit_probability - 0.5) para dar um boost/redução simétrico
        risk_adjustment_factor = 1.0 + (confidence - 0.5) * 0.5 + (profit_probability - 0.5) * 0.75 
        risk_adjustment_factor = np.clip(risk_adjustment_factor, 0.5, 1.5) # Limita o fator de ajuste

        target_risk_per_trade_pct = self.config.MAX_PORTFOLIO_RISK_PERCENT 
        adjusted_risk_pct = target_risk_per_trade_pct * risk_adjustment_factor
        
        # 2. Tamanho da Posição Base (sem alavancagem), pela paridade de risco
        # Este é o valor nocional % do portfólio que deveria ter o risco ajustado.
        position_size_notional_base_pct = adjusted_risk_pct / volatility

        # 3. Determinar a alavancagem final a ser aplicada
        # A alavancagem final será a alavancagem desejada pelo agente, dentro dos limites configurados.
        final_leverage_to_apply = np.clip(desired_leverage, self.config.MIN_LEVERAGE_PER_TRADE, self.config.MAX_LEVERAGE_PER_TRADE)

        # 4. Calcular a margem a ser alocada
        # Margem = (Valor Nocional Base) / Alavancagem Final Aplicada
        # Esta é a porcentagem do portfólio que será usada como MARGEM.
        margin_allocation_pct = position_size_notional_base_pct / (final_leverage_to_apply + np.finfo(float).eps)

        # 5. Limites Finais: Margem 
        # Garante que a margem alocada não exceda a MAX_POSITION_SIZE_PERCENT (margem máxima por trade).
        margin_allocation_pct = min(margin_allocation_pct, self.config.MAX_POSITION_SIZE_PERCENT)

        # Calcula o valor nocional resultante deste trade individual
        resulting_notional_pct = margin_allocation_pct * final_leverage_to_apply

        # Se o resulting_notional_pct exceder o MAX_TOTAL_EXPOSURE_PERCENT, ajustamos a alavancagem
        if resulting_notional_pct > self.config.MAX_TOTAL_EXPOSURE_PERCENT:
            # Reduz a alavancagem para que o nocional resultante não exceda o limite.
            final_leverage_to_apply = self.config.MAX_TOTAL_EXPOSURE_PERCENT / (margin_allocation_pct + np.finfo(float).eps)
            final_leverage_to_apply = np.clip(final_leverage_to_apply, self.config.MIN_LEVERAGE_PER_TRADE, self.config.MAX_LEVERAGE_PER_TRADE)
            
            logger.debug(f"📐 [RISCO] Alavancagem ajustada devido ao limite de exposição total. Nova Alavancagem: {final_leverage_to_apply:.2f}x.")

        logger.debug(f"📐 [RISCO] Cálculo de Posição Final para {symbol}: Confiança={confidence:.2f}, P_Lucro={profit_probability:.2f}, Volatilidade={volatility:.2%}, "
                     f"Margem Alocada={margin_allocation_pct:.2%}, Alavancagem Aplicada={final_leverage_to_apply:.2f}x.")
        
        return float(margin_allocation_pct), float(final_leverage_to_apply)

    def calculate_portfolio_var(self, add_trade_signal: Optional[Signal] = None) -> float:
        """
        Calcula o Value at Risk (VaR) de 1 dia do portfólio usando o método histórico (não-paramétrico).
        Considera posições alavancadas e custos de trade.
        """
        # Se o valor do portfólio for zero ou não houver posições e nenhum trade proposto, VaR é 0.
        if (self.portfolio_value <= 0 and not self.positions) and add_trade_signal is None:
            return 0.0

        # Constrói os pesos do portfólio atual (valor nocional da posição / equity total)
        current_notional_weights: Dict[str, float] = {}
        if self.portfolio_value > 0: 
            for symbol, pos in self.positions.items():
                # Para cálculo de VaR, precisamos do peso nocional da posição.
                # O PortfolioOptimizer já deve ter atualizado o valor nocional (`self.total_notional_value`).
                # Se `pos.unrealized_pnl` estiver atualizado, `pos.quantity` e `pos.entry_price` podem ser usados como proxies
                # ou podemos usar a informação de `self.total_notional_value` se o PortfolioOptimizer a fornecer.
                
                # Para o VaR, a exposição é baseada no valor nocional
                # O peso é a porcentagem da exposição nocional em relação ao equity total do portfólio.
                
                # Como a função `update_portfolio_state` agora recebe `total_notional_value`,
                # podemos calcular a exposição por ativo se o `positions` contiver o valor nocional de cada um.
                # Como o `Position` dataclass não tem `notional_value`, vamos usar uma estimativa baseada na quantidade e entry_price,
                # sabendo que isso pode ser impreciso se o preço de mercado mudou muito desde a entrada.
                
                # Se `pos.unrealized_pnl` está atualizado, podemos usar o preço atual para calcular o nocional.
                # current_price = pos.entry_price + (pos.unrealized_pnl / pos.quantity)
                # position_notional = abs(pos.quantity) * current_price
                
                # Por simplicidade, vamos usar a `self.total_notional_value` para o portfólio atual,
                # e apenas adicionar o trade hipotético para o cálculo projetado.

                # O peso é a porcentagem da exposição nocional em relação ao equity total.
                # Como não temos o valor nocional de cada posição aqui, vamos usar a exposição total (`self.total_notional_value`)
                # e distribuir o peso proporcionalmente à quantidade de cada posição, se for o único ativo.

                # Se estamos negociando apenas um ativo (PRIMARY_PAIR), a exposição é `self.current_leverage_ratio`.
                # Se houver múltiplos ativos, precisaríamos da exposição individual.
                
                # Assumindo um único ativo para o backtest:
                if len(self.positions) == 1:
                    current_notional_weights[symbol] = self.current_leverage_ratio * (1 if pos.quantity >= 0 else -1)

        projected_notional_weights = current_notional_weights.copy()

        # Se um trade hipotético é fornecido, ajusta os pesos para o cálculo do VaR projetado
        if add_trade_signal:
            if not isinstance(add_trade_signal, Signal):
                logger.error(f"❌ [ERRO RISCO] Sinal de trade inválido para cálculo de VaR. Tipo: {type(add_trade_signal)}. Ignorando.")
            else:
                # O valor nocional do trade hipotético (em relação ao equity total)
                # (Capital Total * Margem Alocada % ) * Alavancagem
                signal_notional_value_pct_of_equity = add_trade_signal.position_size_pct * add_trade_signal.leverage
                
                # O sinal do peso depende do lado da ordem
                trade_weight = signal_notional_value_pct_of_equity
                if add_trade_signal.action == OrderSide.SELL:
                    trade_weight *= -1 # Se for uma ordem de venda (abrir short), o peso é negativo

                # Ajusta o peso do símbolo no portfólio projetado
                # Se o símbolo já estiver na posição, o peso líquido muda.
                projected_notional_weights[add_trade_signal.symbol] = \
                    projected_notional_weights.get(add_trade_signal.symbol, 0.0) + trade_weight
                
                logger.debug(f"📊 [RISCO] Calculando VaR projetado com sinal: {add_trade_signal.symbol} {add_trade_signal.action.value} {add_trade_signal.leverage:.2f}x alavancagem.")

        if not projected_notional_weights:
            # Em startup ou se apenas caixa, o VaR é 0 - não é um alerta de risco mas um estado normal
            logger.debug("🛡️ [RISCO] Nenhuns pesos de portfólio (100% Caixa). VaR = 0.0.")
            return 0.0

        # Coleta séries de retornos para os ativos que estão ou estarão no portfólio
        relevant_returns_series: Dict[str, pd.Series] = {}
        for sym, weight in projected_notional_weights.items():
            if sym in self.historical_returns and not self.historical_returns[sym].empty:
                relevant_returns_series[sym] = self.historical_returns[sym]
            elif abs(weight) > 1e-9: 
                logger.warning(f"⚠️ [ALERTE RISCO] Dados históricos de retorno ausentes para '{sym}' para cálculo de VaR. Isso pode afetar a precisão.")

        if not relevant_returns_series:
            logger.warning("⚠️ [ALERTE RISCO] Não há dados históricos de retorno para os ativos relevantes no portfólio. Impossível calcular VaR. Retornando VaR máximo configurado.")
            return self.config.MAX_VAR_1D_PERCENT

        returns_df = pd.DataFrame(relevant_returns_series).dropna()
        
        min_var_samples = 252 
        if len(returns_df) < min_var_samples:
            logger.warning(f"⚠️ [ALERTE RISCO] Dados históricos insuficientes ({len(returns_df)} pontos, mínimo {min_var_samples}) para calcular VaR de forma robusta. Retornando VaR máximo configurado.")
            return self.config.MAX_VAR_1D_PERCENT

        weights_series = pd.Series(projected_notional_weights)
        # Sincroniza pesos com as colunas do DataFrame de retornos usando reindex (mais robusto que align com Index)
        aligned_weights = weights_series.reindex(returns_df.columns).fillna(0.0)
        
        portfolio_hist_returns = returns_df.dot(aligned_weights)
        
        if portfolio_hist_returns.empty:
            logger.error("❌ [ERRO RISCO] Retornos históricos do portfólio estão vazios após cálculo ponderado. Retornando VaR máximo.")
            return self.config.MAX_VAR_1D_PERCENT

        # O VaR de 95% é o 5º percentil dos retornos históricos (a 5% pior perda).
        # Multiplicamos por -1 para obter um valor positivo, representando a perda.
        var_95 = -np.percentile(portfolio_hist_returns, 5) # 5º percentil para 95% de confiança
        
        # Impacto das Funding Fees no VaR
        # Custo diário = Valor Nocional Total * Custo Diário da Alavancagem %
        estimated_daily_funding_cost_pct = self.total_notional_value * self.config.LEVERAGE_COST_PER_DAY_PCT / (self.portfolio_value + np.finfo(float).eps)
        
        var_95_with_fees = var_95 + estimated_daily_funding_cost_pct
        
        logger.debug(f"📊 [RISCO] VaR do Portfólio (95%, 1D) calculado: {var_95:.4f} (base), "
                     f"Funding Cost Estimado: {estimated_daily_funding_cost_pct:.4f}, VaR Final: {var_95_with_fees:.4f}.")
        
        return float(var_95_with_fees) 

    def get_current_risk_profile(self) -> Dict[str, Any]:
        """
        Retorna um resumo do perfil de risco atual do portfólio.
        """
        # Recalcula as métricas para garantir que o snapshot é o mais atual
        portfolio_var = self.calculate_portfolio_var() # VaR do estado atual

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "portfolio_value": self.portfolio_value,
            "total_notional_value": self.total_notional_value,
            "margin_used": self.margin_used,
            "current_leverage_ratio": self.current_leverage_ratio,
            "max_leverage_limit": self.config.MAX_LEVERAGE_PER_TRADE,
            "current_drawdown_pct": self.current_drawdown,
            "max_drawdown_limit_pct": self.config.MAX_DRAWDOWN_PERCENT,
            "portfolio_var_95_pct": portfolio_var,
            "max_var_limit_pct": self.config.MAX_VAR_1D_PERCENT,
            "is_risk_off_mode": (
                self.current_drawdown > self.config.MAX_DRAWDOWN_PERCENT or          # Drawdown crítico
                self.current_leverage_ratio > self.config.MAX_TOTAL_EXPOSURE_PERCENT  # Exposição total excedida
            ),  # Indica se o bot deve entrar em modo defensivo (bloqueia TWAP intra-fatia)
            "open_positions_count": len(self.positions),
            "total_risk_per_trade_pct": self.config.MAX_PORTFOLIO_RISK_PERCENT
        }

    def _calculate_trade_cost_impact(self, signal: Signal, trade_value_notional: float, is_maker_order: bool = False) -> float:
        """
        Simula o impacto total dos custos de um trade (taxas de trading e slippage).
        """
        if trade_value_notional <= 0:
            return 0.0

        # Seleciona a taxa apropriada (Maker ou Taker)
        fee_rate = self.config.MAKER_FEE if is_maker_order else self.config.TAKER_FEE
        
        # Taxa de trading
        trading_fee = trade_value_notional * fee_rate

        # Slippage é o custo adicional
        slippage_cost = trade_value_notional * self.config.PENALTY_SLIPPAGE 

        total_cost = trading_fee + slippage_cost
        
        return total_cost