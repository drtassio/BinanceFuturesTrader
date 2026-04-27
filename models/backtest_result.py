# -----------------------------------------------------------------------------
# ARQUIVO: models/backtest_result.py
# -----------------------------------------------------------------------------

"""
Define o modelo de banco de dados (SQLAlchemy) para armazenar os resultados
detalhados de cada execução de backtest.

Esta tabela é crucial para a análise de performance, comparação de estratégias
e rastreabilidade das versões dos modelos de IA.
"""
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime
import json # Importar json para trabalhar com campos JSON como 'symbols'

# Base declarativa para os modelos SQLAlchemy.
# Todos os modelos de tabela herdarão desta Base.
Base = declarative_base()

class BacktestResult(Base):
    """
    Modelo SQLAlchemy para a tabela 'backtest_results'.
    Armazena um resumo completo e as métricas de performance de uma simulação de backtest.
    """
    __tablename__ = 'backtest_results' # Nome da tabela no banco de dados

    # --- Identificação e Contexto do Backtest ---
    id = Column(Integer, primary_key=True, autoincrement=True, comment="Chave primária e ID único incremental para o registro do backtest.")
    test_id = Column(String(64), unique=True, nullable=False, index=True, comment="ID único gerado para esta execução de backtest (e.g., 'bt_uuid'). Essencial para rastreabilidade.")
    model_version = Column(String(64), nullable=False, comment="Versão ou identificador do modelo de IA ou estratégia que foi testada.")
    start_date = Column(DateTime, nullable=False, comment="Data de início do período histórico utilizado no backtest.")
    end_date = Column(DateTime, nullable=False, comment="Data de fim do período histórico utilizado no backtest.")
    
    # Usamos TEXT para 'symbols' pois pode ser uma lista JSON de múltiplos símbolos.
    # Garante flexibilidade para múltiplos pares no futuro.
    symbols = Column(Text, nullable=False, comment="Lista de símbolos de ativos negociados durante o backtest, armazenados em formato JSON (e.g., '[\"BTCUSDT\", \"ETHUSDT\"]').")
    
    timeframe = Column(String(8), nullable=False, comment="Timeframe dos dados de mercado usados (e.g., '1m', '1h', '4h', '1d').")
    
    # 'parameters' pode ser um JSON mais complexo, como hiperparâmetros específicos da IA ou estratégia.
    parameters = Column(JSON, comment="Dicionário de hiperparâmetros e configurações específicas da estratégia ou modelo usados neste teste, em formato JSON.")
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, comment="Timestamp UTC de quando este registro de resultado de backtest foi criado e salvo no banco de dados.")

    # --- Métricas de Retorno Financeiro ---
    initial_capital = Column(Float, nullable=False, comment="Capital inicial (em USDT ou moeda base) com o qual o backtest começou.")
    final_capital = Column(Float, nullable=False, comment="Capital final (em USDT ou moeda base) ao término do backtest.")
    total_return_pct = Column(Float, nullable=False, comment="Retorno percentual total sobre o capital inicial (final - inicial) / inicial * 100.")
    annualized_return_pct = Column(Float, nullable=False, comment="Retorno percentual anualizado, representando o desempenho em base anual.")

    # --- Métricas de Risco de Portfólio ---
    max_drawdown_pct = Column(Float, nullable=False, comment="Maior queda percentual do valor do portfólio em relação ao seu pico anterior.")
    volatility_pct = Column(Float, nullable=False, comment="Volatilidade anualizada dos retornos do portfólio em porcentagem.")
    var_95_pct = Column(Float, comment="Value at Risk (VaR) de 95% diário em porcentagem. Representa a perda máxima esperada em 95% dos dias.")
    cvar_95_pct = Column(Float, comment="Conditional VaR (CVaR) de 95% diário em porcentagem. Representa a perda média das 5% piores perdas.")

    # --- Ratios de Performance Ajustados ao Risco ---
    sharpe_ratio = Column(Float, comment="Sharpe Ratio: Mede o retorno excedente por unidade de risco (desvio padrão).")
    sortino_ratio = Column(Float, comment="Sortino Ratio: Similar ao Sharpe, mas usa apenas o desvio padrão de retornos negativos (downside deviation).")
    calmar_ratio = Column(Float, comment="Calmar Ratio: Mede o retorno anualizado em relação ao drawdown máximo. Maior é melhor.")

    # --- Estatísticas Detalhadas de Trades ---
    total_trades = Column(Integer, nullable=False, comment="Número total de operações de compra e venda realizadas durante o backtest.")
    winning_trades = Column(Integer, nullable=False, comment="Número de trades que foram fechados com lucro.")
    losing_trades = Column(Integer, nullable=False, comment="Número de trades que foram fechados com prejuízo.")
    win_rate_pct = Column(Float, nullable=False, comment="Taxa de acerto em porcentagem: (winning_trades / total_trades) * 100.")
    avg_trade_pnl_pct = Column(Float, comment="Lucro ou prejuízo médio por trade, em porcentagem do capital inicial por trade.")
    profit_factor = Column(Float, comment="Fator de lucro: (Lucro Bruto Total) / (Prejuízo Bruto Total). Um valor > 1 indica lucratividade.")
    avg_holding_period_hours = Column(Float, comment="Tempo médio que as posições foram mantidas abertas, em horas.")

    # --- Dados Detalhados para Análise (armazenados como JSON para flexibilidade) ---
    # Estes campos podem ser grandes e são usados para visualizações e análises mais profundas.
    equity_curve = Column(JSON, comment="Série temporal completa do valor do portfólio (equity) ao longo do backtest, em formato JSON (lista de dicts {timestamp, equity}).")
    trade_log = Column(JSON, comment="Log detalhado de cada trade individual executado durante o backtest, em formato JSON (lista de dicts com detalhes do trade).")
    monthly_returns = Column(JSON, comment="Retornos agregados mensalmente para o portfólio, em formato JSON.")

    def __repr__(self) -> str:
        """
        Representação em string do objeto para fácil depuração em logs.
        Fornece um resumo conciso das informações mais importantes do backtest.
        """
        return (f"<BacktestResult(id={self.id}, test_id='{self.test_id}', model='{self.model_version}', "
                f"start={self.start_date.strftime('%Y-%m-%d')}, end={self.end_date.strftime('%Y-%m-%d')}, "
                f"return={self.total_return_pct:.2f}%, drawdown={self.max_drawdown_pct:.2f}%, "
                f"sharpe={self.sharpe_ratio:.2f})>")

    def to_dict(self) -> Dict[str, Any]:
        """
        Converte o objeto BacktestResult em um dicionário Python.
        Útil para serialização em JSON ou para exibição.
        """
        data = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        # Converter objetos datetime para string ISO 8601
        for key in ['start_date', 'end_date', 'created_at']:
            if data[key]:
                data[key] = data[key].isoformat()
        
        # JSON fields will already be dicts/lists when loaded by SQLAlchemy's JSON type
        # but if this is a fresh object before saving, they might be strings.
        # Ensure 'symbols' is treated as a list if it's still a JSON string
        if isinstance(data['symbols'], str):
            try:
                data['symbols'] = json.loads(data['symbols'])
            except json.JSONDecodeError:
                pass # Keep as string if not valid JSON

        return data