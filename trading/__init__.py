# -----------------------------------------------------------------------------
# ARQUIVO: trading/__init__.py
# -----------------------------------------------------------------------------

"""
Pacote de Operações de Trading (Trading Operations).

Esta é a camada operacional do sistema, responsável por conectar todos os
componentes de IA e governança com o mercado ao vivo.

Componentes Principais:
- BinanceConnector: Interface de baixo nível para a API da Binance,
  gerenciando requisições, respostas e conexões WebSocket.
- AIController: O orquestrador central que consome dados de mercado, consulta
  a IA e produz um sinal de trading.
- HRLMaster: O agente de RL de alto nível que determina o regime de mercado
  e delega para os especialistas.
- ExecutionEngine: Recebe sinais de trading e os executa de forma inteligente
  usando algoritmos como TWAP e VWAP para minimizar o impacto no mercado.
- RiskManager: Monitora o risco do portfólio em tempo real, valida trades
  e pode acionar paradas de emergência.
- Portfolio: Gerencia o estado atual das posições, capital e PnL.
- OnChainEngine: Coleta e analisa dados on-chain para enriquecer a tomada de decisão.
- TapeEngine: Analisa o fluxo de ordens (market microstructure) em tempo real.
- Backtester: Simula estratégias com dados históricos.
- StressTester: Testa a robustez do sistema em cenários extremos.
- StateRestore: Gerencia a persistência e recuperação do estado do sistema.
"""

