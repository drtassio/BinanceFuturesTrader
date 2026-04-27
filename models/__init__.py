# -----------------------------------------------------------------------------
# ARQUIVO: models/__init__.py
# -----------------------------------------------------------------------------

"""
Pacote de Schemas de Dados e Modelos de Banco de Dados.

Este pacote define a estrutura de dados utilizada em todo o sistema,
garantindo consistência e clareza. Ele contém:

- Schemas de Dados (dataclasses): Para objetos em memória como Posições, Ordens,
  e Sinais de Trading. Isso proporciona type-hinting e validação.

- Modelos de Banco de Dados (SQLAlchemy): Para a persistência de dados
  históricos, como resultados de backtests, trades executados e métricas de
  performance.

A exposição das classes principais aqui permite importações mais limpas,
como `from models import Trade, Position`.
"""


