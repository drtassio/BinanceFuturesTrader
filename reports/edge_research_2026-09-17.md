# Pesquisa de edge por agente — 17/09/2026

Dados: `data/history_causal.parquet` (BTCUSDT perpétuo, 15m, fev/2020–set/2026, montado com o pipeline ao vivo).
Custos: 0,05% por lado (taker + slippage) e funding a cada 8h. Divisão: treino fev/2020–set/2024, validação
set/2024–set/2025, holdout out/2025–set/2026 (BTC −38%). Regras e parâmetros escolhidos **só no treino**.

## Bull — edge encontrado

| Hipótese | Treino | Validação | Holdout |
|---|---|---|---|
| Tendência longa (Donchian 480/240, stops ATR 4h) — professora | +94,3%, PF 2,60 | +15,1% | −3,2% |
| Agente de tendência (DAgger 2), aprovado | +83,7%, PF 4,41 | +18,9%, PF 4,84, DD 7,5% | −2,7%, PF 2,31 |
| **Rally com fluxo** (rompe topo 48h + expansão de volatilidade + agressão/CVD compradores, sai no fundo de 8h) — pesquisa vetorizada | +266%, Sharpe 1,43, DD 17% | +13%, Sharpe 0,76 | +1% |
| **Rally com fluxo** — professora no ambiente, stops ATR 1h | **+182,1%**, PF 4,02, DD 10,4% | **+17,4%**, PF 1,86, DD 8,0% | **+10,1%**, PF 2,47, DD 10,5% |

A regra de rally foi positiva em todos os anos (2020 +45%, 2021 +27%, 2022 +11%, 2023 +29%, 2024 +39%, 2025 +3%, 2026 +15%).
Sem a condição de fluxo, a mesma regra perdeu 5% na validação e 6% no holdout: o tape é o que dá o edge.

## Bear — sem edge validável

| Hipótese | Melhor no treino | Validação / holdout |
|---|---|---|
| Venda de tendência (Donchian 30–120 em 4h) | Sharpe 0,22, DD 41% | val −34% |
| Idem com filtro EMA50<EMA200 em 4h | pior que sem filtro | — |
| Funding extremo + rompimento | Sharpe 0,35, DD 53% | quase nenhum trade |
| Rompimento de fundo 12h–48h + expansão + fluxo vendedor (15m) | Sharpe 0,16, retorno ~0% | val −32% |
| Idem só com tendência maior de baixa (4h) | +77%, Sharpe 0,53, **DD 49%** | val −8%, holdout +1% |
| Venda da exaustão de alta forte / falso rompimento / RSI | Sharpe −0,16 | negativo |
| Meta-rotulagem por eventos de queda (gradient boosting, 52 features de fluxo e tendência) | AUC CV 0,514, −0,10%/trade | AUC 0,484 / 0,504, negativo |

No BTC 15m–4h de 2020–2026, quedas curtas tendem a reverter e a deriva de alta mais o funding penalizam o short.
Nenhum classificador sobre as features disponíveis separa as quedas que continuam (AUC ~0,5).
Hipótese para trabalho futuro: o edge das quedas acentuadas está em dados sem histórico aqui — liquidações,
open interest e profundidade do livro. O tape engine ao vivo poderia começar a gravá-los.

## Ranger — sem edge validável após custos

| Hipótese | Treino | Validação / holdout |
|---|---|---|
| Bollinger em mercado sem tendência (ADX baixo), 1h e 4h | Sharpe negativo | negativo |
| Reversão à média 15m em baixa eficiência, custo taker | −70% a −80% (bruto +0,01–0,02%/trade) | negativo |
| Idem com custo de ordem limitada (0,02%/lado) | +30%, Sharpe 0,53 | val 0%, holdout −6% (edge sumiu a partir de 2023) |
| Meta-rotulagem por eventos de extremo em lateralização | AUC CV 0,524, −0,02%/trade | −0,09% / −0,08% por trade |

Scripts: `scripts/research_event_edges.py` (relatório `reports/event_edges.json`); as grades vetorizadas ficaram
na área de rascunho da sessão e os números acima são os impressos por elas.
