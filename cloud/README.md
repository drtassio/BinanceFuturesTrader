# Treino remoto dos especialistas

Três agentes, três ambientes. Nenhum deles precisa de chave de API: o treino lê
apenas dados históricos públicos já versionados no repositório.

| Agente | Onde | Notebook | Observação |
|---|---|---|---|
| `bull` | Google Colab | `cloud/colab_train.ipynb` | lado com edge medido no holdout |
| `bear` | Kaggle | `cloud/kaggle_train.ipynb` | lado mais frágil, ver abaixo |
| `ranger` | o primeiro que liberar | qualquer um dos dois | mude `AGENT = "ranger"` |

> **Nunca** envie `.env`, chaves ou tokens para Colab ou Kaggle. Mantenha os
> notebooks privados.

## Como rodar

**Colab** — abra `cloud/colab_train.ipynb`, escolha `Ambiente de execução >
Alterar tipo > T4 GPU`, ajuste `AGENT` na primeira célula e execute tudo.

**Kaggle** — crie um notebook privado, importe `cloud/kaggle_train.ipynb`,
ligue *Internet* e *GPU* em Settings, ajuste `AGENT` e execute tudo.

Ambos fazem, na ordem: clone raso, dependências, reconstrução do dataset
causal, geração das features do meta-modelo, **verificação de causalidade**,
treino, veredito e empacotamento dos artefatos.

Se `scripts/verify_causality.py` falhar, o treino não deve acontecer. Esse
script existe porque o dataset anterior expunha o fechamento de um candle de 4h
até 3h45 antes de ele existir, o que tornava sem sentido qualquer métrica.

## O pipeline

```
featured_data.parquet  +  tape (aggressor/trade_count/quote_volume)  +  funding
        |
        v
scripts/build_causal_dataset.py        candles 1h/4h deslocados para o fechamento,
        |                              31 features de tendência, 15 de tape
        v
scripts/build_meta_features.py         barreira tripla + gradient boosting
        |                              walk-forward -> ml_p_long, ml_p_short,
        |                              ml_edge, ml_conf
        v
scripts/verify_causality.py            portão: falha se algo enxerga o futuro
        |
        v
cloud/train_agent.py                   SAC decide tamanho, momento e saída
```

O modelo supervisionado estima *se existe vantagem agora*; o agente decide *o
que fazer com ela*. Separar as duas coisas é o que tira do SAC a tarefa de
achar sinal em 400 colunas ruidosas com 90 mil barras.

## Divisão dos dados

```
treino 70%   |  embargo 768 barras  |  validação 15%  |  embargo  |  holdout 15%
```

O holdout é lido **uma única vez**, no fim. Usá-lo para early stopping o
transformaria em segundo conjunto de validação e todo número sairia otimista.

## O veredito

Um modelo só é marcado como apto a operar se passar em cinco critérios:

- a política **determinística** abre pelo menos 10 posições no holdout
- retorno líquido positivo
- melhor que comprar e segurar no mesmo período
- drawdown máximo abaixo de 35%
- desvio do voto acima de 0,01, isto é, a política não colapsou numa decisão só

O primeiro e o último critério existem por um motivo concreto: uma execução
anterior reportou fator de lucro 3,8 enquanto sua política determinística nunca
abriu posição alguma. Quem negociava era o ruído de exploração, e o bot ao vivo
sempre age de forma determinística.

## Expectativas honestas

Medido no holdout de 2025-09 a 2026-03 (comprar e segurar rendeu **−39%**):

- entrar em toda barra com a geometria de barreira usada: **−0,183%** por trade
- entrar apenas onde o meta-modelo aprova: **+0,17%** por trade
- AUC do meta-modelo no holdout: **0,574** (treino 0,725)

É vantagem real, medida fora da amostra e depois de custos, mas é **fina**. O
lado short tem ordenação instável neste histórico, que é quase todo de alta:
espere que o `bear` reprove no veredito com mais frequência que o `bull`. Isso
é o sistema funcionando, não falhando.

## Retomar um treino interrompido

```bash
python cloud/train_agent.py --agent bull --resume --run-directory cloud/artifacts/bull_<timestamp>
```
