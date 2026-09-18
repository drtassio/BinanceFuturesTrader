# Relatório — padrões do BTC nos últimos 2 anos e como ensinar os agentes

Data: 17/09/2026. Script: `scripts/research_recent_legs.py`. Números completos: `reports/recent_research.json`.

## 1. O que o bot precisa fazer

- **Bull** entra no fundo, quando a subida começa, segura as pequenas correções e sai no topo.
- **Bear** entra no topo, quando a queda começa, segura os pequenos repiques e sai no fundo.
- **Ranger** opera oscilações pequenas com trades curtos que pagam os custos.
- **Movimento grande** é para o Bull ou o Bear; **movimento pequeno** é para o Ranger.
- **O bot troca os agentes nas viradas**, e **trocar não significa operar**: sem oportunidade com vantagem, ele fica de fora. "Sem oportunidade" é um estado válido.

## 2. Como a pesquisa foi feita

- **Dados:** só BTCUSDT perpétuo, candles de 15m, de 17/09/2024 a 17/09/2026 (70.099 candles), montados pelo mesmo pipeline do bot ao vivo.
- **Divisão cronológica:**
  - treino: set/2024–fev/2026 (BTC +19,7%);
  - validação: fev–jun/2026 (BTC −8,0%);
  - holdout: jun–set/2026 (BTC +13,5%).
- **Custos:** 0,05% por lado a mercado (0,02% na variante com ordem limitada), mais funding a cada 8h.
- **Escolha:** regras e parâmetros escolhidos **só no treino**; os modelos são retreinados a cada trimestre com o passado.

### O mercado mudou

A amplitude diária média caiu de 4–6% (2020–2022) para ~3% (2023–2026), e o ATR de 1h caiu de ~1,0–1,4% para ~0,6%. Com o mesmo custo por trade, pernas menores deixam menos lucro, e padrões de 2020–2021 deixaram de valer. Por isso a pesquisa usa só o mercado recente.

## 3. Resultados

### Bull

| Abordagem | Treino | Validação | Holdout |
|---|---|---|---|
| Regra de perna de alta (rompe topo + expansão + fluxo comprador) | +7%, Sharpe 0,46 | −4% a −8% | +4% a +16% |
| **Modelo walk-forward de 24h: comprar quando a previsão está nos 10% mais altos** | — | lucro líquido por trade positivo em **5 de 5 trimestres** fora da amostra: +0,64%, +0,39%, +0,52%, +0,12%, +0,23% | — |

O modelo de 24h é o sinal comprador mais consistente do período recente. Ele lucrou inclusive nos trimestres em que o BTC caiu 22–23%, ou seja, reconheceu as subidas dentro das quedas. **Ainda falta** simular com entradas e saídas reais, sem sobreposição de posições, antes de virar professora.

### Bear

| Abordagem | Treino | Validação | Holdout |
|---|---|---|---|
| **Perna de queda com tendência de 4h de baixa** (rompe o fundo de 24h, só se EMA50 < EMA200 em 4h; sai no topo de 8h ou trailing de 2,5 ATR) | **+24%, Sharpe 0,83, DD 14%**, 97 trades | **+10%, Sharpe 1,68, DD 9%** | −4%, DD 10% (BTC +13,5%) |
| Venda da exaustão de alta forte | ~0% | −7% | −12% |
| Meta-rotulagem por eventos de queda | AUC 0,535 | AUC 0,489 | AUC 0,416 |

Pela primeira vez o Bear tem um candidato positivo no treino **e** na validação. Por trimestre: 2026-T1 +11% e 2026-T2 +7%, nos períodos de queda, e perdas pequenas nas altas. No holdout, com o BTC subindo 13,5%, perdeu 4%, o esperado para um especialista vendido fora do seu tipo de mercado.

### Ranger

| Abordagem | Treino | Validação | Holdout |
|---|---|---|---|
| Reversão à média em lateralização, custo a mercado | −10% | −5% | ~0% |
| Idem com ordem limitada | −2% | −3% | +2% |
| Meta-rotulagem por eventos | AUC 0,512 | AUC 0,485 | AUC 0,516 |

**Sem vantagem validada.** As pernas pequenas do mercado atual não pagam os custos de forma consistente. Sem oportunidade, o Ranger fica desligado.

### Indicadores isolados

**Nenhum** dos 82 indicadores do BTC manteve o mesmo sinal nos 9 trimestres. Nenhum indicador sozinho anuncia as pernas; o padrão está na **combinação** deles, que é o que o modelo walk-forward aprende.

### Reconhecimento de fundos e topos

| Trimestre | AUC fundo | AUC topo | Operar só com isso: Bull / Bear | BTC |
|---|---|---|---|---|
| 2025-T3 | 0,82 | 0,76 | +3,0% / −9,2% | +6% |
| 2025-T4 | 0,84 | 0,87 | +1,3% / +25,9% | −23% |
| 2026-T1 | 0,85 | 0,84 | −17,8% / −2,6% | −22% |
| 2026-T2 | 0,92 | 0,84 | −12,4% / −9,8% | −14% |
| 2026-T3 | 0,83 | 0,89 | −2,1% / −6,9% | +30% |

O modelo **reconhece muito bem quando o preço está numa zona de fundo ou de topo** (AUC 0,76–0,92). Mas operar só com isso perde dinheiro, porque ele não distingue quais fundos e topos viram pernas grandes. Serve como **prontidão para a virada**, não como gatilho de entrada.

## 4. Como ensinar os agentes da melhor maneira

1. **Um professor por agente, só com vantagem comprovada.** O agente aprende a imitar um professor (clonagem + DAgger); o RL só fica se melhorar a validação. Sem professor com vantagem, o agente não opera.
   - **Bull:** entrada pelo modelo walk-forward de 24h (previsão nos 10% mais altos), saída de perna (perde o fundo de 8h ou trailing). Primeiro validar com simulação completa.
   - **Bear:** perna de queda com tendência de 4h de baixa e saída no topo de 8h.
   - **Ranger:** desligado até existir vantagem que pague os custos (próximo teste: execução com ordem limitada no bot ao vivo).
2. **O bot (orquestrador) tem quatro estados:** Bull, Bear, Ranger e **sem oportunidade**.
   - A **prontidão** vem do reconhecimento de fundos e topos (AUC 0,76–0,92): perto de um topo, o bot fica pronto para uma possível queda; perto de um fundo, para uma possível subida.
   - O **gatilho** é sempre a condição de entrada do agente. Sem gatilho, fica de fora.
   - Com sinais em conflito, fica de fora.
3. **Retreino trimestral só com o passado recente,** porque o mercado muda e o modelo precisa acompanhar.
4. **Avaliação do ciclo completo numa conta única:** Bull → sem oportunidade/Bear/Ranger → Bull, com custos, em validação e holdout, e critério de aprovação escrito antes do resultado.
5. **Paridade com o bot ao vivo:** o mesmo espelho já validado (replay no ambiente e mesma posição na corretora) vale para cada agente e para o orquestrador.
