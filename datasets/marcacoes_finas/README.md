# Marcações finas — BTCUSDT perp 15m, 18/09/2024 a 18/09/2026

Todas as pernas do zigue-zague dos últimos 2 anos, marcadas com entrada e saída
como um trader marcaria olhando o gráfico pronto (semana de exemplo:
`graficos/semana_20250716.png`, 25 pernas).

| | Bull (long) | Bear (short) |
|---|---|---|
| Pernas | 1.380 | 1.368 |
| Movimento mediano | 1,55% | 1,55% |
| Capturado líquido mediano (após 0,14% de custo) | +0,54% | +0,52% |
| Soma capturada | +1.116% | +1.068% |
| Acerto | 84,5% | 84,6% |
| Duração mediana | 3,8 h | 3,4 h |
| Tempo posicionado | 43% | 39% |

> **Atenção:** as marcações sabem onde cada perna termina. São **exemplos para
> ensinar os agentes**, não uma estratégia que o bot consiga reproduzir ao vivo.
> Agentes que só imitaram estas marcações perderam na validação (PF 0,6 a 0,8);
> o treino atual testa se o reforço (SAC) consegue extrair lucro delas.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `todas_pernas.parquet` | uma linha por perna: `side`, `start_time`, `entry_time`, `extreme_time`, `exit_time`, preços, `leg_move`, `captured_net`, `hours` |
| `rotulos_por_candle.parquet` | um candle de 15m por linha: `hs_target_position` (+1 long, −1 short, 0 fora), `hs_phase` (inicio, entrada, surf, saida, fora), `hs_leg_id` |
| `bull/pernas_long.parquet`, `bear/pernas_short.parquet` | só as pernas de cada lado |
| `bull/rotulos_long.parquet`, `bear/rotulos_short.parquet` | rótulo por candle só do lado (o outro lado vira 0) |
| `bull/regra.json`, `bear/regra.json` | professor usado nos treinos (`marked_legs`) |
| `bull/resumo.json`, `bear/resumo.json` | os números da tabela acima |
| `graficos/` | 105 semanas com as entradas (▲ long, ▼ short) e saídas (✖); abra `graficos/index.html` |

## Como foram marcadas

`scripts/mark_legs.py --tag fino --swing 0.006 --leg-min 0.009 --speed 0.0005 --confirm 0.0025`

1. Um zigzag divide o preço em ondas com reversão de pelo menos 0,6%.
2. Uma onda é **perna** quando anda pelo menos 0,9% a pelo menos 0,05% por hora;
   ondas menores ou lentas ficam como lateral (fora).
3. **Entrada:** primeiro fechamento 0,25% além do início da perna (o movimento já começou).
4. **Saída:** primeiro fechamento 0,25% de volta a partir do extremo da perna.

## Regerar

```
python scripts/mark_legs.py --tag fino --swing 0.006 --leg-min 0.009 --speed 0.0005 --confirm 0.0025
python scripts/export_marked_legs.py
```
