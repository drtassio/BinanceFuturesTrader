# Suíte de validação científica

## Como rodar

```bash
# Todas as fases
python scripts/validate_pipeline.py

# Apenas a fase 6 (testes científicos)
python scripts/validate_pipeline.py --phase 6

# Pula a fase 7 (end-to-end)
python scripts/validate_pipeline.py --skip 7
```

Saída: `reports/validation/report.json` + log no stdout.
Exit code 0 se nenhuma fase falhou; SKIP não bloqueia.

## Fases

| # | Nome | O que verifica |
|---|---|---|
| 1 | Imports & Smoke | Todos os módulos importam; os 5 bugs P0 que corrigi não regrediram |
| 2 | Features | Indicadores nativos no range correto; ffill > fillna(0); look-ahead probe |
| 3 | Regime | `fit_predict(train_ratio=0.8)` sem leakage; concordância com regime sintético |
| 4 | CV Purgado | PurgedKFold sem overlap; WalkForward respeita ordem temporal |
| 5 | Especialistas | Bull/Bear/Ranger têm `_specialist_direction`; `reset()` limpa stats; PnL SHORT correto |
| 6 | Testes Científicos | DSR, White's RC, block-permutation, ADF, PBO — validados em estratégia com e sem skill |
| 7 | End-to-End | (precisa de `models_ai/` e `checkpoints_ai/` reais) |

## Para rodar a Fase 7 na sua máquina

Você precisa de:
- `models_ai/base_featured_df.pkl` (features pré-computadas)
- `checkpoints_ai/bull_specialist_sac.zip` (e bear/ranger)
- `models_ai/temporal_autoencoder.pth` (autoencoder treinado)

A fase carrega cada artefato, roda o detector de regime, instancia cada
especialista, e chama `decide_action` em uma janela real — sem postar ordens
e sem precisar de chaves da Binance.

## Referências dos testes científicos

- **Deflated Sharpe Ratio**: Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"
- **White's Reality Check**: White (2000), "A Reality Check for Data Snooping"
- **Bootstrap por blocos**: Politis & Romano (1994), "The Stationary Bootstrap"
- **PBO**: Bailey, Borwein, López de Prado (2014), "The Probability of Backtest Overfitting"
- **CV Purgado**: López de Prado (2018), *Advances in Financial Machine Learning*, Cap. 7
