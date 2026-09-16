# Treino remoto dos agentes

Esta pasta contém o mesmo comando para Colab, Kaggle e uma máquina GPU local.
O treino usa apenas `data/featured_data.parquet`; não copie `.env`, chaves ou
tokens para nenhuma plataforma.

## Colab

1. Abra um notebook Python, escolha `Runtime > Change runtime type > GPU` e clone ou envie o repositório.
2. Execute:

```bash
pip install -r cloud/requirements-cloud.txt
python cloud/train_agent.py --agent bull --timesteps 100000
```

Baixe `cloud/artifacts/bull_training_report.json` e o modelo criado em `models_ai/`.

## Kaggle

Crie um Notebook privado, adicione o dataset/repositório como entrada e selecione
GPU em Settings > Accelerator. Execute o mesmo comando com `--agent bear`.

## Terceiro agente

Execute `--agent ranger` no primeiro ambiente que terminar, ou em outra sessão
de Colab/Kaggle. `--resume` permite continuar a partir de um checkpoint salvo.

O holdout temporal é mantido separado do treino. O modelo só deve voltar ao bot
depois de apresentar retorno líquido positivo, drawdown aceitável e resultado
melhor que ficar parado no holdout.
