# Como rodar o bot em outro PC (testnet)

Branch: `feat/staircase-teacher`.

## 1. Baixar o projeto

```
git clone https://github.com/drtassio/BinanceFuturesTrader.git
cd BinanceFuturesTrader
git checkout feat/staircase-teacher
```

## 2. Python e pacotes

Use Python 3.10 (o bot foi treinado e testado com 3.10.9 e scikit-learn 1.6.1;
com outra versao do scikit-learn o scaler do Bull nao carrega e o bot se recusa
a iniciar).

```
py -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
py -m pip install -r requirements-bot.txt
```

Sem placa NVIDIA, instale `torch==2.5.1` sem o `--index-url`.

## 3. Chaves

Copie `.env.example` para `.env` e preencha as chaves da Binance (producao so
para ler dados; testnet para as ordens). O `.env` nunca vai para o git.

## 4. Conferir e ligar

```
py scripts/check_testnet_ready.py
$env:PYTHONIOENCODING="utf-8"; [Console]::OutputEncoding=[Text.Encoding]::UTF8; py run_bot.py
```

Rode um bot so por conta: dois bots na mesma conta duplicam as ordens.
Na primeira vez o bot reconstroi o historico de candles (cerca de 30 s).

## 5. Como ler o terminal

A cada candle de 15m fechado aparece o painel **ESPELHO DOS AGENTES**:

- Cada agente (agente LONG = Bull, aprovado; agente SHORT = Bear, diagnostico) mostra a
  posicao que tem **na simulacao** do ambiente de treino: FORA, COMPRADO ou
  VENDIDO, desde quando, preco de entrada, stop e resultado.
- **CONTA (Binance)**: a posicao real na testnet.
- **ACAO DO BOT** e **POR QUE**: o que o bot faz neste candle.
- A ordem sai quando um agente mostra **★ NOVA ENTRADA** (entrou no candle que
  acabou de fechar). Se o agente ja estava na posicao antes de o bot ligar, o
  bot nao entra atrasado e espera a proxima entrada.
- Logo abaixo do painel sai o grafico dos candles de 15m das ultimas 24h, em
  cores: fundo verde onde o agente LONG estava comprado, vermelho onde o agente
  SHORT estava vendido, ▲ ▼ entradas, ✖ saidas e a linha do preco atual. A mesma
  janela de 72h em imagem fica em `logs/charts/espelho.png`.
- Entre um candle e outro aparece uma linha curta `🪞 [ESPELHO] ...`.

Regime, tape, OBI, sentimento (Fear & Greed) e SHAP aparecem no status, mas sao
so informativos: nao decidem a ordem.
