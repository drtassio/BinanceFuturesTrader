# Bot na Oracle Cloud (VM Ubuntu, 24h)

VM recomendada: `VM.Standard.A1.Flex` (ARM), 2 OCPU / 12 GB, Ubuntu 24.04, regiao
Sao Paulo (a Binance bloqueia IPs dos EUA). A `E2.1.Micro` (1 GB) nao comporta o bot.

## Instalar (na VM, como `ubuntu`)

```
curl -fsSL https://raw.githubusercontent.com/drtassio/BinanceFuturesTrader/feat/staircase-teacher/deploy/oracle/setup.sh -o setup.sh
bash setup.sh
```

Isso instala Python/venv/pacotes (so CPU), baixa o codigo da branch
`feat/staircase-teacher`, testa a carga dos modelos e registra o servico
`bot-trader` no systemd (ainda desligado).

## Colocar o .env (chaves da Binance)

O `.env` nao vai pelo git. Crie-o na VM com o mesmo conteudo do .env do bot:

```
nano /home/ubuntu/BinanceFuturesTrader/.env     # colar o conteudo, salvar
chmod 600 /home/ubuntu/BinanceFuturesTrader/.env
```

## Ligar (um bot so por conta: desligue o do notebook antes)

```
bash setup.sh --start
```

## Operar

```
sudo systemctl status bot-trader          # esta rodando?
tail -f ~/BinanceFuturesTrader/logs/bot_service.log   # painel e logs
sudo systemctl restart bot-trader         # religar (ex.: depois de atualizar)
sudo systemctl stop bot-trader            # parar
bash setup.sh --start                     # atualizar codigo/modelos do git e religar
```

O grafico fica em `~/BinanceFuturesTrader/logs/charts/espelho.png`. O servico
religa o bot sozinho se ele cair e quando a VM reinicia.
