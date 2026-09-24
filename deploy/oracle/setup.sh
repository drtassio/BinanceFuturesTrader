#!/usr/bin/env bash
# Instala o bot numa VM Ubuntu (Oracle Cloud), so CPU, e deixa pronto para o
# servico do systemd. Idempotente: pode rodar de novo para atualizar.
#
#   bash setup.sh              instala/atualiza e roda o checklist
#   bash setup.sh --start      alem disso liga o servico (exige o .env no lugar)
#
# O .env (chaves da Binance) NUNCA vai pelo git: copie-o para
# /home/ubuntu/BinanceFuturesTrader/.env antes de usar --start.
set -euo pipefail

REPO_URL="https://github.com/drtassio/BinanceFuturesTrader.git"
BRANCH="feat/staircase-teacher"
DIR="/home/ubuntu/BinanceFuturesTrader"
TORCH_VERSION="2.5.1"

echo "== 1/6 pacotes do sistema"
sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-dev build-essential pkg-config

MEM_MB=$(free -m | awk '/^Mem:/{print $2}')
if [ "$MEM_MB" -lt 3000 ]; then
  echo "ATENCAO: a VM tem ${MEM_MB} MB de RAM; o bot precisa de ~1.5 GB livres. Use uma A1.Flex com 8+ GB."
fi

echo "== 2/6 codigo ($BRANCH)"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch origin
  git -C "$DIR" checkout "$BRANCH"
  git -C "$DIR" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$DIR"
fi
git -C "$DIR" log --oneline -1

echo "== 3/6 ambiente Python (.venv)"
cd "$DIR"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel setuptools
# torch CPU (x86_64 pelo indice CPU; ARM usa a roda padrao do PyPI, que ja e CPU)
if [ "$(uname -m)" = "x86_64" ]; then
  .venv/bin/pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
else
  .venv/bin/pip install "torch==${TORCH_VERSION}"
fi
.venv/bin/pip install -r deploy/oracle/requirements-server.txt

echo "== 4/6 teste de importacao e dos modelos"
mkdir -p logs
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python - <<'PY'
import sys, json
sys.path.insert(0, ".")
import run_bot  # grafo de imports do bot
from pathlib import Path
from trading import agent_mirror as mirror
for name in ("bull", "bear"):
    agent, contract = mirror.load_specialist(name, Path("models_ai"))
    print("  %s carregado | stop ATR %s | trava 4h %s" % (name, contract.get("stop_atr_timeframe"), contract.get("short_requires_trend_4h", False)))
import joblib
joblib.load("models_ai/crypto_regime_detector.pkl"); joblib.load("models_ai/meta_labeler.joblib")
print("  detector de regime e meta-modelo carregados")
PY

echo "== 5/6 servico do systemd"
sudo cp deploy/oracle/bot-trader.service /etc/systemd/system/bot-trader.service
sudo systemctl daemon-reload
sudo systemctl enable bot-trader.service >/dev/null

echo "== 6/6 checklist da testnet"
if [ -f .env ]; then
  chmod 600 .env
  CUDA_VISIBLE_DEVICES=-1 .venv/bin/python scripts/check_testnet_ready.py || true
else
  echo "  .env ausente: copie-o para $DIR/.env (chmod 600) e rode de novo."
fi

if [ "${1:-}" = "--start" ]; then
  [ -f .env ] || { echo "sem .env: nao ligo o bot"; exit 1; }
  sudo systemctl restart bot-trader.service
  sleep 5
  systemctl --no-pager --lines=0 status bot-trader.service || true
  echo "log: tail -f $DIR/logs/bot_service.log"
fi
echo "pronto."
