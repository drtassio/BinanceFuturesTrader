# -----------------------------------------------------------------------------
# ARQUIVO: run_bot.py
# DESCRIÇÃO: Versão final, completa e validada para operar de forma autônoma,
# com logging, retreinamento automático e verificação de condições para live trading.
# -----------------------------------------------------------------------------

import os
import asyncio
import json
import warnings
import csv
import signal
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
# --- Configurações e Utilitários ---
from config.settings import active_config, TradingConfig, AIConfig, DataConfig
from utils.logger import get_logger

# --- Modelos de Dados ---
from models.trade_schema import Action

# --- Componentes Principais do Bot ---
from trading.binance_connector import BinanceConnector, shutdown_event
from trading.portfolio import PortfolioOptimizer
from trading.risk_manager import RiskManager
from trading.execution_engine import ExecutionEngine
from trading.onchain_engine import OnChainEngine
from trading.tape_engine import TapeEngine
from trading.ai_controller import AIController
from trading.state_restore import StateRestore
from trading.ollama_narrator import OllamaNarrator
from trading.telegram_notifier import TelegramNotifier
from data_provider import DataProvider
from feature_engineering.main import FeatureEngineeringPipeline
from governance.ai_monitor import AIMonitor

# --- Environment Hardening (Audit Fix P0) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)
# Prevent 'init_gesdd failed init' and other MKL/BLAS conflicts on Windows
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=r".*precision lowered by casting to float32.*")  # gymnasium Box, inofensivo

# ============================================================
# 🚀 GPU INITIALIZATION: Configura GPU antes de qualquer modelo
# ============================================================
try:
    import torch as _torch_gpu_init
    if _torch_gpu_init.cuda.is_available():
        # Força inicialização do contexto CUDA no processo principal
        _torch_gpu_init.cuda.set_device(0)
        _dummy = _torch_gpu_init.zeros(1, device='cuda:0')
        del _dummy
        # Otimizações de backend para treinamento
        _torch_gpu_init.backends.cuda.matmul.allow_tf32 = True
        _torch_gpu_init.backends.cudnn.allow_tf32 = True
        _torch_gpu_init.backends.cudnn.benchmark = True
        _torch_gpu_init.backends.cudnn.deterministic = False
        _gpu_name = _torch_gpu_init.cuda.get_device_name(0)
        _vram = _torch_gpu_init.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n🚀 [GPU INIT] CUDA inicializado com sucesso!")
        print(f"   GPU: {_gpu_name}")
        print(f"   VRAM: {_vram:.1f} GB")
        print(f"   TF32: ON | cuDNN Benchmark: ON")
        print(f"   Todos os modelos de RL treinarão na GPU!\n")
    else:
        print("\n⚠️ [GPU INIT] CUDA NÃO disponível! Usando CPU (treinamento será lento).\n")
except Exception as _gpu_init_err:
    print(f"\n⚠️ [GPU INIT] Erro ao inicializar CUDA: {_gpu_init_err}\n")

# --- Configuração Inicial ---
logger = get_logger("run_bot_main")
LOGS_DIR = os.path.join(os.getcwd(), "logs")
TRADES_LOG_FILE = os.path.join(LOGS_DIR, "trades_log.csv")

# --- Estado e Componentes Globais ---
system_state: Dict[str, Any] = {
    "status": "stopped", "trading_active": False, "training_active": False,
    "binance_connection": "desconectado", "last_explanation": {},
    "live_trading_enabled": False, "ai_status": "inicializando",
    "system_health": "unknown", "drift_status": "desconhecido",
    "onchain_pulse": {"signal": "NEUTRAL", "confidence": 0.0},
    "tape_pulse": {"pulse": "NEUTRAL", "confidence": 0.0},
    "recent_trades": [],  # Lista para armazenar trades recentes para o ExecutionEngine
}
system_components: Dict[str, Any] = {}
background_tasks: List[asyncio.Task] = []
training_trade_logs: List[Dict[str, Any]] = []


# --- Funções de Logging e Persistência ---

def save_trade_to_log(trade_data: Dict):
    """Salva os detalhes de um trade em um arquivo CSV para auditoria e retreinamento."""
    file_exists = os.path.isfile(TRADES_LOG_FILE)
    try:
        with open(TRADES_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            fieldnames = [
                'timestamp', 'symbol', 'action', 'quantity', 'price', 'status',
                'leverage', 'profit_probability', 'notional_value', 'order_id', 'client_order_id'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade_data)
        logger.info(f"💾 [AUDITORIA] Trade salvo: {trade_data.get('symbol')} {trade_data.get('action')}")
    except Exception as e:
        logger.error(f"❌ [AUDITORIA] Falha ao salvar trade no log CSV: {e}", exc_info=True)


def log_portfolio_status(portfolio_name: str, portfolio_data: Dict):
    """Registra o status de um portfólio no log."""
    if not portfolio_data: return
    pnl_char = "🟢" if portfolio_data.get('unrealized_pnl', 0) >= 0 else "🔴"
    lev_char = "🟢" if portfolio_data.get('total_leverage_ratio', 0) < 2 else "🟡" if portfolio_data.get('total_leverage_ratio', 0) < 5 else "🔴"
    
    # [MELHORIA] Exibir alavancagem máxima configurada nas posições
    positions = portfolio_data.get('positions', [])
    max_lev = 0
    if positions:
        # Pode ser lista de dicts ou objetos Position (se chamado internamente), mas get_detailed_status retorna dicts
        try:
             max_lev = max(p.get('leverage', 1) for p in positions)
        except: max_lev = 1

    lev_str = f"{portfolio_data.get('total_leverage_ratio', 0):.2f}x"
    if max_lev > 1:
        lev_str += f" (Max: {max_lev}x)"

    divider = "─" * 60
    log_message = (
        f"\n💰 {divider}\n"
        f"   🏦 PORTFÓLIO: {portfolio_name.upper()}\n"
        f"   {divider}\n"
        f"   💵 Equity:        ${portfolio_data.get('total_value', 0):>12,.2f}\n"
        f"   📈 PnL Aberto:    {pnl_char} ${portfolio_data.get('unrealized_pnl', 0):>10,.2f}\n"
        f"   💸 Caixa Disp:    ${portfolio_data.get('cash', 0):>12,.2f}\n"
        f"   🛡️ Margem Usada:  ${portfolio_data.get('margin_used', 0):>12,.2f}\n"
        f"   ⚡ Alavancagem:   {lev_char} {lev_str:<10} | Posições: {portfolio_data.get('positions_count', 0)}\n"
        f"💰 {divider}\n"
    )
    logger.info(log_message)


def log_system_status():
    """Registra um resumo geral do status operacional do sistema."""
    if not system_state['live_trading_enabled']:
        mode = "PAPER TRADING (SIMULADO)"
    elif TradingConfig.BINANCE_TESTNET:
        mode = "TESTNET (ordens na testnet, dinheiro de teste)"
    else:
        mode = "PRODUCAO (DINHEIRO REAL)"

    # --- Regime atual (via tape_pulse ou latest_features) ---
    tape = system_state.get('tape_pulse', {})
    tape_pulse_str = tape.get('pulse', 'NEUTRAL')
    tape_conf_str  = f"{tape.get('confidence', 0.0):.0%}"

    # Regime do modelo (a partir do último feature calculado)
    latest_feat = system_state.get('latest_features')
    regime_map  = {0: "BULL 🐂", 1: "BEAR 🐻", 2: "RANGER 🤠"}
    if latest_feat is not None and hasattr(latest_feat, 'get'):
        regime_code = int(latest_feat.get('regime', 2))
        regime_conf = float(latest_feat.get('regime_confidence', 0.5))
        regime_str  = f"{regime_map.get(regime_code, 'DESCONHECIDO')} ({regime_conf:.0%})"
    else:
        regime_str = "AGUARDANDO DADOS..."

    divider = "─" * 60
    tape_score = tape.get('score', 0.0)
    info, policy_line = "", ""
    if _mirror_policy():
        agents = [a.strip() for a in str(getattr(TradingConfig, 'LIVE_AGENTS', '')).split(',') if a.strip()]
        policy_line = "   🪞 POLITICA:    espelho dos agentes: %s\n" % ", ".join(AGENT_NAMES.get(a, a) for a in agents)
    log_message = (
        f"\n💡 {divider}\n"
        f"   📊 STATUS GERAL ({datetime.now().strftime('%H:%M:%S')})\n"
        f"   {divider}\n"
        f"   🚀 MODO:        {mode}\n"
        f"{policy_line}"
        f"   🏥 SAÚDE:       {system_state['system_health'].upper()} | CONEXÃO: {system_state['binance_connection'].upper()}\n"
        f"   🧠 IA STATUS:   {system_state['ai_status'].upper()} | REGIME: {regime_str}{info}\n"
        f"   📟 TAPE PULSE:  {tape_pulse_str:<10} (Score: {tape_score:+.2f}){info}\n"
        f"   🧠 SENTIMENT:    {system_state['onchain_pulse']['signal']} ({system_state['onchain_pulse'].get('score', 50)}){info}\n"
        f"💡 {divider}\n"
    )
    logger.info(log_message)


# --- Painel do espelho (LIVE_POLICY=agent_mirror) ---------------------------
# No espelho quem decide sao os agentes Bull e Bear repassados no ambiente de
# treino; regime, tape, sentimento e SHAP nao entram na ordem. O painel mostra
# o que decide: a posicao de cada agente na simulacao, a conta e o que o bot faz.
_MIRROR_REASONS = {
    "trade do agente ja em curso: nao persegue":
        "o agente entrou ANTES (bot desligado ou candle anterior). Entrar agora seria atrasado, "
        "com preco e stop diferentes do testado: aguardando a proxima entrada nova.",
    "agente saiu da posicao": "o agente SAIU na simulacao: fechando a posicao da conta (reduceOnly).",
    "sombras em conflito: zerar": "Bull e Bear em lados opostos: zerando a conta por seguranca.",
    "sombras em conflito: ficar de fora": "Bull e Bear em lados opostos: ficando de fora.",
    "ordem desta barra ja enviada; aguardando execucao": "ordem deste candle ja enviada: aguardando execucao.",
    "espelho indisponivel": "espelho ainda nao carregado.",
    "historico incompleto": "historico de candles incompleto: reconstruindo antes de decidir.",
}
_last_mirror_panel = {"key": None}
# Nome mostrado ao usuario: o Bull opera so comprado e o Bear so vendido.
AGENT_NAMES = {"bull": "agente LONG", "bear": "agente SHORT"}


def _mirror_policy() -> bool:
    return str(getattr(TradingConfig, "LIVE_POLICY", "sac")).strip().lower() == "agent_mirror"


def _fear_greed_text(score) -> str:
    """Indice Fear & Greed com o nome da faixa (o sinal BEARISH do bot e leitura contraria)."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "indisponível"
    label = ("medo extremo" if value < 25 else "medo" if value < 45 else "neutro" if value <= 55
             else "ganância" if value < 75 else "ganância extrema")
    return "%d (%s)" % (round(value), label)


def _mirror_reason_text(view) -> str:
    reason = str(view.get("reason") or "")
    if reason.startswith("agente ") and reason.endswith(" entrou"):
        agent = AGENT_NAMES.get(reason.split()[1], reason.split()[1]).upper()
        return "NOVA ENTRADA do %s no candle que acabou de fechar: abrindo a mesma posicao na conta." % agent
    if reason.startswith("vetado pelo risco"):
        return "entrada VETADA pelo gestor de risco (%s)." % reason.split(":", 1)[-1].strip()
    if reason == "posicao igual a do agente" and view.get("account_side", 0) == 0:
        return "nenhum agente em posicao: aguardando uma escada confirmada (2+ degraus) para entrar."
    if reason == "posicao igual a do agente":
        return "conta igual a simulacao: mantendo a posicao; o stop na corretora so aperta."
    return _MIRROR_REASONS.get(reason, reason or "sem motivo")


def log_mirror_panel(ai_controller, signal) -> None:
    """Painel por candle: posicao de cada agente, conta e acao do bot."""
    view = getattr(ai_controller, "mirror_view", None) if ai_controller else None
    if not view:
        reason = (signal.explanation or {}).get("reason", "N/A") if signal else "N/A"
        logger.info("🪞 [ESPELHO] %s", _MIRROR_REASONS.get(reason, reason))
        return
    local_tz = datetime.now().astimezone().tzinfo
    bar_close = pd.Timestamp(view["bar"]) + pd.Timedelta(minutes=15)
    if bar_close.tzinfo is not None:
        when = "%s UTC (%s no seu horario)" % (bar_close.strftime("%d/%m %H:%M"),
                                               bar_close.tz_convert(local_tz).strftime("%H:%M"))
    else:
        when = bar_close.strftime("%d/%m %H:%M")
    price = view.get("close")
    diagnostic = set(view.get("diagnostic") or [])
    icons = {"bull": "🐂", "bear": "🐻"}
    agent_lines, compact = [], []
    for sh in view.get("shadows", []):
        tag = "diagnostico" if sh.agent in diagnostic else "aprovado"
        name = "%s %s (%s)" % (icons.get(sh.agent, "•"), AGENT_NAMES.get(sh.agent, sh.agent), tag)
        if sh.side == 0:
            state = "FORA"
            compact.append("%s FORA" % AGENT_NAMES.get(sh.agent, sh.agent))
        else:
            side_txt = "COMPRADO" if sh.side > 0 else "VENDIDO"
            pnl = sh.side * (price / sh.entry_price - 1) * 100 if price and sh.entry_price else 0.0
            since = ""
            if getattr(sh, "entry_bar", None) is not None:
                eb = pd.Timestamp(sh.entry_bar) + pd.Timedelta(minutes=15)
                since = " desde %s UTC" % eb.strftime("%d/%m %H:%M")
            stop = " | stop {:,.1f}".format(sh.stop_price) if sh.stop_price else ""
            new = "  ★ NOVA ENTRADA" if sh.entered_on_last_bar else ""
            state = "%s%s @ {:,.1f}%s | %+.2f%%%s".format(sh.entry_price) % (side_txt, since, stop, pnl, new)
            compact.append("%s %s %+.2f%%%s" % (AGENT_NAMES.get(sh.agent, sh.agent), side_txt, pnl, " ★" if sh.entered_on_last_bar else ""))
        agent_lines.append("   %-30s na simulacao: %s" % (name, state))
    acc_side = view.get("account_side", 0)
    account = "SEM POSICAO" if acc_side == 0 else "%s %.4f BTC" % (
        "COMPRADA" if acc_side > 0 else "VENDIDA", view.get("account_qty", 0.0))
    action = {"open": "ABRIR POSICAO", "close": "FECHAR POSICAO", "hold": "AGUARDAR"}.get(view.get("action"), "AGUARDAR")
    if signal is not None and signal.action == Action.HOLD:
        action = "AGUARDAR"
    reason_text = _mirror_reason_text(view)
    key = (str(view["bar"]), action, reason_text, acc_side, tuple(s.side for s in view.get("shadows", [])))
    if key == _last_mirror_panel["key"]:
        logger.info("🪞 [ESPELHO] candle %s | %s | conta %s | %s", bar_close.strftime("%H:%M"),
                    " | ".join(compact), account, action)
        return
    _last_mirror_panel["key"] = key
    divider = "─" * 72
    price_txt = "{:,.1f}".format(price) if price else "n/d"
    logger.info(
        "\n🪞 " + divider + "\n"
        "   🪞 ESPELHO DOS AGENTES  •  candle de 15m fechado " + when + "\n"
        "   " + divider + "\n"
        "   💲 BTCUSDT: " + price_txt + "\n" +
        "\n".join(agent_lines) + "\n"
        "   🏦 CONTA (Binance):         " + account + "\n"
        "   ▶  ACAO DO BOT:             " + action + "\n"
        "   📌 POR QUE:                 " + reason_text + "\n"
        "   ⏭  PROXIMA ORDEM:           quando um agente mostrar ★ NOVA ENTRADA, ou quando o\n"
        "                               agente que esta na conta sair na simulacao.\n"
        "   ℹ️  Regime, tape, OBI, sentimento e SHAP sao so informativos: nao decidem a ordem.\n"
        "🪞 " + divider)


# --- Funções de Verificação Pré-voo ---

async def reconcile_positions(portfolio: PortfolioOptimizer, connector: BinanceConnector, real_data: Dict[str, Any]) -> None:
    """
    Reconcilia as posições do portfólio interno com os dados reais da Binance.
    
    Esta função verifica se há discrepâncias entre o estado do portfólio simulado
    e o estado real da conta na exchange, corrigindo qualquer inconsistência.
    """
    try:
        logger.info("🔄 [RECONCILIATION] Iniciando reconciliação de posições...")
        
        # Obtém posições reais da Binance
        real_positions = real_data.get('positions', {})
        real_balance = real_data.get('total_value', 0.0)
        
        # Obtém posições simuladas do portfólio
        simulated_positions = portfolio.positions
        simulated_balance = portfolio.get_total_value()
        
        # Verifica discrepâncias no saldo total
        balance_discrepancy = abs(real_balance - simulated_balance)
        balance_discrepancy_pct = (balance_discrepancy / real_balance * 100) if real_balance > 0 else 0
        
        if balance_discrepancy_pct > 5.0:  # Discrepância maior que 5%
            logger.info(f"🔄 [RECONCILIATION] Sincronizando saldo: Simulado=${simulated_balance:,.2f} → Real=${real_balance:,.2f}")
            
            # Corrige o saldo simulado
            portfolio.cash = real_balance - sum(pos.quantity * pos.entry_price for pos in simulated_positions.values())
            # Força atualização do histórico para refletir o novo saldo imediatamente
            portfolio.update_portfolio_value({}) 
            logger.info(f"✅ [RECONCILIATION] Saldo simulado corrigido para ${portfolio.cash:,.2f}")
        
        # Verifica posições por símbolo
        symbol = TradingConfig.PRIMARY_PAIR
        real_position = real_positions.get(symbol, {})
        simulated_position = simulated_positions.get(symbol)
        
        if real_position and simulated_position:
            real_quantity = real_position.get('quantity', 0.0)
            real_entry_price = real_position.get('entry_price', 0.0)
            
            quantity_discrepancy = abs(real_quantity - simulated_position.quantity)
            price_discrepancy = abs(real_entry_price - simulated_position.entry_price)
            
            if quantity_discrepancy > 0.001 or price_discrepancy > 0.01:  # Tolerâncias
                logger.warning(f"⚠️ [RECONCILIATION] Discrepância na posição {symbol}:")
                logger.warning(f"   Quantidade: Simulado={simulated_position.quantity:.6f}, Real={real_quantity:.6f}")
                logger.warning(f"   Preço: Simulado=${simulated_position.entry_price:.2f}, Real=${real_entry_price:.2f}")
                
                # Corrige a posição simulada
                simulated_position.quantity = real_quantity
                simulated_position.entry_price = real_entry_price
                simulated_position.leverage = real_position.get('leverage', 1)
                simulated_position.liquidation_price = float(real_position.get('liquidationPrice', 0.0) or 0.0)
                simulated_position.mark_price = float(real_position.get('markPrice', 0.0) or 0.0)
                simulated_position.timestamp = datetime.utcnow()
                
                logger.info(f"✅ [RECONCILIATION] Posição {symbol} corrigida.")
        
        # Verifica se há posições reais que não estão no portfólio simulado
        for symbol, real_pos in real_positions.items():
            if symbol not in simulated_positions and real_pos.get('quantity', 0) != 0:
                logger.warning(f"⚠️ [RECONCILIATION] Posição real {symbol} não encontrada no portfólio simulado. Adicionando...")
                
                # Adiciona a posição real ao portfólio simulado
                from models.trade_schema import Position
                new_position = Position(
                    symbol=symbol,
                    quantity=real_pos.get('quantity', 0.0),
                    entry_price=real_pos.get('entry_price', 0.0),
                    timestamp=datetime.utcnow(),
                    leverage=real_pos.get('leverage', 1),
                    liquidation_price=float(real_pos.get('liquidationPrice', 0.0) or 0.0),
                    mark_price=float(real_pos.get('markPrice', 0.0) or 0.0)
                )
                simulated_positions[symbol] = new_position
        
        # Verifica se há posições simuladas que não existem na realidade
        for symbol, sim_pos in list(simulated_positions.items()):
            if symbol not in real_positions and abs(sim_pos.quantity) > 0: # Fix: check quantity > 0
                logger.warning(f"⚠️ [RECONCILIATION] Posição simulada {symbol} não existe na realidade. Removendo...")
                del simulated_positions[symbol]
        
        # Força atualização de métricas do portfólio (alavancagem, equity, etc.)
        portfolio.update_portfolio_value({})
        
        logger.info("✅ [RECONCILIATION] Reconciliação de posições concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ [RECONCILIATION] Erro durante reconciliação: {e}", exc_info=True)

async def ensure_trailing_stops_for_existing_positions(connector, positions: Dict[str, Any], config) -> None:
    """
    Verifica e coloca trailing stops para posições existentes que não têm trailing stop ativo.
    Chamado no startup para garantir que todas as posições estão protegidas.
    
    Args:
        connector: BinanceConnector instance
        positions: Dicionário de posições abertas {symbol: position_data}
        config: TradingConfig com DEFAULT_STOP_LOSS_PCT
    """
    try:
        if not positions:
            logger.info("ℹ️ [TRAILING STOP] Nenhuma posição existente para verificar.")
            return
            
        logger.info(f"🔍 [TRAILING STOP] Verificando trailing stops para {len(positions)} posição(ões)...")
        
        for symbol, pos_data in positions.items():
            quantity = pos_data.get('quantity', 0)
            if quantity == 0:
                continue
                
            # Verifica se já existe trailing stop
            has_trailing = await connector.has_trailing_stop_order(symbol)
            
            if has_trailing:
                logger.info(f"✅ [TRAILING STOP] {symbol} já possui trailing stop ativo.")
                continue
            
            # Precisa colocar trailing stop
            logger.warning(f"⚠️ [TRAILING STOP] {symbol} NÃO possui trailing stop! Colocando agora...")
            
            # Determina o lado do trailing stop (oposto da posição)
            # quantity > 0 = LONG, trailing stop = SELL
            # quantity < 0 = SHORT, trailing stop = BUY
            trailing_side = "SELL" if quantity > 0 else "BUY"
            abs_quantity = abs(quantity)
            
            # Callback rate do config (4% default)
            callback_rate = config.DEFAULT_STOP_LOSS_PCT * 100
            
            result = await connector.place_trailing_stop_order(
                symbol=symbol,
                side=trailing_side,
                quantity=abs_quantity,
                callback_rate=callback_rate
            )
            
            if result:
                logger.info(f"✅ [TRAILING STOP] Trailing stop colocado para {symbol} ({trailing_side} @ {callback_rate}%)")
            else:
                logger.error(f"❌ [TRAILING STOP] Falha ao colocar trailing stop para {symbol}")
                
    except Exception as e:
        logger.error(f"❌ [TRAILING STOP] Erro ao verificar trailing stops: {e}", exc_info=True)

async def check_funds_condition(components: Dict[str, Any], min_balance_usd: float = 100.0) -> bool:
    """Verifica se há fundos suficientes na conta de produção."""
    logger.info(f"🔎 [PRÉ-VOO] Verificando fundos na conta (mínimo: ${min_balance_usd})...")
    try:
        connector = components["binance_connector"]
        summary = await connector.get_account_summary()
        balance = summary.get("total_value", 0.0)
        
        logger.info(f"💰 [PRÉ-VOO] Saldo atual da conta de produção: ${balance:,.2f}")
        
        if balance >= min_balance_usd:
            logger.info("✅ [PRÉ-VOO] CONDIÇÃO DE FUNDOS SUFICIENTES ATENDIDA.")
            return True

    except Exception as e:
        logger.error(f"❌ [PRÉ-VOO] Erro ao verificar fundos da conta: {e}", exc_info=True)

    logger.warning("⚠️ [PRÉ-VOO] CONDIÇÃO DE FUNDOS SUFICIENTES NÃO ATENDIDA.")
    return False


# --- Funções de Inicialização e Lógica do Bot ---

# No arquivo run_bot.py

async def initialize_all_components():
    """Inicializa todos os componentes na ordem correta."""
    logger.info("⚙️ [SETUP] Inicializando todos os componentes do sistema...")
    try:
        # --- Configurações Iniciais ---
        trading_config = TradingConfig()
        ai_config = AIConfig()

        # --- [NEW] SISTEMA DE CONNECTORS DUPLOS ---
        # data_connector: SEMPRE usa produção para obter dados reais do mercado
        # trading_connector: Usa testnet quando BINANCE_TESTNET=True (paper trading)
        
        # Connector para DADOS (sempre produção para dados reais)
        data_connector = BinanceConnector(active_config, force_production=True)
        system_components["data_connector"] = data_connector
        logger.info("📊 [SETUP] Data Connector inicializado (PRODUÇÃO - dados reais).")
        
        # Connector para TRADING (testnet para paper trading, produção para live)
        trading_connector = BinanceConnector(active_config, force_production=False)
        system_components["trading_connector"] = trading_connector
        logger.info("💹 [SETUP] Trading Connector inicializado (modo definido pelo .env).")
        
        # Mantém binance_connector apontando para trading_connector por compatibilidade
        system_components["binance_connector"] = trading_connector
        
        # --- Componentes Base ---
        system_components["feature_pipeline"] = FeatureEngineeringPipeline(ai_config)
        # DataProvider usa data_connector para buscar dados reais
        system_components["data_provider"] = DataProvider(data_connector, system_components["feature_pipeline"])
        system_components["portfolio"] = PortfolioOptimizer(trading_config)
        system_components["risk_manager"] = RiskManager(trading_config)
        system_components["ai_controller"] = AIController(ai_config, trading_config, system_state)
        system_components["onchain_engine"] = OnChainEngine(active_config)
        system_components["onchain_engine"].start() # Inicia imediatamente para carregar o valor real (29)
        
        # TapeEngine usa data_connector para dados reais de mercado
        system_components["tape_engine"] = TapeEngine(data_connector, [trading_config.PRIMARY_PAIR])
        system_components["ai_monitor"] = AIMonitor(log_dir="logs/ai_events")
        system_components["state_restore"] = StateRestore(active_config, ai_config)
        # Narrador IA local via Ollama — análise em linguagem natural no terminal
        # (telegram é passado depois, abaixo, após ser criado)
        _narrator = OllamaNarrator(model="qwen3.5:4b")
        system_components["narrator"] = _narrator

        # Notificações Telegram — alertas instantâneos + relatório horário via LLM
        _tg_token = getattr(active_config, 'TELEGRAM_BOT_TOKEN', '')
        _tg_chat  = getattr(active_config, 'TELEGRAM_CHAT_ID', '')
        _telegram = TelegramNotifier(
            token       = _tg_token,
            chat_id     = _tg_chat,
            narrator    = _narrator,
            ollama_model= "qwen3.5:4b",
        )
        system_components["telegram"] = _telegram
        # Liga o narrator ao telegram para encaminhar análises automaticamente
        _narrator.telegram = _telegram

        # --- Injeção de Dependências no AIController ---
        ai_controller = system_components["ai_controller"]
        ai_controller.set_portfolio(system_components["portfolio"])
        ai_controller.set_risk_manager(system_components["risk_manager"])
        ai_controller.set_ai_monitor(system_components["ai_monitor"])
        ai_controller.set_feature_pipeline(system_components["feature_pipeline"])
        ai_controller.set_tape_engine(system_components["tape_engine"])
        ai_controller.set_onchain_engine(system_components["onchain_engine"])
        ai_controller.set_data_provider(system_components["data_provider"])

        # --- Inicialização final do ExecutionEngine ---
        # ExecutionEngine usa trading_connector para executar ordens (testnet ou produção)
        system_components["execution_engine"] = ExecutionEngine(
            trading_config, trading_connector,
            system_components["portfolio"], system_state,
            trade_log_callback=save_trade_to_log
        )
        
        # [RISK GATE] Injeta RiskManager no ExecutionEngine para gates de risco intra-TWAP
        # Sem isso, ordens TWAP em andamento ignoram drawdown alto e modo risk-off.
        system_components["execution_engine"].set_risk_manager(system_components["risk_manager"])
        
        system_state["status"] = "inicializado"
        logger.info("✅ [SETUP] Todos os componentes inicializados com sucesso.")
    except Exception as e:
        logger.critical(f"🚨 [CRÍTICO SETUP] Falha crítica na inicialização: {e}", exc_info=True)
        shutdown_event.set()

async def main_trading_loop():
    """    
    O loop principal de operação do bot, com pipeline de dados unificado.
    """
    logger.info("📈 [LOOP] Iniciando loop principal de trading...")
    # Desempacota todos os componentes necessários de uma vez
    components = system_components
    data_provider, ai_controller, risk_manager, portfolio = (
        components["data_provider"], components["ai_controller"],
        components["risk_manager"], components["portfolio"]
    )
    execution_engine, tape_engine, feature_pipeline = (
        components["execution_engine"], components["tape_engine"], components["feature_pipeline"]
    )
    ai_monitor = components.get("ai_monitor")  # [XAI] Persiste explicações em JSONL
    narrator   = components.get("narrator")    # [NARRATOR] Análise LLM local no terminal
    telegram   = components.get("telegram")    # [TELEGRAM] Alertas instantâneos
    symbol = TradingConfig.PRIMARY_PAIR

    # [INFO] tape_engine já foi iniciado em main() antes do loop começar.
    # A verificação abaixo é um fallback de segurança apenas.
    if not tape_engine.is_running:
        logger.warning("[LOOP] TapeEngine não estava rodando — iniciando como fallback.")
        await tape_engine.start()
    onchain_engine = components.get("onchain_engine")
    if onchain_engine and not onchain_engine.is_running: onchain_engine.start()

    while not shutdown_event.is_set():
        if not system_state["trading_active"]:
            await asyncio.sleep(2)
            continue
        try:
            # Etapa 1/2: Obter métricas on-chain e features multi-timeframe
            # [SENTIMENT] Coleta o pulso do mercado via Fear & Greed da Binance
            onchain_pulse = onchain_engine.get_onchain_sentiment_signal() if onchain_engine else None
            system_state['onchain_pulse'] = onchain_pulse or {"signal": "NEUTRAL", "confidence": 0.1, "score": 50}
            
            # [FEATURES] Prepara dados para os especialistas
            featured_df = await data_provider.get_latest_features(symbol, tape_metrics=system_state.get('tape_pulse'), sentiment_metrics=onchain_pulse)
            if featured_df is None or featured_df.empty:
                logger.warning("⚠️ [LOOP] Falha ao gerar features para os dados recentes.")
                await asyncio.sleep(10)
                continue
            
            latest_features = featured_df.iloc[-1]
            system_state['latest_features'] = latest_features
            current_price = latest_features.get('close')

            # Atualiza os componentes com os dados mais recentes
            portfolio.update_portfolio_value({symbol: current_price})
            risk_manager.add_historical_data({symbol: featured_df[['close']]})
            risk_manager.update_portfolio_state(
                portfolio.get_total_value(), portfolio.cash, portfolio.positions,
                portfolio.total_notional_value, portfolio.margin_used
            )
            system_state['tape_pulse'] = tape_engine.get_market_pulse(symbol)
            
            # [STATUS] Atualização de Saúde do Sistema
            if risk_manager:
                try:
                    r_prof = risk_manager.get_current_risk_profile()
                    if r_prof.get('is_risk_off_mode'):
                        system_state['system_health'] = "CRÍTICO (Risk Off)"
                        # [TELEGRAM] Alerta risk-off (anti-spam 5 min interno)
                        if telegram:
                            _dd  = r_prof.get('current_drawdown_pct', 0)
                            _lev = portfolio.get_detailed_status().get('total_leverage_ratio', 0)
                            asyncio.create_task(telegram.alert_risk_off(_dd, _lev))
                    elif r_prof.get('current_drawdown_pct', 0) > r_prof.get('max_drawdown_limit_pct', 0.1) * 0.7:
                         system_state['system_health'] = "ATENÇÃO (Drawdown Elevado)"
                    else:
                         system_state['system_health'] = "SAUDÁVEL"
                except Exception:
                    system_state['system_health'] = "ERRO"
            
            # Etapa 3: Gerar e executar a decisão
            if ai_controller.is_trained:
                signal = await ai_controller.generate_trading_decision(featured_df)
                if _mirror_policy():
                    log_mirror_panel(ai_controller, signal)
                    _view = getattr(ai_controller, "mirror_view", None)
                    system_state["mirror_view"] = _view
                    # Grafico dos candles a cada candle de 15m fechado: no terminal
                    # (texto colorido) e em logs/charts/espelho.png.
                    if _view is not None and system_state.get("mirror_chart_bar") != str(_view.get("bar")):
                        system_state["mirror_chart_bar"] = str(_view.get("bar"))
                        try:
                            from trading import mirror_chart
                            # Marcas do grafico a partir das execucoes reais da conta
                            # (inclui trades fechados com o bot parado ou por outra maquina).
                            _conn = getattr(execution_engine, "connector", None)
                            if _conn is not None and hasattr(_conn, "_make_request"):
                                try:
                                    await mirror_chart.sync_fills(_conn, TradingConfig.PRIMARY_PAIR)
                                except Exception as _fills_error:
                                    logger.warning("[ESPELHO] Execucoes da conta nao sincronizadas: %s", _fills_error)
                            _pos = portfolio.positions.get(TradingConfig.PRIMARY_PAIR) if portfolio else None
                            mirror_chart.record_account(_view["bar"], _view.get("account_side", 0),
                                                        getattr(_pos, "entry_price", 0.0) if _pos else 0.0)
                            _hist = ai_controller.mirror_history.frame
                            _paths = dict(getattr(ai_controller, "mirror_paths", {}) or {})
                            _text = mirror_chart.render(_hist, _paths, _view)
                            _mode = os.environ.get("MIRROR_CHART_MODE", "imagem").strip().lower()
                            if _mode == "png":
                                # So redesenha logs/charts/espelho.png (sem imprimir, sem copias).
                                asyncio.create_task(asyncio.to_thread(
                                    mirror_chart.render_png, _hist.copy(), _paths, dict(_view)))
                            elif _mode == "texto":
                                print(_text, flush=True)
                                asyncio.create_task(asyncio.to_thread(
                                    mirror_chart.render_png, _hist.copy(), _paths, dict(_view)))
                            else:
                                # A imagem PNG impressa no terminal (Sixel, Windows Terminal 1.22+).
                                async def _print_chart(h=_hist.copy(), p=_paths, v=dict(_view)):
                                    try:
                                        from trading import sixel
                                        png = await asyncio.to_thread(mirror_chart.render_png, h, p, v)
                                        image = await asyncio.to_thread(sixel.encode, png)
                                        print("\n" + image, flush=True)
                                    except Exception as _img_error:
                                        logger.warning("[ESPELHO] Imagem do grafico nao impressa: %s", _img_error)
                                asyncio.create_task(_print_chart())
                        except Exception as _chart_error:
                            logger.warning("[ESPELHO] Grafico nao gerado: %s", _chart_error)
                    # IA local (Ollama): conta em linguagem natural o que o bot esta vendo,
                    # a partir das mesmas variaveis que os agentes leem. So comenta.
                    if narrator and _view is not None:
                        try:
                            from trading.ollama_narrator import mirror_facts
                            _row = ai_controller.mirror_history.frame.iloc[-1]
                            _tape = system_state.get('tape_pulse', {})
                            _sent = system_state.get('onchain_pulse', {}) or {}
                            _regimes = {0: "bull", 1: "bear", 2: "lateral"}
                            _facts = mirror_facts(_row, _view)
                            _names = {"bull": "agente LONG", "bear": "agente SHORT"}
                            _sides = " | ".join("%s: %s" % (_names.get(s.agent, s.agent),
                                                "fora" if s.side == 0 else "comprado" if s.side > 0 else "vendido")
                                                for s in _view.get("shadows", []))
                            _acc = _view.get("account_side", 0)
                            _action = {"open": "ABRIR POSIÇÃO", "close": "FECHAR POSIÇÃO"}.get(_view.get("action"), "AGUARDAR")
                            _bar_close = pd.Timestamp(_view["bar"]) + pd.Timedelta(minutes=15)
                            await narrator.maybe_narrate_mirror({
                                "symbol": TradingConfig.PRIMARY_PAIR,
                                "price": float(_view.get("close") or current_price or 0.0),
                                "bar": _bar_close.strftime("%d/%m %H:%M"),
                                "facts": _facts["facts"], "agents": _facts["agents"],
                                "action": _action, "reason": _mirror_reason_text(_view),
                                "regime": _regimes.get(int(latest_features.get('regime', 2)), "desconhecido"),
                                "tape_pulse": _tape.get('pulse', 'NEUTRAL'),
                                "tape_score": float(_tape.get('score', 0.0)),
                                "obi": float(_tape.get('obi', 0.0)),
                                "sentiment": _fear_greed_text(_sent.get('score')),
                                "_state_key": _facts["state"],
                                "_mirror_key": (str(_view["bar"]), _action, _acc,
                                                tuple(s.side for s in _view.get("shadows", []))),
                                "_mirror_sub": "%s | conta: %s | bot: %s" % (
                                    _sides, "sem posição" if not _acc else "comprada" if _acc > 0 else "vendida", _action),
                            })
                        except Exception as _narr_error:
                            logger.debug("[NARRATOR] Contexto do espelho indisponivel: %s", _narr_error)
                    if _view is not None:
                        system_state["mirror_reason_text"] = _mirror_reason_text(_view)
                        # Conta mudou sem ordem de fechamento do espelho (stop na corretora).
                        _prev = system_state.get("mirror_account_side")
                        _now = _view.get("account_side", 0)
                        if telegram and _prev is not None and _prev != _now and not system_state.get("mirror_order_pending"):
                            asyncio.create_task(telegram.alert_mirror_account_change(
                                _prev, _view, portfolio.get_total_value() if portfolio else 0.0))
                        system_state["mirror_account_side"] = _now
                        system_state["mirror_order_pending"] = False

                # Mirrored policies: keep the exchange stop on the environment's
                # stop, so a stop fills inside the bar as in the backtest.
                stop_target = getattr(ai_controller, "mirror_stop_target", None)
                if stop_target and hasattr(execution_engine, "sync_mirror_stop"):
                    try:
                        await execution_engine.sync_mirror_stop(*stop_target)
                    except Exception as stop_error:
                        logger.error("[MIRROR] Falha ao sincronizar o stop: %s", stop_error, exc_info=True)
                    ai_controller.mirror_stop_target = None

                if signal and signal.action != Action.HOLD:
                    order = await execution_engine.submit_order(signal)
                    if _mirror_policy() and telegram and getattr(ai_controller, "mirror_view", None):
                        # A mudança de lado da conta que esta ordem causa não é
                        # um stop na corretora: não alerta duas vezes.
                        system_state["mirror_order_pending"] = True
                    if _mirror_policy() and order is not None and getattr(ai_controller, "mirror_view", None):
                        # Trade real do bot: marca no grafico no candle desta ordem.
                        try:
                            from trading import mirror_chart
                            _mv = ai_controller.mirror_view
                            _new_side = 0 if _mv.get("action") == "close" else (1 if signal.action == Action.BUY else -1)
                            mirror_chart.record_account(_mv["bar"], _new_side, _mv.get("close") or 0.0)
                        except Exception as _rec_error:
                            logger.warning("[ESPELHO] Trade real nao registrado no grafico: %s", _rec_error)
                        asyncio.create_task(telegram.alert_mirror_order(
                            dict(ai_controller.mirror_view), signal, order is not None))
                    # [XAI] Persiste evento de ordem enviada no AIMonitor
                    if ai_monitor and order is not None:
                        ai_monitor.log_event(
                            event_type = "ORDER_SUBMITTED",
                            message    = f"Ordem {signal.action.value} submetida para {signal.symbol}",
                            level      = "INFO",
                            context    = {
                                "order_id":      str(getattr(order, 'id', 'N/A')),
                                "symbol":        signal.symbol,
                                "action":        signal.action.value,
                                # Signal nao tem quantidade nem preco: vem da ordem.
                                "quantity":      getattr(order, 'quantity', None),
                                "price":         getattr(order, 'price', None),
                                "stop_loss":     signal.stop_loss,
                                "take_profit":   signal.take_profit,
                                "confidence":    signal.confidence,
                                "explanation":   system_state.get('last_explanation', {}).get('narrative', ''),
                            }
                        )
                elif signal and signal.action == Action.HOLD:
                    # [VETO RISK] Se o trade foi vetado pela margem, notifica o ExecutionEngine
                    # para não adicionar outras ordens (como stops) que possam gerar conflito.
                    reason = signal.explanation.get('reason', '')
                    if "REJEITADO" in reason and "Margem" in reason:
                        execution_engine.register_margin_veto(signal.symbol)

            else:
                logger.warning("⚠️ [LOOP] IA não treinada. Pulando geração de decisão.")

        except Exception as e:
            logger.critical(f"🚨 [CRÍTICO LOOP] Erro fatal no loop de trading: {e}", exc_info=True)
            system_state["system_health"] = "critical"
            # [TELEGRAM] Alerta imediato para erro crítico no loop
            if telegram:
                asyncio.create_task(telegram.alert_error(str(e)[:200], level="CRITICAL"))

        await asyncio.sleep(DataConfig.MINUTE_INTERVAL_SECONDS)

async def monitor_and_log_loop():
    """Loop para registrar o status do sistema e portfólios periodicamente."""
    logger.info("📊 [MONITOR] Iniciando loop de monitoramento e logging.")

    # [MELHORIA] Variáveis para reconciliação de posições
    last_reconciliation_time = datetime.utcnow()
    reconciliation_interval_hours = 2  # Reconcilia a cada 2 horas
    # [TELEGRAM] Rastreia início de WARMUP prolongado do tape
    _tape_warmup_since: Optional[datetime] = None

    while not shutdown_event.is_set():
        try:
            log_system_status()
            portfolio = system_components.get("portfolio")
            # Ao vivo o portfolio interno so espelha a Binance: mostrar os dois
            # repetia o mesmo saldo com rotulos diferentes.
            if portfolio and not system_state["live_trading_enabled"]:
                log_portfolio_status("Portfólio de Papel", portfolio.get_detailed_status())
            
            if system_state["live_trading_enabled"] and system_state["binance_connection"] == "conectado":
                if connector := system_components.get("binance_connector"):
                    real_data = await connector.get_account_summary()
                    
                    # [CORREÇÃO] Atualiza margem usada no objeto de portfólio imediatamente
                    # Isso garante que o RiskManager tenha o valor correto para check_trade_approval
                    if portfolio:
                        portfolio.margin_used = real_data.get('margin_used', 0.0)
                        
                        # [SAFETY] Sincroniza preços de liquidação e mark price
                        for s, p_data in real_data.get('positions', {}).items():
                             if s in portfolio.positions:
                                 portfolio.positions[s].liquidation_price = float(p_data.get('liquidation_price', 0.0) or 0.0)
                                 portfolio.positions[s].mark_price = float(p_data.get('mark_price', 0.0) or 0.0)
                        
                    log_portfolio_status("Portfólio Real (Binance)", real_data)
                    
                    # [SAFETY] Checagem de Risco de Liquidação
                    risk_manager = system_components.get("risk_manager")
                    if risk_manager:
                        risk_alerts = risk_manager.check_liquidation_risk()
                        for alert in risk_alerts:
                            logger.critical(alert)
                            # Feedback visual na UI
                            system_state["command_feedback"] = alert

                    # [MELHORIA] Reconciliação de Posições
                    current_time = datetime.utcnow()
                    time_since_last_reconciliation = (current_time - last_reconciliation_time).total_seconds() / 3600
                    
                    if time_since_last_reconciliation >= reconciliation_interval_hours:
                        await reconcile_positions(portfolio, connector, real_data)
                        last_reconciliation_time = current_time

            # ── TELEGRAM: relatório horário + alerta de tape WARMUP ──────────
            _telegram = system_components.get("telegram")
            if _telegram:
                _tape = system_state.get('tape_pulse', {})
                await _telegram.maybe_hourly_report(
                    system_state,
                    system_components.get("portfolio"),
                    _tape,
                )
                # Alerta se tape em WARMUP por mais de 5 min
                if _tape.get('pulse') == 'WARMUP...':
                    if _tape_warmup_since is None:
                        _tape_warmup_since = datetime.utcnow()
                    else:
                        _elapsed = (datetime.utcnow() - _tape_warmup_since).total_seconds()
                        if _elapsed > 300:   # > 5 min
                            asyncio.create_task(_telegram.alert_tape_warmup(_elapsed))
                else:
                    _tape_warmup_since = None   # reset quando sai do WARMUP
            # ─────────────────────────────────────────────────────────────────

            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"❌ [MONITOR] Erro no loop de monitoramento: {e}", exc_info=True)

# No arquivo run_bot.py
# Substitua sua função 'main' inteira por esta versão final e definitiva.

async def main():
    """    
    Orquestra o bot com uma lógica de treinamento/carregamento inequívoca
    baseada na existência e validade dos metadados, com treinamento granular.
    """
    logger.info("🚀 [BOT] Iniciando Bot Shield IA (Modo Automatizado)...")
    os.makedirs(LOGS_DIR, exist_ok=True)

    await initialize_all_components()
    if shutdown_event.is_set(): return

    try:
        # [NEW] Conecta ambos os connectors
        # Data Connector: Produção (dados reais)
        data_connector = system_components.get("data_connector")
        if data_connector:
            await data_connector.connect()
            logger.info("✅ [CONEXÃO] Data Connector (PRODUÇÃO) conectado com sucesso.")
        
        # Trading Connector: Testnet ou Produção (baseado no .env)
        trading_connector = system_components.get("trading_connector")
        if trading_connector:
            await trading_connector.connect()
            mode = "TESTNET" if trading_connector.testnet else "PRODUÇÃO"
            logger.info(f"✅ [CONEXÃO] Trading Connector ({mode}) conectado com sucesso.")
        
        system_state["binance_connection"] = "conectado"
    except Exception as e:
        logger.critical(f"🚨 [CRÍTICO] Falha ao conectar à Binance: {e}. Encerrando.", exc_info=True)
        return
    
    await system_components["execution_engine"].start()
    
    if "tape_engine" in system_components and system_components["tape_engine"]:
        await system_components["tape_engine"].start()
    
    # -------------------------------------------------------------------------
    # [CRITICAL UPDATE] Sincronização de Portfólio com Binance (Testnet/Prod)
    # -------------------------------------------------------------------------
    try:
        trading_connector = system_components.get("trading_connector")
        portfolio = system_components.get("portfolio")
        
        if trading_connector and portfolio and system_state.get("binance_connection") == "conectado":
            logger.info("🔄 [INIT] Sincronizando portfólio com o saldo real da Binance...")
            
            # Obtém saldo e posições reais
            account_info = await trading_connector.get_account_summary()
            
            # Atualiza capital inicial e disponível
            real_balance = account_info.get("total_value", 0.0)
            real_cash = account_info.get("cash", 0.0)
            
            if real_balance > 0:
                portfolio.initial_capital = real_balance
                portfolio.cash = real_cash
                
                # Se houver posições abertas na exchange, importa elas
                real_positions = account_info.get("positions", {})
                if real_positions:
                    logger.info(f"🔄 [INIT] Importando {len(real_positions)} posições abertas da exchange...")
                    
                # Reconcilia explicitamente para garantir alinhamento total
                await reconcile_positions(portfolio, trading_connector, account_info)
                
                system_components["risk_manager"].update_portfolio_state(
                    portfolio.get_total_value(), portfolio.cash, portfolio.positions,
                    portfolio.total_notional_value, portfolio.margin_used
                )
                
                logger.info(f"✅ [INIT] Portfólio sincronizado! Saldo Real: ${real_balance:,.2f} | Disponível: ${real_cash:,.2f}")
            else:
                 logger.warning("⚠️ [INIT] Saldo retornado pela Binance é Zero ou inválido. Usando valor padrão/paper trading.")

    except Exception as e:
        # Verifica se é erro de autenticação (401)
        if "Unauthorized" in str(e) or "401" in str(e):
            logger.warning("⚠️ [INIT] Falha de autenticação na Binance (401). Verifique suas chaves API. O bot continuará, mas o trading real pode falhar.")
        else:
            logger.error(f"❌ [INIT] Falha ao sincronizar portfólio com Binance: {e}", exc_info=False)
    # -------------------------------------------------------------------------

    ai_controller = system_components["ai_controller"]
    
    # O bot opera os agentes aprovados (espelho) ou nada: nao treina ao
    # iniciar, e sem aprovacao valida ele para aqui. O treino e feito fora do
    # bot (cloud/train_guided.py) e o modelo so entra depois de aprovado.
    # Modelo de algum agente ausente: retreina esse agente antes de operar.
    from pathlib import Path as _Path
    from trading.auto_retrain import retrain_missing
    _diag = {a.strip() for a in str(getattr(TradingConfig, 'TESTNET_DIAGNOSTIC_AGENTS', '')).split(',') if a.strip()}
    _agents = [a.strip() for a in str(getattr(TradingConfig, 'LIVE_AGENTS', 'bull')).split(',')
               if a.strip() and a.strip() not in _diag]
    if not await retrain_missing(_Path(str(AIConfig.MODEL_DIR)), _agents):
        logger.critical("🚨 [RETREINO] Nao foi possivel retreinar o modelo que falta. Encerrando.")
        return
    if not await ai_controller.prepare_agent_mirror():
        logger.critical("🚨 [agent_mirror] Politica sem aprovacao ou artefatos validos. Encerrando.")
        return
    system_state["ai_status"] = "pronto"
    logger.info("✅ [PRÉ-VOO] Bot pronto para operar.")

    # Verificação de fundos (informativa - não bloqueia)
    if not TradingConfig.BINANCE_TESTNET:
         await check_funds_condition(system_components)
    else:
         logger.info("ℹ️ [PRÉ-VOO] Verificação de fundos pulada (Modo Testnet/Paper).")

    # [MODIFICAÇÃO] Habilita execução de ordens (Testnet ou Prod)
    # Sempre ativa o modo de operação após o pré-voo, independente do backtest (request do usuário)
    system_state["live_trading_enabled"] = True
    system_state["trading_active"] = True

    mode_label = "LIVE (REAL)" if not TradingConfig.BINANCE_TESTNET else "TESTNET (REAL API)"
    logger.info(f"🚀🚀🚀 [BOT ATIVADO] Iniciando operação em modo {mode_label}. 🚀🚀🚀")

    # [TELEGRAM] Verifica bot + envia alerta de início
    _telegram = system_components.get("telegram")
    if _telegram:
        await _telegram.check_availability()
        _bal = system_components["portfolio"].get_total_value() if system_components.get("portfolio") else 0.0
        _policy = ""
        if _mirror_policy():
            _agents = [a.strip() for a in str(getattr(TradingConfig, 'LIVE_AGENTS', '')).split(',') if a.strip()]
            _diag = {a.strip() for a in str(getattr(TradingConfig, 'TESTNET_DIAGNOSTIC_AGENTS', '')).split(',') if a.strip()}
            _policy = "Espelho dos agentes: " + ", ".join(
                "%s (%s)" % (AGENT_NAMES.get(a, a), "diagnóstico" if a in _diag else "aprovado") for a in _agents)
        await _telegram.alert_bot_started(mode_label, _bal, _policy)
    
    # Salvar estado inicial (checkpoint de partida)
    try:
        state_to_save = {
            "portfolio": portfolio.get_detailed_status(),
            "system_state": system_state 
        } 
        # system_components["state_restore"].save_checkpoint(state_to_save, "startup_state")
    except Exception as e:
        logger.warning(f"⚠️ [INIT] Não foi possível salvar checkpoint inicial: {e}")

    # --- INÍCIO DOS LOOPS ASSÍNCRONOS ---
    background_tasks.append(asyncio.create_task(monitor_and_log_loop()))
    background_tasks.append(asyncio.create_task(main_trading_loop()))
    
    await shutdown_event.wait()

async def shutdown_gracefully():
    # ... (mesmo código de antes)
    if not shutdown_event.is_set():
        logger.info("🛑 [BOT] Iniciando processo de desligamento...")
        shutdown_event.set()

        # [TELEGRAM] Alerta de encerramento (síncrono antes do cleanup)
        _telegram = system_components.get("telegram")
        if _telegram and _telegram._available:
            try:
                await _telegram.alert_bot_stopped("Desligamento normal")
            except Exception:
                pass
        
        # Salvar estado final antes de sair!
        try:
            if system_components.get("state_restore") and system_components.get("portfolio"):
                # Construir estado serializável
                # Simplificação: usando os dados internos do portfolio
                p = system_components["portfolio"]
                
                # Converter posições para dicts
                pos_list = []
                for sym, pos in p.positions.items():
                    pos_list.append({
                        "symbol": sym,
                        "quantity": pos.quantity,
                        "entry_price": pos.entry_price,
                        "timestamp": pos.timestamp.isoformat(),
                        "unrealized_pnl": pos.unrealized_pnl
                    })
                
                # Converter histórico
                hist_dict = {k.isoformat(): v for k, v in p.portfolio_value_history.items()}
                
                state_dict = {
                    "portfolio": {
                        "initial_capital": p.initial_capital,
                        "cash": p.cash,
                        "margin_used": p.margin_used,
                        "total_notional_value": p.total_notional_value,
                        "positions": pos_list,
                        "portfolio_value_history": hist_dict
                    }
                }
                
                system_components["state_restore"].save_checkpoint(state_dict, "shutdown_state")
        except Exception as e:
            logger.error(f"Erro ao salvar estado no shutdown: {e}")

        for task in background_tasks:
            task.cancel()
        await asyncio.gather(*background_tasks, return_exceptions=True)
        
        try:
            if (ee := system_components.get("execution_engine")): await ee.stop()
            if (te := system_components.get("tape_engine")): await te.stop()
            # [FIX AIOHTTP] Fecha AMBOS os conectores (data + trading) para evitar "Unclosed client session".
            # Antes apenas "binance_connector" (alias de trading_connector) era fechado,
            # deixando data_connector com sessão aiohttp aberta → WARNING no shutdown.
            if (dc := system_components.get("data_connector")):
                await dc.close()
            if (tc := system_components.get("trading_connector")):
                await tc.close()
            logger.info("✅ [BOT] Conexões e engines parados com sucesso.")
        except Exception as e:
            logger.critical(f"🚨 [CRÍTICO] Erro na parada de componentes: {e}", exc_info=True)

        logger.info("🏁 [BOT] Bot Shield IA finalizado.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(shutdown_gracefully()))
    except (NotImplementedError, AttributeError): pass

    try:
        loop.run_until_complete(main())
    except asyncio.CancelledError:
        logger.info("Task principal cancelada.")
    except KeyboardInterrupt:
        pass
        pass
    finally:
        if not loop.is_closed():
            loop.run_until_complete(shutdown_gracefully())
            loop.close()
        print("\n🏁 Aplicação finalizada. Verifique os logs.")
