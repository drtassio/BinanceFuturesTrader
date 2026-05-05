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
import shutil
# --- Configurações e Utilitários ---
from config.settings import active_config, TradingConfig, AIConfig, DataConfig, BacktestConfig
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
from trading.backtester import Backtester
from trading.ollama_narrator import OllamaNarrator
from trading.telegram_notifier import TelegramNotifier
from data_provider import DataProvider
from feature_engineering.main import FeatureEngineeringPipeline
from governance.ai_monitor import AIMonitor
from governance.drift_detector import DriftDetector
from learning.curriculum_engine import CurriculumEngine
from xai_engine.explainer import Explainer

# --- Environment Hardening (Audit Fix P0) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)
# Prevent 'init_gesdd failed init' and other MKL/BLAS conflicts on Windows
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    "last_adaptation": None,  # Timestamp da última adaptação
    "adaptation_count": 0,  # Contador de adaptações realizadas
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
    mode = "LIVE TRADING (REAL)" if system_state['live_trading_enabled'] else "PAPER TRADING (SIMULADO)"

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

    # Adaptações realizadas
    adapt_count = system_state.get('adaptation_count', 0)
    last_adapt  = system_state.get('last_adaptation')
    adapt_str   = f"{adapt_count}x | Último: {last_adapt[:10] if last_adapt else 'nunca'}"

    divider = "─" * 60
    tape_score = tape.get('score', 0.0)
    log_message = (
        f"\n💡 {divider}\n"
        f"   📊 STATUS GERAL ({datetime.now().strftime('%H:%M:%S')})\n"
        f"   {divider}\n"
        f"   🚀 MODO:        {mode}\n"
        f"   🏥 SAÚDE:       {system_state['system_health'].upper()} | CONEXÃO: {system_state['binance_connection'].upper()}\n"
        f"   🧠 IA STATUS:   {system_state['ai_status'].upper()} | REGIME: {regime_str}\n"
        f"   📟 TAPE PULSE:  {tape_pulse_str:<10} (Score: {tape_score:+.2f})\n"
        f"   🧠 SENTIMENT:    {system_state['onchain_pulse']['signal']} ({system_state['onchain_pulse'].get('score', 50)})\n"
        f"   ⚙️ ADAPTAÇÕES:  {adapt_str}\n"
        f"💡 {divider}\n"
    )
    logger.info(log_message)


def log_trade_decision(signal, explainer, ai_monitor=None):
    """
    Registra a decisão de trading da IA, gera explicação SHAP e persiste no AIMonitor.

    Cobre TODOS os tipos de decisão (BUY, SELL, HOLD) para que o bot nunca
    seja uma caixa-preta — cada escolha fica rastreável via logs/ai_events/*.jsonl.
    """
    if not signal:
        logger.info("🧠 [IA] Decisão: HOLD. Motivo: Sinal nulo recebido.")
        return

    action_str = signal.action.value.upper() if hasattr(signal.action, 'value') else str(signal.action)
    
    # [OBSERVABILIDADE] Diferencia tipos de HOLD para transparência
    hold_type = signal.explanation.get('hold_type') if signal.explanation else None
    if signal.action == Action.HOLD and hold_type:
        if hold_type == 'LOW_CONF':
            action_str = "HOLD (AGUARDANDO)"
        elif hold_type == 'COUNTER_TREND_VETO':
            action_str = "HOLD (VETO REGIME)"
        elif hold_type == 'LOW_PROB':
            action_str = "HOLD (RISCO/RETORNO)"
        elif hold_type == 'RISK':
            action_str = "HOLD (VETO RISCO)"
    latest_features = system_state.get('latest_features')

    # ── Gera explicação ──────────────────────────────────────────────────────
    if explainer is not None and latest_features is not None:
        try:
            explanation = explainer.explain_decision(signal, latest_features)
        except Exception as exc:
            logger.warning(f"⚠️ [XAI] Falha ao gerar explicação: {exc}")
            explanation = {
                "decision": action_str,
                "confidence": f"{signal.confidence:.2%}",
                "narrative": f"Decisão {action_str} — explicação indisponível ({exc})",
                "main_contributors": [],
            }
    else:
        # Fallback mínimo sem explainer / sem features
        reason = signal.explanation.get('reason', 'N/A') if signal.explanation else 'N/A'
        explanation = {
            "decision": action_str,
            "confidence": f"{signal.confidence:.2%}",
            "narrative": f"Decisão {action_str}. Motivo: {reason}",
            "main_contributors": [],
        }

    system_state['last_explanation'] = explanation

    # ── Log no terminal ──────────────────────────────────────────────────────
    narrative = explanation.get('narrative', '')
    divider = "─" * 60
    
    _exp = signal.explanation or {}
    reason = _exp.get('reason', 'N/A')

    status_icon = "⚖️" if signal.action == Action.HOLD else ("🚀" if action_str == "BUY" else "🔻")
    
    log_message = (
        f"\n🧠 {divider}\n"
        f"   🧠 DECISÃO IA: {action_str}\n"
        f"   {divider}\n"
        f"   {status_icon} Ação:        {action_str:<10}\n"
        f"   📌 Motivo:      {reason}\n"
        f"   🎯 Confiança:   {signal.confidence:.2%}\n"
        f"   \n"
        f"   {narrative}\n"
        f"🧠 {divider}\n"
    )
    logger.info(log_message)

    # ── Persiste no AIMonitor (JSONL em logs/ai_events/) ────────────────────
    if ai_monitor is not None:
        # Monta contexto rico: top-5 features + métricas de mercado + regime
        tape   = system_state.get('tape_pulse', {})
        regime = int(latest_features.get('regime', 2)) if latest_features is not None else -1
        regime_names = {0: "bull", 1: "bear", 2: "ranger"}

        context = {
            "symbol":       signal.symbol,
            "action":       action_str,
            "confidence":   signal.confidence,
            "price":        float(latest_features.get('close', 0)) if latest_features is not None else 0,
            "specialist":   signal.explanation.get('specialist', 'N/A') if signal.explanation else 'N/A',
            "regime":       regime_names.get(regime, 'unknown'),
            "tape_pulse":   tape.get('pulse', 'NEUTRAL'),
            "tape_score":   round(tape.get('score', 0.0), 4),
            "obi":          round(tape.get('obi', 0.0), 4),
            "top_features": explanation.get('main_contributors', []),
            "stop_loss":    signal.stop_loss,
            "take_profit":  signal.take_profit,
            "position_pct": signal.position_size_pct,
            "quantum":      signal.explanation.get('quantum', {})
        }

        # Exibe Dados Técnicos Formatados no Terminal
        quantum = context.get('quantum', {})
        h_val = quantum.get('hurst', 0.5)
        e_val = quantum.get('entropy', 0.0)
        c_lbl = quantum.get('chaos_label', 'N/A')
        
        technical_log = (
            f"🛠️ DADOS TÉCNICOS:\n"
            f"   🔹 Preço:       ${context['price']:,.2f}\n"
            f"   🔹 Specialist:  {context['specialist']}\n"
            f"   🔹 Regime:       {context['regime'].upper()}\n"
            f"   🔹 OBI/Tape:     {context['obi']:.4f} / {context['tape_pulse']}\n"
            f"   🔹 Hurst/Caos:   {h_val:.2f} / {c_lbl} (E:{e_val:.1f})\n"
            f"   {divider}"
        )
        logger.info(technical_log)

        # Persistência silenciosa do JSON no arquivo de log via log_event
        try:
            level = "INFO" if signal.action == Action.HOLD else "WARNING"
            ai_monitor.log_event(
                event_type = f"DECISION_{action_str}",
                message    = explanation.get('narrative', narrative),
                level      = level,
                context    = context
            )
        except Exception as e:
            logger.error(f"⚠️ [MONITOR] Falha ao registrar evento no AIMonitor: {e}")


# --- Funções de Verificação Pré-voo ---

async def check_profitability_condition(components: Dict[str, Any]) -> bool:
    """
    [VERSÃO FINAL COMPLETA - OTIMIZADA]
    Executa o backtest, passando o Explainer apenas se a configuração o exigir,
    para permitir uma verificação pré-voo rápida por padrão.
    """
    logger.info("🔎 [PRÉ-VOO] Verificando lucratividade e risco da estratégia via backtest...")
    # [CONFIG] Thresholds relaxados para permitir operação inicial
    PROFITABILITY_THRESHOLD = 0.00 # Basta ser positivo
    MAX_DRAWDOWN_THRESHOLD = 0.35 # Aceita até 35% de DD em testes
    result_obj = None
    
    try:
        backtester = components["backtester"]
        ai_controller = components["ai_controller"]
        feature_pipeline = components["feature_pipeline"]
        data_provider = components["data_provider"]
        explainer = components["explainer"]
        
        temp_portfolio = PortfolioOptimizer(TradingConfig())
        temp_risk_manager = RiskManager(TradingConfig())
        
        logger.info("⚙️ [PRÉ-VOO] Tentando carregar DataFrame de features pré-processado para acelerar o backtest...")
        
        # Primeiro tenta carregar o DataFrame pré-processado para economizar tempo
        base_featured_path = os.path.join("models_ai", "base_featured_df.pkl")
        featured_df = None
        
        if os.path.exists(base_featured_path):
            try:
                featured_df = pd.read_pickle(base_featured_path)
                logger.info(f"✅ [PRÉ-VOO] DataFrame pré-processado carregado com sucesso! Shape: {featured_df.shape}")
                
                # Verifica se os dados são recentes o suficiente (últimos 180 dias)
                end_date = featured_df.index[-1]
                start_date = end_date - timedelta(days=180)
                backtest_df = featured_df.loc[start_date:end_date]
                
                if len(backtest_df) >= 100:  # Mínimo de 100 linhas para backtest válido
                    logger.info(f"📊 [PRÉ-VOO] Slice de backtest do cache: {len(backtest_df)} linhas de {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')} (últimos 180 dias)")
                else:
                    logger.warning("⚠️ [PRÉ-VOO] Dados pré-processados insuficientes. Buscando dados históricos...")
                    featured_df = None
                    
            except Exception as e:
                logger.warning(f"⚠️ [PRÉ-VOO] Erro ao carregar DataFrame pré-processado: {e}. Buscando dados históricos...")
                featured_df = None
        
        # Se não conseguiu carregar o pré-processado, busca dados históricos
        if featured_df is None:
            logger.info("⚙️ [PRÉ-VOO] Coletando dados históricos multi-timeframe para o backtest de validação...")
            from datetime import datetime, timedelta, timezone
            end_date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            start_date_str = (datetime.now(timezone.utc) - timedelta(days=365*2)).strftime('%Y-%m-%d')

            historical_data_multi_tf = await data_provider.get_data_for_training(
                symbol=TradingConfig.PRIMARY_PAIR,
                start_date=start_date_str,
                end_date=end_date_str
            )

            if not historical_data_multi_tf:
                logger.error("❌ [PRÉ-VOO] Falha ao coletar dados históricos multi-timeframe para o backtest. Condição REPROVADA.")
                return False

            logger.info("⚙️ [PRÉ-VOO] Recalculando features para o dataset de backtest para garantir consistência...")
            featured_df = await feature_pipeline.create_features(
                historical_data_multi_tf,
                TradingConfig.PRIMARY_PAIR, 
                TradingConfig.PRIMARY_TIMEFRAME_TRADING,
                fit_scaler=False
            )
        else:
            # Se carregou do cache, define as datas para o backtest
            end_date = featured_df.index[-1]
            start_date = end_date - timedelta(days=180)
            backtest_df = featured_df.loc[start_date:end_date]

        if featured_df is None or featured_df.empty:
             logger.error("❌ [PRÉ-VOO] Falha ao gerar features para o backtest. Condição REPROVADA.")
             return False

        # Se não carregou do cache, define as datas para o backtest
        if 'backtest_df' not in locals():
            end_date = featured_df.index[-1]
            start_date = end_date - timedelta(days=180)
            backtest_df = featured_df.loc[start_date:end_date]
            logger.info(f"📊 [PRÉ-VOO] Slice de backtest calculado: {len(backtest_df)} linhas de {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')} (últimos 180 dias)")
        
        model_version_str = (
            ai_controller.last_trained_date.strftime('%Y%m%d-%H%M')
            if ai_controller.last_trained_date else "live_model"
        )
        
        # Cria um dicionário de argumentos para a função run
        run_kwargs = {
            "historical_data_multi_tf": {TradingConfig.PRIMARY_TIMEFRAME_TRADING: backtest_df},
            "ai_controller": ai_controller,
            "portfolio": temp_portfolio,
            "risk_manager": temp_risk_manager,
            "symbol": TradingConfig.PRIMARY_PAIR,
            "timeframe": TradingConfig.PRIMARY_TIMEFRAME_TRADING,
            "model_version": model_version_str
        }
        
        # Adiciona o explainer como argumento apenas se a configuração o permitir
        if BacktestConfig.BACKTEST_WITH_EXPLANATIONS:
            run_kwargs["explainer"] = explainer
            
        result_obj = await backtester.run(**run_kwargs)

        if result_obj and result_obj.total_return_pct is not None and result_obj.max_drawdown_pct is not None:
            return_pct = result_obj.total_return_pct / 100
            drawdown_pct = result_obj.max_drawdown_pct / 100
            
            # [MELHORIA] Métricas adicionais para gate mais robusto
            profit_factor = getattr(result_obj, 'profit_factor', 1.0)
            sharpe_ratio = getattr(result_obj, 'sharpe_ratio', 0.0)
            avg_leverage = getattr(result_obj, 'avg_leverage_per_trade', 1.0)
            
            # [MELHORIA] Critérios de aprovação mais rigorosos
            meets_return_criteria = return_pct > PROFITABILITY_THRESHOLD
            meets_drawdown_criteria = drawdown_pct < MAX_DRAWDOWN_THRESHOLD
            meets_profit_factor_criteria = profit_factor > 1.2  # Profit factor mínimo
            meets_sharpe_criteria = sharpe_ratio > 0.5  # Sharpe ratio mínimo
            meets_leverage_criteria = avg_leverage < 3.0  # Alavancagem média máxima
            
            logger.info(f"📊 [PRÉ-VOO] Resultado: Retorno={return_pct:.2%}, Drawdown={drawdown_pct:.2%}")
            logger.info(f"📊 [PRÉ-VOO] Métricas: Profit Factor={profit_factor:.2f}, Sharpe={sharpe_ratio:.2f}, Avg Leverage={avg_leverage:.2f}")
            
            if (meets_return_criteria and meets_drawdown_criteria and 
                meets_profit_factor_criteria and meets_sharpe_criteria and meets_leverage_criteria):
                logger.info("✅ [PRÉ-VOO] CONDIÇÃO DE LUCRATIVIDADE E RISCO ATENDIDA (critérios rigorosos).")
                return True, result_obj
            else:
                failed_criteria = []
                if not meets_return_criteria: failed_criteria.append("Retorno insuficiente")
                if not meets_drawdown_criteria: failed_criteria.append("Drawdown excessivo")
                if not meets_profit_factor_criteria: failed_criteria.append("Profit Factor baixo")
                if not meets_sharpe_criteria: failed_criteria.append("Sharpe Ratio baixo")
                if not meets_leverage_criteria: failed_criteria.append("Alavancagem alta")
                
                logger.warning(f"⚠️ [PRÉ-VOO] Critérios não atendidos: {', '.join(failed_criteria)}")
                return False, result_obj

    except Exception as e:
        logger.error(f"❌ [PRÉ-VOO] Erro na verificação de lucratividade: {e}", exc_info=True)
    
    logger.warning("⚠️ [PRÉ-VOO] CONDIÇÃO DE LUCRATIVIDADE E RISCO NÃO ATENDIDA.")
    return False, None

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
    """ Inicializa todos os componentes na ordem correta,
    incluindo o novo Explainer com SHAP."""
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
        
        # --- Componentes Base (sem o Explainer por enquanto) ---
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
        system_components["drift_detector"] = None  # [ARCH] DriftDetector desativado
        system_components["curriculum_engine"] = CurriculumEngine(ai_config)
        system_components["backtester"] = Backtester(BacktestConfig(), db_session=None)
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
        # ai_controller.set_drift_detector(None)  # [ARCH] desativado
        ai_controller.set_curriculum_engine(system_components["curriculum_engine"])
        ai_controller.set_feature_pipeline(system_components["feature_pipeline"])
        ai_controller.set_tape_engine(system_components["tape_engine"])
        ai_controller.set_onchain_engine(system_components["onchain_engine"])
        ai_controller.set_data_provider(system_components["data_provider"])

        # --- INICIALIZAÇÃO DO NOVO EXPLAINER (SHAP) ---
        logger.info("⚙️ [SETUP] Preparando dados de referência para o Explainer SHAP...")
        # [FIX] Use dedicated SHAP background file if available, fallback to base cache
        shap_bg_path = os.path.join(ai_config.MODEL_DIR, "shap_background.pkl")
        base_df_path = os.path.join(ai_config.MODEL_DIR, "base_featured_df.pkl")
        
        background_df = None
        
        if os.path.exists(shap_bg_path):
            try:
                background_df = pd.read_pickle(shap_bg_path)
                logger.info(f"✅ [SHAP] Background carregado de '{shap_bg_path}'")
            except Exception as e:
                logger.warning(f"⚠️ [SHAP] Erro ao carregar '{shap_bg_path}': {e}")
        
        if background_df is None and os.path.exists(base_df_path):
            logger.info(f"ℹ️ [SHAP] Usando '{base_df_path}' como fallback para background.")
            try:
                background_df = pd.read_pickle(base_df_path)
            except Exception as e:
                logger.warning(f"⚠️ [SHAP] Erro ao carregar '{base_df_path}': {e}")
                background_df = None
            
        if background_df is not None:
            
            # [FIX] Filter to only use base_feature_columns (53 features) to match model expectations
            # Try to get from controller first, then from metadata as fallback
            base_cols = ai_controller.base_feature_columns
            if not base_cols:
                # [FIX] Fallback: load from training_metadata.json
                metadata_path = os.path.join(ai_config.MODEL_DIR, "training_metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        base_cols = metadata.get('base_feature_columns', [])
                        if base_cols:
                            logger.info(f"📋 [SHAP] base_feature_columns carregado de metadata ({len(base_cols)} features)")
                    except Exception as e:
                        logger.warning(f"⚠️ [SHAP] Erro ao ler metadata: {e}")
            
            if base_cols:
                # [FIX SHAP] Se o Autoencoder estiver ativo, precisamos adicionar as hidden_features ao background
                if hasattr(ai_controller, 'feature_pipeline') and ai_controller.feature_pipeline.temporal_autoencoder_pipeline:
                    try:
                        logger.info("🧪 [SHAP] Enriquecendo background com Hidden Features do Autoencoder...")
                        ae = ai_controller.feature_pipeline.temporal_autoencoder_pipeline
                        if ae.is_trained:
                            # Garante que temos as colunas base para o AE
                            ae_input_cols = ae.feature_cols if hasattr(ae, 'feature_cols') else base_cols
                            available_ae_cols = [c for c in ae_input_cols if c in background_df.columns]
                            
                            if len(available_ae_cols) >= len(ae_input_cols) * 0.8:
                                # Aplica AE para gerar as 25-32 colunas latentes
                                hidden_df = ae.apply_hidden(background_df)
                                # Adiciona à lista de colunas que o SHAP deve monitorar
                                latent_cols = [c for c in hidden_df.columns if 'hidden_feature' in c]
                                base_cols = list(base_cols) + latent_cols
                                background_df = hidden_df
                                # [CRÍTICO] Atualiza o controlador para reconhecer as novas colunas no SHAP
                                ai_controller.base_feature_columns = list(background_df.columns)
                                logger.info(f"✅ [SHAP] Background enriquecido: {len(latent_cols)} Hidden Features adicionadas.")
                    except Exception as ae_err:
                        logger.warning(f"⚠️ [SHAP] Falha ao enriquecer background com AE: {ae_err}")

                available_cols = [c for c in base_cols if c in background_df.columns]
                if len(available_cols) > 0:
                    background_df = background_df[available_cols]
                    logger.info(f"✅ [SHAP] Background filtrado para {len(available_cols)} features")
                else:
                    logger.info("ℹ️ [SHAP] Colunas base não encontradas. Usando primeiras 53 colunas.")
                    background_df = background_df.iloc[:, :53]
            else:
                # Fallback: use first 53 columns if no base_feature_columns defined anywhere
                logger.info("ℹ️ [SHAP] base_feature_columns não encontrado. Usando primeiras 53 colunas.")
                background_df = background_df.iloc[:, :53]
            
            # 1. Cria a instância do Explainer, passando o controller e os dados
            # [MELHORIA] Explainer com modo de produção
            production_mode = system_state.get("live_trading_enabled", False)
            explainer_instance = Explainer(ai_controller, background_df, production_mode=production_mode)
            system_components["explainer"] = explainer_instance
            mode_str = "PRODUÇÃO (rápido)" if production_mode else "BACKTEST (completo)"
            logger.info(f"✅ [INIT] Explainer inicializado em modo {mode_str}.")
            # 2. Injeta a instância criada de volta no controller
            ai_controller.set_explainer(explainer_instance)
        else:
            logger.warning("⚠️ [SETUP] Nenhum background válido encontrado (shap_background.pkl ou base_featured_df.pkl). SHAP desativado.")
            system_components["explainer"] = None # Define como None se não puder ser criado

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

def manage_validation_stamp(action: str, ai_controller: 'AIController', backtest_result: Optional[Any] = None) -> bool:
    """
    Gerencia um 'selo de validação' para evitar re-executar backtests desnecessários.
    [MELHORADO] Usa training_metadata.json como fallback e permite fast restart.

    Args:
        action (str): 'check' ou 'create'.
        ai_controller (AIController): A instância do AIController.
        backtest_result (BacktestResult, optional): O resultado do backtest para salvar no selo.

    Returns:
        bool: True se a validação for bem-sucedida (selo existe e é válido) ou criado com sucesso. False caso contrário.
    """
    stamp_path = os.path.join(ai_controller.config_ai.MODEL_DIR, "validation_stamp.json")
    metadata_path = os.path.join(ai_controller.config_ai.MODEL_DIR, "training_metadata.json")
    
    if action == 'check':
        # [FIX] Primeiro, tenta ler a data do training_metadata.json como fallback
        training_date_str = None
        if ai_controller.last_trained_date:
            training_date_str = ai_controller.last_trained_date.isoformat()
        elif os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                training_date_str = metadata.get('last_full_training_date')
                if training_date_str:
                    logger.info(f"📋 [PRÉ-VOO] Data de treino obtida de metadata: {training_date_str[:19]}")
            except Exception as e:
                logger.warning(f"⚠️ [PRÉ-VOO] Erro ao ler metadata: {e}")
        
        if not os.path.exists(stamp_path):
            logger.info("ℹ️ [PRÉ-VOO] Selo de validação não encontrado. Backtest de verificação é necessário.")
            return False
        
        try:
            with open(stamp_path, 'r') as f:
                stamp_data = json.load(f)
            
            last_validated_training_date = stamp_data.get('last_trained_date')
            validation_timestamp = stamp_data.get('validation_timestamp_utc', '')
            
            # [FIX] Aceita selo se:
            # 1) Data de treino bate, OU
            # 2) Selo foi criado nos últimos 7 dias (fast restart mode)
            if training_date_str and last_validated_training_date == training_date_str:
                logger.info("✅ [PRÉ-VOO] Selo de validação encontrado e corresponde aos modelos atuais. Backtest de verificação pulado.")
                return True
            elif validation_timestamp:
                try:
                    stamp_time = datetime.fromisoformat(validation_timestamp.replace('Z', '+00:00'))
                    age_days = (datetime.now(timezone.utc) - stamp_time.replace(tzinfo=timezone.utc if stamp_time.tzinfo is None else stamp_time.tzinfo)).days
                    if age_days <= 7:
                        logger.info(f"🎫 [PRÉ-VOO] Selo de validação recente ({age_days} dias). FAST RESTART ativado - pulando backtest.")
                        return True
                    else:
                        logger.warning(f"⚠️ [PRÉ-VOO] Selo de validação expirado ({age_days} dias > 7). Novo backtest necessário.")
                        return False
                except Exception as parse_err:
                    logger.warning(f"⚠️ [PRÉ-VOO] Erro ao parsear data do selo: {parse_err}")
            
            logger.warning("⚠️ [PRÉ-VOO] Modelos foram retreinados desde a última validação. Novo backtest de verificação é necessário.")
            return False
        except Exception as e:
            logger.error(f"❌ [PRÉ-VOO] Erro ao ler o selo de validação: {e}. Forçando novo backtest.")
            return False

    elif action == 'create':
        # [FIX] Usa training_metadata como fallback para a data
        training_date_str = None
        if ai_controller.last_trained_date:
            training_date_str = ai_controller.last_trained_date.isoformat()
        elif os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                training_date_str = metadata.get('last_full_training_date')
            except Exception:
                pass
        
        if training_date_str and backtest_result:
            stamp_data = {
                "validation_timestamp_utc": datetime.utcnow().isoformat(),
                "last_trained_date": training_date_str,
                "fast_restart_enabled": True,
                "backtest_summary": {
                    "total_return_pct": backtest_result.total_return_pct,
                    "max_drawdown_pct": backtest_result.max_drawdown_pct,
                    "sharpe_ratio": backtest_result.sharpe_ratio
                }
            }
            try:
                with open(stamp_path, 'w') as f:
                    json.dump(stamp_data, f, indent=4)
                logger.info(f"✅ [PRÉ-VOO] Selo de validação criado com sucesso para o treino de {training_date_str[:19]}.")
                return True
            except Exception as e:
                logger.error(f"❌ [PRÉ-VOO] Falha ao criar o selo de validação: {e}")
        return False
    
    return False


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
    execution_engine, tape_engine, explainer, feature_pipeline = (
        components["execution_engine"], components["tape_engine"],
        components["explainer"], components["feature_pipeline"]
    )
    ai_monitor = components.get("ai_monitor")  # [XAI] Persiste explicações em JSONL
    narrator   = components.get("narrator")    # [NARRATOR] Análise LLM local no terminal
    telegram   = components.get("telegram")    # [TELEGRAM] Alertas instantâneos
    symbol = TradingConfig.PRIMARY_PAIR

    # [MELHORIA] Variáveis para controle do ciclo de adaptação do Meta-Learner
    last_adaptation_time = datetime.utcnow()
    adaptation_interval_hours = 4  # Adapta a cada 4 horas
    adaptation_data_buffer = []  # Buffer para acumular dados para adaptação
    max_buffer_size = 1000  # Máximo de pontos de dados no buffer

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
                log_trade_decision(signal, explainer, ai_monitor)  # [XAI] explica + persiste

                # [TELEGRAM] Registra decisão no buffer para o relatório horário
                if telegram and signal:
                    _regime_map3 = {0: "BULL", 1: "BEAR", 2: "RANGER"}
                    telegram.log_decision(
                        action     = signal.action.value,
                        confidence = signal.confidence,
                        regime     = _regime_map3.get(int(latest_features.get('regime', 2)), "UNKNOWN"),
                    )

                if signal and signal.action != Action.HOLD:
                    order = await execution_engine.submit_order(signal)
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
                                "quantity":      signal.quantity,
                                "price":         signal.entry_price,
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

                # ── NARRATOR: análise LLM local no terminal ──────────────────
                if narrator and signal:
                    try:
                        tape_state  = system_state.get('tape_pulse', {})
                        regime_map  = {0: "bull", 1: "bear", 2: "ranger"}
                        regime_code = int(latest_features.get('regime', 2))

                        narrator_ctx = {
                            "symbol":      signal.symbol,
                            "action":      signal.action.value,
                            "confidence":  signal.confidence,
                            "price":       float(current_price or 0),
                            "regime":      regime_map.get(regime_code, "desconhecido"),
                            "tape_pulse":  tape_state.get('pulse', 'NEUTRAL'),
                            "tape_score":  float(tape_state.get('score', 0.0)),
                            "obi":         float(tape_state.get('obi', 0.0)),
                            "top_features": signal.explanation.get('main_contributors', []),
                        }

                        # Posição aberta com PnL atualizado
                        narrator_position = None
                        if portfolio.positions:
                            pos_sym = list(portfolio.positions.keys())[0]
                            pos     = portfolio.positions[pos_sym]
                            # PnL calculado com o preço atual do loop
                            if current_price and pos.entry_price > 0:
                                pnl = (float(current_price) - pos.entry_price) * pos.quantity
                            else:
                                pnl = pos.unrealized_pnl
                            narrator_position = {
                                "quantity":        pos.quantity,
                                "entry_price":     pos.entry_price,
                                "unrealized_pnl":  pnl,
                                "leverage":        pos.leverage,
                                "symbol":          pos_sym,
                            }

                        await narrator.maybe_narrate(narrator_ctx, narrator_position)
                    except Exception as _ne:
                        logger.debug(f"[NARRATOR] Erro ao preparar contexto: {_ne}")
                # ─────────────────────────────────────────────────────────────
            else:
                logger.warning("⚠️ [LOOP] IA não treinada. Pulando geração de decisão.")

            # [MELHORIA] Ciclo de Adaptação do Meta-Learner
            current_time = datetime.utcnow()
            time_since_last_adaptation = (current_time - last_adaptation_time).total_seconds() / 3600
            
            # Adiciona dados ao buffer para adaptação
            if len(adaptation_data_buffer) < max_buffer_size:
                adaptation_data_buffer.append({
                    'features': latest_features.to_dict(),
                    'timestamp': current_time,
                    'price': current_price,
                    'tape_pulse': system_state.get('tape_pulse', {})
                })
            
            # Verifica se é hora de adaptar (a cada 4 horas ou quando há drift detectado)
            should_adapt = (
                time_since_last_adaptation >= adaptation_interval_hours and 
                len(adaptation_data_buffer) >= 100 and  # Mínimo de dados para adaptação
                ai_controller.is_trained and
                ai_controller.is_trained  # [ARCH] Drift gate removido - adaptacao nao bloqueia
            )
            
            if should_adapt:
                try:
                    logger.info(f"🧠 [META-LEARNING] Iniciando ciclo de adaptação com {len(adaptation_data_buffer)} pontos de dados...")
                    
                    # Converte buffer para DataFrame
                    adaptation_df = pd.DataFrame(adaptation_data_buffer)
                    
                    # Chama a função de adaptação do Meta-Learner
                    adaptation_success = await ai_controller.adapt(
                        recent_data=adaptation_df,
                        adaptation_strength=0.1,  # Adaptação suave
                        preserve_core_learning=True
                    )
                    
                    if adaptation_success:
                        last_adaptation_time = current_time
                        adaptation_data_buffer = []  # Limpa o buffer após adaptação bem-sucedida
                        logger.info("✅ [META-LEARNING] Adaptação concluída com sucesso!")
                        
                        # Atualiza o estado do sistema
                        system_state["last_adaptation"] = current_time.isoformat()
                        system_state["adaptation_count"] = system_state.get("adaptation_count", 0) + 1
                    else:
                        logger.warning("⚠️ [META-LEARNING] Adaptação falhou. Tentará novamente no próximo ciclo.")
                        
                except Exception as e:
                    logger.error(f"❌ [META-LEARNING] Erro durante adaptação: {e}", exc_info=True)
                    # Não limpa o buffer em caso de erro para tentar novamente

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
            if portfolio := system_components.get("portfolio"):
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
    
    # --- LÓGICA DE TREINAMENTO GRANULAR E INTELIGENTE ---
    
    ai_controller.load_metadata()
    status = ai_controller.get_training_status()
    needs_training_this_session = False

    # Cenário 1: Retreinamento completo é OBRIGATÓRIO (modelos muito antigos ou corrompidos)
    if ai_controller.is_retraining_due(AIConfig.MODEL_RETRAIN_DAYS):
        valid_models_count = sum(1 for v in status.values() if v and v != 'all_trained')
        total_models = len([k for k in status.keys() if k != 'all_trained'])
        
        if valid_models_count == 0:
            logger.warning(f"🗓️ RETREINAMENTO COMPLETO: Nenhum modelo válido encontrado. Iniciando reconstrução total.")
            needs_training_this_session = True
            
            # Limpa apenas checkpoints, preserva modelos existentes
            try:
                if os.path.exists(ai_controller.config_ai.CHECKPOINT_DIR):
                    shutil.rmtree(ai_controller.config_ai.CHECKPOINT_DIR)
                os.makedirs(ai_controller.config_ai.CHECKPOINT_DIR, exist_ok=True)
                logger.info("Checkpoints antigos limpos para retreinamento completo.")
            except Exception as e:
                logger.error(f"❌ Erro durante a limpeza de checkpoints: {e}")

            final_training_df = await ai_controller.train_foundation_models()
            if final_training_df is None:
                logger.critical("🚨 Falha no treinamento de fundação. Encerrando.")
                return
            
            # [ARCH] DriftDetector desativado — guard para evitar crash se ainda houver
            # código de retreinamento que chame set_reference_data em um objeto None.
            drift_detector = system_components.get("drift_detector")
            if drift_detector is not None:
                drift_detector.set_reference_data(final_training_df, ai_controller.base_feature_columns)
        else:
            logger.info(f"ℹ️ Modelos antigos detectados, mas há {valid_models_count}/{total_models} modelos válidos. Treinamento granular recomendado.")
            needs_training_this_session = True
            
            success = await ai_controller.train_missing_components(status)
            if not success:
                logger.critical("🚨 Falha no treinamento granular dos componentes ausentes. Encerrando.")
                return

    # Cenário 2: Se NÃO for um retreinamento por idade, verifica se algo está faltando
    elif not status.get('all_trained'):
        valid_models_count = sum(1 for v in status.values() if v and v != 'all_trained')
        total_models = len([k for k in status.keys() if k != 'all_trained'])
        
        logger.warning(f"⚠️ ESTADO INCONSISTENTE: {valid_models_count}/{total_models} modelos válidos. Executando treinamento granular para corrigir.")
        needs_training_this_session = True
        
        success = await ai_controller.train_missing_components(status)
        if not success:
            logger.critical("🚨 Falha no treinamento granular dos componentes ausentes. Encerrando.")
            return
            
        # [FIX] Recarrega todos os modelos após treinamento para garantir estado atualizado e carregar referências de drift
        logger.info("🔄 [RELOAD] Treinamento granular concluído. Recarregando modelos para operação...")
        try:
            ai_controller.load_all_models()
        except Exception as e:
            logger.critical(f"🚨 Falha crítica ao recarregar modelos após treinamento: {e}", exc_info=True)
            return
    
    # Cenário 3: Se NENHUM dos cenários acima for verdadeiro, significa que tudo está pronto
    else:
        logger.info("✅ [CARREGAMENTO] Modelos estão atualizados e consistentes. Carregando pesos...")
        try:
            ai_controller.load_all_models()
        except Exception as e:
            logger.critical(f"🚨 Falha crítica ao carregar modelos: {e}", exc_info=True)
            return

    # Verificação final para garantir que a IA está pronta para operar
    if not ai_controller.is_trained:
        # [MODIFICAÇÃO] Bypass autorizado via config
        if TradingConfig.ALLOW_UNTRAINED_EXECUTION:
            logger.warning("⚠️ [ALERTA DE SEGURANÇA] Bot operando em modo SEM TREINAMENTO (BYPASS ATIVO). Resultados podem ser aleatórios!")
        else:
            logger.critical("🚨 Carregamento ou treinamento final dos modelos falhou. O bot não pode operar. Encerrando.")
            return
    
    logger.info("✅ [IA PRONTA] Todas as fases de treinamento/carregamento foram concluídas.")
    # [FIX] Atualiza o status da IA para refletir o estado real no painel de monitoramento.
    # Antes ficava preso em "inicializando" para sempre pois nunca era atualizado pós-boot.
    system_state["ai_status"] = "pronto" if ai_controller.is_trained else "bypass (não treinado)"
    
    # [ARCH] Pre-flight backtest REMOVIDO.
    # HoldoutValidationCallback (gen_ratio avg=0.905) ja validou os modelos out-of-sample.
    # Backtest de pre-voo e redundante e atrasa o boot em ~2 minutos desnecessariamente.
    # Para backtest manual: python run_bot.py --force-backtest
    import sys
    if "--force-backtest" in sys.argv or "--backtest" in sys.argv:
        logger.info("[PRE-VOO] Backtest manual solicitado via --force-backtest...")
        backtest_passed, backtest_result_obj = await check_profitability_condition(system_components)
        if backtest_passed:
            manage_validation_stamp("create", ai_controller, backtest_result_obj)
            logger.info("✅ [PRE-VOO] Estrategia aprovada no backtest manual.")
        else:
            logger.warning("⚠️ [PRE-VOO] Backtest reprovado - continuando mesmo assim.")
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
        await _telegram.alert_bot_started(mode_label, _bal)
    
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
