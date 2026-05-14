# -----------------------------------------------------------------------------
# ARQUIVO: config/settings.py
# -----------------------------------------------------------------------------

"""
Configuration settings for the Quantum Trading System.
"""

import os
from datetime import timedelta
from dotenv import load_dotenv
import logging

logger = logging.getLogger("SettingsConfig")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

load_dotenv()


class Config:
    """Classe de configuração base para todo o sistema."""

    SECRET_KEY = os.environ.get("SESSION_SECRET")
    if not SECRET_KEY:
        raise RuntimeError(
            "SESSION_SECRET nao definida no .env. Fallback hardcoded removido por seguranca."
        )
    if SECRET_KEY == "a_super_secret_key_for_quantum_trading_2024":
        logger.warning(
            "🚨 [ALERTE CONFIG] SECRET_KEY padrão em uso. Por favor, defina SESSION_SECRET no .env para segurança em produção."
        )

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_recycle": 280,
        "pool_pre_ping": True,
        "pool_size": 15,
        "max_overflow": 25,
    }
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        logger.error(
            "âŒ [ERRO CONFIG] VariÃ¡vel de ambiente 'DATABASE_URL' nÃ£o definida. Usando padrÃ£o localhost."
        )
        DATABASE_URL = (
            "postgresql://quantum_user:quantum_pass@localhost:5432/quantum_trading_db"
        )
    SQLALCHEMY_DATABASE_URI = DATABASE_URL

    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
    REDIS_DB = int(os.environ.get("REDIS_DB", 0))
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

    BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET")
    BINANCE_TESTNET = os.environ.get("BINANCE_TESTNET", "True").lower() in (
        "true",
        "1",
        "t",
    )

    # API Keys separadas para Testnet (Paper Trading)
    # Se BINANCE_TESTNET=True e estas chaves existirem, usa elas para trading
    BINANCE_TESTNET_API_KEY = os.environ.get("BINANCE_TESTNET_API_KEY")
    BINANCE_TESTNET_API_SECRET = os.environ.get("BINANCE_TESTNET_API_SECRET")

    # PAPER_TRADING_PURE: Modo de simulação PURO que não usa API autenticada
    # Ideal para regiões com bloqueio geográfico (ex: Brasil - Futures proibidos)
    # Quando True: usa apenas dados públicos (klines) e simula ordens localmente
    PAPER_TRADING_PURE = os.environ.get("PAPER_TRADING_PURE", "False").lower() in (
        "true",
        "1",
        "t",
    )
    if PAPER_TRADING_PURE:
        logger.info(
            "📝 [CONFIG] Modo PAPER_TRADING_PURE ativo. Ordens serão simuladas localmente (sem API de ordens)."
        )

    # FAST_STARTUP: Inicialização rápida para paper trading
    # Quando True: pula verificações desnecessárias se modelos já estão treinados
    FAST_STARTUP = os.environ.get("FAST_STARTUP", "False").lower() in ("true", "1", "t")
    if FAST_STARTUP:
        logger.info(
            "⚡ [CONFIG] Modo FAST_STARTUP ativo. Inicialização otimizada para paper trading."
        )

    # Telegram Bot — alertas instantâneos + relatório horário
    # Obtenha o token via @BotFather e o chat_id via @userinfobot
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logger.critical(
            "ðŸš¨ [CRÃ TICO CONFIG] Chaves de API da Binance nÃ£o definidas. O bot NÃƒO poderÃ¡ operar em modo LIVE ou PAPER."
        )

    pass


class TradingConfig(Config):
    """ConfiguraÃ§Ãµes especÃ­ficas para o sistema de trading e gerenciamento de risco."""

    INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", 100.0))
    if INITIAL_CAPITAL <= 0:
        logger.error(
            "â Œ [ERRO CONFIG] INITIAL_CAPITAL deve ser um valor positivo. Usando padrÃ£o 100000.0."
        )
        INITIAL_CAPITAL = 200

    TRADING_PAIRS = os.environ.get("TRADING_PAIRS", "BTCUSDT").split(",")
    PRIMARY_PAIR = os.environ.get("PRIMARY_PAIR", "BTCUSDT")
    if PRIMARY_PAIR not in TRADING_PAIRS:
        logger.warning(
            f"âš ï¸ [ALERTE CONFIG] PRIMARY_PAIR '{PRIMARY_PAIR}' nÃ£o estÃ¡ na lista TRADING_PAIRS. Pode haver inconsistÃªncias."
        )

    PRICE_COLS = ["open", "high", "low", "close", "volume"]

    # --- ConfiguraÃ§Ãµes Multi-Timeframe ---
    PRIMARY_TIMEFRAME_TRADING = os.environ.get(
        "PRIMARY_TIMEFRAME_TRADING", "15m"
    )  # Exemplo para 15 minutos
    SECONDARY_TIMEFRAMES = os.environ.get("SECONDARY_TIMEFRAMES", "1h,4h").split(
        ","
    )  # Timeframes acima
    MICRO_TIMEFRAMES = os.environ.get("MICRO_TIMEFRAMES", "1m,5m").split(
        ","
    )  # Timeframes abaixo
    ALL_TRADING_TIMEFRAMES = (
        [PRIMARY_TIMEFRAME_TRADING] + SECONDARY_TIMEFRAMES + MICRO_TIMEFRAMES
    )

    MAX_PORTFOLIO_RISK_PERCENT = float(
        os.environ.get("MAX_PORTFOLIO_RISK_PERCENT", 0.02)
    )  # Risco por trade como % do portfÃ³lio
    MAX_POSITION_SIZE_PERCENT = float(
        os.environ.get("MAX_POSITION_SIZE_PERCENT", 0.15)
    )  # Max % do capital para MARGEM (Subiu de 10% para 15%)
    MAX_TOTAL_EXPOSURE_PERCENT = float(
        os.environ.get("MAX_TOTAL_EXPOSURE_PERCENT", 0.80)
    )  # Max % do capital para EXPOSIÃ‡ÃƒO TOTAL (valor nocional)

    # --- ConfiguraÃ§Ãµes de Alavancagem DinÃ¢mica ---
    # MAX_LEVERAGE Ã© a alavancagem mÃ¡xima permitida pela exchange ou pelo seu sistema.
    # MAX_LEVERAGE_PER_TRADE Ã© o limite que o agente pode *predizer* ou que o RiskManager pode *permitir*.
    MAX_LEVERAGE = float(
        os.environ.get("MAX_LEVERAGE", 5.0)
    )  # MÃ¡ximo de alavancagem permitida pela corretora
    MAX_LEVERAGE_PER_TRADE = float(
        os.environ.get("MAX_LEVERAGE_PER_TRADE", 5.0)
    )  # Limite que o agente pode usar (Subiu de 3x para 5x)
    MIN_LEVERAGE_PER_TRADE = float(
        os.environ.get("MIN_LEVERAGE_PER_TRADE", 1.0)
    )  # MÃ­nimo de alavancagem
    LEVERAGE_PREDICTION_RANGE = (
        MIN_LEVERAGE_PER_TRADE,
        MAX_LEVERAGE_PER_TRADE,
    )  # Range para a aÃ§Ã£o de alavancagem do agente
    LEVERAGE_COST_PER_DAY_PCT = float(
        os.environ.get("LEVERAGE_COST_PER_DAY_PCT", 0.0003)
    )  # Custo diÃ¡rio da alavancagem (para funding fee, se aplicÃ¡vel em simulaÃ§Ã£o)

    DEFAULT_STOP_LOSS_PCT = float(os.environ.get("DEFAULT_STOP_LOSS_PCT", 0.03))
    DEFAULT_TAKE_PROFIT_PCT = float(os.environ.get("DEFAULT_TAKE_PROFIT_PCT", 0.09))
    ATR_STOP_LOSS_MULTIPLIER = float(os.environ.get("ATR_STOP_LOSS_MULTIPLIER", 2.5))
    ATR_TAKE_PROFIT_MULTIPLIER = float(
        os.environ.get("ATR_TAKE_PROFIT_MULTIPLIER", 7.5)
    )
    SL_DIST_MULTIPLIER_MIN = float(
        os.environ.get("SL_DIST_MULTIPLIER_MIN", 1.0)
    )  # Evita SLs muito apertados
    SL_DIST_MULTIPLIER_MAX = float(
        os.environ.get("SL_DIST_MULTIPLIER_MAX", 5.0)
    )  # Limite superior
    TP_DIST_MULTIPLIER_MIN = float(
        os.environ.get("TP_DIST_MULTIPLIER_MIN", 0.5)
    )  # Limite inferior para TP
    TP_DIST_MULTIPLIER_MAX = float(
        os.environ.get("TP_DIST_MULTIPLIER_MAX", 10.0)
    )  # Limite superior para TP

    # Stop de CatÃ¡strofe para ambientes de RL (proteÃ§Ã£o hard-stop baseada em ATR)
    CATASTROPHE_SL_MULTIPLIER = float(os.environ.get("CATASTROPHE_SL_MULTIPLIER", 3.0))

    # Novos ranges para multiplicadores de SL/TP em regimes de alta volatilidade
    SL_DIST_MULTIPLIER_MIN_VOLATILE = float(
        os.environ.get("SL_DIST_MULTIPLIER_MIN_VOLATILE", 3.0)
    )
    SL_DIST_MULTIPLIER_MAX_VOLATILE = float(
        os.environ.get("SL_DIST_MULTIPLIER_MAX_VOLATILE", 10.0)
    )
    TP_DIST_MULTIPLIER_MIN_VOLATILE = float(
        os.environ.get("TP_DIST_MULTIPLIER_MIN_VOLATILE", 2.0)
    )
    TP_DIST_MULTIPLIER_MAX_VOLATILE = float(
        os.environ.get("TP_DIST_MULTIPLIER_MAX_VOLATILE", 20.0)
    )

    MAX_VAR_1D_PERCENT = float(os.environ.get("MAX_VAR_1D_PERCENT", 0.05))
    MAX_DRAWDOWN_PERCENT = float(os.environ.get("MAX_DRAWDOWN_PERCENT", 0.15))
    MIN_SHARPE_RATIO = float(os.environ.get("MIN_SHARPE_RATIO", 0.8))
    MIN_SORTINO_RATIO = float(os.environ.get("MIN_SORTINO_RATIO", 1.2))
    KELLY_FRACTION = float(
        os.environ.get("KELLY_FRACTION", 0.6)
    )  # Fração de Kelly (Subiu de 0.5 para 0.6 para mais agressividade)

    # --- Parâmetros de Qualidade do Sinal ---
    MIN_CONFIDENCE_FOR_TRADE = float(os.environ.get("MIN_CONFIDENCE_FOR_TRADE", 0.60))
    MIN_PROFIT_PROBABILITY = float(os.environ.get("MIN_PROFIT_PROBABILITY", 0.55))

    # --- Taxas de Trading Reais da Binance Futures VIP0 (BTCUSDT Perp) ---
    # Documentação: https://www.binance.com/en/fee/trading
    TAKER_FEE = float(os.environ.get("TAKER_FEE", 0.0004))  # 0.04% VIP0 USDT-M Futures
    MAKER_FEE = float(os.environ.get("MAKER_FEE", 0.0002))  # 0.02% VIP0 USDT-M Futures
    BNB_DISCOUNT_FEE_MULTIPLIER = float(
        os.environ.get("BNB_DISCOUNT_FEE_MULTIPLIER", 0.90)
    )  # 10% de desconto ao usar BNB
    USDC_MAKER_FEE = float(
        os.environ.get("USDC_MAKER_FEE", 0.0)
    )  # 0.000% para USDC Maker
    USDC_TAKER_FEE = float(
        os.environ.get("USDC_TAKER_FEE", 0.00036)
    )  # 0.036% para USDC Taker

    _fees_env = os.environ.get("FEES_PERC_TOTAL")
    if _fees_env is not None:
        FEES_PERC_TOTAL = float(_fees_env)
    else:
        FEES_PERC_TOTAL = max(TAKER_FEE, MAKER_FEE) * 2.0 * 100.0

    SLIPPAGE_TOLERANCE_PCT = float(
        os.environ.get("SLIPPAGE_TOLERANCE_PCT", 0.002)
    )  # 0.2%
    _slippage_bps_env = os.environ.get("SLIPPAGE_BPS")
    if _slippage_bps_env is not None:
        SLIPPAGE_BPS = int(float(_slippage_bps_env))
    else:
        total_slip_pct = float(os.environ.get("SLIPPAGE_TOLERANCE_PCT", 0.002))
        SLIPPAGE_BPS = max(1, int(round((total_slip_pct / 2.0) * 10000.0)))

    EXECUTION_TIMEOUT_SECONDS = int(os.environ.get("EXECUTION_TIMEOUT_SECONDS", 45))
    # [FIX 8] Tempo máximo que uma ordem LIMIT pode ficar pendente antes de ser cancelada
    # e re-executada como MARKET. 300s = 5 minutos. Ajustável via .env LIMIT_ORDER_TIMEOUT_SECONDS.
    LIMIT_ORDER_TIMEOUT_SECONDS = int(
        os.environ.get("LIMIT_ORDER_TIMEOUT_SECONDS", 300)
    )
    TWAP_DURATION_MINUTES = int(os.environ.get("TWAP_DURATION_MINUTES", 15))
    VWAP_LOOKBACK_PERIODS = int(os.environ.get("VWAP_LOOKBACK_PERIODS", 30))
    TWAP_THRESHOLD_PCT = float(
        os.environ.get("TWAP_THRESHOLD_PCT", 0.05)
    )  # Percentual do portfÃ³lio para acionar TWAP (0.05 = 5%)
    MIN_CONFIDENCE_FOR_LARGE_TRADE = float(
        os.environ.get("MIN_CONFIDENCE_FOR_LARGE_TRADE", 0.85)
    )

    if not (0 <= MAX_PORTFOLIO_RISK_PERCENT <= 1):
        logger.error(
            "âŒ [ERRO CONFIG] MAX_PORTFOLIO_RISK_PERCENT fora do intervalo [0, 1]."
        )
    if not (0 <= MAX_POSITION_SIZE_PERCENT <= 1):
        logger.error(
            "âŒ [ERRO CONFIG] MAX_POSITION_SIZE_PERCENT fora do intervalo [0, 1]."
        )
    if not (0 <= MAX_TOTAL_EXPOSURE_PERCENT <= 1):
        logger.error(
            "âŒ [ERRO CONFIG] MAX_TOTAL_EXPOSURE_PERCENT fora do intervalo [0, 1]."
        )
    if not (0 <= MAX_DRAWDOWN_PERCENT <= 1):
        logger.error("âŒ [ERRO CONFIG] MAX_DRAWDOWN_PERCENT fora do intervalo [0, 1].")


class AIConfig(Config):
    """ConfiguraÃ§Ãµes para os modelos de InteligÃªncia Artificial e Machine Learning."""

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models_ai")
    CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints_ai")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    HRL_LEARNING_RATE = float(os.environ.get("HRL_LEARNING_RATE", 3e-4))
    HRL_N_STEPS = int(os.environ.get("HRL_N_STEPS", 2048))
    HRL_BATCH_SIZE = int(os.environ.get("HRL_BATCH_SIZE", 64))
    HRL_GAMMA = float(os.environ.get("HRL_GAMMA", 0.99))
    HRL_GAE_LAMBDA = float(os.environ.get("HRL_GAE_LAMBDA", 0.95))
    HRL_CLIP_RANGE = float(os.environ.get("HRL_CLIP_RANGE", 0.2))
    HRL_ENT_COEF = float(os.environ.get("HRL_ENT_COEF", 0.01))
    HRL_VF_COEF = float(os.environ.get("HRL_VF_COEF", 0.5))

    # --- Configurações de Aprendizado Online (HRL/Master Adaptation) ---
    ONLINE_LEARNING_RATE = float(
        os.environ.get("ONLINE_LEARNING_RATE", 1e-5)
    )  # Baixo para estabilidade
    ONLINE_ADAPTATION_WINDOW = int(
        os.environ.get("ONLINE_ADAPTATION_WINDOW", 20)
    )  # Histórico recente de trades

    SPECIALIST_LEARNING_RATE = float(os.environ.get("SPECIALIST_LEARNING_RATE", 1e-4))
    SPECIALIST_BUFFER_SIZE = int(os.environ.get("SPECIALIST_BUFFER_SIZE", 200000))
    SPECIALIST_BATCH_SIZE = int(os.environ.get("SPECIALIST_BATCH_SIZE", 128))
    SPECIALIST_UPDATE_FREQUENCY_STEPS = int(
        os.environ.get("SPECIALIST_UPDATE_FREQUENCY_STEPS", 4)
    )

    # --- ConfiguraÃ§Ãµes MAML (Meta-Learning) ---
    MAML_INNER_LR = float(os.environ.get("MAML_INNER_LR", 0.01))
    MAML_OUTER_LR = float(os.environ.get("MAML_OUTER_LR", 0.001))
    MAML_NUM_INNER_UPDATES = int(
        os.environ.get("MAML_NUM_INNER_UPDATES", 5)
    )  # k-shot learning
    MAML_ADAPTATION_STEPS = int(
        os.environ.get("MAML_ADAPTATION_STEPS", 10)
    )  # Passos para adaptar na inferÃªncia
    MAML_NUM_OUTER_UPDATES = int(
        os.environ.get("MAML_NUM_OUTER_UPDATES", 500)
    )  # NÃºmero de meta-atualizaÃ§Ãµes

    # --- ConfiguraÃ§Ãµes de Autoencoder para PadrÃµes Ocultos ---
    AUTOENCODER_LATENT_DIM = int(
        os.environ.get("AUTOENCODER_LATENT_DIM", 32)
    )  # DimensÃ£o do espaÃ§o latente
    AUTOENCODER_TRAINING_EPOCHS = int(
        os.environ.get("AUTOENCODER_TRAINING_EPOCHS", 200)
    )
    # Ajustado para dar mais tempo de calibração: 32 épocas e 100 trials (pode sobrescrever via env)
    TREND_PREDICTOR_EPOCHS = int(os.environ.get("TREND_PREDICTOR_EPOCHS", 32))
    TREND_PREDICTOR_OPTUNA_TRIALS = int(
        os.environ.get("TREND_PREDICTOR_OPTUNA_TRIALS", 100)
    )
    TREND_PREDICTOR_TOPK_TRIALS = int(os.environ.get("TREND_PREDICTOR_TOPK_TRIALS", 20))

    REGIME_THRESHOLD = float(os.environ.get("REGIME_THRESHOLD", 0.55))

    SECTION_SEPARATOR = "=========================================="

    # Allow execution even if models are not fully verified/trained (unsafe, for testing only)
    ALLOW_UNTRAINED_EXECUTION = os.environ.get(
        "ALLOW_UNTRAINED_EXECUTION", "False"
    ).lower() in ("true", "1", "t")

    SPECIALIST_CONFIG = {
        "BullSpecialist": {},
        "BearSpecialist": {},
        "RangerSpecialist": {},
    }

    # =========================================================================
    # SCIENTIFIC CORRECTIONS: Configurações para Temporal Autoencoder
    # Referência: De Prado (2018) - "Advances in Financial Machine Learning"
    # =========================================================================

    # Stationarity Validation (ADF/KPSS tests + Fractional Differencing)
    ENFORCE_STATIONARITY = os.environ.get("ENFORCE_STATIONARITY", "True").lower() in (
        "true",
        "1",
        "t",
    )

    # Walk-Forward Cross-Validation (Purged K-Fold) [DEFAULT: ON — ANTI-LEAKAGE]
    USE_WALK_FORWARD_CV = os.environ.get("USE_WALK_FORWARD_CV", "True").lower() in (
        "true",
        "1",
        "t",
    )

    # Bottleneck Constraint (minimum compression ratio to prevent identity mapping)
    MIN_COMPRESSION_RATIO = float(os.environ.get("MIN_COMPRESSION_RATIO", 4.0))

    # Embargo Periods (None = auto-calculate based on autocorrelation)
    EMBARGO_BARS = (
        int(os.environ.get("EMBARGO_BARS")) if os.environ.get("EMBARGO_BARS") else None
    )

    CURRICULUM_STAGES = [
        # min_training_runs: mínimo de ciclos de treino antes de avaliar (1 ciclo ≈ N especialistas treinados)
        # max_training_runs: força avanço de estágio mesmo sem atingir critérios (válvula de segurança)
        # NOTA: atr_percentage_1h está em % bruta (range real: 0.014–2.75).
        #        adx_1h está em escala bruta 0–100 (ADX<20=sem tendência, >40=forte).
        {
            "name": "Calm Markets",
            "volatility_max": 0.5,  # ATR% <= 0.5% (barras calmas)
            "trend_clarity_max": 20.0,  # ADX <= 20 (sem tendência clara)
            "episodes": 50000,
            "min_training_runs": 5,
            "max_training_runs": 20,
            "success_criteria": {"sharpe_ratio": 1.5, "max_drawdown_pct": 0.05},
        },
        {
            "name": "Moderate Volatility",
            "volatility_max": 1.5,  # ATR% <= 1.5%
            "trend_clarity_min": 20.0,  # ADX >= 20 (tendência se formando)
            "episodes": 100000,
            "min_training_runs": 5,
            "max_training_runs": 20,
            "success_criteria": {"sharpe_ratio": 1.0, "max_drawdown_pct": 0.10},
        },
        {
            "name": "Sideways & Choppy",
            "volatility_max": 2.0,  # ATR% <= 2.0%
            "trend_clarity_min": 0.0,
            "trend_clarity_max": 25.0,  # ADX <= 25 (tendência fraca/lateral)
            "episodes": 150000,
            "min_training_runs": 5,
            "max_training_runs": 25,
            "success_criteria": {"sharpe_ratio": 0.5, "max_drawdown_pct": 0.08},
        },
        {
            "name": "High Volatility",
            "volatility_max": 3.0,  # ATR% <= 3.0% (cobre ~99% dos dados)
            "trend_clarity_min": 0.0,
            "episodes": 200000,
            "min_training_runs": 5,
            "max_training_runs": 25,
            "success_criteria": {"sharpe_ratio": 0.7, "max_drawdown_pct": 0.12},
        },
        {
            "name": "Extreme Conditions (Crisis)",
            "volatility_max": 5.0,  # ATR% <= 5.0% (inclui tudo)
            "trend_clarity_min": 0.0,
            "episodes": 250000,
            "min_training_runs": 5,
            "max_training_runs": 30,
            "success_criteria": {"sharpe_ratio": 0.3, "max_drawdown_pct": 0.20},
        },
    ]
    # [CONVERGÊNCIA] 500k validado empiricamente: Sharpe 8+, WR 78%, PF 9.6, ent_coef ~0.04.
    # Aumentar via TOTAL_TRAINING_TIMESTEPS no .env se necessário.
    TOTAL_TRAINING_TIMESTEPS = int(os.environ.get("TOTAL_TRAINING_TIMESTEPS", 500_000))
    EVALUATION_FREQUENCY = int(os.environ.get("EVALUATION_FREQUENCY", 10000))
    SAVE_FREQUENCY_STEPS = int(os.environ.get("SAVE_FREQUENCY_STEPS", 50000))

    LOOKBACK_WINDOW_MAIN = int(
        os.environ.get("LOOKBACK_WINDOW_MAIN", 800)
    )  # 800 = 250 warmup + 500 buffer para garantir 32+ rows após merge
    TECHNICAL_INDICATORS = [
        "RSI",
        "MACD",
        "BollingerBands",
        "ATR",
        "EMA_Fast",
        "EMA_Slow",
        "Stochastic",
        "WilliamsR",
        "CCI",
        "ROC",
        "OBV",
        "VWAP",
        "ADX",
    ]
    REGIME_DETECTION_WINDOW = int(os.environ.get("REGIME_DETECTION_WINDOW", 100))
    REGIMES = [
        "trending_up",
        "trending_down",
        "sideways_volatile",
        "sideways_calm",
        "high_volatility_expansion",
    ]
    MIN_CONFIDENCE_FOR_TRADE = float(os.environ.get("MIN_CONFIDENCE_FOR_TRADE", 0.60))

    MIN_CONFIDENCE_FOR_LEVERAGE = float(
        os.environ.get("MIN_CONFIDENCE_FOR_LEVERAGE", 0.75)
    )  # Nova config para alavancagem
    MIN_PROFIT_PROBABILITY = float(
        os.environ.get("MIN_PROFIT_PROBABILITY", 0.55)
    )  # Filtro de qualidade de sinal (P)

    # --- ConfiguraÃ§Ãµes da FunÃ§Ã£o de Recompensa Refinada ---
    REWARD_PNL_WEIGHT = float(
        os.environ.get("REWARD_PNL_WEIGHT", 3.0)
    )  # Aumentado para dar mais peso ao PnL
    REWARD_SHARPE_RATIO_WEIGHT = float(
        os.environ.get("REWARD_SHARPE_RATIO_WEIGHT", 0.25)
    )  # Aumentado para considerar performance
    REWARD_RISK_ADJUSTED_RETURN_WEIGHT = float(
        os.environ.get("REWARD_RISK_ADJUSTED_RETURN_WEIGHT", 0.1)
    )  # Aumentado
    REWARD_PER_STEP_ALIGNED_TREND = float(
        os.environ.get("REWARD_PER_STEP_ALIGNED_TREND", 0.02)
    )  # Nova recompensa por estar alinhado Ã  tendÃªncia

    # --- Penalidades ---
    PENALTY_FEE_MULTIPLIER = float(
        os.environ.get("PENALTY_FEE_MULTIPLIER", 1.0)
    )  # Multiplicador para penalidade de taxa
    PENALTY_SLIPPAGE = float(os.environ.get("PENALTY_SLIPPAGE", 0.001))
    PENALTY_TRADE_FREQUENCY = float(os.environ.get("PENALTY_TRADE_FREQUENCY", 0.0007))
    PENALTY_DRAWDOWN = float(
        os.environ.get("PENALTY_DRAWDOWN", 7.5)
    )  # Aumentado para maior impacto

    # --- Penalidade de Holding DinÃ¢mica ---
    PENALTY_HOLDING_BASE = float(
        os.environ.get("PENALTY_HOLDING_BASE", 0.01)
    )  # Base da penalidade por passo
    PENALTY_HOLDING_FACTOR_PROB_LOW = float(
        os.environ.get("PENALTY_HOLDING_FACTOR_PROB_LOW", 2.0)
    )  # Multiplicador se P for baixo
    PENALTY_HOLDING_FACTOR_PROB_HIGH = float(
        os.environ.get("PENALTY_HOLDING_FACTOR_PROB_HIGH", 0.5)
    )  # Multiplicador se P for alto
    PENALTY_HOLDING_TOO_LONG = float(
        os.environ.get("PENALTY_HOLDING_TOO_LONG", 0.1)
    )  # Penalidade para especialistas Sideways/Range
    RELAXED_GATES_TREND = bool(int(os.environ.get("RELAXED_GATES_TREND", "0")))

    # =========================================================================
    # PARÂMETROS DE REWARD SHAPING CIENTÍFICO (Scientific RL Corrections)
    # Baseado em: AFLM (López de Prado 2018), Haarnoja SAC (2018)
    # =========================================================================

    # Anti-scalping: duração mínima para não ser penalizado (em steps/candles)
    # Ref: AFLM Ch.3 Triple Barrier - trades precisam de tempo para desenvolver
    ANTI_SCALP_MIN_DURATION = int(os.environ.get("ANTI_SCALP_MIN_DURATION", 3))
    ANTI_SCALP_PENALTY_SCALE = float(os.environ.get("ANTI_SCALP_PENALTY_SCALE", 2.0))

    # Conservative Q-Learning: clipping de reward para prevenir Q-overestimation
    # Ref: Kumar et al. 2020 CQL - "Conservative Q-Learning for Offline RL"
    REWARD_MAX_CLIP = float(os.environ.get("REWARD_MAX_CLIP", 15.0))
    REWARD_MIN_CLIP = float(os.environ.get("REWARD_MIN_CLIP", -15.0))

    # Vote Misalignment: penalidade proporcional à confiança do predictor
    # Quando predictor diz BULL com 100% confiança e agente vota SHORT: penalidade máxima
    VOTE_MISALIGN_BASE_PENALTY = float(
        os.environ.get("VOTE_MISALIGN_BASE_PENALTY", 2.0)
    )

    # Grace Period Emergency Exit: % de perda que anula o grace period
    GRACE_EMERGENCY_EXIT_THRESHOLD = float(
        os.environ.get("GRACE_EMERGENCY_EXIT_THRESHOLD", -0.02)
    )

    # Entropy Floor: impede convergência prematura (std da política cai abaixo disto)
    # Ref: Haarnoja et al. 2018 SAC - entropy regularization
    SAC_ENTROPY_FLOOR_LOG_STD = float(os.environ.get("SAC_ENTROPY_FLOOR_LOG_STD", -2.0))

    # Calmar Ratio bonus weight
    CALMAR_RATIO_WEIGHT = float(os.environ.get("CALMAR_RATIO_WEIGHT", 0.5))

    # Minimum hold reward bonus
    MIN_HOLD_DURATION_BONUS_STEPS = int(
        os.environ.get("MIN_HOLD_DURATION_BONUS_STEPS", 3)
    )
    MIN_HOLD_BONUS_VALUE = float(os.environ.get("MIN_HOLD_BONUS_VALUE", 0.3))

    # =========================================================================
    # PARÂMETROS SAC OTIMIZADOS (baseados no Best Trial + correções científicas)
    # =========================================================================

    # Prior Flip Penalty: sincronizado com o melhor trial Optuna (0.891)
    # ATENÇÃO: O HPO encontrou 0.891, mas o código estava usando 0.1.
    # Agora sincronizados para garantir reprodutibilidade do score 3.118
    TREND_PRIOR_FLIP_PENALTY_WEIGHT = float(
        os.environ.get("TREND_PRIOR_FLIP_PENALTY_WEIGHT", 0.891)
    )
    TREND_PRIOR_FLIP_PENALTY_EXP = float(
        os.environ.get("TREND_PRIOR_FLIP_PENALTY_EXP", 1.502)
    )
    TREND_PRIOR_FLIP_THRESHOLD = float(
        os.environ.get("TREND_PRIOR_FLIP_THRESHOLD", 0.331)
    )
    TREND_DURATION_PRESSURE_WEIGHT = float(
        os.environ.get("TREND_DURATION_PRESSURE_WEIGHT", 0.492)
    )
    TREND_DURATION_PRESSURE_START = int(
        os.environ.get("TREND_DURATION_PRESSURE_START", 157)
    )
    TREND_DURATION_PRESSURE_EXP = float(
        os.environ.get("TREND_DURATION_PRESSURE_EXP", 1.102)
    )
    TREND_DRAWDOWN_PENALTY_WEIGHT = float(
        os.environ.get("TREND_DRAWDOWN_PENALTY_WEIGHT", 1.059)
    )

    # SAC Hyperparameters (Best Trial #5, sincronizados)
    SAC_LEARNING_RATE_DEFAULT = float(
        os.environ.get("SAC_LEARNING_RATE_DEFAULT", 1.818e-05)
    )
    SAC_BUFFER_SIZE_DEFAULT = int(os.environ.get("SAC_BUFFER_SIZE_DEFAULT", 50000))
    SAC_BATCH_SIZE_DEFAULT = int(os.environ.get("SAC_BATCH_SIZE_DEFAULT", 1024))
    SAC_GAMMA_DEFAULT = float(os.environ.get("SAC_GAMMA_DEFAULT", 0.9955))
    SAC_TAU_DEFAULT = float(os.environ.get("SAC_TAU_DEFAULT", 0.00487))
    SAC_TRAIN_FREQ_DEFAULT = int(os.environ.get("SAC_TRAIN_FREQ_DEFAULT", 2))
    SAC_TARGET_ENTROPY_DEFAULT = float(
        os.environ.get("SAC_TARGET_ENTROPY_DEFAULT", -3.884)
    )
    SAC_MAX_GRAD_NORM_DEFAULT = float(
        os.environ.get("SAC_MAX_GRAD_NORM_DEFAULT", 0.867)
    )

    # Total timesteps mínimo para treino final (SAC precisa 500k-2M para convergir)
    # Ref: Haarnoja et al. 2018 - SAC training curves
    TREND_FINAL_TIMESTEPS_MIN = int(os.environ.get("TREND_FINAL_TIMESTEPS_MIN", 500000))
    HPO_MIN_TIMESTEPS_PER_TRIAL = int(
        os.environ.get("HPO_MIN_TIMESTEPS_PER_TRIAL", 25000)
    )

    # --- ConfiguraÃ§Ãµes do Modelo Preditivo de Probabilidade de Lucratividade (P) ---
    PROFIT_PREDICTOR_HIDDEN_DIM = int(
        os.environ.get("PROFIT_PREDICTOR_HIDDEN_DIM", 128)
    )
    REWARD_PROFIT_PROB_WEIGHT = float(
        os.environ.get("REWARD_PROFIT_PROB_WEIGHT", 1.0)
    )  # Recompensa por alta P
    REWARD_EXPECTED_PROFIT_WEIGHT = float(
        os.environ.get("REWARD_EXPECTED_PROFIT_WEIGHT", 5.0)
    )  # Recompensa por alto lucro esperado (escalado)

    MODEL_RETRAIN_DAYS = int(os.environ.get("MODEL_RETRAIN_DAYS", 90))
    # [FIX] 50k default: 1.5M / 50k = 30 checkpoints por treino final (era 1000 = 1500 checkpoints).
    # 1000 steps causava overhead massivo: 1500 eval × 10 episódios = 15k execuções de avaliação.
    CHECKPOINT_FREQ: int = int(os.getenv("CHECKPOINT_FREQ", 50000))
    PATIENCE_EARLY_STOPPING: int = int(os.getenv("PATIENCE_EARLY_STOPPING", 500))

    EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", 100))
    EARLY_STOP_MIN_DELTA = float(os.environ.get("EARLY_STOP_MIN_DELTA", 0.001))

    # 🆕 JEPA hyperparams
    JEPA_PREDICTION_HORIZON = int(os.environ.get("JEPA_PREDICTION_HORIZON", 8))
    JEPA_TAU_BASE = float(os.environ.get("JEPA_TAU_BASE", 0.996))
    JEPA_TAU_FINAL = float(os.environ.get("JEPA_TAU_FINAL", 1.0))

    # 🆕 Risk layer
    MAX_DRAWDOWN_INTRADAY = float(os.environ.get("MAX_DRAWDOWN_INTRADAY", 0.05))
    KELLY_FRACTION = float(os.environ.get("KELLY_FRACTION", 0.25))
    FUNDING_THRESHOLD = float(os.environ.get("FUNDING_THRESHOLD", 0.0005))

    # 🆕 Execution layer
    TAKER_FEE = float(os.environ.get("TAKER_FEE", 0.0004))
    MAKER_FEE = float(os.environ.get("MAKER_FEE", 0.0002))
    BASE_SLIPPAGE_BPS = float(os.environ.get("BASE_SLIPPAGE_BPS", 2.0))
    MAX_SLIPPAGE_BPS = float(os.environ.get("MAX_SLIPPAGE_BPS", 10.0))

    # 🆕 Monitoring
    DRIFT_WINDOW_SIZE = int(os.environ.get("DRIFT_WINDOW_SIZE", 1000))
    DRIFT_ALPHA = float(os.environ.get("DRIFT_ALPHA", 0.01))


class DataConfig(Config):
    """ConfiguraÃ§Ãµes para coleta e armazenamento de dados."""

    REALTIME_INTERVAL_SECONDS = int(os.environ.get("REALTIME_INTERVAL_SECONDS", 1))
    MINUTE_INTERVAL_SECONDS = int(os.environ.get("MINUTE_INTERVAL_SECONDS", 60))
    HOUR_INTERVAL_SECONDS = int(os.environ.get("HOUR_INTERVAL_SECONDS", 3600))
    HISTORICAL_DAYS_TO_FETCH = int(os.environ.get("HISTORICAL_DAYS_TO_FETCH", 730))

    _trading_pairs_lower = [pair.lower() for pair in TradingConfig.TRADING_PAIRS]
    BINANCE_STREAMS = (
        [f"{pair}@kline_1m" for pair in _trading_pairs_lower]
        + [f"{pair}@aggTrade" for pair in _trading_pairs_lower[:3]]
        + [f"{pair}@depth20@100ms" for pair in _trading_pairs_lower[:3]]
    )
    REALTIME_DATA_RETENTION_DAYS = int(
        os.environ.get("REALTIME_DATA_RETENTION_DAYS", 14)
    )
    AGGREGATED_DATA_RETENTION_DAYS = int(
        os.environ.get("AGGREGATED_DATA_RETENTION_DAYS", 365 * 5)
    )
    FEATURE_STORE_UPDATE_INTERVAL_SECONDS = int(
        os.environ.get("FEATURE_STORE_UPDATE_INTERVAL_SECONDS", 60)
    )


class BacktestConfig(Config):
    """ConfiguraÃ§Ãµes para o motor de backtesting."""

    START_DATE = os.environ.get("BACKTEST_START_DATE", "2022-01-01")
    END_DATE = os.environ.get("BACKTEST_END_DATE", "2024-01-01")
    INITIAL_CAPITAL = float(os.environ.get("BACKTEST_INITIAL_CAPITAL", 100.0))
    # Usar as taxas reais do TradingConfig para o Backtest tambÃ©m
    MAKER_FEE = TradingConfig.MAKER_FEE
    TAKER_FEE = TradingConfig.TAKER_FEE
    SLIPPAGE_MODEL = os.environ.get("BACKTEST_SLIPPAGE_MODEL", "volume_based")
    SLIPPAGE_CONSTANT = float(os.environ.get("BACKTEST_SLIPPAGE_CONSTANT", 0.0005))
    SLIPPAGE_IMPACT_FACTOR = float(
        os.environ.get("BACKTEST_SLIPPAGE_IMPACT_FACTOR", 5e-7)
    )
    WARMUP_PERIOD: int = int(os.getenv("WARMUP_PERIOD", 200))  # Passos para aquecimento
    EXECUTION_LATENCY_MS_MIN = int(
        os.environ.get("BACKTEST_EXECUTION_LATENCY_MS_MIN", 5)
    )
    EXECUTION_LATENCY_MS_MAX = int(
        os.environ.get("BACKTEST_EXECUTION_LATENCY_MS_MAX", 50)
    )
    BACKTEST_WITH_EXPLANATIONS = os.environ.get(
        "BACKTEST_WITH_EXPLANATIONS", "False"
    ).lower() in ("true", "1", "t")


class MonitoringConfig(Config):
    """ConfiguraÃ§Ãµes para monitoramento e alertas do sistema."""

    PERFORMANCE_WINDOW_HOURS = int(os.environ.get("PERFORMANCE_WINDOW_HOURS", 24))
    ALERT_DRAWDOWN_PCT = float(os.environ.get("ALERT_DRAWDOWN_PCT", 0.10))
    ALERT_ERROR_RATE_PCT = float(os.environ.get("ALERT_ERROR_RATE_PCT", 0.05))
    ALERT_MAX_LATENCY_MS = int(os.environ.get("ALERT_MAX_LATENCY_MS", 1500))
    HEALTH_CHECK_INTERVAL_SECONDS = int(
        os.environ.get("HEALTH_CHECK_INTERVAL_SECONDS", 60)
    )
    LOG_RETENTION_DAYS = int(os.environ.get("LOG_RETENTION_DAYS", 90))
    AI_MONITOR_LOG_RETENTION_DAYS = int(
        os.environ.get("AI_MONITOR_LOG_RETENTION_DAYS", 180)
    )


class StressTestConfig(Config):
    """CenÃ¡rios e configuraÃ§Ãµes para testes de estresse."""

    SCENARIOS = {
        "flash_crash": {
            "price_drop_pct": float(os.environ.get("STRESS_FLASH_CRASH_DROP", 0.30)),
            "duration_minutes": int(os.environ.get("STRESS_FLASH_CRASH_DURATION", 10)),
            "recovery_pct": float(os.environ.get("STRESS_FLASH_CRASH_RECOVERY", 0.5)),
            "max_allowable_drawdown": float(
                os.environ.get("STRESS_FLASH_CRASH_MAX_DRAWDOWN", 0.25)
            ),
        },
        "volatility_shock": {
            "volatility_multiplier": float(
                os.environ.get("STRESS_VOL_SHOCK_MULTIPLIER", 5.0)
            ),
            "duration_hours": int(os.environ.get("STRESS_VOL_SHOCK_DURATION", 6)),
            "max_allowable_drawdown": float(
                os.environ.get("STRESS_VOL_SHOCK_MAX_DRAWDOWN", 0.30)
            ),
        },
        "liquidity_crisis": {
            "spread_multiplier": float(
                os.environ.get("STRESS_LIQ_CRISIS_SPREAD", 10.0)
            ),
            "volume_divider": float(
                os.environ.get("STRESS_LIQ_CRISIS_VOLUME_DIV", 10.0)
            ),
            "duration_hours": int(os.environ.get("STRESS_LIQ_CRISIS_DURATION", 4)),
            "max_allowable_drawdown": float(
                os.environ.get("STRESS_LIQ_CRISIS_MAX_DRAWDOWN", 0.20)
            ),
        },
        "correlation_breakdown": {
            "target_correlation": float(
                os.environ.get("STRESS_CORR_BREAKDOWN_TARGET", -0.9)
            ),
            "duration_days": int(os.environ.get("STRESS_CORR_BREAKDOWN_DURATION", 5)),
            "max_allowable_drawdown": float(
                os.environ.get("STRESS_CORR_BREAKDOWN_MAX_DRAWDOWN", 0.40)
            ),
        },
        "api_outage": {
            "outage_duration_minutes": int(
                os.environ.get("STRESS_API_OUTAGE_DURATION", 120)
            ),
            "partial_outage_pct": float(
                os.environ.get("STRESS_API_OUTAGE_PARTIAL_PCT", 0.8)
            ),
            "max_allowable_drawdown": float(
                os.environ.get("STRESS_API_OUTAGE_MAX_DRAWDOWN", 0.10)
            ),
        },
        "black_swan": {
            "price_drop_pct": float(os.environ.get("STRESS_BLACK_SWAN_DROP", 0.60)),
            "duration_minutes": int(os.environ.get("STRESS_BLACK_SWAN_DURATION", 30)),
            "volatility_multiplier": float(
                os.environ.get("STRESS_BLACK_SWAN_VOL_MULT", 20.0)
            ),
            "liquidity_crisis": os.environ.get(
                "STRESS_BLACK_SWAN_LIQ_CRISIS", "True"
            ).lower()
            in ("true", "1", "t"),
            "max_allowable_drawdown": float(
                os.environ.get("STRESS_BLACK_SWAN_MAX_DRAWDOWN", 0.50)
            ),
        },
    }


class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False
    SQLALCHEMY_ECHO = True
    logger.info("[CONFIG] Modo de Desenvolvimento ativo.")


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    SQLALCHEMY_ECHO = False
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
    logger.info("[CONFIG] Modo de Producao ativo. DEBUG e SQL ECHO desativados.")


class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    logger.info("[CONFIG] Modo de Teste ativo. Usando SQLite em memoria.")


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}

env_config = os.getenv("FLASK_ENV", "default")
active_config = config_by_name[env_config]()

# Log para diagnosticar o valor de PENALTY_FEE
if hasattr(
    active_config, "PENALTY_FEE_MULTIPLIER"
):  # Alterado para PENALTY_FEE_MULTIPLIER
    logger.info(
        f"[CONFIG DEBUG] Valor de PENALTY_FEE_MULTIPLIER carregado: {active_config.PENALTY_FEE_MULTIPLIER}"
    )

logger.info(f"ConfiguraÃ§Ã£o ativa: {active_config.__class__.__name__}")
