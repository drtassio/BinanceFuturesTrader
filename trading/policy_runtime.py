"""Bind approvals to execution code and explicitly non-secret settings."""
import hashlib
import json
from pathlib import Path

RUNTIME_FILES = (
    'config/settings.py', 'cloud/train_agent.py',
    'specialists/trend_specialist.py', 'specialists/base_regime_specialist.py',
    'specialists/bull_specialist.py', 'specialists/bear_specialist.py',
    'specialists/ranger_specialist.py', 'trading/ai_controller.py',
    'trading/execution_engine.py', 'trading/risk_manager.py',
    'trading/policy_exit.py', 'trading/policy_runtime.py',
)
SAFE_AI_SETTINGS = (
    'ECONOMIC_REWARD_ONLY', 'TRAINING_LEVERAGE_CAP', 'INFERENCE_ACTION_THRESHOLD',
    'OOS_MIN_SHARPE', 'OOS_MIN_PROFIT_FACTOR', 'OOS_MAX_DRAWDOWN',
    'OOS_MIN_NET_RETURN', 'OOS_MIN_TRADES',
    'ENABLE_RANGER_RULE_BASED_EXITS',
)
SAFE_TRADING_SETTINGS = (
    'LIVE_POLICY', 'PRIMARY_PAIR', 'MIN_LEVERAGE_PER_TRADE',
    'MAX_LEVERAGE_PER_TRADE', 'MAX_POSITION_SIZE_PERCENT',
    'MIN_CONFIDENCE_FOR_TRADE', 'DEFAULT_STOP_LOSS_PCT',
    'DEFAULT_TAKE_PROFIT_PCT', 'ENABLE_PROFIT_PROBABILITY_GATE',
)


def runtime_fingerprint(ai_config, trading_config=None, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    result = {}
    for filename in RUNTIME_FILES:
        try:
            result[filename] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        except OSError:
            result[filename] = None
    # Never enumerate configs or hash .env: both may contain credentials.
    settings = {'ai': {key: getattr(ai_config, key, None) for key in SAFE_AI_SETTINGS},
                'trading': {key: getattr(trading_config, key, None)
                            for key in SAFE_TRADING_SETTINGS}}
    result['effective_settings'] = hashlib.sha256(
        json.dumps(settings, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return result
