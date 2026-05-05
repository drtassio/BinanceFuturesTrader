
import json
import os

def filter_core_features(all_features):
    # 1. Features sem sufixo de timeframe: latents + meta — SEMPRE incluir
    _ALWAYS_PREFIXES = ('hidden_feature',)
    _ALWAYS_CONTAINS = ('sdae_recon', 'regime_conf', 'tp_prior', 'uncertainty', 'trend_pred', 'qpnl', 'trend_success', 'predicted_')

    # 2. Timeframes a EXCLUIR das features técnicas
    _EXCLUDED_TF = ('_1m', '_5m', '_4h')

    # 3. Padrões técnicos aceitos SOMENTE em _15m e _1h
    _TECH_PATTERNS = (
        'log_return', 'hl_range_pct', 'oc_move_pct', 'close_frac',
        'realized_vol',
        'rsi',
        'macd_hist', 'macd_cross',
        'adx',
        'plus_di', 'minus_di',
        'bb_width', 'bb_squeeze', 'bb_expansion',
        'atr_percentage',
        'money_flow', 'volume_pressure',
        'ema_trend', 'psar_trend', 'cmf'
    )

    core = []
    for f in all_features:
        fl = f.lower()

        # Latents
        if any(fl.startswith(p) for p in _ALWAYS_PREFIXES):
            core.append(f)
            continue

        # Meta features
        if any(p in fl for p in _ALWAYS_CONTAINS):
            core.append(f)
            continue

        # Excluir timeframes
        if any(fl.endswith(tf) or (tf + '_') in fl for tf in _EXCLUDED_TF):
            continue

        # Features técnicas em _15m e _1h
        if any(p in fl for p in _TECH_PATTERNS):
            if any(tf in fl for tf in ('_15m', '_1h')):
                core.append(f)

    return core

# Load metadata
meta_path = 'models_ai/training_metadata.json'
with open(meta_path, 'r') as f:
    data = json.load(f)

all_feats = data['base_feature_columns']
core_feats = filter_core_features(all_feats)

# Adiciona as features de estado que são injetadas no AIController
state_feats = ["pos_side", "pnl", "steps", "sin_h", "cos_h", "sin_d", "cos_d", "prior_dir", "prior_conf"]
final_list = core_feats + state_feats

print(json.dumps(final_list, indent=4))
print(f"\nTotal Core Features: {len(final_list)}")
