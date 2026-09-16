"""Validation-only reference using causal walk-forward edge features.

No labels, holdout inspection, threshold search, orders or model promotion.
Uses the specialist's real costs and stops, not triple-barrier label returns.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class EdgeReference:
    def __init__(self, frame, side):
        self.frame, self.side, self.cursor = frame, side, 0

    def predict(self, observation, deterministic=True):
        row = self.frame.iloc[min(self.cursor, len(self.frame) - 1)]
        self.cursor += 1
        probability = float(row[f'ml_p_{"long" if self.side == 1 else "short"}'])
        edge = self.side * float(row['ml_edge'])
        aligned = (float(row['tp_regime_up']) > float(row['tp_regime_down'])
                   if self.side == 1 else
                   float(row['tp_regime_down']) > float(row['tp_regime_up']))
        active = probability >= 0.60 and edge >= 0.10 and aligned
        vote = self.side * 0.8 if active else -self.side * 0.8
        return np.array([[vote, 3.0, 1.0]], dtype=np.float32), None


def main():
    import joblib
    from cloud.train_agent import evaluate, load_dataset, split_chronological
    from config.settings import AIConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=('bull', 'bear'), required=True)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    _, frame, _ = split_chronological(load_dataset(ROOT / 'data/featured_data_causal.parquet'))
    required = {'ml_p_long', 'ml_p_short', 'ml_edge', 'tp_regime_up', 'tp_regime_down'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing causal edge features: {required - set(frame.columns)}')
    run = args.run_directory.resolve()
    scaler = joblib.load(run / 'models' / f'{args.agent}_specialist_scaler.joblib')
    config = AIConfig()
    config.ECONOMIC_REWARD_ONLY = True

    def make_environment(data, *, env_class, **kwargs):
        return env_class(df=data, config=config, feature_scaler=scaler,
                         specialist_name=f'{args.agent}_specialist', **kwargs)

    agent = SimpleNamespace(model=EdgeReference(frame, 1 if args.agent == 'bull' else -1),
                            feature_columns=list(scaler.feature_names_in_),
                            _make_trend_env=make_environment)
    result = evaluate(agent, frame, args.agent, deterministic=True)
    report = {'policy': 'fixed_causal_edge_reference', 'split': 'validation_only',
              'thresholds': {'probability': 0.60, 'directional_edge': 0.10},
              'metrics': result}
    output = args.output or run / f'{args.agent}_edge_reference_validation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
