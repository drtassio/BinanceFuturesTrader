"""Replay a saved learned policy without training, promotion or orders."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TREND_SKIP_OPTUNA', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=['bull', 'bear', 'ranger'], required=True)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/featured_data_causal.parquet')
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation', 'holdout'],
                        default=['train', 'validation'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_directory.resolve()
    if 'holdout' in args.splits and not (directory / f'{args.agent}_guided_report.json').is_file():
        parser.error('holdout audit requires a completed run; never inspect it to select checkpoints')
    if 'holdout' in args.splits and args.checkpoint is not None:
        parser.error('holdout audits only the final selected model, not candidate checkpoints')
    checkpoint = directory / 'models' / (args.checkpoint or f'{args.agent}_specialist_sac.zip')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    from cloud.train_agent import AGENTS, load_dataset, split_chronological, evaluate
    from config.settings import AIConfig, TradingConfig
    from specialists.trend_specialist import ClippedSAC
    from trading.policy_runtime import runtime_fingerprint
    import torch
    torch.set_num_threads(1)

    frame = load_dataset(args.data)
    frames = dict(zip(['train', 'validation', 'holdout'], split_chronological(frame)))
    config = AIConfig()
    config.MODEL_DIR = str(directory / 'models')
    config.ECONOMIC_REWARD_ONLY = True
    trading = TradingConfig()
    agent = AGENTS[args.agent](config=config, trading_config=trading,
                              input_dim=len(frame.select_dtypes(include='number').columns))
    agent.load_model()
    agent.model = ClippedSAC.load(str(checkpoint), device='cpu')
    results = {}
    for name in dict.fromkeys(args.splits):
        results[name] = evaluate(agent, frames[name], args.agent, deterministic=True)
        print(name, json.dumps(results[name], default=str), flush=True)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Checkpoint changed during audit; results cannot be bound to this file')
    report = dict(agent=args.agent, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                  dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  runtime_hashes=runtime_fingerprint(config, trading), metrics=results,
                  periods={name: [str(frames[name].index.min()), str(frames[name].index.max())]
                           for name in results},
                  approval=False, scope='Environment replay, not full live execution parity')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print('audit:', args.output, flush=True)


if __name__ == '__main__':
    main()
