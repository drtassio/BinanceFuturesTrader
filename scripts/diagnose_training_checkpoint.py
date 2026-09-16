"""Read-only checkpoint diagnosis on validation, never on final holdout.

Does not promote models or overwrite their scalers. Safe during training;
retry later if the checkpoint is being rewritten concurrently.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import io
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import joblib
    import torch
    from stable_baselines3 import SAC
    from cloud.train_agent import load_dataset, split_chronological, evaluate
    from config.settings import AIConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--agent", choices=("bull", "bear", "ranger"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = args.run_directory.resolve()
    checkpoint = run / "checkpoints" / f"{args.agent}_specialist_checkpoints" / "best_model.zip"
    checkpoint_bytes = checkpoint.read_bytes()
    scaler = joblib.load(run / "models" / f"{args.agent}_specialist_scaler.joblib")
    features = list(scaler.feature_names_in_)
    config = AIConfig()
    config.ECONOMIC_REWARD_ONLY = True
    config.MODEL_DIR = str(run / "models")
    config.CHECKPOINT_DIR = str(run / "checkpoints")

    def make_environment(frame, *, env_class, **kwargs):
        return env_class(df=frame, config=config, feature_scaler=scaler,
                         specialist_name=f"{args.agent}_specialist", **kwargs)

    agent = SimpleNamespace(
        model=SAC.load(io.BytesIO(checkpoint_bytes), device="cpu"),
        feature_columns=features, _make_trend_env=make_environment,
    )
    _, validation, _ = split_chronological(
        load_dataset(ROOT / "data" / "featured_data_causal.parquet")
    )
    result = evaluate(agent, validation, args.agent, deterministic=True)
    report = {"checkpoint": str(checkpoint),
              "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
              "split": "validation_only", "phase3_runtime_rules": True,
              "validation_start": str(validation.index.min()),
              "validation_end": str(validation.index.max()), "metrics": result}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
