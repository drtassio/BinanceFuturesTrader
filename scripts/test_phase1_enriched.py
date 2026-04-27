import os
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config.settings import AIConfig, TradingConfig
from learning.trend_predictor import TrendPredictor
from specialists.trend_specialist import TrendSpecialist
from learning.fase1_dataset_filter import prepare_phase1_datasets


def build_enriched_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_path = os.path.join("models_ai", "base_featured_df.pkl")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset base não encontrado em {base_path}")

    df = pd.read_pickle(base_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    ai_cfg = AIConfig()
    trading_cfg = TradingConfig()
    trend_pred = TrendPredictor(ai_cfg, trading_cfg)
    trend_pred.load_model()
    df_enriched = trend_pred.annotate_priors(df)

    # Usa filtro de Fase 1 para reforçar períodos de tendência
    train_df, eval_df = prepare_phase1_datasets(df_enriched)

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    numeric_cols = train_df.select_dtypes(include="number").columns

    for frame in (train_df, eval_df):
        frame.loc[:, numeric_cols] = (
            frame[numeric_cols].ffill().bfill().fillna(0.0)
        )

    scaler = RobustScaler()
    scaler.fit(train_df[ohlcv_cols])
    train_df.loc[:, ohlcv_cols] = scaler.transform(train_df[ohlcv_cols])
    eval_df.loc[:, ohlcv_cols] = scaler.transform(eval_df[ohlcv_cols])

    return train_df, eval_df


def main() -> None:
    train_df, eval_df = build_enriched_frames()

    ai_cfg = AIConfig()
    input_dim = train_df.select_dtypes(include="number").shape[1]
    specialist = TrendSpecialist(ai_cfg, input_dim=input_dim)

    anti_scap_db = os.path.join(
        os.getcwd(), "logs", "trend_specialist_anti_scalping_config.db"
    )
    if os.path.exists(anti_scap_db):
        os.remove(anti_scap_db)

    result = specialist.optimize_anti_scalping_config(
        train_df=train_df,
        eval_df=eval_df,
        n_trials=3,
        n_timesteps=1000,
        n_eval_episodes=2,
    )

    print("=== RESULTADO TESTE FASE 1 ENRIQUECIDO ===")
    print(result)


if __name__ == "__main__":
    main()
