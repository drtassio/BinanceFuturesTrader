import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config.settings import AIConfig
from specialists.trend_specialist import TrendSpecialist


def main():
    base_path = os.path.join('models_ai', 'base_featured_df.pkl')
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset base não encontrado em {base_path}")

    df = pd.read_pickle(base_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'])
        else:
            df.index = pd.to_datetime(df.index)

    train_df, eval_df = train_test_split(df, test_size=0.2, shuffle=False)

    cfg = AIConfig()
    input_dim = df.select_dtypes(include='number').shape[1]
    ts = TrendSpecialist(cfg, input_dim=input_dim)
    # Reinicia estudo da FASE 1 para o teste curto
    anti_scap_db = os.path.join(os.getcwd(), 'logs', 'trend_specialist_anti_scalping_config.db')
    try:
        if os.path.exists(anti_scap_db):
            os.remove(anti_scap_db)
    except Exception:
        pass
    res = ts.optimize_anti_scalping_config(
        train_df=train_df,
        eval_df=eval_df,
        n_trials=3,
        n_timesteps=1000,
        n_eval_episodes=2,
    )
    print("\n=== RESULTADO TESTE CURTO FASE 1 ===")
    print(res)


if __name__ == "__main__":
    main()


