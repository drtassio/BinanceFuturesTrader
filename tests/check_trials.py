import optuna

dbs = {
    'Bull': 'logs/bull_specialist_sac_optuna.db',
    'Bear': 'logs/bear_specialist_sac_optuna.db',
    'Ranger': 'logs/ranger_specialist_sac_optuna.db'
}

for n, p in dbs.items():
    try:
        s = optuna.study.get_all_study_summaries(f"sqlite:///{p}")[0]
        print(f"{n}: {s.n_trials} trials")
    except Exception as e:
        print(f"{n}: ERROR - {e}")
