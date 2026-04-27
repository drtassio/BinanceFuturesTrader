import optuna
import os

db_path = r"c:/Users/TAOLIVE/OneDrive - Daimler Truck/Desktop/aprendiz/Bot Shield/Bot Shield/logs/bull_specialist_sac_optuna.db"

try:
    studies = optuna.get_all_study_summaries(storage=f"sqlite:///{db_path}")
    for s in studies:
        print(f"Estudo: {s.study_name}")
        print(f"Total de Trials: {s.n_trials}")
        study = optuna.load_study(study_name=s.study_name, storage=f"sqlite:///{db_path}")
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"Completos: {len(completed)}")
        if completed:
            print(f"Melhor valor (Recompensa): {study.best_trial.value}")
            print(f"Melhores parâmetros: {study.best_params}")
        
        # Check failed or pruned trials
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        print(f"Podados: {len(pruned)}, Falhos: {len(failed)}")
except Exception as e:
    print(f"Erro: {e}")
