import optuna
import os

db_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\logs\bull_specialist_sac_optuna.db"

if not os.path.exists(db_path):
    print(f"Erro: Arquivo {db_path} não encontrado.")
else:
    try:
        study = optuna.load_study(study_name="bull_specialist_sac_optimization_v3", storage=f"sqlite:///{db_path}")
        print(f"Estudo: {study.study_name}")
        print(f"Total de Trials: {len(study.trials)}")
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        
        print(f"Completos: {len(completed)}")
        print(f"Podados (Pruned): {len(pruned)}")
        print(f"Falhos (Fail): {len(failed)}")
        
        if completed:
            print(f"Melhor trial: {study.best_trial.number}")
            print(f"Melhor valor: {study.best_trial.value}")
            print(f"Melhores parâmetros: {study.best_params}")
        else:
            print("Nenhum trial completo encontrado.")
            
    except Exception as e:
        print(f"Erro ao ler banco de dados: {e}")
