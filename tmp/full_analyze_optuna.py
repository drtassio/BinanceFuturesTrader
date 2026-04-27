import optuna
import os

db_path = r"c:/Users/TAOLIVE/OneDrive - Daimler Truck/Desktop/aprendiz/Bot Shield/Bot Shield/logs/bull_specialist_sac_optuna.db"

def analyze_db():
    if not os.path.exists(db_path):
        print(f"File not found: {db_path}")
        return
        
    try:
        storage = f"sqlite:///{db_path}"
        studies = optuna.get_all_study_summaries(storage=storage)
        print(f"Found {len(studies)} studies in DB.")
        
        for s in studies:
            print("-" * 30)
            print(f"Study: {s.study_name}")
            study = optuna.load_study(study_name=s.study_name, storage=storage)
            
            states = {}
            for t in study.trials:
                state_name = str(t.state).split('.')[-1]
                states[state_name] = states.get(state_name, 0) + 1
            
            print(f"Total Trials: {len(study.trials)}")
            print(f"States: {states}")
            
            if 'COMPLETE' in states:
                print(f"Best Value: {study.best_trial.value}")
                # print(f"Best Params: {study.best_params}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_db()
