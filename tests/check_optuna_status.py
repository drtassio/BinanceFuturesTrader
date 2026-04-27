import optuna
import os

dbs = {
    'BullSpecialist': 'logs/bull_specialist_sac_optuna.db',
    'BearSpecialist': 'logs/bear_specialist_sac_optuna.db', 
    'RangerSpecialist': 'logs/ranger_specialist_sac_optuna.db'
}

print("=" * 60)
print("STATUS ATUAL DOS ESTUDOS OPTUNA")
print("=" * 60)

for name, path in dbs.items():
    exists = os.path.exists(path)
    if not exists:
        print(f"\n{name}: NÃO EXISTE")
        continue
        
    size = os.path.getsize(path)
    print(f"\n{name}:")
    print(f"  Arquivo: {path} ({size/1024:.1f} KB)")
    
    try:
        studies = optuna.study.get_all_study_summaries(f'sqlite:///{path}')
        for s in studies:
            complete = len([t for t in optuna.load_study(study_name=s.study_name, storage=f'sqlite:///{path}').trials if t.state == optuna.trial.TrialState.COMPLETE])
            pruned = len([t for t in optuna.load_study(study_name=s.study_name, storage=f'sqlite:///{path}').trials if t.state == optuna.trial.TrialState.PRUNED])
            print(f"  Study: {s.study_name}")
            print(f"    Total: {s.n_trials} trials")
            print(f"    Complete: {complete}, Pruned: {pruned}")
            print(f"    Status: {'✅ COMPLETO' if s.n_trials >= 100 else f'🔄 FALTA {100 - s.n_trials} trials'}")
    except Exception as e:
        print(f"  Erro: {e}")

print("\n" + "=" * 60)
print("ORDEM DE PROCESSAMENTO: Bull → Bear → Ranger")
print("=" * 60)
