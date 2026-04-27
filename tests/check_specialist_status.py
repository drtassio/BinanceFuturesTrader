"""
Script para verificar status completo de cada specialist (estudo Optuna + modelo treinado)
"""
import optuna
import os

specialists = {
    'BullSpecialist': {
        'db': 'logs/bull_specialist_sac_optuna.db',
        'model': 'models/Bull_Specialist_sac.zip',
        'tensorboard': 'logs/tensorboard/Bull_Specialist'
    },
    'BearSpecialist': {
        'db': 'logs/bear_specialist_sac_optuna.db',
        'model': 'models/Bear_Specialist_sac.zip',
        'tensorboard': 'logs/tensorboard/Bear_Specialist'
    },
    'RangerSpecialist': {
        'db': 'logs/ranger_specialist_sac_optuna.db',
        'model': 'models/Ranger_Specialist_sac.zip',
        'tensorboard': 'logs/tensorboard/Ranger_Specialist'
    }
}

print("=" * 70)
print("STATUS COMPLETO DOS SPECIALISTS")
print("=" * 70)

for name, paths in specialists.items():
    print(f"\n📊 {name}:")
    
    # Verificar Estudo Optuna
    if os.path.exists(paths['db']):
        try:
            studies = optuna.study.get_all_study_summaries(f"sqlite:///{paths['db']}")
            if studies:
                s = studies[0]
                study = optuna.load_study(study_name=s.study_name, storage=f"sqlite:///{paths['db']}")
                complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                total = s.n_trials
                status = "✅ COMPLETO" if total >= 100 else f"🔄 {total}/100 trials"
                print(f"   📁 Estudo: {status} (Complete: {complete}, Pruned: {pruned})")
                if study.best_trial and study.best_trial.state == optuna.trial.TrialState.COMPLETE:
                    print(f"   🏆 Melhor Trial: #{study.best_trial.number} (value={study.best_trial.value:.4f})")
            else:
                print(f"   📁 Estudo: ⚠️ Sem estudos no DB")
        except Exception as e:
            print(f"   📁 Estudo: ❌ Erro - {e}")
    else:
        print(f"   📁 Estudo: ❌ Não existe")
    
    # Verificar Modelo Treinado
    if os.path.exists(paths['model']):
        size_mb = os.path.getsize(paths['model']) / 1e6
        print(f"   🤖 Modelo: ✅ Existe ({size_mb:.1f} MB)")
    else:
        print(f"   🤖 Modelo: ❌ Não treinado")
    
    # Verificar TensorBoard logs
    if os.path.exists(paths['tensorboard']):
        files = os.listdir(paths['tensorboard'])
        print(f"   📈 TensorBoard: ✅ {len(files)} runs")
    else:
        print(f"   📈 TensorBoard: ❌ Sem logs")

print("\n" + "=" * 70)
print("PRÓXIMO PASSO: Specialist com mais progresso será processado primeiro")
print("=" * 70)
