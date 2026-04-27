"""
Script para remover o método _enrich_df_with_trend_predictions que ainda usa TrendPredictor
"""

import re

def fix_enrich_method():
    file_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\trading\ai_controller.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Encontrar o método _enrich_df_with_trend_predictions
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if 'def _enrich_df_with_trend_predictions' in line:
            start_idx = i
        elif start_idx is not None and line.strip() and not line.startswith(' ') and (line.startswith('def ') or line.startswith('async def ') or line.startswith('class ')):
            end_idx = i
            break
    
    if start_idx is None:
        print("Método não encontrado")
        return
    
    if end_idx is None:
        # Se não encontrou o próximo método, procurar até encontrar algo não indentado
        for i in range(start_idx + 1, len(lines)):
            if lines[i].strip() and not lines[i].startswith(' '):
                end_idx = i
                break
    
    # Backup
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path + f'.backup2_{timestamp}'
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backup criado: {backup_path}")
    
    # Novo método simplificado
    new_method = '''    def _enrich_df_with_trend_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona colunas de tendência neutras.
        Nota: TrendPredictor removido - os especialistas usam ScientificRegimeDetector.
        """
        df_enriched = df.copy()
        
        # Adiciona colunas neutras (para compatibilidade com código legado)
        neutral_cols = {
            'trend_pred_neutral': 1.0,
            'trend_pred_uptrend': 0.0,
            'trend_pred_downtrend': 0.0,
            'trend_pred_confidence': 0.5,
            'trend_success_prob': 0.5,
            'predicted_volatility_high': 0.0,
            'predicted_breakout': 0.0,
        }
        for col, val in neutral_cols.items():
            if col not in df_enriched.columns:
                df_enriched[col] = val
        
        logger.debug("Colunas de tendência neutras adicionadas (TrendPredictor removido)")
        return df_enriched

'''
    
    # Substituir
    new_lines = lines[:start_idx] + [new_method] + lines[end_idx:]
    
    # Salvar
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ Método _enrich_df_with_trend_predictions simplificado!")
    print(f"   Removidas {end_idx - start_idx} linhas, adicionadas {len(new_method.split(chr(10)))}")

if __name__ == "__main__":
    try:
        fix_enrich_method()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
