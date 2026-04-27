"""
Script final para limpar TODAS as referências ao trend_predictor
"""

import re

def final_cleanup():
    file_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\trading\ai_controller.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path + f'.backup3_{timestamp}'
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backup criado: {backup_path}")
    
    # 1. Remover TODO o bloco duplicado de _enrich_df_with_trend_predictions (linhas ~1670-1750)
    # Usa um padrão mais agressivo para pegar o método duplicado completo
    pattern = r'(\s*# No arquivo trading/ai_controller\.py\s*\n\s*# Substitua a função _enrich_df_with_trend_predictions.*?def _enrich_df_with_trend_predictions\(self.*?return df_enriched\s*\n)'
    
    matches = list(re.finditer(pattern, content, flags=re.DOTALL))
    print(f"Encontrados {len(matches)} métodos _enrich_df_with_trend_predictions para remover")
    
    if len(matches) > 1:
        # Remove apenas o duplicado (primeiro ou último)
        # Manter apenas o primeiro
        for match in matches[1:]:
            start, end = match.span()
            print(f"Removendo duplicata nas linhas ~{content[:start].count(chr(10)) + 1}")
            content = content[:start] + content[end:]
    
    # 2. Salvar
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Limpeza final concluída!")
    
    # 3. Verificar quantas referências sobraram
    refs = content.count('self.trend_predictor')
    print(f"📊 Referências restantes a 'self.trend_predictor': {refs}")
    
    if refs > 0:
        # Listar as linhas
        lines = content.split('\n')
        print("\n⚠️  Referências restantes:")
        for i, line in enumerate(lines, 1):
            if 'self.trend_predictor' in line and not line.strip().startswith('#'):
                print(f"  Linha {i}: {line.strip()[:80]}")

if __name__ == "__main__":
    try:
        final_cleanup()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
