"""
Remove método duplicado _enrich_df_with_trend_predictions (linhas 1673-1743)
"""

def remove_duplicate_method():
    file_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\trading\ai_controller.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Backup
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path + f'.backup6_{timestamp}'
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backup criado: {backup_path}")
    
    # Remover linhas 1669-1744 (índices 1668-1743)
    # Procurar pelo começo do método duplicado
    start = None
    end = None
    
    for i, line in enumerate(lines):
        if i >= 1668 and '# No arquivo trading/ai_controller.py' in line:
            start = i
        if start and i > start and 'def report_regime_validation' in line:
            end = i
            break
    
    if start and end:
        print(f"Removendo linhas {start+1} a {end} ({end-start} linhas)")
        new_lines = lines[:start] + lines[end:]
    else:
        print("Método duplicado não encontrado, mantendo arquivo original")
        new_lines = lines
    
    # Salvar
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ Método duplicado removido!")
    
    # Testar compilação
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'py_compile', 'trading\\ai_controller.py'],
        cwd=r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield",
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Arquivo compila sem erros!")
        return True
    else:
        print(f"❌ Erro de compilação:\n{result.stderr[:500]}")
        return False

if __name__ == "__main__":
    try:
        remove_duplicate_method()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
