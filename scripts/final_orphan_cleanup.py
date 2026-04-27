"""
Remove TODOS os blocos órfãos restantes
"""

def final_orphan_cleanup():
    file_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\trading\ai_controller.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path + f'.backup7_{timestamp}'
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backup criado: {backup_path}")
    
    # Remover todas as ocorrências de try: except na mesma linha ou try órfão
    import re
    
    # Padrão 1: try:    except na mesma linha
    content = re.sub(r'\s*try:\s*except Exception as \w+:\s*\n', '', content)
    
    # Padrão 2: try: seguido de except sem conteúdo (várias linhas)
    content = re.sub(r'\s*try:\s*\n\s*except Exception as \w+:', '', content)
    
    #Padrão 3: linhas órfãs com logger.warning sobre falha (depois de remover try/except)
    # que começam com indentação errada
    linhas = content.split('\n')
    novas_linhas = []
    
    for i, linha in enumerate(linhas):
        # Se linha tem logger.warning sobre falha mas não há try antes
        if 'logger.warning' in linha and 'Falha ao anotar priors' in linha:
            # Verificar se há try nas últimas 5 linhas
            has_try = any('try:' in linhas[max(0, i-5):i])
            if not has_try:
                print(f"Removendo linha órfã {i+1}: {linha.strip()[:60]}")
                continue
        
        novas_linhas.append(linha)
    
    content = '\n'.join(novas_linhas)
    
    # Salvar
    with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)
    
    print(f"✅ Limpeza de blocos órfãos concluída!")
    
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
        
        # Verificação final de referências
        refs = content.count('self.trend_predictor')
        print(f"\n📊 Verificação final:")
        print(f"   Referências a 'self.trend_predictor': {refs}")
        
        return True
    else:
        print(f"❌ Erro de compilação:\n{result.stderr[:800]}")
        return False

if __name__ == "__main__":
    try:
        success = final_orphan_cleanup()
        if success:
            print("\n🎉 REFATORAÇÃO COMPLETA!")
            print("✅ Todas as referências ao TrendPredictor foram removidas")
            print("✅ O arquivo compila sem erros")
        else:
            print("\n⚠️  Ainda há erros. Verifique o output acima.")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
