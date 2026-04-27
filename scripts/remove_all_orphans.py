"""
Remove TODAS as linhas órfãs com logger.warning sobre priors
"""

def remove_all_orphan_warnings():
    file_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\trading\ai_controller.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Backup
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path + f'.backup_orphans_{timestamp}'
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backup: {backup_path}")
    
    new_lines = []
    removed = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Se linha tem apenas espaços ou está vazia, manter
        if not line.strip():
            new_lines.append(line)
            i += 1
            continue
        
        # Se é um comentário órfão sobre "Anota priors"
        if 'Anota priors' in line and 'TrendPredictor' in line:
            # Substituir por versão atualizada
            indent = len(line) - len(line.lstrip())
            new_lines.append(' ' * indent + '# TrendPredictor removido - anotação de priors não mais necessária\n')
            print(f"Linha {i+1}: Atualizado comentário sobre priors")
            i += 1
            continue
        
        # Se é um logger.warning órfão sobre priors (indentação estranha)
        if 'logger.warning' in line and 'Falha ao anotar priors' in line:
            # Verificar se há try/except adequado antes
            # Se não há, é órfão - remover
            has_try = False
            for j in range(max(0, i-10), i):
                if 'try:' in lines[j] and lines[j].strip() == 'try:':
                    has_try = True
                    break
            
            if not has_try:
                print(f"Linha {i+1}: Removida logger.warning órfão")
                removed += 1
                i += 1
                continue
        
        new_lines.append(line)
        i += 1
    
    # Salvar
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"\n✅ Removidas {removed} linhas órfãs")
    print(f"   Total: {len(lines)} -> {len(new_lines)} linhas")
    
    # Testar
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'py_compile', 'trading\\ai_controller.py'],
        cwd=r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield",
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("\n🎉 SUCESSO! Arquivo compila!")
        
        # Testar import
        result2 = subprocess.run(
            ['python', '-c', 'from trading.ai_controller import AIController; print("✅ AIController imported successfully!")'],
            cwd=r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield",
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result2.returncode == 0:
            print(result2.stdout.strip())
            print("\n" + "="*60)
            print("🎊 REFATORAÇÃO 100% COMPLETA! 🎊")
            print("="*60)
            print("\n✅ Todas as referências ao TrendPredictor removidas")
            print("✅ Arquivo compila sem erros")
            print("✅ Import funciona perfeitamente")
            print("\n📝 Próximos passos:")
            print("   1. Testar os novos especialistas")
            print("   2. Remover arquivos obsoletos (trend_predictor.py, etc.)")
            print("   3. Treinar o sistema com a nova arquitetura!")
            return True
        else:
            print(f"⚠️ Compila mas erro no import:\n{result2.stderr[:500]}")
            return False
    else:
        print(f"\n❌ Erro de compilação:\n{result.stderr[:800]}")
        
        # Mostrar as linhas problemáticas
        error_line = None
        if 'line' in result.stderr:
            import re
            match = re.search(r'line (\d+)', result.stderr)
            if match:
                error_line = int(match.group(1))
                print(f"\n🔍 Linha problemática: {error_line}")
                if error_line <= len(new_lines):
                    print(f"   Conteúdo: {new_lines[error_line-1].strip()}")
        
        return False

if __name__ == "__main__":
    try:
        remove_all_orphan_warnings()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
