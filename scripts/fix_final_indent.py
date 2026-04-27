"""
Correção final do erro de indentação na linha 2194
"""

def fix_final_indentation():
    file_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\trading\ai_controller.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total de linhas: {len(lines)}")
    
    # Backup
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path + f'.backup_final_{timestamp}'
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backup: {backup_path}")
    
    # Corrigir linha 2194 (índice 2193)
    if len(lines) > 2193:
        # Remover as linhas problemáticas 2192-2194
        new_lines = []
        skip_until = -1
        
        for i, line in enumerate(lines):
            # Linha 2191 (índice 2190): df_with_trends = ...
            # Linha 2192 (índice 2191): # Anota priors...
            # Linha 2193 (índice 2192): linha vazia
            # Linha 2194 (índice 2193): logger.warning com indentação errada
            # Linha 2195 (índice 2194): df_fully_enriched = df_with_trends
            
            # Se estamos na linha 2191 (comentário órfão)
            if i == 2191 and 'Anota priors do TrendPredictor' in line:
                # Substituir por comentário atualizado
                indent = '            '
                new_lines.append(f"{indent}# TrendPredictor removido - anotação de priors não mais necessária\n")
                skip_until = 2194  # Pular até df_fully_enriched
                continue
            
            # Pular linhas 2192-2193 (vazia e logger.warning)
            if skip_until >= 0 and i < skip_until:
                continue
            
            if i == skip_until:
                skip_until = -1  # Resetar
            
            new_lines.append(line)
        
        # Salvar
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"✅ Corrigidas {len(lines) - len(new_lines)} linhas problemáticas")
        print(f"   Linhas: {len(lines)} -> {len(new_lines)}")
    
    # Testar compilação
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'py_compile', 'trading\\ai_controller.py'],
        cwd=r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield",
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("\n🎉 SUCESSO! Arquivo compila sem erros!")
        
        # Testar importação
        result2 = subprocess.run(
            ['python', '-c', 'from trading.ai_controller import AIController; print("✅ Import OK")'],
            cwd=r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield",
            capture_output=True,
            text=True
        )
        
        if result2.returncode == 0:
            print(result2.stdout)
            print("\n✅ REFATORAÇÃO 100% COMPLETA!")
        else:
            print(f"⚠️ Erro no import:\n{result2.stderr[:500]}")
        
        return True
    else:
        print(f"\n❌ Erro de compilação:\n{result.stderr[:800]}")
        return False

if __name__ == "__main__":
    try:
        fix_final_indentation()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
