"""
Script para limpar blocos órfãos deixados pela remoção do trend_predictor
"""

def fix_orphan_blocks():
    file_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\trading\ai_controller.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Backup
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path + f'.backup5_{timestamp}'
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backup criado: {backup_path}")
    
    new_lines = []
    i = 0
    fixed_count = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Se encontrar um try: seguido de except: (bloco vazio)
        if line.strip() == 'try:':
            # Verificar se a próxima linha não vazia é um except
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            
            if j < len(lines) and lines[j].strip().startswith('except'):
                # Bloco try vazio! Remover try e except
                print(f"Linha {i+1}: Removendo try/except órfão")
                i = j + 1  # Pular try e except
                # Também pular o conteúdo do except se necessário
                indent = len(lines[j]) - len(lines[j].lstrip())
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip():
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= indent:
                            break
                    i += 1
                fixed_count += 1
                continue
        
        # Se encontrar if output is None sem ter output definido
        if 'if output is None:' in line:
            # Verificar se output foi definido antes
            has_output = any('output =' in lines[max(0, i-10):i])
            if not has_output:
                print(f"Linha {i+1}: Removendo bloco órfão com 'output'")
                # Pular este bloco if
                indent = len(line) - len(line.lstrip())
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip():
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= indent:
                            break
                    i += 1
                fixed_count += 1
                continue
        
        # Se encontrar prior_dir sem output
        if 'prior_dir = getattr(output' in line:
            # Verificar se output existe
            has_output = any('output =' in lines[max(0, i-20):i])
            if not has_output:
                print(f"Linha {i+1}: Removendo bloco órfão com 'prior_dir'")
                # Pular toda a seção até return ou próximo def
                while i < len(lines):
                    if 'return' in lines[i] or lines[i].strip().startswith('def '):
                        if 'return' in lines[i]:
                            i += 1
                        break
                    i += 1
                fixed_count += 1
                continue
        
        new_lines.append(line)
        i += 1
    
    # Salvar
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ Corrigidos {fixed_count} blocos órfãos")
    return fixed_count

if __name__ == "__main__":
    try:
        count = fix_orphan_blocks()
        if count > 0:
            print("\n🔧 Testando compilação...")
            import subprocess
            result = subprocess.run(
                ['python', '-m', 'py_compile', 'trading\\ai_controller.py'],
                cwd=r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield",
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ Arquivo compila sem erros!")
            else:
                print(f"❌ Erro de compilação:\n{result.stderr}")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
