"""
Script para remover as últimas 2 referências ao trend_predictor
"""

def remove_last_references():
    file_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\trading\ai_controller.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Backup
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path + f'.backup4_{timestamp}'
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backup criado: {backup_path}")
    
    # Procurar e remover linhas com self.trend_predictor (não comentadas)
    new_lines = []
    removed_count = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Pular se for comentário (já removido)
        if line.strip().startswith('#') and 'trend_predictor' in line:
            new_lines.append(line)
            i += 1
            continue
        
        # Se encontrar self.trend_predictor em código ativo
        if 'self.trend_predictor' in line:
            print(f"Linha {i+1}: {line.strip()[:80]}")
            
            # Se for uma linha if com trend_predictor, remover todo o bloco
            if 'if not self.trend_predictor' in line:
                # Pular toda a estrutura if/else/try
                indent_level = len(line) - len(line.lstrip())
                i += 1
                skipped = 1
                while i < len(lines):
                    next_line = lines[i]
                    next_indent = len(next_line) - len(next_line.lstrip())
                    
                    # Se a indentação voltar ao nível original ou menor, paramos
                    if next_line.strip() and next_indent <= indent_level:
                        break
                    i += 1
                    skipped += 1
                print(f"  -> Removido bloco de {skipped} linhas")
                removed_count += skipped
                continue
            
            # Se for uma linha simples, apenas pular
            removed_count += 1
            i += 1
            continue
        
        new_lines.append(line)
        i += 1
    
    # Salvar
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ Removidas {removed_count} linhas")
    
    # Verificar
    final_count = ''.join(new_lines).count('self.trend_predictor')
    print(f"📊 Referências restantes: {final_count}")
    
    if final_count == 0:
        print("🎉 SUCESSO! Todas as referências ao self.trend_predictor foram removidas!")
    else:
        print("⚠️  Ainda há referências. Verifique manualmente.")

if __name__ == "__main__":
    try:
        remove_last_references()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
