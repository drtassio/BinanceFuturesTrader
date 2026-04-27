"""
Script para completar a refatoração do ai_controller.py
Remove todas as referências ao TrendPredictor
"""

import re

def refactor_ai_controller():
    file_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\trading\ai_controller.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Remover linha 127: self.trend_predictor = TrendPredictor(...)
    content = re.sub(
        r'\s*self\.trend_predictor = TrendPredictor\(self\.config_ai, self\.config_trading\)\s*\n',
        '',
        content
    )
    
    # 2. Remover comentário sobre TrendPredictor em _initialize_feature_models
    content = re.sub(
        r'\s*# Passa self\.config_ai e self\.config_trading para o TrendPredictor\s*\n',
        '',
        content
    )
    
    # 3. Atualizar docstring de _initialize_feature_models
    content = re.sub(
        r'(\s*Inicializa os modelos de features, passando ambas as configurações\s*\n\s*necessárias para o TrendPredictor\.\s*)',
        '    Inicializa os modelos de features.\n    Nota: TrendPredictor foi removido - detecção de regime via ScientificRegimeDetector.\n    ',
        content
    )
    
    # 4. Remover 'trend_predictor' da lista expected_models (linha ~469)
    content = re.sub(
        r"'autoencoder_model', 'trend_predictor', 'profitability_predictor',",
        "'autoencoder_model', 'profitability_predictor',",
        content
    )
    
    # 5. Remover verificação de trend_predictor em load_all_models (linha ~497)
    content = re.sub(
        r'\s*self\.trend_predictor and self\.trend_predictor\.is_trained and\s*\n',
        '',
        content
    )
    
    # 6. Remover linha do status dict em load_all_models (linha ~506)
    content = re.sub(
        r'\s*"TrendPredictor": self\.trend_predictor\.is_trained if self\.trend_predictor else False,\s*\n',
        '',
        content
    )
    
    # 7. Remover método _annotate_trend_features completo (linhas ~849-900)
    # Procura pelo método e remove até o próximo método
    content = re.sub(
        r'(\s*def _annotate_trend_features\(self.*?\n.*?)((?=\n\s*def\s+)|(?=\n\s*async def\s+))',
        '',
        content,
        flags=re.DOTALL
    )
    
    # 8. Remover método _optimize_trend_predictor completo (linha ~1315)
    content = re.sub(
        r'(\s*def _optimize_trend_predictor\(self.*?\n.*?)((?=\n\s*def\s+)|(?=\n\s*async def\s+))',
        '',
        content,
        flags=re.DOTALL
    )
    
    # 9. Remover bloco de treinamento do TrendPredictor em _full_training_phase1
    # Linhas 1378-1402
    content = re.sub(
        r'\s*labels_df = self\.trend_predictor\.set_labels_from_indicators\(df_with_hidden_features\).*?'
        r'self\._update_model_metadata\(\'trend_predictor\', True\)\s*\n',
        '            # TrendPredictor removido - detecção de regime via ScientificRegimeDetector\n            logger.info("TrendPredictor removido - usando detecção determinística de regime")\n',
        content,
        flags=re.DOTALL
    )
    
    # 10. Remover anotação de priors em build_enriched_dataset (linhas ~1479-1480, 1494-1495)
    content = re.sub(
        r'\s*if self\.trend_predictor and self\.trend_predictor\.is_trained:\s*\n'
        r'\s*df_with_trends = self\.trend_predictor\.annotate_priors\(df_with_trends\)\s*\n',
        '',
        content
    )
    
    # 11. Remover verificação de trend_predictor em execute_trade_decision (linha ~1631)
    content = re.sub(
        r'if not all\(\[self\.trend_predictor, self\.profitability_predictor\]\):',
        'if not self.profitability_predictor:',
        content
    )
    
    # 12. Remover predição de regime em execute_trade_decision (linha ~1654)
    content = re.sub(
        r'\s*market_regime_pred = self\.trend_predictor\.predict\(single_row_df\)\s*\n',
        '                    # market_regime_pred removido - usando ScientificRegimeDetector\n',
        content
    )
    
    # 13. Remover verificação de trend_predictor em trade() (linha ~1701)
    content = re.sub(
        r'if not self\.is_trained or not all\(\[self\.trend_predictor, self\.drift_detector',
        'if not self.is_trained or not all([self.drift_detector',
        content
    )
    
    # 14. Remover predição de tendência em trade() (linhas ~1770-1772)
    content = re.sub(
        r'\s*if self\.trend_predictor and hasattr\(self\.trend_predictor, \'predict\'\):\s*\n'
        r'\s*try:\s*\n'
        r'\s*trend_prediction = self\.trend_predictor\.predict\(pd\.DataFrame\(\[latest_row\]\)\).*?\n'
        r'\s*except.*?\n',
        '',
        content,
        flags=re.DOTALL
    )
    
    # 15. Remover adaptação do trend_predictor em adapt_online() (linhas ~1887-1888)
    content = re.sub(
        r'\s*if self\.trend_predictor and hasattr\(self\.trend_predictor, \'adapt\'\):\s*\n'
        r'\s*trend_adapt_success = await self\.trend_predictor\.adapt\(.*?\n',
        '',
        content,
        flags=re.DOTALL
    )
    
    # Salvar arquivo
    if content != original_content:
        # Backup
        import shutil
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = file_path + f'.backup_{timestamp}'
        shutil.copy2(file_path, backup_path)
        print(f"✅ Backup criado: {backup_path}")
        
        # Salvar refatoração
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Refatoração concluída!")
        print(f"📊 Linhas modificadas: {original_content.count(chr(10))} -> {content.count(chr(10))}")
        
        # Verificar se ainda há referências
        remaining = content.count('trend_predictor')
        if remaining > 0:
            print(f"⚠️  ATENÇÃO: Ainda há {remaining} referências a 'trend_predictor' (podem ser em comentários)")
        else:
            print("✅ Todas as referências ao trend_predictor foram removidas!")
    else:
        print("ℹ️  Nenhuma mudança necessária")

if __name__ == "__main__":
    try:
        refactor_ai_controller()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
