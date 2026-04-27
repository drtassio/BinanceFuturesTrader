import os

file_path = r'c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\trading\ai_controller.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the malformed line (should be around line 2191)
target_idx = None
for i, line in enumerate(lines):
    if '# --- Etapa 2: Predicao de Rentabilidade ---' in line and '\\n' in line:
        target_idx = i
        break

if target_idx is None:
    print("Malformed line not found, checking for predict_profitability")
    for i, line in enumerate(lines):
        if 'self.profitability_predictor.predict_profitability(' in line:
            # Already fixed or different pattern
            print(f"Found predict_profitability at line {i+1}: {line.strip()[:80]}")
            exit(0)
    exit(1)

print(f"Found malformed line at index {target_idx}")

# Replace the malformed line with proper multi-line code
new_lines = lines[:target_idx]
new_lines.append('                    # --- Etapa 2: Predicao de Rentabilidade ---\n')
new_lines.append('                    # [FIX] ProfitabilityPredictor foi REMOVIDO do pipeline.\n')
new_lines.append('                    # Usamos score neutro (0.5) como padrao quando nao disponivel.\n')
new_lines.append('                    if self.profitability_predictor is not None:\n')
new_lines.append('                        # O ProfitabilityPredictor espera sequencias\n')
new_lines.append('                        feature_sequence = row.values.reshape(1, 1, -1)  # (batch, seq_len, features)\n')
new_lines.append('                        context_features = np.array([0.0, 0.0, 0.0, 0.0])  # Contexto padrao\n')
new_lines.append('                        prob, _, _, _ = self.profitability_predictor.predict_profitability(\n')
new_lines.append('                            feature_sequence, context_features\n')
new_lines.append('                        )\n')
new_lines.append('                    else:\n')
new_lines.append('                        # Sem predictor, usa probabilidade neutra\n')
new_lines.append('                        prob = 0.5\n')
new_lines.append('\n')
# Skip the malformed line
new_lines.extend(lines[target_idx + 1:])

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"SUCCESS: Fixed malformed line. Lines: {len(lines)} -> {len(new_lines)}")
