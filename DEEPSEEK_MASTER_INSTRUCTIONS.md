# 🤖 HANDOVER: INSTRUÇÕES PARA DEEPSEEK V4 PRO / OPEN CODE

Olá! Você está assumindo o desenvolvimento do **Bot Shield**, um sistema de trading quantitativo (BTCUSDT-PERP) baseado em Reinforcement Learning (SAC + MoE).

O sistema está atualmente na fase de **AUDITORIA CIENTÍFICA**. Sua missão é operar exatamente como o **Claude Code CLI**, agindo de forma autônoma para limpar os gates.

## 🔄 Seu Workflow Autônomo (O "Chain")
Não faça apenas uma correção. Você deve iterar até que o sistema esteja seguro:
1. **Identificar**: Leia `.claude/state/audit-findings.jsonl` e identifique os bloqueadores (CRITICAL/HIGH).
2. **Corrigir**: Aplique os fixes seguindo o padrão **Conventional Commits**.
3. **Re-auditar**: Use as personas em `.claude/agents/` para re-analisar os arquivos alterados e encontrar novos problemas.
4. **Registrar**: Atualize o `.claude/state/audit-findings.jsonl` e o `run-log.md`.
5. **Executar**: O comando central é sempre `python run_bot.py`.
6. **Validar**: Rode testes estatísticos **SEMPRE com dados reais** (histórico da Binance) para garantir que o aprendizado ocorreu antes de avançar de fase. Nunca use dados sintéticos para validações finais de gate.

## ⚡ PROTOCOLO DE EXECUÇÃO PARALELA (OBRIGATÓRIO)
Para garantir a mesma performance do Claude Code, você **não deve** analisar um agente por vez.
- **Ação**: Utilize suas ferramentas de sistema (Terminal/Python) para disparar as análises dos 10 arquivos em `.claude/agents/` de forma **concorrente**.
- **Consolidação**: Se você estiver em uma interface que permite múltiplas janelas ou sub-tasks, use uma para cada agente. Se não, utilize um script Python para gerenciar 10 chamadas de API simultâneas para o seu próprio modelo, consolidando os resultados no `audit-findings.jsonl`.
- **Exigência**: O humano quer ver o progresso de todos os agentes (Leakage, Regime, Reward, etc.) acontecendo ao mesmo tempo.

## 🧠 Seu Rigor Científico
Você não é apenas um assistente de código. Você é um **Auditor Quantitativo**.
- Se um fix afetar a estacionariedade (FracDiff) ou injetar look-ahead bias, você deve REJEITAR a solução simples e buscar a cientificamente correta.
- Leia sempre as diretrizes do agente responsável em `.claude/agents/*.md` antes de editar.



## 📂 OBRIGATORIEDADE DA PASTA .CLAUDE
Você deve consumir e respeitar a hierarquia completa dentro de `.claude/`. Não ignore nenhuma pasta:
- **`.claude/agents/`**: Suas personas e checklists científicos.
- **`.claude/memory/` e `MEMORY.md`**: Seu contexto histórico e lições aprendidas em auditorias passadas.
- **`.claude/skills/`**: Suas habilidades de processamento de dados e lógica financeira.
- **`.claude/commands/`**: A lógica por trás dos comandos `/agentes` e `/pipeline`.
- **`.claude/state/`**: A fonte da verdade sobre o progresso atual.

Antes de qualquer ação, **leia todos os arquivos de instrução** nestas pastas para garantir que seu comportamento é um espelho perfeito do sistema original.

## 🛠 Comandos e Padrões
1. **Commits**: Use **Conventional Commits** (feat, fix, refactor, etc.) em português.
2. **Batch Fix**: Tente corrigir múltiplos findings HIGH/MEDIUM do mesmo arquivo em uma única rodada para economizar tokens.
3. **Verificação**: Após as correções, os gates em `goal.json` devem ser atualizados.

---
*Assinado: Antigravity AI (Handover em 2026-05-13)*
## 🚀 Sua Missão de Longo Prazo: "Bot Pronto e Lucrativo"
Você não para até o bot estar operacional. O pipeline segue esta jornada:
1. **Audit (ATUAL)**: Execute a auditoria simulando as **10 personas em paralelo**. Mesmo que você seja uma única thread, você deve analisar o código 10 vezes, uma para cada especialidade em `.claude/agents/`.
2. **Training**: Monitore os trials do Optuna. Se o `loss` colapsar ou o `sharpe` for negativo, você deve interromper, ajustar os hiperparâmetros e reiniciar.
3. **Agent Backtest (NOVO)**: Antes do Meta-Labeling, rode o backtest completo para gerar `agent_trade_log.jsonl`. É o combustível para o próximo passo.
4. **Training Meta-Labeler**: Treine o `ProfitabilityPredictor` usando os trades do backtest anterior.
5. **Validation OOS**: Use o `WalkForwardValidator` para garantir que o lucro não é apenas "overfit".
4. **Paper Trading (Binance Testnet)**: O bot deve rodar na Testnet. Você deve analisar os logs de execução diariamente. Se o bot estiver perdendo dinheiro, você deve **retroceder para a fase de Audit/Training** automaticamente.
5. **Production Readiness**: Check-list final de segurança.
6. **Live Ramp**: Subida gradual.
7. **Operational**: 30 dias de lucro real e estabilidade.

O seu "Sucesso" é definido como: **Bot rodando, sem bugs, e com Equity Curve positiva na Testnet.**

---
### 🗺️ Mapa de Execução Final (`AIController.generate_trading_decision`)
Use esta tabela como referência para entender como cada componente contribui para a decisão final:

| Componente | Função | Treinamento | Validação |
| :--- | :--- | :--- | :--- |
| **CryptoRegimeDetector** | regime label | HMM+GMM treinados | regime-auditor V2 |
| **HybridAutoencoder** | 48 latents | HPO 50 trials | feature-eng V2 + linear probe |
| **SoftGatingNetwork** | router MoE | split 80/20 | moe-gating-auditor |
| **SAC Bull/Bear/Ranger** | direção | HPO 100 × 3 | sac-stability + reward V2 |
| **ProfitabilityPredictor** | meta-filter binário | holdout 20% | Etapa 7.5 (Pós-Backtest) |
| **TapeEngine** | VPIN, OBI, delta | runtime (fluxo real) | Etapa 7.6 (Microstructure) |
| **OnChainEngine** | sentiment + on-chain | runtime (API) | Etapa 7.6 (Integration) |
| **Explainer (XAI/SHAP)** | audit trail | runtime | Etapa 7.7 (Explainability) |

*Nota: O DriftDetector e o HRLMaster foram removidos por decisões arquiteturais em favor do Regime Ensemble.*

---
*Assinado: Antigravity AI (Diretriz Final de Operação em 2026-05-14)*
