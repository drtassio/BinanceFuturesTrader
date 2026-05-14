# System Prompt — Padrão de Commits

Ao criar commits neste repositório, siga ESTRITAMENTE o padrão **Conventional Commits**:

## Regras
- SEMPRE use o formato: `<tipo>(<escopo>): <descrição curta>`
- NUNCA use emojis nas mensagens de commit
- SEMPRE escreva a descrição em **português**
- Use `git commit` sem flag `-m` para abrir o editor e visualizar o template `.gitmessage`
- Se outro agente de IA já realizou um commit no mesmo escopo, use `git commit --amend` para complementar em vez de criar um novo commit

## Tipos válidos
| Tipo     | Uso                                   |
|----------|---------------------------------------|
| feat     | Nova funcionalidade                   |
| fix      | Correção de bug                       |
| docs     | Documentação                          |
| style    | Formatação (sem impacto no código)    |
| refactor | Refatoração (nem bug, nem feat)       |
| test     | Adição/correção de testes             |
| chore    | Manutenção (deps, builds, etc.)       |
| perf     | Melhoria de performance               |
| ci       | Integração contínua                   |
| build    | Sistema de build ou dependências      |

## Exemplos
```
feat(parser): adiciona suporte a logs do Antigravity
fix(auth): corrige falha de validação no token JWT
refactor(db): extrai lógica de conexão para módulo separado
perf(api): reduz latência com cache em memória
```

## Importante
O hook `commit-msg` está ativo neste repositório e **REJEITARÁ** commits que não seguirem este padrão.

---

# Workflow científico autônomo

Este repositório opera sob auditoria científica contínua. Pipeline end-to-end:
**audit → training → validation_oos → paper_trading → production_readiness → live_ramp**

## 🤝 Handover para Outras IAs (DeepSeek/OpenCode)
Se você for uma IA assumindo este projeto durante um reset de tokens:
1. Leia **DEEPSEEK_MASTER_INSTRUCTIONS.md** no root.
2. Siga as personas em `.claude/agents/`.
3. Foque em resolver os findings em `.claude/state/audit-findings.jsonl`.


## Comandos custom (no Claude Code CLI)

| Comando | Função |
|---------|--------|
| `/goal show` | Mostra meta + KPIs alvo + status dos gates |
| `/goal gates` | Lista apenas bloqueadores ativos |
| `/goal measure` | Atualiza gates baseado em findings + métricas |
| `/agentes list` | Lista os 10 agentes científicos |
| `/agentes run <nome\|all>` | Roda 1 ou todos auditores em paralelo |
| `/agentes chain` | Loop audit → fix → re-audit até zero CRITICAL |
| `/agentes report` | Gera `.claude/state/audit-report.md` |
| `/pipeline start` | Inicia/retoma loop autônomo |
| `/pipeline pause` | Pausa loop |
| `/pipeline status` | Estado completo |

## Agentes científicos (`.claude/agents/`)

1. `leakage-auditor` — vazamento temporal, CPCV, embargo, scaler global
2. `regime-auditor` — HMM causal, drift, structural breaks
3. `reward-auditor` — 13 components, shaping vs PnL, hacking
4. `replay-buffer-auditor` — replace=True, imbalance, stale
5. `sac-stability-auditor` — Q-overest, alpha, critic collapse
6. `moe-gating-auditor` — expert collapse, load balance
7. `feature-eng-auditor` — FracDiff, Triple Barrier, Meta-Label
8. `execution-realism-auditor` — slippage, funding, latency
9. `validation-auditor` — walk-forward, DSR, OOS integrity
10. `production-readiness` — síntese final, recomenda go/no-go

## Estado persistente (`.claude/state/`)

- `goal.json` — meta, KPIs, gates, estágios, telegram config
- `audit-findings.jsonl` — 1 finding por linha (agent, severity, file, issue, fix)
- `run-log.md` — log human-readable de transições e decisões
- `audit-report.md` — relatório consolidado (gerado por `/agentes report`)

## Pontos onde humano decide (não autônomo)

1. `/goal set` na configuração inicial
2. Findings CRITICAL em `scientific_corrections.py`, `trend_specialist.py` (reward/env), `execution_engine.py`, `risk_layer.py`
3. Transição paper → production_readiness (validação final)
4. Transição production_readiness → live (autorização de capital real)
5. Override de gate ou abort de pipeline

Em todos esses pontos, o sistema notifica via Telegram e aguarda resposta.

## Integração Telegram

`telegram_bridge.py` lê `goal.json` e responde comandos do usuário. Notifica
automaticamente em: transições de fase, findings CRITICAL, KPI diário durante
paper, drift detection, kill-switch ativações.

Chat autorizado: definido em `.env` (`TELEGRAM_CHAT_ID`).
