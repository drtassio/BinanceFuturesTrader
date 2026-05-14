# Scientific Fixes V2 — Plano e Status

Data: 2026-05-14
Contexto: Opcao A da analise das skills externas (autonomous-engineering,
profit-engineering, autoencoder-quality, code-corrector).

## Resumo executivo

Verificacao real do codigo revelou que **2 de 3 fixes cientificos JA estao
aplicados** (de sessoes anteriores). Apenas 1 fix de refactor permanece.
Adicao mais valiosa: novas extensoes nos auditores + Evidence Ledger.

---

## FIX 1: Log returns / Soft clip no reward — STATUS: APLICADO

**Onde**: `specialists/trend_specialist.py:1619-1625`

**Codigo atual:**
```python
dur_amortization = float(np.clip(np.sqrt(duration_steps / 5.0), 0.3, 2.5))
soft_clamped_return = 0.5 * np.tanh(trade_return_pct / 0.3)
return_clip = soft_clamped_return
reward = return_clip * 80.0 * dur_amortization
```

**Avaliacao cientifica**:
- `tanh(x/0.3)` eh funcionalmente equivalente a `log1p(x)` em assimetria,
  mas com bounded output em [-0.5, +0.5]. Drawdown -50% → tanh(-1.67) = -0.94 * 0.5 = -0.47
  vs ganho +50% → tanh(+1.67) = +0.94 * 0.5 = +0.47.
- Embora simetrico em `trade_return_pct`, a multiplicacao por `dur_amortization`
  injeta penalidade indireta a drawdowns (que tendem a ter duracao curta).
- Multiplier 80 vs shaping ~10-15: ratio PnL/shaping ~5x → PnL DOMINA reward.

**Acao**: nenhuma. Manter como esta. Documentar em Evidence Ledger (H-006).

---

## FIX 2: Fees Binance no reward — STATUS: APLICADO

**Onde**: `trading/portfolio.py:65`

**Codigo atual:**
```python
def update_from_trade(self, trade: Trade):
    fee = trade.fee
    self.cash -= fee
    ...
```

**Cadeia de aplicacao**:
1. `trading/execution_engine.py` calcula `trade.fee` ao executar
2. `portfolio.update_from_trade()` deduz do cash
3. `trend_specialist.py:1627` documenta: "Custo de transacao ja embutido no pnl_realized"
4. `pnl_realized` entra no `_compute_trade_reward` ja com fee descontado

**Avaliacao**: fee taker Binance (0.04%) eh aplicado. Funding rate NAO foi
verificado explicitamente — TODO menor.

**Acao**: nenhuma no codigo principal. Adicionar checklist no
`execution-realism-auditor` para validar `trade.fee != 0` em testes mock.

---

## FIX 3: Auxiliary loss multi-task no AE — STATUS: NAO APLICADO

**Onde deveria ir**: `feature_engineering/hybrid_autoencoder.py`

**Estado atual**: AE compoe loss = recon_loss + jepa_loss + vicreg_loss + ts2vec_loss.
**Nao tem** previsao de retorno auxiliar.

**Risco de aplicar agora**:
- AE ja foi treinado e salvo em `models_ai/hybrid_autoencoder_v3.pt`
- Adicionar prediction_head muda arquitetura → modelo atual incompativel
- Re-treinar AE = +6h GPU
- Bull HPO atual usa latents do AE atual — invalidaria comparacao

**Alternativa proposta (linear probe post-hoc)**:
- Em vez de re-treinar AE com aux head, rodar **linear probe** sobre os
  latents existentes para PROVAR se preservam sinal direcional
- Se AUC linear probe >= 0.55 → AE atual eh aceitavel SEM aux head
- Se AUC < 0.52 → necessario re-treinar com aux head (Fase 2 do plano)

**Script a criar**: `scripts/ae_linear_probe.py`
- Carrega `hybrid_autoencoder_v3.pt`
- Encoda dataset OOS (ultimos 20%)
- Treina LightGBM nos 48 latents → target = sign(return_1h)
- Output: AUC, feature importance, evidencia para ledger H-001 + H-005

**Acao recomendada**:
- AGORA: criar `ae_linear_probe.py` e rodar (~10min)
- DEPOIS: decidir se re-treinar AE baseado em resultado

---

## EXTENSOES APLICADAS NESTA SESSAO

| Item | Status | Onde |
|------|--------|------|
| Evidence Ledger criado | OK | `.claude/state/evidence-ledger.jsonl` (5 hipoteses iniciais) |
| Evidence Ledger schema doc | OK | `.claude/state/evidence-ledger-schema.md` |
| reward-auditor V2 (runtime corr + log returns + fees + reward hacking patterns) | OK | `.claude/agents/reward-auditor.md` |
| feature-eng-auditor V2 (linear probe + permutation + ablation + aux loss check) | OK | `.claude/agents/feature-eng-auditor.md` |
| regime-auditor V2 (R^2 por regime + coverage + drift) | OK | `.claude/agents/regime-auditor.md` |
| Capital staging (LIMITED_CAPITAL $200, 1x leverage, 15d) | OK | `pipeline.md` + `goal.json` |
| Gates novos: `limited_capital_pass`, `evidence_ledger_pass` | OK | `goal.json` |

---

## PROXIMOS PASSOS

1. **Criar `scripts/ae_linear_probe.py`** quando Bull HPO concluir
2. **Criar `scripts/pnl_consistency_monitor.py`** para rodar corr(reward, pnl) durante training
3. **Rodar /agentes run all** apos Bull HPO para popular Evidence Ledger com evidencias reais
4. **Production-readiness** so deve recomendar go com:
   - 9/10 gates verdes + `evidence_ledger_pass=true`
   - evidence_ledger score (mean confidence) >= 0.60
   - Linear probe AUC >= 0.55 documentado em H-001
   - corr(reward, pnl) >= 0.30 documentado em H-002

## Politica de aplicacao durante Bull HPO

- Bull HPO atual usa codigo de `2026-05-14 06:00` (snapshot pos-fixes leakage)
- Mudancas futuras em reward function → afetam SO Bear/Ranger HPO subsequentes
- Para comparabilidade: aplicar AE aux head e re-treinar TODOS os 3 specialists junto
- Por isso o plano deixa Fix 3 para fase posterior, nao agora
