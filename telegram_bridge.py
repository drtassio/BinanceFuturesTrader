"""
Ponte Telegram <-> Pipeline Autonomo (v2).

Integra com .claude/state/goal.json para mostrar estado real, gates,
KPIs e fases. Aceita comandos para controlar o pipeline.

Diferencas vs v1:
- Le goal.json em vez de pipeline_state interno
- Suporta /goal show, /goal gates, /agentes report
- /override <gate> para forcar transicao (com warning)
- /kpi daily roda /goal measure e responde com snapshot
- Confirmacao dupla para /abort
- Audit-findings.jsonl integrado em /status

Uso: python telegram_bridge.py
"""

import asyncio
import aiohttp
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
API_URL = f"https://api.telegram.org/bot{TOKEN}"

PROJECT_ROOT = Path(__file__).resolve().parent
GOAL_PATH = PROJECT_ROOT / ".claude" / "state" / "goal.json"
FINDINGS_PATH = PROJECT_ROOT / ".claude" / "state" / "audit-findings.jsonl"
RUNLOG_PATH = PROJECT_ROOT / ".claude" / "state" / "run-log.md"
AUDIT_REPORT_PATH = PROJECT_ROOT / ".claude" / "state" / "audit-report.md"

# Estado runtime (volatile — fonte de verdade eh goal.json)
runtime = {
    "paused": False,
    "last_command_offset": 0,
    "pending_confirmations": {},  # {"action": "abort", "expires_at": ts}
}


# ============================================================================
# IO helpers
# ============================================================================

def read_goal() -> Dict[str, Any]:
    try:
        with open(GOAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e), "current_phase": "unknown"}


def write_goal(data: Dict[str, Any]) -> bool:
    try:
        with open(GOAL_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def read_findings(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not FINDINGS_PATH.exists():
        return []
    findings = []
    try:
        with open(FINDINGS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    findings.append(json.loads(line))
    except Exception:
        pass
    if limit:
        findings = findings[-limit:]
    return findings


def append_runlog(entry: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    block = f"\n\n## [{ts}] {entry}\n"
    try:
        with open(RUNLOG_PATH, "a", encoding="utf-8") as f:
            f.write(block)
    except Exception:
        pass


# ============================================================================
# Telegram API
# ============================================================================

async def send_message(text: str, parse_mode: str = "HTML") -> bool:
    if not TOKEN or not CHAT_ID:
        print(f"[TG-OFFLINE] {text[:200]}")
        return False
    try:
        url = f"{API_URL}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text[:4096], "parse_mode": parse_mode}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=15) as resp:
                result = await resp.json()
                return result.get("ok", False)
    except Exception as e:
        print(f"[TELEGRAM] Erro ao enviar: {e}")
        return False


async def get_updates(offset: int = 0) -> list:
    if not TOKEN:
        return []
    try:
        url = f"{API_URL}/getUpdates"
        params = {"offset": offset, "timeout": 30, "allowed_updates": ["message"]}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=35) as resp:
                result = await resp.json()
                if result.get("ok"):
                    return result.get("result", [])
    except Exception:
        pass
    return []


# ============================================================================
# Render helpers
# ============================================================================

def _gate_emoji(passed: bool) -> str:
    return "OK" if passed else "--"


def render_status() -> str:
    goal = read_goal()
    findings = read_findings()
    critical = [f for f in findings if f.get("severity") == "critical"]
    high = [f for f in findings if f.get("severity") == "high"]

    phase = goal.get("current_phase", "unknown")
    stages = goal.get("stages", {})
    stage_info = stages.get(phase, {})
    status_str = stage_info.get("status", "n/a")
    started = stage_info.get("started_at", "n/a")

    return (
        f"<b>STATUS DO PIPELINE</b>\n"
        f"-----------------------\n"
        f"Fase: <b>{phase}</b>\n"
        f"Status: {status_str}\n"
        f"Iniciada: {started}\n"
        f"Pausado: {'sim' if runtime['paused'] else 'nao'}\n"
        f"\n"
        f"Findings ativos:\n"
        f"  CRITICAL: <b>{len(critical)}</b>\n"
        f"  HIGH:     <b>{len(high)}</b>\n"
        f"\n"
        f"Use /goal gates para ver bloqueadores."
    )


def render_goal_show() -> str:
    g = read_goal()
    if "error" in g:
        return f"Erro lendo goal.json: {g['error']}"

    gates = g.get("scientific_gates", {})
    kpis = g.get("kpis", {})

    lines = [
        f"<b>META</b>",
        "-----------------------",
        f"{g.get('primary', 'n/a')[:200]}",
        "",
        f"<b>Fase atual:</b> {g.get('current_phase', 'n/a')}",
        "",
        "<b>KPIs alvo:</b>",
    ]
    for k, v in kpis.items():
        lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("<b>Gates cientificos:</b>")
    for gate, passed in gates.items():
        lines.append(f"  [{_gate_emoji(passed)}] {gate}")

    return "\n".join(lines)


def render_goal_gates() -> str:
    g = read_goal()
    gates = g.get("scientific_gates", {})
    findings = read_findings()
    crit_high = [f for f in findings if f.get("severity") in ("critical", "high")]

    blockers = [k for k, v in gates.items() if not v]
    if not blockers:
        return "Todos os gates OK. Pipeline pode avancar."

    lines = ["<b>BLOQUEADORES</b>", "-----------------------"]
    for b in blockers:
        agent_name = b.replace("_pass", "").replace("_audit", "-auditor")
        related = [f for f in crit_high if agent_name in f.get("agent", "")]
        lines.append(f"\n[--] <b>{b}</b>")
        if related:
            for f in related[:3]:
                lines.append(
                    f"  {f.get('severity', '?').upper()}: {f.get('file', '?')}\n"
                    f"  {f.get('issue', '?')[:120]}"
                )
        else:
            lines.append(f"  (sem findings registrados — rode /agentes run {agent_name})")

    return "\n".join(lines)


def render_findings_top(n: int = 5) -> str:
    findings = read_findings()
    crit = [f for f in findings if f.get("severity") == "critical"][:n]
    high = [f for f in findings if f.get("severity") == "high"][:n]
    all_f = crit + high[: max(0, n - len(crit))]
    if not all_f:
        return "Nenhum finding CRITICAL/HIGH ativo."
    lines = ["<b>TOP FINDINGS</b>", "-----------------------"]
    for i, f in enumerate(all_f[:n], 1):
        sev = f.get("severity", "?").upper()
        lines.append(
            f"\n{i}. [{sev}] {f.get('file', '?')}\n"
            f"   {f.get('issue', '?')[:140]}\n"
            f"   fix: {f.get('fix', '?')[:120]}"
        )
    return "\n".join(lines)


# ============================================================================
# Command handlers
# ============================================================================

async def handle_status(_: str) -> str:
    return render_status()


async def handle_goal(args: str) -> str:
    sub = args.strip().split(None, 1)
    cmd = sub[0] if sub else "show"

    if cmd == "show" or cmd == "":
        return render_goal_show()
    elif cmd == "gates":
        return render_goal_gates()
    elif cmd == "measure":
        # Trigger measure (placeholder — Claude Code roda /goal measure manualmente)
        append_runlog("/goal measure solicitado via Telegram")
        return ("Pedido de measure registrado. Aguarde o pipeline rodar /goal measure "
                "ou execute manualmente no CLI.")
    elif cmd == "history":
        g = read_goal()
        history = g.get("history", [])[-10:]
        if not history:
            return "Sem historico."
        lines = ["<b>HISTORICO (ultimos 10)</b>", "-----------------------"]
        for h in history:
            lines.append(f"\n{h.get('timestamp', '?')[:19]} | {h.get('event', '?')}")
        return "\n".join(lines)
    else:
        return f"Subcomando /goal desconhecido: {cmd}. Use: show, gates, measure, history."


async def handle_pause(_: str) -> str:
    runtime["paused"] = True
    append_runlog("Pipeline PAUSADO por humano via Telegram")
    return "Pipeline pausado. Use /resume para retomar."


async def handle_resume(_: str) -> str:
    runtime["paused"] = False
    append_runlog("Pipeline RETOMADO por humano via Telegram")
    return "Pipeline retomado."


async def handle_agentes(args: str) -> str:
    sub = args.strip().split(None, 1)
    cmd = sub[0] if sub else "list"
    if cmd == "list":
        return (
            "<b>AGENTES CIENTIFICOS (10)</b>\n"
            "-----------------------\n"
            "1. leakage-auditor\n"
            "2. regime-auditor\n"
            "3. reward-auditor\n"
            "4. replay-buffer-auditor\n"
            "5. sac-stability-auditor\n"
            "6. moe-gating-auditor\n"
            "7. feature-eng-auditor\n"
            "8. execution-realism-auditor\n"
            "9. validation-auditor\n"
            "10. production-readiness\n"
            "\nExecute /agentes run all no CLI Claude Code."
        )
    elif cmd == "report":
        if not AUDIT_REPORT_PATH.exists():
            return "audit-report.md ainda nao gerado. Rode /agentes report no CLI."
        try:
            txt = AUDIT_REPORT_PATH.read_text(encoding="utf-8")[:3800]
            return f"<pre>{txt}</pre>"
        except Exception as e:
            return f"Erro lendo report: {e}"
    elif cmd == "findings":
        return render_findings_top(5)
    else:
        return f"Subcomando /agentes desconhecido: {cmd}. Use: list, report, findings."


async def handle_kpi(_: str) -> str:
    return render_goal_show()


async def handle_override(args: str) -> str:
    if not args.strip():
        return ("Uso: /override <gate_name>\n"
                "Exemplo: /override leakage_audit_pass\n"
                "PERIGOSO: marca gate como aprovado sem evidencia.")
    gate = args.strip()
    g = read_goal()
    gates = g.get("scientific_gates", {})
    if gate not in gates:
        return f"Gate desconhecido: {gate}. Veja /goal show."
    # Confirmacao dupla
    key = f"override:{gate}"
    if runtime["pending_confirmations"].get(key):
        gates[gate] = True
        if "history" not in g:
            g["history"] = []
        g["history"].append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": f"override_gate:{gate}",
            "actor": "humano_telegram",
        })
        write_goal(g)
        runtime["pending_confirmations"].pop(key, None)
        append_runlog(f"OVERRIDE manual de gate {gate} por Telegram")
        return f"Gate {gate} marcado como aprovado (override manual)."
    runtime["pending_confirmations"][key] = datetime.utcnow().isoformat()
    return (f"CONFIRMAR override do gate {gate}?\n"
            f"Envie /override {gate} novamente em 60s para confirmar.")


async def handle_abort(_: str) -> str:
    key = "abort"
    if runtime["pending_confirmations"].get(key):
        g = read_goal()
        g["current_phase"] = "audit"
        for k in g.get("scientific_gates", {}):
            g["scientific_gates"][k] = False
        if "history" not in g:
            g["history"] = []
        g["history"].append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": "PIPELINE_ABORTED",
            "actor": "humano_telegram",
        })
        write_goal(g)
        runtime["pending_confirmations"].pop(key, None)
        append_runlog("PIPELINE ABORTADO via Telegram")
        return "Pipeline ABORTADO. Estado resetado para fase audit."
    runtime["pending_confirmations"][key] = datetime.utcnow().isoformat()
    return ("CONFIRMAR ABORT do pipeline?\n"
            "Envie /abort novamente em 60s para confirmar.\n"
            "Isto reseta gates e fase para audit.")


async def handle_help(_: str) -> str:
    return (
        "<b>COMANDOS DISPONIVEIS</b>\n"
        "-----------------------\n"
        "/status — fase atual + findings\n"
        "/goal show — KPIs e gates\n"
        "/goal gates — bloqueadores ativos\n"
        "/goal measure — pede ao pipeline para medir\n"
        "/goal history — ultimos eventos\n"
        "/agentes list — lista agentes\n"
        "/agentes report — relatorio de auditoria\n"
        "/agentes findings — top findings\n"
        "/kpi — snapshot atual (alias /goal show)\n"
        "/pause /resume — controla loop\n"
        "/override &lt;gate&gt; — forca gate (PERIGO)\n"
        "/abort — aborta pipeline (PERIGO)\n"
        "/help — esta mensagem"
    )


COMMAND_TABLE = {
    "/status": handle_status,
    "/goal": handle_goal,
    "/pause": handle_pause,
    "/resume": handle_resume,
    "/agentes": handle_agentes,
    "/agents": handle_agentes,  # alias en
    "/kpi": handle_kpi,
    "/override": handle_override,
    "/abort": handle_abort,
    "/help": handle_help,
    "/start": handle_help,
}


async def dispatch(text: str) -> str:
    text = text.strip()
    parts = text.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    handler = COMMAND_TABLE.get(cmd)
    if handler:
        try:
            return await handler(args)
        except Exception as e:
            return f"Erro processando {cmd}: {e}"
    return f"Comando nao reconhecido: <code>{text[:50]}</code>. /help para listar."


# ============================================================================
# Notificacoes do pipeline
# ============================================================================

async def notify_phase_transition(from_phase: str, to_phase: str, details: str = "") -> None:
    await send_message(
        f"<b>FASE: {from_phase} -> {to_phase}</b>\n"
        f"-----------------------\n"
        f"{details}\n"
        f"{datetime.utcnow().strftime('%H:%M:%S UTC')}"
    )
    append_runlog(f"TRANSICAO {from_phase} -> {to_phase}: {details}")


async def notify_critical_finding(finding: Dict[str, Any]) -> None:
    await send_message(
        f"<b>FINDING CRITICAL</b>\n"
        f"-----------------------\n"
        f"Agente: {finding.get('agent', '?')}\n"
        f"Arquivo: {finding.get('file', '?')}\n"
        f"Issue: {finding.get('issue', '?')[:300]}\n"
        f"Fix: {finding.get('fix', '?')[:300]}"
    )


async def notify_daily_kpi() -> None:
    """Chamada agendada — envia snapshot diario."""
    await send_message(render_goal_show())


async def request_human_approval(reason: str, options: List[str] = None) -> None:
    options = options or ["GO", "STOP", "EXTEND"]
    await send_message(
        f"<b>APROVACAO HUMANA REQUERIDA</b>\n"
        f"-----------------------\n"
        f"{reason}\n"
        f"\n"
        f"Responda: {' | '.join(options)}"
    )


# ============================================================================
# Watchdog — detecta inatividade, rate limits, erros
# ============================================================================

WATCHDOG_STATE = {
    "last_mtime": 0.0,
    "last_alert_at": 0.0,
    "consecutive_silent_alerts": 0,
    "last_rate_limit_seen": None,        # timestamp da ultima vez que viu 429
    "rate_limit_reset_at": None,         # quando deve resetar (estimado)
    # Progress tracking (novo)
    "last_findings_count": 0,            # total de linhas em audit-findings.jsonl
    "last_findings_by_severity": {},     # contagem por CRITICAL/HIGH/MEDIUM/LOW
    "last_phase": None,                  # current_phase em goal.json
    "last_gates_passed": 0,              # quantos scientific_gates = true
    "last_runlog_size": 0,               # bytes de run-log.md
    "initialized": False,                # primeira passagem nao alerta tudo como "novo"
}

# Padroes a detectar em run-log.md e audit-findings.jsonl (ultimas linhas)
ERROR_PATTERNS = [
    (re.compile(r"429|rate.?limit|Limite de uso", re.IGNORECASE), "rate_limit"),
    (re.compile(r"API Error|api.error", re.IGNORECASE), "api_error"),
    (re.compile(r"OOM|out of memory|CUDA out of memory", re.IGNORECASE), "oom"),
    (re.compile(r"Traceback|Exception:", re.IGNORECASE), "traceback"),
    (re.compile(r"CRITICAL", re.IGNORECASE), "critical_finding"),
]

# Janelas de alerta (minutos sem update -> alerta)
SILENCE_THRESHOLDS_MIN = [20, 60, 180]  # 20min, 1h, 3h


def _max_mtime() -> float:
    paths = [GOAL_PATH, FINDINGS_PATH, RUNLOG_PATH]
    return max((p.stat().st_mtime if p.exists() else 0.0) for p in paths)


def _scan_recent_errors(lines_to_scan: int = 80) -> List[Dict[str, str]]:
    """Le ultimas linhas de run-log + audit-findings e busca padroes."""
    hits = []
    for path in [RUNLOG_PATH, FINDINGS_PATH]:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                # Le ultimas N linhas eficiente
                f.seek(0, 2)
                size = f.tell()
                read_size = min(size, 64 * 1024)
                f.seek(max(0, size - read_size))
                content = f.read()
            lines = content.splitlines()[-lines_to_scan:]
            for line in lines:
                for pattern, kind in ERROR_PATTERNS:
                    if pattern.search(line):
                        hits.append({
                            "file": path.name,
                            "kind": kind,
                            "snippet": line.strip()[:200],
                        })
                        break
        except Exception:
            continue
    return hits


def _parse_rate_limit_reset(text: str) -> Optional[datetime]:
    """Extrai 'Sera resetado em: 2h 9min 28s' do texto."""
    m = re.search(r"resetado em:\s*(?:(\d+)h)?\s*(?:(\d+)min)?\s*(?:(\d+)s)?", text, re.IGNORECASE)
    if not m:
        return None
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    if h + mi + s == 0:
        return None
    return datetime.utcnow() + timedelta(hours=h, minutes=mi, seconds=s)


# Padroes positivos no run-log que merecem notificacao
PROGRESS_PATTERNS = [
    (re.compile(r"FIX|CORRECAO|CORRIGIDO|corrigid", re.IGNORECASE), "fix_applied"),
    (re.compile(r"AUDITORIA COMPLETA|AUDIT COMPLETED|AUDITORIA FINALIZADA", re.IGNORECASE), "audit_done"),
    (re.compile(r"FASE \d|PHASE \d|TRANSICAO|PIPELINE START|transitando", re.IGNORECASE), "phase_change"),
    (re.compile(r"HPO TRIAL|TRAINING.*COMPLETE|TREINO.*COMPLET", re.IGNORECASE), "training_progress"),
    (re.compile(r"GATE.*OK|GATE.*PASSED|gate_passed", re.IGNORECASE), "gate_passed"),
]


def _count_findings_by_severity() -> Dict[str, int]:
    """Conta findings em audit-findings.jsonl por severity."""
    counts = {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
    if not FINDINGS_PATH.exists():
        return counts
    try:
        with open(FINDINGS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                counts["total"] += 1
                try:
                    obj = json.loads(line)
                    sev = obj.get("severity", "").lower()
                    if sev in counts:
                        counts[sev] += 1
                except Exception:
                    continue
    except Exception:
        pass
    return counts


def _gates_passed_count() -> int:
    g = read_goal()
    gates = g.get("scientific_gates", {})
    return sum(1 for v in gates.values() if v is True)


def _get_recent_runlog_events(max_chars: int = 4000) -> List[Dict[str, str]]:
    """Retorna eventos recentes no run-log.md classificados por tipo."""
    if not RUNLOG_PATH.exists():
        return []
    events = []
    try:
        with open(RUNLOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_chars))
            content = f.read()
        # Pega ultima entry (entre ## headers)
        chunks = content.split("\n## ")
        for chunk in chunks[-3:]:  # ultimas 3 entries
            for pattern, kind in PROGRESS_PATTERNS:
                if pattern.search(chunk):
                    first_line = chunk.split("\n")[0][:200]
                    events.append({"kind": kind, "snippet": first_line.strip()})
                    break
    except Exception:
        pass
    return events


async def watchdog_loop(poll_seconds: int = 60):
    """Monitora atividade do pipeline e notifica Telegram em silencio/erro/progresso."""
    print("[WATCHDOG] Iniciando")
    WATCHDOG_STATE["last_mtime"] = _max_mtime()
    # Inicializa baseline sem alertar
    WATCHDOG_STATE["last_findings_by_severity"] = _count_findings_by_severity()
    WATCHDOG_STATE["last_findings_count"] = WATCHDOG_STATE["last_findings_by_severity"]["total"]
    _g = read_goal()
    WATCHDOG_STATE["last_phase"] = _g.get("current_phase")
    WATCHDOG_STATE["last_gates_passed"] = _gates_passed_count()
    WATCHDOG_STATE["last_runlog_size"] = RUNLOG_PATH.stat().st_size if RUNLOG_PATH.exists() else 0
    WATCHDOG_STATE["initialized"] = True

    while True:
        try:
            current_mtime = _max_mtime()
            now = time.time()

            # 0. PROGRESS DETECTION (novo) — notifica eventos positivos
            await _check_progress_and_notify()

            # 1. Atividade detectada?
            if current_mtime > WATCHDOG_STATE["last_mtime"]:
                # Reset contador
                if WATCHDOG_STATE["consecutive_silent_alerts"] > 0:
                    await send_message(
                        f"<b>PIPELINE RETOMOU ATIVIDADE</b>\n"
                        f"-----------------------\n"
                        f"Apos silencio. Ultima atualizacao: agora\n"
                        f"{datetime.utcnow().strftime('%H:%M:%S UTC')}"
                    )
                WATCHDOG_STATE["last_mtime"] = current_mtime
                WATCHDOG_STATE["consecutive_silent_alerts"] = 0
                WATCHDOG_STATE["last_alert_at"] = 0.0

            # 2. Silencio prolongado?
            silence_min = (now - WATCHDOG_STATE["last_mtime"]) / 60.0
            threshold = SILENCE_THRESHOLDS_MIN[
                min(WATCHDOG_STATE["consecutive_silent_alerts"], len(SILENCE_THRESHOLDS_MIN) - 1)
            ]
            if silence_min >= threshold:
                # Limita: 1 alerta por threshold cruzado
                last_alert_min_ago = (now - WATCHDOG_STATE["last_alert_at"]) / 60.0
                if last_alert_min_ago > 10:  # nao spammar
                    g = read_goal()
                    phase = g.get("current_phase", "?")
                    errors = _scan_recent_errors()
                    error_summary = ""
                    rate_limit_hit = any(e["kind"] == "rate_limit" for e in errors)

                    if rate_limit_hit:
                        # Tenta extrair tempo de reset
                        for e in errors:
                            if e["kind"] == "rate_limit":
                                reset_at = _parse_rate_limit_reset(e["snippet"])
                                if reset_at:
                                    WATCHDOG_STATE["rate_limit_reset_at"] = reset_at
                                    error_summary = (
                                        f"\nRATE LIMIT detectado. Reset estimado: "
                                        f"{reset_at.strftime('%H:%M:%S UTC')}"
                                    )
                                    break
                    elif errors:
                        error_summary = "\nErros recentes detectados:\n" + "\n".join(
                            f"- [{e['kind']}] {e['snippet'][:120]}" for e in errors[-3:]
                        )

                    await send_message(
                        f"<b>PIPELINE SILENCIOSO</b>\n"
                        f"-----------------------\n"
                        f"Sem atualizacao ha <b>{int(silence_min)}min</b>\n"
                        f"Fase: {phase}\n"
                        f"{error_summary}\n"
                        f"\n"
                        f"Use /status para mais info."
                    )
                    WATCHDOG_STATE["last_alert_at"] = now
                    WATCHDOG_STATE["consecutive_silent_alerts"] += 1

            # 3. Rate limit estimado expirou?
            if WATCHDOG_STATE["rate_limit_reset_at"]:
                if datetime.utcnow() >= WATCHDOG_STATE["rate_limit_reset_at"]:
                    await send_message(
                        "<b>RATE LIMIT DEVE TER RESETADO</b>\n"
                        "-----------------------\n"
                        "Janela estimada terminou. Retome o CLI ou /pipeline start."
                    )
                    WATCHDOG_STATE["rate_limit_reset_at"] = None

        except Exception as e:
            print(f"[WATCHDOG] erro: {e}")

        await asyncio.sleep(poll_seconds)


async def _check_progress_and_notify():
    """Detecta progresso positivo desde a ultima checagem e notifica."""
    if not WATCHDOG_STATE.get("initialized"):
        return

    # 1. Mudanca de fase em goal.json
    g = read_goal()
    current_phase = g.get("current_phase")
    if current_phase and current_phase != WATCHDOG_STATE["last_phase"]:
        await send_message(
            f"<b>TRANSICAO DE FASE</b>\n"
            f"-----------------------\n"
            f"{WATCHDOG_STATE['last_phase']} -> <b>{current_phase}</b>\n"
            f"{datetime.utcnow().strftime('%H:%M:%S UTC')}"
        )
        WATCHDOG_STATE["last_phase"] = current_phase

    # 2. Novos gates passados
    current_gates = _gates_passed_count()
    if current_gates > WATCHDOG_STATE["last_gates_passed"]:
        delta = current_gates - WATCHDOG_STATE["last_gates_passed"]
        gates = g.get("scientific_gates", {})
        passed_now = [k for k, v in gates.items() if v is True]
        await send_message(
            f"<b>GATES CIENTIFICOS</b>\n"
            f"-----------------------\n"
            f"+{delta} gate(s) aprovados. Total: <b>{current_gates}/{len(gates)}</b>\n"
            f"Verdes: {', '.join(passed_now[-3:])}"
        )
        WATCHDOG_STATE["last_gates_passed"] = current_gates

    # 3. Novos findings (especialmente CRITICAL/HIGH)
    current_findings = _count_findings_by_severity()
    last = WATCHDOG_STATE["last_findings_by_severity"]
    if current_findings["total"] > last.get("total", 0):
        delta_total = current_findings["total"] - last.get("total", 0)
        delta_crit = current_findings["critical"] - last.get("critical", 0)
        delta_high = current_findings["high"] - last.get("high", 0)

        if delta_crit > 0:
            # Critical merece alerta dedicado
            await send_message(
                f"<b>NOVO FINDING CRITICAL</b>\n"
                f"-----------------------\n"
                f"+{delta_crit} CRITICAL detectado. Total CRITICAL ativo: {current_findings['critical']}\n"
                f"Use /agentes findings para detalhes."
            )
        elif delta_high > 0:
            await send_message(
                f"<b>NOVOS FINDINGS HIGH</b>\n"
                f"-----------------------\n"
                f"+{delta_high} HIGH. Total findings: {current_findings['total']}\n"
                f"(C={current_findings['critical']} H={current_findings['high']} M={current_findings['medium']} L={current_findings['low']})"
            )
        elif delta_total >= 5:
            # Bulk de findings menores
            await send_message(
                f"<b>AUDITORIA EM ANDAMENTO</b>\n"
                f"-----------------------\n"
                f"+{delta_total} findings ({current_findings['medium']}M + {current_findings['low']}L)\n"
                f"Total: {current_findings['total']}"
            )
        WATCHDOG_STATE["last_findings_by_severity"] = current_findings
        WATCHDOG_STATE["last_findings_count"] = current_findings["total"]

    # 4. Eventos recentes no run-log (fixes aplicados, etc)
    current_size = RUNLOG_PATH.stat().st_size if RUNLOG_PATH.exists() else 0
    if current_size > WATCHDOG_STATE["last_runlog_size"] + 100:  # >100 bytes de mudanca
        events = _get_recent_runlog_events()
        # Notifica APENAS o evento mais relevante (prioridade: fix > phase > training > audit > gate)
        priority = ["fix_applied", "phase_change", "audit_done", "training_progress", "gate_passed"]
        for kind in priority:
            event = next((e for e in events if e["kind"] == kind), None)
            if event:
                kind_emoji = {
                    "fix_applied": "FIX APLICADO",
                    "phase_change": "TRANSICAO",
                    "audit_done": "AUDITORIA",
                    "training_progress": "TREINO",
                    "gate_passed": "GATE",
                }.get(kind, "EVENTO")
                await send_message(
                    f"<b>{kind_emoji}</b>\n"
                    f"-----------------------\n"
                    f"{event['snippet'][:300]}"
                )
                break
        WATCHDOG_STATE["last_runlog_size"] = current_size


# ============================================================================
# Loop principal
# ============================================================================

async def telegram_listener():
    print(f"[TELEGRAM] Iniciando (chat_id={CHAT_ID})")
    await send_message(
        "<b>TELEGRAM BRIDGE v2 ATIVO</b>\n"
        "-----------------------\n"
        "Pipeline cientifico autonomo conectado.\n"
        "/help para comandos.\n"
        f"{datetime.utcnow().strftime('%H:%M:%S UTC')}"
    )

    offset = 0
    while True:
        updates = await get_updates(offset)
        for update in updates:
            offset = update["update_id"] + 1
            msg = update.get("message", {})
            text = msg.get("text", "")
            chat = msg.get("chat", {}).get("id")

            if str(chat) != str(CHAT_ID):
                continue  # ignora chats nao autorizados

            if text.startswith("/"):
                response = await dispatch(text)
                await send_message(response)

        await asyncio.sleep(1)


async def _main():
    """Roda listener (comandos) + watchdog (monitoramento) em paralelo."""
    await asyncio.gather(
        telegram_listener(),
        watchdog_loop(poll_seconds=60),
    )


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("[TELEGRAM] Encerrado.")
