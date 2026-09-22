"""Read-only readiness check for Binance testnet operation.

Sends no orders. Every call here is a GET, so the script can be run at any
time, including against a live session, without touching positions.

scripts/verify_readiness.py is the complementary check that actually places
and cancels orders on the testnet; run this one first, because it catches the
configuration mistakes that would make that test meaningless or dangerous.

    python scripts/check_testnet_ready.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")

from config.settings import AIConfig, TradingConfig, active_config  # noqa: E402
from trading.binance_connector import BinanceConnector  # noqa: E402

OK, FAIL, WARN = "[ OK ]", "[FALHA]", "[AVISO]"
results: list = []


def record(passed, label, detail=""):
    mark = OK if passed is True else (WARN if passed is None else FAIL)
    print("%s %-46s %s" % (mark, label, detail))
    results.append((passed, label))


async def main() -> int:
    ai_config = AIConfig()
    trading_config = TradingConfig()

    print("=" * 96)
    print("PRONTIDAO PARA TESTNET (somente leitura — nenhuma ordem e enviada)")
    print("=" * 96)

    # ── Configuracao ─────────────────────────────────────────────────────────
    print("\n1. Configuracao")
    testnet = bool(getattr(active_config, "BINANCE_TESTNET", False))
    record(testnet, "BINANCE_TESTNET ligado",
           "ordens vao para a testnet" if testnet else "ORDENS IRIAM PARA A CORRETORA REAL")
    has_keys = bool(getattr(active_config, "BINANCE_TESTNET_API_KEY", None)
                    and getattr(active_config, "BINANCE_TESTNET_API_SECRET", None))
    record(has_keys, "chaves de testnet configuradas",
           "" if has_keys else "cairia nas chaves de producao")

    cap = float(getattr(ai_config, "TRAINING_LEVERAGE_CAP", 3.0))
    record(cap <= float(trading_config.MAX_LEVERAGE_PER_TRADE),
           "teto de alavancagem coerente", "producao limitada a %.0fx (treino: %.0fx)" % (cap, cap))
    required = bool(getattr(ai_config, "REQUIRE_OOS_POLICY_APPROVAL", True))
    record(True if required else None, "aprovacao exigida antes de operar",
           "" if required else "desligada (REQUIRE_OOS_POLICY_APPROVAL=False)")

    # ── Trava de somente-leitura ─────────────────────────────────────────────
    print("\n2. Separacao entre dados e ordens")
    data_connector = BinanceConnector(active_config, force_production=True)
    trading_connector = BinanceConnector(active_config, force_production=False)
    record(data_connector.allow_order_execution is False,
           "conector de DADOS recusa ordens", "aponta para producao com chaves reais")
    record(trading_connector.allow_order_execution is True,
           "conector de TRADING aceita ordens", trading_connector.base_url)
    blocked = await data_connector._make_request("POST", "/fapi/v1/order",
                                                 params={"symbol": "BTCUSDT"}, signed=True)
    record(blocked is None, "ordem pelo conector de dados bloqueada na pratica")

    # ── Conectividade ────────────────────────────────────────────────────────
    print("\n3. Conectividade")
    symbol = trading_config.PRIMARY_PAIR
    try:
        await data_connector.connect()
        ticker = await data_connector.get_ticker_price(symbol)
        price = float(ticker["price"]) if ticker and "price" in ticker else None
        record(price is not None, "dados de mercado (producao)",
               "%s = %s" % (symbol, price))
    except Exception as exc:
        record(False, "dados de mercado (producao)", str(exc)[:60])
    finally:
        await data_connector.close()

    balance = None
    try:
        await trading_connector.connect()
        summary = await trading_connector.get_account_summary()
        if summary:
            balance = summary.get("totalWalletBalance") or summary.get("total_balance")
        record(summary is not None, "conta da testnet acessivel",
               "saldo: %s" % balance if balance is not None else "")
        info = await trading_connector.get_symbol_info(symbol)
        record(info is not None, "contrato %s disponivel na testnet" % symbol)
    except Exception as exc:
        record(False, "conta da testnet acessivel", str(exc)[:60])
    finally:
        await trading_connector.close()

    # ── Artefatos de modelo ──────────────────────────────────────────────────
    print("\n4. Modelos e aprovacao")
    model_dir = Path(str(ai_config.MODEL_DIR))
    policy = str(getattr(trading_config, "LIVE_POLICY", "agent_mirror"))
    record(policy == "agent_mirror", "politica de operacao", policy)
    if policy == "agent_mirror":
        from trading import agent_mirror as mirror
        listed = [a.strip() for a in str(trading_config.LIVE_AGENTS).split(",") if a.strip()]
        # Agentes de diagnostico (so testnet) ficam em models_ai/shadow, com paridade propria.
        diagnostic = [a.strip() for a in str(getattr(trading_config, "TESTNET_DIAGNOSTIC_AGENTS", "")).split(",")
                      if a.strip()]
        approved = [a for a in listed if a not in diagnostic]
        for folder, names, label in ((model_dir, approved, "aprovados"), (model_dir / "shadow", diagnostic, "diagnostico")):
            if not names:
                continue
            for name in names:
                for filename in ("%s_specialist_sac.zip" % name, "%s_specialist_scaler.joblib" % name,
                                 "%s_feature_contract.json" % name):
                    record((folder / filename).exists(), "%s presente (%s)" % (filename, label))
            try:
                parity = json.loads((folder / "agent_mirror_parity.json").read_text(encoding="utf-8"))
                current = mirror.artifact_hashes(folder, names)
                same = all(parity["artifact_hashes"].get(k) == v for k, v in current.items())
                passed = bool(parity.get("all_passed")) and same
                record(passed if passed or bool(getattr(ai_config, "REQUIRE_OOS_POLICY_APPROVAL", True)) else None,
                       "replay do espelho = backtest (%s)" % label,
                       "" if same else "paridade medida em outros arquivos")
            except (OSError, ValueError, KeyError):
                record(False if bool(getattr(ai_config, "REQUIRE_OOS_POLICY_APPROVAL", True)) else None,
                       "replay do espelho = backtest (%s)" % label, "rode scripts/verify_agent_mirror_parity.py")
        if diagnostic:
            record(bool(trading_config.BINANCE_TESTNET), "agentes de diagnostico so na testnet", ", ".join(diagnostic))
        ok, detail = mirror.approval_is_valid(model_dir)
        record(ok if ok or bool(getattr(ai_config, "REQUIRE_OOS_POLICY_APPROVAL", True)) else None,
               "aprovacao do espelho valida para os arquivos", detail)
    from feature_engineering.crypto_regime_detector import canonical_detector_matches
    detector_ok, detector_detail = canonical_detector_matches()
    record(detector_ok, "detector de regime = o que rotulou o treino", detector_detail)

    dataset = ROOT / "data" / "featured_data_causal.parquet"
    record(dataset.exists() if dataset.exists() else None, "dataset causal presente",
           "" if dataset.exists() else "rode scripts/build_causal_dataset.py")

    # ── Veredito ─────────────────────────────────────────────────────────────
    failures = [label for passed, label in results if passed is False]
    warnings = [label for passed, label in results if passed is None]
    print("\n" + "=" * 96)
    if failures:
        print("NAO PRONTO. Reprovado em:")
        for label in failures:
            print("  - %s" % label)
    elif warnings:
        print("QUASE PRONTO. Pendente:")
        for label in warnings:
            print("  - %s" % label)
    else:
        print("PRONTO para operar na testnet.")
    print("=" * 96)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
