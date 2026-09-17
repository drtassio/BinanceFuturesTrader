"""Does the live pipeline hand the specialists the same numbers training did?

Rebuilds features exactly the way the running bot does — the same GET requests
to Binance (read-only data connector), the same 852-bar window per timeframe,
the same funding alignment, the same FeatureEngineeringPipeline.create_features
— but ending at a moment that also exists in the training dataset. Then it
compares, column by column, every feature the specialists observe.

A mismatch here means a model that trained on one set of numbers would trade
on another. Sends no orders.

    python scripts/verify_live_parity.py --end "2026-03-10 00:00"
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")
os.chdir(ROOT)


async def build_live_frame(end: pd.Timestamp, skip_meta: bool = False) -> pd.DataFrame:
    from config.settings import AIConfig, TradingConfig, active_config
    from data_provider import DataProvider
    from feature_engineering.main import FeatureEngineeringPipeline
    from trading.binance_connector import BinanceConnector

    if skip_meta:
        # Diagnostico: sem isto o primeiro insumo ausente do meta-modelo
        # interrompe tudo e esconde as demais colunas que faltam ao vivo.
        FeatureEngineeringPipeline._add_meta_features = lambda self, df: df

    ai_config, trading_config = AIConfig(), TradingConfig()
    connector = BinanceConnector(active_config, force_production=True)
    await connector.connect()
    try:
        pipeline = FeatureEngineeringPipeline(ai_config)
        provider = DataProvider(connector, pipeline)
        symbol = trading_config.PRIMARY_PAIR
        end_ms = int(end.timestamp() * 1000)
        points = ai_config.LOOKBACK_WINDOW_MAIN + 52
        raw = {}
        for interval in provider.all_timeframes:
            frame = await provider._fetch_and_format_data(symbol, interval, limit=points, end_ts=end_ms)
            if frame is None or frame.empty:
                raise RuntimeError("sem klines para %s" % interval)
            # O bot so transforma em feature o candle ja fechado em 'end'
            # (DataProvider.get_latest_features). Na busca historica o candle
            # aberto em 'end' vem completo: mante-lo seria olhar o futuro.
            interval_ms = provider._get_interval_milliseconds(interval)
            raw[interval] = frame.loc[(frame.index.asi8 // 10**6 + interval_ms) <= end_ms]
        primary = trading_config.PRIMARY_TIMEFRAME_TRADING
        primary_df = raw[primary].copy()
        start_ms = int((primary_df.index.min() - pd.Timedelta(days=1)).timestamp() * 1000)
        rows = await connector.get_funding_rate_history(symbol, start_ms, end_ms)
        if rows:
            funding = pd.DataFrame(rows)
            funding.index = pd.to_datetime(funding["fundingTime"], unit="ms", utc=True)
            series = funding.sort_index()["fundingRate"].astype(float)
            primary_df["funding_rate"] = series.reindex(primary_df.index, method="ffill").fillna(0.0)
            rates = primary_df["funding_rate"]
            std = rates.rolling(32, min_periods=4).std().replace(0.0, np.nan)
            primary_df["funding_rate_z_32"] = ((rates - rates.rolling(32, min_periods=4).mean()) / std).fillna(0.0).clip(-8, 8)
            raw[primary] = primary_df
        featured = await pipeline.create_features(raw, symbol, primary, fit_scaler=False)
        if featured is None or featured.empty:
            raise RuntimeError("create_features devolveu vazio")
        # Priors de regime como o caminho de decisao os monta.
        featured = pipeline.apply_hidden_features(featured)
        return featured
    finally:
        await connector.close()


def specialist_columns(train: pd.DataFrame) -> list:
    from config.settings import AIConfig
    from specialists.bull_specialist import BullTradingEnv

    sample = train.iloc[-800:].ffill().fillna(0.0)
    env = BullTradingEnv(df=sample, config=AIConfig(), mode="training", specialist_name="bull_specialist")
    return list(env.feature_columns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", default="2026-03-10 00:00")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "featured_data_causal.parquet")
    parser.add_argument("--rows", type=int, default=16, help="ultimas barras comparadas")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="diferenca maxima aceita, em desvios-padrao da coluna no treino")
    parser.add_argument("--skip-meta", action="store_true", help="nao gerar ml_* (diagnostico)")
    args = parser.parse_args()

    end = pd.Timestamp(args.end, tz="UTC")
    train = pd.read_parquet(args.data).sort_index()
    if train.index.tz is None:
        train.index = train.index.tz_localize("UTC")
    columns = specialist_columns(train)
    print("especialista observa %d colunas de mercado" % len(columns))

    live = asyncio.run(build_live_frame(end, skip_meta=args.skip_meta))
    import joblib
    bundle_path = ROOT / "models_ai" / "meta_labeler.joblib"
    if bundle_path.exists():
        meta_inputs = list(joblib.load(bundle_path)["features"])
        absent = [c for c in meta_inputs if c not in live.columns]
        print("entradas do meta-modelo ausentes ao vivo: %d %s" % (len(absent), absent))
    all_missing = sorted(set(train.columns) - set(live.columns))
    print("colunas do dataset que o pipeline ao vivo nao produz: %d" % len(all_missing))
    (ROOT / "data" / "live_missing_columns.json").write_text(
        __import__("json").dumps(all_missing, indent=1), encoding="utf-8")
    if live.index.tz is None:
        live.index = live.index.tz_localize("UTC")

    # As ultimas barras completas da janela ao vivo, que tambem existem no treino.
    common = live.index.intersection(train.index)
    common = common[common < end][-args.rows:]
    print("barras comparadas: %d (%s -> %s)\n" % (len(common), common.min(), common.max()))
    if len(common) == 0:
        print("nenhuma barra em comum")
        return 1

    # As colunas ml_* do dataset sao walk-forward: vieram de modelos treinados
    # so com o passado de cada bloco. O pacote ao vivo foi ajustado em todo o
    # historico e ja viu estas barras. Comparar as duas mede diferenca de
    # modelo, nao de pipeline. O teste correto aplica o MESMO pacote as
    # entradas do treino e as entradas ao vivo: se as entradas batem, as
    # saidas batem.
    train_view = train.loc[common].copy()
    if not args.skip_meta and bundle_path.exists():
        from feature_engineering.causal_features import add_regime_priors
        from learning.meta_labeler import predict_bundle
        bundle = joblib.load(bundle_path)
        scored, _ = add_regime_priors(train_view)
        for column, values in predict_bundle(bundle, scored).items():
            train_view[column] = values.to_numpy()

    missing = [c for c in columns if c not in live.columns and not (args.skip_meta and c.startswith("ml_"))]
    rows = []
    for column in columns:
        if column in missing or column not in live.columns:
            continue
        a = train_view[column].astype(float).to_numpy()
        b = live.loc[common, column].astype(float).to_numpy()
        # Escala = variacao tipica da coluna no treino. Dividir pelo proprio
        # valor explode em barras onde a feature passa perto de zero.
        typical = float(np.nanstd(train[column].astype(float).to_numpy()))
        scale = max(typical, 1e-9)
        rel = float(np.nanmax(np.abs(a - b)) / scale)
        rows.append((rel, column, float(a[-1]), float(b[-1])))
    rows.sort(reverse=True)

    bad = [r for r in rows if not np.isfinite(r[0]) or r[0] > args.tolerance]
    print("%-34s %12s %14s %14s" % ("coluna", "dif/desvio", "treino", "ao vivo"))
    for rel, column, a, b in rows[:25]:
        flag = "  <-- DIVERGE" if (not np.isfinite(rel) or rel > args.tolerance) else ""
        print("%-34s %12.4f %14.6g %14.6g%s" % (column, rel, a, b, flag))
    print()
    print("ausentes ao vivo: %d %s" % (len(missing), missing[:15]))
    print("divergentes (> %.2f desvio): %d de %d" % (args.tolerance, len(bad), len(rows)))
    if missing or bad:
        print("\nPARIDADE REPROVADA")
        return 1
    print("\nPARIDADE OK: o bot ao vivo entrega ao especialista os mesmos numeros do treino.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
