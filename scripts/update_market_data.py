"""
update_market_data.py
Traz os dados de mercado até o último candle de 15m FECHADO.

1. data/market_raw/*.parquet  - klines 15m (perp/spot BTC, perp ETH) e funding, API pública.
2. data/featured_data.parquet - estende com o MESMO pipeline do bot
   (DataProvider -> FeatureEngineeringPipeline.create_features), em todos os
   timeframes que o bot usa (TradingConfig.ALL_TRADING_TIMEFRAMES).
   Antes de gravar, recalcula um trecho que já existe e compara com o arquivo,
   para garantir que as linhas novas saem iguais às antigas.

    python scripts/update_market_data.py            # tudo
    python scripts/update_market_data.py --raw-only
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

RAW_DIR = ROOT_DIR / "data/market_raw"
FEATURED = ROOT_DIR / "data/featured_data.parquet"
KLINE_COLS = ["open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
              "taker_buy_base", "taker_buy_quote"]
RAW_SOURCES = {
    "btc_perp_15m.parquet": ("https://fapi.binance.com/fapi/v1/klines", "BTCUSDT", 1500),
    "eth_perp_15m.parquet": ("https://fapi.binance.com/fapi/v1/klines", "ETHUSDT", 1500),
    "btc_spot_15m.parquet": ("https://api.binance.com/api/v3/klines", "BTCUSDT", 1000),
}
WARMUP_DAYS = 200          # EMA200 do 4h precisa de ~133 dias para convergir
PARITY_DAYS = 14           # trecho já existente recalculado para comparação
PARITY_TOLERANCE = 1e-3    # diferença relativa mediana aceitável por coluna


def last_closed_15m() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC").tz_convert(None)
    return now.floor("15min") - pd.Timedelta("15min")   # abertura do último candle já fechado


def _get(url, params):
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 ** attempt)
    r.raise_for_status()


def update_klines(name, url, symbol, limit):
    path = RAW_DIR / name
    old = pd.read_parquet(path)
    last_open = last_closed_15m()
    start = old.index.max() + pd.Timedelta("15min")
    rows = []
    while start <= last_open:
        batch = _get(url, {"symbol": symbol, "interval": "15m", "limit": limit,
                           "startTime": int(start.tz_localize("UTC").timestamp() * 1000)})
        if not batch:
            break
        rows += batch
        start = pd.Timestamp(batch[-1][0], unit="ms") + pd.Timedelta("15min")
        time.sleep(0.2)
    if not rows:
        print(f"   {name}: já atualizado até {old.index.max()}")
        return
    new = pd.DataFrame([r[1:11] for r in rows], columns=KLINE_COLS, dtype=float)
    new.index = pd.to_datetime([r[0] for r in rows], unit="ms")
    new.index.name = old.index.name
    new = new[new.index <= last_open]
    out = pd.concat([old, new])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.to_parquet(path)
    print(f"   {name}: +{len(new)} candles → {out.index.max()}")


def update_funding():
    path = RAW_DIR / "btc_funding.parquet"
    old = pd.read_parquet(path)
    start = old.index.max() + pd.Timedelta("1ms")
    rows = _get("https://fapi.binance.com/fapi/v1/fundingRate",
                {"symbol": "BTCUSDT", "limit": 1000, "startTime": int(start.tz_localize("UTC").timestamp() * 1000)})
    if not rows:
        print(f"   btc_funding.parquet: já atualizado até {old.index.max()}")
        return
    new = pd.DataFrame({"fundingRate": [float(r["fundingRate"]) for r in rows]},
                       index=pd.to_datetime([r["fundingTime"] for r in rows], unit="ms").floor("s"))
    new.index.name = old.index.name
    new = new[new.index > old.index.max()]
    if new.empty:
        print(f"   btc_funding.parquet: já atualizado até {old.index.max()}")
        return
    out = pd.concat([old, new])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.to_parquet(path)
    print(f"   btc_funding.parquet: +{len(new)} taxas → {out.index.max()}")


async def build_featured(start: datetime, end: datetime) -> pd.DataFrame:
    from config.settings import active_config, AIConfig, TradingConfig
    from data_provider import DataProvider
    from feature_engineering.main import FeatureEngineeringPipeline
    from trading.binance_connector import BinanceConnector

    connector = BinanceConnector(active_config, force_production=True)   # dados reais, só leitura
    await connector.connect()
    try:
        pipeline = FeatureEngineeringPipeline(AIConfig())
        provider = DataProvider(connector, pipeline)
        start_ts, end_ts = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        raw = {}
        for tf in TradingConfig.ALL_TRADING_TIMEFRAMES:
            _, df = await provider._fetch_historical_batch(TradingConfig.PRIMARY_PAIR, tf, start_ts, end_ts)
            raw[tf] = df
        # featured_data.parquet is the base BEFORE the causal treatment and the
        # meta-model: build_causal_dataset.py applies both on top of it, while
        # the live create_features applies them inline. Stop at the base here.
        import feature_engineering.causal_features as causal
        original = causal.build_causal_features
        # [FIX B4] Lambda signature must accept **kwargs (e.g. df5) passed by create_features
        causal.build_causal_features = lambda df, **kwargs: (df, {"trend_features": [], "tape_features": []})
        pipeline._add_meta_features = lambda df: df
        try:
            featured = await pipeline.create_features(raw, TradingConfig.PRIMARY_PAIR, "15m", fit_scaler=False)
        finally:
            causal.build_causal_features = original
        return featured, raw
    finally:
        await connector.close()


def _naive(df):
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def _splice_obv(col, old, fresh, raw, common):
    """NativeIndicators.obv = X / rolling_std_200(X), com X = soma acumulada do volume com sinal.
    Um início diferente só soma uma constante C a X (o desvio não muda), então
    obv_antigo = (X + C) / s. Estima C no trecho comum e renormaliza."""
    tf = col.split("_", 1)[1]
    d = _naive(raw[tf].copy())
    vol = d["volume"].replace(0, 1e-9)
    x = (vol * np.sign(d["close"].diff()).fillna(0)).cumsum()
    s = x.rolling(200, min_periods=1).std() + 1e-9
    if tf != "15m":   # mesmo deslocamento anti-lookahead do create_features
        x.index = x.index + pd.Timedelta(tf) - pd.Timedelta("15min")
        s.index = x.index
    x = x.reindex(fresh.index, method="ffill")
    s = s.reindex(fresh.index, method="ffill")
    c = (old.loc[common, col].astype(float) * s.loc[common] - x.loc[common]).median()
    return ((x + c) / s).astype(old[col].dtype)


def update_featured(force: bool = False, micro_ok: bool = False):
    old = pd.read_parquet(FEATURED)
    old_end = old.index.max()
    last_open = last_closed_15m()
    if old_end >= last_open:
        print(f"   featured_data já atualizado até {old_end}")
        return

    start = (old_end - pd.Timedelta(days=WARMUP_DAYS + PARITY_DAYS)).tz_localize("UTC").to_pydatetime()
    end = (last_open + pd.Timedelta("15min")).tz_localize("UTC").to_pydatetime()
    fresh, raw = asyncio.run(build_featured(start, end))
    fresh = _naive(fresh)
    fresh = fresh[fresh.index <= last_open].copy()

    # "*_tf_*" columns come from an old historical builder that the live
    # pipeline never produces; build_causal_dataset.py drops them. New rows
    # carry NaN there instead of a value the bot could not reproduce.
    legacy = [c for c in old.columns if "_tf_" in c and c not in fresh.columns]
    for col in legacy:
        fresh[col] = np.nan
    if legacy:
        print(f"   {len(legacy)} colunas legadas *_tf_* (descartadas no dataset causal) ficam vazias nas linhas novas")
    missing = [c for c in old.columns if c not in fresh.columns]
    extra = [c for c in fresh.columns if c not in old.columns]
    if missing:
        raise SystemExit(f"❌ pipeline não gerou {len(missing)} colunas do arquivo atual: {missing[:10]}")

    # Paridade: o trecho recalculado tem de bater com o que já está no arquivo
    lo = old_end - pd.Timedelta(days=PARITY_DAYS)
    a, b = old.loc[lo:old_end], fresh.loc[lo:old_end, old.columns]
    common = a.index.intersection(b.index)
    num = [c for c in old.columns if pd.api.types.is_numeric_dtype(old[c]) and c not in legacy]
    a_num, b_num = a.loc[common, num].astype(float), b.loc[common, num].astype(float)
    rel = ((a_num - b_num).abs() / (a_num.abs() + 1e-9)).median()
    # Indicadores acumulados (OBV, PVT, ADL) dependem de onde a série começa: se a
    # diferença no trecho comum é uma constante, emenda somando essa constante.
    for col in rel[rel > PARITY_TOLERANCE].index:
        diff = a_num[col] - b_num[col]
        if diff.std() <= PARITY_TOLERANCE * (a_num[col].abs().mean() + 1e-9):
            fresh[col] = fresh[col].astype(float) + diff.mean()
            rel[col] = ((a_num[col] - fresh.loc[common, col]).abs() / (a_num[col].abs() + 1e-9)).median()
            print(f"   {col}: série acumulada emendada com deslocamento constante {diff.mean():+.4g}")
        elif col.startswith("obv_"):
            fresh[col] = _splice_obv(col, old, fresh, raw, common)
            rel[col] = ((a_num[col] - fresh.loc[common, col]).abs() / (a_num[col].abs() + 1e-9)).median()
            print(f"   {col}: OBV reconstruído a partir da soma acumulada e renormalizado")
    bad = rel[rel > PARITY_TOLERANCE].sort_values(ascending=False)
    print(f"   paridade em {len(common)} candles já existentes: {len(num) - len(bad)}/{len(num)} colunas iguais")
    by_tf = pd.Series([c.rsplit("_", 1)[-1] if c.rsplit("_", 1)[-1] in ("1m", "5m", "15m", "1h", "4h") else "base"
                       for c in bad.index], dtype=object).value_counts()
    if len(bad):
        print("   divergentes por timeframe: " + ", ".join(f"{tf}={n}" for tf, n in by_tf.items()))
    if micro_ok:
        # The specialists never observe *_1m/*_5m (trend_specialist._EXCLUDED_TF).
        # The shipped base file cannot be reproduced there, so the new rows keep
        # what the live bot computes and the check applies to everything else.
        bad = bad[[not (c.endswith("_1m") or c.endswith("_5m")) for c in bad.index]]
    if len(bad):
        print("   colunas divergentes (diferença relativa mediana):")
        print(bad.head(15).to_string())

    new = fresh.loc[fresh.index > old_end, old.columns]
    for c in old.columns:
        if new[c].dtype != old[c].dtype:
            try:
                new[c] = new[c].astype(old[c].dtype)
            except (TypeError, ValueError):
                pass
    out = pd.concat([old, new])
    out.index.name = old.index.name
    gaps = out.index.to_series().diff().dropna()
    print(f"   intervalos fora de 15m: {(gaps != pd.Timedelta('15min')).sum()}")
    if len(bad) and not force:
        candidate = FEATURED.with_name("featured_data_candidate.parquet")
        out.to_parquet(candidate)
        raise SystemExit(f"⚠️ paridade falhou: {FEATURED.name} NÃO foi alterado; resultado em {candidate.name} "
                         f"(use --force para gravar mesmo assim)")
    out.to_parquet(FEATURED)
    print(f"   featured_data: +{len(new)} candles ({new.index.min()} → {new.index.max()}); "
          f"colunas extras ignoradas: {len(extra)}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="grava o featured_data mesmo com paridade divergente")
    ap.add_argument("--allow-micro-divergence", action="store_true",
                    help="aceita divergencia so em colunas *_1m/*_5m, que os especialistas nao observam")
    args = ap.parse_args()
    logging.disable(logging.INFO)

    print(f"Último candle de 15m fechado: {last_closed_15m()} UTC")
    print("1) Dados brutos (API pública da Binance)")
    for name, (url, symbol, limit) in RAW_SOURCES.items():
        update_klines(name, url, symbol, limit)
    update_funding()
    if not args.raw_only:
        print("2) featured_data.parquet (pipeline do bot)")
        update_featured(args.force, args.allow_micro_divergence)


if __name__ == "__main__":
    main()
