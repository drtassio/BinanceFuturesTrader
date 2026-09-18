"""Forward-test block: bars after the dataset ends, built the way the live bot builds them.

The dataset stops on 2026-03-21. Everything after that has never been used
for training, rule selection, model selection or the meta-model fit, so it is
the one block where a specialist can be tested without any of those choices
having seen it. And unlike a rebuilt historical frame, these rows come out of
the running bot's own pipeline — the same klines requests, the same funding
alignment, the same FeatureEngineeringPipeline and meta-model bundle — so a
result here is a result on what the bot will actually observe.

The pipeline turns an 852-bar request into ~600 closed rows, and only the last
~200 of them carry the same meta-model and regime-prior values they would have
as the newest bar (measured: identical up to 200 bars from the end, different
from 300 on, while the inputs warm up). The live bot only ever decides on the
newest bar, so each window contributes just its stable tail and the block is
stitched from successive windows. Consecutive windows overlap, and the
overlap is compared: if the same bar gets different values from two windows,
the frame is not stable enough to test on and the script fails.

The regime is the exception, and it cannot be stitched. The detector labels a
window with HMM Viterbi decoding plus a segment smoother, so a bar's regime
changes when later bars enter the window. The live bot only ever reads the
regime of its newest bar, so every bar here gets the regime the detector gives
it as the last bar of the 600 closed bars ending on it (checked: identical to
the live pipeline's newest bar on 12 of 12 holdout samples, confidence,
tp_prior_dir and ml_p_long included, to four decimals). The regime priors and
the meta-model columns, which read the regime, are then rebuilt from it.

Acceptance is not the overlap: rows further from a window's end carry a larger
EMA-200 start-up residual, so the overlap can disagree slightly while every
row is still what the bot would see. The test that matters is run last: for a
random sample of bars, the live pipeline is rebuilt ending on that bar and its
newest bar is compared with the block, column by column over what the
specialist and the meta-model read. The block is accepted when the regime
matches on at least 95% of the sample and no other column is off by more than
the tolerance. (Measured on 40 bars: 39 identical; the one regime difference
came from the live pipeline running the detector with the still-forming
candle in the window; the bot now builds features from closed candles only.)
Read-only: GET requests through the production data connector, no orders.

    python scripts/build_forward_dataset.py --start "2026-03-21 03:00"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TREND_SKIP_OPTUNA", "1")
os.chdir(ROOT)

BAR = pd.Timedelta(minutes=15)
REGIME_WINDOW = 600
REGIME_DERIVED_PREFIXES = ("regime", "tp_", "ml_", "trend_pred_")
STABLE_TAIL = 200
WARMUP = 350


def _regime_chunk(job):
    """Newest-bar regime for each position, from the 600 closed bars ending on it."""
    path, positions = job
    from feature_engineering.crypto_regime_detector import CryptoRegimeDetector

    frame = pd.read_parquet(path)
    detector = CryptoRegimeDetector.load()
    out = []
    for i in positions:
        result = detector.predict(frame.iloc[i - REGIME_WINDOW + 1:i + 1]).iloc[-1]
        out.append((int(i), int(result["regime"]), float(result["confidence"])))
    return out


def causal_regime(block: pd.DataFrame, first: int, workers: int, scratch: Path) -> pd.DataFrame:
    import joblib
    from multiprocessing import Pool
    from feature_engineering.causal_features import add_regime_priors
    from learning.meta_labeler import predict_bundle

    block.to_parquet(scratch)
    positions = np.arange(first, len(block))
    jobs = [(str(scratch), chunk) for chunk in np.array_split(positions, workers * 8) if len(chunk)]
    began = time.monotonic()
    with Pool(workers) as pool:
        results = [row for part in pool.imap_unordered(_regime_chunk, jobs) for row in part]
    print("regime causal de %d barras em %.0fs" % (len(results), time.monotonic() - began), flush=True)
    results.sort()
    out = block.iloc[[r[0] for r in results]].copy()
    regime = np.array([r[1] for r in results])
    out["regime"] = regime
    out["regime_val"] = regime
    out["regime_confidence"] = np.array([r[2] for r in results])
    for name, code in (("regime_is_bull", 0), ("regime_is_bear", 1), ("regime_is_ranger", 2)):
        if name in out.columns:
            out[name] = (regime == code).astype(int)
    out, _ = add_regime_priors(out)
    out["tp_uncertainty"] = 1.0 - out["tp_prior_conf"].astype(float)
    for target, source in (("trend_pred_uptrend", "tp_regime_up"), ("trend_pred_downtrend", "tp_regime_down"),
                           ("trend_pred_neutral", "tp_regime_sideways")):
        if target in out.columns:
            out[target] = out[source]
    bundle = joblib.load(ROOT / "models_ai" / "meta_labeler.joblib")
    for column, values in predict_bundle(bundle, out).items():
        out[column] = values.to_numpy()
    return out


def checked_columns(contracts) -> set:
    """What must not depend on the window: what the specialists and the meta-model
    read, plus the prices the environment trades on. Cumulative indicators such as
    OBV depend on where the window starts and are read by neither."""
    import joblib

    columns = {"open", "high", "low", "close", "volume", "atr_15m", "funding_rate"}
    columns.update(joblib.load(ROOT / "models_ai" / "meta_labeler.joblib")["features"])
    for path in contracts:
        columns.update(json.loads(Path(path).read_text(encoding="utf-8"))["feature_columns"])
    return {c for c in columns if not c.startswith(REGIME_DERIVED_PREFIXES)}


async def build(start: pd.Timestamp, end: pd.Timestamp, step: int, overlap: int, tolerance: float, checked: set):
    from config.settings import AIConfig, TradingConfig, active_config
    from data_provider import DataProvider
    from feature_engineering.main import FeatureEngineeringPipeline
    from trading.binance_connector import BinanceConnector
    from trading.teacher_policy import closed_bars

    ai_config, trading_config = AIConfig(), TradingConfig()
    connector = BinanceConnector(active_config, force_production=True)
    await connector.connect()
    pieces, worst = [], {}
    try:
        pipeline = FeatureEngineeringPipeline(ai_config)
        provider = DataProvider(connector, pipeline)
        symbol = trading_config.PRIMARY_PAIR
        primary = trading_config.PRIMARY_TIMEFRAME_TRADING
        points = ai_config.LOOKBACK_WINDOW_MAIN + 52
        cursor = start + BAR * step
        while True:
            window_end = min(cursor, end)
            began = time.monotonic()
            end_ms = int(window_end.timestamp() * 1000)
            raw = {}
            for interval in provider.all_timeframes:
                frame = await provider._fetch_and_format_data(symbol, interval, limit=points, end_ts=end_ms)
                if frame is None or frame.empty:
                    raise RuntimeError("sem klines para %s ate %s" % (interval, window_end))
                interval_ms = provider._get_interval_milliseconds(interval)
                raw[interval] = frame.loc[(frame.index.asi8 // 10**6 + interval_ms) <= end_ms]
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
                raise RuntimeError("create_features vazio em %s" % window_end)
            featured = pipeline.apply_hidden_features(featured)
            if featured.index.tz is None:
                featured.index = featured.index.tz_localize("UTC")
            featured = closed_bars(featured, window_end)
            if step + overlap > STABLE_TAIL or len(featured) < STABLE_TAIL + WARMUP:
                raise RuntimeError("janela sem cauda estavel suficiente: %d linhas" % len(featured))
            piece = featured.iloc[-(step + overlap):]
            if pieces:
                shared = pieces[-1].index.intersection(piece.index)
                for column in checked:
                    if column not in piece.columns or column not in pieces[-1].columns or not len(shared):
                        continue
                    a = pieces[-1].loc[shared, column].astype(float).to_numpy()
                    b = piece.loc[shared, column].astype(float).to_numpy()
                    scale = max(float(np.nanstd(piece[column].astype(float))), 1e-9)
                    diff = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b)) / scale
                    worst[column] = max(worst.get(column, 0.0), float(np.nanmax(diff)) if diff.size else 0.0)
                pieces[-1] = pieces[-1].loc[pieces[-1].index < piece.index[0]]
            pieces.append(piece)
            top = sorted(worst.items(), key=lambda kv: -kv[1])[:3]
            print("janela ate %s: %d barras (%.0fs) | pior sobreposicao (desvios): %s" % (
                window_end, len(piece), time.monotonic() - began,
                " ".join("%s=%.2g" % kv for kv in top if kv[1] > 0) or "0"), flush=True)
            if window_end >= end:
                break
            cursor = cursor + BAR * step
    finally:
        await connector.close()
    block = pd.concat(pieces).sort_index()
    block = block.loc[(block.index >= start) & ~block.index.duplicated(keep="last")]
    stable = all(v <= tolerance for v in worst.values())
    return block, worst, stable


def validate_against_live(block: pd.DataFrame, checked: set, samples: int, seed: int, tolerance: float,
                          compare_regime: bool = True) -> dict:
    from scripts.verify_live_parity import build_live_frame
    from trading.teacher_policy import closed_bars

    extra = {"regime_confidence", "tp_prior_conf", "tp_prior_dir", "ml_p_long", "ml_p_short", "ml_edge", "ml_conf"}
    columns = sorted(c for c in (checked | extra if compare_regime else checked) if c in block.columns)
    scale = block[columns].astype(float).std().replace(0.0, 1e-9)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(block.index, size=min(samples, len(block)), replace=False))
    regime_equal, worst, bars = 0, {}, []
    for bar in chosen:
        live = asyncio.run(build_live_frame(bar + BAR))
        if live.index.tz is None:
            live.index = live.index.tz_localize("UTC")
        live = closed_bars(live, bar + BAR)
        if live.index[-1] != bar:
            raise RuntimeError("pipeline ao vivo terminou em %s, esperado %s" % (live.index[-1], bar))
        same_regime = (not compare_regime) or int(live.loc[bar, "regime"]) == int(block.loc[bar, "regime"])
        regime_equal += same_regime
        diff = (live.loc[bar, columns].astype(float) - block.loc[bar, columns].astype(float)).abs() / scale
        if same_regime:
            for column, value in diff.items():
                worst[column] = max(worst.get(column, 0.0), float(value))
        bars.append({"bar": str(bar), "same_regime": bool(same_regime), "max_std": float(diff.max())})
    top = dict(sorted(worst.items(), key=lambda kv: -kv[1])[:8])
    share = regime_equal / max(len(chosen), 1)
    accepted = share >= 0.95 and all(v <= tolerance for v in worst.values())
    print("validacao contra o ultimo candle ao vivo: regime igual em %d/%d | pior coluna (regime igual): %s | aceito: %s" % (
        regime_equal, len(chosen), top, accepted))
    return {"samples": len(chosen), "regime_equal_share": share, "worst_columns_in_std": top,
            "accepted": accepted, "bars": bars}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-03-21 03:00")
    parser.add_argument("--end", default=None, help="padrao: agora")
    parser.add_argument("--step", type=int, default=150, help="barras novas por janela")
    parser.add_argument("--overlap", type=int, default=50)
    # A EMA de 200 barras ainda guarda ~2% do ponto de partida de uma janela de
    # 600 linhas: a mesma barra difere em ate ~0.07 desvio entre janelas. O bot
    # ao vivo tem o mesmo residuo; verify_live_parity aceita ate 0.05 no ultimo
    # candle. Acima de 0.1 algo depende de fato da janela.
    parser.add_argument("--tolerance", type=float, default=0.1, help="em desvios-padrao da coluna")
    # overlap_worst_columns_in_std fica no relatorio como diagnostico.
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--validate-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--validate-only", action="store_true", help="so revalida um bloco ja montado")
    # Regime e meta-modelo nao mostraram poder preditivo (IC ~0.01; AUC ~0.5 em
    # validacao, holdout e no bloco futuro) e o regime causal por barra custa
    # ~0.3s cada. Para montar historicos longos eles podem ser omitidos: as
    # colunas derivadas ficam NaN para que nada as use por engano.
    parser.add_argument("--skip-regime", action="store_true")
    parser.add_argument("--contract", action="append", default=[],
                        help="feature_contract.json de uma execucao; repetivel")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "forward_live.parquet")
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC").floor("15min")
    checked = checked_columns(args.contract)
    meta_path = Path(str(args.output) + ".meta.json")
    if args.validate_only:
        block = pd.read_parquet(args.output)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["live_validation"] = validate_against_live(block, checked, args.validate_samples, args.seed, args.tolerance)
        meta["stable"] = bool(meta["live_validation"]["accepted"]) and not meta.get("gaps")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return 0 if meta["stable"] else 1

    # O regime do primeiro candle testado precisa de 600 barras fechadas antes dele.
    warm_start = start - BAR * (REGIME_WINDOW + 10)
    stitched, worst, _ = asyncio.run(build(warm_start, end, args.step, args.overlap, args.tolerance, checked))
    first = int(np.searchsorted(stitched.index, start))
    if first < REGIME_WINDOW - 1:
        raise RuntimeError("historico insuficiente antes de %s para o regime causal" % start)
    if args.skip_regime:
        block = stitched.iloc[first:].copy()
        derived = [c for c in block.columns if str(c).startswith(REGIME_DERIVED_PREFIXES)]
        block[derived] = np.nan
    else:
        block = causal_regime(stitched, first, args.workers, Path(str(args.output) + ".stitched.parquet"))
    gaps = block.index.to_series().diff().dropna()
    missing_bars = int((gaps > BAR).sum())
    block.to_parquet(args.output)
    validation = validate_against_live(block, checked, args.validate_samples, args.seed, args.tolerance,
                                       compare_regime=not args.skip_regime)
    stable = bool(validation["accepted"])
    if args.skip_regime:
        print("regime omitido: colunas %s ficaram NaN" % (REGIME_DERIVED_PREFIXES,))
    worst_columns = dict(sorted(worst.items(), key=lambda kv: -kv[1])[:10])
    meta = {"start": str(block.index[0]), "end": str(block.index[-1]), "rows": len(block),
            "gaps": missing_bars, "overlap_worst_columns_in_std": worst_columns, "stable": stable,
            "regime": ("omitted (NaN)" if args.skip_regime else
                       "causal: newest bar of the %d closed bars ending on each bar" % REGIME_WINDOW),
            "checked_columns": len(checked), "contracts": [str(c) for c in args.contract],
            "live_validation": validation,
            "built_at": pd.Timestamp.now(tz="UTC").isoformat(), "source": "live pipeline (read-only)"}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n%d barras de %s a %s | lacunas: %d | janelas consistentes: %s" % (
        len(block), meta["start"], meta["end"], missing_bars, stable))
    return 0 if stable and missing_bars == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
