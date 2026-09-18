"""
train_leg_vision.py
Ensina um modelo de visão a reconhecer, no gráfico, as pernas marcadas por
scripts/mark_legs.py, usando só o que já aconteceu até o candle atual.

Transformação visual: em cada candle de 15m o gráfico das últimas 24h
(96 candles de 15m) e das últimas 48h em 1h (só candles de 1h já FECHADOS) vira
uma imagem de ALTURA x largura com três canais: corpo de alta, corpo de baixa e
pavio. Volume e fluxo agressor entram como uma faixa embaixo. Uma CNN pequena
lê as duas imagens e diz: perna de alta, perna de baixa ou lateral.

Blocos cronológicos com embargo de 2 dias:
  treino     início dos rótulos .. TRAIN_END
  validação  .. VAL_END    (escolhe época e limiares de entrada/saída)
  holdout    .. fim        (lido uma única vez, no fim)

Saídas:
  models_ai/leg_vision.pt                pesos + normalização + limiares
  reports/leg_vision_report.json         métricas por bloco
  data/leg_vision_signals.parquet        p_long / p_short por candle (fora do treino)
"""

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
HEIGHT = 48                 # linhas de preço da imagem
W15, W1H = 96, 48           # candles por imagem
TRAIN_END = pd.Timestamp("2025-12-31 23:45")
VAL_END = pd.Timestamp("2026-04-30 23:45")
EMBARGO = pd.Timedelta("2D")
COST = 0.0014


def load_bars():
    raw = pd.read_parquet(ROOT / "data/market_raw/btc_perp_15m.parquet")
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw["flow"] = (2 * raw["taker_buy_base"] - raw["volume"]) / raw["volume"].replace(0, np.nan)
    one_h = raw.resample("1h", closed="left", label="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "taker_buy_base": "sum"}).dropna()
    one_h["flow"] = (2 * one_h["taker_buy_base"] - one_h["volume"]) / one_h["volume"].replace(0, np.nan)
    return raw, one_h


def raster(o, h, l, c, v, f, height=HEIGHT):
    """(N, T) arrays -> (N, 4, height+8, T) uint8 images, each window scaled to its own range."""
    n, t = c.shape
    lo = l.min(1, keepdims=True)
    hi = h.max(1, keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    to_row = lambda x: np.clip(((x - lo) / span * (height - 1)).round().astype(int), 0, height - 1)
    rows = np.arange(height)[None, :, None]
    r_low, r_high = to_row(l)[:, None, :], to_row(h)[:, None, :]
    r_bot, r_top = to_row(np.minimum(o, c))[:, None, :], to_row(np.maximum(o, c))[:, None, :]
    wick = (rows >= r_low) & (rows <= r_high)
    body = (rows >= r_bot) & (rows <= r_top)
    up = (c >= o)[:, None, :]
    img = np.zeros((n, 4, height + 8, t), dtype=np.uint8)
    img[:, 0, :height] = (body & up) * 255
    img[:, 1, :height] = (body & ~up) * 255
    img[:, 2, :height] = wick * 255
    # faixa de baixo: volume relativo (altura) com o sinal do fluxo agressor no canal 3
    vol = v / np.maximum(v.max(1, keepdims=True), 1e-9)
    vrow = (vol * 7).round().astype(int)[:, None, :]
    band = np.arange(8)[None, :, None] < vrow
    img[:, 2, height:] = band * 255
    img[:, 3, height:] = (band * np.clip((np.nan_to_num(f)[:, None, :] + 1) * 127.5, 0, 255)).astype(np.uint8)
    return img[:, :, ::-1, :].copy()   # preço alto em cima, como no gráfico


def windows(frame: pd.DataFrame, ends: np.ndarray, width: int):
    idx = ends[:, None] + np.arange(-width + 1, 1)[None, :]
    cols = [frame[k].to_numpy(float)[idx] for k in ("open", "high", "low", "close", "volume", "flow")]
    return raster(*cols)


class LegNet(nn.Module):
    def __init__(self):
        super().__init__()
        def branch():
            return nn.Sequential(
                nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((2, 6)), nn.Flatten())
        self.b15, self.b1h = branch(), branch()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * 48 * 12, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x15, x1h):
        return self.head(torch.cat([self.b15(x15), self.b1h(x1h)], 1))


def build_dataset(raw, one_h, labels):
    pos15 = raw.index.get_indexer(labels.index)
    # candle de 1h que o candle de 15m em t já enxerga: o último que FECHOU até t+15m
    closed_1h = (labels.index + pd.Timedelta("15min")).floor("1h") - pd.Timedelta("1h")
    pos1h = one_h.index.get_indexer(closed_1h)
    ok = (pos15 >= W15) & (pos1h >= W1H)
    y = labels["hs_target_position"].to_numpy().astype(np.int64)[ok] + 1   # 0 short, 1 fora, 2 long
    return labels.index[ok], pos15[ok], pos1h[ok], y


def batches(raw, one_h, p15, p1h, device):
    x15 = torch.from_numpy(windows(raw, p15, W15)).to(device).float() / 255.0
    x1h = torch.from_numpy(windows(one_h, p1h, W1H)).to(device).float() / 255.0
    return x15, x1h


def predict(model, raw, one_h, p15, p1h, device, bs=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(p15), bs):
            x15, x1h = batches(raw, one_h, p15[s:s + bs], p1h[s:s + bs], device)
            out.append(torch.softmax(model(x15, x1h), 1).cpu().numpy())
    return np.concatenate(out)


def trade(times, close, prob, enter, stay):
    """Entra quando a prob. da perna passa de `enter`, segura enquanto fica acima de `stay`."""
    pos, entry, trades = 0, 0.0, []
    for i in range(len(prob)):
        pl, ps = prob[i, 2], prob[i, 0]
        if pos == 0:
            if pl >= enter and pl > ps:
                pos, entry, t0 = 1, close[i], times[i]
            elif ps >= enter and ps > pl:
                pos, entry, t0 = -1, close[i], times[i]
        else:
            keep = (pl if pos > 0 else ps) >= stay
            if not keep or i == len(prob) - 1:
                trades.append((t0, times[i], pos, pos * (close[i] / entry - 1) - COST))
                pos = 0
    return pd.DataFrame(trades, columns=["entry", "exit", "side", "net"])


def summary(t):
    if t.empty:
        return {"trades": 0, "net_compound": 0.0, "pf": 0.0, "win_rate": 0.0, "long": 0, "short": 0, "max_dd": 0.0}
    w, l = t.net[t.net > 0].sum(), -t.net[t.net <= 0].sum()
    eq = (1 + t.net).cumprod()
    return {"trades": int(len(t)), "net_compound": float(eq.iloc[-1] - 1), "pf": float(w / l) if l else float("inf"),
            "win_rate": float((t.net > 0).mean()), "long": int((t.side > 0).sum()), "short": int((t.side < 0).sum()),
            "max_dd": float((1 - eq / eq.cummax()).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw, one_h = load_bars()
    labels = pd.read_parquet(ROOT / "data/teacher_leg_labels.parquet")
    labels.index = pd.to_datetime(labels.index)
    times, p15, p1h, y = build_dataset(raw, one_h, labels)
    tr = times <= TRAIN_END - EMBARGO
    va = (times > TRAIN_END) & (times <= VAL_END - EMBARGO)
    ho = times > VAL_END
    print("dispositivo %s | treino %d | validação %d | holdout %d candles" % (device, tr.sum(), va.sum(), ho.sum()))

    model = LegNet().to(device)
    counts = np.bincount(y[tr], minlength=3)
    weight = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    idx_tr, idx_va = np.nonzero(tr)[0], np.nonzero(va)[0]
    best = (np.inf, None, -1)
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(idx_tr)
        total = 0.0
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            x15, x1h = batches(raw, one_h, p15[b], p1h[b], device)
            loss = F.cross_entropy(model(x15, x1h), torch.as_tensor(y[b], device=device), weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        pv = predict(model, raw, one_h, p15[idx_va], p1h[idx_va], device)
        val_loss = float(F.nll_loss(torch.log(torch.as_tensor(pv) + 1e-9), torch.as_tensor(y[idx_va]),
                                    weight=weight.cpu()))
        acc = (pv.argmax(1) == y[idx_va]).mean()
        print("época %2d: perda treino %.4f | validação perda %.4f acerto %.1f%%" % (epoch, total / len(order), val_loss, acc * 100))
        if val_loss < best[0]:
            best = (val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch)
        elif epoch - best[2] >= 5:
            break
    model.load_state_dict(best[1])
    print("melhor época na validação: %d" % best[2])

    close = raw["close"].to_numpy(float)
    probs = {name: predict(model, raw, one_h, p15[m], p1h[m], device) for name, m in (("treino", tr), ("validacao", va), ("holdout", ho))}
    # limiares escolhidos SÓ na validação
    grid = []
    for enter in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
        for stay in (0.25, 0.3, 0.35, 0.4, 0.45):
            if stay >= enter:
                continue
            t = trade(times[va], close[p15[va]], probs["validacao"], enter, stay)
            s = summary(t)
            score = s["net_compound"] / max(s["max_dd"], 0.02) if s["trades"] >= 20 and s["pf"] > 1 else -np.inf
            grid.append((score, enter, stay, s))
    grid.sort(key=lambda g: g[0], reverse=True)
    _, enter, stay, _ = grid[0]
    report = {"thresholds": {"enter": enter, "stay": stay}, "best_epoch": best[2],
              "periods": {k: [str(times[m].min()), str(times[m].max())] for k, m in (("treino", tr), ("validacao", va), ("holdout", ho))}}
    for name, m in (("treino", tr), ("validacao", va), ("holdout", ho)):
        t = trade(times[m], close[p15[m]], probs[name], enter, stay)
        report[name] = summary(t)
        acc = (probs[name].argmax(1) == y[m]).mean()
        report[name]["accuracy"] = float(acc)
        s = report[name]
        print("%-10s trades %4d (long %d / short %d) | composto %+7.1f%% | PF %.2f | acerto trades %.0f%% | DD %.1f%% | acerto rótulo %.1f%%"
              % (name, s["trades"], s["long"], s["short"], s["net_compound"] * 100, s["pf"], s["win_rate"] * 100,
                 s["max_dd"] * 100, acc * 100))
    print("limiares (escolhidos na validação): entra >= %.2f, segura >= %.2f" % (enter, stay))

    out = pd.DataFrame(np.concatenate([probs["validacao"], probs["holdout"]]),
                       index=np.concatenate([times[va], times[ho]]), columns=["p_short", "p_flat", "p_long"])
    out.to_parquet(ROOT / "data/leg_vision_signals.parquet")
    torch.save({"state_dict": best[1], "thresholds": report["thresholds"], "height": HEIGHT, "w15": W15, "w1h": W1H},
               ROOT / "models_ai/leg_vision.pt")
    (ROOT / "reports/leg_vision_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
