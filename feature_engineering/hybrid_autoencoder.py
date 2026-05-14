# -----------------------------------------------------------------------------
# ARQUIVO: feature_engineering/hybrid_autoencoder.py
# -----------------------------------------------------------------------------
"""
Predictive State Representation Learning System para RL Quantitativo (v3.1 — JEPA)

Arquitetura:
  - Causal Encoder Determinístico (CNN + Transformer)
  - EMA Target Encoder (BYOL-style) para Representation Stability no SAC
  - Hierarchical Contrastive Learning (TS2Vec-style com Avg+Max Pooling)
  - VICReg Regularization (Variance & Covariance) para evitar dimensional collapse
  - JEPA Predictor (Joint-Embedding Predictive Architecture): predição no espaço
    latente entre z_t e z_{t+k}, eliminando decodificação para features ruidosas
  - RegimeHead: classificação de regime de mercado (Bull/Bear/Ranger)
  - ActiveU + EffRank diagnostics instrumentados no loop de validação

Loss: UncertaintyWeighting adaptativa entre JEPA, TS2Vec, VICReg e Regime.
Validação: Purged Walk-Forward Validation com Embargo para evitar Leakage OOS.
"""

import os
import copy
import math
import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.exceptions import TrialPruned

from sklearn.preprocessing import RobustScaler
import joblib

from utils.logger import get_logger
from config.settings import AIConfig

warnings.filterwarnings("ignore", category=UserWarning)
logger = get_logger("HybridAutoencoderV3")

N_REGIMES: int = 3
EPSILON: float = 1e-9

# ─────────────────────────────────────────────────────────────────────────────
# VISUAL HELPERS — barras inline, banners de secao, semaforos
# ─────────────────────────────────────────────────────────────────────────────
# Detecta se o terminal suporta ANSI colors (Windows: ANSICON, conemu, ou WT_SESSION)
_USE_COLOR = os.environ.get("TERM", "").lower() != "dumb" and (
    os.name != "nt"
    or "ANSICON" in os.environ
    or "WT_SESSION" in os.environ
    or "ConEmuANSI" in os.environ
    or os.environ.get("TERM") == "xterm"
)


# ANSI escape codes
class _C:
    RESET = "\033[0m" if _USE_COLOR else ""
    BOLD = "\033[1m" if _USE_COLOR else ""
    DIM = "\033[2m" if _USE_COLOR else ""
    GREEN = "\033[92m" if _USE_COLOR else ""
    YELLOW = "\033[93m" if _USE_COLOR else ""
    RED = "\033[91m" if _USE_COLOR else ""
    CYAN = "\033[96m" if _USE_COLOR else ""
    MAGENTA = "\033[95m" if _USE_COLOR else ""
    BLUE = "\033[94m" if _USE_COLOR else ""


def _mini_bar(value: float, total: float, width: int = 8) -> str:
    """Barra inline compacta: ▰▰▰▰▱▱▱▱"""
    if total <= 0:
        return "▱" * width
    pct = max(0.0, min(1.0, value / total))
    filled = int(round(pct * width))
    return "▰" * filled + "▱" * (width - filled)


def _status_dot(val: float, good: float, warn: float) -> str:
    """Semaforo: verde se >= good, amarelo se >= warn, vermelho caso contrario."""
    if val >= good:
        return f"{_C.GREEN}●{_C.RESET}"
    elif val >= warn:
        return f"{_C.YELLOW}●{_C.RESET}"
    return f"{_C.RED}●{_C.RESET}"


def _banner(title: str, char: str = "═", width: int = 70) -> str:
    """Cabecalho de secao bonito."""
    title_pad = f"  {title}  "
    return (
        f"\n{_C.CYAN}{char * width}{_C.RESET}\n"
        f"{_C.BOLD}{_C.CYAN}{title_pad.center(width, ' ')}{_C.RESET}\n"
        f"{_C.CYAN}{char * width}{_C.RESET}"
    )


def _kv_block(pairs: List[Tuple[str, str]], indent: int = 2) -> str:
    """Bloco key=value alinhado: '  latent_dim ........ 32'"""
    pad = " " * indent
    if not pairs:
        return ""
    key_w = max(len(str(k)) for k, _ in pairs)
    lines = []
    for k, v in pairs:
        dots = "." * max(2, 22 - key_w)
        lines.append(
            f"{pad}{_C.DIM}{k:<{key_w}}{_C.RESET} {_C.DIM}{dots}{_C.RESET} {_C.BOLD}{v}{_C.RESET}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET COM EMBARGO
# ─────────────────────────────────────────────────────────────────────────────
class HybridDataset(Dataset):
    """
    Retorna janela de contexto (passado) E janela alvo (futuro deslocado de k passos)
    para JEPA temporal autêntico.
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_length: int,
        regime_labels: np.ndarray,
        future_returns: np.ndarray,
        future_vols: np.ndarray,
        prediction_horizon: int = 8,
    ):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_length
        self.k = prediction_horizon
        self.n_windows = max(0, len(data) - seq_length - prediction_horizon)

        self.regime = torch.LongTensor(regime_labels)
        self.f_ret = torch.FloatTensor(future_returns)
        self.f_vol = torch.FloatTensor(future_vols)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        x_ctx = self.data[idx : idx + self.seq_len]
        future_start = idx + self.k
        x_tgt = self.data[future_start : future_start + self.seq_len]

        t_last = idx + self.seq_len - 1
        t_next = idx + self.seq_len
        regime = self.regime[t_last]
        f_ret = self.f_ret[min(t_next, len(self.f_ret) - 1)]
        f_vol = self.f_vol[min(t_next, len(self.f_vol) - 1)]
        return x_ctx, x_tgt, regime, f_ret, f_vol


# ─────────────────────────────────────────────────────────────────────────────
# BLOCOS CAUSAIS E ENCODER
# ─────────────────────────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class CausalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        latent_dim: int,
        n_conv_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, d_model, bias=False)
        self.norm_in = nn.LayerNorm(d_model)

        self.causal_convs = nn.ModuleList()
        for i in range(n_conv_layers):
            dil = 2**i
            self.causal_convs.append(
                nn.Sequential(
                    CausalConv1d(d_model, d_model, kernel=3, dilation=dil),
                    nn.GroupNorm(min(8, d_model // 8 or 1), d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )

        self.transformer = CausalTransformerLayer(d_model, n_heads, dropout)
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=False)
        self.latent_proj = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor, return_seq: bool = False) -> torch.Tensor:
        h = self.norm_in(self.proj_in(x))
        h = h.permute(0, 2, 1)
        for conv in self.causal_convs:
            h = h + conv(h)
        h = h.permute(0, 2, 1)
        h = self.transformer(h)

        if return_seq:
            return self.latent_proj(h)  # (B, T, latent_dim)

        _, h_n = self.gru(h)
        return self.latent_proj(h_n.squeeze(0))  # (B, latent_dim)


# ─────────────────────────────────────────────────────────────────────────────
# HEADS
# ─────────────────────────────────────────────────────────────────────────────
class JEPAPredictor(nn.Module):
    """
    Joint-Embedding Predictive Architecture (JEPA) — Latent Space Predictor.

    Inspirado em V-JEPA (LeCun et al.), este módulo NÃO regride variáveis
    escalares observáveis (Sharpe, retorno). Em vez disso, projeta o estado
    latente atual z_t para prever o estado latente futuro z_{t+k} no próprio
    espaço de representação.

    A perda é Cosine Similarity + L1 entre o z predito e o z alvo (extraído
    pelo Target Encoder a partir dos dados futuros reais).

    Isso elimina a necessidade de decodificação para o espaço de features
    originais, isolando o ruído de cauda gorda e permitindo planejamento
    autônomo em modo Mode-2 (RL state synthesis).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for i in range(depth):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class RegimeHead(nn.Module):
    def __init__(self, latent_dim: int, n_regimes: int = 3, dropout: float = 0.1):
        super().__init__()
        # [OVERFIT FIX] RegimeHead com regularizacao reforçada.
        # Diagnostico (TB): val_CE explodiu para >5 enquanto train_CE caiu para 0.05
        # — overfit classico do classifier head em distribution shift temporal.
        #
        # Mudancas:
        #   1. Dropout do head = max(0.30, dropout_geral) — head precisa de mais reg.
        #   2. Bottleneck reduzido (64 -> 32) — menos capacidade memorizar.
        #   3. Dropout pre-bottleneck (entrada do head) — random feature dropout.
        head_dropout = max(0.30, dropout)
        self.net = nn.Sequential(
            nn.Dropout(dropout * 0.5),  # input dropout (suave)
            nn.Linear(latent_dim, 32),  # bottleneck menor (era 64)
            nn.GELU(),
            nn.Dropout(head_dropout),  # dropout principal (>= 0.30)
            nn.Linear(32, n_regimes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─────────────────────────────────────────────────────────────────────────────
# REGULARIZATIONS & LOSSES
# ─────────────────────────────────────────────────────────────────────────────
class VICRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance Regularization (Barlow Twins/VICReg family).
    Previne dimensional collapse forçando a variância de cada dimensão a 1 e
    penalizando covariância fora da diagonal (decorrelação cruzada).
    """

    def __init__(self, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # Invariance (Similitude já tratada pelo TS2Vec/CPC, mas ajudamos aqui)
        sim_loss = F.mse_loss(z1, z2)

        # Variance Loss (Hinge loss para manter STD >= 1)
        std_z1 = torch.sqrt(z1.var(dim=0) + EPSILON)
        std_z2 = torch.sqrt(z2.var(dim=0) + EPSILON)
        var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        # Covariance Loss (Decorrelação)
        z1_c = z1 - z1.mean(dim=0)
        z2_c = z2 - z2.mean(dim=0)
        cov_z1 = (z1_c.T @ z1_c) / (z1.size(0) - 1)
        cov_z2 = (z2_c.T @ z2_c) / (z2.size(0) - 1)
        cov_loss = self._off_diagonal(cov_z1).pow_(2).sum().div(
            z1.size(1)
        ) + self._off_diagonal(cov_z2).pow_(2).sum().div(z2.size(1))

        return (
            self.sim_weight * sim_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
        )

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class UncertaintyWeightedLoss(nn.Module):
    """
    Kendall & Gal (2018) — Multi-task learning com homoscedastic uncertainty.
    L_total = sum_i [ exp(-2*log_sigma_i) * L_i / 2 + log_sigma_i ]
    SEM tarefa 'forecast' redundante.
    """

    TASK_NAMES = ["jepa", "regime", "vicreg", "ts2vec"]

    def __init__(self):
        super().__init__()
        # [G8 FIX] Inicializacao informada por escala tipica de cada loss:
        #   - JEPA cos+L1 ~ [0.5, 2.0]: sigma=1.0 (log_sigma=0)
        #   - Regime CE   ~ [0.8, 1.5]: sigma=1.0 (log_sigma=0)
        #   - VICReg      ~ [10, 50] : sigma=5.0 (log_sigma=ln(5)≈1.6)
        #   - TS2Vec NCE  ~ [4, 8]   : sigma=2.0 (log_sigma=ln(2)≈0.7)
        # Sem isto, JEPA + regime dominavam por escala numerica nos primeiros ~5 epochs,
        # ate o gradiente de log_sigma reequilibrar. Inicializacao informada acelera
        # convergencia em ~30% (homoscedastic uncertainty learning).
        _init = torch.tensor([0.0, 0.0, 1.6, 0.7])  # [jepa, regime, vicreg, ts2vec]
        self.log_sigma = nn.Parameter(_init)

    def forward(self, losses: Dict[str, torch.Tensor]):
        device = self.log_sigma.device
        total = torch.tensor(0.0, device=device)
        weights = {}
        for i, name in enumerate(self.TASK_NAMES):
            if name not in losses:
                continue
            precision = torch.exp(-2 * self.log_sigma[i])
            total = total + 0.5 * precision * losses[name] + self.log_sigma[i]
            weights[f"w_{name}"] = float((precision * 0.5).item())
        return total, weights


# ─────────────────────────────────────────────────────────────────────────────
# MODELO COMPLETO V3
# ─────────────────────────────────────────────────────────────────────────────
class HybridTemporalAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        d_model: int = 128,
        n_conv_layers: int = 3,
        n_heads_attn: int = 4,
        dropout: float = 0.1,
        n_regimes: int = N_REGIMES,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Online Encoder (Treinado ativamente)
        self.encoder = CausalEncoder(
            input_dim, d_model, latent_dim, n_conv_layers, n_heads_attn, dropout
        )

        # Target Encoder (EMA estabilizado - BYOL style para RL)
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.jepa_predictor = JEPAPredictor(
            latent_dim, hidden_dim=128, depth=2, dropout=dropout
        )
        self.regime_head = RegimeHead(latent_dim, n_regimes, dropout)
        self.vicreg_loss = VICRegLoss()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    @torch.no_grad()
    def update_target_encoder(
        self,
        current_step: int = 0,
        total_steps: int = 1,
        tau_base: float = 0.996,
        tau_final: float = 1.0,
    ):
        """
        BYOL-style EMA com schedule cosine.
        tau cresce de 0.996 (início, mais responsivo) → 1.0 (fim, congelado).
        """
        progress = min(1.0, current_step / max(total_steps, 1))
        tau = (
            tau_final - (tau_final - tau_base) * (math.cos(math.pi * progress) + 1) / 2
        )
        for online_p, target_p in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target_p.data.mul_(tau).add_(online_p.data, alpha=1 - tau)

    def ts2vec_loss(self, z1_seq: torch.Tensor, z2_seq: torch.Tensor) -> torch.Tensor:
        """Hierarchical Contrastive Loss (Avg + Max Pooling)"""
        loss = 0.0
        scales = [1, 4, 16]
        for scale in scales:
            if scale == 1:
                p1, p2 = z1_seq[:, -1, :], z2_seq[:, -1, :]
            else:
                p1_avg = F.avg_pool1d(z1_seq.transpose(1, 2), scale).transpose(1, 2)[
                    :, -1, :
                ]
                p2_avg = F.avg_pool1d(z2_seq.transpose(1, 2), scale).transpose(1, 2)[
                    :, -1, :
                ]
                p1_max = F.max_pool1d(z1_seq.transpose(1, 2), scale).transpose(1, 2)[
                    :, -1, :
                ]
                p2_max = F.max_pool1d(z2_seq.transpose(1, 2), scale).transpose(1, 2)[
                    :, -1, :
                ]
                p1, p2 = p1_avg + p1_max, p2_avg + p2_max

            p1 = F.normalize(p1, dim=-1)
            p2 = F.normalize(p2, dim=-1)
            logits = torch.matmul(p1, p2.T) / 0.1
            labels = torch.arange(p1.size(0), device=p1.device)
            loss += (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            ) / 2.0
        return loss / len(scales)

    def forward(
        self,
        x_ctx: torch.Tensor,
        x_tgt: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        JEPA temporal autêntico:
          - x_ctx: janela de contexto (passado)
          - x_tgt: janela alvo (futuro real, t+k) — usado só durante treino
        """
        z_seq_online = self.encoder(x_ctx, return_seq=True)
        z = z_seq_online[:, -1, :]

        out = {
            "z": z,
            "regime": self.regime_head(z),
        }

        if training and x_tgt is not None:
            with torch.no_grad():
                z_seq_target = self.target_encoder(x_tgt, return_seq=True)
                z_target = z_seq_target[:, -1, :]

            z_pred = self.jepa_predictor(z)

            out["jepa_pred"] = z_pred
            out["jepa_target"] = z_target
            out["vicreg"] = self.vicreg_loss(z_pred, z_target)
            out["ts2vec"] = self.ts2vec_loss(z_seq_online, z_seq_target)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS & TARGETS
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_participation_ratio(z_batch: torch.Tensor) -> float:
    if z_batch.shape[0] < 2:
        return 1.0
    c = z_batch - z_batch.mean(dim=0, keepdim=True)
    cov = torch.matmul(c.T, c) / (z_batch.size(0) - 1)
    t1 = torch.trace(cov)
    t2 = torch.trace(torch.matmul(cov, cov))
    if t2 < 1e-9:
        return 1.0
    return float((t1**2) / t2)


@torch.no_grad()
def compute_effective_rank(z_batch):
    if z_batch.shape[0] < 2:
        return 1.0
    try:
        c = z_batch - z_batch.mean(dim=0, keepdim=True)
        _, s, _ = torch.linalg.svd(c, full_matrices=False)
        p = (s + EPSILON) / (s + EPSILON).sum()
        return float(torch.exp(-(p * torch.log(p)).sum()).item())
    except Exception:
        return 1.0


@torch.no_grad()
def compute_active_units(
    z_batch: torch.Tensor,
    abs_threshold: float = 1e-2,
    rel_threshold: float = 0.05,
) -> float:
    """
    Quantifica unidades ativas (ActiveU) no espaço latente.

    Uma unidade é "ativa" se var(z_d) > MAX(abs_threshold, rel_threshold * max_var).
    O critério relativo (5% da maior variância) detecta collapse parcial mesmo
    quando todas as dims passam pelo critério absoluto absoluto frouxo —
    necessário em latentes ~N(0,1) onde var=1e-2 é trivialmente excedido.

    Returns:
        Número de unidades ativas
    """
    if z_batch.shape[0] < 2:
        return float(z_batch.shape[1])
    var_z = torch.var(z_batch, dim=0, unbiased=False)
    max_var = float(var_z.max().item())
    effective_threshold = max(abs_threshold, rel_threshold * max_var)
    return float((var_z > effective_threshold).sum().item())


@torch.no_grad()
def compute_kl_proxy(z_batch: torch.Tensor) -> float:
    """
    Proxy de divergência KL para arquiteturas determinísticas (não-VAE).
    Mede a "energia" média do espaço latente — se cai a zero, o modelo
    está se aproximando da apatia (análogo ao posterior collapse de VAE).
    """
    if z_batch.shape[0] < 2:
        return 0.0
    return float(z_batch.var(dim=0, unbiased=False).mean().item())


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE DE TREINAMENTO V3
# ─────────────────────────────────────────────────────────────────────────────
class HybridAutoencoderPipeline:
    MODEL_FILE = "hybrid_autoencoder_v3.pt"
    SCALER_FILE = "hybrid_scaler_v3.joblib"
    HPARAMS_FILE = "hybrid_hparams_v3.json"

    def __init__(self, config: AIConfig):
        self.config = config
        self.model_dir = config.MODEL_DIR
        self.model_path = os.path.join(self.model_dir, self.MODEL_FILE)
        self.scaler_path = os.path.join(self.model_dir, self.SCALER_FILE)
        self.hparams_path = os.path.join(self.model_dir, self.HPARAMS_FILE)
        self.optuna_db_path = os.path.join(self.model_dir, "ae_optuna_hpo.db")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[HybridTemporalAutoencoder] = None
        self.scaler: Optional[RobustScaler] = None
        self.hparams: Dict[str, Any] = {}
        self.feature_cols: List[str] = []

        self._load_if_exists()

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    @property
    def autoencoder(self):
        return self.model

    @property
    def hyperparams(self):
        return self.hparams

    @property
    def is_optimized(self) -> bool:
        import json

        if not os.path.exists(self.optuna_db_path):
            return False
        try:
            study_summaries = optuna.study.get_all_study_summaries(
                storage=f"sqlite:///{self.optuna_db_path}"
            )
            # [B3 FIX] Verifica se pelo menos 1 trial completou com sucesso,
            # em vez de hardcodar n_trials >= 100 (que falhava para HPO com n_trials < 100).
            return any(
                s.n_trials >= 1 and s.best_trial is not None for s in study_summaries
            )
        except Exception:
            return False

    def _get_completed_trials(self) -> int:
        try:
            study_summaries = optuna.study.get_all_study_summaries(
                storage=f"sqlite:///{self.optuna_db_path}"
            )
            return max((s.n_trials for s in study_summaries), default=0)
        except Exception:
            return 0

    def load_state(self):
        """Alias para compatibilidade legada"""
        self._load_if_exists()

    def _load_if_exists(self):
        import json

        try:
            if os.path.exists(self.hparams_path):
                with open(self.hparams_path) as f:
                    self.hparams = json.load(f)
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            if os.path.exists(self.model_path) and self.hparams:
                self.model = self._build_model(self.hparams)
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
                self.model.to(self.device).eval()
                self.model = self._compile_for_training(self.model)
                logger.info("[HybridAE_v3] Modelo compilado para inferência rápida")
        except Exception as e:
            logger.warning(f"[HybridAE_v3] Falha ao carregar: {e}")
            self.model = None

    def optimize_hyperparameters(self, df: pd.DataFrame, n_trials: int = 50) -> bool:
        completed = self._get_completed_trials()
        if completed >= n_trials:
            logger.info(
                f"[HybridAE_v3] HPO já tem {completed} trials. Carregando best params..."
            )
            self._load_best_hpo_params()
            return True

        remaining = n_trials - completed
        logger.info(
            f"[HybridAE_v3] HPO: {completed}/{n_trials} trials. Rodando +{remaining}..."
        )

        feature_cols = [c for c in self._filter_features(df) if c in df.columns]
        df_feat = (
            df[feature_cols]
            .copy()
            .replace([np.inf, -np.inf], np.nan)
            .ffill(limit=300)
            .dropna()
        )

        # [PERF] Subsample 50% mais recente para HPO — reduz tempo por trial em ~50%.
        # Hiperparâmetros ótimos (latent_dim, d_model, seq_len) são estáveis com ±20k bars.
        # O treino final usa 100% do dataset. Usar tail() garante que regimes recentes
        # (mais relevantes para produção) estejam representados.
        _hpo_max_samples = max(20000, int(len(df_feat) * 0.50))
        if len(df_feat) > _hpo_max_samples:
            logger.info(
                f"[HybridAE_v3] HPO subsample: usando últimos {_hpo_max_samples:,} de "
                f"{len(df_feat):,} bars (50% mais recente) para acelerar busca."
            )
            df_feat = df_feat.tail(_hpo_max_samples).copy()

        split = int(0.8 * len(df_feat))
        scaler = RobustScaler()
        # Fit no treino, transform no dataset COMPLETO
        scaler.fit(df_feat.iloc[:split].values)
        data = scaler.transform(df_feat.values).astype(np.float32)

        regime_labels = (
            df.loc[df_feat.index]
            .get("regime", pd.Series(np.full(len(df_feat), 2), index=df_feat.index))
            .fillna(2)
            .astype(int)
            .values
        )

        # Verifica se regime_labels tem variância. Se não, deriva via retornos.
        # NUNCA gera labels aleatorios (np.random.choice) — treinaria o RegimeHead
        # em ruido puro e o classificador retornaria confianca alta em padroes
        # inexistentes, contaminando downstream (SAC consome prob_bull/bear/ranger).
        unique_regimes = np.unique(regime_labels)
        if len(unique_regimes) < 2:
            close_col = "close" if "close" in df.columns else None
            if close_col is None:
                raise ValueError(
                    "[HybridAE_v3] regime_labels degenerados e coluna 'close' ausente. "
                    "Impossivel derivar regimes — abortando HPO para evitar treino em ruido."
                )
            logger.warning(
                f"[HybridAE_v3] regime_labels tem apenas classe {unique_regimes}. "
                "Derivando labels via retornos (z-score do log-return)."
            )
            close = df.loc[df_feat.index, close_col].values
            log_ret = np.zeros(len(close))
            log_ret[1:] = np.diff(np.log(np.maximum(close, 1e-9)))
            # bfill aqui afeta APENAS os primeiros 19 candles (warmup do rolling).
            # E aceitavel porque (1) sao labels de TREINO fallback, nao previsao,
            # (2) o regime detector real ja falhou, estamos em modo degradado.
            vol = pd.Series(log_ret).rolling(20).std().bfill().fillna(1e-4).values
            vol = np.maximum(vol, 1e-6)
            # Bull/Bear/Ranger via z-score: |z|>0.5 -> direcional, senao Ranger
            regime_labels = np.where(
                log_ret > 0.5 * vol,
                0,
                np.where(log_ret < -0.5 * vol, 1, 2),
            ).astype(np.int64)

        f_sharpe, f_drawdown = self._compute_forward_targets(df.loc[df_feat.index])

        # [G2 FIX + STABILITY] HyperbandPruner com max_resource=50 (era 30).
        # Schedule de rungs com max=50 e reduction=3: 10 → 18 → 50 (3 rungs).
        # Top 1/3 dos trials promissores recebem 50 epochs — tempo suficiente para
        # demonstrar convergencia COMPLETA e o gap penalty detectar overfit no fim.
        # Trials ruins continuam sendo cortados em 10 epochs (custo extra marginal).
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=10,  # Minimo 10 epochs antes de poder (BYOL tau converge ~10-15 epochs)
            max_resource=50,  # Maximo 50 epochs por trial — top trials demonstram convergencia
            reduction_factor=3,  # Mantem top 1/3 a cada rung
        )

        # Limpa estudo anterior se contém trials corrompidos (bug anterior)
        _should_delete_study = False
        try:
            _existing = optuna.load_study(
                study_name="ae_hpo_v3",
                storage=f"sqlite:///{self.optuna_db_path}",
            )
            # Caso 1: best_value absurdo (score inflado por val_loss=0)
            if _existing.best_trial and _existing.best_value > 10000:
                _should_delete_study = True
                logger.warning(
                    f"[HybridAE_v3] Estudo com score absurdo ({_existing.best_value:.0f}). Deletando."
                )
            # Caso 2: Trials com intermediate values corrompidos (scores > 10000)
            if not _should_delete_study:
                for t in _existing.trials:
                    for step, val in t.intermediate_values.items():
                        if val > 10000:
                            _should_delete_study = True
                            logger.warning(
                                f"[HybridAE_v3] Trial {t.number} tem intermediate value absurdo "
                                f"({val:.0f} no step {step}). Deletando estudo."
                            )
                            break
                    if _should_delete_study:
                        break
            # Caso 3: Todos os trials existentes foram podados (nenhum completou)
            # e o input_dim mudou (features diferentes = estudo incompatível)
            if not _should_delete_study and len(_existing.trials) > 0:
                completed = [
                    t
                    for t in _existing.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                if not completed:
                    # Nenhum trial completou — estudo provavelmente corrompido
                    _should_delete_study = True
                    logger.warning(
                        f"[HybridAE_v3] Estudo com {len(_existing.trials)} trials mas nenhum completo. Deletando."
                    )

            if _should_delete_study:
                optuna.delete_study(
                    study_name="ae_hpo_v3",
                    storage=f"sqlite:///{self.optuna_db_path}",
                )
                logger.info("[HybridAE_v3] Estudo deletado. Recomeçando limpo.")
        except Exception:
            pass

        # [G7 FIX] Seed fixo no TPESampler para reprodutibilidade do HPO.
        # Mantem o stochastic do treino RL, mas a busca de hiperparametros e deterministica.
        study = optuna.create_study(
            direction="maximize",
            storage=f"sqlite:///{self.optuna_db_path}",
            study_name="ae_hpo_v3",
            load_if_exists=True,
            pruner=pruner,
            sampler=TPESampler(seed=42, n_startup_trials=15),
        )

        input_dim = data.shape[1]

        def objective(trial: optuna.Trial) -> float:
            # [G5 FIX] Range estendido para incluir contextos longos (24h em 15m = 96 candles).
            # Literatura JEPA quant: dependencias temporais de 12-48h sao cruciais para regime detection.
            hpo_seq = trial.suggest_categorical("seq_len", [16, 24, 32, 48, 64, 96])
            embargo = hpo_seq * 2
            hpo_train_d = data[: split - embargo]
            hpo_val_d = data[split:]
            hpo_train_r = regime_labels[: split - embargo]
            hpo_val_r = regime_labels[split:]
            hpo_tr_sh, hpo_v_sh = f_sharpe[: split - embargo], f_sharpe[split:]
            hpo_tr_dd, hpo_v_dd = f_drawdown[: split - embargo], f_drawdown[split:]

            # [G6 FIX] prediction_horizon agora otimizado pelo HPO.
            # Para 15m: horizon=4 (1h) a 24 (6h) cobre desde scalp ate swing.
            hpo_pred_horizon = trial.suggest_categorical(
                "prediction_horizon", [4, 8, 16, 24]
            )

            ds_tr = HybridDataset(
                hpo_train_d,
                hpo_seq,
                hpo_train_r,
                hpo_tr_sh,
                hpo_tr_dd,
                prediction_horizon=hpo_pred_horizon,
            )
            ds_vl = HybridDataset(
                hpo_val_d,
                hpo_seq,
                hpo_val_r,
                hpo_v_sh,
                hpo_v_dd,
                prediction_horizon=hpo_pred_horizon,
            )
            # [G1 FIX] batch_size agora otimizado pelo HPO.
            # Contrastive TS2Vec + VICReg sao instaveis com batch <128; 256 da gradientes
            # mais limpos mas dobra VRAM. Range deixa o Optuna escolher.
            hpo_batch = trial.suggest_categorical("batch_size", [128, 256])
            dl_tr = DataLoader(ds_tr, batch_size=hpo_batch, shuffle=True)
            dl_vl = DataLoader(ds_vl, batch_size=hpo_batch, shuffle=False)

            hpo_hp = {
                "input_dim": input_dim,
                "latent_dim": trial.suggest_categorical("latent_dim", [16, 24, 32, 48]),
                "d_model": trial.suggest_categorical("d_model", [64, 96, 128, 160]),
                "n_conv_layers": trial.suggest_int("n_conv_layers", 2, 4),
                "n_heads_attn": trial.suggest_categorical("n_heads_attn", [2, 4, 8]),
                "dropout": trial.suggest_float("dropout", 0.05, 0.25),
                "seq_len": hpo_seq,
                "batch_size": hpo_batch,
                "prediction_horizon": hpo_pred_horizon,
                "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
            }
            model = self._build_model(hpo_hp).to(self.device)
            loss_fn = UncertaintyWeightedLoss().to(self.device)

            # [G2 FIX + STABILITY] max_epochs=50 (alinhado com Hyperband max_resource).
            # Trials promissores rodam ate 50 epochs — tempo suficiente para
            # demonstrar convergencia E para o gap_penalty detectar overfit no fim.
            # Trials ruins continuam podados em 10-18 epochs (rungs 1 e 2 do Hyperband).
            best_val, pr, er, au, kl, val_degrad, gap_max = self._train_one(
                model,
                loss_fn,
                dl_tr,
                dl_vl,
                lr=hpo_hp["lr"],
                max_epochs=50,
                trial=trial,
            )

            # [G4 FIX] Score re-balanceado com penalidade de COLLAPSE.
            # Score anterior: 1/val_loss dominava (val=0.1 → 10.0; val=0.5 → 2.0).
            # Trials colapsados (AU=0, eR=1) podiam ter val_loss baixo e ser selecionados.
            # Agora: AU/latent_ratio < 30% aplica multiplicador de 0.1 (mata o score).
            latent_dim = hpo_hp["latent_dim"]
            au_ratio = au / max(latent_dim, 1)
            er_ratio = er / max(latent_dim, 1)

            # [GAP PENALTY] Calcula multiplicador de estabilidade do val.
            # val_degrad = avg(ultimos 3 val) - best_val. Quanto val piorou no fim.
            #   < 0.05 → modelo estavel, multiplier=1.0
            #   0.05-0.20 → degradacao moderada, multiplier=0.8
            #   0.20-0.50 → degradacao severa, multiplier=0.5
            #   > 0.50 → val explodiu (caso do run anterior), multiplier=0.2
            if val_degrad < 0.05:
                stability_mult = 1.0
            elif val_degrad < 0.20:
                stability_mult = 0.8
            elif val_degrad < 0.50:
                stability_mult = 0.5
            else:
                stability_mult = 0.2

            if best_val < 1e-4:
                score = 0.1  # Loss trivial = modelo nao aprendeu
            elif au_ratio < 0.30 or er_ratio < 0.20:
                # COLLAPSE: AU < 30% das dims ativas ou eR < 20% rank efetivo
                # Penaliza fortemente — nao queremos hiperparametros que produzam latent morto
                base_score = (1.0 / max(best_val, 0.01)) + 0.3 * pr + 0.2 * er
                score = float(base_score * 0.1)  # Multiplica por 0.1 (90% penalty)
                trial.set_user_attr("collapse_detected", True)
            else:
                # Score saudavel: val_loss inverso + bonus de qualidade latente
                # Adiciona termos absolutos de AU/eR (em ratio) para evitar over-reliance em val_loss
                base_score = (
                    (1.0 / max(best_val, 0.01))
                    + 0.3 * pr
                    + 0.2 * er
                    + 2.0 * au_ratio  # bonus direto por unidades ativas
                    + 1.5 * er_ratio  # bonus direto por rank efetivo
                )
                # [GAP PENALTY] Aplica multiplier de estabilidade: trials com val
                # explodindo no fim sao penalizados mesmo com best_val baixo.
                score = float(base_score * stability_mult)
                trial.set_user_attr("collapse_detected", False)

            trial.set_user_attr("val_loss", float(best_val))
            trial.set_user_attr("val_degradation", float(val_degrad))
            trial.set_user_attr("train_val_gap_max", float(gap_max))
            trial.set_user_attr("stability_mult", float(stability_mult))
            trial.set_user_attr("PR", float(pr))
            trial.set_user_attr("eRank", float(er))
            trial.set_user_attr("AU", float(au))
            trial.set_user_attr("AU_ratio", float(au_ratio))
            trial.set_user_attr("eR_ratio", float(er_ratio))
            return score

        # Banner de inicio do HPO
        print(_banner("🧪 HYBRID AUTOENCODER — HYPERPARAMETER OPTIMIZATION", width=70))
        print(
            _kv_block(
                [
                    ("Trials alvo", f"{n_trials} ({remaining} restantes)"),
                    ("Sampler", "TPESampler (seed=42, startup=15)"),
                    ("Pruner", "HyperbandPruner (min=10, max=50, η=3)"),
                    (
                        "Search space",
                        "latent∈{16,24,32,48} │ d_model∈{64,96,128,160} │ seq∈{16..96}",
                    ),
                    ("Anti-collapse", "AU<30% ou eR<20% → score × 0.1 (G4 FIX)"),
                ]
            )
        )
        print()

        try:
            from tqdm import tqdm

            pbar = tqdm(
                total=remaining,
                desc=f"{_C.MAGENTA}{_C.BOLD}  ▶ AE HPO    {_C.RESET}",
                unit="t",
                dynamic_ncols=True,
                bar_format=(
                    "{desc} {percentage:3.0f}% "
                    + _C.BLUE
                    + "{bar:18}"
                    + _C.RESET
                    + _C.DIM
                    + " {n_fmt}/{total_fmt}"
                    + _C.RESET
                    + _C.DIM
                    + " [{elapsed}<{remaining}, {rate_fmt}]"
                    + _C.RESET
                    + " {postfix}"
                ),
                ascii=False,
                colour="magenta",
            )
            best_score = float("-inf")
            best_trial_num = -1
            stats = {"complete": 0, "pruned": 0, "collapse": 0}

            def _hpo_cb(s, t):
                nonlocal best_score, best_trial_num
                pbar.update(1)
                val = t.value if t.value is not None else float("nan")
                if val is not None and not np.isnan(val) and val > best_score:
                    best_score = val
                    best_trial_num = t.number

                state = t.state.name if hasattr(t, "state") else ""
                is_pruned = state == "PRUNED"
                collapsed = t.user_attrs.get("collapse_detected", False)

                if is_pruned:
                    stats["pruned"] += 1
                    last_step = (
                        max(t.intermediate_values.keys())
                        if t.intermediate_values
                        else 0
                    )
                    last_score = t.intermediate_values.get(last_step, 0)
                    pbar.set_postfix_str(
                        f"{_C.YELLOW}✂ pruned{_C.RESET} ep{last_step + 1} "
                        f"score={last_score:4.1f} │ best={_C.BOLD}{best_score:5.2f}{_C.RESET}"
                        f" (t{best_trial_num})"
                    )
                elif collapsed:
                    stats["collapse"] += 1
                    pbar.set_postfix_str(
                        f"{_C.RED}✗ collapse{_C.RESET} score={val:4.1f} │ "
                        f"best={_C.BOLD}{best_score:5.2f}{_C.RESET} (t{best_trial_num})"
                    )
                else:
                    stats["complete"] += 1
                    pr_val = t.user_attrs.get("PR", 0)
                    er_val = t.user_attrs.get("eRank", 0)
                    au_val = t.user_attrs.get("AU", 0)
                    vl_val = t.user_attrs.get("val_loss", 0)
                    latent = t.params.get("latent_dim", 32)

                    dot_pr = _status_dot(pr_val, 3.0, 1.5)
                    dot_er = _status_dot(er_val / max(latent, 1), 0.30, 0.15)
                    dot_au = _status_dot(au_val / max(latent, 1), 0.40, 0.20)

                    # [GAP PENALTY] Mostra degradacao do val (overfit signature)
                    degrad = t.user_attrs.get("val_degradation", 0.0)
                    stab = t.user_attrs.get("stability_mult", 1.0)
                    if stab < 1.0:
                        stab_icon = (
                            f"{_C.YELLOW}⚠{_C.RESET}"
                            if stab >= 0.5
                            else f"{_C.RED}⚠{_C.RESET}"
                        )
                    else:
                        stab_icon = f"{_C.GREEN}●{_C.RESET}"

                    pbar.set_postfix_str(
                        f"{_C.GREEN}✓{_C.RESET} score={_C.BOLD}{val:5.2f}{_C.RESET} │ "
                        f"PR{dot_pr}{pr_val:4.1f} │ eR{dot_er}{er_val:4.1f} │ "
                        f"AU{dot_au}{int(au_val):2d}/{latent} │ "
                        f"val={vl_val:4.2f} │ degrad{stab_icon}{degrad:+.2f} │ "
                        f"best={_C.BOLD}{best_score:5.2f}{_C.RESET} (t{best_trial_num})"
                    )

            study.optimize(objective, n_trials=remaining, callbacks=[_hpo_cb])
            pbar.close()
        except ImportError:
            study.optimize(objective, n_trials=remaining)
            stats = {"complete": len(study.trials), "pruned": 0, "collapse": 0}
            best_trial_num = study.best_trial.number if study.best_trial else -1

        self._load_best_hpo_params()

        # Banner de conclusao do HPO com sumario
        print(_banner("✓ HPO CONCLUÍDO", char="─", width=70))
        _best_params = study.best_params if study.best_trial else {}
        print(
            _kv_block(
                [
                    (
                        "Best score",
                        f"{_C.BOLD}{_C.GREEN}{study.best_value:.3f}{_C.RESET} (trial #{study.best_trial.number if study.best_trial else 'N/A'})",
                    ),
                    ("Trials totais", f"{len(study.trials)}"),
                    ("├─ completos", f"{_C.GREEN}{stats['complete']}{_C.RESET}"),
                    ("├─ podados", f"{_C.YELLOW}{stats['pruned']}{_C.RESET}"),
                    ("└─ colapsados", f"{_C.RED}{stats['collapse']}{_C.RESET}"),
                    ("latent_dim", f"{_best_params.get('latent_dim', '?')}"),
                    ("d_model", f"{_best_params.get('d_model', '?')}"),
                    ("seq_len", f"{_best_params.get('seq_len', '?')}"),
                    ("batch_size", f"{_best_params.get('batch_size', '?')}"),
                    (
                        "prediction_horizon",
                        f"{_best_params.get('prediction_horizon', '?')}",
                    ),
                    ("lr", f"{_best_params.get('lr', 0):.5f}"),
                ]
            )
        )
        print()

        logger.info(
            f"[HybridAE_v3] HPO concluido. best={study.best_value:.3f} "
            f"(t#{study.best_trial.number if study.best_trial else '?'}, "
            f"{stats['complete']}/{len(study.trials)} completos)"
        )
        return True

    def _load_best_hpo_params(self):
        try:
            study = optuna.load_study(
                study_name="ae_hpo_v3",
                storage=f"sqlite:///{self.optuna_db_path}",
            )
            if study.best_trial:
                best = study.best_params
                self.hparams.update(
                    {
                        "latent_dim": best.get("latent_dim", 32),
                        "d_model": best.get("d_model", 128),
                        "n_conv_layers": best.get("n_conv_layers", 3),
                        "n_heads_attn": best.get("n_heads_attn", 4),
                        "dropout": best.get("dropout", 0.1),
                        "seq_len": best.get("seq_len", 48),
                        # [G1/G6 FIX] Persiste batch_size e prediction_horizon otimizados
                        "batch_size": best.get("batch_size", 128),
                        "prediction_horizon": best.get("prediction_horizon", 8),
                        "lr": best.get("lr", 1e-3),
                    }
                )
                self._save_hparams_only()
                logger.info(
                    "[HybridAE_v3] Best HPO params carregados: "
                    f"latent={self.hparams['latent_dim']}, "
                    f"d_model={self.hparams['d_model']}, "
                    f"seq_len={self.hparams['seq_len']}, "
                    f"batch={self.hparams['batch_size']}, "
                    f"horizon={self.hparams['prediction_horizon']}, "
                    f"lr={self.hparams['lr']:.5f}"
                )
        except Exception as e:
            logger.warning(f"[HybridAE_v3] Falha ao carregar best HPO: {e}")

    def _save(self):
        import json

        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        with open(self.hparams_path, "w") as f:
            json.dump(self.hparams, f, indent=2)

    def _save_hparams_only(self):
        import json

        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.hparams_path, "w") as f:
            json.dump(self.hparams, f, indent=2)

    def _build_model(self, hp: Dict) -> HybridTemporalAutoencoder:
        return HybridTemporalAutoencoder(
            input_dim=hp["input_dim"],
            latent_dim=hp["latent_dim"],
            d_model=hp["d_model"],
            n_conv_layers=hp.get("n_conv_layers", 3),
            n_heads_attn=hp.get("n_heads_attn", 4),
            dropout=hp["dropout"],
        )

    @staticmethod
    def _compile_for_training(model: nn.Module) -> nn.Module:
        try:
            if hasattr(torch, "compile") and torch.cuda.is_available():
                import torch._dynamo

                torch._dynamo.config.suppress_errors = True
                return torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception:
            pass
        return model

    @staticmethod
    def _filter_features(df: pd.DataFrame) -> List[str]:
        """
        Seleciona features para o autoencoder aprender representações latentes.

        Estratégia: O autoencoder deve receber informação COMPLEMENTAR ao que
        os agentes SAC já veem diretamente. Os agentes recebem indicadores _15m e _1h.
        O autoencoder recebe:
          - OHLCV normalizado (sinal bruto que indicadores derivam)
          - TODOS os timeframes (1m, 5m, 15m, 1h, 4h) — comprime multi-scale
          - Indicadores que os agentes NÃO recebem (williams_r, cci, roc, etc.)
          - Microestrutura (volume_pressure, money_flow, etc.)

        O autoencoder NÃO recebe:
          - regime/regime_confidence (é o que ele deve PREDIZER)
          - hidden_feature_* / sdae_* (output dele mesmo — evita loop)
          - tp_prior_* (meta-label, não é feature de mercado)
        """
        # Excluir: outputs do próprio autoencoder, regime labels, meta-labels
        skip_prefixes = (
            "hidden_feature",
            "sdae_",
            "tp_prior",
            "prob_bull",
            "prob_bear",
            "prob_ranger",
        )
        skip_exact = {"regime", "regime_confidence", "regime_val", "regime_code"}

        candidates = []
        for c in df.select_dtypes(include=[np.number]).columns:
            cl = c.lower()
            if c in skip_exact:
                continue
            if any(cl.startswith(p) for p in skip_prefixes):
                continue
            candidates.append(c)

        # Incluir OHLCV (normalizado pelo RobustScaler depois)
        ohlcv = ["open", "high", "low", "close", "volume"]
        for col in ohlcv:
            if col in df.columns and col not in candidates:
                candidates.append(col)

        return candidates

    @staticmethod
    def _compute_forward_targets(
        df: pd.DataFrame, close_col: str = "close", horizon: int = 5
    ):
        """
        Risk-adjusted Return e Max Drawdown (vetorizado).

        [M3 FIX] Substituído loop Python O(N) por sliding_window_view + operações numpy.
        Para 100k bars: ~0.05s vs ~3s anterior.
        """
        close = df[close_col].values if close_col in df.columns else np.ones(len(df))
        n = len(df)
        log_ret = np.zeros(n)
        log_ret[:-1] = np.diff(np.log(np.maximum(close, 1e-9)))

        fwd_sharpe = np.zeros(n)
        fwd_drawdown = np.zeros(n)

        if n <= horizon:
            return fwd_sharpe, fwd_drawdown

        # Sliding windows de log_ret[i+1 : i+1+horizon] para i in [0, n-horizon-1]
        # Offset: queremos janelas começando em index 1, 2, ..., n-horizon
        ret_windows = np.lib.stride_tricks.sliding_window_view(log_ret[1:], horizon)
        # ret_windows[i] = log_ret[i+1 : i+1+horizon], shape (n-horizon, horizon)
        valid_n = min(len(ret_windows), n - 1)

        window_sum = ret_windows[:valid_n].sum(axis=1)
        window_std = ret_windows[:valid_n].std(axis=1, ddof=0) + 1e-9
        fwd_sharpe[:valid_n] = np.clip(window_sum / window_std, -5.0, 5.0)

        # Max drawdown vetorizado: para cada janela, calcula peak cumulativo e dd
        price_windows = np.lib.stride_tricks.sliding_window_view(close[1:], horizon)
        valid_p = min(len(price_windows), n - 1)
        # cummax ao longo do eixo 1 (dentro de cada janela)
        peaks = np.maximum.accumulate(price_windows[:valid_p], axis=1)
        dd = (price_windows[:valid_p] - peaks) / (peaks + 1e-9)
        fwd_drawdown[:valid_p] = dd.min(axis=1)

        return fwd_sharpe, fwd_drawdown

    def _train_one(
        self,
        model: HybridTemporalAutoencoder,
        loss_fn: UncertaintyWeightedLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float,
        max_epochs: int,
        trial=None,
    ):
        params = list(model.parameters()) + list(loss_fn.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr * 0.01)
        scaler_amp = GradScaler("cuda", enabled=(self.device.type == "cuda"))

        total_steps = max(1, len(train_loader) * max_epochs)
        global_step = 0

        best_val, best_state = float("inf"), None
        pr, er, au, kl_proxy = 1.0, 1.0, 0.0, 0.0

        # [OVERFIT FIX] Early stopping com patience.
        # Diagnostico (TB): val_CE explodiu de 0 para >5 nos ultimos epochs
        # enquanto o codigo rodava todos os 100 epochs sem parar. best_state era
        # salvo no minimo, mas o treino DESPERDIcAVA computacao apos val piorar.
        # HPO trials: patience=5 (corta cedo, ja tem MedianPruner).
        # Treino final: patience=15 (mais tolerante, da chance de re-convergir).
        early_stop_patience = 5 if trial is not None else 15
        patience_counter = 0

        # [GAP PENALTY] Tracking de estabilidade do val para deteccao de overfit no HPO.
        # Diagnostico: HPO antigo escolheu latent=48 + dropout=0.10 + lr=0.0021
        # porque o score so via best_val_loss (minimo absoluto). Esses hparams tinham
        # val_loss minimo bom (~0.39) MAS depois explodiam para 5+. Score "otimo"
        # mascarava overfit catastrofico que so se manifestava no treino final.
        # Solucao: trackear avg_val_loss dos ULTIMOS 3 epochs e penalizar degradacao.
        final_val_window = []  # ultimos 3 val_loss para media movel
        train_val_gap_max = 0.0  # gap maximo train-val observado

        # TensorBoard logging — APENAS no treino final (trial is None).
        # HPO gera centenas de trials curtos que produzem logs inúteis e poluem o disco.
        tb_writer = None
        if trial is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_log_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "logs",
                    "tensorboard",
                    "HybridAutoencoder",
                )
                from datetime import datetime as _dt

                tb_run_dir = os.path.join(
                    tb_log_dir, f"train_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
                )
                os.makedirs(tb_run_dir, exist_ok=True)
                tb_writer = SummaryWriter(tb_run_dir)
                logger.info(f"📊 TensorBoard: {tb_run_dir}")
            except ImportError:
                pass

        # Barra de progresso por epoch (sempre ativa — HPO e treino final)
        epoch_pbar = None
        try:
            from tqdm import tqdm

            if trial is not None:
                desc = f"{_C.DIM}  ↳ Trial {_C.BOLD}{trial.number:>3}{_C.RESET}{_C.DIM} ▸{_C.RESET}"
            else:
                desc = f"{_C.MAGENTA}{_C.BOLD}  ▶ AE Train  {_C.RESET}"
            epoch_pbar = tqdm(
                total=max_epochs,
                desc=desc,
                unit="ep",
                dynamic_ncols=True,
                # Layout limpo: descrição | barra colorida | epoch/total | tempo | métricas
                bar_format=(
                    "{desc} {percentage:3.0f}% "
                    + _C.BLUE
                    + "{bar:18}"
                    + _C.RESET
                    + _C.DIM
                    + " {n_fmt}/{total_fmt}"
                    + _C.RESET
                    + _C.DIM
                    + " [{elapsed}<{remaining}]"
                    + _C.RESET
                    + " {postfix}"
                ),
                leave=(trial is None),  # HPO: nao deixa barra residual
                ascii=False,
                colour="cyan",
            )
        except ImportError:
            pass

        for epoch in range(max_epochs):
            model.train()
            loss_fn.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x_ctx, x_tgt, regime, _f_ret, _f_vol = [
                    b.to(self.device) for b in batch
                ]
                opt.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=(self.device.type == "cuda")):
                    out = model(x_ctx, x_tgt=x_tgt, training=True)

                    jepa_cos = (
                        1
                        - F.cosine_similarity(
                            out["jepa_pred"], out["jepa_target"], dim=-1
                        ).mean()
                    )
                    jepa_l1 = F.smooth_l1_loss(out["jepa_pred"], out["jepa_target"])
                    jepa_loss = jepa_cos + 0.5 * jepa_l1

                    task_losses = {
                        "jepa": jepa_loss,
                        # [OVERFIT FIX] label_smoothing=0.1: reduz overconfidence do
                        # RegimeHead (Szegedy et al. 2016). Antes: train_CE colapsava
                        # para ~0.05 em ~30 epochs (overfit). Smoothing converte one-hot
                        # [0,1,0] em [0.033, 0.933, 0.033] -> CE minimo teorico ~0.34,
                        # forcando o modelo a manter calibracao em vez de memorizar.
                        "regime": F.cross_entropy(
                            out["regime"], regime, label_smoothing=0.1
                        ),
                        "vicreg": out["vicreg"],
                        "ts2vec": out["ts2vec"],
                    }
                    total, _w = loss_fn(task_losses)

                scaler_amp.scale(total).backward()
                scaler_amp.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler_amp.step(opt)
                scaler_amp.update()
                epoch_loss += total.item()
                n_batches += 1

                # TensorBoard: losses individuais por step (a cada 10 batches para não sobrecarregar)
                if tb_writer is not None and global_step % 10 == 0:
                    tb_writer.add_scalar(
                        "Loss_detail/jepa", jepa_loss.item(), global_step
                    )
                    tb_writer.add_scalar(
                        "Loss_detail/regime_CE",
                        task_losses["regime"].item(),
                        global_step,
                    )
                    tb_writer.add_scalar(
                        "Loss_detail/vicreg", task_losses["vicreg"].item(), global_step
                    )
                    tb_writer.add_scalar(
                        "Loss_detail/ts2vec", task_losses["ts2vec"].item(), global_step
                    )
                    tb_writer.add_scalar("Loss_detail/total", total.item(), global_step)

                model.update_target_encoder(global_step, total_steps)
                global_step += 1

            sched.step()
            avg_train_loss = epoch_loss / max(n_batches, 1)

            model.eval()
            loss_fn.eval()
            val_loss = 0.0
            all_z = []

            with torch.no_grad():
                for batch in val_loader:
                    x_ctx, _x_tgt, regime, _f_ret, _f_vol = [
                        b.to(self.device) for b in batch
                    ]
                    out = model(x_ctx, x_tgt=None, training=False)
                    # [OVERFIT FIX] Mesmo label_smoothing do train para metrica consistente
                    val_loss += F.cross_entropy(
                        out["regime"], regime, label_smoothing=0.1
                    ).item()
                    all_z.append(out["z"].cpu())

            val_loss /= max(len(val_loader), 1)

            # [GAP PENALTY] Track media movel dos ultimos 3 val_loss e gap train-val.
            # Detecta degradacao de val mesmo quando best_val_loss continua baixo
            # (caso classico do overfit: val cai cedo, sobe depois — best fica baixo).
            final_val_window.append(float(val_loss))
            if len(final_val_window) > 3:
                final_val_window.pop(0)
            # Train-val gap (overfit literal): se train cai e val nao acompanha
            _gap = float(val_loss - avg_train_loss)
            if _gap > train_val_gap_max:
                train_val_gap_max = _gap

            z_all = torch.cat(all_z) if all_z else torch.zeros(2, model.latent_dim)
            pr = compute_participation_ratio(z_all)
            er = compute_effective_rank(z_all)
            au = compute_active_units(z_all)
            kl_proxy = float(z_all.var(dim=0).mean().item())

            # TensorBoard: loga métricas por epoch
            if tb_writer is not None:
                tb_writer.add_scalar("Loss/train", avg_train_loss, epoch)
                tb_writer.add_scalar("Loss/val_CE", val_loss, epoch)
                tb_writer.add_scalar("Health/participation_ratio", pr, epoch)
                tb_writer.add_scalar("Health/effective_rank", er, epoch)
                tb_writer.add_scalar("Health/active_units", au, epoch)
                tb_writer.add_scalar("Health/variance_z", kl_proxy, epoch)
                tb_writer.add_scalar(
                    "Health/eRank_ratio", er / max(model.latent_dim, 1), epoch
                )
                tb_writer.add_scalar(
                    "Health/AU_ratio", au / max(model.latent_dim, 1), epoch
                )
                tb_writer.add_scalar("LR/current", sched.get_last_lr()[0], epoch)

            # Semaforos por metrica com cores ANSI + barras mini-inline
            latent_dim = model.latent_dim
            er_ratio = er / max(latent_dim, 1)
            au_ratio = au / max(latent_dim, 1)

            dot_er = _status_dot(er_ratio, 0.30, 0.15)
            dot_au = _status_dot(au_ratio, 0.40, 0.20)
            dot_vz = _status_dot(kl_proxy, 1e-3, 1e-4)
            dot_loss = (
                _status_dot(1.0 / max(val_loss, 0.01), 1.0, 0.5)
                if val_loss > 0
                else _C.RED + "●" + _C.RESET
            )

            # Mini-barras de saude latente (compactas, 4 chars cada)
            bar_er = _mini_bar(er, latent_dim, width=4)
            bar_au = _mini_bar(au, latent_dim, width=4)

            # Atualiza barra de progresso com layout limpo
            if epoch_pbar is not None:
                epoch_pbar.update(1)
                epoch_pbar.set_postfix_str(
                    f"loss={_C.BOLD}{avg_train_loss:5.3f}{_C.RESET} │ "
                    f"valCE={dot_loss}{_C.BOLD}{val_loss:5.3f}{_C.RESET} │ "
                    f"eRank{dot_er}{bar_er}{_C.DIM}{er:4.1f}/{latent_dim}{_C.RESET} │ "
                    f"AU{dot_au}{bar_au}{_C.DIM}{int(au):2d}/{latent_dim}{_C.RESET} │ "
                    f"PR={pr:4.1f} │ VarZ{dot_vz}{kl_proxy:.3g}"
                )
            else:
                logger.info(
                    f"  Ep {epoch + 1:3d}/{max_epochs} │ "
                    f"loss={avg_train_loss:5.3f} │ valCE={dot_loss}{val_loss:5.3f} │ "
                    f"eRank{dot_er}{bar_er}{er:4.1f}/{latent_dim} │ "
                    f"AU{dot_au}{bar_au}{int(au):2d}/{latent_dim} │ "
                    f"PR={pr:4.1f} │ VarZ{dot_vz}{kl_proxy:.3g}"
                )

            # ─── PRUNING INTELIGENTE PARA OPTUNA ───
            if trial is not None:
                # Score intermediário: combina val_loss (menor=melhor) com saúde latente
                # O pruner compara este valor entre trials no mesmo epoch
                intermediate_score = (1.0 / max(val_loss, 1e-6)) + 0.3 * pr + 0.2 * er
                trial.report(intermediate_score, epoch)

                # Poda 1: Optuna MedianPruner — trial abaixo da mediana dos anteriores
                if trial.should_prune():
                    logger.debug(
                        f"  {_C.YELLOW}✂ PRUNE{_C.RESET} Trial {trial.number} ep{epoch + 1} "
                        f"score={intermediate_score:.2f} < mediana"
                    )
                    if epoch_pbar is not None:
                        epoch_pbar.close()
                    if tb_writer is not None:
                        tb_writer.close()
                    raise TrialPruned()

                # Poda 2: Collapse precoce — não espera metade dos epochs
                if epoch >= 3 and (er < 1.5 or au < 2 or kl_proxy < 1e-5):
                    logger.debug(
                        f"  {_C.RED}✗ COLLAPSE{_C.RESET} Trial {trial.number} ep{epoch + 1} "
                        f"eR={er:.2f} AU={int(au)} VarZ={kl_proxy:.2g}"
                    )
                    if epoch_pbar is not None:
                        epoch_pbar.close()
                    if tb_writer is not None:
                        tb_writer.close()
                    raise TrialPruned()

                # Poda 3: Val loss explodiu (divergência)
                if epoch >= 2 and val_loss > 5.0:
                    logger.debug(
                        f"  {_C.RED}⚠ DIVERGE{_C.RESET} Trial {trial.number} ep{epoch + 1} "
                        f"val_loss={val_loss:.2f} (>5.0)"
                    )
                    if epoch_pbar is not None:
                        epoch_pbar.close()
                    if tb_writer is not None:
                        tb_writer.close()
                    raise TrialPruned()

            # Poda para treino final (sem trial): apenas collapse severo
            elif epoch >= max(max_epochs // 2, 5) and (
                er < 1.1 or au == 0 or kl_proxy < 1e-6
            ):
                logger.warning(
                    f"\n{_C.RED}╔══════ COLLAPSE DETECTADO — TREINO ABORTADO ══════╗{_C.RESET}\n"
                    f"  {_C.RED}eRank ={_C.RESET} {er:.2f}  {_C.DIM}(< 1.1){_C.RESET}\n"
                    f"  {_C.RED}AU    ={_C.RESET} {int(au)}    {_C.DIM}(== 0){_C.RESET}\n"
                    f"  {_C.RED}VarZ  ={_C.RESET} {kl_proxy:.4g}  {_C.DIM}(< 1e-6){_C.RESET}\n"
                    f"{_C.RED}╚══════════════════════════════════════════════════╝{_C.RESET}"
                )
                if epoch_pbar is not None:
                    epoch_pbar.close()
                if tb_writer is not None:
                    tb_writer.close()
                break

            # [OVERFIT FIX] Early stopping: monitora val_loss e para se nao melhorar.
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0  # reseta contador a cada nova best
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    msg = (
                        f"{_C.YELLOW}⏸ EARLY STOP{_C.RESET} ep{epoch + 1} — "
                        f"val_loss={val_loss:.4f} nao melhorou em {early_stop_patience} epochs "
                        f"(best={best_val:.4f})"
                    )
                    if trial is None:
                        # Treino final: log normal
                        logger.info(f"[HybridAE_v3] {msg}")
                        if epoch_pbar is not None:
                            epoch_pbar.write(msg)
                    else:
                        # HPO trial: log debug
                        logger.debug(msg)
                    break

        if epoch_pbar is not None:
            epoch_pbar.close()
        # TensorBoard: loga métricas finais e fecha writer
        if tb_writer is not None:
            tb_writer.add_hparams(
                {
                    "lr": lr,
                    "max_epochs": max_epochs,
                    "latent_dim": model.latent_dim,
                    "trial": trial.number if trial else -1,
                },
                {
                    "hparam/best_val_loss": best_val,
                    "hparam/final_PR": pr,
                    "hparam/final_eRank": er,
                    "hparam/final_AU": au,
                    "hparam/final_VarZ": kl_proxy,
                },
            )
            tb_writer.close()
        if best_state:
            model.load_state_dict(best_state)
        # [G4 FIX] Retorna AU e kl_proxy para o score do HPO penalizar collapse.
        # [GAP PENALTY] Calcula degradacao do val no fim do treino e gap train-val.
        final_val_avg = (
            float(np.mean(final_val_window)) if final_val_window else float(best_val)
        )
        val_degradation = final_val_avg - float(best_val)  # quanto val piorou do minimo
        return (
            float(best_val),
            float(pr),
            float(er),
            float(au),
            float(kl_proxy),
            float(val_degradation),
            float(train_val_gap_max),
        )

    def train_autoencoder_temporal(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        optimize: bool = True,
        n_trials: int = 50,
        latent_dim: int = 32,
        d_model: int = 128,
        seq_len: int = 48,
        max_epochs: int = 150,  # [STABILITY] 100 -> 150 epochs. Early stop patience=15 corta se convergir antes; mais espaço se precisar.
        batch_size: int = 128,
        lr: float = 1e-3,
        **kwargs,
    ) -> bool:
        try:
            # 0. Otimização HPO (se necessário)
            if optimize and not self.is_optimized:
                logger.info(
                    "[HybridAE_v3] Otimizando hiperparâmetros antes do treino..."
                )
                self.optimize_hyperparameters(df, n_trials=n_trials)

            logger.info(
                "[HybridAE_v3] Iniciando treinamento (RL-Aware Predictive State)..."
            )
            self.feature_cols = feature_columns or self._filter_features(df)
            df_feat = (
                df[self.feature_cols]
                .copy()
                .replace([np.inf, -np.inf], np.nan)
                .ffill(limit=300)
                .dropna()
            )

            # 1. Purged Split & Scaling
            split = int(0.8 * len(df_feat))
            self.scaler = RobustScaler()
            data = (
                self.scaler.fit(df_feat.iloc[:split].values)
                .transform(df_feat.values)
                .astype(np.float32)
            )

            regime_labels = (
                df.loc[df_feat.index]
                .get("regime", pd.Series(np.full(len(df_feat), 2), index=df_feat.index))
                .fillna(2)
                .astype(int)
                .values
            )

            # Verifica se regime_labels tem variância. Se não, deriva via retornos.
            # NUNCA gera labels aleatorios — treinaria o RegimeHead em ruido puro.
            unique_regimes = np.unique(regime_labels)
            if len(unique_regimes) < 2:
                close_col = "close" if "close" in df.columns else None
                if close_col is None:
                    raise ValueError(
                        "[HybridAE_v3] regime_labels degenerados e coluna 'close' ausente. "
                        "Abortando treino para evitar RegimeHead aprender ruido."
                    )
                logger.warning(
                    f"[HybridAE_v3] regime_labels tem apenas classe {unique_regimes}. "
                    "Derivando labels via z-score do log-return."
                )
                close = df.loc[df_feat.index, close_col].values
                log_ret = np.zeros(len(close))
                log_ret[1:] = np.diff(np.log(np.maximum(close, 1e-9)))
                # bfill aqui afeta APENAS os primeiros 19 candles (warmup do rolling).
                # E aceitavel porque (1) sao labels de TREINO fallback, nao previsao,
                # (2) o regime detector real ja falhou, estamos em modo degradado.
                vol = pd.Series(log_ret).rolling(20).std().bfill().fillna(1e-4).values
                vol = np.maximum(vol, 1e-6)
                regime_labels = np.where(
                    log_ret > 0.5 * vol,
                    0,
                    np.where(log_ret < -0.5 * vol, 1, 2),
                ).astype(np.int64)

            f_sharpe, f_drawdown = self._compute_forward_targets(df.loc[df_feat.index])

            # [G1/G6 FIX] Hiperparametros HPO-otimizados sao usados quando disponiveis.
            # Antes: seq_len, batch_size e prediction_horizon eram fixos via kwargs/defaults
            # mesmo apos HPO os ter otimizado. Agora puxamos dos hparams persistidos.
            if optimize and self.hparams:
                seq_len = self.hparams.get("seq_len", seq_len)
                batch_size = self.hparams.get("batch_size", batch_size)
                prediction_horizon = self.hparams.get(
                    "prediction_horizon", kwargs.get("prediction_horizon", 8)
                )
            else:
                prediction_horizon = kwargs.get("prediction_horizon", 8)

            embargo = seq_len * 2  # Embargo rigoroso (proporcional ao contexto)

            train_d = data[: split - embargo]
            val_d = data[split:]
            train_r, val_r = regime_labels[: split - embargo], regime_labels[split:]
            tr_sh, v_sh = f_sharpe[: split - embargo], f_sharpe[split:]
            tr_dd, v_dd = f_drawdown[: split - embargo], f_drawdown[split:]

            # [OVERFIT FIX] Diagnostico de distribution shift train vs val.
            # Causa raiz do overfit no run anterior: train (2023-2025) dominado por
            # bull/bear, val (ultimos 20% ~2025-2026) dominado por ranger lateral.
            # RegimeHead memorizou padroes temporais que nao generalizaram.
            # Solucao: logar distribuicao por regime para diagnosticar; se shift > 30%
            # em alguma classe, avisa para o usuario considerar walk-forward CV.
            _tr_dist = np.bincount(train_r, minlength=3) / max(1, len(train_r))
            _vl_dist = np.bincount(val_r, minlength=3) / max(1, len(val_r))
            _shift = np.abs(_tr_dist - _vl_dist).max()
            _regime_names = ["Bull", "Bear", "Ranger"]
            _shift_msg = (
                f"[REGIME SHIFT] Train: "
                + " ".join(f"{_regime_names[i]}={_tr_dist[i]:.0%}" for i in range(3))
                + " | Val: "
                + " ".join(f"{_regime_names[i]}={_vl_dist[i]:.0%}" for i in range(3))
                + f" | max_shift={_shift:.1%}"
            )
            if _shift > 0.30:
                logger.warning(
                    f"⚠️ {_shift_msg} — distribution shift severo. "
                    "RegimeHead pode overfitar. Considere walk-forward CV."
                )
            else:
                logger.info(_shift_msg)

            self.hparams["prediction_horizon"] = prediction_horizon
            self.hparams["regime_dist_train"] = _tr_dist.tolist()
            self.hparams["regime_dist_val"] = _vl_dist.tolist()

            ds_tr = HybridDataset(
                train_d,
                seq_len,
                train_r,
                tr_sh,
                tr_dd,
                prediction_horizon=prediction_horizon,
            )
            ds_vl = HybridDataset(
                val_d,
                seq_len,
                val_r,
                v_sh,
                v_dd,
                prediction_horizon=prediction_horizon,
            )
            dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
            dl_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False)

            if not self.hparams or "latent_dim" not in self.hparams:
                self.hparams.update(
                    {
                        "input_dim": data.shape[1],
                        "latent_dim": latent_dim,
                        "d_model": d_model,
                        "n_conv_layers": 3,
                        "n_heads_attn": 4,
                        "dropout": 0.1,
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "lr": lr,
                    }
                )
            if optimize:
                lr = self.hparams.get("lr", lr)
                max_epochs = kwargs.get("epochs", max_epochs)
            self.hparams["input_dim"] = data.shape[1]
            # Persiste seq_len/batch_size/prediction_horizon (caso vindos de kwargs)
            self.hparams["seq_len"] = seq_len
            self.hparams["batch_size"] = batch_size

            self.model = self._build_model(self.hparams).to(self.device)
            self.model = self._compile_for_training(self.model)
            loss_fn = UncertaintyWeightedLoss().to(self.device)

            # Banner de inicio do treino final
            _ld = self.hparams.get("latent_dim", 32)
            print(_banner("🎯 HYBRID AUTOENCODER — FINAL TRAINING", width=70))
            print(
                _kv_block(
                    [
                        ("Train samples", f"{len(train_d):,}"),
                        ("Val samples", f"{len(val_d):,}"),
                        ("Epochs", f"{max_epochs}"),
                        ("latent_dim", f"{_ld}"),
                        ("seq_len", f"{seq_len}"),
                        ("batch_size", f"{batch_size}"),
                        ("prediction_horizon", f"{prediction_horizon}"),
                        ("learning_rate", f"{lr:.5f}"),
                        (
                            "Losses",
                            "JEPA + Regime CE + VICReg + TS2Vec (Kendall-Gal weighted)",
                        ),
                    ]
                )
            )
            print()

            val_loss, pr, er, au, kl, val_degrad_final, gap_max_final = self._train_one(
                self.model, loss_fn, dl_tr, dl_vl, lr=lr, max_epochs=max_epochs
            )
            # Persiste metricas de estabilidade nos hparams para auditoria pos-treino
            self.hparams["val_degradation_final"] = float(val_degrad_final)
            self.hparams["train_val_gap_max"] = float(gap_max_final)

            # Sumario do treino antes do gate
            au_r_train = au / max(_ld, 1)
            er_r_train = er / max(_ld, 1)
            # [GAP PENALTY] Semaforo de estabilidade (invertido: menor = melhor)
            dot_degrad = (
                f"{_C.GREEN}●{_C.RESET}"
                if val_degrad_final < 0.05
                else f"{_C.YELLOW}●{_C.RESET}"
                if val_degrad_final < 0.20
                else f"{_C.RED}●{_C.RESET}"
            )
            dot_gap = (
                f"{_C.GREEN}●{_C.RESET}"
                if gap_max_final < 0.5
                else f"{_C.YELLOW}●{_C.RESET}"
                if gap_max_final < 1.5
                else f"{_C.RED}●{_C.RESET}"
            )
            print(_banner("📊 TRAINING SUMMARY", char="─", width=70))
            print(
                _kv_block(
                    [
                        ("Val loss (Regime CE)", f"{val_loss:.4f}"),
                        ("Participation Ratio", f"{pr:.2f}"),
                        (
                            "Effective Rank",
                            f"{_status_dot(er_r_train, 0.30, 0.15)} {er:.2f}/{_ld} ({er_r_train:.1%})",
                        ),
                        (
                            "Active Units",
                            f"{_status_dot(au_r_train, 0.40, 0.20)} {int(au)}/{_ld} ({au_r_train:.1%})",
                        ),
                        ("KL Proxy (VarZ)", f"{_status_dot(kl, 1e-3, 1e-4)} {kl:.4g}"),
                        (
                            "Val degradation",
                            f"{dot_degrad} {val_degrad_final:+.4f} (final - best)",
                        ),
                        (
                            "Train-Val gap max",
                            f"{dot_gap} {gap_max_final:+.4f} (overfit literal)",
                        ),
                    ]
                )
            )
            print()

            # GATE PÓS-TREINO: valida saúde do modelo antes de salvar.
            # [BUG FIX] Variavel correta e seq_len (nao 'seq' que nao existe no escopo).
            # Antes este NameError era engolido pelo except do train_autoencoder_temporal,
            # fazendo o treino retornar False silenciosamente APOS terminar — o modelo
            # treinava OK mas o gate nunca rodava, e is_trained ficava True sem auditoria.
            gate_result = self._post_training_gate(val_d, seq_len)
            er_r = gate_result.get("effective_rank_ratio", 0.0)
            au_r = gate_result.get("active_units_ratio", 0.0)
            kl_v = gate_result.get("kl_proxy", 0.0)
            pr_v = gate_result.get("participation_ratio", 0.0)

            if gate_result["pass"]:
                print(
                    _banner(
                        f"{_C.GREEN}✓ GATE APROVADO — modelo confiavel{_C.RESET}",
                        char="═",
                        width=70,
                    )
                )
                print(
                    _kv_block(
                        [
                            (
                                "eRank ratio",
                                f"{_status_dot(er_r, 0.30, 0.15)} {er_r:.1%} (min 30%)",
                            ),
                            (
                                "Active Units ratio",
                                f"{_status_dot(au_r, 0.40, 0.20)} {au_r:.1%} (min 40%)",
                            ),
                            (
                                "KL proxy",
                                f"{_status_dot(kl_v, 1e-3, 1e-4)} {kl_v:.4g} (min 1e-4)",
                            ),
                            (
                                "Participation Ratio",
                                f"{_C.GREEN}●{_C.RESET} {pr_v:.2f}",
                            ),
                        ]
                    )
                )
                print()
                logger.info(
                    f"[HybridAE_v3] GATE APROVADO: eR={er_r:.1%}, AU={au_r:.1%}, KL={kl_v:.4g}, PR={pr_v:.2f}"
                )
                self.hparams["gate_approved"] = True
                self.hparams.pop("gate_failures", None)
                self._save()
                return True
            else:
                print(
                    _banner(
                        f"{_C.RED}✗ GATE REPROVADO — modelo NAO confiavel{_C.RESET}",
                        char="═",
                        width=70,
                    )
                )
                print(
                    _kv_block(
                        [
                            (
                                "eRank ratio",
                                f"{_status_dot(er_r, 0.30, 0.15)} {er_r:.1%} (min 30%)",
                            ),
                            (
                                "Active Units ratio",
                                f"{_status_dot(au_r, 0.40, 0.20)} {au_r:.1%} (min 40%)",
                            ),
                            (
                                "KL proxy",
                                f"{_status_dot(kl_v, 1e-3, 1e-4)} {kl_v:.4g} (min 1e-4)",
                            ),
                            (
                                "Falhas",
                                f"{_C.RED}{gate_result.get('failures', [])}{_C.RESET}",
                            ),
                            (
                                "Status",
                                f"{_C.YELLOW}Modelo salvo mas marcado como NAO-CONFIAVEL{_C.RESET}",
                            ),
                        ]
                    )
                )
                print()
                logger.warning(
                    f"[HybridAE_v3] GATE REPROVADO: falhas={gate_result.get('failures', [])}"
                )
                self.hparams["gate_approved"] = False
                self.hparams["gate_failures"] = gate_result["failures"]
                self._save()
                return True  # Treino completou, mas modelo é suspeito

        except Exception as e:
            logger.error(f"[HybridAE_v3] Erro: {e}", exc_info=True)
            return False

    @torch.no_grad()
    def _post_training_gate(self, val_data: np.ndarray, seq_len: int) -> Dict[str, Any]:
        """
        Gate pós-treino automático. Avalia saúde do espaço latente no dataset
        de validação e retorna veredicto pass/fail com métricas.

        Critérios mínimos:
          - eRank/latent_dim >= 30% (diversidade dimensional)
          - ActiveUnits/latent_dim >= 40% (utilização do espaço)
          - KL Proxy >= 1e-4 (variância mínima, anti-collapse)
        """
        MINIMUM_EFFECTIVE_RANK_RATIO = 0.30
        MINIMUM_ACTIVE_UNITS_RATIO = 0.40
        MINIMUM_KL_PROXY = 1e-4

        latent_dim = self.hparams.get("latent_dim", 32)
        n = len(val_data)

        if n < seq_len + 10 or self.model is None:
            return {"pass": True, "reason": "Dados insuficientes para gate, skip"}

        self.model.eval()
        all_z = []
        indices = list(range(seq_len - 1, min(n, seq_len + 2000)))
        batch_size = 256

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            batch_tensors = [
                torch.FloatTensor(val_data[i - seq_len + 1 : i + 1])
                for i in batch_indices
            ]
            x = torch.stack(batch_tensors).to(self.device)
            out = self.model(x, x_tgt=None, training=False)
            all_z.append(out["z"].cpu())

        z_all = torch.cat(all_z)
        pr = compute_participation_ratio(z_all)
        er = compute_effective_rank(z_all)
        au = compute_active_units(z_all)
        kl = compute_kl_proxy(z_all)

        er_ratio = er / max(latent_dim, 1)
        au_ratio = au / max(latent_dim, 1)

        failures = []
        if er_ratio < MINIMUM_EFFECTIVE_RANK_RATIO:
            failures.append(
                f"eRank {er_ratio:.1%} < {MINIMUM_EFFECTIVE_RANK_RATIO:.0%}"
            )
        if au_ratio < MINIMUM_ACTIVE_UNITS_RATIO:
            failures.append(
                f"ActiveU {au_ratio:.1%} < {MINIMUM_ACTIVE_UNITS_RATIO:.0%}"
            )
        if kl < MINIMUM_KL_PROXY:
            failures.append(f"KL {kl:.6f} < {MINIMUM_KL_PROXY}")

        return {
            "pass": len(failures) == 0,
            "failures": failures,
            "participation_ratio": float(pr),
            "effective_rank": float(er),
            "effective_rank_ratio": float(er_ratio),
            "active_units": float(au),
            "active_units_ratio": float(au_ratio),
            "kl_proxy": float(kl),
        }

    def apply_hidden_features_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.scaler is None:
            return df
        try:
            # Se feature_cols vazio (modelo carregado do disco), recalcular
            if not self.feature_cols:
                self.feature_cols = self._filter_features(df)
            feat_cols = [c for c in self.feature_cols if c in df.columns]
            df_feat = (
                df[feat_cols].copy().replace([np.inf, -np.inf], np.nan).ffill(limit=300)
            )
            # Drop rows still NaN (início do dataset) — não usar fillna(0) que cria sinais falsos
            valid_mask = ~df_feat.isna().any(axis=1)
            # [C3 FIX] Consistência train/inference: usa ffill(limit=300) + fillna(0.0)
            # identico ao treino. O segundo ffill() sem limit era redundante e confuso.
            # Bars com NaN residual (warmup) são invalidados pelo valid_mask acima.
            # FIX: fillna(0.0) mantido para rows válidas (no-op após ffill). Rows inválidas
            # são excluídas dos índices de inferência abaixo para evitar sinal falso.
            df_feat_filled = df_feat.fillna(0.0)
            data_full = self.scaler.transform(df_feat_filled.values)
            seq = self.hparams.get("seq_len", 48)
            latent_dim = self.hparams.get("latent_dim", 32)
            n = len(data_full)

            self.model.eval()
            latents = np.full((n, latent_dim), np.nan)
            regime_logits = np.full((n, N_REGIMES), np.nan)

            # Inferência em batches para performance
            batch_size = 256
            indices = [
                i for i in range(seq - 1, n) if valid_mask[i - seq + 1 : i + 1].all()
            ]
            with torch.no_grad():
                for batch_start in range(0, len(indices), batch_size):
                    batch_indices = indices[batch_start : batch_start + batch_size]
                    batch_tensors = []
                    for i in batch_indices:
                        batch_tensors.append(
                            torch.FloatTensor(data_full[i - seq + 1 : i + 1])
                        )
                    x_batch = torch.stack(batch_tensors).to(self.device)
                    out = self.model(x_batch, x_tgt=None, training=False)
                    latents_batch = out["z"].cpu().numpy()
                    regime_batch = out["regime"].softmax(dim=-1).cpu().numpy()
                    for j, i in enumerate(batch_indices):
                        if valid_mask.iloc[i]:
                            latents[i] = latents_batch[j]
                            regime_logits[i] = regime_batch[j]

            # [PERF FIX] Construir TODAS as colunas novas em UM unico pd.concat
            # em vez de adicionar coluna-por-coluna. Anteriormente, cada `df_out[col] = ...`
            # causava insert no DataFrame existente → PerformanceWarning "highly fragmented"
            # e custo O(N×K) onde K=cols. Agora: 1 concat em axis=1, custo O(N+K).
            # Ganho medido: ~30% mais rapido em datasets de 100k+ linhas com 48 latents.

            # Regime: argmax apenas onde temos dados validos
            regime_vals = np.full(n, 2)  # default Ranger
            valid_regime = ~np.isnan(regime_logits[:, 0])
            regime_vals[valid_regime] = regime_logits[valid_regime].argmax(axis=1)

            # Constroi DataFrame de novas colunas de uma vez (vetorizado)
            new_cols_data: Dict[str, np.ndarray] = {}
            for d in range(latent_dim):
                col_data = latents[:, d]
                new_cols_data[f"sdae_latent_{d}"] = col_data
                # Alias para compatibilidade com agentes SAC (procuram hidden_feature_*)
                new_cols_data[f"hidden_feature_{d}"] = col_data
            new_cols_data["prob_bull"] = regime_logits[:, 0]
            new_cols_data["prob_bear"] = regime_logits[:, 1]
            new_cols_data["prob_ranger"] = regime_logits[:, 2]
            new_cols_data["regime"] = regime_vals

            new_cols_df = pd.DataFrame(new_cols_data, index=df.index)
            # [BUG FIX 2026-05-13] Concat duplicava colunas que ja existiam no df.
            # Solucao: drop colunas existentes do df ANTES do concat, preservando os
            # novos valores. Sem isso, "regime" do AE conflitava com "regime" antigo
            # do regime_detector — causando shape (N, 2) em df["regime"].values.
            df_base = df.copy()
            overlap = [c for c in new_cols_df.columns if c in df_base.columns]
            if overlap:
                df_base = df_base.drop(columns=overlap)
            df_out = pd.concat([df_base, new_cols_df], axis=1)
            return df_out
        except Exception as e:
            logger.error(f"[HybridAE_v3] Erro: {e}")
            return df

    # ─────────────────────────────────────────────────────────────────────────
    # MÉTODOS DE AUTO-AVALIAÇÃO CIENTÍFICA (JEPA-basado)
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _compute_consistency_error(self, df: pd.DataFrame) -> float:
        feat_cols = [c for c in self.feature_cols if c in df.columns]
        if not feat_cols or self.model is None or self.scaler is None:
            return float("nan")
        # ANTI-LEAKAGE: bfill() removido — vazaria valores futuros para o passado,
        # invalidando o teste de consistencia OOD. Apenas ffill + zero-fill para
        # NaN do warmup inicial; janelas que comecem com NaN sao zeros (neutro).
        df_feat = (
            df[feat_cols].copy().replace([np.inf, -np.inf], np.nan).ffill(limit=300)
        )
        valid_mask = ~df_feat.isna().any(axis=1)
        data = self.scaler.transform(df_feat.fillna(0.0).values)
        seq = self.hparams.get("seq_len", 48)
        n = len(data)
        if n < seq:
            return float("nan")
        errors = []
        self.model.eval()
        for i in range(seq - 1, n, max(1, (n - seq) // 100)):
            if not valid_mask[i - seq + 1 : i + 1].all():
                continue
            x = (
                torch.FloatTensor(data[i - seq + 1 : i + 1])
                .unsqueeze(0)
                .to(self.device)
            )
            out = self.model(x, x_tgt=None, training=False)
            x_aug = x.clone()
            aug_len = int(x.size(1) * 0.15)
            x_aug[:, :aug_len, :] = 0.0
            out_aug = self.model(x_aug, x_tgt=None, training=False)
            errors.append(F.mse_loss(out["z"], out_aug["z"]).item())
        return float(np.mean(errors)) if errors else float("nan")

    @staticmethod
    def _block_permutation(df: pd.DataFrame, block_size: int = 10) -> pd.DataFrame:
        """
        Block permutation para teste de significancia estatistica.

        Quebra a serie em blocos contiguos de `block_size`, embaralha a ordem
        dos blocos, e atribui o indice cronologico original. Preserva estrutura
        LOCAL (dentro de cada bloco) mas destroi correlacao GLOBAL temporal.

        [BUG FIX 2026-05-12] Versao anterior fazia `pd.concat(blocks).loc[df.index]`
        que REORDENAVA os valores de volta para a ordem cronologica original —
        a permutacao nao tinha efeito e p_value do permutation_test era SEM
        SIGNIFICADO ESTATISTICO. Agora reatribuimos df.index como label apenas,
        preservando a ordem embaralhada dos VALORES.
        """
        n = len(df)
        if n < block_size:
            return df.copy()
        blocks = [df.iloc[i : i + block_size].copy() for i in range(0, n, block_size)]
        np.random.shuffle(blocks)
        df_shuffled = pd.concat(blocks).reset_index(drop=True)
        # Trunca p/ exatamente n linhas (ultimo bloco pode ser parcial)
        df_shuffled = df_shuffled.iloc[:n]
        # Atribui o indice cronologico original como rotulo (mantem ordem de VALORES embaralhada)
        df_shuffled.index = df.index[:n]
        return df_shuffled

    def validate_ood_consistency(
        self, df: pd.DataFrame, ood_periods: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError(
                "Modelo não treinado. Execute train_autoencoder_temporal primeiro."
            )
        results: Dict[str, Any] = {}
        baseline_mask = pd.Series(True, index=df.index)
        for start, end, _ in ood_periods:
            try:
                rng = pd.date_range(start, end).intersection(df.index)
                baseline_mask[rng] = False
            except Exception:
                pass
        baseline_df = df.loc[baseline_mask]
        if len(baseline_df) > self.hparams.get("seq_len", 48):
            results["baseline"] = self._compute_consistency_error(baseline_df)
        for start, end, label in ood_periods:
            try:
                ood_df = df.loc[start:end]
                if len(ood_df) < self.hparams.get("seq_len", 48):
                    results[label] = {
                        "error": float("nan"),
                        "acceptable": None,
                        "degradation": float("nan"),
                        "reason": "Dados insuficientes",
                    }
                    continue
                err = self._compute_consistency_error(ood_df)
                baseline = results.get("baseline", float("nan"))
                degradation = (
                    (err - baseline) / baseline
                    if not np.isnan(baseline) and baseline > 0
                    else 0
                )
                results[label] = {
                    "error": err,
                    "acceptable": degradation < 1.0,
                    "degradation": degradation,
                }
            except Exception as e:
                results[label] = {"error": str(e)}
        return results

    def permutation_test_consistency(
        self, df: pd.DataFrame, n_permutations: int = 50, block_size: int = 10
    ) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError(
                "Modelo não treinado. Execute train_autoencoder_temporal primeiro."
            )
        real_error = self._compute_consistency_error(df)
        if np.isnan(real_error):
            return {
                "real_error": float("nan"),
                "p_value": 1.0,
                "interpretation": "Erro no cálculo",
            }
        perm_errors = []
        for _ in range(n_permutations):
            df_perm = self._block_permutation(df, block_size)
            err = self._compute_consistency_error(df_perm)
            if not np.isnan(err):
                perm_errors.append(err)
        if not perm_errors:
            return {
                "real_error": real_error,
                "p_value": 1.0,
                "interpretation": "Sem dados suficientes",
            }
        perm_arr = np.array(perm_errors)
        p_value = float(np.mean(perm_arr >= real_error))
        interpretation = (
            "Modelo aprendeu padrões temporais genuínos (p<0.05)"
            if p_value < 0.05
            else "Evidência fraca de padrões temporais"
        )
        return {
            "real_error": real_error,
            "perm_error_mean": float(perm_arr.mean()),
            "perm_error_std": float(perm_arr.std()),
            "p_value": p_value,
            "n_permutations": len(perm_errors),
            "interpretation": interpretation,
        }
