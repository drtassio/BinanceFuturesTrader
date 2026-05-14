"""
Profitability Predictor — Meta-Labeling for Trade Filtering.

Based on Lopez de Prado (2018) "Advances in Financial ML", Chapter 3-4.
Primary model (RL agent) → SIDE (long/short/flat).
Meta-labeler → P(trade_profitable | entry_features).

Only trades with P(profit) > threshold pass the filter.

Model: LogisticRegression + TimeSeriesSplit CV (anti-leakage).
Label: pnl > 2 × taker_fee × notional  (fee-adjusted profitability).
Threshold: calibrated via precision-recall curve (maximize F1).
"""

import asyncio
import numpy as np
import pandas as pd
import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
)
from sklearn.model_selection import TimeSeriesSplit
import joblib

from utils.logger import get_logger
from config.settings import AIConfig

logger = get_logger("ProfitabilityPredictor")

# ── VISUAL HELPERS ────────────────────────────────────────────────────────────



def _bar(value: float, width: int = 20, lo: float = 0.0, hi: float = 1.0) -> str:
    """ASCII progress bar scaled between lo and hi."""
    filled = int(round((value - lo) / max(hi - lo, 1e-9) * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)

def _rating(auc: float) -> str:
    if auc >= 0.75:
        return "🟢 Excelente"
    if auc >= 0.65:
        return "🟡 Bom"
    if auc >= 0.55:
        return "🟠 Fraco"
    return "🔴 Insuficiente"

def _print_metrics_panel(metrics: "MetaLabelMetrics") -> None:
    """Print a rich-style metrics panel after meta-labeler training."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(no_wrap=True)
        table.add_column()

        def metric_row(label, value_str, bar_val, lo=0.0, hi=1.0):
            table.add_row(label, value_str, _bar(bar_val, 18, lo, hi))

        table.add_row("[bold]Amostras[/bold]", f"{metrics.n_samples:,}", "")
        table.add_row(
            "[bold]Balanceamento[/bold]", f"{metrics.class_balance:.1%} positivos", ""
        )
        table.add_row("[bold]CV Folds[/bold]", f"TimeSeriesSplit (5 folds)", "")
        table.add_row("─" * 20, "─" * 22, "─" * 18)
        metric_row(
            "ROC-AUC ",
            f"{metrics.roc_auc:.4f}  {_rating(metrics.roc_auc)}",
            metrics.roc_auc,
            0.5,
            1.0,
        )
        metric_row("F1-Score", f"{metrics.f1:.4f}", metrics.f1, 0.0, 1.0)
        metric_row("Precisão", f"{metrics.precision:.4f}", metrics.precision, 0.0, 1.0)
        metric_row("Recall  ", f"{metrics.recall:.4f}", metrics.recall, 0.0, 1.0)
        metric_row(
            "Brier   ",
            f"{metrics.brier:.4f}  (↓ melhor)",
            1.0 - metrics.brier,
            0.0,
            1.0,
        )
        table.add_row("─" * 20, "─" * 22, "─" * 18)
        table.add_row(
            "[bold]Threshold[/bold]",
            f"{metrics.profit_threshold:.3f} (calibrado P-R)",
            "",
        )
        metric_row(
            "Pass Rate",
            f"{metrics.passed_rate:.1%} dos trades",
            metrics.passed_rate,
            0.0,
            1.0,
        )
        metric_row(
            "WR Filtrado",
            f"{metrics.profit_rate_passed:.1%}",
            metrics.profit_rate_passed,
            0.0,
            1.0,
        )
        table.add_row(
            "[bold]Melhoria vs baseline[/bold]",
            f"[{'green' if metrics.improvement_vs_baseline > 0 else 'red'}]{metrics.improvement_vs_baseline:+.1%}[/]",
            "",
        )

        panel = Panel(
            table,
            title="[bold magenta]🎯 META-LABELING — PROFITABILITY PREDICTOR[/bold magenta]",
            subtitle="[dim]Lopez de Prado (2018) AFLM Cap. 3-4[/dim]",
            border_style="magenta",
        )
        console.print(panel)

    except ImportError:
        # Fallback sem rich
        w = 62
        print("\n" + "╔" + "═" * w + "╗")
        print("║" + " 🎯 META-LABELING — PROFITABILITY PREDICTOR".center(w) + "║")
        print("╠" + "═" * w + "╣")
        def row(k, v):
            print(f"║  {k:<20} {v:<38}║")
        row("Amostras:", f"{metrics.n_samples:,}")
        row("Balanceamento:", f"{metrics.class_balance:.1%} positivos")
        row("CV:", "TimeSeriesSplit (5 folds)")
        print("╠" + "─" * w + "╣")
        row("ROC-AUC:", f"{metrics.roc_auc:.4f}  {_rating(metrics.roc_auc)}")
        row("F1-Score:", f"{metrics.f1:.4f}  {_bar(metrics.f1)}")
        row("Precisão:", f"{metrics.precision:.4f}  {_bar(metrics.precision)}")
        row("Recall:", f"{metrics.recall:.4f}  {_bar(metrics.recall)}")
        row("Brier (↓):", f"{metrics.brier:.4f}  {_bar(1 - metrics.brier)}")
        print("╠" + "─" * w + "╣")
        row("Threshold:", f"{metrics.profit_threshold:.3f} (calibrado P-R)")
        row("Pass Rate:", f"{metrics.passed_rate:.1%}")
        row("WR Filtrado:", f"{metrics.profit_rate_passed:.1%}")
        row("Melhoria:", f"{metrics.improvement_vs_baseline:+.1%} vs baseline")
        print("╚" + "═" * w + "╝\n")


# ── DATACLASS ─────────────────────────────────────────────────────────────────



@dataclass
class MetaLabelMetrics:
    """Metrics for meta-labeling model evaluation."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    brier: float = 0.0
    n_samples: int = 0
    class_balance: float = 0.0
    profit_threshold: float = 0.55
    passed_rate: float = 0.0
    profit_rate_passed: float = 0.0
    improvement_vs_baseline: float = 0.0
    cv_auc_mean: float = 0.0
    cv_auc_std: float = 0.0
    feature_importances: Dict[str, float] = field(default_factory=dict)


# ── MAIN CLASS ────────────────────────────────────────────────────────────────



class ProfitabilityPredictor:
    """
    Meta-labeling binary classifier: P(trade_profitable | entry_features).

    Scientific design:
    - LogisticRegression (fast, interpretable, well-calibrated)
    - TimeSeriesSplit 5-fold CV: respects temporal order (Lopez de Prado 2018 Ch.7)
    - Fee-adjusted label: pnl > 2 × taker_fee × notional (nets fees)
    - Threshold calibrated via Precision-Recall curve (maximize F1)
    - Isotonic calibration via CalibratedClassifierCV
    - Feature importance via model.coef_
    """

    TAKER_FEE_DEFAULT = 0.0004  # Binance Futures taker fee

    def __init__(
        self,
        input_dim: Optional[int] = None,
        profit_threshold: float = 0.55,
        min_samples: int = 50,
        model_dir: Optional[str] = None,
        config: Optional[AIConfig] = None,
        taker_fee: Optional[float] = None,
    ):
        self.input_dim = input_dim
        self.profit_threshold = profit_threshold
        self.min_samples = min_samples
        self.model_dir = model_dir or os.path.join(os.getcwd(), "models_ai")
        self.config = config or AIConfig()
        self.taker_fee = taker_fee or getattr(
            self.config, "TAKER_FEE", self.TAKER_FEE_DEFAULT
        )
        os.makedirs(self.model_dir, exist_ok=True)

        self.model: Optional[CalibratedClassifierCV] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self.feature_names: Optional[List[str]] = None
        self.last_metrics: Optional[MetaLabelMetrics] = None
        self.n_trades_seen = 0
        self.n_trades_at_last_train = 0

    # ── LABEL ENGINEERING ─────────────────────────────────────────────────────

    def _compute_labels(
        self,
        pnls: np.ndarray,
        notionals: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fee-adjusted profitability label.

        A trade is labeled +1 only if PnL exceeds the round-trip taker fee.
        This avoids labeling micro-profit trades (that net negative after fees) as wins.

        Lopez de Prado (2018) AFLM Ch.4: the meta-labeler must learn
        REAL profitability, not accounting profitability.
        """
        if notionals is not None and len(notionals) == len(pnls):
            fee_threshold = 2.0 * self.taker_fee * np.abs(notionals)
        else:
            # Without notional, use a conservative 0 threshold (safe fallback)
            fee_threshold = np.zeros_like(pnls)
        return (pnls > fee_threshold).astype(np.int64)

    # ── THRESHOLD CALIBRATION ─────────────────────────────────────────────────

    @staticmethod
    def _calibrate_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Find threshold that maximizes F1 on the validation set.
        Falls back to 0.55 if curve is degenerate.
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls + 1e-9),
            0.0,
        )
        # Exclude the last element (recall=0, precision=1, no threshold)
        if len(thresholds) == 0:
            return 0.55
        best_idx = int(np.argmax(f1_scores[:-1]))
        best_thresh = float(thresholds[best_idx])
        # Clamp to sensible range
        return float(np.clip(best_thresh, 0.45, 0.75))

    # ── TRAINING ──────────────────────────────────────────────────────────────

    def train(
        self,
        entry_features: np.ndarray,
        pnls: np.ndarray,
        notionals: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        progress_callback=None,
    ) -> MetaLabelMetrics:
        """
        Train the meta-labeler with fee-adjusted labels and TimeSeriesSplit CV.

        Args:
            entry_features: (N, D) features at trade entry time
            pnls:           (N,) realized PnL values (NOT binary)
            notionals:      (N,) notional value of each trade (for fee threshold)
            feature_names:  Column names for interpretability
            progress_callback: Optional callable(step, total)

        Returns:
            MetaLabelMetrics with full evaluation
        """
        X = np.asarray(entry_features, dtype=np.float64)
        pnls_arr = np.asarray(pnls, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Fee-adjusted labels (Lopez de Prado 2018 Ch.4)
        y = self._compute_labels(pnls_arr, notionals)

        n = len(X)
        n_pos = int(y.sum())
        n_steps = 6

        if progress_callback:
            progress_callback(0, n_steps)

        if n < self.min_samples:
            logger.warning(
                f"[MetaLabel] Apenas {n} amostras (min={self.min_samples}). Modelo não treinado."
            )
            return MetaLabelMetrics(n_samples=n)

        if n_pos < 10 or (n - n_pos) < 10:
            logger.warning(
                f"[MetaLabel] Classes desequilibradas demais: {n_pos}/{n} positivos. Abortando."
            )
            return MetaLabelMetrics(n_samples=n, class_balance=float(y.mean()))

        # ── STEP 1: Scale ────────────────────────────────────────────────────
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        if progress_callback:
            progress_callback(1, n_steps)

        # ── STEP 2: TimeSeriesSplit CV for AUC estimate ───────────────────────
        n_folds = min(5, n // max(self.min_samples, 20))
        n_folds = max(2, n_folds)
        tscv = TimeSeriesSplit(n_splits=n_folds)

        cv_aucs = []
        base_lr = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=0.1,
            solver="lbfgs",
            random_state=42,
        )

        try:
            from tqdm import tqdm as _tqdm


            fold_iter = _tqdm(
                enumerate(tscv.split(X_scaled)),
                total=n_folds,
                desc="  ↳ MetaLabel CV",
                unit="fold",
                leave=False,
            )
        except ImportError:
            fold_iter = enumerate(tscv.split(X_scaled))

        for fold_idx, (tr_idx, val_idx) in fold_iter:
            if len(np.unique(y[val_idx])) < 2:
                continue
            lr_fold = LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                C=0.1,
                solver="lbfgs",
                random_state=42,
            )
            lr_fold.fit(X_scaled[tr_idx], y[tr_idx])
            proba_fold = lr_fold.predict_proba(X_scaled[val_idx])[:, 1]
            try:
                cv_aucs.append(roc_auc_score(y[val_idx], proba_fold))
            except Exception:
                pass

        cv_auc_mean = float(np.mean(cv_aucs)) if cv_aucs else 0.0
        cv_auc_std = float(np.std(cv_aucs)) if cv_aucs else 0.0
        if progress_callback:
            progress_callback(2, n_steps)

        # ── STEP 3: Final model on 80% train, calibrate on 20% val ──────────
        split_idx = int(n * 0.80)
        X_tr, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_tr, y_val = y[:split_idx], y[split_idx:]

        base_lr.fit(X_tr, y_tr)

        # Isotonic calibration (Platt scaling via CalibratedClassifierCV on val set)
        # We calibrate on the val set only (no refit) to avoid leakage
        if len(X_val) >= 20 and len(np.unique(y_val)) == 2:
            try:
                calibrated = CalibratedClassifierCV(
                    base_lr, cv="prefit", method="isotonic"
                )
                calibrated.fit(X_val, y_val)
                self.model = calibrated
            except Exception:
                self.model = base_lr
        else:
            self.model = base_lr

        if progress_callback:
            progress_callback(3, n_steps)

        if self.model is None:
            logger.error("❌ [MetaLabel] Falha ao inicializar o modelo antes da avaliação.")
            return MetaLabelMetrics(n_samples=n)

        # ── STEP 4: Evaluate on validation set ───────────────────────────────
        y_proba = self.model.predict_proba(X_val)[:, 1]

        # Calibrate threshold via P-R curve
        calibrated_threshold = self._calibrate_threshold(y_val, y_proba)
        self.profit_threshold = calibrated_threshold

        y_pred = (y_proba >= calibrated_threshold).astype(int)

        passed_mask = y_proba >= calibrated_threshold
        profit_rate_passed = (
            float(y_val[passed_mask].mean()) if passed_mask.sum() > 0 else 0.0
        )
        improvement = (
            profit_rate_passed - float(y_val.mean()) if passed_mask.sum() > 0 else 0.0
        )

        if progress_callback:
            progress_callback(4, n_steps)

        # ── STEP 5: Feature importance from base LR coefficients ─────────────
        feature_importances: Dict[str, float] = {}
        try:
            coef = base_lr.coef_[0]  # shape (D,)
            if feature_names and len(feature_names) == len(coef):
                abs_coef = np.abs(coef)
                top_idx = np.argsort(abs_coef)[::-1][:15]
                feature_importances = {
                    feature_names[i]: float(coef[i]) for i in top_idx
                }
        except Exception:
            pass

        metrics = MetaLabelMetrics(
            accuracy=float(accuracy_score(y_val, y_pred)),
            precision=float(precision_score(y_val, y_pred, zero_division=0)),
            recall=float(recall_score(y_val, y_pred, zero_division=0)),
            f1=float(f1_score(y_val, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_val, y_proba))
            if len(np.unique(y_val)) > 1
            else 0.5,
            brier=float(brier_score_loss(y_val, y_proba)),
            n_samples=n,
            class_balance=float(y.mean()),
            profit_threshold=calibrated_threshold,
            passed_rate=float(passed_mask.mean()),
            profit_rate_passed=profit_rate_passed,
            improvement_vs_baseline=improvement,
            cv_auc_mean=cv_auc_mean,
            cv_auc_std=cv_auc_std,
            feature_importances=feature_importances,
        )

        if progress_callback:
            progress_callback(5, n_steps)

        # ── STEP 6: Save + display ────────────────────────────────────────────
        self.is_trained = True
        self.input_dim = X.shape[1]
        self.feature_names = feature_names
        self.last_metrics = metrics
        self.n_trades_at_last_train = n
        self._save()
        self._log_tensorboard(metrics)


        _print_metrics_panel(metrics)

        logger.info(
            f"[MetaLabel] ✅ Treinado: n={n} | CV-AUC={cv_auc_mean:.3f}±{cv_auc_std:.3f} "
            f"| val-AUC={metrics.roc_auc:.3f} | threshold={calibrated_threshold:.3f} "
            f"| pass_rate={metrics.passed_rate:.1%} | improvement={improvement:+.1%}"
        )

        if progress_callback:
            progress_callback(n_steps, n_steps)

        return metrics

    def _log_tensorboard(self, metrics: MetaLabelMetrics):
        """Log meta-labeling metrics to TensorBoard."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return
        tb = getattr(self, "_tb_writer", None)
        if tb is None:
            try:
                import os, datetime

                log_dir = os.path.join(os.getcwd(), "logs", "tensorboard", "MetaLabel")
                self._tb_writer = SummaryWriter(log_dir)
                tb = self._tb_writer
                self._tb_step = 0
            except Exception:
                return
        step = getattr(self, "_tb_step", 0)
        self._tb_step = step + 1

        tb.add_scalar("MetaLabel/AUC", metrics.roc_auc, step)
        tb.add_scalar("MetaLabel/F1", metrics.f1, step)
        tb.add_scalar("MetaLabel/Precision", metrics.precision, step)
        tb.add_scalar("MetaLabel/Recall", metrics.recall, step)
        tb.add_scalar("MetaLabel/Brier", metrics.brier, step)
        tb.add_scalar("MetaLabel/Threshold", metrics.profit_threshold, step)
        tb.add_scalar("MetaLabel/PassRate", metrics.passed_rate, step)
        tb.add_scalar("MetaLabel/ProfitPassed", metrics.profit_rate_passed, step)
        tb.add_scalar("MetaLabel/Improvement", metrics.improvement_vs_baseline, step)
        tb.add_scalar("MetaLabel/NSamples", metrics.n_samples, step)
        tb.add_scalar("MetaLabel/ClassBalance", metrics.class_balance, step)
        tb.add_scalar("MetaLabel/CV_AUC_Mean", metrics.cv_auc_mean, step)
        tb.add_scalar("MetaLabel/CV_AUC_Std", metrics.cv_auc_std, step)

        # Feature importance (top 10)
        if metrics.feature_importances:
            top = sorted(
                metrics.feature_importances.items(), key=lambda x: x[1], reverse=True
            )[:10]
            for name, imp in top:
                safe_name = name.replace("/", "_").replace(" ", "_")[:50]
                tb.add_scalar(f"MetaLabel/FeatureImportance/{safe_name}", imp, step)


    def train_from_trade_log(
        self,
        featured_df: pd.DataFrame,
        trade_log: List[Dict[str, Any]],
        progress_callback=None,
    ) -> Optional[MetaLabelMetrics]:
        """
        Train from a DataFrame of features + list of closed trade dicts.

        Trade dict must have:
            entry_time  (str or datetime) — timestamp of trade entry
            pnl         (float)           — realized PnL in quote currency
            notional    (float, optional) — notional value for fee-adjusted label

        Args:
            featured_df: DataFrame with market features (datetime index)
            trade_log:   List of closed trade dicts
            progress_callback: Optional callable(step, total)
        """
        if not trade_log:
            logger.warning("[MetaLabel] Trade log vazio. Treino abortado.")
            return None

        feature_cols = [
            c
            for c in featured_df.columns
            if c
            not in (
                "open",
                "high",
                "low",
                "close",
                "volume",
                "timestamp",
                "date",
                "entry_time",
                "exit_time",
                "pnl",
                "regime",
                "regime_name",
            )
            and featured_df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]

        if self.feature_names:
            valid = [c for c in self.feature_names if c in featured_df.columns]
            if valid:
                feature_cols = valid

        X_list, pnl_list, notional_list = [], [], []

        for trade in trade_log:
            try:
                # Support both "entry_time" and "timestamp" keys
                entry_ts = pd.to_datetime(
                trade.get("entry_time") or trade.get("timestamp")
                )
                pnl = float(trade.get("pnl", 0))

                idx = featured_df.index.get_indexer([entry_ts], method="pad")
                if idx[0] < 0:
                    continue
                row = featured_df.iloc[idx[0]]
                feat = row[feature_cols].values.astype(np.float64)
                if np.isnan(feat).any():
                    continue

                X_list.append(feat)
                pnl_list.append(pnl)
                notional_list.append(float(trade.get("notional_value", 0) or 0))
            except (KeyError, IndexError, ValueError):
                continue

        if len(X_list) < self.min_samples:
            logger.warning(
                f"[MetaLabel] {len(X_list)} amostras válidas < min={self.min_samples}."
            )
            return None

        X = np.array(X_list, dtype=np.float64)
        pnls = np.array(pnl_list, dtype=np.float64)
        notionals = np.array(notional_list, dtype=np.float64)

        return self.train(
            X,
            pnls,
            notionals=notionals,
            feature_names=feature_cols,
            progress_callback=progress_callback,
        )

    # ── INFERENCE ─────────────────────────────────────────────────────────────

    def predict(self, entry_features: np.ndarray) -> float:
        """
        Predict P(profit | entry_features).

        Returns:
            Probability in [0, 1]. Returns 0.5 if model not trained (neutral).
        """
        if not self.is_trained or self.model is None:
            return 0.5

        X = np.asarray(entry_features, dtype=np.float64).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except ValueError:
                return 0.5

        proba = float(self.model.predict_proba(X)[0, 1])
        return float(np.clip(proba, 0.0, 1.0))

    def should_execute(self, entry_features: np.ndarray) -> Tuple[bool, float]:
        """
        Check if trade should be executed based on meta-label.

        Returns:
            (should_execute: bool, probability: float)
        """
        prob = self.predict(entry_features)
        return prob >= self.profit_threshold, prob

    def needs_retraining(
        self, new_trades: int = 0, retrain_interval: int = 200
    ) -> bool:
        """
        True if enough new trades have accumulated to justify retraining.
        Lopez de Prado (2018) Ch.4: meta-labeler should be updated continuously.
        """
        delta = (self.n_trades_seen + new_trades) - self.n_trades_at_last_train
        return delta >= retrain_interval

    # ── ADAPT (async compatibility for ai_controller) ─────────────────────────

    async def adapt(
        self,
        recent_data: Any = None,
        adaptation_strength: float = 0.5,
        preserve_core_learning: bool = True,
    ) -> bool:
        """
        Async adapter for ai_controller periodic adaptation loop.

        Full retraining is triggered externally via train_from_trade_log().
        This method is a no-op stub to satisfy the interface contract.
        """
        logger.debug(
            "[MetaLabel] adapt() chamado — retraining via trade_log separadamente."
        )
        return True

    # ── PERSISTENCE ───────────────────────────────────────────────────────────

    def _save(self):
        path = os.path.join(self.model_dir, "profitability_predictor.pth")
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "input_dim": self.input_dim,
                "last_metrics": self.last_metrics,
                "profit_threshold": self.profit_threshold,
                "n_trades_at_last_train": self.n_trades_at_last_train,
                "taker_fee": self.taker_fee,
            },
            path,
        )
        logger.debug(f"[MetaLabel] Modelo salvo em {path}")

    @classmethod
    def load(cls, model_dir: Optional[str] = None, **kwargs) -> "ProfitabilityPredictor":
        model_dir = model_dir or os.path.join(os.getcwd(), "models_ai")
        path = os.path.join(model_dir, "profitability_predictor.pth")
        obj = cls(model_dir=model_dir, **kwargs)
        if os.path.exists(path):
            try:
                data = joblib.load(path)
                obj.model = data.get("model")
                obj.scaler = data.get("scaler")
                obj.feature_names = data.get("feature_names")
                obj.input_dim = data.get("input_dim")
                obj.last_metrics = data.get("last_metrics")
                obj.profit_threshold = data.get(
                    "profit_threshold", obj.profit_threshold
                )
                obj.n_trades_at_last_train = data.get("n_trades_at_last_train", 0)
                obj.taker_fee = data.get("taker_fee", obj.taker_fee)
                obj.is_trained = obj.model is not None
                if obj.is_trained and obj.last_metrics:
                    m = obj.last_metrics
                    logger.info(
                        f"[MetaLabel] Carregado: auc={m.roc_auc:.3f} | "
                        f"threshold={obj.profit_threshold:.3f} | n={m.n_samples}"
                    )
            except Exception as e:
                logger.warning(f"[MetaLabel] Falha ao carregar: {e}")
        return obj

    def save_model(self, path: str = None):
        self._save()

    def load_model(self, path: str = None) -> "ProfitabilityPredictor":
        p = path or os.path.join(self.model_dir, "profitability_predictor.pth")
        if os.path.exists(p):
            try:
                data = joblib.load(p)
                self.model = data.get("model")
                self.scaler = data.get("scaler")
                self.feature_names = data.get("feature_names")
                self.input_dim = data.get("input_dim")
                self.last_metrics = data.get("last_metrics")
                self.profit_threshold = data.get(
                    "profit_threshold", self.profit_threshold
                )
                self.n_trades_at_last_train = data.get("n_trades_at_last_train", 0)
                self.taker_fee = data.get("taker_fee", self.taker_fee)
                self.is_trained = self.model is not None
            except Exception as e:
                logger.warning(f"[MetaLabel] load_model falhou: {e}")
        return self
