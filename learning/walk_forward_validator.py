# -----------------------------------------------------------------------------
# ARQUIVO: learning/walk_forward_validator.py
# -----------------------------------------------------------------------------
"""
Walk-Forward Analysis Framework for Scientific Hyperparameter Optimization.

This module implements proper nested cross-validation for financial time series,
avoiding look-ahead bias and providing robust out-of-sample validation.

Scientific References:
- Bailey et al. (2017) "The Probability of Backtest Overfitting"
- Pardo (2011) "The Evaluation and Optimization of Trading Strategies"
- López de Prado (2018) "Advances in Financial Machine Learning"
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Iterator, Callable
from dataclasses import dataclass
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, MedianPruner

from utils.logger import get_logger

logger = get_logger("WalkForwardValidator")


@dataclass
class WalkForwardFold:
    """Represents a single Walk-Forward fold."""
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    gap_size: int
    
    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start
        
    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


@dataclass 
class WalkForwardMetrics:
    """Aggregated metrics from Walk-Forward validation."""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    n_folds: int
    fold_metrics: List[Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'avg_trade_duration': self.avg_trade_duration,
            'n_folds': self.n_folds
        }


class WalkForwardValidator:
    """
    Scientific Walk-Forward Analysis for Hyperparameter Optimization.
    
    Key features:
    - Rolling window cross-validation with temporal ordering
    - Embargo period to prevent information leakage
    - Aggregated metrics across folds
    - Support for multiple objective functions
    
    Timeline visualization:
    |---Train_1---|gap|--Test_1--|
            |---Train_2---|gap|--Test_2--|
                    |---Train_3---|gap|--Test_3--|
                            ...
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.7,  # Fraction of available data for training
        gap_periods: int = 50,  # Embargo period between train and test
        min_train_samples: int = 1000,
        min_test_samples: int = 200,
        expanding_window: bool = False  # If True, train window grows; if False, slides
    ):
        """
        Args:
            n_splits: Number of Walk-Forward folds
            train_ratio: Ratio of training data within each fold
            gap_periods: Number of periods between train end and test start (embargo)
            min_train_samples: Minimum samples required in training set
            min_test_samples: Minimum samples required in test set
            expanding_window: If True, use expanding window (anchor start); else sliding
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap_periods = gap_periods
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        self.expanding_window = expanding_window
        
        logger.info(
            f"✅ WalkForwardValidator initialized: "
            f"n_splits={n_splits}, train_ratio={train_ratio}, "
            f"gap={gap_periods}, expanding={expanding_window}"
        )
        
    def split(self, df: pd.DataFrame) -> Iterator[Tuple[WalkForwardFold, pd.DataFrame, pd.DataFrame]]:
        """
        Generate Walk-Forward train/test splits.
        
        Yields:
            Tuple of (WalkForwardFold, train_df, test_df)
        """
        n = len(df)
        
        # Calculate test size for each fold
        total_test_size = int(n * (1 - self.train_ratio))
        test_size_per_fold = max(self.min_test_samples, total_test_size // self.n_splits)
        
        for fold_idx in range(self.n_splits):
            # Calculate test window position (from the end, backwards)
            test_end = n - (self.n_splits - fold_idx - 1) * test_size_per_fold
            test_start = test_end - test_size_per_fold
            
            # Calculate train window (before test, with gap)
            train_end = test_start - self.gap_periods
            
            if self.expanding_window:
                # Expanding: always start from beginning
                train_start = 0
            else:
                # Sliding: maintain fixed train size
                train_size = int((test_start - self.gap_periods) * self.train_ratio)
                train_start = max(0, train_end - train_size)
            
            # Validate sizes
            if train_end - train_start < self.min_train_samples:
                logger.warning(
                    f"⚠️ Fold {fold_idx}: Insufficient training data "
                    f"({train_end - train_start} < {self.min_train_samples}). Skipping."
                )
                continue
                
            if test_end - test_start < self.min_test_samples:
                logger.warning(
                    f"⚠️ Fold {fold_idx}: Insufficient test data "
                    f"({test_end - test_start} < {self.min_test_samples}). Skipping."
                )
                continue
                
            fold = WalkForwardFold(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                gap_size=self.gap_periods
            )
            
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            logger.debug(
                f"📊 Fold {fold_idx}: Train[{train_start}:{train_end}] ({len(train_df)}), "
                f"Test[{test_start}:{test_end}] ({len(test_df)})"
            )
            
            yield fold, train_df, test_df
            
    def evaluate(
        self,
        df: pd.DataFrame,
        train_fn: Callable[[pd.DataFrame, Dict[str, Any]], Any],
        eval_fn: Callable[[Any, pd.DataFrame], Dict[str, float]],
        hyperparams: Dict[str, Any]
    ) -> WalkForwardMetrics:
        """
        Perform Walk-Forward evaluation with given hyperparameters.
        
        Args:
            df: Full dataset
            train_fn: Function(train_df, hyperparams) -> trained_model
            eval_fn: Function(model, test_df) -> dict of metrics
            hyperparams: Hyperparameters to evaluate
            
        Returns:
            WalkForwardMetrics with aggregated results
        """
        fold_metrics = []
        
        for fold, train_df, test_df in self.split(df):
            try:
                # Train on this fold
                model = train_fn(train_df, hyperparams)
                
                # Evaluate on test set
                metrics = eval_fn(model, test_df)
                metrics['fold_idx'] = fold.fold_idx
                fold_metrics.append(metrics)
                
                logger.info(
                    f"📈 Fold {fold.fold_idx}: Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
                    f"Sortino={metrics.get('sortino_ratio', 0):.3f}, "
                    f"MaxDD={metrics.get('max_drawdown', 0):.2%}"
                )
                
            except Exception as e:
                logger.error(f"❌ Fold {fold.fold_idx} failed: {e}", exc_info=True)
                # Add penalty metrics for failed fold
                fold_metrics.append({
                    'fold_idx': fold.fold_idx,
                    'sharpe_ratio': -2.0,
                    'sortino_ratio': -2.0,
                    'max_drawdown': 1.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'error': str(e)
                })
                
        return self._aggregate_metrics(fold_metrics)
        
    def _aggregate_metrics(self, fold_metrics: List[Dict[str, float]]) -> WalkForwardMetrics:
        """
        Aggregate metrics across folds.
        
        Uses:
        - Mean for risk-adjusted returns (Sharpe, Sortino, Calmar)
        - WORST CASE for Max Drawdown (conservative)
        - Weighted average by trades for win rate
        """
        if not fold_metrics:
            return WalkForwardMetrics(
                sharpe_ratio=-999, sortino_ratio=-999, max_drawdown=1.0,
                calmar_ratio=-999, win_rate=0, profit_factor=0,
                total_trades=0, avg_trade_duration=0, n_folds=0,
                fold_metrics=[]
            )
            
        # Extract arrays
        sharpe = [m.get('sharpe_ratio', 0) for m in fold_metrics]
        sortino = [m.get('sortino_ratio', 0) for m in fold_metrics]
        max_dd = [m.get('max_drawdown', 0) for m in fold_metrics]
        calmar = [m.get('calmar_ratio', 0) for m in fold_metrics]
        win_rate = [m.get('win_rate', 0) for m in fold_metrics]
        pf = [m.get('profit_factor', 0) for m in fold_metrics]
        trades = [m.get('num_trades', 0) for m in fold_metrics]
        duration = [m.get('avg_trade_duration', 0) for m in fold_metrics]
        
        total_trades = sum(trades)
        
        # Weighted averages where appropriate
        if total_trades > 0:
            weights = [t / total_trades for t in trades]
            weighted_win_rate = sum(w * wr for w, wr in zip(weights, win_rate))
            weighted_duration = sum(w * d for w, d in zip(weights, duration))
        else:
            weighted_win_rate = np.mean(win_rate) if win_rate else 0
            weighted_duration = np.mean(duration) if duration else 0
            
        return WalkForwardMetrics(
            sharpe_ratio=float(np.mean(sharpe)),
            sortino_ratio=float(np.mean(sortino)),
            max_drawdown=float(np.max(max_dd)),  # Worst case
            calmar_ratio=float(np.mean(calmar)),
            win_rate=float(weighted_win_rate),
            profit_factor=float(np.mean(pf)),
            total_trades=int(total_trades),
            avg_trade_duration=float(weighted_duration),
            n_folds=len(fold_metrics),
            fold_metrics=fold_metrics
        )


class WalkForwardOptimizer:
    """
    Optuna-based hyperparameter optimization using Walk-Forward validation.
    
    This ensures that HPO doesn't overfit to a single train/test split.
    """
    
    def __init__(
        self,
        validator: WalkForwardValidator,
        objective_metric: str = 'sortino_ratio',
        direction: str = 'maximize',
        n_trials: int = 100,
        timeout: Optional[int] = 3600,  # 1 hour default
        study_name: str = 'walk_forward_hpo',
        storage_path: Optional[str] = None,
        n_jobs: int = 1
    ):
        """
        Args:
            validator: WalkForwardValidator instance
            objective_metric: Which metric to optimize
            direction: 'maximize' or 'minimize'
            n_trials: Number of Optuna trials
            timeout: Maximum optimization time in seconds
            study_name: Optuna study name
            storage_path: Path to SQLite database for persistence
            n_jobs: Number of parallel trials
        """
        self.validator = validator
        self.objective_metric = objective_metric
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.n_jobs = n_jobs
        
        if storage_path:
            self.storage = f"sqlite:///{storage_path}"
        else:
            self.storage = None
            
        # Samplers for efficient search
        self.sampler = TPESampler(
            n_startup_trials=10,
            multivariate=True,
            group=True
        )
        
        # Pruner for early stopping of bad trials
        self.pruner = HyperbandPruner(
            min_resource=1,
            max_resource=self.validator.n_splits,
            reduction_factor=3
        )
        
        logger.info(
            f"✅ WalkForwardOptimizer initialized: "
            f"objective={objective_metric}, direction={direction}, n_trials={n_trials}"
        )
        
    def optimize(
        self,
        df: pd.DataFrame,
        train_fn: Callable[[pd.DataFrame, Dict[str, Any]], Any],
        eval_fn: Callable[[Any, pd.DataFrame], Dict[str, float]],
        search_space: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], WalkForwardMetrics]:
        """
        Run optimization with Walk-Forward validation.
        
        Args:
            df: Full dataset
            train_fn: Training function
            eval_fn: Evaluation function
            search_space: Optuna search space definition
                Example: {
                    'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-3},
                    'batch_size': {'type': 'categorical', 'choices': [128, 256, 512]}
                }
                
        Returns:
            Tuple of (best_hyperparams, best_metrics)
        """
        
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            hyperparams = self._sample_hyperparams(trial, search_space)
            
            try:
                # Run Walk-Forward evaluation
                metrics = self.validator.evaluate(df, train_fn, eval_fn, hyperparams)
                
                # Report intermediate values for pruning
                trial.report(getattr(metrics, self.objective_metric), step=metrics.n_folds)
                
                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
                return getattr(metrics, self.objective_metric)
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"❌ Trial {trial.number} failed: {e}")
                return -999 if self.direction == 'maximize' else 999
                
        # Create or load study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True
        )
        
        logger.info(f"🔬 Starting optimization: {self.n_trials} trials, {self.timeout}s timeout")
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # Get best results
        best_params = study.best_params
        best_metrics = self.validator.evaluate(df, train_fn, eval_fn, best_params)
        
        logger.info(
            f"✅ Optimization complete. Best {self.objective_metric}: "
            f"{getattr(best_metrics, self.objective_metric):.4f}"
        )
        logger.info(f"📊 Best params: {best_params}")
        
        return best_params, best_metrics
        
    def _sample_hyperparams(
        self, 
        trial: optuna.Trial, 
        search_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sample hyperparameters according to search space definition."""
        params = {}
        
        for name, config in search_space.items():
            param_type = config.get('type', 'uniform')
            
            if param_type == 'uniform':
                params[name] = trial.suggest_float(
                    name, config['low'], config['high']
                )
            elif param_type == 'loguniform':
                params[name] = trial.suggest_float(
                    name, config['low'], config['high'], log=True
                )
            elif param_type == 'int':
                params[name] = trial.suggest_int(
                    name, config['low'], config['high']
                )
            elif param_type == 'categorical':
                params[name] = trial.suggest_categorical(
                    name, config['choices']
                )
            elif param_type == 'discrete_uniform':
                params[name] = trial.suggest_float(
                    name, config['low'], config['high'], step=config.get('step', 1)
                )
            else:
                logger.warning(f"Unknown param type {param_type} for {name}")
                
        return params


# Convenience function for quick validation
def quick_walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    gap: int = 50
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Simple function to get Walk-Forward splits without full validator.
    """
    validator = WalkForwardValidator(n_splits=n_splits, gap_periods=gap)
    return [(train, test) for _, train, test in validator.split(df)]
