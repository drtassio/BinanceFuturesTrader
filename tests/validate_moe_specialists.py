# -----------------------------------------------------------------------------
# ARQUIVO: tests/validate_moe_specialists.py
# -----------------------------------------------------------------------------
"""
Scientific Validation Suite for MoE Specialists.

This script provides comprehensive validation tests to verify:
1. Regime detector accuracy against historical price action
2. Specialist learning quality via profit metrics
3. Statistical significance of predictions

Usage:
    python tests/validate_moe_specialists.py
    python tests/validate_moe_specialists.py --regime-only
    python tests/validate_moe_specialists.py --specialist bull
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

from utils.logger import get_logger

logger = get_logger("MoEValidator")


# ============================================================================
# REGIME DETECTOR VALIDATION
# ============================================================================

class RegimeDetectorValidator:
    """
    Validates ScientificRegimeDetector accuracy.
    
    Methodology:
    1. Label ground truth regimes based on future returns
    2. Compare detector predictions vs ground truth
    3. Calculate precision, recall, F1 per regime
    4. Analyze regime transition accuracy
    """
    
    def __init__(self, forward_window: int = 20, return_threshold: float = 0.02):
        """
        Args:
            forward_window: Bars to look forward for ground truth
            return_threshold: Min return to classify as bull/bear (2% default)
        """
        self.forward_window = forward_window
        self.return_threshold = return_threshold
        
    def create_ground_truth(self, df: pd.DataFrame) -> np.ndarray:
        """
        Creates ground truth regime labels based on future price action.
        
        Ground Truth Logic:
        - Bull (0): Forward return > threshold
        - Bear (1): Forward return < -threshold
        - Ranger (2): |Forward return| <= threshold
        """
        # Calculate forward returns
        forward_returns = df['close'].pct_change(self.forward_window).shift(-self.forward_window)
        
        # Create labels
        labels = np.full(len(df), 2)  # Default = Ranger
        labels[forward_returns > self.return_threshold] = 0   # Bull
        labels[forward_returns < -self.return_threshold] = 1  # Bear
        
        return labels
        
    def validate(self, df: pd.DataFrame, predictions: np.ndarray) -> Dict[str, float]:
        """
        Validates detector predictions against ground truth.
        
        Returns:
            Dict with accuracy, per-class precision/recall/F1, and analysis
        """
        ground_truth = self.create_ground_truth(df)
        
        # Remove NaN rows from ground truth (forward returns calculation)
        valid_mask = ~np.isnan(ground_truth.astype(float))
        gt_valid = ground_truth[valid_mask].astype(int)
        pred_valid = predictions[valid_mask].astype(int)
        
        # Overall accuracy
        accuracy = (gt_valid == pred_valid).mean()
        
        # Per-class metrics
        regime_names = ['bull', 'bear', 'ranger']
        metrics = {'accuracy': accuracy}
        
        for i, name in enumerate(regime_names):
            tp = np.sum((pred_valid == i) & (gt_valid == i))
            fp = np.sum((pred_valid == i) & (gt_valid != i))
            fn = np.sum((pred_valid != i) & (gt_valid == i))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[f'{name}_precision'] = precision
            metrics[f'{name}_recall'] = recall
            metrics[f'{name}_f1'] = f1
            
        # Confusion matrix (simplified)
        confusion = np.zeros((3, 3), dtype=int)
        for gt, pred in zip(gt_valid, pred_valid):
            confusion[gt, pred] += 1
        metrics['confusion_matrix'] = confusion.tolist()
        
        # Regime transition accuracy (how well does it detect regime changes?)
        gt_changes = np.diff(gt_valid) != 0
        pred_changes = np.diff(pred_valid) != 0
        transition_agreement = (gt_changes == pred_changes).mean()
        metrics['transition_accuracy'] = transition_agreement
        
        return metrics
        
    def print_report(self, metrics: Dict[str, float]):
        """Prints formatted validation report."""
        print("\n" + "=" * 70)
        print("📊 REGIME DETECTOR VALIDATION REPORT")
        print("=" * 70)
        
        print(f"\n🎯 Overall Accuracy: {metrics['accuracy']:.2%}")
        print(f"🔄 Transition Accuracy: {metrics['transition_accuracy']:.2%}")
        
        print("\n📈 Per-Regime Metrics:")
        print("-" * 50)
        print(f"{'Regime':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 50)
        
        for regime in ['bull', 'bear', 'ranger']:
            p = metrics[f'{regime}_precision']
            r = metrics[f'{regime}_recall']
            f1 = metrics[f'{regime}_f1']
            print(f"{regime.upper():<12} {p:.2%}        {r:.2%}        {f1:.2%}")
            
        print("\n📊 Confusion Matrix:")
        print("         Predicted")
        print("         Bull  Bear  Ranger")
        confusion = np.array(metrics['confusion_matrix'])
        for i, name in enumerate(['Bull  ', 'Bear  ', 'Ranger']):
            row = confusion[i]
            print(f"True {name}: {row[0]:5d} {row[1]:5d} {row[2]:5d}")
            
        # Quality assessment
        print("\n📋 Quality Assessment:")
        avg_f1 = (metrics['bull_f1'] + metrics['bear_f1'] + metrics['ranger_f1']) / 3
        
        if avg_f1 > 0.7:
            print("   ✅ EXCELLENT: Detector is performing well (F1 > 70%)")
        elif avg_f1 > 0.5:
            print("   ⚠️ MODERATE: Detector needs improvement (F1 50-70%)")
        else:
            print("   ❌ POOR: Detector is unreliable (F1 < 50%)")
            
        if metrics['transition_accuracy'] < 0.6:
            print("   ⚠️ WARNING: Low transition accuracy - may miss regime changes")
            
        print("=" * 70 + "\n")


# ============================================================================
# SPECIALIST LEARNING VALIDATION
# ============================================================================

class SpecialistValidator:
    """
    Validates specialist learning quality.
    
    Methodology:
    1. Out-of-sample testing on held-out data
    2. Profit-based metrics (not just loss)
    3. Statistical significance tests
    4. Regime-specific performance analysis
    """
    
    def __init__(self, min_trades: int = 30, confidence_level: float = 0.95):
        self.min_trades = min_trades
        self.confidence_level = confidence_level
        
    def simulate_trades(
        self, 
        specialist, 
        df: pd.DataFrame, 
        initial_capital: float = 10000.0
    ) -> Dict[str, float]:
        """
        Simulates trading with the specialist and calculates metrics.
        """
        from specialists.trend_specialist import TrendFollowingEnv
        
        # Create environment
        env = TrendFollowingEnv(df)
        
        # Track trades
        trades = []
        capital = initial_capital
        position = 0
        entry_price = 0
        
        # Run simulation
        obs = env.reset()
        done = False
        
        while not done:
            # Get action from specialist
            action, _ = specialist.predict(obs, deterministic=True)
            
            # Execute
            obs, reward, done, info = env.step(action)
            
            # Track trade
            if action != 0 and position == 0:  # Entry
                position = 1 if action > 0 else -1
                entry_price = info.get('price', df.iloc[env.current_step]['close'])
            elif action == 0 and position != 0:  # Exit
                exit_price = info.get('price', df.iloc[env.current_step]['close'])
                pnl = position * (exit_price - entry_price) / entry_price
                trades.append({
                    'pnl': pnl,
                    'direction': 'long' if position > 0 else 'short',
                    'entry': entry_price,
                    'exit': exit_price
                })
                capital *= (1 + pnl)
                position = 0
                
        # Calculate metrics
        if len(trades) < self.min_trades:
            return {
                'sufficient_trades': False,
                'n_trades': len(trades),
                'message': f'Insufficient trades ({len(trades)} < {self.min_trades})'
            }
            
        pnls = [t['pnl'] for t in trades]
        
        # Basic metrics
        total_return = (capital - initial_capital) / initial_capital
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
        
        # Risk metrics
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
        
        # Drawdown
        cumulative = np.cumprod(1 + np.array(pnls))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Statistical significance (t-test against 0)
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(pnls, 0)
        is_significant = p_value < (1 - self.confidence_level)
        
        return {
            'sufficient_trades': True,
            'n_trades': len(trades),
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        }
        
    def validate_learning(self, metrics: Dict[str, float]) -> Dict[str, any]:
        """
        Evaluates if specialist has learned effectively.
        
        Returns validation verdict with reasons.
        """
        if not metrics.get('sufficient_trades', False):
            return {
                'verdict': 'INSUFFICIENT_DATA',
                'passed': False,
                'reasons': [metrics.get('message', 'Not enough trades')]
            }
            
        reasons = []
        passed = True
        
        # Check 1: Positive returns
        if metrics['total_return'] <= 0:
            reasons.append(f"❌ Negative total return: {metrics['total_return']:.2%}")
            passed = False
        else:
            reasons.append(f"✅ Positive return: {metrics['total_return']:.2%}")
            
        # Check 2: Win rate above 45%
        if metrics['win_rate'] < 0.45:
            reasons.append(f"⚠️ Low win rate: {metrics['win_rate']:.2%}")
            passed = False
        else:
            reasons.append(f"✅ Good win rate: {metrics['win_rate']:.2%}")
            
        # Check 3: Positive Sharpe
        if metrics['sharpe_ratio'] < 0.5:
            reasons.append(f"⚠️ Low Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            passed = False
        else:
            reasons.append(f"✅ Good Sharpe: {metrics['sharpe_ratio']:.2f}")
            
        # Check 4: Statistical significance
        if not metrics['is_significant']:
            reasons.append(f"⚠️ Results not statistically significant (p={metrics['p_value']:.3f})")
        else:
            reasons.append(f"✅ Statistically significant (p={metrics['p_value']:.3f})")
            
        # Check 5: Acceptable drawdown
        if metrics['max_drawdown'] < -0.30:
            reasons.append(f"⚠️ High max drawdown: {metrics['max_drawdown']:.2%}")
        else:
            reasons.append(f"✅ Acceptable drawdown: {metrics['max_drawdown']:.2%}")
            
        return {
            'verdict': 'PASSED' if passed else 'NEEDS_IMPROVEMENT',
            'passed': passed,
            'reasons': reasons,
            'metrics': metrics
        }
        
    def print_report(self, specialist_name: str, validation: Dict):
        """Prints formatted specialist validation report."""
        print("\n" + "=" * 70)
        print(f"📊 {specialist_name.upper()} SPECIALIST VALIDATION REPORT")
        print("=" * 70)
        
        verdict = validation['verdict']
        if verdict == 'PASSED':
            print(f"\n🎉 Verdict: ✅ PASSED - Specialist has learned effectively!")
        elif verdict == 'NEEDS_IMPROVEMENT':
            print(f"\n⚠️ Verdict: NEEDS IMPROVEMENT - Some metrics below threshold")
        else:
            print(f"\n❌ Verdict: {verdict}")
            
        print("\n📋 Validation Checks:")
        for reason in validation['reasons']:
            print(f"   {reason}")
            
        if 'metrics' in validation:
            m = validation['metrics']
            print(f"\n📈 Key Metrics:")
            print(f"   • Total Return: {m.get('total_return', 0):.2%}")
            print(f"   • Win Rate: {m.get('win_rate', 0):.2%}")
            print(f"   • Sharpe Ratio: {m.get('sharpe_ratio', 0):.2f}")
            print(f"   • Max Drawdown: {m.get('max_drawdown', 0):.2%}")
            print(f"   • Number of Trades: {m.get('n_trades', 0)}")
            
        print("=" * 70 + "\n")


# ============================================================================
# TENSORBOARD VS REAL METRICS
# ============================================================================

class TensorBoardTrustAnalyzer:
    """
    Analyzes reliability of TensorBoard metrics.
    
    TensorBoard shows TRAINING metrics, not TRADING metrics.
    This class helps understand the gap.
    """
    
    @staticmethod
    def explain_metrics():
        """Explains what TensorBoard metrics mean vs real performance."""
        
        explanation = """
╔══════════════════════════════════════════════════════════════════════╗
║              TENSORBOARD METRICS vs REAL TRADING PERFORMANCE         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  ⚠️  TensorBoard metrics ≠ Trading performance!                      ║
║                                                                       ║
║  What TensorBoard shows (Training Metrics):                          ║
║  ─────────────────────────────────────────                           ║
║  • loss: Neural network training loss (lower = better fit)           ║
║  • accuracy: Classification accuracy on regime labels                ║
║  • entropy: Prediction uncertainty (higher = more exploration)       ║
║  • episode_reward: Reward in simulated environment                   ║
║                                                                       ║
║  What REALLY matters (Trading Metrics):                              ║
║  ─────────────────────────────────────                               ║
║  • Sharpe Ratio: Risk-adjusted returns                               ║
║  • Win Rate: Percentage of profitable trades                         ║
║  • Max Drawdown: Largest peak-to-trough decline                      ║
║  • Profit Factor: Gross profit / Gross loss                          ║
║  • Statistical Significance: Results not due to chance               ║
║                                                                       ║
║  🔍 KEY INSIGHT:                                                      ║
║  A model with LOW loss can still LOSE money!                         ║
║  Always validate with OUT-OF-SAMPLE trading simulation.              ║
║                                                                       ║
║  ✅ RECOMMENDED VALIDATION WORKFLOW:                                  ║
║  1. Train model (watch TensorBoard for convergence)                  ║
║  2. Run this validation script on held-out data                      ║
║  3. Check statistical significance of results                        ║
║  4. Only deploy if validation passes                                 ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""
        print(explanation)


# ============================================================================
# MAIN VALIDATION RUNNER
# ============================================================================

def run_regime_detector_validation(df: pd.DataFrame):
    """Run regime detector validation."""
    from feature_engineering.regime_detector import ScientificRegimeDetector
    
    print("\n🔍 Running Regime Detector Validation...")
    
    detector = ScientificRegimeDetector()
    predictions = detector.detect_regimes(df)
    
    validator = RegimeDetectorValidator(forward_window=20, return_threshold=0.02)
    metrics = validator.validate(df, predictions)
    validator.print_report(metrics)
    
    return metrics


def run_specialist_validation(specialist_name: str, df: pd.DataFrame):
    """Run specialist validation."""
    print(f"\n🔍 Running {specialist_name} Specialist Validation...")
    
    # Load specialist
    from specialists.bull_specialist import BullSpecialist
    from specialists.bear_specialist import BearSpecialist
    from specialists.ranger_specialist import RangerSpecialist
    from config.settings import AIConfig, TradingConfig
    
    specialists = {
        'bull': BullSpecialist,
        'bear': BearSpecialist,
        'ranger': RangerSpecialist
    }
    
    if specialist_name not in specialists:
        print(f"❌ Unknown specialist: {specialist_name}")
        return None
        
    config = AIConfig()
    trading_config = TradingConfig()
    
    specialist = specialists[specialist_name](
        config=config,
        trading_config=trading_config,
        input_dim=64
    )
    
    # Check if trained
    if not specialist.is_trained:
        print(f"⚠️ {specialist_name} specialist is not trained. Skipping validation.")
        return None
        
    validator = SpecialistValidator()
    
    # Split data for out-of-sample testing
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]
    
    metrics = validator.simulate_trades(specialist, test_df)
    validation = validator.validate_learning(metrics)
    validator.print_report(specialist_name, validation)
    
    return validation


def main():
    """Main validation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MoE Specialist Validation Suite")
    parser.add_argument('--regime-only', action='store_true', help='Only validate regime detector')
    parser.add_argument('--specialist', type=str, help='Validate specific specialist')
    parser.add_argument('--explain', action='store_true', help='Explain TensorBoard metrics')
    
    args = parser.parse_args()
    
    if args.explain:
        TensorBoardTrustAnalyzer.explain_metrics()
        return
        
    # Load data
    print("\n📊 Loading validation data...")
    try:
        from feature_engineering.temporal_autoencoder import TemporalAutoencoderPipeline
        from config.settings import AIConfig
        
        pipeline = TemporalAutoencoderPipeline(AIConfig())
        df = pipeline.get_featured_dataframe()
        
        if df is None or df.empty:
            print("❌ Could not load data for validation")
            return
            
        print(f"   Loaded {len(df):,} samples")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
        
    # Run validations
    print("\n" + "=" * 70)
    print("🧪 MOE SPECIALIST VALIDATION SUITE")
    print("=" * 70)
    
    # Always explain TensorBoard first
    TensorBoardTrustAnalyzer.explain_metrics()
    
    # Regime detector validation
    regime_metrics = run_regime_detector_validation(df)
    
    # Specialist validation
    if args.regime_only:
        return
        
    if args.specialist:
        run_specialist_validation(args.specialist, df)
    else:
        for name in ['bull', 'bear', 'ranger']:
            run_specialist_validation(name, df)
            
    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
