# -----------------------------------------------------------------------------
# ARQUIVO: scripts/train_moe_specialists.py
# -----------------------------------------------------------------------------
"""
Granular MoE Training Orchestrator with TensorBoard and Console Metrics.

This script handles the complete training pipeline for the Mixture of Experts
trading system. It automatically:
1. Checks which components need optimization/training
2. Trains the Soft Gating Network with TensorBoard logging
3. Trains each specialist with Walk-Forward validation
4. Optimizes hyperparameters via Optuna
5. Initializes online learning infrastructure

Usage:
    python scripts/train_moe_specialists.py --full          # Full training from scratch
    python scripts/train_moe_specialists.py --check         # Check status only
    python scripts/train_moe_specialists.py --specialist bull  # Train specific specialist
    python scripts/train_moe_specialists.py --optimize-only # Only run HPO
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
import time

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# TensorBoard for training visualization
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("⚠️ TensorBoard not available. Install with: pip install tensorboard")

from utils.logger import get_logger, LOG_LEVEL_DEBUG, setup_phase_logger
from config.settings import AIConfig, TradingConfig

# MoE Components
from specialists.soft_gating_network import SoftGatingNetwork, create_gating_network
from specialists.base_regime_specialist import BaseRegimeSpecialist
from specialists.bull_specialist import BullSpecialist
from specialists.bear_specialist import BearSpecialist
from specialists.ranger_specialist import RangerSpecialist

# Training infrastructure
from learning.walk_forward_validator import WalkForwardValidator, WalkForwardOptimizer
from learning.online_learner import OnlineTrendLearner, create_online_learner

# Data loading
from feature_engineering.temporal_autoencoder import TemporalAutoencoderPipeline

# 🔬 Dr. Tensor: New Ensemble Regime Detector (HMM + GMM + ADX + Funding)
from feature_engineering.crypto_regime_detector import CryptoRegimeDetector, RegimeConfig

logger = get_logger("MoETrainer", LOG_LEVEL_DEBUG)


# ============================================================================
# CONSOLE METRICS DISPLAY
# ============================================================================

class MetricsDisplay:
    """Rich console display for training metrics."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.start_time = time.time()
        
    def print_header(self, title: str):
        print("\n" + "=" * 70)
        print(f"📊 {title}")
        print("=" * 70)
        
    def print_epoch_metrics(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        elapsed = time.time() - self.start_time
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"[Epoch {epoch:3d}/{total_epochs}] {elapsed:6.1f}s | {metrics_str}")
        
    def print_summary(self, final_metrics: Dict[str, float]):
        elapsed = time.time() - self.start_time
        print("\n" + "-" * 70)
        print(f"✅ {self.component_name} Complete!")
        print(f"   Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        for k, v in final_metrics.items():
            if isinstance(v, float):
                print(f"   • {k}: {v:.4f}")
        print("-" * 70 + "\n")
        
    def print_status_table(self, status: Dict[str, Any]):
        print("\n╔" + "═" * 68 + "╗")
        print(f"║ {'Component':<25} {'Trained':<12} {'Optimized':<12} {'Last Update':<17} ║")
        print("╠" + "═" * 68 + "╣")
        
        for component, info in status.items():
            if isinstance(info, dict):
                trained = "✅" if info.get('trained', False) else "❌"
                optimized = "✅" if info.get('optimized', False) else "❌"
                last = str(info.get('last_update', 'Never'))[:10] if info.get('last_update') else 'Never'
                print(f"║ {component:<25} {trained:<12} {optimized:<12} {last:<17} ║")
                
        print("╚" + "═" * 68 + "╝\n")


# ============================================================================
# TENSORBOARD LOGGER
# ============================================================================

class MoETensorBoardLogger:
    """TensorBoard logger for MoE training metrics."""
    
    def __init__(self, log_dir: str = "logs/tensorboard/MoE_Training"):
        self.log_dir = log_dir
        self.writer: Optional[SummaryWriter] = None
        self.global_step = 0
        
        if HAS_TENSORBOARD:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_path = f"{log_dir}/{timestamp}"
            os.makedirs(full_path, exist_ok=True)
            self.writer = SummaryWriter(full_path)
            logger.info(f"📊 TensorBoard: {full_path}")
            print(f"\n📊 TensorBoard logging to: {full_path}")
            print(f"   Launch with: tensorboard --logdir={log_dir}")
            
    def log_scalars(self, tag: str, metrics: Dict[str, float], step: Optional[int] = None):
        if self.writer is None:
            return
        step = step if step is not None else self.global_step
        for key, value in metrics.items():
            self.writer.add_scalar(f"{tag}/{key}", value, step)
        self.global_step += 1
        
    def log_text(self, tag: str, text: str):
        if self.writer:
            self.writer.add_text(tag, text, self.global_step)
        
    def close(self):
        if self.writer:
            self.writer.close()


# ============================================================================
# TRAINING STATUS TRACKER
# ============================================================================

class MoETrainingStatus:
    """Tracks training status of all MoE components."""
    
    STATUS_FILE = "logs/moe_training_status.json"
    
    def __init__(self):
        self.status = self._load_status()
        
    def _load_status(self) -> Dict[str, Any]:
        if os.path.exists(self.STATUS_FILE):
            try:
                with open(self.STATUS_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load status: {e}")
                
        return {
            'crypto_regime_detector': {'trained': False, 'optimized': False, 'last_update': None},
            'soft_gating_network': {'trained': False, 'optimized': False, 'last_update': None},
            'bull_specialist': {'trained': False, 'optimized': False, 'last_update': None},
            'bear_specialist': {'trained': False, 'optimized': False, 'last_update': None},
            'ranger_specialist': {'trained': False, 'optimized': False, 'last_update': None},
            'online_learner': {'initialized': False, 'last_update': None},
            'walk_forward_validated': False,
            'last_full_training': None
        }
        
    def save(self):
        os.makedirs(os.path.dirname(self.STATUS_FILE), exist_ok=True)
        with open(self.STATUS_FILE, 'w') as f:
            json.dump(self.status, f, indent=2, default=str)
            
    def mark_trained(self, component: str):
        if component in self.status:
            self.status[component]['trained'] = True
            self.status[component]['last_update'] = datetime.now().isoformat()
            self.save()
            
    def mark_optimized(self, component: str):
        if component in self.status:
            self.status[component]['optimized'] = True
            self.status[component]['last_update'] = datetime.now().isoformat()
            self.save()
            
    def needs_training(self, component: str) -> bool:
        if component not in self.status:
            return True
        return not self.status[component].get('trained', False)
        
    def needs_optimization(self, component: str) -> bool:
        if component not in self.status:
            return True
        return not self.status[component].get('optimized', False)
        
    def get_summary(self) -> str:
        lines = ["=" * 60, "MoE Training Status", "=" * 60]
        for component, info in self.status.items():
            if isinstance(info, dict):
                trained = "✅" if info.get('trained', False) else "❌"
                optimized = "✅" if info.get('optimized', False) else "❌"
                lines.append(f"{component}: Trained={trained} Optimized={optimized}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# MOE TRAINING ORCHESTRATOR
# ============================================================================

class MoETrainingOrchestrator:
    """Orchestrates the complete MoE training pipeline with metrics logging."""
    
    def __init__(
        self,
        config: Optional[AIConfig] = None,
        trading_config: Optional[TradingConfig] = None,
        device: str = 'auto',
        enable_tensorboard: bool = True,
        verbose: bool = True
    ):
        self.config = config or AIConfig()
        self.trading_config = trading_config or TradingConfig()
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu'
        self.verbose = verbose
        
        self.status = MoETrainingStatus()
        self.hpo_config = self._load_hpo_config()
        
        # Logging
        self.display = MetricsDisplay("MoE")
        self.tb_logger = MoETensorBoardLogger() if enable_tensorboard else None
        
        # Components
        self.gating_network: Optional[SoftGatingNetwork] = None
        self.specialists: Dict[str, BaseRegimeSpecialist] = {}
        self.online_learner: Optional[OnlineTrendLearner] = None
        
        # 🔬 Dr. Tensor: Ensemble Regime Detector (HMM + GMM + ADX + Funding)
        self.regime_detector: Optional[CryptoRegimeDetector] = None
        self.regime_df: Optional[pd.DataFrame] = None
        
        # Data
        self.train_df: Optional[pd.DataFrame] = None
        self.latent_dim: int = 64
        
        logger.info(f"🚀 MoETrainingOrchestrator initialized on {self.device}")
        
    def _load_hpo_config(self) -> Dict[str, Any]:
        config_path = ROOT_DIR / "config" / "hpo_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
        
    def load_data(self) -> pd.DataFrame:
        """Load training data."""
        logger.info("📊 Loading training data...")
        
        try:
            pipeline = TemporalAutoencoderPipeline(self.config)
            
            if hasattr(pipeline, 'get_featured_dataframe'):
                self.train_df = pipeline.get_featured_dataframe()
            else:
                # Try multiple fallback locations
                fallback_paths = [
                    ROOT_DIR / "data" / "featured_data.parquet",
                    ROOT_DIR / "models_ai" / "base_featured_df.pkl",  # Cached training data
                ]
                
                for path in fallback_paths:
                    if path.exists():
                        if path.suffix == '.parquet':
                            self.train_df = pd.read_parquet(path)
                        else:
                            self.train_df = pd.read_pickle(path)
                        logger.info(f"✅ Loaded data from: {path}")
                        break
                else:
                    raise FileNotFoundError(
                        "No training data found. Run the bot first to generate data or provide featured_data.parquet"
                    )
                    
            latent_cols = [c for c in self.train_df.columns if c.startswith('latent_')]
            self.latent_dim = len(latent_cols) if latent_cols else 64
            
            logger.info(f"✅ Loaded {len(self.train_df)} samples, latent_dim={self.latent_dim}")
            if self.verbose:
                print(f"   Loaded {len(self.train_df):,} samples, latent_dim={self.latent_dim}")
            return self.train_df
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
            
    def initialize_components(self):
        """Initialize all MoE components."""
        logger.info("🔧 Initializing MoE components...")
        
        self.gating_network = create_gating_network(self.config, self.latent_dim, self.device)
        
        # Try loading existing
        gating_path = self.hpo_config.get('model_paths', {}).get('soft_gating_network', 'models_ai/soft_gating_network.pt')
        if os.path.exists(gating_path):
            try:
                self.gating_network = SoftGatingNetwork.load(gating_path, self.device)
                logger.info(f"✅ Loaded gating network from {gating_path}")
            except Exception as e:
                logger.warning(f"Could not load gating network: {e}")
                
        # Initialize specialists
        for name, cls in [('bull', BullSpecialist), ('bear', BearSpecialist), ('ranger', RangerSpecialist)]:
            try:
                self.specialists[name] = cls(
                    config=self.config,
                    trading_config=self.trading_config,
                    input_dim=self.latent_dim
                )
                logger.info(f"✅ Initialized {name} specialist")
            except Exception as e:
                logger.error(f"❌ Failed to init {name}: {e}")
    
    # ============================================================================
    # 🔬 DR. TENSOR: REGIME DETECTOR TRAINING (FIRST STEP)
    # ============================================================================
    
    def train_regime_detector(self, force: bool = False):
        """
        🔬 Train CryptoRegimeDetector as FIRST step in pipeline.
        
        Uses ensemble of HMM + GMM + ADX + Funding for regime detection.
        Must be trained BEFORE gating network and autoencoder.
        """
        if not force and not self.status.needs_training('crypto_regime_detector'):
            logger.info("⏩ Regime detector already trained")
            # Try to load cached regime labels
            regime_cache = ROOT_DIR / "models_ai" / "regime_labels.parquet"
            if regime_cache.exists():
                self.regime_df = pd.read_parquet(regime_cache)
                logger.info(f"✅ Loaded cached regime labels: {len(self.regime_df)} samples")
                return
            
        self.display.print_header("🔬 Training CryptoRegimeDetector (Step 1)")
        
        if self.train_df is None:
            self.load_data()
        
        # Configure detector for crypto
        config = RegimeConfig(
            adx_period=9,                    # Shorter for crypto
            adx_trend_threshold=20.0,        # Lower threshold
            confidence_threshold=0.75,       # Higher for crypto noise
            min_regime_duration=8,           # Anti-whipsaw
            weight_hmm=0.35,
            weight_gmm=0.30,
            weight_adx=0.20,
            weight_funding=0.15
        )
        
        self.regime_detector = CryptoRegimeDetector(config)
        
        print("   🧠 Fitting HMM + GMM + ADX + Funding ensemble...")
        start = time.time()
        
        try:
            # Fit and predict regimes
            self.regime_df = self.regime_detector.fit_predict(self.train_df)
            
            # Validate quality
            metrics = self.regime_detector.validate_regimes()
            
            elapsed = time.time() - start
            
            # Log to TensorBoard
            if self.tb_logger:
                self.tb_logger.log_scalars("RegimeDetector", {
                    'persistence_bars': metrics['avg_persistence_bars'],
                    'model_agreement': metrics['avg_model_agreement'],
                    'mean_confidence': metrics['mean_confidence'],
                    'transition_rate': metrics['transition_rate']
                })
            
            # Show results
            if self.verbose:
                print(f"\\n   📊 Regime Distribution:")
                print(f"      Bull:   {metrics['bull_pct']*100:.1f}%")
                print(f"      Bear:   {metrics['bear_pct']*100:.1f}%")
                print(f"      Ranger: {metrics['sideways_pct']*100:.1f}%")
                print(f"\\n   📈 Quality Metrics:")
                print(f"      Persistence:  {metrics['avg_persistence_bars']:.1f} bars {'✅' if metrics['persistence_ok'] else '❌'}")
                print(f"      Agreement:    {metrics['avg_model_agreement']*100:.1f}% {'✅' if metrics['agreement_ok'] else '❌'}")
                print(f"      Confidence:   {metrics['mean_confidence']*100:.1f}% {'✅' if metrics['confidence_ok'] else '❌'}")
            
            # Add regime labels to training dataframe
            self.train_df['regime'] = self.regime_df['regime'].values
            self.train_df['regime_confidence'] = self.regime_df['confidence'].values
            
            # Cache for later use
            regime_cache = ROOT_DIR / "models_ai" / "regime_labels.parquet"
            os.makedirs(regime_cache.parent, exist_ok=True)
            self.regime_df.to_parquet(regime_cache)
            
            # Mark as trained
            self.status.mark_trained('crypto_regime_detector')
            
            self.display.print_summary({
                'duration_sec': elapsed,
                'n_samples': len(self.regime_df),
                'persistence': metrics['avg_persistence_bars'],
                'agreement': metrics['avg_model_agreement'],
                'all_checks_passed': metrics['all_checks_passed']
            })
            
        except Exception as e:
            logger.error(f"❌ Regime detector training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
                
    def train_gating_network(self, force: bool = False):
        """Train gating network with TensorBoard logging."""
        if not force and not self.status.needs_training('soft_gating_network'):
            logger.info("⏩ Gating network already trained")
            return
            
        self.display.print_header("Training Soft Gating Network")
        
        # Prepare data
        latent_cols = [c for c in self.train_df.columns if c.startswith('latent_')]
        if not latent_cols:
            latent_cols = self.train_df.select_dtypes(include=np.number).columns.tolist()[:self.latent_dim]
            
        X = torch.FloatTensor(self.train_df[latent_cols[:self.latent_dim]].values).to(self.device)
        
        # 🔬 Dr. Tensor: Get regimes from CryptoRegimeDetector (already trained)
        if self.regime_df is None:
            logger.warning("⚠️ Regime detector not trained yet! Training now...")
            self.train_regime_detector(force=True)
        
        y = torch.LongTensor(self.regime_df['regime'].values).to(self.device)
        
        # Show regime distribution
        unique, counts = torch.unique(y, return_counts=True)
        regime_names = ['bull', 'bear', 'ranger']
        if self.verbose:
            print("   Regime distribution:")
            for u, c in zip(unique.cpu().numpy(), counts.cpu().numpy()):
                print(f"      {regime_names[u]}: {c:,} samples ({c/len(y)*100:.1f}%)")
        
        # Training
        optimizer = torch.optim.Adam(self.gating_network.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        epochs = self.hpo_config.get('training_defaults', {}).get('gating_epochs', 100)
        self.gating_network.train()
        
        print(f"\n   Training for {epochs} epochs...\n")
        
        best_acc = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.gating_network(X, return_entropy=True)
            
            ce_loss = F.cross_entropy(output['logits'], y)
            lb_loss = self.gating_network.compute_load_balancing_loss(reset=True)
            loss = ce_loss + 0.1 * lb_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gating_network.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Metrics
            with torch.no_grad():
                accuracy = (output['selected_expert'] == y).float().mean().item()
                entropy = output['entropy'].mean().item()
            
            if accuracy > best_acc:
                best_acc = accuracy
            
            # TensorBoard
            if self.tb_logger:
                self.tb_logger.log_scalars("GatingNetwork", {
                    'loss': loss.item(),
                    'accuracy': accuracy,
                    'entropy': entropy,
                    'lr': scheduler.get_last_lr()[0]
                }, step=epoch)
            
            # Console (every 10 epochs)
            if (epoch + 1) % 10 == 0:
                self.display.print_epoch_metrics(epoch + 1, epochs, {
                    'loss': loss.item(),
                    'accuracy': accuracy,
                    'entropy': entropy
                })
                
        # Save
        gating_path = self.hpo_config.get('model_paths', {}).get('soft_gating_network', 'models_ai/soft_gating_network.pt')
        os.makedirs(os.path.dirname(gating_path), exist_ok=True)
        self.gating_network.save(gating_path)
        self.status.mark_trained('soft_gating_network')
        
        self.display.print_summary({'best_accuracy': best_acc, 'final_loss': loss.item()})
        
    def train_specialist(self, name: str, force: bool = False):
        """Train a specialist with TensorBoard logging."""
        if name not in self.specialists:
            logger.error(f"❌ Unknown specialist: {name}")
            return
            
        if not force and not self.status.needs_training(f'{name}_specialist'):
            logger.info(f"⏩ {name} specialist already trained")
            return
            
        self.display.print_header(f"Training {name.upper()} Specialist")
        
        specialist = self.specialists[name]
        timesteps = self.hpo_config.get('training_defaults', {}).get('specialist_timesteps', 500000)
        
        if self.verbose:
            print(f"   Target timesteps: {timesteps:,}")
        
        try:
            start = time.time()
            tb_log_dir = f"logs/tensorboard/{name}_specialist"
            
            metrics = specialist.train_model(
                df=self.train_df,
                total_timesteps=timesteps,
                tensorboard_log=tb_log_dir
            )
            
            elapsed = time.time() - start
            self.status.mark_trained(f'{name}_specialist')
            
            final = metrics if isinstance(metrics, dict) else {}
            final['training_time_min'] = elapsed / 60
            self.display.print_summary(final)
            
        except Exception as e:
            logger.error(f"❌ Failed to train {name}: {e}")
            
    def initialize_online_learner(self):
        """Initialize online learning."""
        logger.info("🔧 Initializing online learner...")
        
        try:
            self.online_learner = create_online_learner(
                model=self.gating_network,
                config=self.config,
                device=self.device
            )
            
            self.status.status['online_learner']['initialized'] = True
            self.status.status['online_learner']['last_update'] = datetime.now().isoformat()
            self.status.save()
            
            logger.info("✅ Online learner initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to init online learner: {e}")
            
    def check_and_train(self) -> bool:
        """Check status and train what's needed."""
        if self.verbose:
            self.display.print_header("MoE Component Status")
            self.display.print_status_table(self.status.status)
        
        needs_work = False
        
        self.load_data()
        self.initialize_components()
        
        # 🔬 Dr. Tensor: Regime detector MUST be first
        if self.status.needs_training('crypto_regime_detector'):
            print("\n📌 [STEP 1] Regime detector needs training")
            self.train_regime_detector()
            needs_work = True
        
        if self.status.needs_training('soft_gating_network'):
            print("\n📌 [STEP 2] Gating network needs training")
            self.train_gating_network()
            needs_work = True
            
        for name in ['bull', 'bear', 'ranger']:
            if self.status.needs_training(f'{name}_specialist'):
                print(f"\n📌 {name} specialist needs training")
                self.train_specialist(name)
                needs_work = True
                
        if not self.status.status['online_learner'].get('initialized', False):
            print("\n📌 Initializing online learner")
            self.initialize_online_learner()
            needs_work = True
            
        if not needs_work:
            print("\n✅ All MoE components are trained and ready!")
            
        if self.tb_logger:
            self.tb_logger.close()
            
        return needs_work
        
    def run_full_pipeline(self, force: bool = False):
        """Run complete training pipeline."""
        self.display.print_header("Full MoE Training Pipeline")
        start_time = datetime.now()
        
        try:
            self.load_data()
            self.initialize_components()
            
            # 🔬 Dr. Tensor: STEP 1 - Train regime detector FIRST
            self.train_regime_detector(force=force)
            
            # STEP 2 - Train gating network (uses regime labels)
            self.train_gating_network(force=force)
            
            # STEP 3 - Train specialists
            for name in ['bull', 'bear', 'ranger']:
                self.train_specialist(name, force=force)
                
            # STEP 4 - Initialize online learner
            self.initialize_online_learner()
            
            self.status.status['last_full_training'] = datetime.now().isoformat()
            self.status.save()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n✅ Full pipeline completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
        finally:
            if self.tb_logger:
                self.tb_logger.close()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for MoE training."""
    parser = argparse.ArgumentParser(description="MoE Training Orchestrator")
    parser.add_argument('--full', action='store_true', help='Full training pipeline')
    parser.add_argument('--force', action='store_true', help='Force retraining')
    parser.add_argument('--check', action='store_true', help='Check status only')
    parser.add_argument('--specialist', type=str, help='Train specific specialist')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard')
    
    args = parser.parse_args()
    
    setup_phase_logger("MoE Training", "logs/moe_training.log")
    
    orchestrator = MoETrainingOrchestrator(
        device=args.device,
        enable_tensorboard=not args.no_tensorboard
    )
    
    if args.check:
        orchestrator.display.print_status_table(orchestrator.status.status)
        return
        
    if args.full:
        orchestrator.run_full_pipeline(force=args.force)
    elif args.specialist:
        orchestrator.load_data()
        orchestrator.initialize_components()
        orchestrator.train_specialist(args.specialist, force=args.force)
    else:
        orchestrator.check_and_train()


if __name__ == "__main__":
    main()
