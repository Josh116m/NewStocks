import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import concurrent.futures
from datetime import datetime, timedelta
import time
import joblib
import os
import psutil
import GPUtil
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""
    # Data parameters
    sequence_length: int = 20
    train_test_split: float = 0.8
    validation_split: float = 0.1
    
    # Training parameters
    max_epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    
    # Ensemble parameters
    n_cv_folds: int = 5
    use_time_series_cv: bool = True
    calibrate_probabilities: bool = True
    
    # Resource management
    n_cpu_jobs: int = -1  # Use all CPUs
    use_gpu: bool = torch.cuda.is_available()
    mixed_precision: bool = True
    
    # Performance monitoring
    monitor_resources: bool = True
    profile_performance: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 5  # Save every N epochs
    
    # Target performance
    target_accuracy: float = 0.85
    max_training_hours: float = 6.0


class ResourceMonitor:
    """Monitor system resources during training."""
    
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        self.is_monitoring = False
        
    def start_monitoring(self):
        """Start resource monitoring in a separate thread."""
        self.is_monitoring = True
        self._monitor_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._monitor_future = self._monitor_thread.submit(self._monitor_loop)
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.shutdown(wait=True)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            # CPU and Memory
            self.cpu_usage.append(psutil.cpu_percent(interval=1))
            self.memory_usage.append(psutil.virtual_memory().percent)
            
            # GPU if available
            if torch.cuda.is_available():
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_usage.append(gpus[0].load * 100)
                    self.gpu_memory.append(gpus[0].memoryUtil * 100)
                else:
                    self.gpu_usage.append(0)
                    self.gpu_memory.append(0)
            
            self.timestamps.append(datetime.now())
            time.sleep(1)  # Monitor every second
    
    def get_summary(self) -> Dict[str, float]:
        """Get resource usage summary."""
        if not self.cpu_usage:
            return {}
        
        summary = {
            'avg_cpu_usage': np.mean(self.cpu_usage),
            'max_cpu_usage': np.max(self.cpu_usage),
            'avg_memory_usage': np.mean(self.memory_usage),
            'max_memory_usage': np.max(self.memory_usage),
        }
        
        if self.gpu_usage:
            summary.update({
                'avg_gpu_usage': np.mean(self.gpu_usage),
                'max_gpu_usage': np.max(self.gpu_usage),
                'avg_gpu_memory': np.mean(self.gpu_memory),
                'max_gpu_memory': np.max(self.gpu_memory),
            })
        
        return summary
    
    def plot_usage(self, save_path: Optional[str] = None):
        """Plot resource usage over time."""
        if not self.timestamps:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Convert timestamps to elapsed time
        start_time = self.timestamps[0]
        elapsed_times = [(t - start_time).total_seconds() / 60 for t in self.timestamps]
        
        # CPU and Memory
        ax1 = axes[0]
        ax1.plot(elapsed_times, self.cpu_usage, label='CPU Usage', color='blue')
        ax1.plot(elapsed_times, self.memory_usage, label='Memory Usage', color='green')
        ax1.set_ylabel('Usage (%)')
        ax1.set_title('System Resource Usage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # GPU
        if self.gpu_usage:
            ax2 = axes[1]
            ax2.plot(elapsed_times, self.gpu_usage, label='GPU Usage', color='red')
            ax2.plot(elapsed_times, self.gpu_memory, label='GPU Memory', color='orange')
            ax2.set_ylabel('Usage (%)')
            ax2.set_xlabel('Time (minutes)')
            ax2.set_title('GPU Resource Usage')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class CheckpointManager:
    """Manage model checkpoints for warm-starting."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, models: Dict[str, Any], 
                       epoch: int, metrics: Dict[str, float],
                       config: PipelineConfig):
        """Save checkpoint with models and metadata."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pkl"
        
        checkpoint = {
            'models': models,
            'epoch': epoch,
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(checkpoint, checkpoint_path)
        logger.info(f"💾 Saved checkpoint to {checkpoint_path}")
        
        # Save metrics separately for easy tracking
        metrics_path = self.checkpoint_dir / "training_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        all_metrics.append({
            'epoch': epoch,
            'timestamp': checkpoint['timestamp'],
            **metrics
        })
        
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pkl"))
        if not checkpoints:
            return None
        
        # Sort by epoch number
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        
        logger.info(f"📂 Loading checkpoint from {latest}")
        return joblib.load(latest)
    
    def cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints to save space."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pkl"),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
                logger.info(f"🗑️  Removed old checkpoint: {checkpoint}")


class OptimizedTrainingPipeline:
    """Main training pipeline with optimization and monitoring."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.resource_monitor = ResourceMonitor()
        
        # Component storage
        self.feature_engineer = None
        self.regime_detector = None
        self.lstm_model = None
        self.ensemble = None
        
        # Training history
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'ensemble_acc': [],
            'training_time': []
        }
        
        # Set device
        self.device = torch.device('cuda' if config.use_gpu else 'cpu')
        logger.info(f"🖥️  Using device: {self.device}")
        
    def prepare_data(self, raw_data: pd.DataFrame) -> Tuple[Dict[str, Any], ...]:
        """
        Prepare data for training with all preprocessing steps.
        
        Args:
            raw_data: Raw stock data from Polygon.io
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info("📊 Preparing data for training...")
        
        # Initialize feature engineer
        from advanced_feature_engineering import AdvancedFeatureEngineer
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Extract market data (SPY) for regime detection
        market_data = raw_data[raw_data['ticker'] == 'SPY'].copy()
        if market_data.empty:
            logger.warning("⚠️  No SPY data found for regime detection")
            market_data = None
        else:
            # Ensure market data has proper datetime index for regime detection
            market_data['date'] = pd.to_datetime(market_data['date'])
            market_data = market_data.set_index('date').sort_index()

        # Engineer features
        logger.info("🔧 Engineering features...")
        data_with_features = self.feature_engineer.compute_all_features(raw_data, market_data)

        # Initialize and fit regime detector
        if market_data is not None:
            from regime_detector import DataDrivenRegimeDetector
            self.regime_detector = DataDrivenRegimeDetector()
            self.regime_detector.fit_regime_model(market_data)

            # Get regime labels for all data
            regime_history = self.regime_detector.get_regime_history(market_data)

            # Ensure date columns are compatible for merging
            data_with_features['date'] = pd.to_datetime(data_with_features['date'])
            regime_history['date'] = pd.to_datetime(regime_history['date'])

            data_with_features = data_with_features.merge(
                regime_history[['date', 'regime_id']],
                on='date',
                how='left'
            )
        else:
            # Default regime if no market data
            data_with_features['regime_id'] = 0
        
        # Create target variable (next day return > 0)
        data_with_features['target'] = (
            data_with_features.groupby('ticker')['returns'].shift(-1) > 0
        ).astype(int)
        
        # Remove NaN values
        data_with_features = data_with_features.dropna()
        
        # Split by time
        n_samples = len(data_with_features)
        train_end = int(n_samples * self.config.train_test_split)
        val_end = train_end + int(n_samples * self.config.validation_split)
        
        train_data = data_with_features.iloc[:train_end]
        val_data = data_with_features.iloc[train_end:val_end]
        test_data = data_with_features.iloc[val_end:]
        
        logger.info(f"📈 Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Prepare features for different models
        feature_cols = [col for col in data_with_features.columns 
                       if col not in ['ticker', 'date', 'target', 'regime_id', 
                                     'open', 'high', 'low', 'close', 'volume']]
        
        # Prepare LSTM features
        from multi_stream_lstm import prepare_features_for_lstm
        lstm_features_train = prepare_features_for_lstm(train_data, self.feature_engineer)
        lstm_features_val = prepare_features_for_lstm(val_data, self.feature_engineer)
        lstm_features_test = prepare_features_for_lstm(test_data, self.feature_engineer)
        
        # Package data
        train_package = {
            'X': train_data[feature_cols].values,
            'y': train_data['target'].values,
            'regime_ids': train_data['regime_id'].values,
            'lstm_features': lstm_features_train,
            'feature_names': feature_cols
        }
        
        val_package = {
            'X': val_data[feature_cols].values,
            'y': val_data['target'].values,
            'regime_ids': val_data['regime_id'].values,
            'lstm_features': lstm_features_val,
            'feature_names': feature_cols
        }
        
        test_package = {
            'X': test_data[feature_cols].values,
            'y': test_data['target'].values,
            'regime_ids': test_data['regime_id'].values,
            'lstm_features': lstm_features_test,
            'feature_names': feature_cols
        }
        
        return train_package, val_package, test_package
    
    def train_lstm_model(self, train_data: Dict[str, Any], 
                        val_data: Dict[str, Any],
                        checkpoint: Optional[Dict] = None) -> Any:
        """Train the multi-stream LSTM model."""
        logger.info("🧠 Training Multi-Stream LSTM...")
        
        from multi_stream_lstm import MultiStreamLSTM, MultiStreamLSTMModule, StockDataset
        from torch.utils.data import DataLoader
        import pytorch_lightning as pl
        
        # Determine feature dimensions
        feature_dims = {
            'short': train_data['lstm_features']['short'].shape[1],
            'medium': train_data['lstm_features']['medium'].shape[1],
            'long': train_data['lstm_features']['long'].shape[1]
        }
        
        # Create datasets
        train_dataset = StockDataset(
            train_data['lstm_features'],
            train_data['y'],
            train_data['regime_ids'],
            self.config.sequence_length
        )
        
        val_dataset = StockDataset(
            val_data['lstm_features'],
            val_data['y'],
            val_data['regime_ids'],
            self.config.sequence_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model
        if checkpoint and 'lstm' in checkpoint['models']:
            logger.info("♻️  Loading LSTM from checkpoint...")
            self.lstm_model = checkpoint['models']['lstm']
        else:
            self.lstm_model = MultiStreamLSTM(feature_dims)
        
        # Create Lightning module
        pl_module = MultiStreamLSTMModule(
            self.lstm_model,
            learning_rate=self.config.learning_rate
        )
        
        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator='gpu' if self.config.use_gpu else 'cpu',
            devices=1 if self.config.use_gpu else 'auto',
            precision=16 if self.config.mixed_precision else 32,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    mode='min'
                ),
                pl.callbacks.ModelCheckpoint(
                    dirpath=self.config.checkpoint_dir,
                    filename='lstm-{epoch:02d}-{val_loss:.2f}',
                    save_top_k=1,
                    monitor='val_loss',
                    mode='min'
                )
            ],
            gradient_clip_val=1.0,
            log_every_n_steps=10
        )
        
        # Train
        trainer.fit(pl_module, train_loader, val_loader)
        
        # Get best model
        self.lstm_model = pl_module.model
        
        return self.lstm_model
    
    def train_ensemble(self, train_data: Dict[str, Any],
                      val_data: Dict[str, Any],
                      checkpoint: Optional[Dict] = None) -> Any:
        """Train the stacked ensemble."""
        logger.info("🎯 Training Stacked Ensemble...")
        
        from stacked_ensemble import StackedEnsemblePredictor
        
        # Initialize ensemble
        self.ensemble = StackedEnsemblePredictor(
            lstm_model=self.lstm_model,
            n_folds=self.config.n_cv_folds,
            use_time_series_cv=self.config.use_time_series_cv,
            calibrate_probabilities=self.config.calibrate_probabilities
        )
        
        # Combine train and validation for ensemble training
        X_combined = np.vstack([train_data['X'], val_data['X']])
        y_combined = np.hstack([train_data['y'], val_data['y']])
        regime_combined = np.hstack([train_data['regime_ids'], val_data['regime_ids']])
        
        # Combine LSTM features
        lstm_combined = {}
        for key in ['short', 'medium', 'long']:
            lstm_combined[key] = np.vstack([
                train_data['lstm_features'][key],
                val_data['lstm_features'][key]
            ])
        
        # Train ensemble
        self.ensemble.fit(
            X_combined,
            y_combined,
            regime_labels=regime_combined,
            feature_names=train_data['feature_names'],
            lstm_features=lstm_combined
        )
        
        return self.ensemble
    
    def evaluate_performance(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        logger.info("📊 Evaluating performance...")
        
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
        
        # Get predictions
        predictions = self.ensemble.predict_proba(
            test_data['X'],
            test_data['regime_ids'],
            test_data['lstm_features']
        )
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(test_data['y'], predictions.round()),
            'auc': roc_auc_score(test_data['y'], predictions),
            'precision': precision_score(test_data['y'], predictions.round()),
            'recall': recall_score(test_data['y'], predictions.round())
        }
        
        # Get performance by regime
        regime_performance = {}
        for regime_id in np.unique(test_data['regime_ids']):
            mask = test_data['regime_ids'] == regime_id
            if mask.sum() > 10:  # Need enough samples
                regime_performance[f'regime_{regime_id}_acc'] = accuracy_score(
                    test_data['y'][mask],
                    predictions[mask].round()
                )
        
        metrics.update(regime_performance)
        
        return metrics
    
    def run_training_pipeline(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            raw_data: Raw stock data
            
        Returns:
            Dictionary with trained models and performance metrics
        """
        logger.info("🚀 Starting Optimized Training Pipeline")
        logger.info(f"⏰ Target: {self.config.target_accuracy*100:.1f}% accuracy in {self.config.max_training_hours} hours")
        
        start_time = time.time()
        
        # Start resource monitoring
        if self.config.monitor_resources:
            self.resource_monitor.start_monitoring()
        
        try:
            # Check for existing checkpoint
            checkpoint = self.checkpoint_manager.load_latest_checkpoint()
            if checkpoint:
                logger.info(f"📂 Resuming from epoch {checkpoint['epoch']}")
            
            # Prepare data
            train_data, val_data, test_data = self.prepare_data(raw_data)
            
            # Train LSTM
            lstm_start = time.time()
            self.train_lstm_model(train_data, val_data, checkpoint)
            lstm_time = time.time() - lstm_start
            logger.info(f"⏱️  LSTM training time: {lstm_time/60:.1f} minutes")
            
            # Train ensemble
            ensemble_start = time.time()
            self.train_ensemble(train_data, val_data, checkpoint)
            ensemble_time = time.time() - ensemble_start
            logger.info(f"⏱️  Ensemble training time: {ensemble_time/60:.1f} minutes")
            
            # Evaluate performance
            test_metrics = self.evaluate_performance(test_data)
            
            # Total training time
            total_time = time.time() - start_time
            
            # Print results
            logger.info("\n" + "="*60)
            logger.info("📈 TRAINING COMPLETE!")
            logger.info("="*60)
            logger.info(f"⏱️  Total training time: {total_time/3600:.2f} hours")
            logger.info(f"🎯 Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
            logger.info(f"📊 Test AUC: {test_metrics['auc']:.4f}")
            logger.info(f"📊 Precision: {test_metrics['precision']:.4f}")
            logger.info(f"📊 Recall: {test_metrics['recall']:.4f}")
            
            # Check if target achieved
            if test_metrics['accuracy'] >= self.config.target_accuracy:
                logger.info(f"✅ TARGET ACHIEVED! {test_metrics['accuracy']*100:.2f}% >= {self.config.target_accuracy*100:.1f}%")
            else:
                logger.info(f"❌ Target not reached: {test_metrics['accuracy']*100:.2f}% < {self.config.target_accuracy*100:.1f}%")
            
            # Save final models
            self.save_final_models(test_metrics)
            
            # Get resource summary
            if self.config.monitor_resources:
                self.resource_monitor.stop_monitoring()
                resource_summary = self.resource_monitor.get_summary()
                logger.info("\n📊 Resource Usage Summary:")
                for key, value in resource_summary.items():
                    logger.info(f"   {key}: {value:.1f}%")
                
                # Plot resource usage
                self.resource_monitor.plot_usage(
                    os.path.join(self.config.checkpoint_dir, "resource_usage.png")
                )
            
            # Return results
            results = {
                'models': {
                    'feature_engineer': self.feature_engineer,
                    'regime_detector': self.regime_detector,
                    'lstm': self.lstm_model,
                    'ensemble': self.ensemble
                },
                'metrics': test_metrics,
                'training_time': total_time,
                'resource_usage': resource_summary if self.config.monitor_resources else None
            }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            if self.config.monitor_resources:
                self.resource_monitor.stop_monitoring()
            raise
    
    def save_final_models(self, metrics: Dict[str, float]):
        """Save all trained models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(self.config.checkpoint_dir) / f"final_models_{timestamp}"
        save_dir.mkdir(exist_ok=True)
        
        # Save each component
        joblib.dump(self.feature_engineer, save_dir / "feature_engineer.pkl")
        joblib.dump(self.regime_detector, save_dir / "regime_detector.pkl")
        torch.save(self.lstm_model.state_dict(), save_dir / "lstm_model.pth")
        self.ensemble.save_ensemble(str(save_dir / "ensemble.pkl"))
        
        # Save metrics
        with open(save_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"💾 Models saved to {save_dir}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        if not self.training_history['epochs']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        ax1 = axes[0, 0]
        ax1.plot(self.training_history['epochs'], self.training_history['train_loss'], label='Train Loss')
        ax1.plot(self.training_history['epochs'], self.training_history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2 = axes[0, 1]
        ax2.plot(self.training_history['epochs'], self.training_history['train_acc'], label='Train Acc')
        ax2.plot(self.training_history['epochs'], self.training_history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Ensemble accuracy
        ax3 = axes[1, 0]
        ax3.plot(self.training_history['epochs'], self.training_history['ensemble_acc'], 
                label='Ensemble Acc', color='green')
        ax3.axhline(y=self.config.target_accuracy, color='r', linestyle='--', 
                   label=f'Target ({self.config.target_accuracy:.2%})')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Ensemble Accuracy vs Target')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Training time
        ax4 = axes[1, 1]
        ax4.plot(self.training_history['epochs'], self.training_history['training_time'])
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (hours)')
        ax4.set_title('Cumulative Training Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage
def run_example_pipeline():
    """Run an example training pipeline."""
    # Configure pipeline
    config = PipelineConfig(
        sequence_length=20,
        max_epochs=30,
        batch_size=256,
        target_accuracy=0.82,  # Realistic target
        max_training_hours=6.0,
        checkpoint_dir="./model_checkpoints"
    )
    
    # Initialize pipeline
    pipeline = OptimizedTrainingPipeline(config)
    
    # Load your data here
    # For this example, we'll create synthetic data
    logger.info("📊 Creating synthetic data for example...")
    
    # Simulate loading data (replace with actual data loading)
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'SPY', 'AMZN']
    
    data = []
    for ticker in tickers:
        for date in dates:
            if date.weekday() < 5:  # Weekdays only
                base_price = 100 + np.random.randn() * 10
                data.append({
                    'ticker': ticker,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': base_price + np.random.randn(),
                    'high': base_price + abs(np.random.randn()) * 2,
                    'low': base_price - abs(np.random.randn()) * 2,
                    'close': base_price + np.random.randn() * 0.5,
                    'volume': int(1000000 + np.random.randn() * 100000)
                })
    
    raw_data = pd.DataFrame(data)
    
    # Run training pipeline
    results = pipeline.run_training_pipeline(raw_data)
    
    # Plot training history
    pipeline.plot_training_history("./model_checkpoints/training_history.png")
    
    return results


if __name__ == "__main__":
    # Run example
    results = run_example_pipeline()
