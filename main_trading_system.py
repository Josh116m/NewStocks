"""
Complete Stock Trading Prediction System
Integrates all components for daily trading predictions with 80-85% accuracy target
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import torch
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# GPU acceleration setup
try:
    import cupy as cp
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        DEVICE = torch.device('cuda')
        print(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        DEVICE = torch.device('cpu')
        print("💻 Using CPU - GPU not available")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    DEVICE = torch.device('cpu')
    print("⚠️ CuPy not available - using CPU only")

# Import all our modules
from advanced_feature_engineering import AdvancedFeatureEngineer
from regime_detector import DataDrivenRegimeDetector
from multi_stream_lstm import MultiStreamLSTM, prepare_features_for_lstm
from stacked_ensemble import StackedEnsemblePredictor
from training_pipeline import OptimizedTrainingPipeline, PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockTradingPredictor:
    """
    Main class for stock trading predictions.
    Integrates all components for end-to-end prediction.
    GPU-accelerated when available.
    """

    def __init__(self, model_dir: str = "./models", use_gpu: bool = True):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # GPU configuration
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device = DEVICE if self.use_gpu else torch.device('cpu')

        if self.use_gpu:
            logger.info(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("💻 Using CPU for predictions")

        # Component storage
        self.feature_engineer = None
        self.regime_detector = None
        self.lstm_model = None
        self.ensemble = None

        # Performance tracking
        self.prediction_history = []
        self.daily_performance = {}
        
    def train_from_polygon_data(self, data_file: str, 
                               config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
        """
        Train the complete system from Polygon.io data.
        
        Args:
            data_file: Path to the downloaded Polygon data (CSV or Parquet)
            config: Pipeline configuration (uses defaults if None)
            
        Returns:
            Dictionary with training results
        """
        logger.info("🚀 Starting Stock Trading Predictor Training")
        logger.info(f"📁 Loading data from: {data_file}")
        
        # Load data
        if data_file.endswith('.parquet'):
            raw_data = pd.read_parquet(data_file)
        else:
            raw_data = pd.read_csv(data_file)
        
        logger.info(f"📊 Loaded {len(raw_data):,} records for {raw_data['ticker'].nunique()} stocks")
        logger.info(f"📅 Date range: {raw_data['date'].min()} to {raw_data['date'].max()}")
        
        # Use default config if not provided
        if config is None:
            config = PipelineConfig(
                sequence_length=20,
                train_test_split=0.8,
                validation_split=0.1,
                max_epochs=50,
                batch_size=256,
                learning_rate=1e-3,
                early_stopping_patience=10,
                n_cv_folds=5,
                use_time_series_cv=True,
                calibrate_probabilities=True,
                target_accuracy=0.82,
                max_training_hours=6.0,
                checkpoint_dir=str(self.model_dir / "checkpoints"),
                use_gpu=self.use_gpu,  # Pass GPU configuration
                mixed_precision=self.use_gpu  # Enable mixed precision if GPU available
            )
        
        # Initialize and run training pipeline
        pipeline = OptimizedTrainingPipeline(config)
        results = pipeline.run_training_pipeline(raw_data)
        
        # Store trained models
        self.feature_engineer = results['models']['feature_engineer']
        self.regime_detector = results['models']['regime_detector']
        self.lstm_model = results['models']['lstm']
        self.ensemble = results['models']['ensemble']

        # Move models to GPU if available
        self._move_models_to_device()

        # Save training results
        self._save_training_results(results)

        return results

    def _move_models_to_device(self):
        """Move PyTorch models to the appropriate device (GPU/CPU)."""
        if self.lstm_model is not None and hasattr(self.lstm_model, 'to'):
            self.lstm_model = self.lstm_model.to(self.device)
            logger.info(f"📱 LSTM model moved to {self.device}")

        # Ensure feature engineer uses GPU if available
        if self.feature_engineer is not None and hasattr(self.feature_engineer, 'use_gpu'):
            self.feature_engineer.use_gpu = self.use_gpu
            logger.info(f"🔧 Feature engineer GPU usage: {self.use_gpu}")

    def predict_next_day(self, current_data: pd.DataFrame,
                        lookback_days: int = 60) -> pd.DataFrame:
        """
        Make predictions for the next trading day.
        
        Args:
            current_data: Recent market data (at least lookback_days of history)
            lookback_days: Days of history needed for features
            
        Returns:
            DataFrame with predictions for each stock
        """
        if self.ensemble is None:
            raise ValueError("Models must be trained or loaded before making predictions")
        
        logger.info(f"🔮 Making predictions for {current_data['ticker'].nunique()} stocks")
        
        # Engineer features
        data_with_features = self.feature_engineer.compute_all_features(current_data)
        
        # Detect current market regime
        market_data = current_data[current_data['ticker'] == 'SPY']
        if not market_data.empty and len(market_data) >= lookback_days:
            regime_name, confidence, context = self.regime_detector.detect_current_regime(
                market_data.tail(lookback_days)
            )
            logger.info(f"📊 Current market regime: {regime_name} (confidence: {confidence:.2%})")
        else:
            regime_name = "unknown"
            confidence = 0.0
            context = {}
        
        # Prepare predictions for each stock
        predictions = []
        
        for ticker in data_with_features['ticker'].unique():
            ticker_data = data_with_features[data_with_features['ticker'] == ticker].tail(lookback_days)
            
            if len(ticker_data) < self.feature_engineer.short_window:
                continue
            
            # Get latest features
            latest_features = ticker_data.iloc[-1]
            
            # Prepare feature array
            feature_cols = [col for col in ticker_data.columns
                          if col not in ['ticker', 'date', 'target', 'regime_id',
                                       'open', 'high', 'low', 'close', 'volume']]

            # Get feature values and ensure they are numeric
            feature_values = ticker_data[feature_cols].iloc[-1:].copy()

            # Convert any non-numeric columns to numeric, replacing errors with NaN
            for col in feature_cols:
                feature_values[col] = pd.to_numeric(feature_values[col], errors='coerce')

            # Fill any NaN values with 0 (or median if available)
            feature_values = feature_values.fillna(0)

            X = feature_values.values.reshape(1, -1)

            # Handle feature count mismatch with trained models
            expected_features = None
            expected_feature_names = None

            if self.ensemble and hasattr(self.ensemble, 'fitted_base_models'):
                for model_name, model in self.ensemble.fitted_base_models.items():
                    if hasattr(model, 'n_features_in_'):
                        expected_features = model.n_features_in_
                        break

                # Try to get the exact feature names from saved feature importance
                if hasattr(self.ensemble, 'base_model_importance'):
                    for model_name, importance in self.ensemble.base_model_importance.items():
                        if importance and len(importance) == expected_features:
                            expected_feature_names = list(importance.keys())
                            break

            if expected_features and X.shape[1] != expected_features:
                if expected_feature_names and len(feature_cols) > expected_features:
                    # Filter to only the expected features in the correct order
                    filtered_feature_cols = [col for col in expected_feature_names if col in feature_cols]
                    if len(filtered_feature_cols) == expected_features:
                        # Recompute X with only the expected features
                        feature_values_filtered = ticker_data[filtered_feature_cols].iloc[-1:].copy()
                        for col in filtered_feature_cols:
                            feature_values_filtered[col] = pd.to_numeric(feature_values_filtered[col], errors='coerce')
                        feature_values_filtered = feature_values_filtered.fillna(0)
                        X = feature_values_filtered.values.reshape(1, -1)
                        logger.info(f"✅ Filtered features from {len(feature_cols)} to {expected_features} for {ticker}")
                    else:
                        logger.warning(f"⚠️ Could not match expected features, truncating from {X.shape[1]} to {expected_features} for {ticker}")
                        X = X[:, :expected_features]
                elif X.shape[1] < expected_features:
                    # Pad with zeros if we have fewer features
                    padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                    X = np.hstack([X, padding])
                    logger.warning(f"⚠️ Padded features from {X.shape[1] - padding.shape[1]} to {expected_features} for {ticker}")
                elif X.shape[1] > expected_features:
                    # Truncate if we have more features
                    X = X[:, :expected_features]
                    logger.warning(f"⚠️ Truncated features from {X.shape[1]} to {expected_features} for {ticker}")
            
            # Prepare LSTM features
            lstm_features = prepare_features_for_lstm(
                ticker_data,
                self.feature_engineer,
                sequence_length=20
            )

            # Move LSTM features to GPU if available
            if self.use_gpu and lstm_features is not None:
                for key in lstm_features:
                    if key != 'feature_names' and isinstance(lstm_features[key], np.ndarray):
                        lstm_features[key] = torch.tensor(lstm_features[key], dtype=torch.float32).to(self.device).cpu().numpy()

            # Get regime ID
            if 'regime_id' in ticker_data.columns:
                regime_id = ticker_data['regime_id'].iloc[-1:].values
            else:
                regime_id = np.array([0])  # Default regime

            # Make prediction
            try:
                # Pass LSTM features to ensemble - let ensemble handle reshaping
                lstm_pred_features = {k: v for k, v in lstm_features.items() if k != 'feature_names'}

                pred_proba = self.ensemble.predict_proba(
                    X,
                    regime_id,
                    lstm_pred_features
                )
                
                # Get explanation
                explanation = self.ensemble.explain_prediction(
                    X, regime_id, lstm_features, sample_idx=0
                )
                
                predictions.append({
                    'ticker': ticker,
                    'date': ticker_data['date'].iloc[-1],
                    'current_price': ticker_data['close'].iloc[-1],
                    'prediction_proba': pred_proba[0],
                    'prediction': 'BUY' if pred_proba[0] > 0.5 else 'SELL',
                    'confidence': explanation['confidence'],
                    'regime': regime_name,
                    'regime_confidence': confidence
                })
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to predict for {ticker}: {str(e)}")
                continue
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame(predictions)

        # Check if we have any predictions
        if predictions_df.empty:
            logger.warning("⚠️ No successful predictions generated")
            return predictions_df

        # Sort by prediction probability (strongest signals first)
        if 'prediction_proba' in predictions_df.columns:
            predictions_df = predictions_df.sort_values('prediction_proba', ascending=False)
        else:
            logger.warning("⚠️ No prediction_proba column found, returning unsorted predictions")
        
        # Add position sizing based on confidence
        predictions_df['suggested_position_size'] = self._calculate_position_sizes(predictions_df)
        
        # Save predictions
        self._save_daily_predictions(predictions_df)
        
        return predictions_df
    
    def backtest_strategy(self, historical_data: pd.DataFrame,
                         start_date: str, end_date: str,
                         initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Backtest the trading strategy on historical data.
        
        Args:
            historical_data: Historical stock data
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting capital
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"📊 Running backtest from {start_date} to {end_date}")
        
        # Filter data for backtest period
        backtest_data = historical_data[
            (historical_data['date'] >= start_date) & 
            (historical_data['date'] <= end_date)
        ]
        
        # Initialize portfolio
        portfolio = {
            'cash': initial_capital,
            'positions': {},
            'history': [],
            'daily_returns': []
        }
        
        # Get unique dates
        dates = sorted(backtest_data['date'].unique())
        
        for i, date in enumerate(dates[60:]):  # Need 60 days history
            # Get data up to current date
            current_data = backtest_data[backtest_data['date'] <= date]
            
            # Make predictions
            predictions = self.predict_next_day(current_data)
            
            # Execute trades based on predictions
            portfolio = self._execute_trades(portfolio, predictions, current_data, date)
            
            # Calculate daily performance
            portfolio_value = self._calculate_portfolio_value(portfolio, current_data)
            daily_return = (portfolio_value - initial_capital) / initial_capital
            
            portfolio['history'].append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': portfolio['cash'],
                'n_positions': len(portfolio['positions']),
                'daily_return': daily_return
            })
            
            if i % 20 == 0:  # Progress update
                logger.info(f"📈 Backtest progress: {date} - Portfolio: ${portfolio_value:,.2f}")
        
        # Calculate final metrics
        backtest_results = self._calculate_backtest_metrics(portfolio, initial_capital)
        
        # Save backtest results
        self._save_backtest_results(backtest_results)
        
        return backtest_results
    
    def _calculate_position_sizes(self, predictions_df: pd.DataFrame) -> np.ndarray:
        """Calculate position sizes based on Kelly Criterion and confidence."""
        # Simplified Kelly Criterion with safety factor
        kelly_fraction = 0.25  # Conservative Kelly fraction
        
        position_sizes = []
        for _, row in predictions_df.iterrows():
            # Win probability
            p = row['prediction_proba']
            # Loss probability
            q = 1 - p
            # Assumed win/loss ratio (can be calculated from historical data)
            b = 1.5  # Win 1.5x what we risk
            
            # Kelly formula: f = (p*b - q) / b
            kelly = (p * b - q) / b
            
            # Apply safety factor and confidence scaling
            position_size = max(0, kelly * kelly_fraction * row['confidence'])
            
            # Cap maximum position size
            position_size = min(position_size, 0.1)  # Max 10% per position
            
            position_sizes.append(position_size)
        
        return np.array(position_sizes)
    
    def _execute_trades(self, portfolio: Dict, predictions: pd.DataFrame,
                       current_data: pd.DataFrame, date: str) -> Dict:
        """Execute trades based on predictions."""
        # Limit number of positions
        max_positions = 10
        
        # Get current prices
        current_prices = current_data[current_data['date'] == date].set_index('ticker')['close']
        
        # Close positions for SELL signals
        for ticker in list(portfolio['positions'].keys()):
            if ticker in predictions['ticker'].values:
                pred_row = predictions[predictions['ticker'] == ticker].iloc[0]
                if pred_row['prediction'] == 'SELL' and ticker in current_prices:
                    # Sell position
                    shares = portfolio['positions'][ticker]['shares']
                    sell_price = current_prices[ticker]
                    portfolio['cash'] += shares * sell_price
                    del portfolio['positions'][ticker]
                    logger.debug(f"📉 Sold {ticker}: {shares} shares @ ${sell_price:.2f}")
        
        # Open new positions for BUY signals
        buy_signals = predictions[predictions['prediction'] == 'BUY'].head(max_positions)
        
        for _, row in buy_signals.iterrows():
            ticker = row['ticker']
            if ticker not in portfolio['positions'] and ticker in current_prices:
                # Calculate position size
                position_value = portfolio['cash'] * row['suggested_position_size']
                buy_price = current_prices[ticker]
                shares = int(position_value / buy_price)
                
                if shares > 0 and portfolio['cash'] >= shares * buy_price:
                    # Buy position
                    portfolio['positions'][ticker] = {
                        'shares': shares,
                        'entry_price': buy_price,
                        'entry_date': date
                    }
                    portfolio['cash'] -= shares * buy_price
                    logger.debug(f"📈 Bought {ticker}: {shares} shares @ ${buy_price:.2f}")
        
        return portfolio
    
    def _calculate_portfolio_value(self, portfolio: Dict, current_data: pd.DataFrame) -> float:
        """Calculate total portfolio value."""
        total_value = portfolio['cash']
        
        if portfolio['positions']:
            latest_date = current_data['date'].max()
            current_prices = current_data[current_data['date'] == latest_date].set_index('ticker')['close']
            
            for ticker, position in portfolio['positions'].items():
                if ticker in current_prices:
                    total_value += position['shares'] * current_prices[ticker]
        
        return total_value
    
    def _calculate_backtest_metrics(self, portfolio: Dict, initial_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics."""
        history_df = pd.DataFrame(portfolio['history'])
        
        # Calculate returns
        history_df['daily_return_pct'] = history_df['portfolio_value'].pct_change()
        
        # Calculate metrics
        total_return = (history_df['portfolio_value'].iloc[-1] - initial_capital) / initial_capital
        
        # Sharpe ratio (assuming 252 trading days)
        sharpe_ratio = (history_df['daily_return_pct'].mean() * 252) / (history_df['daily_return_pct'].std() * np.sqrt(252))
        
        # Maximum drawdown
        cumulative = (1 + history_df['daily_return_pct']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_days = (history_df['daily_return_pct'] > 0).sum()
        total_days = len(history_df) - 1
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': history_df['portfolio_value'].iloc[-1],
            'total_trades': len(portfolio['history']),
            'avg_daily_return': history_df['daily_return_pct'].mean(),
            'return_volatility': history_df['daily_return_pct'].std(),
            'history': history_df
        }
        
        return metrics
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results to disk."""
        save_path = self.model_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        save_dict = {
            'metrics': results['metrics'],
            'training_time': results['training_time'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        logger.info(f"💾 Training results saved to {save_path}")
    
    def _save_daily_predictions(self, predictions_df: pd.DataFrame):
        """Save daily predictions."""
        date_str = datetime.now().strftime('%Y%m%d')
        save_path = self.model_dir / f"predictions_{date_str}.csv"
        predictions_df.to_csv(save_path, index=False)
        logger.info(f"💾 Predictions saved to {save_path}")
    
    def _save_backtest_results(self, results: Dict[str, Any]):
        """Save backtest results."""
        save_path = self.model_dir / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert DataFrame to dict for JSON
        save_dict = {k: v for k, v in results.items() if k != 'history'}
        save_dict['summary'] = {
            'total_return': results['total_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'win_rate': results['win_rate']
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        # Save detailed history as CSV
        history_path = self.model_dir / f"backtest_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results['history'].to_csv(history_path, index=False)
        
        logger.info(f"💾 Backtest results saved to {save_path}")
    
    def load_models(self, model_path: str):
        """Load pre-trained models."""
        logger.info(f"📂 Loading models from {model_path}")
        
        import joblib
        
        model_dir = Path(model_path)
        
        # Load each component
        self.feature_engineer = joblib.load(model_dir / "feature_engineer.pkl")

        # Ensure feature engineer has required attributes for compatibility
        if not hasattr(self.feature_engineer, 'enable_cache'):
            self.feature_engineer.enable_cache = True
        if not hasattr(self.feature_engineer, 'cache_dir'):
            self.feature_engineer.cache_dir = Path("cache/features")
            self.feature_engineer.cache_dir.mkdir(parents=True, exist_ok=True)
        if not hasattr(self.feature_engineer, 'n_jobs'):
            self.feature_engineer.n_jobs = -1

        self.regime_detector = joblib.load(model_dir / "regime_detector.pkl")
        self.ensemble = StackedEnsemblePredictor()
        self.ensemble.load_ensemble(str(model_dir / "ensemble.pkl"))
        
        # Load LSTM
        if (model_dir / "lstm_model.pth").exists():
            # Reconstruct LSTM architecture with correct dimensions from saved model
            feature_dims = {'short': 15, 'medium': 13, 'long': 9}  # Match saved model dims
            self.lstm_model = MultiStreamLSTM(feature_dims)

            # Load state dict with error handling for dimension mismatches
            try:
                self.lstm_model.load_state_dict(torch.load(model_dir / "lstm_model.pth", map_location=self.device))

                # Ensure device attribute is properly set after loading
                self.lstm_model.device = self.device
                self.lstm_model = self.lstm_model.to(self.device)
                self.lstm_model.eval()
                logger.info("✅ LSTM model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ LSTM loading failed: {e}")
                self.lstm_model = None
        
        logger.info("✅ Models loaded successfully")


def main():
    """Main function to run the complete trading system."""
    # Example usage
    logger.info("🚀 Stock Trading Prediction System")
    logger.info("=" * 60)
    
    # Initialize predictor
    predictor = StockTradingPredictor(model_dir="./trading_models")
    
    # Option 1: Train from scratch
    if not (Path("./trading_models") / "ensemble.pkl").exists():
        logger.info("🔧 Training new models...")
        
        # Download data using the Polygon downloader
        from polygon_downloader import download_2year_data
        data = download_2year_data()
        
        # Save data
        data_file = "stock_data_2years.parquet"
        data.to_parquet(data_file)
        
        # Train models
        training_results = predictor.train_from_polygon_data(data_file)
        
        logger.info(f"✅ Training complete! Accuracy: {training_results['metrics']['accuracy']*100:.2f}%")
    else:
        # Option 2: Load existing models
        logger.info("📂 Loading existing models...")
        predictor.load_models("./trading_models/final_models_20240101_120000")  # Update path
    
    # Make daily predictions
    logger.info("\n🔮 Making predictions for tomorrow...")
    
    # Load recent data (last 60+ days)
    recent_data = pd.read_parquet("stock_data_2years.parquet")
    recent_data = recent_data[recent_data['date'] >= '2023-10-01']  # Last 3 months
    
    # Get predictions
    predictions = predictor.predict_next_day(recent_data)
    
    # Display top predictions
    logger.info("\n📊 Top 10 Stock Predictions for Tomorrow:")
    logger.info("=" * 80)
    print(predictions[['ticker', 'prediction', 'prediction_proba', 'confidence', 
                      'suggested_position_size']].head(10).to_string(index=False))
    
    # Run backtest
    logger.info("\n📈 Running backtest...")
    backtest_results = predictor.backtest_strategy(
        recent_data,
        start_date='2023-10-01',
        end_date='2024-01-01',
        initial_capital=100000
    )
    
    logger.info(f"\n📊 Backtest Results:")
    logger.info(f"   Total Return: {backtest_results['total_return']*100:.2f}%")
    logger.info(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    logger.info(f"   Max Drawdown: {backtest_results['max_drawdown']*100:.2f}%")
    logger.info(f"   Win Rate: {backtest_results['win_rate']*100:.2f}%")
    logger.info(f"   Final Portfolio Value: ${backtest_results['final_value']:,.2f}")
    
    logger.info("\n✅ System ready for daily trading!")


if __name__ == "__main__":
    main()
