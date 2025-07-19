"""
Daily Stock Analysis Script
Runs predictions for all available tickers for the next trading day
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Optimized CPU processing setup
import torch
GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    print(f"� GPU detected: {torch.cuda.get_device_name(0)} (using CPU for optimal performance)")
else:
    print("💻 Weekly Analysis using optimized CPU processing")

# Import our main trading system
from main_trading_system import StockTradingPredictor
from advanced_feature_engineering import AdvancedFeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def optimize_data_types(data):
    """Optimize data types for better memory usage and processing speed."""
    # Ensure optimal data types are maintained
    if 'ticker' in data.columns and data['ticker'].dtype != 'category':
        data['ticker'] = data['ticker'].astype('category')

    # Ensure float32 for price columns
    for col in ['open', 'high', 'low', 'close']:
        if col in data.columns and data[col].dtype != 'float32':
            data[col] = data[col].astype('float32')

    return data

# Remove the old predict_without_lstm function - we'll use the working predict_next_day method instead


def run_daily_analysis():
    """Run analysis for all available tickers and rank them for the next trading day."""
    logger.info("🚀 DAILY STOCK ANALYSIS - Finding Best Options for Next Trading Day")
    logger.info("=" * 70)

    # Initialize predictor with optimized CPU processing
    predictor = StockTradingPredictor(model_dir="./test_checkpoints", use_gpu=False)

    # Load models using the same approach as the working test script
    model_path = "./test_checkpoints/final_models_20250713_185556"
    if Path(model_path).exists():
        logger.info("📂 Loading existing models...")
        try:
            predictor.load_models(model_path)
            logger.info("✅ Models loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Model loading issue: {e}")
            # Load manually with optimized feature engineer (same as test_5_stocks.py)
            import joblib
            from stacked_ensemble import StackedEnsemblePredictor

            predictor.feature_engineer = AdvancedFeatureEngineer(use_gpu=False, enable_cache=True, n_jobs=-1)
            predictor.regime_detector = joblib.load(f"{model_path}/regime_detector.pkl")
            predictor.ensemble = StackedEnsemblePredictor()
            predictor.ensemble.load_ensemble(f"{model_path}/ensemble.pkl")
            predictor.lstm_model = None
            logger.info("✅ Models loaded manually with optimized feature engineer")
    else:
        logger.error("❌ No trained models found! Please train models first.")
        return None
    
    # Ensure we have the latest data before making predictions
    logger.info("🔄 Checking for latest data...")
    data_file = "Chris's_Copy/stock_data_2years_20250716_190514.csv"

    try:
        # Check if data needs updating
        if Path(data_file).exists():
            df_check = pd.read_csv(data_file)
            df_check['date'] = pd.to_datetime(df_check['date'])
            latest_data_date = df_check['date'].max()

            # Check if data is from today or yesterday (accounting for market hours)
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)

            if latest_data_date.date() >= yesterday:
                logger.info(f"✅ Data is current (through {latest_data_date.strftime('%Y-%m-%d')})")
            else:
                logger.info(f"🔄 Data may need updating (latest: {latest_data_date.strftime('%Y-%m-%d')})")
                logger.info("💡 Consider running: cd Chris's_Copy && python update_latest_data.py")
        else:
            logger.warning(f"⚠️ Data file not found: {data_file}")
    except Exception as e:
        logger.warning(f"⚠️ Data freshness check failed: {e}")

    # Load recent data with optimized preprocessing
    logger.info("📊 Loading and preprocessing stock data...")
    try:
        # Load data with optimized dtypes for memory efficiency
        # Use the most recent data file from Chris's_Copy directory
        recent_data = pd.read_csv("Chris's_Copy/stock_data_2years_20250716_190514.csv")

        # Optimize dtypes after loading
        recent_data['ticker'] = recent_data['ticker'].astype('category')
        for col in ['open', 'high', 'low', 'close']:
            recent_data[col] = recent_data[col].astype('float32')
        recent_data['volume'] = recent_data['volume'].astype('int64')
        recent_data['date'] = pd.to_datetime(recent_data['date'], format='mixed')

        # Optimize data for processing
        logger.info(f"📊 Loaded {len(recent_data):,} records for {recent_data['ticker'].nunique()} tickers")
        logger.info(f"📅 Date range: {recent_data['date'].min()} to {recent_data['date'].max()}")

        # Sort data for optimal processing (important for rolling operations)
        recent_data = recent_data.sort_values(['ticker', 'date']).reset_index(drop=True)

        # Remove any duplicate records
        initial_len = len(recent_data)
        recent_data = recent_data.drop_duplicates(subset=['ticker', 'date'], keep='last')
        if len(recent_data) < initial_len:
            logger.info(f"🧹 Removed {initial_len - len(recent_data)} duplicate records")

        # Filter out tickers with insufficient data (less than 200 days for better predictions)
        ticker_counts = recent_data['ticker'].value_counts()
        valid_tickers = ticker_counts[ticker_counts >= 200].index

        # Filter to approximately 4,179 most actively traded stocks
        # Calculate average daily volume for each ticker
        ticker_avg_volume = recent_data.groupby('ticker')['volume'].mean().sort_values(ascending=False)

        # Select top 4,179 tickers by average volume that also have sufficient data
        top_tickers = ticker_avg_volume[ticker_avg_volume.index.isin(valid_tickers)].head(4179).index
        recent_data = recent_data[recent_data['ticker'].isin(top_tickers)]

        logger.info(f"📈 Using {len(top_tickers)} most actively traded tickers with sufficient data (≥200 days)")
        logger.info(f"🎯 Sample tickers: {', '.join(list(top_tickers[:10]))}{'...' if len(top_tickers) > 10 else ''}")

        # Optimize memory usage
        logger.info(f"💾 Memory usage: {recent_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return None
    
    # Get predictions for all tickers using the working approach
    logger.info("\n🔮 Making predictions for all tickers...")
    try:
        predictions = predictor.predict_next_day(recent_data)

        if predictions is None or predictions.empty:
            logger.error("❌ No predictions generated!")
            return None

        # Calculate the prediction target date (next trading day after latest data)
        latest_data_date = recent_data['date'].max()
        # Add 1 business day to get next trading day
        prediction_target_date = pd.bdate_range(start=latest_data_date, periods=2)[1]

        # Add prediction target date to all predictions
        predictions['prediction_date'] = prediction_target_date
        predictions['data_through_date'] = latest_data_date

        logger.info(f"📅 Data through: {latest_data_date.strftime('%Y-%m-%d')}")
        logger.info(f"🎯 Predicting for: {prediction_target_date.strftime('%Y-%m-%d')} (next trading day)")

        # Sort by prediction probability (confidence in BUY signal)
        predictions_sorted = predictions.sort_values('prediction_proba', ascending=False)
        
        # Display results
        logger.info(f"\n📊 DAILY STOCK RANKINGS - BEST OPTIONS FOR {prediction_target_date.strftime('%Y-%m-%d')}:")
        logger.info("=" * 80)

        # Filter for BUY signals only and get top 10
        buy_predictions = predictions_sorted[predictions_sorted['prediction'] == 'BUY'].head(10)

        print(f"\n🏆 TOP 10 BUY RECOMMENDATIONS FOR {prediction_target_date.strftime('%Y-%m-%d')}:")
        print("=" * 60)

        for i, (_, row) in enumerate(buy_predictions.iterrows(), 1):
            confidence_emoji = "🔥" if row['confidence'] > 0.8 else "⭐" if row['confidence'] > 0.6 else "📈"

            print(f"{i:2d}. {confidence_emoji} {row['ticker']:6s} 🚀 BUY "
                  f"| Prob: {row['prediction_proba']:.3f} | Conf: {row['confidence']:.3f} "
                  f"| Price: ${row['current_price']:.2f}")

        # Show detailed analysis for top 10 BUY picks
        logger.info("\n🔍 DETAILED ANALYSIS - TOP 10 BUY PICKS:")
        logger.info("=" * 60)

        for i, (_, row) in enumerate(buy_predictions.iterrows(), 1):
            print(f"\n#{i} {row['ticker']} - {row['prediction']}")
            print(f"   Current Price: ${row['current_price']:.2f}")
            print(f"   Prediction Probability: {row['prediction_proba']:.3f}")
            print(f"   Model Confidence: {row['confidence']:.3f}")
            print(f"   Market Regime: {row['regime']}")
            if 'suggested_position_size' in row:
                print(f"   Suggested Position Size: {row['suggested_position_size']}")

        # Also show overall top performers regardless of signal
        logger.info("\n📊 OVERALL TOP PERFORMERS (All Signals):")
        logger.info("=" * 50)

        for i, (_, row) in enumerate(predictions_sorted.head(20).iterrows(), 1):
            confidence_emoji = "🔥" if row['confidence'] > 0.8 else "⭐" if row['confidence'] > 0.6 else "📈"
            prediction_emoji = "🚀" if row['prediction'] == 'BUY' else "⬇️" if row['prediction'] == 'SELL' else "📊"

            print(f"{i:2d}. {confidence_emoji} {row['ticker']:6s} {prediction_emoji} {row['prediction']:4s} "
                  f"| Prob: {row['prediction_proba']:.3f} | Conf: {row['confidence']:.3f} "
                  f"| Price: ${row['current_price']:.2f}")
        
        # Create predictions directory if it doesn't exist
        Path("predictions").mkdir(exist_ok=True)

        # Save results with prediction target date in filename
        prediction_date_str = prediction_target_date.strftime('%Y%m%d')
        timestamp = datetime.now().strftime('%H%M%S')
        output_file = f"predictions/daily_predictions_for_{prediction_date_str}_{timestamp}.csv"
        predictions_sorted.to_csv(output_file, index=False)

        logger.info(f"\n💾 Full analysis saved to: {output_file}")
        logger.info(f"📊 Predictions for next trading day: {prediction_target_date.strftime('%Y-%m-%d')}")
        
        # Summary statistics
        buy_signals = len(predictions_sorted[predictions_sorted['prediction'] == 'BUY'])
        sell_signals = len(predictions_sorted[predictions_sorted['prediction'] == 'SELL'])
        avg_confidence = predictions_sorted['confidence'].mean()
        
        logger.info(f"\n📈 SUMMARY:")
        logger.info(f"   Total Tickers Analyzed: {len(predictions_sorted)}")
        logger.info(f"   BUY Signals: {buy_signals}")
        logger.info(f"   SELL Signals: {sell_signals}")
        logger.info(f"   Average Confidence: {avg_confidence:.3f}")
        
        # Show model performance metrics
        metrics_file = Path(model_path) / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            logger.info(f"\n🎯 MODEL PERFORMANCE:")
            logger.info(f"   Accuracy: {metrics['accuracy']*100:.1f}%")
            logger.info(f"   AUC Score: {metrics['auc']:.3f}")
            logger.info(f"   Precision: {metrics['precision']:.3f}")
            logger.info(f"   Recall: {metrics['recall']:.3f}")
        
        return predictions_sorted
        
    except Exception as e:
        logger.error(f"❌ Error making predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the daily analysis
    results = run_daily_analysis()

    if results is not None:
        print("\n" + "="*60)
        print("🎉 DAILY ANALYSIS COMPLETE!")
        print("="*60)
        print(f"✅ Analyzed {len(results)} stocks")
        print(f"📊 Results saved to predictions/ directory")
        print(f"🚀 Ready for next trading day decisions!")
    else:
        print("\n❌ Analysis failed. Please check the logs above.")


