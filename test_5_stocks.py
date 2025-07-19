"""
Test weekly analysis with 5 random stocks to measure performance
"""
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from weekly_analysis import run_weekly_analysis
from main_trading_system import StockTradingPredictor
from advanced_feature_engineering import AdvancedFeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_5_random_stocks():
    """Test the weekly analysis with 5 randomly selected stocks."""
    print("🎯 TESTING WEEKLY ANALYSIS WITH 5 RANDOM STOCKS")
    print("=" * 60)
    
    # Load the data to see available tickers
    print('📊 Loading stock data to check available tickers...')
    start_load = time.time()
    
    # Load with optimized dtypes
    df = pd.read_csv('data/stock_data_2year.csv')
    df['ticker'] = df['ticker'].astype('category')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype('float32')
    df['volume'] = df['volume'].astype('int64')
    df['date'] = pd.to_datetime(df['date'])
    
    load_time = time.time() - start_load
    
    print(f'✅ Data loaded in {load_time:.2f} seconds')
    print(f'   Total records: {len(df):,}')
    print(f'   Date range: {df["date"].min()} to {df["date"].max()}')
    print(f'   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')

    # Get available tickers with sufficient data
    ticker_counts = df['ticker'].value_counts()
    valid_tickers = ticker_counts[ticker_counts >= 200].index  # At least 200 days of data
    
    print(f'   Available tickers with ≥200 days: {len(valid_tickers)}')
    
    # Pick 5 random tickers
    np.random.seed(42)  # For reproducible results
    selected_tickers = np.random.choice(valid_tickers, size=5, replace=False)
    print(f'\n🎯 Selected 5 random tickers: {list(selected_tickers)}')

    # Filter data to only selected tickers
    test_data = df[df['ticker'].isin(selected_tickers)].copy()
    test_data = test_data.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f'\n📊 Test Dataset Summary:')
    for ticker in selected_tickers:
        ticker_data = test_data[test_data['ticker'] == ticker]
        print(f'   {ticker}: {len(ticker_data)} records ({ticker_data["date"].min().date()} to {ticker_data["date"].max().date()})')
    
    print(f'\n   Total test records: {len(test_data):,}')
    print(f'   Test data memory: {test_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
    
    # Initialize predictor
    print(f'\n🚀 Initializing optimized predictor...')
    init_start = time.time()
    
    predictor = StockTradingPredictor(model_dir="./test_checkpoints", use_gpu=False)
    
    # Load models
    model_path = "./test_checkpoints/final_models_20250713_185556"
    try:
        predictor.load_models(model_path)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"⚠️ Model loading issue: {e}")
        # Load manually with optimized feature engineer
        import joblib
        from stacked_ensemble import StackedEnsemblePredictor
        
        predictor.feature_engineer = AdvancedFeatureEngineer(use_gpu=False, enable_cache=True, n_jobs=-1)
        predictor.regime_detector = joblib.load(f"{model_path}/regime_detector.pkl")
        predictor.ensemble = StackedEnsemblePredictor()
        predictor.ensemble.load_ensemble(f"{model_path}/ensemble.pkl")
        predictor.lstm_model = None
        print("✅ Models loaded manually with optimized feature engineer")
    
    init_time = time.time() - init_start
    print(f'   Initialization time: {init_time:.2f} seconds')
    
    # Run feature engineering
    print(f'\n🔧 Running feature engineering...')
    feature_start = time.time()
    
    try:
        data_with_features = predictor.feature_engineer.compute_all_features(test_data)
        feature_time = time.time() - feature_start
        
        print(f'✅ Feature engineering completed in {feature_time:.2f} seconds')
        print(f'   Original columns: {len(test_data.columns)}')
        print(f'   Total columns after features: {len(data_with_features.columns)}')
        print(f'   Features created: {len(data_with_features.columns) - len(test_data.columns)}')
        print(f'   Final data memory: {data_with_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
        
    except Exception as e:
        print(f'❌ Feature engineering failed: {e}')
        import traceback
        traceback.print_exc()
        return None
    
    # Run predictions
    print(f'\n🔮 Making predictions...')
    pred_start = time.time()
    
    try:
        predictions = predictor.predict_next_day(test_data)
        pred_time = time.time() - pred_start
        
        if predictions is not None and not predictions.empty:
            print(f'✅ Predictions completed in {pred_time:.2f} seconds')
            print(f'   Predictions generated: {len(predictions)}')
            
            # Show results
            print(f'\n📊 PREDICTION RESULTS:')
            predictions_sorted = predictions.sort_values('prediction_proba', ascending=False)
            
            for i, (_, row) in enumerate(predictions_sorted.iterrows(), 1):
                confidence_emoji = "🔥" if row['confidence'] > 0.8 else "⭐" if row['confidence'] > 0.6 else "📈"
                prediction_emoji = "🚀" if row['prediction'] == 'BUY' else "⬇️"
                
                print(f"   {i}. {confidence_emoji} {row['ticker']:6s} {prediction_emoji} {row['prediction']:4s} "
                      f"| Prob: {row['prediction_proba']:.3f} | Conf: {row['confidence']:.3f} "
                      f"| Price: ${row['current_price']:.2f}")
        else:
            print(f'❌ No predictions generated')
            pred_time = time.time() - pred_start
            
    except Exception as e:
        print(f'❌ Prediction failed: {e}')
        import traceback
        traceback.print_exc()
        pred_time = time.time() - pred_start
    
    # Performance summary
    total_time = load_time + init_time + feature_time + pred_time
    
    print(f'\n📊 PERFORMANCE SUMMARY:')
    print(f'   Data loading:        {load_time:.2f}s')
    print(f'   Model initialization: {init_time:.2f}s') 
    print(f'   Feature engineering:  {feature_time:.2f}s')
    print(f'   Predictions:         {pred_time:.2f}s')
    print(f'   ─────────────────────────────────')
    print(f'   Total time:          {total_time:.2f}s')
    
    print(f'\n💾 MEMORY EFFICIENCY:')
    print(f'   Original data:       {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
    print(f'   Test data (5 stocks): {test_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
    if 'data_with_features' in locals():
        print(f'   With features:       {data_with_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
    
    print(f'\n🎉 TEST COMPLETE!')
    print(f'   Processing 5 stocks took {total_time:.2f} seconds')
    print(f'   Estimated time for all {len(valid_tickers)} stocks: {total_time * len(valid_tickers) / 5 / 60:.1f} minutes')
    
    return {
        'total_time': total_time,
        'feature_time': feature_time,
        'pred_time': pred_time,
        'stocks_processed': 5,
        'records_processed': len(test_data)
    }

if __name__ == "__main__":
    results = test_5_random_stocks()
