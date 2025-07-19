"""
Test script for feature engineering caching functionality.
This script demonstrates how the caching system works and its performance benefits.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
import logging
from advanced_feature_engineering import AdvancedFeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_data():
    """Load sample stock data for testing."""
    try:
        # Try to load the actual data file
        data_file = "data/stock_data_2year.csv"
        if Path(data_file).exists():
            logger.info(f"📊 Loading data from {data_file}")
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            logger.warning(f"⚠️ Data file {data_file} not found, creating sample data")
            return create_sample_data()
    except Exception as e:
        logger.warning(f"⚠️ Error loading data: {e}, creating sample data")
        return create_sample_data()

def create_sample_data():
    """Create sample stock data for testing if real data is not available."""
    logger.info("🔧 Creating sample stock data for testing...")
    
    # Create sample data for 3 tickers over 500 days
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
    
    data = []
    for ticker in tickers:
        # Generate realistic stock price data
        np.random.seed(hash(ticker) % 2**32)  # Consistent random data per ticker
        
        initial_price = np.random.uniform(100, 300)
        prices = [initial_price]
        
        for i in range(1, len(dates)):
            # Random walk with slight upward bias
            change = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # Ensure positive prices
        
        for i, date in enumerate(dates):
            # Generate OHLCV data
            close = prices[i]
            high = close * np.random.uniform(1.0, 1.05)
            low = close * np.random.uniform(0.95, 1.0)
            open_price = np.random.uniform(low, high)
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'ticker': ticker,
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
    
    df = pd.DataFrame(data)
    logger.info(f"✅ Created sample data: {len(df)} records for {len(tickers)} tickers")
    return df

def test_caching_performance():
    """Test the performance benefits of caching."""
    logger.info("🚀 Testing feature engineering caching performance...")
    
    # Load data
    df = load_sample_data()
    logger.info(f"📊 Loaded {len(df)} records for {df['ticker'].nunique()} tickers")
    
    # Test 1: First run (no cache)
    logger.info("\n" + "="*60)
    logger.info("TEST 1: First run (building cache)")
    logger.info("="*60)
    
    fe_cached = AdvancedFeatureEngineer(use_gpu=True, enable_cache=True)
    
    start_time = time.time()
    result1 = fe_cached.compute_all_features(df)
    first_run_time = time.time() - start_time
    
    logger.info(f"✅ First run completed in {first_run_time:.2f} seconds")
    logger.info(f"📊 Output shape: {result1.shape}")
    
    # Test 2: Second run (using cache)
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Second run (using cache)")
    logger.info("="*60)
    
    start_time = time.time()
    result2 = fe_cached.compute_all_features(df)
    second_run_time = time.time() - start_time
    
    logger.info(f"✅ Second run completed in {second_run_time:.2f} seconds")
    logger.info(f"📊 Output shape: {result2.shape}")
    
    # Test 3: Incremental update (add new data)
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Incremental update (new data)")
    logger.info("="*60)
    
    # Add 10 new days of data
    last_date = df['date'].max()
    new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='D')
    
    new_data = []
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].tail(1)
        last_price = ticker_data['close'].iloc[0]
        
        for date in new_dates:
            # Generate new price data
            change = np.random.normal(0.001, 0.02)
            new_price = last_price * (1 + change)
            high = new_price * np.random.uniform(1.0, 1.05)
            low = new_price * np.random.uniform(0.95, 1.0)
            open_price = np.random.uniform(low, high)
            volume = np.random.randint(1000000, 10000000)
            
            new_data.append({
                'ticker': ticker,
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': new_price,
                'volume': volume
            })
            last_price = new_price
    
    new_df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
    
    start_time = time.time()
    result3 = fe_cached.compute_all_features(new_df)
    incremental_time = time.time() - start_time
    
    logger.info(f"✅ Incremental update completed in {incremental_time:.2f} seconds")
    logger.info(f"📊 Output shape: {result3.shape}")
    
    # Test 4: Force recompute
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Force recompute (ignore cache)")
    logger.info("="*60)
    
    start_time = time.time()
    result4 = fe_cached.compute_all_features(df, force_recompute=True)
    recompute_time = time.time() - start_time
    
    logger.info(f"✅ Force recompute completed in {recompute_time:.2f} seconds")
    logger.info(f"📊 Output shape: {result4.shape}")
    
    # Performance summary
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*60)
    logger.info(f"First run (no cache):     {first_run_time:.2f} seconds")
    logger.info(f"Second run (cached):      {second_run_time:.2f} seconds")
    logger.info(f"Incremental update:       {incremental_time:.2f} seconds")
    logger.info(f"Force recompute:          {recompute_time:.2f} seconds")
    logger.info(f"")
    logger.info(f"Cache speedup:            {first_run_time/second_run_time:.1f}x faster")
    logger.info(f"Incremental speedup:      {first_run_time/incremental_time:.1f}x faster")
    
    # Test cache management
    logger.info("\n" + "="*60)
    logger.info("CACHE MANAGEMENT")
    logger.info("="*60)
    
    # Show cache directory contents
    cache_dir = fe_cached.cache_dir
    cache_files = list(cache_dir.glob("*.pkl"))
    logger.info(f"📁 Cache directory: {cache_dir}")
    logger.info(f"📄 Cache files: {len(cache_files)}")
    for file in cache_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        logger.info(f"   {file.name} ({size_mb:.1f} MB)")
    
    # Test cache clearing
    logger.info("\n🗑️ Testing cache clearing...")
    fe_cached.clear_cache()
    
    cache_files_after = list(cache_dir.glob("*.pkl"))
    logger.info(f"📄 Cache files after clearing: {len(cache_files_after)}")
    
    return {
        'first_run_time': first_run_time,
        'cached_run_time': second_run_time,
        'incremental_time': incremental_time,
        'recompute_time': recompute_time,
        'speedup': first_run_time / second_run_time if second_run_time > 0 else 0
    }

def test_trend_strength_caching():
    """Test specific trend strength feature caching."""
    logger.info("\n🚀 Testing trend strength feature caching...")
    
    df = load_sample_data()
    fe = AdvancedFeatureEngineer(use_gpu=True, enable_cache=True)
    
    # Test cached trend strength computation
    start_time = time.time()
    result = fe.compute_trend_strength_features_cached(df)
    trend_time = time.time() - start_time
    
    logger.info(f"✅ Trend strength features computed in {trend_time:.2f} seconds")
    logger.info(f"📊 Output shape: {result.shape}")
    
    # Test second run (should use cache)
    start_time = time.time()
    result2 = fe.compute_trend_strength_features_cached(df)
    cached_trend_time = time.time() - start_time
    
    logger.info(f"✅ Cached trend strength features in {cached_trend_time:.2f} seconds")
    logger.info(f"🚀 Trend caching speedup: {trend_time/cached_trend_time:.1f}x faster")

if __name__ == "__main__":
    logger.info("🧪 Starting feature engineering caching tests...")
    
    try:
        # Test overall caching performance
        performance_results = test_caching_performance()
        
        # Test specific trend strength caching
        test_trend_strength_caching()
        
        logger.info("\n🎉 All caching tests completed successfully!")
        logger.info(f"💾 Overall caching provides {performance_results['speedup']:.1f}x speedup")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
