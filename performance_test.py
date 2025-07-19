"""
Performance test for optimized CPU-based feature engineering
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from advanced_feature_engineering import AdvancedFeatureEngineer

def generate_test_data(n_tickers=10, n_days=500):
    """Generate synthetic stock data for testing."""
    print(f"📊 Generating test data: {n_tickers} tickers, {n_days} days each")
    
    data = []
    base_date = datetime.now() - timedelta(days=n_days)
    
    for ticker_idx in range(n_tickers):
        ticker = f"TEST{ticker_idx:03d}"
        
        # Generate realistic price data with random walk
        np.random.seed(ticker_idx)  # Reproducible results
        base_price = 50 + np.random.random() * 100
        
        for day in range(n_days):
            date = base_date + timedelta(days=day)
            
            # Random walk with some volatility
            price_change = np.random.normal(0, 0.02)  # 2% daily volatility
            base_price *= (1 + price_change)
            
            # Generate OHLC data
            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = base_price * (1 + np.random.normal(0, 0.005))
            close = base_price
            volume = int(np.random.lognormal(10, 1))  # Log-normal volume distribution
            
            data.append({
                'ticker': ticker,
                'date': date.strftime('%Y-%m-%d'),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df):,} records")
    return df

def test_feature_engineering_performance():
    """Test the performance of different feature engineering approaches."""
    print("🚀 PERFORMANCE TEST: Optimized CPU Feature Engineering")
    print("=" * 60)
    
    # Generate test data
    test_data = generate_test_data(n_tickers=20, n_days=300)  # 6,000 records total
    
    print(f"\n📊 Test Data Summary:")
    print(f"   Total records: {len(test_data):,}")
    print(f"   Tickers: {test_data['ticker'].nunique()}")
    print(f"   Date range: {test_data['date'].min()} to {test_data['date'].max()}")
    print(f"   Memory usage: {test_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Test 1: Original approach (single-threaded)
    print(f"\n🔄 Test 1: Single-threaded CPU processing")
    fe_single = AdvancedFeatureEngineer(use_gpu=False, n_jobs=1, enable_cache=False)
    
    start_time = time.time()
    result_single = fe_single.compute_trend_strength_features(test_data.copy())
    result_single = fe_single.compute_momentum_features(result_single)
    single_time = time.time() - start_time
    
    print(f"   ⏱️  Time: {single_time:.2f} seconds")
    print(f"   📈 Features created: {len(result_single.columns) - len(test_data.columns)}")
    
    # Test 2: Parallel processing approach
    print(f"\n⚡ Test 2: Multi-threaded CPU processing")
    fe_parallel = AdvancedFeatureEngineer(use_gpu=False, n_jobs=-1, enable_cache=False)
    
    start_time = time.time()
    result_parallel = fe_parallel.compute_features_parallel(test_data.copy())
    parallel_time = time.time() - start_time
    
    print(f"   ⏱️  Time: {parallel_time:.2f} seconds")
    print(f"   📈 Features created: {len(result_parallel.columns) - len(test_data.columns)}")
    print(f"   🚀 Speedup: {single_time/parallel_time:.2f}x")
    
    # Test 3: With caching enabled
    print(f"\n💾 Test 3: With caching enabled")
    fe_cached = AdvancedFeatureEngineer(use_gpu=False, n_jobs=-1, enable_cache=True)
    
    # First run (cache miss)
    start_time = time.time()
    result_cached_1 = fe_cached.compute_trend_strength_features(test_data.copy())
    result_cached_1 = fe_cached.compute_momentum_features(result_cached_1)
    cached_time_1 = time.time() - start_time
    
    # Second run (cache hit)
    start_time = time.time()
    result_cached_2 = fe_cached.compute_trend_strength_features(test_data.copy())
    result_cached_2 = fe_cached.compute_momentum_features(result_cached_2)
    cached_time_2 = time.time() - start_time
    
    print(f"   ⏱️  First run (cache miss): {cached_time_1:.2f} seconds")
    print(f"   ⏱️  Second run (cache hit): {cached_time_2:.2f} seconds")
    print(f"   🚀 Cache speedup: {cached_time_1/cached_time_2:.2f}x")
    
    # Validate results are consistent
    print(f"\n✅ Validation:")
    
    # Check if parallel and single-threaded results are similar
    common_features = [col for col in result_single.columns if col in result_parallel.columns]
    differences = []
    
    for feature in common_features:
        if feature not in ['ticker', 'date']:
            single_vals = result_single[feature].dropna()
            parallel_vals = result_parallel[feature].dropna()
            
            if len(single_vals) > 0 and len(parallel_vals) > 0:
                # Compare using correlation (should be very close to 1.0)
                if len(single_vals) == len(parallel_vals):
                    corr = np.corrcoef(single_vals, parallel_vals)[0, 1]
                    if not np.isnan(corr):
                        differences.append(abs(1.0 - corr))
    
    if differences:
        max_diff = max(differences)
        avg_diff = np.mean(differences)
        print(f"   📊 Feature correlation: avg={1-avg_diff:.6f}, min={1-max_diff:.6f}")
        if max_diff < 0.001:
            print(f"   ✅ Results are consistent (max difference: {max_diff:.6f})")
        else:
            print(f"   ⚠️  Results differ (max difference: {max_diff:.6f})")
    
    # Performance summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   Single-threaded: {single_time:.2f}s")
    print(f"   Multi-threaded:  {parallel_time:.2f}s ({single_time/parallel_time:.2f}x faster)")
    print(f"   With caching:    {cached_time_2:.2f}s ({single_time/cached_time_2:.2f}x faster)")
    
    # Memory efficiency
    print(f"\n💾 MEMORY EFFICIENCY:")
    print(f"   Input data:      {test_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   With features:   {result_parallel.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   Memory increase: {(result_parallel.memory_usage(deep=True).sum() / test_data.memory_usage(deep=True).sum() - 1) * 100:.1f}%")
    
    # Clean up cache
    fe_cached.clear_cache()
    
    return {
        'single_time': single_time,
        'parallel_time': parallel_time,
        'cached_time_cold': cached_time_1,
        'cached_time_warm': cached_time_2,
        'speedup_parallel': single_time / parallel_time,
        'speedup_cache': cached_time_1 / cached_time_2,
        'features_created': len(result_parallel.columns) - len(test_data.columns)
    }

if __name__ == "__main__":
    results = test_feature_engineering_performance()
    
    print(f"\n🎉 OPTIMIZATION COMPLETE!")
    print(f"   Overall speedup: {results['speedup_parallel']:.2f}x with parallel processing")
    print(f"   Cache speedup: {results['speedup_cache']:.2f}x with caching")
    print(f"   Features created: {results['features_created']}")
