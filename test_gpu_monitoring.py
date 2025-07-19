"""
Test GPU monitoring during feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from advanced_feature_engineering import AdvancedFeatureEngineer, GPUMonitor

def test_gpu_monitoring():
    """Test GPU monitoring with feature engineering"""
    print("🚀 Testing GPU monitoring during feature engineering...")
    
    # Create test data
    dates = pd.date_range('2023-01-01', '2023-06-01', freq='D')
    n_tickers = 100  # Use fewer tickers for faster testing
    
    data_list = []
    for i, ticker in enumerate([f'TEST{i:03d}' for i in range(n_tickers)]):
        ticker_data = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'open': np.random.randn(len(dates)).cumsum() + 100 + i,
            'high': np.random.randn(len(dates)).cumsum() + 102 + i,
            'low': np.random.randn(len(dates)).cumsum() + 98 + i,
            'close': np.random.randn(len(dates)).cumsum() + 100 + i,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        data_list.append(ticker_data)
    
    test_data = pd.concat(data_list, ignore_index=True)
    print(f"📊 Created test data: {len(test_data)} rows, {test_data['ticker'].nunique()} tickers")
    
    # Initialize GPU monitoring
    gpu_monitor = None
    try:
        gpu_monitor = GPUMonitor()
        gpu_monitor.start_monitoring(interval=0.5)
        print("🚀 GPU monitoring started")
    except Exception as e:
        print(f"⚠️ Could not start GPU monitoring: {e}")
    
    # Initialize AdvancedFeatureEngineer with GPU
    print("🚀 Initializing AdvancedFeatureEngineer with GPU...")
    fe = AdvancedFeatureEngineer(use_gpu=True)
    
    # Run feature engineering
    print("📈 Computing features with GPU acceleration...")
    start_time = time.time()
    
    try:
        # Test different feature engineering methods
        print("📈 Computing trend strength features...")
        result1 = fe.compute_trend_strength_features(test_data)
        print(f"✅ Trend strength features completed: {len(result1)} rows")
        
        print("📈 Computing volatility features...")
        result2 = fe.compute_volatility_features(test_data)
        print(f"✅ Volatility features completed: {len(result2)} rows")
        
        print("📈 Computing momentum features...")
        result3 = fe.compute_momentum_features(test_data)
        print(f"✅ Momentum features completed: {len(result3)} rows")
        
        print("📈 Computing all features...")
        result_all = fe.compute_all_features(test_data)
        print(f"✅ All features completed: {len(result_all)} rows")
        
    except Exception as e:
        print(f"❌ Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = time.time()
    print(f"⏱️ Total processing time: {end_time - start_time:.2f} seconds")
    
    # Stop GPU monitoring
    if gpu_monitor:
        try:
            gpu_monitor.stop_monitoring()
            print("🛑 GPU monitoring stopped")
        except Exception as e:
            print(f"⚠️ Error stopping GPU monitoring: {e}")
    
    print("✅ GPU monitoring test completed!")

if __name__ == "__main__":
    test_gpu_monitoring()
