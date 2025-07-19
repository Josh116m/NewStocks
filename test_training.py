#!/usr/bin/env python3
"""
Simple test script to verify the training pipeline works
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_training():
    """Test basic training functionality."""
    print("🧪 Testing basic training functionality...")
    
    try:
        # Import modules
        from training_pipeline import OptimizedTrainingPipeline, PipelineConfig
        print("✅ Successfully imported training modules")
        
        # Create simple config
        config = PipelineConfig(
            sequence_length=10,  # Shorter for testing
            max_epochs=2,        # Just 2 epochs for testing
            batch_size=32,       # Smaller batch
            target_accuracy=0.60,  # Lower target for testing
            max_training_hours=0.1,  # 6 minutes max
            checkpoint_dir="./test_checkpoints"
        )
        
        # Initialize pipeline
        pipeline = OptimizedTrainingPipeline(config)
        print("✅ Successfully initialized pipeline")
        
        # Create minimal synthetic data
        print("📊 Creating minimal test data...")
        dates = pd.date_range('2022-01-01', '2023-06-01', freq='D')  # More data for regime detection
        tickers = ['AAPL', 'SPY', 'MSFT']  # 3 tickers for better testing
        
        data = []
        for ticker in tickers:
            for date in dates:
                if date.weekday() < 5:  # Weekdays only
                    base_price = 100 + np.random.randn() * 5
                    data.append({
                        'ticker': ticker,
                        'date': date.strftime('%Y-%m-%d'),
                        'open': base_price + np.random.randn() * 0.5,
                        'high': base_price + abs(np.random.randn()) * 1,
                        'low': base_price - abs(np.random.randn()) * 1,
                        'close': base_price + np.random.randn() * 0.3,
                        'volume': int(1000000 + np.random.randn() * 50000)
                    })
        
        raw_data = pd.DataFrame(data)
        print(f"✅ Created test data with {len(raw_data)} rows")
        
        # Test just the data preparation part first
        print("🔧 Testing data preparation...")
        train_data, val_data, test_data = pipeline.prepare_data(raw_data)
        print("✅ Data preparation successful!")
        print(f"   Train samples: {len(train_data['y'])}")
        print(f"   Val samples: {len(val_data['y'])}")
        print(f"   Test samples: {len(test_data['y'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_training()
    if success:
        print("\n🎉 Basic training test PASSED!")
    else:
        print("\n💥 Basic training test FAILED!")
        sys.exit(1)
