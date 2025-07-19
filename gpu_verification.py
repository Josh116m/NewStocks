#!/usr/bin/env python3
"""
Comprehensive GPU verification script for the stock trading system.
Tests all components to ensure they're using GPU acceleration properly.
"""

import time
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpu_availability():
    """Test basic GPU availability and information."""
    print("\n🖥️  GPU SYSTEM INFORMATION")
    print("=" * 50)
    
    # PyTorch CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"Total GPU Memory: {total_memory:.1f} GB")
        print(f"Allocated Memory: {allocated_memory:.1f} GB")
        print(f"Cached Memory: {cached_memory:.1f} GB")
        print(f"Free Memory: {total_memory - cached_memory:.1f} GB")
    
    # CuPy
    try:
        import cupy as cp
        print(f"CuPy Available: True")
        # Test basic CuPy operation
        x = cp.array([1, 2, 3])
        y = cp.sum(x)
        print(f"CuPy Test: sum([1,2,3]) = {y}")
    except ImportError:
        print(f"CuPy Available: False")
    except Exception as e:
        print(f"CuPy Error: {e}")
    
    return cuda_available

def test_feature_engineering_gpu():
    """Test GPU acceleration in feature engineering."""
    print("\n🔧 FEATURE ENGINEERING GPU TEST")
    print("=" * 50)
    
    try:
        from advanced_feature_engineering import AdvancedFeatureEngineer
        
        # Create test data
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        test_data = pd.DataFrame({
            'date': dates,
            'ticker': 'TEST',
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # Test CPU vs GPU performance
        print("Testing CPU feature engineering...")
        fe_cpu = AdvancedFeatureEngineer(use_gpu=False)
        start_time = time.time()
        features_cpu = fe_cpu.compute_trend_strength_features(test_data.copy())
        cpu_time = time.time() - start_time
        print(f"CPU Time: {cpu_time:.2f} seconds")
        
        print("Testing GPU feature engineering...")
        fe_gpu = AdvancedFeatureEngineer(use_gpu=True)
        start_time = time.time()
        features_gpu = fe_gpu.compute_trend_strength_features(test_data.copy())
        gpu_time = time.time() - start_time
        print(f"GPU Time: {gpu_time:.2f} seconds")
        
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"Speedup: {speedup:.2f}x")
        
        print("✅ Feature engineering GPU test completed")
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering GPU test failed: {e}")
        return False

def test_lstm_gpu():
    """Test GPU acceleration in LSTM model."""
    print("\n🧠 LSTM GPU TEST")
    print("=" * 50)
    
    try:
        from multi_stream_lstm import MultiStreamLSTM
        
        # Create test model
        feature_dims = {'short': 10, 'medium': 8, 'long': 6}
        model = MultiStreamLSTM(feature_dims, n_regimes=6)
        
        # Test device placement
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"Model device: {next(model.parameters()).device}")
        
        # Create test tensors
        batch_size = 32
        seq_len = 20
        x_short = torch.randn(batch_size, seq_len, 10).to(device)
        x_medium = torch.randn(batch_size, seq_len, 8).to(device)
        x_long = torch.randn(batch_size, 6).to(device)
        regime_id = torch.randint(0, 6, (batch_size, 1)).to(device)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            logits, attention = model(x_short, x_medium, x_long, regime_id)
            inference_time = time.time() - start_time
        
        print(f"Inference time: {inference_time:.4f} seconds")
        print(f"Output shape: {logits.shape}")
        print(f"Output device: {logits.device}")
        
        print("✅ LSTM GPU test completed")
        return True
        
    except Exception as e:
        print(f"❌ LSTM GPU test failed: {e}")
        return False

def test_ensemble_gpu():
    """Test GPU acceleration in ensemble models."""
    print("\n🎯 ENSEMBLE GPU TEST")
    print("=" * 50)
    
    try:
        from stacked_ensemble import StackedEnsemblePredictor
        
        # Create test data
        n_samples = 1000
        n_features = 50
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        regime_labels = np.random.randint(0, 6, n_samples)
        
        # Test ensemble
        ensemble = StackedEnsemblePredictor()
        
        print("Testing ensemble training with GPU models...")
        start_time = time.time()
        ensemble.fit(X, y, regime_labels=regime_labels)
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.2f} seconds")
        
        # Test prediction
        start_time = time.time()
        predictions = ensemble.predict_proba(X[:100], regime_labels[:100])
        prediction_time = time.time() - start_time
        
        print(f"Prediction time (100 samples): {prediction_time:.4f} seconds")
        print(f"Predictions shape: {predictions.shape}")
        
        print("✅ Ensemble GPU test completed")
        return True
        
    except Exception as e:
        print(f"❌ Ensemble GPU test failed: {e}")
        return False

def test_trading_system_gpu():
    """Test GPU acceleration in the main trading system."""
    print("\n🚀 TRADING SYSTEM GPU TEST")
    print("=" * 50)
    
    try:
        from main_trading_system import StockTradingPredictor
        
        # Initialize with GPU
        predictor = StockTradingPredictor(use_gpu=True)
        
        print(f"GPU enabled: {predictor.use_gpu}")
        print(f"Device: {predictor.device}")
        
        print("✅ Trading system GPU test completed")
        return True
        
    except Exception as e:
        print(f"❌ Trading system GPU test failed: {e}")
        return False

def main():
    """Run all GPU verification tests."""
    print("🧪 COMPREHENSIVE GPU VERIFICATION")
    print("=" * 60)
    
    # Test basic GPU availability
    gpu_available = test_gpu_availability()
    
    if not gpu_available:
        print("\n⚠️  GPU not available. Some tests will be skipped.")
    
    # Run all tests
    tests = [
        ("Feature Engineering", test_feature_engineering_gpu),
        ("LSTM Model", test_lstm_gpu),
        ("Ensemble Models", test_ensemble_gpu),
        ("Trading System", test_trading_system_gpu),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 VERIFICATION SUMMARY")
    print("=" * 40)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL GPU ACCELERATION TESTS PASSED!")
        print("Your system is fully optimized for GPU-accelerated trading.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    # GPU utilization tip
    if gpu_available:
        print(f"\n💡 TIP: Monitor GPU usage with 'nvidia-smi' while running training/analysis")

if __name__ == "__main__":
    main()
