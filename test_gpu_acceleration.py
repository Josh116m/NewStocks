#!/usr/bin/env python3
"""
Test script to verify GPU acceleration is working for all ML frameworks.
"""

import time
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import GPUtil

def test_pytorch_gpu():
    """Test PyTorch GPU acceleration."""
    print("\n🔥 Testing PyTorch GPU acceleration...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available for PyTorch")
        return False
    
    try:
        # Create test data
        device = torch.device('cuda')
        x = torch.randn(1000, 100).to(device)
        y = torch.randn(1000, 1).to(device)
        
        # Simple linear model
        model = torch.nn.Linear(100, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        
        # Time training
        start_time = time.time()
        for _ in range(100):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        
        gpu_time = time.time() - start_time
        print(f"✅ PyTorch GPU training completed in {gpu_time:.2f}s")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch GPU test failed: {e}")
        return False

def test_xgboost_gpu():
    """Test XGBoost GPU acceleration."""
    print("\n🚀 Testing XGBoost GPU acceleration...")
    
    try:
        # Create test data
        X, y = make_classification(n_samples=10000, n_features=50, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test GPU training
        start_time = time.time()
        model_gpu = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            tree_method='hist',
            device='cuda',
            random_state=42
        )
        model_gpu.fit(X_train, y_train)
        gpu_time = time.time() - start_time
        
        # Test CPU training for comparison
        start_time = time.time()
        model_cpu = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            tree_method='hist',
            device='cpu',
            random_state=42
        )
        model_cpu.fit(X_train, y_train)
        cpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time
        print(f"✅ XGBoost GPU training completed")
        print(f"   GPU time: {gpu_time:.2f}s")
        print(f"   CPU time: {cpu_time:.2f}s")
        print(f"   Speedup: {speedup:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ XGBoost GPU test failed: {e}")
        return False

def test_lightgbm_gpu():
    """Test LightGBM GPU acceleration."""
    print("\n⚡ Testing LightGBM GPU acceleration...")
    
    try:
        # Create test data
        X, y = make_classification(n_samples=10000, n_features=50, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test GPU training
        start_time = time.time()
        model_gpu = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            device='gpu',
            random_state=42,
            verbose=-1
        )
        model_gpu.fit(X_train, y_train)
        gpu_time = time.time() - start_time
        
        # Test CPU training for comparison
        start_time = time.time()
        model_cpu = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            device='cpu',
            random_state=42,
            verbose=-1
        )
        model_cpu.fit(X_train, y_train)
        cpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time
        print(f"✅ LightGBM GPU training completed")
        print(f"   GPU time: {gpu_time:.2f}s")
        print(f"   CPU time: {cpu_time:.2f}s")
        print(f"   Speedup: {speedup:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ LightGBM GPU test failed: {e}")
        return False

def show_gpu_info():
    """Display GPU information."""
    print("\n🖥️  GPU Information:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Device Count: {torch.cuda.device_count()}")
        print(f"   Current Device: {torch.cuda.current_device()}")
        print(f"   Device Name: {torch.cuda.get_device_name(0)}")
        
        # GPU utilization
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"   GPU Load: {gpu.load*100:.1f}%")
            print(f"   GPU Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB ({gpu.memoryUtil*100:.1f}%)")

def main():
    """Run all GPU acceleration tests."""
    print("🧪 GPU Acceleration Test Suite")
    print("=" * 50)
    
    show_gpu_info()
    
    results = []
    results.append(test_pytorch_gpu())
    results.append(test_xgboost_gpu())
    results.append(test_lightgbm_gpu())
    
    print("\n📊 Summary:")
    print("=" * 30)
    frameworks = ["PyTorch", "XGBoost", "LightGBM"]
    for framework, success in zip(frameworks, results):
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"   {framework}: {status}")
    
    if all(results):
        print("\n🎉 All GPU acceleration is working! Your RTX 3060 is ready for ML workloads.")
    else:
        print("\n⚠️  Some GPU acceleration failed. Check the errors above.")

if __name__ == "__main__":
    main()
