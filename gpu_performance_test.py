"""
Test GPU vs CPU performance for stock data processing operations
"""
import torch
import cupy as cp
import time
import numpy as np
import pandas as pd

def test_gpu_vs_cpu_performance():
    print('🔍 Testing GPU vs CPU performance for stock data operations...')
    
    # Test data size similar to your stock data
    data_size = 50000  # Typical size for 2 years of daily data across multiple tickers
    data = np.random.randn(data_size).astype(np.float32)
    
    print(f'📊 Testing with {data_size:,} data points')
    
    # CPU test - rolling operations
    print('\n💻 CPU Performance Test:')
    start_time = time.time()
    for i in range(5):
        # Simulate typical feature engineering operations
        rolling_mean = pd.Series(data).rolling(20).mean().values
        rolling_std = pd.Series(data).rolling(20).std().values
        rolling_max = pd.Series(data).rolling(20).max().values
        rolling_min = pd.Series(data).rolling(20).min().values
    cpu_time = time.time() - start_time
    print(f'   Time: {cpu_time:.3f}s')
    
    # GPU test with transfers
    if torch.cuda.is_available():
        print('\n🚀 GPU Performance Test (with data transfers):')
        start_time = time.time()
        for i in range(5):
            # Transfer to GPU, process, transfer back
            gpu_data = cp.asarray(data)
            
            # Manual rolling operations (simplified)
            result1 = cp.zeros_like(gpu_data)
            result2 = cp.zeros_like(gpu_data)
            result3 = cp.zeros_like(gpu_data)
            result4 = cp.zeros_like(gpu_data)
            
            # Simple operations to test GPU utilization
            for j in range(20, len(gpu_data)):
                window = gpu_data[j-20:j]
                result1[j] = cp.mean(window)
                result2[j] = cp.std(window)
                result3[j] = cp.max(window)
                result4[j] = cp.min(window)
            
            # Transfer back to CPU
            _ = cp.asnumpy(result1)
            _ = cp.asnumpy(result2)
            _ = cp.asnumpy(result3)
            _ = cp.asnumpy(result4)
            
        gpu_time_with_transfers = time.time() - start_time
        print(f'   Time: {gpu_time_with_transfers:.3f}s')
        print(f'   Speedup vs CPU: {cpu_time/gpu_time_with_transfers:.2f}x')
        
        # GPU test without constant transfers
        print('\n🚀 GPU Performance Test (minimal transfers):')
        gpu_data = cp.asarray(data)
        start_time = time.time()
        
        for i in range(5):
            # Process on GPU without transfers
            result1 = cp.zeros_like(gpu_data)
            result2 = cp.zeros_like(gpu_data)
            result3 = cp.zeros_like(gpu_data)
            result4 = cp.zeros_like(gpu_data)
            
            for j in range(20, len(gpu_data)):
                window = gpu_data[j-20:j]
                result1[j] = cp.mean(window)
                result2[j] = cp.std(window)
                result3[j] = cp.max(window)
                result4[j] = cp.min(window)
        
        # Only transfer final results
        final_results = [cp.asnumpy(r) for r in [result1, result2, result3, result4]]
        gpu_time_minimal_transfers = time.time() - start_time
        
        print(f'   Time: {gpu_time_minimal_transfers:.3f}s')
        print(f'   Speedup vs CPU: {cpu_time/gpu_time_minimal_transfers:.2f}x')
        
        # Analysis
        print('\n📊 Performance Analysis:')
        print(f'   CPU is faster than GPU with transfers: {gpu_time_with_transfers > cpu_time}')
        print(f'   CPU is faster than GPU minimal transfers: {gpu_time_minimal_transfers > cpu_time}')
        
        if gpu_time_with_transfers > cpu_time:
            print('   ⚠️  GPU overhead from data transfers makes CPU faster')
            print('   💡 Recommendation: Use CPU for small datasets or frequent transfers')
        
        if gpu_time_minimal_transfers < cpu_time:
            print('   ✅ GPU is faster when minimizing transfers')
            print('   💡 Recommendation: Batch GPU operations and minimize transfers')
        
        # Memory transfer overhead test
        print('\n🔄 Memory Transfer Overhead Test:')
        transfer_start = time.time()
        for i in range(10):
            gpu_data = cp.asarray(data)
            cpu_data = cp.asnumpy(gpu_data)
        transfer_time = time.time() - transfer_start
        print(f'   10 round-trip transfers: {transfer_time:.3f}s')
        print(f'   Transfer overhead per operation: {transfer_time/10:.3f}s')
        
    else:
        print('❌ GPU not available')

if __name__ == "__main__":
    test_gpu_vs_cpu_performance()
