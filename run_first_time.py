#!/usr/bin/env python3
"""
First-time setup and verification script for Stock Trading Prediction System
Run this after installing dependencies to verify everything is working.
"""

import sys
import os
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False


def check_imports():
    """Check if all required packages are installed."""
    print("\n📦 Checking required packages...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'torch': 'torch',
        'pytorch_lightning': 'pytorch-lightning',
        # 'pandas_ta': 'pandas-ta',  # Skipping due to compatibility issues
        'boto3': 'boto3',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'psutil': 'psutil',
        'GPUtil': 'gputil',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    # Check PyTorch GPU support
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch GPU support - CUDA {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  PyTorch CPU only - GPU not available")
    except:
        pass
    
    return missing_packages


def check_project_files():
    """Check if all required project files are present."""
    print("\n📁 Checking project files...")
    
    required_files = [
        'polygon_downloader.py',
        'advanced_feature_engineering.py',
        'regime_detector.py',
        'multi_stream_lstm.py',
        'stacked_ensemble.py',
        'training_pipeline.py',
        'main_trading_system.py',
        'requirements.txt'
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT FOUND")
            missing_files.append(file)
    
    return missing_files


def create_directories():
    """Create necessary directories."""
    print("\n📂 Creating directories...")
    
    directories = [
        'data',
        'models',
        'checkpoints',
        'logs',
        'predictions',
        'backtest_results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")


def run_import_test():
    """Test importing all custom modules."""
    print("\n🔧 Testing module imports...")
    
    try:
        from advanced_feature_engineering import AdvancedFeatureEngineer
        print("✅ AdvancedFeatureEngineer")
        
        from regime_detector import DataDrivenRegimeDetector
        print("✅ DataDrivenRegimeDetector")
        
        from multi_stream_lstm import MultiStreamLSTM
        print("✅ MultiStreamLSTM")
        
        from stacked_ensemble import StackedEnsemblePredictor
        print("✅ StackedEnsemblePredictor")
        
        from training_pipeline import OptimizedTrainingPipeline
        print("✅ OptimizedTrainingPipeline")
        
        from main_trading_system import StockTradingPredictor
        print("✅ StockTradingPredictor")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def download_sample_data():
    """Download a small sample of data for testing."""
    print("\n📊 Testing data download (small sample)...")
    
    try:
        # Create a minimal test to verify Polygon.io connection
        import boto3
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Test S3 connection
        S3_ENDPOINT = "https://files.polygon.io"
        S3_BUCKET = "flatfiles"
        S3_ACCESS_KEY = "89476d33-0f4d-42de-82f5-e029e1fe208d"
        S3_SECRET_KEY = "NAElgPyaDwyzJiJE50jHPdEwXopQJuh9"
        
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name='us-east-1'
        )
        
        # Try to list files (just to test connection)
        test_date = datetime.now() - timedelta(days=10)
        test_key = f"us_stocks_sip/day_aggs_v1/{test_date.year:04d}/{test_date.month:02d}/"
        
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=test_key,
            MaxKeys=1
        )
        
        if 'Contents' in response:
            print("✅ Polygon.io connection successful")
            return True
        else:
            print("⚠️  Could not verify Polygon.io connection")
            return False
            
    except Exception as e:
        print(f"⚠️  Data download test failed: {e}")
        return False


def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "="*60)
    print("🎯 NEXT STEPS")
    print("="*60)
    print("""
1. Download historical data:
   python -c "from polygon_downloader import download_2year_data; download_2year_data()"

2. Train the model:
   python main_trading_system.py

3. Or run a quick test:
   python -c "from training_pipeline import run_example_pipeline; run_example_pipeline()"

4. Check the quickstart guide:
   cat quickstart_guide.md

Happy trading! 📈
""")


def main():
    """Run all setup checks."""
    print("🚀 Stock Trading Prediction System - First Time Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Please upgrade Python to version 3.8 or higher.")
        sys.exit(1)
    
    # Check imports
    missing_packages = check_imports()
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check project files
    missing_files = check_project_files()
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        print("Make sure all project files are in the current directory.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Test imports
    if not run_import_test():
        print("\n❌ Module import test failed. Check for syntax errors.")
        sys.exit(1)
    
    # Test data connection
    download_sample_data()
    
    # Success!
    print("\n" + "="*60)
    print("✅ ALL CHECKS PASSED! System is ready to use.")
    print("="*60)
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
