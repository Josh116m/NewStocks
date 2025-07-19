#!/usr/bin/env python3
"""
Setup script for Weekly Stock Analysis System
Checks dependencies and system requirements
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required. Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True

def check_required_packages():
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
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            importlib.import_module(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    return missing_packages

def install_packages(packages):
    """Install missing packages."""
    if not packages:
        return True
    
    print(f"\n🔧 Installing missing packages: {', '.join(packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def check_system_files():
    """Check if all required system files are present."""
    print("\n📁 Checking system files...")
    
    required_files = [
        "weekly_analysis.py",
        "main_trading_system.py",
        "advanced_feature_engineering.py",
        "regime_detector.py",
        "stacked_ensemble.py",
        "multi_stream_lstm.py",
        "simple_ta.py",
        "download_fresh_data.py",
        "simple_2year_data_downloader.py",
        "requirements.txt",
        "models/final_models_20250713_185556/ensemble.pkl",
        "models/final_models_20250713_185556/regime_detector.pkl",
        "models/final_models_20250713_185556/lstm_model.pth",
        "data/sample_stock_data.csv"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    return missing_files

def test_import():
    """Test importing the main modules."""
    print("\n🧪 Testing module imports...")
    
    try:
        from main_trading_system import StockTradingPredictor
        print("✅ StockTradingPredictor")
        
        from advanced_feature_engineering import AdvancedFeatureEngineer
        print("✅ AdvancedFeatureEngineer")
        
        from regime_detector import DataDrivenRegimeDetector
        print("✅ DataDrivenRegimeDetector")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 WEEKLY STOCK ANALYSIS SYSTEM - SETUP")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return False
    
    # Check system files
    missing_files = check_system_files()
    if missing_files:
        print(f"\n❌ Setup failed: Missing files: {missing_files}")
        return False
    
    # Check packages
    missing_packages = check_required_packages()
    
    # Install missing packages if any
    if missing_packages:
        install_success = install_packages(missing_packages)
        if not install_success:
            print("\n❌ Setup failed: Could not install required packages")
            return False
    
    # Test imports
    if not test_import():
        print("\n❌ Setup failed: Module import errors")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("✅ All dependencies installed")
    print("✅ All files present")
    print("✅ System ready to use")
    print("\n🚀 Next steps:")
    print("   1. Download fresh data: python download_fresh_data.py")
    print("   2. Run analysis: python weekly_analysis.py")
    print("\n💡 Or use sample data directly:")
    print("   python weekly_analysis.py")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
