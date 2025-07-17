#!/usr/bin/env python3
"""
Simple runner script for Weekly Stock Analysis
Provides easy execution with error handling and guidance
"""

import sys
import os
from pathlib import Path

def check_setup():
    """Quick check if system is ready."""
    required_files = [
        "weekly_analysis.py",
        "models/final_models_20250713_185556/ensemble.pkl",
        "data/sample_stock_data.csv"
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        print("❌ System not properly set up!")
        print("Missing files:", missing)
        print("\n🔧 Please run setup first:")
        print("   python setup.py")
        return False
    
    return True

def run_analysis():
    """Run the weekly analysis with error handling."""
    print("🚀 STARTING WEEKLY STOCK ANALYSIS")
    print("=" * 50)
    
    if not check_setup():
        return False
    
    try:
        # Import and run the analysis
        from weekly_analysis import run_weekly_analysis
        
        print("📊 Running analysis...")
        results = run_weekly_analysis()
        
        if results is not None:
            print("\n" + "="*60)
            print("🎉 ANALYSIS COMPLETE!")
            print("="*60)
            print(f"✅ Analyzed {len(results)} stocks")
            print(f"📊 Results saved to predictions/ directory")
            print(f"🚀 Check the output above for recommendations!")
            return True
        else:
            print("\n❌ Analysis failed. Check the error messages above.")
            return False
            
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\n🔧 Try installing dependencies:")
        print("   pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\n🔧 If this persists, check:")
        print("   1. All files are present")
        print("   2. Dependencies are installed")
        print("   3. Data file is valid")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("Weekly Stock Analysis Runner")
        print("\nUsage:")
        print("  python run_analysis.py     # Run analysis")
        print("  python run_analysis.py -h  # Show this help")
        print("\nFirst time setup:")
        print("  python setup.py           # Check and install dependencies")
        return
    
    success = run_analysis()
    
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("   1. Run 'python setup.py' to check installation")
        print("   2. Ensure you have at least 4GB RAM available")
        print("   3. Check that data files are present")
        print("   4. See README.md for detailed instructions")

if __name__ == "__main__":
    main()
