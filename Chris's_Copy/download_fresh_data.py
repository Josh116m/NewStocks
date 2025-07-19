#!/usr/bin/env python3
"""
Fresh Data Downloader for Weekly Stock Analysis System
Downloads the latest 2 years of stock data from Polygon.io

This script includes API credentials and can download fresh data automatically.
Run this periodically to keep your data current.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import boto3
        import pandas as pd
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("🔧 Install with: pip install boto3 pandas")
        return False

def download_fresh_data():
    """Download fresh 2-year stock data."""
    print("🚀 FRESH DATA DOWNLOADER")
    print("=" * 50)
    print("📡 Downloading latest 2 years of stock data from Polygon.io")
    print("🔑 Using included API credentials")
    print()
    
    if not check_dependencies():
        return False
    
    try:
        # Import the downloader
        from simple_2year_data_downloader import download_2year_data
        
        # Create data directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)
        
        # Backup existing data if it exists
        existing_data = "data/stock_data_2year.csv"
        if Path(existing_data).exists():
            backup_name = f"data/stock_data_2year_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.rename(existing_data, backup_name)
            print(f"💾 Backed up existing data to: {backup_name}")
        
        print("📥 Starting download...")
        print("⏱️  This may take 10-20 minutes depending on your connection")
        print()
        
        # Download the data
        data = download_2year_data()
        
        if data is not None and not data.empty:
            # Save to the expected location
            output_file = "data/stock_data_2year.csv"
            data.to_csv(output_file, index=False)
            
            print(f"\n🎉 DOWNLOAD COMPLETE!")
            print("=" * 50)
            print(f"✅ Fresh data saved to: {output_file}")
            print(f"📊 Total records: {len(data):,}")
            print(f"📈 Unique stocks: {data['ticker'].nunique():,}")
            print(f"📅 Date range: {data['date'].min()} to {data['date'].max()}")
            print()
            print("🚀 You can now run the weekly analysis with fresh data!")
            print("   python weekly_analysis.py")
            
            return True
        else:
            print("❌ Download failed - no data received")
            return False
            
    except Exception as e:
        print(f"❌ Download error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Ensure you have enough disk space (2-3 GB)")
        print("   3. Try running again (sometimes network issues are temporary)")
        return False

def check_current_data():
    """Check what data we currently have."""
    data_file = "data/stock_data_2year.csv"
    sample_file = "data/sample_stock_data.csv"
    
    print("📊 CURRENT DATA STATUS")
    print("=" * 30)
    
    if Path(data_file).exists():
        try:
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            latest_date = df['date'].max()
            days_old = (datetime.now() - latest_date).days
            
            print(f"✅ Full dataset found: {data_file}")
            print(f"   Records: {len(df):,}")
            print(f"   Stocks: {df['ticker'].nunique():,}")
            print(f"   Latest date: {latest_date.date()}")
            print(f"   Age: {days_old} days old")
            
            if days_old > 7:
                print(f"⚠️  Data is {days_old} days old - consider updating")
                return False
            else:
                print("✅ Data is current")
                return True
                
        except Exception as e:
            print(f"❌ Error reading {data_file}: {e}")
            return False
    elif Path(sample_file).exists():
        print(f"📝 Sample data found: {sample_file}")
        print("💡 This is limited sample data (10 stocks)")
        print("🚀 Download fresh data for full analysis")
        return False
    else:
        print("❌ No data found")
        print("🚀 Download fresh data to get started")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("Fresh Data Downloader")
            print("\nUsage:")
            print("  python download_fresh_data.py        # Download fresh data")
            print("  python download_fresh_data.py check  # Check current data status")
            print("  python download_fresh_data.py -h     # Show this help")
            print("\nFeatures:")
            print("  • Downloads latest 2 years of stock data")
            print("  • Includes 10,000+ stocks")
            print("  • Updates automatically to current date")
            print("  • Backs up existing data before updating")
            print("  • Uses included API credentials")
            return
        elif sys.argv[1] == 'check':
            check_current_data()
            return
    
    # Check current data first
    if check_current_data():
        response = input("\n🤔 Data appears current. Download fresh data anyway? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("✅ Keeping existing data")
            return
    
    # Download fresh data
    success = download_fresh_data()
    
    if not success:
        print("\n💡 Tips:")
        print("   • Make sure you have a stable internet connection")
        print("   • Ensure you have 2-3 GB of free disk space")
        print("   • The download includes 500+ days of data for 10,000+ stocks")
        print("   • If download fails, you can still use the sample data")

if __name__ == "__main__":
    main()
