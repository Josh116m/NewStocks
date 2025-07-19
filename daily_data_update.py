"""
Daily Data Update Script
Run this script daily to ensure you have the latest stock data
"""

import logging
from pathlib import Path
from datetime import datetime
from update_latest_data import ensure_fresh_data, check_data_freshness

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to update daily data"""
    print("🚀 DAILY DATA UPDATE")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Find the data file
    data_file = "stock_data_2years_20250716_190514.csv"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        print("💡 Please run simple_2year_data_downloader.py first to download initial data")
        return False
    
    print(f"📂 Found data file: {data_file}")
    
    # Check current data status
    current_date, latest_available = check_data_freshness(data_file)
    
    if current_date and latest_available:
        print(f"📊 Current data through: {current_date}")
        print(f"📅 Latest available: {latest_available}")
        
        if current_date >= latest_available:
            print("✅ Your data is already up to date!")
            print("🎯 Ready to run predictions!")
            return True
        else:
            print(f"🔄 Data needs updating...")
            print("📥 Downloading latest data...")
            
            # Update the data
            success = ensure_fresh_data(data_file, max_age_days=0)
            
            if success:
                print("✅ Data update completed successfully!")
                print("🎯 Ready to run predictions!")
                print()
                print("💡 You can now run:")
                print("   cd .. && python weekly_analysis.py")
                return True
            else:
                print("❌ Data update failed")
                print("💡 You can still run predictions with existing data")
                return False
    else:
        print("❌ Could not determine data status")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*50)
        print("🎉 DAILY UPDATE COMPLETE!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("⚠️ UPDATE HAD ISSUES")
        print("="*50)
        print("💡 Check the messages above for details")
