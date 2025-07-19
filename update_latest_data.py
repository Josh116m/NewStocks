"""
Update Latest Data
Downloads the most recent trading data and updates existing dataset
"""

import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import os
from simple_2year_data_downloader import setup_s3_client, find_trading_days, download_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_latest_available_date():
    """Get the latest available trading date from S3"""
    try:
        s3_client = setup_s3_client()
        # Check last 10 days to find the most recent available
        available_dates = find_trading_days(s3_client, target_days=10)

        if available_dates:
            # Sort dates to get the most recent (find_trading_days returns oldest first)
            dates_sorted = sorted([d['date'] for d in available_dates], reverse=True)
            latest_date = dates_sorted[0]  # Most recent
            logger.info(f"📅 Latest available trading date: {latest_date}")
            return latest_date
        else:
            logger.warning("⚠️ No trading dates found")
            return None

    except Exception as e:
        logger.error(f"❌ Error checking latest date: {e}")
        return None

def check_data_freshness(data_file):
    """Check how fresh our current data is"""
    try:
        if not Path(data_file).exists():
            logger.info(f"📂 Data file {data_file} doesn't exist")
            return None, None
        
        # Load existing data
        if data_file.endswith('.parquet'):
            df = pd.read_parquet(data_file)
        else:
            df = pd.read_csv(data_file)
        
        df['date'] = pd.to_datetime(df['date'])
        latest_data_date = df['date'].max().strftime('%Y-%m-%d')
        
        # Get latest available date
        latest_available_date = get_latest_available_date()
        
        logger.info(f"📊 Current data through: {latest_data_date}")
        logger.info(f"📅 Latest available: {latest_available_date}")
        
        return latest_data_date, latest_available_date
        
    except Exception as e:
        logger.error(f"❌ Error checking data freshness: {e}")
        return None, None

def download_missing_days(start_date, end_date, existing_tickers=None):
    """Download data for missing days"""
    try:
        s3_client = setup_s3_client()
        
        # Find available dates in the range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get all available dates and filter to our range
        all_dates = find_trading_days(s3_client, target_days=50)  # Get more than we need
        
        target_dates = []
        for date_info in all_dates:
            date_dt = datetime.strptime(date_info['date'], '%Y-%m-%d')
            if start_dt <= date_dt <= end_dt:
                target_dates.append(date_info)
        
        if not target_dates:
            logger.info("📅 No new trading days to download")
            return pd.DataFrame()
        
        logger.info(f"📥 Downloading {len(target_dates)} missing days...")
        
        all_data = []
        for i, date_info in enumerate(target_dates):
            date_str = date_info['date']
            file_key = date_info['file_key']
            
            logger.info(f"📊 Downloading {date_str} ({i+1}/{len(target_dates)})")
            
            day_data = download_file(s3_client, file_key)
            if day_data is not None and not day_data.empty:
                # Filter to existing tickers if specified
                if existing_tickers is not None:
                    day_data = day_data[day_data['ticker'].isin(existing_tickers)]
                
                all_data.append(day_data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"✅ Downloaded {len(combined_df):,} new records")
            return combined_df
        else:
            logger.warning("⚠️ No new data downloaded")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ Error downloading missing days: {e}")
        return pd.DataFrame()

def update_data_file(data_file, backup=True):
    """Update existing data file with latest available data"""
    logger.info(f"🔄 Updating data file: {data_file}")
    
    # Check current data freshness
    current_date, latest_available = check_data_freshness(data_file)
    
    if current_date is None or latest_available is None:
        logger.error("❌ Could not determine data freshness")
        return False
    
    if current_date >= latest_available:
        logger.info("✅ Data is already up to date!")
        return True
    
    # Calculate missing days
    current_dt = datetime.strptime(current_date, '%Y-%m-%d')
    latest_dt = datetime.strptime(latest_available, '%Y-%m-%d')
    missing_days = (latest_dt - current_dt).days
    
    logger.info(f"📊 Need to update {missing_days} days of data")
    
    try:
        # Load existing data
        if data_file.endswith('.parquet'):
            existing_df = pd.read_parquet(data_file)
        else:
            existing_df = pd.read_csv(data_file)
        
        # Get list of existing tickers
        existing_tickers = existing_df['ticker'].unique().tolist()
        logger.info(f"🎯 Updating {len(existing_tickers)} tickers")
        
        # Backup existing file if requested
        if backup:
            backup_file = data_file.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            backup_file = backup_file.replace('.parquet', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet')
            
            if data_file.endswith('.parquet'):
                existing_df.to_parquet(backup_file, index=False)
            else:
                existing_df.to_csv(backup_file, index=False)
            logger.info(f"💾 Backed up existing data to: {backup_file}")
        
        # Download missing data
        next_day = (current_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        new_data = download_missing_days(next_day, latest_available, existing_tickers)
        
        if not new_data.empty:
            # Combine with existing data
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            new_data['date'] = pd.to_datetime(new_data['date'])
            
            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
            combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)
            
            # Save updated data
            if data_file.endswith('.parquet'):
                combined_df.to_parquet(data_file, index=False)
            else:
                combined_df.to_csv(data_file, index=False)
            
            logger.info(f"✅ Updated data saved to: {data_file}")
            logger.info(f"📊 Added {len(new_data):,} new records")
            logger.info(f"📅 New date range: {combined_df['date'].min().strftime('%Y-%m-%d')} to {combined_df['date'].max().strftime('%Y-%m-%d')}")
            
            return True
        else:
            logger.warning("⚠️ No new data to add")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error updating data file: {e}")
        return False

def ensure_fresh_data(data_file, max_age_days=1):
    """Ensure data file is fresh, update if needed"""
    logger.info(f"🔍 Checking data freshness (max age: {max_age_days} days)")
    
    current_date, latest_available = check_data_freshness(data_file)
    
    if current_date is None:
        logger.error("❌ Could not check data freshness")
        return False
    
    if latest_available is None:
        logger.error("❌ Could not determine latest available date")
        return False
    
    # Calculate age in days
    current_dt = datetime.strptime(current_date, '%Y-%m-%d')
    latest_dt = datetime.strptime(latest_available, '%Y-%m-%d')
    age_days = (latest_dt - current_dt).days
    
    if age_days <= max_age_days:
        logger.info(f"✅ Data is fresh (age: {age_days} days)")
        return True
    else:
        logger.info(f"🔄 Data needs update (age: {age_days} days)")
        return update_data_file(data_file)

if __name__ == "__main__":
    # Test the update functionality
    logger.info("🧪 Testing data update functionality...")
    
    # Check current data file
    data_file = "stock_data_2years_20250716_190514.csv"
    
    if Path(data_file).exists():
        logger.info(f"📂 Found existing data file: {data_file}")
        
        # Check freshness
        current_date, latest_available = check_data_freshness(data_file)
        
        if current_date and latest_available:
            if current_date < latest_available:
                logger.info("🔄 Data needs updating...")
                success = update_data_file(data_file)
                if success:
                    logger.info("✅ Data update completed successfully!")
                else:
                    logger.error("❌ Data update failed")
            else:
                logger.info("✅ Data is already up to date!")
        else:
            logger.error("❌ Could not determine data status")
    else:
        logger.error(f"❌ Data file not found: {data_file}")
        logger.info("💡 Run simple_2year_data_downloader.py first to download initial data")
