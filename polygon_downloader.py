"""
Polygon.io Data Downloader
Downloads historical stock data from Polygon.io S3 flat files
"""

import boto3
import pandas as pd
from datetime import datetime, timedelta
import gzip
import io
import time
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Polygon.io S3 Configuration
S3_ENDPOINT = "https://files.polygon.io"
S3_BUCKET = "flatfiles"
S3_ACCESS_KEY = "89476d33-0f4d-42de-82f5-e029e1fe208d"
S3_SECRET_KEY = "NAElgPyaDwyzJiJE50jHPdEwXopQJuh9"

def setup_s3_client():
    """Setup S3 client for Polygon.io flat files"""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name='us-east-1'
    )

def find_trading_days(s3_client, target_days=504):
    """Find available trading days going back from today"""
    logger.info(f"🔍 Searching for {target_days} trading days...")
    
    current_date = datetime.now()
    available_dates = []
    days_checked = 0
    max_days_to_check = target_days * 3  # Check up to 3x to account for weekends/holidays
    
    while len(available_dates) < target_days and days_checked < max_days_to_check:
        # Skip weekends
        if current_date.weekday() < 5:
            year, month, day = current_date.year, current_date.month, current_date.day
            file_key = f"us_stocks_sip/day_aggs_v1/{year:04d}/{month:02d}/{year:04d}-{month:02d}-{day:02d}.csv.gz"
            
            try:
                # Check if file exists
                s3_client.head_object(Bucket=S3_BUCKET, Key=file_key)
                available_dates.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'file_key': file_key
                })
                if len(available_dates) % 50 == 0:  # Progress every 50 days
                    logger.info(f"📅 Found {len(available_dates)} trading days...")
            except:
                pass  # File doesn't exist, skip silently
        
        current_date -= timedelta(days=1)
        days_checked += 1
    
    logger.info(f"✅ Found {len(available_dates)} trading days")
    return available_dates

def download_day_data(s3_client, file_key, target_symbols=None):
    """Download and process data for a single day"""
    try:
        # Download the file
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        
        # Decompress and read CSV
        with gzip.GzipFile(fileobj=io.BytesIO(response['Body'].read())) as gz:
            df = pd.read_csv(gz)
        
        # Filter for target symbols if specified
        if target_symbols:
            df = df[df['ticker'].isin(target_symbols)]
        
        # Basic data cleaning
        df = df.dropna()
        df = df[df['volume'] > 0]  # Remove zero volume days
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading {file_key}: {e}")
        return None

def download_2year_data(symbols=None, output_dir="./data"):
    """
    Download 2 years of stock data
    
    Args:
        symbols: List of stock symbols to download. If None, downloads top liquid stocks
        output_dir: Directory to save the data
    """
    # Default symbols - top liquid stocks
    if symbols is None:
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'TLT', 'VIX'
        ]
    
    logger.info(f"📊 Starting download for {len(symbols)} symbols")
    logger.info(f"Symbols: {', '.join(symbols)}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Setup S3 client
    s3_client = setup_s3_client()
    
    # Find available trading days (2 years ≈ 504 trading days)
    available_dates = find_trading_days(s3_client, target_days=504)
    
    if not available_dates:
        logger.error("❌ No trading days found")
        return
    
    # Download data
    all_data = []
    successful_downloads = 0
    
    for i, date_info in enumerate(available_dates):
        date_str = date_info['date']
        file_key = date_info['file_key']
        
        logger.info(f"📥 Downloading {date_str} ({i+1}/{len(available_dates)})")
        
        day_data = download_day_data(s3_client, file_key, symbols)
        
        if day_data is not None and not day_data.empty:
            day_data['date'] = date_str
            all_data.append(day_data)
            successful_downloads += 1
            
            # Save progress every 50 days
            if successful_downloads % 50 == 0:
                logger.info(f"💾 Progress: {successful_downloads} days downloaded")
        
        # Small delay to be respectful
        time.sleep(0.1)
    
    if all_data:
        # Combine all data
        logger.info("🔄 Combining all data...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save to file
        output_file = Path(output_dir) / "stock_data_2year.csv"
        combined_df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Download complete!")
        logger.info(f"📁 Saved to: {output_file}")
        logger.info(f"📊 Total records: {len(combined_df):,}")
        logger.info(f"📅 Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        logger.info(f"🏢 Symbols: {sorted(combined_df['ticker'].unique())}")
        
        return str(output_file)
    else:
        logger.error("❌ No data downloaded")
        return None

if __name__ == "__main__":
    # Download data when run directly
    download_2year_data()
