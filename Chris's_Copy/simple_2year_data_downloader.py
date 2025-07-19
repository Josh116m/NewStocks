import boto3
import pandas as pd
from datetime import datetime, timedelta
import gzip
import io
import time

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
    print(f"🔍 Searching for {target_days} trading days...")
    
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
                    print(f"📅 Found {len(available_dates)} trading days...")
            except:
                pass  # File doesn't exist, skip silently
        
        current_date -= timedelta(days=1)
        days_checked += 1
    
    # Sort oldest to newest
    available_dates.reverse()
    print(f"✅ Found {len(available_dates)} trading days from {available_dates[0]['date']} to {available_dates[-1]['date']}")
    return available_dates

def download_file(s3_client, file_key):
    """Download and parse a single CSV file"""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        with gzip.GzipFile(fileobj=io.BytesIO(response['Body'].read())) as gz_file:
            content = gz_file.read().decode('utf-8')
        
        df = pd.read_csv(io.StringIO(content))
        date_str = file_key.split('/')[-1].replace('.csv.gz', '')
        df['date'] = date_str
        return df
    except Exception as e:
        print(f"❌ Error downloading {file_key}: {e}")
        return None

def download_2year_data():
    """Download 2 years of stock data and save to files"""
    print("🚀 SIMPLE 2-YEAR STOCK DATA DOWNLOADER")
    print("=" * 50)
    
    # Setup S3 client
    print("🔧 Setting up connection...")
    s3_client = setup_s3_client()
    
    # Find available trading days (504 = ~2 years)
    available_dates = find_trading_days(s3_client, target_days=504)
    
    if not available_dates:
        print("❌ No data found!")
        return
    
    # Download all data
    print(f"\n📥 Downloading {len(available_dates)} files...")
    all_data = []
    
    for i, date_info in enumerate(available_dates, 1):
        # Show progress every 10%
        if i % (len(available_dates) // 10) == 0 or i == len(available_dates):
            progress = (i / len(available_dates)) * 100
            print(f"📊 Progress: {progress:.0f}% ({i}/{len(available_dates)}) - {date_info['date']}")
        
        df = download_file(s3_client, date_info['file_key'])
        if df is not None:
            all_data.append(df)
        
        time.sleep(0.01)  # Small delay to be respectful
    
    if not all_data:
        print("❌ No data downloaded successfully!")
        return
    
    # Combine all data
    print("\n🔄 Combining data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Basic data cleaning
    print("🧹 Cleaning data...")
    initial_count = len(combined_df)
    combined_df = combined_df.dropna(subset=['ticker', 'open', 'close', 'high', 'low', 'volume'])
    combined_df = combined_df[combined_df['volume'] >= 0]
    combined_df = combined_df[combined_df['open'] > 0]
    combined_df = combined_df[combined_df['close'] > 0]
    final_count = len(combined_df)
    
    print(f"✅ Cleaned: {initial_count:,} → {final_count:,} records")
    
    # Save files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    csv_filename = f"stock_data_2years_{timestamp}.csv"
    combined_df.to_csv(csv_filename, index=False)
    print(f"💾 Saved CSV: {csv_filename}")
    
    # Save as Parquet (more efficient)
    try:
        parquet_filename = f"stock_data_2years_{timestamp}.parquet"
        combined_df.to_parquet(parquet_filename, index=False)
        print(f"💾 Saved Parquet: {parquet_filename}")
    except ImportError:
        print("⚠️  Parquet not available (install pyarrow for better performance)")
    
    # Summary
    print(f"\n📊 DOWNLOAD COMPLETE!")
    print("=" * 50)
    print(f"📈 Total records: {len(combined_df):,}")
    print(f"📈 Unique stocks: {combined_df['ticker'].nunique():,}")
    print(f"📈 Trading days: {combined_df['date'].nunique()}")
    print(f"📈 Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"📈 Total volume: {combined_df['volume'].sum():,} shares")
    
    return combined_df

if __name__ == "__main__":
    data = download_2year_data()
