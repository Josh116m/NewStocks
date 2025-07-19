"""
Polygon.io REST API Downloader
Downloads current/latest stock data using direct REST API calls
This gets the most up-to-date data including today's trading data
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import List, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Polygon.io REST API Configuration
POLYGON_API_KEY = "89476d33-0f4d-42de-82f5-e029e1fe208d"
POLYGON_BASE_URL = "https://api.polygon.io"

def get_current_trading_day():
    """Get the current or most recent trading day"""
    today = datetime.now()
    
    # If it's weekend, go back to Friday
    if today.weekday() == 5:  # Saturday
        today = today - timedelta(days=1)
    elif today.weekday() == 6:  # Sunday
        today = today - timedelta(days=2)
    
    return today.strftime('%Y-%m-%d')

def get_ticker_list():
    """Get list of active tickers from Polygon API"""
    url = f"{POLYGON_BASE_URL}/v3/reference/tickers"
    params = {
        'apikey': POLYGON_API_KEY,
        'market': 'stocks',
        'active': 'true',
        'limit': 1000  # Get top 1000 most active
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data:
            tickers = [result['ticker'] for result in data['results']]
            logger.info(f"📊 Retrieved {len(tickers)} active tickers")
            return tickers
        else:
            logger.warning("No tickers found in API response")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching ticker list: {e}")
        # Return default list if API fails
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'TLT', 'VIX', 'AMD', 'INTC',
            'BABA', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'DIS', 'UBER', 'LYFT'
        ]

def get_daily_bars(ticker: str, date: str) -> Optional[dict]:
    """Get daily OHLCV data for a specific ticker and date"""
    url = f"{POLYGON_BASE_URL}/v1/open-close/{ticker}/{date}"
    params = {'apikey': POLYGON_API_KEY}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'OK':
            return {
                'ticker': ticker,
                'date': date,
                'open': data.get('open'),
                'high': data.get('high'),
                'low': data.get('low'),
                'close': data.get('close'),
                'volume': data.get('volume', 0)
            }
        else:
            return None
            
    except Exception as e:
        logger.debug(f"Error fetching data for {ticker} on {date}: {e}")
        return None

def get_aggregates_range(ticker: str, start_date: str, end_date: str) -> List[dict]:
    """Get aggregate bars for a ticker over a date range"""
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'apikey': POLYGON_API_KEY,
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if data.get('status') == 'OK' and 'results' in data:
            for bar in data['results']:
                # Convert timestamp to date
                date_str = datetime.fromtimestamp(bar['t'] / 1000).strftime('%Y-%m-%d')
                results.append({
                    'ticker': ticker,
                    'date': date_str,
                    'open': bar['o'],
                    'high': bar['h'],
                    'low': bar['l'],
                    'close': bar['c'],
                    'volume': bar['v']
                })
        
        return results
        
    except Exception as e:
        logger.debug(f"Error fetching aggregates for {ticker}: {e}")
        return []

def download_current_day_data(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Download current day data for specified tickers"""
    if tickers is None:
        tickers = get_ticker_list()[:100]  # Limit to top 100 for speed
    
    current_date = get_current_trading_day()
    logger.info(f"📅 Downloading data for {current_date}")
    logger.info(f"🎯 Target tickers: {len(tickers)} stocks")
    
    all_data = []
    successful = 0
    
    for i, ticker in enumerate(tickers):
        if i % 10 == 0:
            logger.info(f"📊 Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%)")
        
        data = get_daily_bars(ticker, current_date)
        if data:
            all_data.append(data)
            successful += 1
        
        # Rate limiting - be respectful to API
        time.sleep(0.1)
    
    logger.info(f"✅ Successfully downloaded {successful}/{len(tickers)} tickers")
    
    if all_data:
        df = pd.DataFrame(all_data)
        return df
    else:
        return pd.DataFrame()

def download_recent_data(days: int = 5, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Download recent data for the last N trading days"""
    if tickers is None:
        tickers = get_ticker_list()[:50]  # Limit for speed
    
    end_date = get_current_trading_day()
    start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days*2)).strftime('%Y-%m-%d')
    
    logger.info(f"📅 Downloading data from {start_date} to {end_date}")
    logger.info(f"🎯 Target tickers: {len(tickers)} stocks")
    
    all_data = []
    successful = 0
    
    for i, ticker in enumerate(tickers):
        if i % 10 == 0:
            logger.info(f"📊 Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%)")
        
        ticker_data = get_aggregates_range(ticker, start_date, end_date)
        if ticker_data:
            all_data.extend(ticker_data)
            successful += 1
        
        # Rate limiting
        time.sleep(0.12)  # 5 calls per second limit
    
    logger.info(f"✅ Successfully downloaded {successful}/{len(tickers)} tickers")
    
    if all_data:
        df = pd.DataFrame(all_data)
        # Sort by date and ticker
        df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
        return df
    else:
        return pd.DataFrame()

def update_existing_data(existing_file: str, output_file: str = None) -> str:
    """Update existing data file with latest data"""
    logger.info(f"📂 Loading existing data from {existing_file}")
    
    try:
        if existing_file.endswith('.parquet'):
            existing_df = pd.read_parquet(existing_file)
        else:
            existing_df = pd.read_csv(existing_file)
        
        # Get latest date in existing data
        existing_df['date'] = pd.to_datetime(existing_df['date'])
        latest_date = existing_df['date'].max()
        
        logger.info(f"📅 Latest date in existing data: {latest_date.strftime('%Y-%m-%d')}")
        
        # Check if we need to update
        current_date = datetime.strptime(get_current_trading_day(), '%Y-%m-%d')
        
        if latest_date.date() >= current_date.date():
            logger.info("✅ Data is already up to date!")
            return existing_file
        
        # Get unique tickers from existing data
        tickers = existing_df['ticker'].unique().tolist()
        logger.info(f"🎯 Updating {len(tickers)} tickers")
        
        # Download recent data to fill the gap
        days_to_update = (current_date - latest_date).days + 1
        new_data = download_recent_data(days=days_to_update, tickers=tickers)
        
        if not new_data.empty:
            # Convert date column to datetime for consistency
            new_data['date'] = pd.to_datetime(new_data['date'])
            
            # Filter out data we already have
            new_data = new_data[new_data['date'] > latest_date]
            
            if not new_data.empty:
                # Combine with existing data
                combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)
                
                # Save updated data
                if output_file is None:
                    output_file = existing_file
                
                if output_file.endswith('.parquet'):
                    combined_df.to_parquet(output_file, index=False)
                else:
                    combined_df.to_csv(output_file, index=False)
                
                logger.info(f"✅ Updated data saved to {output_file}")
                logger.info(f"📊 Added {len(new_data)} new records")
                logger.info(f"📅 New date range: {combined_df['date'].min().strftime('%Y-%m-%d')} to {combined_df['date'].max().strftime('%Y-%m-%d')}")
                
                return output_file
            else:
                logger.info("✅ No new data to add")
                return existing_file
        else:
            logger.warning("⚠️ Failed to download new data")
            return existing_file
            
    except Exception as e:
        logger.error(f"❌ Error updating data: {e}")
        return existing_file

if __name__ == "__main__":
    # Test the API
    logger.info("🧪 Testing Polygon REST API...")

    # Test recent data download (last 3 days)
    logger.info("📅 Testing recent data download...")
    recent_data = download_recent_data(days=3, tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'])

    if not recent_data.empty:
        print(f"\n📊 Recent Data Sample:")
        print(recent_data.head(10))
        print(f"\n📅 Date range: {recent_data['date'].min()} to {recent_data['date'].max()}")
        print(f"📈 Unique tickers: {recent_data['ticker'].nunique()}")
        print(f"📊 Total records: {len(recent_data)}")

        # Save test data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"recent_data_{timestamp}.csv"
        recent_data.to_csv(output_file, index=False)
        logger.info(f"💾 Test data saved to {output_file}")
    else:
        logger.warning("⚠️ No recent data retrieved")

    # Also test current day for comparison
    logger.info("\n📅 Testing current day download...")
    current_data = download_current_day_data(['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'])

    if not current_data.empty:
        print(f"\n📊 Current Day Data Sample:")
        print(current_data.head())

        # Save test data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"current_day_data_{timestamp}.csv"
        current_data.to_csv(output_file, index=False)
        logger.info(f"💾 Current day data saved to {output_file}")
    else:
        logger.info("ℹ️ Current day data not available (markets may be closed or data not ready)")
