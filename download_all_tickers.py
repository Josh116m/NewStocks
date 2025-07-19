"""
Download comprehensive stock data for all major tickers
"""

import logging
from polygon_downloader import download_2year_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_comprehensive_ticker_list():
    """Get a comprehensive list of major tickers to download."""
    
    # Major Tech Stocks
    tech_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 
        'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO',
        'PYPL', 'UBER', 'LYFT', 'ZOOM', 'DOCU', 'SHOP', 'SQ', 'TWLO'
    ]
    
    # Major Financial Stocks
    financial_stocks = [
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
        'AXP', 'BLK', 'SCHW', 'CB', 'MMC', 'AON', 'PGR', 'TRV', 'ALL', 'MET'
    ]
    
    # Major Healthcare/Pharma
    healthcare_stocks = [
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
        'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'CELG', 'ISRG', 'ILMN', 'MRNA'
    ]
    
    # Major Consumer/Retail
    consumer_stocks = [
        'AMZN', 'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'SBUX', 'NKE', 'DIS',
        'COST', 'TGT', 'LOW', 'TJX', 'ROST', 'ULTA', 'LULU', 'BBY', 'GPS'
    ]
    
    # Major Industrial/Energy
    industrial_energy = [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'KMI', 'OKE', 'WMB', 'EPD',
        'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC'
    ]
    
    # Major Utilities/REITs
    utilities_reits = [
        'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'ES', 'ED',
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'VTR', 'O'
    ]
    
    # ETFs and Market Indices
    etfs_indices = [
        'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'AGG', 'LQD', 'HYG', 'TLT',
        'GLD', 'SLV', 'USO', 'UNG', 'VIX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXL', 'SPXS',
        'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLY'
    ]
    
    # Crypto-related stocks
    crypto_stocks = [
        'COIN', 'MSTR', 'RIOT', 'MARA', 'CLSK', 'BITF', 'CAN', 'BTBT', 'SOS'
    ]
    
    # Emerging/Growth stocks
    growth_stocks = [
        'ROKU', 'PELOTON', 'ZM', 'PTON', 'SNOW', 'PLTR', 'RBLX', 'U', 'DKNG',
        'PENN', 'DRAFT', 'FUBO', 'NKLA', 'LCID', 'RIVN', 'F', 'GM', 'FORD'
    ]
    
    # Combine all lists
    all_tickers = (
        tech_stocks + financial_stocks + healthcare_stocks + 
        consumer_stocks + industrial_energy + utilities_reits + 
        etfs_indices + crypto_stocks + growth_stocks
    )
    
    # Remove duplicates and sort
    unique_tickers = sorted(list(set(all_tickers)))
    
    logger.info(f"📊 Compiled {len(unique_tickers)} unique tickers")
    logger.info(f"Categories: Tech({len(tech_stocks)}), Financial({len(financial_stocks)}), "
                f"Healthcare({len(healthcare_stocks)}), Consumer({len(consumer_stocks)}), "
                f"Industrial/Energy({len(industrial_energy)}), Utilities/REITs({len(utilities_reits)}), "
                f"ETFs({len(etfs_indices)}), Crypto({len(crypto_stocks)}), Growth({len(growth_stocks)})")
    
    return unique_tickers

def download_all_available_data():
    """Download data for all available tickers."""
    
    logger.info("🚀 COMPREHENSIVE STOCK DATA DOWNLOAD")
    logger.info("=" * 60)
    
    # Get comprehensive ticker list
    all_tickers = get_comprehensive_ticker_list()
    
    logger.info(f"🎯 Target tickers: {', '.join(all_tickers[:10])}... (and {len(all_tickers)-10} more)")
    
    # Download data
    try:
        output_file = download_2year_data(symbols=all_tickers, output_dir="./data")
        
        if output_file:
            logger.info("✅ DOWNLOAD SUCCESSFUL!")
            
            # Load and analyze the downloaded data
            import pandas as pd
            df = pd.read_csv(output_file)
            
            logger.info(f"\n📊 DOWNLOAD SUMMARY:")
            logger.info(f"   Total Records: {len(df):,}")
            logger.info(f"   Unique Tickers: {df['ticker'].nunique()}")
            logger.info(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"   File Size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            # Show ticker breakdown
            ticker_counts = df['ticker'].value_counts()
            logger.info(f"\n🏢 AVAILABLE TICKERS ({len(ticker_counts)}):")
            
            # Group tickers by count for better display
            full_data_tickers = ticker_counts[ticker_counts >= 400].index.tolist()
            partial_data_tickers = ticker_counts[ticker_counts < 400].index.tolist()
            
            logger.info(f"   Full Data (400+ days): {len(full_data_tickers)} tickers")
            if len(full_data_tickers) <= 20:
                logger.info(f"     {', '.join(full_data_tickers)}")
            else:
                logger.info(f"     {', '.join(full_data_tickers[:20])}... (and {len(full_data_tickers)-20} more)")
            
            if partial_data_tickers:
                logger.info(f"   Partial Data (<400 days): {len(partial_data_tickers)} tickers")
                if len(partial_data_tickers) <= 10:
                    logger.info(f"     {', '.join(partial_data_tickers)}")
                else:
                    logger.info(f"     {', '.join(partial_data_tickers[:10])}... (and {len(partial_data_tickers)-10} more)")
            
            # Show sample of data
            logger.info(f"\n📈 SAMPLE DATA:")
            sample_ticker = ticker_counts.index[0]
            sample_data = df[df['ticker'] == sample_ticker].tail(5)
            logger.info(f"   Latest 5 days for {sample_ticker}:")
            for _, row in sample_data.iterrows():
                logger.info(f"     {row['date']}: ${row['close']:.2f} (Vol: {row['volume']:,})")
            
            return output_file
            
        else:
            logger.error("❌ Download failed!")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error during download: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run comprehensive data download."""
    
    # Check if we already have data
    import os
    existing_file = "./data/stock_data_2year.csv"
    
    if os.path.exists(existing_file):
        logger.info(f"📁 Found existing data file: {existing_file}")
        
        # Check the existing data
        import pandas as pd
        try:
            df = pd.read_csv(existing_file)
            logger.info(f"   Current data: {len(df):,} records, {df['ticker'].nunique()} tickers")
            logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            
            # Ask if user wants to re-download
            response = input("\n🤔 Re-download data? This will overwrite existing data. (y/N): ")
            if response.lower() not in ['y', 'yes']:
                logger.info("📊 Using existing data. Run analysis with current data.")
                return existing_file
                
        except Exception as e:
            logger.warning(f"⚠️ Error reading existing file: {e}. Will re-download.")
    
    # Download new data
    return download_all_available_data()

if __name__ == "__main__":
    result = main()
    
    if result:
        print(f"\n🎉 SUCCESS! Data saved to: {result}")
        print("🚀 Ready to run comprehensive weekly analysis!")
        print("\nNext steps:")
        print("  1. Run: python simple_weekly_analysis.py")
        print("  2. Or run: python weekly_analysis.py (for advanced ML analysis)")
    else:
        print("\n❌ Download failed. Please check the logs above.")
