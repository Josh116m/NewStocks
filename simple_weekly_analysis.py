"""
Simple Weekly Stock Analysis
Provides basic technical analysis and recommendations for available stocks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_technical_indicators(df):
    """Calculate basic technical indicators for stock analysis."""
    
    # Sort by date
    df = df.sort_values('date')
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    df['returns_5d'] = df['close'].pct_change(5)
    df['returns_20d'] = df['close'].pct_change(20)
    
    # Moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # RSI (simplified)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volatility
    df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Price momentum
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    return df


def generate_signals(df):
    """Generate buy/sell signals based on technical indicators."""
    
    latest = df.iloc[-1]
    
    signals = []
    score = 0
    
    # Moving average signals
    if latest['close'] > latest['sma_5'] > latest['sma_20']:
        signals.append("✅ Price above short-term MAs")
        score += 2
    elif latest['close'] < latest['sma_5'] < latest['sma_20']:
        signals.append("❌ Price below short-term MAs")
        score -= 2
    
    # RSI signals
    if latest['rsi'] < 30:
        signals.append("🔥 RSI oversold - potential buy")
        score += 3
    elif latest['rsi'] > 70:
        signals.append("⚠️ RSI overbought - potential sell")
        score -= 2
    elif 40 <= latest['rsi'] <= 60:
        signals.append("📊 RSI neutral")
        score += 1
    
    # Bollinger Band signals
    if latest['bb_position'] < 0.2:
        signals.append("🎯 Near lower Bollinger Band - potential buy")
        score += 2
    elif latest['bb_position'] > 0.8:
        signals.append("⚠️ Near upper Bollinger Band - potential sell")
        score -= 1
    
    # Volume signals
    if latest['volume_ratio'] > 1.5:
        signals.append("📈 High volume activity")
        score += 1
    elif latest['volume_ratio'] < 0.5:
        signals.append("📉 Low volume activity")
        score -= 1
    
    # Momentum signals
    if latest['momentum_20'] > 0.05:
        signals.append("🚀 Strong 20-day momentum")
        score += 2
    elif latest['momentum_20'] < -0.05:
        signals.append("⬇️ Weak 20-day momentum")
        score -= 2
    
    # Volatility signals
    if latest['volatility_20'] > 0.4:
        signals.append("⚡ High volatility - risky")
        score -= 1
    elif latest['volatility_20'] < 0.2:
        signals.append("😌 Low volatility - stable")
        score += 1
    
    # Generate recommendation
    if score >= 5:
        recommendation = "STRONG BUY"
        confidence = min(0.9, 0.6 + (score - 5) * 0.05)
    elif score >= 2:
        recommendation = "BUY"
        confidence = 0.6 + (score - 2) * 0.05
    elif score >= -1:
        recommendation = "HOLD"
        confidence = 0.5 + abs(score) * 0.02
    elif score >= -4:
        recommendation = "SELL"
        confidence = 0.6 + abs(score + 1) * 0.05
    else:
        recommendation = "STRONG SELL"
        confidence = min(0.9, 0.6 + abs(score + 4) * 0.05)
    
    return {
        'recommendation': recommendation,
        'confidence': confidence,
        'score': score,
        'signals': signals,
        'rsi': latest['rsi'],
        'bb_position': latest['bb_position'],
        'momentum_20': latest['momentum_20'],
        'volatility_20': latest['volatility_20'],
        'volume_ratio': latest['volume_ratio']
    }


def run_simple_weekly_analysis():
    """Run simple weekly analysis for all available tickers."""
    logger.info("🚀 SIMPLE WEEKLY STOCK ANALYSIS")
    logger.info("=" * 60)
    
    # Load stock data
    try:
        data = pd.read_csv("data/stock_data_2year.csv")
        data['date'] = pd.to_datetime(data['date'])
        
        # Get recent data (last 100 days for better indicators)
        cutoff_date = data['date'].max() - timedelta(days=100)
        recent_data = data[data['date'] >= cutoff_date]
        
        logger.info(f"📊 Loaded data for {recent_data['ticker'].nunique()} tickers")
        logger.info(f"📅 Date range: {recent_data['date'].min()} to {recent_data['date'].max()}")
        
        tickers = recent_data['ticker'].unique()
        logger.info(f"🎯 Available tickers: {', '.join(tickers)}")
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return None
    
    # Analyze each ticker
    results = []
    
    for ticker in tickers:
        logger.info(f"\n🔍 Analyzing {ticker}...")
        
        ticker_data = recent_data[recent_data['ticker'] == ticker].copy()
        
        if len(ticker_data) < 50:
            logger.warning(f"⚠️ Insufficient data for {ticker}")
            continue
        
        # Calculate technical indicators
        ticker_data = calculate_technical_indicators(ticker_data)
        
        # Generate signals
        analysis = generate_signals(ticker_data)
        
        # Get current price info
        latest = ticker_data.iloc[-1]
        
        results.append({
            'ticker': ticker,
            'current_price': latest['close'],
            'recommendation': analysis['recommendation'],
            'confidence': analysis['confidence'],
            'score': analysis['score'],
            'rsi': analysis['rsi'],
            'bb_position': analysis['bb_position'],
            'momentum_20': analysis['momentum_20'],
            'volatility_20': analysis['volatility_20'],
            'volume_ratio': analysis['volume_ratio'],
            'signals': analysis['signals']
        })
    
    # Sort by score (best opportunities first)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)
    
    # Display results
    logger.info("\n📊 WEEKLY STOCK RANKINGS:")
    logger.info("=" * 80)
    
    print("\n🏆 TOP RECOMMENDATIONS FOR THIS WEEK:")
    print("=" * 50)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        rec_emoji = {"STRONG BUY": "🔥", "BUY": "🚀", "HOLD": "📊", "SELL": "⬇️", "STRONG SELL": "💀"}
        emoji = rec_emoji.get(row['recommendation'], "📈")
        
        print(f"{i}. {emoji} {row['ticker']:6s} {row['recommendation']:12s} "
              f"| Score: {row['score']:3.0f} | Conf: {row['confidence']:.2f} "
              f"| Price: ${row['current_price']:.2f}")
    
    # Detailed analysis for each stock
    logger.info("\n🔍 DETAILED ANALYSIS:")
    logger.info("=" * 50)
    
    for _, row in results_df.iterrows():
        print(f"\n📈 {row['ticker']} - {row['recommendation']}")
        print(f"   Current Price: ${row['current_price']:.2f}")
        print(f"   Confidence: {row['confidence']:.2f}")
        print(f"   Technical Score: {row['score']}")
        print(f"   RSI: {row['rsi']:.1f}")
        print(f"   20-day Momentum: {row['momentum_20']:.2%}")
        print(f"   Volatility: {row['volatility_20']:.2f}")
        print(f"   Volume Ratio: {row['volume_ratio']:.2f}")
        print("   Key Signals:")
        for signal in row['signals']:
            print(f"     • {signal}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"predictions/simple_weekly_analysis_{timestamp}.csv"
    
    # Create predictions directory if it doesn't exist
    import os
    os.makedirs("predictions", exist_ok=True)
    
    results_df.to_csv(output_file, index=False)
    logger.info(f"\n💾 Analysis saved to: {output_file}")
    
    # Summary
    buy_signals = len(results_df[results_df['recommendation'].isin(['BUY', 'STRONG BUY'])])
    sell_signals = len(results_df[results_df['recommendation'].isin(['SELL', 'STRONG SELL'])])
    hold_signals = len(results_df[results_df['recommendation'] == 'HOLD'])
    
    logger.info(f"\n📈 SUMMARY:")
    logger.info(f"   Total Stocks Analyzed: {len(results_df)}")
    logger.info(f"   BUY Recommendations: {buy_signals}")
    logger.info(f"   HOLD Recommendations: {hold_signals}")
    logger.info(f"   SELL Recommendations: {sell_signals}")
    
    return results_df


if __name__ == "__main__":
    # Run the simple analysis
    results = run_simple_weekly_analysis()
    
    if results is not None:
        print("\n" + "="*60)
        print("🎉 SIMPLE WEEKLY ANALYSIS COMPLETE!")
        print("="*60)
        print(f"✅ Analyzed {len(results)} stocks")
        print(f"📊 Results saved to predictions/ directory")
        print(f"🚀 Ready for trading decisions!")
    else:
        print("\n❌ Analysis failed. Please check the logs above.")
