"""
Simple Technical Analysis Functions
A lightweight replacement for pandas-ta with basic indicators
"""

import pandas as pd
import numpy as np

def sma(series, window):
    """Simple Moving Average"""
    return series.rolling(window=window).mean()

def ema(series, window):
    """Exponential Moving Average"""
    return series.ewm(span=window).mean()

def rsi(series, window=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence)"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })

def bollinger_bands(series, window=20, std_dev=2):
    """Bollinger Bands"""
    sma_line = sma(series, window)
    std = series.rolling(window=window).std()
    
    return pd.DataFrame({
        'BB_Upper': sma_line + (std * std_dev),
        'BB_Middle': sma_line,
        'BB_Lower': sma_line - (std * std_dev)
    })

def stochastic(high, low, close, k_window=14, d_window=3):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return pd.DataFrame({
        'STOCH_K': k_percent,
        'STOCH_D': d_percent
    })

def atr(high, low, close, window=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()

def williams_r(high, low, close, window=14):
    """Williams %R"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    
    return -100 * ((highest_high - close) / (highest_high - lowest_low))

def cci(high, low, close, window=20):
    """Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=window).mean()
    mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
    
    return (typical_price - sma_tp) / (0.015 * mad)

def momentum(series, window=10):
    """Price Momentum"""
    return series / series.shift(window) - 1

def rate_of_change(series, window=10):
    """Rate of Change"""
    return ((series - series.shift(window)) / series.shift(window)) * 100

def add_all_indicators(df):
    """
    Add all technical indicators to a DataFrame
    Expected columns: open, high, low, close, volume
    """
    result = df.copy()
    
    # Price-based indicators
    result['SMA_10'] = sma(df['close'], 10)
    result['SMA_20'] = sma(df['close'], 20)
    result['SMA_50'] = sma(df['close'], 50)
    result['EMA_12'] = ema(df['close'], 12)
    result['EMA_26'] = ema(df['close'], 26)
    
    # RSI
    result['RSI'] = rsi(df['close'])
    
    # MACD
    macd_data = macd(df['close'])
    result = pd.concat([result, macd_data], axis=1)
    
    # Bollinger Bands
    bb_data = bollinger_bands(df['close'])
    result = pd.concat([result, bb_data], axis=1)
    
    # Stochastic
    stoch_data = stochastic(df['high'], df['low'], df['close'])
    result = pd.concat([result, stoch_data], axis=1)
    
    # ATR
    result['ATR'] = atr(df['high'], df['low'], df['close'])
    
    # Williams %R
    result['WILLIAMS_R'] = williams_r(df['high'], df['low'], df['close'])
    
    # CCI
    result['CCI'] = cci(df['high'], df['low'], df['close'])
    
    # Momentum indicators
    result['MOMENTUM_10'] = momentum(df['close'], 10)
    result['ROC_10'] = rate_of_change(df['close'], 10)
    
    # Volume indicators
    result['VOLUME_SMA_10'] = sma(df['volume'], 10)
    result['VOLUME_RATIO'] = df['volume'] / result['VOLUME_SMA_10']
    
    return result

# Alias for compatibility
def ta_add_all(df):
    """Alias for add_all_indicators for compatibility"""
    return add_all_indicators(df)
