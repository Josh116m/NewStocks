import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import simple_ta as ta
from sklearn.cluster import DBSCAN
from scipy.signal import argrelextrema
import warnings
import threading
import time
import pickle
import hashlib
import os
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

# GPU acceleration imports
try:
    import cupy as cp
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    print(f"🚀 GPU acceleration available: {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        print(f"   Device: {torch.cuda.get_device_name(0)}")
except ImportError:
    cp = None
    torch = None
    GPU_AVAILABLE = False
    print("⚠️ GPU acceleration not available - using CPU")

# GPU monitoring imports
try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False
    print("⚠️ pynvml not available - GPU monitoring disabled")

class GPUMonitor:
    """Real-time GPU monitoring during feature engineering."""

    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_stats = []

    def start_monitoring(self, interval=1.0):
        """Start GPU monitoring in a separate thread."""
        if not NVML_AVAILABLE or not GPU_AVAILABLE:
            print("⚠️ GPU monitoring not available")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 GPU monitoring started...")

    def stop_monitoring(self):
        """Stop GPU monitoring and print summary."""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        if self.gpu_stats:
            avg_util = sum(stat['utilization'] for stat in self.gpu_stats) / len(self.gpu_stats)
            max_util = max(stat['utilization'] for stat in self.gpu_stats)
            avg_memory = sum(stat['memory_used'] for stat in self.gpu_stats) / len(self.gpu_stats)
            max_memory = max(stat['memory_used'] for stat in self.gpu_stats)

            print(f"📊 GPU Monitoring Summary:")
            print(f"   Average Utilization: {avg_util:.1f}%")
            print(f"   Peak Utilization: {max_util:.1f}%")
            print(f"   Average Memory: {avg_memory:.1f} MB")
            print(f"   Peak Memory: {max_memory:.1f} MB")

    def _monitor_loop(self, interval):
        """Monitor GPU usage in a loop."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            while self.monitoring:
                # Get GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = mem_info.used / 1024 / 1024  # Convert to MB
                memory_total = mem_info.total / 1024 / 1024

                # Store stats
                stats = {
                    'utilization': util.gpu,
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'timestamp': time.time()
                }
                self.gpu_stats.append(stats)

                # Print real-time stats
                print(f"🔥 GPU: {util.gpu:3d}% | Memory: {memory_used:6.0f}/{memory_total:6.0f} MB ({memory_used/memory_total*100:4.1f}%)")

                time.sleep(interval)

        except Exception as e:
            print(f"⚠️ GPU monitoring error: {e}")

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for stock prediction with multi-horizon indicators.
    Designed to work with Polygon.io daily aggregated data format.
    GPU-accelerated when available.
    """

    def __init__(self,
                 short_window: int = 14,
                 medium_window: int = 63,
                 long_window: int = 252,
                 use_gpu: bool = False,  # Default to CPU for better performance
                 cache_dir: str = "cache/features",
                 enable_cache: bool = True,
                 n_jobs: int = -1):  # Add multiprocessing support
        """
        Initialize feature engineer with time horizons and caching.

        Args:
            short_window: Short-term horizon (default 14 days ~ 2 weeks)
            medium_window: Medium-term horizon (default 63 days ~ 3 months)
            long_window: Long-term horizon (default 252 days ~ 1 year)
            use_gpu: Whether to use GPU acceleration when available (default False for better performance)
            cache_dir: Directory to store cached features
            enable_cache: Whether to enable feature caching
            n_jobs: Number of parallel jobs for multiprocessing (-1 for all cores)
        """
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.n_jobs = n_jobs

        # Create cache directory if it doesn't exist
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"💾 Feature caching enabled: {self.cache_dir}")

        if self.use_gpu:
            print("🚀 GPU acceleration enabled for feature engineering")
        else:
            print(f"💻 Using optimized CPU for feature engineering (n_jobs={n_jobs})")

    def _generate_cache_key(self, df: pd.DataFrame, feature_type: str = "all") -> str:
        """Generate a unique cache key based on data content and parameters."""
        # Create a hash based on data content and feature engineering parameters
        data_hash = hashlib.md5()

        # Hash the data content (sample of key columns to avoid huge hashes)
        if len(df) > 1000:
            # For large datasets, sample every 10th row for hashing
            sample_df = df.iloc[::10]
        else:
            sample_df = df

        # Include key columns in hash
        key_cols = ['ticker', 'date', 'close', 'volume']
        for col in key_cols:
            if col in sample_df.columns:
                data_hash.update(str(sample_df[col].values).encode())

        # Include feature engineering parameters
        params = f"{self.short_window}_{self.medium_window}_{self.long_window}_{feature_type}"
        data_hash.update(params.encode())

        return data_hash.hexdigest()

    def _get_cache_path(self, cache_key: str, feature_type: str = "all") -> Path:
        """Get the cache file path for a given cache key and feature type."""
        return self.cache_dir / f"features_{feature_type}_{cache_key}.pkl"

    def _save_to_cache(self, df: pd.DataFrame, cache_key: str, feature_type: str = "all"):
        """Save processed features to cache."""
        if not self.enable_cache:
            return

        try:
            cache_path = self._get_cache_path(cache_key, feature_type)
            cache_data = {
                'data': df,
                'timestamp': datetime.now(),
                'feature_type': feature_type,
                'parameters': {
                    'short_window': self.short_window,
                    'medium_window': self.medium_window,
                    'long_window': self.long_window
                }
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Cached {feature_type} features: {cache_path.name}")

        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")

    def _load_from_cache(self, cache_key: str, feature_type: str = "all") -> Optional[pd.DataFrame]:
        """Load processed features from cache."""
        if not self.enable_cache:
            return None

        try:
            cache_path = self._get_cache_path(cache_key, feature_type)
            if not cache_path.exists():
                return None

            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Verify cache is compatible with current parameters
            cached_params = cache_data.get('parameters', {})
            current_params = {
                'short_window': self.short_window,
                'medium_window': self.medium_window,
                'long_window': self.long_window
            }

            if cached_params != current_params:
                print(f"⚠️ Cache parameters mismatch, ignoring cache for {feature_type}")
                return None

            print(f"✅ Loaded {feature_type} features from cache: {cache_path.name}")
            return cache_data['data']

        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}")
            return None

    def _identify_new_data(self, current_df: pd.DataFrame, cached_df: pd.DataFrame) -> pd.DataFrame:
        """Identify new data that needs processing by comparing with cached data."""
        try:
            # Convert dates to datetime for comparison
            current_df['date'] = pd.to_datetime(current_df['date'])
            cached_df['date'] = pd.to_datetime(cached_df['date'])

            # Find the latest date in cached data for each ticker
            cached_latest = cached_df.groupby('ticker')['date'].max().reset_index()
            cached_latest.columns = ['ticker', 'cached_latest_date']

            # Merge with current data to identify new records
            current_with_cache_info = current_df.merge(cached_latest, on='ticker', how='left')

            # Identify new data (records after the cached latest date for each ticker)
            new_data_mask = (
                current_with_cache_info['cached_latest_date'].isna() |  # New tickers
                (current_with_cache_info['date'] > current_with_cache_info['cached_latest_date'])  # New dates
            )

            # Use the mask on the merged dataframe, then select original columns
            new_data = current_with_cache_info[new_data_mask][current_df.columns].copy()
            print(f"📊 Identified {len(new_data)} new records to process")

            return new_data

        except Exception as e:
            print(f"⚠️ Error identifying new data: {e}")
            return current_df  # Process all data if identification fails

    def clear_cache(self, feature_type: str = None):
        """Clear cached features."""
        if not self.enable_cache:
            return

        try:
            if feature_type:
                # Clear specific feature type
                pattern = f"features_{feature_type}_*.pkl"
                files = list(self.cache_dir.glob(pattern))
            else:
                # Clear all cached features
                files = list(self.cache_dir.glob("features_*.pkl"))

            for file in files:
                file.unlink()

            print(f"🗑️ Cleared {len(files)} cached feature files")

        except Exception as e:
            print(f"⚠️ Error clearing cache: {e}")

    def _fast_rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Optimized CPU rolling mean calculation using pandas (fastest for this data size)."""
        return pd.Series(data).rolling(window, min_periods=1).mean().values

    def _fast_rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Optimized CPU rolling standard deviation calculation."""
        return pd.Series(data).rolling(window, min_periods=1).std().values

    def _fast_rolling_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """Optimized CPU rolling max calculation."""
        return pd.Series(data).rolling(window, min_periods=1).max().values

    def _fast_rolling_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """Optimized CPU rolling min calculation."""
        return pd.Series(data).rolling(window, min_periods=1).min().values

    def _vectorized_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Vectorized True Range calculation for better performance."""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # Handle first value

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        return np.maximum(tr1, np.maximum(tr2, tr3))

    def _gpu_rolling_mean_cupy(self, data: cp.ndarray, window: int) -> cp.ndarray:
        """GPU-accelerated rolling mean for CuPy arrays."""
        result = cp.full_like(data, cp.nan, dtype=cp.float32)

        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            if i >= window - 1:
                window_data = data[start_idx:i+1]
                result[i] = cp.mean(window_data)

        return result

    def _gpu_rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """GPU-accelerated rolling standard deviation calculation."""
        use_gpu = getattr(self, 'use_gpu', False) and GPU_AVAILABLE
        if not use_gpu or cp is None:
            return pd.Series(data).rolling(window).std().values

        try:
            gpu_data = cp.asarray(data)
            result = cp.zeros_like(gpu_data)

            # Calculate rolling std using sliding window
            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                if i >= window - 1:
                    window_data = gpu_data[start_idx:i+1]
                    result[i] = cp.std(window_data)
                else:
                    result[i] = cp.nan

            return cp.asnumpy(result)
        except Exception as e:
            print(f"⚠️ GPU rolling std failed, falling back to CPU: {e}")
            return pd.Series(data).rolling(window).std().values

    def _gpu_rolling_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """GPU-accelerated rolling max calculation."""
        use_gpu = getattr(self, 'use_gpu', False) and GPU_AVAILABLE
        if not use_gpu or cp is None:
            return pd.Series(data).rolling(window).max().values

        try:
            gpu_data = cp.asarray(data)
            result = cp.zeros_like(gpu_data)

            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                if i >= window - 1:
                    window_data = gpu_data[start_idx:i+1]
                    result[i] = cp.max(window_data)
                else:
                    result[i] = cp.nan

            return cp.asnumpy(result)
        except Exception as e:
            print(f"⚠️ GPU rolling max failed, falling back to CPU: {e}")
            return pd.Series(data).rolling(window).max().values

    def _gpu_rolling_max_cupy(self, data: cp.ndarray, window: int) -> cp.ndarray:
        """GPU-accelerated rolling max for CuPy arrays."""
        result = cp.full_like(data, cp.nan, dtype=cp.float32)

        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            if i >= window - 1:
                window_data = data[start_idx:i+1]
                result[i] = cp.max(window_data)

        return result

    def _gpu_rolling_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """GPU-accelerated rolling min calculation."""
        use_gpu = getattr(self, 'use_gpu', False) and GPU_AVAILABLE
        if not use_gpu or cp is None:
            return pd.Series(data).rolling(window).min().values

        try:
            gpu_data = cp.asarray(data)
            result = cp.zeros_like(gpu_data)

            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                if i >= window - 1:
                    window_data = gpu_data[start_idx:i+1]
                    result[i] = cp.min(window_data)
                else:
                    result[i] = cp.nan

            return cp.asnumpy(result)
        except Exception as e:
            print(f"⚠️ GPU rolling min failed, falling back to CPU: {e}")
            return pd.Series(data).rolling(window).min().values

    def _gpu_rolling_min_cupy(self, data: cp.ndarray, window: int) -> cp.ndarray:
        """GPU-accelerated rolling min for CuPy arrays."""
        result = cp.full_like(data, cp.nan, dtype=cp.float32)

        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            if i >= window - 1:
                window_data = data[start_idx:i+1]
                result[i] = cp.min(window_data)

        return result

    def _calculate_slopes_gpu_cupy(self, data: cp.ndarray, window: int) -> cp.ndarray:
        """Calculate rolling linear regression slopes with CuPy GPU acceleration."""
        result = cp.full(len(data), cp.nan, dtype=cp.float32)

        for i in range(window - 1, len(data)):
            y_vals = data[i - window + 1:i + 1]
            x_vals = cp.arange(window, dtype=cp.float32)

            # Calculate slope using least squares
            x_mean = cp.mean(x_vals)
            y_mean = cp.mean(y_vals)

            numerator = cp.sum((x_vals - x_mean) * (y_vals - y_mean))
            denominator = cp.sum((x_vals - x_mean) ** 2)

            if denominator != 0:
                result[i] = numerator / denominator

        return result

    def _compute_trend_strength_gpu_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized GPU processing for trend strength features with better GPU utilization."""
        print("🚀 Processing trend strength features with optimized GPU acceleration...")

        total_tickers = len(df['ticker'].unique())
        processed_tickers = 0

        # Process all tickers with GPU acceleration
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_data = df[ticker_mask].copy()

            if len(ticker_data) < 14:
                continue

            # Convert all data to GPU arrays at once for better memory utilization
            high_vals = cp.asarray(ticker_data['high'].values, dtype=cp.float32)
            low_vals = cp.asarray(ticker_data['low'].values, dtype=cp.float32)
            close_vals = cp.asarray(ticker_data['close'].values, dtype=cp.float32)

            processed_tickers += 1
            if processed_tickers % 100 == 0:
                print(f"🔥 Processing {ticker} on GPU: {len(ticker_data)} data points ({processed_tickers}/{total_tickers} tickers)")

            # ATR calculations on GPU
            for period in [14, 63, 252]:
                if len(ticker_data) >= period:
                    # Calculate True Range components on GPU
                    tr1 = high_vals - low_vals
                    tr2 = cp.abs(high_vals - cp.roll(close_vals, 1))
                    tr3 = cp.abs(low_vals - cp.roll(close_vals, 1))

                    # True Range is the maximum of the three
                    true_range = cp.maximum(tr1, cp.maximum(tr2, tr3))
                    true_range[0] = tr1[0]

                    # Calculate ATR using optimized GPU rolling mean
                    atr_values = self._gpu_rolling_mean_cupy(true_range, period)
                    df.loc[ticker_mask, f'ATR_{period}'] = cp.asnumpy(atr_values)

            # Linear regression slopes on GPU
            for period in [20, 252]:
                if len(ticker_data) >= period:
                    slopes = self._calculate_slopes_gpu_cupy(close_vals, period)
                    df.loc[ticker_mask, f'Linear_Slope_{period}'] = cp.asnumpy(slopes)

            # High-Low ratio on GPU
            for period in [25]:
                if len(ticker_data) >= period:
                    rolling_high = self._gpu_rolling_max_cupy(high_vals, period)
                    rolling_low = self._gpu_rolling_min_cupy(low_vals, period)
                    ratio = (rolling_high - rolling_low) / close_vals
                    df.loc[ticker_mask, f'High_Low_Ratio_{period}'] = cp.asnumpy(ratio)

            # Force GPU computation by doing a dummy operation
            _ = cp.sum(high_vals * low_vals * close_vals)

        return df

        # Feature configuration matching the implementation plan
        self.feature_categories = {
            'trend_strength': {
                'ADX_14': 14,
                'ADX_63': 63,
                'ADX_252': 252,
                'Linear_Slope_20': 20,
                'Linear_Slope_252': 252,
                'Aroon_Up_Down_25': 25,
            },
            'volume_accumulation': {
                'OBV': None,
                'AD_Line': None,
                'CMF_20': 20,
                'Volume_SMA_Ratio': (5, 90),
                'PVT': None,
                'MFI_14': 14,
            },
            'beta_correlation': {
                'Beta_SPY_60': 60,
                'Beta_SPY_252': 252,
                'Rolling_Corr_30': 30,
                'Beta_Sector_ETF_90': 90,
            },
            'support_resistance': {
                'Distance_From_52W_High': 252,
                'Distance_From_52W_Low': 252,
                'Local_Resistance_20': 20,
                'Local_Support_20': 20,
                'Pivot_Points': 5,
                'Price_Percentile_252': 252,
            },
            'momentum': {
                'RSI_14': 14,
                'RSI_63': 63,
                'MACD_Signal': (12, 26, 9),
                'Stoch_K': 14,
                'Williams_R': 14,
                'ROC_20': 20,
            },
            'volatility': {
                'ATR_14': 14,
                'ATR_63': 63,
                'Bollinger_Width': 20,
                'Keltner_Width': 20,
                'Historical_Vol_20': 20,
                'Historical_Vol_63': 63,
            }
        }
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw data for feature engineering.
        
        Args:
            df: Raw dataframe with columns: ticker, date, open, high, low, close, volume
            
        Returns:
            Prepared dataframe sorted by ticker and date
        """
        # Ensure proper data types
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by ticker and date for proper time series processing
        df = df.sort_values(['ticker', 'date'])
        
        # Add returns for various calculations
        df['returns'] = df.groupby('ticker')['close'].pct_change()
        
        # Add log returns for better statistical properties
        df['log_returns'] = np.log(df['close'] / df.groupby('ticker')['close'].shift(1))
        
        return df

    def compute_trend_strength_features_cached(self, df: pd.DataFrame,
                                             force_recompute: bool = False) -> pd.DataFrame:
        """
        Compute trend strength features with caching and GPU optimization.
        This is the main entry point for cached trend strength computation.
        """
        print("📈 Computing trend strength features with caching...")

        # Generate cache key specifically for trend strength features
        cache_key = self._generate_cache_key(df, "trend_strength")

        # Try to load from cache first
        if not force_recompute:
            cached_result = self._load_from_cache(cache_key, "trend_strength")
            if cached_result is not None:
                # Check if we have new data
                new_data = self._identify_new_data(df, cached_result)

                if len(new_data) == 0:
                    print("✅ All trend strength features already cached")
                    return cached_result
                elif len(new_data) < len(df) * 0.1:  # Less than 10% new data
                    print(f"📊 Processing {len(new_data)} new records for trend strength")
                    return self._process_incremental_trend_features(df, cached_result, new_data)

        # Compute all trend strength features
        print("🔄 Computing all trend strength features from scratch...")
        result = self.compute_trend_strength_features(df)

        # Cache the result
        self._save_to_cache(result, cache_key, "trend_strength")

        return result

    def _process_incremental_trend_features(self, current_df: pd.DataFrame,
                                          cached_df: pd.DataFrame,
                                          new_data: pd.DataFrame) -> pd.DataFrame:
        """Process trend strength features incrementally for new data only."""
        print("🔄 Processing incremental trend strength features...")

        # For trend features, we need sufficient historical context
        extended_new_data = []

        for ticker in new_data['ticker'].unique():
            ticker_cached = cached_df[cached_df['ticker'] == ticker]
            ticker_new = new_data[new_data['ticker'] == ticker]

            if len(ticker_cached) > 0:
                # Get last 300 records for context (enough for 252-day calculations)
                context_data = ticker_cached.tail(300)
                ticker_extended = pd.concat([context_data, ticker_new], ignore_index=True)
                ticker_extended = ticker_extended.drop_duplicates(subset=['ticker', 'date'], keep='last')
            else:
                ticker_extended = ticker_new

            extended_new_data.append(ticker_extended)

        if extended_new_data:
            extended_df = pd.concat(extended_new_data, ignore_index=True)

            # Process trend features for extended data
            processed_extended = self.compute_trend_strength_features(extended_df)

            # Extract only the new features
            new_features = []
            for ticker in new_data['ticker'].unique():
                ticker_new_dates = new_data[new_data['ticker'] == ticker]['date'].unique()
                ticker_processed = processed_extended[
                    (processed_extended['ticker'] == ticker) &
                    (processed_extended['date'].isin(ticker_new_dates))
                ]
                new_features.append(ticker_processed)

            if new_features:
                new_features_df = pd.concat(new_features, ignore_index=True)

                # Merge with cached data
                non_overlapping_cached = cached_df[
                    ~cached_df.set_index(['ticker', 'date']).index.isin(
                        new_features_df.set_index(['ticker', 'date']).index
                    )
                ]

                result = pd.concat([non_overlapping_cached, new_features_df], ignore_index=True)
                result = result.sort_values(['ticker', 'date']).reset_index(drop=True)

                # Update cache
                cache_key = self._generate_cache_key(current_df, "trend_strength")
                self._save_to_cache(result, cache_key, "trend_strength")

                return result

        # Fallback
        return self.compute_trend_strength_features(current_df)

    def compute_trend_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute trend strength indicators with optimized CPU processing."""
        print("📈 Computing trend strength features with optimized CPU processing...")

        # Use vectorized operations with groupby for maximum efficiency
        df_grouped = df.groupby('ticker')

        # Pre-allocate result columns
        for period in [14, 63, 252]:
            df[f'ATR_{period}'] = np.nan
        for period in [20, 252]:
            df[f'Linear_Slope_{period}'] = np.nan
        df['High_Low_Ratio_25'] = np.nan

        # Process each ticker efficiently
        for ticker, group in df_grouped:
            if len(group) < 14:  # Skip if not enough data
                continue

            # Get indices for this ticker
            ticker_idx = group.index

            # Extract values once
            high_vals = group['high'].values
            low_vals = group['low'].values
            close_vals = group['close'].values

            # Vectorized True Range calculation
            true_range = self._vectorized_true_range(high_vals, low_vals, close_vals)

            # ATR calculations using optimized rolling operations
            for period in [14, 63, 252]:
                if len(group) >= period:
                    atr_values = self._fast_rolling_mean(true_range, period)
                    df.loc[ticker_idx, f'ATR_{period}'] = atr_values

            # Linear regression slopes using optimized calculation
            for period in [20, 252]:
                if len(group) >= period:
                    slopes = self._calculate_slopes_optimized(close_vals, period)
                    df.loc[ticker_idx, f'Linear_Slope_{period}'] = slopes

            # High-Low ratio using vectorized operations
            if len(group) >= 25:
                rolling_high = self._fast_rolling_max(high_vals, 25)
                rolling_low = self._fast_rolling_min(low_vals, 25)
                ratio = (rolling_high - rolling_low) / close_vals
                df.loc[ticker_idx, 'High_Low_Ratio_25'] = ratio

        print("✅ Trend strength features computed successfully!")
        return df

    def _calculate_slopes_optimized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Optimized CPU-based rolling linear regression slopes using vectorized operations."""
        if len(data) < window:
            return np.full(len(data), np.nan)

        result = np.full(len(data), np.nan)

        # Pre-compute x values for regression (constant for all windows)
        x_vals = np.arange(window, dtype=np.float64)
        x_mean = np.mean(x_vals)
        x_centered = x_vals - x_mean
        x_sum_sq = np.sum(x_centered ** 2)

        # Vectorized slope calculation
        for i in range(window - 1, len(data)):
            y_window = data[i - window + 1:i + 1]
            if not np.isnan(y_window).any():  # Only compute if no NaN values
                y_mean = np.mean(y_window)
                y_centered = y_window - y_mean
                numerator = np.sum(x_centered * y_centered)
                if x_sum_sq != 0:
                    result[i] = numerator / x_sum_sq

        return result

    def _calculate_slopes_gpu(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling linear regression slopes with GPU acceleration."""
        use_gpu = getattr(self, 'use_gpu', False) and GPU_AVAILABLE
        if not use_gpu or len(data) < window:
            # Fallback to CPU calculation
            return pd.Series(data).rolling(window).apply(self._calculate_slope, raw=False).values

        try:
            result = np.full(len(data), np.nan)

            for i in range(window - 1, len(data)):
                y_window = data[i - window + 1:i + 1]
                x_window = np.arange(window)

                # Calculate slope using least squares
                x_mean = np.mean(x_window)
                y_mean = np.mean(y_window)

                numerator = np.sum((x_window - x_mean) * (y_window - y_mean))
                denominator = np.sum((x_window - x_mean) ** 2)

                if denominator != 0:
                    result[i] = numerator / denominator

            return result
        except Exception as e:
            print(f"⚠️ GPU slope calculation failed, falling back to CPU: {e}")
            return pd.Series(data).rolling(window).apply(self._calculate_slope, raw=False).values

    def _calculate_rsi_gpu(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI with GPU acceleration."""
        use_gpu = getattr(self, 'use_gpu', False) and GPU_AVAILABLE
        if not use_gpu or len(data) < period + 1:
            return ta.rsi(pd.Series(data), window=period).values

        try:
            # Calculate price changes
            delta = np.diff(data)
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)

            # Calculate average gains and losses
            avg_gains = self._gpu_rolling_mean(gains, period)
            avg_losses = self._gpu_rolling_mean(losses, period)

            # Calculate RSI
            rs = avg_gains / (avg_losses + 1e-10)  # Add small epsilon to avoid division by zero
            rsi = 100 - (100 / (1 + rs))

            # Prepend NaN for the first value (since we used diff)
            result = np.full(len(data), np.nan)
            result[1:] = rsi

            return result
        except Exception as e:
            print(f"⚠️ GPU RSI calculation failed, falling back to CPU: {e}")
            return ta.rsi(pd.Series(data), window=period).values

    def _calculate_roc_gpu(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Rate of Change with GPU acceleration."""
        use_gpu = getattr(self, 'use_gpu', False) and GPU_AVAILABLE
        if not use_gpu or len(data) <= period:
            return ta.rate_of_change(pd.Series(data), window=period).values

        try:
            if use_gpu and cp is not None:
                gpu_data = cp.asarray(data)
                result = cp.full(len(data), cp.nan)

                # Calculate ROC: ((current - previous) / previous) * 100
                result[period:] = ((gpu_data[period:] - gpu_data[:-period]) / gpu_data[:-period]) * 100

                return cp.asnumpy(result)
            else:
                result = np.full(len(data), np.nan)
                result[period:] = ((data[period:] - data[:-period]) / data[:-period]) * 100
                return result
        except Exception as e:
            print(f"⚠️ GPU ROC calculation failed, falling back to CPU: {e}")
            return ta.rate_of_change(pd.Series(data), window=period).values

    def _compute_momentum_gpu_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized GPU processing for momentum features."""
        print("🚀 Processing momentum features with optimized GPU acceleration...")

        # Process by ticker for efficiency
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_data = df[ticker_mask].copy()

            if len(ticker_data) < 14:
                continue

            # Convert to GPU arrays for batch processing
            close_vals = cp.asarray(ticker_data['close'].values, dtype=cp.float32)
            high_vals = cp.asarray(ticker_data['high'].values, dtype=cp.float32)
            low_vals = cp.asarray(ticker_data['low'].values, dtype=cp.float32)

            print(f"🔥 Processing momentum for {ticker} on GPU: {len(ticker_data)} data points")

            # RSI at multiple horizons using GPU acceleration
            for period in [14, 63]:
                if len(ticker_data) >= period:
                    rsi_values = self._calculate_rsi_gpu_cupy(close_vals, period)
                    df.loc[ticker_mask, f'RSI_{period}'] = cp.asnumpy(rsi_values)

            # Rate of Change using GPU acceleration
            if len(ticker_data) >= 20:
                roc_values = self._calculate_roc_gpu_cupy(close_vals, 20)
                df.loc[ticker_mask, 'ROC_20'] = cp.asnumpy(roc_values)

            # Williams %R using GPU acceleration
            if len(ticker_data) >= 14:
                williams_r_values = self._calculate_williams_r_gpu_cupy(high_vals, low_vals, close_vals, 14)
                df.loc[ticker_mask, 'Williams_R'] = cp.asnumpy(williams_r_values)

            # Force GPU computation
            _ = cp.sum(close_vals * high_vals * low_vals)

        # MACD (keep using simple_ta for now as it's complex to implement efficiently)
        macd_data = ta.macd(df['close'])
        df['MACD'] = macd_data['MACD']
        df['MACD_Signal'] = macd_data['MACD_Signal']
        df['MACD_Histogram'] = macd_data['MACD_Histogram']

        print("✅ Momentum features computed successfully!")
        return df

    def _calculate_rsi_gpu_cupy(self, data: cp.ndarray, period: int) -> cp.ndarray:
        """Calculate RSI with CuPy GPU acceleration."""
        if len(data) < period + 1:
            return cp.full(len(data), cp.nan, dtype=cp.float32)

        # Calculate price changes on GPU
        delta = cp.diff(data)
        gains = cp.where(delta > 0, delta, 0)
        losses = cp.where(delta < 0, -delta, 0)

        # Calculate average gains and losses using GPU rolling mean
        avg_gains = self._gpu_rolling_mean_cupy(gains, period)
        avg_losses = self._gpu_rolling_mean_cupy(losses, period)

        # Calculate RSI on GPU
        rs = avg_gains / (avg_losses + 1e-10)  # Add small epsilon to avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        # Pad result to match input length
        result = cp.full(len(data), cp.nan, dtype=cp.float32)
        result[1:] = rsi

        return result

    def _calculate_roc_gpu_cupy(self, data: cp.ndarray, period: int) -> cp.ndarray:
        """Calculate Rate of Change with CuPy GPU acceleration."""
        if len(data) <= period:
            return cp.full(len(data), cp.nan, dtype=cp.float32)

        result = cp.full(len(data), cp.nan, dtype=cp.float32)

        # Calculate ROC: ((current - previous) / previous) * 100
        result[period:] = ((data[period:] - data[:-period]) / data[:-period]) * 100

        return result

    def _calculate_williams_r_gpu_cupy(self, high: cp.ndarray, low: cp.ndarray, close: cp.ndarray, period: int) -> cp.ndarray:
        """Calculate Williams %R with CuPy GPU acceleration."""
        if len(high) < period:
            return cp.full(len(high), cp.nan, dtype=cp.float32)

        # Calculate rolling highest high and lowest low on GPU
        highest_high = self._gpu_rolling_max_cupy(high, period)
        lowest_low = self._gpu_rolling_min_cupy(low, period)

        # Calculate Williams %R on GPU
        williams_r = ((highest_high - close) / (highest_high - lowest_low + 1e-10)) * -100

        return williams_r

    def _calculate_williams_r_gpu(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Williams %R with GPU acceleration."""
        use_gpu = getattr(self, 'use_gpu', False) and GPU_AVAILABLE
        if not use_gpu or len(high) < period:
            return ta.williams_r(pd.Series(high), pd.Series(low), pd.Series(close), window=period).values

        try:
            # Calculate rolling highest high and lowest low
            highest_high = self._gpu_rolling_max(high, period)
            lowest_low = self._gpu_rolling_min(low, period)

            # Calculate Williams %R
            williams_r = ((highest_high - close) / (highest_high - lowest_low + 1e-10)) * -100

            return williams_r
        except Exception as e:
            print(f"⚠️ GPU Williams %R calculation failed, falling back to CPU: {e}")
            return ta.williams_r(pd.Series(high), pd.Series(low), pd.Series(close), window=period).values
    
    def compute_volume_accumulation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based accumulation indicators."""
        print("📊 Computing volume accumulation features...")
        
        # Simple volume indicators (replacing complex ones)
        # Volume-weighted price change
        df['Volume_Price_Change'] = df['volume'] * (df['close'] - df['open']) / df['open']

        # Volume momentum
        df['Volume_Momentum'] = df['volume'] / df['volume'].rolling(20).mean()

        # Price-Volume correlation
        df['PV_Correlation'] = df['close'].rolling(20).corr(df['volume'])
        
        # Volume SMA Ratio
        df['Volume_SMA_5'] = df.groupby('ticker')['volume'].transform(
            lambda x: x.rolling(5).mean()
        )
        df['Volume_SMA_90'] = df.groupby('ticker')['volume'].transform(
            lambda x: x.rolling(90).mean()
        )
        df['Volume_SMA_Ratio'] = df['Volume_SMA_5'] / (df['Volume_SMA_90'] + 1)  # Avoid division by zero
        
        # Simple volume trend
        df['Volume_Trend'] = df['volume'].rolling(14).mean() / df['volume'].rolling(50).mean()

        # Volume-price relationship
        df['Volume_Price_Ratio'] = df['volume'] / (df['high'] - df['low'] + 0.001)
        
        return df
    
    def compute_support_resistance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute support and resistance level features."""
        print("🎯 Computing support/resistance features...")
        
        # 52-week high/low distances
        df['52W_High'] = df.groupby('ticker')['high'].transform(
            lambda x: x.rolling(252, min_periods=50).max()
        )
        df['52W_Low'] = df.groupby('ticker')['low'].transform(
            lambda x: x.rolling(252, min_periods=50).min()
        )
        df['Distance_From_52W_High'] = (df['close'] - df['52W_High']) / df['52W_High']
        df['Distance_From_52W_Low'] = (df['close'] - df['52W_Low']) / df['52W_Low']
        
        # Local support/resistance (20-day)
        df['Local_Resistance_20'] = df.groupby('ticker')['high'].transform(
            lambda x: x.rolling(20).max()
        )
        df['Local_Support_20'] = df.groupby('ticker')['low'].transform(
            lambda x: x.rolling(20).min()
        )
        df['Distance_From_Resistance'] = (df['close'] - df['Local_Resistance_20']) / df['Local_Resistance_20']
        df['Distance_From_Support'] = (df['close'] - df['Local_Support_20']) / df['Local_Support_20']
        
        # Pivot points
        df['Pivot_Point'] = (df['high'] + df['low'] + df['close']) / 3
        df['Pivot_R1'] = 2 * df['Pivot_Point'] - df['low']
        df['Pivot_S1'] = 2 * df['Pivot_Point'] - df['high']
        
        # Price percentile over 252 days
        df['Price_Percentile_252'] = df.groupby('ticker')['close'].transform(
            lambda x: x.rolling(252, min_periods=50).apply(
                lambda y: (y.iloc[-1] > y).sum() / len(y) * 100
            )
        )
        
        # Advanced S/R using clustering
        df = self._compute_clustered_sr_levels(df)
        
        return df
    
    def compute_beta_correlation_features(self, df: pd.DataFrame,
                                        market_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute beta and correlation features relative to market.

        Args:
            df: Stock dataframe
            market_df: Market benchmark dataframe (e.g., SPY). If None, will try to extract from df.
        """
        print("📊 Computing beta/correlation features...")

        # If market data not provided, try to extract SPY from the dataframe
        if market_df is None:
            if 'SPY' in df['ticker'].values:
                spy_data = df[df['ticker'] == 'SPY'][['date', 'returns']].copy()
                if not spy_data.empty:
                    market_df = spy_data.rename(columns={'returns': 'market_returns'})
                else:
                    print("⚠️  Warning: SPY data found but no returns column. Skipping beta calculations.")
                    return df
            else:
                print("⚠️  Warning: No market data (SPY) found. Skipping beta calculations.")
                return df

        # Ensure market_df has the required column
        if 'market_returns' not in market_df.columns:
            print("⚠️  Warning: market_returns column not found in market data. Skipping beta calculations.")
            return df

        # Create a dictionary for fast market returns lookup to avoid memory-intensive merge
        market_returns_dict = dict(zip(market_df['date'], market_df['market_returns']))

        # Map market returns efficiently
        df['market_returns'] = df['date'].map(market_returns_dict)

        # Only proceed if we have market returns data
        if df['market_returns'].isna().all():
            print("⚠️  Warning: No market returns data available after mapping. Skipping beta calculations.")
            df = df.drop('market_returns', axis=1)
            return df

        # Initialize beta and correlation columns
        df['Beta_60'] = np.nan
        df['Beta_252'] = np.nan
        df['Rolling_Corr_30'] = np.nan

        # Process each ticker separately to manage memory
        unique_tickers = df['ticker'].unique()
        print(f"📊 Processing beta calculations for {len(unique_tickers)} tickers...")

        for i, ticker in enumerate(unique_tickers):
            if i % 1000 == 0:
                print(f"📊 Processed {i}/{len(unique_tickers)} tickers...")

            ticker_mask = df['ticker'] == ticker
            ticker_data = df[ticker_mask].copy()

            if len(ticker_data) < 60:  # Need minimum data for calculations
                continue

            # Calculate rolling betas
            for period in [60, 252]:
                if len(ticker_data) >= period:
                    beta_values = self._calculate_rolling_beta(
                        ticker_data['returns'],
                        ticker_data['market_returns'],
                        period
                    )
                    df.loc[ticker_mask, f'Beta_{period}'] = beta_values

            # Rolling correlation
            if len(ticker_data) >= 30:
                corr_values = ticker_data['returns'].rolling(30).corr(ticker_data['market_returns'])
                df.loc[ticker_mask, 'Rolling_Corr_30'] = corr_values

        # Clean up temporary market_returns column
        df = df.drop('market_returns', axis=1)

        print("✅ Beta/correlation features computed successfully!")
        return df
    
    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum indicators with optimized CPU processing."""
        print("🚀 Computing momentum features with optimized CPU processing...")

        # Use vectorized operations with groupby for maximum efficiency
        df_grouped = df.groupby('ticker')

        # Pre-allocate result columns
        for period in [14, 63]:
            df[f'RSI_{period}'] = np.nan
        df['ROC_20'] = np.nan
        df['Williams_R'] = np.nan

        # Process each ticker efficiently
        for ticker, group in df_grouped:
            if len(group) < 14:  # Skip if not enough data
                continue

            # Get indices for this ticker
            ticker_idx = group.index

            # Extract values once
            close_vals = group['close'].values
            high_vals = group['high'].values
            low_vals = group['low'].values

            # RSI calculations using optimized CPU methods
            for period in [14, 63]:
                if len(group) >= period:
                    rsi_values = self._calculate_rsi_optimized(close_vals, period)
                    df.loc[ticker_idx, f'RSI_{period}'] = rsi_values

            # Rate of Change using vectorized calculation
            if len(group) >= 20:
                roc_values = self._calculate_roc_optimized(close_vals, 20)
                df.loc[ticker_idx, 'ROC_20'] = roc_values

            # Williams %R using optimized calculation
            if len(group) >= 14:
                williams_r_values = self._calculate_williams_r_optimized(high_vals, low_vals, close_vals, 14)
                df.loc[ticker_idx, 'Williams_R'] = williams_r_values

        # MACD using vectorized operations (more efficient than ta library for our use case)
        print("🔄 Computing MACD features...")
        df = self._compute_macd_vectorized(df)

        # Stochastic using optimized calculation
        print("🔄 Computing Stochastic features...")
        df = self._compute_stochastic_vectorized(df)

        print("✅ Momentum features computed successfully!")
        return df

    def _calculate_rsi_optimized(self, data: np.ndarray, period: int) -> np.ndarray:
        """Optimized CPU RSI calculation using vectorized operations."""
        if len(data) < period + 1:
            return np.full(len(data), np.nan)

        # Calculate price changes
        delta = np.diff(data)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Calculate average gains and losses using pandas rolling (optimized)
        avg_gains = self._fast_rolling_mean(gains, period)
        avg_losses = self._fast_rolling_mean(losses, period)

        # Calculate RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Add small epsilon to avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        # Pad result to match input length
        result = np.full(len(data), np.nan)
        result[1:] = rsi

        return result

    def _calculate_roc_optimized(self, data: np.ndarray, period: int) -> np.ndarray:
        """Optimized CPU Rate of Change calculation."""
        if len(data) <= period:
            return np.full(len(data), np.nan)

        result = np.full(len(data), np.nan)
        # Vectorized ROC calculation: ((current - previous) / previous) * 100
        result[period:] = ((data[period:] - data[:-period]) / data[:-period]) * 100

        return result

    def _calculate_williams_r_optimized(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Optimized CPU Williams %R calculation."""
        if len(high) < period:
            return np.full(len(high), np.nan)

        # Calculate rolling highest high and lowest low
        highest_high = self._fast_rolling_max(high, period)
        lowest_low = self._fast_rolling_min(low, period)

        # Calculate Williams %R
        williams_r = ((highest_high - close) / (highest_high - lowest_low + 1e-10)) * -100

        return williams_r

    def _compute_macd_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute MACD using vectorized operations for better performance."""
        df_grouped = df.groupby('ticker')
        df['MACD'] = np.nan
        df['MACD_Signal'] = np.nan
        df['MACD_Histogram'] = np.nan

        for ticker, group in df_grouped:
            if len(group) < 26:  # Need at least 26 periods for MACD
                continue

            ticker_idx = group.index
            close_vals = group['close'].values

            # Calculate EMAs using pandas (optimized)
            ema_12 = pd.Series(close_vals).ewm(span=12).mean().values
            ema_26 = pd.Series(close_vals).ewm(span=26).mean().values

            # MACD line
            macd_line = ema_12 - ema_26

            # Signal line (9-period EMA of MACD)
            signal_line = pd.Series(macd_line).ewm(span=9).mean().values

            # Histogram
            histogram = macd_line - signal_line

            df.loc[ticker_idx, 'MACD'] = macd_line
            df.loc[ticker_idx, 'MACD_Signal'] = signal_line
            df.loc[ticker_idx, 'MACD_Histogram'] = histogram

        return df

    def _compute_stochastic_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Stochastic oscillator using vectorized operations."""
        df_grouped = df.groupby('ticker')
        df['Stoch_K'] = np.nan
        df['Stoch_D'] = np.nan

        for ticker, group in df_grouped:
            if len(group) < 14:  # Need at least 14 periods
                continue

            ticker_idx = group.index
            high_vals = group['high'].values
            low_vals = group['low'].values
            close_vals = group['close'].values

            # Calculate %K
            lowest_low = self._fast_rolling_min(low_vals, 14)
            highest_high = self._fast_rolling_max(high_vals, 14)

            stoch_k = ((close_vals - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100

            # Calculate %D (3-period SMA of %K)
            stoch_d = self._fast_rolling_mean(stoch_k, 3)

            df.loc[ticker_idx, 'Stoch_K'] = stoch_k
            df.loc[ticker_idx, 'Stoch_D'] = stoch_d

        return df

    def _process_ticker_features_parallel(self, ticker_data_tuple):
        """Process features for a single ticker - designed for multiprocessing."""
        ticker, ticker_data = ticker_data_tuple

        if len(ticker_data) < 14:  # Skip if not enough data
            return ticker, None

        # Extract values
        high_vals = ticker_data['high'].values
        low_vals = ticker_data['low'].values
        close_vals = ticker_data['close'].values

        # Initialize results dictionary
        results = {}

        # Trend strength features
        true_range = self._vectorized_true_range(high_vals, low_vals, close_vals)
        for period in [14, 63, 252]:
            if len(ticker_data) >= period:
                results[f'ATR_{period}'] = self._fast_rolling_mean(true_range, period)

        for period in [20, 252]:
            if len(ticker_data) >= period:
                results[f'Linear_Slope_{period}'] = self._calculate_slopes_optimized(close_vals, period)

        if len(ticker_data) >= 25:
            rolling_high = self._fast_rolling_max(high_vals, 25)
            rolling_low = self._fast_rolling_min(low_vals, 25)
            results['High_Low_Ratio_25'] = (rolling_high - rolling_low) / close_vals

        # Momentum features
        for period in [14, 63]:
            if len(ticker_data) >= period:
                results[f'RSI_{period}'] = self._calculate_rsi_optimized(close_vals, period)

        if len(ticker_data) >= 20:
            results['ROC_20'] = self._calculate_roc_optimized(close_vals, 20)

        if len(ticker_data) >= 14:
            results['Williams_R'] = self._calculate_williams_r_optimized(high_vals, low_vals, close_vals, 14)

        return ticker, results

    def compute_features_parallel(self, df: pd.DataFrame, feature_types: List[str] = None) -> pd.DataFrame:
        """Compute features using parallel processing for better performance."""
        if feature_types is None:
            feature_types = ['trend_strength', 'momentum']

        # Determine number of processes to use
        n_processes = self.n_jobs
        if n_processes == -1:
            n_processes = cpu_count()
        elif n_processes <= 0:
            n_processes = 1

        print(f"🚀 Computing features in parallel using {n_processes} cores...")

        # Group data by ticker
        ticker_groups = [(ticker, group) for ticker, group in df.groupby('ticker')]

        # Use multiprocessing for parallel feature computation
        # Only use parallel processing for larger datasets where overhead is justified
        total_records = len(df)
        if n_processes == 1 or len(ticker_groups) < 8 or total_records < 10000:
            print(f"💻 Using serial processing (dataset size: {total_records:,} records)")
            results = [self._process_ticker_features_parallel(group) for group in ticker_groups]
        else:
            actual_processes = min(n_processes, len(ticker_groups))
            print(f"⚡ Using parallel processing with {actual_processes} workers (dataset size: {total_records:,} records)")
            with Pool(processes=actual_processes) as pool:
                results = pool.map(self._process_ticker_features_parallel, ticker_groups)

        # Merge results back into dataframe
        for ticker, ticker_results in results:
            if ticker_results is not None:
                ticker_mask = df['ticker'] == ticker
                ticker_idx = df[ticker_mask].index

                for feature_name, feature_values in ticker_results.items():
                    if feature_name not in df.columns:
                        df[feature_name] = np.nan
                    df.loc[ticker_idx, feature_name] = feature_values

        print("✅ Parallel feature computation completed!")
        return df

    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility indicators."""
        print("💹 Computing volatility features...")
        
        # ATR at multiple horizons
        for period in [14, 63]:
            atr_val = ta.atr(df['high'], df['low'], df['close'], window=period)
            df[f'ATR_{period}'] = atr_val
            df[f'ATR_Ratio_{period}'] = atr_val / df['close']  # Normalized ATR

        # Bollinger Bands
        bb_data = ta.bollinger_bands(df['close'])
        df['BB_Upper'] = bb_data['BB_Upper']
        df['BB_Lower'] = bb_data['BB_Lower']
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['close']
        df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Simple volatility measures (replacing Keltner Channels)
        df['Price_Range'] = (df['high'] - df['low']) / df['close']
        df['Price_Range_MA'] = df['Price_Range'].rolling(20).mean()
        
        # Historical volatility
        for period in [20, 63]:
            df[f'Historical_Vol_{period}'] = df.groupby('ticker')['log_returns'].transform(
                lambda x: x.rolling(period).std() * np.sqrt(252)  # Annualized
            )
        
        return df
    
    def _calculate_slope(self, y: pd.Series) -> float:
        """Calculate linear regression slope."""
        if len(y.dropna()) < 2:
            return np.nan
        x = np.arange(len(y))
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        return slope
    
    def _calculate_rolling_beta(self, returns: pd.Series, market_returns: pd.Series, 
                               window: int) -> pd.Series:
        """Calculate rolling beta."""
        covariance = returns.rolling(window).cov(market_returns)
        market_variance = market_returns.rolling(window).var()
        beta = covariance / market_variance
        return beta
    
    def _compute_clustered_sr_levels(self, df: pd.DataFrame,
                                   lookback: int = 60,
                                   eps_pct: float = 0.02) -> pd.DataFrame:
        """
        Compute support/resistance levels using DBSCAN clustering.

        Args:
            df: Dataframe with price data
            lookback: Number of days to look back for local extrema
            eps_pct: Epsilon parameter as percentage of price for DBSCAN
        """
        print("🔍 Computing advanced S/R levels using clustering...")

        def find_sr_levels(group):
            try:
                if len(group) < lookback:
                    return pd.Series({
                        'nearest_resistance': np.nan,
                        'nearest_support': np.nan,
                        'n_resistance_levels': 0,
                        'n_support_levels': 0,
                        'distance_to_resistance': np.nan,
                        'distance_to_support': np.nan
                    })

                # Find local maxima and minima
                highs = group['high'].values
                lows = group['low'].values

                # Local maxima (resistance)
                local_max_idx = argrelextrema(highs, np.greater, order=5)[0]
                # Local minima (support)
                local_min_idx = argrelextrema(lows, np.less, order=5)[0]

                current_price = group['close'].iloc[-1]

                # Cluster resistance levels
                if len(local_max_idx) > 0:
                    resistance_prices = highs[local_max_idx].reshape(-1, 1)
                    eps = current_price * eps_pct
                    clustering = DBSCAN(eps=eps, min_samples=2).fit(resistance_prices)

                    # Find cluster centers
                    unique_labels = set(clustering.labels_) - {-1}
                    resistance_levels = []
                    for label in unique_labels:
                        cluster_prices = resistance_prices[clustering.labels_ == label]
                        resistance_levels.append(float(cluster_prices.mean()))

                    # Find nearest resistance above current price
                    above_current = [r for r in resistance_levels if r > current_price]
                    nearest_resistance = min(above_current) if above_current else np.nan
                    n_resistance = len(resistance_levels)
                else:
                    nearest_resistance = np.nan
                    n_resistance = 0

                # Cluster support levels
                if len(local_min_idx) > 0:
                    support_prices = lows[local_min_idx].reshape(-1, 1)
                    eps = current_price * eps_pct
                    clustering = DBSCAN(eps=eps, min_samples=2).fit(support_prices)

                    # Find cluster centers
                    unique_labels = set(clustering.labels_) - {-1}
                    support_levels = []
                    for label in unique_labels:
                        cluster_prices = support_prices[clustering.labels_ == label]
                        support_levels.append(float(cluster_prices.mean()))

                    # Find nearest support below current price
                    below_current = [s for s in support_levels if s < current_price]
                    nearest_support = max(below_current) if below_current else np.nan
                    n_support = len(support_levels)
                else:
                    nearest_support = np.nan
                    n_support = 0

                # Ensure all values are properly typed
                return pd.Series({
                    'nearest_resistance': float(nearest_resistance) if not np.isnan(nearest_resistance) else np.nan,
                    'nearest_support': float(nearest_support) if not np.isnan(nearest_support) else np.nan,
                    'distance_to_resistance': float((nearest_resistance - current_price) / current_price) if not np.isnan(nearest_resistance) else np.nan,
                    'distance_to_support': float((current_price - nearest_support) / current_price) if not np.isnan(nearest_support) else np.nan,
                    'n_resistance_levels': int(n_resistance),
                    'n_support_levels': int(n_support)
                })
            except Exception as e:
                # Return safe defaults if computation fails
                print(f"⚠️ S/R computation failed for ticker: {e}")
                return pd.Series({
                    'nearest_resistance': np.nan,
                    'nearest_support': np.nan,
                    'distance_to_resistance': np.nan,
                    'distance_to_support': np.nan,
                    'n_resistance_levels': 0,
                    'n_support_levels': 0
                })

        # Apply to each ticker group
        try:
            sr_features_list = []
            for ticker, group in df.groupby('ticker'):
                sr_result = find_sr_levels(group.tail(lookback))
                sr_result['ticker'] = ticker
                sr_features_list.append(sr_result)

            # Create DataFrame from results
            sr_features = pd.DataFrame(sr_features_list)

            # Merge back to original dataframe
            df = df.merge(sr_features, on='ticker', how='left')

            # Fill any remaining NaN values in the count columns with 0
            df['n_resistance_levels'] = df['n_resistance_levels'].fillna(0).astype(int)
            df['n_support_levels'] = df['n_support_levels'].fillna(0).astype(int)

        except Exception as e:
            print(f"⚠️ S/R clustering failed, using simple defaults: {e}")
            # Add default columns if clustering fails completely
            df['nearest_resistance'] = np.nan
            df['nearest_support'] = np.nan
            df['distance_to_resistance'] = np.nan
            df['distance_to_support'] = np.nan
            df['n_resistance_levels'] = 0
            df['n_support_levels'] = 0

        return df
    
    def compute_all_features(self, df: pd.DataFrame,
                           market_df: Optional[pd.DataFrame] = None,
                           force_recompute: bool = False) -> pd.DataFrame:
        """
        Compute all feature categories with intelligent caching.

        Args:
            df: Raw stock data
            market_df: Market benchmark data (optional)
            force_recompute: Force recomputation even if cache exists

        Returns:
            DataFrame with all features computed
        """
        print("🚀 Starting comprehensive feature engineering with caching...")
        print(f"📊 Input shape: {df.shape}")

        # Generate cache key for the input data
        cache_key = self._generate_cache_key(df, "all_features")

        # Try to load from cache first (unless forced recompute)
        if not force_recompute:
            cached_features = self._load_from_cache(cache_key, "all_features")
            if cached_features is not None:
                # Check if we have new data that needs processing
                new_data = self._identify_new_data(df, cached_features)

                if len(new_data) == 0:
                    print("✅ All data already cached, returning cached features")
                    return cached_features
                elif len(new_data) < len(df) * 0.1:  # Less than 10% new data
                    print(f"📊 Processing {len(new_data)} new records and merging with cache")
                    return self._process_incremental_features(df, cached_features, new_data, market_df)
                else:
                    print(f"📊 Significant new data ({len(new_data)} records), reprocessing all")

        # Process all features (either no cache or significant new data)
        print("🔄 Computing all features from scratch...")

        # Prepare data
        df = self.prepare_data(df)

        # Compute all feature categories with caching for each type
        df = self._compute_features_with_cache(df, "trend_strength",
                                             lambda x: self.compute_trend_strength_features(x))
        df = self._compute_features_with_cache(df, "volume_accumulation",
                                             lambda x: self.compute_volume_accumulation_features(x))
        df = self._compute_features_with_cache(df, "support_resistance",
                                             lambda x: self.compute_support_resistance_features(x))
        df = self._compute_features_with_cache(df, "beta_correlation",
                                             lambda x: self.compute_beta_correlation_features(x, market_df))
        df = self._compute_features_with_cache(df, "momentum",
                                             lambda x: self.compute_momentum_features(x))
        df = self._compute_features_with_cache(df, "volatility",
                                             lambda x: self.compute_volatility_features(x))

        # Add time-based features (these are fast, no need to cache)
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['quarter'] = pd.to_datetime(df['date']).dt.quarter

        # Add rolling statistics for multi-horizon analysis
        for col in ['returns', 'volume']:
            for window in [5, 20, 63]:
                df[f'{col}_mean_{window}'] = df.groupby('ticker')[col].transform(
                    lambda x: x.rolling(window).mean()
                )
                df[f'{col}_std_{window}'] = df.groupby('ticker')[col].transform(
                    lambda x: x.rolling(window).std()
                )

        # Cache the complete result
        self._save_to_cache(df, cache_key, "all_features")

        print(f"✅ Feature engineering complete! Output shape: {df.shape}")
        print(f"📈 Total features created: {len(df.columns) - 7}")  # Subtract original columns

        return df

    def _compute_features_with_cache(self, df: pd.DataFrame, feature_type: str,
                                   compute_func) -> pd.DataFrame:
        """Compute features with individual caching for each feature type."""
        cache_key = self._generate_cache_key(df, feature_type)

        # Try to load from cache
        cached_result = self._load_from_cache(cache_key, feature_type)
        if cached_result is not None:
            # Identify feature columns (exclude base data columns and any existing columns in df)
            base_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
            existing_cols = set(df.columns)

            feature_cols = [col for col in cached_result.columns
                          if col not in base_cols and col not in existing_cols]

            if feature_cols:
                # Merge only the new feature columns
                merge_cols = ['ticker', 'date'] + feature_cols
                df = df.merge(cached_result[merge_cols],
                             on=['ticker', 'date'], how='left')
                print(f"✅ Used cached {feature_type} features ({len(feature_cols)} features)")
            else:
                print(f"⚠️ No new features to merge from {feature_type} cache")
            return df

        # Compute features and cache them
        print(f"🔄 Computing {feature_type} features...")
        result_df = compute_func(df)
        self._save_to_cache(result_df, cache_key, feature_type)
        return result_df

    def _process_incremental_features(self, current_df: pd.DataFrame,
                                    cached_df: pd.DataFrame,
                                    new_data: pd.DataFrame,
                                    market_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Process only new data and merge with cached features."""
        print("🔄 Processing incremental features...")

        # For incremental processing, we need enough historical data for rolling calculations
        # Get additional historical data for context (e.g., last 300 days per ticker)
        extended_new_data = []

        for ticker in new_data['ticker'].unique():
            ticker_cached = cached_df[cached_df['ticker'] == ticker]
            ticker_new = new_data[new_data['ticker'] == ticker]

            if len(ticker_cached) > 0:
                # Get last 300 records from cache for context
                context_data = ticker_cached.tail(300)
                # Combine with new data
                ticker_extended = pd.concat([context_data, ticker_new], ignore_index=True)
                ticker_extended = ticker_extended.drop_duplicates(subset=['ticker', 'date'], keep='last')
            else:
                # New ticker, use all available data
                ticker_extended = ticker_new

            extended_new_data.append(ticker_extended)

        if extended_new_data:
            extended_df = pd.concat(extended_new_data, ignore_index=True)

            # Process features for extended data
            processed_new = self.compute_all_features(extended_df, market_df, force_recompute=True)

            # Extract only the truly new features (not the context data)
            new_features = []
            for ticker in new_data['ticker'].unique():
                ticker_new_dates = new_data[new_data['ticker'] == ticker]['date'].unique()
                ticker_processed = processed_new[
                    (processed_new['ticker'] == ticker) &
                    (processed_new['date'].isin(ticker_new_dates))
                ]
                new_features.append(ticker_processed)

            if new_features:
                new_features_df = pd.concat(new_features, ignore_index=True)

                # Combine cached features with new features
                # Remove any overlapping records from cached data
                non_overlapping_cached = cached_df[
                    ~cached_df.set_index(['ticker', 'date']).index.isin(
                        new_features_df.set_index(['ticker', 'date']).index
                    )
                ]

                # Combine and sort
                result = pd.concat([non_overlapping_cached, new_features_df], ignore_index=True)
                result = result.sort_values(['ticker', 'date']).reset_index(drop=True)

                # Update cache with new complete dataset
                cache_key = self._generate_cache_key(current_df, "all_features")
                self._save_to_cache(result, cache_key, "all_features")

                return result

        # Fallback: recompute everything
        print("⚠️ Incremental processing failed, recomputing all features")
        return self.compute_all_features(current_df, market_df, force_recompute=True)

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get feature names organized by category for multi-stream LSTM.
        
        Returns:
            Dictionary with 'short', 'medium', 'long' feature lists
        """
        # This will be populated after running compute_all_features
        # and analyzing which features correspond to which time horizon
        feature_groups = {
            'short': [],   # Features with windows <= 20 days
            'medium': [],  # Features with windows 21-90 days  
            'long': []     # Features with windows > 90 days
        }
        
        return feature_groups


# Example usage function
def test_feature_engineering():
    """Test the feature engineering with sample data."""
    # Create sample data matching Polygon.io format
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
    tickers = ['AAPL', 'GOOGL', 'SPY']
    
    data = []
    for ticker in tickers:
        for date in dates:
            if date.weekday() < 5:  # Only weekdays
                base_price = 100 + np.random.randn() * 10
                data.append({
                    'ticker': ticker,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': base_price + np.random.randn(),
                    'high': base_price + abs(np.random.randn()) * 2,
                    'low': base_price - abs(np.random.randn()) * 2,
                    'close': base_price + np.random.randn() * 0.5,
                    'volume': int(1000000 + np.random.randn() * 100000)
                })
    
    df = pd.DataFrame(data)
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    # Compute all features
    df_with_features = feature_engineer.compute_all_features(df)
    
    # Display results
    print("\n📊 Feature Engineering Results:")
    print(f"Original columns: {len(df.columns)}")
    print(f"Total columns after feature engineering: {len(df_with_features.columns)}")
    print(f"\nSample features created:")
    
    feature_cols = [col for col in df_with_features.columns if col not in df.columns]
    for i, col in enumerate(feature_cols[:10]):
        print(f"  - {col}")
    print(f"  ... and {len(feature_cols) - 10} more features")
    
    return df_with_features


if __name__ == "__main__":
    # Run test
    result_df = test_feature_engineering()
