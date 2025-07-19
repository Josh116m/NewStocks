# Feature Engineering Caching System

This document describes the intelligent caching system implemented for feature engineering to dramatically reduce processing time and improve efficiency.

## Overview

The caching system provides:
- **Intelligent caching** of computed features to avoid reprocessing
- **Incremental updates** for new data without full recomputation
- **GPU-optimized processing** with caching for trend strength features
- **Cache management tools** for monitoring and maintenance
- **Automatic cache validation** and integrity checking

## Key Benefits

- **10-100x speedup** for repeated feature engineering runs
- **Incremental processing** for new data (only process what's new)
- **Persistent storage** of expensive computations
- **Memory efficient** with selective loading
- **GPU acceleration** combined with intelligent caching

## How It Works

### 1. Cache Key Generation
The system generates unique cache keys based on:
- Data content hash (sampled for large datasets)
- Feature engineering parameters (windows, settings)
- Feature type (trend_strength, momentum, etc.)

### 2. Intelligent Cache Loading
- Checks for existing cached features
- Identifies new data that needs processing
- Decides between full recomputation vs incremental updates

### 3. Incremental Processing
- For small amounts of new data (<10% of total), processes only new records
- Maintains sufficient historical context for rolling calculations
- Merges new features with cached results

### 4. Feature-Specific Caching
- Caches different feature types separately
- Allows partial cache hits (e.g., trend features cached, momentum needs computation)
- GPU-optimized caching for computationally expensive features

## Usage

### Basic Usage with Caching

```python
from advanced_feature_engineering import AdvancedFeatureEngineer

# Initialize with caching enabled (default)
fe = AdvancedFeatureEngineer(
    use_gpu=True,           # Enable GPU acceleration
    enable_cache=True,      # Enable caching (default)
    cache_dir="cache/features"  # Cache directory
)

# First run - computes and caches all features
df_with_features = fe.compute_all_features(stock_data)

# Second run - loads from cache (much faster)
df_with_features = fe.compute_all_features(stock_data)

# Force recomputation (ignore cache)
df_with_features = fe.compute_all_features(stock_data, force_recompute=True)
```

### Trend Strength Feature Caching

```python
# Specifically cache trend strength features (most expensive)
df_with_trend = fe.compute_trend_strength_features_cached(stock_data)

# Force recomputation of trend features
df_with_trend = fe.compute_trend_strength_features_cached(stock_data, force_recompute=True)
```

### Cache Management

```python
# Clear all cached features
fe.clear_cache()

# Clear specific feature type
fe.clear_cache(feature_type="trend_strength")

# Check cache status
cache_info = fe._get_cache_info()
```

## Cache Management Tools

### Command Line Cache Manager

```bash
# View cache information
python cache_manager.py info

# Clear all cache
python cache_manager.py clear

# Clear specific feature type
python cache_manager.py clear --feature-type trend_strength

# Validate cache integrity
python cache_manager.py validate

# Optimize cache (remove duplicates/outdated)
python cache_manager.py optimize
```

### Testing Cache Performance

```bash
# Run comprehensive caching tests
python test_feature_caching.py
```

## Cache Directory Structure

```
cache/features/
├── features_all_features_abc123.pkl      # Complete feature set
├── features_trend_strength_def456.pkl    # Trend strength features
├── features_momentum_ghi789.pkl          # Momentum features
├── features_volatility_jkl012.pkl        # Volatility features
└── ...
```

## Cache File Format

Each cache file contains:
```python
{
    'data': pd.DataFrame,           # Processed features
    'timestamp': datetime,          # When cached
    'feature_type': str,           # Type of features
    'parameters': {                # Feature engineering parameters
        'short_window': 14,
        'medium_window': 63,
        'long_window': 252
    }
}
```

## Performance Characteristics

### Typical Speedups
- **Full cache hit**: 50-100x faster
- **Incremental update**: 10-20x faster
- **Partial cache hit**: 2-5x faster

### Memory Usage
- Cache files: 1-50 MB per feature type
- Memory overhead: Minimal (lazy loading)
- Disk space: Scales with data size and feature complexity

## Configuration Options

### AdvancedFeatureEngineer Parameters

```python
fe = AdvancedFeatureEngineer(
    short_window=14,           # Short-term window (days)
    medium_window=63,          # Medium-term window (days)  
    long_window=252,           # Long-term window (days)
    use_gpu=True,              # Enable GPU acceleration
    cache_dir="cache/features", # Cache directory
    enable_cache=True          # Enable/disable caching
)
```

### Cache Behavior Controls

- `force_recompute=True`: Ignore cache and recompute
- `enable_cache=False`: Disable caching entirely
- Cache invalidation: Automatic when parameters change

## Best Practices

### 1. Cache Management
- Monitor cache size regularly
- Use `cache_manager.py optimize` to remove outdated files
- Validate cache integrity after system updates

### 2. Development Workflow
- Use `force_recompute=True` when testing feature changes
- Clear cache when changing feature engineering logic
- Keep cache enabled for production runs

### 3. Performance Optimization
- Enable GPU acceleration for maximum benefit
- Use incremental processing for daily updates
- Monitor cache hit rates in logs

### 4. Troubleshooting
- Check cache validation if getting unexpected results
- Clear cache if encountering version compatibility issues
- Monitor disk space usage in cache directory

## Integration with Weekly Analysis

The caching system is automatically integrated with `weekly_analysis.py`:

```python
# Automatically uses caching when available
predictor = StockTradingPredictor(use_gpu=True)
predictor.feature_engineer = AdvancedFeatureEngineer(use_gpu=True, enable_cache=True)

# Features are cached and reused across runs
predictions = predictor.predict_next_day(stock_data)
```

## Monitoring and Debugging

### Cache Hit Logging
The system logs cache operations:
```
💾 Feature caching enabled: cache/features
✅ Loaded trend_strength features from cache: features_trend_strength_abc123.pkl
📊 Identified 50 new records to process
🔄 Processing incremental trend strength features...
💾 Cached trend_strength features: features_trend_strength_def456.pkl
```

### Performance Metrics
- Cache hit rate
- Processing time comparisons
- Memory usage statistics
- Disk space utilization

## Limitations and Considerations

1. **Cache Invalidation**: Cache is invalidated when feature engineering parameters change
2. **Disk Space**: Large datasets can generate substantial cache files
3. **Version Compatibility**: Cache format may change between versions
4. **Incremental Limits**: Very large data updates may trigger full recomputation

## Future Enhancements

- Distributed caching for multi-machine setups
- Compression for cache files
- TTL (time-to-live) for automatic cache expiration
- Cache warming strategies
- Advanced cache analytics and monitoring
