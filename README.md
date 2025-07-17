# Weekly Stock Analysis System

A complete machine learning-based stock analysis system that provides weekly buy/sell recommendations with high accuracy predictions.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Weekly Analysis
```bash
python weekly_analysis.py
```

The system will automatically:
- Load pre-trained models
- Use sample data (or your own data if provided)
- Generate weekly stock recommendations
- Save results to `predictions/` directory

## 📊 System Overview

This system uses advanced machine learning to analyze stocks and provide trading recommendations:

- **Accuracy**: 77.1% prediction accuracy
- **AUC Score**: 0.813 (Very Good)
- **Precision**: 82.6% (High precision)
- **Market Regime Detection**: Automatically detects market conditions
- **Risk Management**: Kelly criterion position sizing

## 🔧 System Components

### Core Files
- `weekly_analysis.py` - Main analysis script
- `main_trading_system.py` - Core prediction system
- `advanced_feature_engineering.py` - Feature engineering pipeline
- `regime_detector.py` - Market regime detection
- `stacked_ensemble.py` - Ensemble prediction models
- `multi_stream_lstm.py` - LSTM neural network
- `simple_ta.py` - Technical analysis indicators
- `download_fresh_data.py` - Fresh data downloader (API included)
- `simple_2year_data_downloader.py` - Core download functions

### Models Directory
- `models/final_models_20250713_185556/` - Pre-trained models
  - `ensemble.pkl` - Ensemble model
  - `regime_detector.pkl` - Market regime detector
  - `lstm_model.pth` - LSTM neural network
  - `feature_engineer.pkl` - Feature engineering pipeline
  - `metrics.json` - Model performance metrics

### Data Directory
- `data/sample_stock_data.csv` - Sample data (10 popular stocks)
- `data/` - Place your own stock data here

## 📈 Getting Fresh Data

### Option 1: Download Fresh Data (Recommended)
**🔑 API credentials included - no setup required!**

```bash
python download_fresh_data.py
```

This will:
- Download latest 2 years of data (10,000+ stocks)
- Update to current date automatically
- Backup existing data before updating
- Take 10-20 minutes depending on connection

### Option 2: Check Current Data Status
```bash
python download_fresh_data.py check
```

### Option 3: Use Your Own Data

1. **Data Format Required:**
   ```csv
   ticker,date,open,high,low,close,volume
   AAPL,2023-01-01,150.00,152.00,149.00,151.00,1000000
   ```

2. **Place your data file:**
   - Save as `data/stock_data_2year.csv`
   - System will automatically use it instead of sample data

3. **Data Requirements:**
   - At least 200 trading days per ticker for reliable predictions
   - Recent data (within last 2 years recommended)
   - Standard OHLCV format

## 🎯 Understanding Results

### Output Format
The system generates:
1. **Top 10 BUY Recommendations** - Highest probability picks
2. **Detailed Analysis** - Price, probability, confidence, regime
3. **Overall Top Performers** - All signals ranked
4. **Summary Statistics** - Total analyzed, buy/sell signals
5. **CSV Export** - Full results in `predictions/` directory

### Key Metrics
- **Prediction Probability**: 0.0-1.0 (higher = more likely to go up)
- **Model Confidence**: 0.0-1.0 (higher = more reliable prediction)
- **Market Regime**: Current market condition (bull/bear/sideways + volatility)
- **Position Size**: Recommended position size (Kelly criterion)

### Market Regimes
- `bull_low_vol` - Rising market, low volatility
- `bull_high_vol` - Rising market, high volatility  
- `bear_low_vol` - Falling market, low volatility
- `bear_high_vol` - Falling market, high volatility
- `sideways_low_vol` - Flat market, low volatility
- `sideways_high_vol` - Flat market, high volatility

## ⚙️ Configuration

### GPU Support
- System automatically detects GPU availability
- Uses CPU by default for optimal performance
- To force GPU usage, modify `use_gpu=True` in `weekly_analysis.py`

### Memory Optimization
- Automatic data type optimization
- Efficient caching system
- Handles large datasets (tested with 10,000+ stocks)

### Caching
- Feature engineering results are cached for speed
- Cache stored in `cache/features/` directory
- Automatically invalidated when data changes

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **No Data Found**
   - Ensure `data/sample_stock_data.csv` exists
   - Or provide your own `data/stock_data_2year.csv`

3. **Memory Issues**
   - Reduce dataset size
   - Close other applications
   - Use 64-bit Python

4. **Slow Performance**
   - System uses caching - first run is slower
   - Subsequent runs are much faster
   - Consider reducing number of stocks

### Performance Tips
- First run takes longer (building cache)
- Subsequent runs are 5-10x faster
- Use SSD storage for better I/O performance
- 8GB+ RAM recommended for large datasets

## 📋 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB free disk space

### Recommended
- Python 3.9+
- 8GB+ RAM
- SSD storage
- Multi-core CPU

### Optional
- NVIDIA GPU (CUDA support)
- 16GB+ RAM for very large datasets

## 🚨 Important Notes

### Risk Disclaimer
- This system is for educational/research purposes
- Past performance does not guarantee future results
- Always do your own research before trading
- Consider position sizing and risk management
- Markets can be unpredictable

### Data Privacy
- No personal data is collected or transmitted
- All processing happens locally
- No internet connection required for analysis
- Models are pre-trained and included

## 📞 Support

For technical issues:
1. Check this README first
2. Verify all dependencies are installed
3. Ensure data format is correct
4. Check system requirements

## 🔄 Updates

This system includes:
- Pre-trained models (July 2025)
- Sample data through July 2025
- Latest feature engineering pipeline
- Optimized performance settings

For newer models or data, you would need to retrain the system with fresh data.

---

**Happy Trading! 📈**
