import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class DataDrivenRegimeDetector:
    """
    Advanced market regime detection using Gaussian Mixture Models.
    Identifies 6 distinct market regimes based on historical patterns.
    """
    
    def __init__(self, 
                 n_regimes: int = 6,
                 persistence_window: int = 3,
                 confidence_threshold: float = 0.6):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of market regimes to identify
            persistence_window: Days to average for regime stability
            confidence_threshold: Minimum confidence for regime assignment
        """
        self.n_regimes = n_regimes
        self.persistence_window = persistence_window
        self.confidence_threshold = confidence_threshold
        
        self.regime_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.regime_characteristics = None
        
        # Regime naming based on characteristics (will be updated after fitting)
        self.regime_names = {
            0: 'bull_low_vol',
            1: 'bull_high_vol', 
            2: 'bear_low_vol',
            3: 'bear_high_vol',
            4: 'sideways_low_vol',
            5: 'sideways_high_vol'
        }
        
        # Color scheme for visualization
        self.regime_colors = {
            'bull_low_vol': '#2E7D32',      # Dark green
            'bull_high_vol': '#81C784',     # Light green
            'bear_low_vol': '#D32F2F',      # Dark red
            'bear_high_vol': '#EF5350',     # Light red
            'sideways_low_vol': '#1976D2',  # Dark blue
            'sideways_high_vol': '#64B5F6'  # Light blue
        }
    
    def extract_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract features for regime detection from market data.
        
        Args:
            market_data: DataFrame with market index data (e.g., SPY)
            
        Returns:
            Feature array for regime detection
        """
        # Ensure we have required columns
        required_cols = ['close', 'high', 'low', 'volume']
        if not all(col in market_data.columns for col in required_cols):
            raise ValueError(f"Market data must contain columns: {required_cols}")
        
        # Calculate returns
        market_data['returns'] = market_data['close'].pct_change()
        market_data['log_returns'] = np.log(market_data['close'] / market_data['close'].shift(1))
        
        features = pd.DataFrame(index=market_data.index)
        
        # Volatility features at multiple scales
        for window in [5, 10, 20, 60]:
            features[f'volatility_{window}'] = market_data['returns'].rolling(window).std() * np.sqrt(252)
        
        # Volatility ratios (regime change indicators)
        features['vol_ratio_5_20'] = features['volatility_5'] / features['volatility_20']
        features['vol_ratio_10_60'] = features['volatility_10'] / features['volatility_60']
        
        # Trend strength features (adapt windows based on available data)
        max_data_len = len(market_data)
        # Use smaller windows for limited data to ensure we get valid features
        if max_data_len >= 200:
            trend_windows = [20, 50, 200]
        elif max_data_len >= 100:
            trend_windows = [10, 20, 50]
        else:
            trend_windows = [5, 10, 20]

        for window in trend_windows:
            if max_data_len >= window + 5:  # Reduced buffer requirement
                features[f'sma_{window}'] = market_data['close'].rolling(window).mean()
                features[f'trend_strength_{window}'] = (
                    (market_data['close'] - features[f'sma_{window}']) / features[f'sma_{window}']
                )
        
        # Momentum features (adapt to available data)
        momentum_windows = [5, 20, 60] if max_data_len >= 80 else [5, 10, 20]
        for window in momentum_windows:
            if max_data_len >= window + 5:
                features[f'return_{window}d'] = market_data['returns'].rolling(window).sum()
                features[f'momentum_{window}'] = market_data['close'].pct_change(window)
        
        # Market breadth proxy (using volume patterns)
        features['volume_ratio'] = (
            market_data['volume'].rolling(5).mean() / 
            market_data['volume'].rolling(20).mean()
        )
        
        # Price range features (market stress indicators)
        features['high_low_ratio'] = (
            (market_data['high'] - market_data['low']) / market_data['close']
        ).rolling(10).mean()
        
        # Skewness and kurtosis (distribution shape)
        features['skew_20'] = market_data['returns'].rolling(20).skew()
        features['kurt_20'] = market_data['returns'].rolling(20).kurt()
        
        # Drawdown features
        rolling_max = market_data['close'].expanding().max()
        features['drawdown'] = (market_data['close'] - rolling_max) / rolling_max
        features['drawdown_duration'] = features['drawdown'].groupby(
            (features['drawdown'] == 0).cumsum()
        ).cumcount()
        
        # Store feature names for later use
        self.feature_names = features.columns.tolist()

        # More robust handling of NaN values
        # First, forward fill and backward fill
        features_filled = features.fillna(method='ffill').fillna(method='bfill')

        # For any remaining NaNs, fill with reasonable defaults
        for col in features_filled.columns:
            if features_filled[col].isna().any():
                if 'volatility' in col:
                    # Use a reasonable default volatility (e.g., 20% annualized)
                    features_filled[col] = features_filled[col].fillna(0.2)
                elif 'ratio' in col:
                    # Use 1.0 for ratios
                    features_filled[col] = features_filled[col].fillna(1.0)
                elif 'return' in col or 'momentum' in col or 'trend_strength' in col:
                    # Use 0 for returns and momentum
                    features_filled[col] = features_filled[col].fillna(0.0)
                elif 'drawdown' in col:
                    # Use 0 for drawdown
                    features_filled[col] = features_filled[col].fillna(0.0)
                elif 'skew' in col:
                    # Use 0 for skewness (normal distribution)
                    features_filled[col] = features_filled[col].fillna(0.0)
                elif 'kurt' in col:
                    # Use 3 for kurtosis (normal distribution)
                    features_filled[col] = features_filled[col].fillna(3.0)
                else:
                    # Default to column mean or 0
                    mean_val = features_filled[col].mean()
                    features_filled[col] = features_filled[col].fillna(mean_val if not pd.isna(mean_val) else 0.0)

        # Final check - should have no NaNs now
        features_clean = features_filled.dropna()

        # If we still have no data (shouldn't happen), create minimal synthetic data
        if len(features_clean) == 0:
            print("⚠️ Creating synthetic regime features due to insufficient data")
            # Create one row of reasonable default values
            default_row = {}
            for col in features.columns:
                if 'volatility' in col:
                    default_row[col] = 0.2  # 20% volatility
                elif 'ratio' in col:
                    default_row[col] = 1.0
                elif 'sma' in col:
                    default_row[col] = market_data['close'].iloc[-1] if len(market_data) > 0 else 100.0
                else:
                    default_row[col] = 0.0

            features_clean = pd.DataFrame([default_row])

        return features_clean.values
    
    def fit_regime_model(self, market_data: pd.DataFrame, 
                        verbose: bool = True) -> 'DataDrivenRegimeDetector':
        """
        Learn market regimes from historical data.
        
        Args:
            market_data: Historical market index data
            verbose: Whether to print fitting progress
            
        Returns:
            Fitted regime detector
        """
        if verbose:
            print("🔍 Fitting regime detection model...")
        
        # Extract features
        features = self.extract_regime_features(market_data)
        
        if verbose:
            print(f"📊 Extracted {features.shape[1]} features from {features.shape[0]} samples")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit Gaussian Mixture Model
        self.regime_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            n_init=10,
            random_state=42,
            verbose=verbose
        )
        
        self.regime_model.fit(features_scaled)
        
        # Analyze regime characteristics
        self._analyze_regime_characteristics(features, features_scaled)
        
        # Validate and potentially relabel regimes
        self._validate_regime_labels(features, market_data)
        
        if verbose:
            print("✅ Regime model fitted successfully!")
            self._print_regime_summary()
        
        return self
    
    def _analyze_regime_characteristics(self, features: np.ndarray, 
                                      features_scaled: np.ndarray) -> None:
        """Analyze the characteristics of each regime."""
        # Get regime assignments
        regime_labels = self.regime_model.predict(features_scaled)
        
        # Create DataFrame for analysis
        feature_df = pd.DataFrame(features, columns=self.feature_names)
        feature_df['regime'] = regime_labels
        
        # Calculate regime characteristics
        self.regime_characteristics = {}
        
        for regime in range(self.n_regimes):
            regime_data = feature_df[feature_df['regime'] == regime]
            
            characteristics = {
                'n_samples': len(regime_data),
                'pct_samples': len(regime_data) / len(feature_df) * 100,
                'avg_return_20d': regime_data['return_20d'].mean(),
                'avg_volatility_20': regime_data['volatility_20'].mean(),
                'avg_trend_strength': self._get_avg_trend_strength(regime_data),
                'avg_drawdown': regime_data['drawdown'].mean(),
                'feature_means': regime_data.drop('regime', axis=1).mean().to_dict()
            }
            
            self.regime_characteristics[regime] = characteristics
    
    def _validate_regime_labels(self, features: np.ndarray, 
                               market_data: pd.DataFrame) -> None:
        """
        Validate and relabel regimes based on their characteristics.
        """
        # Sort regimes by return and volatility characteristics
        regime_summary = []
        
        for regime_id, chars in self.regime_characteristics.items():
            regime_summary.append({
                'regime_id': regime_id,
                'avg_return': chars['avg_return_20d'],
                'avg_volatility': chars['avg_volatility_20'],
                'trend_strength': chars['avg_trend_strength'],
                'n_samples': chars['n_samples']
            })
        
        regime_df = pd.DataFrame(regime_summary)
        
        # Categorize regimes
        new_regime_names = {}
        
        # Sort by returns to identify bull/bear/sideways
        regime_df['return_rank'] = regime_df['avg_return'].rank()
        regime_df['vol_rank'] = regime_df['avg_volatility'].rank()
        
        for _, row in regime_df.iterrows():
            regime_id = row['regime_id']
            
            # Determine market direction
            if row['return_rank'] > 4:  # Top 2 regimes by return
                direction = 'bull'
            elif row['return_rank'] < 3:  # Bottom 2 regimes by return
                direction = 'bear'
            else:  # Middle 2 regimes
                direction = 'sideways'
            
            # Determine volatility level
            if row['vol_rank'] <= 3:
                vol_level = 'low_vol'
            else:
                vol_level = 'high_vol'
            
            new_regime_names[regime_id] = f"{direction}_{vol_level}"
        
        self.regime_names = new_regime_names
    
    def detect_current_regime(self, current_market_data: pd.DataFrame) -> Tuple[str, float, Dict]:
        """
        Detect current market regime with confidence score.

        Args:
            current_market_data: Recent market data (should include at least 60 days)

        Returns:
            Tuple of (regime_name, confidence, regime_probabilities)
        """
        if self.regime_model is None:
            raise ValueError("Model must be fitted before detecting regimes!")

        # Extract features
        features = self.extract_regime_features(current_market_data)

        if len(features) == 0:
            # Return default regime if insufficient data
            print("⚠️  Warning: Insufficient data for regime detection, using default regime")
            return "unknown", 0.0, {
                'regime_probabilities': {},
                'is_transition': False,
                'secondary_regime': None,
                'transition_probability': 0.0
            }
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get regime probabilities for recent period
        if len(features_scaled) >= self.persistence_window:
            # Use persistence window for stability
            recent_features = features_scaled[-self.persistence_window:]
            regime_probs = self.regime_model.predict_proba(recent_features).mean(axis=0)
        else:
            # Use all available data
            regime_probs = self.regime_model.predict_proba(features_scaled).mean(axis=0)
        
        # Get most likely regime
        regime_id = np.argmax(regime_probs)
        confidence = regime_probs[regime_id]
        regime_name = self.regime_names[regime_id]
        
        # Create probability dictionary
        regime_prob_dict = {
            self.regime_names[i]: prob 
            for i, prob in enumerate(regime_probs)
        }
        
        # Add additional context
        context = {
            'regime_probabilities': regime_prob_dict,
            'confidence': confidence,
            'is_transition': confidence < self.confidence_threshold,
            'secondary_regime': self.regime_names[np.argsort(regime_probs)[-2]] if confidence < 0.8 else None,
            'current_volatility': features[-1, self.feature_names.index('volatility_20')] if len(features) > 0 else None,
            'current_trend': self._get_trend_strength_value(features) if len(features) > 0 else None
        }
        
        return regime_name, confidence, context

    def _get_trend_strength_value(self, features):
        """Get the best available trend strength value from features."""
        # Try to get the longest-term trend strength available
        for trend_col in ['trend_strength_20', 'trend_strength_10', 'trend_strength_5']:
            if trend_col in self.feature_names:
                return features[-1, self.feature_names.index(trend_col)]
        return None

    def _get_avg_trend_strength(self, regime_data):
        """Get average trend strength using the best available column."""
        # Try to get the longest-term trend strength available
        for trend_col in ['trend_strength_20', 'trend_strength_10', 'trend_strength_5']:
            if trend_col in regime_data.columns:
                return regime_data[trend_col].mean()
        return 0.0

    def get_regime_history(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Get historical regime assignments for the given data.
        
        Args:
            market_data: Historical market data
            
        Returns:
            DataFrame with dates and regime assignments
        """
        # Extract features
        features = self.extract_regime_features(market_data)
        features_scaled = self.scaler.transform(features)
        
        # Get regime predictions
        regime_ids = self.regime_model.predict(features_scaled)
        regime_probs = self.regime_model.predict_proba(features_scaled)
        
        # Create results dataframe
        valid_dates = market_data.index[len(market_data) - len(features):]
        
        results = pd.DataFrame({
            'date': valid_dates,
            'regime_id': regime_ids,
            'regime_name': [self.regime_names[rid] for rid in regime_ids],
            'confidence': regime_probs.max(axis=1)
        })
        
        # Add regime probabilities
        for i in range(self.n_regimes):
            results[f'prob_{self.regime_names[i]}'] = regime_probs[:, i]
        
        return results
    
    def plot_regime_analysis(self, market_data: pd.DataFrame, 
                           save_path: Optional[str] = None) -> None:
        """
        Create comprehensive regime analysis plots.
        
        Args:
            market_data: Market data with price information
            save_path: Path to save the plot (optional)
        """
        # Get regime history
        regime_history = self.get_regime_history(market_data)
        
        # Align data
        market_data_aligned = market_data.iloc[len(market_data) - len(regime_history):]
        
        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        
        # 1. Price chart with regime coloring
        ax1 = axes[0]
        for regime_name in self.regime_names.values():
            regime_mask = regime_history['regime_name'] == regime_name
            if regime_mask.any():
                regime_dates = regime_history[regime_mask]['date']
                regime_prices = market_data_aligned.loc[regime_mask, 'close']
                ax1.scatter(regime_dates, regime_prices, 
                          c=self.regime_colors[regime_name],
                          label=regime_name, alpha=0.7, s=10)
        
        ax1.plot(regime_history['date'], market_data_aligned['close'], 
                'k-', linewidth=0.5, alpha=0.5)
        ax1.set_ylabel('Price')
        ax1.set_title('Market Price with Regime Classifications')
        ax1.legend(loc='best', ncol=3)
        ax1.grid(True, alpha=0.3)
        
        # 2. Regime probabilities
        ax2 = axes[1]
        bottom = np.zeros(len(regime_history))
        for regime_name in self.regime_names.values():
            prob_col = f'prob_{regime_name}'
            ax2.fill_between(regime_history['date'], bottom, 
                           bottom + regime_history[prob_col],
                           color=self.regime_colors[regime_name],
                           label=regime_name, alpha=0.8)
            bottom += regime_history[prob_col]
        
        ax2.set_ylabel('Regime Probability')
        ax2.set_title('Regime Probabilities Over Time')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 3. Confidence scores
        ax3 = axes[2]
        ax3.plot(regime_history['date'], regime_history['confidence'], 
                'b-', linewidth=1.5)
        ax3.axhline(y=self.confidence_threshold, color='r', 
                   linestyle='--', label=f'Threshold ({self.confidence_threshold})')
        ax3.fill_between(regime_history['date'], 0, regime_history['confidence'],
                        where=regime_history['confidence'] < self.confidence_threshold,
                        color='red', alpha=0.2, label='Low Confidence')
        ax3.set_ylabel('Confidence')
        ax3.set_title('Regime Detection Confidence')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Regime duration analysis
        ax4 = axes[3]
        regime_changes = regime_history['regime_name'].ne(regime_history['regime_name'].shift())
        regime_groups = regime_changes.cumsum()
        regime_durations = regime_history.groupby(regime_groups).agg({
            'date': ['first', 'last'],
            'regime_name': 'first'
        })
        
        for _, regime_period in regime_durations.iterrows():
            start_date = regime_period[('date', 'first')]
            end_date = regime_period[('date', 'last')]
            regime_name = regime_period[('regime_name', 'first')]
            
            ax4.barh(0, (end_date - start_date).days, 
                    left=start_date, height=0.8,
                    color=self.regime_colors[regime_name],
                    edgecolor='black', linewidth=0.5)
        
        ax4.set_ylim(-0.5, 0.5)
        ax4.set_yticks([])
        ax4.set_xlabel('Date')
        ax4.set_title('Regime Duration Timeline')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _print_regime_summary(self) -> None:
        """Print summary of detected regimes."""
        print("\n📊 Regime Characteristics Summary:")
        print("=" * 80)
        
        for regime_id, regime_name in self.regime_names.items():
            chars = self.regime_characteristics[regime_id]
            print(f"\n🎯 Regime {regime_id}: {regime_name}")
            print(f"   - Samples: {chars['n_samples']} ({chars['pct_samples']:.1f}%)")
            print(f"   - Avg 20-day return: {chars['avg_return_20d']*100:.2f}%")
            print(f"   - Avg volatility (annual): {chars['avg_volatility_20']*100:.1f}%")
            print(f"   - Avg trend strength: {chars['avg_trend_strength']*100:.2f}%")
            print(f"   - Avg drawdown: {chars['avg_drawdown']*100:.2f}%")
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted regime model."""
        model_dict = {
            'regime_model': self.regime_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'regime_names': self.regime_names,
            'regime_characteristics': self.regime_characteristics,
            'n_regimes': self.n_regimes,
            'persistence_window': self.persistence_window,
            'confidence_threshold': self.confidence_threshold
        }
        joblib.dump(model_dict, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a previously fitted regime model."""
        model_dict = joblib.load(filepath)
        self.regime_model = model_dict['regime_model']
        self.scaler = model_dict['scaler']
        self.feature_names = model_dict['feature_names']
        self.regime_names = model_dict['regime_names']
        self.regime_characteristics = model_dict['regime_characteristics']
        self.n_regimes = model_dict['n_regimes']
        self.persistence_window = model_dict['persistence_window']
        self.confidence_threshold = model_dict['confidence_threshold']
        print(f"✅ Model loaded from {filepath}")


# Example usage and testing
def test_regime_detector():
    """Test the regime detector with synthetic market data."""
    # Create synthetic market data with different regimes
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    # Simulate different market regimes
    n_days = len(dates)
    prices = np.zeros(n_days)
    volumes = np.zeros(n_days)
    
    # Starting price
    prices[0] = 100
    
    # Create regime periods
    regime_periods = [
        (0, 200, 'bull_low_vol', 0.0005, 0.01),      # Steady bull
        (200, 400, 'bull_high_vol', 0.0003, 0.025),   # Volatile bull
        (400, 600, 'bear_high_vol', -0.0004, 0.03),   # Crash
        (600, 800, 'sideways_low_vol', 0, 0.008),     # Recovery/sideways
        (800, 1000, 'bull_low_vol', 0.0004, 0.012),   # New bull
        (1000, n_days, 'bear_low_vol', -0.0002, 0.015) # Slow bear
    ]
    
    for start, end, regime, drift, vol in regime_periods:
        for i in range(start, min(end, n_days-1)):
            returns = np.random.normal(drift, vol)
            prices[i+1] = prices[i] * (1 + returns)
            volumes[i] = np.random.lognormal(14, 0.5)  # Log-normal volume
    
    # Add high/low based on daily volatility
    daily_range = np.abs(np.random.normal(0, 0.02, n_days))
    highs = prices * (1 + daily_range/2)
    lows = prices * (1 - daily_range/2)
    
    # Create market dataframe
    market_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    # Remove weekends
    market_data = market_data[market_data['date'].dt.dayofweek < 5]
    market_data.set_index('date', inplace=True)
    
    print("🎯 Testing Regime Detector with Synthetic Data")
    print("=" * 50)
    
    # Initialize and fit regime detector
    detector = DataDrivenRegimeDetector(n_regimes=6)
    detector.fit_regime_model(market_data)
    
    # Detect current regime (using last 60 days)
    recent_data = market_data.tail(60)
    regime_name, confidence, context = detector.detect_current_regime(recent_data)
    
    print(f"\n📍 Current Regime: {regime_name}")
    print(f"📊 Confidence: {confidence:.2%}")
    print(f"🔄 Is Transition: {context['is_transition']}")
    
    if context['secondary_regime']:
        print(f"🔄 Secondary Regime: {context['secondary_regime']}")
    
    print("\n📈 All Regime Probabilities:")
    for regime, prob in context['regime_probabilities'].items():
        print(f"   {regime}: {prob:.2%}")
    
    # Plot analysis
    detector.plot_regime_analysis(market_data)
    
    return detector, market_data


if __name__ == "__main__":
    # Run test
    detector, market_data = test_regime_detector()
