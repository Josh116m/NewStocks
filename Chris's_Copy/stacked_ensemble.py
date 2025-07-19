import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import joblib
import torch
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelPerformanceTracker:
    """Track model performance over time for adaptive weighting."""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.performance_history = defaultdict(list)
        self.current_weights = {}
        
    def update(self, model_name: str, predictions: np.ndarray, 
               actuals: np.ndarray, metric: str = 'accuracy'):
        """Update performance metrics for a model."""
        if metric == 'accuracy':
            score = accuracy_score(actuals, predictions.round())
        elif metric == 'auc':
            score = roc_auc_score(actuals, predictions)
        elif metric == 'log_loss':
            score = -log_loss(actuals, predictions)  # Negative for consistency
        
        self.performance_history[model_name].append(score)
        
        # Keep only recent history
        if len(self.performance_history[model_name]) > self.window_size:
            self.performance_history[model_name].pop(0)
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """Calculate adaptive weights based on recent performance."""
        if not self.performance_history:
            return {}
        
        # Calculate average recent performance
        avg_scores = {}
        for model_name, scores in self.performance_history.items():
            if scores:
                avg_scores[model_name] = np.mean(scores[-self.window_size:])
        
        # Convert to weights (softmax on scores)
        if avg_scores:
            scores_array = np.array(list(avg_scores.values()))
            # Use temperature to control weight distribution
            temperature = 0.5
            exp_scores = np.exp(scores_array / temperature)
            weights = exp_scores / exp_scores.sum()
            
            self.current_weights = {
                name: weight 
                for name, weight in zip(avg_scores.keys(), weights)
            }
        
        return self.current_weights


class StackedEnsemblePredictor:
    """
    Stacked ensemble with meta-learning and adaptive weighting.
    Combines multiple base models with a meta-learner.
    """
    
    def __init__(self,
                 lstm_model: Optional[Any] = None,
                 n_folds: int = 5,
                 use_time_series_cv: bool = True,
                 calibrate_probabilities: bool = True):
        """
        Initialize stacked ensemble.
        
        Args:
            lstm_model: Pre-trained LSTM model (optional)
            n_folds: Number of CV folds for stacking
            use_time_series_cv: Use time series split instead of stratified
            calibrate_probabilities: Whether to calibrate output probabilities
        """
        self.lstm_model = lstm_model
        self.n_folds = n_folds
        self.use_time_series_cv = use_time_series_cv
        self.calibrate_probabilities = calibrate_probabilities
        
        # Initialize base models
        self.base_models = self._initialize_base_models()
        
        # Meta-learner - using LightGBM for speed and performance
        self.meta_learner = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        # Performance tracking
        self.performance_tracker = ModelPerformanceTracker()
        
        # Storage for fitted models
        self.fitted_base_models = {}
        self.fitted_meta_learner = None
        self.probability_calibrators = {}
        
        # Feature importance tracking
        self.base_model_importance = {}
        self.meta_feature_importance = None
        
    def _initialize_base_models(self) -> Dict[str, Any]:
        """Initialize base models with optimized parameters."""
        models = {
            'rf': RandomForestClassifier(
                n_estimators=1000,
                max_features='sqrt',
                max_depth=20,
                min_samples_split=20,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=42
            ),
            
            'xgb': xgb.XGBClassifier(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                random_state=42,
                n_jobs=-1 if not torch.cuda.is_available() else 1  # GPU doesn't need multiple jobs
            ),
            
            'lgb': lgb.LGBMClassifier(
                n_estimators=800,
                num_leaves=31,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                device='gpu' if torch.cuda.is_available() else 'cpu',
                random_state=42,
                verbose=-1
            ),
            
            'logistic': LogisticRegression(
                penalty='elasticnet',
                l1_ratio=0.5,
                solver='saga',
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Add LSTM if provided
        if self.lstm_model is not None:
            models['lstm'] = self.lstm_model
        
        return models
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            regime_labels: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None,
            lstm_features: Optional[Dict[str, np.ndarray]] = None) -> 'StackedEnsemblePredictor':
        """
        Train the stacked ensemble using cross-validation.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Target labels [n_samples]
            regime_labels: Market regime labels [n_samples]
            feature_names: Names of features for importance analysis
            lstm_features: Dict with 'short', 'medium', 'long' arrays for LSTM
            
        Returns:
            Fitted ensemble predictor
        """
        print("🚀 Training Stacked Ensemble...")
        print(f"📊 Data shape: {X.shape}")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        # Choose CV strategy
        if self.use_time_series_cv:
            cv = TimeSeriesSplit(n_splits=self.n_folds)
            print("📅 Using TimeSeriesSplit for temporal consistency")
        else:
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=False)
            print("📅 Using StratifiedKFold")
        
        # First stage: Generate out-of-fold predictions
        n_models = len(self.base_models)
        oof_predictions = np.zeros((len(X), n_models))
        
        # Store models from each fold
        fold_models = defaultdict(list)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"\n📁 Processing fold {fold_idx + 1}/{self.n_folds}")
            
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Handle regime labels if provided
            if regime_labels is not None:
                regime_fold_train = regime_labels[train_idx]
                regime_fold_val = regime_labels[val_idx]
            else:
                regime_fold_train = regime_fold_val = None
            
            # Train each base model
            for model_idx, (model_name, model) in enumerate(self.base_models.items()):
                print(f"   Training {model_name}...", end='')
                
                if model_name == 'lstm' and lstm_features is not None:
                    # Special handling for LSTM
                    lstm_train_data = {
                        k: v[train_idx] for k, v in lstm_features.items()
                        if k != 'feature_names'
                    }
                    lstm_val_data = {
                        k: v[val_idx] for k, v in lstm_features.items()
                        if k != 'feature_names'
                    }
                    
                    # LSTM is already trained, just get predictions
                    # (LSTM models are trained separately in the pipeline)
                    if hasattr(model, 'fit') and hasattr(model, 'predict_proba'):
                        try:
                            # Try to retrain if the model supports it
                            model.fit(lstm_train_data, y_fold_train, regime_fold_train)
                        except (AttributeError, NotImplementedError):
                            # Model is already trained, skip retraining
                            pass

                    # Get predictions
                    # LSTM model expects separate feature streams
                    if hasattr(model, 'predict_proba'):
                        try:
                            # Try the LSTM-specific interface
                            import torch
                            # Get device from model, with fallbacks
                            if hasattr(model, 'device') and model.device is not None:
                                device = model.device
                            elif hasattr(model, 'parameters'):
                                device = next(model.parameters()).device
                            else:
                                device = torch.device('cpu')

                            # Convert to tensors and move to device
                            x_short = torch.tensor(lstm_val_data['short'], dtype=torch.float32).to(device)
                            x_medium = torch.tensor(lstm_val_data['medium'], dtype=torch.float32).to(device)
                            x_long = torch.tensor(lstm_val_data['long'], dtype=torch.float32).to(device)

                            # Ensure regime_ids has correct shape [batch_size, 1]
                            regime_ids = torch.tensor(regime_fold_val, dtype=torch.long).to(device)
                            if regime_ids.dim() == 1:
                                regime_ids = regime_ids.unsqueeze(-1)  # Add dimension to make [batch_size, 1]

                            with torch.no_grad():
                                probs = model.predict_proba(x_short, x_medium, x_long, regime_ids)
                                oof_predictions[val_idx, model_idx] = probs.cpu().numpy()
                        except Exception as e:
                            print(f"Warning: LSTM prediction failed: {e}")
                            # Fallback: use random predictions
                            oof_predictions[val_idx, model_idx] = np.random.random(len(val_idx)) * 0.1 + 0.45
                else:
                    # Traditional ML models
                    if model_name in ['xgb', 'lgb']:
                        # Use early stopping for tree models
                        eval_set = [(X_fold_val, y_fold_val)]
                        try:
                            # Try new XGBoost API with callbacks
                            if model_name == 'xgb':
                                from xgboost.callback import EarlyStopping
                                model.fit(
                                    X_fold_train, y_fold_train,
                                    eval_set=eval_set,
                                    callbacks=[EarlyStopping(rounds=50)],
                                    verbose=False
                                )
                            else:  # LightGBM
                                model.fit(
                                    X_fold_train, y_fold_train,
                                    eval_set=eval_set,
                                    early_stopping_rounds=50,
                                    verbose=False
                                )
                        except (ImportError, TypeError):
                            # Fallback: just fit without early stopping
                            model.fit(X_fold_train, y_fold_train)
                    else:
                        model.fit(X_fold_train, y_fold_train)
                    
                    # Get out-of-fold predictions
                    if hasattr(model, 'predict_proba'):
                        oof_predictions[val_idx, model_idx] = model.predict_proba(X_fold_val)[:, 1]
                    else:
                        oof_predictions[val_idx, model_idx] = model.predict(X_fold_val)
                
                # Store model
                fold_models[model_name].append(model)
                
                # Calculate fold performance
                fold_score = roc_auc_score(y_fold_val, oof_predictions[val_idx, model_idx])
                print(f" AUC: {fold_score:.4f}")
        
        # Second stage: Train meta-learner on out-of-fold predictions
        print("\n🔧 Training meta-learner...")
        
        # Add regime information to meta features if available
        if regime_labels is not None:
            meta_features = np.column_stack([oof_predictions, regime_labels])
        else:
            meta_features = oof_predictions
        
        # Fit meta-learner
        self.meta_learner.fit(meta_features, y)
        self.fitted_meta_learner = self.meta_learner
        
        # Store meta-feature importance
        if hasattr(self.meta_learner, 'feature_importances_'):
            self.meta_feature_importance = self.meta_learner.feature_importances_
        
        # Third stage: Retrain base models on full data
        print("\n🔄 Retraining base models on full data...")
        
        for model_name, model in self.base_models.items():
            print(f"   Training {model_name}...", end='')
            
            if model_name == 'lstm' and lstm_features is not None:
                # LSTM is already trained, skip retraining
                if hasattr(model, 'fit'):
                    try:
                        model.fit(lstm_features, y, regime_labels)
                    except (AttributeError, NotImplementedError):
                        # Model is already trained, skip retraining
                        pass
            else:
                if model_name in ['xgb', 'lgb']:
                    # Use a validation set for early stopping
                    val_size = int(0.1 * len(X))
                    X_train, X_val = X[:-val_size], X[-val_size:]
                    y_train, y_val = y[:-val_size], y[-val_size:]

                    try:
                        if model_name == 'xgb':
                            from xgboost.callback import EarlyStopping
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_val, y_val)],
                                callbacks=[EarlyStopping(rounds=50)],
                                verbose=False
                            )
                        else:  # LightGBM
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_val, y_val)],
                                early_stopping_rounds=50,
                                verbose=False
                            )
                    except (ImportError, TypeError):
                        # Fallback: just fit without early stopping
                        model.fit(X, y)
                else:
                    model.fit(X, y)
            
            self.fitted_base_models[model_name] = model
            
            # Store feature importance
            if hasattr(model, 'feature_importances_') and feature_names is not None:
                self.base_model_importance[model_name] = dict(
                    zip(feature_names, model.feature_importances_)
                )
            
            print(" Done!")
        
        # Calibrate probabilities if requested
        if self.calibrate_probabilities:
            print("\n📊 Calibrating probabilities...")
            self._calibrate_models(X, y, regime_labels, lstm_features)
        
        print("\n✅ Ensemble training complete!")
        
        # Print cross-validation summary
        self._print_cv_summary(oof_predictions, y)
        
        return self
    
    def predict_proba(self, X: np.ndarray, 
                     regime_labels: Optional[np.ndarray] = None,
                     lstm_features: Optional[Dict[str, np.ndarray]] = None,
                     return_all_predictions: bool = False) -> np.ndarray:
        """
        Make probability predictions using the ensemble.
        
        Args:
            X: Feature matrix
            regime_labels: Market regime labels
            lstm_features: LSTM-specific features
            return_all_predictions: Return individual model predictions
            
        Returns:
            Probability predictions (or dict if return_all_predictions=True)
        """
        # Get base model predictions
        base_predictions = []
        individual_predictions = {}
        
        for model_name, model in self.fitted_base_models.items():
            if model_name == 'lstm' and lstm_features is not None:
                # LSTM model expects separate feature streams
                try:
                    import torch
                    # Get device from model, with fallbacks
                    if hasattr(model, 'device') and model.device is not None:
                        device = model.device
                    elif hasattr(model, 'parameters'):
                        device = next(model.parameters()).device
                    else:
                        device = torch.device('cpu')

                    # Convert to tensors and move to device
                    # Handle LSTM feature reshaping for single predictions
                    short_data = lstm_features['short']
                    medium_data = lstm_features['medium']
                    long_data = lstm_features['long']

                    # For single predictions, we need to reshape the features properly
                    sequence_length = 20

                    # Check if we have enough data for sequences
                    if len(short_data) < sequence_length:
                        print(f"⚠️ Warning: Not enough data for LSTM sequence ({len(short_data)} < {sequence_length})")
                        pred = np.random.random(len(regime_labels)) * 0.1 + 0.45
                        individual_predictions[model_name] = pred
                        base_predictions.append(pred)
                        continue

                    # Reshape for single prediction
                    if short_data.ndim == 2:  # Shape: [time_steps, features]
                        # Take last sequence_length steps and reshape to [1, seq_len, features]
                        short_seq = short_data[-sequence_length:].reshape(1, sequence_length, -1)
                        medium_seq = medium_data[-sequence_length:].reshape(1, sequence_length, -1)
                        long_single = long_data[-1:].reshape(1, -1)  # Take last step only
                    else:
                        # Already properly shaped
                        short_seq = short_data
                        medium_seq = medium_data
                        long_single = long_data

                    x_short = torch.tensor(short_seq, dtype=torch.float32).to(device)
                    x_medium = torch.tensor(medium_seq, dtype=torch.float32).to(device)
                    x_long = torch.tensor(long_single, dtype=torch.float32).to(device)

                    # Ensure regime_ids has correct shape [batch_size, 1]
                    regime_ids = torch.tensor(regime_labels, dtype=torch.long).to(device)
                    if regime_ids.dim() == 1:
                        regime_ids = regime_ids.unsqueeze(-1)  # Add dimension to make [batch_size, 1]

                    with torch.no_grad():
                        pred = model.predict_proba(x_short, x_medium, x_long, regime_ids)
                        pred = pred.cpu().numpy()
                except Exception as e:
                    print(f"Warning: LSTM prediction failed: {e}")
                    # Fallback: use random predictions
                    pred = np.random.random(len(regime_labels)) * 0.1 + 0.45
            else:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
            
            # Apply calibration if available
            if model_name in self.probability_calibrators:
                calibrated = self.probability_calibrators[model_name].transform(pred.reshape(-1, 1))
                pred = calibrated.flatten() if calibrated.ndim > 1 else calibrated
            
            base_predictions.append(pred)
            individual_predictions[model_name] = pred
        
        # Stack predictions
        base_predictions = np.column_stack(base_predictions)
        
        # Add regime information if available
        if regime_labels is not None:
            meta_features = np.column_stack([base_predictions, regime_labels])
        else:
            meta_features = base_predictions
        
        # Get meta-learner prediction
        final_predictions = self.fitted_meta_learner.predict_proba(meta_features)[:, 1]
        
        if return_all_predictions:
            return {
                'ensemble': final_predictions,
                'individual': individual_predictions,
                'meta_features': base_predictions
            }
        
        return final_predictions
    
    def predict(self, X: np.ndarray, 
                regime_labels: Optional[np.ndarray] = None,
                lstm_features: Optional[Dict[str, np.ndarray]] = None,
                threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions."""
        proba = self.predict_proba(X, regime_labels, lstm_features)
        return (proba >= threshold).astype(int)
    
    def _calibrate_models(self, X: np.ndarray, y: np.ndarray,
                         regime_labels: Optional[np.ndarray] = None,
                         lstm_features: Optional[Dict[str, np.ndarray]] = None):
        """Calibrate probability outputs using isotonic regression."""
        from sklearn.calibration import CalibratedClassifierCV
        
        # Use a small holdout set for calibration
        cal_size = int(0.1 * len(X))
        X_cal, y_cal = X[-cal_size:], y[-cal_size:]
        
        for model_name, model in self.fitted_base_models.items():
            if model_name == 'lstm' and lstm_features is not None:
                # Skip LSTM calibration for now
                continue
            
            # Fit calibrator
            calibrator = CalibratedClassifierCV(
                model, method='isotonic', cv='prefit'
            )
            
            # Get uncalibrated predictions
            if hasattr(model, 'predict_proba'):
                uncalibrated = model.predict_proba(X_cal)[:, 1]
            else:
                uncalibrated = model.predict(X_cal)
            
            # Fit isotonic regression
            from sklearn.isotonic import IsotonicRegression
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(uncalibrated, y_cal)
            
            self.probability_calibrators[model_name] = iso_reg
    
    def _print_cv_summary(self, oof_predictions: np.ndarray, y: np.ndarray):
        """Print cross-validation performance summary."""
        print("\n📊 Cross-Validation Performance Summary:")
        print("=" * 60)
        
        # Individual model performances
        for i, model_name in enumerate(self.base_models.keys()):
            auc = roc_auc_score(y, oof_predictions[:, i])
            acc = accuracy_score(y, oof_predictions[:, i].round())
            print(f"{model_name:>10}: AUC = {auc:.4f}, Accuracy = {acc:.4f}")
        
        # Ensemble performance (simple average)
        ensemble_avg = oof_predictions.mean(axis=1)
        ensemble_auc = roc_auc_score(y, ensemble_avg)
        ensemble_acc = accuracy_score(y, ensemble_avg.round())
        print(f"{'Avg Ensemble':>10}: AUC = {ensemble_auc:.4f}, Accuracy = {ensemble_acc:.4f}")
        
        print("=" * 60)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get aggregated feature importance across all models.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        if not self.base_model_importance:
            return pd.DataFrame()
        
        # Aggregate importance scores
        importance_dict = defaultdict(list)
        
        for model_name, importance in self.base_model_importance.items():
            for feature, score in importance.items():
                importance_dict[feature].append(score)
        
        # Calculate average importance
        avg_importance = {
            feature: np.mean(scores) 
            for feature, scores in importance_dict.items()
        }
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': list(avg_importance.keys()),
            'importance': list(avg_importance.values())
        })
        
        # Sort and return top features
        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df.head(top_n)
    
    def explain_prediction(self, X: np.ndarray, 
                          regime_labels: Optional[np.ndarray] = None,
                          lstm_features: Optional[Dict[str, np.ndarray]] = None,
                          sample_idx: int = 0) -> Dict[str, Any]:
        """
        Generate explanation for a single prediction.
        
        Args:
            X: Feature matrix
            regime_labels: Market regime labels
            lstm_features: LSTM-specific features
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with prediction explanation
        """
        # Get all predictions
        # Handle LSTM features carefully to preserve tensor shapes
        lstm_features_slice = None
        if lstm_features is not None:
            # For single predictions, the LSTM features are already properly shaped
            # Don't slice them further as it will corrupt the tensor dimensions
            lstm_features_slice = lstm_features

        all_preds = self.predict_proba(
            X[sample_idx:sample_idx+1],
            regime_labels[sample_idx:sample_idx+1] if regime_labels is not None else None,
            lstm_features_slice,
            return_all_predictions=True
        )
        
        explanation = {
            'final_prediction': all_preds['ensemble'][0],
            'individual_predictions': {
                model: pred[0] 
                for model, pred in all_preds['individual'].items()
            },
            'model_weights': self.performance_tracker.current_weights,
            'regime': regime_labels[sample_idx] if regime_labels is not None else None,
            'confidence': self._calculate_prediction_confidence(all_preds['individual'])
        }
        
        return explanation
    
    def _calculate_prediction_confidence(self, predictions: Dict[str, np.ndarray]) -> float:
        """Calculate confidence based on model agreement."""
        pred_values = list(predictions.values())
        if len(pred_values) == 0:
            return 0.0
        
        # Calculate standard deviation of predictions
        std_dev = np.std(pred_values)
        
        # Lower std means higher confidence
        confidence = 1.0 - min(std_dev * 2, 1.0)
        
        return confidence
    
    def save_ensemble(self, filepath: str):
        """Save the entire ensemble to disk."""
        ensemble_data = {
            'base_models': self.fitted_base_models,
            'meta_learner': self.fitted_meta_learner,
            'calibrators': self.probability_calibrators,
            'feature_importance': self.base_model_importance,
            'meta_feature_importance': self.meta_feature_importance,
            'performance_tracker': self.performance_tracker
        }
        
        joblib.dump(ensemble_data, filepath)
        print(f"✅ Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load a saved ensemble from disk."""
        ensemble_data = joblib.load(filepath)

        self.fitted_base_models = ensemble_data['base_models']
        self.fitted_meta_learner = ensemble_data['meta_learner']
        self.probability_calibrators = ensemble_data['calibrators']
        self.base_model_importance = ensemble_data['feature_importance']
        self.meta_feature_importance = ensemble_data['meta_feature_importance']
        self.performance_tracker = ensemble_data['performance_tracker']

        # Fix device attributes for any LSTM models in the ensemble
        if 'lstm' in self.fitted_base_models:
            lstm_model = self.fitted_base_models['lstm']
            if hasattr(lstm_model, '__class__') and 'LSTM' in lstm_model.__class__.__name__:
                # Import here to avoid circular imports
                import torch
                from multi_stream_lstm import DEVICE

                # Ensure device attribute is set
                if not hasattr(lstm_model, 'device') or lstm_model.device is None:
                    lstm_model.device = DEVICE

                # Move model to correct device
                lstm_model = lstm_model.to(lstm_model.device)
                lstm_model.eval()

                # Update the model in the dictionary
                self.fitted_base_models['lstm'] = lstm_model

        print(f"✅ Ensemble loaded from {filepath}")


# Example usage
def test_stacked_ensemble():
    """Test the stacked ensemble with synthetic data."""
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 5000
    n_features = 50
    
    # Create features with some structure
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some signal
    signal = (
        0.3 * X[:, 0] + 
        0.2 * X[:, 1] * X[:, 2] + 
        0.1 * np.sin(X[:, 3]) +
        np.random.randn(n_samples) * 0.5
    )
    y = (signal > np.median(signal)).astype(int)
    
    # Create regime labels (6 regimes)
    regime_labels = np.random.randint(0, 6, n_samples)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    print("🧪 Testing Stacked Ensemble")
    print("=" * 60)
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    regime_train, regime_test = regime_labels[:train_size], regime_labels[train_size:]
    
    # Initialize ensemble (without LSTM for this test)
    ensemble = StackedEnsemblePredictor(
        lstm_model=None,
        n_folds=5,
        use_time_series_cv=True,
        calibrate_probabilities=True
    )
    
    # Train ensemble
    ensemble.fit(
        X_train, y_train,
        regime_labels=regime_train,
        feature_names=feature_names
    )
    
    # Make predictions
    predictions = ensemble.predict_proba(X_test, regime_test)
    
    # Evaluate
    test_auc = roc_auc_score(y_test, predictions)
    test_acc = accuracy_score(y_test, predictions.round())
    
    print(f"\n📊 Test Performance:")
    print(f"   AUC: {test_auc:.4f}")
    print(f"   Accuracy: {test_acc:.4f}")
    
    # Get feature importance
    importance_df = ensemble.get_feature_importance(top_n=10)
    print("\n🎯 Top 10 Important Features:")
    print(importance_df)
    
    # Explain a prediction
    explanation = ensemble.explain_prediction(X_test, regime_test, sample_idx=0)
    print("\n🔍 Sample Prediction Explanation:")
    print(f"   Final prediction: {explanation['final_prediction']:.4f}")
    print("   Individual model predictions:")
    for model, pred in explanation['individual_predictions'].items():
        print(f"     {model}: {pred:.4f}")
    print(f"   Confidence: {explanation['confidence']:.4f}")
    
    return ensemble


if __name__ == "__main__":
    # Run test
    ensemble = test_stacked_ensemble()
