import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

# GPU device management
def get_device():
    """Get the best available device (GPU if available, otherwise CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU for LSTM: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device('cpu')
        print("💻 Using CPU for LSTM")
        return device

# Global device setting
DEVICE = get_device()


class StockDataset(Dataset):
    """Custom dataset for multi-stream stock data."""
    
    def __init__(self, 
                 features_dict: Dict[str, np.ndarray],
                 labels: np.ndarray,
                 regime_ids: np.ndarray,
                 sequence_length: int = 20):
        """
        Initialize dataset.
        
        Args:
            features_dict: Dictionary with 'short', 'medium', 'long' feature arrays
            labels: Binary labels (0/1)
            regime_ids: Market regime IDs for each sample
            sequence_length: Length of sequences for LSTM
        """
        self.features_dict = features_dict
        self.labels = labels
        self.regime_ids = regime_ids
        self.sequence_length = sequence_length
        
        # Validate data
        n_samples = len(labels)
        for key, features in features_dict.items():
            if len(features) != n_samples:
                raise ValueError(f"Feature {key} has {len(features)} samples, expected {n_samples}")
    
    def __len__(self):
        return len(self.labels) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Get sequences for each time horizon
        short_seq = self.features_dict['short'][idx:idx + self.sequence_length]
        medium_seq = self.features_dict['medium'][idx:idx + self.sequence_length]
        long_feat = self.features_dict['long'][idx + self.sequence_length - 1]
        
        # Get label and regime for the last timestep
        label = self.labels[idx + self.sequence_length - 1]
        regime_id = self.regime_ids[idx + self.sequence_length - 1]
        
        return {
            'short': torch.FloatTensor(short_seq),
            'medium': torch.FloatTensor(medium_seq),
            'long': torch.FloatTensor(long_feat),
            'regime_id': torch.LongTensor([regime_id]),
            'label': torch.FloatTensor([label])
        }


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for regime conditioning."""
    
    def __init__(self, feature_dim: int, conditioning_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.gamma_layer = nn.Linear(conditioning_dim, feature_dim)
        self.beta_layer = nn.Linear(conditioning_dim, feature_dim)
        
        # Initialize gamma to 1 and beta to 0 for identity transformation
        nn.init.ones_(self.gamma_layer.weight)
        nn.init.zeros_(self.gamma_layer.bias)
        nn.init.zeros_(self.beta_layer.weight)
        nn.init.zeros_(self.beta_layer.bias)
    
    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.
        
        Args:
            features: Input features [batch_size, feature_dim]
            conditioning: Conditioning vector [batch_size, conditioning_dim]
            
        Returns:
            Modulated features
        """
        gamma = 1 + self.gamma_layer(conditioning)  # Centered at 1
        beta = self.beta_layer(conditioning)
        
        return gamma * features + beta


class MultiStreamLSTM(nn.Module):
    """Multi-stream LSTM with cross-attention and regime conditioning."""

    def __init__(self,
                 feature_dims: Dict[str, int],
                 n_regimes: int = 6,
                 hidden_sizes: Dict[str, int] = None,
                 dropout_rate: float = 0.2,
                 device: Optional[torch.device] = None):
        """
        Initialize multi-stream LSTM.

        Args:
            feature_dims: Dictionary with dimensions for 'short', 'medium', 'long' features
            n_regimes: Number of market regimes
            hidden_sizes: Hidden sizes for each stream (default: auto-calculated)
            dropout_rate: Dropout rate for regularization
            device: Device to use (GPU/CPU). If None, auto-detects best device.
        """
        super().__init__()

        self.feature_dims = feature_dims
        self.n_regimes = n_regimes
        self.device = device if device is not None else DEVICE
        
        # Default hidden sizes if not provided
        if hidden_sizes is None:
            hidden_sizes = {
                'short': 64,
                'medium': 48,
                'long': 32
            }
        self.hidden_sizes = hidden_sizes
        
        # Short-term LSTM (processes high-frequency features)
        self.short_lstm = nn.LSTM(
            input_size=feature_dims['short'],
            hidden_size=hidden_sizes['short'],
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True
        )
        self.short_output_size = hidden_sizes['short'] * 2  # Bidirectional
        
        # Medium-term LSTM (processes medium-frequency features)
        self.medium_lstm = nn.LSTM(
            input_size=feature_dims['medium'],
            hidden_size=hidden_sizes['medium'],
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True
        )
        self.medium_output_size = hidden_sizes['medium'] * 2  # Bidirectional
        
        # Long-term feedforward (processes low-frequency features)
        self.long_ff = nn.Sequential(
            nn.Linear(feature_dims['long'], hidden_sizes['long'] * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes['long'] * 2, hidden_sizes['long']),
            nn.ReLU()
        )
        
        # Cross-attention between short and medium streams
        # Use the smaller dimension to avoid issues
        self.attention_dim = min(self.short_output_size, self.medium_output_size)

        # Projection layers to match attention dimension
        self.short_proj = nn.Linear(self.short_output_size, self.attention_dim)
        self.medium_proj = nn.Linear(self.medium_output_size, self.attention_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=4,  # Reduced to ensure divisibility
            dropout=dropout_rate,
            batch_first=True
        )

        # Attention normalization layers
        self.attention_norm = nn.LayerNorm(self.attention_dim)
        
        # Regime embedding
        self.regime_embedding = nn.Embedding(n_regimes, 16)
        
        # FiLM layers for regime conditioning
        total_features = self.attention_dim + self.medium_output_size + hidden_sizes['long']
        self.film_layer = FiLMLayer(total_features, 16)

        # Final fusion layers
        self.fusion_input_size = total_features + 16  # Features + regime embedding
        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Binary classification
        )
        
        # Attention weight storage for interpretability
        self.last_attention_weights = None

        # Move model to device
        self.to(self.device)

    def to_device(self, device: torch.device):
        """Move model to specified device."""
        self.device = device
        return self.to(device)

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to ensure device is properly set."""
        result = super().load_state_dict(state_dict, strict=strict)
        # Ensure device is set after loading
        if not hasattr(self, 'device') or self.device is None:
            self.device = DEVICE
        return result

    def forward(self, x_short: torch.Tensor, x_medium: torch.Tensor,
                x_long: torch.Tensor, regime_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-stream architecture.
        
        Args:
            x_short: Short-term features [batch, seq_len, features]
            x_medium: Medium-term features [batch, seq_len, features]
            x_long: Long-term features [batch, features]
            regime_id: Market regime IDs [batch, 1]
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        # Ensure all inputs are on the correct device
        x_short = x_short.to(self.device)
        x_medium = x_medium.to(self.device)
        x_long = x_long.to(self.device)
        regime_id = regime_id.to(self.device)

        batch_size = x_short.size(0)

        # Process short-term stream
        short_out, (h_short, c_short) = self.short_lstm(x_short)
        # Take the output from the last timestep
        short_final = short_out[:, -1, :]  # [batch, hidden_size * 2]
        
        # Process medium-term stream
        medium_out, (h_medium, c_medium) = self.medium_lstm(x_medium)
        medium_final = medium_out[:, -1, :]  # [batch, hidden_size * 2]
        
        # Process long-term features
        long_out = self.long_ff(x_long)  # [batch, hidden_size]
        
        # Apply cross-attention: short attending to medium sequence
        # Project to attention dimension
        short_proj = self.short_proj(short_final)  # [batch, attention_dim]
        medium_proj = self.medium_proj(medium_out)  # [batch, seq_len, attention_dim]

        # Reshape for attention mechanism
        short_query = short_proj.unsqueeze(1)  # [batch, 1, attention_dim]
        medium_keys = medium_proj  # [batch, seq_len, attention_dim]

        attended_short, attention_weights = self.cross_attention(
            short_query, medium_keys, medium_keys
        )
        attended_short = attended_short.squeeze(1)  # [batch, attention_dim]

        # Apply residual connection and normalization
        short_combined = self.attention_norm(short_proj + attended_short)
        
        # Store attention weights for interpretability
        self.last_attention_weights = attention_weights.detach()
        
        # Concatenate all stream outputs
        combined_features = torch.cat([
            short_combined,
            medium_final,
            long_out
        ], dim=1)  # [batch, total_features]
        
        # Apply regime conditioning via FiLM
        # Ensure regime_id is properly shaped for embedding lookup
        if regime_id.dim() > 1:
            regime_id_flat = regime_id.squeeze(-1)  # Remove last dimension if present
        else:
            regime_id_flat = regime_id

        # Clamp regime_id to valid range [0, n_regimes-1]
        regime_id_flat = torch.clamp(regime_id_flat, 0, self.n_regimes - 1)

        regime_emb = self.regime_embedding(regime_id_flat)  # [batch, 16]
        modulated_features = self.film_layer(combined_features, regime_emb)
        
        # Final fusion with regime embedding
        final_input = torch.cat([modulated_features, regime_emb], dim=1)
        
        # Get predictions
        logits = self.fusion(final_input)  # [batch, 2]
        
        return logits, attention_weights
    
    def predict_proba(self, x_short: torch.Tensor, x_medium: torch.Tensor,
                     x_long: torch.Tensor, regime_id: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        # Ensure model is in eval mode
        self.eval()

        with torch.no_grad():
            # Ensure inputs are on correct device
            x_short = x_short.to(self.device)
            x_medium = x_medium.to(self.device)
            x_long = x_long.to(self.device)
            regime_id = regime_id.to(self.device)

            logits, _ = self.forward(x_short, x_medium, x_long, regime_id)
            return F.softmax(logits, dim=1)[:, 1]  # Return positive class probability


class MultiStreamLSTMModule(pl.LightningModule):
    """PyTorch Lightning module for training the Multi-Stream LSTM."""
    
    def __init__(self, 
                 model: MultiStreamLSTM,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss function with optional class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def forward(self, batch):
        return self.model(
            batch['short'], 
            batch['medium'], 
            batch['long'], 
            batch['regime_id']
        )
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        logits, _ = self.forward(batch)
        labels = batch['label'].squeeze(-1).long()
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        logits, _ = self.forward(batch)
        labels = batch['label'].squeeze(-1).long()
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Calculate additional metrics
        probs = F.softmax(logits, dim=1)[:, 1]
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', accuracy, on_epoch=True)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'probs': probs,
            'labels': labels
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            ),
            'monitor': 'val_loss',
            'interval': 'epoch'
        }
        
        return [optimizer], [scheduler]


class AttentionVisualizer:
    """Visualize attention weights for interpretability."""
    
    @staticmethod
    def visualize_attention(attention_weights: torch.Tensor,
                          input_labels: Optional[List[str]] = None,
                          save_path: Optional[str] = None):
        """
        Visualize attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weights [batch, heads, seq_len, seq_len]
            input_labels: Labels for sequence positions
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Average over heads and batch
        if len(attention_weights.shape) == 4:
            avg_attention = attention_weights.mean(dim=[0, 1]).cpu().numpy()
        else:
            avg_attention = attention_weights.cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            avg_attention,
            cmap='Blues',
            cbar=True,
            square=True,
            xticklabels=input_labels if input_labels else False,
            yticklabels=input_labels if input_labels else False
        )
        
        plt.title('Cross-Attention Weights (Short attending to Medium)')
        plt.xlabel('Medium Stream Positions')
        plt.ylabel('Short Stream Query')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def prepare_features_for_lstm(df: pd.DataFrame, 
                            feature_engineer,
                            sequence_length: int = 20) -> Dict[str, np.ndarray]:
    """
    Prepare features for multi-stream LSTM from engineered features.
    
    Args:
        df: DataFrame with engineered features
        feature_engineer: AdvancedFeatureEngineer instance
        sequence_length: Sequence length for LSTM
        
    Returns:
        Dictionary with 'short', 'medium', 'long' feature arrays
    """
    # Define feature groups based on time horizons - adjusted to match saved model dimensions
    # Target: 15 short, 13 medium, 9 long features
    short_features = [
        'RSI_14', 'ADX_14', 'ATR_14', 'MFI_14', 'Stoch_K', 'Stoch_D',
        'Williams_R', 'returns_mean_5', 'returns_std_5', 'volume_mean_5',
        'Volume_SMA_Ratio', 'CMF_20', 'ROC_20', 'Linear_Slope_20',
        'volatility_5'  # Reduced to 15 features
    ]

    medium_features = [
        'RSI_63', 'ADX_63', 'ATR_63', 'Beta_60', 'Rolling_Corr_30',
        'returns_mean_20', 'returns_std_20', 'volume_mean_20', 'Historical_Vol_20',
        'trend_strength_50', 'momentum_20', 'volatility_20',
        'MACD'  # Reduced to 13 features
    ]

    long_features = [
        'ADX_252', 'Beta_252', 'Linear_Slope_252', 'Distance_From_52W_High',
        'Distance_From_52W_Low', 'Price_Percentile_252', 'trend_strength_200',
        'OBV', 'drawdown'  # Reduced to 9 features
    ]
    
    # Filter available features and ensure exact counts
    available_short = [f for f in short_features if f in df.columns]
    available_medium = [f for f in medium_features if f in df.columns]
    available_long = [f for f in long_features if f in df.columns]

    # Ensure we have exactly the right number of features for the saved model
    target_counts = {'short': 15, 'medium': 13, 'long': 9}

    # Truncate or pad features to match expected dimensions
    if len(available_short) > target_counts['short']:
        available_short = available_short[:target_counts['short']]
    elif len(available_short) < target_counts['short']:
        # Add fallback features if needed
        fallback_short = ['volatility_10', 'Local_Resistance_20', 'Local_Support_20',
                         'Distance_From_Resistance', 'Distance_From_Support']
        for f in fallback_short:
            if f in df.columns and f not in available_short and len(available_short) < target_counts['short']:
                available_short.append(f)

    if len(available_medium) > target_counts['medium']:
        available_medium = available_medium[:target_counts['medium']]
    elif len(available_medium) < target_counts['medium']:
        # Add fallback features if needed
        fallback_medium = ['returns_mean_63', 'returns_std_63', 'volume_mean_63',
                          'Historical_Vol_63', 'momentum_60', 'volatility_60', 'MACD_Signal', 'MACD_Histogram']
        for f in fallback_medium:
            if f in df.columns and f not in available_medium and len(available_medium) < target_counts['medium']:
                available_medium.append(f)

    if len(available_long) > target_counts['long']:
        available_long = available_long[:target_counts['long']]
    elif len(available_long) < target_counts['long']:
        # Add fallback features if needed
        fallback_long = ['AD_Line', 'PVT', 'day_of_week', 'month', 'quarter',
                        'drawdown_duration', 'n_resistance_levels', 'n_support_levels']
        for f in fallback_long:
            if f in df.columns and f not in available_long and len(available_long) < target_counts['long']:
                available_long.append(f)

    print(f"📊 Feature allocation:")
    print(f"   Short-term: {len(available_short)} features")
    print(f"   Medium-term: {len(available_medium)} features")
    print(f"   Long-term: {len(available_long)} features")

    # Extract feature arrays
    features_dict = {
        'short': df[available_short].values,
        'medium': df[available_medium].values,
        'long': df[available_long].values
    }

    # Note: feature names are stored separately if needed for debugging
    # but not included in the returned dict to avoid issues with StockDataset

    return features_dict


# Example usage
def test_multi_stream_lstm():
    """Test the multi-stream LSTM architecture."""
    # Create synthetic data
    n_samples = 1000
    seq_length = 20
    
    # Feature dimensions
    feature_dims = {
        'short': 20,
        'medium': 21,
        'long': 17
    }
    
    # Generate synthetic features
    features_dict = {
        'short': np.random.randn(n_samples, feature_dims['short']),
        'medium': np.random.randn(n_samples, feature_dims['medium']),
        'long': np.random.randn(n_samples, feature_dims['long'])
    }
    
    # Generate synthetic labels and regime IDs
    labels = np.random.randint(0, 2, n_samples)
    regime_ids = np.random.randint(0, 6, n_samples)
    
    # Create dataset
    dataset = StockDataset(features_dict, labels, regime_ids, seq_length)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = MultiStreamLSTM(feature_dims, n_regimes=6)
    
    # Test forward pass
    batch = next(iter(dataloader))
    logits, attention_weights = model(
        batch['short'], 
        batch['medium'], 
        batch['long'], 
        batch['regime_id']
    )
    
    print("🔍 Model Test Results:")
    print(f"   Input shapes:")
    print(f"     - Short: {batch['short'].shape}")
    print(f"     - Medium: {batch['medium'].shape}")
    print(f"     - Long: {batch['long'].shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Attention shape: {attention_weights.shape}")
    
    # Test probability output
    probs = model.predict_proba(
        batch['short'], 
        batch['medium'], 
        batch['long'], 
        batch['regime_id']
    )
    print(f"   Probability shape: {probs.shape}")
    print(f"   Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model, dataset


if __name__ == "__main__":
    # Run test
    model, dataset = test_multi_stream_lstm()
