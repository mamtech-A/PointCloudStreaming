"""
LSTM-based Bandwidth Prediction Model

This module implements a proper PyTorch LSTM model for bandwidth prediction
with train/test split, hyperparameter tuning, and model saving functionality.
"""

import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import deque
import json
import copy


class BandwidthDataset(Dataset):
    """PyTorch Dataset for bandwidth sequences."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BandwidthLSTM(nn.Module):
    """
    LSTM model for bandwidth prediction.
    
    Architecture:
    - Input layer: sequence_length features
    - LSTM layers with configurable hidden_size and num_layers
    - Dropout for regularization
    - Fully connected output layer
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of features per timestep (1 for univariate bandwidth)
            hidden_size: Number of hidden units in LSTM layers
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate for regularization (applied between LSTM layers)
        """
        super(BandwidthLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers for output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Reshape input if needed: (batch, seq_len) -> (batch, seq_len, 1)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc(last_hidden)
        
        return output.squeeze(-1)


class LSTMPredictor:
    """
    LSTM-based bandwidth predictor with training and inference capabilities.
    
    This class wraps the BandwidthLSTM model and provides methods for:
    - Training with train/test split
    - Hyperparameter tuning
    - Model saving and loading
    - Real-time prediction
    """
    
    def __init__(self, sequence_length=10, hidden_size=64, num_layers=2, 
                 dropout=0.2, learning_rate=0.001, device=None):
        """
        Initialize the LSTM predictor.
        
        Args:
            sequence_length: Number of historical samples for prediction
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            device: Computation device ('cuda' or 'cpu')
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = BandwidthLSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Statistics for normalization
        self.stats = {
            'mean': 0,
            'std': 1,
            'min': 0,
            'max': 100
        }
        
        # History for real-time prediction
        self.history = deque(maxlen=sequence_length)
        self.trained = False
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'test_loss': [],
            'best_epoch': 0,
            'best_test_loss': float('inf')
        }
    
    def _normalize(self, data):
        """Normalize data using stored statistics."""
        return (data - self.stats['mean']) / (self.stats['std'] + 1e-8)
    
    def _denormalize(self, data):
        """Denormalize data using stored statistics."""
        return data * (self.stats['std'] + 1e-8) + self.stats['mean']
    
    def fit(self, X, y, test_size=0.2, epochs=100, batch_size=32, 
            patience=15, verbose=True):
        """
        Train the LSTM model with train/test split.
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length)
            y: Target values of shape (n_samples,)
            test_size: Fraction of data to use for testing
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience (epochs without improvement)
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training history and metrics
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Compute normalization statistics from training data
        all_values = np.concatenate([X.flatten(), y])
        self.stats['mean'] = float(np.mean(all_values))
        self.stats['std'] = float(np.std(all_values)) if np.std(all_values) > 0 else 1.0
        self.stats['min'] = float(np.min(all_values))
        self.stats['max'] = float(np.max(all_values))
        
        # Normalize data
        X_norm = self._normalize(X)
        y_norm = self._normalize(y)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y_norm, test_size=test_size, random_state=42
        )
        
        if verbose:
            print(f"Dataset split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Create data loaders
        train_dataset = BandwidthDataset(X_train, y_train)
        test_dataset = BandwidthDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_model_state = None
        best_test_loss = float('inf')
        epochs_without_improvement = 0
        
        self.training_history = {
            'train_loss': [],
            'test_loss': [],
            'best_epoch': 0,
            'best_test_loss': float('inf')
        }
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Evaluation phase
            self.model.eval()
            test_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_losses.append(loss.item())
            
            avg_test_loss = np.mean(test_losses)
            
            # Record history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['test_loss'].append(avg_test_loss)
            
            # Check for improvement
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                self.training_history['best_epoch'] = epoch
                self.training_history['best_test_loss'] = best_test_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, "
                      f"Test Loss: {avg_test_loss:.6f}")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"Loaded best model from epoch {self.training_history['best_epoch']+1} "
                      f"with test loss: {best_test_loss:.6f}")
        
        self.trained = True
        
        # Compute final metrics on test set
        test_predictions = self.predict(X_test)
        y_test_denorm = self._denormalize(y_test)
        test_predictions_denorm = self._denormalize(test_predictions)
        
        mae = np.mean(np.abs(y_test_denorm - test_predictions_denorm))
        # Use a larger epsilon to avoid division issues with values near zero
        nonzero_mask = np.abs(y_test_denorm) > 0.1  # Filter out near-zero values for MAPE
        if np.sum(nonzero_mask) > 0:
            mape = np.mean(np.abs((y_test_denorm[nonzero_mask] - test_predictions_denorm[nonzero_mask]) / y_test_denorm[nonzero_mask])) * 100
        else:
            mape = 0.0
        rmse = np.sqrt(np.mean((y_test_denorm - test_predictions_denorm) ** 2))
        
        metrics = {
            'mae': float(mae),
            'mape': float(mape),
            'rmse': float(rmse),
            'best_epoch': self.training_history['best_epoch'],
            'best_test_loss': float(best_test_loss)
        }
        
        if verbose:
            print(f"\nFinal Test Metrics:")
            print(f"  MAE: {mae:.4f} Mbps")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  RMSE: {rmse:.4f} Mbps")
        
        return metrics
    
    def predict(self, X):
        """
        Predict bandwidth for input sequences.
        
        Args:
            X: Input sequences (normalized or raw)
            
        Returns:
            Predicted bandwidth values
        """
        self.model.eval()
        
        # Convert to numpy if needed
        if isinstance(X, list):
            X = np.array(X)
        
        # Handle single sequence
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def update(self, bandwidth_mbps):
        """
        Update history with a new bandwidth measurement.
        
        Args:
            bandwidth_mbps: New bandwidth measurement in Mbps
        """
        self.history.append(bandwidth_mbps)
    
    def predict_next(self):
        """
        Predict next bandwidth value based on current history.
        
        Returns:
            Predicted bandwidth in Mbps
        """
        if len(self.history) < self.sequence_length:
            # Not enough history, return average
            return np.mean(list(self.history)) if len(self.history) > 0 else self.stats['mean']
        
        # Convert history to normalized sequence
        seq = np.array(list(self.history))
        seq_norm = self._normalize(seq)
        
        # Get prediction
        pred_norm = self.predict(seq_norm)
        
        # Denormalize
        pred = self._denormalize(pred_norm)
        
        # Clamp to reasonable range
        pred = np.clip(pred, self.stats['min'], self.stats['max'])
        
        return float(pred[0]) if isinstance(pred, np.ndarray) else float(pred)
    
    def save(self, path):
        """
        Save model to file.
        
        Args:
            path: Path to save the model (.pkl file)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'stats': self.stats,
            'trained': self.trained,
            'training_history': self.training_history
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load(self, path):
        """
        Load model from file.
        
        Args:
            path: Path to the saved model (.pkl file)
            
        Returns:
            Self for method chaining
        """
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.sequence_length = save_dict['sequence_length']
        self.hidden_size = save_dict['hidden_size']
        self.num_layers = save_dict['num_layers']
        self.dropout = save_dict['dropout']
        self.learning_rate = save_dict['learning_rate']
        self.stats = save_dict['stats']
        self.trained = save_dict['trained']
        self.training_history = save_dict.get('training_history', {})
        
        # Reinitialize model with correct architecture
        self.model = BandwidthLSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(save_dict['model_state_dict'])
        
        # Reinitialize history
        self.history = deque(maxlen=self.sequence_length)
        
        return self


def load_bandwidth_trace(log_path):
    """
    Load bandwidth trace from a log file.
    
    Format: timestamp ms_since_start lat lon bytes_received ms_interval
    Returns: list of bandwidth values in bps
    """
    bandwidths = []
    with open(log_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            bytes_received = int(parts[4])
            ms_interval = int(parts[5])
            if ms_interval == 0:
                continue
            # Calculate throughput in bps
            bps = (bytes_received * 8) / (ms_interval / 1000.0)
            bandwidths.append(bps)
    return bandwidths


def prepare_dataset(bandwidth_dir, sequence_length=10):
    """
    Prepare dataset from all bandwidth log files.
    
    Args:
        bandwidth_dir: Directory containing bandwidth log files
        sequence_length: Number of historical samples to use
        
    Returns:
        X: Input sequences (historical bandwidth values in Mbps)
        y: Target values (next bandwidth value in Mbps)
    """
    X, y = [], []
    
    # Get all log files
    log_files = [f for f in os.listdir(bandwidth_dir) if f.endswith('.log')]
    
    for log_file in log_files:
        log_path = os.path.join(bandwidth_dir, log_file)
        bandwidths = load_bandwidth_trace(log_path)
        
        # Normalize to Mbps for better numerical stability
        bandwidths_mbps = [bw / 1e6 for bw in bandwidths]
        
        # Create sequences
        for i in range(len(bandwidths_mbps) - sequence_length):
            X.append(bandwidths_mbps[i:i + sequence_length])
            y.append(bandwidths_mbps[i + sequence_length])
    
    return np.array(X), np.array(y)


def tune_hyperparameters(X, y, param_grid, test_size=0.2, epochs=50, verbose=True):
    """
    Perform hyperparameter tuning using grid search.
    
    Args:
        X: Input sequences
        y: Target values
        param_grid: Dictionary of hyperparameters to search
        test_size: Fraction of data for testing
        epochs: Training epochs per configuration
        verbose: Print progress
        
    Returns:
        best_params: Best hyperparameter combination
        best_metrics: Metrics for best model
        all_results: All hyperparameter combinations and their results
    """
    all_results = []
    best_params = None
    best_metrics = None
    best_mae = float('inf')
    
    # Generate all combinations
    from itertools import product
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    combinations = list(product(*param_values))
    total_combinations = len(combinations)
    
    if verbose:
        print(f"\nTesting {total_combinations} hyperparameter combinations...")
        print("=" * 60)
    
    for idx, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        if verbose:
            print(f"\n[{idx+1}/{total_combinations}] Testing: {params}")
        
        # Create and train model
        predictor = LSTMPredictor(
            sequence_length=params.get('sequence_length', 10),
            hidden_size=params.get('hidden_size', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2),
            learning_rate=params.get('learning_rate', 0.001)
        )
        
        metrics = predictor.fit(
            X, y, 
            test_size=test_size, 
            epochs=epochs, 
            batch_size=params.get('batch_size', 32),
            patience=10,
            verbose=False
        )
        
        result = {
            'params': params,
            'metrics': metrics
        }
        all_results.append(result)
        
        if verbose:
            print(f"  MAE: {metrics['mae']:.4f}, MAPE: {metrics['mape']:.2f}%, RMSE: {metrics['rmse']:.4f}")
        
        # Check if this is the best model
        if metrics['mae'] < best_mae:
            best_mae = metrics['mae']
            best_params = params
            best_metrics = metrics
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Best hyperparameters: {best_params}")
        print(f"Best MAE: {best_mae:.4f} Mbps")
    
    return best_params, best_metrics, all_results


def train_lstm_model(bandwidth_dir, output_path, sequence_length=10, tune=True, verbose=True):
    """
    Train LSTM model on bandwidth traces with optional hyperparameter tuning.
    
    Args:
        bandwidth_dir: Directory containing bandwidth log files
        output_path: Path to save trained model
        sequence_length: Number of historical samples to use
        tune: Whether to perform hyperparameter tuning
        verbose: Print progress
        
    Returns:
        Trained predictor
    """
    if verbose:
        print(f"Loading bandwidth traces from {bandwidth_dir}...")
    
    X, y = prepare_dataset(bandwidth_dir, sequence_length)
    
    if verbose:
        print(f"Dataset size: {len(X)} sequences")
        print(f"Bandwidth range: {y.min():.2f} - {y.max():.2f} Mbps")
        print(f"Mean bandwidth: {y.mean():.2f} Mbps, Std: {y.std():.2f} Mbps")
    
    if tune:
        # Define hyperparameter search space
        param_grid = {
            'hidden_size': [32, 64, 128],
            'num_layers': [1, 2],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.0005]
        }
        
        # Perform hyperparameter tuning
        best_params, best_metrics, all_results = tune_hyperparameters(
            X, y, param_grid, 
            test_size=0.2, 
            epochs=50, 
            verbose=verbose
        )
        
        # Save tuning results
        results_path = output_path.replace('.pkl', '_tuning_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_metrics': best_metrics,
                'all_results': all_results
            }, f, indent=2)
        
        if verbose:
            print(f"\nTuning results saved to: {results_path}")
            print(f"\nRetraining final model with best hyperparameters...")
        
        # Train final model with best params
        predictor = LSTMPredictor(
            sequence_length=sequence_length,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout'],
            learning_rate=best_params['learning_rate']
        )
    else:
        # Use default hyperparameters
        predictor = LSTMPredictor(
            sequence_length=sequence_length,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001
        )
    
    # Train model
    if verbose:
        print("\nTraining final model...")
    
    metrics = predictor.fit(
        X, y,
        test_size=0.2,
        epochs=100,
        batch_size=32,
        patience=15,
        verbose=verbose
    )
    
    # Save model
    if verbose:
        print(f"\nSaving model to {output_path}...")
    
    predictor.save(output_path)
    
    # Also save as best model
    best_model_path = output_path.replace('.pkl', '_best.pkl')
    predictor.save(best_model_path)
    
    if verbose:
        print(f"Best model saved to: {best_model_path}")
    
    # Test prediction on a sample
    if verbose:
        print(f"\nSample predictions:")
        test_sequences = X[:5]
        test_actual = y[:5]
        
        for i in range(min(5, len(test_sequences))):
            seq_norm = predictor._normalize(test_sequences[i])
            pred_norm = predictor.predict(seq_norm)
            pred = predictor._denormalize(pred_norm)
            print(f"  Actual: {test_actual[i]:.2f} Mbps, Predicted: {pred[0]:.2f} Mbps")
    
    return predictor


# Backward compatibility: SimpleLSTM wrapper for existing code
class SimpleLSTM:
    """
    Backward-compatible wrapper for LSTMPredictor.
    
    This allows existing code using SimpleLSTM to work with the new LSTM model.
    """
    
    def __init__(self, sequence_length=10, alpha=0.3):
        """Initialize with backward-compatible interface."""
        self.sequence_length = sequence_length
        self.alpha = alpha  # Not used, kept for compatibility
        self._predictor = None
        self.history = deque(maxlen=sequence_length)
        self.trained = False
        self.stats = {
            'mean': 0,
            'std': 1,
            'min': 0,
            'max': 100
        }
    
    def fit(self, X, y):
        """Train the model."""
        self._predictor = LSTMPredictor(
            sequence_length=self.sequence_length,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001
        )
        
        self._predictor.fit(X, y, test_size=0.2, epochs=50, verbose=False)
        self.stats = self._predictor.stats.copy()
        self.trained = True
        return self
    
    def predict(self, X):
        """Predict bandwidth."""
        if self._predictor is None or not self.trained:
            # Fallback to last value
            if isinstance(X, np.ndarray):
                if len(X.shape) == 1:
                    return X[-1]
                return np.array([seq[-1] for seq in X])
            return X[-1]
        
        X_norm = self._predictor._normalize(X)
        pred_norm = self._predictor.predict(X_norm)
        return self._predictor._denormalize(pred_norm)
    
    def update(self, bandwidth_mbps):
        """Update history."""
        self.history.append(bandwidth_mbps)
        if self._predictor is not None:
            self._predictor.update(bandwidth_mbps)
    
    def predict_next(self):
        """Predict next bandwidth."""
        if self._predictor is None or not self.trained:
            return np.mean(list(self.history)) if len(self.history) > 0 else self.stats['mean']
        
        return self._predictor.predict_next()
    
    def save(self, path):
        """Save model."""
        if self._predictor is not None:
            self._predictor.save(path)
        else:
            # Save basic stats for compatibility
            with open(path, 'wb') as f:
                pickle.dump({
                    'sequence_length': self.sequence_length,
                    'alpha': self.alpha,
                    'trained': self.trained,
                    'stats': self.stats
                }, f)
    
    def load(self, path):
        """Load model."""
        try:
            # Try loading as new LSTM model
            self._predictor = LSTMPredictor(sequence_length=self.sequence_length)
            self._predictor.load(path)
            self.stats = self._predictor.stats.copy()
            self.trained = self._predictor.trained
            self.sequence_length = self._predictor.sequence_length
            self.history = deque(maxlen=self.sequence_length)
        except (KeyError, TypeError):
            # Fall back to old format
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.sequence_length = data['sequence_length']
                self.alpha = data.get('alpha', 0.3)
                self.trained = data['trained']
                self.stats = data['stats']
                self.history = deque(maxlen=self.sequence_length)
                self._predictor = None
        return self


if __name__ == "__main__":
    import sys
    
    # Get project root directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
    except NameError:
        project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    
    bandwidth_dir = os.path.join(project_root, 'bandwith')
    model_dir = os.path.join(project_root, 'models')
    output_path = os.path.join(model_dir, 'bandwidth_lstm.pkl')
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(bandwidth_dir):
        print(f"Error: Bandwidth directory not found at {bandwidth_dir}")
        sys.exit(1)
    
    print("="*80)
    print("LSTM Bandwidth Prediction Model Training")
    print("="*80)
    
    model = train_lstm_model(bandwidth_dir, output_path, tune=True)
    
    print("\n" + "="*80)
    print("✅ Model trained and saved successfully!")
    print("="*80)
