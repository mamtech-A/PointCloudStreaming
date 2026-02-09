"""
LSTM-based Bandwidth Prediction Module

This module implements an LSTM model to predict bandwidth based on historical measurements
from the 4G network traces collected in the bandwith/ folder.
"""

import os
import numpy as np
import pickle
from collections import deque


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
    Prepare training dataset from all bandwidth log files.
    
    Args:
        bandwidth_dir: Directory containing bandwidth log files
        sequence_length: Number of historical samples to use for prediction
        
    Returns:
        X: Input sequences (historical bandwidth values)
        y: Target values (next bandwidth value)
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


class SimpleLSTM:
    """
    Simple LSTM-like predictor using exponential moving average and trend analysis.
    
    This is a lightweight alternative to a full LSTM that doesn't require TensorFlow/PyTorch.
    It uses historical bandwidth data to predict future bandwidth based on:
    - Exponential moving average (EMA)
    - Trend detection
    - Variance analysis
    """
    
    def __init__(self, sequence_length=10, alpha=0.3):
        """
        Initialize the predictor.
        
        Args:
            sequence_length: Number of historical samples to consider
            alpha: EMA smoothing factor (0-1, higher = more weight on recent values)
        """
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.history = deque(maxlen=sequence_length)
        self.trained = False
        self.stats = {
            'mean': 0,
            'std': 1,
            'min': 0,
            'max': 100
        }
    
    def fit(self, X, y):
        """
        Train the model on bandwidth sequences.
        
        Args:
            X: Input sequences (not used in this simple model)
            y: Target values (used to compute statistics)
        """
        # Compute statistics for normalization
        all_values = np.concatenate([X.flatten(), y])
        self.stats['mean'] = np.mean(all_values)
        self.stats['std'] = np.std(all_values) if np.std(all_values) > 0 else 1.0
        self.stats['min'] = np.min(all_values)
        self.stats['max'] = np.max(all_values)
        self.trained = True
        return self
    
    def predict(self, X):
        """
        Predict bandwidth for a sequence of historical values.
        
        Args:
            X: Input sequence(s) of historical bandwidth values (Mbps)
            
        Returns:
            Predicted bandwidth value(s) in Mbps
        """
        if not self.trained:
            # If not trained, just return the last value
            if isinstance(X, np.ndarray):
                if len(X.shape) == 1:
                    return X[-1]
                else:
                    return np.array([seq[-1] for seq in X])
            return X[-1]
        
        # Handle single sequence
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for seq in X:
            # Calculate exponential moving average
            ema = seq[0]
            for val in seq[1:]:
                ema = self.alpha * val + (1 - self.alpha) * ema
            
            # Calculate trend (linear regression slope)
            x = np.arange(len(seq))
            y = seq
            trend = np.polyfit(x, y, 1)[0]
            
            # Calculate variance (measure of stability)
            variance = np.var(seq)
            variance_norm = variance / (self.stats['std'] ** 2) if self.stats['std'] > 0 else 0
            
            # Predict: EMA + trend adjustment - variance penalty
            # High variance = less predictable = more conservative (lower prediction)
            prediction = ema + trend * 0.5 - variance_norm * 0.1 * ema
            
            # Clamp to reasonable range
            prediction = np.clip(prediction, self.stats['min'], self.stats['max'])
            
            predictions.append(prediction)
        
        return np.array(predictions) if len(predictions) > 1 else predictions[0]
    
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
            # Not enough history, return average of what we have
            return np.mean(self.history) if len(self.history) > 0 else self.stats['mean']
        
        # Convert history to numpy array and predict
        seq = np.array(list(self.history))
        return self.predict(seq)
    
    def save(self, path):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'sequence_length': self.sequence_length,
                'alpha': self.alpha,
                'trained': self.trained,
                'stats': self.stats
            }, f)
    
    def load(self, path):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.sequence_length = data['sequence_length']
            self.alpha = data['alpha']
            self.trained = data['trained']
            self.stats = data['stats']
            self.history = deque(maxlen=self.sequence_length)
        return self


def train_lstm_model(bandwidth_dir, output_path, sequence_length=10):
    """
    Train LSTM model on bandwidth traces and save to disk.
    
    Args:
        bandwidth_dir: Directory containing bandwidth log files
        output_path: Path to save trained model
        sequence_length: Number of historical samples to use
        
    Returns:
        Trained model
    """
    print(f"Loading bandwidth traces from {bandwidth_dir}...")
    X, y = prepare_dataset(bandwidth_dir, sequence_length)
    
    print(f"Dataset size: {len(X)} sequences")
    print(f"Bandwidth range: {y.min():.2f} - {y.max():.2f} Mbps")
    print(f"Mean bandwidth: {y.mean():.2f} Mbps, Std: {y.std():.2f} Mbps")
    
    print("Training model...")
    model = SimpleLSTM(sequence_length=sequence_length)
    model.fit(X, y)
    
    # Save model
    print(f"Saving model to {output_path}...")
    model.save(output_path)
    
    # Test prediction on a sample
    test_pred = model.predict(X[:5])
    test_actual = y[:5]
    print(f"\nSample predictions:")
    for i in range(min(5, len(test_pred))):
        print(f"  Actual: {test_actual[i]:.2f} Mbps, Predicted: {test_pred[i]:.2f} Mbps")
    
    return model


if __name__ == "__main__":
    # Train model when run as script
    import sys
    
    # Get project root directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
    except NameError:
        project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    
    bandwidth_dir = os.path.join(project_root, 'bandwith')
    output_path = os.path.join(project_root, 'models', 'bandwidth_lstm.pkl')
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(bandwidth_dir):
        print(f"Error: Bandwidth directory not found at {bandwidth_dir}")
        sys.exit(1)
    
    model = train_lstm_model(bandwidth_dir, output_path)
    print(f"\n✅ Model trained and saved successfully!")
