#!/usr/bin/env python3
"""
LSTM Bandwidth Prediction Model Training for Google Colab

This is a self-contained script that can be run on Google Colab to train
an LSTM model for bandwidth prediction with hyperparameter tuning and
visualization of results.

Usage in Google Colab:
    1. Upload this file to Colab or copy/paste into a notebook cell
    2. Upload your bandwidth log files to the 'bandwith/' folder (note: folder name matches existing repository)
    3. Run the script

The script will:
    - Load and prepare the bandwidth dataset
    - Perform hyperparameter grid search
    - Train the best model
    - Plot hyperparameter search results
    - Plot training curves
    - Plot prediction results
    - Save the trained model
"""

# ===========================
# 1. IMPORTS AND SETUP
# ===========================

import os
import sys
import numpy as np
import pickle
import json
import copy
from collections import deque
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("Running locally (not in Colab)")

# Install required packages if in Colab
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                   'torch', 'scikit-learn', 'matplotlib', 'seaborn'])

# Import ML libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Installing scikit-learn...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
    from sklearn.model_selection import train_test_split

# Import plotting libraries
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    print("Installing seaborn...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'seaborn'])
    import seaborn as sns
    sns.set_theme(style="whitegrid")

# ===========================
# 2. CONSTANTS AND CONFIG
# ===========================

# Model training constants
GRADIENT_CLIP_MAX_NORM = 1.0
MAPE_ZERO_THRESHOLD = 0.1  # Mbps

# Default hyperparameter search space
DEFAULT_PARAM_GRID = {
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.0005]
}

# Random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ===========================
# 3. PYTORCH DATASET
# ===========================

class BandwidthDataset(Dataset):
    """PyTorch Dataset for bandwidth sequences."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===========================
# 4. LSTM MODEL ARCHITECTURE
# ===========================

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


# ===========================
# 5. LSTM PREDICTOR CLASS
# ===========================

class LSTMPredictor:
    """
    LSTM-based bandwidth predictor with training and inference capabilities.
    """
    
    def __init__(self, sequence_length=10, hidden_size=64, num_layers=2, 
                 dropout=0.2, learning_rate=0.001, device=None):
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
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = BandwidthLSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Statistics for normalization
        self.stats = {'mean': 0, 'std': 1, 'min': 0, 'max': 100}
        
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
        """Train the LSTM model with train/test split."""
        X = np.array(X)
        y = np.array(y)
        
        # Compute normalization statistics
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
            X_norm, y_norm, test_size=test_size, random_state=RANDOM_SEED
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM
                )
                
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
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"Loaded best model from epoch {self.training_history['best_epoch']+1}")
        
        self.trained = True
        
        # Compute final metrics on test set
        test_predictions = self.predict(X_test)
        y_test_denorm = self._denormalize(y_test)
        test_predictions_denorm = self._denormalize(test_predictions)
        
        mae = np.mean(np.abs(y_test_denorm - test_predictions_denorm))
        nonzero_mask = np.abs(y_test_denorm) > MAPE_ZERO_THRESHOLD
        if np.sum(nonzero_mask) > 0:
            mape = np.mean(np.abs(
                (y_test_denorm[nonzero_mask] - test_predictions_denorm[nonzero_mask]) 
                / y_test_denorm[nonzero_mask]
            )) * 100
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
        """Predict bandwidth for input sequences."""
        self.model.eval()
        
        if isinstance(X, list):
            X = np.array(X)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def update(self, bandwidth_mbps):
        """Update history with a new bandwidth measurement."""
        self.history.append(bandwidth_mbps)
    
    def predict_next(self):
        """Predict next bandwidth value based on current history."""
        if len(self.history) < self.sequence_length:
            return np.mean(list(self.history)) if len(self.history) > 0 else self.stats['mean']
        
        seq = np.array(list(self.history))
        seq_norm = self._normalize(seq)
        pred_norm = self.predict(seq_norm)
        pred = self._denormalize(pred_norm)
        pred = np.clip(pred, self.stats['min'], self.stats['max'])
        
        return float(pred[0]) if isinstance(pred, np.ndarray) else float(pred)
    
    def save(self, path):
        """Save model to file."""
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
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
        """Load model from file."""
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
        
        self.model = BandwidthLSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.history = deque(maxlen=self.sequence_length)
        
        return self


# ===========================
# 6. DATA LOADING FUNCTIONS
# ===========================

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
    
    log_files = [f for f in os.listdir(bandwidth_dir) if f.endswith('.log')]
    print(f"Found {len(log_files)} bandwidth trace files")
    
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


# ===========================
# 7. HYPERPARAMETER TUNING
# ===========================

def tune_hyperparameters(X, y, param_grid, test_size=0.2, epochs=50, verbose=True):
    """
    Perform hyperparameter tuning using grid search.
    
    Returns:
        best_params: Best hyperparameter combination
        best_metrics: Metrics for best model
        all_results: All hyperparameter combinations and their results
    """
    all_results = []
    best_params = None
    best_metrics = None
    best_mae = float('inf')
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    combinations = list(product(*param_values))
    total_combinations = len(combinations)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing {total_combinations} hyperparameter combinations...")
        print(f"{'='*60}")
    
    for idx, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        if verbose:
            print(f"\n[{idx+1}/{total_combinations}] Testing: {params}")
        
        # Create and train model
        predictor = LSTMPredictor(
            sequence_length=X.shape[1],
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
            'metrics': metrics,
            'training_history': predictor.training_history.copy()
        }
        all_results.append(result)
        
        if verbose:
            print(f"  MAE: {metrics['mae']:.4f}, MAPE: {metrics['mape']:.2f}%, "
                  f"RMSE: {metrics['rmse']:.4f}")
        
        # Check if this is the best model
        if metrics['mae'] < best_mae:
            best_mae = metrics['mae']
            best_params = params
            best_metrics = metrics
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Best hyperparameters: {best_params}")
        print(f"Best MAE: {best_mae:.4f} Mbps")
        print(f"{'='*60}")
    
    return best_params, best_metrics, all_results


# ===========================
# 8. PLOTTING FUNCTIONS
# ===========================

def plot_hyperparameter_search_results(all_results, save_path=None):
    """
    Plot comprehensive hyperparameter search results.
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Extract data for plotting
    hidden_sizes = sorted(set(r['params']['hidden_size'] for r in all_results))
    num_layers_list = sorted(set(r['params']['num_layers'] for r in all_results))
    dropouts = sorted(set(r['params']['dropout'] for r in all_results))
    learning_rates = sorted(set(r['params']['learning_rate'] for r in all_results))
    
    metrics_data = {
        'hidden_size': [],
        'num_layers': [],
        'dropout': [],
        'learning_rate': [],
        'mae': [],
        'mape': [],
        'rmse': []
    }
    
    for r in all_results:
        metrics_data['hidden_size'].append(r['params']['hidden_size'])
        metrics_data['num_layers'].append(r['params']['num_layers'])
        metrics_data['dropout'].append(r['params']['dropout'])
        metrics_data['learning_rate'].append(r['params']['learning_rate'])
        metrics_data['mae'].append(r['metrics']['mae'])
        metrics_data['mape'].append(r['metrics']['mape'])
        metrics_data['rmse'].append(r['metrics']['rmse'])
    
    # 1. Heatmap: Hidden Size vs Num Layers (averaged over other params)
    ax1 = fig.add_subplot(2, 3, 1)
    heatmap_data = np.zeros((len(num_layers_list), len(hidden_sizes)))
    for i, nl in enumerate(num_layers_list):
        for j, hs in enumerate(hidden_sizes):
            maes = [r['metrics']['mae'] for r in all_results 
                    if r['params']['num_layers'] == nl and r['params']['hidden_size'] == hs]
            heatmap_data[i, j] = np.mean(maes) if maes else 0
    
    sns.heatmap(heatmap_data, ax=ax1, annot=True, fmt='.3f', cmap='YlOrRd_r',
                xticklabels=hidden_sizes, yticklabels=num_layers_list)
    ax1.set_xlabel('Hidden Size')
    ax1.set_ylabel('Number of Layers')
    ax1.set_title('MAE by Hidden Size vs Num Layers\n(Lower is Better)')
    
    # 2. Heatmap: Dropout vs Learning Rate (averaged over other params)
    ax2 = fig.add_subplot(2, 3, 2)
    heatmap_data2 = np.zeros((len(dropouts), len(learning_rates)))
    for i, do in enumerate(dropouts):
        for j, lr in enumerate(learning_rates):
            maes = [r['metrics']['mae'] for r in all_results 
                    if r['params']['dropout'] == do and r['params']['learning_rate'] == lr]
            heatmap_data2[i, j] = np.mean(maes) if maes else 0
    
    sns.heatmap(heatmap_data2, ax=ax2, annot=True, fmt='.3f', cmap='YlOrRd_r',
                xticklabels=[f'{lr:.4f}' for lr in learning_rates], 
                yticklabels=[f'{d:.1f}' for d in dropouts])
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Dropout')
    ax2.set_title('MAE by Dropout vs Learning Rate\n(Lower is Better)')
    
    # 3. Bar plot: MAE by Hidden Size
    ax3 = fig.add_subplot(2, 3, 3)
    mae_by_hs = {hs: [] for hs in hidden_sizes}
    for r in all_results:
        mae_by_hs[r['params']['hidden_size']].append(r['metrics']['mae'])
    
    means = [np.mean(mae_by_hs[hs]) for hs in hidden_sizes]
    stds = [np.std(mae_by_hs[hs]) for hs in hidden_sizes]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(hidden_sizes)))
    
    bars = ax3.bar(range(len(hidden_sizes)), means, yerr=stds, capsize=5, color=colors)
    ax3.set_xticks(range(len(hidden_sizes)))
    ax3.set_xticklabels(hidden_sizes)
    ax3.set_xlabel('Hidden Size')
    ax3.set_ylabel('MAE (Mbps)')
    ax3.set_title('MAE by Hidden Size\n(with std dev)')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Line plot: Effect of Dropout
    ax4 = fig.add_subplot(2, 3, 4)
    for lr in learning_rates:
        mae_by_dropout = []
        for do in dropouts:
            maes = [r['metrics']['mae'] for r in all_results 
                    if r['params']['dropout'] == do and r['params']['learning_rate'] == lr]
            mae_by_dropout.append(np.mean(maes) if maes else 0)
        ax4.plot(dropouts, mae_by_dropout, 'o-', label=f'LR={lr}', linewidth=2, markersize=8)
    
    ax4.set_xlabel('Dropout Rate')
    ax4.set_ylabel('MAE (Mbps)')
    ax4.set_title('Effect of Dropout Rate on MAE')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Scatter plot: All configurations ranked by MAE
    ax5 = fig.add_subplot(2, 3, 5)
    sorted_results = sorted(all_results, key=lambda x: x['metrics']['mae'])
    maes = [r['metrics']['mae'] for r in sorted_results]
    mapes = [r['metrics']['mape'] for r in sorted_results]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(maes)))
    scatter = ax5.scatter(range(len(maes)), maes, c=colors, s=100, alpha=0.8)
    
    # Highlight top 3
    for i in range(min(3, len(maes))):
        ax5.annotate(f'#{i+1}', (i, maes[i]), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')
    
    ax5.set_xlabel('Configuration Rank')
    ax5.set_ylabel('MAE (Mbps)')
    ax5.set_title('All Configurations Ranked by MAE\n(Green=Best, Red=Worst)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Parallel coordinates-style plot for top 5 configurations
    ax6 = fig.add_subplot(2, 3, 6)
    top_5 = sorted_results[:5]
    
    categories = ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 'MAE']
    x = np.arange(len(categories))
    
    # Normalize values for visualization
    for i, result in enumerate(top_5):
        params = result['params']
        # Normalize each parameter to 0-1 scale for visualization
        values = [
            (params['hidden_size'] - min(hidden_sizes)) / (max(hidden_sizes) - min(hidden_sizes) + 1e-8),
            (params['num_layers'] - min(num_layers_list)) / (max(num_layers_list) - min(num_layers_list) + 1e-8),
            (params['dropout'] - min(dropouts)) / (max(dropouts) - min(dropouts) + 1e-8),
            params['learning_rate'] / max(learning_rates),
            result['metrics']['mae'] / max(maes)
        ]
        ax6.plot(x, values, 'o-', linewidth=2, markersize=10, alpha=0.8,
                label=f"#{i+1}: MAE={result['metrics']['mae']:.3f}")
    
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories, rotation=45, ha='right')
    ax6.set_ylabel('Normalized Value')
    ax6.set_title('Top 5 Configurations Comparison')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Hyperparameter search plot saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_training_history(training_history, save_path=None):
    """
    Plot training and validation loss curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss curves
    ax1 = axes[0]
    epochs = range(1, len(training_history['train_loss']) + 1)
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, training_history['test_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    # Mark best epoch
    best_epoch = training_history['best_epoch'] + 1
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, 
                label=f'Best Epoch ({best_epoch})')
    ax1.scatter([best_epoch], [training_history['best_test_loss']], 
               color='g', s=100, zorder=5, marker='*')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale loss curves
    ax2 = axes[1]
    ax2.semilogy(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.semilogy(epochs, training_history['test_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, 
                label=f'Best Epoch ({best_epoch})')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (MSE) - Log Scale')
    ax2.set_title('Training and Validation Loss (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_predictions(X_test, y_test, predictor, num_samples=100, save_path=None):
    """
    Plot model predictions vs actual values.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Normalize and predict
    X_test_norm = predictor._normalize(X_test)
    predictions_norm = predictor.predict(X_test_norm)
    predictions = predictor._denormalize(predictions_norm)
    
    # Plot 1: Scatter plot - Predicted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(y_test[:num_samples], predictions[:num_samples], alpha=0.5, s=30)
    
    # Perfect prediction line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Bandwidth (Mbps)')
    ax1.set_ylabel('Predicted Bandwidth (Mbps)')
    ax1.set_title('Predicted vs Actual Bandwidth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time series comparison
    ax2 = axes[0, 1]
    sample_size = min(200, len(y_test))
    ax2.plot(range(sample_size), y_test[:sample_size], 'b-', alpha=0.7, 
             label='Actual', linewidth=1.5)
    ax2.plot(range(sample_size), predictions[:sample_size], 'r--', alpha=0.7, 
             label='Predicted', linewidth=1.5)
    ax2.fill_between(range(sample_size), y_test[:sample_size], predictions[:sample_size],
                     alpha=0.2, color='gray')
    
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Bandwidth (Mbps)')
    ax2.set_title('Time Series Comparison (First 200 Samples)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    ax3 = axes[1, 0]
    errors = predictions[:num_samples] - y_test[:num_samples]
    ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax3.axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2, 
                label=f'Mean Error ({np.mean(errors):.3f})')
    
    ax3.set_xlabel('Prediction Error (Mbps)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Prediction Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error vs actual value
    ax4 = axes[1, 1]
    abs_errors = np.abs(errors)
    scatter = ax4.scatter(y_test[:num_samples], abs_errors, 
                          c=abs_errors, cmap='RdYlGn_r', alpha=0.6, s=30)
    plt.colorbar(scatter, ax=ax4, label='Absolute Error')
    
    ax4.set_xlabel('Actual Bandwidth (Mbps)')
    ax4.set_ylabel('Absolute Error (Mbps)')
    ax4.set_title('Prediction Error vs Actual Bandwidth')
    ax4.grid(True, alpha=0.3)
    
    # Add summary statistics as text
    mae = np.mean(np.abs(y_test - predictions))
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    nonzero_mask = np.abs(y_test) > MAPE_ZERO_THRESHOLD
    mape = np.mean(np.abs((y_test[nonzero_mask] - predictions[nonzero_mask]) 
                          / y_test[nonzero_mask])) * 100
    
    stats_text = f'MAE: {mae:.4f} Mbps\nRMSE: {rmse:.4f} Mbps\nMAPE: {mape:.2f}%'
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Predictions plot saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_best_config_summary(best_params, best_metrics, all_results, save_path=None):
    """
    Plot a summary of the best configuration found.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Best hyperparameters bar chart
    ax1 = axes[0]
    params_display = {
        'Hidden Size': best_params['hidden_size'],
        'Num Layers': best_params['num_layers'],
        'Dropout': best_params['dropout'],
        'Learning Rate': best_params['learning_rate'] * 1000  # Scale for visibility
    }
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(params_display)))
    bars = ax1.bar(params_display.keys(), params_display.values(), color=colors)
    ax1.set_ylabel('Value')
    ax1.set_title('Best Hyperparameters\n(LR × 1000 for visibility)')
    
    # Add value labels
    for bar, (key, val) in zip(bars, params_display.items()):
        if key == 'Learning Rate':
            label = f'{val/1000:.4f}'
        else:
            label = f'{val}'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Best metrics
    ax2 = axes[1]
    metrics_display = {
        'MAE\n(Mbps)': best_metrics['mae'],
        'MAPE\n(%)': best_metrics['mape'],
        'RMSE\n(Mbps)': best_metrics['rmse']
    }
    
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(metrics_display)))
    bars = ax2.bar(metrics_display.keys(), metrics_display.values(), color=colors)
    ax2.set_ylabel('Value')
    ax2.set_title('Best Model Performance Metrics')
    
    for bar, val in zip(bars, metrics_display.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: Comparison with all configurations
    ax3 = axes[2]
    all_maes = sorted([r['metrics']['mae'] for r in all_results])
    
    ax3.hist(all_maes, bins=15, edgecolor='black', alpha=0.7, color='lightblue',
             label='All Configurations')
    ax3.axvline(x=best_metrics['mae'], color='r', linestyle='--', linewidth=3,
                label=f"Best: {best_metrics['mae']:.3f}")
    ax3.axvline(x=np.median(all_maes), color='orange', linestyle=':', linewidth=2,
                label=f"Median: {np.median(all_maes):.3f}")
    
    ax3.set_xlabel('MAE (Mbps)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of MAE Across All Configurations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Summary plot saved to: {save_path}")
    
    plt.show()
    
    return fig


# ===========================
# 9. MAIN TRAINING FUNCTION
# ===========================

def train_lstm_model(bandwidth_dir, output_path, sequence_length=10, 
                     param_grid=None, tune=True, verbose=True):
    """
    Train LSTM model on bandwidth traces with hyperparameter tuning.
    
    Args:
        bandwidth_dir: Directory containing bandwidth log files
        output_path: Path to save trained model
        sequence_length: Number of historical samples to use
        param_grid: Hyperparameter search space (dict)
        tune: Whether to perform hyperparameter tuning
        verbose: Print progress
        
    Returns:
        predictor: Trained LSTMPredictor
        all_results: Results from hyperparameter search (if tune=True)
    """
    if verbose:
        print(f"\n{'='*70}")
        print("LSTM Bandwidth Prediction Model Training")
        print(f"{'='*70}")
        print(f"Loading bandwidth traces from: {bandwidth_dir}")
    
    # Prepare dataset
    X, y = prepare_dataset(bandwidth_dir, sequence_length)
    
    if verbose:
        print(f"Dataset size: {len(X)} sequences")
        print(f"Sequence length: {sequence_length}")
        print(f"Bandwidth range: {y.min():.2f} - {y.max():.2f} Mbps")
        print(f"Mean bandwidth: {y.mean():.2f} Mbps, Std: {y.std():.2f} Mbps")
    
    all_results = None
    best_params = None
    
    if tune:
        # Use default param grid if none provided
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID
        
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
            # Convert to JSON serializable format
            json_results = {
                'best_params': best_params,
                'best_metrics': best_metrics,
                'all_results': [{
                    'params': r['params'],
                    'metrics': r['metrics']
                } for r in all_results]
            }
            json.dump(json_results, f, indent=2)
        
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
        print("\n" + "="*70)
        print("Training final model...")
        print("="*70)
    
    metrics = predictor.fit(
        X, y,
        test_size=0.2,
        epochs=100,
        batch_size=32,
        patience=15,
        verbose=verbose
    )
    
    # Save model
    predictor.save(output_path)
    
    # Also save as best model
    best_model_path = output_path.replace('.pkl', '_best.pkl')
    predictor.save(best_model_path)
    
    if verbose:
        print(f"\n{'='*70}")
        print("✅ Training Complete!")
        print(f"{'='*70}")
        print(f"Model saved to: {output_path}")
        print(f"Best model saved to: {best_model_path}")
    
    return predictor, all_results, X, y


# ===========================
# 10. COLAB HELPER FUNCTIONS
# ===========================

def setup_colab_data():
    """
    Setup data directory in Colab environment.
    Creates sample data directory structure.
    """
    if IN_COLAB:
        # Mount Google Drive if needed
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted successfully")
        except Exception as e:
            print(f"Could not mount Drive: {e}")
            print("Continuing without Drive...")
    
    # Create bandwidth directory if it doesn't exist
    # Note: Directory named 'bandwith' to match existing repository structure
    bandwidth_dir = 'bandwith'
    os.makedirs(bandwidth_dir, exist_ok=True)
    
    print(f"\nPlease upload your bandwidth log files to: {bandwidth_dir}/")
    print("Files should be in the format: timestamp ms_since_start lat lon bytes_received ms_interval")
    
    return bandwidth_dir


def check_data_available(bandwidth_dir):
    """
    Check if bandwidth data files are available.
    """
    if not os.path.exists(bandwidth_dir):
        return False, []
    
    log_files = [f for f in os.listdir(bandwidth_dir) if f.endswith('.log')]
    return len(log_files) > 0, log_files


# ===========================
# 11. MAIN EXECUTION
# ===========================

def main():
    """
    Main training function with all plots and outputs.
    """
    print("="*70)
    print("🚀 LSTM Bandwidth Prediction - Colab Training Script")
    print("="*70)
    
    # Setup paths
    if IN_COLAB:
        bandwidth_dir = setup_colab_data()
        model_dir = 'models'
    else:
        # Local path (relative to this script)
        # Note: 'bandwith' folder name matches existing repository structure
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
        bandwidth_dir = os.path.join(script_dir, 'bandwith')
        model_dir = os.path.join(script_dir, 'models')
    
    # Create output directories
    os.makedirs(model_dir, exist_ok=True)
    plots_dir = os.path.join(model_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    output_path = os.path.join(model_dir, 'bandwidth_lstm.pkl')
    
    # Check for data
    data_available, log_files = check_data_available(bandwidth_dir)
    
    if not data_available:
        print(f"\n❌ No bandwidth log files found in: {bandwidth_dir}")
        print("\nPlease upload your bandwidth log files (.log format) to this directory.")
        if IN_COLAB:
            print("\nYou can use the Colab file upload feature:")
            print("  from google.colab import files")
            print("  files.upload()")
        return
    
    print(f"\n✅ Found {len(log_files)} bandwidth trace files")
    print(f"📁 Data directory: {bandwidth_dir}")
    print(f"💾 Model output: {output_path}")
    print(f"📊 Plots directory: {plots_dir}")
    
    # Configuration
    SEQUENCE_LENGTH = 10
    PARAM_GRID = {
        'hidden_size': [32, 64, 128],
        'num_layers': [1, 2],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0005]
    }
    
    print(f"\n📋 Configuration:")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Hyperparameter Grid: {PARAM_GRID}")
    
    # Train model with hyperparameter tuning
    predictor, all_results, X, y = train_lstm_model(
        bandwidth_dir=bandwidth_dir,
        output_path=output_path,
        sequence_length=SEQUENCE_LENGTH,
        param_grid=PARAM_GRID,
        tune=True,
        verbose=True
    )
    
    # Get best parameters and metrics
    if all_results:
        best_result = min(all_results, key=lambda x: x['metrics']['mae'])
        best_params = best_result['params']
        best_metrics = best_result['metrics']
        
        print("\n" + "="*70)
        print("📊 GENERATING VISUALIZATION PLOTS")
        print("="*70)
        
        # Plot 1: Hyperparameter search results
        print("\n📈 Plot 1: Hyperparameter Search Results")
        plot_hyperparameter_search_results(
            all_results,
            save_path=os.path.join(plots_dir, 'hyperparameter_search.png')
        )
        
        # Plot 2: Best configuration summary
        print("\n📈 Plot 2: Best Configuration Summary")
        plot_best_config_summary(
            best_params, best_metrics, all_results,
            save_path=os.path.join(plots_dir, 'best_config_summary.png')
        )
    
    # Plot 3: Training history
    print("\n📈 Plot 3: Training History")
    plot_training_history(
        predictor.training_history,
        save_path=os.path.join(plots_dir, 'training_history.png')
    )
    
    # Split data for prediction plot
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Plot 4: Predictions
    print("\n📈 Plot 4: Model Predictions")
    plot_predictions(
        X_test, y_test, predictor,
        num_samples=min(500, len(X_test)),
        save_path=os.path.join(plots_dir, 'predictions.png')
    )
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📦 Saved Files:")
    print(f"  - Model: {output_path}")
    print(f"  - Best Model: {output_path.replace('.pkl', '_best.pkl')}")
    print(f"  - Tuning Results: {output_path.replace('.pkl', '_tuning_results.json')}")
    print(f"  - Plots: {plots_dir}/")
    
    if all_results:
        print(f"\n🏆 Best Configuration:")
        print(f"  - Hidden Size: {best_params['hidden_size']}")
        print(f"  - Num Layers: {best_params['num_layers']}")
        print(f"  - Dropout: {best_params['dropout']}")
        print(f"  - Learning Rate: {best_params['learning_rate']}")
        print(f"\n📊 Best Metrics:")
        print(f"  - MAE: {best_metrics['mae']:.4f} Mbps")
        print(f"  - MAPE: {best_metrics['mape']:.2f}%")
        print(f"  - RMSE: {best_metrics['rmse']:.4f} Mbps")
    
    print("\n" + "="*70)
    
    # Return objects for further exploration
    return predictor, all_results, X, y


# Run main when executed as script
if __name__ == "__main__":
    predictor, all_results, X, y = main()
