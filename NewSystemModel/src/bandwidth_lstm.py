"""
LSTM-based Bandwidth Prediction Module

This module implements an LSTM model to predict bandwidth based on historical measurements
from the 4G network traces collected in the bandwith/ folder.

The module now uses a proper PyTorch LSTM model with:
- Train/test split for proper evaluation
- Hyperparameter tuning via grid search
- Early stopping to prevent overfitting
- Best model saving
"""

import os
import numpy as np
import pickle
from collections import deque

# Import from the new LSTM model module
from lstm_model import (
    LSTMPredictor,
    SimpleLSTM,
    load_bandwidth_trace,
    prepare_dataset,
    train_lstm_model,
    tune_hyperparameters
)

# Re-export for backward compatibility
__all__ = [
    'LSTMPredictor',
    'SimpleLSTM', 
    'load_bandwidth_trace',
    'prepare_dataset',
    'train_lstm_model',
    'tune_hyperparameters'
]


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
    
    print("="*80)
    print("LSTM Bandwidth Prediction Model Training")
    print("="*80)
    
    # Train with hyperparameter tuning
    model = train_lstm_model(bandwidth_dir, output_path, tune=True)
    
    print("\n" + "="*80)
    print("✅ Model trained and saved successfully!")
    print("="*80)
