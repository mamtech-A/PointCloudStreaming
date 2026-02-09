# LSTM-Based Bandwidth Prediction for Video Quality Selection

This project implements an LSTM-based bandwidth prediction system for adaptive video quality selection in point cloud streaming. The system uses historical 4G network measurements to predict future bandwidth and automatically selects one of three video quality levels to optimize frame rate (FPS).

## Overview

The system consists of two main components:

1. **LSTM Model Training** (`src/lstm_model.py`): Trains a proper PyTorch LSTM bandwidth prediction model with:
   - Train/test split (80/20) for proper evaluation
   - Hyperparameter tuning via grid search
   - Early stopping to prevent overfitting
   - Best model saving based on test set performance

2. **LSTM-Enhanced Edge Node** (`src/network_model.py`): Uses the trained model to predict bandwidth and select appropriate video quality

## Dataset

The bandwidth measurements are located in the `bandwith/` folder. These are real 4G network traces collected in Ghent, Belgium during 2015-2016 using various transportation methods (foot, bicycle, bus, tram, train, car).

### Data Format

Each log file contains bandwidth measurements in the following format:
```
timestamp_ms time_since_start_ms latitude longitude bytes_received interval_ms
```

Example:
```
1453121790686 39287 51.0386528885778 3.73220785642186 2493100 1000
1453121791687 40288 51.0386532820580 3.73222847166054 2763504 1001
```

The throughput (bandwidth) is calculated as: `throughput_bps = (bytes_received * 8) / (interval_ms / 1000)`

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy torch scikit-learn
```

### 2. Train the LSTM Model

```bash
cd NewSystemModel
python3 train_model.py
```

This will:
- Load all 40 bandwidth traces from the `bandwith/` folder
- Split data into train (80%) and test (20%) sets
- Perform hyperparameter tuning via grid search
- Train the best LSTM model configuration
- Save the model to `models/bandwidth_lstm.pkl`
- Save the best model to `models/bandwidth_lstm_best.pkl`
- Save tuning results to `models/bandwidth_lstm_tuning_results.json`

#### Training Options

```bash
# Train with default hyperparameters (skip tuning for faster training)
python3 train_model.py --no-tune

# Customize sequence length
python3 train_model.py --sequence-length 15

# Customize maximum epochs
python3 train_model.py --epochs 150
```

Expected output:
```
================================================================================
🤖 LSTM Bandwidth Prediction Model Training
================================================================================

📁 Found 40 bandwidth trace files
📂 Output directory: models
🔧 Hyperparameter tuning: Enabled
📊 Sequence length: 10

================================================================================
Training Process:
  1. Load and prepare dataset
  2. Split into train (80%) and test (20%) sets
  3. Perform hyperparameter grid search
  4. Train final model with best hyperparameters
  5. Save trained model and best model
================================================================================

Loading bandwidth traces from .../bandwith...
Dataset size: 17636 sequences
Dataset split: 14108 train, 3528 test samples

Testing 36 hyperparameter combinations...
...

Best hyperparameters: {'hidden_size': 64, 'num_layers': 2, ...}
Best MAE: 5.23 Mbps

Final Test Metrics:
  MAE: 5.15 Mbps
  MAPE: 18.5%
  RMSE: 7.82 Mbps

✅ Training complete!
```

### 3. Run the System with LSTM Prediction

```bash
python3 run_lstm.py
```

This will:
- Load the trained LSTM model
- Simulate point cloud streaming with adaptive quality selection
- Use LSTM predictions to choose between 3 quality levels
- Generate detailed logs and statistics

## LSTM Model Architecture

The model uses a proper PyTorch LSTM with the following architecture:

```
Input (sequence_length timesteps) 
    ↓
LSTM Layer(s) [configurable: 1-2 layers, 32-128 hidden units]
    ↓
Dropout [configurable: 0.1-0.3]
    ↓
Fully Connected Layer [hidden_size → hidden_size/2]
    ↓
ReLU Activation
    ↓
Dropout
    ↓
Output Layer [hidden_size/2 → 1]
    ↓
Predicted Bandwidth
```

### Hyperparameter Tuning

The following hyperparameters are tuned via grid search:

| Parameter | Search Space | Description |
|-----------|--------------|-------------|
| `hidden_size` | [32, 64, 128] | Number of LSTM hidden units |
| `num_layers` | [1, 2] | Number of stacked LSTM layers |
| `dropout` | [0.1, 0.2, 0.3] | Dropout rate for regularization |
| `learning_rate` | [0.001, 0.0005] | Adam optimizer learning rate |

### Training Features

- **Train/Test Split**: 80% training, 20% testing for proper evaluation
- **Early Stopping**: Training stops if no improvement after 15 epochs
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **Best Model Saving**: Saves model with lowest test loss
- **Normalization**: Z-score normalization for numerical stability

## How It Works

### LSTM Bandwidth Prediction

The `LSTMPredictor` class uses PyTorch to build a proper LSTM model:

1. **Input**: Sequence of 10 normalized bandwidth measurements
2. **LSTM Processing**: Learns temporal patterns in bandwidth data
3. **Output**: Predicted next bandwidth value

The predictor maintains a sliding window of the last 10 bandwidth measurements and predicts the next value.

### Quality Level Selection

Based on predicted bandwidth, the system selects one of 3 quality levels:

| Quality Level | Bandwidth Range | Representation ID | Description |
|---------------|----------------|-------------------|-------------|
| **High** | > 20 Mbps | rep_id=0 | Best quality for high bandwidth |
| **Medium** | 5-20 Mbps | rep_id=1 | Medium quality for moderate bandwidth |
| **Low** | < 5 Mbps | rep_id=2 | Lowest quality for low bandwidth |

This mapping ensures:
- Stable video quality matching network conditions
- Better FPS by avoiding quality mismatches
- Smooth playback with minimal rebuffering

### EdgeNodeLSTM Class

The `EdgeNodeLSTM` class extends the basic `EdgeNode` with LSTM prediction:

```python
edge = EdgeNodeLSTM(
    server, 
    tcp_params=tcp_params,
    lstm_model_path="models/bandwidth_lstm.pkl",
    use_prediction=True  # Enable LSTM prediction
)
```

Key features:
- Maintains bandwidth history for prediction
- Predicts next bandwidth before each frame
- Selects quality based on prediction
- Tracks prediction accuracy (MAE, MAPE)

## Results

Running the system with LSTM prediction provides:

### Playback Statistics
- Total frames processed
- Real video FPS achieved
- Buffer health and statistics

### LSTM Prediction Statistics
Example output:
```
🤖 LSTM Prediction Statistics:
   Total Predictions: 291
   Mean Predicted BW: 40.70 Mbps
   Mean Actual BW: 42.84 Mbps
   Mean Absolute Error: 5.15 Mbps
   Mean Absolute % Error: 18.5%
```

### Output Files

The system generates several output files:

In `models/` directory (after training):
- `bandwidth_lstm.pkl`: Final trained LSTM model
- `bandwidth_lstm_best.pkl`: Best model from training (lowest test loss)
- `bandwidth_lstm_tuning_results.json`: Hyperparameter tuning results

In `logs/` directory (after running simulation):
- `results.csv`: Per-frame results with quality, bandwidth, and buffer info
- `packets.log`: Detailed TCP packet transmission log
- `buffer.log`: Buffer state changes and rebuffering events

## Configuration

### LSTM Model Parameters

In `src/lstm_model.py`, the following parameters can be configured:

```python
# Model architecture
hidden_size = 64       # Number of LSTM hidden units (tuned: 32, 64, 128)
num_layers = 2         # Number of stacked LSTM layers (tuned: 1, 2)
dropout = 0.2          # Dropout rate for regularization (tuned: 0.1, 0.2, 0.3)

# Training parameters
learning_rate = 0.001  # Adam optimizer learning rate (tuned: 0.001, 0.0005)
batch_size = 32        # Training batch size
epochs = 100           # Maximum training epochs
patience = 15          # Early stopping patience
sequence_length = 10   # Number of historical samples to use
```

### Quality Thresholds

In `src/network_model.py` (EdgeNodeLSTM class):
```python
# Modify _select_quality_level() to adjust thresholds
if bandwidth_mbps >= 20:    # High quality
    return rep_id_0
elif bandwidth_mbps >= 5:   # Medium quality
    return rep_id_1
else:                       # Low quality
    return rep_id_2
```

### Buffer Settings

In `run_lstm.py`:
```python
target_fps = 30.0         # Target playback FPS
buffer_capacity_s = 5.0   # Maximum buffer size in seconds
min_buffer_s = 1.0        # Minimum buffer before playback starts
```

## Comparison: With vs Without LSTM

### Without LSTM (baseline)
```bash
python3 run.py
```
Uses actual bandwidth measurements for quality selection.

### With LSTM
```bash
python3 run_lstm.py
```
Uses LSTM predictions to anticipate bandwidth changes and select quality proactively.

## Project Structure

```
NewSystemModel/
├── bandwith/              # 4G network traces (40 log files)
├── config/
│   └── mpd.xml           # Media presentation description
├── data/
│   └── bandwidth/        # Bandwidth data for simulation
├── models/               # Trained LSTM models (generated)
│   ├── bandwidth_lstm.pkl           # Final trained model
│   ├── bandwidth_lstm_best.pkl      # Best model from training
│   └── bandwidth_lstm_tuning_results.json  # Hyperparameter tuning results
├── src/
│   ├── lstm_model.py     # PyTorch LSTM model implementation
│   ├── bandwidth_lstm.py # Backward-compatible wrapper
│   ├── network_model.py  # EdgeNodeLSTM and simulation
│   └── tcp_protocol.py   # TCP connection simulation
├── train_model.py       # Training script with hyperparameter tuning
├── run.py               # Standard simulation (no LSTM)
└── run_lstm.py          # LSTM-based simulation
```

## Technical Details

### LSTM Model Architecture

The PyTorch LSTM model processes bandwidth sequences through:

1. **LSTM Layers**: Stacked LSTM cells with configurable hidden size and number of layers
2. **Dropout**: Applied between LSTM layers and in fully connected layers
3. **Fully Connected Layers**: Transform LSTM output to bandwidth prediction

The model learns temporal patterns in bandwidth data through backpropagation with:
- Adam optimizer with configurable learning rate
- MSE loss function for regression
- Gradient clipping (max_norm=1.0) to prevent exploding gradients

### Bandwidth History Management

The system maintains:
- `bandwidth_history`: All observed bandwidth measurements
- `prediction_history`: All LSTM predictions
- `quality_history`: Selected quality levels

This enables post-simulation analysis and accuracy tracking.

## Future Improvements

1. **Multi-step Prediction**: Predict multiple future time steps
2. **Buffer-aware Selection**: Incorporate buffer level into quality selection
3. **Online Learning**: Update model during runtime with new measurements
4. **Quality Switching Cost**: Add penalty for frequent quality changes
5. **Attention Mechanism**: Add attention layer for better temporal modeling

## References

- Original bandwidth dataset from research on HTTP Adaptive Streaming (HAS)
- Measurements collected in Ghent, Belgium (2015-2016)
- Based on methodology similar to Riiser et al.

## License

This project is part of the PointCloudStreaming repository.
