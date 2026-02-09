# LSTM-Based Bandwidth Prediction for Video Quality Selection

This project implements an LSTM-based bandwidth prediction system for adaptive video quality selection in point cloud streaming. The system uses historical 4G network measurements to predict future bandwidth and automatically selects one of three video quality levels to optimize frame rate (FPS).

## Overview

The system consists of two main components:

1. **LSTM Model Training** (`src/bandwidth_lstm.py`): Trains a bandwidth prediction model using historical network traces
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
pip install numpy
```

### 2. Train the LSTM Model

```bash
cd NewSystemModel
python3 src/bandwidth_lstm.py
```

This will:
- Load all 40 bandwidth traces from the `bandwith/` folder
- Create ~17,636 training sequences
- Train the LSTM model
- Save the model to `models/bandwidth_lstm.pkl`

Expected output:
```
Loading bandwidth traces from .../bandwith...
Dataset size: 17636 sequences
Bandwidth range: 0.00 - 110.97 Mbps
Mean bandwidth: 30.23 Mbps, Std: 16.65 Mbps
Training model...
✅ Model trained and saved successfully!
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

## How It Works

### LSTM Bandwidth Prediction

The `SimpleLSTM` class uses a lightweight approach that doesn't require TensorFlow or PyTorch:

1. **Exponential Moving Average (EMA)**: Smooths out recent bandwidth measurements
2. **Trend Detection**: Calculates the trend direction using linear regression
3. **Variance Analysis**: Adjusts predictions based on bandwidth stability

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
   Mean Absolute Error: 8.05 Mbps
   Mean Absolute % Error: 22.1%
```

### Output Files

The system generates three log files in `logs/` directory:
- `results.csv`: Per-frame results with quality, bandwidth, and buffer info
- `packets.log`: Detailed TCP packet transmission log
- `buffer.log`: Buffer state changes and rebuffering events

## Configuration

### LSTM Model Parameters

In `src/bandwidth_lstm.py`:
```python
sequence_length = 10  # Number of historical samples to use
alpha = 0.3          # EMA smoothing factor (higher = more weight on recent values)
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
├── src/
│   ├── bandwidth_lstm.py # LSTM model training and prediction
│   ├── network_model.py  # EdgeNodeLSTM and simulation
│   └── tcp_protocol.py   # TCP connection simulation
├── run.py               # Standard simulation (no LSTM)
└── run_lstm.py          # LSTM-based simulation
```

## Technical Details

### SimpleLSTM Algorithm

The prediction formula combines three components:
```
prediction = EMA + (trend × 0.5) - (variance_penalty × 0.1 × EMA)
```

Where:
- **EMA**: Exponential moving average of recent bandwidth
- **Trend**: Linear regression slope of the sequence
- **Variance penalty**: Reduces prediction when bandwidth is unstable

### Bandwidth History Management

The system maintains:
- `bandwidth_history`: All observed bandwidth measurements
- `prediction_history`: All LSTM predictions
- `quality_history`: Selected quality levels

This enables post-simulation analysis and accuracy tracking.

## Future Improvements

1. **Deep Learning**: Replace SimpleLSTM with TensorFlow/PyTorch LSTM for better accuracy
2. **Multi-step Prediction**: Predict multiple future time steps
3. **Buffer-aware Selection**: Incorporate buffer level into quality selection
4. **Online Learning**: Update model during runtime with new measurements
5. **Quality Switching Cost**: Add penalty for frequent quality changes

## References

- Original bandwidth dataset from research on HTTP Adaptive Streaming (HAS)
- Measurements collected in Ghent, Belgium (2015-2016)
- Based on methodology similar to Riiser et al.

## License

This project is part of the PointCloudStreaming repository.
