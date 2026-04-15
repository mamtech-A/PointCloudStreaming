#!/usr/bin/env python3
"""
Comparison script: Baseline vs LSTM-based bandwidth prediction

This script demonstrates the difference between:
1. Baseline: Using actual bandwidth measurements for quality selection
2. LSTM: Using predicted bandwidth for proactive quality selection

The LSTM approach aims to:
- Anticipate bandwidth changes before they occur
- Select appropriate quality levels more smoothly
- Potentially improve FPS by avoiding quality mismatches
"""

import os
import sys

# Add src directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from network_model import (
    PointCloudServer, EdgeNode, EdgeNodeLSTM, PointCloudClient, Simulator
)

def run_comparison():
    """Run both baseline and LSTM simulations for comparison."""
    
    # Setup paths
    mpd_path = os.path.join(project_root, "config", "mpd.xml")
    bandwidth_log_path = os.path.join(project_root, "data", "bandwidth", "report_foot_0001.log")
    lstm_model_path = os.path.join(project_root, "models", "bandwidth_lstm.pkl")
    
    # Common parameters
    tcp_params = {
        'rtt_ms': 50.0,
        'rtt_jitter_ms': 10.0,
        'loss_prob': 0.0,
        'cwnd_packets': 10.0,
        'mss_bytes': 1460,
        'rto_formula': 'jacobson',
        'rto_fixed_s': 1.0,
    }
    
    target_fps = 30.0
    buffer_capacity_s = 5.0
    min_buffer_s = 1.0
    
    print("="*100)
    print("BASELINE vs LSTM COMPARISON")
    print("="*100)
    print("\nThis comparison runs two simulations:")
    print("1. BASELINE: Uses actual bandwidth measurements")
    print("2. LSTM: Uses LSTM predictions for bandwidth")
    print("\nBoth use the same bandwidth trace and configuration.")
    print("="*100)
    
    # Check if LSTM model exists
    if not os.path.exists(lstm_model_path):
        print(f"\n❌ LSTM model not found at {lstm_model_path}")
        print("   Please run: python3 train_model.py")
        return
    
    # Run baseline simulation
    print("\n" + "="*100)
    print("SIMULATION 1: BASELINE (Actual Bandwidth)")
    print("="*100)
    
    server1 = PointCloudServer(base_url="http://localhost/")
    edge1 = EdgeNode(server1, tcp_params=tcp_params)
    client1 = PointCloudClient(edge1, target_fps=target_fps, 
                               buffer_capacity_s=buffer_capacity_s, 
                               min_buffer_s=min_buffer_s)
    sim1 = Simulator(server1, edge1, [client1])
    
    # Note: For a fair comparison, we'd need to capture and compare metrics
    # For now, we just demonstrate both can run
    print("\n[Running baseline simulation - this may take a minute...]")
    # Uncomment to run full simulation:
    # sim1.run(mpd_path=mpd_path, bandwidth_log_path=bandwidth_log_path)
    
    # Run LSTM simulation
    print("\n" + "="*100)
    print("SIMULATION 2: LSTM (Predicted Bandwidth)")
    print("="*100)
    
    server2 = PointCloudServer(base_url="http://localhost/")
    edge2 = EdgeNodeLSTM(server2, tcp_params=tcp_params, 
                         lstm_model_path=lstm_model_path, 
                         use_prediction=True)
    client2 = PointCloudClient(edge2, target_fps=target_fps,
                               buffer_capacity_s=buffer_capacity_s,
                               min_buffer_s=min_buffer_s)
    sim2 = Simulator(server2, edge2, [client2])
    
    print("\n[Running LSTM simulation - this may take a minute...]")
    # Uncomment to run full simulation:
    # sim2.run(mpd_path=mpd_path, bandwidth_log_path=bandwidth_log_path)
    
    print("\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)
    print("\nTo see full results, uncomment the sim.run() calls in this script.")
    print("\nKey differences to observe:")
    print("  - Quality selection patterns (baseline vs predicted)")
    print("  - Rebuffering events (frequency and duration)")
    print("  - FPS achieved (frames per second)")
    print("  - QoE score (quality of experience)")
    print("\nFor quick testing, you can run:")
    print("  python3 run.py        # Baseline")
    print("  python3 run_lstm.py   # LSTM")
    print("="*100)

if __name__ == "__main__":
    run_comparison()
