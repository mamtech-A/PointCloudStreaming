#!/usr/bin/env python3
"""
Script to train the LSTM bandwidth prediction model.

This script:
1. Loads all bandwidth traces from the bandwith/ folder
2. Prepares training data (sequences of 10 measurements)
3. Trains the SimpleLSTM model
4. Saves the trained model to models/bandwidth_lstm.pkl

Usage:
    python3 train_model.py
    
The trained model can then be used with run_lstm.py for bandwidth prediction.
"""

import os
import sys

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
sys.path.insert(0, src_dir)

from bandwidth_lstm import train_lstm_model

def main():
    print("="*80)
    print("LSTM Bandwidth Prediction Model Training")
    print("="*80)
    
    # Set paths
    project_root = script_dir
    bandwidth_dir = os.path.join(project_root, 'bandwith')
    model_dir = os.path.join(project_root, 'models')
    output_path = os.path.join(model_dir, 'bandwidth_lstm.pkl')
    
    # Check if bandwidth directory exists
    if not os.path.exists(bandwidth_dir):
        print(f"❌ Error: Bandwidth directory not found at {bandwidth_dir}")
        print("   Please ensure the 'bandwith' folder exists with .log files.")
        sys.exit(1)
    
    # Count log files
    log_files = [f for f in os.listdir(bandwidth_dir) if f.endswith('.log')]
    if len(log_files) == 0:
        print(f"❌ Error: No .log files found in {bandwidth_dir}")
        sys.exit(1)
    
    print(f"\n📁 Found {len(log_files)} bandwidth trace files")
    print(f"📂 Output directory: {model_dir}")
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Train model
    print("\n" + "="*80)
    model = train_lstm_model(bandwidth_dir, output_path, sequence_length=10)
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print(f"📦 Model saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Run the system with LSTM prediction:")
    print("     python3 run_lstm.py")
    print("  2. Or compare with baseline (no LSTM):")
    print("     python3 run.py")
    print("="*80)

if __name__ == "__main__":
    main()
