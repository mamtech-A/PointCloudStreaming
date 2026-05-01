#!/usr/bin/env python3
"""
Script to train the LSTM bandwidth prediction model.

This script:
1. Loads all bandwidth traces from the bandwith/ folder
2. Splits files into train and test sets (file-level split)
3. Performs hyperparameter tuning via grid search
4. Trains a proper PyTorch LSTM model
5. Saves the trained model to models/bandwidth_lstm.pkl
6. Saves the best model to models/bandwidth_lstm_best.pkl
7. Saves file-level split metadata to models/bandwidth_lstm_split.json

Usage:
    python3 train_model.py [--no-tune]
    
Options:
    --no-tune: Skip hyperparameter tuning and use default parameters

The trained model can then be used with run_lstm.py for bandwidth prediction.
"""

import os
import sys
import argparse

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
sys.path.insert(0, src_dir)

from lstm_model import train_lstm_model, prepare_dataset


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LSTM bandwidth prediction model')
    parser.add_argument('--no-tune', action='store_true', 
                        help='Skip hyperparameter tuning and use default parameters')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Number of historical samples for prediction (default: 10)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs (default: 100)')
    parser.add_argument('--test-size', type=float, default=0.1,
                        help='Fraction of files for test split at file level (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic file-level split (default: 42)')
    args = parser.parse_args()
    
    print("="*80)
    print("🤖 LSTM Bandwidth Prediction Model Training")
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
    print(f"🔧 Hyperparameter tuning: {'Disabled' if args.no_tune else 'Enabled'}")
    print(f"📊 Sequence length: {args.sequence_length}")
    print(f"🧪 File-level test split: {args.test_size:.2f}")
    print(f"🎲 Split random seed: {args.seed}")
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Train model with hyperparameter tuning
    print("\n" + "="*80)
    print("Training Process:")
    print("  1. Load and prepare dataset")
    print("  2. Split files into train/test sets (file-level)")
    if not args.no_tune:
        print("  3. Perform hyperparameter grid search")
        print("  4. Train final model with best hyperparameters")
    else:
        print("  3. Train model with default hyperparameters")
    print("  5. Save trained model and best model")
    print("="*80 + "\n")
    
    model = train_lstm_model(
        bandwidth_dir, 
        output_path, 
        sequence_length=args.sequence_length,
        tune=not args.no_tune,
        verbose=True,
        test_size=args.test_size,
        random_state=args.seed
    )
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print(f"📦 Model saved to: {output_path}")
    print(f"📦 Best model saved to: {output_path.replace('.pkl', '_best.pkl')}")
    print(f"🧾 Split metadata saved to: {output_path.replace('.pkl', '_split.json')}")
    
    if not args.no_tune:
        results_path = output_path.replace('.pkl', '_tuning_results.json')
        print(f"📊 Tuning results saved to: {results_path}")
    
    print("\nNext steps:")
    print("  1. Run the system with LSTM prediction:")
    print("     python3 run_lstm.py")
    print("  2. Or compare with baseline (no LSTM):")
    print("     python3 run.py")
    print("  3. Use a test-only simulation trace from:")
    print(f"     {output_path.replace('.pkl', '_split.json')} -> test_files")
    print("="*80)


if __name__ == "__main__":
    main()
