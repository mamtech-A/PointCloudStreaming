#!/usr/bin/env python3
"""
Script to train the LSTM bandwidth prediction model.

This script:
1. Loads all bandwidth traces from the bandwidth_5g/ folder
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

# Make emoji-rich status prints safe under non-UTF-8 consoles (e.g. cp1256 on redirect).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of files for test split at file level (default: 0.2 '
                             '= 4 held-out test traces with the 21-file 5G dataset)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic file-level split (default: 42)')
    parser.add_argument('--transform', choices=['none', 'log1p'], default='none',
                        help="Input pre-transform: 'log1p' suits the right-skewed 5G "
                             "throughput distribution (default: none)")
    parser.add_argument('--out', default=None,
                        help='Output model path (default: models/bandwidth_lstm.pkl)')
    # Explicit hyperparameters (used with --no-tune, e.g. to retrain the grid
    # winner at a different sequence length or transform without re-tuning).
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    print("="*80)
    print("🤖 LSTM Bandwidth Prediction Model Training")
    print("="*80)
    
    # Set paths
    project_root = script_dir
    bandwidth_dir = os.path.join(project_root, 'bandwidth_5g')
    model_dir = os.path.join(project_root, 'models')
    output_path = args.out or os.path.join(model_dir, 'bandwidth_lstm.pkl')

    # Check if bandwidth directory exists
    if not os.path.exists(bandwidth_dir):
        print(f"❌ Error: Bandwidth directory not found at {bandwidth_dir}")
        print("   Run prepare_5g_traces.py first (see bandwidth_5g/README.md).")
        sys.exit(1)

    # Count trace files
    log_files = [f for f in os.listdir(bandwidth_dir) if f.endswith(('.log', '.csv'))]
    if len(log_files) == 0:
        print(f"❌ Error: No trace files found in {bandwidth_dir}")
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
        random_state=args.seed,
        transform=args.transform,
        epochs=args.epochs,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        learning_rate=args.lr
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
