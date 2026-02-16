#!/bin/bash
# Generic script to run k-fold training/testing sequentially on local machine
# Usage:
# sh scripts/train_kfold_local.sh <config_file> <num_folds>

set -e  # Exit immediately if a command fails

# Project directory (auto-detect repo root or fallback)
PROJECT_DIR=$(pwd)  # Assumes the script is run from the repo root
# Alternatively, uncomment and set manually:
# PROJECT_DIR=$HOME/medical-shortcut-mitigation

# Activate virtual environment
source .venv/bin/activate

# Number of folds
NUM_FOLDS=$2

echo "Running $NUM_FOLDS folds sequentially"

for ((i=0; i<NUM_FOLDS; i++)); do
    echo "=== Fold $i ==="
    python $PROJECT_DIR/train.py --cfg $1 --training/num_folds $NUM_FOLDS --training/fold $i
    python $PROJECT_DIR/test.py --cfg $1 --training/fold $i
done

# Deactivate virtual environment
deactivate

echo "All folds completed successfully!"