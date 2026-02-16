#!/bin/bash
# Generic script to run k-fold testing sequentially on local machine
# Usage:
# sh scripts/test_kfold_local.sh <config_file> <num_folds>

set -e  # Exit on first error

# Project directory (auto-detect repo root or fallback)
PROJECT_DIR=$(pwd)  # Assumes script is run from repo root
# Alternatively, set manually:
# PROJECT_DIR=$HOME/Documents/medical-shortcut-mitigation

# Activate virtual environment
source .venv/bin/activate

NUM_FOLDS=$2

echo "Running testing for $NUM_FOLDS folds sequentially"

for ((i=0; i<NUM_FOLDS; i++)); do
    echo "=== Testing Fold $i ==="
    python $PROJECT_DIR/test.py \
        --cfg $1 \
        --training/fold $i \
        --metrics '["Accuracy","AUROC"]'
done

# Deactivate virtual environment
deactivate

echo "All folds tested successfully!"
