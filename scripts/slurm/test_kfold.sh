#!/bin/bash
# Generic SLURM array job script for k-fold testing
# Usage:
# sbatch --array=0-<num_folds-1> scripts/test_kfold_slurm.sh <config_file> <num_folds>

#SBATCH --job-name=lightning_kfold_test   # Job name
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gres=gpu:1                       # Number of GPUs
#SBATCH --cpus-per-task=16                 # CPUs per task
#SBATCH --mem=50G                          # Memory per node
#SBATCH --time=0-02:00                     # Max runtime (D-HH:MM)
#SBATCH --output=logfiles/%x_%A_%a.out
#SBATCH --error=logfiles/%x_%A_%a.err
# #SBATCH --partition=<your_partition>      # Optional: uncomment/set if needed
# #SBATCH --mail-type=END,FAIL              # Optional: notifications
# #SBATCH --mail-user=your_email@example.com # Optional: set your email
# #SBATCH --array=0-4                       # Array index for folds (override on sbatch)

set -e  # Exit on first error

# Print job info
scontrol show job $SLURM_JOB_ID

# Project directory (auto-detect repo root or fallback)
PROJECT_DIR=$WORK/medical-shortcut-mitigation
# Alternatively, set manually:
# PROJECT_DIR=$(pwd)

# Activate virtual environment
source .venv/bin/activate

# Optional debugging
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Testing Fold $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || echo "No GPU info available"

python $PROJECT_DIR/test.py \
    --cfg $1 \
    --training/fold $SLURM_ARRAY_TASK_ID \
    --metrics '["Accuracy","AUROC"]'