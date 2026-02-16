#!/bin/bash
# Generic SLURM array job script for k-fold training/testing
# Usage:
# sbatch --array=0-<num_folds-1> scripts/train_kfold_slurm.sh <config_file> <num_folds>

#SBATCH --job-name=lightning_kfold          # Name of the job
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --gres=gpu:1                        # Number of GPUs per node
#SBATCH --cpus-per-task=16                  # CPU cores per task
#SBATCH --mem=50G                           # Memory per node
#SBATCH --time=0-02:00                      # Max runtime (D-HH:MM)
#SBATCH --output=logfiles/%x_%A_%a.out
#SBATCH --error=logfiles/%x_%A_%a.err
# #SBATCH --mail-type=END,FAIL              # Optional: uncomment to enable email notifications
# #SBATCH --mail-user=your_email@example.com # Optional: set your email here
# #SBATCH --partition=<your_partition>      # Optional: set partition here
# #SBATCH --array=0-4                       # Array job; override via sbatch command

# Print job info
scontrol show job $SLURM_JOB_ID

# Project directory (update as needed)
PROJECT_DIR=$WORK/medical-shortcut-mitigation/

echo "Starting training"
set -e  # Exit on first error

# Activate virtual environment
source .venv/bin/activate

# Optional debugging flags
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Fold-specific info
echo "Fold $SLURM_ARRAY_TASK_ID starting on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || echo "No GPU info available"

# Run training and testing
python $PROJECT_DIR/train.py --cfg $1 --training/num_folds $2 --training/fold $SLURM_ARRAY_TASK_ID
python $PROJECT_DIR/test.py --cfg $1 --training/fold $SLURM_ARRAY_TASK_ID