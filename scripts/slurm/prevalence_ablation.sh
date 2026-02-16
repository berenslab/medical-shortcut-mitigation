#!/bin/bash
# SLURM script to run k-fold CV for multiple prevalence strengths sequentially on one compute node
# Automatically uses GPUs allocated by SLURM. Optional bucket_samples are included if they exist.
#
# Usage:
# sbatch run_prevalence_ablation_slurm.sh NUM_FOLDS EXPERIMENT_ROOT BASE_CONFIG_DIR METHOD PREV1 PREV2 ...

# ------------------ SLURM DIRECTIVES ------------------
#SBATCH --job-name=prevalence_ablation
#SBATCH --nodes=1
#SBATCH --gres=gpu:2                 # Adjust max GPUs needed
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=1-12:00
#SBATCH --output=logfiles/%x_%j.out
#SBATCH --error=logfiles/%x_%j.err
# #SBATCH --partition=<your_partition>    # Optional: set your partition
# #SBATCH --mail-type=END,FAIL            # Optional: notifications
# #SBATCH --mail-user=your_email@example.com # Optional: email

echo "Job started: $(date)"
scontrol show job $SLURM_JOB_ID

# ------------------ PARSE ARGUMENTS ------------------
NUM_FOLDS=$1
EXPERIMENT_ROOT=$2
BASE_CONFIG_DIR=$3
METHOD=$4
shift 4     # Remaining args are prevalence strengths

BASE_CONFIG="${BASE_CONFIG_DIR}/${METHOD}.yaml"
PROJECT_DIR=$WORK/medical-shortcut-mitigation

echo "Base config: $BASE_CONFIG"
echo "Experiment root: $EXPERIMENT_ROOT"
echo "Base config dir: $BASE_CONFIG_DIR"
echo "Method: $METHOD"
echo "Prevalence strengths: $@"

set -e
source .venv/bin/activate

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Using GPUs allocated by Slurm: $CUDA_VISIBLE_DEVICES"

# ------------------ DETERMINE CORRELATION FORMAT ------------------
CONFIG_FORMAT="none"
if [[ "$BASE_CONFIG_DIR" == *"morpho-mnist"* ]]; then
    CONFIG_FORMAT="list"
elif [[ "$BASE_CONFIG_DIR" == *"kermani-oct"* ]]; then
    CONFIG_FORMAT="dict"
fi
echo "Correlation strength format for this dataset: $CONFIG_FORMAT"

# ------------------ LOOP OVER PREVALENCE STRENGTHS ------------------
for PREV in "$@"; do
    echo "============================================================"
    echo " Running prevalence strength: $PREV"
    echo "============================================================"

    OUT_DIR="${EXPERIMENT_ROOT}/prevalence_ablation/${PREV}/${METHOD}"
    mkdir -p "$OUT_DIR"

    # ------------------ DETERMINE CORRELATION STRENGTH ------------------
    CORRELATION_ARG=""
    if [ "$CONFIG_FORMAT" == "list" ]; then
        TASK1=$PREV
        TASK2=$((100 - PREV))
        CORRELATION_ARG="--data/correlation_strength \"[${TASK1},${TASK2}]\""

    elif [ "$CONFIG_FORMAT" == "dict" ]; then
        TASK1=$(awk "BEGIN {print $PREV/100}")
        TASK2=$(awk "BEGIN {print 1 - $PREV/100}")
        CORRELATION_ARG="--data/correlation_strength \"{\\\"task1\\\":${TASK1},\\\"task2\\\":${TASK2}}\""
    fi

    # ------------------ CHECK BUCKET SAMPLES ------------------
    BUCKET_ARG=""
    BUCKET_SAMPLES_FILE="${BASE_CONFIG_DIR}/prevalence_to_buckets/${PREV}.json"
    if [ -f "$BUCKET_SAMPLES_FILE" ]; then
        BUCKET_SAMPLES=$(cat "$BUCKET_SAMPLES_FILE")
        BUCKET_ARG="--data/bucket_samples \"$BUCKET_SAMPLES\""
    else
        echo "No bucket_samples file found, skipping for this dataset."
    fi

    echo "OUT_DIR              = $OUT_DIR"
    echo "correlation arg      = $CORRELATION_ARG"
    echo "bucket samples arg   = $BUCKET_ARG"

    # ------------------ LOOP OVER FOLDS ------------------
    for ((FOLD=0; FOLD<NUM_FOLDS; FOLD++)); do
        echo "-------------------------------------------------------"
        echo " Fold $((FOLD+1)) / $NUM_FOLDS for prevalence $PREV"
        echo "-------------------------------------------------------"

        # ---------- BUILD TRAIN COMMAND ----------
        TRAIN_CMD="python $PROJECT_DIR/train.py"
        TRAIN_CMD+=" --cfg \"$BASE_CONFIG\""
        TRAIN_CMD+=" --training/num_folds $NUM_FOLDS"
        TRAIN_CMD+=" --training/fold $FOLD"
        TRAIN_CMD+=" --out_dir \"$OUT_DIR\""
        [ -n "$CORRELATION_ARG" ] && TRAIN_CMD+=" $CORRELATION_ARG"
        [ -n "$BUCKET_ARG" ] && TRAIN_CMD+=" $BUCKET_ARG"

        echo "Running TRAIN command:"
        echo "$TRAIN_CMD"
        eval $TRAIN_CMD

        # ---------- BUILD TEST COMMAND ----------
        TEST_CMD="python $PROJECT_DIR/test.py"
        TEST_CMD+=" --cfg \"$BASE_CONFIG\""
        TEST_CMD+=" --training/fold $FOLD"
        TEST_CMD+=" --out_dir \"$OUT_DIR\""
        [ -n "$CORRELATION_ARG" ] && TEST_CMD+=" $CORRELATION_ARG"
        [ -n "$BUCKET_ARG" ] && TEST_CMD+=" $BUCKET_ARG"

        echo "Running TEST command:"
        echo "$TEST_CMD"
        eval $TEST_CMD

    done
done

echo "All prevalence strengths completed at: $(date)"