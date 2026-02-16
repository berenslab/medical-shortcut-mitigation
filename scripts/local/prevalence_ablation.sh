#!/bin/bash
# Generic local script to run k-fold CV for multiple prevalence strengths
# Integrates config-format detection, optional bucket_samples,
# and correlation-strength logic.
#
# Usage:
# sh scripts/prevalence_ablation_local.sh NUM_FOLDS EXPERIMENT_ROOT BASE_CONFIG_DIR METHOD DEVICES PREV1 PREV2 ...

set -e  # Exit on first error

# ------------------ PARSE ARGUMENTS ------------------
NUM_FOLDS=$1
EXPERIMENT_ROOT=$2
BASE_CONFIG_DIR=$3
METHOD=$4
DEVICES=$5      # JSON list, e.g. "[0,1]"
shift 5         # Remaining args → prevalence strengths

BASE_CONFIG="${BASE_CONFIG_DIR}/${METHOD}.yaml"
PROJECT_DIR=$(pwd)  # Assumes script is run from repo root; adjust if needed

echo "Base config:         $BASE_CONFIG"
echo "Experiment root:     $EXPERIMENT_ROOT"
echo "Base config dir:     $BASE_CONFIG_DIR"
echo "Method:              $METHOD"
echo "GPU devices:         $DEVICES"
echo "Prevalence strengths: $@"

# Activate virtual environment
source .venv/bin/activate

export PYTHONFAULTHANDLER=1
echo "Using GPU devices: $DEVICES"

# ------------------ DETERMINE CORRELATION FORMAT ------------------
CONFIG_FORMAT="none"
if [[ "$BASE_CONFIG_DIR" == *"morpho-mnist"* ]]; then
    CONFIG_FORMAT="list"
elif [[ "$BASE_CONFIG_DIR" == *"kermani-oct"* ]]; then
    CONFIG_FORMAT="dict"
fi
echo "Correlation strength format: $CONFIG_FORMAT"

# ------------------ LOOP OVER PREVALENCE STRENGTHS ------------------
for PREV in "$@"; do
    echo "============================================================"
    echo " Running prevalence strength: $PREV"
    echo "============================================================"

    OUT_DIR="${EXPERIMENT_ROOT}/prevalence_ablation/${PREV}/${METHOD}"
    mkdir -p "$OUT_DIR"

    # ---------- BUILD CORRELATION ARG ----------
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

    # ---------- CHECK FOR BUCKET SAMPLES ----------
    BUCKET_ARG=""
    BUCKET_SAMPLES_FILE="${BASE_CONFIG_DIR}/prevalence_to_buckets/${PREV}.json"
    if [ -f "$BUCKET_SAMPLES_FILE" ]; then
        BUCKET_SAMPLES=$(cat "$BUCKET_SAMPLES_FILE")
        BUCKET_ARG="--data/bucket_samples \"$BUCKET_SAMPLES\""
    else
        echo "No bucket_samples file found for PREV=$PREV → skipping."
    fi

    echo "OUT_DIR            = $OUT_DIR"
    echo "correlation arg    = $CORRELATION_ARG"
    echo "bucket samples arg = $BUCKET_ARG"

    # ---------- LOOP OVER FOLDS ----------
    for ((FOLD=0; FOLD<NUM_FOLDS; FOLD++)); do
        echo "-------------------------------------------------------"
        echo " Fold $((FOLD+1)) / $NUM_FOLDS for prevalence $PREV"
        echo "-------------------------------------------------------"

        # ---------- TRAIN COMMAND ----------
        TRAIN_CMD="python $PROJECT_DIR/train.py"
        TRAIN_CMD+=" --cfg \"$BASE_CONFIG\""
        TRAIN_CMD+=" --training/num_folds $NUM_FOLDS"
        TRAIN_CMD+=" --training/fold $FOLD"
        TRAIN_CMD+=" --devices \"$DEVICES\""
        TRAIN_CMD+=" --out_dir \"$OUT_DIR\""
        [ -n "$CORRELATION_ARG" ] && TRAIN_CMD+=" $CORRELATION_ARG"
        [ -n "$BUCKET_ARG" ] && TRAIN_CMD+=" $BUCKET_ARG"

        echo "Running TRAIN command:"
        echo "$TRAIN_CMD"
        eval $TRAIN_CMD

        # ---------- TEST COMMAND ----------
        TEST_CMD="python $PROJECT_DIR/test.py"
        TEST_CMD+=" --cfg \"$BASE_CONFIG\""
        TEST_CMD+=" --training/fold $FOLD"
        TEST_CMD+=" --devices \"$DEVICES\""
        TEST_CMD+=" --out_dir \"$OUT_DIR\""
        [ -n "$CORRELATION_ARG" ] && TEST_CMD+=" $CORRELATION_ARG"
        [ -n "$BUCKET_ARG" ] && TEST_CMD+=" $BUCKET_ARG"

        echo "Running TEST command:"
        echo "$TEST_CMD"
        eval $TEST_CMD
    done
done

# Deactivate virtual environment
deactivate

echo "All prevalence strengths completed successfully!"