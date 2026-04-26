#!/bin/bash
# Run v2 analyses on eyecog server
# Fixes PCA bias with random projections, velocity-space PCA, and probe-informed projection
#
# Data locations on eyecog:
#   trajectories_0shot: /data/thanhdo/trajectories_0shot/
#   trajectories_8shot: /data/thanhdo/trajectories_8shot/
#   trajectories (current): ~/p1_geom_characterization/data/trajectories/
#
# Usage:
#   ssh eyecog
#   cd ~/p1_geom_characterization
#   bash scripts/analysis/run_v2_analyses.sh [0shot|8shot|current]

set -euo pipefail

# Force Python to flush stdout immediately (otherwise tee gets nothing until script ends)
export PYTHONUNBUFFERED=1

# ============================================================================
# Configuration
# ============================================================================

MODELS="olmo3_base olmo3_sft olmo3_rl_zero olmo3_think"
TASKS="gsm8k humaneval logiqa"
MAX_SAMPLES=500
N_PROJ_DIMS=64
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================================================
# Parse arguments
# ============================================================================

DATA_VARIANT="${1:-0shot}"

case "$DATA_VARIANT" in
    0shot)
        DATA_DIR="/data/thanhdo/trajectories_0shot"
        ;;
    8shot)
        DATA_DIR="/data/thanhdo/trajectories_8shot"
        ;;
    current)
        DATA_DIR="$HOME/p1_geom_characterization/data/trajectories"
        ;;
    *)
        echo "ERROR: Unknown data variant '$DATA_VARIANT'"
        echo "Usage: $0 [0shot|8shot|current]"
        echo "  0shot   - /data/thanhdo/trajectories_0shot/ (default)"
        echo "  8shot   - /data/thanhdo/trajectories_8shot/"
        echo "  current - ~/p1_geom_characterization/data/trajectories/"
        exit 1
        ;;
esac

# ============================================================================
# Timing
# ============================================================================

OVERALL_START=$(date +%s)
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "============================================================"
echo "  ManiVer v2 Analyses Runner"
echo "============================================================"
echo "  Date:         $(timestamp)"
echo "  Data variant: $DATA_VARIANT"
echo "  Data dir:     $DATA_DIR"
echo "  Models:       $MODELS"
echo "  Tasks:        $TASKS"
echo "  Max samples:  $MAX_SAMPLES"
echo "  Proj dims:    $N_PROJ_DIMS"
echo "  Project dir:  $PROJECT_DIR"
echo "============================================================"

# ============================================================================
# Validate data directory
# ============================================================================

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    echo "Check that you're on the eyecog server and the data is available."
    exit 1
fi

echo ""
echo "--- Checking available trajectory files ---"
FILE_COUNT=0
MISSING_COUNT=0
AVAILABLE_COMBOS=()
for model in $MODELS; do
    for task in $TASKS; do
        h5_file="$DATA_DIR/$model/${task}_trajectories.h5"
        if [ -f "$h5_file" ]; then
            size=$(du -h "$h5_file" | cut -f1)
            echo "  FOUND: $model/$task ($size)"
            FILE_COUNT=$((FILE_COUNT + 1))
            AVAILABLE_COMBOS+=("$model/$task")
        else
            echo "  MISSING: $model/$task"
            MISSING_COUNT=$((MISSING_COUNT + 1))
        fi
    done
done

echo ""
echo "  Total: $FILE_COUNT found, $MISSING_COUNT missing"

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "ERROR: No trajectory files found in $DATA_DIR"
    echo "Expected files like: $DATA_DIR/<model>/<task>_trajectories.h5"
    exit 1
fi

# ============================================================================
# Activate conda environment
# ============================================================================

echo ""
echo "--- Activating conda environment ---"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
echo "  Python: $(which python)"
echo "  Python version: $(python --version 2>&1)"

# ============================================================================
# Check/install signatory
# ============================================================================

echo ""
echo "--- Checking signatory installation ---"
if python -c "import signatory; print(f'  signatory {signatory.__version__} installed')" 2>/dev/null; then
    echo "  signatory is available"
else
    echo "  signatory not found, attempting install..."
    pip install signatory==1.2.6.1.9.0 2>/dev/null || echo "  WARNING: signatory install failed - may need manual install"
    # Verify after install attempt
    if python -c "import signatory" 2>/dev/null; then
        echo "  signatory installed successfully"
    else
        echo "  WARNING: signatory still not available. Path signature analysis will be limited."
    fi
fi

# ============================================================================
# Create output directory
# ============================================================================

OUTPUT_DIR="$PROJECT_DIR/results/v2_${DATA_VARIANT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo ""
echo "  Output directory: $OUTPUT_DIR"
echo "  Log directory:    $LOG_DIR"

# ============================================================================
# Step 1: Path Signature Analysis v2
# ============================================================================

echo ""
echo "============================================================"
echo "  Step 1: Path Signature Analysis v2"
echo "  $(timestamp)"
echo "============================================================"

SIGNATURE_SCRIPT="$PROJECT_DIR/scripts/analysis/path_signature_analysis_v2.py"
SIGNATURE_LOG="$LOG_DIR/path_signature_v2.log"

if [ ! -f "$SIGNATURE_SCRIPT" ]; then
    echo "ERROR: Script not found: $SIGNATURE_SCRIPT"
    echo "Skipping path signature analysis."
    STEP1_STATUS="SKIPPED (script not found)"
else
    STEP1_START=$(date +%s)
    MODELS_CSV=$(echo $MODELS | tr ' ' ',')

    echo "  Running: python $SIGNATURE_SCRIPT"
    echo "    --data-dir $DATA_DIR"
    echo "    --models $MODELS_CSV"
    echo "    --output-dir $OUTPUT_DIR"
    echo "    --max-samples $MAX_SAMPLES"
    echo "    --n-proj-dims $N_PROJ_DIMS"
    echo "  Log: $SIGNATURE_LOG"
    echo ""

    set +e
    python "$SIGNATURE_SCRIPT" \
        --data-dir "$DATA_DIR" \
        --models "$MODELS_CSV" \
        --output-dir "$OUTPUT_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --n-proj-dims "$N_PROJ_DIMS" \
        2>&1 | tee "$SIGNATURE_LOG"
    STEP1_EXIT=${PIPESTATUS[0]}
    set -e

    STEP1_END=$(date +%s)
    STEP1_ELAPSED=$((STEP1_END - STEP1_START))
    STEP1_MINS=$((STEP1_ELAPSED / 60))
    STEP1_SECS=$((STEP1_ELAPSED % 60))

    if [ "$STEP1_EXIT" -eq 0 ]; then
        STEP1_STATUS="SUCCESS (${STEP1_MINS}m ${STEP1_SECS}s)"
        echo ""
        echo "  Path signature analysis completed in ${STEP1_MINS}m ${STEP1_SECS}s"
    else
        STEP1_STATUS="FAILED (exit code $STEP1_EXIT, ${STEP1_MINS}m ${STEP1_SECS}s)"
        echo ""
        echo "  WARNING: Path signature analysis failed (exit code $STEP1_EXIT)"
        echo "  Check log: $SIGNATURE_LOG"
    fi
fi

# ============================================================================
# Step 2: Phase 3 Dynamical Analysis v2 (per model)
# ============================================================================

echo ""
echo "============================================================"
echo "  Step 2: Phase 3 Dynamical Analysis (per model)"
echo "  $(timestamp)"
echo "============================================================"

# Try v2 first, fall back to original
DYNAMICAL_SCRIPT="$PROJECT_DIR/scripts/analysis/phase3_dynamical_analysis_v2.py"
DYNAMICAL_SCRIPT_NAME="phase3_dynamical_analysis_v2.py"
if [ ! -f "$DYNAMICAL_SCRIPT" ]; then
    echo "  NOTE: v2 script not found, falling back to phase3_dynamical_analysis.py"
    DYNAMICAL_SCRIPT="$PROJECT_DIR/scripts/analysis/phase3_dynamical_analysis.py"
    DYNAMICAL_SCRIPT_NAME="phase3_dynamical_analysis.py"
fi

if [ ! -f "$DYNAMICAL_SCRIPT" ]; then
    echo "ERROR: No dynamical analysis script found."
    echo "  Looked for: phase3_dynamical_analysis_v2.py"
    echo "  Looked for: phase3_dynamical_analysis.py"
    echo "Skipping dynamical analysis."
    STEP2_STATUS="SKIPPED (script not found)"
else
    STEP2_START=$(date +%s)
    STEP2_TOTAL=0
    STEP2_SUCCESS=0
    STEP2_FAILED=0
    STEP2_SKIPPED=0

    echo "  Using: $DYNAMICAL_SCRIPT_NAME"
    echo ""

    for model in $MODELS; do
        # Determine which tasks have data for this model
        AVAILABLE_TASKS=""
        for task in $TASKS; do
            h5_file="$DATA_DIR/$model/${task}_trajectories.h5"
            if [ -f "$h5_file" ]; then
                if [ -z "$AVAILABLE_TASKS" ]; then
                    AVAILABLE_TASKS="$task"
                else
                    AVAILABLE_TASKS="$AVAILABLE_TASKS,$task"
                fi
            fi
        done

        if [ -z "$AVAILABLE_TASKS" ]; then
            echo "  SKIP: $model - no trajectory files found"
            STEP2_SKIPPED=$((STEP2_SKIPPED + 1))
            continue
        fi

        STEP2_TOTAL=$((STEP2_TOTAL + 1))
        MODEL_LOG="$LOG_DIR/phase3_dynamical_${model}.log"
        MODEL_OUTPUT="$OUTPUT_DIR/phase3_dynamical_${model}.json"

        echo "  --- $model (tasks: $AVAILABLE_TASKS) ---"
        echo "    Log: $MODEL_LOG"

        MODEL_START=$(date +%s)
        set +e
        python "$DYNAMICAL_SCRIPT" \
            --data-dir "$DATA_DIR" \
            --model "$model" \
            --tasks "$AVAILABLE_TASKS" \
            --output "$MODEL_OUTPUT" \
            --max-samples "$MAX_SAMPLES" \
            2>&1 | tee "$MODEL_LOG"
        MODEL_EXIT=${PIPESTATUS[0]}
        set -e

        MODEL_END=$(date +%s)
        MODEL_ELAPSED=$((MODEL_END - MODEL_START))
        MODEL_MINS=$((MODEL_ELAPSED / 60))
        MODEL_SECS=$((MODEL_ELAPSED % 60))

        if [ "$MODEL_EXIT" -eq 0 ]; then
            echo "    Completed in ${MODEL_MINS}m ${MODEL_SECS}s"
            STEP2_SUCCESS=$((STEP2_SUCCESS + 1))
        else
            echo "    FAILED (exit code $MODEL_EXIT) after ${MODEL_MINS}m ${MODEL_SECS}s"
            STEP2_FAILED=$((STEP2_FAILED + 1))
        fi
        echo ""
    done

    STEP2_END=$(date +%s)
    STEP2_ELAPSED=$((STEP2_END - STEP2_START))
    STEP2_MINS=$((STEP2_ELAPSED / 60))
    STEP2_SECS=$((STEP2_ELAPSED % 60))
    STEP2_STATUS="${STEP2_SUCCESS}/${STEP2_TOTAL} succeeded, ${STEP2_FAILED} failed, ${STEP2_SKIPPED} skipped (${STEP2_MINS}m ${STEP2_SECS}s)"
fi

# ============================================================================
# Summary
# ============================================================================

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$((OVERALL_END - OVERALL_START))
OVERALL_MINS=$((OVERALL_ELAPSED / 60))
OVERALL_SECS=$((OVERALL_ELAPSED % 60))

echo ""
echo "============================================================"
echo "  ANALYSIS COMPLETE"
echo "============================================================"
echo "  Finished:       $(timestamp)"
echo "  Total time:     ${OVERALL_MINS}m ${OVERALL_SECS}s"
echo "  Data variant:   $DATA_VARIANT"
echo "  Data dir:       $DATA_DIR"
echo ""
echo "  Step 1 (Path Signatures v2): $STEP1_STATUS"
echo "  Step 2 (Dynamical Analysis):  $STEP2_STATUS"
echo ""
echo "  Output directory: $OUTPUT_DIR"
echo ""

# List output files
echo "  Output files:"
if [ -d "$OUTPUT_DIR" ]; then
    for ext in csv json; do
        for f in "$OUTPUT_DIR"/*."$ext"; do
            if [ -f "$f" ]; then
                size=$(du -h "$f" | cut -f1)
                echo "    $(basename "$f") ($size)"
            fi
        done
    done
fi

echo ""
echo "  Log files:"
if [ -d "$LOG_DIR" ]; then
    for f in "$LOG_DIR"/*.log; do
        if [ -f "$f" ]; then
            size=$(du -h "$f" | cut -f1)
            echo "    $(basename "$f") ($size)"
        fi
    done
fi

echo ""
echo "============================================================"
