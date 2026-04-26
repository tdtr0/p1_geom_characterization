#!/bin/bash
#
# Monitor SLURM job progress
# Run on LOGIN NODE to check job status and view logs
#
# Usage:
#   bash monitor_slurm_job.sh [job_id]
#

JOB_ID=$1

if [ -z "$JOB_ID" ]; then
    echo "Usage: bash monitor_slurm_job.sh [job_id]"
    echo ""
    echo "Your running jobs:"
    squeue -u $USER
    exit 1
fi

echo "========================================="
echo "SLURM Job Monitor"
echo "========================================="
echo "Job ID: $JOB_ID"
echo "Time: $(date)"
echo ""

# Check job status
echo "Job Status:"
squeue -j $JOB_ID 2>/dev/null || {
    echo "Job $JOB_ID not found in queue (may have completed or failed)"
    echo ""
}

# Find log files
echo ""
echo "Searching for log files..."

# Check common locations
TEST_OUT="$HOME/logiqa_test_out.txt"
TEST_ERR="$HOME/logiqa_test_err.txt"
FULL_OUT="$HOME/logiqa_collection_out.txt"
FULL_ERR="$HOME/logiqa_collection_err.txt"

if [ -f "$TEST_OUT" ]; then
    echo "Found test stdout: $TEST_OUT"
    STDOUT_FILE="$TEST_OUT"
fi

if [ -f "$FULL_OUT" ]; then
    echo "Found collection stdout: $FULL_OUT"
    STDOUT_FILE="$FULL_OUT"
fi

if [ -f "$TEST_ERR" ]; then
    echo "Found test stderr: $TEST_ERR"
    STDERR_FILE="$TEST_ERR"
fi

if [ -f "$FULL_ERR" ]; then
    echo "Found collection stderr: $FULL_ERR"
    STDERR_FILE="$FULL_ERR"
fi

# Show last 30 lines of stdout
if [ -n "$STDOUT_FILE" ]; then
    echo ""
    echo "========================================="
    echo "Latest Output (last 30 lines):"
    echo "========================================="
    tail -30 "$STDOUT_FILE"
else
    echo ""
    echo "No stdout file found yet. Job may not have started."
fi

# Check for errors
if [ -n "$STDERR_FILE" ] && [ -s "$STDERR_FILE" ]; then
    echo ""
    echo "========================================="
    echo "Errors (if any):"
    echo "========================================="
    tail -30 "$STDERR_FILE"
fi

# Check for completion markers
if [ -n "$STDOUT_FILE" ]; then
    echo ""
    echo "========================================="
    echo "Progress Indicators:"
    echo "========================================="

    # Check if collection started
    if grep -q "OPTIMIZED LogiQA Collection\|FULLY OPTIMIZED LogiQA Collection" "$STDOUT_FILE" 2>/dev/null; then
        echo "✓ Collection script started"
    fi

    # Check if vLLM loaded
    if grep -q "vLLM loaded" "$STDOUT_FILE" 2>/dev/null; then
        echo "✓ vLLM model loaded"
    fi

    # Check if HF model loaded
    if grep -q "HF loaded" "$STDOUT_FILE" 2>/dev/null; then
        echo "✓ HuggingFace model loaded"
    fi

    # Check progress
    if grep -q "olmo3_sft/logiqa" "$STDOUT_FILE" 2>/dev/null; then
        PROGRESS=$(grep "olmo3_sft/logiqa" "$STDOUT_FILE" | tail -1)
        echo "Progress: $PROGRESS"
    fi

    # Check if B2 upload started
    if grep -q "Uploading to Backblaze B2" "$STDOUT_FILE" 2>/dev/null; then
        echo "✓ Collection complete, uploading to B2..."
    fi

    # Check if fully done
    if grep -q "All done!" "$STDOUT_FILE" 2>/dev/null; then
        echo "✓ JOB COMPLETED SUCCESSFULLY!"
    fi

    # Check for test completion
    if grep -q "TEST PASSED!" "$STDOUT_FILE" 2>/dev/null; then
        echo "✓ TEST JOB PASSED!"
    fi
fi

echo ""
echo "========================================="
echo "Commands:"
echo "========================================="
echo "To follow output live:"
echo "  tail -f $STDOUT_FILE"
echo ""
echo "To cancel job:"
echo "  scancel $JOB_ID"
echo ""
